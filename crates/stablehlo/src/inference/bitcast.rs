use super::{InferenceError, bound, bounds_encoding, dimensions, with_shape_and_encoding};
use crate::{ElementType, StableHloOp, TensorType, Type};

/// Infers bitcast dimensions from the input and selected result elements.
///
/// The result template supplies ordinary or quantized elements, not dimensions
/// or encoding. Equal-width conversions preserve the shape; narrower elements
/// append a minor bit-width ratio, and wider elements remove that ratio.
/// Surviving dimension bounds are projected from the operand. Other encodings
/// are dropped rather than guessing how to transform their storage maps.
///
/// This implements the shape equations in bitcast_convert C1/C2; the pinned
/// reference supplies a verifier, not a dedicated result-inference routine.
pub fn infer_bitcast_convert_type(
    operand: &Type,
    result_template: &Type,
) -> Result<Type, InferenceError> {
    let (input_width, output_width) = widths(operand, result_template)?;
    let mut shape = dimensions(operand).expect("checked ranked input").to_vec();
    let mut bounds = (0..shape.len())
        .map(|axis| bound(operand, axis))
        .collect::<Vec<_>>();
    if input_width > output_width {
        shape.push(Some(input_width / output_width));
        bounds.push(None);
    } else if input_width < output_width {
        let Some(axis) = shape.len().checked_sub(1) else {
            return Err(InferenceError::InvalidBitcastMinorExtent);
        };
        validate_minor(operand, axis, output_width / input_width)?;
        shape.pop();
        bounds.pop();
    }
    let output = with_shape_and_encoding(result_template, shape, bounds_encoding(bounds));
    output
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    validate(operand, &output)?;
    Ok(output)
}

pub(crate) fn validate(input: &Type, output: &Type) -> Result<(), InferenceError> {
    let (input_width, output_width) = widths(input, output)?;
    if input_width == output_width {
        super::infer_common_shape(&[input, output])?;
        return Ok(());
    }
    let (small, big, ratio) = if input_width < output_width {
        (input, output, output_width / input_width)
    } else {
        (output, input, input_width / output_width)
    };
    let small_shape = dimensions(small).expect("checked ranked type");
    let big_shape = dimensions(big).expect("checked ranked type");
    if small_shape.len().checked_sub(1) != Some(big_shape.len()) {
        return Err(InferenceError::RankMismatch {
            expected: big_shape.len() + 1,
            actual: small_shape.len(),
        });
    }
    for axis in 0..big_shape.len() {
        super::infer_common_dimension(&[(small, axis), (big, axis)])?;
    }
    validate_minor(small, big_shape.len(), ratio)
}

fn validate_minor(ty: &Type, axis: usize, ratio: u64) -> Result<(), InferenceError> {
    let required = Type::from(TensorType::static_shape(ElementType::I1, [ratio]));
    super::infer_common_dimension(&[(ty, axis), (&required, 0)])
        .map_err(|_| InferenceError::InvalidBitcastMinorExtent)?;
    Ok(())
}

fn widths(input: &Type, output: &Type) -> Result<(u64, u64), InferenceError> {
    for (index, ty) in [input, output].into_iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        let accepted = if index == 0 {
            StableHloOp::BitcastConvert.accepts_fixed_operand_type(0, ty)
        } else {
            StableHloOp::BitcastConvert.accepts_fixed_result_type(0, ty)
        };
        if !accepted {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let element = |ty: &Type| match ty {
        Type::Tensor(tensor) => tensor.element(),
        Type::QuantizedTensor(tensor) => tensor.storage_type(),
        _ => unreachable!("checked bitcast tensor domain"),
    };
    let (input, output) = (element(input), element(output));
    let complex = |element| matches!(element, ElementType::C64 | ElementType::C128);
    if complex(input) != complex(output) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let input = u64::from(input.bit_width().expect("checked bitcast element"));
    let output = u64::from(output.bit_width().expect("checked bitcast element"));
    if !input.max(output).is_multiple_of(input.min(output)) {
        return Err(InferenceError::NonIntegralBitcastRatio { input, output });
    }
    Ok((input, output))
}
