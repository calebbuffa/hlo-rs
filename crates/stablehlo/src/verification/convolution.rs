//! Shared convolution inference, with operation-specific element constraints.

use std::collections::HashMap;

use super::{
    BuildError, convolution_dot_quantization_compatible, inferred_shape_matches,
    optional_integer_attribute, shaped_dimensions, window,
};
use crate::{
    Attribute, ConvolutionOptions, ElementType, Instruction, ScalarValue, StableHloAttribute,
    StableHloOp, Type, ValueId,
};

fn invalid(instruction: &Instruction) -> BuildError {
    BuildError::InvalidDimensionNumbers {
        operation: instruction.operation().name(),
    }
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if !matches!(opcode, StableHloOp::Convolution | StableHloOp::DynamicConv) {
        return Ok(());
    }
    let lhs = &available[&instruction.operands()[0]];
    let rhs = &available[&instruction.operands()[1]];
    let rank = shaped_dimensions(lhs)
        .expect("verified tensor operand")
        .len();
    if rank < 2 {
        return Err(invalid(instruction));
    }
    let padding = if opcode == StableHloOp::DynamicConv {
        let shape = shaped_dimensions(&available[&instruction.operands()[2]])
            .expect("verified padding tensor");
        if shape != [Some((rank - 2) as u64), Some(2)] {
            return Err(invalid(instruction));
        }
        vec![None; rank - 2]
    } else {
        window::padding(instruction, rank - 2)?
    };
    verify_with_padding(instruction, lhs, rhs, &padding)
}

pub(super) fn verify_with_padding(
    instruction: &Instruction,
    lhs: &Type,
    rhs: &Type,
    padding: &[Option<[i128; 2]>],
) -> Result<(), BuildError> {
    let attributes = instruction
        .operation()
        .attributes()
        .expect("StableHLO attributes");
    let Some(Attribute::StableHlo(attribute)) = attributes.get("dimension_numbers") else {
        unreachable!("verified convolution dimension numbers");
    };
    let StableHloAttribute::ConvDimensionNumbers(dimensions) = attribute.as_ref() else {
        unreachable!("verified convolution dimension numbers");
    };
    let group = |name| match attributes.get(name) {
        Some(Attribute::Integer { value, .. }) => {
            i64::try_from(*value).map_err(|_| invalid(instruction))
        }
        _ => unreachable!("verified positive group count"),
    };
    let strides = optional_integer_attribute(instruction, "window_strides")?;
    let lhs_dilation = optional_integer_attribute(instruction, "lhs_dilation")?;
    let rhs_dilation = optional_integer_attribute(instruction, "rhs_dilation")?;
    let reversal = match attributes.get("window_reversal") {
        Some(Attribute::DenseArray {
            element: ElementType::I1,
            values,
        }) => Some(
            values
                .iter()
                .map(|value| match value {
                    ScalarValue::Bool(value) => *value,
                    _ => unreachable!("verified boolean array"),
                })
                .collect::<Vec<_>>(),
        ),
        None => None,
        _ => unreachable!("verified reversal attribute"),
    };
    let options = ConvolutionOptions {
        dimension_numbers: dimensions,
        window_strides: strides.as_deref(),
        lhs_dilation: lhs_dilation.as_deref(),
        rhs_dilation: rhs_dilation.as_deref(),
        window_reversal: reversal.as_deref(),
        padding: None,
        feature_group_count: group("feature_group_count")?,
        batch_group_count: group("batch_group_count")?,
    };
    let error = |error| BuildError::TypeInference {
        operation: instruction.operation().name(),
        error,
    };
    let expected =
        crate::inference::infer_convolution_shape(lhs, rhs, options, padding).map_err(error)?;
    let result = &instruction.results()[0];
    crate::inference::validate_convolution_result(lhs, rhs, result.ty(), options).map_err(error)?;
    let elements_match = match (lhs, rhs, result.ty()) {
        (Type::Tensor(a), Type::Tensor(b), Type::Tensor(c)) => {
            a.element() == c.element()
                && (a.element() == b.element()
                    || (a.element().is_float()
                        && a.element().bit_width() == Some(8)
                        && b.element().is_float()
                        && b.element().bit_width() == Some(8)))
        }
        (_, _, result) => {
            convolution_dot_quantization_compatible(lhs, rhs, result)
                && matches!(rhs, Type::QuantizedTensor(rhs) if rhs.quantization_dimension()
                    .is_none_or(|axis| i64::try_from(axis) == Ok(dimensions.kernel_output_feature_dimension)))
                && match result {
                    Type::QuantizedTensor(result) => {
                        result.quantization_dimension().is_none_or(|axis| {
                            i64::try_from(axis) == Ok(dimensions.output_feature_dimension)
                        })
                    }
                    _ => true,
                }
        }
    };
    if !inferred_shape_matches(&expected, result.ty()) || !elements_match {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation().name(),
            result: result.value(),
        });
    }
    Ok(())
}
