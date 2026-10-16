use super::{InferenceError, dimensions, with_shape_and_encoding};
use crate::{StableHloOp, TensorType, Type};

/// Infers dequantization's expressed-element tensor using the operand shape.
///
/// Follows `inferUniformDequantizeOp`: the input encoding is not copied.
pub fn infer_uniform_dequantize_type(operand: &Type) -> Result<Type, InferenceError> {
    checked_operand(StableHloOp::UniformDequantize, operand)?;
    let Type::QuantizedTensor(tensor) = operand else {
        unreachable!("checked quantized operand");
    };
    Ok(TensorType::new(tensor.expressed_type(), tensor.dimensions().to_vec()).into())
}

/// Infers quantization or requantization using explicitly selected result elements.
///
/// `result_template` supplies quantization metadata, not shape or encoding.
/// Follows `inferUniformQuantizeOp` for the shape, then verifies the selected
/// result's expressed type and per-axis parameter count. Input encodings are
/// not copied. Declared-result shape/bound compatibility is checked separately
/// by operation verification.
pub fn infer_uniform_quantize_type(
    operand: &Type,
    result_template: &Type,
) -> Result<Type, InferenceError> {
    let opcode = StableHloOp::UniformQuantize;
    checked_operand(opcode, operand)?;
    result_template
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    if !opcode.accepts_fixed_result_type(0, result_template) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let Type::QuantizedTensor(result) = result_template else {
        unreachable!("checked quantized result");
    };
    let expressed = match operand {
        Type::Tensor(tensor) => tensor.element(),
        Type::QuantizedTensor(tensor) => tensor.expressed_type(),
        _ => unreachable!("checked float/quantized operand"),
    };
    if result.expressed_type() != expressed {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let result = with_shape_and_encoding(
        result_template,
        dimensions(operand)
            .expect("checked ranked operand")
            .to_vec(),
        None,
    );
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    Ok(result)
}

fn checked_operand(opcode: StableHloOp, operand: &Type) -> Result<(), InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    if !opcode.accepts_fixed_operand_type(0, operand) {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    Ok(())
}
