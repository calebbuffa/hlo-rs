use super::{InferenceError, encoding};
use crate::{ElementType, StableHloOp, TensorType, Type};

/// Infers conversion to explicitly selected nonquantized elements.
///
/// Follows `inferConvertOp`: shape and encoding come from the operand, without
/// refinement from a declared result. The selected element must satisfy the
/// operation schema, and the retained encoding must remain valid for it.
pub fn infer_convert_type(operand: &Type, element: ElementType) -> Result<Type, InferenceError> {
    validate(operand, element)?;
    let Type::Tensor(input) = operand else {
        unreachable!("checked nonquantized tensor operand")
    };
    let mut output = TensorType::new(element, input.dimensions().to_vec());
    if let Some(encoding) = encoding(operand) {
        output = output.with_encoding(encoding.clone());
    }
    let output = Type::from(output);
    output
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    Ok(output)
}

pub(crate) fn validate(operand: &Type, element: ElementType) -> Result<(), InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    if !StableHloOp::Convert.accepts_fixed_operand_type(0, operand) {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    let output = Type::from(TensorType::scalar(element));
    output
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    if !StableHloOp::Convert.accepts_fixed_result_type(0, &output) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    Ok(())
}
