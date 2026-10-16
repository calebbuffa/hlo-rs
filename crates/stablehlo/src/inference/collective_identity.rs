use super::InferenceError;
use crate::{ElementType, StableHloOp, Type};

/// Infers collective-broadcast results, omitting the optional final root tensor.
///
/// Every data result retains its operand's exact type. With `has_dynamic_root`,
/// the last operand must be a rank-one i32 tensor with one entry per data
/// operand. Process groups, channel metadata and runtime root values are
/// operation-verifier responsibilities.
pub fn infer_collective_broadcast_types(
    operands: &[Type],
    has_dynamic_root: bool,
) -> Result<Vec<Type>, InferenceError> {
    let data = if has_dynamic_root {
        let Some((root, data)) = operands.split_last() else {
            return Err(InferenceError::EmptyInputs);
        };
        validate_dynamic_root(root, data.len())?;
        data
    } else {
        operands
    };
    if data.is_empty() {
        return Err(InferenceError::EmptyInputs);
    }
    for (index, ty) in data.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::CollectiveBroadcast.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    Ok(data.to_vec())
}

/// Infers collective-permute's unchanged operand type.
///
/// Source/target pair uniqueness and channels remain operation-verifier checks.
pub fn infer_collective_permute_type(operand: &Type) -> Result<Type, InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    if !StableHloOp::CollectivePermute.accepts_fixed_operand_type(0, operand) {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    Ok(operand.clone())
}

pub(crate) fn validate_dynamic_root(root: &Type, data_count: usize) -> Result<(), InferenceError> {
    root.verify().map_err(|error| InferenceError::InvalidType {
        index: data_count,
        error,
    })?;
    if !matches!(root, Type::Tensor(tensor) if tensor.element() == ElementType::I32
        && tensor.dimensions() == [Some(data_count as u64)])
    {
        return Err(InferenceError::IncompatibleTypes { index: data_count });
    }
    Ok(())
}
