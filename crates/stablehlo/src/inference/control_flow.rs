use super::{InferenceError, infer_least_specific_type};
use crate::{StableHloOp, Type};

/// Infers tuple construction, retaining every operand type and nested metadata.
pub fn infer_tuple_type(operands: &[Type]) -> Result<Type, InferenceError> {
    check_operands(StableHloOp::Tuple, operands)?;
    Ok(Type::Tuple(operands.to_vec()))
}

/// Infers tuple projection. Negative and out-of-range i32 indices are errors.
pub fn infer_get_tuple_element_type(operand: &Type, index: i32) -> Result<Type, InferenceError> {
    check_fixed_operand(StableHloOp::GetTupleElement, operand)?;
    let Type::Tuple(elements) = operand else {
        unreachable!("checked tuple operand")
    };
    usize::try_from(index)
        .ok()
        .and_then(|index| elements.get(index))
        .cloned()
        .ok_or(InferenceError::TupleIndexOutOfBounds {
            index,
            size: elements.len(),
        })
}

/// Infers optimization-barrier results without changing operand metadata.
pub fn infer_optimization_barrier_types(operands: &[Type]) -> Result<Vec<Type>, InferenceError> {
    check_operands(StableHloOp::OptimizationBarrier, operands)?;
    Ok(operands.to_vec())
}

/// Infers while results from operands, without refining them from the body.
///
/// Condition/body argument and return contracts remain region-verifier checks.
pub fn infer_while_types(operands: &[Type]) -> Result<Vec<Type>, InferenceError> {
    check_operands(StableHloOp::While, operands)?;
    Ok(operands.to_vec())
}

/// Joins the two branches of an if operation using the reference's
/// least-specific-type rule. Branch slices contain terminator operand types;
/// the region verifier separately checks zero arguments and valid terminators.
pub fn infer_if_types(
    predicate: &Type,
    true_results: &[Type],
    false_results: &[Type],
) -> Result<Vec<Type>, InferenceError> {
    infer_conditional_types(StableHloOp::If, predicate, &[true_results, false_results])
}

/// Joins a nonempty case branch list, retaining first-branch element metadata.
///
/// Branch slices contain terminator operand types. The selector must be a
/// scalar i32 tensor. Region structure remains a verifier responsibility.
pub fn infer_case_types(index: &Type, branches: &[Vec<Type>]) -> Result<Vec<Type>, InferenceError> {
    infer_conditional_types(
        StableHloOp::Case,
        index,
        &branches.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
}

pub(crate) fn infer_conditional_types(
    opcode: StableHloOp,
    selector: &Type,
    branches: &[&[Type]],
) -> Result<Vec<Type>, InferenceError> {
    check_fixed_operand(opcode, selector)?;
    if !matches!(selector, Type::Tensor(tensor) if tensor.rank() == 0) {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    let Some(first) = branches.first() else {
        return Err(InferenceError::EmptyBranches);
    };
    for (branch, results) in branches.iter().enumerate() {
        if results.len() != first.len() {
            return Err(InferenceError::BranchResultCountMismatch {
                branch,
                expected: first.len(),
                actual: results.len(),
            });
        }
        for (index, ty) in results.iter().enumerate() {
            ty.verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            if !opcode.accepts_variadic_result_type(0, ty) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
            if !crate::verification::compatible_value_types(&first[index], ty) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
        }
    }
    (0..first.len())
        .map(|index| {
            infer_least_specific_type(
                &branches
                    .iter()
                    .map(|types| types[index].clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn check_fixed_operand(opcode: StableHloOp, ty: &Type) -> Result<(), InferenceError> {
    ty.verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    if !opcode.accepts_fixed_operand_type(0, ty) {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    Ok(())
}

fn check_operands(opcode: StableHloOp, operands: &[Type]) -> Result<(), InferenceError> {
    for (index, ty) in operands.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !opcode.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    Ok(())
}
