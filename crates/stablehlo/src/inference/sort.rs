use super::{InferenceError, shape_witnesses::common_shape};
use crate::{StableHloOp, Type};

/// Infers sort results in operand order without refining their metadata.
///
/// Matches `inferSortOp`: every result retains its corresponding input's
/// shape, elements and encoding. Checks the common shape and signed dimension
/// range required by `verifySortOp`; comparator-region validation remains
/// part of operation verification.
pub fn infer_sort_types(inputs: &[Type], dimension: i64) -> Result<Vec<Type>, InferenceError> {
    validate(&inputs.iter().collect::<Vec<_>>(), dimension)?;
    Ok(inputs.to_vec())
}

pub(crate) fn validate(inputs: &[&Type], dimension: i64) -> Result<(), InferenceError> {
    for (index, ty) in inputs.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::Sort.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let shape = common_shape(inputs)?;
    let rank = shape.len();
    let signed_rank = rank as i128;
    if i128::from(dimension) < -signed_rank || i128::from(dimension) >= signed_rank {
        return Err(InferenceError::InvalidSortDimension { dimension, rank });
    }
    Ok(())
}
