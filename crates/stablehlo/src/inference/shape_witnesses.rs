use super::{InferenceError, Specificity, bound, dimensions, merge_dimension};
use crate::Type;

pub(crate) fn common_dimension(
    witnesses: &[(&Type, usize)],
) -> Result<Option<u64>, InferenceError> {
    if witnesses.is_empty() {
        return Err(InferenceError::EmptyInputs);
    }
    let mut size = None;
    let mut upper = None;
    for (index, &(ty, axis)) in witnesses.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        let shape = dimensions(ty).ok_or(InferenceError::IncompatibleTypes { index })?;
        let &value = shape
            .get(axis)
            .ok_or(InferenceError::DimensionOutOfBounds {
                dimension: axis,
                rank: shape.len(),
            })?;
        (size, upper) =
            merge_dimension(axis, size, value, upper, bound(ty, axis), Specificity::Most)?;
    }
    Ok(size)
}

/// Checks equal shapes independently of element types, retaining all bounds.
pub(crate) fn common_shape(types: &[&Type]) -> Result<Vec<Option<u64>>, InferenceError> {
    Ok(common_witness(types)?.sizes)
}

pub(super) struct ShapeWitness {
    pub sizes: Vec<Option<u64>>,
    pub bounds: Vec<Option<u64>>,
}

pub(super) fn common_witness(types: &[&Type]) -> Result<ShapeWitness, InferenceError> {
    let Some(first) = types.first() else {
        return Err(InferenceError::EmptyInputs);
    };
    for (index, ty) in types.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
    }
    let shape = dimensions(first).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    let rank = shape.len();
    let mut sizes = shape.to_vec();
    let mut upper = (0..rank).map(|axis| bound(first, axis)).collect::<Vec<_>>();
    for (index, ty) in types.iter().enumerate().skip(1) {
        let shape = dimensions(ty).ok_or(InferenceError::IncompatibleTypes { index })?;
        if shape.len() != rank {
            return Err(InferenceError::RankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
        for axis in 0..rank {
            (sizes[axis], upper[axis]) = merge_dimension(
                axis,
                sizes[axis],
                shape[axis],
                upper[axis],
                bound(ty, axis),
                Specificity::Most,
            )?;
        }
    }
    Ok(ShapeWitness {
        sizes,
        bounds: upper,
    })
}
