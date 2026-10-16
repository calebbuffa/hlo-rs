use super::{
    InferenceError, Specificity, bound, bounds, bounds_encoding, dimensions, encoding,
    merge_dimension, with_shape_and_encoding,
};
use crate::{Type, verification::tensor_element_types_match};

/// Infers a concatenate result, including static sizes and dynamic bounds.
///
/// Follows `TypeInference.cpp::inferConcatenateOp` and
/// `Base.cpp::inferConcatenatedDimAndBound` from the pinned reference.
/// Element types (including quantization parameters) must match exactly.
/// Size and bound sums must fit the MLIR signed 64-bit dimension domain.
pub fn infer_concatenate_type(types: &[Type], dimension: usize) -> Result<Type, InferenceError> {
    let first = types.first().ok_or(InferenceError::EmptyInputs)?;
    for (index, ty) in types.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
    }
    let first_shape = dimensions(first).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    let rank = first_shape.len();
    if dimension >= rank {
        return Err(InferenceError::DimensionOutOfBounds { dimension, rank });
    }
    let mut sizes = vec![None; rank];
    let mut output_bounds = vec![None; rank];
    sizes[dimension] = Some(0);
    for (index, ty) in types.iter().enumerate() {
        if !tensor_element_types_match(first, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
        let shape = dimensions(ty).ok_or(InferenceError::IncompatibleTypes { index })?;
        if shape.len() != rank {
            return Err(InferenceError::RankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
        for axis in 0..rank {
            let (size, upper) = if axis == dimension {
                concatenate_dimension(
                    axis,
                    sizes[axis],
                    shape[axis],
                    output_bounds[axis],
                    bound(ty, axis),
                )?
            } else {
                merge_dimension(
                    axis,
                    sizes[axis],
                    shape[axis],
                    output_bounds[axis],
                    bound(ty, axis),
                    Specificity::Most,
                )?
            };
            sizes[axis] = size;
            output_bounds[axis] = upper;
        }
    }
    let encoding = if types.iter().any(|ty| bounds(ty).is_some()) {
        bounds_encoding(output_bounds)
    } else {
        types.iter().find_map(encoding).cloned()
    };
    let result = with_shape_and_encoding(first, sizes, encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}

fn concatenate_dimension(
    dimension: usize,
    lhs: Option<u64>,
    rhs: Option<u64>,
    left_bound: Option<u64>,
    right_bound: Option<u64>,
) -> Result<(Option<u64>, Option<u64>), InferenceError> {
    let add = |lhs: u64, rhs: u64| {
        lhs.checked_add(rhs)
            .filter(|sum| *sum <= i64::MAX as u64)
            .ok_or(InferenceError::DimensionOverflow { dimension })
    };
    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
        Ok((Some(add(lhs, rhs)?), None))
    } else if let (Some(lhs), Some(rhs)) = (lhs.or(left_bound), rhs.or(right_bound)) {
        Ok((None, Some(add(lhs, rhs)?)))
    } else {
        Ok((None, None))
    }
}
