use super::{InferenceError, bound, bounds, dimensions, encoding, with_shape_and_encoding};
use crate::Type;

/// Infers a statically indexed slice using the pinned `inferSliceOp` rules.
///
/// Each range must fit the operand's static size or dynamic upper bound.
/// Result dimensions are static ceil-divisions of the ranges. Bounds encodings
/// are removed once all dimensions are static; unrelated encodings are retained.
pub fn infer_slice_type(
    operand: &Type,
    start_indices: &[i64],
    limit_indices: &[i64],
    strides: &[i64],
) -> Result<Type, InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    let shape = dimensions(operand).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    for (attribute, values) in [
        ("start_indices", start_indices),
        ("limit_indices", limit_indices),
        ("strides", strides),
    ] {
        if values.len() != shape.len() {
            return Err(InferenceError::AttributeLengthMismatch {
                attribute,
                expected: shape.len(),
                actual: values.len(),
            });
        }
    }
    let mut sizes = Vec::with_capacity(shape.len());
    for (dimension, ((&start, &limit), &stride)) in start_indices
        .iter()
        .zip(limit_indices)
        .zip(strides)
        .enumerate()
    {
        if start < 0 || limit < start || stride <= 0 {
            return Err(InferenceError::InvalidSliceRange {
                dimension,
                start,
                limit,
                stride,
            });
        }
        if let Some(size_or_bound) = shape[dimension].or_else(|| bound(operand, dimension))
            && limit as u64 > size_or_bound
        {
            return Err(InferenceError::SliceLimitExceeded {
                dimension,
                limit,
                size_or_bound,
            });
        }
        sizes.push(Some(((limit - start) as u64).div_ceil(stride as u64)));
    }
    let encoding = if bounds(operand).is_some_and(|bounds| !bounds.is_empty()) {
        None
    } else {
        encoding(operand).cloned()
    };
    let result = with_shape_and_encoding(operand, sizes, encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}
