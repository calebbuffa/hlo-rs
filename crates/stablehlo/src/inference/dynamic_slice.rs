use super::{InferenceError, bound, dimensions, with_shape_and_encoding};
use crate::{Type, verification::tensor_element_types_match};

/// Infers a dynamic slice from its operand, scalar index types and slice sizes.
///
/// The result has static dimensions and no encoding, following the pinned
/// `inferDynamicSliceOp`. A known upper bound also limits feasible slice sizes.
pub fn infer_dynamic_slice_type(
    operand: &Type,
    start_indices: &[Type],
    slice_sizes: &[i64],
) -> Result<Type, InferenceError> {
    let shape = tensor_shape(operand, 0)?;
    validate_indices(start_indices, shape.len(), 1)?;
    if slice_sizes.len() != shape.len() {
        return Err(InferenceError::AttributeLengthMismatch {
            attribute: "slice_sizes",
            expected: shape.len(),
            actual: slice_sizes.len(),
        });
    }
    let mut sizes = Vec::with_capacity(shape.len());
    for (dimension, &size) in slice_sizes.iter().enumerate() {
        let size = u64::try_from(size)
            .map_err(|_| InferenceError::InvalidSliceSize { dimension, size })?;
        if let Some(limit) = shape[dimension].or_else(|| bound(operand, dimension))
            && size > limit
        {
            return Err(InferenceError::SliceSizeExceeded {
                dimension,
                size,
                size_or_bound: limit,
            });
        }
        sizes.push(Some(size));
    }
    Ok(with_shape_and_encoding(operand, sizes, None))
}

/// Infers a dynamic update slice, retaining the operand's type and encoding.
///
/// Checks update shape feasibility and exact element metadata, including
/// per-tensor quantization, as well as homogeneous scalar integer index types.
pub fn infer_dynamic_update_slice_type(
    operand: &Type,
    update: &Type,
    start_indices: &[Type],
) -> Result<Type, InferenceError> {
    let shape = tensor_shape(operand, 0)?;
    let update_shape = tensor_shape(update, 1)?;
    if shape.len() != update_shape.len() {
        return Err(InferenceError::RankMismatch {
            expected: shape.len(),
            actual: update_shape.len(),
        });
    }
    validate_indices(start_indices, shape.len(), 2)?;
    if !tensor_element_types_match(operand, update) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    for (dimension, &size) in update_shape.iter().enumerate() {
        if let (Some(size), Some(limit)) =
            (size, shape[dimension].or_else(|| bound(operand, dimension)))
            && size > limit
        {
            return Err(InferenceError::SliceSizeExceeded {
                dimension,
                size,
                size_or_bound: limit,
            });
        }
    }
    Ok(operand.clone())
}

fn tensor_shape(ty: &Type, index: usize) -> Result<&[Option<u64>], InferenceError> {
    ty.verify()
        .map_err(|error| InferenceError::InvalidType { index, error })?;
    if matches!(ty, Type::QuantizedTensor(tensor) if tensor.quantization_dimension().is_some()) {
        return Err(InferenceError::IncompatibleTypes { index });
    }
    dimensions(ty).ok_or(InferenceError::IncompatibleTypes { index })
}

fn validate_indices(indices: &[Type], rank: usize, offset: usize) -> Result<(), InferenceError> {
    if indices.len() != rank {
        return Err(InferenceError::AttributeLengthMismatch {
            attribute: "start_indices",
            expected: rank,
            actual: indices.len(),
        });
    }
    for (index, ty) in indices.iter().enumerate() {
        let index = offset + index;
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !crate::verification::is_scalar_integer(ty)
            || !tensor_element_types_match(&indices[0], ty)
        {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    Ok(())
}
