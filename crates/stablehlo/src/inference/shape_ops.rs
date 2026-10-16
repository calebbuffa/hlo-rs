use super::{InferenceError, dimensions, with_shape_and_encoding};
use crate::Type;

/// Infers a reverse result, retaining the operand's type, bounds and encoding.
///
/// Validates the unique in-range axes required by `verifyReverseOp`, then
/// follows `inferReverseOp`'s identity type rule.
pub fn infer_reverse_type(operand: &Type, axes: &[i64]) -> Result<Type, InferenceError> {
    let shape = tensor_shape(operand)?;
    let mut seen = vec![false; shape.len()];
    for &axis in axes {
        let entry = usize::try_from(axis)
            .ok()
            .and_then(|axis| seen.get_mut(axis))
            .ok_or_else(|| InferenceError::InvalidDimensions {
                rank: shape.len(),
                dimensions: axes.to_vec(),
            })?;
        if std::mem::replace(entry, true) {
            return Err(InferenceError::InvalidDimensions {
                rank: shape.len(),
                dimensions: axes.to_vec(),
            });
        }
    }
    Ok(operand.clone())
}

/// Infers the legacy leading-axis broadcast shape.
///
/// Follows the pinned `inferBroadcastOp`, which prepends nonnegative static
/// extents and drops the input encoding rather than extending its bounds.
pub fn infer_broadcast_type(
    operand: &Type,
    broadcast_sizes: &[i64],
) -> Result<Type, InferenceError> {
    let shape = tensor_shape(operand)?;
    let mut sizes = broadcast_sizes
        .iter()
        .enumerate()
        .map(|(dimension, &size)| {
            u64::try_from(size)
                .map(Some)
                .map_err(|_| InferenceError::InvalidBroadcastSize { dimension, size })
        })
        .collect::<Result<Vec<_>, _>>()?;
    sizes.extend_from_slice(shape);
    Ok(with_shape_and_encoding(operand, sizes, None))
}

fn tensor_shape(operand: &Type) -> Result<&[Option<u64>], InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    if matches!(operand, Type::QuantizedTensor(tensor) if tensor.quantization_dimension().is_some())
    {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    dimensions(operand).ok_or(InferenceError::IncompatibleTypes { index: 0 })
}
