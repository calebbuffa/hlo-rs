use super::{
    InferenceError, bound, bounds, bounds_encoding, dimensions, encoding, with_shape_and_encoding,
};
use crate::{QuantizedTensorType, Type};

/// Infers transposed dimensions and bounds from an exact axis permutation.
///
/// Follows the pinned `inferTransposeOp` shape/bounds rule and `transpose_c4`
/// for per-axis quantization: the output quantization axis is the position of
/// the input quantization axis in the permutation.
pub fn infer_transpose_type(operand: &Type, permutation: &[i64]) -> Result<Type, InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    let shape = dimensions(operand).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    let invalid = || InferenceError::InvalidPermutation {
        rank: shape.len(),
        permutation: permutation.to_vec(),
    };
    if permutation.len() != shape.len() {
        return Err(invalid());
    }
    let mut seen = vec![false; shape.len()];
    let mut axes = Vec::with_capacity(shape.len());
    for &axis in permutation {
        let axis = usize::try_from(axis).map_err(|_| invalid())?;
        let slot = seen.get_mut(axis).ok_or_else(invalid)?;
        if std::mem::replace(slot, true) {
            return Err(invalid());
        }
        axes.push(axis);
    }
    let sizes = axes.iter().map(|&axis| shape[axis]).collect::<Vec<_>>();
    let encoding = if bounds(operand).is_some_and(|bounds| !bounds.is_empty()) {
        bounds_encoding(axes.iter().map(|&axis| bound(operand, axis)).collect())
    } else {
        encoding(operand).cloned()
    };
    let result = if let Type::QuantizedTensor(tensor) = operand
        && let Some(input_axis) = tensor.quantization_dimension()
    {
        let (min, max) = tensor.storage_range();
        let mut tensor = QuantizedTensorType::new(
            sizes,
            tensor.storage_type(),
            min,
            max,
            tensor.expressed_type(),
            axes.iter().position(|&axis| axis == input_axis),
            tensor.scales().collect(),
            tensor.zero_points().to_vec(),
        );
        if let Some(encoding) = encoding {
            tensor = tensor.with_encoding(encoding);
        }
        Type::QuantizedTensor(tensor)
    } else {
        with_shape_and_encoding(operand, sizes, encoding)
    };
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}
