use super::{InferenceError, bound, bounds_encoding, dimensions, with_shape_and_encoding};
use crate::{ElementType, TensorType, Type};

/// Infers the scalar i32 result of the pinned `inferGetDimensionSizeOp`.
pub fn infer_get_dimension_size_type(
    operand: &Type,
    dimension: usize,
) -> Result<Type, InferenceError> {
    checked_shape(operand, dimension)?;
    Ok(TensorType::scalar(ElementType::I32).into())
}

/// Infers `set_dimension_size` dimensions and bounds.
///
/// Follows the pinned `inferSetDimensionSizeOp`: a static input extent becomes
/// a bound, the selected dimension becomes dynamic, and a constant equal to
/// that bound restores a static dimension. Other constants do not refine the
/// result. Unrelated encodings are discarded. This is type inference, not
/// runtime validation of the size value.
pub fn infer_set_dimension_size_type(
    operand: &Type,
    size: &Type,
    dimension: usize,
    constant_size: Option<i32>,
) -> Result<Type, InferenceError> {
    let shape = checked_shape(operand, dimension)?;
    if matches!(operand, Type::QuantizedTensor(tensor) if tensor.quantization_dimension().is_some())
    {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    size.verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    if !matches!(size, Type::Tensor(tensor) if tensor.rank() == 0 && tensor.element() == ElementType::I32)
    {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let mut output_bounds = (0..shape.len())
        .map(|axis| bound(operand, axis))
        .collect::<Vec<_>>();
    let mut sizes = shape.to_vec();
    output_bounds[dimension] = shape[dimension].or(output_bounds[dimension]);
    sizes[dimension] = None;
    if let Some(size) = constant_size.and_then(|size| u64::try_from(size).ok())
        && output_bounds[dimension] == Some(size)
    {
        sizes[dimension] = Some(size);
        output_bounds[dimension] = None;
    }
    let result = with_shape_and_encoding(operand, sizes, bounds_encoding(output_bounds));
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}

fn checked_shape(operand: &Type, dimension: usize) -> Result<&[Option<u64>], InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    let shape = dimensions(operand).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    if dimension >= shape.len() {
        return Err(InferenceError::DimensionOutOfBounds {
            dimension,
            rank: shape.len(),
        });
    }
    Ok(shape)
}
