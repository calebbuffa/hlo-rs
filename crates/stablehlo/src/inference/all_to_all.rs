use super::{
    InferenceError, bound, bounds, bounds_encoding, dimensions, encoding, with_shape_and_encoding,
};
use crate::{StableHloOp, Type};

/// Infers each all-to-all result's split/concatenated dimensions and bounds.
///
/// Follows `inferAllToAllOp`, retaining each operand's own elements and encoding.
/// Replica groups and channel constraints are checked by operation verification.
pub fn infer_all_to_all_types(
    operands: &[Type],
    split_dimension: i64,
    concat_dimension: i64,
    split_count: i64,
) -> Result<Vec<Type>, InferenceError> {
    if split_count <= 0 {
        return Err(InferenceError::InvalidSplitCount { count: split_count });
    }
    if split_dimension < 0 || concat_dimension < 0 {
        return Err(InferenceError::InvalidDimensions {
            rank: operands.first().and_then(dimensions).map_or(0, <[_]>::len),
            dimensions: vec![split_dimension, concat_dimension],
        });
    }
    let count = split_count as u64;
    operands
        .iter()
        .enumerate()
        .map(|(index, operand)| {
            operand
                .verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            if !StableHloOp::AllToAll.accepts_variadic_operand_type(0, operand) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
            let shape = dimensions(operand).expect("checked ranked tensor domain");
            let axis = |value| {
                usize::try_from(value)
                    .ok()
                    .filter(|&axis| axis < shape.len())
                    .ok_or_else(|| InferenceError::InvalidDimensions {
                        rank: shape.len(),
                        dimensions: vec![split_dimension, concat_dimension],
                    })
            };
            let split = axis(split_dimension)?;
            let concat = axis(concat_dimension)?;
            if let Some(size) = shape[split]
                && !size.is_multiple_of(count)
            {
                return Err(InferenceError::NonDivisibleSplit {
                    dimension: split,
                    size,
                    count,
                });
            }
            let transform = |mut values: Vec<Option<u64>>| -> Result<_, InferenceError> {
                values[split] = values[split].map(|value| value / count);
                values[concat] = values[concat]
                    .map(|value| {
                        value
                            .checked_mul(count)
                            .filter(|&value| value <= i64::MAX as u64)
                            .ok_or(InferenceError::DimensionOverflow { dimension: concat })
                    })
                    .transpose()?;
                Ok(values)
            };
            let result_shape = transform(shape.to_vec())?;
            let encoding = if bounds(operand).is_some_and(|bounds| !bounds.is_empty()) {
                bounds_encoding(transform(
                    (0..shape.len()).map(|axis| bound(operand, axis)).collect(),
                )?)
            } else {
                encoding(operand).cloned()
            };
            let result = with_shape_and_encoding(operand, result_shape, encoding);
            result
                .verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            Ok(result)
        })
        .collect()
}
