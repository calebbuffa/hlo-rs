use super::{
    InferenceError, bound, bounds, bounds_encoding, dimensions, encoding, infer_all_reduce_types,
    shape_witnesses::common_dimension, with_shape_and_encoding,
};
use crate::{StableHloOp, Type};

/// Infers all-gather shapes for an explicitly known process-group size.
///
/// Implements all_gather C6, retaining each input's elements and non-bounds
/// encoding. Dynamic bounds are multiplied, constrained by the int64 extent
/// domain. Groups/channels are checked separately by operation verification.
pub fn infer_all_gather_types(
    operands: &[Type],
    dimension: i64,
    group_size: i64,
) -> Result<Vec<Type>, InferenceError> {
    let count = checked_group_size(group_size)?;
    if dimension < 0 {
        return Err(InferenceError::InvalidDimensions {
            rank: operands.first().and_then(dimensions).map_or(0, <[_]>::len),
            dimensions: vec![dimension],
        });
    }
    operands
        .iter()
        .enumerate()
        .map(|(index, operand)| {
            checked_tensor(StableHloOp::AllGather, operand, index)?;
            transform(operand, dimension, count, true)
        })
        .collect()
}

/// Infers reduce-scatter shape/bounds and scalar reducer-return elements.
///
/// Implements reduce_scatter C1, C8-C9. Dynamic bounds divide down to the
/// largest attainable output extent. Reducer argument/promotion contracts and
/// process-group/channel metadata remain operation-verifier responsibilities.
pub fn infer_reduce_scatter_type(
    operand: &Type,
    accumulator: &Type,
    dimension: i64,
    group_size: i64,
) -> Result<Type, InferenceError> {
    let count = checked_group_size(group_size)?;
    let inferred = infer_all_reduce_types(std::slice::from_ref(operand), accumulator)?;
    transform(&inferred[0], dimension, count, false)
}

fn checked_group_size(size: i64) -> Result<u64, InferenceError> {
    u64::try_from(size)
        .ok()
        .filter(|&size| size != 0)
        .ok_or(InferenceError::InvalidGroupSize { size })
}

fn checked_tensor(opcode: StableHloOp, operand: &Type, index: usize) -> Result<(), InferenceError> {
    operand
        .verify()
        .map_err(|error| InferenceError::InvalidType { index, error })?;
    if !opcode.accepts_variadic_operand_type(0, operand) {
        return Err(InferenceError::IncompatibleTypes { index });
    }
    Ok(())
}

fn axis(ty: &Type, dimension: i64) -> Result<usize, InferenceError> {
    let rank = dimensions(ty)
        .ok_or(InferenceError::IncompatibleTypes { index: 0 })?
        .len();
    usize::try_from(dimension)
        .ok()
        .filter(|&axis| axis < rank)
        .ok_or_else(|| InferenceError::InvalidDimensions {
            rank,
            dimensions: vec![dimension],
        })
}

fn transform(
    operand: &Type,
    dimension: i64,
    count: u64,
    gather: bool,
) -> Result<Type, InferenceError> {
    let axis = axis(operand, dimension)?;
    let mut shape = dimensions(operand)
        .expect("checked ranked operand")
        .to_vec();
    let (minimum, maximum) = extent_interval(operand, axis)?;
    if gather {
        if minimum > i64::MAX as u64 / count {
            return Err(InferenceError::DimensionOverflow { dimension: axis });
        }
        shape[axis] = shape[axis].map(|size| size * count);
    } else {
        if maximum < count {
            return Err(InferenceError::InvalidCollectiveExtent { dimension: axis });
        }
        if let Some(size) = shape[axis] {
            if !size.is_multiple_of(count) {
                return Err(InferenceError::NonDivisibleSplit {
                    dimension: axis,
                    size,
                    count,
                });
            }
            shape[axis] = Some(size / count);
        }
    }
    let result_encoding = if bounds(operand).is_some_and(|bounds| !bounds.is_empty()) {
        let mut output_bounds = (0..shape.len())
            .map(|axis| bound(operand, axis))
            .collect::<Vec<_>>();
        output_bounds[axis] = output_bounds[axis].map(|upper| {
            if gather {
                upper.min(i64::MAX as u64 / count) * count
            } else {
                upper / count
            }
        });
        bounds_encoding(output_bounds)
    } else {
        encoding(operand).cloned()
    };
    let result = with_shape_and_encoding(operand, shape, result_encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}

fn extent_interval(ty: &Type, axis: usize) -> Result<(u64, u64), InferenceError> {
    let shape = dimensions(ty).expect("checked tensor");
    let low = shape[axis].unwrap_or(1);
    let high = shape[axis]
        .or_else(|| bound(ty, axis))
        .unwrap_or(i64::MAX as u64);
    if low == 0 || high < low {
        return Err(InferenceError::InvalidCollectiveExtent { dimension: axis });
    }
    Ok((low, high))
}

/// Checks `larger[axis] = smaller[axis] * group_size` for a nonzero extent.
///
/// An unresolved group size means a positive integer factor, not permission
/// to ignore bounds, prefix shape or static divisibility.
pub(crate) fn validate_exchange_shape(
    smaller: &Type,
    larger: &Type,
    dimension: i64,
    group_size: Option<u64>,
) -> Result<(), InferenceError> {
    let axis = axis(smaller, dimension)?;
    let small_shape = dimensions(smaller).expect("checked tensor");
    let large_shape = dimensions(larger).ok_or(InferenceError::IncompatibleTypes { index: 1 })?;
    if small_shape.len() != large_shape.len() {
        return Err(InferenceError::RankMismatch {
            expected: small_shape.len(),
            actual: large_shape.len(),
        });
    }
    for i in 0..small_shape.len() {
        if i != axis {
            common_dimension(&[(smaller, i), (larger, i)])?;
        }
    }
    let (small_low, small_high) = extent_interval(smaller, axis)?;
    let (large_low, large_high) = extent_interval(larger, axis)?;
    let possible = match group_size {
        Some(0) => false,
        Some(count) => {
            small_low.max(large_low.div_ceil(count)) <= small_high.min(large_high / count)
        }
        None if small_low == small_high => {
            large_low.div_ceil(small_low).max(1) <= large_high / small_low
        }
        None => true,
    };
    if !possible {
        return Err(InferenceError::InvalidCollectiveExtent { dimension: axis });
    }
    Ok(())
}
