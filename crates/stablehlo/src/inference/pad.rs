use super::{
    InferenceError, bound, bounds, bounds_encoding, dimensions, encoding, with_shape_and_encoding,
};
use crate::{Type, verification::tensor_element_types_match};

/// Infers padding sizes and bounds using the pinned `inferPadOp` rules.
///
/// Negative edge padding crops dimensions; interior padding must be nonnegative.
/// The padding value must be a scalar with identical element metadata. Signed
/// 64-bit size and bound overflow is reported rather than wrapped.
pub fn infer_pad_type(
    operand: &Type,
    padding_value: &Type,
    edge_padding_low: &[i64],
    edge_padding_high: &[i64],
    interior_padding: &[i64],
) -> Result<Type, InferenceError> {
    for (index, ty) in [operand, padding_value].into_iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
    }
    let shape = dimensions(operand).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    if dimensions(padding_value) != Some(&[][..])
        || !tensor_element_types_match(operand, padding_value)
    {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    for (attribute, values) in [
        ("edge_padding_low", edge_padding_low),
        ("edge_padding_high", edge_padding_high),
        ("interior_padding", interior_padding),
    ] {
        if values.len() != shape.len() {
            return Err(InferenceError::AttributeLengthMismatch {
                attribute,
                expected: shape.len(),
                actual: values.len(),
            });
        }
    }
    let mut sizes = vec![None; shape.len()];
    let mut output_bounds = vec![None; shape.len()];
    for dimension in 0..shape.len() {
        let interior = interior_padding[dimension];
        if interior < 0 {
            return Err(InferenceError::NegativeInteriorPadding {
                dimension,
                value: interior,
            });
        }
        let Some(size_or_bound) = shape[dimension].or_else(|| bound(operand, dimension)) else {
            continue;
        };
        let size = i128::from(size_or_bound);
        let padded = size
            + i128::from(edge_padding_low[dimension])
            + i128::from(edge_padding_high[dimension])
            + (size - 1).max(0) * i128::from(interior);
        if padded < 0 {
            return Err(InferenceError::NegativePaddingResult { dimension });
        }
        if padded > i128::from(i64::MAX) {
            return Err(InferenceError::DimensionOverflow { dimension });
        }
        if shape[dimension].is_some() {
            sizes[dimension] = Some(padded as u64);
        } else {
            output_bounds[dimension] = Some(padded as u64);
        }
    }
    let encoding = if bounds(operand).is_some_and(|bounds| !bounds.is_empty()) {
        bounds_encoding(output_bounds)
    } else {
        encoding(operand).cloned()
    };
    let result = with_shape_and_encoding(operand, sizes, encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}
