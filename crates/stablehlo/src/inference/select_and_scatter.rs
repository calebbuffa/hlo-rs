use super::{
    InferenceError, bound, dimensions, encoding,
    shape_witnesses::common_witness,
    window::{output_extent, output_shape},
    with_shape_and_encoding,
};
use crate::{StableHloOp, Type};

/// Infers select-and-scatter results with the input's shape and encoding and
/// the scatter computation's scalar return elements (`inferSelectAndScatterOp`).
///
/// Source-window geometry and bounds are checked. Select/scatter region bodies,
/// initialization values and accumulator promotion remain verifier checks.
pub fn infer_select_and_scatter_type(
    operand: &Type,
    source: &Type,
    accumulator: &Type,
    window_dimensions: &[i64],
    window_strides: Option<&[i64]>,
    padding: Option<&[[i64; 2]]>,
) -> Result<Type, InferenceError> {
    validate(
        &[operand],
        source,
        window_dimensions,
        window_strides,
        padding,
    )?;
    if !crate::verification::tensor_element_types_match(operand, source) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    accumulator
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 2, error })?;
    if !StableHloOp::SelectAndScatter.accepts_fixed_result_type(0, accumulator) {
        return Err(InferenceError::IncompatibleTypes { index: 2 });
    }
    let shape = dimensions(accumulator).expect("checked domain");
    if !shape.is_empty() {
        return Err(InferenceError::NonScalarAccumulator {
            index: 0,
            rank: shape.len(),
        });
    }
    let result = with_shape_and_encoding(
        accumulator,
        dimensions(operand).expect("checked operand").to_vec(),
        encoding(operand).cloned(),
    );
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 2, error })?;
    Ok(result)
}

pub(crate) fn validate(
    input_witnesses: &[&Type],
    source: &Type,
    window_dimensions: &[i64],
    window_strides: Option<&[i64]>,
    padding: Option<&[[i64; 2]]>,
) -> Result<(), InferenceError> {
    for (index, ty) in input_witnesses.iter().copied().chain([source]).enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::SelectAndScatter.accepts_fixed_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let input = common_witness(input_witnesses)?;
    let rank = input.sizes.len();
    let source_shape = dimensions(source).expect("checked source");
    if source_shape.len() != rank {
        return Err(InferenceError::RankMismatch {
            expected: rank,
            actual: source_shape.len(),
        });
    }
    let window = window_dimensions
        .iter()
        .enumerate()
        .map(|(dimension, &value)| {
            if value <= 0 {
                Err(InferenceError::NonPositiveWindowValue {
                    attribute: "window_dimensions",
                    dimension,
                    value,
                })
            } else {
                Ok(Some(value as u64))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ones = vec![1; rank];
    let strides = window_strides.unwrap_or(&ones);
    let padding = padding.map_or_else(
        || vec![Some([0, 0]); rank],
        |values| {
            values
                .iter()
                .map(|p| Some([i128::from(p[0]), i128::from(p[1])]))
                .collect()
        },
    );
    output_shape(&input.sizes, &window, strides, &ones, &ones, &padding)?;
    for axis in 0..rank {
        let extent = |size| {
            output_extent(
                size,
                window[axis].expect("positive window"),
                strides[axis],
                1,
                1,
                padding[axis].expect("known padding"),
            )
        };
        let minimum = extent(input.sizes[axis].unwrap_or(0));
        let maximum = extent(
            input.sizes[axis]
                .or(input.bounds[axis])
                .unwrap_or(i64::MAX as u64),
        );
        let source_minimum = i128::from(source_shape[axis].unwrap_or(0));
        let source_maximum = i128::from(
            source_shape[axis]
                .or(bound(source, axis))
                .unwrap_or(i64::MAX as u64),
        );
        if source_minimum > maximum || minimum > source_maximum {
            return Err(InferenceError::InvalidWindowSource { dimension: axis });
        }
    }
    Ok(())
}
