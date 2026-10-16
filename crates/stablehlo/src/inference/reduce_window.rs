use super::{
    InferenceError, bound, bounds, bounds_encoding, dimensions, shape_witnesses::common_shape,
    window::output_shape, with_shape_and_encoding,
};
use crate::{StableHloOp, Type};

/// Infers windowed reduction results from scalar reducer return types.
///
/// Shapes and bounds are transformed independently for each corresponding
/// input, following `inferReduceWindowOp`. Elements come from reducer returns.
/// This validates input shapes and window geometry, not reducer bodies,
/// initialization values or accumulator promotion.
pub fn infer_reduce_window_types(
    inputs: &[Type],
    accumulators: &[Type],
    window_dimensions: &[i64],
    window_strides: Option<&[i64]>,
    base_dilations: Option<&[i64]>,
    window_dilations: Option<&[i64]>,
    padding: Option<&[[i64; 2]]>,
) -> Result<Vec<Type>, InferenceError> {
    if inputs.is_empty() {
        return Err(InferenceError::EmptyInputs);
    }
    if inputs.len() != accumulators.len() {
        return Err(InferenceError::AccumulatorCountMismatch {
            expected: inputs.len(),
            actual: accumulators.len(),
        });
    }
    for (index, ty) in inputs.iter().chain(accumulators).enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::ReduceWindow.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let shape = common_shape(&inputs.iter().collect::<Vec<_>>())?;
    let rank = shape.len();
    let window = window_dimensions
        .iter()
        .enumerate()
        .map(|(dimension, &value)| {
            if value <= 0 {
                return Err(InferenceError::NonPositiveWindowValue {
                    attribute: "window_dimensions",
                    dimension,
                    value,
                });
            }
            Ok(Some(value as u64))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let ones = vec![1; rank];
    let strides = window_strides.unwrap_or(&ones);
    let base = base_dilations.unwrap_or(&ones);
    let dilation = window_dilations.unwrap_or(&ones);
    let padding = padding.map_or_else(
        || vec![Some([0, 0]); rank],
        |padding| {
            padding
                .iter()
                .map(|p| Some([i128::from(p[0]), i128::from(p[1])]))
                .collect()
        },
    );
    inputs
        .iter()
        .zip(accumulators)
        .enumerate()
        .map(|(index, (input, accumulator))| {
            let accumulator_rank = dimensions(accumulator).expect("checked domain").len();
            if accumulator_rank != 0 {
                return Err(InferenceError::NonScalarAccumulator {
                    index,
                    rank: accumulator_rank,
                });
            }
            let sizes = output_shape(
                dimensions(input).expect("checked domain"),
                &window,
                strides,
                base,
                dilation,
                &padding,
            )?;
            let encoding = if bounds(input).is_some_and(|b| !b.is_empty()) {
                let input_bounds = (0..rank).map(|axis| bound(input, axis)).collect::<Vec<_>>();
                bounds_encoding(output_shape(
                    &input_bounds,
                    &window,
                    strides,
                    base,
                    dilation,
                    &padding,
                )?)
            } else {
                None
            };
            let result = with_shape_and_encoding(accumulator, sizes, encoding);
            result
                .verify()
                .map_err(|error| InferenceError::InvalidType {
                    index: inputs.len() + index,
                    error,
                })?;
            Ok(result)
        })
        .collect()
}
