use super::{
    InferenceError, Specificity, bound, dimensions, encoding, merge_dimension,
    shape_witnesses::common_witness, with_shape_and_encoding,
};
use crate::{ScatterDimensionNumbers, StableHloOp, Type};

/// Infers scatter results from input geometry and scalar update-computation returns.
///
/// Each result retains its corresponding input's dimensions and encoding, with
/// elements selected by the corresponding scalar return, as in `inferScatterOp`.
/// Dimension maps, update windows, batching and common shape/bound witnesses are
/// validated without refining output metadata. Region bodies, scalar argument
/// compatibility and accumulator promotion remain verifier responsibilities.
pub fn infer_scatter_types(
    inputs: &[Type],
    scatter_indices: &Type,
    updates: &[Type],
    dimension_numbers: &ScatterDimensionNumbers,
    accumulators: &[Type],
) -> Result<Vec<Type>, InferenceError> {
    if inputs.len() != updates.len() {
        return Err(InferenceError::UpdateCountMismatch {
            expected: inputs.len(),
            actual: updates.len(),
        });
    }
    if inputs.len() != accumulators.len() {
        return Err(InferenceError::AccumulatorCountMismatch {
            expected: inputs.len(),
            actual: accumulators.len(),
        });
    }
    validate(
        &inputs.iter().collect::<Vec<_>>(),
        scatter_indices,
        &updates.iter().collect::<Vec<_>>(),
        dimension_numbers,
    )?;
    for (index, (input, update)) in inputs.iter().zip(updates).enumerate() {
        if !crate::verification::tensor_element_types_match(input, update) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    inputs
        .iter()
        .zip(accumulators)
        .enumerate()
        .map(|(index, (input, accumulator))| {
            accumulator
                .verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            if !StableHloOp::Scatter.accepts_variadic_operand_type(0, accumulator) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
            let shape = dimensions(accumulator).expect("checked scalar domain");
            if !shape.is_empty() {
                return Err(InferenceError::NonScalarAccumulator {
                    index,
                    rank: shape.len(),
                });
            }
            let result = with_shape_and_encoding(
                accumulator,
                dimensions(input).expect("checked input").to_vec(),
                encoding(input).cloned(),
            );
            result
                .verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            Ok(result)
        })
        .collect()
}

fn axes(values: &[i64], rank: usize, sorted: bool) -> Result<Vec<usize>, InferenceError> {
    let invalid = || InferenceError::InvalidScatterDimensions;
    let mut seen = vec![false; rank];
    let mut axes = Vec::with_capacity(values.len());
    for &value in values {
        let axis = usize::try_from(value).map_err(|_| invalid())?;
        let entry = seen.get_mut(axis).ok_or_else(invalid)?;
        if std::mem::replace(entry, true) {
            return Err(invalid());
        }
        axes.push(axis);
    }
    if sorted && !axes.is_sorted() {
        return Err(invalid());
    }
    Ok(axes)
}

pub(crate) fn validate(
    input_witnesses: &[&Type],
    indices: &Type,
    updates: &[&Type],
    d: &ScatterDimensionNumbers,
) -> Result<(), InferenceError> {
    for (index, ty) in input_witnesses.iter().chain(updates).enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::Scatter.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    indices
        .verify()
        .map_err(|error| InferenceError::InvalidType {
            index: input_witnesses.len(),
            error,
        })?;
    if !StableHloOp::Scatter.accepts_fixed_operand_type(1, indices) {
        return Err(InferenceError::IncompatibleTypes {
            index: input_witnesses.len(),
        });
    }
    let input = common_witness(input_witnesses)?;
    let mut update = common_witness(updates)?;
    let index = dimensions(indices).expect("checked index tensor");
    let invalid = || InferenceError::InvalidScatterDimensions;
    let vector = usize::try_from(d.index_vector_dim)
        .ok()
        .filter(|&axis| axis <= index.len())
        .ok_or_else(invalid)?;
    let window = axes(&d.update_window_dims, update.sizes.len(), true)?;
    let inserted = axes(&d.inserted_window_dims, input.sizes.len(), true)?;
    let input_batches = axes(&d.input_batching_dims, input.sizes.len(), true)?;
    let index_batches = axes(&d.scatter_indices_batching_dims, index.len(), false)?;
    let index_map = axes(&d.scatter_dims_to_operand_dims, input.sizes.len(), false)?;
    if window.len() + inserted.len() + input_batches.len() != input.sizes.len()
        || input_batches.len() != index_batches.len()
        || input_batches
            .iter()
            .any(|axis| inserted.contains(axis) || index_map.contains(axis))
        || index_batches.contains(&vector)
    {
        return Err(invalid());
    }
    merge_dimension(
        vector,
        index.get(vector).copied().unwrap_or(Some(1)),
        Some(index_map.len() as u64),
        bound(indices, vector),
        None,
        Specificity::Most,
    )?;
    let scatter_axes = (0..index.len())
        .filter(|&axis| axis != vector)
        .collect::<Vec<_>>();
    let update_axes = (0..update.sizes.len())
        .filter(|axis| !window.contains(axis))
        .collect::<Vec<_>>();
    if scatter_axes.len() != update_axes.len() {
        return Err(invalid());
    }
    for (&index_axis, &update_axis) in scatter_axes.iter().zip(&update_axes) {
        (update.sizes[update_axis], update.bounds[update_axis]) = merge_dimension(
            update_axis,
            update.sizes[update_axis],
            index[index_axis],
            update.bounds[update_axis],
            bound(indices, index_axis),
            Specificity::Most,
        )?;
    }
    for (&input_axis, &index_axis) in input_batches.iter().zip(&index_batches) {
        let position = scatter_axes
            .iter()
            .position(|&axis| axis == index_axis)
            .expect("batch axis excludes vector axis");
        let update_axis = update_axes[position];
        merge_dimension(
            input_axis,
            input.sizes[input_axis],
            update.sizes[update_axis],
            input.bounds[input_axis],
            update.bounds[update_axis],
            Specificity::Most,
        )?;
    }
    let input_windows = (0..input.sizes.len())
        .filter(|axis| !inserted.contains(axis) && !input_batches.contains(axis));
    for (&update_axis, input_axis) in window.iter().zip(input_windows) {
        if let (Some(size), Some(upper)) = (
            update.sizes[update_axis],
            input.sizes[input_axis].or(input.bounds[input_axis]),
        ) && size > upper
        {
            return Err(InferenceError::SliceSizeExceeded {
                dimension: input_axis,
                size,
                size_or_bound: upper,
            });
        }
    }
    Ok(())
}
