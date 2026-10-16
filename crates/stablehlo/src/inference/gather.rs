use super::{
    InferenceError, Specificity, bound, bounds_encoding, dimensions, encoding, merge_dimension,
    with_shape_and_encoding,
};
use crate::{GatherDimensionNumbers, StableHloOp, Type};

/// Infers gather shape and index-derived bounds using the pinned `inferGatherOp`.
/// Operand element metadata is retained; only index batch bounds are propagated.
pub fn infer_gather_type(
    operand: &Type,
    start_indices: &Type,
    dimension_numbers: &GatherDimensionNumbers,
    slice_sizes: &[i64],
) -> Result<Type, InferenceError> {
    let sizes = slice_sizes
        .iter()
        .enumerate()
        .map(|(dimension, &size)| {
            u64::try_from(size)
                .map(Some)
                .map_err(|_| InferenceError::InvalidSliceSize { dimension, size })
        })
        .collect::<Result<Vec<_>, _>>()?;
    infer(operand, start_indices, dimension_numbers, &sizes)
}

/// Infers dynamic gather with unknown slice values from the slice vector type.
/// Known values are checked separately when verifying constant operands.
pub fn infer_dynamic_gather_type(
    operand: &Type,
    start_indices: &Type,
    slice_sizes: &Type,
    dimension_numbers: &GatherDimensionNumbers,
) -> Result<Type, InferenceError> {
    slice_sizes
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 2, error })?;
    if !StableHloOp::DynamicGather.accepts_fixed_operand_type(2, slice_sizes) {
        return Err(InferenceError::IncompatibleTypes { index: 2 });
    }
    let rank = dimensions(operand)
        .ok_or(InferenceError::IncompatibleTypes { index: 0 })?
        .len();
    if dimensions(slice_sizes) != Some(&[Some(rank as u64)]) {
        return Err(InferenceError::IncompatibleTypes { index: 2 });
    }
    infer(operand, start_indices, dimension_numbers, &vec![None; rank])
}

fn axes(values: &[i64], rank: usize, sorted: bool) -> Result<Vec<usize>, InferenceError> {
    let invalid = || InferenceError::InvalidDimensions {
        rank,
        dimensions: values.to_vec(),
    };
    let mut seen = vec![false; rank];
    let mut axes = Vec::with_capacity(values.len());
    for &value in values {
        let axis = usize::try_from(value).map_err(|_| invalid())?;
        let slot = seen.get_mut(axis).ok_or_else(invalid)?;
        if std::mem::replace(slot, true) {
            return Err(invalid());
        }
        axes.push(axis);
    }
    if sorted && !axes.is_sorted() {
        return Err(invalid());
    }
    Ok(axes)
}

pub(crate) fn infer(
    operand: &Type,
    indices: &Type,
    d: &GatherDimensionNumbers,
    sizes: &[Option<u64>],
) -> Result<Type, InferenceError> {
    for (index, ty) in [operand, indices].iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::Gather.accepts_fixed_operand_type(index, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let input = dimensions(operand).expect("checked gather domain");
    let index = dimensions(indices).expect("checked gather domain");
    let invalid = || InferenceError::InvalidGatherDimensions;
    if sizes.len() != input.len()
        || input.len()
            != d.offset_dims.len() + d.collapsed_slice_dims.len() + d.operand_batching_dims.len()
    {
        return Err(invalid());
    }
    let vector = usize::try_from(d.index_vector_dim)
        .ok()
        .filter(|&axis| axis <= index.len())
        .ok_or_else(invalid)?;
    let vector_size = index.get(vector).copied().unwrap_or(Some(1));
    merge_dimension(
        vector,
        vector_size,
        Some(d.start_index_map.len() as u64),
        bound(indices, vector),
        None,
        Specificity::Most,
    )?;
    let batch_axes = (0..index.len())
        .filter(|&axis| axis != vector)
        .collect::<Vec<_>>();
    let rank = batch_axes.len() + d.offset_dims.len();
    let offsets = axes(&d.offset_dims, rank, true)?;
    let collapsed = axes(&d.collapsed_slice_dims, input.len(), true)?;
    let input_batches = axes(&d.operand_batching_dims, input.len(), true)?;
    let index_batches = axes(&d.start_indices_batching_dims, index.len(), false)?;
    let index_map = axes(&d.start_index_map, input.len(), false)?;
    if input_batches.len() != index_batches.len()
        || input_batches
            .iter()
            .any(|axis| collapsed.contains(axis) || index_map.contains(axis))
        || index_batches.contains(&vector)
    {
        return Err(invalid());
    }
    for (&a, &b) in input_batches.iter().zip(&index_batches) {
        merge_dimension(
            a,
            input[a],
            index[b],
            bound(operand, a),
            bound(indices, b),
            Specificity::Most,
        )?;
    }
    for (axis, &size) in sizes.iter().enumerate() {
        if let Some(size) = size {
            if size > i64::MAX as u64
                || (size > 1 && (collapsed.contains(&axis) || input_batches.contains(&axis)))
            {
                return Err(invalid());
            }
            if let Some(limit) = input[axis].or_else(|| bound(operand, axis))
                && size > limit
            {
                return Err(InferenceError::SliceSizeExceeded {
                    dimension: axis,
                    size,
                    size_or_bound: limit,
                });
            }
        }
    }
    let mut slice_axes =
        (0..input.len()).filter(|axis| !collapsed.contains(axis) && !input_batches.contains(axis));
    let mut batches = batch_axes.into_iter();
    let mut shape = Vec::with_capacity(rank);
    let mut upper = Vec::with_capacity(rank);
    for axis in 0..rank {
        if offsets.contains(&axis) {
            shape.push(sizes[slice_axes.next().expect("checked offset cardinality")]);
            upper.push(None);
        } else {
            let batch = batches.next().expect("checked batch cardinality");
            shape.push(index[batch]);
            upper.push(bound(indices, batch));
        }
    }
    let encoding = if upper.is_empty() {
        encoding(indices).cloned()
    } else {
        bounds_encoding(upper)
    };
    let result = with_shape_and_encoding(operand, shape, encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}

pub(crate) fn validate_result(
    operand: &Type,
    indices: &Type,
    result: &Type,
    d: &GatherDimensionNumbers,
    sizes: &[Option<u64>],
) -> Result<(), InferenceError> {
    let inferred = infer(operand, indices, d, sizes)?;
    if !crate::verification::tensor_element_types_match(operand, result) {
        return Err(InferenceError::IncompatibleTypes { index: 2 });
    }
    super::infer_most_specific_type(&[inferred, result.clone()])?;
    let input = dimensions(operand).expect("checked domain");
    let index = dimensions(indices).expect("checked domain");
    let output = dimensions(result).expect("checked compatible result");
    for (&output_axis, input_axis) in d.offset_dims.iter().zip((0..input.len()).filter(|axis| {
        !d.collapsed_slice_dims.contains(&(*axis as i64))
            && !d.operand_batching_dims.contains(&(*axis as i64))
    })) {
        if let Some(size) = output[output_axis as usize]
            && let Some(limit) = input[input_axis].or_else(|| bound(operand, input_axis))
            && size > limit
        {
            return Err(InferenceError::SliceSizeExceeded {
                dimension: input_axis,
                size,
                size_or_bound: limit,
            });
        }
    }
    let vector = d.index_vector_dim as usize;
    let output_batches = (0..output.len())
        .filter(|axis| !d.offset_dims.contains(&(*axis as i64)))
        .collect::<Vec<_>>();
    let index_batches = (0..index.len())
        .filter(|&axis| axis != vector)
        .collect::<Vec<_>>();
    for (&a, &b) in d
        .operand_batching_dims
        .iter()
        .zip(&d.start_indices_batching_dims)
    {
        let (a, b) = (a as usize, b as usize);
        let position = index_batches
            .iter()
            .position(|&axis| axis == b)
            .expect("checked batch map");
        let c = output_batches[position];
        let (size, upper) = merge_dimension(
            a,
            input[a],
            index[b],
            bound(operand, a),
            bound(indices, b),
            Specificity::Most,
        )?;
        merge_dimension(
            a,
            size,
            output[c],
            upper,
            bound(result, c),
            Specificity::Most,
        )?;
    }
    Ok(())
}
