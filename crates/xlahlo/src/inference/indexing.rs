use super::{axes, compatible_size, infer_reduce};
use crate::proto::xla_data::{GatherDimensionNumbers, ProgramShapeProto, ScatterDimensionNumbers};
use crate::{Dimension, Error, Result, Shape, UNBOUNDED_SIZE};

fn sorted_axes(rank: usize, dimensions: &[i64]) -> Result<()> {
    axes(rank, dimensions, false)?;
    if dimensions.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::new(
            "dimensions",
            "window/collapsed/batching dimensions must be strictly increasing",
        ));
    }
    Ok(())
}

fn expanded_indices(indices: &Shape, index_vector_dim: i64) -> Result<Vec<Dimension>> {
    let rank = indices.rank()?;
    if !indices.element_type().is_integral() {
        return Err(Error::new(
            "type-domain",
            "gather/scatter indices must have integral element type",
        ));
    }
    if index_vector_dim < 0 || index_vector_dim > rank as i64 {
        return Err(Error::new(
            "dimensions",
            "index vector dimension must be in [0,indices rank]",
        ));
    }
    let mut dimensions = (0..rank)
        .map(|axis| indices.dimension(axis))
        .collect::<Result<Vec<_>>>()?;
    if index_vector_dim == rank as i64 {
        dimensions.push(Dimension::Static(1));
    }
    Ok(dimensions)
}

fn bound(dimension: Dimension) -> i64 {
    match dimension {
        Dimension::Static(n) | Dimension::Bounded(n) => n,
        Dimension::Unbounded => UNBOUNDED_SIZE,
    }
}

pub fn infer_gather(
    input: &Shape,
    indices: &Shape,
    dimensions: &GatherDimensionNumbers,
    slice_sizes: &[i64],
) -> Result<Shape> {
    let rank = input.rank()?;
    let expanded = expanded_indices(indices, dimensions.index_vector_dim)?;
    let output_rank = dimensions
        .offset_dims
        .len()
        .checked_add(expanded.len() - 1)
        .ok_or_else(|| Error::new("overflow", "gather result rank overflow"))?;
    sorted_axes(output_rank, &dimensions.offset_dims)?;
    axes(rank, &dimensions.start_index_map, false)?;
    sorted_axes(rank, &dimensions.collapsed_slice_dims)?;
    sorted_axes(rank, &dimensions.operand_batching_dims)?;
    axes(
        expanded.len(),
        &dimensions.start_indices_batching_dims,
        false,
    )?;
    if dimensions.operand_batching_dims.len() != dimensions.start_indices_batching_dims.len()
        || !compatible_size(
            dimensions.start_index_map.len() as i64,
            bound(expanded[dimensions.index_vector_dim as usize]),
        )
    {
        return Err(Error::new(
            "dimensions",
            "gather index vector or batching cardinality mismatch",
        ));
    }
    if slice_sizes.len() != rank
        || rank
            != dimensions.offset_dims.len()
                + dimensions.collapsed_slice_dims.len()
                + dimensions.operand_batching_dims.len()
    {
        return Err(Error::new(
            "dimensions",
            "gather requires one slice size per input axis partitioned into offset, collapsed and batching axes",
        ));
    }
    for (axis, &size) in slice_sizes.iter().enumerate() {
        if input.dimensions()[axis] != UNBOUNDED_SIZE
            && (size < 0 || size > input.dimensions()[axis])
        {
            return Err(Error::new(
                "slice",
                "gather slice size is outside operand bounds",
            ));
        }
    }
    for &axis in dimensions
        .collapsed_slice_dims
        .iter()
        .chain(&dimensions.operand_batching_dims)
    {
        if slice_sizes[axis as usize] > 1 {
            return Err(Error::new(
                "slice",
                "gather collapsed/batching slice sizes cannot exceed one",
            ));
        }
    }
    let mut offsets = (0..rank).filter(|axis| {
        !dimensions.collapsed_slice_dims.contains(&(*axis as i64))
            && !dimensions.operand_batching_dims.contains(&(*axis as i64))
    });
    let mut batches = expanded
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != dimensions.index_vector_dim as usize);
    let mut result = Vec::with_capacity(output_rank);
    for axis in 0..output_rank {
        if dimensions.offset_dims.contains(&(axis as i64)) {
            let source = offsets.next().ok_or_else(|| {
                Error::new(
                    "dimensions",
                    "gather has overlapping collapsed and batching dimensions",
                )
            })?;
            let size = slice_sizes[source];
            result.push(if size == input.dimensions()[source] {
                input.dimension(source)?
            } else {
                Dimension::Static(size)
            });
        } else {
            result.push(
                *batches
                    .next()
                    .ok_or_else(|| Error::new("dimensions", "gather missing batch dimension"))?
                    .1,
            );
        }
    }
    Shape::dynamic_array(input.element_type(), result)?.with_default_layout()
}

pub fn infer_scatter(
    operands: &[Shape],
    dimensions: &ScatterDimensionNumbers,
    reducer: &ProgramShapeProto,
) -> Result<Shape> {
    if operands.len() < 3 || operands.len().is_multiple_of(2) {
        return Err(Error::new(
            "arity",
            "scatter requires N inputs, indices and N updates with N >= 1",
        ));
    }
    let count = operands.len() / 2;
    let inputs = &operands[..count];
    let updates = &operands[count + 1..];
    let expanded = expanded_indices(&operands[count], dimensions.index_vector_dim)?;
    if dimensions.input_batching_dims.len() != dimensions.scatter_indices_batching_dims.len() {
        return Err(Error::new(
            "dimensions",
            "scatter input/index batching cardinalities differ",
        ));
    }
    axes(
        expanded.len(),
        &dimensions.scatter_indices_batching_dims,
        false,
    )?;
    if !compatible_size(
        dimensions.scatter_dims_to_operand_dims.len() as i64,
        bound(expanded[dimensions.index_vector_dim as usize]),
    ) {
        return Err(Error::new(
            "dimensions",
            "scatter index vector size and operand mapping differ",
        ));
    }
    for (input, update) in inputs.iter().zip(updates) {
        let rank = input.rank()?;
        let update_rank = update.rank()?;
        sorted_axes(update_rank, &dimensions.update_window_dims)?;
        sorted_axes(rank, &dimensions.inserted_window_dims)?;
        sorted_axes(rank, &dimensions.input_batching_dims)?;
        axes(rank, &dimensions.scatter_dims_to_operand_dims, false)?;
        if rank
            != dimensions.update_window_dims.len()
                + dimensions.inserted_window_dims.len()
                + dimensions.input_batching_dims.len()
            || update_rank != expanded.len() - 1 + dimensions.update_window_dims.len()
        {
            return Err(Error::new(
                "dimensions",
                "scatter input/update window rank mismatch",
            ));
        }
        let mut window_sources = (0..rank).filter(|axis| {
            !dimensions.inserted_window_dims.contains(&(*axis as i64))
                && !dimensions.input_batching_dims.contains(&(*axis as i64))
        });
        for &axis in &dimensions.update_window_dims {
            let source = window_sources.next().ok_or_else(|| {
                Error::new(
                    "dimensions",
                    "scatter overlapping inserted and batching dimensions",
                )
            })?;
            if update.dimensions()[axis as usize] > input.dimensions()[source] {
                return Err(Error::new(
                    "slice",
                    "scatter update window exceeds operand bounds",
                ));
            }
        }
        let mut index_dimensions = expanded
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != dimensions.index_vector_dim as usize);
        for axis in 0..update_rank {
            if !dimensions.update_window_dims.contains(&(axis as i64)) {
                let index = *index_dimensions
                    .next()
                    .ok_or_else(|| Error::new("dimensions", "scatter missing index dimension"))?
                    .1;
                if !compatible_size(update.dimensions()[axis], bound(index)) {
                    return Err(Error::new(
                        "shape",
                        "scatter update batch bounds and index bounds differ",
                    ));
                }
            }
        }
    }
    let reducer_operands = updates
        .iter()
        .chain(inputs)
        .map(|shape| Shape::scalar(shape.element_type()))
        .collect::<Result<Vec<_>>>()?;
    infer_reduce(&reducer_operands, &[], reducer)?;
    if count == 1 {
        Ok(inputs[0].clone())
    } else {
        Shape::tuple(inputs.to_vec())
    }
}
