use crate::{
    Diagnostic,
    attributes::{Axes, GatherDimensionNumbers, ScatterDimensionNumbers},
    shape::{self, Dimension, UNBOUNDED_SIZE},
    types::tensor,
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

use super::super::{axes, require};

fn expanded_indices(ctx: &Context, indices: TypeHandle, vector: usize) -> Result<Vec<Dimension>> {
    let indices = tensor(ctx, indices)?;
    require(indices.element.is_integer(), "indices must be integral")?;
    require(
        vector <= indices.dimensions.len(),
        &format!(
            "index vector dimension {vector} out of range for indices rank {}",
            indices.dimensions.len()
        ),
    )?;
    let mut output = indices
        .dimensions
        .iter()
        .zip(&indices.dynamic_dimensions)
        .map(|(&size, &dynamic)| {
            if size == UNBOUNDED_SIZE {
                Dimension::Unbounded
            } else if dynamic {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            }
        })
        .collect::<Vec<_>>();
    if vector == output.len() {
        output.push(Dimension::Static(1));
    }
    Ok(output)
}
fn bound(dimension: Dimension) -> i64 {
    match dimension {
        Dimension::Static(size) | Dimension::Bounded(size) => size,
        Dimension::Unbounded => UNBOUNDED_SIZE,
    }
}
fn compatible_size(left: i64, right: i64) -> bool {
    left == right || left == UNBOUNDED_SIZE || right == UNBOUNDED_SIZE
}
/// Infer the output shape for gather index and window partitions.
pub fn infer_gather(
    ctx: &Context,
    input: TypeHandle,
    indices: TypeHandle,
    dims: &GatherDimensionNumbers,
    sizes: &Axes,
) -> Result<TypeHandle> {
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    let vector = usize::try_from(dims.index_vector_dim).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "index_vector_dim",
            message: format!(
                "gather index vector dimension {} must be nonnegative",
                dims.index_vector_dim
            ),
        })
    })?;
    let expanded = expanded_indices(ctx, indices, vector)?;
    require(
        vector < expanded.len(),
        "gather index vector dimension out of range",
    )?;
    let output_rank = dims
        .offset_dims
        .0
        .len()
        .checked_add(expanded.len() - 1)
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: "gather result rank overflow".into(),
            })
        })?;
    let offsets = axes(output_rank, &dims.offset_dims.0, true)?;
    let collapsed = axes(rank, &dims.collapsed_slice_dims.0, true)?;
    let batching = axes(rank, &dims.operand_batching_dims.0, true)?;
    axes(expanded.len(), &dims.start_indices_batching_dims.0, false)?;
    axes(rank, &dims.start_index_map.0, false)?;
    let vector_size = bound(expanded[vector]);
    require(
        vector_size == UNBOUNDED_SIZE || vector_size == dims.start_index_map.0.len() as i64,
        "gather index vector and start map cardinalities differ",
    )?;
    require(
        sizes.0.len() == rank,
        "gather requires one slice size per input axis",
    )?;
    require(
        rank == dims.offset_dims.0.len()
            + dims.collapsed_slice_dims.0.len()
            + dims.operand_batching_dims.0.len(),
        "gather input partition mismatch",
    )?;
    require(
        dims.operand_batching_dims.0.len() == dims.start_indices_batching_dims.0.len(),
        "gather operand and index batching cardinalities differ",
    )?;
    for axis in 0..rank {
        require(
            !(batching[axis] && collapsed[axis]),
            "gather dimensions overlap",
        )?;
        let size = sizes.0[axis];
        require(size >= 0, "negative gather slice size")?;
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE || size <= input_ty.dimensions[axis],
            "gather slice size exceeds input bound",
        )?;
        require(
            !(collapsed[axis] || batching[axis]) || size <= 1,
            "collapsed or batching gather slice exceeds one",
        )?;
    }
    let mut windows = (0..rank).filter(|&axis| !collapsed[axis] && !batching[axis]);
    let mut batches = expanded
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != vector);
    let mut dimensions = Vec::with_capacity(output_rank);
    for is_offset in offsets {
        if is_offset {
            let source = windows.next().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: "gather output offset has no remaining input window dimension".into(),
                })
            })?;
            dimensions.push(if sizes.0[source] == input_ty.dimensions[source] {
                shape::dimension(ctx, input, source)?
            } else {
                Dimension::Static(sizes.0[source])
            });
        } else {
            dimensions.push(
                *batches
                    .next()
                    .ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: "gather output batch has no remaining index dimension".into(),
                        })
                    })?
                    .1,
            );
        }
    }
    require(
        windows.next().is_none() && batches.next().is_none(),
        "gather output partition mismatch",
    )?;
    shape::with_default_layout(
        ctx,
        shape::dynamic_array(ctx, input_ty.element, dimensions)?,
    )
}
/// Infer the output shape for a single-input scatter.
pub fn infer_scatter(
    ctx: &Context,
    input: TypeHandle,
    indices: TypeHandle,
    update: TypeHandle,
    dims: &ScatterDimensionNumbers,
) -> Result<TypeHandle> {
    infer_variadic_scatter(ctx, &[input], indices, &[update], dims)
}
/// Infer the result type for variadic scatter inputs and updates.
///
/// A single input retains its type; multiple inputs produce a tuple.
pub fn infer_variadic_scatter(
    ctx: &Context,
    inputs: &[TypeHandle],
    indices: TypeHandle,
    updates: &[TypeHandle],
    dims: &ScatterDimensionNumbers,
) -> Result<TypeHandle> {
    require(
        !inputs.is_empty() && inputs.len() == updates.len(),
        "scatter requires equally many inputs and updates",
    )?;
    let vector = usize::try_from(dims.index_vector_dim).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "index_vector_dim",
            message: format!(
                "scatter index vector dimension {} must be nonnegative",
                dims.index_vector_dim
            ),
        })
    })?;
    let expanded = expanded_indices(ctx, indices, vector)?;
    require(
        vector < expanded.len(),
        "scatter index vector dimension out of range",
    )?;
    let mut results = Vec::with_capacity(inputs.len());
    for (&input, &update) in inputs.iter().zip(updates) {
        let input_ty = tensor(ctx, input)?;
        let update_ty = tensor(ctx, update)?;
        let rank = input_ty.dimensions.len();
        let update_rank = update_ty.dimensions.len();
        let update_windows = axes(update_rank, &dims.update_window_dims.0, true)?;
        let inserted = axes(rank, &dims.inserted_window_dims.0, true)?;
        let batching = axes(rank, &dims.input_batching_dims.0, true)?;
        axes(expanded.len(), &dims.scatter_indices_batching_dims.0, false)?;
        axes(rank, &dims.scatter_dims_to_operand_dims.0, false)?;
        require(
            update_rank == expanded.len() - 1 + dims.update_window_dims.0.len(),
            "scatter update rank differs from index and window dimensions",
        )?;
        require(
            rank == dims.update_window_dims.0.len()
                + dims.inserted_window_dims.0.len()
                + dims.input_batching_dims.0.len(),
            "scatter window partition mismatch",
        )?;
        let vector_size = bound(expanded[vector]);
        require(
            vector_size == UNBOUNDED_SIZE
                || vector_size == dims.scatter_dims_to_operand_dims.0.len() as i64,
            "scatter index vector and map cardinalities differ",
        )?;
        require(
            dims.input_batching_dims.0.len() == dims.scatter_indices_batching_dims.0.len(),
            "scatter input and index batching cardinalities differ",
        )?;
        require(
            dims.inserted_window_dims
                .0
                .iter()
                .all(|&axis| !batching[axis as usize]),
            "scatter batching and inserted dimensions overlap",
        )?;
        let mut input_windows = (0..rank).filter(|&axis| !inserted[axis] && !batching[axis]);
        let mut index_dimensions = expanded
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != vector);
        for (axis, &is_window) in update_windows.iter().enumerate() {
            if is_window {
                let source = input_windows.next().ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Inference {
                        message: format!(
                            "scatter update axis {axis} has no input window dimension"
                        ),
                    })
                })?;
                require(
                    input_ty.dimensions[source] == UNBOUNDED_SIZE
                        || update_ty.dimensions[axis] <= input_ty.dimensions[source],
                    "scatter update window exceeds input",
                )?;
            } else {
                let index_size = *index_dimensions
                    .next()
                    .ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: format!(
                                "scatter update axis {axis} has no index batch dimension"
                            ),
                        })
                    })?
                    .1;
                require(
                    compatible_size(update_ty.dimensions[axis], bound(index_size)),
                    "scatter update batch size differs from indices",
                )?;
            }
        }
        require(
            input_windows.next().is_none() && index_dimensions.next().is_none(),
            "scatter update partition mismatch",
        )?;
        require(
            input_ty
                .element
                .same_ignoring_fp_precision(update_ty.element),
            "scatter input and update element types differ",
        )?;
        results.push(input);
    }
    if results.len() == 1 {
        Ok(results[0])
    } else {
        shape::tuple(ctx, results)
    }
}
