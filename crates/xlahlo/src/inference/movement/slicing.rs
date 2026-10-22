use crate::{
    Diagnostic,
    attributes::{Axes, Slice},
    shape::{self, Dimension, Layout, UNBOUNDED_SIZE},
    types::{self, TensorType, tensor},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};

use super::super::require;

fn validate_index_shapes(ctx: &Context, operand_rank: usize, indices: &[TypeHandle]) -> Result<()> {
    if let Some(&index) = indices.first() {
        let index = shape::array_view(ctx, index)?;
        let index_ty = tensor(ctx, index)?;
        if index_ty.dimensions.len() == 1 {
            require(
                indices.len() == 1,
                "index vector must be the only index operand",
            )?;
            require(
                index_ty.element.is_integer() && index_ty.dimensions == [operand_rank as i64],
                "index vector must be integral and match the operand rank",
            )?;
            return Ok(());
        }
    }
    require(
        indices.len() == operand_rank,
        &format!(
            "index count {} must match operand rank {operand_rank}",
            indices.len()
        ),
    )?;
    if let Some(&first) = indices.first() {
        let first_array = shape::array_view(ctx, first)?;
        let first_ty = tensor(ctx, first_array)?;
        require(
            first_ty.dimensions.is_empty() && first_ty.element.is_integer(),
            "indices must be integral scalars",
        )?;
        for &index in &indices[1..] {
            let index = shape::array_view(ctx, index)?;
            require(
                shape::compatible(ctx, first_array, index),
                "indices must have the same scalar type and shape",
            )?;
        }
    }
    Ok(())
}
/// Infer the shape of a strided slice.
///
/// Only bounded input dynamism propagates, and unit output extents are static.
pub fn infer_slice(ctx: &Context, input: TypeHandle, geometry: &Slice) -> Result<TypeHandle> {
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    require(
        geometry.starts.0.len() == rank
            && geometry.limits.0.len() == rank
            && geometry.strides.0.len() == rank,
        "slice geometry rank mismatch",
    )?;
    let mut dimensions = Vec::with_capacity(rank);
    let mut dynamic = Vec::with_capacity(rank);
    for axis in 0..rank {
        let (start, limit, stride) = (
            geometry.starts.0[axis],
            geometry.limits.0[axis],
            geometry.strides.0[axis],
        );
        require(start >= 0, "negative slice start")?;
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE || limit <= input_ty.dimensions[axis],
            "slice limit exceeds input bound",
        )?;
        require(
            limit >= start && stride > 0,
            "invalid slice bounds or stride",
        )?;
        let extent = limit.checked_sub(start).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!(
                    "slice extent overflows at axis {axis}: start {start}, limit {limit}"
                ),
            })
        })?;
        let size = extent / stride + i64::from(extent % stride != 0);
        dimensions.push(size);
        dynamic.push(
            input_ty.dynamic_dimensions[axis]
                && input_ty.dimensions[axis] != UNBOUNDED_SIZE
                && size != 1,
        );
    }
    let mut output = input_ty.clone();
    output.dimensions = dimensions;
    output.dynamic_dimensions = dynamic;
    output.layout = vec![Layout::descending(rank)];
    output.verify(ctx)?;
    Ok(TensorType::instantiate(output, ctx).into())
}
/// Infer the bounded result of a runtime-indexed slice.
pub fn infer_dynamic_slice(
    ctx: &Context,
    input: TypeHandle,
    indices: &[TypeHandle],
    sizes: &Axes,
) -> Result<TypeHandle> {
    let input = shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    validate_index_shapes(ctx, rank, indices)?;
    require(sizes.0.len() == rank, "dynamic slice sizes must match rank")?;
    let mut dimensions = Vec::with_capacity(rank);
    for (axis, &size) in sizes.0.iter().enumerate() {
        require(size >= 0, "dynamic slice size must be nonnegative")?;
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE || size <= input_ty.dimensions[axis],
            "dynamic slice size exceeds input bound",
        )?;
        dimensions.push(
            if input_ty.dynamic_dimensions[axis] && size > 1 && size == input_ty.dimensions[axis] {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            },
        );
    }
    let output = shape::dynamic_array(ctx, input_ty.element, dimensions)?;
    shape::with_default_layout(ctx, output)
}
/// Infer the result of updating an array at runtime indices.
pub fn infer_dynamic_update_slice(
    ctx: &Context,
    input: TypeHandle,
    update: TypeHandle,
    indices: &[TypeHandle],
) -> Result<TypeHandle> {
    let input_array = shape::array_view(ctx, input)?;
    let update = shape::array_view(ctx, update)?;
    let input_ty = tensor(ctx, input_array)?;
    let update_ty = tensor(ctx, update)?;
    let rank = input_ty.dimensions.len();
    validate_index_shapes(ctx, rank, indices)?;
    require(
        update_ty.dimensions.len() == rank,
        "dynamic update rank mismatch",
    )?;
    require(
        update_ty
            .element
            .same_ignoring_fp_precision(input_ty.element),
        "dynamic update element type mismatch",
    )?;
    for axis in 0..rank {
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE
                || update_ty.dimensions[axis] <= input_ty.dimensions[axis],
            "dynamic update exceeds input bound",
        )?;
    }
    let mut output = input_ty.clone();
    for axis in 0..rank {
        output.dynamic_dimensions[axis] = input_ty.dynamic_dimensions[axis]
            || update_ty.dynamic_dimensions[axis]
                && update_ty.dimensions[axis] == input_ty.dimensions[axis];
    }
    output.verify(ctx)?;
    let output: TypeHandle = TensorType::instantiate(output, ctx).into();
    if input.deref(ctx).is::<types::BufferType>() {
        shape::buffer(ctx, output)
    } else {
        Ok(output)
    }
}
