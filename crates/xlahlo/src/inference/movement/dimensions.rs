use crate::{
    attributes::{Axes, ShuffleMode},
    shape,
    types::{ElementType, tensor},
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

use super::super::{axes, require};

/// Validate and return the input type of a copy.
pub fn infer_copy(ctx: &Context, input: TypeHandle) -> Result<TypeHandle> {
    require(
        crate::types::is_hlo_type(ctx, input),
        "copy operand must be an HLO type",
    )?;
    Ok(input)
}
/// Infer the S32 scalar returned by get-dimension-size.
pub fn infer_get_dimension_size(
    ctx: &Context,
    input: TypeHandle,
    axis: usize,
) -> Result<TypeHandle> {
    let input = shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input)?;
    require(
        axis < input_ty.dimensions.len(),
        &format!(
            "get-dimension-size axis {axis} out of range for rank {}",
            input_ty.dimensions.len()
        ),
    )?;
    require(
        input_ty.dimensions[axis] <= i64::from(i32::MAX),
        "dimension exceeds S32 limit",
    )?;
    let scalar = shape::scalar(ctx, ElementType::S32)?;
    shape::with_default_layout(ctx, scalar)
}
/// Infer an input type with the selected dimension marked dynamic.
pub fn infer_set_dimension_size(
    ctx: &Context,
    input: TypeHandle,
    size: TypeHandle,
    axis: usize,
) -> Result<TypeHandle> {
    let input_array = shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input_array)?;
    require(
        axis < input_ty.dimensions.len(),
        &format!(
            "set-dimension-size axis {axis} out of range for rank {}",
            input_ty.dimensions.len()
        ),
    )?;
    require(
        input_ty.dimensions[axis] <= i64::from(i32::MAX),
        "dimension exceeds S32 limit",
    )?;
    let size = shape::array_view(ctx, size)?;
    let size_ty = tensor(ctx, size)?;
    require(
        size_ty.dimensions.is_empty() && size_ty.element == ElementType::S32,
        "dimension size must be S32 scalar",
    )?;
    shape::with_dynamic_dimension(ctx, input, axis, true)
}
/// Validate shuffle axes and shifts, preserving the input type.
pub fn infer_shuffle(
    ctx: &Context,
    input: TypeHandle,
    dimensions: &Axes,
    mode: &ShuffleMode,
) -> Result<TypeHandle> {
    let rank = shape::rank(ctx, input)?;
    require(
        !dimensions.0.is_empty(),
        "shuffle must include at least one dimension",
    )?;
    require(
        dimensions
            .0
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == dimensions.0.len(),
        "a shuffle dimension is repeated",
    )?;
    axes(rank, &dimensions.0, false)?;
    require(
        mode.shifts.0.len() == dimensions.0.len(),
        "shuffle dimensions and shifts must have the same size",
    )?;
    Ok(input)
}
