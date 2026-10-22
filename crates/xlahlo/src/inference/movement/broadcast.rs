use crate::{attributes::Axes, shape::UNBOUNDED_SIZE, types::tensor};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

use super::super::{axes, require};

/// Validate a broadcast mapping and return the declared output type.
pub fn infer_broadcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    dimensions: &Axes,
) -> Result<TypeHandle> {
    check_broadcast(ctx, input, output, dimensions, true, true)?;
    Ok(output)
}
pub(crate) fn verify_broadcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    dimensions: &Axes,
    require_order: bool,
) -> Result<()> {
    check_broadcast(ctx, input, output, dimensions, require_order, false)
}
fn check_broadcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    dimensions: &Axes,
    require_order: bool,
    reject_unbounded_output: bool,
) -> Result<()> {
    let input = tensor(ctx, input)?;
    let output = tensor(ctx, output)?;
    axes(output.dimensions.len(), &dimensions.0, require_order)?;
    require(
        dimensions.0.len() == input.dimensions.len(),
        "broadcast mapping rank mismatch",
    )?;
    if reject_unbounded_output {
        require(
            !output.dimensions.contains(&UNBOUNDED_SIZE),
            "unbounded output is not supported by broadcast inference",
        )?;
    }
    for (axis, &target) in dimensions.0.iter().enumerate() {
        let target = target as usize;
        if input.dimensions[axis] != UNBOUNDED_SIZE {
            require(
                input.dimensions[axis] == 1 || input.dimensions[axis] == output.dimensions[target],
                "broadcast mapped dimensions differ",
            )?;
            require(
                input.dynamic_dimensions[axis] == output.dynamic_dimensions[target],
                "broadcast mapped dynamism differs",
            )?;
        }
    }
    Ok(())
}
