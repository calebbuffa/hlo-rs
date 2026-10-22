use crate::{
    Diagnostic,
    attributes::Axes,
    shape,
    types::{TensorType, tensor},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};

use super::super::{axes, require};

/// Validate array operands and return the declared bitcast result.
pub fn infer_bitcast(ctx: &Context, input: TypeHandle, output: TypeHandle) -> Result<TypeHandle> {
    tensor(ctx, shape::array_view(ctx, input)?)?;
    tensor(ctx, shape::array_view(ctx, output)?)?;
    Ok(output)
}
/// Infer dimensions and layout after a permutation.
pub fn infer_transpose(ctx: &Context, input: TypeHandle, permutation: &Axes) -> Result<TypeHandle> {
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    axes(rank, &permutation.0, false)?;
    require(
        permutation.0.len() == rank,
        "transpose requires a full permutation",
    )?;
    let mut inverse = vec![0usize; rank];
    for (target, &source) in permutation.0.iter().enumerate() {
        inverse[source as usize] = target;
    }
    let mut output = input_ty.clone();
    output.dimensions = permutation
        .0
        .iter()
        .map(|&axis| input_ty.dimensions[axis as usize])
        .collect();
    output.dynamic_dimensions = permutation
        .0
        .iter()
        .map(|&axis| input_ty.dynamic_dimensions[axis as usize])
        .collect();
    for layout in &mut output.layout {
        for axis in &mut layout.minor_to_major {
            let source = usize::try_from(*axis).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("negative transpose layout axis {axis} for rank {rank}"),
                })
            })?;
            *axis = i64::try_from(*inverse.get(source).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("transpose layout axis {source} out of range for rank {rank}"),
                })
            })?)
            .map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("transpose layout axis {source} exceeds i64"),
                })
            })?;
        }
    }
    output.verify(ctx)?;
    Ok(TensorType::instantiate(output, ctx).into())
}
/// Validate reversed axes and return the unchanged type.
pub fn infer_reverse(ctx: &Context, input: TypeHandle, dimensions: &Axes) -> Result<TypeHandle> {
    let rank = shape::rank(ctx, input)?;
    require(
        dimensions
            .0
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == dimensions.0.len(),
        "a reverse dimension is repeated",
    )?;
    axes(rank, &dimensions.0, false)?;
    Ok(input)
}
