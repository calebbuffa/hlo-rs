use super::UNBOUNDED_SIZE;
use crate::types::TensorType;
use pliron::{context::Context, r#type::TypeHandle};

/// Compare shape structure, ignoring layouts and dynamic flags; unbounded extents match any bound.
pub fn compatible(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    compare(ctx, a, b, false, false, true)
}

/// Compare compatible shapes while also allowing differing floating-point precisions.
pub fn compatible_ignoring_fp_precision(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    compare(ctx, a, b, true, false, true)
}

/// Compare types, bounds, and dynamic flags without considering physical layouts.
pub fn equal_ignoring_layout(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    compare(ctx, a, b, false, false, false)
}

/// Compare structure and dynamic flags, ignoring layout and floating-point precision.
pub fn equal_ignoring_layout_and_fp_precision(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    compare(ctx, a, b, true, false, false)
}

/// Compare complete shapes using semantic layout equality.
pub fn equal_including_layout(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    compare(ctx, a, b, false, true, false)
}

fn compare(
    ctx: &Context,
    a: TypeHandle,
    b: TypeHandle,
    fp: bool,
    layout: bool,
    compatible: bool,
) -> bool {
    let at = a.deref(ctx);
    let bt = b.deref(ctx);
    if let (Some(ta), Some(tb)) = (
        at.downcast_ref::<TensorType>(),
        bt.downcast_ref::<TensorType>(),
    ) {
        return (ta.element == tb.element || fp && ta.element.is_float() && tb.element.is_float())
            && ta.dimensions.len() == tb.dimensions.len()
            && ta.dimensions.iter().zip(&tb.dimensions).all(|(&a, &b)| {
                a == b || compatible && (a == UNBOUNDED_SIZE || b == UNBOUNDED_SIZE)
            })
            && (compatible || ta.dynamic_dimensions == tb.dynamic_dimensions)
            && (!layout
                || match (super::layout(ctx, a), super::layout(ctx, b)) {
                    (Ok(None), Ok(None)) => true,
                    (Ok(Some(a)), Ok(Some(b))) => a.equal(ctx, &b),
                    _ => false,
                });
    }
    if at.get_type_id() != bt.get_type_id() {
        return false;
    }
    let ac = super::children(ctx, a);
    let bc = super::children(ctx, b);
    ac.len() == bc.len()
        && ac
            .iter()
            .zip(bc.iter())
            .all(|(&a, &b)| compare(ctx, a, b, fp, layout, compatible))
}
