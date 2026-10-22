use super::{Dimension, UNBOUNDED_SIZE};
use crate::{
    Diagnostic,
    types::PrimitiveType,
    types::{
        BufferType, ElementType, OpaqueType, TensorType, TokenType, TupleType, tensor,
        tensor_structure,
    },
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};
use std::{cell::Ref, slice};

/// Whether the type is a native array, excluding buffers.
pub fn is_array(ctx: &Context, shape: TypeHandle) -> bool {
    shape.deref(ctx).is::<TensorType>()
}

/// Whether the type is an array or an array-backed buffer.
pub fn is_array_or_buffer(ctx: &Context, shape: TypeHandle) -> bool {
    let ty = shape.deref(ctx);
    ty.is::<TensorType>() || ty.is::<BufferType>()
}

/// Whether any descendant array has an unbounded dynamic dimension.
pub fn has_unbounded(ctx: &Context, shape: TypeHandle) -> bool {
    let ty = shape.deref(ctx);
    if let Some(ty) = ty.downcast_ref::<TensorType>() {
        ty.dimensions.contains(&UNBOUNDED_SIZE)
    } else {
        super::children(ctx, shape)
            .iter()
            .any(|&child| super::has_unbounded(ctx, child))
    }
}

/// Whether every descendant array specifies a physical layout.
pub fn has_complete_layout(ctx: &Context, shape: TypeHandle) -> Result<bool> {
    if super::is_array_or_buffer(ctx, shape) {
        return Ok(super::layout(ctx, shape)?.is_some());
    }
    for &child in super::children(ctx, shape).iter() {
        if !super::has_complete_layout(ctx, child)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the type is a native tuple.
pub fn is_tuple(ctx: &Context, shape: TypeHandle) -> bool {
    shape.deref(ctx).is::<TupleType>()
}

/// Return array rank, rejecting structural types.
pub fn rank(ctx: &Context, shape: TypeHandle) -> Result<usize> {
    Ok(tensor_structure(ctx, shape)?.dimensions.len())
}

/// Borrows verified array dimension bounds without copying the interned type.
///
/// The borrow is tied to `ctx`; the array type must not be mutably borrowed
/// while this view is alive. Structural types, including array-backed buffers,
/// return an error; use [`super::array_view`] to select a buffer's array first.
pub fn dimensions(ctx: &Context, shape: TypeHandle) -> Result<Ref<'_, [i64]>> {
    Ok(Ref::map(tensor(ctx, shape)?, |ty| ty.dimensions.as_slice()))
}

/// Return the numeric element format of an array.
pub fn element_type(ctx: &Context, shape: TypeHandle) -> Result<ElementType> {
    Ok(tensor_structure(ctx, shape)?.element)
}

/// Returns the native primitive type, including structural primitives.
///
/// Arrays return the primitive corresponding to their numeric element format.
/// Tuples, tokens, opaque values, and buffers return their structural primitive.
/// A non-HLO type handle returns an error; this query does not recursively
/// verify the contents of a raw tuple or buffer.
pub fn primitive_type(ctx: &Context, shape: TypeHandle) -> Result<PrimitiveType> {
    let ty = shape.deref(ctx);
    if let Some(ty) = ty.downcast_ref::<TensorType>() {
        return Ok(ty.element.into());
    }
    if ty.is::<TupleType>() {
        return Ok(PrimitiveType::Tuple);
    }
    if ty.is::<TokenType>() {
        return Ok(PrimitiveType::Token);
    }
    if ty.is::<OpaqueType>() {
        return Ok(PrimitiveType::Opaque);
    }
    if ty.is::<BufferType>() {
        return Ok(PrimitiveType::Buffer);
    }
    Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
        message: "expected native HLO type".into(),
    }))
}

/// Return an array axis's extent and dynamic classification.
///
/// Checks structural cardinalities and the requested axis in constant time.
/// Use `Verify::verify` to validate the entire raw type before consuming it.
pub fn dimension(ctx: &Context, shape: TypeHandle, axis: usize) -> Result<Dimension> {
    let ty = tensor_structure(ctx, shape)?;
    let &size = ty.dimensions.get(axis).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "axis out of range".into(),
        })
    })?;
    if (size < 0 && !(size == UNBOUNDED_SIZE && ty.dynamic_dimensions[axis]))
        || ty.layout.first().is_some_and(|layout| {
            layout.minor_to_major.get(axis).is_some_and(|&value| {
                usize::try_from(value).map_or(true, |value| value >= ty.dimensions.len())
            })
        })
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "malformed array dimension or layout axis".into(),
        }));
    }
    Ok(if size == UNBOUNDED_SIZE {
        Dimension::Unbounded
    } else if ty.dynamic_dimensions[axis] {
        Dimension::Bounded(size)
    } else {
        Dimension::Static(size)
    })
}

/// Borrow tuple elements or a buffer's storage type; arrays have no children.
pub fn children(ctx: &Context, shape: TypeHandle) -> Ref<'_, [TypeHandle]> {
    Ref::map(shape.deref(ctx), |ty| {
        if let Some(ty) = ty.downcast_ref::<TupleType>() {
            ty.elements.as_slice()
        } else if let Some(ty) = ty.downcast_ref::<BufferType>() {
            slice::from_ref(&ty.array)
        } else {
            &[]
        }
    })
}

/// Count tuple-tree leaves; a non-tuple contributes one leaf.
pub fn leaf_count(ctx: &Context, shape: TypeHandle) -> usize {
    if super::is_tuple(ctx, shape) {
        super::children(ctx, shape)
            .iter()
            .map(|&child| super::leaf_count(ctx, child))
            .sum()
    } else {
        1
    }
}

/// Whether any descendant array has a bounded or unbounded runtime extent.
pub fn is_dynamic(ctx: &Context, shape: TypeHandle) -> bool {
    let ty = shape.deref(ctx);
    if let Some(ty) = ty.downcast_ref::<TensorType>() {
        ty.dynamic_dimensions.contains(&true)
    } else {
        super::children(ctx, shape)
            .iter()
            .any(|&child| super::is_dynamic(ctx, child))
    }
}

/// Multiply array bounds, rejecting unbounded extents and overflow.
pub fn element_count(ctx: &Context, shape: TypeHandle) -> Result<i64> {
    let ty = tensor(ctx, shape)?;
    if ty.dimensions.contains(&UNBOUNDED_SIZE) {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "unbounded shape has no fixed element count".into(),
        }));
    }
    ty.dimensions.iter().try_fold(1i64, |n, &d| {
        n.checked_mul(d).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "shape element count overflow".into(),
            })
        })
    })
}

/// Return an array or a buffer's underlying array without copying its type.
pub fn array_view(ctx: &Context, shape: TypeHandle) -> Result<TypeHandle> {
    let array = shape
        .deref(ctx)
        .downcast_ref::<BufferType>()
        .map_or(shape, |ty| ty.array);
    tensor(ctx, array)?;
    Ok(array)
}
