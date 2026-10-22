use super::{Dimension, UNBOUNDED_SIZE};
use crate::{
    Diagnostic,
    types::PrimitiveType,
    types::{BufferType, OpaqueType, TensorType, TokenType, TupleType},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};

/// Interns a static array without imposing a physical layout.
///
/// `dimensions` lists extents in logical axis order. An empty list
/// produces a scalar; zero extents are allowed. `element` accepts either a
/// numeric [`ElementType`](crate::types::ElementType) or a numeric
/// [`PrimitiveType`].
///
/// # Errors
///
/// Returns an error for a structural primitive, negative extent, or storage
/// size that cannot be represented by the native shape model.
pub fn array(
    ctx: &Context,
    element: impl Into<PrimitiveType>,
    dimensions: impl Into<Vec<i64>>,
) -> Result<TypeHandle> {
    let dimensions = dimensions.into();
    let ty = TensorType {
        element: element.into().array_element()?,
        dynamic_dimensions: vec![false; dimensions.len()],
        dimensions,
        layout: Vec::new(),
    };
    ty.verify(ctx)?;
    Ok(TensorType::instantiate(ty, ctx).into())
}

/// Interns a rank-zero numeric array with no specified physical layout.
///
/// This is equivalent to [`array()`] with an empty dimension list. Structural
/// primitives, including tokens and tuples, return an error.
pub fn scalar(ctx: &Context, element: impl Into<PrimitiveType>) -> Result<TypeHandle> {
    super::array(ctx, element, Vec::new())
}

/// Interns an array with explicit static, bounded, and unbounded dimensions.
///
/// Each [`Dimension`] supplies an extent or runtime bound for one logical
/// axis. The returned type has no specified physical layout; use
/// [`super::with_layout`] to attach one without changing this type.
///
/// # Errors
///
/// Returns an error for structural primitives, negative static extents or
/// dynamic bounds, or a bounded storage size that overflows the shape model.
pub fn dynamic_array(
    ctx: &Context,
    element: impl Into<PrimitiveType>,
    dimensions: Vec<Dimension>,
) -> Result<TypeHandle> {
    let element = element.into().array_element()?;
    let mut sizes = Vec::with_capacity(dimensions.len());
    let mut dynamic_dimensions = Vec::with_capacity(dimensions.len());
    for dimension in dimensions {
        let (size, dynamic) = match dimension {
            Dimension::Static(n) => (n, false),
            Dimension::Bounded(n) if n >= 0 => (n, true),
            Dimension::Bounded(_) => {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "negative dynamic bound".into(),
                }));
            }
            Dimension::Unbounded => (UNBOUNDED_SIZE, true),
        };
        sizes.push(size);
        dynamic_dimensions.push(dynamic);
    }
    let ty = TensorType {
        element,
        dimensions: sizes,
        layout: Vec::new(),
        dynamic_dimensions,
    };
    ty.verify(ctx)?;
    Ok(TensorType::instantiate(ty, ctx).into())
}

/// Interns an ordered tuple after validating every child type.
///
/// Children may be arrays, nested tuples, tokens, opaque values, or buffers.
/// An empty tuple is valid. All child handles must belong to `ctx`.
///
/// # Errors
///
/// Returns an error for a non-HLO child, an invalid child shape or layout, or
/// a type tree reaching the native nesting limit of 128.
pub fn tuple(ctx: &Context, elements: Vec<TypeHandle>) -> Result<TypeHandle> {
    TupleType::get(ctx, elements)
}

/// Return the interned sequencing token type.
pub fn token(ctx: &Context) -> TypeHandle {
    TokenType::get(ctx)
}

/// Return the interned opaque type.
pub fn opaque(ctx: &Context) -> TypeHandle {
    OpaqueType::get(ctx)
}

/// Interns a buffer whose underlying storage type is a verified native array.
///
/// Returns an error if `array` is a structural type or contains invalid
/// dimensions or layout metadata. The array's type handle is shared, not copied.
pub fn buffer(ctx: &Context, array: TypeHandle) -> Result<TypeHandle> {
    BufferType::get(ctx, array)
}
