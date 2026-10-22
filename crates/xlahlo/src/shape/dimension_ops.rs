use super::{Layout, validate_layout};
use crate::{
    Diagnostic,
    types::PrimitiveType,
    types::{BufferType, TensorType, tensor},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};
use std::cell::Ref;

/// Returns an array or buffer with a different dimension bound.
///
/// Preserves the axis's dynamic flag and all layout metadata. The original
/// interned type is never modified. Returns an error for an out-of-range axis,
/// an invalid bound, or a malformed source type.
pub fn with_dimension_bound(
    ctx: &Context,
    shape: TypeHandle,
    axis: usize,
    bound: i64,
) -> Result<TypeHandle> {
    let array = super::array_view(ctx, shape)?;
    let mut result = tensor(ctx, array)?.clone();
    result.verify(ctx)?;
    let dimension = result.dimensions.get_mut(axis).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: format!("dimension axis {axis} out of range"),
        })
    })?;
    *dimension = bound;
    result.verify(ctx)?;
    let result = TensorType::instantiate(result, ctx).into();
    if array == shape {
        Ok(result)
    } else {
        super::buffer(ctx, result)
    }
}

/// Returns an array or buffer with a different dynamic flag for one axis.
///
/// Preserves the bound and layout and leaves the original type unchanged.
/// Returns an error for an invalid axis or source type, or when attempting
/// to make an unbounded dimension static.
pub fn with_dynamic_dimension(
    ctx: &Context,
    shape: TypeHandle,
    axis: usize,
    dynamic: bool,
) -> Result<TypeHandle> {
    let array = super::array_view(ctx, shape)?;
    let mut result = tensor(ctx, array)?.clone();
    result.verify(ctx)?;
    let flag = result.dynamic_dimensions.get_mut(axis).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: format!("dimension axis {axis} out of range"),
        })
    })?;
    *flag = dynamic;
    result.verify(ctx)?;
    let result = TensorType::instantiate(result, ctx).into();
    if array == shape {
        Ok(result)
    } else {
        super::buffer(ctx, result)
    }
}

/// Delete one array or buffer axis, preserving retained bounds, dynamic
/// flags and the layout's OpenXLA dimension-deletion semantics.
pub fn delete_dimension(ctx: &Context, shape: TypeHandle, dimension: i64) -> Result<TypeHandle> {
    super::delete_dimensions(ctx, shape, &[dimension])
}

/// Delete distinct logical axes, expressed in the original array's rank.
///
/// Input order is immaterial. Negative, duplicate and out-of-range axes
/// are errors. The original interned type is never modified.
pub fn delete_dimensions(
    ctx: &Context,
    shape: TypeHandle,
    dimensions: &[i64],
) -> Result<TypeHandle> {
    let array = super::array_view(ctx, shape)?;
    let mut result = tensor(ctx, array)?.clone();
    result.verify(ctx)?;
    let mut deleted = vec![false; result.dimensions.len()];
    for &dimension in dimensions {
        let valid = usize::try_from(dimension)
            .ok()
            .and_then(|axis| deleted.get_mut(axis))
            .is_some_and(|slot| !std::mem::replace(slot, true));
        if !valid {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: format!("duplicate or out-of-range dimension {dimension}"),
            }));
        }
    }
    if dimensions.is_empty() {
        return Ok(shape);
    }
    for layout in &mut result.layout {
        for (axis, &delete) in deleted.iter().enumerate().rev() {
            if delete {
                layout.delete_dimension(axis as i64)?;
            }
        }
    }
    result.dimensions = result
        .dimensions
        .into_iter()
        .zip(&deleted)
        .filter_map(|(extent, &delete)| (!delete).then_some(extent))
        .collect();
    result.dynamic_dimensions = result
        .dynamic_dimensions
        .into_iter()
        .zip(&deleted)
        .filter_map(|(dynamic, &delete)| (!delete).then_some(dynamic))
        .collect();
    result.verify(ctx)?;
    let result = TensorType::instantiate(result, ctx).into();
    if array == shape {
        Ok(result)
    } else {
        super::buffer(ctx, result)
    }
}

/// Keep the axes selected by a predicate on original logical axis indices.
///
/// The predicate is evaluated once per axis in ascending order. As in
/// `ShapeUtil::FilterDimensions`, only arrays are accepted, not buffers.
pub fn filter_dimensions(
    ctx: &Context,
    shape: TypeHandle,
    mut keep: impl FnMut(i64) -> bool,
) -> Result<TypeHandle> {
    let source = tensor(ctx, shape)?;
    source.verify(ctx)?;
    let deleted = (0..source.dimensions.len())
        .map(|axis| axis as i64)
        .filter(|&axis| !keep(axis))
        .collect::<Vec<_>>();
    super::delete_dimensions(ctx, shape, &deleted)
}

/// Borrow the optional array or buffer layout without copying its metadata.
pub fn layout(ctx: &Context, shape: TypeHandle) -> Result<Option<Ref<'_, Layout>>> {
    let shape = super::array_view(ctx, shape)?;
    let ty = tensor(ctx, shape)?;
    Ok(Ref::filter_map(ty, |ty| ty.layout.first()).ok())
}

/// Rebuild an array or buffer with a checked physical layout.
pub fn with_layout(ctx: &Context, shape: TypeHandle, mut layout: Layout) -> Result<TypeHandle> {
    if let Some(ty) = shape.deref(ctx).downcast_ref::<BufferType>() {
        let array = super::with_layout(ctx, ty.array, layout)?;
        return super::buffer(ctx, array);
    }
    if layout.tail_padding_alignment_in_elements == 0 {
        layout.tail_padding_alignment_in_elements = 1;
    }
    validate_layout(&layout, &tensor(ctx, shape)?.dimensions)?;
    let source = tensor(ctx, shape)?;
    let ty = TensorType {
        element: source.element,
        dimensions: source.dimensions.clone(),
        dynamic_dimensions: source.dynamic_dimensions.clone(),
        layout: vec![layout],
    };
    ty.verify(ctx)?;
    Ok(TensorType::instantiate(ty, ctx).into())
}

/// Rebuild an array with its conventional dense descending layout.
pub fn with_default_layout(ctx: &Context, shape: TypeHandle) -> Result<TypeHandle> {
    super::with_layout(ctx, shape, Layout::descending(super::rank(ctx, shape)?))
}

/// Remove physical layouts recursively from arrays, tuples, and buffers.
pub fn without_layout(ctx: &Context, shape: TypeHandle) -> Result<TypeHandle> {
    if super::is_array(ctx, shape) {
        let source = tensor(ctx, shape)?;
        let ty = TensorType {
            element: source.element,
            dimensions: source.dimensions.clone(),
            dynamic_dimensions: source.dynamic_dimensions.clone(),
            layout: Vec::new(),
        };
        return Ok(TensorType::instantiate(ty, ctx).into());
    }
    if super::is_tuple(ctx, shape) {
        return super::tuple(
            ctx,
            super::children(ctx, shape)
                .iter()
                .map(|&ty| super::without_layout(ctx, ty))
                .collect::<Result<_>>()?,
        );
    }
    if shape.deref(ctx).is::<BufferType>() {
        return super::buffer(
            ctx,
            super::without_layout(ctx, super::array_view(ctx, shape)?)?,
        );
    }
    Ok(shape)
}

/// Rebuild an array with a new numeric format and normalized storage-bit metadata.
pub fn with_element_type(
    ctx: &Context,
    shape: TypeHandle,
    element: impl Into<PrimitiveType>,
) -> Result<TypeHandle> {
    let element = element.into().array_element()?;
    let mut ty = tensor(ctx, shape)?.clone();
    ty.element = element;
    if element.storage_bit_width() >= 8 {
        for layout in &mut ty.layout {
            layout.element_size_in_bits = 0;
        }
    }
    ty.verify(ctx)?;
    Ok(TensorType::instantiate(ty, ctx).into())
}
