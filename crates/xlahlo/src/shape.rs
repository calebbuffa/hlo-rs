//! Native shape operations on pliron type handles.
use crate::Diagnostic;
use crate::{
    primitive::PrimitiveType,
    types::{
        BufferType, ElementType, OpaqueType, TensorType, TokenType, TupleType, tensor,
        tensor_structure,
    },
};
use pliron::result::Result;
use pliron::{
    common_traits::Verify,
    context::Context,
    derive::pliron_attr,
    r#type::{Type, TypeHandle},
};
use std::{cell::Ref, slice};

/// Dimension extent for an unbounded dynamic axis.
pub const UNBOUNDED_SIZE: i64 = i64::MIN;
/// Tile marker combining a logical dimension with its neighbor.
pub const COMBINE_DIMENSION: i64 = i64::MIN;
/// Tile marker matching the corresponding bound.
pub const MATCH_DIMENSION: i64 = i64::MIN + 1;

#[pliron_attr(name = "xlahlo.dimension", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Static extent, bounded runtime extent, or unbounded dynamic extent.
pub enum Dimension {
    Static(i64),
    Bounded(i64),
    Unbounded,
}

#[pliron_attr(
    name = "xlahlo.tile",
    format = "`[` vec($dimensions, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Physical tiling dimensions in minor-to-major order.
pub struct Tile {
    pub dimensions: Vec<i64>,
}

#[pliron_attr(
    name = "xlahlo.split_config",
    format = "`<` $dimension `,` `[` vec($split_indices, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Split boundaries for one physical dimension.
pub struct SplitConfig {
    pub dimension: i64,
    pub split_indices: Vec<i64>,
}

#[pliron_attr(name = "xlahlo.dim_level_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Sparse level annotations retained independently from layout equality.
pub enum DimLevelType {
    Dense,
    Compressed,
    Singleton,
    LooseCompressed,
}

#[pliron_attr(
    name = "xlahlo.layout",
    format = "`<` `[` vec($minor_to_major, CharSpace(`,`)) `]` `,` `[` vec($tiles, CharSpace(`,`)) `]` `,` $tail_padding_alignment_in_elements `,` $element_size_in_bits `,` $memory_space `,` `[` vec($index_primitive_type, CharSpace(`,`)) `]` `,` `[` vec($pointer_primitive_type, CharSpace(`,`)) `]` `,` `[` vec($physical_shape, CharSpace(`,`)) `]` `,` `[` vec($split_configs, CharSpace(`,`)) `]` `,` $dynamic_shape_metadata_prefix_bytes `,` `[` vec($dim_level_types, CharSpace(`,`)) `]` `,` `[` vec($dim_unique, CharSpace(`,`)) `]` `,` `[` vec($dim_ordered, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Complete physical layout, including tiling, padding, memory space, and sparse metadata.
pub struct Layout {
    pub minor_to_major: Vec<i64>,
    pub tiles: Vec<Tile>,
    pub tail_padding_alignment_in_elements: i64,
    pub element_size_in_bits: i64,
    pub memory_space: i64,
    pub index_primitive_type: Vec<ElementType>,
    pub pointer_primitive_type: Vec<ElementType>,
    pub physical_shape: Vec<TypeHandle>,
    pub split_configs: Vec<SplitConfig>,
    pub dynamic_shape_metadata_prefix_bytes: i64,
    pub dim_level_types: Vec<DimLevelType>,
    pub dim_unique: Vec<bool>,
    pub dim_ordered: Vec<bool>,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            minor_to_major: Vec::new(),
            tiles: Vec::new(),
            tail_padding_alignment_in_elements: 1,
            element_size_in_bits: 0,
            memory_space: 0,
            index_primitive_type: Vec::new(),
            pointer_primitive_type: Vec::new(),
            physical_shape: Vec::new(),
            split_configs: Vec::new(),
            dynamic_shape_metadata_prefix_bytes: 0,
            dim_level_types: Vec::new(),
            dim_unique: Vec::new(),
            dim_ordered: Vec::new(),
        }
    }
}

impl Layout {
    /// Construct the conventional dense descending minor-to-major order.
    pub fn descending(rank: usize) -> Self {
        Self {
            minor_to_major: (0..rank as i64).rev().collect(),
            ..Self::default()
        }
    }
    /// Delete a logical axis and renumber logical and physical split axes.
    ///
    /// Like OpenXLA `Layout::DeleteDimension`, this leaves tiles and obsolete
    /// sparse annotations unchanged. Invalid axes or permutations are errors
    /// and leave this layout unchanged.
    pub fn delete_dimension(&mut self, dimension: i64) -> Result<()> {
        let rank = self.minor_to_major.len();
        let mut seen = vec![false; rank];
        for &axis in &self.minor_to_major {
            let valid = usize::try_from(axis)
                .ok()
                .and_then(|axis| seen.get_mut(axis))
                .is_some_and(|slot| !std::mem::replace(slot, true));
            if !valid {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "dimension deletion requires a valid layout permutation".into(),
                }));
            }
        }
        let position = self
            .minor_to_major
            .iter()
            .position(|&axis| axis == dimension)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: format!("layout dimension {dimension} out of range"),
                })
            })?;
        let physical_dimension = (rank - 1 - position) as i64;
        self.minor_to_major.remove(position);
        for axis in &mut self.minor_to_major {
            if *axis > dimension {
                *axis -= 1;
            }
        }
        self.split_configs.retain_mut(|split| {
            if split.dimension == physical_dimension {
                return false;
            }
            if split.dimension > physical_dimension {
                split.dimension -= 1;
            }
            true
        });
        Ok(())
    }
    fn equal(&self, ctx: &Context, other: &Self) -> bool {
        self.minor_to_major == other.minor_to_major
            && self.tiles == other.tiles
            && self.tail_padding_alignment_in_elements.max(1)
                == other.tail_padding_alignment_in_elements.max(1)
            && self.element_size_in_bits == other.element_size_in_bits
            && self.memory_space == other.memory_space
            && self.index_primitive_type == other.index_primitive_type
            && self.pointer_primitive_type == other.pointer_primitive_type
            && self.split_configs == other.split_configs
            && self.physical_shape.len() == other.physical_shape.len()
            && self
                .physical_shape
                .iter()
                .zip(&other.physical_shape)
                .all(|(&a, &b)| Shape::equal_including_layout(ctx, a, b))
    }
}

/// Compare layouts semantically, ignoring metadata prefixes and obsolete annotations.
pub fn layouts_equal(ctx: &Context, a: &Layout, b: &Layout) -> bool {
    a.equal(ctx, b)
}

/// Validate the pinned LayoutUtil rules, including sparse-layout restrictions.
pub fn validate_layout(layout: &Layout, dimensions: &[i64]) -> Result<()> {
    if layout.minor_to_major.is_empty() && layout.memory_space != 0 {
        return Ok(());
    }
    if layout.minor_to_major.len() != dimensions.len() {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "layout rank mismatch".into(),
        }));
    }
    let mut seen = vec![false; dimensions.len()];
    for &axis in &layout.minor_to_major {
        let axis = usize::try_from(axis).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "negative layout axis".into(),
            })
        })?;
        let slot = seen.get_mut(axis).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "layout axis out of range".into(),
            })
        })?;
        if std::mem::replace(slot, true) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "duplicate layout axis".into(),
            }));
        }
    }
    if layout.tail_padding_alignment_in_elements <= 0 || layout.element_size_in_bits < 0 {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "nonpositive layout alignment or negative element size".into(),
        }));
    }
    if !layout.index_primitive_type.is_empty()
        || !layout.pointer_primitive_type.is_empty()
        || !layout.physical_shape.is_empty()
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "sparse physical/index/pointer fields rejected by pinned LayoutUtil".into(),
        }));
    }
    for tile in &layout.tiles {
        if tile.dimensions.is_empty()
            || tile
                .dimensions
                .iter()
                .any(|&n| n <= 0 && n != COMBINE_DIMENSION && n != MATCH_DIMENSION)
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "invalid tile dimension".into(),
            }));
        }
    }
    let mut split_axes = vec![false; dimensions.len()];
    for split in &layout.split_configs {
        let axis = usize::try_from(split.dimension).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "negative split axis".into(),
            })
        })?;
        let slot = split_axes.get_mut(axis).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "split axis out of range".into(),
            })
        })?;
        if std::mem::replace(slot, true) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "duplicate split axis".into(),
            }));
        }
        let logical_axis = layout.minor_to_major[dimensions.len() - 1 - axis] as usize;
        let mut previous = 0;
        for &index in &split.split_indices {
            if index <= previous || index >= dimensions[logical_axis] {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "split indices must increase within the physical dimension".into(),
                }));
            }
            previous = index;
        }
    }
    Ok(())
}

/// A namespace, not a runtime shape wrapper: every shape is an interned pliron
/// TensorType, TupleType, TokenType, OpaqueType or BufferType.
pub struct Shape;

impl Shape {
    /// Construct a static array without imposing a physical layout.
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
    /// Construct a rank-zero array without imposing a physical layout.
    pub fn scalar(ctx: &Context, element: impl Into<PrimitiveType>) -> Result<TypeHandle> {
        Self::array(ctx, element, Vec::new())
    }
    /// Construct an array with explicit static, bounded, and unbounded dimensions.
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
    /// Construct a checked tuple of native HLO types.
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
    /// Construct a buffer whose storage is a native array type.
    pub fn buffer(ctx: &Context, array: TypeHandle) -> Result<TypeHandle> {
        BufferType::get(ctx, array)
    }
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
            Self::children(ctx, shape)
                .iter()
                .any(|&child| Self::has_unbounded(ctx, child))
        }
    }
    /// Whether every descendant array specifies a physical layout.
    pub fn has_complete_layout(ctx: &Context, shape: TypeHandle) -> Result<bool> {
        if Self::is_array_or_buffer(ctx, shape) {
            return Ok(Self::layout(ctx, shape)?.is_some());
        }
        for &child in Self::children(ctx, shape).iter() {
            if !Self::has_complete_layout(ctx, child)? {
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
    /// Borrow dimension bounds without copying the interned type.
    pub fn dimensions(ctx: &Context, shape: TypeHandle) -> Result<Ref<'_, [i64]>> {
        Ok(Ref::map(tensor(ctx, shape)?, |ty| ty.dimensions.as_slice()))
    }
    /// Return the numeric element format of an array.
    pub fn element_type(ctx: &Context, shape: TypeHandle) -> Result<ElementType> {
        Ok(tensor_structure(ctx, shape)?.element)
    }
    /// Return the primitive kind, including tuple, token, opaque, and buffer kinds.
    pub fn primitive_type(ctx: &Context, shape: TypeHandle) -> Result<PrimitiveType> {
        let ty = shape.deref(ctx);
        if let Some(ty) = ty.downcast_ref::<TensorType>() {
            return Ok(ty.element.into());
        }
        if ty.is::<TupleType>() {
            return Ok(PrimitiveType::TUPLE);
        }
        if ty.is::<TokenType>() {
            return Ok(PrimitiveType::TOKEN);
        }
        if ty.is::<OpaqueType>() {
            return Ok(PrimitiveType::OPAQUE_TYPE);
        }
        if ty.is::<BufferType>() {
            return Ok(PrimitiveType::BUFFER);
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
        if Self::is_tuple(ctx, shape) {
            Self::children(ctx, shape)
                .iter()
                .map(|&child| Self::leaf_count(ctx, child))
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
            Self::children(ctx, shape)
                .iter()
                .any(|&child| Self::is_dynamic(ctx, child))
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
    /// Replace an array or buffer dimension's bound, preserving its dynamic
    /// flag and all layout metadata, as in `Shape::set_dimensions` with no
    /// dynamic override. The original interned type is never modified.
    pub fn set_dimension(
        ctx: &Context,
        shape: TypeHandle,
        axis: usize,
        bound: i64,
    ) -> Result<TypeHandle> {
        let array = Self::array_view(ctx, shape)?;
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
            Self::buffer(ctx, result)
        }
    }
    /// Change an array or buffer axis's dynamic flag without changing its
    /// bound or layout. An unbounded dimension cannot be made static.
    pub fn set_dynamic_dimension(
        ctx: &Context,
        shape: TypeHandle,
        axis: usize,
        dynamic: bool,
    ) -> Result<TypeHandle> {
        let array = Self::array_view(ctx, shape)?;
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
            Self::buffer(ctx, result)
        }
    }
    /// Delete one array or buffer axis, preserving retained bounds, dynamic
    /// flags and the layout's OpenXLA dimension-deletion semantics.
    pub fn delete_dimension(
        ctx: &Context,
        shape: TypeHandle,
        dimension: i64,
    ) -> Result<TypeHandle> {
        Self::delete_dimensions(ctx, shape, &[dimension])
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
        let array = Self::array_view(ctx, shape)?;
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
            Self::buffer(ctx, result)
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
        Self::delete_dimensions(ctx, shape, &deleted)
    }
    /// Borrow the optional array or buffer layout without copying its metadata.
    pub fn layout(ctx: &Context, shape: TypeHandle) -> Result<Option<Ref<'_, Layout>>> {
        let shape = Self::array_view(ctx, shape)?;
        let ty = tensor(ctx, shape)?;
        Ok(Ref::filter_map(ty, |ty| ty.layout.first()).ok())
    }
    /// Rebuild an array or buffer with a checked physical layout.
    pub fn with_layout(ctx: &Context, shape: TypeHandle, mut layout: Layout) -> Result<TypeHandle> {
        if let Some(ty) = shape.deref(ctx).downcast_ref::<BufferType>() {
            let array = Self::with_layout(ctx, ty.array, layout)?;
            return Self::buffer(ctx, array);
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
        Self::with_layout(ctx, shape, Layout::descending(Self::rank(ctx, shape)?))
    }
    /// Remove physical layouts recursively from arrays, tuples, and buffers.
    pub fn without_layout(ctx: &Context, shape: TypeHandle) -> Result<TypeHandle> {
        if Self::is_array(ctx, shape) {
            let source = tensor(ctx, shape)?;
            let ty = TensorType {
                element: source.element,
                dimensions: source.dimensions.clone(),
                dynamic_dimensions: source.dynamic_dimensions.clone(),
                layout: Vec::new(),
            };
            return Ok(TensorType::instantiate(ty, ctx).into());
        }
        if Self::is_tuple(ctx, shape) {
            return Self::tuple(
                ctx,
                Self::children(ctx, shape)
                    .iter()
                    .map(|&ty| Self::without_layout(ctx, ty))
                    .collect::<Result<_>>()?,
            );
        }
        if shape.deref(ctx).is::<BufferType>() {
            return Self::buffer(
                ctx,
                Self::without_layout(ctx, Self::array_view(ctx, shape)?)?,
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
    /// Compare shape structure, ignoring layouts and dynamic flags; unbounded extents match any bound.
    pub fn compatible(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
        Self::compare(ctx, a, b, false, false, true)
    }
    /// Compare compatible shapes while also allowing differing floating-point precisions.
    pub fn compatible_ignoring_fp_precision(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
        Self::compare(ctx, a, b, true, false, true)
    }
    /// Compare types, bounds, and dynamic flags without considering physical layouts.
    pub fn equal_ignoring_layout(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
        Self::compare(ctx, a, b, false, false, false)
    }
    /// Compare structure and dynamic flags, ignoring layout and floating-point precision.
    pub fn equal_ignoring_layout_and_fp_precision(
        ctx: &Context,
        a: TypeHandle,
        b: TypeHandle,
    ) -> bool {
        Self::compare(ctx, a, b, true, false, false)
    }
    /// Compare complete shapes using semantic layout equality.
    pub fn equal_including_layout(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
        Self::compare(ctx, a, b, false, true, false)
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
            return (ta.element == tb.element
                || fp && ta.element.is_float() && tb.element.is_float())
                && ta.dimensions.len() == tb.dimensions.len()
                && ta.dimensions.iter().zip(&tb.dimensions).all(|(&a, &b)| {
                    a == b || compatible && (a == UNBOUNDED_SIZE || b == UNBOUNDED_SIZE)
                })
                && (compatible || ta.dynamic_dimensions == tb.dynamic_dimensions)
                && (!layout
                    || match (Self::layout(ctx, a), Self::layout(ctx, b)) {
                        (Ok(None), Ok(None)) => true,
                        (Ok(Some(a)), Ok(Some(b))) => a.equal(ctx, &b),
                        _ => false,
                    });
        }
        if at.get_type_id() != bt.get_type_id() {
            return false;
        }
        let ac = Self::children(ctx, a);
        let bc = Self::children(ctx, b);
        ac.len() == bc.len()
            && ac
                .iter()
                .zip(bc.iter())
                .all(|(&a, &b)| Self::compare(ctx, a, b, fp, layout, compatible))
    }
    /// Follow checked tuple indices, returning the root for an empty index.
    pub fn subshape(ctx: &Context, mut shape: TypeHandle, index: &[usize]) -> Result<TypeHandle> {
        for &axis in index {
            let next = shape
                .deref(ctx)
                .downcast_ref::<TupleType>()
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "index traverses non-tuple shape".into(),
                    })
                })?
                .elements
                .get(axis)
                .copied()
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "tuple index out of range".into(),
                    })
                })?;
            shape = next;
        }
        Ok(shape)
    }
    /// Return bounded byte storage size, using `pointer_size` for tuple entries.
    pub fn byte_size(ctx: &Context, shape: TypeHandle, pointer_size: i64) -> Result<i64> {
        let ty = shape.deref(ctx);
        if ty.is::<TupleType>() || ty.is::<OpaqueType>() {
            if pointer_size <= 0 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "tuple/opaque size requires positive pointer size".into(),
                }));
            }
            let count = ty
                .downcast_ref::<TupleType>()
                .map_or(1, |t| t.elements.len() as i64);
            return count.checked_mul(pointer_size).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "pointer table byte size overflow".into(),
                })
            });
        }
        if ty.is::<TokenType>() {
            return Ok(0);
        }
        let array = Self::array_view(ctx, shape)?;
        let count = Self::element_count(ctx, array)?;
        let size = Self::layout(ctx, array)?.map_or(0, |l| l.element_size_in_bits);
        if size != 0 {
            let bits = count.checked_mul(size).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "packed array bit size overflow".into(),
                })
            })?;
            Ok(bits / 8 + i64::from(bits % 8 != 0))
        } else {
            count
                .checked_mul(i64::from(
                    Self::element_type(ctx, array)?.bit_width().div_ceil(8),
                ))
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "array byte size overflow".into(),
                    })
                })
        }
    }
    /// Test dense stride compatibility without moving elements; tiles and padding are ignored.
    pub fn reshape_is_bitcast(
        ctx: &Context,
        input: TypeHandle,
        output: TypeHandle,
        ignore_element_type: bool,
    ) -> Result<bool> {
        let input = Self::array_view(ctx, input)?;
        let output = Self::array_view(ctx, output)?;
        fn strides(ctx: &Context, shape: TypeHandle) -> Result<Vec<i64>> {
            let dimensions = Shape::dimensions(ctx, shape)?;
            let layout = Shape::layout(ctx, shape)?.ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "reshape bitcast requires layouts".into(),
                })
            })?;
            if layout.minor_to_major.len() != dimensions.len() {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "reshape bitcast requires complete minor order".into(),
                }));
            }
            let mut result = vec![0; dimensions.len()];
            let mut stride = 1i64;
            for &axis in &layout.minor_to_major {
                result[axis as usize] = stride;
                stride = stride
                    .checked_mul(dimensions[axis as usize])
                    .ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                            message: "reshape stride overflow".into(),
                        })
                    })?;
            }
            Ok(result)
        }
        if Self::layout(ctx, input)?.is_none() || Self::layout(ctx, output)?.is_none() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "reshape bitcast requires layouts".into(),
            }));
        }
        if !ignore_element_type
            && Self::element_type(ctx, input)? != Self::element_type(ctx, output)?
        {
            return Ok(false);
        }
        let count = Self::element_count(ctx, input)?;
        if count != Self::element_count(ctx, output)? {
            return Ok(false);
        }
        if count == 0 {
            return Ok(true);
        }
        let input_strides = strides(ctx, input)?;
        let output_strides = strides(ctx, output)?;
        fn units(
            input: &[i64],
            output: &[i64],
            input_strides: &[i64],
            output_strides: &[i64],
        ) -> Result<bool> {
            let mut logical_stride = 1i64;
            for axis in (0..input.len()).rev() {
                if input[axis] > 1 {
                    let mut remaining = logical_stride;
                    let mut physical = 0i64;
                    for target in (0..output.len()).rev() {
                        physical = physical
                            .checked_add(
                                (remaining % output[target])
                                    .checked_mul(output_strides[target])
                                    .ok_or_else(|| {
                                        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                                            message: "reshape physical index overflow".into(),
                                        })
                                    })?,
                            )
                            .ok_or_else(|| {
                                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                                    message: "reshape physical index overflow".into(),
                                })
                            })?;
                        remaining /= output[target];
                    }
                    if physical != input_strides[axis] {
                        return Ok(false);
                    }
                }
                logical_stride = logical_stride.checked_mul(input[axis]).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "reshape logical stride overflow".into(),
                    })
                })?;
            }
            Ok(true)
        }
        let input = Self::dimensions(ctx, input)?;
        let output = Self::dimensions(ctx, output)?;
        Ok(units(&input, &output, &input_strides, &output_strides)?
            && units(&output, &input, &output_strides, &input_strides)?)
    }
    /// Pinned ArrayDataSize: through the last logical element, without trailing
    /// padding. Only the outer tile indexes; the second resolves MATCH packing.
    /// Return tiled array data bytes, excluding tail-alignment padding.
    pub fn array_data_size(ctx: &Context, shape: TypeHandle) -> Result<i64> {
        let shape = Self::array_view(ctx, shape)?;
        let dimensions = Self::dimensions(ctx, shape)?;
        let rank = dimensions.len();
        Self::element_count(ctx, shape)?;
        let layout = Self::layout(ctx, shape)?.ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "array data size requires layout".into(),
            })
        })?;
        if layout.minor_to_major.len() != rank {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "array data size requires full minor order".into(),
            }));
        }
        let overflow = || {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "array data size overflow".into(),
            })
        };
        let mul = |a: i64, b: i64| a.checked_mul(b).ok_or_else(overflow);
        let add = |a: i64, b: i64| a.checked_add(b).ok_or_else(overflow);
        let divide = |a: i64, b: i64| a.checked_div(b).ok_or_else(overflow);
        let remainder = |a: i64, b: i64| a.checked_rem(b).ok_or_else(overflow);
        let ceil =
            |a: i64, b: i64| -> Result<i64> { add(divide(a, b)?, i64::from(remainder(a, b)? > 0)) };
        let count = if rank == 0 {
            1
        } else if rank == 1 {
            dimensions[0]
        } else {
            let tiles = layout
                .tiles
                .first()
                .map_or(&[][..], |tile| tile.dimensions.as_slice());
            let sub_tiles = layout
                .tiles
                .get(1)
                .map_or(&[][..], |tile| tile.dimensions.as_slice());
            let resolve = |axis: usize, bound: i64| -> Result<i64> {
                let sub_axis = axis as isize + sub_tiles.len() as isize - tiles.len() as isize;
                let packing = if sub_axis >= 0 && (sub_axis as usize) < sub_tiles.len() {
                    sub_tiles[sub_axis as usize].max(1)
                } else if axis.checked_add(2) == Some(tiles.len()) && !sub_tiles.is_empty() {
                    sub_tiles[0].max(1)
                } else {
                    1
                };
                mul(ceil(bound.max(1), packing)?, packing)
            };
            let mut outer = 1;
            for (axis, &tile) in tiles.iter().enumerate() {
                let size = if tile == MATCH_DIMENSION {
                    let minor = tiles.len() - 1 - axis;
                    if minor >= rank {
                        continue;
                    }
                    resolve(axis, dimensions[layout.minor_to_major[minor] as usize])?
                } else {
                    tile
                };
                outer = mul(outer, size)?;
            }
            let (mut inner, mut index) = (1, 0);
            for (minor, &logical) in layout.minor_to_major.iter().enumerate() {
                let bound = dimensions[logical as usize];
                let coordinate = bound.checked_sub(1).ok_or_else(overflow)?;
                if minor < tiles.len() {
                    let axis = tiles.len() - 1 - minor;
                    let tile = if tiles[axis] == MATCH_DIMENSION {
                        resolve(axis, bound)?
                    } else {
                        tiles[axis]
                    };
                    index = add(
                        index,
                        add(
                            mul(outer, divide(coordinate, tile)?)?,
                            mul(inner, remainder(coordinate, tile)?)?,
                        )?,
                    )?;
                    outer = mul(outer, ceil(bound, tile)?)?;
                    inner = mul(inner, tile)?;
                } else {
                    index = add(index, mul(coordinate, outer)?)?;
                    outer = mul(outer, bound)?;
                }
            }
            add(index, 1)?
        };
        if count < 0 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "array data size is negative".into(),
            }));
        }
        if layout.element_size_in_bits != 0 {
            ceil(mul(count, layout.element_size_in_bits)?, 8)
        } else {
            mul(
                count,
                i64::from(Self::element_type(ctx, shape)?.bit_width().div_ceil(8)),
            )
        }
    }
}
