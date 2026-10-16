use protobuf::Enum;

use crate::{
    Error, Result,
    proto::xla_data::{LayoutProto, PrimitiveType, ShapeProto},
};

impl ShapeProto {
    pub(crate) fn array_storage(&self) -> std::borrow::Cow<'_, Self> {
        if self.element_type.enum_value() == Ok(PrimitiveType::BUFFER) {
            let child = &self.tuple_shapes[0];
            if let Some(layout) = self.layout.as_ref() {
                let mut child = child.clone();
                child.layout = Some(layout.clone()).into();
                std::borrow::Cow::Owned(child)
            } else {
                std::borrow::Cow::Borrowed(child)
            }
        } else {
            std::borrow::Cow::Borrowed(self)
        }
    }

    pub(crate) fn logical_children(&self) -> &[Self] {
        if matches!(
            self.element_type.enum_value(),
            Ok(PrimitiveType::TUPLE | PrimitiveType::BUFFER)
        ) {
            &self.tuple_shapes
        } else {
            &[]
        }
    }

    pub(crate) fn logical_dimensions(&self) -> &[i64] {
        if self
            .element_type
            .enum_value()
            .is_ok_and(PrimitiveType::is_array_element)
        {
            &self.dimensions
        } else {
            &[]
        }
    }
}

/// HLO's wire sentinel, not MLIR's `-1`.
pub const UNBOUNDED_SIZE: i64 = i64::MIN;
pub const COMBINE_DIMENSION: i64 = i64::MIN;
pub const MATCH_DIMENSION: i64 = i64::MIN + 1;
const MAX_SHAPE_DEPTH: usize = 128;

impl PrimitiveType {
    pub fn is_integral(self) -> bool {
        matches!(
            self,
            Self::S1
                | Self::S2
                | Self::S4
                | Self::S8
                | Self::S16
                | Self::S32
                | Self::S64
                | Self::U1
                | Self::U2
                | Self::U4
                | Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
        )
    }

    pub fn is_signed_integral(self) -> bool {
        matches!(
            self,
            Self::S1 | Self::S2 | Self::S4 | Self::S8 | Self::S16 | Self::S32 | Self::S64
        )
    }

    pub fn is_floating(self) -> bool {
        matches!(
            self,
            Self::F16
                | Self::BF16
                | Self::F32
                | Self::F64
                | Self::F8E5M2
                | Self::F8E4M3
                | Self::F8E4M3FN
                | Self::F8E4M3B11FNUZ
                | Self::F8E3M4
                | Self::F8E5M2FNUZ
                | Self::F8E4M3FNUZ
                | Self::F4E2M1FN
                | Self::F8E8M0FNU
                | Self::F6E3M2FN
                | Self::F6E2M3FN
        )
    }

    pub fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    pub fn same_ignoring_fp_precision(self, other: Self) -> bool {
        self == other || (self.is_floating() && other.is_floating())
    }

    pub fn is_array_element(self) -> bool {
        self == Self::PRED || self.is_integral() || self.is_floating() || self.is_complex()
    }

    pub fn bit_width(self) -> Option<u32> {
        use PrimitiveType::*;
        Some(match self {
            PRED | S1 | U1 => 1,
            S2 | U2 => 2,
            S4 | U4 | F4E2M1FN => 4,
            F6E3M2FN | F6E2M3FN => 6,
            S8 | U8 | F8E5M2 | F8E4M3 | F8E4M3FN | F8E4M3B11FNUZ | F8E3M4 | F8E5M2FNUZ
            | F8E4M3FNUZ | F8E8M0FNU => 8,
            S16 | U16 | F16 | BF16 => 16,
            S32 | U32 | F32 => 32,
            S64 | U64 | F64 | C64 => 64,
            C128 => 128,
            _ => return None,
        })
    }

    /// Match the pinned upstream range-first precision ordering.
    /// Distinct types with identical ordering keys are an upstream fatal case.
    pub fn higher_precision(self, other: Self) -> Result<Self> {
        fn properties(ty: PrimitiveType) -> Result<(bool, i32, i32, u32, bool)> {
            use PrimitiveType::*;
            if !ty.is_array_element() {
                return Err(Error::new(
                    "primitive",
                    "precision ordering requires array element types",
                ));
            }
            let component = match ty {
                C64 => F32,
                C128 => F64,
                _ => ty,
            };
            let (exponent, digits) = match component {
                F16 => (16, 11),
                BF16 => (128, 8),
                F32 => (128, 24),
                F64 => (1024, 53),
                F8E3M4 => (4, 5),
                F8E4M3 | F8E4M3FNUZ => (8, 4),
                F8E4M3FN => (9, 4),
                F8E4M3B11FNUZ => (5, 4),
                F8E5M2 | F8E5M2FNUZ => (16, 3),
                F8E8M0FNU => (128, 1),
                F4E2M1FN => (3, 2),
                F6E3M2FN => (5, 3),
                F6E2M3FN => (3, 4),
                _ => (-1, -1),
            };
            Ok((
                ty.is_complex(),
                exponent,
                digits,
                component
                    .bit_width()
                    .ok_or_else(|| Error::new("primitive", "missing element bit width"))?,
                component.is_signed_integral(),
            ))
        }
        match properties(self)?.cmp(&properties(other)?) {
            std::cmp::Ordering::Greater => Ok(self),
            std::cmp::Ordering::Less => Ok(other),
            std::cmp::Ordering::Equal if self == other => Ok(self),
            _ => Err(Error::new(
                "precision",
                "distinct primitive types tie in the upstream precision ordering",
            )),
        }
    }

    pub fn hlo_name(self) -> String {
        match self {
            Self::OPAQUE_TYPE => "opaque".into(),
            _ => format!("{self:?}").to_ascii_lowercase(),
        }
    }

    pub fn from_hlo_name(name: &str) -> Result<Self> {
        Self::VALUES
            .iter()
            .copied()
            .find(|ty| ty.hlo_name() == name && *ty != Self::PRIMITIVE_TYPE_INVALID)
            .ok_or_else(|| Error::new("primitive", format!("unknown primitive type {name:?}")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    Static(i64),
    Bounded(i64),
    Unbounded,
}

/// An immutable, validated shape. Raw metadata is retained except for pinned
/// OpenXLA import defaults, such as zero tail alignment becoming one.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape(ShapeProto);

impl Shape {
    pub fn array(element: PrimitiveType, dimensions: impl Into<Vec<i64>>) -> Result<Self> {
        Self::dynamic_array(
            element,
            dimensions
                .into()
                .into_iter()
                .map(Dimension::Static)
                .collect(),
        )
    }

    pub fn scalar(element: PrimitiveType) -> Result<Self> {
        Self::array(element, vec![])
    }

    pub fn dynamic_array(element: PrimitiveType, dimensions: Vec<Dimension>) -> Result<Self> {
        if !element.is_array_element() {
            return Err(Error::new("shape", "array requires an array element type"));
        }
        let mut wire = ShapeProto {
            element_type: element.into(),
            ..Default::default()
        };
        for dimension in dimensions {
            let (extent, dynamic) = match dimension {
                Dimension::Static(n) => (n, false),
                Dimension::Bounded(n) if n >= 0 => (n, true),
                Dimension::Bounded(_) => return Err(Error::new("shape", "negative dynamic bound")),
                Dimension::Unbounded => (UNBOUNDED_SIZE, true),
            };
            wire.dimensions.push(extent);
            wire.is_dynamic_dimension.push(dynamic);
        }
        Self::from_proto(wire)
    }

    pub fn tuple(elements: Vec<Self>) -> Result<Self> {
        Self::from_proto(ShapeProto {
            element_type: PrimitiveType::TUPLE.into(),
            tuple_shapes: elements.into_iter().map(Self::into_proto).collect(),
            ..Default::default()
        })
    }

    pub fn token() -> Self {
        Self(ShapeProto {
            element_type: PrimitiveType::TOKEN.into(),
            ..Default::default()
        })
    }

    pub fn opaque() -> Self {
        Self(ShapeProto {
            element_type: PrimitiveType::OPAQUE_TYPE.into(),
            ..Default::default()
        })
    }

    pub fn buffer(array: Self) -> Result<Self> {
        if !array.is_array() {
            return Err(Error::new("shape", "buffer must wrap an array"));
        }
        Self::from_proto(ShapeProto {
            element_type: PrimitiveType::BUFFER.into(),
            tuple_shapes: vec![array.into_proto()],
            ..Default::default()
        })
    }

    pub fn from_proto(wire: ShapeProto) -> Result<Self> {
        validate_import(&wire, 0)?;
        let mut wire = wire;
        normalize_layout_defaults(&mut wire);
        validate_shape(&wire, 0)?;
        Ok(Self(wire))
    }

    pub fn as_proto(&self) -> &ShapeProto {
        &self.0
    }
    pub fn into_proto(self) -> ShapeProto {
        self.0
    }

    pub fn element_type(&self) -> PrimitiveType {
        // Construction has already excluded unknown enum values.
        self.0.element_type.enum_value_or_default()
    }

    pub fn is_array(&self) -> bool {
        self.element_type().is_array_element()
    }
    pub fn is_tuple(&self) -> bool {
        self.element_type() == PrimitiveType::TUPLE
    }

    /// Number of non-tuple leaves. Empty tuples contribute no leaves.
    pub fn leaf_count(&self) -> usize {
        let mut count = 0;
        let mut pending = vec![&self.0];
        while let Some(shape) = pending.pop() {
            if shape.element_type == PrimitiveType::TUPLE.into() {
                pending.extend(&shape.tuple_shapes);
            } else {
                count += 1;
            }
        }
        count
    }
    pub fn dimensions(&self) -> &[i64] {
        self.0.logical_dimensions()
    }
    pub fn rank(&self) -> Result<usize> {
        if !self.is_array() {
            return Err(Error::new("shape", "expected array"));
        }
        Ok(self.0.dimensions.len())
    }

    pub fn dimension(&self, axis: usize) -> Result<Dimension> {
        let &size = self
            .dimensions()
            .get(axis)
            .ok_or_else(|| Error::new("dimension", "axis out of range"))?;
        Ok(if size == UNBOUNDED_SIZE {
            Dimension::Unbounded
        } else if self
            .0
            .is_dynamic_dimension
            .get(axis)
            .copied()
            .unwrap_or(false)
        {
            Dimension::Bounded(size)
        } else {
            Dimension::Static(size)
        })
    }

    pub fn is_dynamic(&self) -> bool {
        proto_is_dynamic(&self.0)
    }

    pub fn element_count(&self) -> Result<i64> {
        self.rank()?;
        if self.dimensions().contains(&UNBOUNDED_SIZE) {
            return Err(Error::new(
                "unbounded",
                "unbounded shape has no fixed element count",
            ));
        }
        self.dimensions().iter().try_fold(1_i64, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| Error::new("overflow", "shape element count overflow"))
        })
    }

    pub fn layout(&self) -> Option<&LayoutProto> {
        self.0.layout.as_ref().or_else(|| {
            (self.element_type() == PrimitiveType::BUFFER)
                .then(|| self.0.tuple_shapes[0].layout.as_ref())
                .flatten()
        })
    }

    /// The underlying array shape for an array or buffer; the wrapper is unchanged.
    pub fn array_view(&self) -> Result<Self> {
        let array = if self.element_type() == PrimitiveType::BUFFER {
            Self::from_proto(self.0.array_storage().into_owned())?
        } else {
            self.clone()
        };
        array.rank()?;
        Ok(array)
    }

    /// Pinned `ShapeUtil::ReshapeIsBitcast`, using `IndexUtil`'s dense indexing.
    /// Tiles, padding, memory spaces and dynamic flags do not affect this test.
    pub fn reshape_is_bitcast(&self, output: &Self, ignore_element_type: bool) -> Result<bool> {
        if self.element_type() == PrimitiveType::BUFFER
            || output.element_type() == PrimitiveType::BUFFER
        {
            return self
                .array_view()?
                .reshape_is_bitcast(&output.array_view()?, ignore_element_type);
        }
        fn strides(shape: &Shape) -> Result<Vec<i64>> {
            let rank = shape.rank()?;
            let layout = shape
                .layout()
                .ok_or_else(|| Error::new("layout", "reshape bitcast test requires layouts"))?;
            if layout.minor_to_major.len() != rank {
                return Err(Error::new(
                    "layout",
                    "reshape bitcast test requires a complete minor order",
                ));
            }
            let mut result = vec![0; rank];
            let mut stride = 1_i64;
            for &axis in &layout.minor_to_major {
                let axis = usize::try_from(axis)
                    .map_err(|_| Error::new("layout", "negative layout axis"))?;
                let slot = result
                    .get_mut(axis)
                    .ok_or_else(|| Error::new("layout", "layout axis exceeds rank"))?;
                if *slot != 0 {
                    return Err(Error::new("layout", "duplicate layout axis"));
                }
                *slot = stride;
                stride = stride
                    .checked_mul(shape.dimensions()[axis])
                    .ok_or_else(|| Error::new("overflow", "reshape bitcast stride overflow"))?;
            }
            Ok(result)
        }
        self.rank()?;
        output.rank()?;
        if self.layout().is_none() || output.layout().is_none() {
            return Err(Error::new(
                "layout",
                "reshape bitcast test requires layouts",
            ));
        }
        if !ignore_element_type && self.element_type() != output.element_type() {
            return Ok(false);
        }
        let count = self.element_count()?;
        if count != output.element_count()? {
            return Ok(false);
        }
        if count == 0 {
            return Ok(true);
        }
        let input_strides = strides(self)?;
        let output_strides = strides(output)?;
        fn units(
            input: &Shape,
            output: &Shape,
            input_strides: &[i64],
            output_strides: &[i64],
        ) -> Result<bool> {
            let mut logical_stride = 1_i64;
            for axis in (0..input.dimensions().len()).rev() {
                if input.dimensions()[axis] > 1 {
                    let mut remaining = logical_stride;
                    let mut physical = 0_i64;
                    for target in (0..output.dimensions().len()).rev() {
                        let size = output.dimensions()[target];
                        physical = physical
                            .checked_add(
                                (remaining % size)
                                    .checked_mul(output_strides[target])
                                    .ok_or_else(|| {
                                        Error::new("overflow", "reshape physical index overflow")
                                    })?,
                            )
                            .ok_or_else(|| {
                                Error::new("overflow", "reshape physical index overflow")
                            })?;
                        remaining /= size;
                    }
                    if physical != input_strides[axis] {
                        return Ok(false);
                    }
                }
                logical_stride = logical_stride
                    .checked_mul(input.dimensions()[axis])
                    .ok_or_else(|| Error::new("overflow", "reshape logical index overflow"))?;
            }
            Ok(true)
        }
        Ok(units(self, output, &input_strides, &output_strides)?
            && units(output, self, &output_strides, &input_strides)?)
    }

    /// Pinned `ShapeUtil::ByteSizeOf`, not a target's tiled/padded allocation size.
    /// Tuple tables and opaque values require an explicit positive pointer size.
    pub fn byte_size(&self, pointer_size: i64) -> Result<i64> {
        if self.is_tuple() || self.element_type() == PrimitiveType::OPAQUE_TYPE {
            if pointer_size <= 0 {
                return Err(Error::new(
                    "shape-size",
                    "tuple/opaque size requires a positive pointer size",
                ));
            }
            let count = if self.is_tuple() {
                i64::try_from(self.0.tuple_shapes.len())
                    .map_err(|_| Error::new("overflow", "tuple cardinality exceeds i64"))?
            } else {
                1
            };
            return count
                .checked_mul(pointer_size)
                .ok_or_else(|| Error::new("overflow", "pointer table byte size overflow"));
        }
        if self.element_type() == PrimitiveType::TOKEN {
            return Ok(0);
        }
        if self.element_type() == PrimitiveType::BUFFER {
            return self.array_view()?.byte_size(pointer_size);
        }
        let count = self.element_count()?;
        if let Some(layout) = self.layout()
            && layout.element_size_in_bits != 0
        {
            let bits = count
                .checked_mul(layout.element_size_in_bits)
                .ok_or_else(|| Error::new("overflow", "packed array bit size overflow"))?;
            return Ok(bits / 8 + i64::from(bits % 8 != 0));
        }
        count
            .checked_mul(i64::from(
                self.element_type()
                    .bit_width()
                    .ok_or_else(|| Error::new("shape-size", "element type has no fixed size"))?
                    .div_ceil(8),
            ))
            .ok_or_else(|| Error::new("overflow", "array byte size overflow"))
    }

    /// Bytes through the last logical element, excluding trailing padding.
    /// Matches `ShapeUtil::ArrayDataSize`: only the outer tile participates in
    /// indexing; a second tile can resolve MATCH dimensions' packing.
    pub fn array_data_size(&self) -> Result<i64> {
        if self.element_type() == PrimitiveType::BUFFER {
            return self.array_view()?.array_data_size();
        }
        let rank = self.rank()?;
        self.element_count()?;
        let layout = self
            .layout()
            .ok_or_else(|| Error::new("layout", "array data size requires a layout"))?;
        if layout.minor_to_major.len() != rank {
            return Err(Error::new(
                "layout",
                "array data size requires a full minor order",
            ));
        }
        let overflow = || Error::new("overflow", "array data size overflow");
        let mul = |a: i64, b: i64| a.checked_mul(b).ok_or_else(overflow);
        let add = |a: i64, b: i64| a.checked_add(b).ok_or_else(overflow);
        let divide = |a: i64, b: i64| a.checked_div(b).ok_or_else(overflow);
        let remainder = |a: i64, b: i64| a.checked_rem(b).ok_or_else(overflow);
        let ceil =
            |a: i64, b: i64| -> Result<i64> { add(divide(a, b)?, i64::from(remainder(a, b)? > 0)) };
        let count = if rank == 0 {
            1
        } else if rank == 1 {
            self.dimensions()[0]
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
                    resolve(
                        axis,
                        self.dimensions()[layout.minor_to_major[minor] as usize],
                    )?
                } else {
                    tile
                };
                outer = mul(outer, size)?;
            }
            let mut inner = 1;
            let mut index = 0;
            for (minor, &logical) in layout.minor_to_major.iter().enumerate() {
                let bound = self.dimensions()[logical as usize];
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
            return Err(Error::new("shape-size", "array data size is negative"));
        }
        if layout.element_size_in_bits != 0 {
            ceil(mul(count, layout.element_size_in_bits)?, 8)
        } else {
            mul(
                count,
                i64::from(
                    self.element_type()
                        .bit_width()
                        .ok_or_else(|| Error::new("shape-size", "element type has no fixed size"))?
                        .div_ceil(8),
                ),
            )
        }
    }

    pub fn with_layout(mut self, layout: LayoutProto) -> Result<Self> {
        self.0.layout = Some(layout).into();
        Self::from_proto(self.0)
    }

    pub fn with_default_layout(self) -> Result<Self> {
        let rank = self.rank()?;
        self.with_layout(LayoutProto {
            minor_to_major: (0..rank as i64).rev().collect(),
            tail_padding_alignment_in_elements: 1,
            ..Default::default()
        })
    }

    pub fn without_layout(&self) -> Self {
        let mut wire = self.0.clone();
        let mut pending = vec![&mut wire];
        while let Some(wire) = pending.pop() {
            wire.layout.clear();
            if matches!(
                wire.element_type.enum_value(),
                Ok(PrimitiveType::TUPLE | PrimitiveType::BUFFER)
            ) {
                pending.extend(wire.tuple_shapes.iter_mut());
            }
        }
        Self(wire)
    }

    pub fn with_element_type(&self, element: PrimitiveType) -> Result<Self> {
        self.rank()?;
        if !element.is_array_element() {
            return Err(Error::new("shape", "invalid array element replacement"));
        }
        let mut wire = self.0.clone();
        wire.element_type = element.into();
        if (element.bit_width().is_some_and(|bits| bits >= 8) || element == PrimitiveType::PRED)
            && let Some(layout) = wire.layout.as_mut()
        {
            layout.element_size_in_bits = 0;
        }
        Self::from_proto(wire)
    }

    /// Upstream ShapeUtil::Compatible: ignores layout and dynamic flags.
    /// Unbounded dimensions are wildcards; bounded/static extents must match.
    pub fn compatible(&self, other: &Self) -> bool {
        self.compatible_impl(other, false)
    }

    pub fn compatible_ignoring_fp_precision(&self, other: &Self) -> bool {
        self.compatible_impl(other, true)
    }

    pub fn equal_ignoring_layout(&self, other: &Self) -> bool {
        self.equal_ignoring_layout_impl(other, false)
    }

    pub fn equal_ignoring_layout_and_fp_precision(&self, other: &Self) -> bool {
        self.equal_ignoring_layout_impl(other, true)
    }

    fn equal_ignoring_layout_impl(&self, other: &Self, ignore_fp: bool) -> bool {
        let mut pending = vec![(&self.0, &other.0)];
        while let Some((a, b)) = pending.pop() {
            let same_type = a.element_type == b.element_type
                || (ignore_fp
                    && a.element_type
                        .enum_value()
                        .is_ok_and(PrimitiveType::is_floating)
                    && b.element_type
                        .enum_value()
                        .is_ok_and(PrimitiveType::is_floating));
            if !same_type
                || a.logical_dimensions() != b.logical_dimensions()
                || (0..a.logical_dimensions().len()).any(|axis| {
                    a.is_dynamic_dimension.get(axis).copied().unwrap_or(false)
                        != b.is_dynamic_dimension.get(axis).copied().unwrap_or(false)
                })
                || a.logical_children().len() != b.logical_children().len()
            {
                return false;
            }
            pending.extend(a.logical_children().iter().zip(b.logical_children()));
        }
        true
    }

    fn compatible_impl(&self, other: &Self, ignore_fp_precision: bool) -> bool {
        let mut pending = vec![(&self.0, &other.0)];
        while let Some((a, b)) = pending.pop() {
            let same_type = a.element_type == b.element_type
                || (ignore_fp_precision
                    && a.element_type
                        .enum_value()
                        .is_ok_and(PrimitiveType::is_floating)
                    && b.element_type
                        .enum_value()
                        .is_ok_and(PrimitiveType::is_floating));
            if !same_type
                || a.logical_dimensions().len() != b.logical_dimensions().len()
                || !a
                    .logical_dimensions()
                    .iter()
                    .zip(b.logical_dimensions())
                    .all(|(&a, &b)| a == b || a == UNBOUNDED_SIZE || b == UNBOUNDED_SIZE)
                || a.logical_children().len() != b.logical_children().len()
            {
                return false;
            }
            pending.extend(a.logical_children().iter().zip(b.logical_children()));
        }
        true
    }

    /// Exact logical and layout equality under pinned C++ semantics.
    /// Unlike wire equality (`PartialEq`), unknown fields, obsolete layout
    /// annotations and dynamic metadata prefix sizes do not participate.
    pub fn equal_including_layout(&self, other: &Self) -> bool {
        shapes_equal_including_layout(&self.0, &other.0)
    }

    pub fn subshape(&self, index: &[usize]) -> Result<Self> {
        let mut current = &self.0;
        for &axis in index {
            if current.element_type.enum_value_or_default() != PrimitiveType::TUPLE {
                return Err(Error::new(
                    "shape-index",
                    "index traverses a non-tuple shape",
                ));
            }
            current = current
                .tuple_shapes
                .get(axis)
                .ok_or_else(|| Error::new("shape-index", "tuple index out of range"))?;
        }
        Ok(Self(current.clone()))
    }
}

pub(crate) fn layouts_equal(a: &LayoutProto, b: &LayoutProto) -> bool {
    layout_fields_equal(a, b)
        && match (a.physical_shape.as_ref(), b.physical_shape.as_ref()) {
            (None, None) => true,
            (Some(a), Some(b)) => shapes_equal_including_layout(a, b),
            _ => false,
        }
}

fn layout_fields_equal(a: &LayoutProto, b: &LayoutProto) -> bool {
    a.minor_to_major == b.minor_to_major
        && a.tail_padding_alignment_in_elements.max(1)
            == b.tail_padding_alignment_in_elements.max(1)
        && a.element_size_in_bits == b.element_size_in_bits
        && a.memory_space == b.memory_space
        && a.index_primitive_type == b.index_primitive_type
        && a.pointer_primitive_type == b.pointer_primitive_type
        && a.tiles.len() == b.tiles.len()
        && a.tiles
            .iter()
            .zip(&b.tiles)
            .all(|(a, b)| a.dimensions == b.dimensions)
        && a.split_configs.len() == b.split_configs.len()
        && a.split_configs
            .iter()
            .zip(&b.split_configs)
            .all(|(a, b)| a.dimension == b.dimension && a.split_indices == b.split_indices)
}

fn shapes_equal_including_layout(a: &ShapeProto, b: &ShapeProto) -> bool {
    let mut pending = vec![(a.clone(), b.clone())];
    while let Some((a, b)) = pending.pop() {
        if a.element_type.enum_value() == Ok(PrimitiveType::BUFFER) {
            pending.push((
                a.array_storage().into_owned(),
                b.array_storage().into_owned(),
            ));
            continue;
        }
        if a.element_type != b.element_type
            || a.logical_dimensions() != b.logical_dimensions()
            || (0..a.logical_dimensions().len()).any(|axis| {
                a.is_dynamic_dimension.get(axis).copied().unwrap_or(false)
                    != b.is_dynamic_dimension.get(axis).copied().unwrap_or(false)
            })
            || a.logical_children().len() != b.logical_children().len()
        {
            return false;
        }
        match (a.layout.as_ref(), b.layout.as_ref()) {
            (None, None) => {}
            (Some(a), Some(b)) if layout_fields_equal(a, b) => {
                match (a.physical_shape.as_ref(), b.physical_shape.as_ref()) {
                    (None, None) => {}
                    (Some(a), Some(b)) => pending.push((a.clone(), b.clone())),
                    _ => return false,
                }
            }
            _ => return false,
        }
        pending.extend(
            a.logical_children()
                .iter()
                .cloned()
                .zip(b.logical_children().iter().cloned()),
        );
    }
    true
}

fn proto_is_dynamic(wire: &ShapeProto) -> bool {
    let mut pending = vec![wire];
    while let Some(shape) = pending.pop() {
        if shape
            .element_type
            .enum_value()
            .is_ok_and(PrimitiveType::is_array_element)
        {
            if shape.is_dynamic_dimension.iter().any(|dynamic| *dynamic) {
                return true;
            }
        } else {
            pending.extend(shape.logical_children());
        }
    }
    false
}

fn validate_import(wire: &ShapeProto, depth: usize) -> Result<()> {
    let mut pending = vec![(wire, depth)];
    while let Some((wire, depth)) = pending.pop() {
        if depth >= MAX_SHAPE_DEPTH {
            return Err(Error::new("resource-limit", "shape import nesting limit"));
        }
        let element = wire
            .element_type
            .enum_value()
            .map_err(|_| Error::new("shape", "unknown primitive enum value"))?;
        if element.is_array_element() {
            if !wire.is_dynamic_dimension.is_empty()
                && wire.is_dynamic_dimension.len() != wire.dimensions.len()
            {
                return Err(Error::new("shape", "array field cardinality mismatch"));
            }
        } else if element == PrimitiveType::BUFFER
            && (wire.tuple_shapes.len() != 1
                || !wire.tuple_shapes[0]
                    .element_type
                    .enum_value()
                    .is_ok_and(PrimitiveType::is_array_element))
        {
            return Err(Error::new("shape", "buffer requires one array shape"));
        }
        pending.extend(
            wire.logical_children()
                .iter()
                .map(|child| (child, depth + 1)),
        );
        if let Some(layout) = wire.layout.as_ref() {
            if !element.is_array_element() && element != PrimitiveType::BUFFER {
                return Err(Error::new("shape", "nonarray shape has layout"));
            }
            if layout.tail_padding_alignment_in_elements < 0 {
                return Err(Error::new("layout", "negative imported tail alignment"));
            }
            if layout
                .tiles
                .iter()
                .flat_map(|tile| &tile.dimensions)
                .any(|&dimension| {
                    dimension < 0 && dimension != COMBINE_DIMENSION && dimension != MATCH_DIMENSION
                })
            {
                return Err(Error::new("layout", "invalid imported tile dimension"));
            }
            if let Some(physical) = layout.physical_shape.as_ref() {
                pending.push((physical, depth + 1));
            }
        }
    }
    Ok(())
}

fn normalize_layout_defaults(wire: &mut ShapeProto) {
    let mut pending = vec![wire];
    while let Some(wire) = pending.pop() {
        if let Some(layout) = wire.layout.as_mut() {
            if layout.tail_padding_alignment_in_elements == 0 {
                layout.tail_padding_alignment_in_elements = 1;
            }
            if let Some(physical_shape) = layout.physical_shape.as_mut() {
                pending.push(physical_shape);
            }
        }
        if matches!(
            wire.element_type.enum_value(),
            Ok(PrimitiveType::TUPLE | PrimitiveType::BUFFER)
        ) {
            pending.extend(&mut wire.tuple_shapes);
        }
    }
}

fn validate_shape(wire: &ShapeProto, depth: usize) -> Result<()> {
    let mut pending = vec![(std::borrow::Cow::Borrowed(wire), depth)];
    while let Some((wire, depth)) = pending.pop() {
        if depth >= MAX_SHAPE_DEPTH {
            return Err(Error::new("resource-limit", "shape nesting limit"));
        }
        let element = wire
            .element_type
            .enum_value()
            .map_err(|_| Error::new("shape", "unknown primitive enum value"))?;
        match element {
            PrimitiveType::TUPLE => {
                if wire.layout.is_some() {
                    return Err(Error::new("shape", "tuple has layout"));
                }
                pending.extend(
                    wire.tuple_shapes
                        .iter()
                        .cloned()
                        .map(|child| (std::borrow::Cow::Owned(child), depth + 1)),
                );
            }
            PrimitiveType::BUFFER => {
                if wire.tuple_shapes.len() != 1 {
                    return Err(Error::new(
                        "shape",
                        "buffer requires exactly one array shape",
                    ));
                }
                pending.push((
                    std::borrow::Cow::Owned(wire.array_storage().into_owned()),
                    depth + 1,
                ));
                if !wire.tuple_shapes[0]
                    .element_type
                    .enum_value_or_default()
                    .is_array_element()
                {
                    return Err(Error::new("shape", "buffer must wrap array"));
                }
            }
            PrimitiveType::TOKEN | PrimitiveType::OPAQUE_TYPE => {
                if wire.layout.is_some() {
                    return Err(Error::new("shape", "token/opaque has layout"));
                }
            }
            ty if ty.is_array_element() => {
                if !wire.is_dynamic_dimension.is_empty()
                    && wire.is_dynamic_dimension.len() != wire.dimensions.len()
                {
                    return Err(Error::new("shape", "array field cardinality mismatch"));
                }
                let mut count = 1_i64;
                for (axis, &size) in wire.dimensions.iter().enumerate() {
                    if size == UNBOUNDED_SIZE {
                        if !wire
                            .is_dynamic_dimension
                            .get(axis)
                            .copied()
                            .unwrap_or(false)
                        {
                            return Err(Error::new("shape", "unbounded dimension must be dynamic"));
                        }
                    } else {
                        if size < 0 {
                            return Err(Error::new("shape", "negative dimension"));
                        }
                        count = count
                            .checked_mul(size)
                            .ok_or_else(|| Error::new("overflow", "shape dimensions overflow"))?;
                    }
                }
                count
                    .checked_mul(i64::from(ty.bit_width().unwrap_or(0).div_ceil(8)))
                    .ok_or_else(|| Error::new("overflow", "shape byte size overflow"))?;
                if let Some(layout) = wire.layout.as_ref() {
                    validate_layout(layout, &wire.dimensions)?;
                }
            }
            _ => return Err(Error::new("shape", "invalid element type")),
        }
    }
    Ok(())
}

/// Mirrors the pinned LayoutUtil dense-layout checks. Sparse fields are retained
/// by wire messages, but rejected here as they are by that pinned validator.
pub fn validate_layout(layout: &LayoutProto, dimensions: &[i64]) -> Result<()> {
    let fail = |message| Error::new("layout", message);
    if layout.minor_to_major.is_empty() && layout.memory_space != 0 {
        return Ok(());
    }
    if layout.minor_to_major.len() != dimensions.len() {
        return Err(fail("layout rank mismatch"));
    }
    let mut seen = vec![false; dimensions.len()];
    for &axis in &layout.minor_to_major {
        let axis = usize::try_from(axis).map_err(|_| fail("negative layout axis"))?;
        let slot = seen
            .get_mut(axis)
            .ok_or_else(|| fail("layout axis out of range"))?;
        if std::mem::replace(slot, true) {
            return Err(fail("duplicate layout axis"));
        }
    }
    if layout.tail_padding_alignment_in_elements < 0 {
        return Err(fail("negative tail padding alignment"));
    }
    // Layout::FromProto normalizes zero to the C++ default of one.
    if layout.element_size_in_bits < 0 {
        return Err(fail("negative element size"));
    }
    if layout.index_primitive_type.value() != 0
        || layout.pointer_primitive_type.value() != 0
        || layout.physical_shape.is_some()
    {
        return Err(fail(
            "sparse physical/index/pointer fields rejected by pinned LayoutUtil",
        ));
    }
    for tile in &layout.tiles {
        if tile.dimensions.is_empty() {
            return Err(fail("empty tile"));
        }
        if tile
            .dimensions
            .iter()
            .any(|&n| n <= 0 && n != COMBINE_DIMENSION && n != MATCH_DIMENSION)
        {
            return Err(fail("invalid tile dimension"));
        }
    }
    let mut split_axes = vec![false; dimensions.len()];
    for split in &layout.split_configs {
        let axis = usize::try_from(split.dimension).map_err(|_| fail("negative split axis"))?;
        let slot = split_axes
            .get_mut(axis)
            .ok_or_else(|| fail("split axis out of range"))?;
        if std::mem::replace(slot, true) {
            return Err(fail("duplicate split axis"));
        }
        let logical_axis = layout.minor_to_major[dimensions.len() - 1 - axis] as usize;
        let mut previous = 0;
        for &index in &split.split_indices {
            if index <= previous || index >= dimensions[logical_axis] {
                return Err(fail(
                    "split indices must increase within the physical dimension",
                ));
            }
            previous = index;
        }
    }
    Ok(())
}
