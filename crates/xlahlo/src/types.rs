//! Interned native HLO types and numeric array element formats.
use crate::Diagnostic;

use self::ElementType::*;
use crate::shape::{COMBINE_DIMENSION, Layout, MATCH_DIMENSION, UNBOUNDED_SIZE, validate_layout};
use pliron::{
    common_traits::Verify,
    context::Context,
    derive::{pliron_attr, pliron_type},
    result::Result,
    r#type::{Type, TypeHandle},
};
use std::{cell::Ref, cmp::Ordering};

#[pliron_attr(name = "xlahlo.element", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Numeric formats accepted as tensor elements; structural kinds use separate types.
pub enum ElementType {
    Pred,
    S1,
    S2,
    S4,
    S8,
    S16,
    S32,
    S64,
    U8,
    U1,
    U2,
    U4,
    U16,
    U32,
    U64,
    F16,
    BF16,
    F32,
    F64,
    C64,
    C128,
    F8E5M2,
    F8E4M3,
    F8E4M3FN,
    F8E4M3B11FNUZ,
    F8E3M4,
    F8E5M2FNUZ,
    F8E4M3FNUZ,
    F4E2M1FN,
    F8E8M0FNU,
    F6E3M2FN,
    F6E2M3FN,
}

impl ElementType {
    /// Number of significant encoding bits per element.
    pub fn bit_width(self) -> u32 {
        match self {
            Self::Pred | Self::S1 | Self::U1 => 1,
            Self::S2 | Self::U2 => 2,
            Self::S4 | Self::U4 | Self::F4E2M1FN => 4,
            Self::F6E3M2FN | Self::F6E2M3FN => 6,
            Self::S8
            | Self::U8
            | Self::F8E5M2
            | Self::F8E4M3
            | Self::F8E4M3FN
            | Self::F8E4M3B11FNUZ
            | Self::F8E3M4
            | Self::F8E5M2FNUZ
            | Self::F8E4M3FNUZ
            | Self::F8E8M0FNU => 8,
            Self::S16 | Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::S32 | Self::U32 | Self::F32 => 32,
            Self::S64 | Self::U64 | Self::F64 | Self::C64 => 64,
            Self::C128 => 128,
        }
    }

    /// Whether this is a signed or unsigned integer format, excluding predicates.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::S1
                | Self::S2
                | Self::S4
                | Self::U1
                | Self::U2
                | Self::U4
                | Self::S8
                | Self::S16
                | Self::S32
                | Self::S64
                | Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
        )
    }

    /// Whether this is a signed integer format.
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            Self::S1 | Self::S2 | Self::S4 | Self::S8 | Self::S16 | Self::S32 | Self::S64
        )
    }

    /// Whether this is a real floating-point format.
    pub fn is_float(self) -> bool {
        !self.is_integer() && self != Self::Pred && !self.is_complex()
    }

    /// Whether this format stores a pair of real floating-point components.
    pub fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }

    /// Compare formats, treating all real floating-point precisions as equivalent.
    pub fn same_ignoring_fp_precision(self, other: Self) -> bool {
        self == other || self.is_float() && other.is_float()
    }
    /// Storage width used for precision ordering and shape storage-bit normalization.
    pub fn storage_bit_width(self) -> u32 {
        match self {
            Self::Pred | Self::F6E3M2FN | Self::F6E2M3FN => 8,
            _ => self.bit_width(),
        }
    }
    /// Return the lowercase HLO spelling.
    pub fn hlo_name(self) -> String {
        format!("{self:?}").to_ascii_lowercase()
    }
    /// Parse an exact lowercase HLO format name.
    pub fn from_hlo_name(name: &str) -> Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|ty| ty.hlo_name() == name)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: format!("unknown primitive type {name:?}"),
                })
            })
    }
    /// Every supported numeric array format.
    pub const ALL: &'static [Self] = &[
        Self::Pred,
        Self::S1,
        Self::S2,
        Self::S4,
        Self::S8,
        Self::S16,
        Self::S32,
        Self::S64,
        Self::U1,
        Self::U2,
        Self::U4,
        Self::U8,
        Self::U16,
        Self::U32,
        Self::U64,
        Self::F16,
        Self::BF16,
        Self::F32,
        Self::F64,
        Self::C64,
        Self::C128,
        Self::F8E5M2,
        Self::F8E4M3,
        Self::F8E4M3FN,
        Self::F8E4M3B11FNUZ,
        Self::F8E3M4,
        Self::F8E5M2FNUZ,
        Self::F8E4M3FNUZ,
        Self::F4E2M1FN,
        Self::F8E8M0FNU,
        Self::F6E3M2FN,
        Self::F6E2M3FN,
    ];
    /// Select by component range, precision, storage width, and signedness.
    ///
    /// Distinct formats tied by all ordering properties are rejected.
    pub fn higher_precision(self, other: Self) -> Result<Self> {
        fn properties(ty: ElementType) -> (bool, i32, i32, u32, bool) {
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
            (
                ty.is_complex(),
                exponent,
                digits,
                if component == Pred {
                    1
                } else {
                    component.storage_bit_width()
                },
                component.is_signed(),
            )
        }
        match properties(self).cmp(&properties(other)) {
            Ordering::Greater => Ok(self),
            Ordering::Less => Ok(other),
            Ordering::Equal if self == other => Ok(self),
            _ => Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "distinct primitive types tie in precision ordering".into(),
            })),
        }
    }
}

#[pliron_type(
    name = "xlahlo.tensor",
    format = "`<` $element `,` `[` vec($dimensions, CharSpace(`,`)) `]` `,` `[` vec($layout, CharSpace(`,`)) `]` `,` `[` vec($dynamic_dimensions, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Array shape with bounded dimensions, dynamic flags, and one optional layout.
pub struct TensorType {
    pub element: ElementType,
    pub dimensions: Vec<i64>,
    /// Zero or one complete layout records; empty means layout is unspecified.
    pub layout: Vec<Layout>,
    pub dynamic_dimensions: Vec<bool>,
}

impl TensorType {
    /// Construct a static array with a descending dense layout.
    pub fn get(ctx: &Context, element: ElementType, dimensions: Vec<i64>) -> Result<TypeHandle> {
        let layout = (0..dimensions.len()).rev().collect();
        Self::with_layout(ctx, element, dimensions, layout)
    }

    /// Construct a static array with an explicit dense minor order.
    ///
    /// An empty order leaves non-scalar layout unspecified; scalars receive
    /// an explicit rank-zero layout.
    pub fn with_layout(
        ctx: &Context,
        element: ElementType,
        dimensions: Vec<i64>,
        layout: Vec<usize>,
    ) -> Result<TypeHandle> {
        let layout = if dimensions.is_empty() || !layout.is_empty() {
            vec![Layout {
                minor_to_major: layout
                    .into_iter()
                    .map(|axis| {
                        i64::try_from(axis).map_err(|_| {
                            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                                message: "layout axis overflow".into(),
                            })
                        })
                    })
                    .collect::<Result<_>>()?,
                ..Layout::default()
            }]
        } else {
            Vec::new()
        };
        let ty = Self {
            element,
            dynamic_dimensions: vec![false; dimensions.len()],
            dimensions,
            layout,
        };
        ty.verify(ctx)?;
        Ok(Self::instantiate(ty, ctx).into())
    }
}

impl Verify for TensorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.verify_at_depth(ctx, 0)
    }
}

impl TensorType {
    fn verify_at_depth(&self, ctx: &Context, depth: usize) -> Result<()> {
        if depth >= 128 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "shape nesting limit".into(),
            }));
        }
        if self.dynamic_dimensions.len() != self.dimensions.len()
            || self
                .dimensions
                .iter()
                .zip(&self.dynamic_dimensions)
                .any(|(&n, &dynamic)| n < 0 && !(n == UNBOUNDED_SIZE && dynamic))
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "invalid HLO dimension or dynamic flag cardinality".into(),
            }));
        }
        let count = self
            .dimensions
            .iter()
            .filter(|&&n| n != UNBOUNDED_SIZE)
            .try_fold(1i64, |count, &n| count.checked_mul(n));
        if count
            .and_then(|n| n.checked_mul(i64::from(self.element.bit_width().div_ceil(8))))
            .is_none()
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "shape byte size overflow".into(),
            }));
        }
        if self.layout.len() > 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "array has multiple layouts".into(),
            }));
        }
        if let Some(layout) = self.layout.first() {
            if layout.tail_padding_alignment_in_elements < 0
                || layout
                    .tiles
                    .iter()
                    .flat_map(|tile| &tile.dimensions)
                    .any(|&n| n < 0 && n != COMBINE_DIMENSION && n != MATCH_DIMENSION)
            {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "invalid imported layout alignment or tile dimension".into(),
                }));
            }
            if layout.index_primitive_type.len() > 1
                || layout.pointer_primitive_type.len() > 1
                || layout.physical_shape.len() > 1
            {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "multiple singleton layout fields".into(),
                }));
            }
            for &physical in &layout.physical_shape {
                verify_shape_tree(ctx, physical, depth + 1)?;
            }
            validate_layout(layout, &self.dimensions)?;
        }
        Ok(())
    }
}

#[pliron_type(
    name = "xlahlo.tuple",
    format = "`<` vec($elements, CharSpace(`,`)) `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Ordered native HLO element types, including nested tuples.
pub struct TupleType {
    pub elements: Vec<TypeHandle>,
}

impl TupleType {
    /// Intern a tuple after validating its entire type tree.
    pub fn get(ctx: &Context, elements: Vec<TypeHandle>) -> Result<TypeHandle> {
        let ty = Self { elements };
        ty.verify(ctx)?;
        Ok(Self::instantiate(ty, ctx).into())
    }
}

impl Verify for TupleType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for &ty in &self.elements {
            verify_shape_tree(ctx, ty, 1)?;
        }
        Ok(())
    }
}

fn verify_shape_tree(ctx: &Context, root: TypeHandle, initial_depth: usize) -> Result<()> {
    let mut pending = vec![(root, initial_depth)];
    while let Some((ty, depth)) = pending.pop() {
        if depth >= 128 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "shape nesting limit".into(),
            }));
        }
        let ty = ty.deref(ctx);
        if let Some(ty) = ty.downcast_ref::<TensorType>() {
            ty.verify_at_depth(ctx, depth)?;
        } else if let Some(ty) = ty.downcast_ref::<TupleType>() {
            pending.extend(ty.elements.iter().map(|&ty| (ty, depth + 1)));
        } else if let Some(ty) = ty.downcast_ref::<BufferType>() {
            tensor(ctx, ty.array)?;
            pending.push((ty.array, depth + 1));
        } else if !ty.is::<TokenType>() && !ty.is::<OpaqueType>() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "expected native HLO type".into(),
            }));
        }
    }
    Ok(())
}

/// Whether a handle denotes an HLO array, tuple, token, opaque, or buffer type.
pub fn is_hlo_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.is::<TensorType>()
        || ty.is::<TupleType>()
        || ty.is::<TokenType>()
        || ty.is::<OpaqueType>()
        || ty.is::<BufferType>()
}

#[pliron_type(name = "xlahlo.token", format = "`<` `>`", verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Sequencing dependency without an array payload.
pub struct TokenType;

impl TokenType {
    /// Return the context's interned token type.
    pub fn get(ctx: &Context) -> TypeHandle {
        Self::instantiate(Self, ctx).into()
    }
}

#[pliron_type(name = "xlahlo.opaque", format = "`<` `>`", verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Opaque non-array value with no inspectable element structure.
pub struct OpaqueType;

impl OpaqueType {
    /// Return the context's interned opaque type.
    pub fn get(ctx: &Context) -> TypeHandle {
        Self::instantiate(Self, ctx).into()
    }
}

#[pliron_type(name = "xlahlo.buffer", format = "`<` $array `>`")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Mutable storage whose underlying shape must be a native array.
pub struct BufferType {
    pub array: TypeHandle,
}

impl BufferType {
    /// Intern a buffer after checking its array storage type.
    pub fn get(ctx: &Context, array: TypeHandle) -> Result<TypeHandle> {
        let ty = Self { array };
        ty.verify(ctx)?;
        Ok(Self::instantiate(ty, ctx).into())
    }
}

impl Verify for BufferType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        tensor(ctx, self.array)?;
        verify_shape_tree(ctx, self.array, 1)
    }
}

/// Borrow an array type, rejecting foreign types and malformed index-bearing fields.
///
/// Handles must be live and belong to `ctx`. This lightweight guard does not
/// recursively verify physical shapes or repeat full layout verification.
pub fn tensor(ctx: &Context, ty: TypeHandle) -> Result<Ref<'_, TensorType>> {
    let ty = tensor_structure(ctx, ty)?;
    if ty
        .dimensions
        .iter()
        .zip(&ty.dynamic_dimensions)
        .any(|(&size, &dynamic)| size < 0 && !(size == UNBOUNDED_SIZE && dynamic))
        || ty.layout.first().is_some_and(|layout| {
            layout.minor_to_major.iter().any(|&axis| {
                axis < 0 || usize::try_from(axis).map_or(true, |axis| axis >= ty.dimensions.len())
            })
        })
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "malformed array dimensions, dynamic flags, or layout axes".into(),
        }));
    }
    Ok(ty)
}

/// Constant-time structural guard for queries that check only their requested axis.
pub(crate) fn tensor_structure(ctx: &Context, ty: TypeHandle) -> Result<Ref<'_, TensorType>> {
    let ty =
        Ref::filter_map(ty.deref(ctx), |ty| ty.downcast_ref::<TensorType>()).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "expected HLO tensor".into(),
            })
        })?;
    if ty.dimensions.len() != ty.dynamic_dimensions.len()
        || ty.layout.len() > 1
        || ty.layout.first().is_some_and(|layout| {
            (!layout.minor_to_major.is_empty() || layout.memory_space == 0)
                && layout.minor_to_major.len() != ty.dimensions.len()
        })
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: "malformed array dimensions, dynamic flags, or layout axes".into(),
        }));
    }
    Ok(ty)
}
