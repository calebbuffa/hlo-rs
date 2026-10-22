//! Exact native literals, stored in physical minor-to-major order.

pub mod semantics;

use crate::Diagnostic;
use crate::{
    shape::{self, Dimension},
    types::{ElementType, TokenType},
};
use pliron::{
    common_traits::Verify, context::Context, derive::pliron_attr, result::Result,
    r#type::TypeHandle,
};

#[pliron_attr(
    name = "xlahlo.typed_literal",
    format = "`<` $shape `,` `[` vec($storage, CharSpace(`,`)) `]` `,` `[` vec($tuple_literals, CharSpace(`,`)) `]` `,` `[` vec($dynamic_sizes, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Exact physical storage and runtime bounds for an array, tuple, or token literal.
///
/// Arrays carry bytes and, for dynamic shapes, one runtime size per axis.
/// Tuples carry only children, and tokens have no payload. Public fields permit
/// direct construction; `Verify::verify` checks these invariants before use.
pub struct Literal {
    /// Interned array, tuple, or token type that determines payload structure.
    pub shape: TypeHandle,
    /// Little-endian exact encoding; sub-byte elements occupy one byte and
    /// retain unused storage bits. Complex real components precede imaginary.
    pub storage: Vec<u8>,
    /// Ordered child literals for a tuple; empty for arrays and tokens.
    pub tuple_literals: Vec<Literal>,
    /// Runtime extents for every axis of a dynamic array, bounded by its shape.
    pub dynamic_sizes: Vec<i64>,
}

impl Literal {
    /// Construct a static array from physical-order exact element encodings.
    pub fn from_bits(ctx: &Context, shape: TypeHandle, bits: &[u128]) -> Result<Self> {
        if shape::is_dynamic(ctx, shape) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "from_bits requires static dimensions; use from_dynamic_storage_bytes"
                    .into(),
            }));
        }
        let width = shape::element_type(ctx, shape)?.bit_width();
        if width < 128 && bits.iter().any(|&n| n >> width != 0) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "element bits exceed primitive width".into(),
            }));
        }
        let storage = bits
            .iter()
            .flat_map(|n| n.to_le_bytes().into_iter().take(width.div_ceil(8) as usize))
            .collect();
        Self::from_dynamic_storage_bytes(ctx, shape, storage, Vec::new())
    }
    /// Take physical bytes without copying owned storage; borrowed slices are copied.
    pub fn from_storage_bytes(
        ctx: &Context,
        shape: TypeHandle,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<Self> {
        if shape::is_dynamic(ctx, shape) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "dynamic literals require runtime sizes".into(),
            }));
        }
        Self::from_dynamic_storage_bytes(ctx, shape, bytes.into(), Vec::new())
    }
    /// Take physical storage and per-axis runtime sizes, checking bounds and byte count.
    ///
    /// Storage includes the full bounded allocation; unbounded arrays are rejected.
    pub fn from_dynamic_storage_bytes(
        ctx: &Context,
        mut shape: TypeHandle,
        storage: Vec<u8>,
        dynamic_sizes: Vec<i64>,
    ) -> Result<Self> {
        if shape::layout(ctx, shape)?.is_none() {
            shape = shape::with_default_layout(ctx, shape)?;
        }
        let layout = shape::layout(ctx, shape)?;
        if let Some(layout) = layout
            && layout.element_size_in_bits != 0
        {
            let mut layout = (*layout).clone();
            layout.element_size_in_bits = 0;
            shape = shape::with_layout(ctx, shape, layout)?;
        }
        let literal = Self {
            shape,
            storage,
            tuple_literals: Vec::new(),
            dynamic_sizes,
        };
        literal.verify(ctx)?;
        Ok(literal)
    }
    /// Take ownership of tuple children and derive the tuple type from their shapes.
    pub fn tuple(ctx: &Context, elements: Vec<Self>) -> Result<Self> {
        let literal = Self {
            shape: shape::tuple(ctx, elements.iter().map(|v| v.shape).collect())?,
            storage: Vec::new(),
            tuple_literals: elements,
            dynamic_sizes: Vec::new(),
        };
        literal.verify(ctx)?;
        Ok(literal)
    }
    /// Construct a sequencing token with no payload.
    pub fn token(ctx: &Context) -> Self {
        Self {
            shape: shape::token(ctx),
            storage: Vec::new(),
            tuple_literals: Vec::new(),
            dynamic_sizes: Vec::new(),
        }
    }
    /// Return the literal's interned shape without copying its data.
    pub fn shape(&self) -> TypeHandle {
        self.shape
    }
    /// Borrow a checked tuple descendant without copying its storage.
    pub fn sub_literal(&self, ctx: &Context, index: &[usize]) -> Result<&Self> {
        let shape = shape::subshape(ctx, self.shape, index)?;
        let mut current = self;
        for &axis in index {
            current = current.tuple_literals.get(axis).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "missing tuple literal element".into(),
                })
            })?;
        }
        if !shape::equal_including_layout(ctx, shape, current.shape) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "tuple literal child shape mismatch".into(),
            }));
        }
        current.verify(ctx)?;
        Ok(current)
    }
    /// Materialize physical-order element encodings, masking unused sub-byte bits.
    ///
    /// Rejects malformed literals before decoding; tuple and token literals
    /// have no array element encodings.
    pub fn bits(&self, ctx: &Context) -> Result<Vec<u128>> {
        self.verify(ctx)?;
        let width = shape::element_type(ctx, self.shape)?.bit_width();
        let byte_width = width.div_ceil(8) as usize;
        if !self.storage.len().is_multiple_of(byte_width) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "partial element bytes".into(),
            }));
        }
        Ok((0..self.storage.len() / byte_width)
            .map(|index| self.element_bits(index, byte_width, width))
            .collect())
    }
    /// Compare active logical elements exactly, independent of layouts and unused bits.
    pub fn equal_logical_bits(&self, ctx: &Context, other: &Self) -> Result<bool> {
        self.verify(ctx)?;
        other.verify(ctx)?;
        if shape::is_tuple(ctx, self.shape) {
            if !shape::is_tuple(ctx, other.shape)
                || self.tuple_literals.len() != other.tuple_literals.len()
            {
                return Ok(false);
            }
            for (a, b) in self.tuple_literals.iter().zip(&other.tuple_literals) {
                if !a.equal_logical_bits(ctx, b)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if self.shape.deref(ctx).is::<TokenType>() {
            return Ok(other.shape.deref(ctx).is::<TokenType>());
        }
        if !shape::is_array(ctx, other.shape)
            || shape::element_type(ctx, self.shape)? != shape::element_type(ctx, other.shape)?
        {
            return Ok(false);
        }
        let (dimensions, a_strides) = self.geometry(ctx)?;
        let (b_dimensions, b_strides) = other.geometry(ctx)?;
        if dimensions != b_dimensions {
            return Ok(false);
        }
        let width = shape::element_type(ctx, self.shape)?.bit_width();
        let byte_width = width.div_ceil(8) as usize;
        let count = dimensions.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "active element count overflow".into(),
                })
            })
        })?;
        for logical in 0..count {
            let (mut remaining, mut a, mut b) = (logical, 0, 0);
            for axis in (0..dimensions.len()).rev() {
                let coordinate = remaining % dimensions[axis];
                remaining /= dimensions[axis];
                a += coordinate * a_strides[axis];
                b += coordinate * b_strides[axis];
            }
            if self.element_bits(a, byte_width, width) != other.element_bits(b, byte_width, width) {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn element_bits(&self, index: usize, byte_width: usize, width: u32) -> u128 {
        let start = index * byte_width;
        let mut word = [0; 16];
        word[..byte_width].copy_from_slice(&self.storage[start..start + byte_width]);
        let bits = u128::from_le_bytes(word);
        if width == 128 {
            bits
        } else {
            bits & ((1u128 << width) - 1)
        }
    }
    pub(crate) fn geometry(&self, ctx: &Context) -> Result<(Vec<usize>, Vec<usize>)> {
        let bounds = shape::dimensions(ctx, self.shape)?;
        let layout = shape::layout(ctx, self.shape)?.ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "missing literal layout".into(),
            })
        })?;
        if layout.minor_to_major.len() != bounds.len() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                message: "literal requires concrete minor order".into(),
            }));
        }
        let mut strides = vec![0; bounds.len()];
        let mut stride = 1usize;
        for &axis in &layout.minor_to_major {
            strides[axis as usize] = stride;
            stride = stride
                .checked_mul(bounds[axis as usize] as usize)
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "literal stride overflow".into(),
                    })
                })?;
        }
        let dimensions = bounds
            .iter()
            .enumerate()
            .map(|(axis, &bound)| {
                let size = if matches!(
                    shape::dimension(ctx, self.shape, axis)?,
                    Dimension::Bounded(_)
                ) {
                    self.dynamic_sizes[axis]
                } else {
                    bound
                };
                usize::try_from(size).map_err(|_| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "invalid literal dimension".into(),
                    })
                })
            })
            .collect::<Result<_>>()?;
        Ok((dimensions, strides))
    }
}

impl Verify for Literal {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let mut pending = vec![(self, 0)];
        while let Some((literal, depth)) = pending.pop() {
            if depth >= 128 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: "literal nesting limit".into(),
                }));
            }
            literal.shape.deref(ctx).verify(ctx)?;
            if shape::is_tuple(ctx, literal.shape) {
                if !literal.storage.is_empty() || !literal.dynamic_sizes.is_empty() {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "tuple literal has array payload".into(),
                    }));
                }
                let children = shape::children(ctx, literal.shape);
                if children.len() != literal.tuple_literals.len() {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "tuple literal arity mismatch".into(),
                    }));
                }

                for (&ty, child) in children.iter().zip(&literal.tuple_literals) {
                    if !shape::equal_including_layout(ctx, ty, child.shape) {
                        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                            message: "tuple literal child shape mismatch".into(),
                        }));
                    }
                    pending.push((child, depth + 1));
                }
            } else if literal.shape.deref(ctx).is::<TokenType>() {
                if !literal.storage.is_empty()
                    || !literal.dynamic_sizes.is_empty()
                    || !literal.tuple_literals.is_empty()
                {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "token literal has payload".into(),
                    }));
                }
            } else {
                let element = shape::element_type(ctx, literal.shape)?;
                let layout = shape::layout(ctx, literal.shape)?.ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "literal requires physical layout".into(),
                    })
                })?;
                if layout.element_size_in_bits != 0 {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "literal requires default storage bits".into(),
                    }));
                }
                if shape::is_dynamic(ctx, literal.shape) {
                    if literal.dynamic_sizes.len() != shape::rank(ctx, literal.shape)? {
                        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                            message: "dynamic size cardinality".into(),
                        }));
                    }
                    for (axis, &size) in literal.dynamic_sizes.iter().enumerate() {
                        if let Dimension::Bounded(bound) =
                            shape::dimension(ctx, literal.shape, axis)?
                            && (size < 0 || size > bound)
                        {
                            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                                message: "dynamic size out of bounds".into(),
                            }));
                        }
                    }
                } else if !literal.dynamic_sizes.is_empty() {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "static literal has dynamic sizes".into(),
                    }));
                }
                let count =
                    usize::try_from(shape::element_count(ctx, literal.shape)?).map_err(|_| {
                        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                            message: "literal size overflow".into(),
                        })
                    })?;
                if count.checked_mul(element.bit_width().div_ceil(8) as usize)
                    != Some(literal.storage.len())
                {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "payload length mismatch".into(),
                    }));
                }
                if element == ElementType::Pred && literal.storage.iter().any(|&n| n > 1) {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "predicate storage must be zero or one".into(),
                    }));
                }
                if !literal.tuple_literals.is_empty() {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                        message: "array literal has tuple payload".into(),
                    }));
                }
            }
        }
        Ok(())
    }
}

#[pliron::derive::attr_interface_impl]
impl pliron::builtin::attr_interfaces::TypedAttrInterface for Literal {
    fn get_type(&self, _: &Context) -> TypeHandle {
        self.shape
    }
}
