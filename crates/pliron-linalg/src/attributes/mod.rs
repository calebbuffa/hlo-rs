//! Attributes used by `linalg` operations.

mod enums;

pub use enums::{ElementwiseArity, ElementwiseKind, IteratorType, TypeFn, WinogradFmr};
use pliron::derive::pliron_attr;
use pliron_mlir_builtin::attributes::{AffineExpr, AffineMap};

/// Iterator types of a structured operation, one per loop dimension.
#[pliron_attr(
    name = "linalg.iterator_types",
    format = "`[` vec($values, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IteratorTypesAttr {
    /// Loop kinds in iteration-space order.
    pub values: Vec<IteratorType>,
}

impl IteratorTypesAttr {
    /// Creates the attribute.
    pub fn new(values: Vec<IteratorType>) -> Self {
        Self { values }
    }
}

/// Indexing maps of a structured operation, one per operand.
///
/// Each map takes the iteration-space dimensions and produces one index per
/// operand dimension.
#[pliron_attr(
    name = "linalg.indexing_maps",
    format = "`[` vec($maps, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexingMapsAttr {
    /// Maps in operand order.
    pub maps: Vec<AffineMap>,
}

impl IndexingMapsAttr {
    /// Creates the attribute.
    pub fn new(maps: Vec<AffineMap>) -> Self {
        Self { maps }
    }
}

/// Shared integer-array storage used by multiple dialects.
pub use pliron_mlir_builtin::attributes::I64ArrayAttr;

/// Operand conversion of a named or contraction operation.
#[pliron_attr(name = "linalg.type_fn", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeFnAttr {
    /// Conversion function.
    pub value: TypeFn,
}

impl TypeFnAttr {
    /// Creates the attribute.
    pub const fn new(value: TypeFn) -> Self {
        Self { value }
    }
}

/// Scalar computation of `linalg.elementwise`.
#[pliron_attr(name = "linalg.elementwise_kind", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementwiseKindAttr {
    /// Computation kind.
    pub value: ElementwiseKind,
}

impl ElementwiseKindAttr {
    /// Creates the attribute.
    pub const fn new(value: ElementwiseKind) -> Self {
        Self { value }
    }
}

/// Winograd configuration of the `linalg.winograd_*` transforms.
#[pliron_attr(name = "linalg.winograd_fmr", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WinogradFmrAttr {
    /// `F(m, r)` configuration.
    pub value: WinogradFmr,
}

impl WinogradFmrAttr {
    /// Creates the attribute.
    pub const fn new(value: WinogradFmr) -> Self {
        Self { value }
    }
}

/// Returns the identity map `(d0, .., dN-1) -> (d0, .., dN-1)`.
pub fn identity_map(rank: usize) -> AffineMap {
    AffineMap {
        dimensions: rank,
        symbols: 0,
        results: (0..rank).map(AffineExpr::dimension).collect(),
    }
}

/// Returns a projection `(d0, .., dN-1) -> (d[i] for i in dims)`.
///
/// Use it for reduction outputs and broadcast inputs.
pub fn projection_map(rank: usize, dims: &[usize]) -> AffineMap {
    AffineMap {
        dimensions: rank,
        symbols: 0,
        results: dims.iter().copied().map(AffineExpr::dimension).collect(),
    }
}
