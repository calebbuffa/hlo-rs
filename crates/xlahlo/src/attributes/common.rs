//! Shared scalar, collection, and identifier attributes.

use pliron::derive::pliron_attr;
use std::collections::BTreeMap;

list!(
    Strings,
    "xlahlo.strings",
    String,
    "Ordered string values in native attribute assembly."
);
list!(
    Bytes,
    "xlahlo.bytes",
    u8,
    "Exact uninterpreted bytes for instruction payloads."
);
list!(
    Bools,
    "xlahlo.bools",
    bool,
    "Ordered boolean values in native attribute assembly."
);
record!(StringPair, "xlahlo.string_pair", "One frontend attribute key and string value.", {
    /// Frontend-defined annotation name.
    key: String,
    /// Uninterpreted annotation text.
    value: String,
});
list!(
    FrontendAttributes,
    "xlahlo.frontend_attributes",
    StringPair,
    "Frontend key/value annotations with last-write-wins lookup."
);

impl FrontendAttributes {
    /// Normalize duplicate keys by their last value and sort keys lexicographically.
    pub fn new(values: impl IntoIterator<Item = (String, String)>) -> Self {
        let map: BTreeMap<_, _> = values.into_iter().collect();
        Self(
            map.into_iter()
                .map(|(key, value)| StringPair { key, value })
                .collect(),
        )
    }
    /// Borrow the last value for a key without normalizing or copying the record.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .rev()
            .find(|pair| pair.key == name)
            .map(|pair| pair.value.as_str())
    }
}

#[pliron_attr(
    name = "xlahlo.axes",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Ordered dimension indices or shape-index components.
pub struct Axes(pub Vec<i64>);

#[pliron_attr(name = "xlahlo.index", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Nonnegative instruction index, such as a parameter number or tuple element.
pub struct Index(pub u32);

#[pliron_attr(name = "xlahlo.direction", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Relational predicate applied by a comparison.
pub enum Direction {
    /// Equal to.
    Eq,
    /// Not equal to.
    Ne,
    /// Strictly less than.
    Lt,
    /// Less than or equal to.
    Le,
    /// Strictly greater than.
    Gt,
    /// Greater than or equal to.
    Ge,
}

#[pliron_attr(name = "xlahlo.order", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Partial, total, or weak ordering policy.
pub enum ComparisonOrder {
    /// Partial ordering, in which some values may be unordered.
    Partial,
    /// Total ordering that also orders exceptional floating-point values.
    Total,
    /// Weak ordering that permits equivalence classes.
    Weak,
}

#[pliron_attr(name = "xlahlo.precision", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Requested operand precision level.
pub enum Precision {
    /// Lets the backend choose its default operand precision.
    Default,
    /// Requests higher precision than the backend default.
    High,
    /// Requests the backend's highest supported operand precision.
    Highest,
}

#[pliron_attr(name = "xlahlo.fusion_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Execution organization of a fused computation.
pub enum FusionKind {
    /// Computes fused operations in one elementwise loop.
    Loop,
    /// Organizes fusion around input traversal.
    Input,
    /// Organizes fusion around output production.
    Output,
    /// Delegates fusion organization to a backend-specific implementation.
    Custom,
}

#[pliron_attr(
    name = "xlahlo.slice",
    format = "`<` $starts `,` $limits `,` $strides `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Per-axis starts, exclusive limits, and strides for static slicing.
pub struct Slice {
    /// Inclusive starting index along each operand axis.
    pub starts: Axes,
    /// Exclusive ending index along each operand axis.
    pub limits: Axes,
    /// Positive step between selected indices along each axis.
    pub strides: Axes,
}

#[pliron_attr(
    name = "xlahlo.callees",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Ordered symbols naming computations called by an instruction.
pub struct Callees(pub Vec<crate::ir::ComputationName>);

#[pliron_attr(
    name = "xlahlo.flags",
    format = "`<` $sorted `,` $unique `>`",
    verifier = "succ"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Sortedness and uniqueness guarantees for gather or scatter indices.
pub struct IndexFlags {
    /// Whether the producer guarantees indices are sorted.
    pub sorted: bool,
    /// Whether the producer guarantees indices contain no duplicates.
    pub unique: bool,
}

impl Direction {
    /// Negate the comparison predicate.
    pub fn inverse(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::Ge => Self::Lt,
        }
    }
    /// Reverse the comparison operands without changing the result.
    pub fn converse(self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
            same => same,
        }
    }
}
