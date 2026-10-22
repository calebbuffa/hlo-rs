use pliron::derive::pliron_attr;

/// Dimension extent for an unbounded dynamic axis.
pub const UNBOUNDED_SIZE: i64 = i64::MIN;

#[pliron_attr(name = "xlahlo.dimension", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Static extent, bounded runtime extent, or unbounded dynamic extent.
pub enum Dimension {
    /// A compile-time extent that must be nonnegative.
    Static(i64),
    /// A runtime extent with a nonnegative compile-time upper bound.
    Bounded(i64),
    /// A runtime extent without a finite compile-time upper bound.
    Unbounded,
}
