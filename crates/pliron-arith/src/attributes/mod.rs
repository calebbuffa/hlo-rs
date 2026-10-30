//! Attributes and enums used by `arith` operations.
//!
//! Enum and flag attributes carry valid Rust values, so their standalone
//! verification is infallible. Operation verification checks the contextual
//! rules, such as whether a predicate matches the operand element type.

use pliron::{
    combine::Parser,
    context::Context,
    derive::pliron_attr,
    identifier::Identifier,
    irfmt::parsers::delimited_list_parser,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream, StreamLocation},
    printable::{self, Printable},
};
use std::fmt;

mod enums;
pub use enums::{EnumParseError, EnumValueError};

/// Predicate of `arith.cmpf`, with MLIR's numeric encoding.
///
/// Ordered predicates (`o*`) are false when either operand is NaN; unordered
/// predicates (`u*`) are true in that case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CmpFPredicate {
    /// Always false.
    AlwaysFalse = 0,
    /// Ordered and equal.
    Oeq = 1,
    /// Ordered and greater than.
    Ogt = 2,
    /// Ordered and greater than or equal.
    Oge = 3,
    /// Ordered and less than.
    Olt = 4,
    /// Ordered and less than or equal.
    Ole = 5,
    /// Ordered and not equal.
    One = 6,
    /// Ordered: neither operand is NaN.
    Ord = 7,
    /// Unordered or equal.
    Ueq = 8,
    /// Unordered or greater than.
    Ugt = 9,
    /// Unordered or greater than or equal.
    Uge = 10,
    /// Unordered or less than.
    Ult = 11,
    /// Unordered or less than or equal.
    Ule = 12,
    /// Unordered or not equal.
    Une = 13,
    /// Unordered: either operand is NaN.
    Uno = 14,
    /// Always true.
    AlwaysTrue = 15,
}

/// Predicate of `arith.cmpi`, with MLIR's numeric encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CmpIPredicate {
    /// Equal.
    Eq = 0,
    /// Not equal.
    Ne = 1,
    /// Signed less than.
    Slt = 2,
    /// Signed less than or equal.
    Sle = 3,
    /// Signed greater than.
    Sgt = 4,
    /// Signed greater than or equal.
    Sge = 5,
    /// Unsigned less than.
    Ult = 6,
    /// Unsigned less than or equal.
    Ule = 7,
    /// Unsigned greater than.
    Ugt = 8,
    /// Unsigned greater than or equal.
    Uge = 9,
}

/// IEEE-754 rounding mode (`roundingmode`), with MLIR's numeric encoding.
///
/// Matches `llvm::RoundingMode`. When absent, operations round to nearest even.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RoundingMode {
    /// Round to nearest, ties to even.
    ToNearestEven = 0,
    /// Round toward negative infinity.
    Downward = 1,
    /// Round toward positive infinity.
    Upward = 2,
    /// Round toward zero.
    TowardZero = 3,
    /// Round to nearest, ties away from zero.
    ToNearestAway = 4,
}

/// Rounding mode stored on `addf`, `subf`, `mulf`, `divf`, `truncf`,
/// `convertf`, and `scaling_truncf`.
#[pliron_attr(name = "arith.roundingmode", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoundingModeAttr {
    /// Rounding applied to inexact results.
    pub value: RoundingMode,
}

impl RoundingModeAttr {
    /// Creates an attribute for the given rounding mode.
    pub const fn new(value: RoundingMode) -> Self {
        Self { value }
    }
}

/// Typed predicate attribute stored on `arith.cmpf`.
#[pliron_attr(name = "arith.cmpf_predicate", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CmpFPredicateAttr {
    /// Comparison applied to each element pair.
    pub value: CmpFPredicate,
}

impl CmpFPredicateAttr {
    /// Creates an attribute for the given predicate.
    pub const fn new(value: CmpFPredicate) -> Self {
        Self { value }
    }
}

/// Typed predicate attribute stored on `arith.cmpi`.
#[pliron_attr(name = "arith.cmpi_predicate", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CmpIPredicateAttr {
    /// Comparison applied to each element pair.
    pub value: CmpIPredicate,
}

impl CmpIPredicateAttr {
    /// Creates an attribute for the given predicate.
    pub const fn new(value: CmpIPredicate) -> Self {
        Self { value }
    }
}

macro_rules! flag_set {
    (
        $(#[$meta:meta])*
        $type:ident, $name:literal, $separator:literal, $all:expr,
        { $($(#[$fmeta:meta])* $flag:ident = $bit:literal => $spelling:literal),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $type {
            bits: u8,
        }

        impl $type {
            /// Upstream MLIR attribute mnemonic, without the dialect prefix.
            pub const NAME: &'static str = $name;
            /// No flags set.
            pub const NONE: Self = Self { bits: 0 };
            $(
                $(#[$fmeta])*
                pub const $flag: Self = Self { bits: $bit };
            )+
            const ALL_BITS: u8 = $all;

            /// Returns the raw bit mask, matching MLIR's encoding.
            pub const fn bits(self) -> u8 {
                self.bits
            }

            /// Creates a set from a raw mask, rejecting undefined bits.
            pub const fn from_bits(bits: u8) -> Option<Self> {
                if bits & !Self::ALL_BITS == 0 {
                    Some(Self { bits })
                } else {
                    None
                }
            }

            /// Returns whether no flags are set.
            pub const fn is_none(self) -> bool {
                self.bits == 0
            }

            /// Returns whether every flag in `other` is set.
            pub const fn contains(self, other: Self) -> bool {
                self.bits & other.bits == other.bits
            }

            /// Returns the union of two sets.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self { bits: self.bits | other.bits }
            }

            fn names(self) -> Vec<&'static str> {
                let mut names = Vec::new();
                $(
                    if self.contains(Self::$flag) {
                        names.push($spelling);
                    }
                )+
                names
            }

            /// Returns the comma-separated MLIR spelling, or `none`.
            pub fn mlir_spelling(self) -> String {
                if self.is_none() {
                    "none".to_owned()
                } else {
                    self.names().join($separator)
                }
            }

            fn from_name(name: &str) -> Result<Self, EnumParseError> {
                match name {
                    "none" => Ok(Self::NONE),
                    $($spelling => Ok(Self::$flag),)+
                    _ => Err(EnumParseError {
                        enum_name: $name,
                        spelling: name.to_owned(),
                    }),
                }
            }
        }

        impl std::ops::BitOr for $type {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.mlir_spelling())
            }
        }

        impl Printable for $type {
            fn fmt(
                &self,
                _: &Context,
                _: &printable::State,
                formatter: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                let names = if self.is_none() { vec!["none"] } else { self.names() };
                write!(formatter, "<{}>", names.join(","))
            }
        }

        impl Parsable for $type {
            type Arg = ();
            type Parsed = Self;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                _: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let loc = state_stream.loc();
                let names = delimited_list_parser('<', '>', ',', Identifier::parser(()))
                    .parse_stream(state_stream)
                    .into_result()?
                    .0;
                let mut flags = Self::NONE;
                for name in names {
                    match Self::from_name(&name.to_string()) {
                        Ok(flag) => flags = flags.union(flag),
                        Err(error) => return pliron::input_err!(loc, error).into_parse_result(),
                    }
                }
                Ok(flags).into_parse_result()
            }
        }
    };
}

flag_set!(
    /// Floating-point fast-math flags (`#arith.fastmath`).
    ///
    /// Each flag relaxes IEEE-754 semantics to permit additional optimization.
    FastMathFlags, "fastmath", ",", 0x7f, {
        /// Allow reassociation.
        REASSOC = 1 => "reassoc",
        /// Assume no NaN operands or results.
        NNAN = 2 => "nnan",
        /// Assume no infinite operands or results.
        NINF = 4 => "ninf",
        /// Ignore the sign of zero.
        NSZ = 8 => "nsz",
        /// Allow reciprocal approximation.
        ARCP = 16 => "arcp",
        /// Allow floating-point contraction such as fused multiply-add.
        CONTRACT = 32 => "contract",
        /// Allow approximate library functions.
        AFN = 64 => "afn",
    }
);

impl FastMathFlags {
    /// Every fast-math flag (`fast`).
    pub const FAST: Self = Self { bits: 0x7f };
}

flag_set!(
    /// Integer overflow flags (`#arith.overflow`).
    ///
    /// Setting a flag makes the corresponding overflow produce poison.
    IntegerOverflowFlags, "overflow", ", ", 0x3, {
        /// No signed wrap.
        NSW = 1 => "nsw",
        /// No unsigned wrap.
        NUW = 2 => "nuw",
    }
);

/// Fast-math flags stored on floating-point operations.
#[pliron_attr(name = "arith.fastmath", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FastMathFlagsAttr {
    /// Flag set.
    pub value: FastMathFlags,
}

impl FastMathFlagsAttr {
    /// Creates an attribute for the given flags.
    pub const fn new(value: FastMathFlags) -> Self {
        Self { value }
    }
}

/// Integer overflow flags stored on `addi`, `subi`, `muli`, and `shli`.
#[pliron_attr(name = "arith.overflow", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntegerOverflowFlagsAttr {
    /// Flag set.
    pub value: IntegerOverflowFlags,
}

impl IntegerOverflowFlagsAttr {
    /// Creates an attribute for the given flags.
    pub const fn new(value: IntegerOverflowFlags) -> Self {
        Self { value }
    }
}
