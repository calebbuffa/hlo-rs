//! Stable textual spellings and numeric encodings for `linalg` enum values.

use crate::Diagnostic;
use pliron::{
    context::Context,
    identifier::Identifier,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream, StreamLocation},
    printable::{self, Printable},
};
use std::{fmt, str::FromStr};

macro_rules! enum_spelling {
    ($type:ident, $name:literal { $($variant:ident => $spelling:literal = $value:literal),+ $(,)? }) => {
        impl $type {
            /// Upstream MLIR enum name, without the dialect prefix.
            pub const NAME: &'static str = $name;

            /// Every value in numeric-encoding order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// Returns the upstream MLIR spelling for this value.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $spelling),+
                }
            }

            /// Returns the upstream numeric encoding.
            pub const fn value(self) -> u32 {
                match self {
                    $(Self::$variant => $value),+
                }
            }

            /// Returns the value with numeric encoding `value`, if defined.
            pub const fn from_value(value: u32) -> Option<Self> {
                match value {
                    $($value => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }

        impl FromStr for $type {
            type Err = Diagnostic;

            fn from_str(spelling: &str) -> Result<Self, Self::Err> {
                match spelling {
                    $($spelling => Ok(Self::$variant),)+
                    _ => Err(Diagnostic::UnknownSpelling {
                        enum_name: $name,
                        spelling: spelling.to_owned(),
                    }),
                }
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl Printable for $type {
            fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl Parsable for $type {
            type Arg = ();
            type Parsed = Self;

            fn parse<'a>(state_stream: &mut StateStream<'a>, _: Self::Arg) -> ParseResult<'a, Self> {
                let spelling = Identifier::parse(state_stream, ())?.0.to_string();
                match spelling.parse::<Self>() {
                    Ok(value) => Ok(value).into_parse_result(),
                    Err(error) => pliron::input_err!(state_stream.loc(), error).into_parse_result(),
                }
            }
        }
    };
}

/// Loop kind of one iteration-space dimension of a structured operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IteratorType {
    /// Independent iterations that may run in any order.
    Parallel,
    /// Iterations that accumulate into the same output element.
    Reduction,
}

enum_spelling!(IteratorType, "iterator_type" {
    Parallel => "parallel" = 0,
    Reduction => "reduction" = 1,
});

/// Element conversion applied to operands before a named op's computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum TypeFn {
    /// Sign-extending (or float) conversion; the upstream default.
    #[default]
    CastSigned,
    /// Zero-extending conversion of integers.
    CastUnsigned,
}

enum_spelling!(TypeFn, "type_fn" {
    CastSigned => "cast_signed" = 0,
    CastUnsigned => "cast_unsigned" = 1,
});

/// Number of inputs an elementwise kind consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementwiseArity {
    /// One input.
    Unary,
    /// Two inputs.
    Binary,
    /// Three inputs.
    Ternary,
}

impl ElementwiseArity {
    /// Returns the number of inputs.
    pub const fn inputs(self) -> usize {
        match self {
            Self::Unary => 1,
            Self::Binary => 2,
            Self::Ternary => 3,
        }
    }
}

/// Scalar computation of `linalg.elementwise`.
///
/// Encodings follow upstream: unary functions are `0..=24`, binary `25..=34`,
/// and ternary `select` is `35`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum ElementwiseKind {
    Exp,
    Log,
    Abs,
    Ceil,
    Floor,
    NegF,
    Reciprocal,
    Round,
    Sqrt,
    Rsqrt,
    Square,
    Tanh,
    Erf,
    Sin,
    Cos,
    Tan,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Log10,
    Log1p,
    Log2,
    Add,
    Sub,
    Mul,
    Div,
    DivUnsigned,
    MaxSigned,
    MinSigned,
    MaxUnsigned,
    MinUnsigned,
    Powf,
    Select,
}

enum_spelling!(ElementwiseKind, "elementwise_kind" {
    Exp => "exp" = 0,
    Log => "log" = 1,
    Abs => "abs" = 2,
    Ceil => "ceil" = 3,
    Floor => "floor" = 4,
    NegF => "negf" = 5,
    Reciprocal => "reciprocal" = 6,
    Round => "round" = 7,
    Sqrt => "sqrt" = 8,
    Rsqrt => "rsqrt" = 9,
    Square => "square" = 10,
    Tanh => "tanh" = 11,
    Erf => "erf" = 12,
    Sin => "sin" = 13,
    Cos => "cos" = 14,
    Tan => "tan" = 15,
    Acos => "acos" = 16,
    Acosh => "acosh" = 17,
    Asin => "asin" = 18,
    Asinh => "asinh" = 19,
    Atan => "atan" = 20,
    Atanh => "atanh" = 21,
    Log10 => "log10" = 22,
    Log1p => "log1p" = 23,
    Log2 => "log2" = 24,
    Add => "add" = 25,
    Sub => "sub" = 26,
    Mul => "mul" = 27,
    Div => "div" = 28,
    DivUnsigned => "div_unsigned" = 29,
    MaxSigned => "max_signed" = 30,
    MinSigned => "min_signed" = 31,
    MaxUnsigned => "max_unsigned" = 32,
    MinUnsigned => "min_unsigned" = 33,
    Powf => "powf" = 34,
    Select => "select" = 35,
});

impl ElementwiseKind {
    /// Returns the number of inputs this kind consumes.
    pub const fn arity(self) -> ElementwiseArity {
        match self.value() {
            0..=24 => ElementwiseArity::Unary,
            25..=34 => ElementwiseArity::Binary,
            _ => ElementwiseArity::Ternary,
        }
    }
}

/// Winograd `F(m, r)` configuration of the `linalg.winograd_*` transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WinogradFmr {
    /// `F(2, 3)`: 2x2 output tiles, 3x3 filters.
    F2x3,
    /// `F(4, 3)`: 4x4 output tiles, 3x3 filters.
    F4x3,
    /// `F(2, 5)`: 2x2 output tiles, 5x5 filters.
    F2x5,
}

enum_spelling!(WinogradFmr, "winograd_conv2d_fmr" {
    F2x3 => "F_2_3" = 0,
    F4x3 => "F_4_3" = 1,
    F2x5 => "F_2_5" = 2,
});

impl WinogradFmr {
    /// Returns `(m, r)`: output tile size and filter size.
    pub const fn m_r(self) -> (i64, i64) {
        match self {
            Self::F2x3 => (2, 3),
            Self::F4x3 => (4, 3),
            Self::F2x5 => (2, 5),
        }
    }

    /// Returns the transformed tile size `m + r - 1`.
    pub const fn alpha(self) -> i64 {
        let (m, r) = self.m_r();
        m + r - 1
    }
}
