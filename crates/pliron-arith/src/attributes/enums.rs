//! Stable textual spellings for `arith` enum values.

use pliron::{
    context::Context,
    identifier::Identifier,
    parsable::StreamLocation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};
use std::{error::Error, fmt, str::FromStr};

/// A string is not a canonical, case-sensitive `arith` enum spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumParseError {
    pub(super) enum_name: &'static str,
    pub(super) spelling: String,
}

impl EnumParseError {
    /// Returns the enum name whose parser rejected the spelling.
    pub const fn enum_name(&self) -> &'static str {
        self.enum_name
    }

    /// Returns the rejected spelling exactly as supplied.
    pub fn spelling(&self) -> &str {
        &self.spelling
    }
}

/// An integer is not a defined numeric encoding of an `arith` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnumValueError {
    /// Enum name, such as `cmpf_predicate`.
    pub enum_name: &'static str,
    /// Rejected numeric encoding.
    pub value: u64,
}

impl fmt::Display for EnumValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown {} numeric encoding {}",
            self.enum_name, self.value
        )
    }
}

impl Error for EnumValueError {}

impl fmt::Display for EnumParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown {} spelling {:?}",
            self.enum_name, self.spelling
        )
    }
}

impl Error for EnumParseError {}

macro_rules! enum_spelling {
    ($type:ty, $name:literal { $($variant:path => $spelling:literal = $value:literal),+ $(,)? }) => {
        impl $type {
            /// Upstream MLIR enum name, without the dialect prefix.
            pub const NAME: &'static str = $name;

            /// Returns the upstream MLIR spelling for this value.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($variant => $spelling),+
                }
            }
        }

        impl From<$type> for u64 {
            fn from(value: $type) -> Self {
                match value {
                    $($variant => $value),+
                }
            }
        }

        impl TryFrom<u64> for $type {
            type Error = EnumValueError;

            fn try_from(value: u64) -> Result<Self, Self::Error> {
                match value {
                    $($value => Ok($variant)),+,
                    _ => Err(EnumValueError { enum_name: $name, value }),
                }
            }
        }

        impl FromStr for $type {
            type Err = EnumParseError;

            fn from_str(spelling: &str) -> Result<Self, Self::Err> {
                match spelling {
                    $($spelling => Ok($variant)),+,
                    _ => Err(EnumParseError {
                        enum_name: $name,
                        spelling: spelling.to_owned(),
                    }),
                }
            }
        }

        impl fmt::Display for $type {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl Printable for $type {
            fn fmt(
                &self,
                _: &Context,
                _: &printable::State,
                formatter: &mut fmt::Formatter<'_>,
            ) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl Parsable for $type {
            type Arg = ();
            type Parsed = Self;

            fn parse<'a>(
                state_stream: &mut StateStream<'a>,
                _: Self::Arg,
            ) -> ParseResult<'a, Self::Parsed> {
                let identifier = Identifier::parse(state_stream, ())?.0;
                let spelling = identifier.to_string();
                match spelling.parse::<Self>() {
                    Ok(value) => Ok(value).into_parse_result(),
                    Err(error) => pliron::input_err!(state_stream.loc(), error)
                        .into_parse_result(),
                }
            }
        }
    };
}

enum_spelling!(super::CmpFPredicate, "cmpf_predicate" {
    super::CmpFPredicate::AlwaysFalse => "false" = 0,
    super::CmpFPredicate::Oeq => "oeq" = 1,
    super::CmpFPredicate::Ogt => "ogt" = 2,
    super::CmpFPredicate::Oge => "oge" = 3,
    super::CmpFPredicate::Olt => "olt" = 4,
    super::CmpFPredicate::Ole => "ole" = 5,
    super::CmpFPredicate::One => "one" = 6,
    super::CmpFPredicate::Ord => "ord" = 7,
    super::CmpFPredicate::Ueq => "ueq" = 8,
    super::CmpFPredicate::Ugt => "ugt" = 9,
    super::CmpFPredicate::Uge => "uge" = 10,
    super::CmpFPredicate::Ult => "ult" = 11,
    super::CmpFPredicate::Ule => "ule" = 12,
    super::CmpFPredicate::Une => "une" = 13,
    super::CmpFPredicate::Uno => "uno" = 14,
    super::CmpFPredicate::AlwaysTrue => "true" = 15,
});

enum_spelling!(super::CmpIPredicate, "cmpi_predicate" {
    super::CmpIPredicate::Eq => "eq" = 0,
    super::CmpIPredicate::Ne => "ne" = 1,
    super::CmpIPredicate::Slt => "slt" = 2,
    super::CmpIPredicate::Sle => "sle" = 3,
    super::CmpIPredicate::Sgt => "sgt" = 4,
    super::CmpIPredicate::Sge => "sge" = 5,
    super::CmpIPredicate::Ult => "ult" = 6,
    super::CmpIPredicate::Ule => "ule" = 7,
    super::CmpIPredicate::Ugt => "ugt" = 8,
    super::CmpIPredicate::Uge => "uge" = 9,
});

enum_spelling!(super::RoundingMode, "roundingmode" {
    super::RoundingMode::ToNearestEven => "to_nearest_even" = 0,
    super::RoundingMode::Downward => "downward" = 1,
    super::RoundingMode::Upward => "upward" = 2,
    super::RoundingMode::TowardZero => "toward_zero" = 3,
    super::RoundingMode::ToNearestAway => "to_nearest_away" = 4,
});
