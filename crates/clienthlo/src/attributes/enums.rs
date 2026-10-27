//! Stable textual spellings for ClientHLO enum values.

use pliron::{
    context::Context,
    identifier::Identifier,
    parsable::StreamLocation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};
use std::{error::Error, fmt, str::FromStr};

/// A string is not a canonical, case-sensitive ClientHLO enum spelling.
///
/// Use the relevant enum's `as_str` method to obtain accepted spellings. No case
/// folding or whitespace trimming is performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumParseError {
    enum_name: &'static str,
    spelling: String,
}

impl EnumParseError {
    /// Returns the canonical CHLO enum name whose parser rejected the spelling.
    pub const fn enum_name(&self) -> &'static str {
        self.enum_name
    }

    /// Returns the rejected spelling exactly as supplied to `FromStr`.
    pub fn spelling(&self) -> &str {
        &self.spelling
    }
}

/// An integer is not a defined numeric encoding of a ClientHLO enum.
///
/// Inspect [`Self::enum_name`] and [`Self::value`] to report the invalid input.
/// Use `u64::from(value)` to encode a known enum value without loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnumValueError {
    /// Canonical CHLO enum name, such as `comparison_type`.
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
    ($type:ty, $name:literal { $($variant:path => $spelling:literal),+ $(,)? }) => {
        impl $type {
            /// Canonical CHLO enum name, without the dialect prefix.
            pub const NAME: &'static str = $name;

            /// Returns the canonical ClientHLO spelling for this value.
            ///
            /// This case-sensitive spelling is accepted by `FromStr` and emitted
            /// by the pliron printer; it is not the Rust variant name.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($variant => $spelling),+
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

enum_spelling!(super::ComparisonDirection, "comparison_direction" {
    super::ComparisonDirection::Eq => "EQ",
    super::ComparisonDirection::Ne => "NE",
    super::ComparisonDirection::Ge => "GE",
    super::ComparisonDirection::Gt => "GT",
    super::ComparisonDirection::Le => "LE",
    super::ComparisonDirection::Lt => "LT",
});
enum_spelling!(super::ComparisonKind, "comparison_type" {
    super::ComparisonKind::NoType => "NOTYPE",
    super::ComparisonKind::Float => "FLOAT",
    super::ComparisonKind::TotalOrder => "TOTALORDER",
    super::ComparisonKind::Signed => "SIGNED",
    super::ComparisonKind::Unsigned => "UNSIGNED",
    super::ComparisonKind::WeakOrder => "WEAKORDER",
});
enum_spelling!(super::Precision, "precision" {
    super::Precision::Default => "DEFAULT",
    super::Precision::High => "HIGH",
    super::Precision::Highest => "HIGHEST",
});
