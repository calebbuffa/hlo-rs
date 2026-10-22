//! Stable textual spellings for XLAHLO enum values.

use crate::types::PrimitiveType;
use pliron::{
    common_traits::Verify,
    context::Context,
    identifier::Identifier,
    location::Located,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
};
use std::{error::Error, fmt, str::FromStr};

/// Error returned when a string is not a valid XLAHLO enum spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumParseError {
    enum_name: &'static str,
    spelling: String,
}

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

impl EnumParseError {
    /// Returns the external enum name whose parser rejected the input.
    pub fn enum_name(&self) -> &'static str {
        self.enum_name
    }

    /// Borrows the rejected spelling.
    pub fn spelling(&self) -> &str {
        &self.spelling
    }
}

macro_rules! enum_spelling {
    ($type:ty, $name:literal { $($variant:path => $spelling:literal),+ $(,)? }) => {
        impl $type {
            /// Name used for this enum in XLAHLO text.
            pub const NAME: &'static str = $name;

            /// Returns the canonical XLAHLO spelling for this value.
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

        impl Verify for $type {
            fn verify(&self, _: &Context) -> pliron::result::Result<()> {
                Ok(())
            }
        }
    };
}

enum_spelling!(super::FftKind, "fft_type" {
    super::FftKind::Fft => "FFT",
    super::FftKind::Ifft => "IFFT",
    super::FftKind::Rfft => "RFFT",
    super::FftKind::Irfft => "IRFFT",
});
enum_spelling!(super::ChannelKind, "channel_type" {
    super::ChannelKind::DeviceToDevice => "DEVICE_TO_DEVICE",
    super::ChannelKind::DeviceToHost => "DEVICE_TO_HOST",
    super::ChannelKind::HostToDevice => "HOST_TO_DEVICE",
});

enum_spelling!(super::CollectiveOpGroupMode, "collective_op_group_mode" {
    super::CollectiveOpGroupMode::CrossReplica => "CrossReplica",
    super::CollectiveOpGroupMode::CrossPartition => "CrossPartition",
    super::CollectiveOpGroupMode::CrossReplicaAndPartition => "CrossReplicaAndPartition",
    super::CollectiveOpGroupMode::FlattenedId => "FlattenedID",
});

impl Printable for PrimitiveType {
    fn fmt(
        &self,
        _: &Context,
        _: &printable::State,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        formatter.write_str(self.attribute_name())
    }
}

impl Parsable for PrimitiveType {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let identifier = Identifier::parse(state_stream, ())?.0;
        let spelling = identifier.to_string();
        let parsed = match spelling.as_str() {
            "PRIMITIVE_TYPE_INVALID" => Some(Self::Invalid),
            "TUPLE" => Some(Self::Tuple),
            "TOKEN" => Some(Self::Token),
            "OPAQUE_TYPE" => Some(Self::Opaque),
            "BUFFER" => Some(Self::Buffer),
            "PRED" => Some(Self::Pred),
            "S1" => Some(Self::S1),
            "S2" => Some(Self::S2),
            "S4" => Some(Self::S4),
            "S8" => Some(Self::S8),
            "S16" => Some(Self::S16),
            "S32" => Some(Self::S32),
            "S64" => Some(Self::S64),
            "U1" => Some(Self::U1),
            "U2" => Some(Self::U2),
            "U4" => Some(Self::U4),
            "U8" => Some(Self::U8),
            "U16" => Some(Self::U16),
            "U32" => Some(Self::U32),
            "U64" => Some(Self::U64),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "C64" => Some(Self::C64),
            "C128" => Some(Self::C128),
            "F8E5M2" => Some(Self::F8E5M2),
            "F8E4M3" => Some(Self::F8E4M3),
            "F8E4M3FN" => Some(Self::F8E4M3FN),
            "F8E4M3B11FNUZ" => Some(Self::F8E4M3B11FNUZ),
            "F8E3M4" => Some(Self::F8E3M4),
            "F8E5M2FNUZ" => Some(Self::F8E5M2FNUZ),
            "F8E4M3FNUZ" => Some(Self::F8E4M3FNUZ),
            "F4E2M1FN" => Some(Self::F4E2M1FN),
            "F8E8M0FNU" => Some(Self::F8E8M0FNU),
            "F6E3M2FN" => Some(Self::F6E3M2FN),
            "F6E2M3FN" => Some(Self::F6E2M3FN),
            _ => None,
        };
        match parsed {
            Some(value) => Ok(value).into_parse_result(),
            None => pliron::input_err!(
                state_stream.loc(),
                "unknown primitive spelling {spelling:?}"
            )
            .into_parse_result(),
        }
    }
}

impl Verify for PrimitiveType {
    fn verify(&self, _: &Context) -> pliron::result::Result<()> {
        Ok(())
    }
}
