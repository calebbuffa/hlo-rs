//! HLO primitive type catalog, including structural types.
use self::PrimitiveType::*;
use super::ElementType;
use crate::Diagnostic;
use pliron::derive::def_attribute;
use pliron::result::Result;

macro_rules! primitives {
    ($($name:ident => $element:ident),* $(,)?) => {
        #[def_attribute("xlahlo.primitive")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        /// HLO primitive types and pinned numeric identifiers, including structural types.
        pub enum PrimitiveType {
            /// Sentinel for an unspecified or invalid primitive; not a usable value type.
            Invalid,
            /// An ordered collection of independently typed values.
            Tuple,
            /// An ordering dependency with no numeric payload.
            Token,
            /// An implementation-defined opaque value.
            Opaque,
            /// Storage backed by an array type.
            Buffer,
            $(
                #[doc = concat!("The `", stringify!($name), "` numeric array format.")]
                $name
            ),*
        }
        impl PrimitiveType {
            /// Return the pinned OpenXLA numeric identifier.
            pub const fn as_i32(self) -> i32 {
                match self {
                    Invalid => 0, Pred => 1, S8 => 2, S16 => 3, S32 => 4,
                    S64 => 5, U8 => 6, U16 => 7, U32 => 8, U64 => 9, F16 => 10,
                    F32 => 11, F64 => 12, Tuple => 13, Opaque => 14, C64 => 15,
                    BF16 => 16, Token => 17, C128 => 18, F8E5M2 => 19, F8E4M3FN => 20,
                    S4 => 21, U4 => 22, F8E4M3B11FNUZ => 23, F8E5M2FNUZ => 24,
                    F8E4M3FNUZ => 25, S2 => 26, U2 => 27, F8E4M3 => 28, F8E3M4 => 29,
                    S1 => 30, U1 => 31, F4E2M1FN => 32, F8E8M0FNU => 33, Buffer => 34,
                    F6E3M2FN => 35, F6E2M3FN => 36,
                }
            }
            /// Every known numeric and structural primitive type.
            pub const VALUES: &'static [Self] = &[
                Self::Invalid, Self::Tuple, Self::Token, Self::Opaque, Self::Buffer,
                $(Self::$name),*
            ];
            /// Convert a numeric type to an array format, rejecting structural types.
            pub fn array_element(self) -> Result<ElementType> {
                match self {
                    $(Self::$name => Ok(ElementType::$element),)*
                    _ => Err(pliron::verify_error_noloc!(Diagnostic::InvalidShape { message: "array requires numeric primitive type".into() })),
                }
            }
        }
        impl From<ElementType> for PrimitiveType {
            fn from(element: ElementType) -> Self {
                match element { $(ElementType::$element => Self::$name),* }
            }
        }
        impl TryFrom<PrimitiveType> for ElementType {
            type Error = pliron::result::Error;

            fn try_from(ty: PrimitiveType) -> std::result::Result<Self, Self::Error> {
                ty.array_element()
            }
        }
        impl From<PrimitiveType> for i32 {
            fn from(ty: PrimitiveType) -> Self {
                ty.as_i32()
            }
        }
        impl TryFrom<i32> for PrimitiveType {
            type Error = PrimitiveTypeValueError;

            fn try_from(value: i32) -> std::result::Result<Self, Self::Error> {
                Self::VALUES
                    .iter()
                    .copied()
                    .find(|ty| ty.as_i32() == value)
                    .ok_or(PrimitiveTypeValueError(value))
            }
        }
    };
}
primitives!(
    Pred => Pred, S1 => S1, S2 => S2, S4 => S4, S8 => S8, S16 => S16, S32 => S32, S64 => S64,
    U1 => U1, U2 => U2, U4 => U4, U8 => U8, U16 => U16, U32 => U32, U64 => U64,
    F16 => F16, BF16 => BF16, F32 => F32, F64 => F64, C64 => C64, C128 => C128,
    F8E5M2 => F8E5M2, F8E4M3 => F8E4M3, F8E4M3FN => F8E4M3FN,
    F8E4M3B11FNUZ => F8E4M3B11FNUZ, F8E3M4 => F8E3M4, F8E5M2FNUZ => F8E5M2FNUZ,
    F8E4M3FNUZ => F8E4M3FNUZ, F4E2M1FN => F4E2M1FN, F8E8M0FNU => F8E8M0FNU,
    F6E3M2FN => F6E3M2FN, F6E2M3FN => F6E2M3FN,
);

impl PrimitiveType {
    /// Whether the type is an integer format, excluding predicates.
    pub fn is_integral(self) -> bool {
        self.array_element().is_ok_and(ElementType::is_integer)
    }
    /// Whether the type is a signed integer format.
    pub fn is_signed_integral(self) -> bool {
        self.array_element().is_ok_and(ElementType::is_signed)
    }
    /// Whether the type is a real floating-point format.
    pub fn is_floating(self) -> bool {
        self.array_element().is_ok_and(ElementType::is_float)
    }
    /// Whether the type has real and imaginary floating-point components.
    pub fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }
    /// Whether the type can appear as an array element.
    pub fn is_array_element(self) -> bool {
        !matches!(
            self,
            Self::Invalid | Self::Tuple | Self::Token | Self::Opaque | Self::Buffer
        )
    }
    /// Compare types, allowing differing real floating-point precisions.
    pub fn same_ignoring_fp_precision(self, other: Self) -> bool {
        self == other || self.is_floating() && other.is_floating()
    }
    /// Return numeric encoding width; structural types have no width.
    pub fn bit_width(self) -> Option<u32> {
        self.array_element().ok().map(ElementType::bit_width)
    }
    /// Return numeric storage width; structural types have no width.
    pub fn storage_bit_width(self) -> Option<u32> {
        self.array_element()
            .ok()
            .map(ElementType::storage_bit_width)
    }
    /// Select the higher-precision numeric type, rejecting structural types and ties.
    pub fn higher_precision(self, other: Self) -> Result<Self> {
        Ok(self
            .array_element()?
            .higher_precision(other.array_element()?)?
            .into())
    }
    /// Return the lowercase HLO spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Invalid => "invalid",
            Self::Tuple => "tuple",
            Self::Token => "token",
            Self::Opaque => "opaque",
            Self::Buffer => "buffer",
            Self::Pred => "pred",
            Self::S1 => "s1",
            Self::S2 => "s2",
            Self::S4 => "s4",
            Self::S8 => "s8",
            Self::S16 => "s16",
            Self::S32 => "s32",
            Self::S64 => "s64",
            Self::U1 => "u1",
            Self::U2 => "u2",
            Self::U4 => "u4",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::C64 => "c64",
            Self::C128 => "c128",
            Self::F8E5M2 => "f8e5m2",
            Self::F8E4M3 => "f8e4m3",
            Self::F8E4M3FN => "f8e4m3fn",
            Self::F8E4M3B11FNUZ => "f8e4m3b11fnuz",
            Self::F8E3M4 => "f8e3m4",
            Self::F8E5M2FNUZ => "f8e5m2fnuz",
            Self::F8E4M3FNUZ => "f8e4m3fnuz",
            Self::F4E2M1FN => "f4e2m1fn",
            Self::F8E8M0FNU => "f8e8m0fnu",
            Self::F6E3M2FN => "f6e3m2fn",
            Self::F6E2M3FN => "f6e2m3fn",
        }
    }
    /// HLO attribute spelling retained for the framework attribute printer.
    pub(crate) const fn attribute_name(self) -> &'static str {
        match self {
            Self::Invalid => "PRIMITIVE_TYPE_INVALID",
            Self::Tuple => "TUPLE",
            Self::Token => "TOKEN",
            Self::Opaque => "OPAQUE_TYPE",
            Self::Buffer => "BUFFER",
            Self::Pred => "PRED",
            Self::S1 => "S1",
            Self::S2 => "S2",
            Self::S4 => "S4",
            Self::S8 => "S8",
            Self::S16 => "S16",
            Self::S32 => "S32",
            Self::S64 => "S64",
            Self::U1 => "U1",
            Self::U2 => "U2",
            Self::U4 => "U4",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::C64 => "C64",
            Self::C128 => "C128",
            Self::F8E5M2 => "F8E5M2",
            Self::F8E4M3 => "F8E4M3",
            Self::F8E4M3FN => "F8E4M3FN",
            Self::F8E4M3B11FNUZ => "F8E4M3B11FNUZ",
            Self::F8E3M4 => "F8E3M4",
            Self::F8E5M2FNUZ => "F8E5M2FNUZ",
            Self::F8E4M3FNUZ => "F8E4M3FNUZ",
            Self::F4E2M1FN => "F4E2M1FN",
            Self::F8E8M0FNU => "F8E8M0FNU",
            Self::F6E3M2FN => "F6E3M2FN",
            Self::F6E2M3FN => "F6E2M3FN",
        }
    }
}

impl std::fmt::Display for PrimitiveType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for PrimitiveType {
    type Err = PrimitiveTypeNameError;

    fn from_str(name: &str) -> std::result::Result<Self, Self::Err> {
        Self::VALUES
            .iter()
            .copied()
            .find(|ty| ty.as_str() == name)
            .ok_or_else(|| PrimitiveTypeNameError(name.to_owned()))
    }
}

/// An unrecognized lowercase HLO primitive type name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown primitive type {0:?}")]
pub struct PrimitiveTypeNameError(String);

impl PrimitiveTypeNameError {
    /// Borrows the rejected lowercase HLO type name.
    pub fn spelling(&self) -> &str {
        &self.0
    }
}

/// An unrecognized OpenXLA numeric primitive type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown primitive type identifier {0}")]
pub struct PrimitiveTypeValueError(i32);

impl PrimitiveTypeValueError {
    /// Returns the rejected OpenXLA primitive identifier.
    pub const fn value(self) -> i32 {
        self.0
    }
}
