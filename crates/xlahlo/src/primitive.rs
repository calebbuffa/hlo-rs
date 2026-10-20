//! Complete HLO primitive kind catalog, including structural kinds.
use self::PrimitiveType::*;
use crate::Diagnostic;
use crate::types::ElementType;
use pliron::derive::pliron_attr;
use pliron::result::Result;

macro_rules! primitives {
    ($($name:ident => $element:ident),* $(,)?) => {
        #[pliron_attr(name = "xlahlo.primitive", format, verifier = "succ")]
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        /// HLO primitive names and pinned numeric identifiers, including structural kinds.
        pub enum PrimitiveType {
            PRIMITIVE_TYPE_INVALID, TUPLE, TOKEN, OPAQUE_TYPE, BUFFER,
            $($name),*
        }
        impl PrimitiveType {
            /// Return the pinned OpenXLA numeric identifier.
            pub fn value(self) -> i32 {
                match self {
                    PRIMITIVE_TYPE_INVALID => 0, PRED => 1, S8 => 2, S16 => 3, S32 => 4,
                    S64 => 5, U8 => 6, U16 => 7, U32 => 8, U64 => 9, F16 => 10,
                    F32 => 11, F64 => 12, TUPLE => 13, OPAQUE_TYPE => 14, C64 => 15,
                    BF16 => 16, TOKEN => 17, C128 => 18, F8E5M2 => 19, F8E4M3FN => 20,
                    S4 => 21, U4 => 22, F8E4M3B11FNUZ => 23, F8E5M2FNUZ => 24,
                    F8E4M3FNUZ => 25, S2 => 26, U2 => 27, F8E4M3 => 28, F8E3M4 => 29,
                    S1 => 30, U1 => 31, F4E2M1FN => 32, F8E8M0FNU => 33, BUFFER => 34,
                    F6E3M2FN => 35, F6E2M3FN => 36,
                }
            }
            /// Look up a pinned numeric identifier; unknown values have no kind.
            pub fn from_i32(value: i32) -> Option<Self> {
                Self::VALUES.iter().copied().find(|ty| ty.value() == value)
            }
            /// Every known numeric and structural primitive kind.
            pub const VALUES: &'static [Self] = &[
                Self::PRIMITIVE_TYPE_INVALID, Self::TUPLE, Self::TOKEN, Self::OPAQUE_TYPE, Self::BUFFER,
                $(Self::$name),*
            ];
            /// Convert a numeric kind to an array format, rejecting structural kinds.
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
    };
}
primitives!(
    PRED => Pred, S1 => S1, S2 => S2, S4 => S4, S8 => S8, S16 => S16, S32 => S32, S64 => S64,
    U1 => U1, U2 => U2, U4 => U4, U8 => U8, U16 => U16, U32 => U32, U64 => U64,
    F16 => F16, BF16 => BF16, F32 => F32, F64 => F64, C64 => C64, C128 => C128,
    F8E5M2 => F8E5M2, F8E4M3 => F8E4M3, F8E4M3FN => F8E4M3FN,
    F8E4M3B11FNUZ => F8E4M3B11FNUZ, F8E3M4 => F8E3M4, F8E5M2FNUZ => F8E5M2FNUZ,
    F8E4M3FNUZ => F8E4M3FNUZ, F4E2M1FN => F4E2M1FN, F8E8M0FNU => F8E8M0FNU,
    F6E3M2FN => F6E3M2FN, F6E2M3FN => F6E2M3FN,
);

impl PrimitiveType {
    /// Whether the kind is an integer format, excluding predicates.
    pub fn is_integral(self) -> bool {
        self.array_element().is_ok_and(ElementType::is_integer)
    }
    /// Whether the kind is a signed integer format.
    pub fn is_signed_integral(self) -> bool {
        self.array_element().is_ok_and(ElementType::is_signed)
    }
    /// Whether the kind is a real floating-point format.
    pub fn is_floating(self) -> bool {
        self.array_element().is_ok_and(ElementType::is_float)
    }
    /// Whether the kind has real and imaginary floating-point components.
    pub fn is_complex(self) -> bool {
        matches!(self, Self::C64 | Self::C128)
    }
    /// Whether the kind can appear as an array element.
    pub fn is_array_element(self) -> bool {
        !matches!(
            self,
            Self::PRIMITIVE_TYPE_INVALID
                | Self::TUPLE
                | Self::TOKEN
                | Self::OPAQUE_TYPE
                | Self::BUFFER
        )
    }
    /// Compare kinds, allowing differing real floating-point precisions.
    pub fn same_ignoring_fp_precision(self, other: Self) -> bool {
        self == other || self.is_floating() && other.is_floating()
    }
    /// Return numeric encoding width; structural kinds have no width.
    pub fn bit_width(self) -> Option<u32> {
        self.array_element().ok().map(ElementType::bit_width)
    }
    /// Return numeric storage width; structural kinds have no width.
    pub fn storage_bit_width(self) -> Option<u32> {
        self.array_element()
            .ok()
            .map(ElementType::storage_bit_width)
    }
    /// Select the higher-precision numeric kind, rejecting structural kinds and ties.
    pub fn higher_precision(self, other: Self) -> Result<Self> {
        Ok(self
            .array_element()?
            .higher_precision(other.array_element()?)?
            .into())
    }
    /// Return the lowercase HLO spelling, including `opaque` for the opaque kind.
    pub fn hlo_name(self) -> String {
        if self == Self::OPAQUE_TYPE {
            "opaque".into()
        } else {
            format!("{self:?}").to_ascii_lowercase()
        }
    }
    /// Parse an exact lowercase HLO spelling, rejecting the invalid kind.
    pub fn from_hlo_name(name: &str) -> Result<Self> {
        Self::VALUES
            .iter()
            .copied()
            .find(|ty| *ty != Self::PRIMITIVE_TYPE_INVALID && ty.hlo_name() == name)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidShape {
                    message: format!("unknown primitive type {name:?}"),
                })
            })
    }
}
