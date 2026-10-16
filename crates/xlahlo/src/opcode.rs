use std::{fmt, str::FromStr};

use crate::{Error, Result};

macro_rules! opcodes {
    ($($variant:ident => ($name:literal, $arity:expr)),+ $(,)?) => {
        /// The opcode catalog generated from the current `extern/xla` checkout.
        ///
        /// Recognition is not a claim of semantic verification support.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Opcode { $($variant),+ }

        impl Opcode {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn name(self) -> &'static str {
                match self { $(Self::$variant => $name),+ }
            }

            pub const fn arity(self) -> Option<usize> {
                match self { $(Self::$variant => $arity),+ }
            }
        }

        impl FromStr for Opcode {
            type Err = Error;

            fn from_str(name: &str) -> Result<Self> {
                match name {
                    $($name => Ok(Self::$variant)),+,
                    _ => Err(Error::new("opcode", format!("unknown HLO opcode {name:?}"))),
                }
            }
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/opcode_list.rs"));

impl Opcode {
    pub const fn supports_result_accuracy(self) -> bool {
        matches!(
            self,
            Self::Acos
                | Self::Acosh
                | Self::Asin
                | Self::Asinh
                | Self::Atanh
                | Self::Cbrt
                | Self::Cos
                | Self::Cosh
                | Self::Erf
                | Self::Exp
                | Self::Exp2
                | Self::Expm1
                | Self::Log
                | Self::Log1p
                | Self::Log2
                | Self::Logistic
                | Self::Rsqrt
                | Self::Sin
                | Self::Sinh
                | Self::Sqrt
                | Self::Tan
                | Self::Tanh
        )
    }
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
