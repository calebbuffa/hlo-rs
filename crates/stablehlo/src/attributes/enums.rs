//! Textual spellings for StableHLO enumeration values.

use std::{error::Error, fmt, str::FromStr};

/// Error returned when a string is not a valid spelling for an enum.
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

macro_rules! enum_spelling {
    ($type:ty, $name:literal { $($variant:path => $spelling:literal),+ $(,)? }) => {
        impl $type {
            /// StableHLO attribute name associated with the enum.
            pub const NAME: &'static str = $name;

            /// Returns the canonical spelling used by StableHLO text.
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
    };
}

enum_spelling!(super::Precision, "precision" {
    super::Precision::Default => "DEFAULT",
    super::Precision::High => "HIGH",
    super::Precision::Highest => "HIGHEST",
});
enum_spelling!(super::ResultAccuracyMode, "result_accuracy_mode" {
    super::ResultAccuracyMode::Default => "DEFAULT",
    super::ResultAccuracyMode::Highest => "HIGHEST",
    super::ResultAccuracyMode::Tolerance => "TOLERANCE",
});
enum_spelling!(super::FftTypeAttr, "fft_type" {
    super::FftTypeAttr::Fft => "FFT",
    super::FftTypeAttr::Ifft => "IFFT",
    super::FftTypeAttr::Rfft => "RFFT",
    super::FftTypeAttr::Irfft => "IRFFT",
});
enum_spelling!(super::ComparisonDirection, "comparison_direction" {
    super::ComparisonDirection::Eq => "EQ",
    super::ComparisonDirection::Ne => "NE",
    super::ComparisonDirection::Ge => "GE",
    super::ComparisonDirection::Gt => "GT",
    super::ComparisonDirection::Le => "LE",
    super::ComparisonDirection::Lt => "LT",
});
enum_spelling!(super::ComparisonType, "comparison_type" {
    super::ComparisonType::NoType => "NOTYPE",
    super::ComparisonType::Float => "FLOAT",
    super::ComparisonType::TotalOrder => "TOTALORDER",
    super::ComparisonType::Signed => "SIGNED",
    super::ComparisonType::Unsigned => "UNSIGNED",
    super::ComparisonType::WeakOrder => "WEAKORDER",
});
enum_spelling!(super::RngAlgorithmAttr, "rng_algorithm" {
    super::RngAlgorithmAttr::Default => "DEFAULT",
    super::RngAlgorithmAttr::ThreeFry => "THREE_FRY",
    super::RngAlgorithmAttr::Philox => "PHILOX",
});
enum_spelling!(super::RngDistributionAttr, "rng_distribution" {
    super::RngDistributionAttr::Uniform => "UNIFORM",
    super::RngDistributionAttr::Normal => "NORMAL",
});
enum_spelling!(crate::ops::TransposeMode, "transpose" {
    crate::ops::TransposeMode::None => "NO_TRANSPOSE",
    crate::ops::TransposeMode::Transpose => "TRANSPOSE",
    crate::ops::TransposeMode::Adjoint => "ADJOINT",
});

pub(crate) fn is_valid(name: &str, spelling: &str) -> bool {
    match name {
        "precision" => spelling.parse::<super::Precision>().is_ok(),
        "result_accuracy_mode" => spelling.parse::<super::ResultAccuracyMode>().is_ok(),
        "fft_type" => spelling.parse::<super::FftTypeAttr>().is_ok(),
        "comparison_direction" => spelling.parse::<super::ComparisonDirection>().is_ok(),
        "comparison_type" => spelling.parse::<super::ComparisonType>().is_ok(),
        "rng_algorithm" => spelling.parse::<super::RngAlgorithmAttr>().is_ok(),
        "rng_distribution" => spelling.parse::<super::RngDistributionAttr>().is_ok(),
        "transpose" => {
            spelling.parse::<crate::ops::TransposeMode>().is_ok() || spelling == "TRANSPOSE_INVALID"
        }
        _ => false,
    }
}
