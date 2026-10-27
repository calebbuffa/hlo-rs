//! Low-level MLIR assembly and dialect-composition interfaces.
//!
//! Most callers should use [`super::import`] and [`super::export`] instead.
//! These helpers handle individual entities without implying module verification.

use super::{TextError, TextErrorKind, floats};
use crate::types::ElementType;

pub use super::attributes::{parse_attribute, parse_dictionary, parse_named_attributes};
pub use super::module::print_operation_attributes;
pub use super::printer::{
    mlir_attribute as print_attribute, mlir_dictionary as print_dictionary, mlir_type as print_type,
};
pub use super::types::parse_type;
/// Framework interfaces and fallible printer for composing dialect translations.
pub use pliron_mlir_interop::{MlirPrinter, ToMlirAttr, ToMlirOp, ToMlirType};

/// Parses one complete MLIR floating-point literal into its destination bit pattern.
///
/// Decimal literals round directly to the destination, without an intermediate
/// host `f64`. Hexadecimal literals preserve raw bits, including signaling NaNs.
///
/// # Errors
///
/// Returns [`TextError`] when `element` is not a floating-point format or the
/// literal is malformed or cannot be represented by that format's bit width.
pub fn parse_float_bits(input: &str, element: ElementType) -> Result<u64, TextError> {
    if !element.is_float() {
        return Err(TextError {
            offset: 0,
            kind: TextErrorKind::Expected("floating element type"),
        });
    }
    floats::parse(input, element).map_err(|kind| TextError { offset: 0, kind })
}

/// Prints a lossless MLIR hexadecimal floating-point literal.
///
/// Preserves all payload bits, including signed zeros and NaNs, without
/// converting through a host floating-point value.
///
/// # Errors
///
/// Returns [`TextError`] when `element` is not a floating-point format or `bits`
/// contains set bits outside the format's width.
pub fn print_float_bits(bits: u64, element: ElementType) -> Result<String, TextError> {
    if !element.is_float() {
        return Err(TextError {
            offset: 0,
            kind: TextErrorKind::Expected("floating element type"),
        });
    }
    let width = element.bit_width().ok_or(TextError {
        offset: 0,
        kind: TextErrorKind::InvalidNumber,
    })?;
    if width < 64 && bits >> width != 0 {
        return Err(TextError {
            offset: 0,
            kind: TextErrorKind::InvalidNumber,
        });
    }
    Ok(format!("0x{bits:X}"))
}
