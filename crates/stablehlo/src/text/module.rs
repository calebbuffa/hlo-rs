//! Generic MLIR module assembly, without a second IR model.

mod parser;
mod printer;

use super::{TextError, TextErrorKind};
use crate::Attribute;

const ENTRY_ATTRIBUTE: &str = "stablehlo_rust.entry";

fn error(kind: TextErrorKind) -> TextError {
    TextError { offset: 0, kind }
}

fn ir_error(error: crate::BuildError) -> TextError {
    self::error(TextErrorKind::InvalidIr(Box::new(error)))
}

fn dictionary(attribute: Attribute) -> Result<crate::Attributes, TextError> {
    match attribute {
        Attribute::Dictionary(attributes) => Ok(attributes),
        _ => Err(error(TextErrorKind::Expected("attribute dictionary"))),
    }
}

pub(super) use parser::parse;
pub(super) use printer::print;
