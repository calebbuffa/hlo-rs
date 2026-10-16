use crate::{Attribute, AttributeValidationError};

/// Structured builtin MLIR source provenance.
///
/// Names and filenames are byte strings, like MLIR StringAttr. Locations are
/// metadata, not operation attributes; the attribute variant allows locations
/// to appear in aliases and other metadata as permitted by MLIR.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Location {
    Unknown,
    FileLine {
        file: Vec<u8>,
        line: u32,
    },
    FileLineCol {
        file: Vec<u8>,
        line: u32,
        column: u32,
    },
    FileRange {
        file: Vec<u8>,
        start_line: u32,
        start_column: u32,
        end_line: Option<u32>,
        end_column: u32,
    },
    Name {
        name: Vec<u8>,
        child: Option<Box<Location>>,
    },
    CallSite {
        callee: Box<Location>,
        caller: Box<Location>,
    },
    Fused {
        locations: Vec<Location>,
        metadata: Option<Box<Attribute>>,
    },
}

impl Location {
    pub fn verify(&self) -> Result<(), AttributeValidationError> {
        crate::attribute::Attribute::verify_location(self)
    }
}
