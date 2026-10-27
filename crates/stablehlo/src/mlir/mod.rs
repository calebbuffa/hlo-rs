//! StableHLO MLIR assembly, distinct from pliron's internal textual format.
//!
//! Export uses the framework's translation interfaces for every native operation,
//! including nested regions. Import retains the bounded straight-line parser;
//! `pliron-mlir-interop` does not provide an MLIR parser.
//!
//! Use [`import`](crate::mlir::import) and [`export`](fn@crate::mlir::export)
//! for complete modules. Low-level assembly helpers and framework interfaces are
//! grouped under [`assembly`](crate::mlir::assembly).
//!
//! ```
//! use stablehlo::{mlir, pliron::context::Context};
//!
//! let mut context = Context::new();
//! let module = mlir::import(&mut context, "module {}")?;
//! let text = mlir::export(&context, &module)?;
//! assert_eq!(text, "module {\n}");
//! # Ok::<(), mlir::TextError>(())
//! ```

mod affine;
pub mod assembly;
mod attributes;
mod complex;
mod cursor;
mod dialect_attributes;
mod floats;
mod location;
mod module;
mod module_attributes;
mod module_ops;
mod printer;
mod translation;
mod types;
#[cfg(test)]
mod upstream_tests;

use std::fmt;

pub use module::{parse_module as import, print_module as export};

/// A byte position and recoverable assembly error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextError {
    /// Byte offset in the rejected input; zero for export failures.
    pub offset: usize,
    /// Structured reason for the failure.
    pub kind: TextErrorKind,
}

/// Syntax, representation, or verification failure in MLIR assembly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextErrorKind {
    /// Required syntax was absent.
    Expected(&'static str),
    /// A numeric literal was invalid or out of range.
    InvalidNumber,
    /// Unconsumed text followed a complete entity.
    TrailingInput,
    /// Assembly exceeded the parser's nesting limit.
    NestingLimit,
    /// A parsed type violated its constraints.
    InvalidType(String),
    /// A parsed attribute violated its constraints.
    InvalidAttribute(String),
    /// A field or symbol was defined more than once.
    DuplicateField(String),
    /// Constructed IR or an export violated its constraints.
    InvalidIr(String),
    /// An SSA use had no definition in scope.
    UnknownValue(String),
    /// The input requires syntax or a representation not supported by the parser.
    Unsupported(&'static str),
}

impl fmt::Display for TextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "assembly error at byte {}: ", self.offset)?;
        match &self.kind {
            TextErrorKind::Expected(expected) => write!(formatter, "expected {expected}"),
            TextErrorKind::InvalidNumber => formatter.write_str("invalid or out-of-range number"),
            TextErrorKind::TrailingInput => formatter.write_str("unexpected trailing input"),
            TextErrorKind::NestingLimit => formatter.write_str("assembly nesting limit exceeded"),
            TextErrorKind::InvalidType(reason) => write!(formatter, "invalid type: {reason}"),
            TextErrorKind::InvalidAttribute(reason) => {
                write!(formatter, "invalid attribute: {reason}")
            }
            TextErrorKind::DuplicateField(name) => write!(formatter, "duplicate field {name}"),
            TextErrorKind::InvalidIr(reason) => write!(formatter, "invalid IR: {reason}"),
            TextErrorKind::UnknownValue(name) => write!(formatter, "unknown SSA value %{name}"),
            TextErrorKind::Unsupported(feature) => {
                write!(formatter, "unsupported assembly: {feature}")
            }
        }
    }
}

impl std::error::Error for TextError {}
