//! MLIR textual interchange.
//!
//! Type and attribute assembly are available independently of module assembly. Parsing and
//! printing validate the represented type; neither API skips unknown syntax.

use std::fmt;

use crate::{Attribute, AttributeValidationError, Type, TypeError};

mod affine;
mod aliases;
mod arrays;
mod attributes;
mod buffers;
mod complex;
mod cursor;
mod floats;
mod geometry;
mod literals;
mod locations;
mod module;
mod scalars;
mod sparse;
mod sparse_encoding;
mod types;

/// A textual assembly error with a UTF-8 byte offset into the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextError {
    pub offset: usize,
    pub kind: TextErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextErrorKind {
    Expected(&'static str),
    InvalidNumber,
    TrailingInput,
    NestingLimit,
    ResourceLimit,
    InvalidType(TypeError),
    InvalidAttribute(AttributeValidationError),
    InvalidIr(Box<crate::BuildError>),
    UnknownValue(String),
    UnknownAlias(String),
    DuplicateField(String),
    Unsupported(&'static str),
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "assembly error at byte {}: ", self.offset)?;
        match &self.kind {
            TextErrorKind::Expected(expected) => write!(f, "expected {expected}"),
            TextErrorKind::InvalidNumber => write!(f, "invalid or out-of-range number"),
            TextErrorKind::TrailingInput => write!(f, "unexpected trailing input"),
            TextErrorKind::NestingLimit => write!(f, "assembly nesting limit exceeded"),
            TextErrorKind::ResourceLimit => {
                write!(f, "assembly expansion exceeds the resource limit")
            }
            TextErrorKind::InvalidType(error) => write!(f, "invalid type: {error}"),
            TextErrorKind::InvalidAttribute(error) => write!(f, "invalid attribute: {error}"),
            TextErrorKind::InvalidIr(error) => write!(f, "invalid IR: {error}"),
            TextErrorKind::UnknownValue(name) => {
                write!(f, "unknown or non-dominating SSA value %{name}")
            }
            TextErrorKind::UnknownAlias(name) => write!(f, "undefined assembly alias {name}"),
            TextErrorKind::DuplicateField(field) => write!(f, "duplicate field {field}"),
            TextErrorKind::Unsupported(feature) => {
                write!(f, "assembly is not implemented for {feature}")
            }
        }
    }
}

impl std::error::Error for TextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            TextErrorKind::InvalidType(error) => Some(error),
            TextErrorKind::InvalidAttribute(error) => Some(error),
            TextErrorKind::InvalidIr(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

/// Parses one complete MLIR type, allowing whitespace and line comments.
///
/// Errors report byte offsets. Inputs deeper than 128 nested types are rejected
/// rather than recursing without a bound. Unsupported representations return an
/// explicit error, not a substitute type.
pub fn parse_type(input: &str) -> Result<Type, TextError> {
    let mut parser = cursor::Cursor::new(input);
    let ty = parser.ty(0)?;
    parser.finish()?;
    ty.verify().map_err(|error| TextError {
        offset: parser.position(),
        kind: TextErrorKind::InvalidType(error),
    })?;
    Ok(ty)
}

/// Prints a valid type as MLIR assembly.
///
/// The printer reports unimplemented attribute/type assembly explicitly. It
/// never emits Rust debug syntax or silently drops encoding/layout metadata.
pub fn print_type(ty: &Type) -> Result<String, TextError> {
    types::print(ty, 0)
}

/// Parses one complete attribute and validates its represented constraints.
pub fn parse_attribute(input: &str) -> Result<Attribute, TextError> {
    let mut parser = cursor::Cursor::new(input);
    let attribute = parser.attribute(0)?;
    parser.finish()?;
    attribute.verify().map_err(|error| TextError {
        offset: parser.position(),
        kind: TextErrorKind::InvalidAttribute(error),
    })?;
    Ok(attribute)
}

/// Prints attribute assembly without discarding unsupported metadata.
pub fn print_attribute(attribute: &Attribute) -> Result<String, TextError> {
    attributes::print(attribute, 0)
}

/// Parses a module using generic MLIR operations or dialect custom assembly.
///
/// Supports `builtin.module`, `func.func`, `func.call`, `func.return`, and
/// StableHLO operations with nested single-block regions. Standard `module`,
/// `func.func`, `func.call`, return syntax and the pinned dialect's custom
/// operation formats are accepted too, including compact reducers and loops.
/// Top-level functions without an explicit module form an implicit module;
/// empty files form empty modules.
/// File-scoped type and ordinary attribute aliases resolve to canonical values;
/// undefined references and excessive recursive expansion are explicit errors.
/// Structured source locations are retained independently of operation attributes.
/// Trailing operation and block-argument locations also accept forward aliases.
/// Unmodeled metadata and operations are rejected explicitly.
pub fn parse_module(input: &str) -> Result<crate::Module, TextError> {
    module::parse(input)
}

/// Prints a verified module using MLIR's generic operation assembly.
///
/// Consumer-selected entry metadata is encoded explicitly as
/// `stablehlo_rust.entry`; it is not a StableHLO specification attribute.
pub fn print_module(module: &crate::Module) -> Result<String, TextError> {
    module::print(module)
}
