//! MLIR bytecode framing, indexed tables and canonical-module interchange.
//!
//! Versions 0 through 6 follow the pinned MLIR format. [`write_module`] emits
//! textual-fallback payloads and [`read_module`] verifies the reconstructed
//! Rust module. Native builtin, StableHLO and uniform Quant payloads, and
//! StableHLO/Func/Module property schemas are decoded into that same IR.
//! Resources, successors and custom use-list orderings remain unsupported.
//! Successful container inspection alone is not semantic verification.

mod container;
mod payload;
mod primitives;
mod properties;
mod reader;
mod tables;
mod writer;

use std::fmt;

pub use container::{BytecodeContainer, BytecodeSection, SectionId};
pub use primitives::{decode_signed_varint, decode_varint, encode_signed_varint, encode_varint};
pub use reader::read_module;
pub use tables::{BytecodeDialect, BytecodeOperationName, BytecodeProperty, EncodedValue};
pub use writer::write_module;

/// A checked bytecode failure, with an absolute offset into the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeError {
    pub offset: usize,
    pub kind: BytecodeErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BytecodeErrorKind {
    UnexpectedEnd,
    InvalidMagic,
    UnsupportedVersion(u64),
    MissingTerminator,
    InvalidSection(u8),
    DuplicateSection(SectionId),
    MissingSection(SectionId),
    InvalidAlignment(u64),
    InvalidPadding(u8),
    InvalidStringLength,
    InvalidStringTerminator,
    InvalidIndex { table: &'static str, index: u64 },
    InvalidGroupSize,
    InvalidEntrySize,
    InvalidDialectVersionSection,
    OperationCountMismatch,
    TrailingData,
    ResourceLimit,
    InvalidIr(Box<crate::BuildError>),
    InvalidAttribute(Box<crate::AttributeValidationError>),
    InvalidType(Box<crate::TypeError>),
    Assembly(Box<crate::text::TextError>),
    LegacyComputationReferences,
    NestingLimit,
    InvalidUtf8,
    InvalidPayload(&'static str),
    InvalidOperationMask(u8),
    InvalidValue(u64),
    ValueCountMismatch,
    Unsupported(&'static str),
    UnsupportedEncoding { dialect: String, kind: &'static str },
}

impl fmt::Display for BytecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bytecode error at byte {}: ", self.offset)?;
        match &self.kind {
            BytecodeErrorKind::UnexpectedEnd => f.write_str("unexpected end of input"),
            BytecodeErrorKind::InvalidMagic => f.write_str("invalid MLIR bytecode magic"),
            BytecodeErrorKind::UnsupportedVersion(version) => {
                write!(f, "unsupported MLIR bytecode version {version}")
            }
            BytecodeErrorKind::MissingTerminator => {
                f.write_str("producer string is not null-terminated")
            }
            BytecodeErrorKind::InvalidSection(id) => write!(f, "invalid section ID {id}"),
            BytecodeErrorKind::DuplicateSection(id) => write!(f, "duplicate {id:?} section"),
            BytecodeErrorKind::MissingSection(id) => write!(f, "missing {id:?} section"),
            BytecodeErrorKind::InvalidAlignment(alignment) => {
                write!(f, "invalid section alignment {alignment}")
            }
            BytecodeErrorKind::InvalidPadding(byte) => {
                write!(f, "expected alignment byte 0xcb, got 0x{byte:02x}")
            }
            BytecodeErrorKind::InvalidStringLength => f.write_str("invalid string-table length"),
            BytecodeErrorKind::InvalidStringTerminator => {
                f.write_str("string-table entry is not null-terminated")
            }
            BytecodeErrorKind::InvalidIndex { table, index } => {
                write!(f, "invalid {table} index {index}")
            }
            BytecodeErrorKind::InvalidGroupSize => f.write_str("invalid dialect-group size"),
            BytecodeErrorKind::InvalidEntrySize => {
                f.write_str("attribute/type or property entry extends past its data section")
            }
            BytecodeErrorKind::InvalidDialectVersionSection => {
                f.write_str("expected a nested dialect-version section")
            }
            BytecodeErrorKind::OperationCountMismatch => {
                f.write_str("operation-name count does not match encoded groups")
            }
            BytecodeErrorKind::TrailingData => f.write_str("unexpected trailing section data"),
            BytecodeErrorKind::ResourceLimit => {
                f.write_str("bytecode table size exceeds available input or allocation limits")
            }
            BytecodeErrorKind::InvalidUtf8 => f.write_str("assembly payload is not UTF-8"),
            BytecodeErrorKind::InvalidPayload(expected) => write!(f, "expected {expected}"),
            BytecodeErrorKind::InvalidOperationMask(mask) => {
                write!(f, "invalid operation mask 0x{mask:02x}")
            }
            BytecodeErrorKind::InvalidValue(index) => {
                write!(f, "invalid or non-dominating value index {index}")
            }
            BytecodeErrorKind::ValueCountMismatch => {
                f.write_str("region value count does not match definitions")
            }
            BytecodeErrorKind::Unsupported(feature) => {
                write!(f, "bytecode decoding is not implemented for {feature}")
            }
            BytecodeErrorKind::UnsupportedEncoding { dialect, kind } => write!(
                f,
                "custom {kind} bytecode decoding is not implemented for dialect {dialect}"
            ),
            BytecodeErrorKind::InvalidIr(error) => write!(f, "invalid IR: {error}"),
            BytecodeErrorKind::InvalidAttribute(error) => write!(f, "invalid attribute: {error}"),
            BytecodeErrorKind::InvalidType(error) => write!(f, "invalid type: {error}"),
            BytecodeErrorKind::Assembly(error) => write!(f, "invalid payload assembly: {error}"),
            BytecodeErrorKind::LegacyComputationReferences => {
                f.write_str("legacy numeric called-computation references are not MLIR symbols")
            }
            BytecodeErrorKind::NestingLimit => {
                f.write_str("bytecode region or attribute/type nesting limit exceeded")
            }
        }
    }
}

impl std::error::Error for BytecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            BytecodeErrorKind::InvalidIr(error) => Some(error),
            BytecodeErrorKind::InvalidAttribute(error) => Some(error),
            BytecodeErrorKind::InvalidType(error) => Some(error),
            BytecodeErrorKind::Assembly(error) => Some(error),
            _ => None,
        }
    }
}
