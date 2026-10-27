//! pliron implementation of the OpenXLA CHLO dialect.

#![doc = include_str!("../README.md")]
#![deny(missing_docs, unreachable_pub, private_interfaces, private_bounds)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
/// The IR framework used by ClientHLO, re-exported to keep context and handle types aligned.
pub use pliron;
/// Shared error envelope and result type used by fallible IR APIs.
pub use pliron::result::{Error, Result};
/// The underlying StableHLO dialect, including its types, operations, and optional MLIR API.
pub use stablehlo;
pub mod attributes;
mod error;
/// Structured diagnostics and recovery categories for failed dialect operations.
pub use error::{Diagnostic, RecoveryCategory};
pub mod inference;
pub mod ops;
pub mod passes;
