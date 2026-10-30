//! Native pliron implementation of MLIR's `tensor` dialect.
#![doc = include_str!("../README.md")]
#![deny(missing_docs, unreachable_pub, private_interfaces, private_bounds)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

/// IR framework used by the dialect.
pub use pliron;
/// Shared error envelope and result type.
pub use pliron::result::{Error, Result};
/// Arithmetic operations for tensor region bodies.
pub use pliron_arith;
/// Shared tensor types, including ranked and unranked tensors.
pub use pliron_mlir_builtin;

mod error;
/// Structured diagnostics and recovery categories.
pub use error::{Diagnostic, RecoveryCategory};
#[cfg(feature = "mlir-interop")]
pub mod mlir;
pub mod ops;
mod support;
