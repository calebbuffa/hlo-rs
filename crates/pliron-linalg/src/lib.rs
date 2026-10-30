//! pliron implementation of the MLIR `linalg` dialect.

#![doc = include_str!("../README.md")]
#![deny(missing_docs, unreachable_pub, private_interfaces, private_bounds)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
/// The IR framework used by this dialect, re-exported to keep context and handle types aligned.
pub use pliron;
/// Shared error envelope and result type used by fallible IR APIs.
pub use pliron::result::{Error, Result};
/// The `arith` dialect used in structured-op bodies.
pub use pliron_arith;
/// The `math` dialect used in structured-op bodies.
pub use pliron_math;
/// The StableHLO crate that supplies tensor types and affine maps.
pub use stablehlo;
pub mod attributes;
mod error;
/// Structured diagnostics and recovery categories for failed dialect operations.
pub use error::{Diagnostic, RecoveryCategory};
#[cfg(feature = "mlir-interop")]
pub mod mlir;
pub mod ops;
mod support;
