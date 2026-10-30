//! MLIR builtin representations shared by independent pliron dialects.
#![doc = include_str!("../README.md")]
#![deny(missing_docs, unreachable_pub, private_interfaces, private_bounds)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

/// IR framework used by these representations.
pub use pliron;
/// Shared error envelope and result type.
pub use pliron::result::{Error, Result};

pub mod attributes;
mod error;
pub use error::{Diagnostic, RecoveryCategory};
/// Arbitrary-precision integer values for exact attribute payloads.
pub mod integer;
pub mod interfaces;
#[cfg(feature = "mlir-interop")]
pub mod mlir;
pub mod types;
