//! Native StableHLO dialect for the pliron intermediate representation.
//!
//! Public APIs are organized under [`attributes`], [`inference`], [`integer`],
//! [`ops`], [`sparse`], [`types`], and [`verification`].

#![doc = include_str!("../README.md")]
#![deny(missing_docs, unreachable_pub, private_interfaces, private_bounds)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

/// Re-exports pliron's IR types and traits used by the StableHLO API.
pub use pliron;
/// Shared error envelope and result type used by fallible IR APIs.
pub use pliron::result::{Error, Result};

/// StableHLO attributes, including operation metadata and sparse encodings.
pub mod attributes;
mod error;
/// Structured errors and recovery guidance emitted by this crate.
pub use error::{Diagnostic, RecoveryCategory};
pub mod inference;
/// Arbitrary-precision integer values for MLIR integer attributes.
pub mod integer;
/// MLIR text import and export, available with the `mlir-interop` feature.
#[cfg(feature = "mlir-interop")]
pub mod mlir;
pub mod ops;
pub mod sparse;
pub mod types;
pub mod verification;
