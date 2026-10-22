//! Native StableHLO dialect for the pliron intermediate representation.
//!
//! Public APIs are organized under [`attributes`], [`inference`], [`integer`],
//! [`ops`], [`sparse`], [`types`], and [`verification`].

#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

pub use pliron;

/// StableHLO attributes, including operation metadata and sparse encodings.
pub mod attributes;
mod error;
pub use error::{Diagnostic, RecoveryCategory};
pub mod inference;
/// Arbitrary-precision integer values for MLIR integer attributes.
pub mod integer;
pub mod ops;
pub mod sparse;
pub mod types;
pub mod verification;
