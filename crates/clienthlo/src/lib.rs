//! pliron implementation of the OpenXLA CHLO dialect.

#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
/// The IR framework used by ClientHLO, re-exported to keep context and handle types aligned.
pub use pliron;
/// Shared error envelope and result type used by fallible IR APIs.
pub use pliron::result::{Error, Result};
pub mod attributes;
mod error;
pub use error::{Diagnostic, RecoveryCategory};
pub mod inference;
pub mod ops;
