//! StableHLO operation wrappers grouped behind one public operations namespace.
//!
//! Constructors that accept an explicit result type use that type as provided and
//! do not run operation verification. Builders named `new_inferred` document the
//! inference and input checks they perform; those checks do not replace full
//! operation verification. Call [`pliron::common_traits::Verify::verify`] before
//! relying on an operation's invariants.

use crate::Diagnostic;
use pliron::result::Result;

pub(crate) fn builder_option<T>(
    value: Option<T>,
    operation: &'static str,
    reason: &'static str,
) -> Result<T> {
    value.ok_or_else(|| {
        Diagnostic::InvalidOperation {
            operation: operation.to_owned(),
            location: "<builder>".to_owned(),
            reason,
        }
        .into()
    })
}

pub(crate) fn builder_result<T>(value: Result<T>, operation: &'static str) -> Result<T> {
    value.map_err(|error| Diagnostic::builder_failure(operation, error))
}

mod constants;
mod control;
mod elementwise;
mod shape;
mod tensor;
pub use constants::*;
pub use control::*;
pub use elementwise::*;
pub use shape::*;
pub use tensor::*;
mod collective;
mod compute;
mod indexing;
mod reduction;
pub use collective::*;
pub use compute::*;
pub use indexing::*;
pub use reduction::*;
