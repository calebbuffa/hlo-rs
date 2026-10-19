//! StableHLO operation wrappers grouped behind one public operations namespace.

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
