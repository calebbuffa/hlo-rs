//! Native HLO inference and checked geometry utilities.
use crate::Diagnostic;
use pliron::{result::Result, r#type::TypeHandle};

pub mod control;
pub mod elementwise;
pub mod movement;
pub mod numerical;
/// Ordered computation parameter types and its single result type.
pub type Signature = (Vec<TypeHandle>, TypeHandle);

pub(crate) fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return pliron::verify_err_noloc!(Diagnostic::Inference {
            message: message.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn axes(rank: usize, values: &[i64], sorted: bool) -> Result<Vec<bool>> {
    let mut seen = vec![false; rank];
    if sorted && !values.windows(2).all(|v| v[0] < v[1]) {
        return pliron::verify_err_noloc!(Diagnostic::Inference {
            message: format!("axes must be strictly ascending; got {values:?} for rank {rank}"),
        });
    }
    for &axis in values {
        let index = usize::try_from(axis)
            .map_err(|_| pliron::verify_error_noloc!(Diagnostic::InvalidAxis { axis, rank }))?;
        if index >= rank {
            return pliron::verify_err_noloc!(Diagnostic::InvalidAxis { axis, rank });
        }
        if seen[index] {
            return pliron::verify_err_noloc!(Diagnostic::Inference {
                message: format!("axis {axis} is repeated in {values:?}"),
            });
        }
        seen[index] = true;
    }
    Ok(seen)
}
