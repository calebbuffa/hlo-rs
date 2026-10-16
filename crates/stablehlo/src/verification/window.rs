//! Shared window geometry for convolution and windowed reductions.

use super::{BuildError, shaped_dimensions};
use crate::{Attribute, Instruction, ScalarValue};

fn invalid(instruction: &Instruction) -> BuildError {
    BuildError::InvalidDimensionNumbers {
        operation: instruction.operation().name(),
    }
}

pub(super) fn padding(
    instruction: &Instruction,
    rank: usize,
) -> Result<Vec<Option<[i128; 2]>>, BuildError> {
    match instruction
        .operation()
        .attributes()
        .and_then(|attributes| attributes.get("padding"))
    {
        None => Ok(vec![Some([0, 0]); rank]),
        Some(Attribute::DenseElements { ty, values })
            if shaped_dimensions(ty) == Some(&[Some(rank as u64), Some(2)]) =>
        {
            values
                .chunks_exact(2)
                .map(|pair| match (&pair[0], &pair[1]) {
                    (ScalarValue::Integer(low), ScalarValue::Integer(high)) => {
                        Ok(Some([*low, *high]))
                    }
                    _ => Err(invalid(instruction)),
                })
                .collect()
        }
        _ => Err(invalid(instruction)),
    }
}
