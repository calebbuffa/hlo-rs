//! Gather verification shares dimension, slice and bound rules with inference.

use std::collections::HashMap;

use super::{BuildError, integer_attribute};
use crate::{
    Attribute, GatherDimensionNumbers, Instruction, StableHloAttribute, StableHloOp, Type, ValueId,
};

fn dimensions(instruction: &Instruction) -> &GatherDimensionNumbers {
    let Some(Attribute::StableHlo(attribute)) = instruction
        .operation()
        .attributes()
        .and_then(|a| a.get("dimension_numbers"))
    else {
        unreachable!("verified gather dimension attribute");
    };
    let StableHloAttribute::GatherDimensionNumbers(dimensions) = attribute.as_ref() else {
        unreachable!("verified gather dimension attribute");
    };
    dimensions
}

fn inference_error(instruction: &Instruction, error: crate::InferenceError) -> BuildError {
    BuildError::TypeInference {
        operation: instruction.operation().name(),
        error,
    }
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if !matches!(opcode, StableHloOp::Gather | StableHloOp::DynamicGather) {
        return Ok(());
    }
    let input = &available[&instruction.operands()[0]];
    let indices = &available[&instruction.operands()[1]];
    if opcode == StableHloOp::Gather {
        let sizes = integer_attribute(instruction, "slice_sizes")?;
        let sizes = sizes
            .iter()
            .enumerate()
            .map(|(dimension, &size)| {
                u64::try_from(size).map(Some).map_err(|_| {
                    inference_error(
                        instruction,
                        crate::InferenceError::InvalidSliceSize { dimension, size },
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        verify_with_sizes(instruction, input, indices, &sizes)
    } else {
        crate::infer_dynamic_gather_type(
            input,
            indices,
            &available[&instruction.operands()[2]],
            dimensions(instruction),
        )
        .map_err(|e| inference_error(instruction, e))?;
        let rank = super::shaped_dimensions(input)
            .expect("verified ranked input")
            .len();
        verify_with_sizes(instruction, input, indices, &vec![None; rank])
    }
}

pub(super) fn verify_with_sizes(
    instruction: &Instruction,
    input: &Type,
    indices: &Type,
    sizes: &[Option<u64>],
) -> Result<(), BuildError> {
    crate::inference::validate_gather_result(
        input,
        indices,
        instruction.results()[0].ty(),
        dimensions(instruction),
        sizes,
    )
    .map_err(|e| inference_error(instruction, e))
}
