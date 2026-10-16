//! Shared scatter geometry and scalar update computation contracts.

use std::collections::HashMap;

use super::{BuildError, regions, shaped_dimensions};
use crate::{Attribute, Instruction, StableHloAttribute, StableHloOp, Type, ValueId};

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if opcode != StableHloOp::Scatter {
        return Ok(());
    }
    let count = instruction.results().len();
    let inputs = instruction.operands()[..count]
        .iter()
        .map(|id| available[id].clone())
        .collect::<Vec<_>>();
    let updates = instruction.operands()[count + 1..]
        .iter()
        .map(|id| available[id].clone())
        .collect::<Vec<_>>();
    let indices = &available[&instruction.operands()[count]];
    let Some(Attribute::StableHlo(attribute)) = instruction
        .operation()
        .attributes()
        .and_then(|attributes| attributes.get("scatter_dimension_numbers"))
    else {
        unreachable!("verified scatter dimensions");
    };
    let StableHloAttribute::ScatterDimensionNumbers(dimensions) = attribute.as_ref() else {
        unreachable!("verified scatter dimensions");
    };
    let accumulators = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            regions::verify_accumulator(instruction, available, input, index, count).cloned()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let error = |error| BuildError::TypeInference {
        operation: opcode.mnemonic(),
        error,
    };
    let inferred =
        crate::infer_scatter_types(&inputs, indices, &updates, dimensions, &accumulators)
            .map_err(error)?;
    let witnesses = inputs
        .iter()
        .chain(instruction.results().iter().map(|result| result.ty()))
        .collect::<Vec<_>>();
    crate::inference::validate_scatter_shapes(
        &witnesses,
        indices,
        &updates.iter().collect::<Vec<_>>(),
        dimensions,
    )
    .map_err(error)?;
    for (index, expected) in inferred.iter().enumerate() {
        regions::check_result(
            instruction,
            index,
            shaped_dimensions(expected).expect("inferred tensor"),
            expected,
        )?;
    }
    Ok(())
}
