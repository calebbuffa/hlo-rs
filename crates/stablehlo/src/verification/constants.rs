//! Verification that depends on dominating constant SSA operands.

use std::collections::HashMap;

use super::{
    BuildError, inferred_shape_matches, integer_attribute, shaped_dimensions,
    static_element_counts_match,
};
use crate::{Attribute, Computation, Instruction, Operation, ScalarValue, StableHloOp, ValueId};

type Constants<'a> = HashMap<ValueId, &'a [ScalarValue]>;

pub(super) fn verify(computation: &Computation) -> Result<(), BuildError> {
    verify_instructions(computation.instructions(), &Constants::new(), computation)
}

fn verify_instructions<'a>(
    instructions: &'a [Instruction],
    enclosing: &Constants<'a>,
    computation: &Computation,
) -> Result<(), BuildError> {
    let mut constants = enclosing.clone();
    for instruction in instructions {
        verify_operation(instruction, &constants, computation)?;
        let captures = if instruction.operation().is_isolated_from_above() {
            Constants::new()
        } else {
            constants.clone()
        };
        for region in instruction.regions() {
            for block in region.blocks() {
                verify_instructions(block.instructions(), &captures, computation)?;
            }
        }
        if let Operation::StableHlo {
            opcode: StableHloOp::Constant,
            attributes,
        } = instruction.operation()
            && let Some(Attribute::DenseElements { values, .. }) = attributes.get("value")
            && let Some(result) = instruction.result()
        {
            constants.insert(result, values);
        }
    }
    Ok(())
}

fn integers(constants: &Constants<'_>, value: ValueId) -> Option<Vec<i128>> {
    constants
        .get(&value)?
        .iter()
        .map(|value| match value {
            ScalarValue::Integer(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn invalid(instruction: &Instruction, value: ValueId, reason: &'static str) -> BuildError {
    BuildError::InvalidConstantOperand {
        operation: instruction.operation().name(),
        value,
        reason,
    }
}

fn result_shape(instruction: &Instruction) -> &[Option<u64>] {
    // Type/schema verification precedes this traversal.
    shaped_dimensions(instruction.results()[0].ty()).expect("verified shaped result")
}

fn verify_shape(
    instruction: &Instruction,
    value: ValueId,
    shape: &[i128],
) -> Result<Vec<Option<u64>>, BuildError> {
    let result_shape = result_shape(instruction);
    if shape.len() != result_shape.len() {
        return Err(invalid(
            instruction,
            value,
            "shape vector length must equal result rank",
        ));
    }
    let dimensions = shape
        .iter()
        .map(|&size| {
            u64::try_from(size)
                .ok()
                .filter(|&size| size <= i64::MAX as u64)
                .map(Some)
                .ok_or_else(|| {
                    invalid(
                        instruction,
                        value,
                        "shape dimensions must be nonnegative int64 values",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !inferred_shape_matches(&dimensions, instruction.results()[0].ty()) {
        return Err(invalid(
            instruction,
            value,
            "constant shape conflicts with result dimensions",
        ));
    }
    Ok(dimensions)
}

fn verify_operation(
    instruction: &Instruction,
    constants: &Constants<'_>,
    computation: &Computation,
) -> Result<(), BuildError> {
    let Operation::StableHlo { opcode, .. } = instruction.operation() else {
        return Ok(());
    };
    match opcode {
        StableHloOp::SetDimensionSize => {
            let size_operand = instruction.operands()[1];
            if let Some(values) = integers(constants, size_operand) {
                let size = values
                    .first()
                    .and_then(|&value| i32::try_from(value).ok())
                    .ok_or_else(|| {
                        invalid(
                            instruction,
                            size_operand,
                            "size must be a scalar i32 constant",
                        )
                    })?;
                let expected = crate::infer_set_dimension_size_type(
                    computation
                        .value_type(instruction.operands()[0])
                        .expect("verified operand"),
                    computation.value_type(size_operand).expect("verified size"),
                    super::dimension_attribute(instruction)?,
                    Some(size),
                )
                .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
                if crate::infer_most_specific_type(&[
                    expected,
                    instruction.results()[0].ty().clone(),
                ])
                .is_err()
                {
                    return Err(invalid(
                        instruction,
                        size_operand,
                        "constant size conflicts with inferred result dimensions",
                    ));
                }
            }
        }
        StableHloOp::Rng => {
            let operands = instruction.operands();
            let shape_operand = operands[2];
            if let Some(values) = integers(constants, shape_operand) {
                let values = values
                    .into_iter()
                    .map(|value| {
                        i64::try_from(value).map_err(|_| {
                            invalid(
                                instruction,
                                shape_operand,
                                "shape dimensions must be nonnegative int64 values",
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let distribution = match instruction
                    .operation()
                    .attributes()
                    .and_then(|attrs| attrs.get("rng_distribution"))
                {
                    Some(Attribute::Enum { value, .. }) if value == "NORMAL" => {
                        crate::RngDistribution::Normal
                    }
                    _ => crate::RngDistribution::Uniform,
                };
                let inferred = crate::infer_rng_type(
                    computation.value_type(operands[0]).expect("verified RNG a"),
                    computation.value_type(operands[1]).expect("verified RNG b"),
                    computation
                        .value_type(shape_operand)
                        .expect("verified RNG shape"),
                    Some(&values),
                    distribution,
                )
                .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
                if !inferred_shape_matches(
                    shaped_dimensions(&inferred).expect("inferred RNG tensor"),
                    instruction.results()[0].ty(),
                ) {
                    return Err(invalid(
                        instruction,
                        shape_operand,
                        "constant shape conflicts with result dimensions",
                    ));
                }
            }
        }
        StableHloOp::DynamicReshape
        | StableHloOp::DynamicBroadcastInDim
        | StableHloOp::DynamicIota => {
            let shape_operand = match opcode {
                StableHloOp::DynamicIota => instruction.operands()[0],
                _ => instruction.operands()[1],
            };
            let Some(shape) = integers(constants, shape_operand) else {
                return Ok(());
            };
            let dimensions = verify_shape(instruction, shape_operand, &shape)?;
            if *opcode == StableHloOp::DynamicReshape {
                let input_type = computation
                    .value_type(instruction.operands()[0])
                    .expect("verified operand");
                let input = shaped_dimensions(input_type).expect("verified tensor operand");
                if !static_element_counts_match(input, &dimensions) {
                    return Err(invalid(
                        instruction,
                        shape_operand,
                        "reshape changes the element count",
                    ));
                }
                if let crate::Type::QuantizedTensor(output) = instruction.results()[0].ty() {
                    let known =
                        crate::Type::QuantizedTensor(crate::QuantizedTensorType::from_element(
                            dimensions.clone(),
                            output.element().clone(),
                        ));
                    if known.verify().is_err()
                        || !super::reshape_quantization_compatible(input_type, &known)
                    {
                        return Err(invalid(
                            instruction,
                            shape_operand,
                            "constant shape violates reshape quantization constraints",
                        ));
                    }
                }
            }
            if *opcode == StableHloOp::DynamicBroadcastInDim {
                let input = shaped_dimensions(
                    computation
                        .value_type(instruction.operands()[0])
                        .expect("verified operand"),
                )
                .expect("verified tensor operand");
                for (&input_size, axis) in input
                    .iter()
                    .zip(integer_attribute(instruction, "broadcast_dimensions")?)
                {
                    let axis = usize::try_from(axis).expect("verified broadcast dimension");
                    if let Some(input_size) = input_size
                        && input_size != 1
                        && dimensions[axis] != Some(input_size)
                    {
                        return Err(invalid(
                            instruction,
                            shape_operand,
                            "shape cannot broadcast the operand dimensions",
                        ));
                    }
                }
            }
        }
        StableHloOp::DynamicPad => verify_pad(instruction, constants, computation)?,
        StableHloOp::DynamicGather => {
            let value = instruction.operands()[2];
            if let Some(sizes) = integers(constants, value) {
                let sizes = sizes
                    .into_iter()
                    .map(|size| {
                        u64::try_from(size)
                            .ok()
                            .filter(|&size| size <= i64::MAX as u64)
                            .map(Some)
                            .ok_or_else(|| {
                                invalid(
                                    instruction,
                                    value,
                                    "slice sizes must be nonnegative int64 values",
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                super::gather::verify_with_sizes(
                    instruction,
                    computation
                        .value_type(instruction.operands()[0])
                        .expect("verified operand"),
                    computation
                        .value_type(instruction.operands()[1])
                        .expect("verified operand"),
                    &sizes,
                )?;
            }
        }
        StableHloOp::DynamicConv => {
            let value = instruction.operands()[2];
            if let Some(values) = integers(constants, value) {
                let padding = values
                    .chunks_exact(2)
                    .map(|pair| Some([pair[0], pair[1]]))
                    .collect::<Vec<_>>();
                super::convolution::verify_with_padding(
                    instruction,
                    computation
                        .value_type(instruction.operands()[0])
                        .expect("verified operand"),
                    computation
                        .value_type(instruction.operands()[1])
                        .expect("verified operand"),
                    &padding,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn verify_pad(
    instruction: &Instruction,
    constants: &Constants<'_>,
    computation: &Computation,
) -> Result<(), BuildError> {
    let interior_operand = instruction.operands()[4];
    let Some(interior) = integers(constants, interior_operand) else {
        return Ok(());
    };
    if interior.iter().any(|&size| size < 0) {
        return Err(invalid(
            instruction,
            interior_operand,
            "interior padding must be nonnegative",
        ));
    }
    let low_operand = instruction.operands()[2];
    let high_operand = instruction.operands()[3];
    let (Some(low), Some(high)) = (
        integers(constants, low_operand),
        integers(constants, high_operand),
    ) else {
        return Ok(());
    };
    let input = shaped_dimensions(
        computation
            .value_type(instruction.operands()[0])
            .expect("verified operand"),
    )
    .expect("verified tensor operand");
    for (index, (&input, &output)) in input.iter().zip(result_shape(instruction)).enumerate() {
        if let (Some(input), Some(output)) = (input, output) {
            let padding = i128::from(input.saturating_sub(1))
                .checked_mul(interior[index])
                .and_then(|value| value.checked_add(i128::from(input)))
                .and_then(|value| value.checked_add(low[index]))
                .and_then(|value| value.checked_add(high[index]));
            if padding != Some(i128::from(output)) {
                return Err(invalid(
                    instruction,
                    interior_operand,
                    "padding values conflict with the result shape",
                ));
            }
        }
    }
    Ok(())
}
