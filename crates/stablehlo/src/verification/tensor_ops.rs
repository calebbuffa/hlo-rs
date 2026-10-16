//! Shape and element relationships for non-region tensor operations.

use std::collections::HashMap;

use super::{
    BuildError, compatible_shapes, compatible_tensor_types, inferred_shape_matches,
    integer_attribute, shaped_dimensions,
};
use crate::{Attribute, ElementType, Instruction, StableHloOp, Type, ValueId};

fn shape(ty: &Type) -> &[Option<u64>] {
    shaped_dimensions(ty).expect("operand/result schema verified before shape relationships")
}

fn invalid_result(instruction: &Instruction, index: usize) -> BuildError {
    BuildError::InvalidResultType {
        operation: instruction.operation().name(),
        result: instruction.results()[index].value(),
    }
}

fn invalid_dimensions(instruction: &Instruction) -> BuildError {
    BuildError::InvalidDimensionNumbers {
        operation: instruction.operation().name(),
    }
}

fn check_shape(
    instruction: &Instruction,
    index: usize,
    expected: &[Option<u64>],
) -> Result<(), BuildError> {
    if inferred_shape_matches(expected, instruction.results()[index].ty()) {
        Ok(())
    } else {
        Err(invalid_result(instruction, index))
    }
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operands = instruction.operands();
    if matches!(opcode, StableHloOp::Send | StableHloOp::Recv) {
        verify_transfer_channel(instruction, opcode)?;
    }
    match opcode {
        StableHloOp::Add => verify_add(instruction, available)?,
        StableHloOp::AsyncDone => {
            let Type::Future(types) = &available[&operands[0]] else {
                unreachable!("verified future operand");
            };
            if types.len() != 1
                || !compatible_tensor_types(types.first(), Some(instruction.results()[0].ty()))
            {
                return Err(invalid_result(instruction, 0));
            }
        }
        StableHloOp::Dot => {
            let result = instruction.results()[0].ty();
            let expected =
                crate::infer_dot_type(&available[&operands[0]], &available[&operands[1]], result)
                    .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
            if crate::infer_most_specific_type(&[expected, result.clone()]).is_err() {
                return Err(invalid_result(instruction, 0));
            }
        }
        StableHloOp::Broadcast => {
            let input = &available[&operands[0]];
            let broadcast_sizes = integer_attribute(instruction, "broadcast_sizes")?;
            let expected =
                crate::infer_broadcast_type(input, &broadcast_sizes).map_err(|error| {
                    BuildError::TypeInference {
                        operation: instruction.operation().name(),
                        error,
                    }
                })?;
            check_shape(instruction, 0, shape(&expected))?;
            if !super::tensor_element_types_match(input, instruction.results()[0].ty()) {
                return Err(invalid_result(instruction, 0));
            }
            for (axis, size) in shape(instruction.results()[0].ty())[broadcast_sizes.len()..]
                .iter()
                .enumerate()
            {
                if let (Some(size), Some(bound)) = (size, crate::inference::bound(input, axis))
                    && *size > bound
                {
                    return Err(invalid_result(instruction, 0));
                }
            }
        }
        StableHloOp::Cholesky => {
            let input = &available[&operands[0]];
            let inferred =
                crate::infer_cholesky_type(input).map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
            let merged =
                crate::infer_most_specific_type(&[inferred, instruction.results()[0].ty().clone()])
                    .map_err(|_| invalid_result(instruction, 0))?;
            crate::infer_cholesky_type(&merged).map_err(|_| invalid_result(instruction, 0))?;
        }
        StableHloOp::TriangularSolve => verify_triangular_solve(instruction, available)?,
        StableHloOp::BatchNormGrad
        | StableHloOp::BatchNormInference
        | StableHloOp::BatchNormTraining => verify_batch_norm(instruction, opcode, available)?,
        StableHloOp::Fft => verify_fft(instruction, available)?,
        StableHloOp::BitcastConvert => verify_bitcast(instruction, available)?,
        StableHloOp::UniformQuantize | StableHloOp::UniformDequantize => {
            let input = &available[&operands[0]];
            let output = instruction.results()[0].ty();
            let inferred = if opcode == StableHloOp::UniformQuantize {
                crate::infer_uniform_quantize_type(input, output)
            } else {
                crate::infer_uniform_dequantize_type(input)
            }
            .map_err(|error| BuildError::TypeInference {
                operation: instruction.operation().name(),
                error,
            })?;
            if !compatible_shapes(input, output)
                || !super::tensor_element_types_match(&inferred, output)
            {
                return Err(invalid_result(instruction, 0));
            }
        }
        StableHloOp::ReducePrecision
            if !compatible_tensor_types(
                available.get(&operands[0]),
                Some(instruction.results()[0].ty()),
            ) =>
        {
            return Err(invalid_result(instruction, 0));
        }
        StableHloOp::ReplicaId | StableHloOp::PartitionId
            if !shape(instruction.results()[0].ty()).is_empty() =>
        {
            return Err(invalid_result(instruction, 0));
        }
        StableHloOp::RngBitGenerator => {
            let algorithm = match instruction
                .operation()
                .attributes()
                .and_then(|attrs| attrs.get("rng_algorithm"))
            {
                Some(Attribute::Enum { value, .. }) if value == "THREE_FRY" => {
                    crate::RngAlgorithm::ThreeFry
                }
                Some(Attribute::Enum { value, .. }) if value == "PHILOX" => {
                    crate::RngAlgorithm::Philox
                }
                _ => crate::RngAlgorithm::Default,
            };
            let state = &available[&operands[0]];
            crate::infer_rng_bit_generator_types(state, instruction.results()[1].ty(), algorithm)
                .map_err(|error| BuildError::TypeInference {
                operation: instruction.operation().name(),
                error,
            })?;
            let merged = crate::infer_most_specific_type(&[
                state.clone(),
                instruction.results()[0].ty().clone(),
            ])
            .map_err(|_| invalid_result(instruction, 0))?;
            crate::inference::validate_rng_state(&merged, algorithm)
                .map_err(|_| invalid_result(instruction, 0))?;
        }
        StableHloOp::Rng => {
            let lhs = &available[&operands[0]];
            let rhs = &available[&operands[1]];
            let output = instruction.results()[0].ty();
            let attributes = instruction
                .operation()
                .attributes()
                .expect("StableHLO attributes");
            let distribution = match attributes.get("rng_distribution") {
                Some(Attribute::Enum { value, .. }) if value == "NORMAL" => {
                    crate::RngDistribution::Normal
                }
                _ => crate::RngDistribution::Uniform,
            };
            let inferred =
                crate::infer_rng_type(lhs, rhs, &available[&operands[2]], None, distribution)
                    .map_err(|error| BuildError::TypeInference {
                        operation: instruction.operation().name(),
                        error,
                    })?;
            if !inferred_shape_matches(shape(&inferred), output)
                || !super::tensor_element_types_match(&inferred, output)
            {
                return Err(invalid_result(instruction, 0));
            }
        }
        StableHloOp::RealDynamicSlice => {
            let input = &available[&operands[0]];
            let rank = shape(input).len();
            let indices = &available[&operands[1]];
            if operands[1..].iter().any(|operand| {
                shape(&available[operand]) != [Some(rank as u64)] || &available[operand] != indices
            }) {
                return Err(invalid_dimensions(instruction));
            }
            if shape(instruction.results()[0].ty()).len() != rank
                || !super::tensor_element_types_match(input, instruction.results()[0].ty())
            {
                return Err(invalid_result(instruction, 0));
            }
        }
        StableHloOp::Infeed | StableHloOp::Recv => {
            let results = instruction.results();
            let Some(last) = results.last() else {
                return Err(BuildError::InvalidResultArity {
                    operation: opcode.mnemonic(),
                    expected: 1,
                    actual: 0,
                });
            };
            if last.ty() != &Type::Token {
                return Err(invalid_result(instruction, results.len() - 1));
            }
            for (index, result) in results[..results.len() - 1].iter().enumerate() {
                if !matches!(result.ty(), Type::Tensor(_) | Type::QuantizedTensor(_)) {
                    return Err(invalid_result(instruction, index));
                }
            }
            if opcode == StableHloOp::Infeed
                && let Some(Attribute::Array(layouts)) = instruction.operation().attributes().and_then(|attrs| attrs.get("layout"))
                && (layouts.len() != results.len()-1 || layouts.iter().any(|layout|
                    !matches!(layout, Attribute::Array(axes) if axes.iter().all(|axis| matches!(axis, Attribute::Integer { ty: ElementType::I64, .. })))))
            {
                return Err(BuildError::MissingOrInvalidAttribute { operation: opcode.mnemonic(), attribute: "layout" });
            }
        }
        _ => {}
    }
    Ok(())
}

fn verify_transfer_channel(
    instruction: &Instruction,
    opcode: StableHloOp,
) -> Result<(), BuildError> {
    let attributes = instruction
        .operation()
        .attributes()
        .expect("StableHLO attributes");
    let host = matches!(
        attributes.get("is_host_transfer"),
        Some(Attribute::Bool(true))
    );
    let Some(Attribute::StableHlo(attribute)) = attributes.get("channel_handle") else {
        unreachable!("verified channel handle");
    };
    let crate::StableHloAttribute::ChannelHandle(channel) = attribute.as_ref() else {
        unreachable!("verified channel handle");
    };
    let expected = if !host {
        1
    } else if opcode == StableHloOp::Send {
        2
    } else {
        3
    };
    if channel.channel_type != expected {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: opcode.mnemonic(),
            attribute: "channel_handle",
        });
    }
    Ok(())
}

fn verify_add(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let lhs = &available[&instruction.operands()[0]];
    let rhs = &available[&instruction.operands()[1]];
    let result = instruction.results()[0].ty();
    if !compatible_shapes(lhs, rhs) {
        return Err(BuildError::IncompatibleOperands {
            operation: "add",
            lhs: instruction.operands()[0],
            rhs: instruction.operands()[1],
        });
    }
    if !compatible_shapes(lhs, result) || !compatible_shapes(rhs, result) {
        return Err(invalid_result(instruction, 0));
    }
    let compatible = match (lhs, rhs, result) {
        (Type::Tensor(lhs), Type::Tensor(rhs), Type::Tensor(result)) => {
            lhs.element() == rhs.element() && lhs.element() == result.element()
        }
        (Type::QuantizedTensor(lhs), Type::QuantizedTensor(rhs), Type::QuantizedTensor(result)) => {
            let lhs_axis = lhs.quantization_dimension();
            let rhs_axis = rhs.quantization_dimension();
            let result_axis = result.quantization_dimension();
            lhs.storage_type() == rhs.storage_type()
                && lhs.storage_type() == result.storage_type()
                && lhs.expressed_type() == rhs.expressed_type()
                && lhs.expressed_type() == result.expressed_type()
                && if lhs_axis.is_some() || rhs_axis.is_some() {
                    result_axis.is_some()
                        && lhs_axis.is_none_or(|axis| Some(axis) == result_axis)
                        && rhs_axis.is_none_or(|axis| Some(axis) == result_axis)
                } else {
                    result_axis.is_none()
                }
        }
        _ => false,
    };
    if !compatible {
        return Err(invalid_result(instruction, 0));
    }
    Ok(())
}

fn verify_triangular_solve(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let lhs = &available[&instruction.operands()[0]];
    let rhs = &available[&instruction.operands()[1]];
    let attributes = instruction
        .operation()
        .attributes()
        .expect("StableHLO attributes");
    let left = matches!(attributes.get("left_side"), Some(Attribute::Bool(true)));
    if matches!(attributes.get("transpose_a"), Some(Attribute::Enum { value, .. }) if value == "TRANSPOSE_INVALID")
    {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation().name(),
            attribute: "transpose_a",
        });
    }
    let inferred = crate::infer_triangular_solve_type(lhs, rhs, left).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    let merged =
        crate::infer_most_specific_type(&[inferred, instruction.results()[0].ty().clone()])
            .map_err(|_| invalid_result(instruction, 0))?;
    crate::infer_triangular_solve_type(lhs, &merged, left)
        .map_err(|_| invalid_result(instruction, 0))?;
    Ok(())
}

fn verify_batch_norm(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let Some(Attribute::Integer { value: feature, .. }) = instruction
        .operation()
        .attributes()
        .and_then(|attrs| attrs.get("feature_index"))
    else {
        unreachable!("attribute schema verified");
    };
    let feature = usize::try_from(*feature).map_err(|_| invalid_dimensions(instruction))?;
    let inputs = instruction
        .operands()
        .iter()
        .map(|id| &available[id])
        .collect::<Vec<_>>();
    let results = instruction
        .results()
        .iter()
        .map(|result| result.ty())
        .collect::<Vec<_>>();
    crate::inference::validate_batch_norm_types(opcode, &inputs, feature, &results).map_err(
        |error| BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        },
    )
}

fn verify_fft(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let input = &available[&instruction.operands()[0]];
    let lengths = integer_attribute(instruction, "fft_length")?;
    let Some(Attribute::Enum { value: mode, .. }) = instruction
        .operation()
        .attributes()
        .and_then(|attrs| attrs.get("fft_type"))
    else {
        unreachable!("verified fft_type");
    };
    let mode = match mode.as_str() {
        "FFT" => crate::FftType::Fft,
        "IFFT" => crate::FftType::Ifft,
        "RFFT" => crate::FftType::Rfft,
        "IRFFT" => crate::FftType::Irfft,
        _ => unreachable!("verified FFT enum"),
    };
    let expected = crate::infer_fft_type(input, mode, &lengths).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    let result = instruction.results()[0].ty();
    if !super::tensor_element_types_match(&expected, result)
        || crate::infer_most_specific_type(&[expected, result.clone()]).is_err()
    {
        return Err(invalid_result(instruction, 0));
    }
    Ok(())
}

fn verify_bitcast(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let input = &available[&instruction.operands()[0]];
    let output = instruction.results()[0].ty();
    crate::inference::validate_bitcast_types(input, output)
        .map_err(|_| invalid_result(instruction, 0))
}
