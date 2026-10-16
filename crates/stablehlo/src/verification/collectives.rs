//! Collective process groups, dynamic roots and tensor geometry.

use std::collections::{HashMap, HashSet};

use super::{BuildError, regions, shaped_dimensions};
use crate::{
    Attribute, Instruction, MeshReference, ScalarValue, StableHloAttribute, StableHloOp, Type,
    ValueId,
};

fn invalid(instruction: &Instruction, attribute: &'static str) -> BuildError {
    BuildError::MissingOrInvalidAttribute {
        operation: instruction.operation().name(),
        attribute,
    }
}

fn integer(instruction: &Instruction, name: &'static str) -> u64 {
    match instruction
        .operation()
        .attributes()
        .and_then(|attributes| attributes.get(name))
    {
        Some(Attribute::Integer { value, .. }) => *value as u64,
        _ => unreachable!("verified integer attribute"),
    }
}

fn verify_groups(
    instruction: &Instruction,
    equal_size: bool,
    complete: bool,
) -> Result<Option<u64>, BuildError> {
    let attributes = instruction
        .operation()
        .attributes()
        .expect("StableHLO attributes");
    let global = attributes.contains_key("use_global_device_ids");
    if global {
        let channel = match attributes.get("channel_handle") {
            Some(Attribute::StableHlo(attribute)) => match attribute.as_ref() {
                StableHloAttribute::ChannelHandle(channel) => channel.handle,
                _ => unreachable!("verified channel handle"),
            },
            _ => 0,
        };
        if channel <= 0 {
            return Err(invalid(instruction, "channel_handle"));
        }
    }
    let Some(Attribute::DenseElements { ty, values }) = attributes.get("replica_groups") else {
        if let Some(Attribute::StableHlo(attribute)) = attributes.get("replica_groups")
            && let StableHloAttribute::ReplicaGroupMeshAxes(groups) = attribute.as_ref()
            && let MeshReference::Inline(mesh) = &groups.mesh
        {
            if complete && mesh.device_ids.as_ref().is_some_and(|ids| !ids.is_empty()) {
                let ids = mesh.device_ids.as_ref().expect("explicit device IDs");
                let seen = ids.iter().copied().collect::<HashSet<_>>();
                if (0..ids.len()).any(|id| !seen.contains(&(id as i64))) {
                    return Err(invalid(instruction, "replica_groups"));
                }
            }
            return mesh
                .replica_group_size(&groups.axes)
                .map(Some)
                .map_err(|error| BuildError::InvalidAttribute {
                    name: "replica_groups".into(),
                    error: crate::AttributeValidationError::StableHlo(error),
                });
        }
        // Symbolic mesh cardinalities require module-level resolution.
        return Ok(None);
    };
    let Some([Some(rows), Some(columns)]) = shaped_dimensions(ty) else {
        return Err(invalid(instruction, "replica_groups"));
    };
    if global && values.is_empty() {
        return Err(invalid(instruction, "replica_groups"));
    }
    let mut seen = HashSet::new();
    for value in values {
        let ScalarValue::Integer(value) = value else {
            unreachable!("verified dense integer attribute");
        };
        if *value == -1 && !equal_size {
            continue;
        }
        if *value < 0 || !seen.insert(*value) {
            return Err(invalid(instruction, "replica_groups"));
        }
    }
    if complete && (0..seen.len()).any(|index| !seen.contains(&(index as i128))) {
        return Err(invalid(instruction, "replica_groups"));
    }
    if *rows == 0 {
        Ok(None)
    } else {
        Ok(Some(*columns))
    }
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    match opcode {
        StableHloOp::AllGather | StableHloOp::ReduceScatter => {
            let group_size = verify_groups(instruction, true, true)?;
            let attribute = if opcode == StableHloOp::AllGather {
                "all_gather_dim"
            } else {
                "scatter_dimension"
            };
            let axis = i64::try_from(integer(instruction, attribute))
                .map_err(|_| invalid(instruction, attribute))?;
            for (index, &operand) in instruction.operands().iter().enumerate() {
                let input = &available[&operand];
                let result = &instruction.results()[index];
                let output = shaped_dimensions(result.ty()).expect("verified tensor result");
                let (smaller, larger) = if opcode == StableHloOp::AllGather {
                    (input, result.ty())
                } else {
                    (result.ty(), input)
                };
                crate::inference::validate_exchange_shape(smaller, larger, axis, group_size)
                    .map_err(|error| BuildError::TypeInference {
                        operation: instruction.operation().name(),
                        error,
                    })?;
                if opcode == StableHloOp::AllGather {
                    regions::check_result(instruction, index, output, input)?;
                }
            }
        }
        StableHloOp::AllToAll => {
            let group_size = verify_groups(instruction, true, true)?;
            let split = integer(instruction, "split_dimension") as usize;
            let concat = integer(instruction, "concat_dimension") as usize;
            let count = integer(instruction, "split_count");
            if group_size != Some(count) {
                let mesh = matches!(
                    instruction
                        .operation()
                        .attributes()
                        .and_then(|attributes| attributes.get("replica_groups")),
                    Some(Attribute::StableHlo(_))
                );
                if group_size.is_some() || !mesh {
                    return Err(invalid(instruction, "replica_groups"));
                }
            }
            let inputs = instruction
                .operands()
                .iter()
                .map(|operand| available[operand].clone())
                .collect::<Vec<_>>();
            let inferred =
                crate::infer_all_to_all_types(&inputs, split as i64, concat as i64, count as i64)
                    .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
            for (index, expected) in inferred.into_iter().enumerate() {
                regions::check_result(
                    instruction,
                    index,
                    shaped_dimensions(&expected).expect("inferred tensor"),
                    &expected,
                )?;
                let result = &instruction.results()[index];
                let output = shaped_dimensions(result.ty()).expect("verified tensor result");
                if output[concat].is_some_and(|size| !size.is_multiple_of(count))
                    || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
                {
                    return Err(BuildError::InvalidResultType {
                        operation: instruction.operation().name(),
                        result: result.value(),
                    });
                }
            }
        }
        StableHloOp::AllReduce | StableHloOp::CollectiveReduce => {
            verify_groups(instruction, false, true)?;
            if opcode == StableHloOp::CollectiveReduce {
                verify_root(instruction, available)?;
            }
        }
        StableHloOp::CollectiveBroadcast => {
            verify_groups(instruction, true, false)?;
            let inputs = instruction
                .operands()
                .iter()
                .map(|id| available[id].clone())
                .collect::<Vec<_>>();
            let has_dynamic_root = instruction
                .operation()
                .attributes()
                .is_some_and(|attrs| attrs.contains_key("has_dynamic_root"));
            let inferred = crate::infer_collective_broadcast_types(&inputs, has_dynamic_root)
                .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
            for (index, input) in inferred.iter().enumerate() {
                regions::check_result(
                    instruction,
                    index,
                    shaped_dimensions(input).expect("verified tensor operand"),
                    input,
                )?;
                crate::infer_most_specific_type(&[
                    input.clone(),
                    instruction.results()[index].ty().clone(),
                ])
                .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
            }
        }
        StableHloOp::CollectivePermute => {
            let Some(Attribute::DenseElements { ty, values }) = instruction
                .operation()
                .attributes()
                .and_then(|attributes| attributes.get("source_target_pairs"))
            else {
                unreachable!("verified source-target pairs");
            };
            let Some([Some(_), Some(2)]) = shaped_dimensions(ty) else {
                return Err(invalid(instruction, "source_target_pairs"));
            };
            let mut seen = [HashSet::new(), HashSet::new()];
            for (index, value) in values.iter().enumerate() {
                let ScalarValue::Integer(value) = value else {
                    unreachable!("verified integer pair");
                };
                if *value < 0 || !seen[index % 2].insert(*value) {
                    return Err(invalid(instruction, "source_target_pairs"));
                }
            }
            let input = &available[&instruction.operands()[0]];
            let inferred = crate::infer_collective_permute_type(input).map_err(|error| {
                BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                }
            })?;
            regions::check_result(
                instruction,
                0,
                shaped_dimensions(&inferred).expect("inferred tensor"),
                &inferred,
            )?;
            crate::infer_most_specific_type(&[inferred, instruction.results()[0].ty().clone()])
                .map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation().name(),
                    error,
                })?;
        }
        _ => {}
    }
    Ok(())
}

fn verify_root(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if instruction
        .operation()
        .attributes()
        .is_some_and(|attributes| attributes.contains_key("has_dynamic_root"))
    {
        let value = instruction.operands().last().expect("verified cardinality");
        crate::inference::validate_dynamic_root(&available[value], instruction.results().len())
            .map_err(|_| BuildError::InvalidOperandType {
                operation: instruction.operation().name(),
                value: *value,
            })?;
    }
    Ok(())
}
