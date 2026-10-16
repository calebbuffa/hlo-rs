use super::{LayoutComparison, same_shapes};
use crate::proto::hlo::HloInstructionProto;
use crate::{Error, Module, Result, Shape};

pub(super) fn verify(module: &Module) -> Result<()> {
    let mut host_channels: std::collections::BTreeMap<
        i64,
        Vec<(&HloInstructionProto, Vec<Shape>)>,
    > = Default::default();
    for computation in module.computations() {
        let current = module.computation(computation)?;
        let async_wrapped = module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .any(|i| {
                matches!(
                    i.opcode.as_str(),
                    "async-start" | "async-update" | "async-done"
                ) && i.called_computation_ids.contains(&current.id)
            });
        for id in module.instructions(computation)? {
            let instruction = module.instruction(id)?;
            let channel = module.channel_id(id)?;
            match instruction.opcode.as_str() {
                "send" | "recv" => {
                    if let Some(channel) = channel.filter(|_| instruction.is_host_transfer) {
                        let shapes = module
                            .operands(id)?
                            .iter()
                            .map(|&id| module.shape(id))
                            .collect::<Result<Vec<_>>>()?;
                        host_channels
                            .entry(channel)
                            .or_default()
                            .push((instruction, shapes));
                    }
                    if instruction.id == current.root_id && async_wrapped {
                        continue;
                    }
                    let users = module.users(id)?;
                    let done = if instruction.opcode == "send" {
                        "send-done"
                    } else {
                        "recv-done"
                    };
                    if channel.is_some() {
                        if users.len() != 1 {
                            return Err(Error::new("channel","channel-bearing send/recv requires exactly one user, including control_dep").context(&instruction.name));
                        }
                        let user = module.instruction(users[0])?;
                        if user.opcode == done
                            && (channel != module.channel_id(users[0])?
                                || instruction.is_host_transfer != user.is_host_transfer)
                        {
                            return Err(Error::new(
                                "channel",
                                "start/done channel or host-transfer flag differs",
                            )
                            .context(&instruction.name));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    for transfers in host_channels.values() {
        let (reference, reference_operands) = &transfers[0];
        for (instruction, operands) in &transfers[1..] {
            let attributes = |i: &HloInstructionProto| {
                i.frontend_attributes
                    .as_ref()
                    .map(|a| a.map.clone())
                    .unwrap_or_default()
            };
            let accuracy = |i: &HloInstructionProto| {
                let accuracy = i.result_accuracy.get_or_default();
                let tolerance = accuracy.tolerance();
                let mode = match accuracy.specs {
                    Some(crate::proto::xla_data::result_accuracy::Specs::Mode(mode)) => {
                        mode.value()
                    }
                    _ => 0,
                };
                (
                    accuracy.has_tolerance(),
                    tolerance.atol,
                    tolerance.rtol,
                    tolerance.ulps,
                    mode,
                )
            };
            let equal = reference.opcode == instruction.opcode
                && reference.backend_config == instruction.backend_config
                && accuracy(reference) == accuracy(instruction)
                && attributes(reference) == attributes(instruction)
                && same_shapes(
                    &Shape::from_proto(
                        reference
                            .shape
                            .as_ref()
                            .ok_or_else(|| Error::new("shape", "missing transfer shape"))?
                            .clone(),
                    )?,
                    &Shape::from_proto(
                        instruction
                            .shape
                            .as_ref()
                            .ok_or_else(|| Error::new("shape", "missing transfer shape"))?
                            .clone(),
                    )?,
                    LayoutComparison::Exact,
                )
                && reference_operands.len() == operands.len()
                && reference_operands
                    .iter()
                    .zip(operands)
                    .all(|(a, b)| same_shapes(a, b, LayoutComparison::Exact));
            if !equal {
                return Err(Error::new("channel","host transfers sharing a channel must be identical apart from operand identities").context(&instruction.name));
            }
        }
    }
    Ok(())
}
