use super::*;
use crate::proto::xla_data::{CollectiveDeviceListProto, IotaReplicaGroupListProto, ReplicaGroup};

/// Compact or explicit collective device groups carried by HLO instructions.
pub use crate::proto::hlo::hlo_instruction_proto::Replica_group_list as CollectiveDeviceList;

pub fn parse_collective_device_list(source: &str) -> Result<CollectiveDeviceList> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let mut instruction = HloInstructionProto::new();
    parse_groups(&mut parser, &mut instruction)?;
    parser.finish()?;
    instruction
        .replica_group_list
        .ok_or_else(|| Error::new("collective-group", "missing device list"))
}

pub fn print_collective_device_list(value: &CollectiveDeviceList) -> Result<String> {
    let instruction = HloInstructionProto {
        replica_group_list: Some(value.clone()),
        ..Default::default()
    };
    let mut result = String::new();
    print_groups(&instruction, &mut result)?;
    Ok(result)
}

/// Parse explicit groups, materializing compact iotas within the text resource limit.
pub fn parse_replica_groups(source: &str) -> Result<Vec<ReplicaGroup>> {
    match parse_collective_device_list(source)? {
        CollectiveDeviceList::CollectiveDeviceList(list) => Ok(list.replica_groups),
        CollectiveDeviceList::IotaCollectiveDeviceList(iota) => {
            let count = iota
                .num_replica_groups
                .checked_mul(iota.num_devices_per_group)
                .ok_or_else(|| Error::new("overflow", "replica group device count overflow"))?;
            if count > ParseOptions::default().max_literal_elements as i64 {
                return Err(Error::new(
                    "resource-limit",
                    "replica groups exceed materialization limit",
                ));
            }
            Ok((0..iota.num_replica_groups)
                .map(|group| ReplicaGroup {
                    replica_ids: (0..iota.num_devices_per_group)
                        .map(|index| {
                            crate::sharding::iota_value(
                                &iota.iota_reshape_dims,
                                &iota.iota_transpose_perm,
                                group * iota.num_devices_per_group + index,
                            )
                        })
                        .collect(),
                    ..Default::default()
                })
                .collect())
        }
        CollectiveDeviceList::MeshAxesReplicaGroupList(wire) => {
            crate::MeshAxesReplicaGroupList::from_proto(wire)?
                .replica_groups(ParseOptions::default().max_literal_elements)
        }
    }
}

pub fn print_replica_groups(groups: &[ReplicaGroup]) -> Result<String> {
    print_collective_device_list(&CollectiveDeviceList::CollectiveDeviceList(
        CollectiveDeviceListProto {
            replica_groups: groups.to_vec(),
            ..Default::default()
        },
    ))
}

pub(crate) fn source_target_pairs(source: &str) -> Result<Vec<(i64, i64)>> {
    parse_replica_groups(source)?
        .iter()
        .map(|group| match group.replica_ids.as_slice() {
            &[source, target] => Ok((source, target)),
            _ => Err(Error::new(
                "deadlock",
                "source-target group must have two members",
            )),
        })
        .collect()
}

pub(super) fn supported(opcode: Opcode) -> bool {
    matches!(
        attribute_opcode(opcode),
        Opcode::AllGather
            | Opcode::AllReduce
            | Opcode::ReduceScatter
            | Opcode::AllToAll
            | Opcode::RaggedAllToAll
            | Opcode::CollectiveBroadcast
            | Opcode::CollectiveReduce
            | Opcode::CollectivePermute
    )
}

pub(super) fn attribute_opcode(opcode: Opcode) -> Opcode {
    match opcode {
        Opcode::AllGatherStart => Opcode::AllGather,
        Opcode::AllReduceStart => Opcode::AllReduce,
        Opcode::CollectivePermuteStart => Opcode::CollectivePermute,
        _ => opcode,
    }
}

pub(super) fn has_groups(opcode: Opcode) -> bool {
    supported(opcode) && attribute_opcode(opcode) != Opcode::CollectivePermute
}

pub(super) fn parse_groups(
    parser: &mut Parser,
    instruction: &mut HloInstructionProto,
) -> Result<()> {
    if matches!(parser.peek(), "mesh" | "single_device" | "maximal_mesh") {
        let mesh = parser.mesh()?;
        let axes = parser.axes(&mesh, false)?.0;
        let wire = crate::proto::xla_data::MeshAxesReplicaGroupListProto {
            mesh: Some(mesh).into(),
            axes,
            ..Default::default()
        };
        instruction.set_mesh_axes_replica_group_list(
            crate::MeshAxesReplicaGroupList::from_proto(wire)?.into_proto(),
        );
        return Ok(());
    }
    if parser.peek() == "{" {
        let mut list = CollectiveDeviceListProto::new();
        list.replica_groups = parser
            .integer_lists()?
            .into_iter()
            .map(|replica_ids| ReplicaGroup {
                replica_ids,
                ..Default::default()
            })
            .collect();
        instruction.set_collective_device_list(list);
        return Ok(());
    }
    let dimensions = parser.integers("[", "]")?;
    let [count, size] = dimensions.as_slice() else {
        return Err(parser.error("iota collective device list requires two dimensions"));
    };
    parser.expect("<=")?;
    let mut iota = IotaReplicaGroupListProto::new();
    iota.num_replica_groups = *count;
    iota.num_devices_per_group = *size;
    iota.iota_reshape_dims = parser.integers("[", "]")?;
    iota.iota_transpose_perm = if iota.iota_reshape_dims.len() == 1 {
        vec![0]
    } else {
        parser.expect("T")?;
        parser
            .integers("(", ")")?
            .into_iter()
            .map(|axis| {
                i32::try_from(axis).map_err(|_| parser.error("iota transpose axis exceeds i32"))
            })
            .collect::<Result<_>>()?
    };
    instruction.set_iota_collective_device_list(iota);
    crate::verification::collective::groups(instruction, false, true)?;
    Ok(())
}

pub(super) fn print_fields(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
    channel_id: Option<i64>,
    result: &mut String,
) -> Result<()> {
    let opcode = attribute_opcode(instruction.opcode.parse()?);
    if !supported(opcode) {
        return Ok(());
    }
    if opcode == Opcode::CollectivePermute {
        result.push_str(", source_target_pairs={");
        for (index, pair) in instruction.source_target_pairs.iter().enumerate() {
            reject_fields(pair, &["source", "target"])?;
            if index != 0 {
                result.push(',');
            }
            write!(result, "{{{},{}}}", pair.source, pair.target)
                .map_err(|e| Error::new("print", e.to_string()))?;
        }
        result.push('}');
        if !instruction.dynamic_slice_sizes.is_empty() {
            let operands = instruction
                .operand_ids
                .iter()
                .map(|id| {
                    let shape = instructions
                        .iter()
                        .find(|i| i.id == *id)
                        .and_then(|i| i.shape.as_ref())
                        .ok_or_else(|| Error::new("reference", "missing permute operand shape"))?;
                    Shape::from_proto(shape.clone())
                })
                .collect::<Result<Vec<_>>>()?;
            let widths = crate::verification::collective::slice_widths(&operands)?;
            let total = widths.iter().try_fold(0usize, |sum, &width| {
                sum.checked_add(width)
                    .ok_or_else(|| Error::new("overflow", "slice width sum overflow"))
            })?;
            if total != instruction.dynamic_slice_sizes.len() {
                return Err(Error::new(
                    "unsupported",
                    "permute flattened slice sizes do not match the pinned import partition",
                ));
            }
            result.push_str(", slice_sizes={");
            let mut offset = 0;
            for (index, width) in widths.into_iter().enumerate() {
                if index != 0 {
                    result.push(',');
                }
                write!(
                    result,
                    "{{{}}}",
                    instruction.dynamic_slice_sizes[offset..offset + width]
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
                .map_err(|e| Error::new("print", e.to_string()))?;
                offset += width;
            }
            result.push('}');
        }
        if let Some(channel_id) = channel_id {
            write!(result, ", channel_id={channel_id}")
                .map_err(|e| Error::new("print", e.to_string()))?;
        }
        return Ok(());
    }
    result.push_str(", replica_groups=");
    print_groups(instruction, result)?;
    if let Some(channel_id) = channel_id {
        write!(result, ", channel_id={channel_id}")
            .map_err(|e| Error::new("print", e.to_string()))?;
    }
    if matches!(
        opcode,
        Opcode::AllGather
            | Opcode::AllReduce
            | Opcode::ReduceScatter
            | Opcode::AllToAll
            | Opcode::CollectiveReduce
    ) {
        write!(
            result,
            ", constrain_layout={}",
            instruction.constrain_layout
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
    }
    if matches!(
        opcode,
        Opcode::AllGather | Opcode::AllReduce | Opcode::ReduceScatter | Opcode::CollectiveReduce
    ) {
        write!(
            result,
            ", use_global_device_ids={}",
            instruction.use_global_device_ids
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
    }
    if matches!(
        opcode,
        Opcode::CollectiveReduce | Opcode::CollectiveBroadcast
    ) {
        write!(
            result,
            ", has_dynamic_root={}",
            instruction.has_dynamic_root
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
    }
    Ok(())
}

fn print_groups(instruction: &HloInstructionProto, result: &mut String) -> Result<()> {
    if instruction.has_mesh_axes_replica_group_list() {
        if !instruction.replica_groups.is_empty() {
            return Err(Error::new(
                "unsupported",
                "legacy groups shadowed by mesh groups cannot round-trip",
            ));
        }
        let wire = instruction.mesh_axes_replica_group_list();
        reject_fields(wire, &["mesh", "axes"])?;
        crate::MeshAxesReplicaGroupList::from_proto(wire.clone())?;
        let mesh = wire.mesh.as_ref().ok_or_else(|| {
            Error::new("unsupported", "mesh group text requires mesh wire presence")
        })?;
        result.push_str(&sharding::print_mesh(mesh)?);
        result.push(' ');
        result.push_str(&sharding::print_axes(&wire.axes, mesh, true)?);
        return Ok(());
    }
    let list = |values: &[i64]| {
        values
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    };
    if instruction.has_iota_collective_device_list() {
        if !instruction.replica_groups.is_empty() {
            return Err(Error::new(
                "unsupported",
                "legacy replica groups shadowed by iota groups cannot round-trip",
            ));
        }
        let iota = instruction.iota_collective_device_list();
        reject_fields(
            iota,
            &[
                "num_replica_groups",
                "num_devices_per_group",
                "iota_reshape_dims",
                "iota_transpose_perm",
            ],
        )?;
        crate::verification::collective::groups(instruction, false, true)?;
        write!(
            result,
            "[{},{}]<=[{}]",
            iota.num_replica_groups,
            iota.num_devices_per_group,
            list(&iota.iota_reshape_dims),
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
        if iota.iota_reshape_dims.len() > 1 {
            write!(
                result,
                "T({})",
                iota.iota_transpose_perm
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(|e| Error::new("print", e.to_string()))?;
        }
    } else {
        let groups = if !instruction.replica_groups.is_empty() {
            if instruction.has_collective_device_list() {
                return Err(Error::new(
                    "unsupported",
                    "legacy and modern replica groups cannot both round-trip",
                ));
            }
            &instruction.replica_groups
        } else {
            reject_fields(instruction.collective_device_list(), &["replica_groups"])?;
            &instruction.collective_device_list().replica_groups
        };
        result.push('{');
        for (index, group) in groups.iter().enumerate() {
            reject_fields(group, &["replica_ids"])?;
            if index != 0 {
                result.push(',');
            }
            write!(result, "{{{}}}", list(&group.replica_ids))
                .map_err(|e| Error::new("print", e.to_string()))?;
        }
        result.push('}');
    }
    Ok(())
}
