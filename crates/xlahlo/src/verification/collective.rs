use super::{LayoutComparison, same_shapes};
use crate::proto::hlo::HloInstructionProto;
use crate::proto::xla_data::ReplicaGroup;
use crate::{Error, Module, Opcode, Result, Shape, inference};

/// Module configuration not represented by HloModuleProto. Upstream defaults
/// both counts to one and then deliberately skips some participant-count checks.
#[derive(Clone, Copy, Debug)]
pub struct CollectiveContext {
    pub replica_count: i64,
    pub partition_count: i64,
}

impl Default for CollectiveContext {
    fn default() -> Self {
        Self {
            replica_count: 1,
            partition_count: 1,
        }
    }
}

#[derive(Clone, Copy)]
enum GroupMode {
    Replica,
    Partition,
    ReplicaAndPartition,
    Flattened,
}

fn group_mode(channel: bool, global: Option<bool>) -> Result<GroupMode> {
    Ok(match (channel, global) {
        (false, Some(true)) => {
            return Err(Error::new(
                "collective-group",
                "global device IDs require a channel",
            ));
        }
        (false, _) => GroupMode::Replica,
        (true, None) => GroupMode::Partition,
        (true, Some(false)) => GroupMode::ReplicaAndPartition,
        (true, Some(true)) => GroupMode::Flattened,
    })
}

fn participant_count(context: CollectiveContext, mode: GroupMode) -> Result<i64> {
    if context.replica_count <= 0 || context.partition_count <= 0 {
        return Err(Error::new(
            "collective-group",
            "replica and partition counts must be positive",
        ));
    }
    Ok(match mode {
        GroupMode::Replica | GroupMode::ReplicaAndPartition => context.replica_count,
        GroupMode::Partition => context.partition_count,
        GroupMode::Flattened => context
            .replica_count
            .checked_mul(context.partition_count)
            .ok_or_else(|| Error::new("overflow", "replica/partition count product overflow"))?,
    })
}

pub(crate) fn groups(
    instruction: &HloInstructionProto,
    uniform: bool,
    check_replica_groups: bool,
) -> Result<Option<(i64, i64)>> {
    if instruction.has_iota_collective_device_list() {
        let iota = instruction.iota_collective_device_list();
        if iota.num_replica_groups <= 0
            || iota.num_devices_per_group <= 0
            || iota.iota_reshape_dims.is_empty()
            || iota.iota_reshape_dims.len() != iota.iota_transpose_perm.len()
        {
            return Err(Error::new(
                "collective-group",
                "invalid iota group dimensions",
            ));
        }
        let total = iota
            .num_replica_groups
            .checked_mul(iota.num_devices_per_group)
            .ok_or_else(|| Error::new("overflow", "iota device count overflow"))?;
        let mut product = 1i64;
        for &dim in &iota.iota_reshape_dims {
            if dim <= 0 {
                return Err(Error::new(
                    "collective-group",
                    "iota reshape dimensions must be positive",
                ));
            }
            product = product
                .checked_mul(dim)
                .ok_or_else(|| Error::new("overflow", "iota reshape product overflow"))?;
        }
        let permutation = iota
            .iota_transpose_perm
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if product != total
            || permutation.len() != iota.iota_reshape_dims.len()
            || permutation
                .iter()
                .enumerate()
                .any(|(index, &axis)| usize::try_from(axis) != Ok(index))
        {
            return Err(Error::new(
                "collective-group",
                "iota reshape product or transpose permutation is invalid",
            ));
        }
        return Ok(Some((total, iota.num_devices_per_group)));
    }
    if instruction.has_mesh_axes_replica_group_list() {
        let wire = instruction.mesh_axes_replica_group_list();
        let groups = crate::MeshAxesReplicaGroupList::from_proto(wire.clone())?;
        let mesh = wire.mesh.get_or_default();
        if mesh.axes.is_empty() && mesh.device_ids.first().is_some_and(|&id| id != 0) {
            return Err(Error::new(
                "collective-group",
                "single-device collective mesh must use device zero",
            ));
        }
        let total = groups
            .num_replica_groups()
            .checked_mul(groups.num_devices_per_group())
            .ok_or_else(|| Error::new("overflow", "mesh group device count overflow"))?;
        return Ok((total != 0).then_some((total, groups.num_devices_per_group())));
    }
    let groups = if !instruction.replica_groups.is_empty() {
        &instruction.replica_groups
    } else {
        &instruction.collective_device_list().replica_groups
    };
    validate_groups(groups, uniform, check_replica_groups)
}

fn validate_groups(
    groups: &[ReplicaGroup],
    uniform: bool,
    check_replica_groups: bool,
) -> Result<Option<(i64, i64)>> {
    if groups.is_empty() {
        return Ok(None);
    }
    let size = groups[0].replica_ids.len();
    let mut total = 0usize;
    for group in groups {
        if check_replica_groups
            && (group.replica_ids.is_empty() || uniform && group.replica_ids.len() != size)
        {
            return Err(Error::new(
                "collective-group",
                "replica groups must be nonempty and, for this operation, uniform",
            ));
        }
        total = total
            .checked_add(group.replica_ids.len())
            .ok_or_else(|| Error::new("overflow", "replica group count overflow"))?;
    }
    if !check_replica_groups {
        return Ok(Some((
            i64::try_from(total).map_err(|_| Error::new("overflow", "too many replica IDs"))?,
            i64::try_from(size).map_err(|_| Error::new("overflow", "replica group too large"))?,
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    for id in groups.iter().flat_map(|group| &group.replica_ids) {
        if !usize::try_from(*id).is_ok_and(|id| id < total) || !seen.insert(*id) {
            return Err(Error::new(
                "collective-group",
                "replica groups must contain every ID from zero to total-1 exactly once",
            ));
        }
    }
    Ok(Some((
        i64::try_from(total).map_err(|_| Error::new("overflow", "too many replica IDs"))?,
        i64::try_from(size).map_err(|_| Error::new("overflow", "replica group too large"))?,
    )))
}

fn checked_groups(
    instruction: &HloInstructionProto,
    context: CollectiveContext,
    global: Option<bool>,
    uniform: bool,
    channel_present: bool,
    check_replica_groups: bool,
) -> Result<(GroupMode, Option<(i64, i64)>)> {
    let mode = group_mode(channel_present, global)?;
    let expected = check_replica_groups
        .then(|| participant_count(context, mode))
        .transpose()?;
    let groups = groups(instruction, uniform, check_replica_groups)?;
    if let Some((total, _)) = groups {
        if expected.is_some_and(|expected| expected != 1 && total != expected) {
            return Err(Error::new(
                "collective-group",
                "replica-group total differs from module configuration",
            ));
        }
    } else if matches!(mode, GroupMode::Flattened) {
        return Err(Error::new(
            "collective-group",
            "flattened device IDs require explicit groups",
        ));
    }
    Ok((mode, groups))
}

fn subgroup(
    instruction: &HloInstructionProto,
    context: CollectiveContext,
    global: Option<bool>,
    channel_present: bool,
    check_replica_groups: bool,
) -> Result<i64> {
    let (mode, groups) = checked_groups(
        instruction,
        context,
        global,
        true,
        channel_present,
        check_replica_groups,
    )?;
    let count = groups.map(|(_, size)| size).unwrap_or(match mode {
        GroupMode::Partition => context.partition_count,
        _ => context.replica_count,
    });
    if matches!(mode, GroupMode::ReplicaAndPartition) {
        count
            .checked_mul(context.partition_count)
            .ok_or_else(|| Error::new("overflow", "collective subgroup size overflow"))
    } else {
        Ok(count)
    }
}

fn shard_count(
    instruction: &HloInstructionProto,
    declared: &Shape,
    operands: &[Shape],
    gather: bool,
    subgroup: i64,
) -> Result<(i64, i64)> {
    let [axis] = instruction.dimensions.as_slice() else {
        return Err(Error::new(
            "collective",
            "collective requires exactly one dimension",
        ));
    };
    let axis_index = usize::try_from(*axis)
        .map_err(|_| Error::new("collective", "negative collective dimension"))?;
    if operands.is_empty() {
        return Err(Error::new("collective", "collective requires operands"));
    }
    if operands.len() > 1
        && (!declared.is_tuple() || declared.as_proto().tuple_shapes.len() != operands.len())
    {
        return Err(Error::new(
            "collective",
            "variadic collective result requires matching tuple cardinality",
        ));
    }
    let mut counts = None;
    for (index, operand) in operands.iter().enumerate() {
        let result = if operands.len() == 1 {
            declared.clone()
        } else {
            declared.subshape(&[index])?
        };
        if axis_index >= operand.rank()? || axis_index >= result.rank()? {
            return Err(Error::new(
                "collective",
                "collective dimension out of bounds",
            ));
        }
        if index == 0 {
            let (numerator, denominator) = if gather {
                (
                    result.dimensions()[axis_index],
                    operand.dimensions()[axis_index],
                )
            } else {
                (
                    operand.dimensions()[axis_index],
                    result.dimensions()[axis_index],
                )
            };
            if denominator <= 0 || numerator < 0 {
                return Err(Error::new(
                    "collective",
                    "cannot infer collective shard count from zero or unbounded extent",
                ));
            }
            let count = numerator / denominator + i64::from(numerator % denominator != 0);
            if subgroup != 1 && count != subgroup {
                return Err(Error::new(
                    "collective-group",
                    "shape-derived shard count differs from subgroup size",
                ));
            }
            counts = Some(count);
        }
    }
    Ok((
        *axis,
        counts.ok_or_else(|| Error::new("collective", "missing shard count"))?,
    ))
}

pub(super) fn infer(
    instruction: &HloInstructionProto,
    opcode: Opcode,
    declared: &Shape,
    operands: &[Shape],
    context: CollectiveContext,
    channel_present: bool,
    check_replica_groups: bool,
) -> Result<Shape> {
    use Opcode::*;
    if matches!(opcode, CollectivePermute | CollectivePermuteStart) {
        let inplace = verify_permute(instruction, operands, context, channel_present)?;
        return if opcode == CollectivePermute {
            inference::infer_collective_permute(operands, inplace)
        } else {
            let contexts = if declared.is_tuple() {
                declared
                    .as_proto()
                    .tuple_shapes
                    .iter()
                    .skip(2)
                    .cloned()
                    .map(Shape::from_proto)
                    .collect::<Result<Vec<_>>>()?
            } else {
                vec![]
            };
            inference::infer_collective_permute_start(operands, &contexts, inplace)
        };
    }
    let global = if matches!(opcode, AllToAll | RaggedAllToAll) {
        None
    } else {
        Some(instruction.use_global_device_ids)
    };
    let subgroup = match opcode {
        CollectiveBroadcast => 1,
        AllReduce | AllReduceStart | CollectiveReduce | RaggedAllToAll => {
            if check_replica_groups {
                checked_groups(
                    instruction,
                    context,
                    global,
                    opcode == RaggedAllToAll,
                    channel_present,
                    true,
                )?;
            }
            1
        }
        _ => subgroup(
            instruction,
            context,
            global,
            channel_present,
            check_replica_groups,
        )?,
    };
    match opcode {
        AllGather => {
            let (axis, shards) = shard_count(instruction, declared, operands, true, subgroup)?;
            inference::infer_all_gather(operands, axis, shards)
        }
        AllGatherStart => {
            if !declared.is_tuple() || declared.as_proto().tuple_shapes.len() != 2 {
                return Err(Error::new(
                    "collective",
                    "all-gather-start requires an input/result tuple",
                ));
            }
            let (axis, shards) = shard_count(
                instruction,
                &declared.subshape(&[1])?,
                operands,
                true,
                subgroup,
            )?;
            inference::infer_all_gather_start(operands, axis, shards)
        }
        ReduceScatter => {
            let (axis, shards) = shard_count(instruction, declared, operands, false, subgroup)?;
            inference::infer_reduce_scatter(operands, axis, shards)
        }
        AllReduce | AllReduceStart => inference::infer_all_reduce(operands),
        AllToAll => {
            if let [axis] = instruction.dimensions.as_slice() {
                let [operand] = operands else {
                    return Err(Error::new(
                        "collective",
                        "array all-to-all requires one operand",
                    ));
                };
                inference::infer_all_to_all(operand, *axis, *axis, subgroup)
            } else if instruction.dimensions.is_empty() {
                if i64::try_from(operands.len()) != Ok(subgroup) {
                    return Err(Error::new(
                        "collective-group",
                        "tuple all-to-all operand count differs from subgroup size",
                    ));
                }
                inference::infer_all_to_all_tuple(operands)
            } else {
                Err(Error::new(
                    "collective",
                    "all-to-all requires zero or one split dimension",
                ))
            }
        }
        RaggedAllToAll => {
            if operands.len() != 6 {
                return Err(Error::new(
                    "collective",
                    "ragged all-to-all requires six operands",
                ));
            }
            for index in 2..5 {
                if !matches!(operands[index].rank()?, 1 | 2)
                    || !same_shapes(
                        &operands[index],
                        &operands[index + 1],
                        LayoutComparison::IgnoreMemory,
                    )
                {
                    return Err(Error::new(
                        "collective",
                        "ragged offsets/sizes must be identical rank-one/two shapes apart from memory space",
                    ));
                }
            }
            inference::infer_ragged_all_to_all(operands)
        }
        CollectiveReduce => {
            inference::infer_collective_reduce(operands, instruction.has_dynamic_root)
        }
        // The pinned verifier does not forward the dynamic-root flag here.
        CollectiveBroadcast => inference::infer_collective_broadcast(operands, false),
        _ => Err(Error::new("unsupported", "collective inference opcode")),
    }
}

fn buffer_offset(buffer: &Shape, offset: &Shape) -> Result<()> {
    if !offset.is_tuple() {
        return Err(Error::new(
            "collective",
            "permute buffer offsets must be tuples",
        ));
    }
    let children = offset
        .as_proto()
        .tuple_shapes
        .iter()
        .cloned()
        .map(Shape::from_proto)
        .collect::<Result<Vec<_>>>()?;
    let all_arrays = children.iter().all(Shape::is_array);
    let all_tuples = children.iter().all(Shape::is_tuple);
    if !all_arrays && !all_tuples {
        return Err(Error::new(
            "collective",
            "permute offset tuple must contain only arrays or only tuples",
        ));
    }
    let rank = buffer.rank()?;
    if if all_tuples {
        children
            .iter()
            .any(|shape| shape.as_proto().tuple_shapes.len() != rank)
    } else {
        children.len() != rank
    } {
        return Err(Error::new(
            "collective",
            "permute offsets must match buffer rank",
        ));
    }
    Ok(())
}

fn verify_permute(
    instruction: &HloInstructionProto,
    operands: &[Shape],
    context: CollectiveContext,
    channel_present: bool,
) -> Result<bool> {
    let inplace = !instruction.dynamic_slice_sizes.is_empty();
    let allowed = if inplace {
        let [input, output, input_offsets, output_offsets] = operands else {
            return Err(Error::new(
                "collective",
                "in-place permute requires four operands",
            ));
        };
        if input.is_array() && output.is_array() {
            buffer_offset(input, input_offsets)?;
            buffer_offset(output, output_offsets)?;
            input_offsets.as_proto().tuple_shapes.len()
        } else if input.is_tuple() && output.is_tuple() {
            let inputs = &input.as_proto().tuple_shapes;
            let outputs = &output.as_proto().tuple_shapes;
            if inputs.len() != outputs.len()
                || !input_offsets.is_tuple()
                || !output_offsets.is_tuple()
                || input_offsets.as_proto().tuple_shapes.len() != inputs.len()
                || output_offsets.as_proto().tuple_shapes.len() != outputs.len()
            {
                return Err(Error::new(
                    "collective",
                    "in-place permute tuple buffers and offsets have different cardinalities",
                ));
            }
            for (buffer, offset) in inputs
                .iter()
                .zip(&input_offsets.as_proto().tuple_shapes)
                .chain(outputs.iter().zip(&output_offsets.as_proto().tuple_shapes))
            {
                buffer_offset(
                    &Shape::from_proto(buffer.clone())?,
                    &Shape::from_proto(offset.clone())?,
                )?;
            }
            input_offsets
                .as_proto()
                .tuple_shapes
                .first()
                .ok_or_else(|| {
                    Error::new("collective", "in-place tuple permute requires a buffer")
                })?
                .tuple_shapes
                .len()
        } else {
            return Err(Error::new(
                "collective",
                "in-place permute input/output must both be arrays or both tuples",
            ));
        }
    } else {
        1
    };
    let mut limit = if channel_present {
        context.partition_count
    } else {
        context.replica_count
    };
    if limit <= 0 {
        return Err(Error::new(
            "collective-group",
            "permute participant count must be positive",
        ));
    }
    if limit == 1 {
        for pair in &instruction.source_target_pairs {
            for id in [pair.source, pair.target] {
                limit = limit.max(
                    id.checked_add(1)
                        .ok_or_else(|| Error::new("overflow", "permute device ID overflow"))?,
                );
            }
        }
    }
    let mut sources = std::collections::BTreeMap::new();
    let mut targets = std::collections::BTreeMap::new();
    for pair in &instruction.source_target_pairs {
        for (id, seen) in [(pair.source, &mut sources), (pair.target, &mut targets)] {
            if id < 0 || id >= limit {
                return Err(Error::new(
                    "collective-group",
                    "permute source or target ID out of range",
                ));
            }
            let count = seen.entry(id).or_insert(0usize);
            if *count == allowed {
                return Err(Error::new(
                    "collective-group",
                    "permute source or target occurs too often",
                ));
            }
            *count += 1;
        }
    }
    Ok(inplace)
}

pub(crate) fn slice_widths(operands: &[Shape]) -> Result<Vec<usize>> {
    let [input, _, offsets, _] = operands else {
        return Err(Error::new(
            "collective",
            "in-place permute requires four operands",
        ));
    };
    if !offsets.is_tuple() {
        return Err(Error::new(
            "collective",
            "in-place permute offsets must be tuples",
        ));
    }
    let first = offsets.subshape(&[0])?;
    if input.is_tuple() {
        let flat = first.subshape(&[0])?.is_array();
        let mut widths = Vec::new();
        for (index, buffer) in input.as_proto().tuple_shapes.iter().enumerate() {
            let rank = Shape::from_proto(buffer.clone())?.rank()?;
            let count = if flat {
                1
            } else {
                offsets.subshape(&[index])?.as_proto().tuple_shapes.len()
            };
            widths.extend(std::iter::repeat_n(rank, count));
        }
        Ok(widths)
    } else {
        let count = if first.is_tuple() {
            offsets.as_proto().tuple_shapes.len()
        } else {
            1
        };
        Ok(vec![input.rank()?; count])
    }
}

pub(super) fn verify_module(module: &Module) -> Result<()> {
    let mut constrain = None;
    for instruction in module
        .as_proto()
        .computations
        .iter()
        .flat_map(|c| &c.instructions)
    {
        if matches!(
            instruction.opcode.as_str(),
            "all-reduce" | "all-reduce-start"
        ) {
            if constrain.is_some_and(|value| value != instruction.constrain_layout) {
                return Err(Error::new(
                    "collective",
                    "module mixes layout-constrained and unconstrained all-reduces",
                ));
            }
            constrain = Some(instruction.constrain_layout);
        }
    }
    Ok(())
}
