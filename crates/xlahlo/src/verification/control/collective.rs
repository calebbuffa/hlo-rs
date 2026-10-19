//! Replica-group policies, collective geometry, and module consistency.
use super::{VerifyOptions, channel, flag, optional_attr, require, required, same_ignoring_memory};
use crate::Diagnostic;
use crate::{
    attributes::{Axes, Callees, OptionalInteger, ReplicaGroups, SourceTargetPairs},
    inference::{
        Signature, axes,
        control::{self as infer, collective as ci},
    },
    opcode::{Opcode, Opcode::*},
    ops::control::{AllReduceOp, AllReduceStartOp, DynamicSliceSizes},
    replica_groups::{IotaReplicaGroupList, MeshAxesReplicaGroupList},
    shape::Shape,
};
use pliron::result::Result;
use pliron::{
    attribute::Attribute,
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::TypeHandle,
};
use std::cell::Ref;
use std::collections::{HashMap, HashSet};

pub(crate) fn register(ctx: &mut Context) {
    IotaReplicaGroupList::register::<IotaReplicaGroupList>(ctx);
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
            return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "global device IDs require a channel".into()
            }));
        }
        (false, _) => GroupMode::Replica,
        (true, None) => GroupMode::Partition,
        (true, Some(false)) => GroupMode::ReplicaAndPartition,
        (true, Some(true)) => GroupMode::Flattened,
    })
}

fn groups(
    ctx: &Context,
    op: Ptr<Operation>,
    uniform: bool,
    check: bool,
) -> Result<Option<(i64, i64)>> {
    if let Some(iota) = optional_attr::<IotaReplicaGroupList>(ctx, op, "hlo_iota_groups")? {
        require(
            iota.num_replica_groups > 0
                && iota.num_devices_per_group > 0
                && !iota.iota_reshape_dims.is_empty()
                && iota.iota_reshape_dims.len() == iota.iota_transpose_perm.len(),
            "invalid iota group dimensions",
        )?;
        let total = iota
            .num_replica_groups
            .checked_mul(iota.num_devices_per_group)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "iota device count overflow".into()
                })
            })?;
        let mut product = 1i64;
        for &size in &iota.iota_reshape_dims {
            require(size > 0, "iota reshape dimensions must be positive")?;
            product = product.checked_mul(size).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "iota reshape product overflow".into()
                })
            })?;
        }
        axes(
            iota.iota_reshape_dims.len(),
            &iota.iota_transpose_perm,
            false,
        )?;
        require(product == total, "iota reshape product is invalid")?;
        return Ok(Some((total, iota.num_devices_per_group)));
    }
    if let Some(mesh) = optional_attr::<MeshAxesReplicaGroupList>(ctx, op, "hlo_mesh_groups")? {
        mesh.verify(ctx)?;
        require(
            !mesh.mesh.axes.is_empty() || mesh.mesh.device_ids.first().is_none_or(|&id| id == 0),
            "single-device collective mesh must use device zero",
        )?;
        let size = mesh.num_devices_per_group()?;
        let total = mesh
            .num_replica_groups()?
            .checked_mul(size)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "mesh group device count overflow".into()
                })
            })?;
        return Ok((total != 0).then_some((total, size)));
    }
    let groups: Ref<'_, ReplicaGroups> = required(ctx, op, "hlo_groups")?;
    if groups.0.is_empty() {
        return Ok(None);
    }
    let size = groups.0[0].0.len();
    let mut total = 0usize;
    for group in &groups.0 {
        require(
            !check || !group.0.is_empty() && (!uniform || group.0.len() == size),
            "replica groups must be nonempty and, for this operation, uniform",
        )?;
        total = total.checked_add(group.0.len()).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "replica group count overflow".into()
            })
        })?;
    }
    if check {
        let mut seen = HashSet::new();
        for &id in groups.0.iter().flat_map(|g| &g.0) {
            require(
                usize::try_from(id).is_ok_and(|id| id < total) && seen.insert(id),
                "replica groups must contain every ID from zero to total-1 exactly once",
            )?;
        }
    }
    Ok(Some((
        i64::try_from(total).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "too many replica IDs".into()
            })
        })?,
        i64::try_from(size).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "replica group too large".into()
            })
        })?,
    )))
}

fn subgroup(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<i64> {
    let checks_only = matches!(
        opcode,
        Opcode::AllReduce
            | Opcode::AllReduceStart
            | Opcode::CollectiveReduce
            | Opcode::RaggedAllToAll
    );
    if checks_only && !options.check_replica_groups {
        return Ok(1);
    }
    let global = if matches!(opcode, Opcode::AllToAll | Opcode::RaggedAllToAll) {
        None
    } else {
        Some(flag(ctx, op, "hlo_use_global_device_ids")?)
    };
    let mode = group_mode(channel(ctx, op)?.is_some(), global)?;
    let context = options.collective_context;
    let uniform = !matches!(
        opcode,
        Opcode::AllReduce | Opcode::AllReduceStart | Opcode::CollectiveReduce
    );
    let groups = groups(ctx, op, uniform, options.check_replica_groups)?;
    if options.check_replica_groups {
        require(
            context.replica_count > 0 && context.partition_count > 0,
            "replica and partition counts must be positive",
        )?;
        let expected = match mode {
            GroupMode::Replica | GroupMode::ReplicaAndPartition => context.replica_count,
            GroupMode::Partition => context.partition_count,
            GroupMode::Flattened => context
                .replica_count
                .checked_mul(context.partition_count)
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "replica/partition count product overflow".into()
                    })
                })?,
        };
        require(
            groups.is_none_or(|(total, _)| expected == 1 || total == expected),
            "replica-group total differs from module configuration",
        )?;
    }
    require(
        groups.is_some() || !matches!(mode, GroupMode::Flattened),
        "flattened device IDs require explicit groups",
    )?;
    if checks_only {
        return Ok(1);
    }
    let count = groups.map(|(_, size)| size).unwrap_or(match mode {
        GroupMode::Partition => context.partition_count,
        _ => context.replica_count,
    });
    if matches!(mode, GroupMode::ReplicaAndPartition) {
        count.checked_mul(context.partition_count).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "collective subgroup size overflow".into()
            })
        })
    } else {
        Ok(count)
    }
}

fn shard_count(
    ctx: &Context,
    op: Ptr<Operation>,
    result: TypeHandle,
    operands: &[TypeHandle],
    gather: bool,
    subgroup: i64,
) -> Result<(i64, i64)> {
    let dimensions: Ref<'_, Axes> = required(ctx, op, "hlo_dimensions")?;
    let [axis] = dimensions.0.as_slice() else {
        return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "collective requires exactly one dimension".into()
        }));
    };
    let index = usize::try_from(*axis).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "negative collective dimension".into()
        })
    })?;
    require(!operands.is_empty(), "collective requires operands")?;
    let children = Shape::children(ctx, result);
    let results = if operands.len() == 1 {
        std::slice::from_ref(&result)
    } else {
        require(
            Shape::is_tuple(ctx, result),
            "variadic collective result requires a tuple",
        )?;
        &children
    };
    require(
        results.len() == operands.len(),
        "variadic collective result requires matching tuple cardinality",
    )?;
    for (&operand, &result) in operands.iter().zip(results) {
        require(
            index < Shape::rank(ctx, operand)? && index < Shape::rank(ctx, result)?,
            "collective dimension out of bounds",
        )?;
    }
    let a = Shape::dimensions(ctx, operands[0])?[index];
    let b = Shape::dimensions(ctx, results[0])?[index];
    let (numerator, denominator) = if gather { (b, a) } else { (a, b) };
    require(
        denominator > 0 && numerator >= 0,
        "cannot infer collective shard count from zero or unbounded extent",
    )?;
    let count = numerator / denominator + i64::from(numerator % denominator != 0);
    require(
        subgroup == 1 || count == subgroup,
        "shape-derived shard count differs from subgroup size",
    )?;
    Ok((*axis, count))
}

pub(super) fn infer(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<TypeHandle> {
    if opcode == AllReduce {
        require(
            channel(ctx, op)?.is_none_or(|id| id > 0),
            "all-reduce channel ID must be positive when present",
        )?;
    }
    if matches!(opcode, AllReduce | AllReduceStart) {
        let legacy = optional_attr::<OptionalInteger>(ctx, op, "hlo_all_reduce_id")?;
        require(
            !(channel(ctx, op)?.is_some_and(|n| n > 0)
                && matches!(legacy.as_deref(),Some(OptionalInteger::Value(n)) if *n>0)),
            "both channel_id and legacy all_reduce_id are present",
        )?;
    }
    if matches!(opcode, CollectivePermute | CollectivePermuteStart) {
        let inplace = verify_permute(ctx, op, operands, options)?;
        return if opcode == CollectivePermute {
            ci::infer_collective_permute(ctx, operands, inplace)
        } else {
            let children = Shape::children(ctx, result);
            ci::infer_collective_permute_start(
                ctx,
                operands,
                children.get(2..).unwrap_or_default(),
                inplace,
            )
        };
    }
    match opcode {
        AllGatherDone => return ci::infer_all_gather_done(ctx, operands[0]),
        AllReduceDone => return ci::infer_all_reduce_done(ctx, operands[0]),
        CollectivePermuteDone => return ci::infer_collective_permute_done(ctx, operands[0]),
        _ => {}
    }
    if matches!(
        opcode,
        AllReduce | AllReduceStart | ReduceScatter | CollectiveReduce
    ) {
        let callees: Ref<'_, Callees> = required(ctx, op, "hlo_callees")?;
        require(
            callees.0.len() == 1,
            "collective reduction requires one reducer",
        )?;
    }
    let subgroup = if opcode == CollectiveBroadcast {
        1
    } else {
        subgroup(ctx, op, opcode, options)?
    };
    match opcode {
        AllGather => {
            let (axis, count) = shard_count(ctx, op, result, operands, true, subgroup)?;
            ci::infer_all_gather(ctx, operands, axis, count)
        }
        AllGatherStart => {
            require(
                Shape::is_tuple(ctx, result) && Shape::children(ctx, result).len() == 2,
                "all-gather-start requires an input/result tuple",
            )?;
            let output = infer::infer_get_tuple_element(ctx, result, 1)?;
            let (axis, count) = shard_count(ctx, op, output, operands, true, subgroup)?;
            ci::infer_all_gather_start(ctx, operands, axis, count)
        }
        ReduceScatter => {
            let (axis, count) = shard_count(ctx, op, result, operands, false, subgroup)?;
            ci::infer_reduce_scatter(ctx, operands, axis, count)
        }
        AllReduce | AllReduceStart => ci::infer_all_reduce(ctx, operands),
        AllToAll => {
            let dims: Ref<'_, Axes> = required(ctx, op, "hlo_dimensions")?;
            match dims.0.as_slice() {
                [axis] => {
                    require(operands.len() == 1, "array all-to-all requires one operand")?;
                    ci::infer_all_to_all(ctx, operands[0], *axis, *axis, subgroup)
                }
                [] => {
                    require(
                        i64::try_from(operands.len()) == Ok(subgroup),
                        "tuple all-to-all operand count differs from subgroup size",
                    )?;
                    ci::infer_all_to_all_tuple(ctx, operands)
                }
                _ => Err(pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "all-to-all requires zero or one split dimension".into()
                })),
            }
        }
        RaggedAllToAll => {
            require(
                operands.len() == 6,
                "ragged all-to-all requires six operands",
            )?;
            for index in 2..5 {
                require(
                    matches!(Shape::rank(ctx, operands[index])?, 1 | 2)
                        && same_ignoring_memory(ctx, operands[index], operands[index + 1], false)?,
                    "ragged offsets/sizes must be identical rank-one/two shapes apart from memory space",
                )?;
            }
            ci::infer_ragged_all_to_all(ctx, operands)
        }
        CollectiveReduce => {
            ci::infer_collective_reduce(ctx, operands, flag(ctx, op, "hlo_has_dynamic_root")?)
        }
        // Broadcast verification uses static-root geometry.
        CollectiveBroadcast => ci::infer_collective_broadcast(ctx, operands, false),
        _ => Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "unsupported collective inference opcode".into()
        })),
    }
}

fn buffer_offset(ctx: &Context, buffer: TypeHandle, offset: TypeHandle) -> Result<()> {
    require(
        Shape::is_tuple(ctx, offset),
        "permute buffer offsets must be tuples",
    )?;
    let children = Shape::children(ctx, offset);
    let arrays = children.iter().all(|&s| Shape::is_array_or_buffer(ctx, s));
    let tuples = children.iter().all(|&s| Shape::is_tuple(ctx, s));
    require(
        arrays || tuples,
        "permute offset tuple must contain only arrays or only tuples",
    )?;
    let rank = Shape::rank(ctx, Shape::array_view(ctx, buffer)?)?;
    require(
        if tuples {
            children
                .iter()
                .all(|&s| Shape::children(ctx, s).len() == rank)
        } else {
            children.len() == rank
        },
        "permute offsets must match buffer rank",
    )
}

fn verify_permute(
    ctx: &Context,
    op: Ptr<Operation>,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<bool> {
    let slices = optional_attr::<DynamicSliceSizes>(ctx, op, "hlo_dynamic_slice_sizes")?;
    let inplace = slices.is_some_and(|s| !s.0.is_empty());
    let allowed = if inplace {
        require(
            operands.len() == 4,
            "in-place permute requires four operands",
        )?;
        let (input, output, io, oo) = (operands[0], operands[1], operands[2], operands[3]);
        if Shape::is_array_or_buffer(ctx, input) && Shape::is_array_or_buffer(ctx, output) {
            buffer_offset(ctx, input, io)?;
            buffer_offset(ctx, output, oo)?;
            Shape::children(ctx, io).len()
        } else {
            require(
                Shape::is_tuple(ctx, input)
                    && Shape::is_tuple(ctx, output)
                    && Shape::is_tuple(ctx, io)
                    && Shape::is_tuple(ctx, oo),
                "in-place permute input/output must both be arrays or both tuples",
            )?;
            let inputs = Shape::children(ctx, input);
            let outputs = Shape::children(ctx, output);
            let ios = Shape::children(ctx, io);
            let oos = Shape::children(ctx, oo);
            require(
                inputs.len() == outputs.len()
                    && inputs.len() == ios.len()
                    && outputs.len() == oos.len()
                    && !inputs.is_empty(),
                "in-place permute tuple buffers and offsets have different cardinalities",
            )?;
            for (&buffer, &offset) in inputs
                .iter()
                .zip(ios.iter())
                .chain(outputs.iter().zip(oos.iter()))
            {
                buffer_offset(ctx, buffer, offset)?;
            }
            Shape::children(ctx, ios[0]).len()
        }
    } else {
        1
    };
    let pairs: Ref<'_, SourceTargetPairs> = required(ctx, op, "hlo_pairs")?;
    let mut limit = if channel(ctx, op)?.is_some() {
        options.collective_context.partition_count
    } else {
        options.collective_context.replica_count
    };
    require(limit > 0, "permute participant count must be positive")?;
    if limit == 1 {
        for pair in &pairs.0 {
            for id in [pair.source, pair.target] {
                limit = limit.max(id.checked_add(1).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "permute device ID overflow".into()
                    })
                })?);
            }
        }
    }
    let mut sources = HashMap::new();
    let mut targets = HashMap::new();
    for pair in &pairs.0 {
        for (id, seen) in [(pair.source, &mut sources), (pair.target, &mut targets)] {
            require(
                id >= 0 && id < limit,
                "permute source or target ID out of range",
            )?;
            let count = seen.entry(id).or_insert(0usize);
            require(
                *count < allowed,
                "permute source or target occurs too often",
            )?;
            *count += 1;
        }
    }
    Ok(inplace)
}

pub(super) fn verify_reducer(
    _: &Context,
    _: Ptr<Operation>,
    _: &[TypeHandle],
    resolved: &[&Signature],
    _: VerifyOptions,
) -> Result<()> {
    // Reducer cardinality is checked without constraining its scalar signature,
    // including for heterogeneous variadic collectives.
    require(
        resolved.len() == 1,
        "collective reduction requires one reducer",
    )
}

/// Enforce consistent layout-constraint policy across all-reduces in a module.
///
/// `instructions` must contain the module's complete native instruction index.
pub(crate) fn verify_module(ctx: &Context, instructions: &[Ptr<Operation>]) -> Result<()> {
    let mut constrain = None;
    for &op in instructions {
        if Operation::is_op::<AllReduceOp>(op, ctx) || Operation::is_op::<AllReduceStartOp>(op, ctx)
        {
            let value = flag(ctx, op, "hlo_constrain_layout")?;
            require(
                constrain.is_none_or(|previous| previous == value),
                "module mixes layout-constrained and unconstrained all-reduces",
            )?;
            constrain = Some(value);
        }
    }
    Ok(())
}
