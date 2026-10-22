//! Linear block verification. HLO regions are isolated and single-block:
//! indexing definitions once replaces pliron's pairwise dominance searches.
use crate::{
    Diagnostic,
    attributes::Callees,
    inference::Signature,
    opcode::Opcode::*,
    ops::{
        ComputationOp, HloOpInterface, ReturnOp,
        control::{FusionOp, ParameterOp},
    },
    types::{TupleType, is_hlo_type},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{ops::ModuleOp, type_interfaces::FunctionTypeInterface, types::FunctionType},
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    region::Region,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::DefiningEntity,
};
use std::collections::{HashMap, HashSet};

use super::{
    CollectiveContext, VerifyOptions, buffers, control, error, instruction_error, layout, metadata,
    movement, numerical, optional_attr, require, same, sharding,
};
pub(crate) fn single_block(ctx: &Context, region: Ptr<Region>) -> Result<Ptr<BasicBlock>> {
    let data = region.deref(ctx);
    let mut blocks = data.iter(ctx);
    let block = blocks.next().ok_or_else(|| error("region is empty"))?;
    require(blocks.next().is_none(), "HLO regions must have one block")?;
    Ok(block)
}

fn signature(ctx: &Context, computation: &ComputationOp) -> Result<Signature> {
    let signature = computation
        .get_attr_hlo_computation_signature(ctx)
        .ok_or_else(|| error("missing computation signature"))?
        .get_type(ctx);
    let ty = signature.deref(ctx);
    let function = ty
        .downcast_ref::<FunctionType>()
        .ok_or_else(|| error("signature must be builtin.function"))?;
    let (args, results) = (function.arg_types(), function.res_types());
    require(
        results.len() == 1 && args.iter().chain(&results).all(|&ty| is_hlo_type(ctx, ty)),
        "HLO signature needs HLO argument types and one result",
    )?;
    Ok((args, results[0]))
}

pub(crate) fn returned(ctx: &Context, block: Ptr<BasicBlock>) -> Result<TypeHandle> {
    let term = block
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or_else(|| error("missing return"))?;
    require(
        Operation::is_op::<ReturnOp>(term, ctx),
        "HLO block must end with xlahlo.return",
    )?;
    Operation::get_op::<ReturnOp>(term, ctx)
        .ok_or_else(|| error("computation terminator must be xlahlo.return"))?
        .verify(ctx)?;
    Ok(term.deref(ctx).get_operand(0).get_type(ctx))
}

pub(crate) fn computation(ctx: &Context, computation: &ComputationOp) -> Result<()> {
    if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
        name.validate()?;
    }
    let (_, result) = signature(ctx, computation)?;
    computation_structure(ctx, computation, result)
}

fn computation_structure(
    ctx: &Context,
    computation: &ComputationOp,
    result: TypeHandle,
) -> Result<()> {
    let op = computation.get_operation().deref(ctx);
    require(
        op.get_num_operands() == 0
            && op.get_num_results() == 0
            && op.num_regions() == 1
            && op.get_num_successors() == 0,
        "invalid computation structure",
    )?;
    require(
        computation.get_attr_hlo_computation_name(ctx).is_some(),
        "computation requires symbol name",
    )?;
    let block = single_block(ctx, computation.body(ctx))?;
    require(
        block.deref(ctx).get_num_arguments() == 0,
        "computations use parameter ops, not block arguments",
    )?;
    require(
        same(ctx, returned(ctx, block)?, result),
        "computation return type mismatch",
    )
}

/// Verify an entire builtin module (or one self-contained computation).
/// Expected O(operations + operands + geometry payload), excluding type interning.
pub fn verify(ctx: &Context, root: Ptr<Operation>) -> Result<()> {
    verify_with_options(ctx, root, VerifyOptions::default())
}

/// Verify a builtin module or self-contained computation under explicit policies.
///
/// Module configuration may override topology unless `ignore_module_config` is
/// enabled. Missing callees, invalid attributes and unsupported IR return errors.
pub fn verify_with_options(
    ctx: &Context,
    root: Ptr<Operation>,
    mut options: VerifyOptions,
) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(root, ctx) || Operation::is_op::<ComputationOp>(root, ctx),
        "verification root must be builtin.module or xlahlo.computation",
    )?;
    let module_config = if Operation::is_op::<ModuleOp>(root, ctx) {
        Some(crate::ir::config(ctx, root)?)
    } else {
        None
    };
    if let Some(config) = module_config.as_ref()
        && !options.ignore_module_config
        && (config.replica_count != 1 || config.num_partitions != 1 || config.use_spmd_partitioning)
    {
        options.collective_context = CollectiveContext {
            replica_count: config.replica_count,
            partition_count: config.num_partitions,
        };
        options.sharding_context.use_spmd_partitioning = config.use_spmd_partitioning;
    }
    let mut computations = HashMap::new();
    let mut signatures = HashMap::new();
    let mut pending = vec![(root, None)];
    let mut all = Vec::new();
    while let Some((op, mut owner)) = pending.pop() {
        if let Some(computation) = Operation::get_op::<ComputationOp>(op, ctx) {
            require(
                owner.is_none(),
                "computation cannot be nested in another computation",
            )?;
            owner = Some(op);
            let name = computation
                .get_attr_hlo_computation_name(ctx)
                .ok_or_else(|| error("missing computation name"))?;
            require(
                computations.insert(name.clone(), op).is_none(),
                "duplicate computation name",
            )?;
            signatures.insert(op, signature(ctx, &computation)?);
        }
        all.push((op, owner));
        let data = op.deref(ctx);
        if Operation::is_op::<ModuleOp>(op, ctx) {
            require(
                op == root,
                "nested modules are outside HLO pipeline contract",
            )?;
        } else {
            require(owner.is_some(), "module may contain only computations")?;
        }
        for region in data.regions() {
            for block in region.deref(ctx).iter(ctx) {
                pending.extend(block.deref(ctx).iter(ctx).map(|child| (child, owner)));
            }
        }
    }
    let entry = if Operation::is_op::<ModuleOp>(root, ctx) {
        let entry = crate::ir::entry(ctx, root)?;
        *computations.get(&entry).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::UnknownComputation {
                name: entry.to_string(),
            })
        })?
    } else {
        root
    };
    if let Some(layout) = module_config
        .as_ref()
        .and_then(|config| config.entry_computation_layout.as_ref())
    {
        let (parameters, result) = &signatures[&entry];
        require(
            layout.parameter_shapes.len() == parameters.len(),
            "entry computation layout parameter count mismatch",
        )?;
        for (&layout_shape, &parameter_shape) in layout.parameter_shapes.iter().zip(parameters) {
            require(
                crate::shape::equal_including_layout(ctx, layout_shape, parameter_shape),
                "entry computation layout parameter shape mismatch",
            )?;
        }
        require(
            crate::shape::equal_including_layout(ctx, layout.result_shape, *result),
            "entry computation layout result shape mismatch",
        )?;
        // HloVerifier::VerifyEntryComputationLayout: the entry ROOT is compared
        // with the configured result by ShapesSame, i.e. exactly when layout
        // sensitive and by compatibility otherwise.
        let entry_computation = Operation::get_op::<ComputationOp>(entry, ctx)
            .ok_or_else(|| error("entry is not a computation"))?;
        let root_block = single_block(ctx, entry_computation.body(ctx))?;
        require(
            options.same_shapes(ctx, returned(ctx, root_block)?, layout.result_shape),
            "entry root shape differs from the entry computation layout result",
        )?;
    }
    let mut checked_types = HashSet::new();
    let mut instruction_names = HashSet::new();
    let mut has_buffers = false;
    let mut has_sharding = false;
    let mut has_collectives = false;
    let mut has_communication = false;
    let mut fusions = Vec::new();
    let payload_count = optional_attr::<crate::ir::MetadataPayloads>(
        ctx,
        root,
        crate::attributes::keys::METADATA_PAYLOADS,
    )?
    .map_or(0, |payloads| payloads.0.len());
    let stack_frame_count = optional_attr::<crate::ir::ModuleMetadata>(
        ctx,
        root,
        crate::attributes::keys::MODULE_METADATA,
    )?
    .and_then(|metadata| {
        metadata
            .stack_frame_index
            .as_ref()
            .map(|index| index.stack_frames.len())
    })
    .unwrap_or(0);
    let mut calls: HashMap<_, Vec<_>> = signatures.keys().map(|&op| (op, Vec::new())).collect();
    let hlo_dialect = pliron::identifier::Identifier::new("xlahlo").into();
    // Attribute-free arithmetic semantics depend only on opcode and immutable
    // interned types. Structural and instruction-specific policies still run.
    let mut arithmetic_last: [Option<[TypeHandle; 3]>; 3] = [None; 3];
    for &(op, owner) in &all {
        let dynamic = Operation::get_op_dyn(op, ctx);
        require(
            Operation::is_op::<ModuleOp>(op, ctx) || dynamic.get_opid().dialect == hlo_dialect,
            "non-HLO operation in computation",
        )?;
        let data = op.deref(ctx);
        if !has_sharding && !data.attributes.0.is_empty() {
            has_sharding = data
                .attributes
                .0
                .contains_key(&pliron::identifier::Identifier::new(
                    crate::attributes::keys::SHARDING,
                ))
                || data
                    .attributes
                    .0
                    .contains_key(&pliron::identifier::Identifier::new(
                        crate::attributes::keys::NAMED_SHARDING,
                    ));
        }
        for attribute in data.attributes.0.values() {
            attribute.verify(ctx)?;
            attribute.verify_interfaces(ctx)?;
        }
        if let Some(name) =
            optional_attr::<crate::ir::InstructionName>(ctx, op, crate::attributes::keys::NAME)?
        {
            require(
                !name.0.is_empty() && instruction_names.insert(name.0.clone()),
                "invalid or duplicate instruction name",
            )?;
        }
        for ty in data.result_types() {
            verify_type(ctx, ty, &mut checked_types, &mut has_buffers)?;
        }
        let hlo = pliron::op::op_cast::<dyn HloOpInterface>(dynamic.as_ref());
        if let Some(hlo) = hlo {
            if hlo.opcode() == Fusion {
                fusions.push((op, owner));
            }
            has_collectives |= matches!(
                hlo.opcode(),
                AllGather
                    | AllGatherStart
                    | AllGatherDone
                    | AllReduce
                    | AllReduceStart
                    | AllReduceDone
                    | AllToAll
                    | RaggedAllToAll
                    | CollectiveBroadcast
                    | CollectivePermute
                    | CollectivePermuteStart
                    | CollectivePermuteDone
                    | CollectiveReduce
                    | ReduceScatter
            );
            has_communication |= matches!(
                hlo.opcode(),
                CopyStart
                    | CopyDone
                    | AsyncStart
                    | AsyncUpdate
                    | AsyncDone
                    | Send
                    | SendDone
                    | Recv
                    | RecvDone
                    | Infeed
                    | Outfeed
            );
            let arithmetic_key = match hlo.opcode() {
                Add | Multiply | Exp
                    if data.attributes.0.is_empty()
                        && data.get_num_results() == 1
                        && data.num_regions() == 0
                        && data.get_num_successors() == 0
                        && data.get_num_operands() == if hlo.opcode() == Exp { 1 } else { 2 } =>
                {
                    let slot = match hlo.opcode() {
                        Add => 0,
                        Multiply => 1,
                        _ => 2,
                    };
                    Some((
                        slot,
                        [
                            data.get_type(0),
                            data.get_operand(0).get_type(ctx),
                            data.get_operand(data.get_num_operands() - 1).get_type(ctx),
                        ],
                    ))
                }
                _ => None,
            };
            if arithmetic_key.is_none_or(|(slot, key)| arithmetic_last[slot] != Some(key)) {
                hlo.verify_with_options(ctx, options)
                    .map_err(|error| instruction_error(ctx, op, error))?;
                if let Some((slot, key)) = arithmetic_key {
                    arithmetic_last[slot] = Some(key);
                }
            }
            layout::verify_instruction(ctx, op, hlo.opcode(), options)
                .map_err(|error| instruction_error(ctx, op, error))?;
        } else {
            require(
                Operation::is_op::<ModuleOp>(op, ctx)
                    || Operation::is_op::<ComputationOp>(op, ctx)
                    || Operation::is_op::<ReturnOp>(op, ctx),
                "unrecognized HLO instruction",
            )?;
            if let Some(computation) = Operation::get_op::<ComputationOp>(op, ctx) {
                computation_structure(ctx, &computation, signatures[&op].1)?;
            } else {
                dynamic.verify(ctx)?;
            }
        }
        // Interface verification is deliberately not generic recursive verify_op.
        dynamic.verify_interfaces(ctx)?;
        metadata::verify_instruction(ctx, op, options, payload_count, stack_frame_count)?;
        for region in data.regions() {
            for block in region.deref(ctx).iter(ctx) {
                for argument in block.deref(ctx).arguments() {
                    verify_type(
                        ctx,
                        argument.get_type(ctx),
                        &mut checked_types,
                        &mut has_buffers,
                    )?;
                }
                verify_block(ctx, block)?;
            }
        }
        if let Some(parameter) = Operation::get_op::<ParameterOp>(op, ctx) {
            let parent = data
                .get_parent_op(ctx)
                .and_then(|p| Operation::get_op::<ComputationOp>(p, ctx))
                .ok_or_else(|| error("parameter must belong to computation"))?;
            let (args, _) = &signatures[&parent.get_operation()];
            let index = parameter
                .index(ctx)
                .ok_or_else(|| error("missing parameter index"))?
                .0 as usize;
            require(
                args.get(index)
                    .is_some_and(|&ty| same(ctx, ty, data.get_type(0))),
                "parameter index/type differs from signature",
            )?;
        }
        let callees = optional_attr::<Callees>(ctx, op, crate::attributes::keys::CALLEES)?;
        let mut resolved = Vec::new();
        if let Some(callees) = &callees {
            let opcode = hlo
                .map(|hlo| hlo.opcode())
                .ok_or_else(|| error("callee attribute requires an HLO instruction"))?;
            let caller = owner.ok_or_else(|| error("callee requires a computation owner"))?;
            let has_to_apply = matches!(
                opcode,
                crate::opcode::Opcode::AllReduce
                    | crate::opcode::Opcode::AllReduceStart
                    | crate::opcode::Opcode::Call
                    | crate::opcode::Opcode::Map
                    | crate::opcode::Opcode::Reduce
                    | crate::opcode::Opcode::ReduceScatter
                    | crate::opcode::Opcode::CollectiveReduce
                    | crate::opcode::Opcode::ReduceWindow
                    | crate::opcode::Opcode::Scatter
                    | crate::opcode::Opcode::Sort
                    | crate::opcode::Opcode::Scan
            ) || (opcode == crate::opcode::Opcode::CustomCall
                && callees.0.len() == 1);
            let same_thread = matches!(
                opcode,
                crate::opcode::Opcode::While
                    | crate::opcode::Opcode::Conditional
                    | crate::opcode::Opcode::Fusion
            ) || (options.verify_call_nested_computation_thread_name
                && has_to_apply);
            for name in &callees.0 {
                let callee = computations.get(name).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::UnknownComputation {
                        name: name.to_string(),
                    })
                })?;
                if same_thread {
                    require(
                        metadata::execution_thread(ctx, caller)?
                            == metadata::execution_thread(ctx, *callee)?,
                        "callee execution thread differs from caller",
                    )?;
                }
                resolved.push(&signatures[callee]);
                calls
                    .get_mut(&caller)
                    .ok_or_else(|| error("caller is missing from the computation index"))?
                    .push(*callee);
            }
        }
        if let Some(hlo) = hlo {
            verify_callees(ctx, op, hlo.opcode(), options, &resolved)
                .map_err(|error| instruction_error(ctx, op, error))?;
        }
    }
    for &computation in computations.values() {
        crate::ir::instruction_order::verify_order(ctx, computation)?;
    }
    let mut indegree: HashMap<_, usize> = calls.keys().map(|&op| (op, 0)).collect();
    for targets in calls.values() {
        for target in targets {
            *indegree
                .get_mut(target)
                .ok_or_else(|| error("callee is missing from the computation index"))? += 1;
        }
    }
    for (fusion, owner) in fusions {
        let owner = owner.ok_or_else(|| error("fusion requires a computation owner"))?;
        let relaxed = owner != entry && indegree[&owner] == 0;
        control::verify_fusion_aliases(ctx, fusion, options, relaxed)?;
    }
    let mut ready: Vec<_> = indegree
        .iter()
        .filter_map(|(&op, &degree)| (degree == 0).then_some(op))
        .collect();
    let mut count = 0;
    while let Some(op) = ready.pop() {
        count += 1;
        for target in &calls[&op] {
            let degree = indegree
                .get_mut(target)
                .ok_or_else(|| error("callee is missing from the computation index"))?;
            *degree = degree
                .checked_sub(1)
                .ok_or_else(|| error("inconsistent computation call graph indegree"))?;
            if *degree == 0 {
                ready.push(*target);
            }
        }
    }
    require(
        count == calls.len(),
        "recursive computation calls are forbidden",
    )?;
    for (&op, (args, result)) in &signatures {
        for &ty in args.iter().chain(std::iter::once(result)) {
            verify_type(ctx, ty, &mut checked_types, &mut has_buffers)?;
        }
        let block = single_block(ctx, op.deref(ctx).get_region(0))?;
        let count = block
            .deref(ctx)
            .iter(ctx)
            .filter(|&op| Operation::is_op::<ParameterOp>(op, ctx))
            .count();
        require(
            count == args.len(),
            "computation must declare every parameter exactly once",
        )?;
    }
    if Operation::is_op::<ModuleOp>(root, ctx) {
        metadata::verify_aliases_and_donors(ctx, root, &computations, &signatures, options)?;
        metadata::verify_schedule(ctx, root, &computations, &calls)?;
    }
    require(
        options.collective_context.partition_count > 0,
        "partition count must be positive",
    )?;
    if has_sharding {
        sharding::verify(ctx, &all, &computations, options)?;
    }
    if has_buffers {
        buffers::verify(ctx, root, &all, &computations, options)?;
    }
    if has_communication || has_collectives {
        control::asynchronous::verify_graph(ctx, root, &all, &computations, &signatures, options)?;
    }
    if has_collectives {
        control::collective::verify_module(
            ctx,
            &all.iter().map(|(op, _)| *op).collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

fn verify_callees(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: crate::opcode::Opcode,
    options: VerifyOptions,
    resolved: &[&Signature],
) -> Result<()> {
    match opcode {
        Reduce | ReduceWindow | SelectAndScatter | Map | Scan | Sort => {
            numerical::verify_callees(ctx, op, opcode, options, resolved)
        }
        Scatter => movement::verify_callees(ctx, op, opcode, options, resolved),
        Call | While | Conditional | Fusion | CustomCall | AllReduce | AllReduceStart
        | ReduceScatter | CollectiveReduce | AsyncStart | AsyncUpdate | AsyncDone => {
            control::verify_callees(ctx, op, opcode, options, resolved)
        }
        _ => require(
            resolved.is_empty(),
            "opcode does not accept called computations",
        ),
    }
}

fn verify_type(
    ctx: &Context,
    ty: TypeHandle,
    checked: &mut HashSet<TypeHandle>,
    has_buffers: &mut bool,
) -> Result<()> {
    if !checked.insert(ty) {
        return Ok(());
    }
    let data = ty.deref(ctx);
    data.verify(ctx)?;
    data.verify_interfaces(ctx)?;
    if let Some(tuple) = data.downcast_ref::<TupleType>() {
        for &child in &tuple.elements {
            verify_type(ctx, child, checked, has_buffers)?;
        }
    } else if let Some(buffer) = data.downcast_ref::<crate::types::BufferType>() {
        *has_buffers = true;
        verify_type(ctx, buffer.array, checked, has_buffers)?;
    }
    Ok(())
}

fn verify_block(ctx: &Context, block: Ptr<BasicBlock>) -> Result<()> {
    let data = block.deref(ctx);
    let ordered: Vec<_> = data.iter(ctx).collect();
    let positions: HashMap<_, _> = ordered.iter().enumerate().map(|(i, &op)| (op, i)).collect();
    let args: HashSet<_> = data.arguments().collect();
    let mut parameter_indices = HashSet::new();
    for (i, &op) in ordered.iter().enumerate() {
        let data = op.deref(ctx);
        if let Some(parameter) = Operation::get_op::<ParameterOp>(op, ctx)
            && let Some(index) = parameter.index(ctx)
        {
            require(
                parameter_indices.insert(index.0),
                "duplicate parameter index",
            )?;
        }
        require(
            !Operation::is_op::<ReturnOp>(op, ctx) || i + 1 == ordered.len(),
            "return must be last",
        )?;
        for operand in data.operands() {
            match operand.defining_entity() {
                DefiningEntity::Op(def) => require(
                    positions.get(&def).is_some_and(|&pos| pos < i),
                    "operand does not dominate use or captures another region",
                )?,
                DefiningEntity::Block(owner) => require(
                    owner == block && args.contains(&operand),
                    "invalid block argument capture",
                )?,
            }
        }
    }
    if block
        .deref(ctx)
        .get_parent_op(ctx)
        .is_some_and(|parent| Operation::is_op::<FusionOp>(parent, ctx))
    {
        // FusionOp verification checks its parameter graph before this region is visited.
        Ok(())
    } else {
        metadata::verify_control_dependencies(ctx, block)
    }
}
