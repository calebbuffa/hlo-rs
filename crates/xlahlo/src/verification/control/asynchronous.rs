//! Asynchronous binding prefixes, ownership, and channel pairing.
use super::{
    VerifyOptions, channel, flag, optional_attr, require, required, same, same_ignoring_memory,
};
use crate::Diagnostic;
use crate::{
    attributes::{Callees, FrontendAttributes, Index, OutputOperandAliases, ResultAccuracy},
    inference::{Signature, control as infer},
    ir::ExecutionThread,
    literal::Literal,
    opcode::{Opcode, Opcode::*},
    ops::{HloOpInterface, ReturnOp},
    shape::Shape,
    types::{BufferType, TokenType},
};
use pliron::result::Result;
use pliron::{
    builtin::attributes::{StringAttr, TypeAttr},
    context::{Context, Ptr},
    identifier::Identifier,
    op::op_cast,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::{DefiningEntity, Value},
};
use std::{
    cell::{Ref, RefCell},
    collections::{HashMap, HashSet},
};

pub(crate) fn opcode(ctx: &Context, op: Ptr<Operation>) -> Option<Opcode> {
    let dynamic = Operation::get_op_dyn(op, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref()).map(|hlo| hlo.opcode())
}

pub(super) fn infer(
    ctx: &Context,
    op: Ptr<Operation>,
    kind: Opcode,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<TypeHandle> {
    match kind {
        CopyStart => infer::infer_copy_start(ctx, operands[0]),
        CopyDone => {
            let destination = infer::infer_copy_done(ctx, operands[0])?;
            if options.layout_sensitive {
                require(
                    same_ignoring_memory(
                        ctx,
                        destination,
                        infer::infer_get_tuple_element(ctx, operands[0], 1)?,
                        true,
                    )?,
                    "copy context source and destination differ",
                )?;
            }
            Ok(destination)
        }
        Send => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            infer::infer_send(ctx, operands[0])
        }
        SendDone => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            Ok(Shape::token(ctx))
        }
        Recv => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            infer::infer_recv(ctx, infer::infer_get_tuple_element(ctx, result, 0)?)
        }
        RecvDone => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            infer::infer_recv_done(ctx, operands[0])
        }
        Infeed => {
            require(
                operands[0].deref(ctx).is::<TokenType>(),
                "infeed requires a token operand",
            )?;
            Shape::tuple(
                ctx,
                vec![
                    infer::infer_get_tuple_element(ctx, result, 0)?,
                    Shape::token(ctx),
                ],
            )
        }
        Outfeed => {
            require(
                operands[1].deref(ctx).is::<TokenType>(),
                "outfeed requires a token second operand",
            )?;
            let host: Ref<'_, TypeAttr> = required(ctx, op, "hlo_outfeed_shape")?;
            require(
                options.same_shapes(ctx, host.get_type(ctx), operands[0]),
                "outfeed host shape differs from data operand",
            )?;
            Ok(Shape::token(ctx))
        }
        AsyncStart => {
            let callees: Ref<'_, Callees> = required(ctx, op, "hlo_callees")?;
            require(
                callees.0.len() == 1,
                "async-start requires one wrapped computation",
            )?;
            let (bound, _) = infer::async_context(ctx, result)?;
            require(
                bound.len() == operands.len()
                    && bound
                        .iter()
                        .zip(operands)
                        .all(|(&a, &b)| options.same_shapes(ctx, a, b)),
                "async-start operand differs from bound parameter shape",
            )?;
            Ok(result)
        }
        AsyncUpdate => {
            let previous = operands.first().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "async-update needs a predecessor".into()
                })
            })?;
            let (previous_bound, previous_output) = infer::async_context(ctx, *previous)?;
            let (bound, output) = infer::async_context(ctx, result)?;
            require(
                prefix(ctx, previous_output, output, options)?
                    && previous_bound.len() <= bound.len(),
                "async-update must extend predecessor parameter/result prefixes",
            )?;
            for (&a, &b) in previous_bound.iter().zip(bound.iter()) {
                require(
                    prefix(ctx, a, b, options)?,
                    "async-update must extend bound parameter prefixes",
                )?;
            }
            require(
                operands.len() - 1 == bound.len() - previous_bound.len(),
                "async-update new operands do not match newly bound parameters",
            )?;
            require(
                operands[1..]
                    .iter()
                    .zip(&bound[previous_bound.len()..])
                    .all(|(&a, &b)| options.same_shapes(ctx, a, b)),
                "async-update new operand shape differs from newly bound parameter",
            )?;
            Ok(result)
        }
        AsyncDone => {
            infer::async_context(ctx, operands[0])?;
            Ok(result)
        }
        _ => Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "unsupported async inference opcode".into()
        })),
    }
}

fn prefix(ctx: &Context, a: TypeHandle, b: TypeHandle, options: VerifyOptions) -> Result<bool> {
    let mut work = vec![(a, b)];
    while let Some((a, b)) = work.pop() {
        if options.same_shapes(ctx, a, b) {
            continue;
        }
        if Shape::is_tuple(ctx, a) && Shape::children(ctx, a).is_empty() {
            continue;
        }
        if !Shape::is_tuple(ctx, a) || !Shape::is_tuple(ctx, b) {
            return Ok(false);
        }
        let a = Shape::children(ctx, a);
        let b = Shape::children(ctx, b);
        if a.len() > b.len() {
            return Ok(false);
        }
        work.extend(a.iter().copied().zip(b.iter().copied()));
    }
    Ok(true)
}

pub(super) fn verify_callees(
    ctx: &Context,
    _: Ptr<Operation>,
    kind: Opcode,
    result: TypeHandle,
    _: &[TypeHandle],
    resolved: &[&Signature],
    options: VerifyOptions,
) -> Result<()> {
    if kind != Opcode::AsyncStart && resolved.is_empty() {
        return Ok(());
    }
    require(
        resolved.len() == 1,
        "async instruction requires one wrapped computation",
    )?;
    let callee = &resolved[0];
    if kind == Opcode::AsyncDone {
        require(
            options.same_shapes(ctx, result, callee.1),
            "async-done requires full computation result shape",
        )
    } else {
        let (parameters, output) = infer::async_context(ctx, result)?;
        require(
            parameters.len() <= callee.0.len(),
            "bound parameters must prefix wrapped computation",
        )?;
        for (&a, &b) in parameters.iter().zip(&callee.0) {
            require(
                prefix(ctx, a, b, options)?,
                "bound parameters must prefix wrapped computation",
            )?;
        }
        require(
            prefix(ctx, output, callee.1, options)?,
            "bound result must prefix wrapped computation result",
        )
    }
}

fn direct(value: Value) -> Option<Ptr<Operation>> {
    match value.defining_entity() {
        DefiningEntity::Op(op) => Some(op),
        _ => None,
    }
}

fn target(ctx: &Context, op: Ptr<Operation>) -> Result<Option<Ref<'_, StringAttr>>> {
    optional_attr(ctx, op, "hlo_target")
}

fn intermediary(ctx: &Context, op: Ptr<Operation>) -> Result<bool> {
    Ok(matches!(
        opcode(ctx, op),
        Some(
            Opcode::Tuple
                | Opcode::GetTupleElement
                | Opcode::OptimizationBarrier
                | Opcode::Copy
                | Opcode::Parameter
                | Opcode::While
        )
    ) || opcode(ctx, op) == Some(Opcode::CustomCall)
        && target(ctx, op)?.is_some_and(|s| {
            matches!(
                s.as_str(),
                "Sharding"
                    | "LocalToGlobalShape"
                    | "GlobalToLocalShape"
                    | "xla.sdy.LocalToGlobalShape"
                    | "xla.sdy.GlobalToLocalShape"
                    | "xla.sdy.FuncResultSharding"
            )
        }))
}

type ComputationPair = (Ptr<Operation>, Ptr<Operation>);

struct Graph<'a> {
    ctx: &'a Context,
    signatures: &'a HashMap<Ptr<Operation>, Signature>,
    owners: HashMap<Ptr<Operation>, Ptr<Operation>>,
    roots: HashMap<Ptr<Operation>, Value>,
    users: HashMap<Value, Vec<Ptr<Operation>>>,
    callees: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
    callers: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
    instructions: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
    producer_cache: RefCell<HashMap<Value, Option<Ptr<Operation>>>>,
    wrapped_cache: RefCell<HashMap<Ptr<Operation>, Ptr<Operation>>>,
    equality_cache: RefCell<HashMap<ComputationPair, bool>>,
    start_cache: RefCell<HashMap<Ptr<Operation>, Ptr<Operation>>>,
}

impl Graph<'_> {
    fn computations_equal(&self, left: Ptr<Operation>, right: Ptr<Operation>) -> Result<bool> {
        if left == right {
            return Ok(true);
        }
        if let Some(&equal) = self.equality_cache.borrow().get(&(left, right)) {
            return Ok(equal);
        }
        let equal = self.computations_equal_uncached(left, right)?;
        self.equality_cache
            .borrow_mut()
            .insert((left, right), equal);
        Ok(equal)
    }

    fn computations_equal_uncached(
        &self,
        left: Ptr<Operation>,
        right: Ptr<Operation>,
    ) -> Result<bool> {
        let ctx = self.ctx;
        let mut computations = vec![(left, right)];
        let mut seen_computations = HashSet::new();
        let mut seen_values = HashSet::new();
        while let Some((left, right)) = computations.pop() {
            if left == right || !seen_computations.insert((left, right)) {
                continue;
            }
            if execution_thread(ctx, left, "hlo_execution_thread")?
                .as_deref()
                .unwrap_or("main")
                != execution_thread(ctx, right, "hlo_execution_thread")?
                    .as_deref()
                    .unwrap_or("main")
            {
                return Ok(false);
            }
            let left_signature = self.signatures.get(&left).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped computation signature".into()
                })
            })?;
            let right_signature = self.signatures.get(&right).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped computation signature".into()
                })
            })?;
            if left_signature.0.len() != right_signature.0.len() {
                return Ok(false);
            }
            let left_root = self.roots.get(&left).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped root".into()
                })
            })?;
            let right_root = self.roots.get(&right).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing wrapped root".into()
                })
            })?;
            let mut values = vec![(*left_root, *right_root)];
            while let Some((left, right)) = values.pop() {
                if left == right || !seen_values.insert((left, right)) {
                    continue;
                }
                if !Shape::equal_including_layout(ctx, left.get_type(ctx), right.get_type(ctx)) {
                    return Ok(false);
                }
                if let (DefiningEntity::Block(a), DefiningEntity::Block(b)) =
                    (left.defining_entity(), right.defining_entity())
                {
                    if a.deref(ctx).arguments().position(|v| v == left)
                        != b.deref(ctx).arguments().position(|v| v == right)
                    {
                        return Ok(false);
                    }
                    continue;
                }
                let (Some(a), Some(b)) = (direct(left), direct(right)) else {
                    return Ok(false);
                };
                let (Some(kind), Some(other)) = (opcode(ctx, a), opcode(ctx, b)) else {
                    return Ok(false);
                };
                if kind != other || matches!(kind, Opcode::AfterAll | Opcode::AddDependency) {
                    return Ok(false);
                }
                let (a_data, b_data) = (a.deref(ctx), b.deref(ctx));
                if a_data.get_num_operands() != b_data.get_num_operands()
                    || channel(ctx, a)? != channel(ctx, b)?
                {
                    return Ok(false);
                }
                let a_backend = optional_attr::<StringAttr>(ctx, a, "hlo_backend_config")?;
                let b_backend = optional_attr::<StringAttr>(ctx, b, "hlo_backend_config")?;
                if a_backend.as_ref().map_or("", |s| s.as_str())
                    != b_backend.as_ref().map_or("", |s| s.as_str())
                {
                    return Ok(false);
                }
                let a_accuracy = optional_attr::<ResultAccuracy>(ctx, a, "hlo_result_accuracy")?;
                let b_accuracy = optional_attr::<ResultAccuracy>(ctx, b, "hlo_result_accuracy")?;
                if kind.supports_result_accuracy()
                    && !a_accuracy
                        .as_deref()
                        .unwrap_or(&ResultAccuracy::Default)
                        .equal(b_accuracy.as_deref().unwrap_or(&ResultAccuracy::Default))
                {
                    return Ok(false);
                }
                if matches!(kind, Opcode::Constant | Opcode::CustomCall) {
                    match (
                        optional_attr::<Literal>(ctx, a, "hlo_literal")?,
                        optional_attr::<Literal>(ctx, b, "hlo_literal")?,
                    ) {
                        (Some(a), Some(b)) if a.equal_logical_bits(ctx, &b)? => {}
                        (None, None) => {}
                        _ => return Ok(false),
                    }
                }
                let keys: &[&'static str] = match kind {
                    Opcode::Parameter
                    | Opcode::GetTupleElement
                    | Opcode::Iota
                    | Opcode::Concatenate
                    | Opcode::GetDimensionSize
                    | Opcode::SetDimensionSize => &["hlo_index"],
                    Opcode::Compare => &["hlo_direction", "hlo_order"],
                    Opcode::ReducePrecision => &["hlo_precision"],
                    Opcode::Broadcast
                    | Opcode::Transpose
                    | Opcode::Reverse
                    | Opcode::DynamicReshape
                    | Opcode::Reshape
                    | Opcode::Reduce
                    | Opcode::Sort => &["hlo_dimensions"],
                    Opcode::Slice => &["hlo_geometry"],
                    Opcode::DynamicSlice => &["hlo_slice_sizes"],
                    Opcode::Pad => &["hlo_config"],
                    Opcode::Gather => &[
                        "hlo_dimensions",
                        "hlo_slice_sizes",
                        "hlo_flags",
                        "hlo_dimension_numbers",
                    ],
                    Opcode::Scatter => &["hlo_dimensions", "hlo_flags", "hlo_dimension_numbers"],
                    Opcode::Shuffle => &["hlo_dimensions", "hlo_mode"],
                    Opcode::Dot => &[
                        "hlo_dimensions",
                        "hlo_lhs_precision",
                        "hlo_rhs_precision",
                        "hlo_precision",
                        "hlo_sparsity",
                        "hlo_block_scaling",
                    ],
                    Opcode::RaggedDot | Opcode::ScaledDot => &[
                        "hlo_dimensions",
                        "hlo_precision",
                        "hlo_sparsity",
                        "hlo_block_scaling",
                    ],
                    Opcode::Convolution => &[
                        "hlo_dimensions",
                        "hlo_window",
                        "hlo_feature_groups",
                        "hlo_batch_groups",
                        "hlo_kind",
                        "hlo_precision",
                    ],
                    Opcode::ReduceWindow | Opcode::SelectAndScatter => &["hlo_window"],
                    Opcode::Scan => &[
                        "hlo_dimensions",
                        "hlo_num_carries",
                        "hlo_is_associative",
                        "hlo_is_reverse",
                        "hlo_cub_scan",
                    ],
                    Opcode::Fft => &["hlo_kind", "hlo_lengths"],
                    Opcode::TopK => &["hlo_k", "hlo_options"],
                    Opcode::TriangularSolve
                    | Opcode::Cholesky
                    | Opcode::BatchNormTraining
                    | Opcode::BatchNormInference
                    | Opcode::BatchNormGrad => &["hlo_options"],
                    Opcode::Rng => &["hlo_distribution"],
                    Opcode::RngBitGenerator => &["hlo_algorithm"],
                    Opcode::RngGetAndUpdateState => &["hlo_delta"],
                    Opcode::CopyStart => &["hlo_cross_program_prefetch_index"],
                    Opcode::Infeed => &["hlo_infeed_config"],
                    Opcode::Outfeed => &["hlo_outfeed_config", "hlo_outfeed_shape"],
                    Opcode::Send | Opcode::SendDone | Opcode::Recv | Opcode::RecvDone => {
                        &["hlo_frontend_attributes", "hlo_source_target_pairs"]
                    }
                    Opcode::Domain => &["hlo_domain_entry_sharding", "hlo_domain_exit_sharding"],
                    Opcode::CustomCall => &[
                        "hlo_target",
                        "hlo_api_version",
                        "hlo_schedule",
                        "hlo_aliases",
                        "hlo_operand_layouts",
                        "hlo_window",
                        "hlo_dimensions",
                        "hlo_precision",
                        "hlo_padding_type",
                    ],
                    Opcode::Fusion => &["hlo_fusion_kind", "hlo_aliases"],
                    Opcode::AllGather
                    | Opcode::AllGatherStart
                    | Opcode::AllReduce
                    | Opcode::AllReduceStart
                    | Opcode::AllToAll
                    | Opcode::ReduceScatter
                    | Opcode::RaggedAllToAll
                    | Opcode::CollectiveBroadcast
                    | Opcode::CollectiveReduce
                    | Opcode::CollectivePermute
                    | Opcode::CollectivePermuteStart => &[
                        "hlo_dimensions",
                        "hlo_groups",
                        "hlo_iota_groups",
                        "hlo_mesh_groups",
                        "hlo_pairs",
                        "hlo_dynamic_slice_sizes",
                    ],
                    _ => &[],
                };
                for &key in keys {
                    let key = Identifier::new(key);
                    if a_data.attributes.0.get(&key) != b_data.attributes.0.get(&key) {
                        return Ok(false);
                    }
                }
                let flags: &[&'static str] = match kind {
                    Opcode::CustomCall => &["hlo_has_side_effect", "hlo_constrain_layout"],
                    Opcode::Send | Opcode::SendDone | Opcode::Recv | Opcode::RecvDone => {
                        &["hlo_host_transfer"]
                    }
                    Opcode::AllGather
                    | Opcode::AllGatherStart
                    | Opcode::AllReduce
                    | Opcode::AllReduceStart
                    | Opcode::ReduceScatter
                    | Opcode::CollectiveReduce => &[
                        "hlo_use_global_device_ids",
                        "hlo_constrain_layout",
                        "hlo_has_dynamic_root",
                    ],
                    _ => &[],
                };
                for &key in flags {
                    if flag(ctx, a, key)? != flag(ctx, b, key)? {
                        return Ok(false);
                    }
                }
                let a_callees = self.callees.get(&a).map(Vec::as_slice).unwrap_or_default();
                let b_callees = self.callees.get(&b).map(Vec::as_slice).unwrap_or_default();
                if a_callees.len() != b_callees.len() {
                    return Ok(false);
                }
                computations.extend(a_callees.iter().copied().zip(b_callees.iter().copied()));
                values.extend(a_data.operands().zip(b_data.operands()));
                if a_data.num_regions() != b_data.num_regions() {
                    return Ok(false);
                }
                for (a, b) in a_data.regions().zip(b_data.regions()) {
                    let a = super::super::single_block(ctx, a)?;
                    let b = super::super::single_block(ctx, b)?;
                    let a = a.deref(ctx).get_terminator(ctx).ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Verification {
                            message: "missing region return".into()
                        })
                    })?;
                    let b = b.deref(ctx).get_terminator(ctx).ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Verification {
                            message: "missing region return".into()
                        })
                    })?;
                    require(
                        a.deref(ctx).get_num_operands() == 1
                            && b.deref(ctx).get_num_operands() == 1,
                        "invalid region return",
                    )?;
                    values.push((a.deref(ctx).get_operand(0), b.deref(ctx).get_operand(0)));
                }
            }
        }
        Ok(true)
    }

    fn producer(&self, from: Value) -> Result<Option<Ptr<Operation>>> {
        if let Some(&producer) = self.producer_cache.borrow().get(&from) {
            return Ok(producer);
        }
        let producer = self.producer_uncached(from)?;
        self.producer_cache.borrow_mut().insert(from, producer);
        Ok(producer)
    }

    fn producer_uncached(&self, from: Value) -> Result<Option<Ptr<Operation>>> {
        enum Work {
            Visit(Value, Vec<usize>),
            Exit(Ptr<Operation>),
        }
        let ctx = self.ctx;
        let mut pending = vec![Work::Visit(from, Vec::new())];
        let mut visited = HashSet::new();
        let mut active = HashSet::new();
        while let Some(work) = pending.pop() {
            let (value, mut index) = match work {
                Work::Visit(value, index) => (value, index),
                Work::Exit(op) => {
                    active.remove(&op);
                    continue;
                }
            };
            if !visited.insert((value, index.clone())) {
                continue;
            }
            let Some(op) = direct(value) else {
                continue;
            };
            if !active.insert(op) {
                continue;
            }
            pending.push(Work::Exit(op));
            let kind = opcode(ctx, op);
            if matches!(
                kind,
                Some(
                    Opcode::AsyncStart
                        | Opcode::AsyncUpdate
                        | Opcode::AllGatherStart
                        | Opcode::AllReduceStart
                        | Opcode::CollectivePermuteStart
                )
            ) {
                return Ok(Some(op));
            }
            let operands: Vec<_> = op.deref(ctx).operands().collect();
            match kind {
                Some(Opcode::GetTupleElement) => {
                    index.push(required::<Index>(ctx, op, "hlo_index")?.0 as usize);
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index));
                    }
                }
                Some(Opcode::Tuple) => {
                    if let Some(position) = index.pop() {
                        if let Some(&operand) = operands.get(position) {
                            pending.push(Work::Visit(operand, index));
                        }
                    } else {
                        pending.extend(operands.into_iter().map(|v| Work::Visit(v, index.clone())));
                    }
                }
                Some(Opcode::OptimizationBarrier | Opcode::Copy | Opcode::Domain) => {
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index));
                    }
                }
                Some(Opcode::CustomCall) if intermediary(ctx, op)? => {
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index));
                    }
                }
                Some(Opcode::While) => {
                    if let Some(&operand) = operands.first() {
                        pending.push(Work::Visit(operand, index.clone()));
                    }
                    if let Some(callee) = self.callees.get(&op).and_then(|c| c.first())
                        && let Some(&root) = self.roots.get(callee)
                    {
                        pending.push(Work::Visit(root, index));
                    }
                }
                Some(Opcode::Parameter) => {
                    if let Some(owner) = self.owners.get(&op)
                        && let Some(callers) = self.callers.get(owner)
                        && callers.len() == 1
                        && opcode(ctx, callers[0]) == Some(Opcode::While)
                        && self.callees.get(&callers[0]).and_then(|c| c.first()) == Some(owner)
                    {
                        if let Some(&root) = self.roots.get(owner) {
                            pending.push(Work::Visit(root, index.clone()));
                        }
                        if callers[0].deref(ctx).get_num_operands() > 0 {
                            pending.push(Work::Visit(callers[0].deref(ctx).get_operand(0), index));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }

    fn wrapped(&self, from: Ptr<Operation>) -> Result<Ptr<Operation>> {
        if let Some(&callee) = self.wrapped_cache.borrow().get(&from) {
            return Ok(callee);
        }
        let ctx = self.ctx;
        let mut current = from;
        let mut visited = HashSet::new();
        while visited.insert(current) {
            let cached = self.wrapped_cache.borrow().get(&current).copied();
            if let Some(callee) = cached {
                for op in visited {
                    self.wrapped_cache.borrow_mut().insert(op, callee);
                }
                return Ok(callee);
            }
            require(
                matches!(
                    opcode(ctx, current),
                    Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
                ),
                "cannot resolve a wrapped computation",
            )?;
            if let Some(callees) = self.callees.get(&current)
                && !callees.is_empty()
            {
                require(
                    callees.len() == 1,
                    "async instruction has multiple wrapped computations",
                )?;
                let callee = callees[0];
                for op in visited {
                    self.wrapped_cache.borrow_mut().insert(op, callee);
                }
                return Ok(callee);
            }
            let data = current.deref(ctx);
            require(data.get_num_operands() > 0, "missing async predecessor")?;
            current = self.producer(data.get_operand(0))?.ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "cannot resolve a wrapped computation".into()
                })
            })?;
        }
        Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "cycle in async chain".into()
        }))
    }

    fn chain_start(&self, from: Ptr<Operation>) -> Result<Ptr<Operation>> {
        let mut current = from;
        let mut seen = HashSet::new();
        while seen.insert(current) {
            let cached = self.start_cache.borrow().get(&current).copied();
            if let Some(start) = cached {
                for op in seen {
                    self.start_cache.borrow_mut().insert(op, start);
                }
                return Ok(start);
            }
            if opcode(self.ctx, current) == Some(Opcode::AsyncStart) {
                for op in seen {
                    self.start_cache.borrow_mut().insert(op, current);
                }
                return Ok(current);
            }
            require(
                matches!(
                    opcode(self.ctx, current),
                    Some(Opcode::AsyncUpdate | Opcode::AsyncDone)
                ),
                "async-done has no chain start",
            )?;
            let data = current.deref(self.ctx);
            require(data.get_num_operands() > 0, "missing async predecessor")?;
            current = self.producer(data.get_operand(0))?.ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "operand does not trace to an async producer".into()
                })
            })?;
        }
        Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "async chain contains a cycle".into()
        }))
    }

    fn carried(&self, from: Ptr<Operation>) -> Result<bool> {
        let ctx = self.ctx;
        let mut work = vec![from];
        let mut seen = HashSet::new();
        while let Some(op) = work.pop() {
            if !seen.insert(op) {
                continue;
            }
            let data = op.deref(ctx);
            if data.get_num_results() == 0 {
                continue;
            }
            let result = data.get_result(0);
            if let Some(owner) = self.owners.get(&op)
                && self.roots.get(owner) == Some(&result)
            {
                for &caller in self.callers.get(owner).into_iter().flatten() {
                    if opcode(ctx, caller) == Some(Opcode::While)
                        && self
                            .callees
                            .get(&caller)
                            .and_then(|callees| callees.first())
                            == Some(owner)
                    {
                        return Ok(true);
                    }
                    if matches!(
                        opcode(ctx, caller),
                        Some(Opcode::Call | Opcode::Conditional)
                    ) {
                        work.push(caller);
                    }
                }
            }
            for &user in self.users.get(&result).into_iter().flatten() {
                match opcode(ctx, user) {
                    Some(Opcode::While) => return Ok(true),
                    Some(
                        Opcode::AsyncUpdate
                        | Opcode::Tuple
                        | Opcode::GetTupleElement
                        | Opcode::OptimizationBarrier
                        | Opcode::Domain
                        | Opcode::Call
                        | Opcode::Conditional,
                    ) => work.push(user),
                    _ => {}
                }
            }
        }
        Ok(false)
    }
}

fn execution_thread<'a>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    key: &'static str,
) -> Result<Option<Ref<'a, str>>> {
    if key == "hlo_execution_thread" {
        return Ok(optional_attr::<ExecutionThread>(ctx, op, key)?
            .map(|thread| Ref::map(thread, |thread| thread.0.as_str()))
            .filter(|thread| !thread.is_empty()));
    }
    Ok(optional_attr::<StringAttr>(ctx, op, key)?
        .map(|s| Ref::map(s, StringAttr::as_str))
        .filter(|s| !s.is_empty()))
}

/// Check async chains, single ownership, transfer pairing, and optional deadlocks.
///
/// `all` is the complete `(operation, owning computation)` index; computation
/// symbols and signatures must already be resolved and local shapes verified.
pub(crate) fn verify_graph(
    ctx: &Context,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<Identifier, Ptr<Operation>>,
    signatures: &HashMap<Ptr<Operation>, Signature>,
    options: VerifyOptions,
) -> Result<()> {
    let mut graph = Graph {
        ctx,
        signatures,
        owners: HashMap::new(),
        roots: HashMap::new(),
        users: HashMap::new(),
        callees: HashMap::new(),
        callers: HashMap::new(),
        instructions: HashMap::new(),
        producer_cache: RefCell::new(HashMap::new()),
        wrapped_cache: RefCell::new(HashMap::new()),
        equality_cache: RefCell::new(HashMap::new()),
        start_cache: RefCell::new(HashMap::new()),
    };
    for &(op, owner) in all {
        if let Some(owner) = owner {
            graph.owners.insert(op, owner);
            graph.instructions.entry(owner).or_default().push(op);
            if Operation::is_op::<ReturnOp>(op, ctx)
                && op.deref(ctx).get_parent_op(ctx) == Some(owner)
            {
                require(
                    op.deref(ctx).get_num_operands() == 1,
                    "return requires one operand",
                )?;
                graph.roots.insert(owner, op.deref(ctx).get_operand(0));
            }
        }
        let data = op.deref(ctx);
        let mut unique = HashSet::new();
        for operand in data.operands() {
            if unique.insert(operand) {
                graph.users.entry(operand).or_default().push(op);
            }
        }
        if let Some(names) = optional_attr::<Callees>(ctx, op, "hlo_callees")? {
            let mut callees = Vec::new();
            for name in &names.0 {
                let callee = *computations.get(name).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "callee not found".into()
                    })
                })?;
                callees.push(callee);
                graph.callers.entry(callee).or_default().push(op);
            }
            graph.callees.insert(op, callees);
        }
    }
    let mut wrapped = HashSet::new();
    for &(op, _) in all {
        if matches!(
            opcode(ctx, op),
            Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
        ) {
            wrapped.insert(graph.wrapped(op)?);
        }
    }
    verify_pairs(&graph, all, &wrapped)?;
    verify_channels(&graph, all, &wrapped)?;
    verify_generic(&graph, all, &wrapped, signatures, options)?;
    if options.check_for_collective_deadlocks {
        super::deadlocks::verify(ctx, all)?;
    }
    Ok(())
}

fn verify_pairs(
    graph: &Graph<'_>,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    wrapped: &HashSet<Ptr<Operation>>,
) -> Result<()> {
    let ctx = graph.ctx;
    for &(op, owner) in all {
        let kind = opcode(ctx, op);
        let done = match kind {
            Some(Opcode::CopyStart) => Some(Opcode::CopyDone),
            Some(Opcode::Send) => Some(Opcode::SendDone),
            Some(Opcode::Recv) => Some(Opcode::RecvDone),
            Some(Opcode::AllGatherStart) => Some(Opcode::AllGatherDone),
            Some(Opcode::AllReduceStart) => Some(Opcode::AllReduceDone),
            Some(Opcode::CollectivePermuteStart) => Some(Opcode::CollectivePermuteDone),
            _ => None,
        };
        if let Some(done) = done {
            let value = op.deref(ctx).get_result(0);
            if matches!(kind, Some(Opcode::Send | Opcode::Recv))
                && owner
                    .is_some_and(|o| wrapped.contains(&o) && graph.roots.get(&o) == Some(&value))
            {
                continue;
            }
            let mut users = Vec::new();
            for &user in graph.users.get(&value).into_iter().flatten() {
                if Operation::is_op::<ReturnOp>(user, ctx) {
                    continue;
                }
                if !(opcode(ctx, user) == Some(Opcode::CustomCall)
                    && target(ctx, user)?.is_some_and(|s| s.as_str() == "control_dep"))
                {
                    users.push(user);
                }
            }
            require(
                users.len() == 1,
                "asynchronous start requires exactly one allowed non-control_dep consumer",
            )?;
            require(
                opcode(ctx, users[0]) == Some(done) || intermediary(ctx, users[0])?,
                "asynchronous start has invalid consumer",
            )?;
        }
        let start = match kind {
            Some(Opcode::CopyDone) => Some(Opcode::CopyStart),
            Some(Opcode::SendDone) => Some(Opcode::Send),
            Some(Opcode::RecvDone) => Some(Opcode::Recv),
            Some(Opcode::AllGatherDone) => Some(Opcode::AllGatherStart),
            Some(Opcode::AllReduceDone) => Some(Opcode::AllReduceStart),
            Some(Opcode::CollectivePermuteDone) => Some(Opcode::CollectivePermuteStart),
            _ => None,
        };
        if let Some(start) = start {
            let data = op.deref(ctx);
            require(
                data.get_num_operands() == 1,
                "asynchronous done requires one operand",
            )?;
            let operand = data.get_operand(0);
            let direct_kind = direct(operand).and_then(|op| opcode(ctx, op));
            let permitted = direct_kind == Some(start)
                || matches!(kind, Some(Opcode::SendDone | Opcode::RecvDone))
                    && direct_kind == Some(Opcode::GetTupleElement);
            require(
                permitted
                    || graph
                        .producer(operand)?
                        .is_some_and(|op| opcode(ctx, op) == Some(start)),
                "asynchronous done operand does not trace to its required producer",
            )?;
        }
    }
    Ok(())
}

fn verify_channels(
    graph: &Graph<'_>,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    wrapped: &HashSet<Ptr<Operation>>,
) -> Result<()> {
    let ctx = graph.ctx;
    let mut hosts = HashMap::<i64, Ptr<Operation>>::new();
    for &(op, owner) in all {
        if !matches!(opcode(ctx, op), Some(Opcode::Send | Opcode::Recv)) {
            continue;
        }
        let Some(channel_id) = channel(ctx, op)? else {
            continue;
        };
        let data = op.deref(ctx);
        let result = data.get_result(0);
        let host = flag(ctx, op, "hlo_host_transfer")?;
        if host {
            if let Some(&reference) = hosts.get(&channel_id) {
                let reference_data = reference.deref(ctx);
                require(
                    opcode(ctx, reference) == opcode(ctx, op)
                        && Shape::equal_including_layout(
                            ctx,
                            reference_data.get_type(0),
                            data.get_type(0),
                        )
                        && reference_data.get_num_operands() == data.get_num_operands(),
                    "host transfers sharing a channel must be identical apart from operand identities",
                )?;
                for (a, b) in reference_data
                    .operand_types(ctx)
                    .zip(data.operand_types(ctx))
                {
                    require(
                        Shape::equal_including_layout(ctx, a, b),
                        "host transfer operand shapes differ",
                    )?;
                }
                require(
                    optional_attr::<FrontendAttributes>(ctx, reference, "hlo_frontend_attributes")?
                        .as_deref()
                        == optional_attr::<FrontendAttributes>(ctx, op, "hlo_frontend_attributes")?
                            .as_deref()
                        && optional_attr::<StringAttr>(ctx, reference, "hlo_backend_config")?
                            .as_deref()
                            == optional_attr::<StringAttr>(ctx, op, "hlo_backend_config")?
                                .as_deref()
                        && optional_attr::<ResultAccuracy>(ctx, reference, "hlo_result_accuracy")?
                            .as_deref()
                            == optional_attr::<ResultAccuracy>(ctx, op, "hlo_result_accuracy")?
                                .as_deref(),
                    "host transfer attributes differ",
                )?;
            } else {
                hosts.insert(channel_id, op);
            }
        }
        if owner.is_some_and(|o| wrapped.contains(&o) && graph.roots.get(&o) == Some(&result)) {
            continue;
        }
        let users = graph
            .users
            .get(&result)
            .into_iter()
            .flatten()
            .copied()
            .filter(|&u| !Operation::is_op::<ReturnOp>(u, ctx))
            .collect::<Vec<_>>();
        require(
            users.len() == 1,
            "channel-bearing send/recv requires exactly one user, including control_dep",
        )?;
        if matches!(
            opcode(ctx, users[0]),
            Some(Opcode::SendDone | Opcode::RecvDone)
        ) {
            require(
                channel(ctx, users[0])? == Some(channel_id)
                    && flag(ctx, users[0], "hlo_host_transfer")? == host,
                "start/done channel or host-transfer flag differs",
            )?;
        }
    }
    Ok(())
}

fn verify_generic(
    graph: &Graph<'_>,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    wrapped: &HashSet<Ptr<Operation>>,
    signatures: &HashMap<Ptr<Operation>, Signature>,
    options: VerifyOptions,
) -> Result<()> {
    let ctx = graph.ctx;
    let mut starts = Vec::new();
    let mut updates = Vec::new();
    let mut dones = Vec::new();
    for &(op, _) in all {
        let kind = opcode(ctx, op);
        let data = op.deref(ctx);
        for (position, operand) in data.operands().enumerate() {
            if direct(operand).is_some_and(|producer| {
                matches!(
                    opcode(ctx, producer),
                    Some(Opcode::AsyncStart | Opcode::AsyncUpdate)
                )
            }) {
                let allowed = if matches!(kind, Some(Opcode::AsyncUpdate | Opcode::AsyncDone)) {
                    position == 0
                } else {
                    matches!(
                        kind,
                        Some(Opcode::Call | Opcode::Conditional | Opcode::Domain)
                    ) || intermediary(ctx, op)?
                        || Operation::is_op::<ReturnOp>(op, ctx)
                };
                require(
                    allowed,
                    "async producers are only allowed as chaining operand zero or through approved dataflow",
                )?;
            }
        }
        let Some(kind @ (Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)) = kind
        else {
            continue;
        };
        match kind {
            Opcode::AsyncStart => starts.push(op),
            Opcode::AsyncUpdate => updates.push(op),
            _ => dones.push(op),
        }
        let callee = graph.wrapped(op)?;
        let signature = signatures.get(&callee).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "missing wrapped computation signature".into()
            })
        })?;
        verify_callees(
            ctx,
            op,
            kind,
            data.get_type(0),
            &data.operand_types(ctx).collect::<Vec<_>>(),
            &[signature],
            options,
        )?;
        verify_async_aliases(ctx, op, kind, signature, options)?;
        let current = graph.chain_start(op)?;
        let thread = execution_thread(ctx, current, "hlo_async_execution_thread")?;
        let thread = thread.as_deref().unwrap_or("main");
        require(
            thread
                == execution_thread(ctx, callee, "hlo_execution_thread")?
                    .as_deref()
                    .unwrap_or("main"),
            "async execution thread differs from wrapped computation",
        )?;
        let explicit = optional_attr::<StringAttr>(ctx, op, "hlo_async_execution_thread")?;
        require(
            explicit.is_none_or(|s| s.as_str().is_empty() || s.as_str() == thread),
            "async execution thread differs from chain",
        )?;
        if kind != Opcode::AsyncStart {
            let producer = graph.producer(data.get_operand(0))?.ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing async predecessor".into()
                })
            })?;
            require(
                matches!(
                    opcode(ctx, producer),
                    Some(Opcode::AsyncStart | Opcode::AsyncUpdate)
                ) && graph.computations_equal(graph.wrapped(producer)?, callee)?,
                "predecessor has a different wrapped computation",
            )?;
            if kind == Opcode::AsyncDone {
                let (bound, _) = infer::async_context(ctx, producer.deref(ctx).get_type(0))?;
                require(
                    bound.len() == signature.0.len()
                        && bound
                            .iter()
                            .zip(&signature.0)
                            .all(|(&a, &b)| same(ctx, a, b)),
                    "async-done predecessor parameters are not fully bound",
                )?;
            }
        }
    }
    for &callee in wrapped {
        let callers = graph
            .callers
            .get(&callee)
            .map(Vec::as_slice)
            .unwrap_or_default();
        require(
            callers.iter().all(|&op| {
                matches!(
                    opcode(ctx, op),
                    Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
                )
            }),
            "async wrapped computation is called by a non-async instruction",
        )?;
        let starts = callers
            .iter()
            .copied()
            .filter(|&op| opcode(ctx, op) == Some(Opcode::AsyncStart))
            .collect::<Vec<_>>();
        let grouped = starts.len() == 1
            && optional_attr::<FrontendAttributes>(ctx, starts[0], "hlo_frontend_attributes")?
                .is_some_and(|a| a.get("_collectives_group").is_some());
        if execution_thread(ctx, callee, "hlo_execution_thread")?
            .as_deref()
            .unwrap_or("main")
            == "main"
            && !grouped
        {
            for &op in graph.instructions.get(&callee).into_iter().flatten() {
                let data = op.deref(ctx);
                if data.get_num_results() == 0 {
                    continue;
                }
                require(
                    graph.roots.get(&callee) == Some(&data.get_result(0))
                        || matches!(
                            opcode(ctx, op),
                            Some(Opcode::Parameter | Opcode::GetTupleElement)
                        ),
                    "main-thread async computation requires only root, parameter and get-tuple-element instructions unless marked as a collectives group",
                )?;
            }
        }
    }
    let mut matched = HashSet::new();
    let mut visited_updates = HashSet::new();
    for done in dones {
        let mut producer = graph.producer(done.deref(ctx).get_operand(0))?;
        while let Some(op) = producer {
            if opcode(ctx, op) != Some(Opcode::AsyncUpdate) {
                break;
            }
            require(
                visited_updates.insert(op),
                "async-update belongs to multiple done chains or a cycle",
            )?;
            producer = graph.producer(op.deref(ctx).get_operand(0))?;
        }
        let start = producer
            .filter(|&p| opcode(ctx, p) == Some(Opcode::AsyncStart))
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "async-done does not trace back to async-start".into()
                })
            })?;
        require(
            matched.insert(start),
            "async-start has multiple matching async-dones",
        )?;
    }
    for start in starts {
        require(
            matched.contains(&start) || graph.carried(start)?,
            "async-start has no matching async-done",
        )?;
    }
    for update in updates {
        require(
            visited_updates.contains(&update) || graph.carried(update)?,
            "orphan async-update",
        )?;
    }
    Ok(())
}

fn alias_shape_equal(
    ctx: &Context,
    a: TypeHandle,
    b: TypeHandle,
    options: VerifyOptions,
) -> Result<bool> {
    let mut work = vec![(a, b)];
    while let Some((a, b)) = work.pop() {
        if Shape::is_tuple(ctx, a) || Shape::is_tuple(ctx, b) {
            if !Shape::is_tuple(ctx, a) || !Shape::is_tuple(ctx, b) {
                return Ok(false);
            }
            let a = Shape::children(ctx, a);
            let b = Shape::children(ctx, b);
            if a.len() != b.len() {
                return Ok(false);
            }
            work.extend(a.iter().copied().zip(b.iter().copied()));
            continue;
        }
        let a = if a.deref(ctx).is::<BufferType>() {
            Shape::array_view(ctx, a)?
        } else {
            a
        };
        let b = if b.deref(ctx).is::<BufferType>() {
            Shape::array_view(ctx, b)?
        } else {
            b
        };
        if options.layout_sensitive {
            if !options.same_shapes(ctx, a, b) {
                return Ok(false);
            }
        } else if Shape::is_array(ctx, a) && Shape::is_array(ctx, b) {
            if *Shape::dimensions(ctx, a)? != *Shape::dimensions(ctx, b)?
                || Shape::element_type(ctx, a)? != Shape::element_type(ctx, b)?
            {
                return Ok(false);
            }
        } else if !Shape::equal_ignoring_layout(ctx, a, b) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn verify_async_aliases(
    ctx: &Context,
    op: Ptr<Operation>,
    kind: Opcode,
    signature: &Signature,
    options: VerifyOptions,
) -> Result<()> {
    if kind == Opcode::AsyncDone {
        return Ok(());
    }
    let Some(aliases) = optional_attr::<OutputOperandAliases>(ctx, op, "hlo_aliases")? else {
        return Ok(());
    };
    let data = op.deref(ctx);
    let alias_output = if kind == Opcode::AsyncStart {
        let mut children = vec![Shape::tuple(ctx, signature.0.clone())?, signature.1];
        children.extend(
            Shape::children(ctx, data.get_type(0))
                .iter()
                .copied()
                .skip(2),
        );
        Shape::tuple(ctx, children)?
    } else {
        data.get_type(0)
    };
    for alias in &aliases.0 {
        let operand = if kind == Opcode::AsyncUpdate {
            require(
                alias.operand_index == 0
                    && alias.operand_shape_index.0.first() == Some(&2)
                    && alias.output_shape_index.0.first() == Some(&2),
                "async-update aliases may only connect context child two",
            )?;
            require(
                data.get_num_operands() > 0,
                "async-update requires a predecessor",
            )?;
            data.get_operand(0).get_type(ctx)
        } else {
            require(
                !alias.output_shape_index.0.is_empty(),
                "async-start alias output index must not be empty",
            )?;
            *usize::try_from(alias.operand_index)
                .ok()
                .and_then(|i| signature.0.get(i))
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "async-start alias parameter number is out of range".into()
                    })
                })?
        };
        let operand = super::indexed(ctx, operand, &alias.operand_shape_index.0)?;
        let output = super::indexed(ctx, alias_output, &alias.output_shape_index.0)?;
        require(
            alias_shape_equal(ctx, operand, output, options)?,
            "alias operand and output shapes differ",
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::control::{GetTupleElementOp, ParameterOp},
        types::{ElementType, TensorType},
    };
    use pliron::op::Op;

    #[test]
    fn producer_tracing_terminates_when_a_projection_cycle_grows_shape_indices() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let f = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
        let tuple = Shape::tuple(&ctx, vec![f]).unwrap();
        let parameter = ParameterOp::new(&mut ctx, tuple, vec![], Index(0));
        let value = parameter.result(&ctx);
        let projection = GetTupleElementOp::new(&mut ctx, f, vec![value], Index(0));
        let value = projection.result(&ctx);
        Operation::replace_operand(projection.get_operation(), &ctx, 0, value);
        let signatures = HashMap::new();
        let graph = Graph {
            ctx: &ctx,
            signatures: &signatures,
            owners: HashMap::new(),
            roots: HashMap::new(),
            users: HashMap::new(),
            callees: HashMap::new(),
            callers: HashMap::new(),
            instructions: HashMap::new(),
            producer_cache: RefCell::new(HashMap::new()),
            wrapped_cache: RefCell::new(HashMap::new()),
            equality_cache: RefCell::new(HashMap::new()),
            start_cache: RefCell::new(HashMap::new()),
        };
        assert!(graph.producer_uncached(value).unwrap().is_none());
    }
}
