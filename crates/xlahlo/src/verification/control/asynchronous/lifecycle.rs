use super::*;

pub(super) fn execution_thread<'a>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    key: &'static str,
) -> Result<Option<Ref<'a, str>>> {
    if key == crate::attributes::keys::EXECUTION_THREAD {
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
    root: Ptr<Operation>,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
    signatures: &HashMap<Ptr<Operation>, Signature>,
    options: VerifyOptions,
) -> Result<()> {
    let mut graph = build_graph(ctx, all, computations, signatures)?;
    graph.payloads =
        optional_attr::<MetadataPayloads>(ctx, root, crate::attributes::keys::METADATA_PAYLOADS)?
            .map_or_else(MetadataPayloads::default, |payloads| payloads.clone());
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
        super::super::deadlocks::verify(ctx, all)?;
    }
    Ok(())
}

/// Resolve generic async callees through the same dataflow used by verification.
///
/// This only resolves the snapshot; it does not verify signatures or pairing.
pub(crate) fn wrapped_computations(
    ctx: &Context,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
) -> Result<HashMap<Ptr<Operation>, Ptr<Operation>>> {
    let signatures = HashMap::new();
    let graph = build_graph(ctx, all, computations, &signatures)?;
    let mut wrapped = HashMap::new();
    for &(op, _) in all {
        if matches!(
            opcode(ctx, op),
            Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
        ) {
            wrapped.insert(op, graph.wrapped(op)?);
        }
    }
    Ok(wrapped)
}

pub(crate) fn chain_starts(
    ctx: &Context,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
) -> Result<HashMap<Ptr<Operation>, Ptr<Operation>>> {
    let signatures = HashMap::new();
    let graph = build_graph(ctx, all, computations, &signatures)?;
    let mut starts = HashMap::new();
    for &(op, _) in all {
        if matches!(
            opcode(ctx, op),
            Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
        ) {
            starts.insert(op, graph.chain_start(op)?);
        }
    }
    Ok(starts)
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
                        && shape::equal_including_layout(
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
                        shape::equal_including_layout(ctx, a, b),
                        "host transfer operand shapes differ",
                    )?;
                }
                require(
                    same_frontend_attributes(ctx, reference, op)?
                        && graph.backend_config(reference)? == graph.backend_config(op)?
                        && optional_attr::<ResultAccuracy>(
                            ctx,
                            reference,
                            crate::attributes::keys::RESULT_ACCURACY,
                        )?
                        .as_deref()
                        .unwrap_or(&ResultAccuracy::Default)
                        .equal(
                            optional_attr::<ResultAccuracy>(
                                ctx,
                                op,
                                crate::attributes::keys::RESULT_ACCURACY,
                            )?
                            .as_deref()
                            .unwrap_or(&ResultAccuracy::Default),
                        ),
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
                == execution_thread(ctx, callee, crate::attributes::keys::EXECUTION_THREAD)?
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
            && optional_attr::<FrontendAttributes>(
                ctx,
                starts[0],
                crate::attributes::keys::FRONTEND_ATTRIBUTES,
            )?
            .is_some_and(|a| a.get("_collectives_group").is_some());
        if execution_thread(ctx, callee, crate::attributes::keys::EXECUTION_THREAD)?
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
        if shape::is_tuple(ctx, a) || shape::is_tuple(ctx, b) {
            if !shape::is_tuple(ctx, a) || !shape::is_tuple(ctx, b) {
                return Ok(false);
            }
            let a = shape::children(ctx, a);
            let b = shape::children(ctx, b);
            if a.len() != b.len() {
                return Ok(false);
            }
            work.extend(a.iter().copied().zip(b.iter().copied()));
            continue;
        }
        let a = if a.deref(ctx).is::<BufferType>() {
            shape::array_view(ctx, a)?
        } else {
            a
        };
        let b = if b.deref(ctx).is::<BufferType>() {
            shape::array_view(ctx, b)?
        } else {
            b
        };
        if options.layout_sensitive {
            if !options.same_shapes(ctx, a, b) {
                return Ok(false);
            }
        } else if shape::is_array(ctx, a) && shape::is_array(ctx, b) {
            if *shape::dimensions(ctx, a)? != *shape::dimensions(ctx, b)?
                || shape::element_type(ctx, a)? != shape::element_type(ctx, b)?
            {
                return Ok(false);
            }
        } else if !shape::equal_ignoring_layout(ctx, a, b) {
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
    let Some(aliases) =
        optional_attr::<OutputOperandAliases>(ctx, op, crate::attributes::keys::ALIASES)?
    else {
        return Ok(());
    };
    let data = op.deref(ctx);
    let alias_output = if kind == Opcode::AsyncStart {
        let mut children = vec![shape::tuple(ctx, signature.0.clone())?, signature.1];
        children.extend(
            shape::children(ctx, data.get_type(0))
                .iter()
                .copied()
                .skip(2),
        );
        shape::tuple(ctx, children)?
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
        let operand = super::super::indexed(ctx, operand, &alias.operand_shape_index.0)?;
        let output = super::super::indexed(ctx, alias_output, &alias.output_shape_index.0)?;
        require(
            alias_shape_equal(ctx, operand, output, options)?,
            "alias operand and output shapes differ",
        )?;
    }
    Ok(())
}
