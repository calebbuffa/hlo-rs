use super::*;

/// Clone without publishing or editing the source. Errors after allocation erase
/// the entire staging computation, including its operand uses. This routine does
/// not supply a default verifier policy or pretend detached named callees resolve.
pub(crate) fn clone_with_replacements(
    ctx: &mut Context,
    source: &ComputationOp,
    name: crate::ir::ComputationName,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
) -> Result<ComputationClone> {
    let module = owning_module(ctx, source.get_operation());
    let mut context = ComputationCloneContext::new(module, "clone".into());
    clone_in_context(ctx, source, name, replacements, options, &mut context)
}

pub(crate) fn clone_in_context(
    ctx: &mut Context,
    source: &ComputationOp,
    name: crate::ir::ComputationName,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<ComputationClone> {
    let mut transaction = context.clone();
    let result = clone_impl(ctx, source, name, replacements, options, &mut transaction);
    if result.is_ok() {
        *context = transaction;
    }
    result
}

pub(super) fn clone_impl(
    ctx: &mut Context,
    source: &ComputationOp,
    name: crate::ir::ComputationName,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<ComputationClone> {
    require(
        context.in_progress.insert(source.get_operation()),
        "cycle in cloned computation dependencies",
    )?;
    let source_module = source
        .get_operation()
        .deref(ctx)
        .get_parent_block()
        .and_then(|block| block.deref(ctx).get_parent_op(ctx));
    if let Some(module) = source_module {
        require(
            Operation::is_op::<ModuleOp>(module, ctx),
            "source owner is not a module",
        )?;
    }
    if let Some(options) = options {
        crate::verification::verify_with_options(
            ctx,
            source_module.unwrap_or(source.get_operation()),
            options,
        )?;
    }
    let module = context.destination;
    if let Some(module) = module {
        require(
            Operation::is_op::<ModuleOp>(module, ctx),
            "clone destination is not a module",
        )?;
        if Some(module) != source_module
            && let Some(options) = options
        {
            crate::verification::verify_with_options(ctx, module, options)?;
        }
    }
    let mut plan = plan(ctx, source, replacements)?;
    let source_name = source
        .get_attr_hlo_computation_name(ctx)
        .ok_or_else(|| error("clone source has no computation name"))?
        .as_ref()
        .to_string();
    let mut source_instruction_names = HashMap::new();
    for operation in nested_operations(ctx, &plan.postorder) {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            source_instruction_names.insert(operation, name.0.clone());
        }
    }
    // HloModule uniquifies instruction names when adding clones. Native named
    // controls need those final names before the detached clone can be verified.
    uniquify_names(ctx, module, &mut plan, &mut context.instruction_names)?;
    let block_attributes = body(ctx, source)?.deref(ctx).attributes.clone();
    let (embedded_computations, embedded_schedules, async_callees) =
        clone_called_computations(ctx, source_module, &plan, options, context)?;
    let mut computation_names = context.computation_names.clone();
    if let Some(module) = module {
        for operation in nested_operations(ctx, &[module]) {
            if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx)
                && let Some(name) = computation.get_attr_hlo_computation_name(ctx)
            {
                computation_names.insert(name.as_ref().to_string());
            }
        }
    }
    for computation in &embedded_computations {
        if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
            computation_names.insert(name.as_ref().to_string());
        }
    }
    let name =
        crate::ir::ComputationName::new(unique_symbol(&mut computation_names, name.as_ref()));
    context.computation_names = computation_names;
    let signature: TypeHandle =
        FunctionType::get(ctx, plan.parameters.clone(), vec![plan.result]).into();
    let computation = ComputationOp::new(ctx, name.clone(), signature);
    let attributes = source.get_operation().deref(ctx).attributes.clone();
    let location = source.get_operation().deref(ctx).loc();
    {
        let mut data = computation.get_operation().deref_mut(ctx);
        data.attributes = attributes;
        data.attributes
            .0
            .remove(&Identifier::new("hlo_instruction_order"));
        data.set_loc(location);
    }
    computation.set_attr_hlo_computation_name(ctx, name);
    computation.set_attr_hlo_computation_signature(ctx, TypeAttr::new(signature));
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(computation.body(ctx), ctx);
    block.deref_mut(ctx).attributes = block_attributes;
    let outcome = build(
        ctx,
        &computation,
        block,
        replacements,
        &plan,
        &async_callees,
        options,
    )
    .and_then(|mapping| {
        if let Some(options) = options {
            verify_in_source_context(
                ctx,
                module,
                &computation,
                &embedded_computations,
                &embedded_schedules,
                context,
                options,
            )?;
        }
        Ok(mapping)
    });
    match outcome {
        Ok(mapping) => {
            context.in_progress.remove(&source.get_operation());
            context
                .computations
                .insert(source.get_operation(), computation.get_operation());
            context
                .computations_by_name
                .insert(source_name, computation.get_operation());
            for operation in nested_operations(ctx, &plan.postorder) {
                // Pinned extra_parameters use Clone() without the clone context.
                if replacements.extra_parameters.contains(&operation) {
                    continue;
                }
                if let Some(cloned) = mapping.lookup_op(operation) {
                    context.instructions.insert(operation, cloned);
                    if let Some(name) = source_instruction_names.get(&operation) {
                        context.instructions_by_name.insert(name.clone(), cloned);
                    }
                }
            }
            context.schedules.extend(embedded_schedules.iter().cloned());
            Ok(ComputationClone {
                computation,
                embedded_computations,
                embedded_schedules,
                mapping,
            })
        }
        Err(error) => {
            Operation::erase(computation.get_operation(), ctx);
            for embedded in embedded_computations {
                Operation::erase(embedded.get_operation(), ctx);
            }
            Err(error)
        }
    }
}

pub(super) fn nested_operations(ctx: &Context, roots: &[Ptr<Operation>]) -> Vec<Ptr<Operation>> {
    let mut result = Vec::new();
    let mut pending: Vec<_> = roots.iter().rev().copied().collect();
    while let Some(operation) = pending.pop() {
        result.push(operation);
        let mut children = Vec::new();
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                children.extend(block.deref(ctx).iter(ctx));
            }
        }
        pending.extend(children.into_iter().rev());
    }
    result
}

pub(super) type AsyncCallees = HashMap<Ptr<Operation>, Callees>;

pub(super) fn clone_called_computations(
    ctx: &mut Context,
    module: Option<Ptr<Operation>>,
    plan: &Plan,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<(Vec<ComputationOp>, Vec<InstructionSequence>, AsyncCallees)> {
    let mut embedded = Vec::<ComputationOp>::new();
    let mut schedules = Vec::new();
    let result = (|| {
        let mut remapped = HashMap::new();
        for operation in nested_operations(ctx, &plan.postorder) {
            if Operation::is_op::<ReturnOp>(operation, ctx) {
                continue;
            }
            let kind = opcode(ctx, operation)?;
            let callees = optional_attr::<Callees>(ctx, operation, "hlo_callees")?
                .map(|callees| callees.0.clone())
                .unwrap_or_default();
            if kind == Opcode::AsyncStart {
                require(
                    callees.len() == 1,
                    "async start requires one wrapped computation",
                )?;
            }
            if callees.is_empty() {
                continue;
            }
            let owner = owning_module(ctx, operation)
                .or(module)
                .ok_or_else(|| error("named clone dependency needs a source module"))?;
            let mut available = HashMap::new();
            for candidate in nested_operations(ctx, &[owner]) {
                if let Some(computation) = Operation::get_op::<ComputationOp>(candidate, ctx) {
                    let name = computation
                        .get_attr_hlo_computation_name(ctx)
                        .ok_or_else(|| error("missing source callee name"))?;
                    available.insert(name.clone(), candidate);
                }
            }
            let mut names = Vec::new();
            for callee_name in callees {
                let original = *available
                    .get(&callee_name)
                    .ok_or_else(|| error("unresolved clone dependency computation"))?;
                let is_async = matches!(
                    kind,
                    Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone,
                );
                if Some(owner) == context.destination && !is_async {
                    names.push(callee_name);
                    continue;
                }
                let cloned = if let Some(cloned) = context.computation(original) {
                    cloned
                } else {
                    let callee = Operation::get_op::<ComputationOp>(original, ctx)
                        .ok_or_else(|| error("callee is not a native computation"))?;
                    let mut names = HashSet::new();
                    let name = crate::ir::ComputationName::new(unique_symbol(
                        &mut names,
                        &format!("{callee_name}.{}", context.suffix),
                    ));
                    let output = clone_impl(
                        ctx,
                        &callee,
                        name,
                        &Replacements::default(),
                        options,
                        context,
                    )?;
                    let cloned = output.computation.get_operation();
                    schedules.extend(output.embedded_schedules);
                    embedded.extend(output.embedded_computations);
                    embedded.push(output.computation);
                    if kind == Opcode::AsyncStart
                        && optional_attr::<Schedule>(ctx, owner, "hlo_schedule")?.is_some_and(
                            |schedule| {
                                schedule
                                    .0
                                    .iter()
                                    .any(|sequence| sequence.computation == callee_name)
                            },
                        )
                    {
                        let computation = embedded
                            .last()
                            .ok_or_else(|| error("missing async wrapped clone"))?;
                        let order =
                            topological_order(ctx, &ordered_instructions(ctx, computation)?)?;
                        schedules.push(sequence(ctx, computation, &order)?);
                    }
                    cloned
                };
                let computation = Operation::get_op::<ComputationOp>(cloned, ctx)
                    .ok_or_else(|| error("context mapping is not a computation"))?;
                names.push(
                    computation
                        .get_attr_hlo_computation_name(ctx)
                        .ok_or_else(|| error("cloned callee has no name"))?
                        .clone(),
                );
            }
            remapped.insert(operation, Callees(names));
        }
        Ok(remapped)
    })();
    match result {
        Ok(callees) => Ok((embedded, schedules, callees)),
        Err(error) => {
            for computation in embedded {
                Operation::erase(computation.get_operation(), ctx);
            }
            Err(error)
        }
    }
}
