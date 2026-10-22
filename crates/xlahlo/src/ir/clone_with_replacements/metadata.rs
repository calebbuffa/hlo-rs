use super::*;

pub(super) fn uniquify_names(
    ctx: &Context,
    module: Option<Ptr<Operation>>,
    plan: &mut Plan,
    reserved: &mut HashSet<String>,
) -> Result<()> {
    let mut used = reserved.clone();
    let mut pending: Vec<_> = module.into_iter().collect();
    while let Some(operation) = pending.pop() {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            used.insert(name.0.clone());
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                if let Some(parameters) = block
                    .deref(ctx)
                    .attributes
                    .get::<FusionParameters>(&Identifier::new("hlo_fusion_parameters"))
                {
                    used.extend(
                        parameters
                            .0
                            .iter()
                            .filter_map(|parameter| parameter.name.clone()),
                    );
                }
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    // Iteration order must not decide names when a template already has a
    // .clone suffix. Match the deterministic instruction traversal.
    for operation in nested_operations(ctx, &plan.postorder) {
        if let std::collections::hash_map::Entry::Vacant(entry) = plan.names.entry(operation)
            && let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
        {
            require(!name.0.is_empty(), "empty nested instruction name")?;
            entry.insert(name.0.clone());
        }
        if let Some(name) = plan.names.get_mut(&operation) {
            *name = unique_name(&mut used, name);
        } else if !Operation::is_op::<ReturnOp>(operation, ctx) {
            let name = unique_name(&mut used, &format!("{}.clone", opcode(ctx, operation)?));
            plan.names.insert(operation, name);
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                if Operation::is_op::<crate::ops::control::FusionOp>(operation, ctx) {
                    let parameters = crate::ir::fusion_parameters(ctx, block)?;
                    for (index, parameter) in parameters.0.iter().enumerate() {
                        let default = format!("parameter_{index}.clone");
                        let name = parameter.name.as_deref().unwrap_or(&default);
                        plan.parameter_names
                            .insert((block, index), unique_name(&mut used, name));
                    }
                }
            }
        }
    }
    if plan.tuple_root {
        plan.tuple_name = Some(unique_name(&mut used, "tuple.clone"));
    }
    *reserved = used;
    Ok(())
}

pub(super) fn unique_name(used: &mut HashSet<String>, old: &str) -> String {
    let mut candidate = old.to_owned();
    let mut suffix = 0;
    while !used.insert(candidate.clone()) {
        candidate = if suffix == 0 {
            format!("{old}.clone")
        } else {
            format!("{old}.clone.{suffix}")
        };
        suffix += 1;
    }
    candidate
}

pub(super) fn unique_symbol(used: &mut HashSet<String>, requested: &str) -> String {
    let mut base: String = requested
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if !base.starts_with(|character: char| character.is_ascii_alphabetic() || character == '_') {
        base.insert(0, '_');
    }
    let mut candidate = base.clone();
    let mut index = 0;
    while !used.insert(candidate.clone()) {
        candidate = format!("{base}_clone_{index}");
        index += 1;
    }
    candidate
}

pub(super) fn normalize_nested_metadata(
    ctx: &Context,
    template: Ptr<Operation>,
    mapping: &IrMapping,
    plan: &Plan,
) -> Result<()> {
    for original in nested_operations(ctx, &[template]) {
        let cloned = mapping
            .lookup_op(original)
            .ok_or_else(|| error("nested operation has no clone mapping"))?;
        optional_attr::<OriginalValue>(ctx, original, "hlo_original_value")?;
        {
            let mut data = cloned.deref_mut(ctx);
            if let Some(name) = plan.names.get(&original) {
                data.attributes
                    .set(Identifier::new("hlo_name"), InstructionName(name.clone()));
            }
            if !Operation::is_op::<ReturnOp>(original, ctx)
                && !opcode(ctx, original)?.supports_result_accuracy()
                && data
                    .attributes
                    .get::<ResultAccuracy>(&Identifier::new("hlo_result_accuracy"))
                    .is_some()
            {
                data.attributes.set(
                    Identifier::new("hlo_result_accuracy"),
                    ResultAccuracy::Default,
                );
            }
        }
        for region in original.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                let cloned_block = mapping
                    .lookup_block(block)
                    .ok_or_else(|| error("nested block has no clone mapping"))?;
                let mut renamed = HashMap::new();
                let mut ambiguous = HashSet::new();
                for child in block.deref(ctx).iter(ctx) {
                    if let Some(name) = optional_attr::<InstructionName>(ctx, child, "hlo_name")?
                        && renamed
                            .insert(name.0.clone(), plan.names[&child].clone())
                            .is_some()
                    {
                        ambiguous.insert(name.0.clone());
                    }
                }
                let parameters = if Operation::is_op::<crate::ops::control::FusionOp>(original, ctx)
                {
                    Some(crate::ir::fusion_parameters(ctx, block)?)
                } else {
                    None
                };
                if let Some(parameters) = &parameters {
                    for (index, parameter) in parameters.0.iter().enumerate() {
                        if let Some(name) = &parameter.name {
                            let mapped =
                                plan.parameter_names.get(&(block, index)).ok_or_else(|| {
                                    error("fusion parameter lacks a planned clone name")
                                })?;
                            if renamed.insert(name.clone(), mapped.clone()).is_some() {
                                ambiguous.insert(name.clone());
                            }
                        }
                    }
                }
                let remap_names = |names: &[String]| -> Result<Vec<String>> {
                    names
                        .iter()
                        .map(|name| {
                            require(!ambiguous.contains(name), "ambiguous nested control name")?;
                            renamed
                                .get(name)
                                .cloned()
                                .ok_or_else(|| error("unresolved nested control predecessor"))
                        })
                        .collect()
                };
                if let Some(mut parameters) = parameters {
                    for (index, parameter) in parameters.0.iter_mut().enumerate() {
                        parameter.name = plan.parameter_names.get(&(block, index)).cloned();
                        parameter.control_predecessors =
                            remap_names(&parameter.control_predecessors)?;
                    }
                    cloned_block
                        .deref_mut(ctx)
                        .attributes
                        .set(Identifier::new("hlo_fusion_parameters"), parameters);
                }
                for child in block.deref(ctx).iter(ctx) {
                    if let Some(controls) = optional_attr::<ControlPredecessors>(
                        ctx,
                        child,
                        "hlo_control_predecessors",
                    )? {
                        let cloned_child = mapping
                            .lookup_op(child)
                            .ok_or_else(|| error("control target lacks a clone mapping"))?;
                        cloned_child.deref_mut(ctx).attributes.set(
                            Identifier::new("hlo_control_predecessors"),
                            ControlPredecessors(remap_names(&controls.0)?),
                        );
                    }
                }
            }
        }
    }
    for original in nested_operations(ctx, &[template]) {
        if Operation::is_op::<crate::ops::control::FusionOp>(original, ctx) {
            let cloned = mapping
                .lookup_op(original)
                .ok_or_else(|| error("fusion has no clone mapping"))?;
            for region in cloned.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    crate::ir::verify_fusion_parameter_structure(ctx, block)?;
                }
            }
        }
    }
    Ok(())
}

/// Reuse the authoritative module verifier in an isolated copy of the source
/// context. The real module is never used as a temporary publication surface.
pub(super) fn verify_in_source_context(
    ctx: &mut Context,
    module: Option<Ptr<Operation>>,
    computation: &ComputationOp,
    embedded: &[ComputationOp],
    embedded_schedules: &[InstructionSequence],
    context: &ComputationCloneContext,
    options: VerifyOptions,
) -> Result<()> {
    let Some(module) = module else {
        require(
            embedded.is_empty(),
            "embedded computations require a module context",
        )?;
        return crate::verification::verify_with_options(ctx, computation.get_operation(), options);
    };
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let staging = clone_operation(module, ctx, &mut rewriter, &mut IrMapping::new());
    let block = staging
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block();
    let Some(block) = block else {
        Operation::erase(staging, ctx);
        return Err(error("staging module has no block"));
    };
    let mut present = HashSet::new();
    for operation in nested_operations(ctx, &[staging]) {
        if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx)
            && let Some(name) = computation.get_attr_hlo_computation_name(ctx)
        {
            present.insert(name.clone());
        }
    }
    computation.get_operation().insert_at_back(block, ctx);
    if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
        present.insert(name.clone());
    }
    for computation in embedded {
        computation.get_operation().insert_at_back(block, ctx);
        if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
            present.insert(name.clone());
        }
    }
    // Earlier context outputs may remain detached. Copy rather than temporarily
    // relink them: only this transaction's new results are moved into staging.
    let result = (|| {
        for &operation in context.computations.values() {
            let computation = Operation::get_op::<ComputationOp>(operation, ctx)
                .ok_or_else(|| error("cached clone is not a native computation"))?;
            let name = computation
                .get_attr_hlo_computation_name(ctx)
                .ok_or_else(|| error("cached clone computation lacks a name"))?
                .clone();
            if present.insert(name) {
                let cloned = clone_operation(operation, ctx, &mut rewriter, &mut IrMapping::new());
                cloned.insert_at_back(block, ctx);
            }
        }
        if !embedded_schedules.is_empty() || !context.schedules.is_empty() {
            let mut schedule = optional_attr::<Schedule>(ctx, staging, "hlo_schedule")?
                .map(|schedule| schedule.clone())
                .unwrap_or_default();
            let mut names: HashSet<_> = schedule
                .0
                .iter()
                .map(|sequence| sequence.computation.clone())
                .collect();
            for sequence in context.schedules.iter().chain(embedded_schedules) {
                if names.insert(sequence.computation.clone()) {
                    schedule.0.push(sequence.clone());
                } else {
                    require(
                        schedule.0.iter().any(|existing| existing == sequence),
                        "cached clone schedule conflicts with destination schedule",
                    )?;
                }
            }
            staging
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_schedule"), schedule);
        }
        crate::verification::verify_with_options(ctx, staging, options)
    })();
    computation.get_operation().unlink(ctx);
    for computation in embedded {
        computation.get_operation().unlink(ctx);
    }
    Operation::erase(staging, ctx);
    result
}
