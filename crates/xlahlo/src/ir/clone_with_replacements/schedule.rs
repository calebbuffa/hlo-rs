use super::*;

pub(super) fn ordered_instructions(
    ctx: &Context,
    computation: &ComputationOp,
) -> Result<Vec<Ptr<Operation>>> {
    Ok(body(ctx, computation)?
        .deref(ctx)
        .iter(ctx)
        .filter(|&operation| !Operation::is_op::<ReturnOp>(operation, ctx))
        .collect())
}

pub(super) fn topological_order(
    ctx: &Context,
    preferred: &[Ptr<Operation>],
) -> Result<Vec<Ptr<Operation>>> {
    let mut names = HashMap::new();
    for &operation in preferred {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            require(
                names.insert(name.0.clone(), operation).is_none(),
                "duplicate schedule name",
            )?;
        }
    }
    let mut dependencies = HashMap::new();
    for &operation in preferred {
        let mut incoming = HashSet::new();
        for operand in operation.deref(ctx).operands() {
            incoming.insert(operand_definition(ctx, operand)?);
        }
        if let Some(controls) =
            optional_attr::<ControlPredecessors>(ctx, operation, "hlo_control_predecessors")?
        {
            for name in &controls.0 {
                incoming.insert(
                    *names
                        .get(name)
                        .ok_or_else(|| error("unresolved scheduled control predecessor"))?,
                );
            }
        }
        dependencies.insert(operation, incoming);
    }
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    while ordered.len() < preferred.len() {
        let next = preferred
            .iter()
            .copied()
            .find(|operation| {
                !seen.contains(operation)
                    && dependencies[operation]
                        .iter()
                        .all(|dependency| seen.contains(dependency))
            })
            .ok_or_else(|| error("cycle or external dependency in cloned schedule"))?;
        seen.insert(next);
        ordered.push(next);
    }
    Ok(ordered)
}

pub(super) fn sequence(
    ctx: &Context,
    computation: &ComputationOp,
    order: &[Ptr<Operation>],
) -> Result<InstructionSequence> {
    let name = computation
        .get_attr_hlo_computation_name(ctx)
        .ok_or_else(|| error("scheduled clone has no computation name"))?
        .clone();
    let instructions = order
        .iter()
        .map(|&instruction| {
            optional_attr::<InstructionName>(ctx, instruction, "hlo_name")?
                .map(|name| name.0.clone())
                .ok_or_else(|| error("scheduled clone instruction has no name"))
        })
        .collect::<Result<_>>()?;
    Ok(InstructionSequence {
        computation: name,
        instructions,
    })
}

#[cfg(test)]
pub(crate) struct ScheduledComputationClone {
    pub cloned: ComputationClone,
    pub sequence: InstructionSequence,
}

/// The separate pinned schedule companion, not a requirement on ordinary clones.
/// Schedule construction failures dispose the already-detached clone transaction.
#[cfg(test)]
pub(crate) fn clone_with_schedule_and_replacements(
    ctx: &mut Context,
    source: &ComputationOp,
    name: crate::ir::ComputationName,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
) -> Result<ScheduledComputationClone> {
    let module = owning_module(ctx, source.get_operation());
    let mut context = ComputationCloneContext::new(module, "clone".into());
    clone_with_schedule_in_context(ctx, source, name, replacements, options, &mut context)
}

#[cfg(test)]
pub(crate) fn clone_with_schedule_in_context(
    ctx: &mut Context,
    source: &ComputationOp,
    name: crate::ir::ComputationName,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<ScheduledComputationClone> {
    let module = source
        .get_operation()
        .deref(ctx)
        .get_parent_block()
        .and_then(|block| block.deref(ctx).get_parent_op(ctx))
        .ok_or_else(|| error("scheduled clone requires a module"))?;
    let source_name = source
        .get_attr_hlo_computation_name(ctx)
        .ok_or_else(|| error("scheduled source requires a name"))?
        .clone();
    let source_order = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .and_then(|schedule| {
            schedule
                .0
                .iter()
                .find(|sequence| sequence.computation == source_name)
                .cloned()
        })
        .ok_or_else(|| error("source computation is not scheduled"))?;
    let source_instructions = ordered_instructions(ctx, source)?;
    let mut by_name = HashMap::new();
    for instruction in source_instructions {
        if let Some(name) = optional_attr::<InstructionName>(ctx, instruction, "hlo_name")? {
            by_name.insert(name.0.clone(), instruction);
        }
    }
    let mut transaction = context.clone();
    let output = clone_impl(ctx, source, name, replacements, options, &mut transaction)?;
    let result = (|| {
        let mut ordered = Vec::new();
        let mut mapped = HashSet::new();
        let mut after_last_parameter = 0;
        for name in &source_order.instructions {
            let original = *by_name
                .get(name)
                .ok_or_else(|| error("scheduled source instruction is unresolved"))?;
            let Some(selected) = selected(original, replacements) else {
                continue;
            };
            let cloned = output
                .mapping
                .lookup_op(selected)
                .ok_or_else(|| error("scheduled replacement has no clone mapping"))?;
            if mapped.insert(cloned) {
                ordered.push(cloned);
            }
            if opcode(ctx, original)? == Opcode::Parameter {
                after_last_parameter = ordered.len();
            }
        }
        let instructions = ordered_instructions(ctx, &output.computation)?;
        let mut extras = Vec::new();
        for &operation in &instructions {
            if !mapped.contains(&operation) && opcode(ctx, operation)? == Opcode::Parameter {
                extras.push(operation);
            }
        }
        ordered.splice(after_last_parameter..after_last_parameter, extras);
        ordered.extend(
            instructions.iter().copied().filter(|operation| {
                !ordered_contains_mapped_or_parameter(ctx, *operation, &mapped)
            }),
        );
        // Upstream instruction construction always assigns a name. Native
        // constructors allow unnamed extras, so supply deterministic names here.
        let mut used = transaction.instruction_names.clone();
        for operation in nested_operations(ctx, &[module]) {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                used.insert(name.0.clone());
            }
        }
        for &operation in &instructions {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                used.insert(name.0.clone());
            }
        }
        for &operation in &instructions {
            if optional_attr::<InstructionName>(ctx, operation, "hlo_name")?.is_none() {
                let name = unique_name(&mut used, &format!("{}.clone", opcode(ctx, operation)?));
                operation
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_name"), InstructionName(name));
            }
        }
        transaction.instruction_names = used;
        let sequence = sequence(ctx, &output.computation, &ordered)?;
        if let Some(options) = options {
            // Module verification skips dependency checks for unreachable
            // computations; this detached result still needs its supplied order
            // checked at the explicitly verified clone boundary.
            require(
                topological_order(ctx, &ordered)? == ordered,
                "mapped clone schedule violates replacement dependencies",
            )?;
            let mut schedules = output.embedded_schedules.clone();
            schedules.push(sequence.clone());
            verify_in_source_context(
                ctx,
                transaction.destination,
                &output.computation,
                &output.embedded_computations,
                &schedules,
                &transaction,
                options,
            )?;
        }
        Ok(sequence)
    })();
    match result {
        Ok(sequence) => {
            transaction.schedules.push(sequence.clone());
            *context = transaction;
            Ok(ScheduledComputationClone {
                cloned: output,
                sequence,
            })
        }
        Err(error) => {
            Operation::erase(output.computation.get_operation(), ctx);
            for computation in output.embedded_computations {
                Operation::erase(computation.get_operation(), ctx);
            }
            Err(error)
        }
    }
}

#[cfg(test)]
pub(super) fn ordered_contains_mapped_or_parameter(
    ctx: &Context,
    instruction: Ptr<Operation>,
    mapped: &HashSet<Ptr<Operation>>,
) -> bool {
    mapped.contains(&instruction)
        || Operation::is_op::<crate::ops::control::ParameterOp>(instruction, ctx)
}
