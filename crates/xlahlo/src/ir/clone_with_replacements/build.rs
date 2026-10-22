use super::*;

pub(super) fn build(
    ctx: &mut Context,
    computation: &ComputationOp,
    block: Ptr<BasicBlock>,
    replacements: &Replacements,
    plan: &Plan,
    async_callees: &AsyncCallees,
    options: Option<VerifyOptions>,
) -> Result<IrMapping> {
    let mut mapping = IrMapping::new();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    for &template in &plan.postorder {
        // Pre-map EVERY direct operand. clone_operation's same-context fallback
        // is intentionally unreachable, including for external dependencies.
        let operands: Vec<_> = template.deref(ctx).operands().collect();
        for operand in operands {
            let original = operand_definition(ctx, operand)?;
            let selected = selected(original, replacements)
                .ok_or_else(|| error("omitted operand after clone planning"))?;
            let cloned = mapping
                .lookup_op(selected)
                .ok_or_else(|| error("unmapped operand after clone planning"))?;
            mapping.map_value(operand, cloned.deref(ctx).get_result(0));
        }
        let cloned = clone_operation(template, ctx, &mut rewriter, &mut mapping);
        cloned.insert_at_back(block, ctx);
        normalize_nested_metadata(ctx, template, &mapping, plan)?;
        for original in nested_operations(ctx, &[template]) {
            if let Some(callees) = async_callees.get(&original) {
                let cloned = mapping
                    .lookup_op(original)
                    .ok_or_else(|| error("async start missing from clone mapping"))?;
                cloned
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_callees"), callees.clone());
            }
        }
        {
            let mut data = cloned.deref_mut(ctx);
            if let Some(controls) = plan.controls.get(&template) {
                let controls = controls
                    .iter()
                    .map(|predecessor| {
                        plan.names
                            .get(predecessor)
                            .cloned()
                            .ok_or_else(|| error("control predecessor lacks a cloned name"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                data.attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(controls),
                );
            } else if data
                .attributes
                .0
                .contains_key(&Identifier::new("hlo_control_predecessors"))
            {
                data.attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors::default(),
                );
            }
            if let Some(name) = plan.names.get(&template) {
                data.attributes
                    .set(Identifier::new("hlo_name"), InstructionName(name.clone()));
            }
        }
        if let Some(options) = options {
            let dynamic = Operation::get_op_dyn(cloned, ctx);
            op_cast::<dyn HloOpInterface>(dynamic.as_ref())
                .ok_or_else(|| error("cloned instruction lost HLO interface"))?
                .verify_with_options(ctx, options)?;
        }
    }
    let roots: Vec<_> = plan
        .roots
        .iter()
        .map(|&root| {
            mapping
                .lookup_op(root)
                .map(|cloned| cloned.deref(ctx).get_result(0))
                .ok_or_else(|| error("selected root has no clone"))
        })
        .collect::<Result<_>>()?;
    let root = if plan.tuple_root {
        let tuple = TupleOp::new(ctx, plan.result, roots);
        tuple.get_operation().insert_at_back(block, ctx);
        let name = plan
            .tuple_name
            .clone()
            .ok_or_else(|| error("tuple root has no planned unique name"))?;
        tuple
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name));
        tuple.result(ctx)
    } else {
        roots[0]
    };
    ReturnOp::new(ctx, root)
        .get_operation()
        .insert_at_back(block, ctx);
    computation.verify(ctx)?;
    topological_order(ctx, &ordered_instructions(ctx, computation)?)?;
    install_logical_order(ctx, computation, replacements, plan, &mapping)?;
    Ok(mapping)
}

pub(super) fn instruction_name(ctx: &Context, operation: Ptr<Operation>) -> Result<String> {
    optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
        .map(|name| name.0.clone())
        .ok_or_else(|| error("logical clone ordering requires a planned instruction name"))
}

pub(super) fn mapped_member(
    ctx: &Context,
    value: Value,
    replacements: &Replacements,
    mapping: &IrMapping,
) -> Result<Option<Value>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => {
            let Some(template) = selected(operation, replacements) else {
                return Ok(None);
            };
            Ok(mapping
                .lookup_op(template)
                .map(|op| op.deref(ctx).get_result(0)))
        }
        DefiningEntity::Block(_) => Ok(mapping.lookup_value(value)),
    }
}

/// Stable mapped-element ordering with unmatched members appended, as in
/// MappedPtrContainerSorter. Membership comes exclusively from the actual graph.
pub(super) fn mapped_order(
    actual: &[Value],
    preferred: impl IntoIterator<Item = Value>,
) -> Vec<Value> {
    let members: HashSet<_> = actual.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut result: Vec<_> = preferred
        .into_iter()
        .filter(|value| members.contains(value) && seen.insert(*value))
        .collect();
    result.extend(actual.iter().copied().filter(|value| seen.insert(*value)));
    result
}

pub(super) fn value_name(ctx: &Context, value: Value) -> Result<String> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => instruction_name(ctx, operation),
        DefiningEntity::Block(block) => {
            let index = block
                .deref(ctx)
                .arguments()
                .position(|argument| argument == value)
                .ok_or_else(|| error("logical clone parameter is missing"))?;
            crate::ir::fusion_parameters(ctx, block)?.0[index]
                .name
                .clone()
                .ok_or_else(|| error("logical clone parameter has no name"))
        }
    }
}

pub(super) fn install_logical_order(
    ctx: &Context,
    computation: &ComputationOp,
    replacements: &Replacements,
    plan: &Plan,
    mapping: &IrMapping,
) -> Result<()> {
    let actual = ordered_instructions(ctx, computation)?;
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    let mut after_last_parameter = 0;
    for &original in &plan.original {
        let Some(template) = selected(original, replacements) else {
            continue;
        };
        let cloned = mapping
            .lookup_op(template)
            .ok_or_else(|| error("logical instruction has no clone"))?;
        if seen.insert(cloned) {
            ordered.push(cloned);
        }
        if opcode(ctx, original)? == Opcode::Parameter {
            after_last_parameter = ordered.len();
        }
    }
    let mut parameters = Vec::new();
    let mut external = Vec::new();
    for &operation in &actual {
        if seen.insert(operation) {
            if opcode(ctx, operation)? == Opcode::Parameter {
                parameters.push(operation);
            } else {
                external.push(operation);
            }
        }
    }
    ordered.splice(after_last_parameter..after_last_parameter, parameters);
    ordered.extend(external);
    let mut order = InstructionOrder {
        instructions: ordered
            .iter()
            .map(|&operation| instruction_name(ctx, operation))
            .collect::<Result<_>>()?,
        ..Default::default()
    };
    // Before installing logical storage order, capture insertion-derived users
    // and control successors. Source-original permutations then take precedence.
    let mut user_orders = HashMap::new();
    let mut successor_orders = HashMap::new();
    for &operation in &actual {
        let value = operation.deref(ctx).get_result(0);
        user_orders.insert(value, instruction_order::users(ctx, value)?);
        successor_orders.insert(value, instruction_order::control_successors(ctx, value)?);
    }
    for &original in &plan.original {
        let Some(template) = selected(original, replacements) else {
            continue;
        };
        let cloned = mapping
            .lookup_op(template)
            .ok_or_else(|| error("logical source member has no clone"))?;
        let value = cloned.deref(ctx).get_result(0);
        let old_value = original.deref(ctx).get_result(0);
        let mapped = |values: Vec<Value>| -> Result<Vec<Value>> {
            values
                .into_iter()
                .map(|value| mapped_member(ctx, value, replacements, mapping))
                .collect::<Result<Vec<_>>>()
                .map(|values| values.into_iter().flatten().collect())
        };
        let incoming = instruction_order::control_predecessors(ctx, value)?;
        let preferred = mapped(instruction_order::control_predecessors(ctx, old_value)?)?;
        let incoming = mapped_order(&incoming, preferred);
        if !incoming.is_empty() {
            cloned.deref_mut(ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(
                    incoming
                        .into_iter()
                        .map(|value| value_name(ctx, value))
                        .collect::<Result<_>>()?,
                ),
            );
        }
        let users = user_orders
            .get(&value)
            .ok_or_else(|| error("cloned user ordering snapshot is missing"))?;
        user_orders.insert(
            value,
            mapped_order(users, mapped(instruction_order::users(ctx, old_value)?)?),
        );
        let successors = successor_orders
            .get(&value)
            .ok_or_else(|| error("cloned control ordering snapshot is missing"))?;
        successor_orders.insert(
            value,
            mapped_order(
                successors,
                mapped(instruction_order::control_successors(ctx, old_value)?)?,
            ),
        );
    }
    for &operation in &actual {
        let value = operation.deref(ctx).get_result(0);
        let subject = value_name(ctx, value)?;
        let members = user_orders
            .remove(&value)
            .ok_or_else(|| error("missing logical user order"))?;
        if !members.is_empty() {
            order.users.push(OrderPermutation {
                subject: subject.clone(),
                members: members
                    .into_iter()
                    .map(|member| value_name(ctx, member))
                    .collect::<Result<_>>()?,
            });
        }
        let members = successor_orders
            .remove(&value)
            .ok_or_else(|| error("missing logical successor order"))?;
        if !members.is_empty() {
            order.control_successors.push(OrderPermutation {
                subject,
                members: members
                    .into_iter()
                    .map(|member| value_name(ctx, member))
                    .collect::<Result<_>>()?,
            });
        }
    }
    for original in nested_operations(ctx, &plan.postorder) {
        if !Operation::is_op::<crate::ops::control::FusionOp>(original, ctx) {
            continue;
        }
        for region in original.deref(ctx).regions() {
            for source_block in region.deref(ctx).iter(ctx) {
                let cloned_owner = mapping
                    .lookup_op(original)
                    .ok_or_else(|| error("fusion ordering has no cloned owner"))?;
                let source_order = instruction_order::block_value_order(ctx, source_block)?;
                let cloned_order = source_order
                    .iter()
                    .map(|&value| {
                        mapping
                            .lookup_value(value)
                            .ok_or_else(|| error("fusion ordering member has no clone"))
                            .and_then(|value| value_name(ctx, value))
                    })
                    .collect::<Result<Vec<_>>>()?;
                order.blocks.push(OrderPermutation {
                    subject: instruction_name(ctx, cloned_owner)?,
                    members: cloned_order,
                });
                for source_value in source_order {
                    let cloned = mapping
                        .lookup_value(source_value)
                        .ok_or_else(|| error("fusion ordering value has no clone mapping"))?;
                    let subject = value_name(ctx, cloned)?;
                    for (members, records) in [
                        (
                            instruction_order::users(ctx, source_value)?,
                            &mut order.users,
                        ),
                        (
                            instruction_order::control_successors(ctx, source_value)?,
                            &mut order.control_successors,
                        ),
                    ] {
                        if members.is_empty() {
                            continue;
                        }
                        let members = members
                            .into_iter()
                            .map(|member| {
                                mapping
                                    .lookup_value(member)
                                    .ok_or_else(|| error("fusion ordering relation has no clone"))
                                    .and_then(|member| value_name(ctx, member))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        records.push(OrderPermutation {
                            subject: subject.clone(),
                            members,
                        });
                    }
                }
            }
        }
    }
    instruction_order::set_order(ctx, computation.get_operation(), order)
}
