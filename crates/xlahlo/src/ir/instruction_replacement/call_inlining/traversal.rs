//! Module/callee lookup and callee-first computation traversal for `CallInliner`.

use super::*;

pub(super) fn module_for(ctx: &Context, mut operation: Ptr<Operation>) -> Result<Ptr<Operation>> {
    loop {
        if Operation::is_op::<ModuleOp>(operation, ctx) {
            return Ok(operation);
        }
        operation = operation
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("inline_call requires an attached native module"))?;
    }
}

pub(super) fn callee(
    ctx: &Context,
    module: Ptr<Operation>,
    call: Ptr<Operation>,
) -> Result<Ptr<Operation>> {
    let callees = optional_attr::<Callees>(ctx, call, "hlo_callees")?
        .ok_or_else(|| error("call has no callee"))?;
    require(callees.0.len() == 1, "call requires exactly one callee")?;
    let body = module
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .ok_or_else(|| error("module has no computation block"))?;
    let mut found = None;
    for operation in body.deref(ctx).iter(ctx) {
        if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx)
            && computation
                .get_attr_hlo_computation_name(ctx)
                .is_some_and(|name| *name == callees.0[0])
        {
            require(found.replace(operation).is_none(), "ambiguous call target")?;
        }
    }
    found.ok_or_else(|| error("unknown call target"))
}

pub(super) fn body(ctx: &Context, computation: Ptr<Operation>) -> Result<Ptr<BasicBlock>> {
    require(
        Operation::is_op::<ComputationOp>(computation, ctx),
        "call body must belong to a computation",
    )?;
    let regions: Vec<_> = computation.deref(ctx).regions().collect();
    require(regions.len() == 1, "callee requires one region")?;
    let blocks: Vec<_> = regions[0].deref(ctx).iter(ctx).collect();
    require(
        blocks.len() == 1 && blocks[0].deref(ctx).get_num_arguments() == 0,
        "callee requires one native parameter-instruction block",
    )?;
    Ok(blocks[0])
}

pub(super) fn predecessors(ctx: &Context, instruction: Ptr<Operation>) -> Result<Vec<String>> {
    Ok(
        optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
            .map(|controls| controls.0.clone())
            .unwrap_or_default(),
    )
}

pub(super) fn visit(
    ctx: &Context,
    operation: Ptr<Operation>,
    members: &HashSet<Ptr<Operation>>,
    names: &HashMap<String, Ptr<Operation>>,
    colors: &mut HashMap<Ptr<Operation>, bool>,
    order: &mut Vec<Ptr<Operation>>,
    computation_postorder: bool,
) -> Result<()> {
    if let Some(&finished) = colors.get(&operation) {
        return require(finished, "cycle in inline callee dependencies");
    }
    require(
        members.contains(&operation),
        "callee captures an external operand",
    )?;
    colors.insert(operation, false);
    let operands = operation
        .deref(ctx)
        .operands()
        .map(|operand| {
            operand
                .defining_op()
                .ok_or_else(|| error("callee operand is not an instruction"))
        })
        .collect::<Result<Vec<_>>>()?;
    let controls = predecessors(ctx, operation)?
        .iter()
        .map(|name| {
            names
                .get(name)
                .copied()
                .ok_or_else(|| error("unresolved callee control predecessor"))
        })
        .collect::<Result<Vec<_>>>()?;
    let dependencies: Vec<_> = if computation_postorder {
        controls.into_iter().rev().chain(operands).collect()
    } else {
        operands.into_iter().chain(controls).collect()
    };
    for dependency in dependencies {
        visit(
            ctx,
            dependency,
            members,
            names,
            colors,
            order,
            computation_postorder,
        )?;
    }
    colors.insert(operation, true);
    order.push(operation);
    Ok(())
}

pub(super) fn computation_postorder(
    ctx: &Context,
    computation: Ptr<Operation>,
) -> Result<Vec<Ptr<Operation>>> {
    let sources = instruction_order(ctx, computation)?;
    let members: HashSet<_> = sources.iter().copied().collect();
    let mut names = HashMap::new();
    for &source in &sources {
        if let Some(name) = optional_attr::<InstructionName>(ctx, source, "hlo_name")? {
            require(
                names.insert(name.0.clone(), source).is_none(),
                "duplicate wrapped computation instruction name",
            )?;
        }
    }
    let mut colors = HashMap::new();
    let mut order = Vec::new();
    for source in sources {
        if source
            .deref(ctx)
            .get_result(0)
            .uses(ctx)
            .iter()
            .all(|use_| Operation::is_op::<ReturnOp>(use_.user_op(), ctx))
        {
            visit(ctx, source, &members, &names, &mut colors, &mut order, true)?;
        }
    }
    require(
        order.len() == members.len(),
        "wrapped computation contains a dependency cycle",
    )?;
    Ok(order)
}
