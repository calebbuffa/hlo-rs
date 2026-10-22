use super::*;

pub(super) struct Plan {
    pub(super) original: Vec<Ptr<Operation>>,
    pub(super) postorder: Vec<Ptr<Operation>>,
    pub(super) roots: Vec<Ptr<Operation>>,
    pub(super) tuple_root: bool,
    pub(super) tuple_name: Option<String>,
    pub(super) parameters: Vec<TypeHandle>,
    pub(super) result: TypeHandle,
    pub(super) names: HashMap<Ptr<Operation>, String>,
    pub(super) parameter_names: HashMap<(Ptr<BasicBlock>, usize), String>,
    pub(super) controls: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
}

pub(super) fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref())
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| error("clone dependency is not a native HLO instruction"))
}

pub(super) fn selected(
    operation: Ptr<Operation>,
    replacements: &Replacements,
) -> Option<Ptr<Operation>> {
    replacements
        .instructions
        .get(&operation)
        .copied()
        .unwrap_or(Some(operation))
}

pub(super) fn operand_definition(ctx: &Context, value: Value) -> Result<Ptr<Operation>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation)
            if operation.deref(ctx).get_num_results() == 1
                && operation.deref(ctx).get_result(0) == value =>
        {
            Ok(operation)
        }
        _ => Err(error(
            "computation clone operand is not a single HLO result",
        )),
    }
}

pub(super) fn body(ctx: &Context, computation: &ComputationOp) -> Result<Ptr<BasicBlock>> {
    let operation = computation.get_operation().deref(ctx);
    require(
        operation.num_regions() == 1
            && operation.get_num_operands() == 0
            && operation.get_num_results() == 0
            && operation.get_num_successors() == 0,
        "invalid computation clone source",
    )?;
    let blocks: Vec<_> = computation.body(ctx).deref(ctx).iter(ctx).collect();
    require(blocks.len() == 1, "computation cloning requires one block")?;
    require(
        blocks[0].deref(ctx).get_num_arguments() == 0,
        "computation cloning requires parameter instructions",
    )?;
    Ok(blocks[0])
}

pub(super) fn validate_region_closure(ctx: &Context, instruction: Ptr<Operation>) -> Result<()> {
    let mut pending = vec![instruction];
    while let Some(operation) = pending.pop() {
        let regions: Vec<_> = operation.deref(ctx).regions().collect();
        for region in regions {
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            require(blocks.len() == 1, "HLO clone region must have one block")?;
            let block = blocks[0];
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            let definitions: HashSet<_> = children.iter().copied().collect();
            for child in children {
                let data = child.deref(ctx);
                require(
                    data.get_num_successors() == 0,
                    "HLO clone has block successors",
                )?;
                for operand in data.operands() {
                    let inside = match operand.defining_entity() {
                        DefiningEntity::Op(owner) => definitions.contains(&owner),
                        DefiningEntity::Block(owner) => owner == block,
                    };
                    require(inside, "inline clone region captures an external value")?;
                }
                pending.push(child);
            }
        }
    }
    Ok(())
}

pub(super) fn plan(
    ctx: &Context,
    computation: &ComputationOp,
    replacements: &Replacements,
) -> Result<Plan> {
    let block = body(ctx, computation)?;
    let mut physical: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let terminator = physical.pop().ok_or_else(|| error("empty clone source"))?;
    require(
        Operation::is_op::<ReturnOp>(terminator, ctx)
            && terminator.deref(ctx).get_num_operands() == 1,
        "computation clone source needs a return",
    )?;
    let original = instruction_order::instruction_order(ctx, computation.get_operation())?;
    let old_root = operand_definition(ctx, terminator.deref(ctx).get_operand(0))?;
    let originals: HashSet<_> = original.iter().copied().collect();
    require(
        originals.contains(&old_root),
        "root is outside source computation",
    )?;
    for (&key, &replacement) in &replacements.instructions {
        opcode(ctx, key)?;
        if let Some(template) = replacement {
            require(
                template.deref(ctx).get_parent_block().is_none(),
                "replacement value must be an owned detached instruction",
            )?;
        }
    }
    for &parameter in &replacements.extra_parameters {
        require(
            opcode(ctx, parameter)? == Opcode::Parameter
                && parameter.deref(ctx).get_num_operands() == 0
                && parameter.deref(ctx).get_num_results() == 1,
            "extra parameter must be a native parameter without operands",
        )?;
    }
    let (root_inputs, tuple_root) = match &replacements.root {
        CloneRoot::Keep => (vec![old_root], false),
        CloneRoot::Instruction(instruction) => {
            require(
                originals.contains(instruction),
                "new root is outside source computation",
            )?;
            (vec![*instruction], false)
        }
        CloneRoot::Tuple(instructions) => {
            require(
                instructions.iter().all(|op| originals.contains(op)),
                "tuple root contains an instruction outside source computation",
            )?;
            (instructions.clone(), true)
        }
    };
    let roots: Vec<_> = root_inputs
        .into_iter()
        .map(|root| selected(root, replacements).ok_or_else(|| error("clone root is omitted")))
        .collect::<Result<_>>()?;
    let starts = original.iter().filter_map(|&op| selected(op, replacements));
    let mut state = HashMap::<Ptr<Operation>, bool>::new();
    let mut postorder = Vec::new();
    for start in starts {
        let mut pending = vec![(start, false)];
        while let Some((operation, finish)) = pending.pop() {
            if finish {
                state.insert(operation, true);
                postorder.push(operation);
                continue;
            }
            if let Some(&finished) = state.get(&operation) {
                require(finished, "cycle in replaced operand graph")?;
                continue;
            }
            let data = operation.deref(ctx);
            require(
                data.get_num_results() == 1 && data.get_num_successors() == 0,
                "clone instruction requires one result and no block successors",
            )?;
            opcode(ctx, operation)?;
            validate_region_closure(ctx, operation)?;
            state.insert(operation, false);
            pending.push((operation, true));
            for operand in data.operands() {
                let definition = operand_definition(ctx, operand)?;
                let dependency = selected(definition, replacements)
                    .ok_or_else(|| error("replacements omit a surviving data operand"))?;
                pending.push((dependency, false));
            }
        }
    }
    require(
        roots.iter().all(|root| state.get(root) == Some(&true)),
        "new root is not in the cloned dependency graph",
    )?;
    let edges = control_edges(ctx, &original, &postorder, replacements)?;
    let selected_set: HashSet<_> = postorder.iter().copied().collect();
    // Extras are independent Clone() occurrences, not DFS roots or shared
    // context mappings. An occurrence also reached by DFS must not deduplicate
    // away its duplicate parameter index.
    postorder.splice(0..0, replacements.extra_parameters.iter().copied());
    let mut parameters = BTreeMap::new();
    let mut names = HashMap::new();
    let mut used_names = HashSet::new();
    for &operation in &postorder {
        if opcode(ctx, operation)? == Opcode::Parameter {
            let index = optional_attr::<Index>(ctx, operation, "hlo_index")?
                .ok_or_else(|| error("clone parameter requires an index"))?
                .0;
            require(
                operation.deref(ctx).get_num_operands() == 0
                    && parameters
                        .insert(index, operation.deref(ctx).get_type(0))
                        .is_none(),
                "duplicate clone parameter index or parameter operands",
            )?;
        }
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            require(!name.0.is_empty(), "empty output instruction name")?;
            used_names.insert(name.0.clone());
            names.insert(operation, name.0.clone());
        }
        optional_attr::<OriginalValue>(ctx, operation, "hlo_original_value")?;
    }
    let parameter_count =
        u32::try_from(parameters.len()).map_err(|_| error("too many clone parameters"))?;
    require(
        parameters.keys().copied().eq(0..parameter_count),
        "clone parameter indices must be contiguous",
    )?;
    let mut controls = HashMap::<_, Vec<_>>::new();
    for (predecessor, successor) in &edges {
        if !selected_set.contains(predecessor) {
            continue;
        }
        let Some(successor) = selected(*successor, replacements) else {
            continue;
        };
        require(
            selected_set.contains(&successor),
            "control successor has no clone",
        )?;
        if !names.contains_key(predecessor) {
            let mut index = 0;
            let name = loop {
                let name = format!("clone_control.{index}");
                if used_names.insert(name.clone()) {
                    break name;
                }
                index += 1;
            };
            names.insert(*predecessor, name);
        }
        let incoming = controls.entry(successor).or_default();
        if !incoming.contains(predecessor) {
            incoming.push(*predecessor);
        }
    }
    let result = if tuple_root {
        TupleType::get(
            ctx,
            roots.iter().map(|op| op.deref(ctx).get_type(0)).collect(),
        )?
    } else {
        roots[0].deref(ctx).get_type(0)
    };
    Ok(Plan {
        original,
        postorder,
        roots,
        tuple_root,
        tuple_name: None,
        parameters: parameters.into_values().collect(),
        result,
        names,
        parameter_names: HashMap::new(),
        controls,
    })
}

pub(super) fn control_edges(
    ctx: &Context,
    original: &[Ptr<Operation>],
    postorder: &[Ptr<Operation>],
    replacements: &Replacements,
) -> Result<Vec<(Ptr<Operation>, Ptr<Operation>)>> {
    let mut blocks = Vec::new();
    let mut seen_blocks = HashSet::new();
    let mut edges = Vec::new();
    for &instruction in original.iter().chain(postorder) {
        if let Some(block) = instruction.deref(ctx).get_parent_block() {
            if seen_blocks.insert(block) {
                blocks.push(block);
            }
        } else {
            require(
                optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
                    .is_none_or(|predecessors| predecessors.0.is_empty()),
                "detached symbolic controls need explicit template control edges",
            )?;
        }
    }
    for block in blocks {
        let instructions: Vec<_> = block.deref(ctx).iter(ctx).collect();
        let mut names = HashMap::new();
        for &instruction in &instructions {
            if let Some(name) = optional_attr::<InstructionName>(ctx, instruction, "hlo_name")? {
                require(
                    names.insert(name.0.clone(), instruction).is_none(),
                    "duplicate source instruction name",
                )?;
            }
        }
        for instruction in instructions {
            if let Some(predecessors) =
                optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
            {
                for name in &predecessors.0 {
                    let predecessor = *names
                        .get(name)
                        .ok_or_else(|| error("unresolved source control predecessor"))?;
                    edges.push((predecessor, instruction));
                }
            }
        }
    }
    let known: HashSet<_> = original.iter().chain(postorder).copied().collect();
    for &(predecessor, successor) in &replacements.template_control_edges {
        require(
            known.contains(&predecessor) && known.contains(&successor),
            "template control endpoint is outside the clone graph",
        )?;
        edges.push((predecessor, successor));
    }
    Ok(edges)
}
