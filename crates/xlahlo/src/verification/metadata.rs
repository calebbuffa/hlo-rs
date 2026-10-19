use super::{VerifyOptions, error, optional_attr, require};
use crate::{
    attributes::{OpMetadata, Payload},
    inference::Signature,
    ir::{
        self, AliasKind, BufferDonors, ControlPredecessors, ExecutionThread, InputOutputAliases,
        InstructionName, Schedule,
    },
    ops::ReturnOp,
    shape::Shape,
    types::{BufferType, TensorType, TupleType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{attributes::TypeAttr, type_interfaces::FunctionTypeInterface, types::FunctionType},
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::DefiningEntity,
};
use std::collections::{HashMap, HashSet};

fn index(path: &[i64]) -> Result<Vec<usize>> {
    path.iter()
        .map(|&i| usize::try_from(i).map_err(|_| error("negative metadata shape index")))
        .collect()
}

pub(super) fn verify_instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    options: VerifyOptions,
    payload_count: usize,
) -> Result<()> {
    let data = op.deref(ctx);
    if data.get_num_results() != 1 {
        return Ok(());
    }
    let shape = data.get_type(0);
    if let Some(value) =
        optional_attr::<crate::original_value::OriginalValue>(ctx, op, "hlo_original_value")?
    {
        value.verify_shape(ctx, shape)?;
    }
    if Shape::has_unbounded(ctx, shape) {
        require(
            options
                .supported_unbounded_dynamic_op
                .is_some_and(|supported| supported(ctx, op)),
            "unbounded dynamism is disabled for this instruction",
        )?;
    }
    if options.verify_no_host_memory_space {
        let mut pending = vec![shape];
        while let Some(shape) = pending.pop() {
            let ty = shape.deref(ctx);
            if let Some(array) = ty.downcast_ref::<TensorType>() {
                require(
                    !array.layout.iter().any(|layout| layout.memory_space == 5),
                    "instruction shape contains host memory space",
                )?;
            } else if let Some(tuple) = ty.downcast_ref::<TupleType>() {
                pending.extend(tuple.elements.iter().copied());
            } else if let Some(buffer) = ty.downcast_ref::<BufferType>() {
                pending.push(buffer.array);
            }
        }
    }
    if let Some(metadata) = optional_attr::<OpMetadata>(ctx, op, "hlo_metadata")? {
        if let Payload::Id(id) = metadata.metadata_payload {
            require(
                usize::try_from(id).is_ok_and(|id| id < payload_count),
                "metadata payload ID is out of range",
            )?;
        }
        if options.verify_instruction_name_unchanged && !metadata.scheduling_name.is_empty() {
            let name = optional_attr::<InstructionName>(ctx, op, "hlo_name")?
                .ok_or_else(|| error("scheduling metadata requires an instruction name"))?;
            require(
                metadata.scheduling_name == name.0
                    || name.0.contains(".clone")
                    || name.0.contains(".remat"),
                "instruction name differs from its scheduling name",
            )?;
        }
    }
    Ok(())
}

pub(super) fn verify_control_dependencies(ctx: &Context, block: Ptr<BasicBlock>) -> Result<()> {
    let instructions: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let mut names = HashMap::new();
    let mut controls = Vec::new();
    for &op in &instructions {
        if op.deref(ctx).attributes.0.is_empty() {
            continue;
        }
        if let Some(name) = optional_attr::<InstructionName>(ctx, op, "hlo_name")? {
            require(!name.0.is_empty(), "instruction name cannot be empty")?;
            require(
                names.insert(name.0.clone(), op).is_none(),
                "duplicate instruction name",
            )?;
        }
        if let Some(predecessors) =
            optional_attr::<ControlPredecessors>(ctx, op, "hlo_control_predecessors")?
            && !predecessors.0.is_empty()
        {
            controls.push((op, predecessors));
        }
    }
    if controls.is_empty() {
        return Ok(());
    }
    let mut indegree: HashMap<_, usize> = instructions.iter().map(|&op| (op, 0)).collect();
    let mut successors: HashMap<_, Vec<_>> = HashMap::new();
    let mut edge = |from, to| -> Result<()> {
        *indegree
            .get_mut(&to)
            .ok_or_else(|| error("control dependency target is outside its block"))? += 1;
        successors.entry(from).or_default().push(to);
        Ok(())
    };
    for &op in &instructions {
        for operand in op.deref(ctx).operands() {
            if let DefiningEntity::Op(definition) = operand.defining_entity() {
                edge(definition, op)?;
            }
        }
    }
    for (op, predecessors) in controls {
        for predecessor in &predecessors.0 {
            let definition = names
                .get(predecessor)
                .ok_or_else(|| error(format!("unknown control predecessor: {predecessor}")))?;
            require(
                !Operation::is_op::<ReturnOp>(*definition, ctx),
                "return cannot be a control predecessor",
            )?;
            edge(*definition, op)?;
        }
    }
    let mut ready: Vec<_> = indegree
        .iter()
        .filter_map(|(&op, &n)| (n == 0).then_some(op))
        .collect();
    let mut count = 0;
    while let Some(op) = ready.pop() {
        count += 1;
        if let Some(users) = successors.get(&op) {
            for user in users {
                let n = indegree
                    .get_mut(user)
                    .ok_or_else(|| error("control dependency user is outside its block"))?;
                *n = n
                    .checked_sub(1)
                    .ok_or_else(|| error("inconsistent control dependency indegree"))?;
                if *n == 0 {
                    ready.push(*user);
                }
            }
        }
    }
    require(
        count == instructions.len(),
        "cycle in data/control dependencies",
    )
}

pub(crate) fn execution_thread(ctx: &Context, computation: Ptr<Operation>) -> Result<String> {
    let thread = optional_attr::<ExecutionThread>(ctx, computation, "hlo_execution_thread")?
        .map(|thread| thread.0.clone())
        .unwrap_or_default();
    Ok(if thread.is_empty() {
        "main".to_owned()
    } else {
        thread
    })
}

pub(super) fn verify_schedule(
    ctx: &Context,
    module: Ptr<Operation>,
    computations: &HashMap<Identifier, Ptr<Operation>>,
    calls: &HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
) -> Result<()> {
    let Some(schedule) = optional_attr::<Schedule>(ctx, module, "hlo_schedule")? else {
        return Ok(());
    };
    let entry = computations[&ir::entry(ctx, module)?];
    let called: HashSet<_> = calls.values().flatten().copied().collect();
    let mut sequences = HashMap::new();
    let mut threads = HashSet::new();
    for sequence in &schedule.0 {
        let computation = computations
            .get(&sequence.computation)
            .ok_or_else(|| error("unknown scheduled computation"))?;
        require(
            sequences.insert(*computation, sequence).is_none(),
            "duplicate computation schedule",
        )?;
        threads.insert(execution_thread(ctx, *computation)?);
    }
    for &computation in computations.values() {
        let live = computation == entry || called.contains(&computation);
        let sequence = sequences.get(&computation);
        if live && threads.contains(&execution_thread(ctx, computation)?) {
            require(sequence.is_some(), "missing live computation schedule")?;
        }
        let Some(sequence) = sequence else {
            continue;
        };
        let block = super::single_block(ctx, computation.deref(ctx).get_region(0))?;
        let instructions: Vec<_> = block
            .deref(ctx)
            .iter(ctx)
            .filter(|&op| !Operation::is_op::<ReturnOp>(op, ctx))
            .collect();
        let mut names = HashMap::new();
        for &op in &instructions {
            if let Some(name) = optional_attr::<InstructionName>(ctx, op, "hlo_name")? {
                names.insert(name.0.clone(), op);
            }
        }
        let mut ordered = Vec::new();
        for name in &sequence.instructions {
            ordered.push(
                *names
                    .get(name)
                    .ok_or_else(|| error(format!("unknown scheduled instruction: {name}")))?,
            );
        }
        if !live {
            continue;
        }
        require(
            ordered.len() == instructions.len()
                && ordered.iter().copied().collect::<HashSet<_>>().len() == ordered.len(),
            "schedule must contain every instruction exactly once",
        )?;
        let mut seen = HashSet::new();
        for op in ordered {
            for operand in op.deref(ctx).operands() {
                if let DefiningEntity::Op(definition) = operand.defining_entity() {
                    require(
                        seen.contains(&definition),
                        "schedule violates data dependency order",
                    )?;
                }
            }
            if let Some(predecessors) =
                optional_attr::<ControlPredecessors>(ctx, op, "hlo_control_predecessors")?
            {
                for predecessor in &predecessors.0 {
                    require(
                        names.get(predecessor).is_some_and(|op| seen.contains(op)),
                        "schedule violates control dependency order",
                    )?;
                }
            }
            seen.insert(op);
        }
    }
    Ok(())
}

pub(super) fn verify_aliases_and_donors(
    ctx: &Context,
    module: Ptr<Operation>,
    computations: &HashMap<Identifier, Ptr<Operation>>,
    signatures: &HashMap<Ptr<Operation>, Signature>,
    options: VerifyOptions,
) -> Result<()> {
    let entry = ir::entry(ctx, module)?;
    let actual = &signatures[&computations[&entry]];
    let mut declared = None;
    if let Some(program) = optional_attr::<TypeAttr>(ctx, module, "hlo_host_program_shape")? {
        let ty = program.get_type(ctx);
        let data = ty.deref(ctx);
        let function = data
            .downcast_ref::<FunctionType>()
            .ok_or_else(|| error("host program shape must be a function type"))?;
        let inputs = function.arg_types();
        let outputs = function.res_types();
        require(outputs.len() == 1, "host program shape needs one result")?;
        require(
            inputs.len() == actual.0.len()
                && inputs
                    .iter()
                    .zip(&actual.0)
                    .all(|(&a, &b)| options.same_shapes(ctx, a, b))
                && options.same_shapes(ctx, outputs[0], actual.1),
            "host program signature differs from entry",
        )?;
        declared = Some((inputs, outputs[0]));
    }
    let program = declared.as_ref().unwrap_or(actual);
    let parameter = |number: i64, path: &[i64]| {
        let shape = usize::try_from(number)
            .ok()
            .and_then(|i| program.0.get(i))
            .ok_or_else(|| error("entry parameter number out of range"))?;
        Shape::subshape(ctx, *shape, &index(path)?)
    };
    let mut outputs = HashSet::new();
    let mut inputs = HashSet::new();
    let aliases = optional_attr::<InputOutputAliases>(ctx, module, "hlo_input_output_aliases")?;
    if let Some(aliases) = &aliases {
        for alias in &aliases.0 {
            let input = parameter(alias.parameter_number, &alias.parameter_shape_index.0)?;
            let output = Shape::subshape(ctx, program.1, &index(&alias.output_shape_index.0)?)?;
            require(
                Shape::is_array_or_buffer(ctx, input) && Shape::is_array_or_buffer(ctx, output),
                "module aliases must reference array subshapes",
            )?;
            if alias.kind == AliasKind::MustAlias {
                require(
                    Shape::equal_including_layout(ctx, input, output),
                    "must-alias entry shapes and layouts differ",
                )?;
            }
            require(
                outputs.insert(alias.output_shape_index.0.as_slice())
                    && inputs.insert((
                        alias.parameter_number,
                        alias.parameter_shape_index.0.as_slice(),
                    )),
                "module alias input or output is used more than once",
            )?;
            let size = |shape| {
                if Shape::has_unbounded(ctx, shape) {
                    Ok(i64::MIN)
                } else if options.layout_sensitive {
                    options.size_of(ctx, shape)
                } else {
                    Ok(0)
                }
            };
            require(
                size(input)? == size(output)?,
                "aliased buffers have different target sizes",
            )?;
        }
    }
    if let Some(donors) = optional_attr::<BufferDonors>(ctx, module, "hlo_buffer_donors")? {
        for donor in &donors.0 {
            require(
                Shape::is_array_or_buffer(
                    ctx,
                    parameter(donor.parameter_number, &donor.parameter_shape_index.0)?,
                ),
                "buffer donors must reference arrays",
            )?;
            require(
                !inputs.contains(&(
                    donor.parameter_number,
                    donor.parameter_shape_index.0.as_slice(),
                )),
                "an aliased parameter cannot also be a buffer donor",
            )?;
        }
    }
    Ok(())
}
