use super::*;
use crate::ir::execution_thread as thread;

pub(super) fn parse_opcode(name: &str) -> Result<(Opcode, Option<Opcode>)> {
    if let Ok(opcode) = name.parse() {
        return Ok((opcode, None));
    }
    for (suffix, opcode) in [
        ("-start", Opcode::AsyncStart),
        ("-update", Opcode::AsyncUpdate),
        ("-done", Opcode::AsyncDone),
    ] {
        if let Some(wrapped) = name.strip_suffix(suffix) {
            let wrapped: Opcode = wrapped.parse()?;
            if matches!(
                wrapped,
                Opcode::AllGather
                    | Opcode::AllReduce
                    | Opcode::CollectivePermute
                    | Opcode::Copy
                    | Opcode::Recv
                    | Opcode::Send
            ) {
                return Err(Error::new(
                    "opcode",
                    "dedicated async opcode cannot use generic wrapping",
                ));
            }
            return Ok((opcode, Some(wrapped)));
        }
    }
    Err(Error::new("opcode", format!("unknown opcode {name:?}")))
}

pub(super) fn context_shapes(shape: &Shape) -> Result<(Vec<Shape>, Shape)> {
    if !shape.is_tuple() || shape.as_proto().tuple_shapes.len() < 2 {
        return Err(Error::new(
            "async",
            "async start/update requires ((operands), result, ...)",
        ));
    }
    let operands = shape.subshape(&[0])?;
    if !operands.is_tuple() {
        return Err(Error::new("async", "async operand context must be a tuple"));
    }
    Ok((
        (0..operands.as_proto().tuple_shapes.len())
            .map(|i| operands.subshape(&[i]))
            .collect::<Result<_>>()?,
        shape.subshape(&[1])?,
    ))
}

pub(super) fn parameter_count(opcode: Opcode, shape: &Shape) -> Result<usize> {
    let min = if matches!(opcode, Opcode::Dot | Opcode::Convolution) {
        2
    } else {
        opcode.arity().unwrap_or(0)
    };
    Ok(context_shapes(shape)?.0.len().max(min))
}

fn allocate(next: &mut i64) -> Result<i64> {
    let id = *next;
    *next = next
        .checked_add(1)
        .ok_or_else(|| Error::new("overflow", "instruction count overflow"))?;
    Ok(id)
}

pub(super) fn wrap(
    mut root: HloInstructionProto,
    shape: Shape,
    computation: i64,
    next: &mut i64,
    channels: &mut BTreeSet<i64>,
) -> Result<(HloInstructionProto, HloComputationProto)> {
    let opcode: Opcode = root.opcode.parse()?;
    let operands = context_shapes(&shape)?.0;
    let count = parameter_count(opcode, &shape)?;
    let mut outer = HloInstructionProto {
        id: root.id,
        name: root.name.clone(),
        opcode: "async-start".into(),
        shape: Some(shape.into_proto()).into(),
        called_computation_ids: vec![computation],
        async_execution_thread: std::mem::take(&mut root.async_execution_thread),
        sharding: root.sharding.take().into(),
        metadata: root.metadata.take().into(),
        original_value: root.original_value.take().into(),
        statistics_viz: root.statistics_viz.take().into(),
        frontend_attributes: root.frontend_attributes.take().into(),
        backend_config: std::mem::take(&mut root.backend_config),
        ..Default::default()
    };
    if !matches!(
        opcode,
        Opcode::CustomCall | Opcode::Fusion | Opcode::AsyncStart | Opcode::AsyncUpdate
    ) {
        outer.output_operand_aliasing = std::mem::take(&mut root.output_operand_aliasing);
    }
    let old = root.id;
    root.id = allocate(next)?;
    root.name = "async_op".into();
    if channels.remove(&old) {
        channels.insert(root.id);
    }
    let mut instructions = Vec::new();
    for index in 0..count {
        let id = allocate(next)?;
        let shape = match operands.get(index) {
            Some(shape) => shape.clone(),
            None => Shape::array(PrimitiveType::F32, vec![1])?.with_default_layout()?,
        };
        instructions.push(HloInstructionProto {
            id,
            name: "async_param".into(),
            opcode: "parameter".into(),
            parameter_number: index as i64,
            shape: Some(shape.into_proto()).into(),
            ..Default::default()
        });
        if opcode.arity().is_none_or(|arity| index < arity) {
            root.operand_ids.push(id);
        }
    }
    let root_id = root.id;
    instructions.push(root);
    Ok((
        outer,
        HloComputationProto {
            id: computation,
            name: "async_wrapped".into(),
            root_id,
            instructions,
            ..Default::default()
        },
    ))
}

fn update_signature(
    module: &mut HloModuleProto,
    computation: i64,
    operands: &[Shape],
    result: &Shape,
    exact: bool,
    next: &mut i64,
) -> Result<()> {
    let mut names = crate::ir::names::NameUniquer::default();
    for instruction in module.computations.iter().flat_map(|c| &c.instructions) {
        names.unique(&instruction.name)?;
    }
    let body = module
        .computations
        .iter_mut()
        .find(|c| c.id == computation)
        .ok_or_else(|| Error::new("async", "missing wrapped computation"))?;
    let same = |wire: &crate::proto::xla_data::ShapeProto, expected: &Shape| -> Result<bool> {
        let actual = Shape::from_proto(wire.clone())?;
        Ok(if exact {
            actual.equal_including_layout(expected)
        } else {
            actual.compatible(expected)
        })
    };
    let mut added = Vec::new();
    for (index, shape) in operands.iter().enumerate() {
        if let Some(parameter) = body
            .instructions
            .iter_mut()
            .find(|i| i.opcode == "parameter" && i.parameter_number == index as i64)
        {
            if !same(parameter.shape.get_or_default(), shape)? {
                parameter.shape = Some(shape.as_proto().clone()).into();
            }
        } else {
            let id = allocate(next)?;
            body.instructions.push(HloInstructionProto {
                id,
                name: names.unique("async_param")?,
                opcode: "parameter".into(),
                parameter_number: index as i64,
                shape: Some(shape.as_proto().clone()).into(),
                ..Default::default()
            });
            added.push(id);
        }
    }
    let root = body
        .instructions
        .iter_mut()
        .find(|i| i.id == body.root_id)
        .ok_or_else(|| Error::new("async", "missing wrapped root"))?;
    root.operand_ids.extend(added);
    if !same(root.shape.get_or_default(), result)? {
        root.shape = Some(result.as_proto().clone()).into();
    }
    Ok(())
}

pub(super) fn bind_called_signatures(
    module: &mut HloModuleProto,
    generated: &BTreeSet<i64>,
    next: &mut i64,
    channels: &BTreeSet<i64>,
) -> Result<()> {
    for body in &mut module.computations {
        if generated.contains(&body.id) {
            contexts::normalize(body, channels)?;
        }
    }
    for &id in generated {
        let root = module
            .computations
            .iter()
            .find(|c| c.id == id)
            .and_then(|c| c.instructions.iter().find(|i| i.id == c.root_id))
            .ok_or_else(|| Error::new("async", "missing generated wrapped root"))?;
        if !matches!(root.opcode.as_str(), "call" | "fusion") {
            continue;
        }
        let callee = root
            .called_computation_ids
            .first()
            .and_then(|id| module.computations.iter().find(|c| c.id == *id))
            .ok_or_else(|| Error::new("async", "wrapped call/fusion requires a callee"))?;
        let mut parameters = callee
            .instructions
            .iter()
            .filter(|i| i.opcode == "parameter")
            .collect::<Vec<_>>();
        parameters.sort_by_key(|i| i.parameter_number);
        let operands = parameters
            .iter()
            .map(|i| Shape::from_proto(i.shape.get_or_default().clone()))
            .collect::<Result<Vec<_>>>()?;
        let result = callee
            .instructions
            .iter()
            .find(|i| i.id == callee.root_id)
            .ok_or_else(|| Error::new("async", "missing callee root"))?;
        let result = Shape::from_proto(result.shape.get_or_default().clone())?;
        update_signature(module, id, &operands, &result, true, next)?;
    }
    Ok(())
}

pub(super) fn bind_sugar(mut module: Module, sugar: &BTreeMap<i64, Opcode>) -> Result<Module> {
    if sugar.is_empty() {
        return Ok(module);
    }
    let mut next = module
        .wire
        .computations
        .iter()
        .flat_map(|c| &c.instructions)
        .map(|i| i.id)
        .max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(|| Error::new("overflow", "instruction count overflow"))?;
    let mut updates = Vec::new();
    for (&id, &expected) in sugar {
        let owner = module
            .wire
            .computations
            .iter()
            .find(|c| c.instructions.iter().any(|i| i.id == id))
            .ok_or_else(|| Error::new("async", "missing async sugar owner"))?
            .id;
        let handle = module.instruction_handle(owner, id);
        let instruction = module.instruction(handle)?;
        if instruction.opcode == "async-start" {
            continue;
        }
        let Some(root) = wrapped_root(&module, handle)? else {
            continue;
        };
        let body = module
            .wire
            .computations
            .iter()
            .find(|c| c.root_id == root)
            .ok_or_else(|| Error::new("async", "missing async wrapped root"))?;
        let actual = body
            .instructions
            .iter()
            .find(|i| i.id == root)
            .ok_or_else(|| Error::new("async", "missing async wrapped opcode"))?;
        if actual.opcode != expected.name() {
            return Err(Error::new(
                "async",
                "async wrapped opcode differs from chain start",
            ));
        }
        let shape = module.shape(handle)?;
        let (operands, result) = if instruction.opcode == "async-update" {
            context_shapes(&shape)?
        } else {
            let mut current = module.operands(handle)?.first().copied();
            let mut seen = BTreeSet::new();
            let mut start = None;
            while let Some(id) = current {
                if !seen.insert(id) {
                    break;
                }
                let Some(producer) = crate::verification::asynchronous::producer(&module, id)?
                else {
                    break;
                };
                if module.instruction(producer)?.opcode == "async-start" {
                    start = Some(producer);
                    break;
                }
                current = module.operands(producer)?.first().copied();
            }
            let operand = module
                .operands(handle)?
                .first()
                .copied()
                .ok_or_else(|| Error::new("async", "async-done requires an operand"))?;
            let context = module.shape(start.unwrap_or(operand))?;
            (context_shapes(&context)?.0, shape)
        };
        updates.push((body.id, operands, result));
    }
    for (id, operands, result) in updates {
        update_signature(&mut module.wire, id, &operands, &result, false, &mut next)?;
    }
    let config = module.config();
    let channels = module.channel_presence;
    let mut module = Module::from_proto_with_config(module.wire, config)?;
    module.channel_presence = channels;
    Ok(module)
}

fn wrapped_root(module: &Module, mut current: crate::InstructionId) -> Result<Option<i64>> {
    let mut visited = BTreeSet::new();
    while visited.insert(current) {
        let instruction = module.instruction(current)?;
        if !matches!(
            instruction.opcode.as_str(),
            "async-start" | "async-update" | "async-done"
        ) {
            break;
        }
        if let Some(&callee) = instruction.called_computation_ids.first() {
            return Ok(Some(
                module
                    .computation(module.computation_handle(callee))?
                    .root_id,
            ));
        }
        let Some(&operand) = module.operands(current)?.first() else {
            break;
        };
        let Some(producer) = crate::verification::asynchronous::producer(module, operand)? else {
            break;
        };
        current = producer;
    }
    Ok(None)
}

pub(super) fn print_policy(module: &Module, wire: &HloModuleProto) -> Result<BTreeSet<i64>> {
    let mut handles = BTreeMap::new();
    for computation in module.computations() {
        for id in module.instructions(computation)? {
            handles.insert(module.instruction(id)?.id, id);
        }
    }
    let instructions = wire
        .computations
        .iter()
        .flat_map(|c| &c.instructions)
        .map(|i| (i.id, i))
        .collect::<BTreeMap<_, _>>();
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for computation in &wire.computations {
        for instruction in &computation.instructions {
            if !matches!(
                instruction.opcode.as_str(),
                "async-start" | "async-update" | "async-done"
            ) {
                continue;
            }
            let id = handles[&instruction.id];
            if let Some(root) = wrapped_root(module, id)? {
                groups.entry(root).or_default().push(instruction.id);
            } else if !instruction.backend_config.is_empty()
                || instruction.frontend_attributes.is_some()
            {
                return Err(Error::new(
                    "unsupported",
                    "cannot print shared async attributes without a wrapped computation",
                ));
            }
            if instruction.opcode == "async-start" {
                let mut pending = instruction.called_computation_ids.clone();
                let mut visited = BTreeSet::new();
                while let Some(callee) = pending.pop() {
                    if !visited.insert(callee) {
                        continue;
                    }
                    let callee = module.computation(module.computation_handle(callee))?;
                    if thread(&callee.execution_thread)
                        != thread(&instruction.async_execution_thread)
                        || callee.instructions.iter().any(|i| {
                            i.opcode == "async-start"
                                && thread(&i.async_execution_thread)
                                    != thread(&instruction.async_execution_thread)
                        })
                    {
                        return Err(Error::new(
                            "unsupported",
                            "async thread propagation would change raw computation metadata",
                        ));
                    }
                    pending.extend(
                        callee
                            .instructions
                            .iter()
                            .flat_map(|i| i.called_computation_ids.iter().copied()),
                    );
                }
            }
        }
    }
    let mut empty_backend = BTreeSet::new();
    for (root, group) in groups {
        let root = instructions[&root];
        if let Some(last) = group
            .iter()
            .rev()
            .map(|id| instructions[id])
            .find(|i| i.statistics_viz.is_some())
            && !metadata::same_statistics(
                last.statistics_viz.as_ref(),
                root.statistics_viz.as_ref(),
            )
        {
            return Err(Error::new(
                "unsupported",
                "shared async statistics would change the wrapped root",
            ));
        }
        if let Some(last) = group
            .iter()
            .rev()
            .map(|id| instructions[id])
            .find(|i| i.metadata.is_some())
            && last.metadata != root.metadata
        {
            return Err(Error::new(
                "unsupported",
                "shared async metadata would change the wrapped root",
            ));
        }
        if let Some(last) = group
            .iter()
            .rev()
            .map(|id| instructions[id])
            .find(|i| i.original_value.is_some())
            && last.original_value != root.original_value
        {
            return Err(Error::new(
                "unsupported",
                "shared async original value would change the wrapped root",
            ));
        }
        if let Some(last) = group
            .iter()
            .rev()
            .map(|id| instructions[id])
            .find(|i| i.frontend_attributes.is_some())
            && last.frontend_attributes != root.frontend_attributes
        {
            return Err(Error::new(
                "unsupported",
                "shared async frontend attributes would change the wrapped root",
            ));
        }
        if let Some(last) = group
            .iter()
            .rev()
            .map(|id| instructions[id])
            .find(|i| !i.backend_config.is_empty())
            && last.backend_config != root.backend_config
        {
            let &last_id = group
                .last()
                .ok_or_else(|| Error::new("async", "empty async attribute group"))?;
            if root.backend_config.is_empty() && instructions[&last_id].backend_config.is_empty() {
                empty_backend.insert(last_id);
            } else {
                return Err(Error::new(
                    "unsupported",
                    "shared async backend configuration would change the wrapped root",
                ));
            }
        }
    }
    Ok(empty_backend)
}

pub(super) fn normalize(mut module: Module, backend_attributes: &BTreeSet<i64>) -> Result<Module> {
    let ids = module
        .computations()
        .map(|c| module.instructions(c))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut inherited = Vec::new();
    for &id in &ids {
        let instruction = module.instruction(id)?;
        if !matches!(instruction.opcode.as_str(), "async-update" | "async-done") {
            continue;
        }
        let operands = module.operands(id)?;
        let &operand = operands
            .first()
            .ok_or_else(|| Error::new("async", "async-update/done requires an operand"))?;
        if module.instruction(operand)?.opcode == "async-done" {
            return Err(Error::new(
                "async",
                "async-done cannot be an async-update/done predecessor",
            ));
        }
        let mut current = crate::verification::asynchronous::producer(&module, operand)?;
        let mut visited = BTreeSet::new();
        while let Some(producer) = current {
            if !visited.insert(producer) {
                break;
            }
            let value = module.instruction(producer)?;
            if value.opcode == "async-start" {
                if !instruction.called_computation_ids.is_empty()
                    && instruction.called_computation_ids != value.called_computation_ids
                {
                    return Err(Error::new(
                        "async",
                        "explicit wrapped computation differs from inherited computation",
                    ));
                }
                if !instruction.async_execution_thread.is_empty()
                    && instruction.async_execution_thread != value.async_execution_thread
                {
                    return Err(Error::new(
                        "async",
                        "explicit async thread differs from inherited thread",
                    ));
                }
                inherited.push(id);
                break;
            }
            if value.opcode != "async-update" {
                break;
            }
            current = match module.operands(producer)?.first() {
                Some(&operand) => crate::verification::asynchronous::producer(&module, operand)?,
                None => None,
            };
        }
    }
    for id in inherited {
        let owner = module
            .computations()
            .find(|&c| module.instructions(c).is_ok_and(|v| v.contains(&id)))
            .ok_or_else(|| Error::new("async", "missing instruction owner"))?;
        let wire = module.instruction(id)?.id;
        let computation = module.computation(owner)?.id;
        let instruction = module
            .wire
            .computations
            .iter_mut()
            .find(|c| c.id == computation)
            .and_then(|c| c.instructions.iter_mut().find(|i| i.id == wire))
            .ok_or_else(|| Error::new("async", "missing inherited async instruction"))?;
        instruction.called_computation_ids.clear();
        instruction.async_execution_thread.clear();
    }
    for &id in &ids {
        let instruction = module.instruction(id)?;
        if instruction.opcode != "async-start" {
            continue;
        }
        if instruction.called_computation_ids.len() != 1 {
            return Err(Error::new(
                "async",
                "async-start requires one wrapped computation",
            ));
        }
        let thread = instruction.async_execution_thread.clone();
        let mut work = instruction.called_computation_ids.clone();
        let mut visited = BTreeSet::new();
        while let Some(computation) = work.pop() {
            if !visited.insert(computation) {
                continue;
            }
            let computation = module
                .wire
                .computations
                .iter_mut()
                .find(|c| c.id == computation)
                .ok_or_else(|| Error::new("async", "missing wrapped computation"))?;
            computation.execution_thread = thread.clone();
            for instruction in &mut computation.instructions {
                if instruction.opcode == "async-start" {
                    instruction.async_execution_thread = thread.clone();
                }
                work.extend(&instruction.called_computation_ids);
            }
        }
    }
    for id in ids {
        let instruction = module.instruction(id)?;
        let has_backend = backend_attributes.contains(&instruction.id);
        let frontend = instruction.frontend_attributes.clone();
        let backend = instruction.backend_config.clone();
        let metadata = instruction.metadata.clone();
        let original = instruction.original_value.clone();
        let statistics = instruction.statistics_viz.clone();
        if !matches!(
            instruction.opcode.as_str(),
            "async-start" | "async-update" | "async-done"
        ) || !has_backend
            && frontend.is_none()
            && metadata.is_none()
            && original.is_none()
            && statistics.is_none()
        {
            continue;
        }
        if let Some(root) = wrapped_root(&module, id)? {
            let root = module
                .wire
                .computations
                .iter_mut()
                .flat_map(|c| &mut c.instructions)
                .find(|i| i.id == root)
                .ok_or_else(|| Error::new("async", "missing wrapped root"))?;
            if has_backend {
                root.backend_config = backend;
            }
            if frontend.is_some() {
                root.frontend_attributes = frontend;
            }
            if metadata.is_some() {
                root.metadata = metadata;
            }
            if original.is_some() {
                root.original_value = original;
            }
            if statistics.is_some() {
                root.statistics_viz = statistics;
            }
        }
    }
    let wrapped = module
        .wire
        .computations
        .iter()
        .flat_map(|c| &c.instructions)
        .filter(|i| {
            matches!(
                i.opcode.as_str(),
                "async-start" | "async-update" | "async-done"
            )
        })
        .flat_map(|i| i.called_computation_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if module
        .wire
        .computations
        .iter()
        .flat_map(|c| &c.instructions)
        .any(|i| {
            !matches!(
                i.opcode.as_str(),
                "async-start" | "async-update" | "async-done"
            ) && i.called_computation_ids.iter().any(|c| wrapped.contains(c))
        })
    {
        return Err(Error::new(
            "async",
            "async computation is called by a non-async instruction",
        ));
    }
    Ok(module)
}
