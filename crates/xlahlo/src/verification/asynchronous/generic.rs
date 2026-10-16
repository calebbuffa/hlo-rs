use super::{BTreeMap, BTreeSet, Dataflow};
use crate::ir::execution_thread as thread;
use crate::{
    ComputationId, Error, InstructionId, Module, Result, Shape, VerifyOptions,
    proto::xla_data::ShapeProto,
};

fn asynchronous(opcode: &str) -> bool {
    matches!(opcode, "async-start" | "async-update" | "async-done")
}

fn error(message: &str) -> Error {
    Error::new("async", message)
}

fn same(a: &Shape, b: &Shape, options: VerifyOptions) -> bool {
    if options.layout_sensitive {
        crate::verification::same_shapes(a, b, crate::verification::LayoutComparison::Exact)
    } else {
        a.compatible(b)
    }
}

fn prefix(a: &Shape, b: &Shape, options: VerifyOptions) -> Result<bool> {
    let mut work = vec![(a.clone(), b.clone())];
    while let Some((a, b)) = work.pop() {
        if same(&a, &b, options) || a.is_tuple() && a.as_proto().tuple_shapes.is_empty() {
            continue;
        }
        if !a.is_tuple()
            || !b.is_tuple()
            || a.as_proto().tuple_shapes.len() > b.as_proto().tuple_shapes.len()
        {
            return Ok(false);
        }
        for (a, b) in a
            .as_proto()
            .tuple_shapes
            .iter()
            .zip(&b.as_proto().tuple_shapes)
        {
            work.push((Shape::from_proto(a.clone())?, Shape::from_proto(b.clone())?));
        }
    }
    Ok(true)
}

fn context(shape: &Shape) -> Result<(Shape, Shape)> {
    let children = &shape.as_proto().tuple_shapes;
    if !shape.is_tuple() || children.len() < 2 {
        return Err(error(
            "context requires ((bound operands...), bound result, states...)",
        ));
    }
    let parameters = Shape::from_proto(children[0].clone())?;
    if !parameters.is_tuple() {
        return Err(error("context child zero must be a tuple"));
    }
    Ok((parameters, Shape::from_proto(children[1].clone())?))
}

impl Dataflow<'_> {
    fn chain_start(&self, from: InstructionId) -> Result<Option<InstructionId>> {
        let mut current = from;
        let mut visited = BTreeSet::new();
        while visited.insert(current) {
            let instruction = self.module.instruction(current)?;
            if instruction.opcode == "async-start" {
                return Ok(Some(current));
            }
            if !asynchronous(&instruction.opcode) {
                return Ok(None);
            }
            let Some(&operand) = self.module.operands(current)?.first() else {
                return Ok(None);
            };
            let Some(producer) = self.producer(operand)? else {
                return Ok(None);
            };
            current = producer;
        }
        Ok(None)
    }

    fn wrapped(&self, from: InstructionId) -> Result<ComputationId> {
        let mut current = from;
        let mut visited = BTreeSet::new();
        while visited.insert(current) {
            let instruction = self.module.instruction(current)?;
            if !asynchronous(&instruction.opcode) {
                break;
            }
            if let Some(&callee) = instruction.called_computation_ids.first() {
                if instruction.called_computation_ids.len() != 1 {
                    return Err(error("async instruction has multiple wrapped computations"));
                }
                return Ok(self.module.computation_handle(callee));
            }
            let Some(&operand) = self.module.operands(current)?.first() else {
                break;
            };
            let Some(producer) = self.producer(operand)? else {
                break;
            };
            current = producer;
        }
        Err(error("cannot resolve a wrapped computation"))
    }

    fn carried(&self, from: InstructionId) -> Result<bool> {
        let mut work = vec![from];
        let mut visited = BTreeSet::new();
        while let Some(id) = work.pop() {
            if !visited.insert(id) {
                continue;
            }
            let owner = self.owners[&id];
            if id == self.module.root(owner)? {
                let wire = self.module.computation(owner)?.id;
                for computation in self.module.computations() {
                    for caller in self.module.instructions(computation)? {
                        let value = self.module.instruction(caller)?;
                        if !value.called_computation_ids.contains(&wire) {
                            continue;
                        }
                        if value.opcode == "while"
                            && value.called_computation_ids.first() == Some(&wire)
                        {
                            return Ok(true);
                        }
                        if matches!(value.opcode.as_str(), "call" | "conditional") {
                            work.push(caller);
                        }
                    }
                }
            }
            for user in self.module.users(id)? {
                match self.module.instruction(user)?.opcode.as_str() {
                    "while" => return Ok(true),
                    "async-update" | "tuple" | "get-tuple-element" | "opt-barrier" | "domain"
                    | "call" | "conditional" => work.push(user),
                    _ => {}
                }
            }
        }
        Ok(false)
    }
}

fn aliases(
    module: &Module,
    id: InstructionId,
    parameters: &[Shape],
    result: &Shape,
    options: VerifyOptions,
) -> Result<()> {
    let instruction = module.instruction(id)?;
    if instruction.opcode == "async-done" {
        return Ok(());
    }
    let declared = module.shape(id)?;
    let alias_output = if instruction.opcode == "async-start" {
        let mut children = vec![Shape::tuple(parameters.to_vec())?, result.clone()];
        for shape in &declared.as_proto().tuple_shapes[2..] {
            children.push(Shape::from_proto(shape.clone())?);
        }
        Shape::tuple(children)?
    } else {
        declared
    };
    let predecessor = if instruction.opcode == "async-update" {
        Some(module.shape(module.operands(id)?[0])?)
    } else {
        None
    };
    fn subshape(shape: &Shape, index: &[i64]) -> Result<Shape> {
        let index = index
            .iter()
            .map(|&v| usize::try_from(v).map_err(|_| error("negative alias shape index")))
            .collect::<Result<Vec<_>>>()?;
        shape.subshape(&index)
    }
    fn normalize(shape: &ShapeProto, layout_sensitive: bool) -> ShapeProto {
        let mut normalized = shape.clone();
        let mut pending = vec![&mut normalized];
        while let Some(shape) = pending.pop() {
            if shape.element_type.enum_value() == Ok(crate::PrimitiveType::BUFFER) {
                *shape = shape.array_storage().into_owned();
            }
            if !layout_sensitive {
                shape.is_dynamic_dimension.clear();
            }
            pending.extend(&mut shape.tuple_shapes);
        }
        normalized
    }
    for alias in &instruction.output_operand_aliasing {
        let operand = if let Some(predecessor) = &predecessor {
            if alias.operand_index != 0
                || alias.operand_shape_index.first() != Some(&2)
                || alias.output_shape_index.first() != Some(&2)
            {
                return Err(error(
                    "async-update aliases may only connect context child two",
                ));
            }
            predecessor
        } else {
            if alias.output_shape_index.is_empty() {
                return Err(error("async-start alias output index must not be empty"));
            }
            usize::try_from(alias.operand_index)
                .ok()
                .and_then(|i| parameters.get(i))
                .ok_or_else(|| error("async-start alias parameter number is out of range"))?
        };
        let operand = Shape::from_proto(normalize(
            subshape(operand, &alias.operand_shape_index)?.as_proto(),
            options.layout_sensitive,
        ))?;
        let output = Shape::from_proto(normalize(
            subshape(&alias_output, &alias.output_shape_index)?.as_proto(),
            options.layout_sensitive,
        ))?;
        if if options.layout_sensitive {
            !same(&operand, &output, options)
        } else {
            !operand.equal_ignoring_layout(&output)
        } {
            return Err(error("alias operand and output shapes differ"));
        }
    }
    Ok(())
}

pub(super) fn verify(module: &Module, options: VerifyOptions) -> Result<()> {
    let dataflow = Dataflow::new(module)?;
    let mut starts = Vec::new();
    let mut updates = Vec::new();
    let mut dones = Vec::new();
    let mut wrapped = BTreeSet::new();
    for computation in module.computations() {
        for id in module.instructions(computation)? {
            let instruction = module.instruction(id)?;
            for (position, operand) in module.operands(id)?.into_iter().enumerate() {
                if !matches!(
                    module.instruction(operand)?.opcode.as_str(),
                    "async-start" | "async-update"
                ) {
                    continue;
                }
                let allowed =
                    if matches!(instruction.opcode.as_str(), "async-update" | "async-done") {
                        position == 0
                    } else {
                        matches!(
                            instruction.opcode.as_str(),
                            "call" | "conditional" | "domain"
                        ) || super::super::layout::allowed_async_intermediary(instruction)
                    };
                if !allowed {
                    return Err(error(
                        "async producers are only allowed as chaining operand zero or through approved dataflow",
                    ));
                }
            }
            if !asynchronous(&instruction.opcode) {
                continue;
            }
            match instruction.opcode.as_str() {
                "async-start" => {
                    if instruction.called_computation_ids.len() != 1 {
                        return Err(error("async-start requires one wrapped computation"));
                    }
                    starts.push(id);
                }
                "async-update" => updates.push(id),
                _ => dones.push(id),
            }
            let callee = dataflow.wrapped(id)?;
            wrapped.insert(callee);
            if module.computation(callee)?.is_fusion_computation {
                return Err(error(
                    "async wrapped computation cannot be a fusion computation",
                ));
            }
            let start = dataflow.chain_start(id)?;
            let execution_thread = if let Some(start) = start {
                thread(&module.instruction(start)?.async_execution_thread)
            } else {
                thread(&module.computation(callee)?.execution_thread)
            };
            if thread(&module.computation(callee)?.execution_thread) != execution_thread
                || !instruction.async_execution_thread.is_empty()
                    && thread(&instruction.async_execution_thread) != execution_thread
            {
                return Err(error(
                    "async execution thread differs from wrapped computation or chain",
                ));
            }
            let program = module.program_shape(callee)?;
            let parameters = program
                .parameters
                .iter()
                .cloned()
                .map(Shape::from_proto)
                .collect::<Result<Vec<_>>>()?;
            let result = Shape::from_proto(
                program
                    .result
                    .as_ref()
                    .ok_or_else(|| error("wrapped computation has no result"))?
                    .clone(),
            )?;
            let declared = module.shape(id)?;
            let operands = module.operands(id)?;
            if instruction.opcode == "async-done" {
                if operands.len() != 1 || !same(&declared, &result, options) {
                    return Err(error(
                        "async-done requires one operand and the full computation result shape",
                    ));
                }
            } else {
                let (bound, output) = context(&declared)?;
                if !prefix(&bound, &Shape::tuple(parameters.clone())?, options)?
                    || !prefix(&output, &result, options)?
                {
                    return Err(error(
                        "bound parameters and result must be prefixes of wrapped computation shapes",
                    ));
                }
                if instruction.opcode == "async-start" {
                    if operands.len() != bound.as_proto().tuple_shapes.len() {
                        return Err(error(
                            "async-start operand and bound parameter counts differ",
                        ));
                    }
                    for (&operand, expected) in operands.iter().zip(&bound.as_proto().tuple_shapes)
                    {
                        if !same(
                            &module.shape(operand)?,
                            &Shape::from_proto(expected.clone())?,
                            options,
                        ) {
                            return Err(error(
                                "async-start operand differs from bound parameter shape",
                            ));
                        }
                    }
                } else {
                    let &previous = operands
                        .first()
                        .ok_or_else(|| error("async-update needs a predecessor"))?;
                    let (previous_bound, previous_output) = context(&module.shape(previous)?)?;
                    if !prefix(&previous_bound, &bound, options)?
                        || !prefix(&previous_output, &output, options)?
                    {
                        return Err(error(
                            "async-update must extend predecessor parameter/result prefixes",
                        ));
                    }
                    let previous_count = previous_bound.as_proto().tuple_shapes.len();
                    let new_shapes = bound
                        .as_proto()
                        .tuple_shapes
                        .get(previous_count..)
                        .ok_or_else(|| error("async-update removes bound parameters"))?;
                    if operands.len() - 1 != new_shapes.len() {
                        return Err(error(
                            "async-update new operands do not match newly bound parameters",
                        ));
                    }
                    for (&operand, shape) in operands[1..].iter().zip(new_shapes) {
                        if !same(
                            &module.shape(operand)?,
                            &Shape::from_proto(shape.clone())?,
                            options,
                        ) {
                            return Err(error(
                                "async-update new operand shape differs from newly bound parameter",
                            ));
                        }
                    }
                }
            }
            if instruction.opcode != "async-start" {
                let &operand = operands
                    .first()
                    .ok_or_else(|| error("missing async predecessor"))?;
                let producer = dataflow
                    .producer(operand)?
                    .ok_or_else(|| error("operand does not trace to an async producer"))?;
                if !matches!(
                    module.instruction(producer)?.opcode.as_str(),
                    "async-start" | "async-update"
                ) || dataflow.wrapped(producer)? != callee
                {
                    return Err(error("predecessor has a different wrapped computation"));
                }
                if instruction.opcode == "async-done" {
                    let start = start.ok_or_else(|| error("async-done has no chain start"))?;
                    let expected = module.program_shape(dataflow.wrapped(start)?)?.parameters;
                    let expected = Shape::tuple(
                        expected
                            .into_iter()
                            .map(Shape::from_proto)
                            .collect::<Result<_>>()?,
                    )?;
                    let (bound, _) = context(&module.shape(producer)?)?;
                    if !expected.compatible(&bound) {
                        return Err(error(
                            "async-done predecessor parameters are not fully bound",
                        ));
                    }
                }
            }
            aliases(module, id, &parameters, &result, options)?;
        }
    }
    for computation in &wrapped {
        let body = module.computation(*computation)?;
        let wire = body.id;
        let mut callers = module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .filter(|i| i.opcode == "async-start" && i.called_computation_ids.contains(&wire));
        let grouped = callers.next().is_some_and(|caller| {
            caller
                .frontend_attributes
                .as_ref()
                .is_some_and(|attributes| attributes.map.contains_key("_collectives_group"))
        }) && callers.next().is_none();
        if thread(&body.execution_thread) == "main"
            && !grouped
            && body.instructions.iter().any(|instruction| {
                instruction.id != body.root_id
                    && !matches!(
                        instruction.opcode.as_str(),
                        "parameter" | "get-tuple-element"
                    )
            })
        {
            return Err(error(
                "main-thread async computation requires only root, parameter and get-tuple-element instructions unless marked as a collectives group",
            ));
        }
        for caller in module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
        {
            if caller.called_computation_ids.contains(&wire) && !asynchronous(&caller.opcode) {
                return Err(error(
                    "async wrapped computation is called by a non-async instruction",
                ));
            }
        }
    }
    let mut matched = BTreeMap::new();
    let mut visited_updates = BTreeSet::new();
    for done in dones {
        let mut producer = dataflow.producer(module.operands(done)?[0])?;
        while let Some(id) = producer {
            if module.instruction(id)?.opcode != "async-update" {
                break;
            }
            if !visited_updates.insert(id) {
                return Err(error(
                    "async-update belongs to multiple done chains or a cycle",
                ));
            }
            let operands = module.operands(id)?;
            let &operand = operands
                .first()
                .ok_or_else(|| error("async-update has no operand"))?;
            producer = dataflow.producer(operand)?;
        }
        let start = producer
            .filter(|&id| {
                module
                    .instruction(id)
                    .is_ok_and(|i| i.opcode == "async-start")
            })
            .ok_or_else(|| error("async-done does not trace back to async-start"))?;
        if matched.insert(start, done).is_some() {
            return Err(error("async-start has multiple matching async-dones"));
        }
    }
    for start in starts {
        if !matched.contains_key(&start) && !dataflow.carried(start)? {
            return Err(error("async-start has no matching async-done"));
        }
    }
    for update in updates {
        if !visited_updates.contains(&update) && !dataflow.carried(update)? {
            return Err(error("orphan async-update"));
        }
    }
    Ok(())
}
