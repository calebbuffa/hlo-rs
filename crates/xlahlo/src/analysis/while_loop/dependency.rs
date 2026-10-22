//! Pinned scalar dependency tracing, not constant evaluation or HLO dataflow.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReachingParameter {
    number: usize,
    tuple_index: Option<usize>,
}

struct Tracer<'a, 'c> {
    view: &'a ModuleView<'c>,
    active: HashSet<(Ptr<Operation>, Option<usize>)>,
    scalar_active: HashSet<Ptr<Operation>>,
}

pub(super) fn induction_index(view: &ModuleView<'_>, parts: &Parts) -> Result<Option<usize>> {
    let ctx = view.ctx;
    if !is_opcode(ctx, parts.input, Opcode::Tuple)?
        || !is_opcode(ctx, parts.body_root, Opcode::Tuple)?
    {
        return Ok(None);
    }
    let mut tracer = Tracer {
        view,
        active: HashSet::new(),
        scalar_active: HashSet::new(),
    };
    let Some(indices) = tracer.indices(parts.condition_root, parts.condition_parameter)? else {
        return Ok(None);
    };
    if indices.len() != 1 {
        return Ok(None);
    }
    let Some(&index) = indices.iter().next() else {
        return Ok(None);
    };
    let update = operand(ctx, parts.body_root, index)?;
    let update = trace(ctx, update)?;
    let Some(body_indices) = tracer.indices(update, parts.body_parameter)? else {
        return Ok(None);
    };
    Ok((body_indices.len() == 1 && body_indices.contains(&index)).then_some(index))
}

fn scalar_shape(ctx: &Context, value: Value) -> Result<bool> {
    let shape = value.get_type(ctx);
    Ok(shape::is_array(ctx, shape) && shape::rank(ctx, shape)? == 0)
}

fn excluded(op: Opcode) -> bool {
    matches!(
        op,
        Opcode::AllGather
            | Opcode::AllGatherDone
            | Opcode::AllGatherStart
            | Opcode::AllReduce
            | Opcode::AllReduceDone
            | Opcode::AllReduceStart
            | Opcode::AllToAll
            | Opcode::CollectiveBroadcast
            | Opcode::CollectivePermute
            | Opcode::CollectivePermuteDone
            | Opcode::CollectivePermuteStart
            | Opcode::CollectiveReduce
            | Opcode::RaggedAllToAll
            | Opcode::ReduceScatter
            | Opcode::Send
            | Opcode::SendDone
            | Opcode::Recv
            | Opcode::RecvDone
            | Opcode::CustomCall
    )
}

impl Tracer<'_, '_> {
    fn scalar_op(&mut self, value: Value) -> Result<bool> {
        let ctx = self.view.ctx;
        if !scalar_shape(ctx, value)? {
            return Ok(false);
        }
        let Some(op) = value.defining_op() else {
            return Ok(true);
        };
        if excluded(opcode(ctx, op)?) {
            return Ok(false);
        }
        if self.scalar_active.len() >= 128 || !self.scalar_active.insert(op) {
            return Ok(false);
        }
        let result = (|| {
            let mut callees = Vec::new();
            if opcode(ctx, op)? == Opcode::Fusion {
                callees.push(op);
            } else if let Some(names) = optional_attr::<Callees>(ctx, op, "hlo_callees")? {
                for name in &names.0 {
                    callees.push(self.view.resolve(name)?);
                }
            }
            for callee in callees {
                for instruction in block(ctx, callee)?.deref(ctx).iter(ctx) {
                    if !Operation::is_op::<ReturnOp>(instruction, ctx)
                        && !self.scalar_op(instruction.deref(ctx).get_result(0))?
                    {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        })();
        self.scalar_active.remove(&op);
        result
    }

    fn indices(&mut self, output: Value, input: Value) -> Result<Option<HashSet<usize>>> {
        let ctx = self.view.ctx;
        let mut work = vec![output];
        let mut visited = HashSet::new();
        let mut indices = HashSet::new();
        while let Some(value) = work.pop() {
            if !visited.insert(value) {
                continue;
            }
            if value == input || is_opcode(ctx, value, Opcode::Parameter)? {
                return Ok(None);
            }
            let Some(op) = value.defining_op() else {
                return Ok(None);
            };
            match opcode(ctx, op)? {
                Opcode::GetTupleElement => {
                    if !scalar_shape(ctx, value)? {
                        return Ok(None);
                    }
                    let mut source = op.deref(ctx).get_operand(0);
                    while is_opcode(ctx, source, Opcode::Copy)? {
                        source = operand(ctx, source, 0)?;
                    }
                    if source == input {
                        indices.insert(attr::<Index>(ctx, op, "hlo_index")?.0 as usize);
                    } else if is_opcode(ctx, source, Opcode::Tuple)? {
                        // Upstream visits every tuple operand, even unselected ones.
                        work.extend(inputs(ctx, defining(source)?));
                    } else if is_opcode(ctx, source, Opcode::Call)?
                        || is_opcode(ctx, source, Opcode::Fusion)?
                    {
                        if !self.called(
                            defining(source)?,
                            input,
                            Some(attr::<Index>(ctx, op, "hlo_index")?.0 as usize),
                            false,
                            &mut work,
                            &mut indices,
                        )? {
                            return Ok(None);
                        }
                    } else {
                        return Ok(None);
                    }
                }
                Opcode::Copy => work.push(op.deref(ctx).get_operand(0)),
                Opcode::Call | Opcode::Fusion => {
                    if !self.scalar_op(value)?
                        || !self.called(op, input, None, true, &mut work, &mut indices)?
                    {
                        return Ok(None);
                    }
                }
                _ => {
                    if !self.scalar_op(value)? {
                        return Ok(None);
                    }
                    work.extend(inputs(ctx, op));
                }
            }
        }
        Ok((!indices.is_empty()).then_some(indices))
    }

    fn called(
        &mut self,
        caller: Ptr<Operation>,
        input: Value,
        tuple_index: Option<usize>,
        scalar_verified: bool,
        work: &mut Vec<Value>,
        indices: &mut HashSet<usize>,
    ) -> Result<bool> {
        let ctx = self.view.ctx;
        let owner = self.view.callee(caller)?;
        let Some(parameters) = self.reaching(owner, tuple_index, scalar_verified)? else {
            return Ok(false);
        };
        for parameter in parameters {
            let mut arg = caller.deref(ctx).get_operand(parameter.number);
            if let Some(index) = parameter.tuple_index {
                while is_opcode(ctx, arg, Opcode::Copy)? {
                    arg = operand(ctx, arg, 0)?;
                }
                if arg != input {
                    return Ok(false);
                }
                indices.insert(index);
            } else {
                work.push(arg);
            }
        }
        Ok(true)
    }

    fn reaching(
        &mut self,
        owner: Ptr<Operation>,
        tuple_index: Option<usize>,
        scalar_verified: bool,
    ) -> Result<Option<HashSet<ReachingParameter>>> {
        let key = (owner, tuple_index);
        if self.active.len() >= 128 || !self.active.insert(key) {
            return Ok(None);
        }
        let result = self.reaching_inner(owner, tuple_index, scalar_verified);
        self.active.remove(&key);
        result
    }

    fn reaching_inner(
        &mut self,
        owner: Ptr<Operation>,
        tuple_index: Option<usize>,
        scalar_verified: bool,
    ) -> Result<Option<HashSet<ReachingParameter>>> {
        let ctx = self.view.ctx;
        let mut work = vec![(root(ctx, owner)?, tuple_index)];
        let mut visited = HashSet::new();
        let mut parameters = HashSet::new();
        let block = block(ctx, owner)?;
        let mut bindings = HashMap::new();
        if Operation::is_op::<FusionOp>(owner, ctx) {
            for index in 0..block.deref(ctx).get_num_arguments() {
                bindings.insert(block.deref(ctx).get_argument(index), index);
            }
        } else {
            for op in block.deref(ctx).iter(ctx) {
                if Operation::is_op::<crate::ops::control::ParameterOp>(op, ctx) {
                    bindings.insert(
                        op.deref(ctx).get_result(0),
                        attr::<Index>(ctx, op, "hlo_index")?.0 as usize,
                    );
                }
            }
        }
        while let Some((value, tuple_index)) = work.pop() {
            if !visited.insert((value, tuple_index)) {
                continue;
            }
            if tuple_index.is_some() {
                if !shape::is_tuple(ctx, value.get_type(ctx)) {
                    return Ok(None);
                }
            } else if !scalar_shape(ctx, value)? {
                return Ok(None);
            }
            if let Some(&number) = bindings.get(&value) {
                parameters.insert(ReachingParameter {
                    number,
                    tuple_index,
                });
                continue;
            }
            let Some(op) = value.defining_op() else {
                return Ok(None);
            };
            match opcode(ctx, op)? {
                Opcode::Tuple => {
                    let Some(index) = tuple_index else {
                        return Ok(None);
                    };
                    work.push((op.deref(ctx).get_operand(index), None));
                }
                Opcode::GetTupleElement => work.push((
                    op.deref(ctx).get_operand(0),
                    Some(attr::<Index>(ctx, op, "hlo_index")?.0 as usize),
                )),
                Opcode::Copy => work.push((op.deref(ctx).get_operand(0), tuple_index)),
                Opcode::Call | Opcode::Fusion => {
                    let callee = self.view.callee(op)?;
                    let Some(inner) = self.reaching(callee, tuple_index, scalar_verified)? else {
                        return Ok(None);
                    };
                    for parameter in inner {
                        work.push((
                            op.deref(ctx).get_operand(parameter.number),
                            parameter.tuple_index,
                        ));
                    }
                }
                _ => {
                    if tuple_index.is_some() || (!scalar_verified && !self.scalar_op(value)?) {
                        return Ok(None);
                    }
                    work.extend(inputs(ctx, op).into_iter().map(|value| (value, None)));
                }
            }
        }
        Ok(Some(parameters))
    }
}
