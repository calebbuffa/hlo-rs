use crate::{ComputationId, Error, InstructionId, Module, Result};
use std::collections::{BTreeMap, BTreeSet};

mod generic;
pub(super) fn verify(module: &Module, options: crate::VerifyOptions) -> Result<()> {
    generic::verify(module, options)
}

pub(crate) fn producer(module: &Module, from: InstructionId) -> Result<Option<InstructionId>> {
    Dataflow::new(module)?.producer(from)
}

struct Dataflow<'a> {
    module: &'a Module,
    owners: BTreeMap<InstructionId, ComputationId>,
    while_callers: BTreeMap<ComputationId, Vec<InstructionId>>,
}

impl<'a> Dataflow<'a> {
    fn new(module: &'a Module) -> Result<Self> {
        let mut owners = BTreeMap::new();
        let mut while_callers: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for computation in module.computations() {
            for id in module.instructions(computation)? {
                owners.insert(id, computation);
                let instruction = module.instruction(id)?;
                if instruction.opcode == "while" {
                    for &callee in &instruction.called_computation_ids {
                        let callers = while_callers
                            .entry(module.computation_handle(callee))
                            .or_default();
                        if !callers.contains(&id) {
                            callers.push(id);
                        }
                    }
                }
            }
        }
        Ok(Self {
            module,
            owners,
            while_callers,
        })
    }

    fn producer(&self, from: InstructionId) -> Result<Option<InstructionId>> {
        enum Work {
            Enter(InstructionId, Vec<i64>),
            Leave(InstructionId),
        }
        let mut pending = vec![Work::Enter(from, vec![])];
        let mut visited = BTreeSet::new();
        while let Some(work) = pending.pop() {
            let (id, mut index) = match work {
                Work::Leave(id) => {
                    visited.remove(&id);
                    continue;
                }
                Work::Enter(id, index) => (id, index),
            };
            if !visited.insert(id) {
                continue;
            }
            let instruction = self.module.instruction(id)?;
            if matches!(
                instruction.opcode.as_str(),
                "async-start"
                    | "async-update"
                    | "all-gather-start"
                    | "all-reduce-start"
                    | "collective-permute-start"
            ) {
                return Ok(Some(id));
            }
            let operands = self.module.operands(id)?;
            let mut next = Vec::new();
            match instruction.opcode.as_str() {
                "get-tuple-element" => {
                    index.push(instruction.tuple_index);
                    if let Some(&operand) = operands.first() {
                        next.push((operand, index));
                    }
                }
                "tuple" | "opt-barrier" => {
                    if instruction.opcode == "opt-barrier" && operands.len() == 1 {
                        next.push((operands[0], index));
                    } else if let Some(position) = index.pop() {
                        if let Ok(position) = usize::try_from(position)
                            && let Some(&operand) = operands.get(position)
                        {
                            next.push((operand, index));
                        }
                    } else {
                        next.extend(operands.into_iter().map(|operand| (operand, index.clone())));
                    }
                }
                "copy" | "custom-call"
                    if instruction.opcode == "copy"
                        || super::layout::allowed_async_intermediary(instruction) =>
                {
                    if let Some(&operand) = operands.first() {
                        next.push((operand, index));
                    }
                }
                "while" => {
                    if let Some(&body) = instruction.called_computation_ids.first() {
                        next.push((
                            self.module.root(self.module.computation_handle(body))?,
                            index.clone(),
                        ));
                    }
                    if let Some(&operand) = operands.first() {
                        next.push((operand, index));
                    }
                }
                "parameter" => {
                    let owner = self
                        .owners
                        .get(&id)
                        .ok_or_else(|| Error::new("async-pair", "missing parameter owner"))?;
                    if let Some(callers) = self.while_callers.get(owner)
                        && callers.len() == 1
                    {
                        let caller = self.module.instruction(callers[0])?;
                        if caller.called_computation_ids.first()
                            == Some(&self.module.computation(*owner)?.id)
                        {
                            if let Some(&initial) = self.module.operands(callers[0])?.first() {
                                next.push((initial, index.clone()));
                            }
                            next.push((self.module.root(*owner)?, index));
                        }
                    }
                }
                _ => {}
            }
            pending.push(Work::Leave(id));
            pending.extend(
                next.into_iter()
                    .rev()
                    .map(|(id, index)| Work::Enter(id, index)),
            );
        }
        Ok(None)
    }
}

pub(super) fn verify_pairs(module: &Module) -> Result<()> {
    let dataflow = Dataflow::new(module)?;
    let wrapped = module
        .as_proto()
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
    for computation in module.computations() {
        for id in module.instructions(computation)? {
            let instruction = module.instruction(id)?;
            let done = match instruction.opcode.as_str() {
                "copy-start" => Some("copy-done"),
                "send" => Some("send-done"),
                "recv" => Some("recv-done"),
                "all-gather-start" => Some("all-gather-done"),
                "all-reduce-start" => Some("all-reduce-done"),
                "collective-permute-start" => Some("collective-permute-done"),
                _ => None,
            };
            if let Some(done) = done {
                if matches!(instruction.opcode.as_str(), "send" | "recv")
                    && id == module.root(computation)?
                    && wrapped.contains(&module.computation(computation)?.id)
                {
                    continue;
                }
                let mut users = Vec::new();
                for user in module.users(id)? {
                    let value = module.instruction(user)?;
                    if !(value.opcode == "custom-call" && value.custom_call_target == "control_dep")
                    {
                        users.push(value);
                    }
                }
                if users.len() != 1
                    || !(users[0].opcode == done
                        || super::layout::allowed_async_intermediary(users[0]))
                {
                    return Err(Error::new(
                        "async-pair",
                        "asynchronous start requires exactly one allowed non-control_dep consumer",
                    )
                    .context(&instruction.name));
                }
            }
            let start = match instruction.opcode.as_str() {
                "copy-done" => Some("copy-start"),
                "send-done" => Some("send"),
                "recv-done" => Some("recv"),
                "all-gather-done" => Some("all-gather-start"),
                "all-reduce-done" => Some("all-reduce-start"),
                "collective-permute-done" => Some("collective-permute-start"),
                _ => None,
            };
            if let Some(start) = start {
                let operands = module.operands(id)?;
                let [operand] = operands.as_slice() else {
                    return Err(Error::new(
                        "async-pair",
                        "asynchronous done requires one operand",
                    ));
                };
                let direct = module.instruction(*operand)?;
                let permitted = direct.opcode == start
                    || matches!(instruction.opcode.as_str(), "send-done" | "recv-done")
                        && direct.opcode == "get-tuple-element";
                if !permitted
                    && !dataflow
                        .producer(*operand)?
                        .map(|producer| module.instruction(producer).map(|i| i.opcode == start))
                        .transpose()?
                        .unwrap_or(false)
                {
                    return Err(Error::new(
                        "async-pair",
                        "asynchronous done operand does not trace to its required producer",
                    )
                    .context(&instruction.name));
                }
            }
        }
    }
    Ok(())
}
