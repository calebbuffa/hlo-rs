//! Stable dependency-driven schedule repair, following `hlo_schedule.cc`.
use super::{ControlPredecessors, InstructionName, InstructionSequence, Schedule};
use crate::{
    ops::{ComputationOp, ReturnOp},
    verification::{error, optional_attr, require},
};
use pliron::{
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    operation::Operation,
    result::Result,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// Repair an existing module schedule without replacing it with a DFS schedule.
///
/// Surviving ready instructions retain their previous relative order. An old
/// instruction whose dependencies changed is deferred, then emitted as soon as
/// those dependencies complete. Newly ready instructions use a FIFO worklist.
/// Like XLA, self control edges are dropped during repair. No schedule is
/// invented for an unscheduled module. Repairs all ordinary computations, as
/// the pinned TupleSimplifier's no-argument `HloSchedule::Update` does.
/// This is a graph edit, not a verifier; callers supply their verification
/// policy separately. Structural errors precede schedule/name mutations.
/// Name creation or control-edge removal invalidates the affected computation's
/// optional logical order; a schedule-only change preserves that order.
pub fn update_schedule(ctx: &Context, module: Ptr<Operation>) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "schedule update requires a module",
    )?;
    let Some(old) = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .as_deref()
        .cloned()
    else {
        return Ok(());
    };
    let block = module
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .ok_or_else(|| error("scheduled module has no body"))?;
    let mut computations = Vec::new();
    let mut symbols = HashMap::new();
    for operation in block.deref(ctx).iter(ctx) {
        let computation = Operation::get_op::<ComputationOp>(operation, ctx)
            .ok_or_else(|| error("scheduled module contains a non-computation"))?;
        let symbol = computation
            .get_attr_hlo_computation_name(ctx)
            .ok_or_else(|| error("scheduled computation has no symbol"))?
            .as_ref()
            .clone();
        require(
            symbols.insert(symbol.clone(), operation).is_none(),
            "duplicate computation symbol",
        )?;
        computations.push((symbol, computation.body(ctx)));
    }
    let mut previous = HashMap::new();
    for sequence in &old.0 {
        require(
            previous
                .insert(sequence.computation.clone(), sequence.instructions.clone())
                .is_none(),
            "duplicate computation schedule",
        )?;
    }
    let mut sequences = Vec::new();
    let mut names = HashSet::new();
    let mut all = vec![module];
    while let Some(operation) = all.pop() {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            require(
                names.insert(name.0.clone()),
                "duplicate module instruction name",
            )?;
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                if Operation::is_op::<crate::ops::control::FusionOp>(operation, ctx) {
                    for parameter in super::fusion_parameters(ctx, block)?.0 {
                        if let Some(name) = parameter.name {
                            require(names.insert(name), "duplicate module instruction name")?;
                        }
                    }
                }
                all.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    let mut generated_names = Vec::new();
    let mut dropped_self_edges = Vec::new();
    for (symbol, region) in computations {
        let block = region
            .deref(ctx)
            .get_entry_block()
            .ok_or_else(|| error("scheduled computation has no body"))?;
        let instructions: Vec<_> = block
            .deref(ctx)
            .iter(ctx)
            .filter(|&operation| !Operation::is_op::<ReturnOp>(operation, ctx))
            .collect();
        let mut instruction_names = Vec::new();
        let mut indices = HashMap::new();
        let mut by_name = HashMap::new();
        for (index, &operation) in instructions.iter().enumerate() {
            indices.insert(operation, index);
            let name = match optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                Some(name) => name.0.clone(),
                None => {
                    let mut suffix = generated_names.len();
                    let name = loop {
                        let candidate = format!("schedule_update.{suffix}");
                        if names.insert(candidate.clone()) {
                            break candidate;
                        }
                        suffix += 1;
                    };
                    generated_names.push((operation, name.clone()));
                    name
                }
            };
            require(
                by_name.insert(name.clone(), index).is_none(),
                "duplicate scheduled instruction name",
            )?;
            instruction_names.push(name);
        }
        let existing = previous.get(&symbol).cloned().unwrap_or_default();
        let existing_names: HashSet<_> = existing.iter().collect();
        require(
            existing_names.len() == existing.len(),
            "duplicate instruction in old schedule",
        )?;
        let mut indegree = vec![0; instructions.len()];
        let mut successors = vec![Vec::new(); instructions.len()];
        let mut ready = VecDeque::new();
        for (index, &operation) in instructions.iter().enumerate() {
            let ir = operation.deref(ctx);
            let mut predecessors = HashSet::new();
            for operand in ir.operands() {
                let producer = operand
                    .defining_op()
                    .and_then(|producer| indices.get(&producer).copied())
                    .ok_or_else(|| error("scheduled operand is outside its computation"))?;
                predecessors.insert(producer);
            }
            if let Some(controls) =
                optional_attr::<ControlPredecessors>(ctx, operation, "hlo_control_predecessors")?
            {
                let mut updated = controls.clone();
                updated.0.retain(|name| name != &instruction_names[index]);
                if updated != *controls {
                    dropped_self_edges.push((operation, updated));
                }
                for name in &controls.0 {
                    let predecessor = by_name
                        .get(name)
                        .copied()
                        .ok_or_else(|| error("scheduled control predecessor is missing"))?;
                    if predecessor != index {
                        predecessors.insert(predecessor);
                    }
                }
            }
            indegree[index] = predecessors.len();
            for predecessor in predecessors {
                successors[predecessor].push(index);
            }
            if indegree[index] == 0 && !existing_names.contains(&instruction_names[index]) {
                ready.push_back(index);
            }
        }
        let mut replay = Replay {
            indegree,
            successors,
            ready,
            existing: instruction_names
                .iter()
                .map(|name| existing_names.contains(name))
                .collect(),
            deferred: vec![false; instructions.len()],
            scheduled: vec![false; instructions.len()],
            output: Vec::new(),
        };
        replay.drain();
        for name in existing {
            let Some(&index) = by_name.get(&name) else {
                continue;
            };
            if replay.scheduled[index] {
                continue;
            }
            if replay.indegree[index] == 0 {
                replay.emit(index);
                replay.drain();
            } else {
                replay.deferred[index] = true;
            }
        }
        replay.drain();
        require(
            replay.output.len() == instructions.len(),
            "schedule update encountered a dependency cycle",
        )?;
        sequences.push(InstructionSequence {
            computation: symbol,
            instructions: replay
                .output
                .into_iter()
                .map(|index| instruction_names[index].clone())
                .collect(),
        });
    }
    let mut order_owners = HashSet::new();
    for operation in generated_names
        .iter()
        .map(|(operation, _)| operation)
        .chain(dropped_self_edges.iter().map(|(operation, _)| operation))
    {
        let owner = operation
            .deref(ctx)
            .get_parent_block()
            .and_then(|block| block.deref(ctx).get_parent_op(ctx))
            .ok_or_else(|| error("scheduled ordering body is detached"))?;
        require(
            Operation::is_op::<ComputationOp>(owner, ctx),
            "scheduled ordering owner is not a computation",
        )?;
        order_owners.insert(owner);
    }
    for owner in order_owners {
        owner
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&Identifier::new("hlo_instruction_order"));
    }
    for (operation, name) in generated_names {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name));
    }
    for (operation, controls) in dropped_self_edges {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_control_predecessors"), controls);
    }
    module
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_schedule"), Schedule(sequences));
    Ok(())
}

struct Replay {
    indegree: Vec<usize>,
    successors: Vec<Vec<usize>>,
    ready: VecDeque<usize>,
    existing: Vec<bool>,
    deferred: Vec<bool>,
    scheduled: Vec<bool>,
    output: Vec<usize>,
}

impl Replay {
    fn emit(&mut self, instruction: usize) {
        if self.scheduled[instruction] {
            return;
        }
        self.scheduled[instruction] = true;
        self.output.push(instruction);
        for &successor in &self.successors[instruction] {
            self.indegree[successor] -= 1;
            if self.indegree[successor] == 0
                && (!self.existing[successor] || self.deferred[successor])
            {
                self.ready.push_back(successor);
            }
        }
    }

    fn drain(&mut self) {
        while let Some(instruction) = self.ready.pop_front() {
            self.emit(instruction);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::Index,
        ops::{
            control::ParameterOp,
            elementwise::{AddOp, NegateOp},
            movement::CopyOp,
        },
        shape::Shape,
        types::ElementType,
    };
    use pliron::{
        basic_block::BasicBlock, builtin::types::FunctionType, op::Op, printable::Printable,
        r#type::TypeHandle, value::Value,
    };

    fn fixture(ctx: &mut Context) -> (Ptr<Operation>, Ptr<BasicBlock>, TypeHandle, Vec<Value>) {
        crate::register(ctx);
        let ty = Shape::array(ctx, ElementType::F32, vec![2]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("schedule")).get_operation();
        super::super::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let signature = FunctionType::get(ctx, vec![ty, ty], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new("main"), signature);
        computation.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let mut values = Vec::new();
        for index in 0..2 {
            let parameter = ParameterOp::new(ctx, ty, vec![], Index(index));
            parameter.get_operation().insert_at_back(block, ctx);
            name(
                ctx,
                parameter.get_operation(),
                if index == 0 { "p0" } else { "p1" },
            );
            values.push(parameter.result(ctx));
        }
        (module, block, ty, values)
    }

    fn name(ctx: &Context, operation: Ptr<Operation>, value: &str) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(value.into()));
    }

    fn schedule(ctx: &Context, module: Ptr<Operation>, names: &[&str]) {
        module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("main"),
                instructions: names.iter().map(|name| (*name).into()).collect(),
            }]),
        );
    }

    fn sequence(ctx: &Context, module: Ptr<Operation>) -> Vec<String> {
        optional_attr::<Schedule>(ctx, module, "hlo_schedule")
            .unwrap()
            .unwrap()
            .0[0]
            .instructions
            .clone()
    }

    fn ordered(ctx: &Context, block: Ptr<BasicBlock>) -> Ptr<Operation> {
        use super::super::instruction_order::{InstructionOrder, OrderPermutation, set_order};
        let owner = block.deref(ctx).get_parent_op(ctx).unwrap();
        set_order(
            ctx,
            owner,
            InstructionOrder {
                instructions: vec!["p1".into(), "p0".into()],
                users: vec![
                    OrderPermutation {
                        subject: "p0".into(),
                        members: vec![],
                    },
                    OrderPermutation {
                        subject: "p1".into(),
                        members: vec![],
                    },
                ],
                control_successors: vec![
                    OrderPermutation {
                        subject: "p0".into(),
                        members: vec![],
                    },
                    OrderPermutation {
                        subject: "p1".into(),
                        members: vec![],
                    },
                ],
                ..Default::default()
            },
        )
        .unwrap();
        owner
    }

    #[test]
    fn schedule_only_repair_preserves_rich_logical_order() {
        let ctx = &mut Context::new();
        let (module, block, _, values) = fixture(ctx);
        ReturnOp::new(ctx, values[0])
            .get_operation()
            .insert_at_back(block, ctx);
        let owner = ordered(ctx, block);
        let before = owner.deref(ctx).attributes.clone();
        schedule(ctx, module, &["stale", "p0", "p1"]);
        update_schedule(ctx, module).unwrap();
        assert!(owner.deref(ctx).attributes == before);
        assert_eq!(sequence(ctx, module), ["p0", "p1"]);
        crate::verification::verify(ctx, module).unwrap();
    }

    #[test]
    fn generated_names_and_removed_self_edges_invalidate_only_affected_order() {
        for generate_name in [false, true] {
            let ctx = &mut Context::new();
            let (module, block, _, values) = fixture(ctx);
            ReturnOp::new(ctx, values[0])
                .get_operation()
                .insert_at_back(block, ctx);
            let owner = ordered(ctx, block);
            let (other, other_block, _, other_values) = fixture(ctx);
            ReturnOp::new(ctx, other_values[0])
                .get_operation()
                .insert_at_back(other_block, ctx);
            let other_owner = ordered(ctx, other_block);
            let other_before = other.disp(ctx).to_string();
            let p0 = values[0].defining_op().unwrap();
            if generate_name {
                p0.deref_mut(ctx)
                    .attributes
                    .0
                    .remove(&Identifier::new("hlo_name"));
            } else {
                p0.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(vec!["p0".into()]),
                );
            }
            schedule(ctx, module, &["p0", "p1"]);
            update_schedule(ctx, module).unwrap();
            assert!(
                !owner
                    .deref(ctx)
                    .attributes
                    .0
                    .contains_key(&Identifier::new("hlo_instruction_order"))
            );
            assert!(
                other_owner
                    .deref(ctx)
                    .attributes
                    .0
                    .contains_key(&Identifier::new("hlo_instruction_order"))
            );
            assert_eq!(other.disp(ctx).to_string(), other_before);
            crate::verification::verify(ctx, module).unwrap();
        }
    }

    #[test]
    fn schedule_preflight_failure_preserves_ordering_metadata() {
        let ctx = &mut Context::new();
        let (module, block, _, values) = fixture(ctx);
        ReturnOp::new(ctx, values[0])
            .get_operation()
            .insert_at_back(block, ctx);
        let owner = ordered(ctx, block);
        for (value, name) in [(values[0], "p1"), (values[1], "p0")] {
            value.defining_op().unwrap().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(vec![name.into()]),
            );
        }
        schedule(ctx, module, &["p0", "p1"]);
        let before = module.disp(ctx).to_string();
        assert!(update_schedule(ctx, module).is_err());
        assert!(
            owner
                .deref(ctx)
                .attributes
                .0
                .contains_key(&Identifier::new("hlo_instruction_order"))
        );
        assert_eq!(module.disp(ctx).to_string(), before);
    }

    #[test]
    fn unscheduled_module_is_not_implicitly_scheduled() {
        let ctx = &mut Context::new();
        let (module, block, _, values) = fixture(ctx);
        ReturnOp::new(ctx, values[0])
            .get_operation()
            .insert_at_back(block, ctx);
        let before = module.disp(ctx).to_string();
        update_schedule(ctx, module).unwrap();
        assert_eq!(module.disp(ctx).to_string(), before);
    }

    #[test]
    fn update_adds_missing_live_computation_sequences_and_removes_dead_ones() {
        let ctx = &mut Context::new();
        let (module, block, _, values) = fixture(ctx);
        ReturnOp::new(ctx, values[0])
            .get_operation()
            .insert_at_back(block, ctx);
        module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("deleted"),
                instructions: vec!["deleted_instruction".into()],
            }]),
        );
        update_schedule(ctx, module).unwrap();
        let updated = optional_attr::<Schedule>(ctx, module, "hlo_schedule")
            .unwrap()
            .unwrap();
        assert_eq!(updated.0.len(), 1);
        assert_eq!(updated.0[0].computation, Identifier::new("main"));
        assert_eq!(updated.0[0].instructions, ["p0", "p1"]);
        crate::verification::verify(ctx, module).unwrap();
    }

    #[test]
    fn update_prunes_stale_names_and_schedules_new_nodes_without_dfs_reordering() {
        let ctx = &mut Context::new();
        let (module, block, ty, values) = fixture(ctx);
        let first = CopyOp::new(ctx, ty, vec![values[0]]);
        first.get_operation().insert_at_back(block, ctx);
        name(ctx, first.get_operation(), "first");
        let second = CopyOp::new(ctx, ty, vec![values[1]]);
        second.get_operation().insert_at_back(block, ctx);
        name(ctx, second.get_operation(), "second");
        let added = NegateOp::new(ctx, ty, vec![first.result(ctx)]);
        added.get_operation().insert_at_back(block, ctx);
        let sum = AddOp::new(ctx, ty, vec![added.result(ctx), second.result(ctx)]);
        sum.get_operation().insert_at_back(block, ctx);
        name(ctx, sum.get_operation(), "sum");
        ReturnOp::new(ctx, sum.result(ctx))
            .get_operation()
            .insert_at_back(block, ctx);
        schedule(
            ctx,
            module,
            &["p0", "p1", "removed", "first", "second", "sum"],
        );
        update_schedule(ctx, module).unwrap();
        let names = sequence(ctx, module);
        let added_name = optional_attr::<InstructionName>(ctx, added.get_operation(), "hlo_name")
            .unwrap()
            .unwrap()
            .0
            .clone();
        assert_eq!(names, ["p0", "p1", "first", &added_name, "second", "sum"]);
        crate::verification::verify(ctx, module).unwrap();
        let before = module.disp(ctx).to_string();
        update_schedule(ctx, module).unwrap();
        assert_eq!(module.disp(ctx).to_string(), before);
    }

    #[test]
    fn self_control_edges_are_removed_but_other_controls_still_defer_old_entries() {
        let ctx = &mut Context::new();
        let (module, block, ty, values) = fixture(ctx);
        values[0]
            .defining_op()
            .unwrap()
            .deref_mut(ctx)
            .attributes
            .set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(vec!["p0".into(), "p1".into()]),
            );
        let sum = AddOp::new(ctx, ty, values);
        sum.get_operation().insert_at_back(block, ctx);
        name(ctx, sum.get_operation(), "sum");
        ReturnOp::new(ctx, sum.result(ctx))
            .get_operation()
            .insert_at_back(block, ctx);
        schedule(ctx, module, &["p0", "p1", "sum"]);
        update_schedule(ctx, module).unwrap();
        assert_eq!(sequence(ctx, module), ["p1", "p0", "sum"]);
        crate::verification::verify(ctx, module).unwrap();
    }

    #[test]
    fn cycles_and_wrong_attribute_types_fail_without_schedule_or_name_mutation() {
        for wrong_type in [false, true] {
            let ctx = &mut Context::new();
            let (module, block, _, values) = fixture(ctx);
            ReturnOp::new(ctx, values[0])
                .get_operation()
                .insert_at_back(block, ctx);
            schedule(ctx, module, &["p0", "p1"]);
            let p0 = values[0].defining_op().unwrap();
            let p1 = values[1].defining_op().unwrap();
            if wrong_type {
                p0.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    pliron::builtin::attributes::StringAttr::new("not-controls".into()),
                );
            } else {
                p0.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(vec!["p1".into()]),
                );
                p1.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(vec!["p0".into()]),
                );
            }
            let before = module.disp(ctx).to_string();
            assert!(update_schedule(ctx, module).is_err());
            assert_eq!(module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn replay_defers_old_nodes_without_pulling_ready_old_successors_forward() {
        let mut replay = Replay {
            indegree: vec![0, 0, 1, 1, 1],
            successors: vec![vec![3], vec![2], vec![], vec![4], vec![]],
            ready: VecDeque::new(),
            existing: vec![true, true, true, true, false],
            deferred: vec![false; 5],
            scheduled: vec![false; 5],
            output: vec![],
        };
        for index in [2, 0, 1, 3] {
            if replay.scheduled[index] {
                continue;
            }
            if replay.indegree[index] == 0 {
                replay.emit(index);
                replay.drain();
            } else {
                replay.deferred[index] = true;
            }
        }
        assert_eq!(replay.output, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn replay_new_ready_nodes_use_fifo_not_a_new_dfs_schedule() {
        let mut replay = Replay {
            indegree: vec![0, 0, 1, 1, 2],
            successors: vec![vec![2, 3], vec![4], vec![4], vec![], vec![]],
            ready: VecDeque::from([0, 1]),
            existing: vec![false; 5],
            deferred: vec![false; 5],
            scheduled: vec![false; 5],
            output: vec![],
        };
        replay.drain();
        assert_eq!(replay.output, vec![0, 1, 2, 3, 4]);
    }
}
