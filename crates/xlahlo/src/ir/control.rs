//! Instruction control-edge updates for ordinary computation bodies.
use super::{ControlPredecessors, FusionParameters, InstructionName};
use crate::{
    ops::{ComputationOp, HloOpInterface},
    verification::{error, optional_attr, require},
};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    result::Result,
};
use std::collections::{HashMap, HashSet};

struct ControlRecord {
    operation: Ptr<Operation>,
    name: Option<String>,
    predecessors: Option<ControlPredecessors>,
}

fn instruction_block(ctx: &Context, instruction: Ptr<Operation>) -> Result<Ptr<BasicBlock>> {
    let dynamic = Operation::get_op_dyn(instruction, ctx);
    require(
        op_cast::<dyn HloOpInterface>(dynamic.as_ref()).is_some(),
        "control dependency endpoint must be a native instruction",
    )?;
    let block = instruction
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| error("control dependency endpoint is detached"))?;
    let parent = block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| error("control dependency block is detached"))?;
    require(
        Operation::is_op::<ComputationOp>(parent, ctx),
        "ordinary control updates require a computation body",
    )?;
    Ok(block)
}

fn snapshot(ctx: &Context, instruction: Ptr<Operation>) -> Result<Vec<ControlRecord>> {
    let block = instruction_block(ctx, instruction)?;
    let mut records = Vec::new();
    let mut names = HashMap::new();
    for operation in block.deref(ctx).iter(ctx) {
        let name = optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
            .map(|name| name.0.clone());
        if let Some(name) = &name {
            require(!name.is_empty(), "instruction name cannot be empty")?;
            require(
                names.insert(name.clone(), operation).is_none(),
                "duplicate instruction name",
            )?;
        }
        let predecessors =
            optional_attr::<ControlPredecessors>(ctx, operation, "hlo_control_predecessors")?
                .map(|predecessors| predecessors.clone());
        records.push(ControlRecord {
            operation,
            name,
            predecessors,
        });
    }
    for record in &records {
        if let Some(predecessors) = &record.predecessors {
            for predecessor in &predecessors.0 {
                let definition = names
                    .get(predecessor)
                    .ok_or_else(|| error(format!("unknown control predecessor: {predecessor}")))?;
                let dynamic = Operation::get_op_dyn(*definition, ctx);
                require(
                    op_cast::<dyn HloOpInterface>(dynamic.as_ref()).is_some(),
                    "control predecessor must be a native instruction",
                )?;
            }
        }
    }
    Ok(records)
}

fn fresh_control_name(ctx: &Context, instruction: Ptr<Operation>) -> Result<String> {
    let mut root = instruction;
    while let Some(parent) = root.deref(ctx).get_parent_op(ctx) {
        root = parent;
    }
    let mut pending = vec![root];
    let mut names = HashSet::new();
    while let Some(operation) = pending.pop() {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            names.insert(name.0.clone());
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                let data = block.deref(ctx);
                let key = Identifier::new("hlo_fusion_parameters");
                if let Some(parameters) = data.attributes.get::<FusionParameters>(&key) {
                    names.extend(
                        parameters
                            .0
                            .iter()
                            .filter_map(|parameter| parameter.name.clone()),
                    );
                } else {
                    require(
                        !data.attributes.0.contains_key(&key),
                        "fusion parameter metadata has an invalid attribute type",
                    )?;
                }
                pending.extend(data.iter(ctx));
            }
        }
    }
    (0..=names.len())
        .map(|index| format!("control_relay.{index}"))
        .find(|name| !names.contains(name))
        .ok_or_else(|| error("could not allocate a unique control dependency name"))
}

/// Copy a source instruction's incoming and outgoing control dependencies onto
/// another instruction in the same computation, retaining all original edges.
///
/// Duplicate edges are idempotent. Like pinned XLA's `CopyAllControlDepsFrom`,
/// this does not check cycles; complete-module verification checks the resulting
/// data/control graph. An unnamed destination receives a collision-free name
/// only when needed by an outgoing edge. Structural failures precede mutation.
pub fn copy_all_control_deps_from(
    ctx: &Context,
    instruction: Ptr<Operation>,
    source: Ptr<Operation>,
) -> Result<()> {
    let block = instruction_block(ctx, instruction)?;
    require(
        block == instruction_block(ctx, source)?,
        "control dependency endpoints must have the same parent computation",
    )?;
    let records = snapshot(ctx, source)?;
    let source_record = records
        .iter()
        .find(|record| record.operation == source)
        .ok_or_else(|| error("control source is missing from its block"))?;
    let source_name = source_record.name.clone();
    let incoming = source_record
        .predecessors
        .as_ref()
        .map(|predecessors| predecessors.0.clone())
        .unwrap_or_default();
    let outgoing = source_name.as_ref().is_some_and(|name| {
        records.iter().any(|record| {
            record
                .predecessors
                .as_ref()
                .is_some_and(|predecessors| predecessors.0.contains(name))
        })
    });
    let destination = records
        .iter()
        .find(|record| record.operation == instruction)
        .ok_or_else(|| error("control destination is missing from its block"))?;
    let mut destination_name = destination.name.clone();
    let generated_name = if outgoing && destination_name.is_none() {
        let name = fresh_control_name(ctx, instruction)?;
        destination_name = Some(name.clone());
        Some(name)
    } else {
        None
    };
    let mut updates = Vec::new();
    for record in &records {
        let mut additions = Vec::new();
        if record.operation == instruction {
            additions.extend(incoming.iter().cloned());
        }
        if let (Some(source_name), Some(destination_name)) = (&source_name, &destination_name)
            && record
                .predecessors
                .as_ref()
                .is_some_and(|predecessors| predecessors.0.contains(source_name))
        {
            additions.push(destination_name.clone());
        }
        if additions.is_empty() {
            continue;
        }
        let mut predecessors = record.predecessors.clone().unwrap_or_default();
        let before = predecessors.0.len();
        for name in additions {
            if !predecessors.0.contains(&name) {
                predecessors.0.push(name);
            }
        }
        if before != predecessors.0.len() {
            updates.push((record.operation, predecessors));
        }
    }
    if !updates.is_empty() || generated_name.is_some() {
        super::instruction_order::invalidate_order(ctx, block)?;
    }
    for (operation, predecessors) in updates {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_control_predecessors"), predecessors);
    }
    if let Some(name) = generated_name {
        instruction
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name));
    }
    Ok(())
}

/// Drop all incoming and outgoing control edges of an instruction, preserving
/// its name, data uses, schedule and all unrelated metadata.
pub fn drop_all_control_deps(ctx: &Context, instruction: Ptr<Operation>) -> Result<()> {
    let block = instruction_block(ctx, instruction)?;
    let records = snapshot(ctx, instruction)?;
    let name = records
        .iter()
        .find(|record| record.operation == instruction)
        .and_then(|record| record.name.as_deref());
    let mut updates = Vec::new();
    for record in &records {
        let Some(predecessors) = &record.predecessors else {
            continue;
        };
        let mut updated = predecessors.clone();
        if record.operation == instruction {
            updated.0.clear();
        } else if let Some(name) = name {
            updated.0.retain(|predecessor| predecessor != name);
        }
        if updated != *predecessors {
            updates.push((record.operation, updated));
        }
    }
    if !updates.is_empty() {
        super::instruction_order::invalidate_order(ctx, block)?;
    }
    for (operation, predecessors) in updates {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_control_predecessors"), predecessors);
    }
    Ok(())
}
