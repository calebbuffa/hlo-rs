//! HLO-aware instruction cloning for caller-block insertion.
//!
//! The whole-computation clone remains private. This is the narrow shared seam
//! for CloneWithNewOperands metadata and async-owned computation normalization.
use super::edit_staging::tree;
use crate::ir::{
    ControlPredecessors, FusionParameters, InstructionName, InstructionSequence, Schedule,
    instruction_order::InstructionOrder,
};
use crate::{
    attributes::{Callees, ResultAccuracy},
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp},
    verification::{error, optional_attr, require},
};
use pliron::{
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::{IrMapping, clone_operation},
        listener::DummyListener,
        rewriter::IRRewriter,
    },
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    result::Result,
    value::{DefiningEntity, Value},
};
use std::collections::{HashMap, HashSet};

pub(super) fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref())
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| error("instruction cloning requires a native HLO operation"))
}

pub(super) fn value_name(ctx: &Context, value: Value) -> Result<String> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => {
            optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
                .map(|name| name.0.clone())
                .ok_or_else(|| error("instruction has no HLO name"))
        }
        DefiningEntity::Block(block) => {
            let index = block
                .deref(ctx)
                .arguments()
                .position(|argument| argument == value)
                .ok_or_else(|| error("unknown fusion parameter"))?;
            crate::ir::fusion_parameters(ctx, block)?.0[index]
                .name
                .clone()
                .ok_or_else(|| error("fusion parameter has no HLO name"))
        }
    }
}

pub(super) struct InstructionCloner {
    names: HashSet<String>,
    symbols: HashSet<String>,
    active_wrappers: HashSet<Ptr<Operation>>,
}

impl InstructionCloner {
    pub fn new(ctx: &Context, module: Ptr<Operation>) -> Result<Self> {
        let mut names = HashSet::new();
        let mut symbols = HashSet::new();
        for operation in tree(ctx, module) {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                names.insert(name.0.clone());
            }
            if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx)
                && let Some(name) = computation.get_attr_hlo_computation_name(ctx)
            {
                symbols.insert(name.as_ref().to_string());
            }
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    if let Some(parameters) = block
                        .deref(ctx)
                        .attributes
                        .get::<FusionParameters>(&Identifier::new("hlo_fusion_parameters"))
                    {
                        names.extend(
                            parameters
                                .0
                                .iter()
                                .filter_map(|parameter| parameter.name.clone()),
                        );
                    }
                }
            }
        }
        Ok(Self {
            names,
            symbols,
            active_wrappers: HashSet::new(),
        })
    }

    fn fresh(&mut self, base: &str) -> String {
        let mut candidate = base.to_owned();
        let mut index = 0;
        while !self.names.insert(candidate.clone()) {
            candidate = format!("{base}.inline.{index}");
            index += 1;
        }
        candidate
    }

    fn symbol(&mut self, base: &str) -> Result<crate::ir::ComputationName> {
        let mut index = 0;
        loop {
            let candidate = format!("{base}_inline_{index}");
            if self.symbols.insert(candidate.clone()) {
                return Ok(crate::ir::ComputationName::new(candidate));
            }
            index += 1;
        }
    }

    /// Every external operand must already be mapped by the caller. Native
    /// generic cloning's same-context fallback is never permitted.
    pub fn clone_instruction(
        &mut self,
        ctx: &mut Context,
        module: Ptr<Operation>,
        source: Ptr<Operation>,
        before: Ptr<Operation>,
        mapping: &mut IrMapping,
    ) -> Result<Ptr<Operation>> {
        for operand in source.deref(ctx).operands() {
            require(
                mapping.lookup_value(operand).is_some(),
                "unmapped inlining operand",
            )?;
        }
        self.validate_regions(ctx, source)?;
        let clone = clone_operation(
            source,
            ctx,
            &mut IRRewriter::<DummyListener>::default(),
            mapping,
        );
        clone.insert_before(ctx, before);
        self.normalize(ctx, source, mapping)?;
        self.async_owners(ctx, module, source, mapping)?;
        Ok(clone)
    }

    fn validate_regions(&self, ctx: &Context, source: Ptr<Operation>) -> Result<()> {
        for operation in tree(ctx, source) {
            require(
                operation.deref(ctx).get_num_successors() == 0,
                "native inline instruction cannot contain block successors",
            )?;
            for region in operation.deref(ctx).regions() {
                let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
                require(
                    blocks.len() == 1,
                    "inline instruction region requires one native block",
                )?;
                let block = blocks[0];
                let members: HashSet<_> = block.deref(ctx).iter(ctx).collect();
                for child in &members {
                    for operand in child.deref(ctx).operands() {
                        require(
                            match operand.defining_entity() {
                                DefiningEntity::Op(definition) => members.contains(&definition),
                                DefiningEntity::Block(owner) => owner == block,
                            },
                            "inline region captures an external SSA value",
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn normalize(
        &mut self,
        ctx: &Context,
        source: Ptr<Operation>,
        mapping: &IrMapping,
    ) -> Result<()> {
        let mut names = HashMap::new();
        for original in tree(ctx, source) {
            let clone = mapping
                .lookup_op(original)
                .ok_or_else(|| error("missing nested clone"))?;
            if !Operation::is_op::<ReturnOp>(original, ctx)
                && !Operation::is_op::<ComputationOp>(original, ctx)
            {
                let old = optional_attr::<InstructionName>(ctx, original, "hlo_name")?
                    .map(|name| name.0.clone());
                let new = self.fresh(old.as_deref().unwrap_or("inlined"));
                if let Some(old) = old {
                    require(
                        names.insert(old, new.clone()).is_none(),
                        "ambiguous nested clone identity",
                    )?;
                }
                clone
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_name"), InstructionName(new));
                if !opcode(ctx, original)?.supports_result_accuracy()
                    && optional_attr::<ResultAccuracy>(ctx, original, "hlo_result_accuracy")?
                        .is_some()
                {
                    clone.deref_mut(ctx).attributes.set(
                        Identifier::new("hlo_result_accuracy"),
                        ResultAccuracy::Default,
                    );
                }
            }
            if Operation::is_op::<crate::ops::control::FusionOp>(original, ctx) {
                for region in original.deref(ctx).regions() {
                    for block in region.deref(ctx).iter(ctx) {
                        let mut parameters = crate::ir::fusion_parameters(ctx, block)?;
                        for (index, parameter) in parameters.0.iter_mut().enumerate() {
                            let old = parameter.name.clone();
                            let base = old.clone().unwrap_or_else(|| format!("parameter_{index}"));
                            let new = self.fresh(&base);
                            if let Some(old) = old {
                                require(
                                    names.insert(old, new.clone()).is_none(),
                                    "ambiguous fusion parameter identity",
                                )?;
                            }
                            parameter.name = Some(new);
                        }
                        let cloned = mapping
                            .lookup_block(block)
                            .ok_or_else(|| error("missing cloned fusion block"))?;
                        cloned
                            .deref_mut(ctx)
                            .attributes
                            .set(Identifier::new("hlo_fusion_parameters"), parameters);
                    }
                }
            }
        }
        let remap = |old: &str| {
            names
                .get(old)
                .cloned()
                .ok_or_else(|| error("unmapped nested control identity"))
        };
        for original in tree(ctx, source) {
            let clone = mapping
                .lookup_op(original)
                .ok_or_else(|| error("missing nested clone"))?;
            // The caller handles direct instruction controls because they may
            // reference mapped parameters/existing caller values.
            if (original != source || Operation::is_op::<ComputationOp>(source, ctx))
                && let Some(incoming) =
                    optional_attr::<ControlPredecessors>(ctx, original, "hlo_control_predecessors")?
            {
                let incoming = incoming
                    .0
                    .iter()
                    .map(|name| remap(name))
                    .collect::<Result<Vec<_>>>()?;
                clone.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(incoming),
                );
            }
            if let Some(order) =
                optional_attr::<InstructionOrder>(ctx, original, "hlo_instruction_order")?
            {
                let mut order = order.clone();
                for name in &mut order.instructions {
                    *name = remap(name)?;
                }
                for record in order
                    .blocks
                    .iter_mut()
                    .chain(&mut order.users)
                    .chain(&mut order.control_successors)
                {
                    record.subject = remap(&record.subject)?;
                    for name in &mut record.members {
                        *name = remap(name)?;
                    }
                }
                clone
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_instruction_order"), order);
            }
            if Operation::is_op::<crate::ops::control::FusionOp>(original, ctx) {
                for region in original.deref(ctx).regions() {
                    for block in region.deref(ctx).iter(ctx) {
                        let cloned = mapping
                            .lookup_block(block)
                            .ok_or_else(|| error("missing cloned fusion block"))?;
                        let mut parameters = cloned
                            .deref(ctx)
                            .attributes
                            .get::<FusionParameters>(&Identifier::new("hlo_fusion_parameters"))
                            .ok_or_else(|| error("missing cloned parameter metadata"))?
                            .clone();
                        for parameter in &mut parameters.0 {
                            parameter.control_predecessors = parameter
                                .control_predecessors
                                .iter()
                                .map(|name| remap(name))
                                .collect::<Result<Vec<_>>>()?;
                        }
                        cloned
                            .deref_mut(ctx)
                            .attributes
                            .set(Identifier::new("hlo_fusion_parameters"), parameters);
                    }
                }
            }
        }
        Ok(())
    }

    fn async_owners(
        &mut self,
        ctx: &mut Context,
        module: Ptr<Operation>,
        source: Ptr<Operation>,
        mapping: &IrMapping,
    ) -> Result<()> {
        for original in tree(ctx, source) {
            if Operation::is_op::<ReturnOp>(original, ctx)
                || Operation::is_op::<ComputationOp>(original, ctx)
            {
                continue;
            }
            let kind = opcode(ctx, original)?;
            let clone = mapping
                .lookup_op(original)
                .ok_or_else(|| error("missing async clone"))?;
            if kind == Opcode::AsyncStart {
                let callees = optional_attr::<Callees>(ctx, original, "hlo_callees")?
                    .as_deref()
                    .cloned()
                    .ok_or_else(|| error("async start lacks wrapped computation"))?;
                require(
                    callees.0.len() == 1,
                    "async start requires one wrapped computation",
                )?;
                let block = module
                    .deref(ctx)
                    .get_region(0)
                    .deref(ctx)
                    .get_entry_block()
                    .ok_or_else(|| error("module has no body"))?;
                let wrapped = block
                    .deref(ctx)
                    .iter(ctx)
                    .find(|&operation| {
                        Operation::get_op::<ComputationOp>(operation, ctx).is_some_and(
                            |computation| {
                                computation
                                    .get_attr_hlo_computation_name(ctx)
                                    .is_some_and(|name| *name == callees.0[0])
                            },
                        )
                    })
                    .ok_or_else(|| error("unknown async wrapped computation"))?;
                require(
                    self.active_wrappers.insert(wrapped),
                    "recursive async wrapped computation",
                )?;
                self.validate_regions(ctx, wrapped)?;
                let mut wrapped_mapping = IrMapping::new();
                let wrapped_clone = clone_operation(
                    wrapped,
                    ctx,
                    &mut IRRewriter::<DummyListener>::default(),
                    &mut wrapped_mapping,
                );
                wrapped_clone.insert_at_back(block, ctx);
                let symbol = self.symbol(callees.0[0].as_str())?;
                Operation::get_op::<ComputationOp>(wrapped_clone, ctx)
                    .ok_or_else(|| error("async wrapped clone is not a computation"))?
                    .set_attr_hlo_computation_name(ctx, symbol.clone());
                self.normalize(ctx, wrapped, &wrapped_mapping)?;
                self.async_owners(ctx, module, wrapped, &wrapped_mapping)?;
                self.active_wrappers.remove(&wrapped);
                clone.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_callees"),
                    Callees(vec![symbol.clone()]),
                );
                let old_schedule = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
                    .as_deref()
                    .cloned();
                if let Some(mut schedule) = old_schedule
                    && schedule
                        .0
                        .iter()
                        .any(|sequence| sequence.computation == callees.0[0])
                {
                    let mut instructions = Vec::new();
                    for operation in super::computation_postorder(ctx, wrapped)? {
                        let copy = wrapped_mapping
                            .lookup_op(operation)
                            .ok_or_else(|| error("wrapped schedule instruction lacks mapping"))?;
                        instructions.push(value_name(ctx, copy.deref(ctx).get_result(0))?);
                    }
                    schedule.0.push(InstructionSequence {
                        computation: symbol,
                        instructions,
                    });
                    module
                        .deref_mut(ctx)
                        .attributes
                        .set(Identifier::new("hlo_schedule"), schedule);
                }
            } else if matches!(kind, Opcode::AsyncUpdate | Opcode::AsyncDone)
                && optional_attr::<Callees>(ctx, original, "hlo_callees")?.is_some()
            {
                let wrapped = super::async_computation(ctx, module, clone)?;
                let computation = Operation::get_op::<ComputationOp>(wrapped, ctx)
                    .ok_or_else(|| error("async chain target is not a computation"))?;
                let name = computation
                    .get_attr_hlo_computation_name(ctx)
                    .ok_or_else(|| error("unnamed async wrapped clone"))?
                    .clone();
                clone
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_callees"), Callees(vec![name]));
            }
        }
        Ok(())
    }
}
