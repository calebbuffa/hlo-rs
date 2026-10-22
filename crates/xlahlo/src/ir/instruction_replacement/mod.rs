//! Native instruction forwarding and recursive operand cleanup.
//!
//! Follows `HloComputation::ReplaceInstruction` and its default
//! `RemoveInstructionAndUnusedOperands` path. Ordinary parameters are protected;
//! fusion parameters repair their owning fusion's operands when removed.
use super::{
    ControlPredecessors, InstructionName, Schedule, fusion_parameters, remove_fusion_parameter,
    set_fusion_parameters,
};
use crate::{
    analysis::effects::SideEffects,
    attributes::{FrontendAttributes, FusionKind, OpMetadata, OutputOperandAliases, Payload},
    ir::OriginalValue,
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp, control::FusionOp},
    shape,
    sharding::Sharding,
    verification::{VerifyOptions, error, optional_attr, require},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    result::Result,
    r#type::Typed,
    value::{DefiningEntity, Value},
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

mod forwarding;
pub use forwarding::replace_all_uses_with;
mod call_inlining;
pub use call_inlining::{DefusedFusion, defuse};
pub use call_inlining::{InlineCallOptions, InlinedCall, inline_call};
mod while_annotation;
pub use while_annotation::{
    WhileInductionForwarding, WhileLoopAnnotationBatch, WhileLoopAnnotationRequest,
    annotate_while_loops,
};
mod alias;
mod creation_staging;
use alias::{alias_owners, restore_alias_owners, verified_edit};
mod annotations;
use annotations::{Annotations, argument_index};
mod cleanup;
use cleanup::{Cleanup, deduplicate_fusion_operands};
mod control_graph;
use control_graph::{ControlGraph, same_custom_target};

/// Policies corresponding to the pinned `ReplaceInstruction` arguments.
///
/// These are graph-edit policies, not verifier options.
#[derive(Clone, Copy, Debug)]
pub struct ReplaceInstructionOptions {
    /// Preserve source sharding when the replacement does not already carry it.
    pub preserve_sharding: bool,
    /// Relay the source instruction's control dependencies to the replacement.
    pub relay_control_dependency: bool,
    /// Recursively remove operands and fusion parameters made unused by removal.
    pub remove_unused_operands: bool,
    /// Preserve frontend attributes according to the pinned replacement rules.
    pub preserve_frontend_attributes: bool,
}

impl Default for ReplaceInstructionOptions {
    fn default() -> Self {
        Self {
            preserve_sharding: false,
            relay_control_dependency: false,
            remove_unused_operands: true,
            preserve_frontend_attributes: true,
        }
    }
}

/// The changed bit and erased operations, for consumers holding pointer worklists.
#[derive(Default)]
pub struct InstructionReplacement {
    /// Whether the requested edit changed the IR; a policy conflict can return
    /// `false` without treating the request as malformed.
    pub changed: bool,
    /// Operations erased by this edit, including recursively cleaned-up
    /// operands. Pointers are useful for invalidating caller-maintained worklists.
    pub removed: HashSet<Ptr<Operation>>,
}

/// Replace a one-result instruction by a compatible value in the same body.
///
/// A fusion block argument is an instruction endpoint for annotation/control
/// purposes, even though pliron represents it as a `Value`. Conflicting sharding
/// or unrelayed controls produce `changed=false`, matching XLA. Other malformed
/// input is an error, never an optimization nonmatch. Existing replacement
/// annotations are retained except for the pinned metadata/provenance rules.
/// `None` is the raw pinned edit, including its alias-maintenance gap during
/// recursive parameter removal. `Some` stages the complete edit, preserves
/// alias owners, repairs an existing schedule, and verifies before committing.
/// No verification policy is chosen implicitly and no schedule is invented.
/// Actual graph changes invalidate optional logical ordering permutations in
/// their containing computation; this does not preserve XLA's historical order.
/// Verified staging rejects custom verifier callbacks: their identity/context
/// semantics cannot be faithfully evaluated on cloned operations yet.
pub fn replace_instruction(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    replacement: Value,
    options: ReplaceInstructionOptions,
    verification: Option<VerifyOptions>,
) -> Result<InstructionReplacement> {
    if let Some(verification) = verification {
        return verified_edit(
            ctx,
            instruction,
            Edit::Replace(replacement, options),
            verification,
        );
    }
    replace_instruction_impl(ctx, instruction, replacement, options)
}

fn replace_instruction_impl(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    replacement: Value,
    options: ReplaceInstructionOptions,
) -> Result<InstructionReplacement> {
    let original = result(ctx, instruction)?;
    if original == replacement {
        return Ok(InstructionReplacement::default());
    }
    let block = endpoint_block(ctx, original)?;
    require(
        endpoint_block(ctx, replacement)? == block,
        "replacement is outside the instruction body",
    )?;
    require(
        shape::compatible(ctx, original.get_type(ctx), replacement.get_type(ctx)),
        "replacement instruction shapes are incompatible",
    )?;
    let mut controls = ControlGraph::read(ctx, block)?;
    let old_node = controls.index(original)?;
    let replacement_node = controls.index(replacement)?;
    let old_annotations = Annotations::read(ctx, original)?;
    let mut annotations = Annotations::read(ctx, replacement)?;
    if options.preserve_sharding
        && let (Some(old), Some(new)) = (&old_annotations.sharding, &annotations.sharding)
        && !old.equal_semantics(new)?
    {
        return Ok(InstructionReplacement::default());
    }
    if !options.relay_control_dependency && controls.has_controls(old_node) {
        return Ok(InstructionReplacement::default());
    }
    let module = containing_module(ctx, block)?;
    let mut analyses = AnalysisManager::default();
    let effects = SideEffects::compute(module, ctx, &mut analyses)?;
    let operands: Vec<_> = instruction.deref(ctx).operands().collect();
    if let Some(operation) = replacement.defining_op() {
        require(
            !depends_on(ctx, operation, instruction)?,
            "replacement depends on the removed instruction",
        )?;
    }
    require(
        opcode(ctx, instruction)? != Opcode::Parameter,
        "parameter replacement requires caller analysis",
    )?;
    let old_metadata = old_annotations.metadata.clone().unwrap_or_default();
    let new_metadata = annotations.metadata.clone().unwrap_or_default();
    if new_metadata.op_name.is_empty() && !old_metadata.op_name.is_empty() {
        annotations.metadata = Some(old_metadata);
    } else if matches!(new_metadata.metadata_payload, Payload::Absent)
        && !matches!(old_metadata.metadata_payload, Payload::Absent)
    {
        let mut metadata = new_metadata;
        metadata.metadata_payload = old_metadata.metadata_payload;
        annotations.metadata = Some(metadata);
    }
    if options.preserve_frontend_attributes
        && annotations.frontend.0.is_empty()
        && !operands.contains(&replacement)
    {
        annotations.frontend = old_annotations.frontend;
    }
    if let Some(provenance) = old_annotations.original_value {
        annotations.original_value = Some(provenance);
    }
    if annotations.sharding.is_none() {
        annotations.sharding = old_annotations.sharding;
    }
    annotations.validate(ctx, replacement)?;
    if options.relay_control_dependency {
        controls.relay(old_node, replacement_node);
        controls.ensure_names(ctx, module)?;
    }
    // Rename before deleting, updating references in schedules and controls.
    // TupleSimplifier normally replaces different opcodes; shared users of this
    // primitive also need the same-opcode identity rule.
    let replacement_opcode = replacement
        .defining_op()
        .map(|op| opcode(ctx, op))
        .transpose()?
        .unwrap_or(Opcode::Parameter);
    let preserve_name = opcode(ctx, instruction)? == replacement_opcode
        && (replacement_opcode != Opcode::CustomCall
            || same_custom_target(ctx, instruction, replacement.defining_op())?);
    if preserve_name {
        drop(optional_attr::<Schedule>(ctx, module, "hlo_schedule")?);
    }
    let order_owner = order_owner(ctx, block)?;
    invalidate_order(ctx, order_owner);
    if preserve_name && let Some(name) = controls.nodes[old_node].name.take() {
        let previous = controls.nodes[replacement_node].name.replace(name.clone());
        if let Some(previous) = previous {
            rename_schedule(ctx, module, &previous, &name)?;
        }
    }
    controls.write(ctx)?;
    annotations.write(ctx, replacement)?;
    let users: HashSet<_> = original
        .uses(ctx)
        .iter()
        .map(|use_| use_.user_op())
        .collect();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.get_config_mut().set_name_on_value_replacement = false;
    rewriter.replace_value_uses_with(ctx, original, replacement);
    let mut cleanup = Cleanup::new(module, &effects);
    if options.remove_unused_operands {
        cleanup.queue.push_back(original);
        cleanup.initial = Some(instruction);
        cleanup.ignore_controls = options.relay_control_dependency;
        cleanup.drain(ctx, &mut rewriter)?;
    } else {
        require(!original.is_used(ctx), "replaced instruction is still used")?;
        cleanup.record_erased_tree(ctx, instruction);
        rewriter.erase_operation(ctx, instruction);
    }
    for user in users {
        if !cleanup.removed.contains(&user) && Operation::is_op::<FusionOp>(user, ctx) {
            deduplicate_fusion_operands(ctx, user, &mut rewriter, &mut cleanup)?;
        }
    }
    cleanup.drain(ctx, &mut rewriter)?;
    Ok(InstructionReplacement {
        changed: true,
        removed: cleanup.removed,
    })
}

/// Remove a dead instruction and recursively dead pure operands.
///
/// Uses the pinned default caller-analysis policy: ordinary computation
/// parameters are retained; unused fusion parameters remove matching caller
/// operands. The initial instruction may be effectful if the caller explicitly
/// requests its removal, but recursively reached effectful producers are kept.
/// Ignored control dependencies are spliced predecessor-to-successor before
/// deletion, rather than silently discarded. `None` is raw pinned removal;
/// `Some` validates an isolated final edit plan before mutating the source.
pub fn remove_instruction_and_unused_operands(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    ignore_control_dependencies: bool,
    verification: Option<VerifyOptions>,
) -> Result<HashSet<Ptr<Operation>>> {
    if let Some(verification) = verification {
        return verified_edit(
            ctx,
            instruction,
            Edit::Remove(ignore_control_dependencies),
            verification,
        )
        .map(|edit| edit.removed);
    }
    remove_instruction_impl(ctx, instruction, ignore_control_dependencies)
}

fn remove_instruction_impl(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    ignore_control_dependencies: bool,
) -> Result<HashSet<Ptr<Operation>>> {
    let value = result(ctx, instruction)?;
    let block = endpoint_block(ctx, value)?;
    require(!value.is_used(ctx), "cannot remove a live instruction")?;
    let graph = ControlGraph::read(ctx, block)?;
    require(
        ignore_control_dependencies || !graph.has_controls(graph.index(value)?),
        "cannot remove instruction with control dependencies",
    )?;
    require(
        opcode(ctx, instruction)? != Opcode::Parameter,
        "ordinary parameters require caller analysis",
    )?;
    let module = containing_module(ctx, block)?;
    let effects = SideEffects::compute(module, ctx, &mut AnalysisManager::default())?;
    let mut cleanup = Cleanup::new(module, &effects);
    cleanup.initial = Some(instruction);
    cleanup.ignore_controls = ignore_control_dependencies;
    cleanup.queue.push_back(value);
    cleanup.drain(ctx, &mut IRRewriter::<DummyListener>::default())?;
    Ok(cleanup.removed)
}

#[derive(Clone, Copy)]
enum Edit {
    Replace(Value, ReplaceInstructionOptions),
    Forward(Value),
    Remove(bool),
}

pub(super) fn require_rehearsable_policy(options: VerifyOptions) -> Result<()> {
    for (field, present) in [
        (
            "supported_unbounded_dynamic_op",
            options.supported_unbounded_dynamic_op.is_some(),
        ),
        (
            "instruction_can_change_layout",
            options.instruction_can_change_layout.is_some(),
        ),
        ("shape_size", options.shape_size.is_some()),
    ] {
        if present {
            return Err(pliron::arg_error_noloc!(
                crate::Diagnostic::UnsupportedCallback { field }
            ));
        }
    }
    Ok(())
}

fn operation_tree(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut operations = vec![root];
    let mut index = 0;
    while index < operations.len() {
        let operation = operations[index];
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                operations.extend(block.deref(ctx).iter(ctx));
            }
        }
        index += 1;
    }
    operations
}

fn result(ctx: &Context, operation: Ptr<Operation>) -> Result<Value> {
    opcode(ctx, operation)?;
    let ir = operation.deref(ctx);
    require(
        ir.get_num_results() == 1,
        "instruction replacement requires exactly one result",
    )?;
    Ok(ir.get_result(0))
}

fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref())
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| error("instruction edit requires native HLO"))
}

fn endpoint_block(ctx: &Context, value: Value) -> Result<Ptr<BasicBlock>> {
    let block = match value.defining_entity() {
        DefiningEntity::Op(operation) => operation
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("instruction endpoint is detached"))?,
        DefiningEntity::Block(block) => block,
    };
    let parent = block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| error("instruction body is detached"))?;
    require(
        Operation::is_op::<ComputationOp>(parent, ctx) || Operation::is_op::<FusionOp>(parent, ctx),
        "instruction edit requires computation or fusion body",
    )?;
    Ok(block)
}

fn containing_module(ctx: &Context, mut block: Ptr<BasicBlock>) -> Result<Ptr<Operation>> {
    loop {
        let parent = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("instruction body is detached"))?;
        if Operation::is_op::<ModuleOp>(parent, ctx) {
            return Ok(parent);
        }
        block = parent
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("instruction has no containing module"))?;
    }
}

fn order_owner(ctx: &Context, mut block: Ptr<BasicBlock>) -> Result<Ptr<Operation>> {
    loop {
        let owner = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("ordering body is detached"))?;
        if Operation::is_op::<ComputationOp>(owner, ctx) {
            return Ok(owner);
        }
        require(
            Operation::is_op::<FusionOp>(owner, ctx),
            "invalid ordering body owner",
        )?;
        block = owner
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("ordering body has no computation"))?;
    }
}

fn invalidate_order(ctx: &Context, owner: Ptr<Operation>) {
    owner
        .deref_mut(ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_instruction_order"));
}

fn depends_on(ctx: &Context, operation: Ptr<Operation>, needle: Ptr<Operation>) -> Result<bool> {
    let mut pending = vec![operation];
    let mut seen = HashSet::new();
    while let Some(operation) = pending.pop() {
        if operation == needle {
            return Ok(true);
        }
        if !seen.insert(operation) {
            continue;
        }
        opcode(ctx, operation)?;
        pending.extend(
            operation
                .deref(ctx)
                .operands()
                .filter_map(|operand| operand.defining_op()),
        );
    }
    Ok(false)
}

fn rename_schedule(ctx: &Context, module: Ptr<Operation>, old: &str, new: &str) -> Result<()> {
    let schedule = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .as_deref()
        .cloned();
    if let Some(mut schedule) = schedule {
        for sequence in &mut schedule.0 {
            sequence.instructions.retain(|name| name != new);
            for name in &mut sequence.instructions {
                if name == old {
                    *name = new.to_owned();
                }
            }
        }
        module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_schedule"), schedule);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verification::verify_with_options;
    use crate::{
        attributes::{Bytes, Index},
        ir::{self, FusionParameter, FusionParameters, InstructionSequence},
        ir::{OriginalArray, OriginalValueElement},
        ops::control::{AfterAllOp, CustomCallOp, GetTupleElementOp, ParameterOp, TupleOp},
        ops::elementwise::AddOp,
        ops::movement::CopyOp,
        shape::{Dimension, Layout},
        types::ElementType,
    };
    use pliron::irbuild::cloning::{IrMapping, clone_operation};
    use pliron::{
        builtin::{attributes::StringAttr, types::FunctionType},
        op::Op,
        printable::Printable,
        r#type::TypeHandle,
    };

    pub(super) struct Fixture {
        pub(super) module: Ptr<Operation>,
        pub(super) block: Ptr<BasicBlock>,
        pub(super) ty: TypeHandle,
        pub(super) parameters: Vec<Value>,
    }

    pub(super) fn fixture(ctx: &mut Context, inputs: usize) -> Fixture {
        crate::register(ctx);
        let ty = shape::array(ctx, ElementType::F32, vec![2, 3]).unwrap();
        fixture_with_type(ctx, inputs, ty)
    }

    fn fixture_with_type(ctx: &mut Context, inputs: usize, ty: TypeHandle) -> Fixture {
        let module = ModuleOp::new(ctx, Identifier::new("replacement")).get_operation();
        ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let signature = FunctionType::get(ctx, vec![ty; inputs], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new("main"), signature);
        computation.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let mut parameters = Vec::new();
        for index in 0..inputs {
            let parameter = ParameterOp::new(ctx, ty, vec![], Index(index as u32));
            parameter.get_operation().insert_at_back(block, ctx);
            name(ctx, parameter.get_operation(), &format!("p{index}"));
            parameters.push(parameter.result(ctx));
        }
        Fixture {
            module,
            block,
            ty,
            parameters,
        }
    }

    pub(super) fn name(ctx: &Context, operation: Ptr<Operation>, name: &str) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name.into()));
    }

    pub(super) fn controls(ctx: &Context, operation: Ptr<Operation>, predecessors: &[&str]) {
        operation.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(predecessors.iter().map(|name| (*name).into()).collect()),
        );
    }

    pub(super) fn finish(ctx: &mut Context, fixture: &Fixture, value: Value) -> Ptr<Operation> {
        let ret = ReturnOp::new(ctx, value).get_operation();
        ret.insert_at_back(fixture.block, ctx);
        crate::verification::verify(ctx, fixture.module).unwrap();
        ret
    }

    pub(super) fn copy(
        ctx: &mut Context,
        fixture: &Fixture,
        value: Value,
        instruction_name: &str,
    ) -> Ptr<Operation> {
        let operation = CopyOp::new(ctx, value.get_type(ctx), vec![value]).get_operation();
        operation.insert_at_back(fixture.block, ctx);
        name(ctx, operation, instruction_name);
        operation
    }

    fn provenance(instruction_name: &str) -> OriginalValue {
        OriginalValue {
            elements: vec![OriginalValueElement {
                shape_index: vec![],
                original_array: vec![OriginalArray {
                    instruction_name: instruction_name.into(),
                    shape_index: vec![],
                }],
            }],
            ..Default::default()
        }
    }

    #[test]
    fn defaults_are_the_pinned_replacement_arguments() {
        let options = ReplaceInstructionOptions::default();
        assert!(!options.preserve_sharding);
        assert!(!options.relay_control_dependency);
        assert!(options.remove_unused_operands);
        assert!(options.preserve_frontend_attributes);
    }

    #[test]
    fn compatible_layout_replacement_retains_ordinary_parameters_and_transfers_annotations() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let laid_out = shape::with_layout(
            ctx,
            f.ty,
            Layout {
                minor_to_major: vec![0, 1],
                ..Default::default()
            },
        )
        .unwrap();
        let old = CopyOp::new(ctx, laid_out, vec![f.parameters[0]]).get_operation();
        old.insert_at_back(f.block, ctx);
        let metadata = OpMetadata {
            op_name: "source".into(),
            ..Default::default()
        };
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_metadata"), metadata.clone());
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::replicated());
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_original_value"), provenance("old"));
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_frontend_attributes"),
            FrontendAttributes::new([("do_not_flow_back".into(), "true".into())]),
        );
        let ret = finish(ctx, &f, result(ctx, old).unwrap());
        let changed = replace_instruction(
            ctx,
            old,
            f.parameters[0],
            ReplaceInstructionOptions {
                preserve_sharding: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert!(changed.changed);
        assert!(changed.removed.contains(&old));
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
        let annotations = Annotations::read(ctx, f.parameters[0]).unwrap();
        assert_eq!(annotations.metadata, Some(metadata));
        assert!(annotations.sharding.is_some());
        assert_eq!(annotations.original_value, Some(provenance("old")));
        assert!(annotations.frontend.0.is_empty());
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn existing_metadata_keeps_name_but_receives_missing_payload_and_provenance() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "discarded_name".into(),
                metadata_payload: Payload::Value(Bytes(vec![1, 2, 3])),
                ..Default::default()
            },
        );
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("from_old"),
        );
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_frontend_attributes"),
            FrontendAttributes::new([("kept".into(), "yes".into())]),
        );
        let destination = f.parameters[1].defining_op().unwrap();
        destination.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "existing".into(),
                source_line: 42,
                ..Default::default()
            },
        );
        finish(ctx, &f, result(ctx, old).unwrap());
        replace_instruction(ctx, old, f.parameters[1], Default::default(), None).unwrap();
        let annotations = Annotations::read(ctx, f.parameters[1]).unwrap();
        let metadata = annotations.metadata.unwrap();
        assert_eq!(metadata.op_name, "existing");
        assert_eq!(metadata.source_line, 42);
        assert_eq!(
            metadata.metadata_payload,
            Payload::Value(Bytes(vec![1, 2, 3]))
        );
        assert_eq!(annotations.frontend.get("kept"), Some("yes"));
        assert_eq!(annotations.original_value, Some(provenance("from_old")));
    }

    #[test]
    fn incompatible_sharding_is_an_unchanged_nonmatch_without_metadata_leaks() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let old = copy(ctx, &f, f.parameters[0], "old");
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::replicated());
        f.parameters[0]
            .defining_op()
            .unwrap()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::maximal(0));
        finish(ctx, &f, result(ctx, old).unwrap());
        let before = f.module.disp(ctx).to_string();
        let outcome = replace_instruction(
            ctx,
            old,
            f.parameters[0],
            ReplaceInstructionOptions {
                preserve_sharding: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert!(!outcome.changed);
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    #[test]
    fn whole_tuple_style_replacement_skips_incoming_and_outgoing_controls() {
        for outgoing in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 2);
            let old = copy(ctx, &f, f.parameters[0], "old");
            let consumer = copy(ctx, &f, f.parameters[1], "consumer");
            if outgoing {
                controls(ctx, consumer, &["old"]);
            } else {
                controls(ctx, old, &["p1"]);
            }
            let add = AddOp::new(
                ctx,
                f.ty,
                vec![result(ctx, old).unwrap(), result(ctx, consumer).unwrap()],
            );
            add.get_operation().insert_at_back(f.block, ctx);
            finish(ctx, &f, add.result(ctx));
            let before = f.module.disp(ctx).to_string();
            assert!(
                !replace_instruction(ctx, old, f.parameters[0], Default::default(), None)
                    .unwrap()
                    .changed
            );
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn relay_moves_controls_and_stable_schedule_defers_old_nodes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        controls(ctx, old, &["p1"]);
        let consumer = copy(ctx, &f, f.parameters[1], "consumer");
        controls(ctx, consumer, &["old"]);
        let add = AddOp::new(
            ctx,
            f.ty,
            vec![result(ctx, old).unwrap(), result(ctx, consumer).unwrap()],
        );
        add.get_operation().insert_at_back(f.block, ctx);
        name(ctx, add.get_operation(), "sum");
        finish(ctx, &f, add.result(ctx));
        f.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: "main".into(),
                instructions: ["p0", "p1", "old", "consumer", "sum"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            }]),
        );
        replace_instruction(
            ctx,
            old,
            f.parameters[0],
            ReplaceInstructionOptions {
                relay_control_dependency: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        super::super::schedule_update::update_schedule(ctx, f.module).unwrap();
        assert_eq!(
            optional_attr::<ControlPredecessors>(ctx, consumer, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec!["p0"]
        );
        let schedule = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
            .unwrap()
            .unwrap();
        assert_eq!(schedule.0[0].instructions, ["p1", "p0", "consumer", "sum"]);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn same_opcode_replacement_keeps_destination_schedule_position_under_source_name() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let destination = copy(ctx, &f, f.parameters[0], "destination");
        let between = copy(ctx, &f, f.parameters[1], "between");
        let old = copy(ctx, &f, f.parameters[0], "old");
        let sum = AddOp::new(
            ctx,
            f.ty,
            vec![result(ctx, old).unwrap(), result(ctx, between).unwrap()],
        );
        sum.get_operation().insert_at_back(f.block, ctx);
        name(ctx, sum.get_operation(), "sum");
        finish(ctx, &f, sum.result(ctx));
        f.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: "main".into(),
                instructions: ["p0", "p1", "destination", "between", "old", "sum"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            }]),
        );
        replace_instruction(
            ctx,
            old,
            result(ctx, destination).unwrap(),
            Default::default(),
            None,
        )
        .unwrap();
        super::super::schedule_update::update_schedule(ctx, f.module).unwrap();
        assert_eq!(
            optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
                .unwrap()
                .unwrap()
                .0[0]
                .instructions,
            ["p0", "p1", "old", "between", "sum"],
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn fusion_argument_replacement_preserves_annotations_and_parameter_controls() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let fusion = FusionOp::new(ctx, f.ty, f.parameters.clone(), FusionKind::Loop);
        fusion.get_operation().insert_at_back(f.block, ctx);
        let body = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
        body.insert_at_back(fusion.body(ctx), ctx);
        let a = body.deref(ctx).get_argument(0);
        let b = body.deref(ctx).get_argument(1);
        let tuple_type = shape::tuple(ctx, vec![f.ty, f.ty]).unwrap();
        let tuple = TupleOp::new(ctx, tuple_type, vec![a, b]);
        tuple.get_operation().insert_at_back(body, ctx);
        let projection = GetTupleElementOp::new(ctx, f.ty, vec![tuple.result(ctx)], Index(0));
        projection.get_operation().insert_at_back(body, ctx);
        name(ctx, projection.get_operation(), "project");
        controls(ctx, projection.get_operation(), &["arg1"]);
        projection.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "fusion_source".into(),
                ..Default::default()
            },
        );
        projection
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::replicated());
        projection.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("fusion_source"),
        );
        let sum = AddOp::new(ctx, f.ty, vec![projection.result(ctx), b]);
        sum.get_operation().insert_at_back(body, ctx);
        ReturnOp::new(ctx, sum.result(ctx))
            .get_operation()
            .insert_at_back(body, ctx);
        set_fusion_parameters(
            ctx,
            body,
            FusionParameters(vec![
                FusionParameter {
                    name: Some("arg0".into()),
                    ..Default::default()
                },
                FusionParameter {
                    name: Some("arg1".into()),
                    ..Default::default()
                },
            ]),
        )
        .unwrap();
        finish(ctx, &f, fusion.result(ctx));
        replace_instruction(
            ctx,
            projection.get_operation(),
            a,
            ReplaceInstructionOptions {
                preserve_sharding: true,
                relay_control_dependency: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let parameters = fusion_parameters(ctx, body).unwrap();
        assert_eq!(parameters.0.len(), 2); // arg1 remains a control predecessor.
        assert_eq!(parameters.0[0].control_predecessors, vec!["arg1"]);
        assert_eq!(
            parameters.0[0].metadata.as_ref().unwrap().op_name,
            "fusion_source"
        );
        assert!(parameters.0[0].sharding.is_some());
        assert_eq!(
            parameters.0[0].original_value,
            Some(provenance("fusion_source"))
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn malformed_metadata_cross_body_shape_and_data_cycle_errors_precede_rewrite() {
        for case in 0..3 {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1);
            let old = copy(ctx, &f, f.parameters[0], "old");
            let later = copy(ctx, &f, result(ctx, old).unwrap(), "later");
            finish(ctx, &f, result(ctx, later).unwrap());
            let replacement = match case {
                0 => {
                    old.deref_mut(ctx).attributes.set(
                        Identifier::new("hlo_metadata"),
                        StringAttr::new("wrong".into()),
                    );
                    f.parameters[0]
                }
                1 => {
                    let different = shape::array(ctx, ElementType::F32, vec![6]).unwrap();
                    f.parameters[0].set_type(ctx, different);
                    f.parameters[0]
                }
                _ => result(ctx, later).unwrap(),
            };
            let before = f.module.disp(ctx).to_string();
            assert!(replace_instruction(ctx, old, replacement, Default::default(), None).is_err());
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn no_implicit_verification_policy_rejects_or_authorizes_unbounded_shapes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dynamic = shape::dynamic_array(
            ctx,
            ElementType::F32,
            vec![Dimension::Unbounded, Dimension::Static(3)],
        )
        .unwrap();
        let old = CopyOp::new(ctx, dynamic, vec![f.parameters[0]]).get_operation();
        old.insert_at_back(f.block, ctx);
        ReturnOp::new(ctx, result(ctx, old).unwrap())
            .get_operation()
            .insert_at_back(f.block, ctx);
        assert!(crate::verification::verify(ctx, f.module).is_err());
        replace_instruction(ctx, old, f.parameters[0], Default::default(), None).unwrap();
        crate::verification::verify(ctx, f.module).unwrap();
    }

    pub(super) struct AliasFixture {
        pub(super) outer: Fixture,
        pub(super) fusion: Ptr<Operation>,
        pub(super) body: Ptr<BasicBlock>,
        pub(super) projection: Ptr<Operation>,
        pub(super) selected: Value,
    }

    pub(super) fn alias_fixture(
        ctx: &mut Context,
        selected_index: usize,
        inputs: usize,
    ) -> AliasFixture {
        let outer = fixture(ctx, inputs);
        let fusion = FusionOp::new(ctx, outer.ty, outer.parameters.clone(), FusionKind::Loop);
        let fusion_ptr = fusion.get_operation();
        fusion_ptr.insert_at_back(outer.block, ctx);
        name(ctx, fusion_ptr, "fusion");
        let body = BasicBlock::new(ctx, None, vec![outer.ty; inputs]);
        body.insert_at_back(fusion.body(ctx), ctx);
        let a = body.deref(ctx).get_argument(0);
        let b = body.deref(ctx).get_argument(1);
        set_fusion_parameters(
            ctx,
            body,
            FusionParameters(
                (0..inputs)
                    .map(|index| FusionParameter {
                        name: Some(format!("arg{index}")),
                        metadata: Some(OpMetadata {
                            op_name: format!("input{index}"),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .collect(),
            ),
        )
        .unwrap();
        let ty = shape::tuple(ctx, vec![outer.ty, outer.ty]).unwrap();
        let tuple = TupleOp::new(ctx, ty, vec![a, b]);
        tuple.get_operation().insert_at_back(body, ctx);
        name(ctx, tuple.get_operation(), "tuple");
        let projection = GetTupleElementOp::new(
            ctx,
            outer.ty,
            vec![tuple.result(ctx)],
            Index(selected_index as u32),
        );
        let projection_ptr = projection.get_operation();
        projection_ptr.insert_at_back(body, ctx);
        name(ctx, projection_ptr, "project");
        controls(
            ctx,
            projection_ptr,
            &[if selected_index == 0 { "arg1" } else { "arg0" }],
        );
        let root = if inputs == 3 {
            let arguments = vec![projection.result(ctx), body.deref(ctx).get_argument(2)];
            let add = AddOp::new(ctx, outer.ty, arguments);
            add.get_operation().insert_at_back(body, ctx);
            name(ctx, add.get_operation(), "sum");
            add.result(ctx)
        } else {
            projection.result(ctx)
        };
        ReturnOp::new(ctx, root)
            .get_operation()
            .insert_at_back(body, ctx);
        fusion_ptr.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_aliases"),
            OutputOperandAliases(vec![crate::attributes::OutputOperandAliasing {
                output_shape_index: crate::attributes::Axes(vec![]),
                operand_index: 1,
                operand_shape_index: crate::attributes::Axes(vec![]),
            }]),
        );
        finish(ctx, &outer, fusion.result(ctx));
        outer.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: "main".into(),
                instructions: (0..inputs)
                    .map(|index| format!("p{index}"))
                    .chain(std::iter::once("fusion".into()))
                    .collect(),
            }]),
        );
        verify_with_options(ctx, outer.module, VerifyOptions::default()).unwrap();
        AliasFixture {
            outer,
            fusion: fusion_ptr,
            body,
            projection: projection_ptr,
            selected: body.deref(ctx).get_argument(selected_index),
        }
    }

    pub(super) fn forwarding_options() -> ReplaceInstructionOptions {
        ReplaceInstructionOptions {
            preserve_sharding: true,
            relay_control_dependency: true,
            ..Default::default()
        }
    }

    pub(super) fn rich_order(ctx: &Context, block: Ptr<BasicBlock>) -> Ptr<Operation> {
        use super::super::instruction_order::{self, InstructionOrder, OrderPermutation};
        let owner = order_owner(ctx, block).unwrap();
        let root = owner
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let value_name = |value: Value| match value.defining_entity() {
            DefiningEntity::Op(operation) => {
                optional_attr::<InstructionName>(ctx, operation, "hlo_name")
                    .unwrap()
                    .unwrap()
                    .0
                    .clone()
            }
            DefiningEntity::Block(block) => fusion_parameters(ctx, block).unwrap().0
                [argument_index(ctx, block, value).unwrap()]
            .name
            .clone()
            .unwrap(),
        };
        let mut order = InstructionOrder {
            instructions: instruction_order::block_instruction_order(ctx, root)
                .unwrap()
                .into_iter()
                .rev()
                .map(|operation| value_name(result(ctx, operation).unwrap()))
                .collect(),
            ..Default::default()
        };
        let mut pending = vec![root];
        while let Some(body) = pending.pop() {
            let values = instruction_order::block_value_order(ctx, body).unwrap();
            if body != root {
                let fusion = body.deref(ctx).get_parent_op(ctx).unwrap();
                order.blocks.push(OrderPermutation {
                    subject: value_name(result(ctx, fusion).unwrap()),
                    members: values
                        .iter()
                        .rev()
                        .map(|&value| value_name(value))
                        .collect(),
                });
            }
            for value in values {
                order.users.push(OrderPermutation {
                    subject: value_name(value),
                    members: instruction_order::users(ctx, value)
                        .unwrap()
                        .into_iter()
                        .rev()
                        .map(&value_name)
                        .collect(),
                });
                order.control_successors.push(OrderPermutation {
                    subject: value_name(value),
                    members: instruction_order::control_successors(ctx, value)
                        .unwrap()
                        .into_iter()
                        .rev()
                        .map(&value_name)
                        .collect(),
                });
                if let Some(operation) = value.defining_op() {
                    for region in operation.deref(ctx).regions() {
                        pending.extend(region.deref(ctx).iter(ctx));
                    }
                }
            }
        }
        instruction_order::set_order(ctx, owner, order).unwrap();
        owner
    }

    pub(super) fn has_order(ctx: &Context, owner: Ptr<Operation>) -> bool {
        owner
            .deref(ctx)
            .attributes
            .0
            .contains_key(&Identifier::new("hlo_instruction_order"))
    }

    #[test]
    fn actual_raw_and_verified_replacements_invalidate_rich_order_only_on_commit() {
        for verification in [None, Some(VerifyOptions::default())] {
            let ctx = &mut Context::new();
            let f = alias_fixture(ctx, 1, 3);
            let owner = rich_order(ctx, f.body);
            assert!(has_order(ctx, owner));
            replace_instruction(
                ctx,
                f.projection,
                f.selected,
                forwarding_options(),
                verification,
            )
            .unwrap();
            assert!(!has_order(ctx, owner));
            verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
        }
    }

    #[test]
    fn ordering_survives_self_control_sharding_and_malformed_preflight_nonchanges() {
        for (case, verification) in (0..4)
            .flat_map(|case| [None, Some(VerifyOptions::default())].map(|policy| (case, policy)))
        {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 2);
            let old = copy(ctx, &f, f.parameters[0], "old");
            if case == 1 {
                controls(ctx, old, &["p1"]);
            }
            if case == 2 {
                old.deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_sharding"), Sharding::replicated());
                f.parameters[0]
                    .defining_op()
                    .unwrap()
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_sharding"), Sharding::maximal(0));
            }
            finish(ctx, &f, result(ctx, old).unwrap());
            let owner = rich_order(ctx, f.block);
            if case == 3 {
                old.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_metadata"),
                    StringAttr::new("wrong".into()),
                );
            }
            let before = f.module.disp(ctx).to_string();
            let replacement = if case == 0 {
                result(ctx, old).unwrap()
            } else {
                f.parameters[0]
            };
            let outcome = replace_instruction(
                ctx,
                old,
                replacement,
                ReplaceInstructionOptions {
                    preserve_sharding: true,
                    ..Default::default()
                },
                verification,
            );
            if case == 3 {
                assert!(outcome.is_err());
            } else {
                assert!(!outcome.unwrap().changed);
            }
            assert!(has_order(ctx, owner));
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn raw_cleanup_invalidation_does_not_clear_unrelated_computation_order() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        finish(ctx, &f, f.parameters[0]);
        let owner = rich_order(ctx, f.block);
        let signature = FunctionType::get(ctx, vec![f.ty], vec![f.ty]).into();
        let other = ComputationOp::new(ctx, Identifier::new("other"), signature);
        let container = owner.deref(ctx).get_parent_block().unwrap();
        other.get_operation().insert_at_back(container, ctx);
        let other_block = BasicBlock::new(ctx, None, vec![]);
        other_block.insert_at_back(other.body(ctx), ctx);
        let input = ParameterOp::new(ctx, f.ty, vec![], Index(0));
        input.get_operation().insert_at_back(other_block, ctx);
        name(ctx, input.get_operation(), "other_input");
        ReturnOp::new(ctx, input.result(ctx))
            .get_operation()
            .insert_at_back(other_block, ctx);
        let other_owner = rich_order(ctx, other_block);
        let before = other.get_operation().disp(ctx).to_string();
        remove_instruction_and_unused_operands(ctx, dead, false, None).unwrap();
        assert!(!has_order(ctx, owner));
        assert!(has_order(ctx, other_owner));
        assert_eq!(other.get_operation().disp(ctx).to_string(), before);
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn raw_deduplication_invalidates_nested_storage_and_relation_permutations() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 0, 2);
        controls(ctx, f.projection, &[]);
        let old = CopyOp::new(ctx, f.outer.ty, vec![f.outer.parameters[0]]).get_operation();
        old.insert_before(ctx, f.fusion);
        name(ctx, old, "duplicate");
        Operation::replace_operand(f.fusion, ctx, 1, result(ctx, old).unwrap());
        let owner = rich_order(ctx, f.body);
        replace_instruction(ctx, old, f.outer.parameters[0], Default::default(), None).unwrap();
        assert!(!has_order(ctx, owner));
        assert_eq!(f.body.deref(ctx).get_num_arguments(), 1);
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 1);
        verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_final_control_cycle_rejection_preserves_rich_order_and_source() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        controls(ctx, f.parameters[1].defining_op().unwrap(), &["p0"]);
        controls(ctx, old, &["p1"]);
        finish(ctx, &f, result(ctx, old).unwrap());
        let owner = rich_order(ctx, f.block);
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx);
        assert!(
            replace_instruction(
                ctx,
                old,
                f.parameters[0],
                forwarding_options(),
                Some(VerifyOptions::default()),
            )
            .is_err()
        );
        assert!(has_order(ctx, owner));
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert!(f.parameters[0].uses(ctx) == uses);
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn protected_after_all_cleanup_noop_preserves_ordering() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let token = shape::token(ctx);
        let effect = CustomCallOp::new(ctx, token, vec![], StringAttr::new("effect".into()));
        effect.get_operation().insert_at_back(f.block, ctx);
        name(ctx, effect.get_operation(), "effect");
        effect.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_has_side_effect"),
            crate::attributes::Bools(vec![true]),
        );
        let after = AfterAllOp::new(ctx, token, vec![effect.result(ctx)]);
        after.get_operation().insert_at_back(f.block, ctx);
        name(ctx, after.get_operation(), "after");
        finish(ctx, &f, f.parameters[0]);
        let owner = rich_order(ctx, f.block);
        let before = f.module.disp(ctx).to_string();
        assert!(
            remove_instruction_and_unused_operands(ctx, after.get_operation(), false, None)
                .unwrap()
                .is_empty()
        );
        assert!(has_order(ctx, owner));
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    #[test]
    fn verified_removal_commits_without_replacing_surviving_operations() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        let ret = finish(ctx, &f, f.parameters[0]);
        let removed = remove_instruction_and_unused_operands(
            ctx,
            dead,
            false,
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert_eq!(removed, HashSet::from([dead]));
        assert_eq!(ret.deref(ctx).get_parent_block(), Some(f.block));
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_no_match_leaves_source_and_uses_unchanged() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        controls(ctx, old, &["p1"]);
        finish(ctx, &f, result(ctx, old).unwrap());
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx).len();
        let outcome = replace_instruction(
            ctx,
            old,
            f.parameters[0],
            Default::default(),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert!(!outcome.changed);
        assert!(outcome.removed.is_empty());
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert_eq!(f.parameters[0].uses(ctx).len(), uses);
    }

    #[test]
    fn verified_edit_rejects_custom_unbounded_policy_without_implicitly_defaulting() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = shape::dynamic_array(ctx, ElementType::F32, vec![Dimension::Unbounded]).unwrap();
        let f = fixture_with_type(ctx, 1, ty);
        let pair = shape::tuple(ctx, vec![ty, ty]).unwrap();
        let tuple = TupleOp::new(ctx, pair, vec![f.parameters[0], f.parameters[0]]);
        tuple.get_operation().insert_at_back(f.block, ctx);
        let projection = GetTupleElementOp::new(ctx, ty, vec![tuple.result(ctx)], Index(1));
        projection.get_operation().insert_at_back(f.block, ctx);
        ReturnOp::new(ctx, projection.result(ctx))
            .get_operation()
            .insert_at_back(f.block, ctx);
        let before = f.module.disp(ctx).to_string();
        assert!(
            replace_instruction(
                ctx,
                projection.get_operation(),
                f.parameters[0],
                Default::default(),
                Some(VerifyOptions::default()),
            )
            .is_err()
        );
        assert_eq!(f.module.disp(ctx).to_string(), before);
        let options = VerifyOptions {
            supported_unbounded_dynamic_op: Some(|ctx, operation| {
                matches!(
                    opcode(ctx, operation),
                    Ok(Opcode::Parameter | Opcode::Tuple | Opcode::GetTupleElement)
                )
            }),
            ..Default::default()
        };
        let failure = replace_instruction(
            ctx,
            projection.get_operation(),
            f.parameters[0],
            Default::default(),
            Some(options),
        )
        .err()
        .expect("custom verifier callbacks require an explicit future policy contract");
        assert!(matches!(
            failure.err.downcast_ref::<crate::Diagnostic>(),
            Some(crate::Diagnostic::UnsupportedCallback {
                field: "supported_unbounded_dynamic_op"
            })
        ));
        assert_eq!(f.module.disp(ctx).to_string(), before);
        replace_instruction(
            ctx,
            projection.get_operation(),
            f.parameters[0],
            Default::default(),
            None,
        )
        .unwrap();
        verify_with_options(ctx, f.module, options).unwrap();
        assert_eq!(
            f.block
                .deref(ctx)
                .get_terminator(ctx)
                .unwrap()
                .deref(ctx)
                .get_operand(0),
            f.parameters[0]
        );
    }

    thread_local! {
        static POLICY_ORIGINALS: std::cell::RefCell<HashSet<Ptr<Operation>>> =
            std::cell::RefCell::new(HashSet::new());
    }

    fn only_originals(_: &Context, operation: Ptr<Operation>) -> bool {
        POLICY_ORIGINALS.with(|originals| originals.borrow().contains(&operation))
    }

    fn except_originals(_: &Context, operation: Ptr<Operation>) -> bool {
        POLICY_ORIGINALS.with(|originals| !originals.borrow().contains(&operation))
    }

    #[test]
    fn original_identity_layout_policy_false_positive_is_rejected_before_any_mutation() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let a = shape::with_layout(
            ctx,
            f.ty,
            Layout {
                minor_to_major: vec![1, 0],
                ..Default::default()
            },
        )
        .unwrap();
        let b = shape::with_layout(
            ctx,
            f.ty,
            Layout {
                minor_to_major: vec![0, 1],
                ..Default::default()
            },
        )
        .unwrap();
        // A replacement may keep a compatible shape but change its layout.
        f.parameters[0].set_type(ctx, a);
        f.parameters[1].set_type(ctx, b);
        let old = CopyOp::new(ctx, a, vec![f.parameters[0]]).get_operation();
        old.insert_at_back(f.block, ctx);
        name(ctx, old, "old");
        let survivor = CopyOp::new(ctx, a, vec![result(ctx, old).unwrap()]).get_operation();
        survivor.insert_at_back(f.block, ctx);
        name(ctx, survivor, "survivor");
        finish(ctx, &f, result(ctx, survivor).unwrap());
        let owner = rich_order(ctx, f.block);
        POLICY_ORIGINALS.with(|originals| *originals.borrow_mut() = HashSet::from([survivor]));
        let options = VerifyOptions {
            instruction_can_change_layout: Some(except_originals),
            ..Default::default()
        };
        verify_with_options(ctx, f.module, options).unwrap();
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx);
        for _ in 0..2 {
            let failure =
                replace_instruction(ctx, old, f.parameters[1], Default::default(), Some(options))
                    .err()
                    .expect("identity-sensitive layout predicate cannot be rehearsed");
            assert!(matches!(
                failure.err.downcast_ref::<crate::Diagnostic>(),
                Some(crate::Diagnostic::UnsupportedCallback {
                    field: "instruction_can_change_layout"
                })
            ));
            assert_eq!(f.module.disp(ctx).to_string(), before);
            assert!(has_order(ctx, owner));
            assert!(f.parameters[0].uses(ctx) == uses);
        }
        // Demonstrate why the original and scratch policies are not equivalent.
        let mut mapping = IrMapping::new();
        let scratch = clone_operation(
            f.module,
            ctx,
            &mut IRRewriter::<DummyListener>::default(),
            &mut mapping,
        );
        let cloned_old = mapping.lookup_op(old).unwrap();
        let cloned_input = mapping.lookup_value(f.parameters[1]).unwrap();
        replace_instruction_impl(ctx, cloned_old, cloned_input, Default::default()).unwrap();
        verify_with_options(ctx, scratch, options).unwrap();
        Operation::erase(scratch, ctx);
        replace_instruction(
            ctx,
            old,
            f.parameters[1],
            Default::default(),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert!(verify_with_options(ctx, f.module, options).is_err());
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
        POLICY_ORIGINALS.with(|originals| originals.borrow_mut().clear());
    }

    #[test]
    fn original_identity_unbounded_policy_false_negative_fails_closed_and_raw_still_works() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = shape::dynamic_array(ctx, ElementType::F32, vec![Dimension::Unbounded]).unwrap();
        let f = fixture_with_type(ctx, 1, ty);
        let old = copy(ctx, &f, f.parameters[0], "old");
        ReturnOp::new(ctx, result(ctx, old).unwrap())
            .get_operation()
            .insert_at_back(f.block, ctx);
        let owner = rich_order(ctx, f.block);
        POLICY_ORIGINALS.with(|originals| {
            *originals.borrow_mut() = operation_tree(ctx, f.module).into_iter().collect()
        });
        let options = VerifyOptions {
            supported_unbounded_dynamic_op: Some(only_originals),
            ..Default::default()
        };
        verify_with_options(ctx, f.module, options).unwrap();
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx);
        let scratch = clone_operation(
            f.module,
            ctx,
            &mut IRRewriter::<DummyListener>::default(),
            &mut IrMapping::new(),
        );
        assert!(verify_with_options(ctx, scratch, options).is_err());
        Operation::erase(scratch, ctx);
        assert!(
            replace_instruction(ctx, old, f.parameters[0], Default::default(), Some(options))
                .is_err()
        );
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert!(has_order(ctx, owner));
        assert!(f.parameters[0].uses(ctx) == uses);
        replace_instruction(ctx, old, f.parameters[0], Default::default(), None).unwrap();
        verify_with_options(ctx, f.module, options).unwrap();
        POLICY_ORIGINALS.with(|originals| originals.borrow_mut().clear());
    }

    #[test]
    fn custom_shape_size_policy_is_rejected_for_verified_removal_then_default_recovers() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        finish(ctx, &f, f.parameters[0]);
        let owner = rich_order(ctx, f.block);
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx);
        let options = VerifyOptions {
            shape_size: Some(|ctx, shape| shape::byte_size(ctx, shape, -1)),
            ..Default::default()
        };
        let Err(failure) = remove_instruction_and_unused_operands(ctx, dead, false, Some(options))
        else {
            panic!("custom size callbacks cannot be silently substituted");
        };
        assert!(matches!(
            failure.err.downcast_ref::<crate::Diagnostic>(),
            Some(crate::Diagnostic::UnsupportedCallback {
                field: "shape_size"
            })
        ));
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert!(has_order(ctx, owner));
        assert!(f.parameters[0].uses(ctx) == uses);
        remove_instruction_and_unused_operands(ctx, dead, false, Some(VerifyOptions::default()))
            .unwrap();
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn cross_body_replacement_is_rejected_without_mutating_either_body() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let other = fixture_with_type(ctx, 1, f.ty);
        let old = CopyOp::new(ctx, f.ty, vec![f.parameters[0]]);
        old.get_operation().insert_at_back(f.block, ctx);
        finish(ctx, &f, old.result(ctx));
        finish(ctx, &other, other.parameters[0]);
        let before = [
            f.module.disp(ctx).to_string(),
            other.module.disp(ctx).to_string(),
        ];
        assert!(
            replace_instruction(
                ctx,
                old.get_operation(),
                other.parameters[0],
                Default::default(),
                None,
            )
            .is_err()
        );
        assert_eq!(
            [
                f.module.disp(ctx).to_string(),
                other.module.disp(ctx).to_string()
            ],
            before
        );
    }

    #[test]
    fn tuple_projection_forwarding_preserves_all_element_formats_and_dynamic_geometry() {
        for &element in ElementType::ALL {
            for dimension in [
                Dimension::Static(3),
                Dimension::Bounded(3),
                Dimension::Unbounded,
            ] {
                let ctx = &mut Context::new();
                crate::register(ctx);
                let ty = shape::dynamic_array(ctx, element, vec![dimension]).unwrap();
                let f = fixture_with_type(ctx, 1, ty);
                let pair = shape::tuple(ctx, vec![ty, ty]).unwrap();
                let tuple = TupleOp::new(ctx, pair, vec![f.parameters[0], f.parameters[0]]);
                tuple.get_operation().insert_at_back(f.block, ctx);
                let projection = GetTupleElementOp::new(ctx, ty, vec![tuple.result(ctx)], Index(1));
                projection.get_operation().insert_at_back(f.block, ctx);
                let ret = ReturnOp::new(ctx, projection.result(ctx)).get_operation();
                ret.insert_at_back(f.block, ctx);
                let options = crate::verification::VerifyOptions {
                    supported_unbounded_dynamic_op: Some(|ctx, operation| {
                        matches!(
                            opcode(ctx, operation),
                            Ok(Opcode::Parameter | Opcode::Tuple | Opcode::GetTupleElement)
                        )
                    }),
                    ..Default::default()
                };
                crate::verification::verify_with_options(ctx, f.module, options).unwrap();
                replace_instruction(
                    ctx,
                    projection.get_operation(),
                    f.parameters[0],
                    Default::default(),
                    None,
                )
                .unwrap();
                assert_eq!(ret.deref(ctx).get_operand(0).get_type(ctx), ty);
                assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
                crate::verification::verify_with_options(ctx, f.module, options).unwrap();
            }
        }
    }
}
