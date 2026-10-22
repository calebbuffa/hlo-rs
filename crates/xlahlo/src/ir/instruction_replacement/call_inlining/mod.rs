//! Transactional native `CallInliner::Inline`, not the module pass.
//!
//! Based on OpenXLA's `CallInliner::Inline` in `service/call_inliner.cc`.
//! Schedule splicing follows `CallInliner::Run`. Pass eligibility, call-graph
//! iteration and DCE belong to the optimizer consumer.
//!
//! Shares creation-capable rehearsal/publication with verified instruction
//! replacement. Whole-computation cloning remains private.
use super::{ReplaceInstructionOptions, alias_owners, replace_instruction, restore_alias_owners};
use crate::{
    analysis::effects::SideEffects,
    attributes::{Callees, FrontendAttributes, OpMetadata, StringPair},
    ir::{
        ControlPredecessors, InstructionName, Schedule,
        instruction_order::{instruction_order, invalidate_order},
    },
    opcode::Opcode,
    ops::{
        ComputationOp, ReturnOp,
        control::{CallOp, ParameterOp},
    },
    shape,
    verification::{VerifyOptions, error, optional_attr, require},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::cloning::IrMapping,
    linked_list::ContainsLinkedList,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    result::Result,
    r#type::Typed,
    value::Value,
};
use std::collections::{HashMap, HashSet};

mod call_metadata;
pub(super) mod defusing;
pub use defusing::{DefusedFusion, defuse};
pub(super) mod edit_staging;
mod instruction_cloning;
mod propagation;
use propagation::{async_computation, propagate_fusion_attributes, propagate_metadata};
mod traversal;
use instruction_cloning::{InstructionCloner, opcode, value_name};
use traversal::{body, callee, computation_postorder, module_for, predecessors, visit};

/// Policy of the direct Inline API, distinct from module-pass eligibility.
#[derive(Clone, Copy, Debug)]
pub struct InlineCallOptions {
    /// Copy pinned metadata and frontend annotations from the callee onto
    /// eligible cloned instructions and callees.
    pub propagate_metadata: bool,
}

impl Default for InlineCallOptions {
    fn default() -> Self {
        Self {
            propagate_metadata: true,
        }
    }
}

/// Keys retain the source callee identities, including formal parameters.
/// Existing caller and shared callee operations keep their original pointers.
pub struct InlinedCall {
    /// Value replacing the call result, including a newly constructed tuple
    /// when the callee returns multiple values.
    pub root: Value,
    /// Mapping from each source callee instruction to its inlined caller value.
    /// Formal parameter instructions map to the corresponding call operands.
    pub instructions: HashMap<Ptr<Operation>, Value>,
}

/// Inline every callee instruction, including unused/effectful roots.
///
/// `None` omits full verification, not transactionality or structural alias
/// preflight. `Some` also applies the caller's alias layout policy.
/// Call output/operand aliases describe the erased callable boundary. Pinned
/// Inline does not transfer them onto its replacement, including Copy/Add roots
/// (call_inliner.cc:126-151; hlo_computation.cc:2036-2100). Intrinsic cloned
/// instruction aliases and durable module input/output aliases remain intact;
/// no new storage promise is fabricated on the mapped root.
/// Existing schedules are spliced exactly and repaired; absence stays absent.
/// Caller logical-order overrides are invalidated, not silently preserved.
/// Shared callee graphs survive. As in pinned XLA, fusion frontend annotations
/// propagate onto fusible source instructions and control-flow metadata may
/// propagate into shared named control-flow callees.
pub fn inline_call(
    ctx: &mut Context,
    call: Ptr<Operation>,
    options: InlineCallOptions,
    verification: Option<VerifyOptions>,
) -> Result<InlinedCall> {
    require(
        Operation::is_op::<CallOp>(call, ctx),
        "inline_call requires a native Call",
    )?;
    let module = module_for(ctx, call)?;
    let callee = callee(ctx, module, call)?;
    let sources = instruction_order(ctx, callee)?;
    let (operands, result) = crate::verification::instruction_types(ctx, call, Opcode::Call)?;
    // Only Call alias validation uses this policy. Raw editing checks indices
    // and compatible subshapes without choosing a full module verifier policy.
    crate::verification::control::verify_aliases(
        ctx,
        call,
        Opcode::Call,
        result,
        &operands,
        verification.unwrap_or_default(),
    )?;
    edit_staging::staged_edit(
        ctx,
        module,
        verification,
        |ctx, scratch, initial| {
            let staged_call = initial
                .lookup_op(call)
                .ok_or_else(|| error("call has no rehearsal identity"))?;
            let result = inline_impl(ctx, scratch, staged_call, options)?;
            let instructions = sources
                .iter()
                .map(|&source| {
                    let staged_source = initial
                        .lookup_op(source)
                        .ok_or_else(|| error("callee has no rehearsal identity"))?;
                    let value = result
                        .instructions
                        .get(&staged_source)
                        .copied()
                        .ok_or_else(|| error("callee instruction was not inlined"))?;
                    Ok((source, value))
                })
                .collect::<Result<_>>()?;
            Ok(InlinedCall {
                root: result.root,
                instructions,
            })
        },
        |result, live| {
            Ok(InlinedCall {
                root: live.value(result.root)?,
                instructions: result
                    .instructions
                    .into_iter()
                    .map(|(source, value)| Ok((source, live.value(value)?)))
                    .collect::<Result<_>>()?,
            })
        },
    )
}

fn inline_impl(
    ctx: &mut Context,
    module: Ptr<Operation>,
    call: Ptr<Operation>,
    options: InlineCallOptions,
) -> Result<InlinedCall> {
    let callee = callee(ctx, module, call)?;
    let callee_block = body(ctx, callee)?;
    let caller_block = call
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| error("detached Call"))?;
    let sources = instruction_order(ctx, callee)?;
    let members: HashSet<_> = sources.iter().copied().collect();
    let mut names = HashMap::new();
    for &source in &sources {
        require(
            source.deref(ctx).get_num_results() == 1,
            "callee contains a noninstruction",
        )?;
        if let Some(name) = optional_attr::<InstructionName>(ctx, source, "hlo_name")? {
            require(
                names.insert(name.0.clone(), source).is_none(),
                "duplicate callee instruction name",
            )?;
        }
    }
    let terminator = callee_block
        .deref(ctx)
        .iter(ctx)
        .last()
        .ok_or_else(|| error("empty callee"))?;
    require(
        Operation::is_op::<ReturnOp>(terminator, ctx)
            && terminator.deref(ctx).get_num_operands() == 1,
        "callee has no single root",
    )?;
    let source_root = terminator
        .deref(ctx)
        .get_operand(0)
        .defining_op()
        .ok_or_else(|| error("callee root is not an instruction"))?;
    let mut order = Vec::new();
    let mut colors = HashMap::new();
    // HloComputation::Accept visits non-root sinks before the return root.
    let controlled: HashSet<_> = sources
        .iter()
        .map(|&source| predecessors(ctx, source))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    for &source in &sources {
        if source != source_root
            && !source.deref(ctx).get_result(0).is_used(ctx)
            && !optional_attr::<InstructionName>(ctx, source, "hlo_name")?
                .is_some_and(|name| controlled.contains(&name.0))
        {
            visit(
                ctx,
                source,
                &members,
                &names,
                &mut colors,
                &mut order,
                false,
            )?;
        }
    }
    visit(
        ctx,
        source_root,
        &members,
        &names,
        &mut colors,
        &mut order,
        false,
    )?;
    require(
        order.len() == sources.len(),
        "callee has an unreachable dependency cycle",
    )?;
    require(
        call.deref(ctx).get_num_results() == 1
            && shape::compatible(
                ctx,
                call.deref(ctx).get_result(0).get_type(ctx),
                source_root.deref(ctx).get_result(0).get_type(ctx),
            ),
        "call result shape mismatch",
    )?;
    let effects = SideEffects::compute(module, ctx, &mut AnalysisManager::default())?;
    propagate_fusion_attributes(ctx, call, &sources, &effects)?;
    let call_metadata = optional_attr::<OpMetadata>(ctx, call, "hlo_metadata")?
        .as_deref()
        .cloned()
        .unwrap_or_default();
    let incoming = predecessors(ctx, call)?;
    let old_schedule = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .as_deref()
        .cloned();
    let call_name =
        optional_attr::<InstructionName>(ctx, call, "hlo_name")?.map(|name| name.0.clone());
    invalidate_order(ctx, caller_block)?;
    if !controlled.is_empty() {
        let mut caller_controls = super::ControlGraph::read(ctx, caller_block)?;
        caller_controls.ensure_names(ctx, module)?;
        caller_controls.write(ctx)?;
    }
    let mut cloner = InstructionCloner::new(ctx, module)?;
    let mut mapping = IrMapping::new();
    let mut instructions = HashMap::new();
    let mut parameters = HashSet::new();
    for source in order {
        let value = if let Some(parameter) = Operation::get_op::<ParameterOp>(source, ctx) {
            let index = parameter
                .index(ctx)
                .ok_or_else(|| error("parameter lacks index"))?
                .0;
            let index = usize::try_from(index).map_err(|_| error("negative callee parameter"))?;
            require(
                index < call.deref(ctx).get_num_operands() && parameters.insert(index),
                "callee parameter index is invalid or duplicated",
            )?;
            let value = call.deref(ctx).get_operand(index);
            require(
                shape::compatible(
                    ctx,
                    value.get_type(ctx),
                    source.deref(ctx).get_result(0).get_type(ctx),
                ),
                "call operand shape mismatch",
            )?;
            mapping.map_value(source.deref(ctx).get_result(0), value);
            value
        } else {
            let clone = cloner.clone_instruction(ctx, module, source, call, &mut mapping)?;
            let provenance = call_metadata::original_value(ctx, call, source)?;
            clone
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&Identifier::new("hlo_original_value"));
            if let Some(provenance) = provenance {
                clone
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_original_value"), provenance);
            }
            if options.propagate_metadata {
                propagate_metadata(ctx, module, clone, &call_metadata, &mut HashSet::new())?;
            }
            let mut controls = Vec::new();
            for name in predecessors(ctx, source)? {
                let predecessor = names
                    .get(&name)
                    .ok_or_else(|| error("unknown control predecessor"))?;
                let value = instructions
                    .get(predecessor)
                    .copied()
                    .ok_or_else(|| error("unmapped control predecessor"))?;
                let name = value_name(ctx, value)?;
                if !controls.contains(&name) {
                    controls.push(name);
                }
            }
            for name in &incoming {
                if !controls.contains(name) {
                    controls.push(name.clone());
                }
            }
            clone
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&Identifier::new("hlo_control_predecessors"));
            if !controls.is_empty() {
                clone.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(controls),
                );
            }
            clone.deref(ctx).get_result(0)
        };
        instructions.insert(source, value);
    }
    require(
        parameters.len() == call.deref(ctx).get_num_operands(),
        "callee parameter arity mismatch",
    )?;
    let root = instructions[&source_root];
    let provenance = super::Annotations::read(ctx, root)?.original_value;
    let tree = edit_staging::tree(ctx, module);
    let mut aliases = alias_owners(ctx, &tree)?;
    let original = call.deref(ctx).get_result(0);
    for snapshot in &mut aliases {
        for owner in &mut snapshot.owners {
            if *owner == original {
                *owner = root;
            }
        }
    }
    let preserve_frontend_attributes = !call.deref(ctx).operands().any(|operand| operand == root);
    let result = replace_instruction(
        ctx,
        call,
        root,
        ReplaceInstructionOptions {
            preserve_sharding: false,
            relay_control_dependency: true,
            remove_unused_operands: false,
            preserve_frontend_attributes,
        },
        None,
    )?;
    require(result.changed, "Call replacement unexpectedly refused")?;
    if provenance.is_none()
        && let Some(operation) = root.defining_op()
    {
        operation
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&Identifier::new("hlo_original_value"));
    }
    let mut root_annotations = super::Annotations::read(ctx, root)?;
    root_annotations.original_value = provenance;
    root_annotations.write(ctx, root)?;
    restore_alias_owners(ctx, aliases, &result.removed)?;
    if let Some(mut old_schedule) = old_schedule {
        let mut repair = HashSet::new();
        let caller = caller_block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("caller has no computation"))?;
        require(
            Operation::is_op::<ComputationOp>(caller, ctx)
                || Operation::is_op::<crate::ops::control::FusionOp>(caller, ctx),
            "scheduled Call has no native computation or fusion body",
        )?;
        // Inline fusion computations have no independent native schedule entry.
        // Their containing instruction retains its outer-module schedule slot.
        if let Some(caller) = Operation::get_op::<ComputationOp>(caller, ctx)
            && let Some(caller_name) = caller.get_attr_hlo_computation_name(ctx)
            && old_schedule
                .0
                .iter()
                .any(|sequence| sequence.computation == *caller_name)
        {
            repair.insert(caller_name.clone());
            let call_name = call_name
                .as_ref()
                .ok_or_else(|| error("scheduled Call has no name"))?;
            let callee_name = Operation::get_op::<ComputationOp>(callee, ctx)
                .ok_or_else(|| error("scheduled callee is not a computation"))?
                .get_attr_hlo_computation_name(ctx)
                .ok_or_else(|| error("unnamed callee"))?;
            let callee_sequence = old_schedule
                .0
                .iter()
                .find(|sequence| sequence.computation == *callee_name)
                .ok_or_else(|| error("scheduled Call requires a callee sequence"))?;
            let splice = callee_sequence
                .instructions
                .iter()
                .map(|name| {
                    names
                        .get(name)
                        .copied()
                        .ok_or_else(|| error("unresolved scheduled callee instruction"))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|&source| !Operation::is_op::<ParameterOp>(source, ctx))
                .map(|source| value_name(ctx, instructions[&source]))
                .collect::<Result<Vec<_>>>()?;
            let sequence = old_schedule
                .0
                .iter_mut()
                .find(|sequence| sequence.computation == *caller_name)
                .ok_or_else(|| error("scheduled caller sequence disappeared"))?;
            let slot = sequence
                .instructions
                .iter()
                .position(|name| name == call_name)
                .ok_or_else(|| error("scheduled caller omits Call"))?;
            require(
                sequence
                    .instructions
                    .iter()
                    .filter(|name| *name == call_name)
                    .count()
                    == 1,
                "scheduled caller duplicates Call",
            )?;
            sequence.instructions.splice(slot..=slot, splice);
        }
        // Keep new async-owned computation sequences created by the cloner.
        if let Some(current) = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
            .as_deref()
            .cloned()
        {
            for sequence in current.0 {
                if !old_schedule
                    .0
                    .iter()
                    .any(|existing| existing.computation == sequence.computation)
                {
                    repair.insert(sequence.computation.clone());
                    old_schedule.0.push(sequence);
                }
            }
        }
        module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_schedule"), old_schedule);
        crate::ir::schedule_update::update_schedule_for_computations(ctx, module, &repair)?;
    }
    Ok(InlinedCall { root, instructions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Axes, FusionKind, Index, OutputOperandAliases, OutputOperandAliasing},
        ir::{FusionParameter, FusionParameters, InstructionName, InstructionSequence},
        ir::{OriginalArray, OriginalValue, OriginalValueElement},
        ops::{
            control::{CustomCallOp, FusionOp},
            elementwise::AddOp,
            movement::CopyOp,
        },
        types::ElementType,
    };
    use pliron::{
        attribute::Attribute,
        builtin::{attributes::StringAttr, types::FunctionType},
        op::Op,
        parsable::parse_from_str,
        printable::Printable,
        r#type::TypeHandle,
    };

    pub(super) struct Fixture {
        pub(super) module: Ptr<Operation>,
        pub(super) caller: Ptr<Operation>,
        pub(super) callee: Ptr<Operation>,
        pub(super) caller_block: Ptr<BasicBlock>,
        pub(super) callee_block: Ptr<BasicBlock>,
        pub(super) parameter: Value,
        pub(super) formal: Ptr<Operation>,
        pub(super) call: Ptr<Operation>,
        pub(super) ret: Ptr<Operation>,
        pub(super) ty: TypeHandle,
    }

    pub(super) fn attr<T: Attribute>(
        ctx: &Context,
        operation: Ptr<Operation>,
        key: &'static str,
        value: T,
    ) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new(key), value);
    }

    pub(super) fn name(ctx: &Context, operation: Ptr<Operation>, value: &str) {
        attr(ctx, operation, "hlo_name", InstructionName(value.into()));
    }

    pub(super) fn computation(
        ctx: &mut Context,
        module: Ptr<Operation>,
        symbol: &'static str,
        ty: TypeHandle,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>, Ptr<Operation>) {
        let signature = FunctionType::get(ctx, vec![ty], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new(symbol), signature);
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        computation.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let parameter = ParameterOp::new(ctx, ty, vec![], Index(0));
        parameter.get_operation().insert_at_back(block, ctx);
        name(ctx, parameter.get_operation(), &format!("{symbol}.p"));
        (
            computation.get_operation(),
            block,
            parameter.get_operation(),
        )
    }

    pub(super) fn fixture(ctx: &mut Context, identity: bool) -> Fixture {
        crate::register(ctx);
        let ty = shape::array(ctx, ElementType::F32, vec![]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("inlining")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let (callee, callee_block, formal) = computation(ctx, module, "callee", ty);
        let root = if identity {
            formal
        } else {
            let input = formal.deref(ctx).get_result(0);
            let add = AddOp::new(ctx, ty, vec![input, input]).get_operation();
            add.insert_at_back(callee_block, ctx);
            name(ctx, add, "callee.add");
            add
        };
        let result = root.deref(ctx).get_result(0);
        ReturnOp::new(ctx, result)
            .get_operation()
            .insert_at_back(callee_block, ctx);
        let (caller, caller_block, parameter) = computation(ctx, module, "main", ty);
        let parameter = parameter.deref(ctx).get_result(0);
        let call =
            CallOp::new(ctx, ty, vec![parameter], Callees(vec!["callee".into()])).get_operation();
        call.insert_at_back(caller_block, ctx);
        name(ctx, call, "call");
        let value = call.deref(ctx).get_result(0);
        let ret = ReturnOp::new(ctx, value).get_operation();
        ret.insert_at_back(caller_block, ctx);
        crate::verification::verify(ctx, module).unwrap();
        Fixture {
            module,
            caller,
            callee,
            caller_block,
            callee_block,
            parameter,
            formal,
            call,
            ret,
            ty,
        }
    }

    pub(super) fn source_root(ctx: &Context, f: &Fixture) -> Ptr<Operation> {
        f.callee_block
            .deref(ctx)
            .iter(ctx)
            .last()
            .unwrap()
            .deref(ctx)
            .get_operand(0)
            .defining_op()
            .unwrap()
    }

    fn copy_before(
        ctx: &mut Context,
        before: Ptr<Operation>,
        value: Value,
        instruction_name: &str,
    ) -> Ptr<Operation> {
        let ty = value.get_type(ctx);
        let copy = CopyOp::new(ctx, ty, vec![value]).get_operation();
        copy.insert_before(ctx, before);
        name(ctx, copy, instruction_name);
        copy
    }

    pub(super) fn checked(ctx: &mut Context, f: &Fixture) -> InlinedCall {
        inline_call(
            ctx,
            f.call,
            Default::default(),
            Some(VerifyOptions::default()),
        )
        .unwrap()
    }

    pub(super) fn provenance(name: &str) -> OriginalValue {
        OriginalValue {
            elements: vec![OriginalValueElement {
                shape_index: vec![],
                original_array: vec![OriginalArray {
                    instruction_name: name.into(),
                    shape_index: vec![],
                }],
            }],
            ..Default::default()
        }
    }

    pub(super) fn roundtrip(ctx: &Context, module: Ptr<Operation>) {
        let text = module.disp(ctx).to_string();
        let mut fresh = Context::new();
        crate::register(&mut fresh);
        let parsed = parse_from_str(
            pliron::irfmt::parsers::spaced(Operation::top_level_parser()),
            &mut fresh,
            &text,
        )
        .unwrap();
        crate::verification::verify(&fresh, parsed).unwrap();
    }

    #[test]
    fn root_and_all_unused_pure_and_effectful_instructions_are_cloned_without_touching_shared_callee()
     {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let root = source_root(ctx, &f);
        let formal = f.formal.deref(ctx).get_result(0);
        let unused = copy_before(ctx, root, formal, "unused");
        let effect = CustomCallOp::new(ctx, f.ty, vec![], StringAttr::new("side_effect".into()))
            .get_operation();
        effect.insert_before(ctx, root);
        name(ctx, effect, "effect");
        attr(
            ctx,
            effect,
            "hlo_has_side_effect",
            crate::attributes::Bools(vec![true]),
        );
        let callee_before = f.callee.disp(ctx).to_string();
        let source_uses = f.formal.deref(ctx).get_result(0).num_uses(ctx);
        let second = CallOp::new(ctx, f.ty, vec![f.parameter], Callees(vec!["callee".into()]))
            .get_operation();
        second.insert_before(ctx, f.ret);
        name(ctx, second, "shared.call");
        let output = checked(ctx, &f);
        assert_eq!(output.instructions.len(), 4);
        assert_eq!(output.instructions[&f.formal], f.parameter);
        assert_eq!(output.root, output.instructions[&root]);
        assert_ne!(output.instructions[&unused].defining_op().unwrap(), unused);
        assert_ne!(output.instructions[&effect].defining_op().unwrap(), effect);
        assert_eq!(f.ret.deref(ctx).get_operand(0), output.root);
        assert_eq!(callee_before, f.callee.disp(ctx).to_string());
        assert_eq!(source_uses, f.formal.deref(ctx).get_result(0).num_uses(ctx));
        assert_eq!(second.deref(ctx).get_parent_block(), Some(f.caller_block));
        crate::verification::verify(ctx, f.module).unwrap();
        roundtrip(ctx, f.module);
    }

    #[test]
    fn incoming_controls_fan_out_to_every_clone_and_outgoing_controls_follow_mapped_root() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let root = source_root(ctx, &f);
        let formal = f.formal.deref(ctx).get_result(0);
        let inner = copy_before(ctx, root, formal, "inner");
        attr(
            ctx,
            root,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["inner".into(), "callee.p".into()]),
        );
        let before = copy_before(ctx, f.call, f.parameter, "before");
        attr(
            ctx,
            f.call,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["before".into()]),
        );
        let after = copy_before(ctx, f.ret, f.parameter, "after");
        attr(
            ctx,
            after,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["call".into()]),
        );
        let output = checked(ctx, &f);
        let cloned_inner = output.instructions[&inner].defining_op().unwrap();
        assert_eq!(predecessors(ctx, cloned_inner).unwrap(), vec!["before"]);
        let root_controls = predecessors(ctx, output.root.defining_op().unwrap()).unwrap();
        assert!(root_controls.contains(&value_name(ctx, output.instructions[&inner]).unwrap()));
        assert!(root_controls.contains(&"main.p".into()));
        assert!(root_controls.contains(&"before".into()));
        assert_eq!(
            predecessors(ctx, after).unwrap(),
            vec![value_name(ctx, output.root).unwrap()]
        );
        assert_eq!(before.deref(ctx).get_parent_block(), Some(f.caller_block));
        roundtrip(ctx, f.module);
    }

    #[test]
    fn identity_root_keeps_existing_operand_provenance_and_does_not_leak_frontend_annotations_backward()
     {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let parameter = f.parameter.defining_op().unwrap();
        attr(
            ctx,
            parameter,
            "hlo_original_value",
            provenance("caller.parameter"),
        );
        attr(
            ctx,
            f.call,
            "hlo_original_value",
            provenance("call.provenance"),
        );
        attr(
            ctx,
            f.call,
            "hlo_frontend_attributes",
            FrontendAttributes::new([("inlineable".into(), "true".into())]),
        );
        let before = CustomCallOp::new(ctx, f.ty, vec![], StringAttr::new("independent".into()))
            .get_operation();
        before.insert_before(ctx, f.call);
        name(ctx, before, "before");
        attr(
            ctx,
            f.call,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["before".into()]),
        );
        let output = checked(ctx, &f);
        assert_eq!(output.root, f.parameter);
        assert_eq!(f.ret.deref(ctx).get_operand(0), f.parameter);
        assert!(
            optional_attr::<FrontendAttributes>(ctx, parameter, "hlo_frontend_attributes")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            *optional_attr::<OriginalValue>(ctx, parameter, "hlo_original_value")
                .unwrap()
                .unwrap(),
            provenance("caller.parameter")
        );
        assert_eq!(predecessors(ctx, parameter).unwrap(), vec!["before"]);
        assert_eq!(before.deref(ctx).get_parent_block(), Some(f.caller_block));
    }

    #[test]
    fn identity_root_restores_absent_operand_provenance_after_call_replacement() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        attr(
            ctx,
            f.call,
            "hlo_original_value",
            provenance("call.provenance"),
        );
        let output = checked(ctx, &f);
        assert_eq!(output.root, f.parameter);
        assert!(
            optional_attr::<OriginalValue>(
                ctx,
                f.parameter.defining_op().unwrap(),
                "hlo_original_value",
            )
            .unwrap()
            .is_none()
        );
        roundtrip(ctx, f.module);
    }

    #[test]
    fn exact_callee_schedule_splices_the_call_slot_and_absence_never_creates_a_schedule() {
        for scheduled in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, false);
            let root = source_root(ctx, &f);
            let formal = f.formal.deref(ctx).get_result(0);
            let dead = copy_before(ctx, root, formal, "dead");
            let tail = copy_before(ctx, f.ret, f.parameter, "tail");
            if scheduled {
                attr(
                    ctx,
                    f.module,
                    "hlo_schedule",
                    Schedule(vec![
                        InstructionSequence {
                            computation: "main".into(),
                            instructions: vec!["main.p".into(), "call".into(), "tail".into()],
                        },
                        InstructionSequence {
                            computation: "callee".into(),
                            instructions: vec![
                                "callee.p".into(),
                                "callee.add".into(),
                                "dead".into(),
                            ],
                        },
                    ]),
                );
            }
            let output = checked(ctx, &f);
            let schedule = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule").unwrap();
            if scheduled {
                let schedule = schedule.unwrap();
                let caller = schedule
                    .0
                    .iter()
                    .find(|sequence| sequence.computation == "main")
                    .unwrap();
                assert_eq!(
                    caller.instructions,
                    vec![
                        "main.p".into(),
                        value_name(ctx, output.root).unwrap(),
                        value_name(ctx, output.instructions[&dead]).unwrap(),
                        "tail".into(),
                    ]
                );
                assert_eq!(
                    schedule
                        .0
                        .iter()
                        .find(|sequence| sequence.computation == "callee")
                        .unwrap()
                        .instructions,
                    vec!["callee.p", "callee.add", "dead"]
                );
            } else {
                assert!(schedule.is_none());
            }
            assert_eq!(tail.deref(ctx).get_parent_block(), Some(f.caller_block));
        }
    }

    #[test]
    fn inline_schedule_repair_does_not_schedule_or_name_untouched_excluded_threads() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let (worker, worker_block, parameter) = computation(ctx, f.module, "worker", f.ty);
        attr(
            ctx,
            worker,
            "hlo_execution_thread",
            crate::ir::ExecutionThread("worker".into()),
        );
        parameter
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&Identifier::new("hlo_name"));
        let input = parameter.deref(ctx).get_result(0);
        let copy = CopyOp::new(ctx, f.ty, vec![input]);
        copy.get_operation().insert_at_back(worker_block, ctx);
        let output = copy.result(ctx);
        ReturnOp::new(ctx, output)
            .get_operation()
            .insert_at_back(worker_block, ctx);
        attr(
            ctx,
            f.module,
            "hlo_schedule",
            Schedule(vec![
                InstructionSequence {
                    computation: "main".into(),
                    instructions: vec!["main.p".into(), "call".into()],
                },
                InstructionSequence {
                    computation: "callee".into(),
                    instructions: vec!["callee.p".into(), "callee.add".into()],
                },
            ]),
        );
        let before = worker.disp(ctx).to_string();
        let output = checked(ctx, &f);
        assert_eq!(before, worker.disp(ctx).to_string());
        assert_eq!(
            copy.get_operation().deref(ctx).get_parent_block(),
            Some(worker_block)
        );
        let schedule = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
            .unwrap()
            .unwrap();
        assert_eq!(schedule.0.len(), 2);
        assert!(
            !schedule
                .0
                .iter()
                .any(|sequence| sequence.computation == "worker")
        );
        assert_eq!(
            schedule
                .0
                .iter()
                .find(|sequence| sequence.computation == "main")
                .unwrap()
                .instructions,
            ["main.p".to_owned(), value_name(ctx, output.root).unwrap()]
        );
        drop(schedule);
        roundtrip(ctx, f.module);
    }

    #[test]
    fn nested_fusion_parameters_controls_and_native_regions_are_remapped_without_external_values() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let ret = f.callee_block.deref(ctx).iter(ctx).last().unwrap();
        let input = f.formal.deref(ctx).get_result(0);
        let fusion = FusionOp::new(ctx, f.ty, vec![input, input], FusionKind::Loop);
        fusion.get_operation().insert_before(ctx, ret);
        name(ctx, fusion.get_operation(), "fusion");
        let block = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
        block.insert_at_back(fusion.body(ctx), ctx);
        let left = block.deref(ctx).get_argument(0);
        let right = block.deref(ctx).get_argument(1);
        let add = AddOp::new(ctx, f.ty, vec![left, right]);
        add.get_operation().insert_at_back(block, ctx);
        name(ctx, add.get_operation(), "inner.add");
        attr(
            ctx,
            add.get_operation(),
            "hlo_control_predecessors",
            ControlPredecessors(vec!["right".into()]),
        );
        block.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_fusion_parameters"),
            FusionParameters(vec![
                FusionParameter {
                    name: Some("left".into()),
                    ..Default::default()
                },
                FusionParameter {
                    name: Some("right".into()),
                    ..Default::default()
                },
            ]),
        );
        let root = add.result(ctx);
        ReturnOp::new(ctx, root)
            .get_operation()
            .insert_at_back(block, ctx);
        Operation::replace_operand(ret, ctx, 0, fusion.result(ctx));
        let output = checked(ctx, &f);
        let cloned = output.root.defining_op().unwrap();
        let cloned_block = cloned
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        assert_ne!(cloned_block, block);
        let parameters = crate::ir::fusion_parameters(ctx, cloned_block).unwrap();
        assert_ne!(parameters.0[0].name.as_deref(), Some("left"));
        assert_ne!(parameters.0[1].name.as_deref(), Some("right"));
        let cloned_add = cloned_block.deref(ctx).iter(ctx).next().unwrap();
        assert_eq!(
            cloned_add.deref(ctx).get_operand(0),
            cloned_block.deref(ctx).get_argument(0)
        );
        assert_eq!(
            predecessors(ctx, cloned_add).unwrap(),
            vec![parameters.0[1].name.clone().unwrap()]
        );
        assert_eq!(block.deref(ctx).get_num_arguments(), 2);
        roundtrip(ctx, f.module);
    }

    #[test]
    fn calls_inside_fusion_bodies_preserve_outer_schedule_and_existing_argument_identity() {
        for scheduled in [false, true] {
            for identity in [false, true] {
                let ctx = &mut Context::new();
                let f = fixture(ctx, identity);
                let fusion = FusionOp::new(ctx, f.ty, vec![f.parameter], FusionKind::Loop);
                let fusion_op = fusion.get_operation();
                fusion_op.insert_before(ctx, f.ret);
                name(ctx, fusion_op, "caller.fusion");
                let block = BasicBlock::new(ctx, None, vec![f.ty]);
                block.insert_at_back(fusion.body(ctx), ctx);
                let argument = block.deref(ctx).get_argument(0);
                block.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_fusion_parameters"),
                    FusionParameters(vec![FusionParameter {
                        name: Some("fusion.p".into()),
                        ..Default::default()
                    }]),
                );
                f.call.unlink(ctx);
                f.call.insert_at_back(block, ctx);
                Operation::replace_operand(f.call, ctx, 0, argument);
                let value = f.call.deref(ctx).get_result(0);
                let ret = ReturnOp::new(ctx, value).get_operation();
                ret.insert_at_back(block, ctx);
                Operation::replace_operand(f.ret, ctx, 0, fusion.result(ctx));
                attr(
                    ctx,
                    f.call,
                    "hlo_original_value",
                    provenance("call.provenance"),
                );
                if scheduled {
                    let mut callee = vec!["callee.p".into()];
                    if !identity {
                        callee.push("callee.add".into());
                    }
                    attr(
                        ctx,
                        f.module,
                        "hlo_schedule",
                        Schedule(vec![
                            InstructionSequence {
                                computation: "callee".into(),
                                instructions: callee,
                            },
                            InstructionSequence {
                                computation: "main".into(),
                                instructions: vec!["main.p".into(), "caller.fusion".into()],
                            },
                        ]),
                    );
                }
                let output = checked(ctx, &f);
                assert_eq!(ret.deref(ctx).get_operand(0), output.root);
                assert_eq!(block.deref(ctx).get_argument(0), argument);
                if identity {
                    assert_eq!(output.root, argument);
                    assert!(
                        crate::ir::fusion_parameters(ctx, block).unwrap().0[0]
                            .original_value
                            .is_none()
                    );
                } else {
                    let root = output.root.defining_op().unwrap();
                    assert_eq!(root.deref(ctx).get_parent_block(), Some(block));
                    assert_eq!(root.deref(ctx).get_operand(0), argument);
                }
                let schedule = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule").unwrap();
                if scheduled {
                    assert_eq!(
                        schedule
                            .as_ref()
                            .unwrap()
                            .0
                            .iter()
                            .find(|sequence| sequence.computation == "main")
                            .unwrap()
                            .instructions,
                        ["main.p", "caller.fusion"]
                    );
                } else {
                    assert!(schedule.is_none());
                }
                drop(schedule);
                roundtrip(ctx, f.module);
            }
        }
    }

    #[test]
    fn caller_fusion_operand_dedup_preserves_alias_owners_and_survivor_pointers() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let input = f.call.deref(ctx).get_result(0);
        let fusion = FusionOp::new(ctx, f.ty, vec![input, f.parameter], FusionKind::Loop);
        fusion.get_operation().insert_before(ctx, f.ret);
        name(ctx, fusion.get_operation(), "consumer");
        let block = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
        block.insert_at_back(fusion.body(ctx), ctx);
        let left = block.deref(ctx).get_argument(0);
        let right = block.deref(ctx).get_argument(1);
        let add = AddOp::new(ctx, f.ty, vec![left, right]);
        add.get_operation().insert_at_back(block, ctx);
        name(ctx, add.get_operation(), "consumer.add");
        let root = add.result(ctx);
        ReturnOp::new(ctx, root)
            .get_operation()
            .insert_at_back(block, ctx);
        attr(
            ctx,
            fusion.get_operation(),
            "hlo_aliases",
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![]),
                operand_index: 1,
                operand_shape_index: Axes(vec![]),
            }]),
        );
        Operation::replace_operand(f.ret, ctx, 0, fusion.result(ctx));
        checked(ctx, &f);
        assert_eq!(fusion.get_operation().deref(ctx).get_num_operands(), 1);
        assert_eq!(block.deref(ctx).get_num_arguments(), 1);
        let aliases =
            optional_attr::<OutputOperandAliases>(ctx, fusion.get_operation(), "hlo_aliases")
                .unwrap()
                .unwrap();
        assert_eq!(aliases.0[0].operand_index, 0);
        assert_eq!(
            fusion.get_operation().deref(ctx).get_operand(0),
            f.parameter
        );
        assert_eq!(
            add.get_operation().deref(ctx).get_parent_block(),
            Some(block)
        );
    }

    #[test]
    fn call_boundary_aliases_expire_without_fabricating_root_aliases() {
        use crate::ir::{AliasEntry, AliasKind, InputOutputAliases};
        for root_kind in [Opcode::Parameter, Opcode::Copy, Opcode::Add] {
            for verified in [false, true] {
                for scheduled in [false, true] {
                    let ctx = &mut Context::new();
                    let f = fixture(ctx, root_kind != Opcode::Add);
                    if root_kind == Opcode::Copy {
                        let ret = f.callee_block.deref(ctx).iter(ctx).last().unwrap();
                        let input = f.formal.deref(ctx).get_result(0);
                        let copy = copy_before(ctx, ret, input, "callee.copy");
                        let value = copy.deref(ctx).get_result(0);
                        Operation::replace_operand(ret, ctx, 0, value);
                    }
                    attr(
                        ctx,
                        f.call,
                        "hlo_aliases",
                        OutputOperandAliases(vec![OutputOperandAliasing {
                            output_shape_index: Axes(vec![]),
                            operand_index: 0,
                            operand_shape_index: Axes(vec![]),
                        }]),
                    );
                    let module_aliases = InputOutputAliases(vec![AliasEntry {
                        output_shape_index: Axes(vec![]),
                        parameter_number: 0,
                        parameter_shape_index: Axes(vec![]),
                        kind: AliasKind::MustAlias,
                    }]);
                    attr(
                        ctx,
                        f.module,
                        "hlo_input_output_aliases",
                        module_aliases.clone(),
                    );
                    if scheduled {
                        let mut instructions = vec!["callee.p".into()];
                        if root_kind != Opcode::Parameter {
                            instructions.push(
                                if root_kind == Opcode::Copy {
                                    "callee.copy"
                                } else {
                                    "callee.add"
                                }
                                .into(),
                            );
                        }
                        attr(
                            ctx,
                            f.module,
                            "hlo_schedule",
                            Schedule(vec![
                                InstructionSequence {
                                    computation: "main".into(),
                                    instructions: vec!["main.p".into(), "call".into()],
                                },
                                InstructionSequence {
                                    computation: "callee".into(),
                                    instructions,
                                },
                            ]),
                        );
                    }
                    let source = f.callee.disp(ctx).to_string();
                    let output = inline_call(
                        ctx,
                        f.call,
                        Default::default(),
                        verified.then(VerifyOptions::default),
                    )
                    .unwrap();
                    assert_eq!(
                        opcode(ctx, output.root.defining_op().unwrap()).unwrap(),
                        root_kind
                    );
                    assert!(
                        optional_attr::<OutputOperandAliases>(
                            ctx,
                            output.root.defining_op().unwrap(),
                            "hlo_aliases"
                        )
                        .unwrap()
                        .is_none()
                    );
                    assert_eq!(source, f.callee.disp(ctx).to_string());
                    assert_eq!(
                        *optional_attr::<InputOutputAliases>(
                            ctx,
                            f.module,
                            "hlo_input_output_aliases"
                        )
                        .unwrap()
                        .unwrap(),
                        module_aliases
                    );
                    roundtrip(ctx, f.module);
                }
            }
        }
    }

    #[test]
    fn malformed_call_boundary_aliases_roll_back_under_raw_and_verified_policies() {
        for verified in [false, true] {
            for case in 0..6 {
                let ctx = &mut Context::new();
                let f = fixture(ctx, false);
                let mut alias = OutputOperandAliasing {
                    output_shape_index: Axes(vec![]),
                    operand_index: 0,
                    operand_shape_index: Axes(vec![]),
                };
                match case {
                    0 => alias.operand_index = -1,
                    1 => alias.operand_index = 1,
                    2 => alias.output_shape_index = Axes(vec![-1]),
                    3 => alias.operand_shape_index = Axes(vec![0]),
                    _ => {}
                }
                if case == 4 {
                    attr(
                        ctx,
                        f.call,
                        "hlo_aliases",
                        StringAttr::new("invalid".into()),
                    );
                } else {
                    attr(
                        ctx,
                        f.call,
                        "hlo_aliases",
                        OutputOperandAliases(vec![alias]),
                    );
                }
                if case == 5 {
                    let different = shape::array(ctx, ElementType::S32, vec![]).unwrap();
                    let result = f.call.deref(ctx).get_result(0);
                    result.set_type(ctx, different);
                }
                let before = f.module.disp(ctx).to_string();
                let uses = f.parameter.num_uses(ctx);
                for _ in 0..2 {
                    assert!(
                        inline_call(
                            ctx,
                            f.call,
                            Default::default(),
                            verified.then(VerifyOptions::default)
                        )
                        .is_err()
                    );
                    assert_eq!(f.module.disp(ctx).to_string(), before);
                    assert_eq!(f.parameter.num_uses(ctx), uses);
                    assert_eq!(f.call.deref(ctx).get_parent_block(), Some(f.caller_block));
                }
                f.call
                    .deref_mut(ctx)
                    .attributes
                    .0
                    .remove(&Identifier::new("hlo_aliases"));
                if case == 5 {
                    let result = f.call.deref(ctx).get_result(0);
                    result.set_type(ctx, f.ty);
                }
                checked(ctx, &f);
            }
        }
    }

    #[test]
    fn tuple_projection_and_transitive_intrinsic_aliases_survive_boundary_erasure() {
        use crate::ops::control::{GetTupleElementOp, TupleOp};
        for verified in [false, true] {
            let ctx = &mut Context::new();
            crate::register(ctx);
            let scalar = shape::array(ctx, ElementType::F32, vec![]).unwrap();
            let pair = shape::tuple(ctx, vec![scalar, scalar]).unwrap();
            let module = ModuleOp::new(ctx, Identifier::new("tuple_aliases")).get_operation();
            crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
            let (callee, callee_block, parameter) = computation(ctx, module, "callee", pair);
            let input = parameter.deref(ctx).get_result(0);
            let mut values = Vec::new();
            let mut intrinsic = Vec::new();
            let alias = OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![]),
                operand_index: 0,
                operand_shape_index: Axes(vec![]),
            }]);
            for index in [1, 0] {
                let projection = GetTupleElementOp::new(ctx, scalar, vec![input], Index(index));
                projection.get_operation().insert_at_back(callee_block, ctx);
                name(
                    ctx,
                    projection.get_operation(),
                    &format!("projection.{index}"),
                );
                let mut value = projection.result(ctx);
                for depth in 0..2 {
                    let custom = CustomCallOp::new(
                        ctx,
                        scalar,
                        vec![value],
                        StringAttr::new("alias".into()),
                    );
                    custom.get_operation().insert_at_back(callee_block, ctx);
                    name(
                        ctx,
                        custom.get_operation(),
                        &format!("alias.{index}.{depth}"),
                    );
                    attr(ctx, custom.get_operation(), "hlo_aliases", alias.clone());
                    intrinsic.push(custom.get_operation());
                    value = custom.result(ctx);
                }
                values.push(value);
            }
            let tuple = TupleOp::new(ctx, pair, values);
            tuple.get_operation().insert_at_back(callee_block, ctx);
            name(ctx, tuple.get_operation(), "repacked");
            let value = tuple.result(ctx);
            ReturnOp::new(ctx, value)
                .get_operation()
                .insert_at_back(callee_block, ctx);
            let (_, block, parameter) = computation(ctx, module, "main", pair);
            let input = parameter.deref(ctx).get_result(0);
            let call = CallOp::new(ctx, pair, vec![input], Callees(vec!["callee".into()]));
            call.get_operation().insert_at_back(block, ctx);
            name(ctx, call.get_operation(), "call");
            attr(
                ctx,
                call.get_operation(),
                "hlo_aliases",
                OutputOperandAliases(vec![
                    OutputOperandAliasing {
                        output_shape_index: Axes(vec![0]),
                        operand_index: 0,
                        operand_shape_index: Axes(vec![1]),
                    },
                    OutputOperandAliasing {
                        output_shape_index: Axes(vec![1]),
                        operand_index: 0,
                        operand_shape_index: Axes(vec![0]),
                    },
                ]),
            );
            let value = call.result(ctx);
            let ret = ReturnOp::new(ctx, value).get_operation();
            ret.insert_at_back(block, ctx);
            let source = callee.disp(ctx).to_string();
            let output = inline_call(
                ctx,
                call.get_operation(),
                Default::default(),
                verified.then(VerifyOptions::default),
            )
            .unwrap();
            assert_eq!(source, callee.disp(ctx).to_string());
            assert_eq!(ret.deref(ctx).get_operand(0), output.root);
            let root = output.root.defining_op().unwrap();
            assert_eq!(opcode(ctx, root).unwrap(), Opcode::Tuple);
            assert!(
                optional_attr::<OutputOperandAliases>(ctx, root, "hlo_aliases")
                    .unwrap()
                    .is_none()
            );
            for source in intrinsic {
                let cloned = output.instructions[&source].defining_op().unwrap();
                assert_eq!(
                    *optional_attr::<OutputOperandAliases>(ctx, cloned, "hlo_aliases")
                        .unwrap()
                        .unwrap(),
                    alias
                );
            }
            roundtrip(ctx, module);
        }
    }

    #[test]
    fn nested_call_keeps_its_own_alias_contract_not_outer_operand_indices() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let (leaf, leaf_block, leaf_parameter) = computation(ctx, f.module, "leaf", f.ty);
        let input = leaf_parameter.deref(ctx).get_result(0);
        ReturnOp::new(ctx, input)
            .get_operation()
            .insert_at_back(leaf_block, ctx);
        let ret = f.callee_block.deref(ctx).iter(ctx).last().unwrap();
        let input = f.formal.deref(ctx).get_result(0);
        let nested = CallOp::new(ctx, f.ty, vec![input], Callees(vec!["leaf".into()]));
        nested.get_operation().insert_before(ctx, ret);
        name(ctx, nested.get_operation(), "nested.alias");
        let aliases = OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(vec![]),
            operand_index: 0,
            operand_shape_index: Axes(vec![]),
        }]);
        attr(ctx, nested.get_operation(), "hlo_aliases", aliases.clone());
        attr(ctx, f.call, "hlo_aliases", aliases.clone());
        let value = nested.result(ctx);
        Operation::replace_operand(ret, ctx, 0, value);
        let source = f.callee.disp(ctx).to_string();
        let leaf_before = leaf.disp(ctx).to_string();
        let first = checked(ctx, &f);
        let root = first.root.defining_op().unwrap();
        assert_eq!(
            *optional_attr::<OutputOperandAliases>(ctx, root, "hlo_aliases")
                .unwrap()
                .unwrap(),
            aliases
        );
        let second = inline_call(ctx, root, Default::default(), Some(Default::default())).unwrap();
        assert_eq!(second.root, f.parameter);
        assert!(
            optional_attr::<OutputOperandAliases>(
                ctx,
                f.parameter.defining_op().unwrap(),
                "hlo_aliases"
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(source, f.callee.disp(ctx).to_string());
        assert_eq!(leaf_before, leaf.disp(ctx).to_string());
        roundtrip(ctx, f.module);
    }

    #[test]
    fn malformed_callee_controls_types_and_unsupported_policies_roll_back_and_recover() {
        for case in 0..4 {
            let ctx = &mut Context::new();
            let f = fixture(ctx, false);
            let root = source_root(ctx, &f);
            match case {
                0 => attr(
                    ctx,
                    root,
                    "hlo_control_predecessors",
                    ControlPredecessors(vec!["missing".into()]),
                ),
                1 => {
                    let different = shape::array(ctx, ElementType::F32, vec![7]).unwrap();
                    let result = f.call.deref(ctx).get_result(0);
                    result.set_type(ctx, different);
                }
                2 => attr(ctx, f.call, "hlo_callees", Callees(vec!["missing".into()])),
                _ => {}
            }
            let mut policy = VerifyOptions::default();
            if case == 3 {
                policy.supported_unbounded_dynamic_op = Some(|_, _| true);
            }
            let before = f.module.disp(ctx).to_string();
            let uses = f.parameter.num_uses(ctx);
            for _ in 0..2 {
                assert!(inline_call(ctx, f.call, Default::default(), Some(policy)).is_err());
                assert_eq!(before, f.module.disp(ctx).to_string());
                assert_eq!(uses, f.parameter.num_uses(ctx));
            }
            match case {
                0 => {
                    root.deref_mut(ctx)
                        .attributes
                        .0
                        .remove(&Identifier::new("hlo_control_predecessors"));
                }
                1 => {
                    let result = f.call.deref(ctx).get_result(0);
                    result.set_type(ctx, f.ty);
                }
                2 => attr(ctx, f.call, "hlo_callees", Callees(vec!["callee".into()])),
                _ => {}
            }
            checked(ctx, &f);
            roundtrip(ctx, f.module);
        }
    }

    #[test]
    fn staging_adopts_new_tree_and_rolls_back_failed_projection_without_changing_survivors() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameter.num_uses(ctx);
        let result: Result<Value> = edit_staging::staged_edit(
            ctx,
            f.module,
            None,
            |ctx, _, initial| {
                let call = initial.lookup_op(f.call).unwrap();
                let input = initial.lookup_value(f.parameter).unwrap();
                let copy = copy_before(ctx, call, input, "created");
                Ok(copy.deref(ctx).get_result(0))
            },
            |_, _| Err(error("intentional projected-output failure")),
        );
        assert!(result.is_err());
        assert_eq!(before, f.module.disp(ctx).to_string());
        assert_eq!(uses, f.parameter.num_uses(ctx));
        let output = edit_staging::staged_edit(
            ctx,
            f.module,
            Some(Default::default()),
            |ctx, _, initial| {
                let call = initial.lookup_op(f.call).unwrap();
                let input = initial.lookup_value(f.parameter).unwrap();
                let copy = copy_before(ctx, call, input, "created");
                Ok(copy.deref(ctx).get_result(0))
            },
            |result, live| live.value(result),
        )
        .unwrap();
        assert_eq!(
            output.defining_op().unwrap().deref(ctx).get_parent_block(),
            Some(f.caller_block)
        );
        assert_eq!(f.call.deref(ctx).get_parent_block(), Some(f.caller_block));
        assert_eq!(
            f.caller
                .deref(ctx)
                .get_region(0)
                .deref(ctx)
                .get_entry_block(),
            Some(f.caller_block)
        );
    }

    #[test]
    fn nested_calls_are_cloned_then_independently_inlined_without_mutating_shared_callers() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let (leaf, leaf_block, leaf_parameter) = computation(ctx, f.module, "leaf", f.ty);
        let input = leaf_parameter.deref(ctx).get_result(0);
        let add = AddOp::new(ctx, f.ty, vec![input, input]);
        add.get_operation().insert_at_back(leaf_block, ctx);
        name(ctx, add.get_operation(), "leaf.add");
        attr(
            ctx,
            add.get_operation(),
            "hlo_metadata",
            OpMetadata {
                op_name: "leaf".into(),
                ..Default::default()
            },
        );
        let value = add.result(ctx);
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(leaf_block, ctx);
        let ret = f.callee_block.deref(ctx).iter(ctx).last().unwrap();
        let input = f.formal.deref(ctx).get_result(0);
        let nested = CallOp::new(ctx, f.ty, vec![input], Callees(vec!["leaf".into()]));
        nested.get_operation().insert_before(ctx, ret);
        name(ctx, nested.get_operation(), "nested");
        attr(
            ctx,
            nested.get_operation(),
            "hlo_metadata",
            OpMetadata {
                op_name: "inner".into(),
                ..Default::default()
            },
        );
        Operation::replace_operand(ret, ctx, 0, nested.result(ctx));
        attr(
            ctx,
            f.call,
            "hlo_metadata",
            OpMetadata {
                op_name: "outer".into(),
                ..Default::default()
            },
        );
        let source = f.callee.disp(ctx).to_string();
        let leaf_before = leaf.disp(ctx).to_string();
        let first = checked(ctx, &f);
        assert_eq!(
            opcode(ctx, first.root.defining_op().unwrap()).unwrap(),
            Opcode::Call
        );
        assert_eq!(leaf_before, leaf.disp(ctx).to_string());
        let second = inline_call(
            ctx,
            first.root.defining_op().unwrap(),
            Default::default(),
            Some(Default::default()),
        )
        .unwrap();
        assert_eq!(
            opcode(ctx, second.root.defining_op().unwrap()).unwrap(),
            Opcode::Add
        );
        assert_eq!(
            optional_attr::<OpMetadata>(ctx, second.root.defining_op().unwrap(), "hlo_metadata")
                .unwrap()
                .unwrap()
                .op_name,
            "outer/inner/leaf"
        );
        assert_eq!(source, f.callee.disp(ctx).to_string());
        assert_eq!(leaf_before, leaf.disp(ctx).to_string());
        assert_eq!(f.ret.deref(ctx).get_operand(0), second.root);
        roundtrip(ctx, f.module);
    }

    #[test]
    fn would_create_control_cycle_rolls_back_under_explicit_verification_policy() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let before = copy_before(ctx, f.call, f.parameter, "depends_on_parameter");
        attr(
            ctx,
            f.call,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["depends_on_parameter".into()]),
        );
        let text = f.module.disp(ctx).to_string();
        for _ in 0..2 {
            assert!(
                inline_call(ctx, f.call, Default::default(), Some(Default::default())).is_err()
            );
            assert_eq!(text, f.module.disp(ctx).to_string());
            assert_eq!(before.deref(ctx).get_parent_block(), Some(f.caller_block));
        }
        f.call
            .deref_mut(ctx)
            .attributes
            .0
            .remove(&Identifier::new("hlo_control_predecessors"));
        checked(ctx, &f);
    }

    #[test]
    fn staging_rejects_survivor_relocation_and_no_external_values_escape_on_failure() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let text = f.module.disp(ctx).to_string();
        let result: Result<()> = edit_staging::staged_edit(
            ctx,
            f.module,
            None,
            |ctx, _, initial| {
                let call = initial.lookup_op(f.call).unwrap();
                let source_root = initial.lookup_op(f.formal).unwrap();
                call.unlink(ctx);
                call.insert_before(ctx, source_root);
                Ok(())
            },
            |(), _| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(text, f.module.disp(ctx).to_string());
        assert_eq!(f.parameter.num_uses(ctx), 1);
        checked(ctx, &f);
    }

    #[test]
    fn staging_rejects_created_external_capture_and_failed_output_projection_before_publication() {
        for external_capture in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, true);
            let text = f.module.disp(ctx).to_string();
            let uses = f.parameter.num_uses(ctx);
            let result: Result<Value> = edit_staging::staged_edit(
                ctx,
                f.module,
                None,
                |ctx, _, initial| {
                    let call = initial.lookup_op(f.call).unwrap();
                    let operand = if external_capture {
                        f.parameter
                    } else {
                        initial.lookup_value(f.parameter).unwrap()
                    };
                    let created = CopyOp::new(ctx, f.ty, vec![operand]);
                    created.get_operation().insert_before(ctx, call);
                    Ok(created.result(ctx))
                },
                |_, live| live.value(f.parameter),
            );
            assert!(result.is_err());
            assert_eq!(text, f.module.disp(ctx).to_string());
            assert_eq!(uses, f.parameter.num_uses(ctx));
            assert_eq!(f.call.deref(ctx).get_parent_block(), Some(f.caller_block));
            checked(ctx, &f);
        }
    }

    #[test]
    fn nameless_valid_native_calls_inline_without_renaming_shared_callee_instructions() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let root = source_root(ctx, &f);
        for operation in [f.call, f.formal, root, f.parameter.defining_op().unwrap()] {
            operation
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&Identifier::new("hlo_name"));
        }
        let source = f.callee.disp(ctx).to_string();
        let output = checked(ctx, &f);
        assert_eq!(source, f.callee.disp(ctx).to_string());
        assert_eq!(output.instructions[&f.formal], f.parameter);
        assert_eq!(
            opcode(ctx, output.root.defining_op().unwrap()).unwrap(),
            Opcode::Add
        );
        roundtrip(ctx, f.module);
    }

    #[test]
    fn two_parameters_can_map_the_same_caller_value_without_new_parameter_instructions() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let second = ParameterOp::new(ctx, f.ty, vec![], Index(1));
        let root = source_root(ctx, &f);
        second.get_operation().insert_before(ctx, root);
        name(ctx, second.get_operation(), "second.parameter");
        Operation::replace_operand(root, ctx, 1, second.result(ctx));
        let signature = FunctionType::get(ctx, vec![f.ty, f.ty], vec![f.ty]).into();
        Operation::get_op::<ComputationOp>(f.callee, ctx)
            .unwrap()
            .set_attr_hlo_computation_signature(
                ctx,
                pliron::builtin::attributes::TypeAttr::new(signature),
            );
        Operation::push_operand(f.call, ctx, f.parameter);
        let output = checked(ctx, &f);
        assert_eq!(output.instructions[&f.formal], f.parameter);
        assert_eq!(output.instructions[&second.get_operation()], f.parameter);
        let root = output.root.defining_op().unwrap();
        assert_eq!(root.deref(ctx).get_operand(0), f.parameter);
        assert_eq!(root.deref(ctx).get_operand(1), f.parameter);
        assert_eq!(
            f.caller_block
                .deref(ctx)
                .iter(ctx)
                .filter(|&operation| Operation::is_op::<ParameterOp>(operation, ctx))
                .count(),
            1
        );
    }
}
