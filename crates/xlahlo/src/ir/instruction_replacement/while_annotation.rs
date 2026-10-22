//! Atomic While annotation and constant-forwarding batch.
use super::{
    alias_owners, containing_module, endpoint_block, operation_tree, order_owner,
    restore_alias_owners,
};
use crate::{
    analysis::effects::SideEffects,
    attributes::{Bytes, Index, Payload},
    ir::{
        BackendConfigPayload, InstructionName,
        instruction_order::{control_predecessors, control_successors},
        set_backend_config_payload,
    },
    literal::Literal,
    ops::{
        ComputationOp,
        control::{ConstantOp, FusionOp, GetTupleElementOp, WhileOp},
    },
    scalar::semantics::Scalar,
    shape,
    verification::{VerifyOptions, error, optional_attr, require},
};
use pliron::{
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    result::Result,
    r#type::Typed,
    value::Value,
};
use std::collections::HashSet;

/// Forwarding intent; callers supply a separately proved final integer value.
#[derive(Clone, Copy, Debug)]
pub struct WhileInductionForwarding {
    /// Element of the While state tuple whose matching result extractions are
    /// replaced with constants.
    pub tuple_index: usize,
    /// Caller-proved post-loop value to encode in each forwarding constant.
    pub final_value: Scalar,
}

/// Already encoded config bytes and optional proved forwarding intent.
///
/// Encoding/decoding policy belongs to the caller's existing codec. This batch
/// publishes an inline payload without changing legacy bytes or shared tables.
#[derive(Clone, Debug)]
pub struct WhileLoopAnnotationRequest {
    /// Native While operation receiving the inline backend-config payload.
    pub while_op: Ptr<Operation>,
    /// Already encoded bytes to store as the operation's backend config.
    pub backend_config: Bytes,
    /// Optional caller-proved value to forward through matching result
    /// `GetTupleElement` operations.
    pub induction_forwarding: Option<WhileInductionForwarding>,
}

#[derive(Default)]
/// Values created while forwarding proven While induction results.
pub struct WhileLoopAnnotationBatch {
    /// One new constant per matching result GTE, including unused GTEs.
    pub constants: Vec<Value>,
}

fn uncontrolled(ctx: &Context, value: Value) -> Result<bool> {
    Ok(control_predecessors(ctx, value)?.is_empty() && control_successors(ctx, value)?.is_empty())
}

fn fresh(names: &mut HashSet<String>) -> Result<String> {
    for suffix in 0..usize::MAX {
        let name = format!("while_induction_constant_{suffix}");
        if names.insert(name.clone()) {
            return Ok(name);
        }
    }
    Err(error("While annotation instruction names exhausted"))
}

/// Publish the entire batch atomically using the existing native edit plan.
///
/// `None` omits whole-module verification, not ownership, control/effect, alias
/// or transactional preflight. `Some` uses the existing verifier callback policy.
/// Original While/GTE operations survive; this is forwarding, not DCE.
/// Inline fusion loops allow metadata-only requests; forwarding would retain
/// dead fusion operations and is rejected even without whole-module verification.
pub fn annotate_while_loops(
    ctx: &mut Context,
    module: Ptr<Operation>,
    requests: &[WhileLoopAnnotationRequest],
    verification: Option<VerifyOptions>,
) -> Result<WhileLoopAnnotationBatch> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "While annotation requires a module",
    )?;
    let mut seen = HashSet::new();
    for request in requests {
        require(
            seen.insert(request.while_op),
            "duplicate While annotation request",
        )?;
        require(
            Operation::is_op::<WhileOp>(request.while_op, ctx),
            "annotation target is not a While",
        )?;
        let value = request.while_op.deref(ctx).get_result(0);
        require(
            containing_module(ctx, endpoint_block(ctx, value)?)? == module,
            "While annotation target is outside the supplied module",
        )?;
    }
    if requests.is_empty() {
        return Ok(WhileLoopAnnotationBatch::default());
    }
    super::call_inlining::edit_staging::staged_edit(
        ctx,
        module,
        verification,
        |ctx, scratch, mapping| {
            let effects = SideEffects::compute(scratch, ctx, &mut AnalysisManager::default())?;
            let operations = operation_tree(ctx, scratch);
            let mut aliases = alias_owners(ctx, &operations)?;
            let mut names = HashSet::new();
            for &operation in &operations {
                if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                    names.insert(name.0.clone());
                }
                for region in operation.deref(ctx).regions() {
                    for bb in region.deref(ctx).iter(ctx) {
                        let bb = bb.deref(ctx);
                        let key = Identifier::new("hlo_fusion_parameters");
                        let parameters = bb.attributes.get::<crate::ir::FusionParameters>(&key);
                        require(
                            parameters.is_some() || !bb.attributes.0.contains_key(&key),
                            "mistyped fusion parameter metadata",
                        )?;
                        if let Some(parameters) = parameters {
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
            let mut constants = Vec::new();
            let mut removed = HashSet::new();
            let mut changed_computations = HashSet::new();
            for request in requests {
                let while_op = mapping
                    .lookup_op(request.while_op)
                    .ok_or_else(|| error("staged While target is missing"))?;
                let value = while_op.deref(ctx).get_result(0);
                if let Some(forwarding) = request.induction_forwarding {
                    require(
                        uncontrolled(ctx, value)?,
                        "controlled While cannot forward an induction value",
                    )?;
                    require(
                        !effects.has_side_effect(while_op)?,
                        "effectful While cannot forward an induction value",
                    )?;
                    let shape =
                        shape::subshape(ctx, value.get_type(ctx), &[forwarding.tuple_index])?;
                    require(
                        shape::is_array(ctx, shape)
                            && !shape::is_dynamic(ctx, shape)
                            && shape::rank(ctx, shape)? == 0
                            && shape::element_type(ctx, shape)?.is_integer()
                            && shape::element_type(ctx, shape)? == forwarding.final_value.element(),
                        "While forwarding requires an exact static integer scalar format",
                    )?;
                    let users: HashSet<_> =
                        value.uses(ctx).iter().map(|use_| use_.user_op()).collect();
                    let mut gtes = Vec::new();
                    // Keep physical block order, independent of use-list ordering.
                    for operation in endpoint_block(ctx, value)?.deref(ctx).iter(ctx) {
                        if !users.contains(&operation)
                            || !Operation::is_op::<GetTupleElementOp>(operation, ctx)
                        {
                            continue;
                        }
                        let index = optional_attr::<Index>(ctx, operation, "hlo_index")?
                            .ok_or_else(|| error("forwarded GTE has no typed index"))?
                            .0 as usize;
                        if index == forwarding.tuple_index {
                            gtes.push(operation);
                        }
                    }
                    for gte in gtes {
                        require(
                            !gte.deref(ctx)
                                .get_parent_op(ctx)
                                .is_some_and(|owner| Operation::is_op::<FusionOp>(owner, ctx)),
                            "induction forwarding inside fusion cannot retain dead GTEs",
                        )?;
                        let original = gte.deref(ctx).get_result(0);
                        require(
                            uncontrolled(ctx, original)?,
                            "controlled GTE cannot forward an induction value",
                        )?;
                        let declared = original.get_type(ctx);
                        require(
                            declared == shape,
                            "forwarded GTE differs from the exact tuple subshape",
                        )?;
                        let literal =
                            Literal::from_bits(ctx, declared, &[forwarding.final_value.bits()])?;
                        let name = fresh(&mut names)?;
                        let constant =
                            ConstantOp::new(ctx, declared, vec![], literal).get_operation();
                        constant
                            .deref_mut(ctx)
                            .attributes
                            .set(Identifier::new("hlo_name"), InstructionName(name));
                        constant.insert_before(ctx, gte);
                        let replacement = constant.deref(ctx).get_result(0);
                        for snapshot in &mut aliases {
                            for owner in &mut snapshot.owners {
                                if *owner == original {
                                    *owner = replacement;
                                }
                            }
                        }
                        let outcome =
                            super::forwarding::replace_all_uses_impl(ctx, gte, replacement)?;
                        removed.extend(outcome.removed);
                        let owner = order_owner(ctx, endpoint_block(ctx, original)?)?;
                        super::invalidate_order(ctx, owner);
                        let symbol = Operation::get_op::<ComputationOp>(owner, ctx)
                            .ok_or_else(|| error("forwarding owner is not a computation"))?
                            .get_attr_hlo_computation_name(ctx)
                            .ok_or_else(|| error("forwarding computation has no symbol"))?
                            .clone();
                        changed_computations.insert(symbol);
                        constants.push(replacement);
                    }
                }
                set_backend_config_payload(
                    ctx,
                    while_op,
                    BackendConfigPayload(Payload::Value(request.backend_config.clone())),
                )?;
            }
            restore_alias_owners(ctx, aliases, &removed)?;
            if !changed_computations.is_empty() {
                crate::ir::schedule_update::update_schedule_for_computations(
                    ctx,
                    scratch,
                    &changed_computations,
                )?;
            }
            Ok(constants)
        },
        |constants, live| {
            Ok(WhileLoopAnnotationBatch {
                constants: constants
                    .into_iter()
                    .map(|value| live.value(value))
                    .collect::<Result<_>>()?,
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Bools, Callees, ComparisonOrder, Direction, Index},
        ir::{self, ControlPredecessors, InstructionSequence, MetadataPayloads, Schedule},
        ops::{
            ReturnOp,
            control::{CustomCallOp, ParameterOp, TupleOp},
            elementwise::{AddOp, CompareOp},
            movement::CopyOp,
        },
        types::ElementType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{attributes::StringAttr, types::FunctionType},
        printable::Printable,
        r#type::TypeHandle,
    };

    struct Fixture {
        module: Ptr<Operation>,
        main: Ptr<BasicBlock>,
        loops: Vec<Ptr<Operation>>,
        gtes: Vec<Ptr<Operation>>,
        ret: Ptr<Operation>,
        scalar: TypeHandle,
        body: Ptr<Operation>,
    }

    #[test]
    fn inline_fusion_forwarding_is_rejected_without_dce_and_metadata_only_is_valid() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let instructions: Vec<_> = f
            .main
            .deref(ctx)
            .iter(ctx)
            .filter(|&op| op != f.ret)
            .collect();
        let fusion = crate::ops::control::FusionOp::new(
            ctx,
            f.scalar,
            vec![],
            crate::attributes::FusionKind::Loop,
        );
        fusion.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName("annotation_inline_fusion".into()),
        );
        fusion.get_operation().insert_before(ctx, f.ret);
        let bb = BasicBlock::new(ctx, None, vec![]);
        bb.insert_at_back(fusion.body(ctx), ctx);
        for instruction in instructions {
            instruction.unlink(ctx);
            instruction.insert_at_back(bb, ctx);
        }
        let result = f.gtes[0].deref(ctx).get_result(0);
        finish(ctx, bb, result);
        Operation::replace_operand(f.ret, ctx, 0, fusion.result(ctx));
        crate::verification::verify(ctx, f.module).unwrap();
        let before = f.module.disp(ctx).to_string();
        for verification in [None, Some(Default::default())] {
            assert!(
                annotate_while_loops(ctx, f.module, &[request(f.loops[0], true)], verification)
                    .is_err()
            );
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
        let result = annotate_while_loops(
            ctx,
            f.module,
            &[request(f.loops[0], false)],
            Some(Default::default()),
        )
        .unwrap();
        assert!(result.constants.is_empty());
        crate::verification::verify(ctx, f.module).unwrap();
    }

    fn append(ctx: &Context, bb: Ptr<BasicBlock>, op: Ptr<Operation>) -> Value {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName(format!("annotation_{index}")),
        );
        op.insert_at_back(bb, ctx);
        op.deref(ctx).get_result(0)
    }

    fn computation(
        ctx: &mut Context,
        module: Ptr<Operation>,
        name: &'static str,
        inputs: Vec<TypeHandle>,
        result: TypeHandle,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>, Vec<Value>) {
        let signature = FunctionType::get(ctx, inputs.clone(), vec![result]).into();
        let comp = ComputationOp::new(ctx, Identifier::new(name), signature);
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        comp.get_operation().insert_at_back(container, ctx);
        let bb = BasicBlock::new(ctx, None, vec![]);
        bb.insert_at_back(comp.body(ctx), ctx);
        let args = inputs
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let p = ParameterOp::new(ctx, ty, vec![], Index(i as u32)).get_operation();
                append(ctx, bb, p)
            })
            .collect();
        (comp.get_operation(), bb, args)
    }

    fn finish(ctx: &mut Context, bb: Ptr<BasicBlock>, value: Value) -> Ptr<Operation> {
        let ret = ReturnOp::new(ctx, value).get_operation();
        ret.insert_at_back(bb, ctx);
        ret
    }

    fn constant(ctx: &mut Context, bb: Ptr<BasicBlock>, scalar: TypeHandle, n: u128) -> Value {
        let literal = Literal::from_bits(ctx, scalar, &[n]).unwrap();
        let op = ConstantOp::new(ctx, scalar, vec![], literal).get_operation();
        append(ctx, bb, op)
    }

    fn fixture(ctx: &mut Context, loop_count: usize, duplicate_gte: bool) -> Fixture {
        crate::register(ctx);
        let module = ModuleOp::new(ctx, Identifier::new("annotation")).get_operation();
        ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let scalar = shape::scalar(ctx, ElementType::S64).unwrap();
        let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
        let state = shape::tuple(ctx, vec![scalar]).unwrap();
        let (body, bb, args) = computation(ctx, module, "body", vec![state], state);
        let gte = GetTupleElementOp::new(ctx, scalar, vec![args[0]], Index(0)).get_operation();
        let i = append(ctx, bb, gte);
        let one = constant(ctx, bb, scalar, 1);
        let add = AddOp::new(ctx, scalar, vec![i, one]).get_operation();
        let next = append(ctx, bb, add);
        let tuple = TupleOp::new(ctx, state, vec![next]).get_operation();
        let root = append(ctx, bb, tuple);
        finish(ctx, bb, root);
        let (_, bb, args) = computation(ctx, module, "condition", vec![state], pred);
        let gte = GetTupleElementOp::new(ctx, scalar, vec![args[0]], Index(0)).get_operation();
        let i = append(ctx, bb, gte);
        let five = constant(ctx, bb, scalar, 5);
        let cmp = CompareOp::new(
            ctx,
            pred,
            vec![i, five],
            Direction::Lt,
            ComparisonOrder::Total,
        )
        .get_operation();
        let cmp = append(ctx, bb, cmp);
        finish(ctx, bb, cmp);
        let (_, main, _) = computation(ctx, module, "main", vec![], scalar);
        let zero = constant(ctx, main, scalar, 0);
        let tuple = TupleOp::new(ctx, state, vec![zero]).get_operation();
        let input = append(ctx, main, tuple);
        let mut loops = Vec::new();
        let mut gtes = Vec::new();
        for _ in 0..loop_count {
            let op = WhileOp::new(
                ctx,
                state,
                vec![input],
                Callees(vec!["body".into(), "condition".into()]),
            )
            .get_operation();
            let result = append(ctx, main, op);
            loops.push(op);
            for _ in 0..if duplicate_gte { 2 } else { 1 } {
                let gte =
                    GetTupleElementOp::new(ctx, scalar, vec![result], Index(0)).get_operation();
                append(ctx, main, gte);
                gtes.push(gte);
            }
        }
        let result = gtes[0].deref(ctx).get_result(0);
        let ret = finish(ctx, main, result);
        crate::verification::verify(ctx, module).unwrap();
        Fixture {
            module,
            main,
            loops,
            gtes,
            ret,
            scalar,
            body,
        }
    }

    fn request(op: Ptr<Operation>, forward: bool) -> WhileLoopAnnotationRequest {
        WhileLoopAnnotationRequest {
            while_op: op,
            backend_config: Bytes(br#"{"known_trip_count":{"n":"5"}}"#.to_vec()),
            induction_forwarding: forward.then(|| WhileInductionForwarding {
                tuple_index: 0,
                final_value: Scalar::from_bits(ElementType::S64, 5).unwrap(),
            }),
        }
    }

    fn run(
        ctx: &mut Context,
        f: &Fixture,
        requests: &[WhileLoopAnnotationRequest],
    ) -> Result<WhileLoopAnnotationBatch> {
        annotate_while_loops(ctx, f.module, requests, Some(VerifyOptions::default()))
    }

    #[test]
    fn new_constant_per_result_gte_including_dead_gte_retains_original_nodes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, true);
        let old = operation_tree(ctx, f.module);
        let result = run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        assert_eq!(result.constants.len(), 2);
        assert_ne!(result.constants[0], result.constants[1]);
        let live = operation_tree(ctx, f.module);
        assert!(old.iter().all(|op| live.contains(op)));
        assert_eq!(f.ret.deref(ctx).get_operand(0), result.constants[0]);
        assert!(
            f.gtes
                .iter()
                .all(|gte| !gte.deref(ctx).get_result(0).is_used(ctx))
        );
        for value in result.constants {
            assert_eq!(value.get_type(ctx), f.scalar);
            let op = value.defining_op().unwrap();
            let literal = optional_attr::<Literal>(ctx, op, "hlo_literal")
                .unwrap()
                .unwrap();
            assert_eq!(Scalar::read(ctx, &literal, &[]).unwrap().bits(), 5);
            assert!(op.deref(ctx).get_parent_block().is_some());
        }
    }

    #[test]
    fn annotation_only_keeps_structure_order_and_exact_legacy_shared_payloads() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2, false);
        let legacy = StringAttr::new("unrecognized legacy bytes".into());
        f.loops[0]
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_backend_config"), legacy.clone());
        let table = MetadataPayloads(vec![Bytes(b"opaque shared bytes".to_vec())]);
        f.module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_metadata_payloads"), table.clone());
        for &op in &f.loops {
            set_backend_config_payload(ctx, op, BackendConfigPayload(Payload::Id(0))).unwrap();
        }
        let old = operation_tree(ctx, f.module);
        let other = f.loops[1].deref(ctx).attributes.clone();
        assert!(
            run(ctx, &f, &[request(f.loops[0], false)])
                .unwrap()
                .constants
                .is_empty()
        );
        assert_eq!(operation_tree(ctx, f.module), old);
        assert_eq!(f.loops[1].deref(ctx).attributes, other);
        assert_eq!(
            optional_attr::<StringAttr>(ctx, f.loops[0], "hlo_backend_config")
                .unwrap()
                .unwrap()
                .as_str(),
            legacy.as_str()
        );
        assert_eq!(
            *optional_attr::<MetadataPayloads>(ctx, f.module, "hlo_metadata_payloads")
                .unwrap()
                .unwrap(),
            table
        );
        assert!(matches!(
            crate::ir::backend_config_payload(ctx, f.loops[0]).unwrap(),
            Some(BackendConfigPayload(Payload::Value(_)))
        ));
    }

    #[test]
    fn later_invalid_forward_rolls_back_earlier_scratch_constants_and_configs() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2, true);
        let before = f.module.disp(ctx).to_string();
        let operations = operation_tree(ctx, f.module);
        let mut second = request(f.loops[1], true);
        second.induction_forwarding.as_mut().unwrap().final_value =
            Scalar::from_bits(ElementType::S32, 5).unwrap();
        assert!(run(ctx, &f, &[request(f.loops[0], true), second]).is_err());
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert_eq!(operation_tree(ctx, f.module), operations);
        let retry = run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        assert_eq!(retry.constants.len(), 2);
    }

    #[test]
    fn duplicate_nonwhile_and_outside_requests_are_diagnostics_without_writes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let other = fixture(ctx, 1, false);
        let before = f.module.disp(ctx).to_string();
        for requests in [
            vec![request(f.loops[0], false), request(f.loops[0], true)],
            vec![request(f.gtes[0], false)],
            vec![request(other.loops[0], false)],
        ] {
            assert!(run(ctx, &f, &requests).is_err());
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    fn set_control(ctx: &Context, source: Ptr<Operation>, target: Ptr<Operation>) {
        let name = optional_attr::<InstructionName>(ctx, source, "hlo_name")
            .unwrap()
            .unwrap()
            .0
            .clone();
        target.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(vec![name]),
        );
    }

    #[test]
    fn controlled_while_or_gte_forwards_refused_atomically_but_annotation_allowed() {
        for target_while in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1, false);
            let input = f.loops[0].deref(ctx).get_operand(0).defining_op().unwrap();
            set_control(
                ctx,
                input,
                if target_while { f.loops[0] } else { f.gtes[0] },
            );
            let before = f.module.disp(ctx).to_string();
            assert!(run(ctx, &f, &[request(f.loops[0], true)]).is_err());
            assert_eq!(f.module.disp(ctx).to_string(), before);
            run(ctx, &f, &[request(f.loops[0], false)]).unwrap();
        }
    }

    #[test]
    fn transitive_body_effect_refuses_forwarding_without_executing_or_erasing() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let bb = f
            .body
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let ret = bb.deref(ctx).iter(ctx).last().unwrap();
        let op = CustomCallOp::new(ctx, f.scalar, vec![], StringAttr::new("effect".into()))
            .get_operation();
        op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName("annotation_effect".into()),
        );
        op.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
        op.insert_before(ctx, ret);
        let before = f.module.disp(ctx).to_string();
        assert!(run(ctx, &f, &[request(f.loops[0], true)]).is_err());
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    fn schedule(ctx: &Context, f: &Fixture) {
        let container = f
            .module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let sequences = container
            .deref(ctx)
            .iter(ctx)
            .map(|comp| {
                let bb = comp
                    .deref(ctx)
                    .get_region(0)
                    .deref(ctx)
                    .get_entry_block()
                    .unwrap();
                let name = Operation::get_op::<ComputationOp>(comp, ctx)
                    .unwrap()
                    .get_attr_hlo_computation_name(ctx)
                    .unwrap()
                    .clone();
                InstructionSequence {
                    computation: name,
                    instructions: bb
                        .deref(ctx)
                        .iter(ctx)
                        .filter(|&op| !Operation::is_op::<ReturnOp>(op, ctx))
                        .map(|op| {
                            optional_attr::<InstructionName>(ctx, op, "hlo_name")
                                .unwrap()
                                .unwrap()
                                .0
                                .clone()
                        })
                        .collect(),
                }
            })
            .collect();
        f.module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_schedule"), Schedule(sequences));
    }

    #[test]
    fn existing_schedule_is_repaired_and_unrelated_sequences_unchanged() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, true);
        schedule(ctx, &f);
        let old = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
            .unwrap()
            .unwrap()
            .clone();
        let result = run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        let new = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
            .unwrap()
            .unwrap()
            .clone();
        assert_eq!(&new.0[..2], &old.0[..2]);
        assert_eq!(
            new.0[2].instructions.len(),
            old.0[2].instructions.len() + result.constants.len()
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn unscheduled_batch_does_not_invent_schedule_or_remove_dead_gtes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, true);
        run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        assert!(
            optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
                .unwrap()
                .is_none()
        );
        assert!(
            f.gtes
                .iter()
                .all(|op| op.deref(ctx).get_parent_block() == Some(f.main))
        );
    }

    #[test]
    fn invalid_subshape_index_and_callback_policy_leave_ir_unchanged() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let before = f.module.disp(ctx).to_string();
        let mut invalid = request(f.loops[0], true);
        invalid.induction_forwarding.as_mut().unwrap().tuple_index = 1;
        assert!(run(ctx, &f, &[invalid]).is_err());
        let options = VerifyOptions {
            instruction_can_change_layout: Some(|_, _| true),
            ..Default::default()
        };
        assert!(
            annotate_while_loops(ctx, f.module, &[request(f.loops[0], true)], Some(options))
                .is_err()
        );
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    #[test]
    fn raw_policy_keeps_atomicity_and_empty_batch_preserves_exact_ir() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let before = f.module.disp(ctx).to_string();
        assert!(
            annotate_while_loops(ctx, f.module, &[], None)
                .unwrap()
                .constants
                .is_empty()
        );
        assert_eq!(f.module.disp(ctx).to_string(), before);
        let mut wrong = request(f.loops[0], true);
        wrong.induction_forwarding.as_mut().unwrap().final_value =
            Scalar::from_bits(ElementType::F64, 5).unwrap();
        assert!(annotate_while_loops(ctx, f.module, &[wrong], None).is_err());
        assert_eq!(f.module.disp(ctx).to_string(), before);
        annotate_while_loops(ctx, f.module, &[request(f.loops[0], true)], None).unwrap();
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn malformed_payload_type_is_not_silently_repaired_by_verified_batch() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        f.loops[0]
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_backend_config_payload"), Index(0));
        let before = f.module.disp(ctx).to_string();
        assert!(run(ctx, &f, &[request(f.loops[0], false)]).is_err());
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    #[test]
    fn forwarding_copies_recovery_provenance_without_copying_unrelated_metadata() {
        use crate::{
            attributes::OpMetadata,
            ir::{OriginalArray, OriginalValue, OriginalValueElement},
        };
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let provenance = OriginalValue {
            elements: vec![OriginalValueElement {
                shape_index: vec![],
                original_array: vec![OriginalArray {
                    instruction_name: "recorded_iv".into(),
                    shape_index: vec![],
                }],
            }],
            ..Default::default()
        };
        f.gtes[0]
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_original_value"), provenance.clone());
        f.gtes[0].deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "not copied".into(),
                ..Default::default()
            },
        );
        let result = run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        let constant = result.constants[0].defining_op().unwrap();
        assert_eq!(
            *optional_attr::<OriginalValue>(ctx, constant, "hlo_original_value")
                .unwrap()
                .unwrap(),
            provenance
        );
        assert!(
            optional_attr::<OpMetadata>(ctx, constant, "hlo_metadata")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn global_constant_names_skip_existing_instruction_names() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, true);
        let original = f.gtes[0].deref(ctx).get_result(0);
        let copied = CopyOp::new(ctx, f.scalar, vec![original]).get_operation();
        copied.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName("while_induction_constant_0".into()),
        );
        copied.insert_before(ctx, f.ret);
        let result = run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        let names: Vec<_> = result
            .constants
            .iter()
            .map(|value| {
                optional_attr::<InstructionName>(ctx, value.defining_op().unwrap(), "hlo_name")
                    .unwrap()
                    .unwrap()
                    .0
                    .clone()
            })
            .collect();
        assert_eq!(
            names,
            ["while_induction_constant_1", "while_induction_constant_2"]
        );
        assert_eq!(copied.deref(ctx).get_operand(0), result.constants[0]);
    }

    #[test]
    fn aliases_of_forwarded_fusion_operands_retain_their_new_owner() {
        use crate::{
            attributes::{Axes, FusionKind, OutputOperandAliases, OutputOperandAliasing},
            ops::control::FusionOp,
        };
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1, false);
        let old = f.gtes[0].deref(ctx).get_result(0);
        let fusion = FusionOp::new(ctx, f.scalar, vec![old, old], FusionKind::Loop);
        fusion.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName("annotation_alias_fusion".into()),
        );
        fusion.get_operation().insert_before(ctx, f.ret);
        let bb = BasicBlock::new(ctx, None, vec![f.scalar; 2]);
        bb.insert_at_back(fusion.body(ctx), ctx);
        let arguments: Vec<_> = bb.deref(ctx).arguments().collect();
        let add = AddOp::new(ctx, f.scalar, arguments).get_operation();
        let root = append(ctx, bb, add);
        finish(ctx, bb, root);
        fusion.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_aliases"),
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![]),
                operand_index: 1,
                operand_shape_index: Axes(vec![]),
            }]),
        );
        Operation::replace_operand(f.ret, ctx, 0, fusion.result(ctx));
        let result = run(ctx, &f, &[request(f.loops[0], true)]).unwrap();
        assert_eq!(fusion.get_operation().deref(ctx).get_num_operands(), 1);
        assert_eq!(
            fusion.get_operation().deref(ctx).get_operand(0),
            result.constants[0]
        );
        assert_eq!(
            optional_attr::<OutputOperandAliases>(ctx, fusion.get_operation(), "hlo_aliases")
                .unwrap()
                .unwrap()
                .0[0]
                .operand_index,
            0
        );
    }
}
