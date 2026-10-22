//! Forwarding-only `HloInstruction::ReplaceAllUsesWith` behavior.
use super::*;

/// Redirect uses to a compatible value, retaining the original instruction.
///
/// Ports `HloInstruction::ReplaceAllUsesWith` and `CopyOriginalValue` at native
/// implementation. Unlike `replace_instruction`, this does not erase dead
/// producers, relay controls, inherit metadata/sharding, or rename either node.
/// `None` performs the raw edit; `Some` rehearses and verifies the complete edit
/// with the supplied policy before committing, including schedule/alias repair.
/// The verified policy's custom-callback limitation is shared with replacement.
pub fn replace_all_uses_with(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    replacement: Value,
    verification: Option<VerifyOptions>,
) -> Result<InstructionReplacement> {
    if let Some(verification) = verification {
        return verified_edit(ctx, instruction, Edit::Forward(replacement), verification);
    }
    replace_all_uses_impl(ctx, instruction, replacement)
}

/// Raw forwarding, without removing the original instruction or its operands.
///
/// A replacement that directly uses the original keeps that operand, as in
/// upstream's copy-of-self case. Other redirects must not create data cycles.
/// Controls, names, schedules and metadata are not relayed. Compatible recovery
/// provenance is copied, and noncustom fusion operands are deduplicated using
/// the same native graph-edit machinery as instruction replacement.
pub(super) fn replace_all_uses_impl(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    replacement: Value,
) -> Result<InstructionReplacement> {
    let original = result(ctx, instruction)?;
    if original == replacement {
        return Ok(InstructionReplacement::default());
    }
    let block = endpoint_block(ctx, original)?;
    require(
        endpoint_block(ctx, replacement)? == block,
        "forwarding replacement is outside the instruction body",
    )?;
    require(
        shape::compatible_ignoring_fp_precision(
            ctx,
            original.get_type(ctx),
            replacement.get_type(ctx),
        ),
        "forwarding instruction shapes are incompatible",
    )?;
    let replacement_op = replacement.defining_op();
    let users: HashSet<_> = original
        .uses(ctx)
        .iter()
        .map(|usage| usage.user_op())
        .filter(|user| Some(*user) != replacement_op)
        .collect();
    for &user in &users {
        require(
            user.deref(ctx).get_parent_block() == Some(block),
            "forwarding user is outside the instruction body",
        )?;
        if let Some(replacement) = replacement_op {
            require(
                !depends_on(ctx, replacement, user)?,
                "forwarding would create a data dependency cycle",
            )?;
        }
    }
    let old = Annotations::read(ctx, original)?;
    let mut annotations = Annotations::read(ctx, replacement)?;
    let mut provenance_changed = false;
    if shape::compatible(ctx, original.get_type(ctx), replacement.get_type(ctx))
        && let Some(provenance) = old.original_value
    {
        provenance_changed = annotations
            .original_value
            .as_ref()
            .map(|previous| previous.equal(&provenance))
            .transpose()?
            != Some(true);
        annotations.original_value = Some(provenance);
        annotations.validate(ctx, replacement)?;
    }
    if users.is_empty() && !provenance_changed {
        return Ok(InstructionReplacement::default());
    }
    let module = containing_module(ctx, block)?;
    let effects = SideEffects::compute(module, ctx, &mut AnalysisManager::default())?;
    let owner = order_owner(ctx, block)?;
    invalidate_order(ctx, owner);
    original.replace_some_uses_with(
        ctx,
        |_, usage| Some(usage.user_op()) != replacement_op,
        &replacement,
    );
    if provenance_changed {
        annotations.write(ctx, replacement)?;
    }
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.get_config_mut().set_name_on_value_replacement = false;
    let mut cleanup = Cleanup::new(module, &effects);
    for user in users {
        if Operation::is_op::<FusionOp>(user, ctx) {
            deduplicate_fusion_operands(ctx, user, &mut rewriter, &mut cleanup)?;
        }
    }
    Ok(InstructionReplacement {
        changed: true,
        removed: cleanup.removed,
    })
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        attributes::{Axes, Index, OutputOperandAliasing},
        ir::{self, InstructionSequence, instruction_replacement::replace_all_uses_with},
        ir::{OriginalArray, OriginalValueElement},
        ops::{control::ParameterOp, elementwise::AddOp, movement::CopyOp},
        types::ElementType,
    };
    use pliron::{builtin::types::FunctionType, op::Op, printable::Printable, r#type::TypeHandle};

    struct Fixture {
        module: Ptr<Operation>,
        block: Ptr<BasicBlock>,
        ty: TypeHandle,
        parameter: Value,
    }

    fn fixture(ctx: &mut Context) -> Fixture {
        crate::register(ctx);
        let ty = shape::array(ctx, ElementType::F32, vec![2]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("forward")).get_operation();
        ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let signature = FunctionType::get(ctx, vec![ty], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new("main"), signature);
        computation.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let parameter = ParameterOp::new(ctx, ty, vec![], Index(0));
        parameter.get_operation().insert_at_back(block, ctx);
        name(ctx, parameter.get_operation(), "p");
        Fixture {
            module,
            block,
            ty,
            parameter: parameter.result(ctx),
        }
    }

    fn name(ctx: &Context, operation: Ptr<Operation>, name: &str) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name.into()));
    }

    fn copy(ctx: &mut Context, fixture: &Fixture, input: Value, name_: &str) -> CopyOp {
        let copy = CopyOp::new(ctx, fixture.ty, vec![input]);
        copy.get_operation().insert_at_back(fixture.block, ctx);
        name(ctx, copy.get_operation(), name_);
        copy
    }

    fn provenance(name: &str) -> OriginalValue {
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

    fn finish(ctx: &mut Context, fixture: &Fixture, root: Value) -> Ptr<Operation> {
        let ret = ReturnOp::new(ctx, root).get_operation();
        ret.insert_at_back(fixture.block, ctx);
        crate::verification::verify(ctx, fixture.module).unwrap();
        ret
    }

    #[test]
    fn redirects_repeated_operands_and_root_without_erasing_old_instruction() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        let add = AddOp::new(ctx, fixture.ty, vec![old.result(ctx); 2]);
        add.get_operation().insert_at_back(fixture.block, ctx);
        name(ctx, add.get_operation(), "add");
        let ret = finish(ctx, &fixture, old.result(ctx));
        let schedule = Schedule(vec![InstructionSequence {
            computation: "main".into(),
            instructions: vec!["p".into(), "old".into(), "add".into()],
        }]);
        fixture
            .module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_schedule"), schedule.clone());
        let change = replace_all_uses_impl(ctx, old.get_operation(), fixture.parameter).unwrap();
        assert!(change.changed && change.removed.is_empty());
        assert_eq!(ret.deref(ctx).get_operand(0), fixture.parameter);
        assert!(
            add.get_operation()
                .deref(ctx)
                .operands()
                .all(|v| v == fixture.parameter)
        );
        assert_eq!(
            old.get_operation().deref(ctx).get_parent_block(),
            Some(fixture.block)
        );
        assert!(!old.result(ctx).is_used(ctx));
        assert_eq!(
            *fixture
                .module
                .deref(ctx)
                .attributes
                .get::<Schedule>(&Identifier::new("hlo_schedule"))
                .unwrap(),
            schedule
        );
        crate::verification::verify(ctx, fixture.module).unwrap();
        assert!(
            !replace_all_uses_impl(ctx, old.get_operation(), fixture.parameter)
                .unwrap()
                .changed
        );
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn copy_of_self_keeps_its_original_operand_and_changes_return() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        let replacement = copy(ctx, &fixture, old.result(ctx), "replacement");
        let ret = finish(ctx, &fixture, old.result(ctx));
        replace_all_uses_impl(ctx, old.get_operation(), replacement.result(ctx)).unwrap();
        assert_eq!(
            replacement.get_operation().deref(ctx).get_operand(0),
            old.result(ctx)
        );
        assert_eq!(ret.deref(ctx).get_operand(0), replacement.result(ctx));
        crate::verification::verify(ctx, fixture.module).unwrap();
        assert!(
            !replace_all_uses_impl(ctx, old.get_operation(), replacement.result(ctx))
                .unwrap()
                .changed
        );
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn recovery_only_forwarding_reports_change_without_erasing_dead_source() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        old.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("original"),
        );
        finish(ctx, &fixture, fixture.parameter);
        assert!(!old.result(ctx).is_used(ctx));
        let change = replace_all_uses_with(
            ctx,
            old.get_operation(),
            fixture.parameter,
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert!(change.changed && change.removed.is_empty());
        let parameter = fixture.parameter.defining_op().unwrap();
        assert_eq!(
            *parameter
                .deref(ctx)
                .attributes
                .get::<OriginalValue>(&Identifier::new("hlo_original_value"))
                .unwrap(),
            provenance("original")
        );
        assert_eq!(
            old.get_operation().deref(ctx).get_parent_block(),
            Some(fixture.block)
        );
        assert!(
            !replace_all_uses_with(
                ctx,
                old.get_operation(),
                fixture.parameter,
                Some(VerifyOptions::default())
            )
            .unwrap()
            .changed
        );
        crate::verification::verify(ctx, fixture.module).unwrap();
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn forwarding_parameter_uses_preserves_parameter_and_computation_signature() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let replacement = copy(ctx, &fixture, fixture.parameter, "replacement");
        let ret = finish(ctx, &fixture, fixture.parameter);
        let parent = fixture.block.deref(ctx).get_parent_op(ctx).unwrap();
        let signature = Operation::get_op::<ComputationOp>(parent, ctx)
            .unwrap()
            .get_attr_hlo_computation_signature(ctx)
            .unwrap()
            .get_type(ctx);
        replace_all_uses_with(
            ctx,
            fixture.parameter.defining_op().unwrap(),
            replacement.result(ctx),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert_eq!(ret.deref(ctx).get_operand(0), replacement.result(ctx));
        assert_eq!(
            replacement.get_operation().deref(ctx).get_operand(0),
            fixture.parameter
        );
        assert_eq!(
            Operation::get_op::<ComputationOp>(parent, ctx)
                .unwrap()
                .get_attr_hlo_computation_signature(ctx)
                .unwrap()
                .get_type(ctx),
            signature
        );
        assert_eq!(
            fixture
                .parameter
                .defining_op()
                .unwrap()
                .deref(ctx)
                .get_parent_block(),
            Some(fixture.block)
        );
        crate::verification::verify(ctx, fixture.module).unwrap();
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn forwards_only_recovery_provenance_and_keeps_control_endpoints() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        let successor = copy(ctx, &fixture, fixture.parameter, "successor");
        old.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("original"),
        );
        old.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "not forwarded".into(),
                ..Default::default()
            },
        );
        successor.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(vec!["old".into()]),
        );
        finish(ctx, &fixture, old.result(ctx));
        let before = fixture
            .parameter
            .defining_op()
            .unwrap()
            .deref(ctx)
            .attributes
            .clone();
        replace_all_uses_impl(ctx, old.get_operation(), fixture.parameter).unwrap();
        let parameter = fixture.parameter.defining_op().unwrap();
        let ir = parameter.deref(ctx);
        assert_eq!(
            *ir.attributes
                .get::<OriginalValue>(&Identifier::new("hlo_original_value"))
                .unwrap(),
            provenance("original")
        );
        assert!(
            ir.attributes
                .get::<OpMetadata>(&Identifier::new("hlo_metadata"))
                .is_none()
        );
        assert_eq!(
            ir.attributes
                .get::<InstructionName>(&Identifier::new("hlo_name"))
                .unwrap()
                .0,
            before
                .get::<InstructionName>(&Identifier::new("hlo_name"))
                .unwrap()
                .0
        );
        drop(ir);
        assert_eq!(
            successor
                .get_operation()
                .deref(ctx)
                .attributes
                .get::<ControlPredecessors>(&Identifier::new("hlo_control_predecessors"))
                .unwrap()
                .0,
            vec!["old"]
        );
        crate::verification::verify(ctx, fixture.module).unwrap();
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn rejects_indirect_cycle_without_mutating_graph() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        let middle = copy(ctx, &fixture, old.result(ctx), "middle");
        let replacement = copy(ctx, &fixture, middle.result(ctx), "replacement");
        finish(ctx, &fixture, old.result(ctx));
        let before = fixture.module.disp(ctx).to_string();
        assert!(replace_all_uses_impl(ctx, old.get_operation(), replacement.result(ctx)).is_err());
        assert_eq!(fixture.module.disp(ctx).to_string(), before);
        crate::verification::verify(ctx, fixture.module).unwrap();
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn precision_changing_forward_does_not_copy_recovery_provenance() {
        use crate::ops::elementwise::ConvertOp;
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        old.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("original"),
        );
        let half = shape::array(ctx, ElementType::BF16, vec![2]).unwrap();
        let replacement = ConvertOp::new(ctx, half, vec![fixture.parameter]);
        replacement
            .get_operation()
            .insert_at_back(fixture.block, ctx);
        name(ctx, replacement.get_operation(), "replacement");
        let result = ConvertOp::new(ctx, fixture.ty, vec![old.result(ctx)]);
        result.get_operation().insert_at_back(fixture.block, ctx);
        name(ctx, result.get_operation(), "result");
        finish(ctx, &fixture, result.result(ctx));
        replace_all_uses_impl(ctx, old.get_operation(), replacement.result(ctx)).unwrap();
        assert_eq!(
            result.get_operation().deref(ctx).get_operand(0),
            replacement.result(ctx)
        );
        assert!(
            replacement
                .get_operation()
                .deref(ctx)
                .attributes
                .get::<OriginalValue>(&Identifier::new("hlo_original_value"))
                .is_none()
        );
        crate::verification::verify(ctx, fixture.module).unwrap();
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn incompatible_shape_and_cross_body_replacements_are_premutation_errors() {
        use crate::ops::elementwise::ConvertOp;
        let ctx = &mut Context::new();
        let first = fixture(ctx);
        let second = fixture(ctx);
        let old = copy(ctx, &first, first.parameter, "old");
        let integer = shape::array(ctx, ElementType::S32, vec![2]).unwrap();
        let incompatible = ConvertOp::new(ctx, integer, vec![first.parameter]);
        incompatible
            .get_operation()
            .insert_at_back(first.block, ctx);
        name(ctx, incompatible.get_operation(), "integer");
        finish(ctx, &first, old.result(ctx));
        finish(ctx, &second, second.parameter);
        let before = first.module.disp(ctx).to_string();
        for replacement in [incompatible.result(ctx), second.parameter] {
            assert!(replace_all_uses_impl(ctx, old.get_operation(), replacement).is_err());
            assert_eq!(first.module.disp(ctx).to_string(), before);
        }
        assert!(
            !replace_all_uses_impl(ctx, old.get_operation(), old.result(ctx))
                .unwrap()
                .changed
        );
        crate::verification::verify(ctx, first.module).unwrap();
        crate::verification::verify(ctx, second.module).unwrap();
        Operation::erase(first.module, ctx);
        Operation::erase(second.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn verified_forward_keeps_source_identities_and_roundtrips_in_fresh_context() {
        use pliron::{irfmt::parsers::spaced, parsable::parse_from_str};
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        let replacement = copy(ctx, &fixture, old.result(ctx), "replacement");
        let ret = finish(ctx, &fixture, old.result(ctx));
        old.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(vec!["p".into()]),
        );
        old.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_original_value"), provenance("source"));
        fixture.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: "main".into(),
                instructions: vec!["p".into(), "old".into(), "replacement".into()],
            }]),
        );
        let change = replace_all_uses_with(
            ctx,
            old.get_operation(),
            replacement.result(ctx),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert!(change.changed && change.removed.is_empty());
        assert_eq!(ret.deref(ctx).get_operand(0), replacement.result(ctx));
        assert_eq!(
            replacement.get_operation().deref(ctx).get_operand(0),
            old.result(ctx)
        );
        assert_eq!(
            old.get_operation().deref(ctx).get_parent_block(),
            Some(fixture.block)
        );
        crate::verification::verify(ctx, fixture.module).unwrap();
        let text = fixture.module.disp(ctx).to_string();
        let fresh = &mut Context::new();
        crate::register(fresh);
        let parsed = parse_from_str(spaced(Operation::top_level_parser()), fresh, &text).unwrap();
        crate::verification::verify(fresh, parsed).unwrap();
        let snapshot = |ctx: &Context, module: Ptr<Operation>| {
            operation_tree(ctx, module)
                .into_iter()
                .map(|operation| {
                    let kind = if Operation::is_op::<ModuleOp>(operation, ctx) {
                        "module"
                    } else if Operation::is_op::<ComputationOp>(operation, ctx) {
                        "computation"
                    } else if Operation::is_op::<ParameterOp>(operation, ctx) {
                        "parameter"
                    } else if Operation::is_op::<CopyOp>(operation, ctx) {
                        "copy"
                    } else {
                        assert!(Operation::is_op::<ReturnOp>(operation, ctx));
                        "return"
                    };
                    let ir = operation.deref(ctx);
                    let mut attributes: Vec<_> = ir
                        .attributes
                        .0
                        .iter()
                        .filter(|(key, _)| key.to_string().starts_with("hlo_"))
                        .map(|(key, value)| (key.to_string(), value.disp(ctx).to_string()))
                        .collect();
                    attributes.sort();
                    let types: Vec<_> = ir
                        .results()
                        .map(|value| value.get_type(ctx).disp(ctx).to_string())
                        .collect();
                    let operands: Vec<_> = ir
                        .operands()
                        .map(|value| {
                            value
                                .defining_op()
                                .unwrap()
                                .deref(ctx)
                                .attributes
                                .get::<InstructionName>(&Identifier::new("hlo_name"))
                                .unwrap()
                                .0
                                .clone()
                        })
                        .collect();
                    (kind, attributes, types, operands)
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(snapshot(ctx, fixture.module), snapshot(fresh, parsed));
        Operation::erase(parsed, fresh);
        assert!(fresh.is_ir_empty());
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn verified_invalid_output_rolls_back_and_valid_retry_works() {
        use crate::ops::elementwise::ConvertOp;
        let ctx = &mut Context::new();
        let fixture = fixture(ctx);
        let old = copy(ctx, &fixture, fixture.parameter, "old");
        let half = shape::array(ctx, ElementType::BF16, vec![2]).unwrap();
        let incompatible_root = ConvertOp::new(ctx, half, vec![fixture.parameter]);
        incompatible_root
            .get_operation()
            .insert_at_back(fixture.block, ctx);
        name(ctx, incompatible_root.get_operation(), "half");
        let ret = finish(ctx, &fixture, old.result(ctx));
        let before = fixture.module.disp(ctx).to_string();
        assert!(
            replace_all_uses_with(
                ctx,
                old.get_operation(),
                incompatible_root.result(ctx),
                Some(VerifyOptions::default())
            )
            .is_err()
        );
        assert_eq!(fixture.module.disp(ctx).to_string(), before);
        assert_eq!(ret.deref(ctx).get_operand(0), old.result(ctx));
        assert!(
            replace_all_uses_with(
                ctx,
                old.get_operation(),
                fixture.parameter,
                Some(VerifyOptions::default())
            )
            .unwrap()
            .changed
        );
        assert_eq!(ret.deref(ctx).get_operand(0), fixture.parameter);
        crate::verification::verify(ctx, fixture.module).unwrap();
        Operation::erase(fixture.module, ctx);
        assert!(ctx.is_ir_empty());
    }

    #[test]
    fn deduplicates_noncustom_fusion_operands_and_repairs_aliases() {
        for (kind, verified) in [
            (FusionKind::Loop, false),
            (FusionKind::Loop, true),
            (FusionKind::Custom, false),
            (FusionKind::Custom, true),
        ] {
            let ctx = &mut Context::new();
            let fixture = fixture(ctx);
            let old = copy(ctx, &fixture, fixture.parameter, "old");
            let fusion = FusionOp::new(
                ctx,
                fixture.ty,
                vec![fixture.parameter, old.result(ctx)],
                kind,
            );
            fusion.get_operation().insert_at_back(fixture.block, ctx);
            name(ctx, fusion.get_operation(), "fusion");
            let body = BasicBlock::new(ctx, None, vec![fixture.ty; 2]);
            body.insert_at_back(fusion.body(ctx), ctx);
            let arguments = body.deref(ctx).arguments().collect();
            let add = AddOp::new(ctx, fixture.ty, arguments);
            add.get_operation().insert_at_back(body, ctx);
            ReturnOp::new(ctx, add.result(ctx))
                .get_operation()
                .insert_at_back(body, ctx);
            fusion.get_operation().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_aliases"),
                OutputOperandAliases(vec![OutputOperandAliasing {
                    output_shape_index: Axes(vec![]),
                    operand_index: 1,
                    operand_shape_index: Axes(vec![]),
                }]),
            );
            finish(ctx, &fixture, fusion.result(ctx));
            replace_all_uses_with(
                ctx,
                old.get_operation(),
                fixture.parameter,
                verified.then(VerifyOptions::default),
            )
            .unwrap();
            let expected = if kind == FusionKind::Custom { 2 } else { 1 };
            assert_eq!(
                fusion.get_operation().deref(ctx).get_num_operands(),
                expected,
                "kind={kind:?}, verified={verified}"
            );
            assert_eq!(body.deref(ctx).get_num_arguments(), expected);
            if kind != FusionKind::Custom {
                assert!(
                    add.get_operation()
                        .deref(ctx)
                        .operands()
                        .all(|v| v == body.deref(ctx).get_argument(0))
                );
            }
            assert_eq!(
                fusion
                    .get_operation()
                    .deref(ctx)
                    .attributes
                    .get::<OutputOperandAliases>(&Identifier::new("hlo_aliases"))
                    .unwrap()
                    .0[0]
                    .operand_index,
                if kind == FusionKind::Custom { 1 } else { 0 },
                "kind={kind:?}, verified={verified}"
            );
            assert_eq!(
                old.get_operation().deref(ctx).get_parent_block(),
                Some(fixture.block)
            );
            crate::verification::verify(ctx, fixture.module).unwrap();
            Operation::erase(fixture.module, ctx);
            assert!(ctx.is_ir_empty());
        }
    }
}
