//! Recursive dead-operand cleanup and fusion operand repair.

use super::*;

pub(super) struct Cleanup<'a> {
    pub(super) module: Ptr<Operation>,
    pub(super) effects: &'a SideEffects,
    pub(super) removed: HashSet<Ptr<Operation>>,
    pub(super) queue: VecDeque<Value>,
    pub(super) initial: Option<Ptr<Operation>>,
    pub(super) ignore_controls: bool,
}

impl<'a> Cleanup<'a> {
    pub(super) fn new(module: Ptr<Operation>, effects: &'a SideEffects) -> Self {
        Self {
            module,
            effects,
            removed: HashSet::new(),
            queue: VecDeque::new(),
            initial: None,
            ignore_controls: false,
        }
    }

    pub(super) fn record_erased_tree(&mut self, ctx: &Context, operation: Ptr<Operation>) {
        let mut pending = vec![operation];
        while let Some(operation) = pending.pop() {
            if !self.removed.insert(operation) {
                continue;
            }
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    pending.extend(block.deref(ctx).iter(ctx));
                }
            }
        }
    }

    pub(super) fn drain(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut IRRewriter<DummyListener>,
    ) -> Result<()> {
        loop {
            let mut arguments: HashMap<Ptr<BasicBlock>, BTreeSet<usize>> = HashMap::new();
            while let Some(value) = self.queue.pop_front() {
                if let Some(operation) = value.defining_op() {
                    if self.removed.contains(&operation) {
                        continue;
                    }
                    if value.is_used(ctx) || opcode(ctx, operation)? == Opcode::Parameter {
                        continue;
                    }
                    if Some(operation) != self.initial && self.effects.has_side_effect(operation)? {
                        continue;
                    }
                    if Some(operation) == self.initial
                        && opcode(ctx, operation)? == Opcode::AfterAll
                        && after_all_depends_on_effect(ctx, operation, self.effects)?
                    {
                        continue;
                    }
                    let block = endpoint_block(ctx, value)?;
                    let mut graph = ControlGraph::read(ctx, block)?;
                    let index = graph.index(value)?;
                    if !self.ignore_controls && graph.has_controls(index) {
                        continue;
                    }
                    let order_owner = order_owner(ctx, block)?;
                    if self.ignore_controls {
                        graph.splice(index);
                        graph.ensure_names(ctx, self.module)?;
                        invalidate_order(ctx, order_owner);
                        graph.write(ctx)?;
                    }
                    invalidate_order(ctx, order_owner);
                    self.queue.extend(operation.deref(ctx).operands());
                    self.record_erased_tree(ctx, operation);
                    rewriter.erase_operation(ctx, operation);
                } else if !value.is_used(ctx) {
                    let block = endpoint_block(ctx, value)?;
                    let index = argument_index(ctx, block, value)?;
                    let mut graph = ControlGraph::read(ctx, block)?;
                    let node = graph.index(value)?;
                    if !self.ignore_controls && graph.has_controls(node) {
                        continue;
                    }
                    let order_owner = order_owner(ctx, block)?;
                    if self.ignore_controls {
                        graph.splice(node);
                        graph.ensure_names(ctx, self.module)?;
                        invalidate_order(ctx, order_owner);
                        graph.write(ctx)?;
                    }
                    arguments.entry(block).or_default().insert(index);
                }
            }
            if arguments.is_empty() {
                return Ok(());
            }
            for (block, indices) in arguments {
                let fusion = block
                    .deref(ctx)
                    .get_parent_op(ctx)
                    .ok_or_else(|| error("fusion parameter has no caller"))?;
                for index in indices.into_iter().rev() {
                    let operand = fusion.deref(ctx).get_operand(index);
                    let owner = order_owner(ctx, block)?;
                    invalidate_order(ctx, owner);
                    remove_fusion_parameter(ctx, block, index)?;
                    if !fusion
                        .deref(ctx)
                        .operands()
                        .any(|remaining| remaining == operand)
                    {
                        self.queue.push_back(operand);
                    }
                }
            }
        }
    }
}

pub(super) fn after_all_depends_on_effect(
    ctx: &Context,
    instruction: Ptr<Operation>,
    effects: &SideEffects,
) -> Result<bool> {
    for operand in instruction.deref(ctx).operands() {
        if let Some(mut operation) = operand.defining_op() {
            if opcode(ctx, operation)? == Opcode::GetTupleElement
                && let Some(producer) = operation.deref(ctx).get_operand(0).defining_op()
            {
                operation = producer;
            }
            let users: HashSet<_> = operand
                .uses(ctx)
                .into_iter()
                .map(|use_| use_.user_op())
                .filter(|&user| !Operation::is_op::<ReturnOp>(user, ctx))
                .collect();
            if effects.has_side_effect(operation)? && users.len() == 1 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

pub(super) fn deduplicate_fusion_operands(
    ctx: &mut Context,
    fusion: Ptr<Operation>,
    rewriter: &mut IRRewriter<DummyListener>,
    cleanup: &mut Cleanup<'_>,
) -> Result<()> {
    let fusion_op =
        Operation::get_op::<FusionOp>(fusion, ctx).ok_or_else(|| error("expected fusion"))?;
    if fusion_op
        .get_attr_hlo_fusion_kind(ctx)
        .is_some_and(|kind| *kind == FusionKind::Custom)
    {
        return Ok(());
    }
    let block = fusion_op
        .body(ctx)
        .deref(ctx)
        .get_entry_block()
        .ok_or_else(|| error("fusion has no body"))?;
    let operands: Vec<_> = fusion.deref(ctx).operands().collect();
    let mut first = HashMap::new();
    let mut duplicates = Vec::new();
    for (index, operand) in operands.into_iter().enumerate() {
        if let Some(&kept) = first.get(&operand) {
            let argument = block.deref(ctx).get_argument(index);
            let replacement = block.deref(ctx).get_argument(kept);
            let owner = order_owner(ctx, block)?;
            invalidate_order(ctx, owner);
            rewriter.replace_value_uses_with(ctx, argument, replacement);
            // ReplaceAllUses copies provenance, not metadata or controls.
            let original = Annotations::read(ctx, argument)?.original_value;
            if let Some(original) = original {
                let mut annotations = Annotations::read(ctx, replacement)?;
                annotations.original_value = Some(original);
                annotations.write(ctx, replacement)?;
            }
            duplicates.push((index, kept));
        } else {
            first.insert(operand, index);
        }
    }
    for (index, kept) in duplicates.into_iter().rev() {
        remap_removed_operand_aliases(ctx, fusion, index, kept)?;
        remove_fusion_parameter(ctx, block, index)?;
    }
    // Keep the original cleanup snapshot; deduplication creates no instructions.
    cleanup.drain(ctx, rewriter)
}

pub(super) fn remap_removed_operand_aliases(
    ctx: &Context,
    fusion: Ptr<Operation>,
    removed: usize,
    replacement: usize,
) -> Result<()> {
    let Some(mut aliases) = optional_attr::<OutputOperandAliases>(ctx, fusion, "hlo_aliases")?
        .as_deref()
        .cloned()
    else {
        return Ok(());
    };
    for alias in &mut aliases.0 {
        let index = usize::try_from(alias.operand_index)
            .map_err(|_| error("negative fusion alias operand"))?;
        if index == removed {
            alias.operand_index = i64::try_from(replacement)
                .map_err(|_| error("fusion alias operand index overflow"))?;
        } else if index > removed {
            alias.operand_index -= 1;
        }
    }
    fusion
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_aliases"), aliases);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{controls, copy, finish, fixture, has_order, name, rich_order};
    use super::*;
    use crate::attributes::Index;
    use crate::ops::control::AfterAllOp;
    use crate::ops::control::CustomCallOp;
    use crate::ops::control::GetTupleElementOp;
    use crate::ops::control::TupleOp;
    use crate::ops::elementwise::AddOp;
    use crate::ops::movement::CopyOp;
    use crate::verification::verify_with_options;
    use pliron::builtin::attributes::StringAttr;
    use pliron::{op::Op, printable::Printable};

    #[test]
    fn recursive_cleanup_removes_pure_operands_but_keeps_effects_and_parameters() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let pure = copy(ctx, &f, f.parameters[0], "pure");
        let effect = CustomCallOp::new(ctx, f.ty, vec![], StringAttr::new("effect".into()));
        effect.get_operation().insert_at_back(f.block, ctx);
        effect.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_has_side_effect"),
            crate::attributes::Bools(vec![true]),
        );
        let tuple_type = shape::tuple(ctx, vec![f.ty, f.ty]).unwrap();
        let tuple = TupleOp::new(
            ctx,
            tuple_type,
            vec![result(ctx, pure).unwrap(), effect.result(ctx)],
        );
        tuple.get_operation().insert_at_back(f.block, ctx);
        let projection = GetTupleElementOp::new(ctx, f.ty, vec![tuple.result(ctx)], Index(0));
        projection.get_operation().insert_at_back(f.block, ctx);
        let ret = finish(ctx, &f, projection.result(ctx));
        let outcome = replace_instruction(
            ctx,
            projection.get_operation(),
            f.parameters[0],
            Default::default(),
            None,
        )
        .unwrap();
        assert!(outcome.removed.contains(&pure));
        assert!(outcome.removed.contains(&tuple.get_operation()));
        assert!(outcome.removed.contains(&projection.get_operation()));
        assert!(!outcome.removed.contains(&effect.get_operation()));
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn after_all_effect_guard_applies_to_the_initial_removal_only() {
        for recursive in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1);
            let token = shape::token(ctx);
            let effect = CustomCallOp::new(ctx, token, vec![], StringAttr::new("effect".into()));
            effect.get_operation().insert_at_back(f.block, ctx);
            effect.get_operation().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_has_side_effect"),
                crate::attributes::Bools(vec![true]),
            );
            let after = AfterAllOp::new(ctx, token, vec![effect.result(ctx)]);
            after.get_operation().insert_at_back(f.block, ctx);
            let requested = if recursive {
                let tuple_ty = shape::tuple(ctx, vec![token]).unwrap();
                let tuple = TupleOp::new(ctx, tuple_ty, vec![after.result(ctx)]);
                tuple.get_operation().insert_at_back(f.block, ctx);
                tuple.get_operation()
            } else {
                after.get_operation()
            };
            finish(ctx, &f, f.parameters[0]);
            let removed =
                remove_instruction_and_unused_operands(ctx, requested, false, None).unwrap();
            assert_eq!(removed.contains(&after.get_operation()), recursive);
            assert!(!removed.contains(&effect.get_operation()));
            crate::verification::verify(ctx, f.module).unwrap();
        }
    }

    #[test]
    fn recursive_cleanup_splices_dead_operand_controls_instead_of_dropping_order() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        controls(ctx, dead, &["p0"]);
        let user = copy(ctx, &f, f.parameters[1], "user");
        controls(ctx, user, &["dead"]);
        finish(ctx, &f, result(ctx, user).unwrap());
        let removed = remove_instruction_and_unused_operands(ctx, dead, true, None).unwrap();
        assert!(removed.contains(&dead));
        assert_eq!(
            optional_attr::<ControlPredecessors>(ctx, user, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec!["p0"]
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn cleanup_removes_unused_fusion_arguments_and_recursively_dead_caller_operands() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let outer_dead = copy(ctx, &f, f.parameters[1], "outer_dead");
        let fusion = FusionOp::new(
            ctx,
            f.ty,
            vec![f.parameters[0], result(ctx, outer_dead).unwrap()],
            FusionKind::Loop,
        );
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
        ReturnOp::new(ctx, projection.result(ctx))
            .get_operation()
            .insert_at_back(body, ctx);
        finish(ctx, &f, fusion.result(ctx));
        let outcome =
            replace_instruction(ctx, projection.get_operation(), a, Default::default(), None)
                .unwrap();
        assert!(outcome.removed.contains(&outer_dead));
        assert_eq!(body.deref(ctx).get_num_arguments(), 1);
        assert_eq!(fusion.get_operation().deref(ctx).get_num_operands(), 1);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn replacement_deduplicates_noncustom_fusion_operands() {
        for kind in [FusionKind::Loop, FusionKind::Custom] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1);
            let old = copy(ctx, &f, f.parameters[0], "old");
            let fusion = FusionOp::new(
                ctx,
                f.ty,
                vec![result(ctx, old).unwrap(), f.parameters[0]],
                kind,
            );
            fusion.get_operation().insert_at_back(f.block, ctx);
            let body = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
            body.insert_at_back(fusion.body(ctx), ctx);
            let arguments = vec![
                body.deref(ctx).get_argument(0),
                body.deref(ctx).get_argument(1),
            ];
            let add = AddOp::new(ctx, f.ty, arguments);
            add.get_operation().insert_at_back(body, ctx);
            ReturnOp::new(ctx, add.result(ctx))
                .get_operation()
                .insert_at_back(body, ctx);
            fusion.get_operation().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_aliases"),
                OutputOperandAliases(vec![crate::attributes::OutputOperandAliasing {
                    output_shape_index: crate::attributes::Axes(vec![]),
                    operand_index: 1,
                    operand_shape_index: crate::attributes::Axes(vec![]),
                }]),
            );
            finish(ctx, &f, fusion.result(ctx));
            replace_instruction(ctx, old, f.parameters[0], Default::default(), None).unwrap();
            assert_eq!(
                body.deref(ctx).get_num_arguments(),
                if kind == FusionKind::Custom { 2 } else { 1 }
            );
            assert_eq!(
                optional_attr::<OutputOperandAliases>(ctx, fusion.get_operation(), "hlo_aliases",)
                    .unwrap()
                    .unwrap()
                    .0[0]
                    .operand_index,
                if kind == FusionKind::Custom { 1 } else { 0 },
            );
            crate::verification::verify(ctx, f.module).unwrap();
        }
    }

    #[test]
    fn current_pin_after_all_protects_only_the_sole_effectful_operand_user() {
        for extra_user in [false, true] {
            for tuple_projection in [false, true] {
                let ctx = &mut Context::new();
                let f = fixture(ctx, 1);
                let token = shape::token(ctx);
                let effect_type = if tuple_projection {
                    shape::tuple(ctx, vec![token]).unwrap()
                } else {
                    token
                };
                let effect =
                    CustomCallOp::new(ctx, effect_type, vec![], StringAttr::new("effect".into()));
                effect.get_operation().insert_at_back(f.block, ctx);
                name(ctx, effect.get_operation(), "effect");
                effect.get_operation().deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_has_side_effect"),
                    crate::attributes::Bools(vec![true]),
                );
                let operand = if tuple_projection {
                    let gte =
                        GetTupleElementOp::new(ctx, token, vec![effect.result(ctx)], Index(0));
                    gte.get_operation().insert_at_back(f.block, ctx);
                    name(ctx, gte.get_operation(), "token");
                    gte.result(ctx)
                } else {
                    effect.result(ctx)
                };
                let after = AfterAllOp::new(ctx, token, vec![operand, operand]);
                after.get_operation().insert_at_back(f.block, ctx);
                name(ctx, after.get_operation(), "after");
                if extra_user {
                    let anchor = CustomCallOp::new(
                        ctx,
                        f.ty,
                        vec![operand],
                        StringAttr::new("anchor".into()),
                    );
                    anchor.get_operation().insert_at_back(f.block, ctx);
                    name(ctx, anchor.get_operation(), "anchor");
                    anchor.get_operation().deref_mut(ctx).attributes.set(
                        Identifier::new("hlo_has_side_effect"),
                        crate::attributes::Bools(vec![true]),
                    );
                }
                finish(ctx, &f, f.parameters[0]);
                let owner = rich_order(ctx, f.block);
                let before = f.module.disp(ctx).to_string();
                let removed =
                    remove_instruction_and_unused_operands(ctx, after.get_operation(), false, None)
                        .unwrap();
                assert_eq!(removed.contains(&after.get_operation()), extra_user);
                assert_eq!(has_order(ctx, owner), !extra_user);
                if !extra_user {
                    assert_eq!(f.module.disp(ctx).to_string(), before);
                }
                verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
            }
        }
    }

    #[test]
    fn disabling_operand_cleanup_retains_dead_producers_and_given_names() {
        use pliron::common_traits::Named;
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let producer = CopyOp::new(ctx, f.ty, vec![f.parameters[0]]);
        producer.get_operation().insert_at_back(f.block, ctx);
        let old = CopyOp::new(ctx, f.ty, vec![producer.result(ctx)]);
        old.get_operation().insert_at_back(f.block, ctx);
        old.result(ctx)
            .set_name(ctx, Some(Identifier::new("old_ssa")));
        let before_name = f.parameters[1].given_name(ctx);
        let ret = finish(ctx, &f, old.result(ctx));
        let edit = replace_instruction(
            ctx,
            old.get_operation(),
            f.parameters[1],
            ReplaceInstructionOptions {
                remove_unused_operands: false,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(edit.removed, HashSet::from([old.get_operation()]));
        assert_eq!(
            producer.get_operation().deref(ctx).get_parent_block(),
            Some(f.block)
        );
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[1]);
        assert_eq!(f.parameters[1].given_name(ctx), before_name);
        crate::verification::verify(ctx, f.module).unwrap();
    }
}
