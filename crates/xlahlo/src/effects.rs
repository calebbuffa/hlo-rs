//! Channel-sensitive local and transitive HLO effects on a module snapshot.
use std::collections::{HashMap, VecDeque};

use pliron::{
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    result::Result,
};

use crate::{
    attributes::Callees,
    ops::{ComputationOp, HloOpInterface, ReturnOp},
    verification::{
        control::{async_wrapped_computations, has_local_side_effect_with_context},
        error, optional_attr,
    },
};

/// Query effects of a single native HLO instruction without visiting callees.
///
/// Pass the containing module configuration's channel-ID sensitivity to match
/// `HasSideEffectNoRecurse`. Layout-constrained collectives remain effectful
/// independently of that policy. This is not an IR verifier; malformed consumed
/// metadata and non-HLO operations are errors.
pub fn has_local_side_effect(
    ctx: &Context,
    operation: Ptr<Operation>,
    channel_id_sensitive: bool,
) -> Result<bool> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    if op_cast::<dyn HloOpInterface>(&*dynamic).is_none() {
        return Err(error("local side-effect query requires an HLO instruction"));
    }
    has_local_side_effect_with_context(ctx, operation, channel_id_sensitive)
}

/// Native analysis of effects, including inline regions and named callees.
///
/// Compute on the containing module. Channel IDs are semantically significant
/// unless its configuration enables SPMD partitioning, matching HLO's
/// `ChannelIdSensitive` policy. Cycles use the least fixed point
/// of local effects; this is not a termination analysis or an IR verifier.
/// Unresolved callees, unknown operations, and queries outside the snapshot are
/// errors. Invalidate after structural or semantic mutation.
pub struct SideEffects {
    effects: HashMap<Ptr<Operation>, bool>,
}

impl SideEffects {
    /// Whether an operation, its inline regions, or its callees have effects.
    pub fn has_side_effect(&self, operation: Ptr<Operation>) -> Result<bool> {
        self.effects
            .get(&operation)
            .copied()
            .ok_or_else(|| error("side-effect query is outside the analyzed module"))
    }
}

impl Analysis for SideEffects {
    fn name(&self) -> &str {
        "hlo-side-effects"
    }

    fn compute(module: Ptr<Operation>, ctx: &Context, _: &mut AnalysisManager) -> Result<Self> {
        if !Operation::is_op::<ModuleOp>(module, ctx) {
            return Err(error("side-effect analysis requires a module"));
        }
        let channel_id_sensitive = crate::ir::config(ctx, module)?.channel_id_sensitive();
        let mut effects = HashMap::new();
        let mut dependents: HashMap<_, Vec<_>> = HashMap::new();
        let mut computations = HashMap::new();
        let mut calls = Vec::new();
        let mut pending = vec![(module, None)];
        let mut all = Vec::new();
        let mut work = VecDeque::new();
        while let Some((operation, mut owner)) = pending.pop() {
            let dynamic = Operation::get_op_dyn(operation, ctx);
            let opcode = op_cast::<dyn HloOpInterface>(&*dynamic).map(|hlo| hlo.opcode());
            let hlo = opcode.is_some();
            if !hlo
                && !Operation::is_op::<ModuleOp>(operation, ctx)
                && !Operation::is_op::<ComputationOp>(operation, ctx)
                && !Operation::is_op::<ReturnOp>(operation, ctx)
            {
                return Err(error(
                    "side-effect analysis encountered a non-HLO operation",
                ));
            }
            let local =
                hlo && has_local_side_effect_with_context(ctx, operation, channel_id_sensitive)?;
            effects.insert(operation, local);
            if local {
                work.push_back(operation);
            }
            if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx) {
                owner = Some(operation);
                let name = computation
                    .get_attr_hlo_computation_name(ctx)
                    .ok_or_else(|| error("computation has no name"))?;
                if computations
                    .insert(name.as_ref().clone(), operation)
                    .is_some()
                {
                    return Err(error("duplicate computation in side-effect analysis"));
                }
            }
            if let Some(callees) = optional_attr::<Callees>(ctx, operation, "hlo_callees")? {
                calls.push((operation, callees.0.clone()));
            }
            all.push((operation, owner));
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    for child in block.deref(ctx).iter(ctx) {
                        dependents.entry(child).or_default().push(operation);
                        pending.push((child, owner));
                    }
                }
            }
        }
        for (call, names) in calls {
            for name in names {
                let computation = computations
                    .get(&name)
                    .ok_or_else(|| error("callee not found in side-effect analysis"))?;
                dependents.entry(*computation).or_default().push(call);
            }
            for (operation, computation) in async_wrapped_computations(ctx, &all, &computations)? {
                dependents.entry(computation).or_default().push(operation);
            }
        }
        if dependents
            .keys()
            .any(|operation| !effects.contains_key(operation))
        {
            return Err(error("effect dependency is outside the analyzed module"));
        }
        while let Some(operation) = work.pop_front() {
            for dependent in dependents.get(&operation).into_iter().flatten() {
                let effect = effects
                    .get_mut(dependent)
                    .ok_or_else(|| error("effect propagation left the module"))?;
                if !*effect {
                    *effect = true;
                    work.push_back(*dependent);
                }
            }
        }
        Ok(Self { effects })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Bools, FusionKind},
        opcode::Opcode,
        ops::control::{CallOp, CustomCallOp, FusionOp},
        shape::Shape,
        types::ElementType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{attributes::StringAttr, types::FunctionType},
        identifier::Identifier,
        op::Op,
        r#type::TypeHandle,
    };

    #[test]
    fn native_pass_manager_invalidates_effects_after_mutation() {
        use pliron::{
            irbuild::IRStatus,
            pass::{Pass, PassResult, Passes},
        };

        struct EnableEffect(Ptr<Operation>);
        impl Pass for EnableEffect {
            fn name(&self) -> &str {
                "enable-test-effect"
            }

            fn run(
                &mut self,
                module: Ptr<Operation>,
                ctx: &mut Context,
                analyses: &mut AnalysisManager,
            ) -> Result<PassResult> {
                assert!(
                    !analyses
                        .get_analysis::<SideEffects>(module, ctx)?
                        .has_side_effect(self.0)?
                );
                self.0
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
                let mut result = PassResult::default();
                result.ir_changed = IRStatus::Changed;
                Ok(result)
            }
        }

        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = Shape::array(ctx, ElementType::F32, vec![0]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("effects")).get_operation();
        let (_, block) = computation(ctx, module, "main", ty);
        let custom = CustomCallOp::new(ctx, ty, vec![], StringAttr::new("effect_probe".into()));
        custom.get_operation().insert_at_back(block, ctx);
        let mut analyses = AnalysisManager::default();
        let mut passes = Passes::default();
        passes.add_pass(EnableEffect(custom.get_operation()));
        passes.run(module, ctx, &mut analyses).unwrap();
        assert!(analyses.try_get_analysis::<SideEffects>(module).is_none());
        assert!(
            analyses
                .get_analysis::<SideEffects>(module, ctx)
                .unwrap()
                .has_side_effect(custom.get_operation())
                .unwrap()
        );
    }

    fn computation(
        ctx: &mut Context,
        module: Ptr<Operation>,
        name: &'static str,
        ty: TypeHandle,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>) {
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let signature = FunctionType::get(ctx, vec![], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new(name), signature);
        computation.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        (computation.get_operation(), block)
    }

    #[test]
    fn collective_effects_follow_module_spmd_policy_and_layout_constraints() {
        use crate::{
            attributes::{Axes, OptionalInteger, ReplicaGroups, SourceTargetPairs},
            ir::ModuleConfig,
            ops::control::{
                AllGatherOp, AllReduceOp, AllToAllOp, CollectivePermuteOp, CollectiveReduceOp,
                ParameterOp, ReduceScatterOp,
            },
        };

        for code in [
            Opcode::AllGather,
            Opcode::AllReduce,
            Opcode::AllToAll,
            Opcode::ReduceScatter,
            Opcode::CollectiveReduce,
            Opcode::CollectivePermute,
        ] {
            // Exercise the local policy independently of collective geometry.
            for spmd in [false, true] {
                for channel in [false, true] {
                    for constrain_layout in [false, true] {
                        let ctx = &mut Context::new();
                        crate::register(ctx);
                        let ty = Shape::array(ctx, ElementType::F32, vec![1]).unwrap();
                        let module =
                            ModuleOp::new(ctx, Identifier::new("spmd_effects")).get_operation();
                        crate::ir::set_config(
                            ctx,
                            module,
                            ModuleConfig {
                                use_spmd_partitioning: spmd,
                                ..ModuleConfig::default()
                            },
                        )
                        .unwrap();
                        let (worker, block) = computation(ctx, module, "worker", ty);
                        let parameter =
                            ParameterOp::new(ctx, ty, vec![], crate::attributes::Index(0));
                        parameter.get_operation().insert_at_back(block, ctx);
                        let input = parameter.result(ctx);
                        let operation = match code {
                            Opcode::AllGather => AllGatherOp::new(
                                ctx,
                                ty,
                                vec![input],
                                Axes(vec![0]),
                                ReplicaGroups::default(),
                            )
                            .get_operation(),
                            Opcode::AllReduce => AllReduceOp::new(
                                ctx,
                                ty,
                                vec![input],
                                Callees(vec![]),
                                ReplicaGroups::default(),
                            )
                            .get_operation(),
                            Opcode::AllToAll => AllToAllOp::new(
                                ctx,
                                ty,
                                vec![input],
                                Axes(vec![0]),
                                ReplicaGroups::default(),
                            )
                            .get_operation(),
                            Opcode::ReduceScatter => ReduceScatterOp::new(
                                ctx,
                                ty,
                                vec![input],
                                Axes(vec![0]),
                                Callees(vec![]),
                                ReplicaGroups::default(),
                            )
                            .get_operation(),
                            Opcode::CollectiveReduce => CollectiveReduceOp::new(
                                ctx,
                                ty,
                                vec![input],
                                Callees(vec![]),
                                ReplicaGroups::default(),
                            )
                            .get_operation(),
                            Opcode::CollectivePermute => CollectivePermuteOp::new(
                                ctx,
                                ty,
                                vec![input],
                                SourceTargetPairs::default(),
                            )
                            .get_operation(),
                            _ => unreachable!(),
                        };
                        operation.insert_at_back(block, ctx);
                        if channel {
                            operation
                                .deref_mut(ctx)
                                .attributes
                                .set(Identifier::new("hlo_channel"), OptionalInteger::Value(1));
                        }
                        operation.deref_mut(ctx).attributes.set(
                            Identifier::new("hlo_constrain_layout"),
                            Bools(vec![constrain_layout]),
                        );
                        let output = operation.deref(ctx).get_result(0);
                        ReturnOp::new(ctx, output)
                            .get_operation()
                            .insert_at_back(block, ctx);
                        let (main, block) = computation(ctx, module, "main", ty);
                        let call =
                            CallOp::new(ctx, ty, vec![], Callees(vec![Identifier::new("worker")]));
                        call.get_operation().insert_at_back(block, ctx);
                        ReturnOp::new(ctx, call.result(ctx))
                            .get_operation()
                            .insert_at_back(block, ctx);
                        let analysis =
                            SideEffects::compute(module, ctx, &mut AnalysisManager::default())
                                .unwrap();
                        let expected = (channel && !spmd)
                            || (constrain_layout && code != Opcode::CollectivePermute);
                        for instruction in [operation, worker, call.get_operation(), main, module] {
                            assert_eq!(
                                analysis.has_side_effect(instruction).unwrap(),
                                expected,
                                "{code}, spmd={spmd}, channel={channel}, layout={constrain_layout}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_module_configuration_is_an_effect_analysis_error() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let module = ModuleOp::new(ctx, Identifier::new("invalid_config")).get_operation();
        module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_module_config"),
            StringAttr::new("not a configuration".into()),
        );
        assert!(SideEffects::compute(module, ctx, &mut AnalysisManager::default()).is_err());
    }

    #[test]
    fn verified_collective_module_recomputes_effects_after_spmd_policy_change() {
        use crate::{
            attributes::{Index, OptionalInteger, SourceTarget, SourceTargetPairs},
            ir::ModuleConfig,
            ops::control::{CollectivePermuteOp, ParameterOp},
        };
        use pliron::{
            builtin::attributes::TypeAttr,
            irbuild::IRStatus,
            pass::{Pass, PassResult, Passes},
        };

        struct EnableSpmd(Ptr<Operation>);
        impl Pass for EnableSpmd {
            fn name(&self) -> &str {
                "enable-spmd-test"
            }

            fn run(
                &mut self,
                module: Ptr<Operation>,
                ctx: &mut Context,
                analyses: &mut AnalysisManager,
            ) -> Result<PassResult> {
                assert!(
                    analyses
                        .get_analysis::<SideEffects>(module, ctx)?
                        .has_side_effect(self.0)?
                );
                let mut config = crate::ir::config(ctx, module)?;
                config.use_spmd_partitioning = true;
                crate::ir::set_config(ctx, module, config)?;
                let mut result = PassResult::default();
                result.ir_changed = IRStatus::Changed;
                Ok(result)
            }
        }

        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = Shape::array(ctx, ElementType::F32, vec![1]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("verified_spmd")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        // Counts deliberately do not determine channel sensitivity upstream.
        crate::ir::set_config(
            ctx,
            module,
            ModuleConfig {
                replica_count: 1,
                num_partitions: 1,
                ..ModuleConfig::default()
            },
        )
        .unwrap();
        let (main, block) = computation(ctx, module, "main", ty);
        let signature = FunctionType::get(ctx, vec![ty], vec![ty]).into();
        Operation::get_op::<ComputationOp>(main, ctx)
            .unwrap()
            .set_attr_hlo_computation_signature(ctx, TypeAttr::new(signature));
        let parameter = ParameterOp::new(ctx, ty, vec![], Index(0));
        parameter.get_operation().insert_at_back(block, ctx);
        let collective = CollectivePermuteOp::new(
            ctx,
            ty,
            vec![parameter.result(ctx)],
            SourceTargetPairs(vec![SourceTarget {
                source: 0,
                target: 0,
            }]),
        );
        collective.get_operation().insert_at_back(block, ctx);
        collective
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_channel"), OptionalInteger::Value(1));
        ReturnOp::new(ctx, collective.result(ctx))
            .get_operation()
            .insert_at_back(block, ctx);
        crate::verification::verify(ctx, module).unwrap();
        let mut analyses = AnalysisManager::default();
        let mut passes = Passes::default();
        passes.add_pass(EnableSpmd(collective.get_operation()));
        passes.run(module, ctx, &mut analyses).unwrap();
        assert!(analyses.try_get_analysis::<SideEffects>(module).is_none());
        crate::verification::verify(ctx, module).unwrap();
        let effects = analyses.get_analysis::<SideEffects>(module, ctx).unwrap();
        for instruction in [collective.get_operation(), main, module] {
            assert!(!effects.has_side_effect(instruction).unwrap());
        }
    }

    #[test]
    fn local_and_transitive_effects_include_inline_fusion_and_all_element_types() {
        for &element in ElementType::ALL {
            for effect in [false, true] {
                let ctx = &mut Context::new();
                crate::register(ctx);
                let ty = Shape::array(ctx, element, vec![0]).unwrap();
                let module = ModuleOp::new(ctx, Identifier::new("effects")).get_operation();
                crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
                let (leaf, leaf_block) = computation(ctx, module, "leaf", ty);
                let fusion = FusionOp::new(ctx, ty, vec![], FusionKind::Loop);
                fusion.get_operation().insert_at_back(leaf_block, ctx);
                let body = BasicBlock::new(ctx, None, vec![]);
                body.insert_at_back(fusion.body(ctx), ctx);
                let custom =
                    CustomCallOp::new(ctx, ty, vec![], StringAttr::new("effect_probe".into()));
                custom
                    .get_operation()
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_has_side_effect"), Bools(vec![effect]));
                custom.get_operation().insert_at_back(body, ctx);
                ReturnOp::new(ctx, custom.result(ctx))
                    .get_operation()
                    .insert_at_back(body, ctx);
                ReturnOp::new(ctx, fusion.result(ctx))
                    .get_operation()
                    .insert_at_back(leaf_block, ctx);
                let (middle, middle_block) = computation(ctx, module, "middle", ty);
                let first = CallOp::new(ctx, ty, vec![], Callees(vec![Identifier::new("leaf")]));
                first.get_operation().insert_at_back(middle_block, ctx);
                ReturnOp::new(ctx, first.result(ctx))
                    .get_operation()
                    .insert_at_back(middle_block, ctx);
                let (main, main_block) = computation(ctx, module, "main", ty);
                let second = CallOp::new(ctx, ty, vec![], Callees(vec![Identifier::new("middle")]));
                second.get_operation().insert_at_back(main_block, ctx);
                ReturnOp::new(ctx, second.result(ctx))
                    .get_operation()
                    .insert_at_back(main_block, ctx);
                crate::verification::verify(ctx, module).unwrap();
                let mut analyses = AnalysisManager::default();
                let analysis = analyses.get_analysis::<SideEffects>(module, ctx).unwrap();
                for operation in [
                    custom.get_operation(),
                    fusion.get_operation(),
                    leaf,
                    first.get_operation(),
                    middle,
                    second.get_operation(),
                    main,
                    module,
                ] {
                    assert_eq!(analysis.has_side_effect(operation).unwrap(), effect);
                }
                assert!(
                    !analysis
                        .has_side_effect(main_block.deref(ctx).iter(ctx).last().unwrap())
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn unresolved_callees_and_out_of_snapshot_queries_are_errors() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = Shape::array(ctx, ElementType::F32, vec![0]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("effects")).get_operation();
        let (_, block) = computation(ctx, module, "main", ty);
        let call = CallOp::new(ctx, ty, vec![], Callees(vec![Identifier::new("missing")]));
        call.get_operation().insert_at_back(block, ctx);
        assert!(SideEffects::compute(module, ctx, &mut AnalysisManager::default()).is_err());
        call.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_callees"), Callees(vec![]));
        let analysis = SideEffects::compute(module, ctx, &mut AnalysisManager::default()).unwrap();
        let outside = CustomCallOp::new(ctx, ty, vec![], StringAttr::new("outside".into()));
        assert!(analysis.has_side_effect(outside.get_operation()).is_err());
        assert!(
            SideEffects::compute(call.get_operation(), ctx, &mut AnalysisManager::default())
                .is_err()
        );
    }

    #[test]
    fn recursive_call_graph_effects_reach_a_fixed_point() {
        for effect in [false, true] {
            let ctx = &mut Context::new();
            crate::register(ctx);
            let ty = Shape::array(ctx, ElementType::F32, vec![0]).unwrap();
            let module = ModuleOp::new(ctx, Identifier::new("recursive")).get_operation();
            let (a, a_block) = computation(ctx, module, "a", ty);
            let (b, b_block) = computation(ctx, module, "b", ty);
            let to_b = CallOp::new(ctx, ty, vec![], Callees(vec![Identifier::new("b")]));
            to_b.get_operation().insert_at_back(a_block, ctx);
            let to_a = CallOp::new(ctx, ty, vec![], Callees(vec![Identifier::new("a")]));
            to_a.get_operation().insert_at_back(b_block, ctx);
            let custom = CustomCallOp::new(ctx, ty, vec![], StringAttr::new("effect_probe".into()));
            custom
                .get_operation()
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![effect]));
            custom.get_operation().insert_at_back(a_block, ctx);
            let analysis =
                SideEffects::compute(module, ctx, &mut AnalysisManager::default()).unwrap();
            for operation in [a, b, to_a.get_operation(), to_b.get_operation()] {
                assert_eq!(analysis.has_side_effect(operation).unwrap(), effect);
            }
        }
    }

    #[test]
    fn asynchronous_updates_and_done_inherit_wrapped_effects() {
        use crate::ops::control::{AsyncDoneOp, AsyncStartOp, AsyncUpdateOp};
        for effect in [false, true] {
            let ctx = &mut Context::new();
            crate::register(ctx);
            let ty = Shape::array(ctx, ElementType::F32, vec![0]).unwrap();
            let module = ModuleOp::new(ctx, Identifier::new("async_effects")).get_operation();
            crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
            let (_, work) = computation(ctx, module, "work", ty);
            let custom = CustomCallOp::new(ctx, ty, vec![], StringAttr::new("effect_probe".into()));
            custom
                .get_operation()
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![effect]));
            custom.get_operation().insert_at_back(work, ctx);
            ReturnOp::new(ctx, custom.result(ctx))
                .get_operation()
                .insert_at_back(work, ctx);
            let (_, main) = computation(ctx, module, "main", ty);
            let inputs = Shape::tuple(ctx, vec![]).unwrap();
            let state = Shape::array(ctx, ElementType::S32, vec![]).unwrap();
            let async_ty = Shape::tuple(ctx, vec![inputs, ty, state]).unwrap();
            let start = AsyncStartOp::new(
                ctx,
                async_ty,
                vec![],
                Callees(vec![Identifier::new("work")]),
            );
            start.get_operation().insert_at_back(main, ctx);
            let update = AsyncUpdateOp::new(ctx, async_ty, vec![start.result(ctx)]);
            update.get_operation().insert_at_back(main, ctx);
            let done = AsyncDoneOp::new(ctx, ty, vec![update.result(ctx)]);
            done.get_operation().insert_at_back(main, ctx);
            ReturnOp::new(ctx, done.result(ctx))
                .get_operation()
                .insert_at_back(main, ctx);
            crate::verification::verify(ctx, module).unwrap();
            let analysis =
                SideEffects::compute(module, ctx, &mut AnalysisManager::default()).unwrap();
            for operation in [
                start.get_operation(),
                update.get_operation(),
                done.get_operation(),
            ] {
                assert_eq!(analysis.has_side_effect(operation).unwrap(), effect);
            }
        }
    }

    #[test]
    fn asynchronous_effects_follow_tuple_forwarding_without_tainting_other_elements() {
        use crate::{
            attributes::Index,
            ops::control::{AsyncDoneOp, AsyncStartOp, AsyncUpdateOp, GetTupleElementOp, TupleOp},
        };
        for effect in [false, true] {
            let ctx = &mut Context::new();
            crate::register(ctx);
            let ty = Shape::array(ctx, ElementType::F32, vec![0]).unwrap();
            let module = ModuleOp::new(ctx, Identifier::new("forwarded_effects")).get_operation();
            crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
            let (_, work) = computation(ctx, module, "work", ty);
            let custom = CustomCallOp::new(ctx, ty, vec![], StringAttr::new("effect_probe".into()));
            custom
                .get_operation()
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![effect]));
            custom.get_operation().insert_at_back(work, ctx);
            ReturnOp::new(ctx, custom.result(ctx))
                .get_operation()
                .insert_at_back(work, ctx);
            let (_, main) = computation(ctx, module, "main", ty);
            let empty = Shape::tuple(ctx, vec![]).unwrap();
            let state = Shape::array(ctx, ElementType::S32, vec![]).unwrap();
            let async_ty = Shape::tuple(ctx, vec![empty, ty, state]).unwrap();
            let start = AsyncStartOp::new(
                ctx,
                async_ty,
                vec![],
                Callees(vec![Identifier::new("work")]),
            );
            start.get_operation().insert_at_back(main, ctx);
            let unrelated =
                CustomCallOp::new(ctx, ty, vec![], StringAttr::new("unrelated_effect".into()));
            unrelated
                .get_operation()
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            unrelated.get_operation().insert_at_back(main, ctx);
            let packed_ty = Shape::tuple(ctx, vec![ty, async_ty]).unwrap();
            let packed = TupleOp::new(
                ctx,
                packed_ty,
                vec![unrelated.result(ctx), start.result(ctx)],
            );
            packed.get_operation().insert_at_back(main, ctx);
            let forwarded =
                GetTupleElementOp::new(ctx, async_ty, vec![packed.result(ctx)], Index(1));
            forwarded.get_operation().insert_at_back(main, ctx);
            let update = AsyncUpdateOp::new(ctx, async_ty, vec![forwarded.result(ctx)]);
            update.get_operation().insert_at_back(main, ctx);
            let packed = TupleOp::new(
                ctx,
                packed_ty,
                vec![unrelated.result(ctx), update.result(ctx)],
            );
            packed.get_operation().insert_at_back(main, ctx);
            let forwarded =
                GetTupleElementOp::new(ctx, async_ty, vec![packed.result(ctx)], Index(1));
            forwarded.get_operation().insert_at_back(main, ctx);
            let done = AsyncDoneOp::new(ctx, ty, vec![forwarded.result(ctx)]);
            done.get_operation().insert_at_back(main, ctx);
            ReturnOp::new(ctx, done.result(ctx))
                .get_operation()
                .insert_at_back(main, ctx);
            crate::verification::verify(ctx, module).unwrap();
            let analysis =
                SideEffects::compute(module, ctx, &mut AnalysisManager::default()).unwrap();
            for operation in [update.get_operation(), done.get_operation()] {
                assert_eq!(analysis.has_side_effect(operation).unwrap(), effect);
            }
        }
    }

    #[test]
    fn malformed_effect_flags_and_unknown_operations_are_errors() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = Shape::array(ctx, ElementType::F32, vec![0]).unwrap();
        let module = ModuleOp::new(ctx, Identifier::new("effects")).get_operation();
        let (_, block) = computation(ctx, module, "main", ty);
        let custom = CustomCallOp::new(ctx, ty, vec![], StringAttr::new("effect_probe".into()));
        custom
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_has_side_effect"), Bools(vec![]));
        custom.get_operation().insert_at_back(block, ctx);
        assert!(SideEffects::compute(module, ctx, &mut AnalysisManager::default()).is_err());
        custom
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_has_side_effect"), Bools(vec![false]));
        pliron::builtin::ops::FuncOp::new(
            ctx,
            Identifier::new("unknown"),
            FunctionType::get(ctx, vec![], vec![]),
        )
        .get_operation()
        .insert_at_back(block, ctx);
        assert!(SideEffects::compute(module, ctx, &mut AnalysisManager::default()).is_err());
    }
}
