//! Fusion, async and metadata propagation across an inlined call boundary.

use super::*;

pub(super) fn propagate_fusion_attributes(
    ctx: &Context,
    call: Ptr<Operation>,
    sources: &[Ptr<Operation>],
    effects: &SideEffects,
) -> Result<()> {
    let Some(frontend) = optional_attr::<FrontendAttributes>(ctx, call, "hlo_frontend_attributes")?
        .as_deref()
        .cloned()
    else {
        return Ok(());
    };
    let mut frontend = frontend;
    if frontend.get("composite.name").is_some() {
        frontend.0.retain(|attribute| {
            !matches!(
                attribute.key.as_str(),
                "composite.name" | "composite.attributes" | "composite.version"
            )
        });
        call.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_frontend_attributes"), frontend.clone());
    }
    let fuse = if frontend.get("MUST_FUSE").is_some() {
        "MUST_FUSE"
    } else {
        "MAXIMAL_FUSE"
    };
    let annotations: Vec<_> = [fuse, "mosaic_fusion_group"]
        .into_iter()
        .filter_map(|key| {
            frontend.get(key).map(|value| StringPair {
                key: key.into(),
                value: value.into(),
            })
        })
        .collect();
    if annotations.is_empty() {
        return Ok(());
    }
    for &source in sources {
        let fusible = match opcode(ctx, source)? {
            Opcode::Domain
            | Opcode::Parameter
            | Opcode::While
            | Opcode::Conditional
            | Opcode::Call => false,
            Opcode::Fusion | Opcode::Map | Opcode::Reduce | Opcode::ReduceWindow => true,
            Opcode::Rng => {
                source
                    .deref(ctx)
                    .get_result(0)
                    .uses(ctx)
                    .into_iter()
                    .map(|use_| use_.user_op())
                    .collect::<HashSet<_>>()
                    .len()
                    <= 1
            }
            _ => !effects.has_side_effect(source)?,
        };
        if fusible {
            let mut attributes =
                optional_attr::<FrontendAttributes>(ctx, source, "hlo_frontend_attributes")?
                    .as_deref()
                    .cloned()
                    .unwrap_or_default();
            for annotation in &annotations {
                attributes
                    .0
                    .retain(|previous| previous.key != annotation.key);
                attributes.0.push(annotation.clone());
            }
            source
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_frontend_attributes"), attributes);
        }
    }
    Ok(())
}

pub(super) fn async_computation(
    ctx: &Context,
    module: Ptr<Operation>,
    instruction: Ptr<Operation>,
) -> Result<Ptr<Operation>> {
    let mut computations = HashMap::new();
    let mut operations = Vec::new();
    for operation in edit_staging::tree(ctx, module) {
        let mut owner = Some(operation);
        while let Some(candidate) = owner {
            if Operation::is_op::<ComputationOp>(candidate, ctx) {
                break;
            }
            owner = candidate.deref(ctx).get_parent_op(ctx);
        }
        operations.push((operation, owner));
        if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx) {
            let name = computation
                .get_attr_hlo_computation_name(ctx)
                .ok_or_else(|| error("unnamed async computation"))?;
            require(
                computations.insert(name.clone(), operation).is_none(),
                "ambiguous async computation",
            )?;
        }
    }
    let starts = crate::verification::control::async_chain_starts(ctx, &operations, &computations)?;
    let start = starts
        .get(&instruction)
        .ok_or_else(|| error("async clone has no chain start"))?;
    let callees = optional_attr::<Callees>(ctx, *start, "hlo_callees")?
        .ok_or_else(|| error("async start has no wrapped computation"))?;
    require(
        callees.0.len() == 1,
        "async start has multiple wrapped computations",
    )?;
    computations
        .get(&callees.0[0])
        .copied()
        .ok_or_else(|| error("unknown async wrapped computation"))
}

pub(super) fn propagate_metadata(
    ctx: &Context,
    module: Ptr<Operation>,
    instruction: Ptr<Operation>,
    metadata: &OpMetadata,
    active: &mut HashSet<Ptr<Operation>>,
) -> Result<()> {
    if metadata.op_name.is_empty() && metadata.stack_frame_id == 0 {
        return Ok(());
    }
    require(
        active.insert(instruction),
        "recursive metadata propagation call graph",
    )?;
    let kind = opcode(ctx, instruction)?;
    if matches!(
        kind,
        Opcode::While
            | Opcode::Conditional
            | Opcode::AsyncStart
            | Opcode::AsyncUpdate
            | Opcode::AsyncDone
    ) {
        let module_body = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .ok_or_else(|| error("module has no body"))?;
        let mut callees = optional_attr::<Callees>(ctx, instruction, "hlo_callees")?
            .as_deref()
            .map(|callees| callees.0.clone())
            .unwrap_or_default();
        if !matches!(kind, Opcode::AsyncUpdate | Opcode::AsyncDone) {
            require(
                !callees.is_empty(),
                "control-flow metadata propagation has no callees",
            )?;
        }
        if matches!(
            kind,
            Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone
        ) {
            let wrapped = async_computation(ctx, module, instruction)?;
            let computation = Operation::get_op::<ComputationOp>(wrapped, ctx)
                .ok_or_else(|| error("async metadata target is not a computation"))?;
            callees = vec![
                computation
                    .get_attr_hlo_computation_name(ctx)
                    .ok_or_else(|| error("unnamed async wrapped computation"))?
                    .clone(),
            ];
        }
        for symbol in &callees {
            let target = module_body
                .deref(ctx)
                .iter(ctx)
                .find(|&operation| {
                    Operation::get_op::<ComputationOp>(operation, ctx).is_some_and(|computation| {
                        computation
                            .get_attr_hlo_computation_name(ctx)
                            .is_some_and(|name| *name == *symbol)
                    })
                })
                .ok_or_else(|| error("metadata propagation has an unknown control-flow callee"))?;
            for operation in instruction_order(ctx, target)? {
                propagate_metadata(ctx, module, operation, metadata, active)?;
            }
        }
    }
    call_metadata::propagate(
        ctx,
        module,
        instruction,
        &metadata.op_name,
        metadata.stack_frame_id,
    )?;
    active.remove(&instruction);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{
        attr, checked, computation, fixture, name, provenance, roundtrip, source_root,
    };
    use super::*;
    use crate::attributes::Index;
    use crate::attributes::Payload;
    use crate::ir::InstructionSequence;
    use crate::ir::OriginalValue;
    use crate::ops::control::ParameterOp;
    use crate::ops::control::{AsyncDoneOp, AsyncStartOp, AsyncUpdateOp};
    use crate::types::ElementType;
    use pliron::builtin::types::FunctionType;
    use pliron::{op::Op, printable::Printable};

    #[test]
    fn original_value_call_hierarchy_absence_empty_and_prefix_rules_restore_root_after_replacement()
    {
        for hierarchy in [None, Some(None), Some(Some("")), Some(Some("outer"))] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, false);
            let root = source_root(ctx, &f);
            attr(ctx, root, "hlo_original_value", provenance("inside"));
            if let Some(hierarchy) = hierarchy {
                let mut value = provenance("call");
                value.call_hierarchy = hierarchy.into_iter().map(str::to_owned).collect();
                attr(ctx, f.call, "hlo_original_value", value);
            }
            let output = checked(ctx, &f);
            let result = optional_attr::<OriginalValue>(
                ctx,
                output.root.defining_op().unwrap(),
                "hlo_original_value",
            )
            .unwrap()
            .as_deref()
            .cloned();
            match hierarchy {
                Some(Some("")) => assert_eq!(result.unwrap(), provenance("inside")),
                Some(Some("outer")) => assert_eq!(result.unwrap(), provenance("outer/inside")),
                _ => assert!(result.is_none()),
            }
            assert_eq!(
                *optional_attr::<OriginalValue>(ctx, root, "hlo_original_value")
                    .unwrap()
                    .unwrap(),
                provenance("inside")
            );
        }
    }

    #[test]
    fn metadata_prefix_payload_and_fusion_annotations_use_pinned_fusibility_and_source_mutation_rules()
     {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let root = source_root(ctx, &f);
        attr(
            ctx,
            root,
            "hlo_metadata",
            OpMetadata {
                op_name: "/inside/".into(),
                ..Default::default()
            },
        );
        attr(
            ctx,
            f.call,
            "hlo_metadata",
            OpMetadata {
                op_name: "outside/".into(),
                metadata_payload: Payload::Absent,
                ..Default::default()
            },
        );
        attr(
            ctx,
            root,
            "hlo_frontend_attributes",
            FrontendAttributes::new([("MUST_FUSE".into(), "inner".into())]),
        );
        attr(
            ctx,
            f.call,
            "hlo_frontend_attributes",
            FrontendAttributes::new([
                ("MUST_FUSE".into(), "outer".into()),
                ("MAXIMAL_FUSE".into(), "ignored".into()),
                ("mosaic_fusion_group".into(), "mosaic".into()),
                ("composite.name".into(), "composite".into()),
                ("composite.attributes".into(), "{}".into()),
                ("composite.version".into(), "1".into()),
            ]),
        );
        let output = checked(ctx, &f);
        let cloned = output.root.defining_op().unwrap();
        assert_eq!(
            optional_attr::<OpMetadata>(ctx, cloned, "hlo_metadata")
                .unwrap()
                .unwrap()
                .op_name,
            "outside/inside"
        );
        assert_eq!(
            optional_attr::<OpMetadata>(ctx, root, "hlo_metadata")
                .unwrap()
                .unwrap()
                .op_name,
            "/inside/"
        );
        for instruction in [root, cloned] {
            let frontend =
                optional_attr::<FrontendAttributes>(ctx, instruction, "hlo_frontend_attributes")
                    .unwrap()
                    .unwrap();
            assert_eq!(frontend.get("MUST_FUSE"), Some("outer"));
            assert_eq!(frontend.get("mosaic_fusion_group"), Some("mosaic"));
            assert_eq!(frontend.get("MAXIMAL_FUSE"), None);
            assert_eq!(frontend.get("composite.name"), None);
        }
        assert!(
            optional_attr::<FrontendAttributes>(ctx, f.formal, "hlo_frontend_attributes")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn metadata_can_be_disabled_without_disabling_original_value_rules() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, false);
        let root = source_root(ctx, &f);
        attr(
            ctx,
            root,
            "hlo_metadata",
            OpMetadata {
                op_name: "inside".into(),
                ..Default::default()
            },
        );
        attr(ctx, root, "hlo_original_value", provenance("inside"));
        attr(
            ctx,
            f.call,
            "hlo_metadata",
            OpMetadata {
                op_name: "outside".into(),
                ..Default::default()
            },
        );
        let output = inline_call(
            ctx,
            f.call,
            InlineCallOptions {
                propagate_metadata: false,
            },
            Some(Default::default()),
        )
        .unwrap();
        let cloned = output.root.defining_op().unwrap();
        assert_eq!(
            optional_attr::<OpMetadata>(ctx, cloned, "hlo_metadata")
                .unwrap()
                .unwrap()
                .op_name,
            "inside"
        );
        assert!(
            optional_attr::<OriginalValue>(ctx, cloned, "hlo_original_value")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn async_start_inlining_clones_wrapped_owner_and_retains_source_chain() {
        for scheduled in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, true);
            let (work, work_block, work_parameter) = computation(ctx, f.module, "work", f.ty);
            let work_value = work_parameter.deref(ctx).get_result(0);
            ReturnOp::new(ctx, work_value)
                .get_operation()
                .insert_at_back(work_block, ctx);
            let inputs = shape::tuple(ctx, vec![f.ty]).unwrap();
            let state = shape::array(ctx, ElementType::S32, vec![]).unwrap();
            let async_type = shape::tuple(ctx, vec![inputs, f.ty, state]).unwrap();
            let ret = f.callee_block.deref(ctx).iter(ctx).last().unwrap();
            let input = f.formal.deref(ctx).get_result(0);
            let start =
                AsyncStartOp::new(ctx, async_type, vec![input], Callees(vec!["work".into()]));
            start.get_operation().insert_before(ctx, ret);
            name(ctx, start.get_operation(), "start");
            let input = start.result(ctx);
            let update = AsyncUpdateOp::new(ctx, async_type, vec![input]);
            update.get_operation().insert_before(ctx, ret);
            name(ctx, update.get_operation(), "update");
            let input = update.result(ctx);
            let done = AsyncDoneOp::new(ctx, f.ty, vec![input]);
            done.get_operation().insert_before(ctx, ret);
            name(ctx, done.get_operation(), "done");
            Operation::replace_operand(ret, ctx, 0, done.result(ctx));
            attr(
                ctx,
                f.call,
                "hlo_metadata",
                OpMetadata {
                    op_name: "caller".into(),
                    ..Default::default()
                },
            );
            if scheduled {
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
                            instructions: vec![
                                "callee.p".into(),
                                "start".into(),
                                "update".into(),
                                "done".into(),
                            ],
                        },
                        InstructionSequence {
                            computation: "work".into(),
                            instructions: vec!["work.p".into()],
                        },
                    ]),
                );
            }
            let before = work.disp(ctx).to_string();
            let output = checked(ctx, &f);
            let cloned_start = output.instructions[&start.get_operation()]
                .defining_op()
                .unwrap();
            let callee = optional_attr::<Callees>(ctx, cloned_start, "hlo_callees")
                .unwrap()
                .unwrap()
                .0[0]
                .clone();
            assert_ne!(callee, "work");
            assert_eq!(before, work.disp(ctx).to_string());
            assert_eq!(
                optional_attr::<Callees>(ctx, start.get_operation(), "hlo_callees")
                    .unwrap()
                    .unwrap()
                    .0[0],
                "work"
            );
            assert_eq!(
                output.root.defining_op().unwrap().deref(ctx).get_operand(0),
                output.instructions[&update.get_operation()]
            );
            let schedule = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
                .unwrap()
                .as_deref()
                .cloned();
            if scheduled {
                let schedule = schedule.unwrap();
                assert_eq!(
                    schedule
                        .0
                        .iter()
                        .find(|sequence| sequence.computation == "main")
                        .unwrap()
                        .instructions,
                    vec![
                        "main.p".to_owned(),
                        value_name(ctx, output.instructions[&start.get_operation()]).unwrap(),
                        value_name(ctx, output.instructions[&update.get_operation()]).unwrap(),
                        value_name(ctx, output.root).unwrap()
                    ]
                );
                let wrapped = schedule
                    .0
                    .iter()
                    .find(|sequence| sequence.computation == callee)
                    .unwrap();
                assert_eq!(wrapped.instructions.len(), 1);
                assert_ne!(wrapped.instructions[0], "work.p");
            } else {
                assert!(schedule.is_none());
            }
            roundtrip(ctx, f.module);
        }
    }

    #[test]
    fn control_flow_metadata_propagates_into_shared_callees_but_calls_stop_recursion() {
        use crate::{
            literal::Literal,
            ops::control::{ConstantOp, WhileOp},
        };
        let ctx = &mut Context::new();
        let f = fixture(ctx, true);
        let (_, loop_block, loop_parameter) = computation(ctx, f.module, "loop_body", f.ty);
        let value = loop_parameter.deref(ctx).get_result(0);
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(loop_block, ctx);
        let pred = shape::array(ctx, ElementType::Pred, vec![]).unwrap();
        let signature = FunctionType::get(ctx, vec![f.ty], vec![pred]).into();
        let condition = ComputationOp::new(ctx, Identifier::new("condition"), signature);
        let container = f
            .module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        condition.get_operation().insert_at_back(container, ctx);
        let condition_block = BasicBlock::new(ctx, None, vec![]);
        condition_block.insert_at_back(condition.body(ctx), ctx);
        let parameter = ParameterOp::new(ctx, f.ty, vec![], Index(0));
        parameter
            .get_operation()
            .insert_at_back(condition_block, ctx);
        name(ctx, parameter.get_operation(), "condition.p");
        let literal = Literal::from_bits(ctx, pred, &[0]).unwrap();
        let constant = ConstantOp::new(ctx, pred, vec![], literal);
        constant
            .get_operation()
            .insert_at_back(condition_block, ctx);
        name(ctx, constant.get_operation(), "condition.false");
        let value = constant.result(ctx);
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(condition_block, ctx);
        let ret = f.callee_block.deref(ctx).iter(ctx).last().unwrap();
        let input = f.formal.deref(ctx).get_result(0);
        let while_op = WhileOp::new(
            ctx,
            f.ty,
            vec![input],
            Callees(vec!["loop_body".into(), "condition".into()]),
        );
        while_op.get_operation().insert_before(ctx, ret);
        name(ctx, while_op.get_operation(), "loop");
        let value = while_op.result(ctx);
        Operation::replace_operand(ret, ctx, 0, value);
        attr(
            ctx,
            f.call,
            "hlo_metadata",
            OpMetadata {
                op_name: "outer".into(),
                ..Default::default()
            },
        );
        checked(ctx, &f);
        for operation in [
            loop_parameter,
            parameter.get_operation(),
            constant.get_operation(),
        ] {
            assert_eq!(
                optional_attr::<OpMetadata>(ctx, operation, "hlo_metadata")
                    .unwrap()
                    .unwrap()
                    .op_name,
                "outer"
            );
        }
        assert!(
            optional_attr::<OpMetadata>(ctx, while_op.get_operation(), "hlo_metadata")
                .unwrap()
                .is_none()
        );
        assert!(
            optional_attr::<OpMetadata>(ctx, f.formal, "hlo_metadata")
                .unwrap()
                .is_none()
        );
        roundtrip(ctx, f.module);
    }

    #[test]
    fn typed_backend_sharding_payload_and_order_absence_survive_native_clone_and_replacement_rules()
    {
        use crate::{
            attributes::Bytes,
            ir::{
                BackendConfigPayload,
                instruction_order::{InstructionOrder, set_order},
            },
            sharding::Sharding,
        };
        for explicit_order in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, false);
            let root = source_root(ctx, &f);
            let backend = BackendConfigPayload(Payload::Value(Bytes(vec![1, 2, 3])));
            attr(ctx, root, "hlo_backend_config_payload", backend.clone());
            attr(ctx, root, "hlo_sharding", Sharding::replicated());
            attr(ctx, f.call, "hlo_sharding", Sharding::maximal(0));
            let metadata = OpMetadata {
                op_name: "inner".into(),
                metadata_payload: Payload::Value(Bytes(vec![4, 5])),
                ..Default::default()
            };
            attr(ctx, root, "hlo_metadata", metadata.clone());
            attr(
                ctx,
                f.call,
                "hlo_metadata",
                OpMetadata {
                    op_name: "outer".into(),
                    metadata_payload: Payload::Value(Bytes(vec![9])),
                    ..Default::default()
                },
            );
            if explicit_order {
                set_order(
                    ctx,
                    f.caller,
                    InstructionOrder {
                        instructions: vec!["call".into(), "main.p".into()],
                        ..Default::default()
                    },
                )
                .unwrap();
                set_order(
                    ctx,
                    f.callee,
                    InstructionOrder {
                        instructions: vec!["callee.add".into(), "callee.p".into()],
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            let source = f.callee.disp(ctx).to_string();
            let output = checked(ctx, &f);
            let clone = output.root.defining_op().unwrap();
            assert_eq!(
                *optional_attr::<BackendConfigPayload>(ctx, clone, "hlo_backend_config_payload")
                    .unwrap()
                    .unwrap(),
                backend
            );
            assert_eq!(
                *optional_attr::<Sharding>(ctx, clone, "hlo_sharding")
                    .unwrap()
                    .unwrap(),
                Sharding::replicated()
            );
            assert_eq!(
                optional_attr::<OpMetadata>(ctx, clone, "hlo_metadata")
                    .unwrap()
                    .unwrap()
                    .metadata_payload,
                metadata.metadata_payload
            );
            assert_eq!(
                optional_attr::<OpMetadata>(ctx, clone, "hlo_metadata")
                    .unwrap()
                    .unwrap()
                    .op_name,
                "outer/inner"
            );
            assert!(
                optional_attr::<InstructionOrder>(ctx, f.caller, "hlo_instruction_order")
                    .unwrap()
                    .is_none()
            );
            assert_eq!(source, f.callee.disp(ctx).to_string());
            roundtrip(ctx, f.module);
        }
    }
}
