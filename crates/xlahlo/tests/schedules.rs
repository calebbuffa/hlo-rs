use xlahlo::proto::hlo::{HloScheduleProto, hlo_schedule_proto::InstructionSequence};
use xlahlo::{Module, PrimitiveType as P, Shape, text};

fn schedule(module: &Module) -> HloScheduleProto {
    let mut schedule = HloScheduleProto::new();
    for computation in module.to_proto().unwrap().computations {
        schedule.sequences.insert(
            computation.id,
            InstructionSequence {
                instruction_ids: computation.instructions.iter().map(|i| i.id).collect(),
                ..Default::default()
            },
        );
    }
    schedule
}

#[test]
fn schedules_require_data_control_order_and_exact_instruction_coverage() {
    let source = "HloModule m ENTRY main { p = f32[] parameter(0) q = f32[] parameter(1), control-predecessors={p} ROOT n = f32[] add(p,q) }";
    let mut module = text::parse_module(source).unwrap();
    let valid = schedule(&module);
    module.set_schedule(Some(valid.clone())).unwrap();
    module.verify(Default::default()).unwrap();
    let before = module.to_proto().unwrap();
    for kind in 0..4 {
        let mut invalid = valid.clone();
        let sequence = invalid.sequences.values_mut().next().unwrap();
        match kind {
            0 => sequence.instruction_ids.swap(0, 1),
            1 => sequence.instruction_ids.reverse(),
            2 => {
                sequence.instruction_ids.pop();
            }
            _ => sequence.instruction_ids.push(sequence.instruction_ids[0]),
        }
        assert!(module.set_schedule(Some(invalid)).is_err());
        assert_eq!(module.to_proto().unwrap(), before);
    }
}

#[test]
fn schedule_covers_live_callees_in_represented_threads_but_not_dead_computations() {
    let source = r#"
HloModule m
callee { ROOT p = f32[] parameter(0) }
dead { ROOT p = f32[] parameter(0) }
ENTRY main { p = f32[] parameter(0) ROOT c = f32[] call(p), to_apply=callee }
"#;
    let mut module = text::parse_module(source).unwrap();
    let ids: std::collections::BTreeMap<_, _> = module
        .as_proto()
        .computations
        .iter()
        .map(|c| (c.name.clone(), c.id))
        .collect();
    let mut valid = schedule(&module);
    valid.sequences.remove(&ids["dead"]);
    module.set_schedule(Some(valid.clone())).unwrap();
    module.verify_schedule().unwrap();
    valid.sequences.remove(&ids["callee"]);
    module.set_schedule(Some(valid)).unwrap();
    assert_eq!(module.verify_schedule().unwrap_err().category, "schedule");
    assert_eq!(
        module.verify(Default::default()).unwrap_err().category,
        "schedule"
    );
}

#[test]
fn imported_dead_and_fusion_schedules_follow_native_verification_policy() {
    let source = r#"HloModule m
fused { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
dead { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
ENTRY main { p = f32[] parameter(0) ROOT f = f32[] fusion(p), kind=kLoop, calls=fused }
"#;
    let module = text::parse_module(source).unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.schedule = Some(schedule(&module)).into();
    let computations = wire
        .computations
        .iter()
        .map(|computation| (computation.name.as_str(), computation))
        .collect::<std::collections::BTreeMap<_, _>>();
    let dead_id = computations["dead"].id;
    let fused_id = computations["fused"].id;
    let dead_sequence = wire
        .schedule
        .as_mut()
        .unwrap()
        .sequences
        .get_mut(&dead_id)
        .unwrap();
    dead_sequence.instruction_ids.clear();
    let fused_sequence = wire
        .schedule
        .as_mut()
        .unwrap()
        .sequences
        .get_mut(&fused_id)
        .unwrap();
    fused_sequence.instruction_ids.reverse();
    Module::from_proto(wire).unwrap().verify_schedule().unwrap();
}

#[test]
fn empty_and_explicit_main_threads_share_policy_checks_and_schedule_coverage() {
    let source = "HloModule m callee { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[] parameter(0) ROOT r = f32[] call(p), to_apply=callee }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    let callee = wire
        .computations
        .iter_mut()
        .find(|c| c.name == "callee")
        .unwrap();
    callee.execution_thread = "main".into();
    let callee_id = callee.id;
    let mut module = Module::from_proto(wire).unwrap();
    module
        .verify(xlahlo::VerifyOptions {
            verify_call_nested_computation_thread_name: true,
            ..Default::default()
        })
        .unwrap();
    let mut schedules = schedule(&module);
    module.set_schedule(Some(schedules.clone())).unwrap();
    module.verify_schedule().unwrap();
    schedules.sequences.remove(&callee_id);
    module.set_schedule(Some(schedules)).unwrap();
    assert_eq!(module.verify_schedule().unwrap_err().category, "schedule");
}

#[test]
fn changing_graphs_invalidates_schedule_and_verified_edits_roll_back_semantic_failures() {
    let mut module = text::parse_module(
        "HloModule m ENTRY main { p = f32[] parameter(0) ROOT n = f32[] negate(p) }",
    )
    .unwrap();
    module.set_schedule(Some(schedule(&module))).unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    let before = module.to_proto().unwrap();
    assert!(
        module
            .edit_verified(Default::default(), |staged| staged
                .set_shape(root, Shape::scalar(P::S32)?))
            .is_err()
    );
    assert_eq!(module.to_proto().unwrap(), before);
    module
        .edit_verified(Default::default(), |staged| {
            staged.edit_instruction_fields(root, |i| {
                i.opcode = "abs".into();
                Ok(())
            })
        })
        .unwrap();
    assert!(module.as_proto().schedule.is_none());
    module.verify(Default::default()).unwrap();
}

#[test]
fn scheduled_module_header_uses_source_order_and_excludes_only_fusions() {
    let source = "HloModule m, is_scheduled=true fused { p = f32[] parameter(0) ROOT n = f32[] negate(p) } dead { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[] parameter(0) ROOT f = f32[] fusion(p), kind=kLoop, calls=fused }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let schedule = module.as_proto().schedule.as_ref().unwrap();
    assert_eq!(schedule.sequences.len(), 2);
    for computation in &module.as_proto().computations {
        if computation.name == "fused" {
            assert!(!schedule.sequences.contains_key(&computation.id));
        } else {
            assert_eq!(
                schedule.sequences[&computation.id].instruction_ids,
                computation
                    .instructions
                    .iter()
                    .map(|i| i.id)
                    .collect::<Vec<_>>()
            );
        }
    }
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("is_scheduled=true"));
    let parsed = text::parse_module(&printed).unwrap();
    parsed.verify(Default::default()).unwrap();
    assert_eq!(text::print_module(&parsed).unwrap(), printed);
}

#[test]
fn scheduled_header_checks_boolean_uniqueness_and_dependency_order() {
    let source = "HloModule m ENTRY main { ROOT p = f32[] parameter(0) }";
    assert!(
        text::parse_module(&source.replace("HloModule m", "HloModule m, is_scheduled=false"))
            .unwrap()
            .as_proto()
            .schedule
            .is_none()
    );
    for header in [
        "HloModule m, is_scheduled=1",
        "HloModule m, is_scheduled=true, is_scheduled=false",
    ] {
        assert!(text::parse_module(&source.replace("HloModule m", header)).is_err());
    }
    for instructions in [
        "ROOT n = f32[] negate(p) p = f32[] parameter(0)",
        "q = f32[] parameter(1), control-predecessors={p} ROOT p = f32[] parameter(0)",
    ] {
        let source = format!("HloModule m, is_scheduled=true ENTRY main {{ {instructions} }}");
        assert_eq!(
            text::parse_module(&source).unwrap_err().category,
            "schedule"
        );
    }
}

#[test]
fn scheduled_text_builds_sequences_after_async_thread_and_payload_normalization() {
    let source = "HloModule m, is_scheduled=true work { p = f32[] parameter(0) ROOT n = f32[] negate(p) } ENTRY main { p = f32[] parameter(0) s = ((f32[]),f32[],s32[]) async-start(p), calls=work, async_execution_thread=\"worker\", backend_config=\"opaque\" ROOT d = f32[] async-done(s) }";
    let module = text::parse_module(source).unwrap();
    assert_eq!(
        module.as_proto().schedule.as_ref().unwrap().sequences.len(),
        2
    );
    let work = module
        .as_proto()
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap();
    assert_eq!(work.execution_thread, "worker");
    assert_eq!(
        work.instructions
            .iter()
            .find(|i| i.name == "n")
            .unwrap()
            .backend_config,
        b"opaque"
    );
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    parsed.verify(Default::default()).unwrap();
    assert_eq!(text::print_module(&parsed).unwrap(), printed);
}

#[test]
fn schedule_printing_preserves_backing_order_instead_of_export_topological_tie_breaks() {
    let source = "HloModule m ENTRY main { p = f32[] parameter(0) q = f32[] parameter(1) ROOT n = f32[] add(p,q) }";
    let mut module = text::parse_module(source).unwrap();
    let mut schedule = schedule(&module);
    schedule
        .sequences
        .values_mut()
        .next()
        .unwrap()
        .instruction_ids
        .swap(0, 1);
    module.set_schedule(Some(schedule)).unwrap();
    assert_eq!(
        text::print_module(&module).unwrap_err().category,
        "unsupported"
    );
    let mut wire = module.to_proto().unwrap();
    wire.computations[0].instructions.swap(0, 1);
    let module = Module::from_proto(wire).unwrap();
    module.verify_schedule().unwrap();
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    assert_eq!(
        parsed.as_proto().computations[0]
            .instructions
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        ["q", "p", "n"]
    );
    parsed.verify(Default::default()).unwrap();
}

#[test]
fn schedule_printing_rejects_partial_schedules_and_unknown_schedule_metadata() {
    let source = "HloModule m, is_scheduled=true dead { ROOT p = f32[] parameter(0) } ENTRY main { ROOT p = f32[] parameter(0) }";
    let module = text::parse_module(source).unwrap();
    for kind in 0..3 {
        let mut wire = module.to_proto().unwrap();
        let dead = wire
            .computations
            .iter()
            .find(|c| c.name == "dead")
            .unwrap()
            .id;
        let schedule = wire.schedule.as_mut().unwrap();
        match kind {
            0 => {
                schedule.sequences.remove(&dead);
            }
            1 => schedule
                .special_fields
                .mut_unknown_fields()
                .add_varint(1000, 1),
            _ => schedule
                .sequences
                .values_mut()
                .next()
                .unwrap()
                .special_fields
                .mut_unknown_fields()
                .add_varint(1000, 1),
        }
        let module = Module::from_proto(wire).unwrap();
        assert_eq!(
            text::print_module(&module).unwrap_err().category,
            "unsupported"
        );
    }
}

#[test]
fn schedules_of_other_threads_do_not_require_main_thread_sequences() {
    let module = text::parse_module("HloModule m other { ROOT p = f32[] parameter(0) } ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
    let mut wire = module.to_proto().unwrap();
    let other = wire
        .computations
        .iter_mut()
        .find(|c| c.name == "other")
        .unwrap();
    other.execution_thread = "worker".into();
    let mut sched = HloScheduleProto::new();
    sched.sequences.insert(
        other.id,
        InstructionSequence {
            instruction_ids: other.instructions.iter().map(|i| i.id).collect(),
            ..Default::default()
        },
    );
    wire.schedule = Some(sched).into();
    Module::from_proto(wire).unwrap().verify_schedule().unwrap();
}
