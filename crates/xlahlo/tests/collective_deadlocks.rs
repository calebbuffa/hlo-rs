use xlahlo::{
    Module, VerifyOptions,
    proto::hlo::{HloScheduleProto, hlo_schedule_proto::InstructionSequence},
    text,
};

fn policy() -> VerifyOptions {
    VerifyOptions {
        check_for_collective_deadlocks: true,
        ..Default::default()
    }
}

fn schedule(module: &mut Module) {
    let mut wire = HloScheduleProto::new();
    for computation in module.computations() {
        wire.sequences.insert(
            module.computation(computation).unwrap().id,
            InstructionSequence {
                instruction_ids: module
                    .instructions(computation)
                    .unwrap()
                    .iter()
                    .map(|&id| module.instruction(id).unwrap().id)
                    .collect(),
                ..Default::default()
            },
        );
    }
    module.set_schedule(Some(wire)).unwrap();
}

fn transfer(index: usize, kind: &str, attributes: &str) -> String {
    let channel = index + 1;
    if kind == "s" {
        format!(
            "x{index} = () send(p,t), channel_id={channel}{attributes} d{index} = token[] send-done(x{index}), channel_id={channel}"
        )
    } else {
        format!(
            "x{index} = (f32[1]) recv(t), channel_id={channel}{attributes} d{index} = (f32[1],token[]) recv-done(x{index}), channel_id={channel}"
        )
    }
}

fn module(events: &[(&str, &str)]) -> Module {
    let mut source = String::from(
        "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT z = f32[] add(a,b) } ENTRY main { p = f32[1] parameter(0) t = token[] after-all() ",
    );
    for (index, &(kind, pairs)) in events.iter().enumerate() {
        if kind == "c" {
            source.push_str(&format!(
                "c{index} = f32[1] all-gather(p), dimensions={{0}}, replica_groups={{}} "
            ));
        } else if kind == "a" {
            source.push_str(&format!(
                "a{index} = f32[1] all-reduce(p), replica_groups={{}}, to_apply=sum "
            ));
        } else {
            let attributes =
                format!(", frontend_attributes={{_xla_send_recv_source_target_pairs=\"{pairs}\"}}");
            source.push_str(&transfer(index, kind, &attributes));
            source.push(' ');
        }
    }
    source.push_str("ROOT result = f32[1] copy(p) }");
    text::parse_module(&source).unwrap()
}

#[test]
fn paired_and_disjoint_pipelined_transfers_match_in_both_directions() {
    for events in [
        vec![("s", "{{0,1}}"), ("r", "{{0,1}}")],
        vec![("r", "{{0,1}}"), ("s", "{{0,1}}")],
        vec![
            ("s", "{{0,1}}"),
            ("s", "{{2,3}}"),
            ("r", "{{0,1}}"),
            ("r", "{{2,3}}"),
        ],
        vec![
            ("r", "{{0,1}}"),
            ("r", "{{2,3}}"),
            ("s", "{{0,1}}"),
            ("s", "{{2,3}}"),
        ],
        vec![("s", "{}"), ("r", "{}")],
        vec![("s", "{{0,-1}}"), ("r", "{{0,-1}}")],
        vec![("s", "[2,2]<=[4]"), ("r", "[2,2]<=[4]")],
        vec![
            ("s", "mesh[\\\"x\\\"=2] {\\\"x\\\"}"),
            ("r", "mesh[\\\"x\\\"=2] {\\\"x\\\"}"),
        ],
    ] {
        let mut module = module(&events);
        schedule(&mut module);
        module.verify(policy()).unwrap();
    }
}

#[test]
fn cycles_duplicates_unmatched_transfers_and_intervening_collectives_fail() {
    for events in [
        vec![("s", "{{0,0}}")],
        vec![("s", "{{0,1},{1,0}}")],
        vec![("s", "{{0,1},{0,2}}")],
        vec![("s", "{{0,2},{1,2}}")],
        vec![("s", "{{0,1}}"), ("s", "{{1,0}}")],
        vec![("r", "{{0,1}}"), ("r", "{{1,0}}")],
        vec![("s", "{{0,1}}"), ("s", "{{0,2}}")],
        vec![("s", "{{0,1}}"), ("s", "{{2,1}}")],
        vec![("s", "{{0,1}}")],
        vec![("r", "{{0,1}}")],
        vec![("s", "{{0,1}}"), ("r", "{{2,3}}")],
        vec![("s", "{{0,1}}"), ("c", ""), ("r", "{{0,1}}")],
        vec![("r", "{{0,1}}"), ("c", ""), ("s", "{{0,1}}")],
        vec![("s", "{{0,1}}"), ("r", "{{0, 1}}")],
    ] {
        let mut module = module(&events);
        module.verify(policy()).unwrap();
        schedule(&mut module);
        module.verify(Default::default()).unwrap();
        assert_eq!(
            module.verify(policy()).unwrap_err().category,
            "deadlock",
            "{events:?}"
        );
    }
}

#[test]
fn malformed_pair_attributes_and_negative_sources_return_checked_errors() {
    for pairs in [
        "missing",
        "{{0,1,2}}",
        "{{0}}",
        "{{-1,0}}",
        "{{0,-2}}",
        "{{0,1}} trailing",
        "[1,3]<=[3]",
    ] {
        let mut module = module(&[("s", pairs)]);
        schedule(&mut module);
        assert!(module.verify(policy()).is_err(), "{pairs}");
    }
    let source = "HloModule m ENTRY main { p = f32[1] parameter(0) t = token[] after-all() s = () send(p,t), channel_id=1 d = token[] send-done(s), channel_id=1 ROOT result = f32[1] copy(p) }";
    let mut module = text::parse_module(source).unwrap();
    schedule(&mut module);
    assert_eq!(module.verify(policy()).unwrap_err().category, "deadlock");
}

#[test]
fn host_and_pipeline_transfers_are_exempt_before_attribute_parsing() {
    for attribute in [
        ", is_host_transfer=true",
        ", frontend_attributes={_xla_send_recv_pipeline=\"\"}",
    ] {
        let mut source = format!(
            "HloModule m ENTRY main {{ p = f32[1] parameter(0) t = token[] after-all() {} ROOT result = f32[1] copy(p) }}",
            transfer(0, "s", attribute)
        );
        if attribute.contains("is_host_transfer") {
            source = source.replace(
                "send-done(x0), channel_id=1",
                "send-done(x0), channel_id=1, is_host_transfer=true",
            );
        }
        let mut module = text::parse_module(&source).unwrap();
        schedule(&mut module);
        module.verify(policy()).unwrap();
    }
}

#[test]
fn pinned_state_reset_and_callee_branch_quirks_are_retained() {
    for events in [
        vec![("s", "{{0,1}}"), ("r", "{{2,3}}"), ("r", "{{0,1}}")],
        vec![("s", "{{0,1}}"), ("a", ""), ("r", "{{0,1}}")],
    ] {
        let mut module = module(&events);
        schedule(&mut module);
        module.verify(policy()).unwrap();
    }
}

#[test]
fn checker_traverses_backing_instruction_order_not_schedule_order() {
    let mut module = module(&[("s", "{{0,1}}"), ("c", ""), ("r", "{{0,1}}")]);
    schedule(&mut module);
    let mut wire = module.as_proto().schedule.as_ref().unwrap().clone();
    let entry = module.computation(module.entry().unwrap()).unwrap();
    let collective = entry
        .instructions
        .iter()
        .find(|instruction| instruction.opcode == "all-gather")
        .unwrap()
        .id;
    let sequence = wire.sequences.get_mut(&entry.id).unwrap();
    sequence.instruction_ids.retain(|&id| id != collective);
    sequence.instruction_ids.insert(2, collective);
    module.set_schedule(Some(wire)).unwrap();
    module.verify_schedule().unwrap();
    assert_eq!(module.verify(policy()).unwrap_err().category, "deadlock");
}

#[test]
fn send_and_recv_expectations_propagate_across_called_computations() {
    let source = r#"HloModule m
sender { p = f32[1] parameter(0) t = token[] after-all()
s = () send(p,t), channel_id=1, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
ROOT d = token[] send-done(s), channel_id=1 }
receiver { t = token[] after-all()
r = (f32[1]) recv(t), channel_id=2, frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
ROOT d = (f32[1],token[]) recv-done(r), channel_id=2 }
ENTRY main { p = f32[1] parameter(0) s = token[] call(p), to_apply=sender
r = (f32[1],token[]) call(), to_apply=receiver ROOT result = f32[1] copy(p) }"#;
    let mut module = text::parse_module(source).unwrap();
    schedule(&mut module);
    module.verify(policy()).unwrap();
    let source = source.replace(
        "r = (f32[1],token[]) call()",
        "g = f32[1] all-gather(p), dimensions={0}, replica_groups={} r = (f32[1],token[]) call()",
    );
    let mut module = text::parse_module(&source).unwrap();
    schedule(&mut module);
    assert_eq!(module.verify(policy()).unwrap_err().category, "deadlock");
}

#[test]
fn async_groups_isolate_state_and_use_the_pinned_parent_expectation_guard() {
    for outer in [false, true] {
        for balanced in [false, true] {
            let inner = transfer(
                10,
                "s",
                ", frontend_attributes={_xla_send_recv_source_target_pairs=\"{{2,3}}\"}",
            );
            let inner_recv = if balanced {
                transfer(
                    11,
                    "r",
                    ", frontend_attributes={_xla_send_recv_source_target_pairs=\"{{2,3}}\"}",
                )
            } else {
                String::new()
            };
            let outer_send = if outer {
                transfer(
                    0,
                    "s",
                    ", frontend_attributes={_xla_send_recv_source_target_pairs=\"{{0,1}}\"}",
                )
            } else {
                String::new()
            };
            let outer_recv = if outer {
                transfer(
                    1,
                    "r",
                    ", frontend_attributes={_xla_send_recv_source_target_pairs=\"{{0,1}}\"}",
                )
            } else {
                String::new()
            };
            let source = format!(
                "HloModule m work {{ p = f32[1] parameter(0) t = token[] after-all() {inner} {inner_recv} ROOT result = f32[1] copy(p) }} ENTRY main {{ p = f32[1] parameter(0) t = token[] after-all() {outer_send} a = ((f32[1]),f32[1],s32[]) async-start(p), calls=work, async_execution_thread=\"worker\" d = f32[1] async-done(a) {outer_recv} ROOT result = f32[1] copy(p) }}"
            );
            let mut module = text::parse_module(&source).unwrap();
            schedule(&mut module);
            assert_eq!(
                module.verify(policy()).is_ok(),
                !outer || balanced,
                "{source}"
            );
        }
    }
}

#[test]
fn compact_groups_are_limited_before_materialization() {
    let mut module = module(&[("s", "[1000001,2]<=[2000002]")]);
    schedule(&mut module);
    assert_eq!(
        module.verify(policy()).unwrap_err().category,
        "resource-limit"
    );
}

#[test]
fn duplicate_detection_preserves_the_native_int32_narrowing_quirk() {
    let pairs = "{{0,2},{1,4294967298}}";
    let mut module = module(&[("s", pairs), ("r", pairs)]);
    schedule(&mut module);
    let error = module.verify(policy()).unwrap_err();
    assert_eq!(error.category, "deadlock");
    assert!(error.message.contains("duplicate"));
}
