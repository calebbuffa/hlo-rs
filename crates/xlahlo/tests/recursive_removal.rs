use xlahlo::{EffectContext, InstructionId, Module, text};

const DIAMOND: &str = include_str!("fixtures/recursive_removal.hlo");

fn named(module: &Module, name: &str) -> InstructionId {
    module
        .instructions(module.entry().unwrap())
        .unwrap()
        .into_iter()
        .find(|&id| module.instruction(id).unwrap().name == name)
        .unwrap()
}

#[test]
fn shared_and_repeated_operands_are_reconsidered_after_their_last_use_disappears() {
    let mut module = text::parse_module(DIAMOND).unwrap();
    let dead = named(&module, "dead");
    let left = named(&module, "left");
    let shared = named(&module, "shared");
    let parameter = named(&module, "p");
    let removed = module.remove_instruction_and_unused_operands(dead).unwrap();
    assert_eq!(removed, [dead, left, shared]);
    for id in removed {
        assert!(module.instruction(id).is_err());
    }
    module.instruction(parameter).unwrap();
    module.verify(Default::default()).unwrap();
    assert_eq!(
        module.instructions(module.entry().unwrap()).unwrap().len(),
        2
    );
    for source in [
        DIAMOND.replace("copy(p)", "add(p,shared)"),
        DIAMOND.replace("negate(p)", "negate(p), control-predecessors={p}"),
        DIAMOND.replace("copy(p)", "copy(p), control-predecessors={shared}"),
    ] {
        let mut module = text::parse_module(&source).unwrap();
        let dead = named(&module, "dead");
        let left = named(&module, "left");
        assert_eq!(
            module.remove_instruction_and_unused_operands(dead).unwrap(),
            [dead, left]
        );
        module.instruction(named(&module, "shared")).unwrap();
        module.verify(Default::default()).unwrap();
    }
}

#[test]
fn unsafe_targets_leave_the_entire_graph_and_semantic_state_unchanged() {
    for source in [
        DIAMOND.replace("copy(p)", "add(p,dead)"),
        DIAMOND.replace("copy(p)", "copy(p), control-predecessors={dead}"),
        DIAMOND.replace("add(shared,left)", "add(shared,left), control-predecessors={p}"),
        "HloModule m ENTRY main { p = f32[] parameter(0) dead = f32[] custom-call(p), custom_call_target=\"effect\", custom_call_has_side_effect=true ROOT r = f32[] copy(p) }".to_owned(),
    ] {
        let mut module = text::parse_module(&source).unwrap();
        let dead = named(&module, "dead");
        let before = module.to_proto().unwrap();
        assert!(module.remove_instruction_and_unused_operands(dead).is_err());
        assert_eq!(module.to_proto().unwrap(), before);
    }
    let mut module = text::parse_module(DIAMOND).unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    let parameter = named(&module, "p");
    let other = module.fork().unwrap();
    assert!(
        module
            .remove_instruction_and_unused_operands(other.root(other.entry().unwrap()).unwrap())
            .is_err()
    );
    assert!(module.remove_instruction_and_unused_operands(root).is_err());
    assert!(
        module
            .remove_instruction_and_unused_operands(parameter)
            .is_err()
    );
    let mut module = text::parse_module(
        "HloModule m ENTRY main { dead = f32[] parameter(0) ROOT r = f32[] constant(0) }",
    )
    .unwrap();
    let parameter = named(&module, "dead");
    assert!(
        module
            .remove_instruction_and_unused_operands(parameter)
            .is_err()
    );
}

#[test]
fn parameters_are_retained_even_when_cleanup_makes_them_unused() {
    let mut module = text::parse_module(
        "HloModule m ENTRY main { p = f32[] parameter(0) q = f32[] parameter(1) dead = f32[] negate(q) ROOT r = f32[] copy(p) }"
    ).unwrap();
    let dead = named(&module, "dead");
    let q = named(&module, "q");
    assert_eq!(
        module.remove_instruction_and_unused_operands(dead).unwrap(),
        [dead]
    );
    module.instruction(q).unwrap();
    assert!(module.users(q).unwrap().is_empty());
    module.verify(Default::default()).unwrap();
}

#[test]
fn cleanup_preserves_recursive_effects_and_respects_semantic_channel_policy() {
    let source = "HloModule m body { a = f32[2] parameter(0) e = () custom-call(), custom_call_target=\"effect\", custom_call_has_side_effect=true ROOT r = f32[2] copy(a) } ENTRY main { p = f32[2] parameter(0) c = f32[2] call(p), to_apply=body n = f32[2] negate(c) dead = f32[2] add(n,n) ROOT r = f32[2] copy(p) }";
    let mut module = text::parse_module(source).unwrap();
    let dead = named(&module, "dead");
    let n = named(&module, "n");
    assert_eq!(
        module.remove_instruction_and_unused_operands(dead).unwrap(),
        [dead, n]
    );
    module.instruction(named(&module, "c")).unwrap();
    module.verify(Default::default()).unwrap();
    for channel in [None, Some(-1), Some(0), Some(3)] {
        for sensitive in [true, false] {
            let source = "HloModule m ENTRY main { p = f32[2] parameter(0) c = f32[2] collective-permute(p), source_target_pairs={{0,0}} dead = f32[2] negate(c) ROOT r = f32[2] copy(p) }";
            let mut module = text::parse_module(source).unwrap();
            let c = named(&module, "c");
            module.set_channel_id(c, channel).unwrap();
            let dead = named(&module, "dead");
            let removed = module
                .remove_instruction_and_unused_operands_with_effect_context(
                    dead,
                    EffectContext {
                        channel_id_sensitive: sensitive,
                    },
                )
                .unwrap();
            assert_eq!(removed.contains(&c), channel.is_none() || !sensitive);
            module.verify(Default::default()).unwrap();
        }
    }
}

#[test]
fn native_after_all_barrier_keeps_effect_sources_including_one_tuple_projection() {
    for source in [
        "HloModule m ENTRY main { p = f32[] parameter(0) t = token[] after-all() e = token[] outfeed(p,t) dead = token[] after-all(e) ROOT r = token[] after-all() }",
        "HloModule m ENTRY main { t = token[] after-all() e = (f32[],token[]) infeed(t) g = token[] get-tuple-element(e), index=1 dead = token[] after-all(g) ROOT r = token[] after-all() }",
    ] {
        let mut module = text::parse_module(source).unwrap();
        let dead = named(&module, "dead");
        let before = module.to_proto().unwrap();
        assert!(
            module
                .remove_instruction_and_unused_operands(dead)
                .unwrap()
                .is_empty()
        );
        assert_eq!(module.to_proto().unwrap(), before);
        module.verify(Default::default()).unwrap();
    }
    let mut module = text::parse_module(
        "HloModule m ENTRY main { t = token[] after-all() dead = token[] after-all(t) ROOT r = token[] after-all() }"
    ).unwrap();
    let dead = named(&module, "dead");
    let t = named(&module, "t");
    assert_eq!(
        module.remove_instruction_and_unused_operands(dead).unwrap(),
        [dead, t]
    );
}

#[test]
fn cleanup_invalidates_schedules_and_preserves_surviving_payloads() {
    let source = DIAMOND
        .replace("HloModule cleanup", "HloModule cleanup, is_scheduled=true")
        .replace("copy(p)", "copy(p), metadata={op_name=\"keep\"}");
    let mut wire = text::parse_module(&source).unwrap().to_proto().unwrap();
    wire.special_fields
        .mut_unknown_fields()
        .add_varint(2000, 42);
    let mut module = Module::from_proto(wire).unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    let root_wire = module.instruction(root).unwrap().clone();
    let dead = named(&module, "dead");
    module.remove_instruction_and_unused_operands(dead).unwrap();
    assert!(module.as_proto().schedule.is_none());
    assert_eq!(module.instruction(root).unwrap(), &root_wire);
    assert_eq!(
        module.as_proto().special_fields.unknown_fields().get(2000),
        Some(protobuf::UnknownValueRef::Varint(42))
    );
    module.verify(Default::default()).unwrap();
}
