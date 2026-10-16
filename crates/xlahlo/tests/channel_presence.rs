use xlahlo::{
    CollectiveContext, EffectContext, Error, Instruction, Module, Opcode, PrimitiveType, Shape,
    VerifyOptions, text,
};

const PERMUTE: &str = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT c = f32[2] collective-permute(p), source_target_pairs={{0,0}} }";
const SEND: &str = "HloModule m ENTRY main { p = f32[2] parameter(0) t = token[] after-all() s = () send(p,t) ROOT d = token[] send-done(s) }";
const REDUCER: &str =
    "sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT r = f32[] add(a,b) }";

#[test]
fn text_retains_optional_presence_and_wire_import_uses_native_positive_ids() {
    for channel in [None, Some(-1), Some(0), Some(3)] {
        for base in [PERMUTE, SEND] {
            let attribute = channel.map_or_else(String::new, |id| format!(", channel_id={id}"));
            let source = if base == SEND {
                base.replace("send(p,t)", &format!("send(p,t){attribute}"))
                    .replace("send-done(s)", &format!("send-done(s){attribute}"))
            } else {
                base.replace(
                    "collective-permute(p)",
                    &format!("collective-permute(p){attribute}"),
                )
            };
            let module = text::parse_module(&source).unwrap();
            module.verify(Default::default()).unwrap();
            let root = module.root(module.entry().unwrap()).unwrap();
            assert_eq!(module.channel_id(root).unwrap(), channel);
            let printed = text::print_module(&module).unwrap();
            assert_eq!(printed.contains("channel_id="), channel.is_some());
            let reparsed = text::parse_module(&printed).unwrap();
            assert_eq!(
                reparsed
                    .channel_id(reparsed.root(reparsed.entry().unwrap()).unwrap())
                    .unwrap(),
                channel
            );
            let imported = Module::from_proto(module.to_proto().unwrap()).unwrap();
            assert_eq!(
                imported
                    .channel_id(imported.root(imported.entry().unwrap()).unwrap())
                    .unwrap(),
                channel.filter(|&id| id > 0)
            );
            imported.verify(Default::default()).unwrap();
            if channel.is_some_and(|id| id < 0) {
                assert_eq!(
                    text::print_module(&imported).unwrap_err().category,
                    "unsupported"
                );
            }
        }
    }
    assert!(
        text::parse_module(&SEND.replace("send-done(s)", "send-done(s), channel_id=0")).is_err()
    );
    assert!(
        text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0), channel_id=0 }")
            .is_err()
    );
}

#[test]
fn prototype_and_module_edits_forks_and_clones_preserve_semantic_presence() {
    for channel in [None, Some(-1), Some(0), Some(3)] {
        let prototype = Instruction::new(
            "c",
            Opcode::CollectivePermute,
            Shape::array(PrimitiveType::F32, vec![2]).unwrap(),
        )
        .with_channel_id(channel)
        .unwrap();
        assert_eq!(prototype.channel_id(), channel);
        assert_eq!(
            prototype.has_local_side_effect(Default::default()).unwrap(),
            channel.is_some()
        );
        assert!(
            !prototype
                .has_local_side_effect(EffectContext {
                    channel_id_sensitive: false,
                })
                .unwrap()
        );
        let prototype = prototype
            .with_fields(|i| i.backend_config = b"kept".to_vec())
            .unwrap();
        assert_eq!(prototype.channel_id(), channel);
        let mut module = text::parse_module(PERMUTE).unwrap();
        let computation = module.entry().unwrap();
        let root = module.root(computation).unwrap();
        let inserted = module
            .add_instruction(computation, prototype, &module.operands(root).unwrap(), &[])
            .unwrap();
        assert_eq!(module.channel_id(inserted).unwrap(), channel);
        module.set_channel_id(root, channel).unwrap();
        module
            .edit_instruction_fields(root, |i| {
                i.backend_config = b"kept".to_vec();
                Ok(())
            })
            .unwrap();
        assert_eq!(module.channel_id(root).unwrap(), channel);
        let fork = module.fork().unwrap();
        assert!(fork.channel_id(root).is_err());
        assert_eq!(
            fork.channel_id(fork.root(fork.entry().unwrap()).unwrap())
                .unwrap(),
            channel
        );
        let clone = module.clone_computation(computation, "copy").unwrap();
        assert_eq!(
            module.channel_id(clone.instructions[&root]).unwrap(),
            channel
        );
        let wire = module.to_proto().unwrap();
        assert!(
            module
                .edit(|staged| {
                    staged.set_channel_id(root, Some(99))?;
                    Err(Error::new("test", "rollback"))
                })
                .is_err()
        );
        assert_eq!(module.channel_id(root).unwrap(), channel);
        assert_eq!(module.to_proto().unwrap(), wire);
        module.set_channel_id(root, None).unwrap();
        assert_eq!(module.channel_id(root).unwrap(), None);
        let parameter = module.operands(root).unwrap()[0];
        assert!(module.set_channel_id(parameter, Some(0)).is_err());
    }
    let mut module = text::parse_module(PERMUTE).unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    module.set_channel_id(root, Some(-1)).unwrap();
    module
        .edit_instruction_fields(root, |i| {
            i.channel_id = 0;
            Ok(())
        })
        .unwrap();
    assert_eq!(module.channel_id(root).unwrap(), None);
    let prototype = Instruction::new(
        "c",
        Opcode::CollectivePermute,
        Shape::scalar(PrimitiveType::F32).unwrap(),
    )
    .with_channel_id(Some(-1))
    .unwrap()
    .with_fields(|i| i.channel_id = 0)
    .unwrap();
    assert_eq!(prototype.channel_id(), None);
    assert!(
        Instruction::new(
            "p",
            Opcode::Parameter,
            Shape::scalar(PrimitiveType::F32).unwrap()
        )
        .with_channel_id(Some(0))
        .is_err()
    );
}

#[test]
fn host_transfer_equivalence_includes_explicit_zero_and_negative_channels() {
    for channel in [None, Some(-1), Some(0), Some(3)] {
        let attribute = channel.map_or_else(String::new, |id| format!(", channel_id={id}"));
        let source = format!(
            "HloModule m ENTRY main {{ a = f32[2] parameter(0) b = f32[3] parameter(1) t = token[] after-all() s0 = () send(a,t), is_host_transfer=true{attribute} send1 = () send(b,t), is_host_transfer=true{attribute} d0 = token[] send-done(s0), is_host_transfer=true{attribute} d1 = token[] send-done(send1), is_host_transfer=true{attribute} ROOT r = (token[],token[]) tuple(d0,d1) }}"
        );
        assert_eq!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_ok(),
            channel.is_none()
        );
    }
}

#[test]
fn zero_and_negative_presence_control_safe_deletion_and_fusion_liveness() {
    let dead = PERMUTE.replace("ROOT c =", "c =").replace(
        "source_target_pairs={{0,0}}",
        "source_target_pairs={{0,0}} ROOT r = f32[2] copy(p)",
    );
    for channel in [None, Some(-1), Some(0), Some(3)] {
        let mut module = text::parse_module(&dead).unwrap();
        let c = module
            .instructions(module.entry().unwrap())
            .unwrap()
            .into_iter()
            .find(|&id| module.instruction(id).unwrap().opcode == "collective-permute")
            .unwrap();
        module.set_channel_id(c, channel).unwrap();
        assert_eq!(module.remove_instruction(c).is_err(), channel.is_some());
        if channel.is_some() {
            module
                .remove_instruction_with_effect_context(
                    c,
                    EffectContext {
                        channel_id_sensitive: false,
                    },
                )
                .unwrap();
        }
        module.validate_structure().unwrap();
        let attribute = channel.map_or_else(String::new, |id| format!(", channel_id={id}"));
        let source = format!(
            "HloModule m body {{ p = f32[2] parameter(0) c = f32[2] collective-permute(p), source_target_pairs={{{{0,0}}}}{attribute} ROOT r = f32[2] copy(p) }} ENTRY main {{ x = f32[2] parameter(0) ROOT f = f32[2] fusion(x), kind=kLoop, calls=body }}"
        );
        assert_eq!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_ok(),
            channel.is_some()
        );
    }
}

#[test]
fn collective_geometry_uses_presence_and_all_reduce_start_keeps_its_distinct_policy() {
    for channel in [None, Some(-1), Some(0), Some(3)] {
        let attribute = channel.map_or_else(String::new, |id| format!(", channel_id={id}"));
        let size = if channel.is_some() { 12 } else { 4 };
        let source = format!(
            "HloModule m ENTRY main {{ p = f32[2] parameter(0) ROOT g = f32[{size}] all-gather(p), dimensions={{0}}, replica_groups={{{{0,1}}}}{attribute} }}"
        );
        text::parse_module(&source)
            .unwrap()
            .verify(VerifyOptions {
                collective_context: CollectiveContext {
                    replica_count: 2,
                    partition_count: 3,
                },
                ..Default::default()
            })
            .unwrap();
        let source = format!(
            "HloModule m {REDUCER} ENTRY main {{ p = f32[2] parameter(0) ROOT r = f32[2] all-reduce(p), to_apply=sum, replica_groups={{{{0}}}}{attribute} }}"
        );
        let module = text::parse_module(&source).unwrap();
        assert_eq!(
            module.verify(Default::default()).is_ok(),
            channel.is_none_or(|id| id > 0)
        );
        let asynchronous = format!(
            "HloModule m {REDUCER} ENTRY main {{ p = f32[2] parameter(0) s = f32[2] all-reduce-start(p), to_apply=sum, replica_groups={{{{0}}}}{attribute} ROOT d = f32[2] all-reduce-done(s) }}"
        );
        text::parse_module(&asynchronous)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}

#[test]
fn legacy_channels_and_unprintable_optional_pairs_are_explicit() {
    let source = format!(
        "HloModule m {REDUCER} ENTRY main {{ p = f32[2] parameter(0) ROOT r = f32[2] all-reduce(p), to_apply=sum, replica_groups={{{{0}}}} }}"
    );
    let mut wire = text::parse_module(&source).unwrap().to_proto().unwrap();
    let i = wire
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
        .find(|i| i.opcode == "all-reduce")
        .unwrap();
    i.all_reduce_id = 7;
    let mut module = Module::from_proto(wire).unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    assert_eq!(module.channel_id(root).unwrap(), Some(7));
    module.verify(Default::default()).unwrap();
    module
        .edit_instruction_fields(root, |i| {
            i.channel_id = 8;
            Ok(())
        })
        .unwrap();
    assert!(module.verify(Default::default()).is_err());
    module.set_channel_id(root, Some(0)).unwrap();
    assert_eq!(module.instruction(root).unwrap().all_reduce_id, 0);
    let mut module = text::parse_module(SEND).unwrap();
    let done = module.root(module.entry().unwrap()).unwrap();
    module.set_channel_id(done, Some(0)).unwrap();
    assert_eq!(
        text::print_module(&module).unwrap_err().category,
        "unsupported"
    );
}
