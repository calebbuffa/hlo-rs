use xlahlo::{
    EffectContext, Instruction, InstructionId, Module, Opcode, PrimitiveType, Shape, text,
};

fn find(module: &Module, name: &str) -> InstructionId {
    module
        .computations()
        .flat_map(|computation| module.instructions(computation).unwrap())
        .find(|&id| module.instruction(id).unwrap().name == name)
        .unwrap()
}

#[test]
fn every_opcode_local_effect_matches_the_pinned_classification() {
    let unconditional = [
        Opcode::Send,
        Opcode::SendDone,
        Opcode::Recv,
        Opcode::RecvDone,
        Opcode::Rng,
        Opcode::RngGetAndUpdateState,
        Opcode::Infeed,
        Opcode::Outfeed,
        Opcode::AllReduceStart,
        Opcode::AllReduceDone,
        Opcode::AllGatherStart,
        Opcode::AllGatherDone,
        Opcode::CollectiveBroadcast,
        Opcode::CollectivePermuteStart,
        Opcode::CollectivePermuteDone,
    ];
    let constrained = [
        Opcode::AllToAll,
        Opcode::AllGather,
        Opcode::AllReduce,
        Opcode::ReduceScatter,
        Opcode::CollectiveReduce,
    ];
    for &opcode in Opcode::ALL {
        for channel in [0, 1, 17] {
            for constrain_layout in [false, true] {
                for custom_call_has_side_effect in [false, true] {
                    for channel_id_sensitive in [false, true] {
                        let instruction = Instruction::new(
                            "i",
                            opcode,
                            Shape::scalar(PrimitiveType::F32).unwrap(),
                        )
                        .with_fields(|wire| {
                            wire.channel_id = channel;
                            wire.constrain_layout = constrain_layout;
                            wire.custom_call_has_side_effect = custom_call_has_side_effect;
                        })
                        .unwrap();
                        let expected = unconditional.contains(&opcode)
                            || (constrained.contains(&opcode) && constrain_layout)
                            || ((constrained.contains(&opcode)
                                || opcode == Opcode::CollectivePermute)
                                && channel > 0
                                && channel_id_sensitive)
                            || (opcode == Opcode::CustomCall && custom_call_has_side_effect);
                        assert_eq!(
                            instruction
                                .has_local_side_effect(EffectContext {
                                    channel_id_sensitive
                                })
                                .unwrap(),
                            expected,
                            "{opcode} channel={channel} constrain={constrain_layout}"
                        );
                    }
                }
            }
        }
    }
}

fn calls(effect: bool) -> Module {
    let source = format!(
        "HloModule m leaf {{ a = f32[] parameter(0) e = f32[] custom-call(), custom_call_target=\"effect\", custom_call_has_side_effect={effect} ROOT r = f32[] copy(a) }} middle {{ b = f32[] parameter(0) ROOT s = f32[] call(b), to_apply=leaf }} ENTRY main {{ p = f32[] parameter(0) d = f32[] call(p), to_apply=middle forwarded = f32[] copy(d) ROOT out = f32[] copy(p) }}"
    );
    text::parse_module(&source).unwrap()
}

#[test]
fn effects_include_dead_callee_instructions_but_not_data_operands() {
    let context = EffectContext::default();
    let mut module = calls(true);
    module.verify(Default::default()).unwrap();
    let call = find(&module, "d");
    let forwarded = find(&module, "forwarded");
    assert!(
        !module
            .instruction_has_local_side_effect(call, context)
            .unwrap()
    );
    assert!(module.instruction_has_side_effect(call, context).unwrap());
    assert!(
        !module
            .instruction_has_side_effect(forwarded, context)
            .unwrap()
    );
    for computation in module.computations() {
        assert!(
            module
                .computation_has_side_effect(computation, context)
                .unwrap()
        );
    }
    module.remove_instruction(forwarded).unwrap();
    let before = module.to_proto().unwrap();
    assert!(module.remove_instruction(call).is_err());
    assert_eq!(module.to_proto().unwrap(), before);
    let foreign = calls(false);
    assert!(foreign.instruction_has_side_effect(call, context).is_err());
}

#[test]
fn pure_calls_and_custom_calls_are_removable_but_parameters_and_controls_are_not() {
    let mut module = calls(false);
    let context = EffectContext::default();
    for computation in module.computations() {
        assert!(
            !module
                .computation_has_side_effect(computation, context)
                .unwrap()
        );
    }
    let forwarded = find(&module, "forwarded");
    let call = find(&module, "d");
    module.remove_instruction(forwarded).unwrap();
    module.remove_instruction(call).unwrap();
    assert!(module.instruction_has_side_effect(call, context).is_err());
    module.remove_instruction(find(&module, "e")).unwrap();
    let parameter = find(&module, "p");
    let main = module
        .computations()
        .find(|&c| module.computation(c).unwrap().name == "main")
        .unwrap();
    let unused = module
        .add_instruction(
            main,
            Instruction::new(
                "unused",
                Opcode::Parameter,
                module.shape(parameter).unwrap(),
            )
            .with_fields(|wire| wire.parameter_number = 1)
            .unwrap(),
            &[],
            &[],
        )
        .unwrap();
    assert_eq!(
        module.remove_instruction(unused).unwrap_err().category,
        "unsupported"
    );
    let spare = module
        .add_instruction(
            main,
            Instruction::new("spare", Opcode::Copy, module.shape(parameter).unwrap()),
            &[parameter],
            &[],
        )
        .unwrap();
    module.add_control_dependency(parameter, spare).unwrap();
    let before = module.to_proto().unwrap();
    assert_eq!(
        module.remove_instruction(spare).unwrap_err().category,
        "live-instruction"
    );
    assert_eq!(module.to_proto().unwrap(), before);
    module.remove_control_dependency(parameter, spare).unwrap();
    module.remove_instruction(spare).unwrap();
    module.verify(Default::default()).unwrap();
}

#[test]
fn spmd_channel_policy_can_allow_otherwise_effectful_dead_collectives() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) c = f32[2] collective-permute(p), source_target_pairs={{0,0}}, channel_id=1 ROOT r = f32[2] copy(p) }";
    let mut module = text::parse_module(source).unwrap();
    let collective = find(&module, "c");
    assert!(module.remove_instruction(collective).is_err());
    module
        .remove_instruction_with_effect_context(
            collective,
            EffectContext {
                channel_id_sensitive: false,
            },
        )
        .unwrap();
    module.verify(Default::default()).unwrap();
}
