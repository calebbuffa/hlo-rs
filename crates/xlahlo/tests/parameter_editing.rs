use xlahlo::{ComputationId, InstructionId, Module, text};

fn parameter(module: &Module, computation: ComputationId, number: i64) -> InstructionId {
    module
        .instructions(computation)
        .unwrap()
        .into_iter()
        .find(|&id| {
            let instruction = module.instruction(id).unwrap();
            instruction.opcode == "parameter" && instruction.parameter_number == number
        })
        .unwrap()
}

#[test]
fn unused_call_parameters_update_every_caller_and_preserve_repeated_uses() {
    let mut module = text::parse_module("HloModule m body { a = f32[] parameter(0) b = f32[] parameter(1) c = f32[] parameter(2) ROOT n = f32[] negate(b) } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) z = f32[] parameter(2) c1 = f32[] call(x,y,z), to_apply=body c2 = f32[] call(z,y,x), to_apply=body ROOT r = (f32[],f32[]) tuple(c1,c2) }").unwrap();
    module.verify(Default::default()).unwrap();
    let body = module
        .computations()
        .find(|&c| module.computation(c).unwrap().name == "body")
        .unwrap();
    let entry = module.entry().unwrap();
    let x = parameter(&module, entry, 0);
    let y = parameter(&module, entry, 1);
    let z = parameter(&module, entry, 2);
    let removed = parameter(&module, body, 0);
    let old_b = parameter(&module, body, 1);
    let old_c = parameter(&module, body, 2);
    let replacements = module.remove_unused_parameter(removed, false).unwrap();
    assert_eq!(replacements.len(), 2);
    assert!(module.instruction(removed).is_err());
    assert!(module.instruction(old_b).is_err());
    let new_b = replacements[&old_b];
    assert_eq!(module.instruction(new_b).unwrap().parameter_number, 0);
    let calls = module
        .instructions(entry)
        .unwrap()
        .into_iter()
        .filter(|&id| module.instruction(id).unwrap().opcode == "call")
        .collect::<Vec<_>>();
    assert_eq!(module.operands(calls[0]).unwrap(), [y, z]);
    assert_eq!(module.operands(calls[1]).unwrap(), [y, x]);
    module.verify(Default::default()).unwrap();
    module
        .remove_unused_parameter(replacements[&old_c], false)
        .unwrap();
    assert_eq!(module.operands(calls[0]).unwrap(), [y]);
    assert_eq!(module.operands(calls[1]).unwrap(), [y]);
    assert!(module.instruction(x).is_ok());
    assert!(module.instruction(z).is_ok());
    module.verify(Default::default()).unwrap();
}

#[test]
fn entry_deletion_is_explicit_and_renumbers_metadata_without_losing_unknown_fields() {
    let mut module = text::parse_module(
        "HloModule m ENTRY main { a = f32[] parameter(0) ROOT b = f32[] parameter(1) }",
    )
    .unwrap();
    let entry = module.entry().unwrap();
    let unused = parameter(&module, entry, 0);
    let before = module.to_proto().unwrap();
    assert!(module.remove_unused_parameter(unused, false).is_err());
    assert_eq!(module.to_proto().unwrap(), before);
    let mut wire = module.to_proto_with_program_shapes().unwrap();
    wire.cross_program_prefetches
        .push(xlahlo::proto::hlo::CrossProgramPrefetch {
            parameter: 1,
            ..Default::default()
        });
    wire.input_output_alias = Some(xlahlo::proto::hlo::HloInputOutputAliasProto {
        entries: vec![
            xlahlo::proto::hlo::hlo_input_output_alias_proto::AliasEntryProto {
                parameter_number: 1,
                ..Default::default()
            },
        ],
        ..Default::default()
    })
    .into();
    wire.computations[0].instructions[1]
        .special_fields
        .mut_unknown_fields()
        .add_varint(900, 42);
    module = Module::from_proto(wire).unwrap();
    let entry = module.entry().unwrap();
    let old_root = module.root(entry).unwrap();
    let unused = parameter(&module, entry, 0);
    let replacements = module.remove_unused_parameter(unused, true).unwrap();
    assert_eq!(module.root(entry).unwrap(), replacements[&old_root]);
    assert!(module.instruction(old_root).is_err());
    assert!(
        module
            .instruction(module.root(entry).unwrap())
            .unwrap()
            .special_fields
            .unknown_fields()
            .get(900)
            .is_some()
    );
    assert_eq!(
        module
            .as_proto()
            .input_output_alias
            .as_ref()
            .unwrap()
            .entries[0]
            .parameter_number,
        0
    );
    assert_eq!(module.as_proto().cross_program_prefetches[0].parameter, 0);
    assert!(module.as_proto().host_program_shape.is_none());
    module.verify(Default::default()).unwrap();
}

#[test]
fn fusion_aliases_and_async_start_contexts_are_renumbered() {
    for (operation, shape, attributes, tail) in [
        (
            "fusion",
            "f32[]",
            ", kind=kLoop, calls=body, output_to_operand_aliasing={{}: (1,{})}",
            "",
        ),
        (
            "async-start",
            "((f32[],f32[]),f32[],s32[])",
            ", calls=body",
            "ROOT done = f32[] async-done(r)",
        ),
        (
            "async-start",
            "((f32[],f32[]),f32[],s32[])",
            ", calls=body, output_to_operand_aliasing={{0,1}: (1,{})}",
            "ROOT done = f32[] async-done(r)",
        ),
    ] {
        let root = if tail.is_empty() { "ROOT " } else { "" };
        let source = format!(
            "HloModule m body {{ a = f32[] parameter(0) ROOT b = f32[] parameter(1) }} ENTRY main {{ x = f32[] parameter(0) y = f32[] parameter(1) {root}r = {shape} {operation}(x,y){attributes} {tail} }}"
        );
        let mut module = text::parse_module(&source).unwrap();
        assert_eq!(
            module.verify(Default::default()).is_ok(),
            operation != "fusion"
        );
        let body = module
            .computations()
            .find(|&c| module.computation(c).unwrap().name == "body")
            .unwrap();
        let unused = parameter(&module, body, 0);
        module.remove_unused_parameter(unused, false).unwrap();
        module.verify(Default::default()).unwrap();
        let entry = module.entry().unwrap();
        let caller = module
            .instructions(entry)
            .unwrap()
            .into_iter()
            .find(|&id| module.instruction(id).unwrap().opcode == operation)
            .unwrap();
        assert_eq!(
            module.operands(caller).unwrap(),
            [parameter(&module, entry, 1)]
        );
        let caller = module.instruction(caller).unwrap();
        if operation == "fusion" {
            assert_eq!(caller.output_operand_aliasing[0].operand_index, 0);
        } else {
            assert_eq!(
                caller.shape.as_ref().unwrap().tuple_shapes[0]
                    .tuple_shapes
                    .len(),
                1
            );
            if let Some(alias) = caller.output_operand_aliasing.first() {
                assert_eq!(alias.operand_index, 0);
                assert_eq!(alias.output_shape_index, [0, 0]);
            }
        }
    }
}

#[test]
fn removed_slot_metadata_and_nontrivial_callers_cannot_be_silently_discarded() {
    for source in [
        "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] parameter(1) } ENTRY main { x = f32[] parameter(0) y = f32[] parameter(1) ROOT c = f32[] fusion(x,y), kind=kLoop, calls=body, output_to_operand_aliasing={{}: (0,{})} }",
        "HloModule m body { a = f32[] parameter(0) ROOT b = f32[] constant(0) } ENTRY main { x = f32[2] parameter(0) ROOT r = f32[2] map(x), to_apply=body }",
        "HloModule m, input_output_alias={{}: (0,{},may-alias)} ENTRY main { a = f32[] parameter(0) ROOT b = f32[] parameter(1) }",
    ] {
        let mut module = text::parse_module(source).unwrap();
        assert_eq!(
            module.verify(Default::default()).is_ok(),
            !source.contains("fusion")
        );
        let owner = module.computations().next().unwrap();
        let unused = parameter(&module, owner, 0);
        let before = module.to_proto().unwrap();
        assert!(module.remove_unused_parameter(unused, true).is_err());
        assert_eq!(module.to_proto().unwrap(), before);
    }
}

#[test]
fn replacement_parameters_keep_control_edges_and_deleted_handles_stay_stale() {
    let mut module = text::parse_module("HloModule m ENTRY main { a = f32[] parameter(0) b = f32[] parameter(1) ROOT c = f32[] parameter(2), control-predecessors={b} }").unwrap();
    let entry = module.entry().unwrap();
    let a = parameter(&module, entry, 0);
    let b = parameter(&module, entry, 1);
    let c = parameter(&module, entry, 2);
    let before = module.to_proto().unwrap();
    assert!(module.remove_unused_parameter(b, true).is_err());
    assert_eq!(module.to_proto().unwrap(), before);
    let replacements = module.remove_unused_parameter(a, true).unwrap();
    assert_eq!(
        module
            .instruction(replacements[&c])
            .unwrap()
            .control_predecessor_ids,
        [module.instruction(replacements[&b]).unwrap().id]
    );
    module.verify(Default::default()).unwrap();
}

#[test]
fn entry_donor_and_prefetch_references_protect_removed_slots() {
    let module = text::parse_module(
        "HloModule m ENTRY main { a = f32[] parameter(0) ROOT b = f32[] parameter(1) }",
    )
    .unwrap();
    for donor in [false, true] {
        let mut wire = module.to_proto().unwrap();
        if donor {
            wire.buffer_donor = Some(xlahlo::proto::hlo::HloBufferDonorProto {
                entries: vec![Default::default()],
                ..Default::default()
            })
            .into();
        } else {
            wire.cross_program_prefetches.push(Default::default());
        }

        let mut module = Module::from_proto(wire.clone()).unwrap();
        let entry = module.entry().unwrap();
        let unused = parameter(&module, entry, 0);
        assert!(module.remove_unused_parameter(unused, true).is_err());
        assert_eq!(module.to_proto().unwrap(), wire);
    }
}

#[test]
fn roots_live_parameters_and_foreign_handles_cannot_be_removed() {
    for source in [
        "HloModule m ENTRY main { ROOT a = f32[] parameter(0) }",
        "HloModule m ENTRY main { a = f32[] parameter(0) ROOT n = f32[] negate(a) }",
    ] {
        let mut module = text::parse_module(source).unwrap();
        let entry = module.entry().unwrap();
        let p = parameter(&module, entry, 0);
        let before = module.to_proto().unwrap();
        assert_eq!(
            module
                .remove_unused_parameter(p, true)
                .unwrap_err()
                .category,
            "live-instruction"
        );
        assert_eq!(module.to_proto().unwrap(), before);
        assert!(
            module
                .fork()
                .unwrap()
                .remove_unused_parameter(p, true)
                .is_err()
        );
    }
}
