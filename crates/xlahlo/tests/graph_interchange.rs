use xlahlo::{Instruction, Module, Opcode, PrimitiveType as P, Shape, text};

#[test]
fn native_export_materializes_signatures_without_changing_raw_presence() {
    let module = text::parse_module("HloModule m body { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[] parameter(0) ROOT c = f32[] call(p), to_apply=body }").unwrap();
    assert!(module.as_proto().host_program_shape.is_none());
    assert!(module.to_proto().unwrap().host_program_shape.is_none());
    let mut wire = module.to_proto_with_program_shapes().unwrap();
    assert!(wire.computations.iter().all(|c| c.program_shape.is_some()));
    assert_eq!(
        wire.host_program_shape.as_ref().unwrap().parameter_names,
        ["p0"]
    );
    wire.host_program_shape
        .as_mut()
        .unwrap()
        .special_fields
        .mut_unknown_fields()
        .add_varint(900, 42);
    wire.computations[0]
        .program_shape
        .as_mut()
        .unwrap()
        .special_fields
        .mut_unknown_fields()
        .add_varint(901, 43);
    let module = Module::from_proto(wire.clone()).unwrap();
    assert_eq!(module.to_proto_with_program_shapes().unwrap(), wire);
    let mut incomplete = wire.clone();
    incomplete
        .host_program_shape
        .as_mut()
        .unwrap()
        .parameter_names
        .clear();
    incomplete.computations[0]
        .program_shape
        .as_mut()
        .unwrap()
        .parameter_names
        .clear();
    let imported = Module::from_proto(incomplete.clone()).unwrap();
    assert_eq!(imported.to_proto_with_program_shapes().unwrap(), wire);
    assert_eq!(imported.as_proto(), &incomplete);
    incomplete
        .host_program_shape
        .as_mut()
        .unwrap()
        .parameter_names = vec!["a".into(), "b".into()];
    assert!(
        Module::from_proto(incomplete)
            .unwrap()
            .to_proto_with_program_shapes()
            .is_err()
    );
}

#[test]
fn instruction_names_are_module_unique_in_text_construction_cloning_and_import() {
    let mut module = Module::new("m");
    for (index, name) in ["body", "main"].iter().enumerate() {
        let computation = module.add_computation(*name).unwrap();
        let root = module
            .add_instruction(
                computation,
                Instruction::new("p", Opcode::Parameter, Shape::scalar(P::F32).unwrap()),
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(
            module.instruction(root).unwrap().name,
            if index == 0 { "p" } else { "p.1" }
        );
        module.set_root(computation, root).unwrap();
        module.set_entry(computation).unwrap();
    }
    module.verify(Default::default()).unwrap();
    let source = module.computations().next().unwrap();
    let clone = module.clone_computation(source, "cloned").unwrap();
    let root = module.root(clone.computation).unwrap();
    assert_eq!(module.instruction(root).unwrap().name, "p.clone");
    let clone = module.clone_computation(source, "cloned_again").unwrap();
    let root = module.root(clone.computation).unwrap();
    assert_eq!(module.instruction(root).unwrap().name, "p.clone.1");
    module.verify(Default::default()).unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.computations[1].instructions[0].name = wire.computations[0].instructions[0].name.clone();
    assert_eq!(
        Module::from_proto(wire).unwrap_err().category,
        "instruction"
    );
}

#[test]
fn indexed_metadata_payloads_are_checked_and_raw_wire_is_retained() {
    let module = text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0), metadata={metadata_payload=\"inline\"} }").unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.payloads = vec![b"indexed".to_vec()];
    for (id, valid) in [(0, true), (-1, false), (1, false), (i64::MAX, false)] {
        wire.computations[0].instructions[0]
            .metadata
            .as_mut()
            .unwrap()
            .metadata_payload
            .as_mut()
            .unwrap()
            .set_id(id);
        let result = Module::from_proto(wire.clone());
        assert_eq!(result.is_ok(), valid, "{id}");
        if let Ok(module) = result {
            module.verify(Default::default()).unwrap();
            assert_eq!(module.to_proto().unwrap(), wire);
        }
    }
    let mut module = module;
    let entry = module.entry().unwrap();
    let root = module.root(entry).unwrap();
    let mut invalid = module.instruction(root).unwrap().clone();
    invalid.name = "invalid".into();
    invalid
        .metadata
        .as_mut()
        .unwrap()
        .metadata_payload
        .as_mut()
        .unwrap()
        .set_id(-1);
    let before = module.to_proto().unwrap();
    assert!(
        module
            .add_instruction(entry, Instruction::from_proto(invalid).unwrap(), &[], &[])
            .is_err()
    );
    assert!(
        module
            .edit_instruction_fields(root, |i| {
                i.metadata
                    .as_mut()
                    .unwrap()
                    .metadata_payload
                    .as_mut()
                    .unwrap()
                    .set_id(0);
                Ok(())
            })
            .is_err()
    );
    assert_eq!(module.to_proto().unwrap(), before);
}
