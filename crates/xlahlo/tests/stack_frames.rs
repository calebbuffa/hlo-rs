use xlahlo::{Module, text};

const SOURCE: &str = include_str!("fixtures/stack_frames.hlo");

#[test]
fn stack_frames_are_interned_by_first_use_and_assembly_labels_are_not_ids() {
    let module = text::parse_module(SOURCE).unwrap();
    module.verify(Default::default()).unwrap();
    let index = module.as_proto().stack_frame_index.as_ref().unwrap();
    assert_eq!(index.file_names, ["code.rs"]);
    assert_eq!(index.function_names, ["caller", "callee"]);
    assert_eq!(index.stack_frames.len(), 2);
    assert_eq!(index.stack_frames[1].parent_frame_id, 1);
    assert_eq!(index.file_locations[1].end_column, 10);
    for instruction in &module.as_proto().computations[0].instructions {
        assert_eq!(instruction.metadata.as_ref().unwrap().stack_frame_id, 2);
    }
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    assert_eq!(
        module.as_proto().stack_frame_index,
        reparsed.as_proto().stack_frame_index
    );
}

#[test]
fn invalid_references_and_cycles_follow_native_recovery() {
    for source in [
        SOURCE.replace("stack_frame_id=3", "stack_frame_id=99"),
        SOURCE.replace("file_location_id=1", "file_location_id=-1"),
        SOURCE.replace("file_name_id=2", "file_name_id=0"),
        SOURCE.replace("function_name_id=1", "function_name_id=99"),
        SOURCE.replace(
            "file_location_id=1 parent_frame_id=1",
            "file_location_id=1 parent_frame_id=3",
        ),
    ] {
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        assert!(
            module.as_proto().computations[0]
                .instructions
                .iter()
                .any(|i| i.metadata.as_ref().unwrap().stack_frame_id == 0)
        );
    }
    let source =
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), metadata={stack_frame_id=8} }";
    let module = text::parse_module(source).unwrap();
    assert_eq!(
        module.as_proto().computations[0].instructions[0]
            .metadata
            .as_ref()
            .unwrap()
            .stack_frame_id,
        0
    );
}

#[test]
fn malformed_tables_and_unprintable_raw_state_fail_explicitly() {
    for source in [
        SOURCE.replace("FunctionNames", "FunctionNamesMissing"),
        SOURCE.replace(" line=3", ""),
        SOURCE.replace(" line=3", " line=2147483648"),
        SOURCE.replace("parent_frame_id=1", "parent_frame_id=-2147483648"),
    ] {
        assert!(text::parse_module(&source).is_err());
    }
    let module = text::parse_module(SOURCE).unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.stack_frame_index
        .as_mut()
        .unwrap()
        .file_names
        .push("unused".into());
    assert!(text::print_module(&Module::from_proto(wire).unwrap()).is_err());
    let mut wire = module.to_proto().unwrap();
    wire.stack_frame_index.as_mut().unwrap().stack_frames[0]
        .special_fields
        .mut_unknown_fields()
        .add_varint(9000, 1);
    assert!(text::print_module(&Module::from_proto(wire).unwrap()).is_err());
}
