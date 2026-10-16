use xlahlo::{Module, text};

#[test]
fn recovery_keys_are_ordered_and_first_duplicate_wins() {
    let source = r#"HloModule m, origin_recovery_table={
 {"z" {2}}:{"new" {1}}
 {"a"}:{"first"}
 {"a"}:{"ignored"}
 } ENTRY main { ROOT c = s32[] constant(1) }"#;
    let module = text::parse_module(source).unwrap();
    let entries = &module
        .as_proto()
        .original_value_recovery_table
        .get_or_default()
        .entries;
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries[0]
            .old_original_array
            .get_or_default()
            .instruction_name,
        "a"
    );
    assert_eq!(
        entries[0]
            .new_original_array
            .get_or_default()
            .instruction_name,
        "first"
    );
    let printed = text::print_module(&module).unwrap();
    assert_eq!(
        text::parse_module(&printed)
            .unwrap()
            .as_proto()
            .original_value_recovery_table,
        module.as_proto().original_value_recovery_table,
    );
    assert!(
        text::parse_module(
            r#"HloModule m, origin_recovery_table={{"a"}:{"b"},{"c"}:{"d"}}
 ENTRY main { ROOT c = s32[] constant(1) }"#,
        )
        .is_err()
    );
}

#[test]
fn quoted_recovery_modules_preserve_nested_metadata_and_binary_graphs() {
    let source = r#"HloModule m, num_partitions=2, origin_recovery_table={
 {"old"}:{"new"}, "HloModule recovery, frontend_attributes={foo=\"bar\"}, backend_config=\"\\000\\377\"
 ENTRY main { ROOT c = s32[] constant(7), metadata={op_name=\"nested\"} }"
 } ENTRY main { ROOT c = s32[] constant(1) }"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let nested = module
        .as_proto()
        .original_value_recovery_table
        .get_or_default()
        .entries[0]
        .recovery_module
        .get_or_default();
    assert_eq!(
        nested.frontend_attributes.get_or_default().map["foo"],
        "bar"
    );
    assert_eq!(nested.backend_config, [0, 255]);
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    assert_eq!(
        module.as_proto().original_value_recovery_table,
        reparsed.as_proto().original_value_recovery_table
    );
    let bytes = xlahlo::interchange::encode(&module.to_proto().unwrap()).unwrap();
    let imported = Module::from_proto_with_config(
        xlahlo::interchange::decode(&bytes, Default::default()).unwrap(),
        module.config(),
    )
    .unwrap();
    assert_eq!(text::print_module(&imported).unwrap(), printed);
}

#[test]
fn recovery_limits_are_shared_across_nested_module_parses() {
    let source = r#"HloModule m, origin_recovery_table={
 {"a"}:{"b"}, "HloModule n ENTRY main { ROOT c = s32[] constant(1) }"
 } ENTRY main { ROOT c = s32[] constant(1) }"#;
    assert!(
        text::parse_module_with_options(
            source,
            text::ParseOptions {
                max_bytes: source.len(),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(text::parse_module(source).is_ok());
    let invalid = source.replace("HloModule n ENTRY", "HloModule n, num_partitions=2 ENTRY");
    assert!(text::parse_module(&invalid).is_err());
    let invalid = source.replace("constant(1) }\"", "constant(bad) }\"");
    assert!(text::parse_module(&invalid).is_err());
}

#[test]
fn recovery_printer_does_not_drop_noncanonical_or_unknown_wire_payloads() {
    let source = r#"HloModule m, origin_recovery_table={{"a"}:{"b"} {"c"}:{"d"}}
 ENTRY main { ROOT c = s32[] constant(1) }"#;
    let module = text::parse_module(source).unwrap();
    for change in 0..5 {
        let mut wire = module.to_proto().unwrap();
        let table = wire.original_value_recovery_table.mut_or_insert_default();
        match change {
            0 => table.entries.reverse(),
            1 => table.entries.clear(),
            2 => table.entries[0].old_original_array.clear(),
            3 => table.entries[0]
                .special_fields
                .mut_unknown_fields()
                .add_varint(1234, 1),
            _ => table.entries.push(table.entries[0].clone()),
        }
        let imported = Module::from_proto(wire).unwrap();
        assert!(text::print_module(&imported).is_err());
    }
}

#[test]
fn empty_recovery_tables_follow_native_absent_serialization() {
    let module = text::parse_module(
        "HloModule m, origin_recovery_table={} ENTRY main { ROOT c = s32[] constant(1) }",
    )
    .unwrap();
    assert!(module.as_proto().original_value_recovery_table.is_none());
}
