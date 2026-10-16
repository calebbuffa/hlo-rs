use xlahlo::{Module, text};

#[test]
fn integer_attributes_share_native_unsigned_token_wrapping_and_signed_range_checks() {
    for (token, expected) in [
        ("18446744073709551615", Some(-1)),
        ("18446744073709551614", Some(-2)),
        ("-2147483648", Some(i32::MIN)),
        ("2147483647", Some(i32::MAX)),
        ("2147483648", None),
        ("-2147483649", None),
        ("18446744073709551616", None),
        ("-9223372036854775809", None),
        ("+1", None),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), metadata={{source_line={token}}} }}"
        );
        let module = text::parse_module(&source);
        assert_eq!(module.is_ok(), expected.is_some(), "{token}");
        if let (Ok(module), Some(expected)) = (module, expected) {
            assert_eq!(
                module.as_proto().computations[0].instructions[0]
                    .metadata
                    .as_ref()
                    .unwrap()
                    .source_line,
                expected
            );
        }
    }
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT g = f32[2] all-gather(p), dimensions={0}, replica_groups={{0}}, channel_id=18446744073709551615 }";
    let module = text::parse_module(source).unwrap();
    assert_eq!(
        module
            .channel_id(module.root(module.entry().unwrap()).unwrap())
            .unwrap(),
        Some(-1)
    );
    module.verify(Default::default()).unwrap();
}

#[test]
fn parameter_replication_follows_non_tuple_leaf_counts_including_empty_tuples() {
    for (shape, replication, count) in [
        ("f32[]", "{true}", 1),
        ("token[]", "{false}", 1),
        ("()", "{}", 0),
        ("((),())", "{}", 0),
        ("(f32[],(s32[],()),token[])", "{true,false,true}", 3),
        ("b(f32[2])", "{false}", 1),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = {shape} parameter(0), parameter_replication={replication} }}"
        );
        let module = text::parse_module(&source).unwrap();
        let root = module.root(module.entry().unwrap()).unwrap();
        assert_eq!(module.shape(root).unwrap().leaf_count(), count);
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        assert_eq!(
            module.instruction(root).unwrap().parameter_replication,
            reparsed
                .instruction(reparsed.root(reparsed.entry().unwrap()).unwrap())
                .unwrap()
                .parameter_replication
        );
        assert_eq!(text::print_module(&reparsed).unwrap(), printed);
        let invalid = source.replace(replication, "{true,false,true,false}");
        assert!(text::parse_module(&invalid).is_err());
    }
    for replication in ["{}", "{true,false}", "{1}", "{true,}"] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), parameter_replication={replication} }}"
        );
        assert!(text::parse_module(&source).is_err());
    }
    let mut wire = text::parse_module(
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), parameter_replication={true} }",
    )
    .unwrap()
    .to_proto()
    .unwrap();
    wire.computations[0].instructions[0]
        .parameter_replication
        .as_mut()
        .unwrap()
        .replicated_at_leaf_buffers
        .clear();
    assert_eq!(
        text::print_module(&Module::from_proto(wire).unwrap())
            .unwrap_err()
            .category,
        "unsupported"
    );
}

#[test]
fn statistics_preserve_order_duplicates_indices_and_every_double_bit_pattern() {
    for statistics in [
        "{}",
        "{ignored=17}",
        "{ignored=-3,a=1,a=-0.0,b=1e20}",
        "{ignored=18446744073709551615,a=18446744073709551615}",
        "{visualizing_index=999,a=-7.0,b=150.0}",
        "{visualizing_index=0,a=nan(0x1),b=-nan(0x8000000000001),c=nan,d=1e999,e=-1e999}",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), statistics={statistics} }}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let stats = module.as_proto().computations[0].instructions[0]
            .statistics_viz
            .as_ref()
            .unwrap();
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        let again = reparsed.as_proto().computations[0].instructions[0]
            .statistics_viz
            .as_ref()
            .unwrap();
        assert_eq!(stats.stat_index_to_visualize, again.stat_index_to_visualize);
        assert_eq!(stats.statistics.len(), again.statistics.len());
        for (a, b) in stats.statistics.iter().zip(&again.statistics) {
            assert_eq!(a.stat_name, b.stat_name);
            assert_eq!(a.stat_val.to_bits(), b.stat_val.to_bits());
        }
        assert_eq!(text::print_module(&reparsed).unwrap(), printed);
    }
    for statistics in [
        "{a=0,b=inf}",
        "{a=0,b=-inf}",
        "{a=0,b=.5}",
        "{a=0,b=+1}",
        "{a=0,b=18446744073709551616}",
        "{a=0,}",
        "{a=0 b=1}",
        "{a=0.5}",
        "{a=0,\"quoted\"=1}",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), statistics={statistics} }}"
        );
        assert!(text::parse_module(&source).is_err(), "{statistics}");
    }
    let mut wire = text::parse_module(
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), statistics={a=0,b=1} }",
    )
    .unwrap()
    .to_proto()
    .unwrap();
    wire.computations[0].instructions[0]
        .statistics_viz
        .as_mut()
        .unwrap()
        .statistics[0]
        .special_fields
        .mut_unknown_fields()
        .add_varint(2000, 3);
    assert_eq!(
        text::print_module(&Module::from_proto(wire).unwrap())
            .unwrap_err()
            .category,
        "unsupported"
    );
}

#[test]
fn source_metadata_profiles_and_inline_payload_bytes_round_trip() {
    let source = r#"HloModule m ENTRY main { ROOT p = f32[] parameter(0), metadata={op_type="Parameter" op_name="p" source_file="code.rs" source_line=12 source_end_line=14 source_column=2 source_end_column=8 stack_frame_id=3 deduplicated_name="p.original" scheduling_name="p" profile_type={0} metadata_payload="\000\377\""} }"#;
    let module = text::parse_module(source).unwrap();
    let before = module.as_proto().computations[0].instructions[0]
        .metadata
        .as_ref()
        .unwrap();
    assert_eq!(
        before.metadata_payload.as_ref().unwrap().value(),
        [0, 255, 34]
    );
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    assert_eq!(
        before,
        reparsed.as_proto().computations[0].instructions[0]
            .metadata
            .as_ref()
            .unwrap()
    );
}

#[test]
fn unsupported_metadata_profiles_and_payload_references_fail_explicitly() {
    let module =
        text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0), metadata={} }")
            .unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.computations[0].instructions[0]
        .metadata
        .as_mut()
        .unwrap()
        .size_of_generated_code_in_bytes = 4;
    assert!(text::print_module(&Module::from_proto(wire).unwrap()).is_err());
    let mut wire = module.to_proto().unwrap();
    wire.payloads.push(b"indexed".to_vec());
    wire.computations[0].instructions[0]
        .metadata
        .as_mut()
        .unwrap()
        .metadata_payload = Some(xlahlo::proto::xla_data::Payload::new()).into();
    wire.computations[0].instructions[0]
        .metadata
        .as_mut()
        .unwrap()
        .metadata_payload
        .as_mut()
        .unwrap()
        .set_id(0);
    assert!(text::print_module(&Module::from_proto(wire).unwrap()).is_err());
    for metadata in [
        "{source_line=2147483648}",
        "{op_name=\"a\",op_name=\"b\"}",
        "{unknown=3}",
        "{profile_type={999}}",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), metadata={metadata} }}"
        );
        assert!(text::parse_module(&source).is_err());
    }
}

#[test]
fn raw_frontend_and_backend_dictionaries_preserve_exact_bytes_without_json_validation() {
    let dictionary =
        r#"{ "unicode": "\u0061", "quoted": "a}b\"{", "nested": { "values": [1,2,true] } }"#;
    let source = format!(
        "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), frontend_attributes={{configuration={dictionary},plain=\"value\"}}, backend_config={dictionary} }}"
    );
    let module = text::parse_module(&source).unwrap();
    let instruction = &module.as_proto().computations[0].instructions[0];
    assert_eq!(instruction.backend_config, dictionary.as_bytes());
    assert_eq!(
        instruction.frontend_attributes.as_ref().unwrap().map["configuration"],
        dictionary
    );
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    assert_eq!(
        instruction.backend_config,
        reparsed.as_proto().computations[0].instructions[0].backend_config
    );
    assert_eq!(
        instruction.frontend_attributes,
        reparsed.as_proto().computations[0].instructions[0].frontend_attributes
    );
    // The pinned lexer balances objects and double-quoted strings, not JSON syntax.
    let source =
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), backend_config={not valid JSON} }";
    assert_eq!(
        text::parse_module(source).unwrap().as_proto().computations[0].instructions[0]
            .backend_config,
        b"{not valid JSON}"
    );
    for source in [
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), backend_config={",
        "HloModule m ENTRY main { ROOT p = f32[] parameter(0), backend_config={\"unterminated} }",
    ] {
        assert!(text::parse_module(source).is_err());
    }
}
