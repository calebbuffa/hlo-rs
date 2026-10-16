use xlahlo::{
    interchange,
    proto::xla_data::{
        FrontendAttributes, OpMetadata, OriginalValueProto, ParameterReplication, StatisticsViz,
    },
    text,
};

#[test]
fn frontend_dictionaries_use_the_standalone_lexer_context() {
    let source = r#" // leading comment
        {z="last", a={"nested": {"text": "} // literal", "array": [1,2]}}, z="overwritten"}"#;
    let value = text::parse_frontend_attributes(source).unwrap();
    assert_eq!(value.map["z"], "overwritten");
    assert_eq!(
        value.map["a"],
        r#"{"nested": {"text": "} // literal", "array": [1,2]}}"#
    );
    let printed = text::print_frontend_attributes(&value).unwrap();
    assert_eq!(text::parse_frontend_attributes(&printed).unwrap(), value);
    let module = text::parse_module(&format!(
        "HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), frontend_attributes={source} }}"
    ))
    .unwrap();
    assert_eq!(
        module.as_proto().computations[0].instructions[0]
            .frontend_attributes
            .as_ref(),
        Some(&value)
    );
}

#[test]
fn standalone_statistics_preserve_double_bits_and_index_semantics() {
    for source in [
        "{}",
        "{ignored=18446744073709551615,a=nan(0x123),a=-0.0,b=1e999}",
    ] {
        let value = text::parse_statistics(source).unwrap();
        let printed = text::print_statistics(&value).unwrap();
        let after = text::parse_statistics(&printed).unwrap();
        assert_eq!(
            interchange::encode(&value).unwrap(),
            interchange::encode(&after).unwrap()
        );
    }
}

#[test]
fn standalone_original_values_share_normalization_and_lossless_printing() {
    for source in [
        r#"{}"#,
        r#"{{"p"{0}}}"#,
        r#"{({"p"},{},{"q"{1}}),["parent"]}"#,
        "{[synthetic_call]}",
    ] {
        let value = text::parse_original_value(source).unwrap();
        let printed = text::print_original_value(&value).unwrap();
        assert_eq!(text::parse_original_value(&printed).unwrap(), value);
    }
}

#[test]
fn replication_cardinality_remains_a_module_context_requirement() {
    for source in ["{}", "{true}", "{false,true,false}"] {
        let value = text::parse_parameter_replication(source).unwrap();
        let printed = text::print_parameter_replication(&value).unwrap();
        assert_eq!(text::parse_parameter_replication(&printed).unwrap(), value);
    }
    assert!(text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0), parameter_replication={true,false} }").is_err());
    assert!(
        text::parse_module(
            "HloModule m ENTRY main { ROOT p = f32[] parameter(0), parameter_replication={true} }"
        )
        .is_ok()
    );
}

#[test]
fn standalone_metadata_retains_locations_profile_types_and_payloads() {
    let value = text::parse_metadata(r#"{op_name="op" source_file="x.rs" source_line=18446744073709551615 profile_type={0} metadata_payload="bytes"}"#).unwrap();
    let printed = text::print_metadata(&value).unwrap();
    assert_eq!(text::parse_metadata(&printed).unwrap(), value);
    assert_eq!(value.source_line, -1);
}

#[test]
fn every_standalone_parser_consumes_all_input_and_rejects_malformed_syntax() {
    for source in ["{} trailing", "{", "{}", "{true,}"] {
        if source != "{}" {
            assert!(text::parse_parameter_replication(source).is_err());
        }
    }
    for source in ["{} trailing", "{"] {
        assert!(text::parse_original_value(source).is_err());
        assert!(text::parse_frontend_attributes(source).is_err());
        assert!(text::parse_statistics(source).is_err());
        assert!(text::parse_metadata(source).is_err());
    }
    assert!(text::parse_frontend_attributes(r#"{a={"x":1}"#).is_err());
    assert!(text::parse_statistics("{ignored=0,a=inf}").is_err());
    assert!(text::parse_metadata("{unrecognized=1}").is_err());
}

#[test]
fn standalone_printers_refuse_unknown_wire_fields() {
    let mut original = OriginalValueProto::new();
    original
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert!(text::print_original_value(&original).is_err());
    let mut frontend = FrontendAttributes::new();
    frontend
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert!(text::print_frontend_attributes(&frontend).is_err());
    let mut statistics = StatisticsViz::new();
    statistics
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert!(text::print_statistics(&statistics).is_err());
    let mut replication = ParameterReplication::new();
    replication
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert!(text::print_parameter_replication(&replication).is_err());
    let mut metadata = OpMetadata::new();
    metadata
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert!(text::print_metadata(&metadata).is_err());
}
