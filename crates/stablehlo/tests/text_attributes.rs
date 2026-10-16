use std::collections::BTreeMap;

use stablehlo::{
    ArrayOfTiling, Attribute, ChannelHandle, DotDimensionNumbers, IndexTensor, StableHloAttribute,
    Tiling,
    text::{TextErrorKind, parse_attribute, parse_type, print_attribute, print_type},
};

fn round_trip(input: &str) -> Attribute {
    let attribute = parse_attribute(input).unwrap_or_else(|error| panic!("{input}: {error}"));
    let printed = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&printed).unwrap(), attribute, "{printed}");
    assert_eq!(
        print_attribute(&parse_attribute(&printed).unwrap()).unwrap(),
        printed
    );
    attribute
}

#[test]
fn dimension_structs_match_reference_defaults_and_arbitrary_order() {
    // StablehloOps.cpp: the custom dimension structs permit omitted fields.
    let empty = round_trip("#stablehlo.dot<>");
    assert_eq!(
        empty,
        Attribute::StableHlo(Box::new(StableHloAttribute::DotDimensionNumbers(
            DotDimensionNumbers::default()
        )))
    );
    for input in [
        "#stablehlo.dot<rhs_contracting_dimensions = [0], lhs_contracting_dimensions = [1]>",
        "#stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [3]>",
        "#stablehlo.dot<lhs_batching_dimensions = [1, 0]>",
        "#stablehlo.gather<>",
        "#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>",
        "#stablehlo.gather<operand_batching_dims = [0], start_indices_batching_dims = [1]>",
        "#stablehlo.scatter<>",
        "#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>",
        "#stablehlo.scatter<input_batching_dims = [0], scatter_indices_batching_dims = [1],>",
    ] {
        round_trip(input);
    }
}

#[test]
fn duplicate_unknown_and_invalid_dimension_fields_are_rejected() {
    let duplicate = parse_attribute(
        "#stablehlo.dot<lhs_batching_dimensions = [], lhs_batching_dimensions = []>",
    )
    .unwrap_err();
    assert_eq!(
        duplicate.kind,
        TextErrorKind::DuplicateField("lhs_batching_dimensions".into())
    );
    for input in [
        "#stablehlo.dot<not_a_field = []>",
        "#stablehlo.gather<offset_dims = [-1]>",
        "#stablehlo.scatter<index_vector_dim = -1>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn channel_handles_preserve_signed_values_and_require_all_fields() {
    assert_eq!(
        round_trip("#stablehlo.channel_handle<type = 1, handle = -7>"),
        Attribute::StableHlo(Box::new(StableHloAttribute::ChannelHandle(ChannelHandle {
            handle: -7,
            channel_type: 1
        })))
    );
    for input in [
        "#stablehlo.channel_handle<>",
        "#stablehlo.channel_handle<handle = 1>",
        "#stablehlo.channel_handle<type = 1>",
        "#stablehlo.channel_handle<handle = 1, type = 2, type = 3>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn aliases_and_algorithms_use_ods_field_order() {
    round_trip(
        "#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = [2, 0]>",
    );
    round_trip(
        "#stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
    );
    for input in [
        "#stablehlo.output_operand_alias<operand_index = 1, output_tuple_indices = [], operand_tuple_indices = []>",
        "#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = -1, operand_tuple_indices = []>",
        "#stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 0, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn bounds_have_custom_and_generated_spellings() {
    let custom = round_trip("#stablehlo.bounds<?, 2, 4>");
    let generated = round_trip("#stablehlo.type_extensions<bounds = [?, 2, 4]>");
    assert_eq!(custom, generated);
    assert_eq!(
        print_attribute(&generated).unwrap(),
        "#stablehlo.bounds<?, 2, 4>"
    );
}

#[test]
fn builtin_attributes_and_nested_metadata_round_trip() {
    for input in [
        "unit",
        "true",
        "false",
        "-9223372036854775808 : i64",
        "9223372036854775807 : i64",
        "0xFF : i64",
        "-0xFE : i64",
        "-1.25e-100 : f64",
        "-0.0 : f64",
        "[]",
        "[true, 4 : i64, \"value\", tensor<2xf32>]",
        "{}",
        "{a, nested = {name = \"text\", dims = [1 : i64, 2 : i64]}}",
        "{decomposition = @foo::@bar, ty = tensor<?xf32, #stablehlo.bounds<4>>}",
        "@foo",
        "@\"foo.bar\"::@\"name with spaces\"",
    ] {
        round_trip(input);
    }

    assert_eq!(
        parse_attribute("{flag}").unwrap(),
        Attribute::Dictionary(BTreeMap::from([("flag".into(), Attribute::Unit)]))
    );
}

#[test]
fn hexadecimal_float_assembly_preserves_nonfinite_payloads() {
    for bits in [
        0_u64,
        1,
        0x8000000000000000,
        0x3ff0000000000000,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x7ff8000000000123,
        0x7ff0000000000001,
    ] {
        let attribute = Attribute::float_bits(bits);
        let input = format!("0x{bits:016X} : f64");
        assert_eq!(parse_attribute(&input).unwrap(), attribute);
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
    assert_eq!(
        parse_attribute("0xFFFFFFFFFFFFFFFF : i64").unwrap(),
        Attribute::integer(-1)
    );
    assert_eq!(
        parse_attribute("-0.0 : f64").unwrap(),
        Attribute::float_bits((-0.0_f64).to_bits())
    );
    assert!(parse_attribute("0x10000000000000000 : f64").is_err());
}
#[test]
fn string_escaping_is_byte_exact_for_utf8_and_controls() {
    for input in [
        "\"plain\"",
        r#""quotes: \" and slash: \\""#,
        r#""\00\n\t\0D""#,
        "\"λ😀\"",
        r#""\CE\BB\F0\9F\98\80""#,
        r#""\FF""#,
        r#""\00\80\FE\FF""#,
        r#""binary\FFpayload" : i32"#,
        r#""typed text" : tensor<2xf32>"#,
    ] {
        round_trip(input);
    }
    assert_eq!(
        parse_attribute("\"λ😀\"").unwrap(),
        parse_attribute(r#""\CE\BB\F0\9F\98\80""#).unwrap()
    );
    for input in [r#""\q""#, r#""\0""#, "\"bad\nnewline\"", "\"unterminated"] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn type_encodings_use_the_shared_attribute_parser_and_validate_metadata() {
    let ty = parse_type("tensor<f32, {tag = \"tag\", data = [1 : i64, true]}>").unwrap();
    assert_eq!(parse_type(&print_type(&ty).unwrap()).unwrap(), ty);
    assert!(parse_type("tensor<f32, #stablehlo.gather<offset_dims = [-1]>>").is_err());
    assert!(parse_type("tensor<f32, {a = 1, a = 2}>").is_err());
}

#[test]
fn tilings_represent_arrays_of_rank_one_index_tensors() {
    // StablehloAttrs.td: StableHLO_TilingAttr is not a flat integer array.
    let tiles = Tiling(vec![IndexTensor(vec![2, 4]), IndexTensor(vec![1, 8])]);
    let attribute = Attribute::from(ArrayOfTiling(vec![tiles.clone()]));
    attribute.verify().unwrap();
    assert_eq!(tiles.0.len(), 2);
    assert_eq!(tiles.0[1].0, vec![1, 8]);
}

#[test]
fn malformed_attribute_inputs_are_explicit_errors() {
    for input in [
        "1.0 : i32",
        "1 : f32",
        "[1,]",
        "{a = 1, a = 2}",
        "true garbage",
        "#unknown.attr<1>",
        "#stablehlo.unknown<>",
        "#stablehlo.bounds<1,>",
        "{1 = 2}",
        "@1invalid",
        "@foo::bar",
        "\"text\" \"more\"",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
    assert_eq!(
        parse_attribute(&print_attribute(&Attribute::Bytes(vec![0xff])).unwrap()).unwrap(),
        Attribute::Bytes(vec![0xff])
    );
}

#[test]
fn nesting_limit_covers_attributes_inside_tensor_types() {
    let nested = format!("{}unit{}", "[".repeat(127), "]".repeat(127));
    let parsed = parse_attribute(&nested).unwrap();
    assert_eq!(print_attribute(&parsed).unwrap(), nested);

    let arrays = format!("{}unit{}", "[".repeat(129), "]".repeat(129));
    assert_eq!(
        parse_attribute(&arrays).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    let mixed = format!(
        "{}tensor<f32>{}",
        "tensor<f32, ".repeat(129),
        ">".repeat(129)
    );
    assert_eq!(
        parse_type(&mixed).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    let mut attribute = Attribute::Unit;
    for _ in 0..129 {
        attribute = Attribute::Array(vec![attribute]);
    }
    assert_eq!(
        print_attribute(&attribute).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    let mut ty = stablehlo::Type::from(stablehlo::TensorType::scalar(stablehlo::ElementType::F32));
    for _ in 0..129 {
        ty = stablehlo::TensorType::scalar(stablehlo::ElementType::F32)
            .with_encoding(Attribute::Type(Box::new(ty)))
            .into();
    }
    assert_eq!(
        print_type(&ty).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
}
