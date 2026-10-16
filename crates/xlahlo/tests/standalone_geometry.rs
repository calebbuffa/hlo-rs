use xlahlo::{
    Shape,
    proto::xla_data::{LayoutProto, PaddingConfig},
    text,
};

#[test]
fn shape_lists_are_brace_delimited_and_share_module_shape_defaults() {
    for source in [
        "{}",
        "{f32[2], (s32[],token[]), b(f32[3])}",
        "{f32[<=4]{0:S(5)}}",
    ] {
        let values = text::parse_shape_list(source).unwrap();
        let printed = text::print_shape_list(&values).unwrap();
        assert_eq!(text::parse_shape_list(&printed).unwrap(), values);
    }
    for source in ["", "f32[2],s32[]", "{f32[2],}", "{} trailing", "{{}}"] {
        assert!(text::parse_shape_list(source).is_err(), "{source}");
    }
    let mut wire = Shape::token().into_proto();
    wire.dimensions.push(7);
    assert!(text::print_shape_list(&[Shape::from_proto(wire).unwrap()]).is_err());
}

#[test]
fn standalone_layouts_do_not_require_rank_or_permutation_validity() {
    for source in [
        "{}",
        "{1,0}",
        "{7,7,-1}",
        "{18446744073709551615}",
        "{1,0:T(2,*)L(8)E(4)S(5)SC(0:1,2)M(16)}",
    ] {
        let value = text::parse_layout(source).unwrap();
        let printed = text::print_layout(&value).unwrap();
        assert_eq!(text::parse_layout(&printed).unwrap(), value);
    }
    for source in ["", "{0,}", "{0} trailing", "{0:E(x)}"] {
        assert!(text::parse_layout(source).is_err(), "{source}");
    }
    let mut wire = LayoutProto::new();
    wire.special_fields.mut_unknown_fields().add_varint(999, 1);
    assert!(text::print_layout(&wire).is_err());
}

#[test]
fn padding_uses_signed_decimal_components_and_preserves_negative_values() {
    for source in [
        "0_0",
        "-1_2_3x4_-5_0",
        "-9223372036854775808_9223372036854775807",
    ] {
        let value = text::parse_padding_config(source).unwrap();
        let printed = text::print_padding_config(&value).unwrap();
        assert_eq!(text::parse_padding_config(&printed).unwrap(), value);
    }
    for source in [
        "",
        "1",
        "0_0_0_0",
        "0_0x",
        "0_0 trailing",
        "18446744073709551615_0",
        "0_0.5",
        "0_0_-1",
        "+1_0",
    ] {
        assert!(text::parse_padding_config(source).is_err(), "{source}");
    }
    assert!(text::print_padding_config(&PaddingConfig::new()).is_err());
    let mut wire = text::parse_padding_config("0_0").unwrap();
    wire.dimensions[0]
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    assert!(text::print_padding_config(&wire).is_err());
    let mut wire = text::parse_padding_config("0_0").unwrap();
    wire.dimensions[0].interior_padding = -1;
    assert!(text::print_padding_config(&wire).is_err());
}

#[test]
fn module_padding_and_standalone_padding_share_the_same_payload() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0) ROOT pad = f32[4] pad(p,z), padding=-1_3 }";
    let module = text::parse_module(source).unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    assert_eq!(
        module
            .instruction(root)
            .unwrap()
            .padding_config
            .as_ref()
            .unwrap(),
        &text::parse_padding_config("-1_3").unwrap()
    );
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    parsed.verify(Default::default()).unwrap();
}
