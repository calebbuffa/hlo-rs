use xlahlo::{Module, PrimitiveType as P, Shape, text};

#[test]
fn rng_contracts_follow_distribution_domains_and_scalar_parameters() {
    for (element, distribution, valid) in [
        ("f32", "rng_uniform", true),
        ("f64", "rng_normal", true),
        ("s32", "rng_uniform", true),
        ("pred", "rng_uniform", true),
        ("s32", "rng_normal", false),
        ("c64", "rng_uniform", false),
        ("f32", "rng_invalid", false),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ a = {element}[] parameter(0) b = {element}[] parameter(1) ROOT r = {element}[3] rng(a,b), distribution={distribution} }}"
        );
        let module = text::parse_module(&source).unwrap();
        assert_eq!(module.verify(Default::default()).is_ok(), valid);
        let printed = text::print_module(&module).unwrap();
        assert_eq!(
            text::parse_module(&printed)
                .unwrap()
                .verify(Default::default())
                .is_ok(),
            valid
        );
    }
    let source = "HloModule m ENTRY main { a = f32[] parameter(0) ROOT r = f32[3] rng(a,a), distribution=rng_uniform }";
    for source in [
        source.replace("rng(a,a)", "rng(a)"),
        source.replace("a = f32[]", "a = f32[1]"),
        source.replace("r = f32[3]", "r = f64[3]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}

#[test]
fn rng_bit_generator_retains_upstreams_permissive_output_contract() {
    for (output, valid) in [
        ("(u64[2],u32[4])", true),
        ("(u64[3],u32[4])", false),
        ("(u64[2],u32[4],u32[])", false),
        ("u32[4]", true),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ s = u64[2] parameter(0) ROOT r = {output} rng-bit-generator(s), algorithm=rng_philox }}"
        );
        let module = text::parse_module(&source).unwrap();
        assert_eq!(module.verify(Default::default()).is_ok(), valid);
        let printed = text::print_module(&module).unwrap();
        assert_eq!(
            text::parse_module(&printed)
                .unwrap()
                .verify(Default::default())
                .is_ok(),
            valid
        );
    }
    for opcode in ["replica-id", "partition-id"] {
        let source = format!("HloModule m ENTRY main {{ ROOT r = u32[] {opcode}() }}");
        text::parse_module(&source)
            .unwrap()
            .verify(Default::default())
            .unwrap();
        assert!(
            text::parse_module(&source.replace("u32[]", "s32[]"))
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let source = "HloModule m ENTRY main { ROOT r = u64[2] rng-get-and-update-state(), delta=-1 }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("delta=-1"));
    assert!(
        text::parse_module(&source.replace("u64[2]", "u64[3]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn feeds_and_dependency_tokens_have_checked_signatures() {
    let source = r#"
HloModule feeds
ENTRY main {
  tok = token[] after-all()
  data = (f32[3],token[]) infeed(tok), infeed_config="input\n\"name\\"
  value = f32[3] get-tuple-element(data), index=0
  incoming = token[] get-tuple-element(data), index=1
  output = token[] outfeed(value,incoming), outfeed_config='output\tname'
  ROOT result = f32[3] add-dependency(value,output)
}
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("outfeed_shape=f32[3]"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for source in [
        source
            .replace(
                "tok = token[] after-all()",
                "bad = f32[] parameter(0) tok = token[] after-all()",
            )
            .replace("infeed(tok)", "infeed(bad)"),
        source.replace("outfeed(value,incoming)", "outfeed(value,value)"),
        source.replace(
            "add-dependency(value,output)",
            "add-dependency(value,value)",
        ),
        source.replace("outfeed_config='output\\tname'", "outfeed_shape=f32[4]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let mut wire = module.to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "infeed")
        .unwrap()
        .shape = Some(
        Shape::tuple(vec![
            Shape::array(P::F32, vec![3]).unwrap(),
            Shape::scalar(P::U32).unwrap(),
        ])
        .unwrap()
        .into_proto(),
    )
    .into();
    assert!(
        Module::from_proto(wire)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    let source =
        "HloModule m ENTRY main { t = token[] after-all() ROOT i = (f32[],u32[]) infeed(t) }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let wire = module.to_proto().unwrap();
    let result = wire.computations[0]
        .instructions
        .iter()
        .find(|i| i.opcode == "infeed")
        .unwrap();
    assert_eq!(
        result.shape.as_ref().unwrap().tuple_shapes[1]
            .element_type
            .enum_value()
            .unwrap(),
        P::TOKEN
    );
}

#[test]
fn quoted_feed_configs_preserve_every_byte_and_c_escapes() {
    let source = r#"HloModule m ENTRY main { t = token[] after-all() ROOT i = (f32[],token[]) infeed(t), infeed_config="" }"#;
    let module = text::parse_module(source).unwrap();
    let bytes: Vec<u8> = (0..=255).collect();
    let mut wire = module.to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "infeed")
        .unwrap()
        .infeed_config = bytes.clone();
    let printed = text::print_module(&Module::from_proto(wire).unwrap()).unwrap();
    let reparsed = text::parse_module(&printed).unwrap().to_proto().unwrap();
    assert_eq!(
        reparsed.computations[0]
            .instructions
            .iter()
            .find(|i| i.opcode == "infeed")
            .unwrap()
            .infeed_config,
        bytes
    );
    let escaped = source.replace(
        "infeed_config=\"\"",
        r#"infeed_config="\a\b\f\n\r\t\v\\\'\"\?\101\x42\u0043\U0001F600""#,
    );
    let wire = text::parse_module(&escaped).unwrap().to_proto().unwrap();
    let value = &wire.computations[0]
        .instructions
        .iter()
        .find(|i| i.opcode == "infeed")
        .unwrap()
        .infeed_config;
    assert_eq!(
        &value[..14],
        &[
            7, 8, 12, 10, 13, 9, 11, b'\\', b'\'', b'"', b'?', b'A', b'B', b'C'
        ]
    );
    assert_eq!(&value[14..], &[0xf0, 0x9f, 0x98, 0x80]);
    for bad in [
        r#""\q""#,
        r#""\400""#,
        r#""\x100""#,
        r#""\uD800""#,
        r#""\x""#,
        r#""\UFFFFFFFF""#,
    ] {
        assert!(text::parse_module(&source.replace("\"\"", bad)).is_err());
    }
}
