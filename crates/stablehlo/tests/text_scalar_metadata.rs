use stablehlo::{
    Attribute, ElementType, TensorType, Type,
    text::{parse_attribute, parse_type, print_attribute, print_type},
};

#[test]
fn byte_strings_preserve_every_byte_without_utf8_loss() {
    let bytes: Vec<u8> = (0..=255).collect();
    let attribute = Attribute::Bytes(bytes.clone());
    let text = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&text).unwrap(), attribute);
    for ty in [Type::Element(ElementType::I32), Type::Token] {
        let attribute = Attribute::TypedString {
            value: bytes.clone(),
            ty: Box::new(ty),
        };
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
    assert_eq!(
        parse_attribute("\"a\tb\rc\0d\"").unwrap(),
        Attribute::String("a\tb\rc\0d".into())
    );
    assert!(parse_attribute("{\"\" = unit}").is_err());
    assert!(parse_attribute("@\"nonutf8\\FF\"").is_err());
}

#[test]
fn signless_i64_literals_accept_the_full_bit_pattern_range() {
    for (text, expected) in [
        ("9223372036854775808 : i64", i64::MIN),
        ("18446744073709551615 : i64", -1),
        ("0xffffffffffffffff : i64", -1),
        ("-9223372036854775808 : i64", i64::MIN),
        ("0 : i64", 0),
    ] {
        assert_eq!(
            parse_attribute(text).unwrap(),
            Attribute::integer(expected),
            "{text}"
        );
    }
    assert_eq!(
        parse_attribute("array<i64: 18446744073709551615, 9223372036854775808>").unwrap(),
        Attribute::integers(vec![-1, i64::MIN]),
    );
    for text in [
        "-0 : i64",
        "-0x0 : i64",
        "+1 : i64",
        "-+1 : i64",
        "18446744073709551616 : i64",
    ] {
        assert!(parse_attribute(text).is_err(), "{text}");
    }
    for ty in ["i1", "i8", "si8", "ui8", "index"] {
        assert!(
            parse_attribute(&format!("dense<-0> : tensor<{ty}>")).is_err(),
            "{ty}"
        );
    }
}

#[test]
fn float_literals_follow_pinned_lexer_not_host_number_grammar() {
    for literal in ["1e0", ".5", "+1.0", "1.0e", "-0x0", "1", "NaN", "inf"] {
        for text in [
            format!("{literal} : f64"),
            format!("array<f64: {literal}>"),
            format!("dense<{literal}> : tensor<f64>"),
        ] {
            assert!(parse_attribute(&text).is_err(), "{text}");
        }
    }
    for literal in ["0.", "-0.", "1.0e+2", "1.0E-2", "0x8000000000000000"] {
        let attribute = parse_attribute(&format!("{literal} : f64")).unwrap();
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
}

#[test]
fn emitted_float_scales_and_scalars_have_lexically_valid_decimal_points() {
    for value in [0.0_f64, -0.0, 1.0, 2.0, 1e100, f64::MIN_POSITIVE, f64::MAX] {
        let text = print_attribute(&Attribute::float_bits(value.to_bits())).unwrap();
        assert!(text.split(':').next().unwrap().contains('.'), "{text}");
        assert_eq!(
            parse_attribute(&text).unwrap(),
            Attribute::float_bits(value.to_bits())
        );
    }
    let quantized = parse_type("tensor<!quant.uniform<i8:f32, 2.0>>").unwrap();
    assert!(print_type(&quantized).unwrap().contains("2.0e0"));
    let tensor: Type = TensorType::scalar(ElementType::F32).into();
    let attribute = Attribute::TypedString {
        value: b"metadata".to_vec(),
        ty: Box::new(tensor),
    };
    assert_eq!(
        parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
        attribute
    );
}

#[test]
fn bare_identifiers_and_symbol_names_match_pinned_lexer() {
    for text in [
        "@$invalid",
        "@invalid-name",
        "{$invalid = unit}",
        "{invalid-name = unit}",
    ] {
        assert!(parse_attribute(text).is_err(), "{text}");
    }
    for text in ["@valid$._1", "@\"quoted-name\"", "{valid$._1 = unit}"] {
        let attribute = parse_attribute(text).unwrap();
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
}
