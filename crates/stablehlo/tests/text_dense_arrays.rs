use stablehlo::{
    Attribute, ElementType, ScalarValue, Type,
    text::{parse_attribute, parse_type, print_attribute},
};

fn round_trip(text: &str) -> Attribute {
    let attribute = parse_attribute(text).unwrap_or_else(|error| panic!("{text}: {error}"));
    assert_eq!(
        parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
        attribute
    );
    attribute
}

#[test]
fn dense_integer_arrays_preserve_all_modeled_byte_aligned_types() {
    for spelling in [
        "i8", "i16", "i32", "i64", "si8", "si16", "si32", "si64", "ui8", "ui16", "ui32", "ui64",
    ] {
        let Type::Element(element) = parse_type(spelling).unwrap() else {
            panic!()
        };
        assert_eq!(
            round_trip(&format!("array<{spelling}>")),
            Attribute::DenseArray {
                element,
                values: vec![]
            }
        );
        assert_eq!(
            round_trip(&format!("array<{spelling}: 0, 1, 2>")),
            Attribute::DenseArray {
                element,
                values: vec![
                    ScalarValue::Integer(0),
                    ScalarValue::Integer(1),
                    ScalarValue::Integer(2)
                ]
            }
        );
    }
    for (text, expected) in [
        ("array<i8: 255>", -1),
        ("array<ui8: 255>", 255),
        ("array<ui8: -1>", 255),
        ("array<ui8: -128>", 128),
        ("array<si8: -128>", -128),
        ("array<ui64: 18446744073709551615>", u64::MAX as i128),
        ("array<ui64: -1>", u64::MAX as i128),
    ] {
        let Attribute::DenseArray { values, .. } = round_trip(text) else {
            panic!()
        };
        assert_eq!(values, [ScalarValue::Integer(expected)], "{text}");
    }
    assert_eq!(
        round_trip("array<i1: false, true>"),
        Attribute::bools(vec![false, true])
    );
}

#[test]
fn dense_float_arrays_cover_all_byte_aligned_formats_and_payload_bits() {
    for spelling in [
        "f16",
        "bf16",
        "f32",
        "f64",
        "f8E3M4",
        "f8E4M3",
        "f8E4M3FN",
        "f8E4M3FNUZ",
        "f8E4M3B11FNUZ",
        "f8E5M2",
        "f8E5M2FNUZ",
        "f8E8M0FNU",
    ] {
        let Type::Element(element) = parse_type(spelling).unwrap() else {
            panic!()
        };
        assert_eq!(
            round_trip(&format!("array<{spelling}>")),
            Attribute::DenseArray {
                element,
                values: vec![]
            }
        );
        let Attribute::Float { bits, .. } = parse_attribute(&format!("1.0 : {spelling}")).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            round_trip(&format!("array<{spelling}: 1.0, 0.0>")),
            Attribute::DenseArray {
                element,
                values: vec![
                    ScalarValue::FloatBits(u128::from(bits)),
                    ScalarValue::FloatBits(0)
                ]
            }
        );
        let mask = (1_u128 << element.bit_width().unwrap()) - 1;
        let Attribute::DenseArray { values, .. } =
            round_trip(&format!("array<{spelling}: 0x{mask:X}>"))
        else {
            panic!()
        };
        assert_eq!(values, [ScalarValue::FloatBits(mask)]);
    }
    let parsed = round_trip("array<f32: 1.00000005960464477539062500000001>");
    assert_eq!(
        parsed,
        Attribute::DenseArray {
            element: ElementType::F32,
            values: vec![ScalarValue::FloatBits(0x3f800001)]
        }
    );
}

#[test]
fn array_grammar_and_type_restrictions_match_pinned_mlir_parser() {
    for spelling in [
        "i2",
        "i4",
        "si2",
        "si4",
        "ui2",
        "ui4",
        "tf32",
        "f4E2M1FN",
        "f6E2M3FN",
        "f6E3M2FN",
        "index",
        "complex<f32>",
        "tensor<f32>",
    ] {
        assert!(
            parse_attribute(&format!("array<{spelling}>")).is_err(),
            "{spelling}"
        );
    }
    for text in [
        "array<i1: 0>",
        "array<i1: 1>",
        "array<i8: true>",
        "array<f32: 1>",
        "array<si8: 128>",
        "array<ui8: -129>",
        "array<ui8: -0>",
        "array<i8: 256>",
        "array<f16: 0x10000>",
        "array<i8: 0,>",
        "array<f8E8M0FNU: -0.0>",
        "array<i8:>",
    ] {
        assert!(parse_attribute(text).is_err(), "{text}");
    }
}

#[test]
fn programmatic_arrays_cannot_smuggle_wrong_element_kinds_or_widths() {
    for attribute in [
        Attribute::DenseArray {
            element: ElementType::I4,
            values: vec![],
        },
        Attribute::DenseArray {
            element: ElementType::Index,
            values: vec![],
        },
        Attribute::DenseArray {
            element: ElementType::F16,
            values: vec![ScalarValue::FloatBits(0x10000)],
        },
        Attribute::DenseArray {
            element: ElementType::I8,
            values: vec![ScalarValue::Integer(128)],
        },
        Attribute::DenseArray {
            element: ElementType::I1,
            values: vec![ScalarValue::Integer(1)],
        },
        Attribute::DenseArray {
            element: ElementType::I64,
            values: vec![ScalarValue::FloatBits(0)],
        },
    ] {
        assert!(attribute.verify().is_err(), "{attribute:?}");
        assert!(print_attribute(&attribute).is_err(), "{attribute:?}");
    }
}
