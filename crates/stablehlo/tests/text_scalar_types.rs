use stablehlo::{
    Attribute, Attributes, ElementType, ScalarValue, StableHloOp, TensorType, Type,
    text::{parse_attribute, parse_type, print_attribute},
};

mod common;
use common::computation;

fn round_trip(input: &str) -> Attribute {
    let attribute = parse_attribute(input).unwrap_or_else(|error| panic!("{input}: {error}"));
    let output = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&output).unwrap(), attribute, "{output}");
    attribute
}

#[test]
fn integer_scalars_preserve_signedness_width_and_index_type() {
    for spelling in [
        "i2", "i4", "i8", "i16", "i32", "i64", "si2", "si4", "si8", "si16", "si32", "si64", "ui2",
        "ui4", "ui8", "ui16", "ui32", "ui64", "index",
    ] {
        let Type::Element(ty) = parse_type(spelling).unwrap() else {
            panic!()
        };
        assert_eq!(
            round_trip(&format!("1 : {spelling}")),
            Attribute::Integer { ty, value: 1 },
        );
    }
    for (input, expected) in [
        ("255 : i8", -1),
        ("255 : ui8", 255),
        ("-128 : si8", -128),
        ("0xffffffff : i32", -1),
        ("18446744073709551615 : ui64", u64::MAX as i128),
        ("-9223372036854775808 : index", i64::MIN as i128),
    ] {
        let Attribute::Integer { value, .. } = round_trip(input) else {
            panic!()
        };
        assert_eq!(value, expected);
    }
    assert_eq!(round_trip("1 : i1"), Attribute::Bool(true));
    assert_eq!(round_trip("-1 : i1"), Attribute::Bool(true));
    assert_eq!(round_trip("0 : i1"), Attribute::Bool(false));
    assert_ne!(round_trip("1 : i32"), round_trip("1 : i64"));
    for input in [
        "128 : si8",
        "256 : i8",
        "-129 : i8",
        "-1 : ui8",
        "-0 : ui8",
        "9223372036854775808 : index",
        "1 : tensor<i32>",
        "1 : complex<f32>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn all_modeled_float_formats_preserve_scalar_type_and_exact_bits() {
    for spelling in [
        "f4E2M1FN",
        "f6E2M3FN",
        "f6E3M2FN",
        "f16",
        "bf16",
        "f32",
        "f64",
        "tf32",
        "f8E3M4",
        "f8E4M3",
        "f8E4M3FN",
        "f8E4M3FNUZ",
        "f8E4M3B11FNUZ",
        "f8E5M2",
        "f8E5M2FNUZ",
        "f8E8M0FNU",
    ] {
        let Type::Element(ty) = parse_type(spelling).unwrap() else {
            panic!()
        };
        let Attribute::Float { ty: parsed, bits } = round_trip(&format!("1.0 : {spelling}")) else {
            panic!()
        };
        assert_eq!(parsed, ty);
        let Attribute::DenseElements { values, .. } =
            parse_attribute(&format!("dense<1.0> : tensor<{spelling}>")).unwrap()
        else {
            panic!()
        };
        assert_eq!(values, [ScalarValue::FloatBits(u128::from(bits))]);
        let width = ty.bit_width().unwrap();
        round_trip(&format!("0x{:X} : {spelling}", (1_u128 << width) - 1));
        assert!(parse_attribute(&format!("0x{:X} : {spelling}", 1_u128 << width)).is_err());
    }
    assert_eq!(
        round_trip("1.00000005960464477539062500000001 : f32"),
        Attribute::f32_bits(0x3f800001),
    );
    assert_ne!(round_trip("1.0 : f16"), round_trip("1.0 : bf16"));
}

#[test]
fn constructed_scalars_are_verified_before_printing_or_operation_use() {
    for attribute in [
        Attribute::Integer {
            ty: ElementType::I8,
            value: 128,
        },
        Attribute::Integer {
            ty: ElementType::Ui8,
            value: -1,
        },
        Attribute::Integer {
            ty: ElementType::F32,
            value: 1,
        },
        Attribute::Float {
            ty: ElementType::I32,
            bits: 0,
        },
        Attribute::Float {
            ty: ElementType::F16,
            bits: 0x10000,
        },
    ] {
        assert!(attribute.verify().is_err(), "{attribute:?}");
        assert!(print_attribute(&attribute).is_err(), "{attribute:?}");
    }
}

#[test]
fn operation_integer_attributes_enforce_ods_width_not_only_value_range() {
    let tensor: Type = TensorType::scalar(ElementType::F32).into();
    let tuple = Type::Tuple(vec![tensor.clone()]);
    computation(
        StableHloOp::GetTupleElement,
        vec![tuple.clone()],
        vec![tensor.clone()],
        [("index".into(), round_trip("0 : i32"))].into(),
    )
    .unwrap();
    assert!(
        computation(
            StableHloOp::GetTupleElement,
            vec![tuple],
            vec![tensor.clone()],
            [("index".into(), round_trip("0 : i64"))].into(),
        )
        .is_err()
    );
    let attrs = |exponent: &str, mantissa: &str| -> Attributes {
        [
            ("exponent_bits".into(), round_trip(exponent)),
            ("mantissa_bits".into(), round_trip(mantissa)),
        ]
        .into()
    };
    computation(
        StableHloOp::ReducePrecision,
        vec![tensor.clone()],
        vec![tensor.clone()],
        attrs("8 : i32", "10 : i32"),
    )
    .unwrap();
    for attributes in [attrs("8 : i64", "10 : i32"), attrs("8 : i32", "10 : i64")] {
        assert!(
            computation(
                StableHloOp::ReducePrecision,
                vec![tensor.clone()],
                vec![tensor.clone()],
                attributes
            )
            .is_err()
        );
    }
    for (version, valid) in [("1 : i32", true), ("1 : i64", false), ("1 : ui32", false)] {
        let attributes = [
            (
                "call_target_name".into(),
                Attribute::String("target".into()),
            ),
            ("api_version".into(), round_trip(version)),
            ("backend_config".into(), Attribute::Bytes(vec![0xff])),
        ]
        .into();
        assert_eq!(
            computation(StableHloOp::CustomCall, vec![], vec![], attributes).is_ok(),
            valid
        );
    }
}
