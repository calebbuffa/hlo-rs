use stablehlo::{
    Attribute, Computation, ElementType, Instruction, Operation, StableHloOp, TensorType, Type,
    ValueId,
    text::{parse_attribute, print_attribute},
};

#[test]
fn all_reference_enum_categories_preserve_their_identity() {
    // StablehloEnums.td; custom_call_api_version is an i32 integer attribute,
    // not a dialect EnumAttr.
    for (name, values) in [
        ("precision", &["DEFAULT", "HIGH", "HIGHEST"][..]),
        (
            "result_accuracy_mode",
            &["DEFAULT", "HIGHEST", "TOLERANCE"][..],
        ),
        ("fft_type", &["FFT", "IFFT", "RFFT", "IRFFT"][..]),
        (
            "comparison_direction",
            &["EQ", "NE", "GE", "GT", "LE", "LT"][..],
        ),
        (
            "comparison_type",
            &[
                "NOTYPE",
                "FLOAT",
                "TOTALORDER",
                "SIGNED",
                "UNSIGNED",
                "WEAKORDER",
            ][..],
        ),
        (
            "transpose",
            &["TRANSPOSE_INVALID", "NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"][..],
        ),
        ("rng_distribution", &["UNIFORM", "NORMAL"][..]),
        ("rng_algorithm", &["DEFAULT", "THREE_FRY", "PHILOX"][..]),
    ] {
        for value in values {
            let input = if name == "result_accuracy_mode" {
                format!("#stablehlo.{name}<{value}>")
            } else {
                format!("#stablehlo<{name} {value}>")
            };
            let attribute = parse_attribute(&input).unwrap();
            if name == "result_accuracy_mode" {
                assert_eq!(
                    parse_attribute(&format!("#stablehlo<{name}<{value}>>")).unwrap(),
                    attribute
                );
            }
            assert_eq!(
                attribute,
                Attribute::Enum {
                    dialect: "stablehlo".into(),
                    name: name.into(),
                    value: (*value).into(),
                }
            );
            assert_eq!(print_attribute(&attribute).unwrap(), input);
        }
    }
    assert_ne!(
        parse_attribute("#stablehlo<precision DEFAULT>").unwrap(),
        parse_attribute("#stablehlo<rng_algorithm DEFAULT>").unwrap()
    );
}

#[test]
fn invalid_enum_category_and_value_combinations_are_rejected() {
    for input in [
        "#stablehlo<precision PHILOX>",
        "#stablehlo<rng_algorithm HIGH>",
        "#stablehlo<comparison_type INVALID>",
        "#stablehlo<fft_type <FFT>>",
        "#stablehlo<precision HIGH garbage>",
        "#stablehlo.precision DEFAULT",
        "#stablehlo.precision<DEFAULT>",
        "#stablehlo.result_accuracy_mode <DEFAULT>",
        "#stablehlo <precision DEFAULT>",
        "#stablehlo.channel_handle <handle = 1, type = 2>",
        "#stablehlo.precision PHILOX",
        "#stablehlo.rng_algorithm HIGH",
        "#stablehlo.result_accuracy_mode DEFAULT",
        "#stablehlo.comparison_type INVALID",
        "#stablehlo.fft_type <FFT>",
        "#stablehlo.precision HIGH garbage",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
    assert!(
        print_attribute(&Attribute::Enum {
            dialect: "stablehlo".into(),
            name: "rng_algorithm".into(),
            value: "HIGH".into(),
        })
        .is_err()
    );
}

fn compare(direction: Attribute) -> Result<Computation, stablehlo::BuildError> {
    let input = Type::from(TensorType::scalar(ElementType::F32));
    Computation::try_new(
        "compare",
        vec![
            Instruction::new(
                Operation::parameter(0, "lhs"),
                vec![],
                vec![],
                ValueId(0),
                input.clone(),
            ),
            Instruction::new(
                Operation::parameter(1, "rhs"),
                vec![],
                vec![],
                ValueId(1),
                input,
            ),
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Compare,
                    [("comparison_direction".into(), direction)].into(),
                ),
                vec![ValueId(0), ValueId(1)],
                vec![],
                ValueId(2),
                Type::from(TensorType::scalar(ElementType::I1)),
            ),
        ],
        ValueId(2),
    )
}

#[test]
fn operation_verification_requires_the_enum_category_not_just_the_value() {
    compare(parse_attribute("#stablehlo<comparison_direction EQ>").unwrap()).unwrap();
    assert!(
        compare(Attribute::Enum {
            dialect: "stablehlo".into(),
            name: "precision".into(),
            value: "EQ".into(),
        })
        .is_err()
    );
}

#[test]
fn dense_builtin_arrays_are_not_confused_with_attribute_arrays() {
    for attribute in [
        Attribute::integers(vec![]),
        Attribute::integers(vec![-1, 0, i64::MAX]),
        Attribute::bools(vec![]),
        Attribute::bools(vec![false, true]),
        Attribute::floats_bits(vec![]),
        Attribute::floats_bits(vec![0, 1, 0x8000000000000000, 0x7ff8000000000001]),
    ] {
        let output = print_attribute(&attribute).unwrap();
        assert_eq!(parse_attribute(&output).unwrap(), attribute, "{output}");
    }
    assert_eq!(
        parse_attribute("array<i64: 1, 2>").unwrap(),
        Attribute::integers(vec![1, 2])
    );
    assert_eq!(
        parse_attribute("array<f64: 1.0, -0.0>").unwrap(),
        Attribute::floats_bits(vec![1.0_f64.to_bits(), (-0.0_f64).to_bits()])
    );
    assert!(matches!(
        parse_attribute("[1, 2]").unwrap(),
        Attribute::Array(_)
    ));
    for input in [
        "array<i64:>",
        "array<i1: 2>",
        "array<i32: 4294967296>",
        "array<i64: 1,>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}
