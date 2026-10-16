use xlahlo::{
    Instruction, Module, Opcode, PrimitiveType, Shape,
    proto::xla_data::{
        ResultAccuracy,
        result_accuracy::{Mode, Specs, Tolerance},
    },
    text,
};

fn unary(opcode: Opcode, accuracy: ResultAccuracy) -> xlahlo::Result<Instruction> {
    Instruction::new(
        "u",
        opcode,
        Shape::array(PrimitiveType::F32, vec![2]).unwrap(),
    )
    .with_fields(|wire| wire.result_accuracy = Some(accuracy).into())
}

#[test]
fn all_eligible_unary_opcodes_retain_accuracy_modes_and_tolerances() {
    let expected = [
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atanh",
        "cbrt",
        "cosine",
        "cosh",
        "erf",
        "exponential",
        "exponential-minus-one",
        "exp2",
        "log",
        "log-plus-one",
        "log2",
        "logistic",
        "rsqrt",
        "sine",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ];
    let mut actual = Opcode::ALL
        .iter()
        .copied()
        .filter(|opcode| opcode.supports_result_accuracy())
        .map(Opcode::name)
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    assert_eq!(actual, expected);
    for &opcode in Opcode::ALL {
        if !opcode.supports_result_accuracy() {
            continue;
        }
        for accuracy in [
            "{mode=default}",
            "{mode=highest}",
            "{tolerance={}}",
            "{tolerance={atol=1.25,rtol=0.001,ulps=3}}",
            "{tolerance={atol=1e30,rtol=-0.0,ulps=3}}",
        ] {
            let source = format!(
                "HloModule m ENTRY main {{ p = f32[2] parameter(0) ROOT u = f32[2] {opcode}(p), result_accuracy={accuracy} }}"
            );
            let module =
                text::parse_module(&source).unwrap_or_else(|error| panic!("{source}: {error}"));
            module.verify(Default::default()).unwrap();
            let printed = text::print_module(&module).unwrap();
            let reparsed = text::parse_module(&printed).unwrap();
            assert_eq!(
                module.as_proto().computations[0].instructions[1].result_accuracy,
                reparsed.as_proto().computations[0].instructions[1].result_accuracy
            );
        }
    }
}

#[test]
fn accuracy_import_checks_modes_nonnegative_and_nan_but_allows_infinity() {
    for mode in [Mode::DEFAULT as i32, Mode::HIGHEST as i32, 999] {
        let result = unary(
            Opcode::Exp,
            ResultAccuracy {
                specs: Some(Specs::Mode(protobuf::EnumOrUnknown::from_i32(mode))),
                ..Default::default()
            },
        );
        assert_eq!(result.is_ok(), mode != 999);
    }
    for (atol, rtol, ulps, valid) in [
        (0.0, 0.0, 0, true),
        (-0.0, 0.0, 1, true),
        (-0.1, 0.0, 0, false),
        (0.0, -0.1, 0, false),
        (0.0, 0.0, -1, false),
        (f64::NAN, 0.0, 0, false),
        (0.0, f64::NAN, 0, false),
        (f64::INFINITY, 0.0, i64::MAX, true),
    ] {
        let result = unary(
            Opcode::Exp,
            ResultAccuracy {
                specs: Some(Specs::Tolerance(Tolerance {
                    atol,
                    rtol,
                    ulps,
                    ..Default::default()
                })),
                ..Default::default()
            },
        );
        assert_eq!(result.is_ok(), valid);
    }
    assert!(unary(Opcode::Exp, ResultAccuracy::default()).is_ok());
}

#[test]
fn accuracy_text_rejects_wrong_opcodes_invalid_specs_and_lossy_integer_conversions() {
    for accuracy in [
        "{}",
        "{mode=high}",
        "{mode=highest,tolerance={}}",
        "{tolerance={atol=-1}}",
        "{tolerance={rtol=nan}}",
        "{tolerance={ulps=-1}}",
        "{tolerance={ulps=inf}}",
        "{tolerance={ulps=9223372036854775808}}",
        "{tolerance={unknown=1}}",
        "{tolerance={atol=18446744073709551615}}",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = f32[2] parameter(0) ROOT u = f32[2] exponential(p), result_accuracy={accuracy} }}"
        );
        assert!(text::parse_module(&source).is_err(), "{accuracy}");
    }
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT u = f32[2] abs(p), result_accuracy={mode=highest} }";
    assert!(text::parse_module(source).is_err());
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT u = f32[2] exponential(p), result_accuracy={tolerance={ulps=-0.75,atol=1,atol=2}} }";
    let module = text::parse_module(source).unwrap();
    let accuracy = module.as_proto().computations[0].instructions[1]
        .result_accuracy
        .as_ref()
        .unwrap();
    assert_eq!(accuracy.tolerance().ulps, 0);
    assert_eq!(accuracy.tolerance().atol, 2.0);
    let mut wire = module.to_proto().unwrap();
    wire.computations[0].instructions[1]
        .result_accuracy
        .as_mut()
        .unwrap()
        .mut_tolerance()
        .ulps = i64::MAX;
    assert!(text::print_module(&Module::from_proto(wire).unwrap()).is_err());
}
