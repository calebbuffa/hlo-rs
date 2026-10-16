use xlahlo::{Instruction, Literal, Module, Opcode, PrimitiveType as P, Shape, interchange, text};

fn scalar(ty: &str, value: &str) -> xlahlo::Result<Literal> {
    let module = text::parse_module(&format!(
        "HloModule m ENTRY main {{ ROOT c = {ty}[] constant({value}) }}"
    ))?;
    Literal::from_proto(
        module.as_proto().computations[0].instructions[0]
            .literal
            .as_ref()
            .unwrap()
            .clone(),
    )
}
fn roundtrip(ty: P, bits: &[u128]) {
    let shape = Shape::array(ty, vec![bits.len() as i64])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let literal = Literal::from_bits(shape.clone(), bits).unwrap();
    let mut module = Module::new("m");
    let main = module.add_computation("main").unwrap();
    let instruction = Instruction::new("c", Opcode::Constant, shape)
        .with_fields(|wire| wire.literal = Some(literal.as_proto().clone()).into())
        .unwrap();
    let root = module.add_instruction(main, instruction, &[], &[]).unwrap();
    module.set_root(main, root).unwrap();
    module.set_entry(main).unwrap();
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    let after = parsed.as_proto().computations[0].instructions[0]
        .literal
        .as_ref()
        .unwrap();
    assert_eq!(
        interchange::encode(literal.as_proto()).unwrap(),
        interchange::encode(after).unwrap(),
        "{printed}"
    );
}

#[test]
fn integer_tokens_follow_pinned_signed_bit_interpretation_and_predicate_conversion() {
    for (ty, value, bits) in [
        ("s64", "18446744073709551615", u128::from(u64::MAX)),
        ("u64", "-1", u128::from(u64::MAX)),
        ("u64", "18446744073709551615", u128::from(u64::MAX)),
        ("s8", "18446744073709551615", 255),
        ("pred", "-27", 1),
        ("pred", "0", 0),
        ("pred", "true", 1),
    ] {
        assert_eq!(scalar(ty, value).unwrap().bits().unwrap(), [bits]);
    }
    for (ty, value) in [
        ("s8", "128"),
        ("u8", "-1"),
        ("u64", "18446744073709551616"),
        ("s64", "-9223372036854775809"),
        ("s32", "+1"),
        ("s32", "true"),
        ("pred", "1.5"),
    ] {
        assert!(scalar(ty, value).is_err(), "{ty} {value}");
    }
    assert_eq!(
        scalar("f64", "18446744073709551615")
            .unwrap()
            .bits()
            .unwrap(),
        [u128::from((-1.0_f64).to_bits())]
    );
}

#[test]
fn binary32_and_binary64_preserve_signed_zero_subnormals_infinity_and_nan_payloads() {
    roundtrip(
        P::F32,
        &[
            0,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x7f80_0001,
            0xffc0_0007,
        ],
    );
    roundtrip(
        P::F64,
        &[
            0,
            0x8000_0000_0000_0000,
            1,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            0xfff8_0000_0000_0007,
        ],
    );
    for value in [
        "1e400",
        "1e40",
        "nan(0x800000)",
        "nan(0x0)",
        "nan(0x10000000000000)",
        "+inf",
        "NaN",
        "nan( 0x1)",
        "nan(0x1 )",
        "nan(0x+1)",
    ] {
        assert!(scalar("f32", value).is_err(), "{value}");
    }
    assert_eq!(
        scalar("f32", "nan(0x400001)").unwrap().bits().unwrap(),
        [0x7fc0_0001]
    );
    assert_eq!(
        scalar("f32", "3.4028235e38").unwrap().bits().unwrap(),
        [u128::from(f32::MAX.to_bits())]
    );
    assert!(scalar("c64", "(3.4028235e38,0)").is_err());
}

#[test]
fn complex_literals_use_component_range_checks_and_the_pinned_nan_handler() {
    roundtrip(
        P::C64,
        &[
            u128::from(1_f32.to_bits()) | (u128::from((-2_f32).to_bits()) << 32),
            0x8000_0000,
            0x7fc0_0001_7fc0_0001,
        ],
    );
    roundtrip(
        P::C128,
        &[
            u128::from(1_f64.to_bits()) | (u128::from((-2_f64).to_bits()) << 64),
            u128::from(1_f64.to_bits()) | (u128::from(0x7ff0_0000_0000_0001_u64) << 64),
        ],
    );
    roundtrip(
        P::C64,
        &[
            u128::from(1_f32.to_bits()) | (u128::from(0xffc0_0017_u32) << 32),
            0x7f80_0001_7f80_0001,
        ],
    );
    roundtrip(
        P::C64,
        &[u128::from(f32::MAX.to_bits()) | (u128::from(f32::MIN.to_bits()) << 32)],
    );
    assert!(scalar("c64", "(1e40,0)").is_err());
    assert!(scalar("c64", "(0,1e40)").is_err());
    assert!(scalar("c64", "2").is_err());
    let literal = scalar("c64", "(nan(0x400007),3)").unwrap();
    assert_eq!(literal.bits().unwrap(), [0x7fc0_0007_7fc0_0007]);
    let literal = scalar("c128", "(-nan(0x8000000000007),5)").unwrap();
    assert_eq!(
        literal.bits().unwrap(),
        [u128::from(0xfff8_0000_0000_0007_u64) | (u128::from(0xfff8_0000_0000_0007_u64) << 64)]
    );
}

#[test]
fn every_low_precision_encoding_roundtrips_including_all_payloads_and_signed_zeros() {
    for ty in [
        P::F16,
        P::BF16,
        P::F8E3M4,
        P::F8E4M3,
        P::F8E5M2,
        P::F8E4M3FN,
        P::F8E4M3FNUZ,
        P::F8E4M3B11FNUZ,
        P::F8E5M2FNUZ,
        P::F8E8M0FNU,
        P::F4E2M1FN,
        P::F6E2M3FN,
        P::F6E3M2FN,
    ] {
        let bits = (0..1_u128 << ty.bit_width().unwrap()).collect::<Vec<_>>();
        roundtrip(ty, &bits);
    }
}

#[test]
fn low_precision_rounding_ranges_and_special_values_follow_native_constructors() {
    for (ty, value, bits) in [
        ("f16", "65519", 0x7bff),
        ("f16", "1.00048828125", 0x3c00),
        ("f16", "1.00146484375", 0x3c02),
        ("f16", "-0.0", 0x8000),
        ("f16", "nan(0x201)", 0x7e01),
        ("bf16", "nan(0x41)", 0x7fc1),
        ("bf16", "1.00390625", 0x3f80),
        ("bf16", "1.01171875", 0x3f82),
        ("f8e3m4", "inf", 0x70),
        ("f8e4m3", "-inf", 0xf8),
        ("f8e5m2", "nan(0x1)", 0x7d),
        ("f8e4m3fn", "inf", 0x7f),
        ("f8e4m3fn", "-nan", 0xff),
        ("f8e4m3fn", "449", 0x7e),
        ("f8e4m3fnuz", "-0.0", 0),
        ("f8e5m2fnuz", "nan", 0x80),
        ("f8e4m3b11fnuz", "-inf", 0x80),
        ("f4e2m1fn", "1.25", 2),
        ("f4e2m1fn", "1.75", 4),
        ("f4e2m1fn", "1e300", 7),
        ("f4e2m1fn", "inf", 7),
        ("f4e2m1fn", "-inf", 15),
        ("f4e2m1fn", "nan", 8),
        ("f4e2m1fn", "-nan", 0),
        ("f6e2m3fn", "1e300", 31),
        ("f6e3m2fn", "-1e300", 63),
        ("f8e8m0fnu", "1.5", 128),
        ("f8e8m0fnu", "1e-300", 0),
        ("f8e8m0fnu", "inf", 255),
    ] {
        assert_eq!(
            scalar(ty, value).unwrap().bits().unwrap(),
            [bits],
            "{ty} {value}"
        );
    }

    for (ty, value) in [
        ("f16", "65520"),
        ("bf16", "3.4e38"),
        ("f16", "nan(0x400)"),
        ("bf16", "nan(0x80)"),
        ("f8e5m2", "nan(0x4)"),
        ("f8e4m3fn", "nan(0x4)"),
        ("f4e2m1fn", "nan(0x1)"),
        ("f8e4m3fn", "465"),
        ("f8e5m2fnuz", "1e30"),
        ("f8e8m0fnu", "0"),
        ("f8e8m0fnu", "-1"),
        ("f8e8m0fnu", "-0.0"),
    ] {
        assert!(scalar(ty, value).is_err(), "{ty} {value}");
    }
    // Eigen's double constructors narrow through float, unlike float8 conversion.
    assert_eq!(
        scalar("f16", "1.0004882812500002").unwrap().bits().unwrap(),
        [0x3c00]
    );
    assert_eq!(
        scalar("bf16", "1.0039062500000002")
            .unwrap()
            .bits()
            .unwrap(),
        [0x3f80]
    );
    assert_eq!(
        scalar("f8e4m3", "1.0625000000000002")
            .unwrap()
            .bits()
            .unwrap(),
        [0x39]
    );
}

#[test]
fn float_attributes_use_double_tokens_with_strict_range_checks() {
    let source = |value: &str| {
        format!(
            "HloModule m ENTRY main {{ a = f32[2,3] parameter(0) v = f32[3] parameter(1) ROOT n = (f32[2,3],f32[3],f32[3]) batch-norm-training(a,v,v), epsilon={value}, feature_index=1 }}"
        )
    };
    for value in [
        "-0.0",
        "3.4028234663852886e38",
        "18446744073709551615",
        "nan(0x8000020000000)",
    ] {
        let module = text::parse_module(&source(value)).unwrap();
        let printed = text::print_module(&module).unwrap();
        let parsed = text::parse_module(&printed).unwrap();
        assert_eq!(
            module.as_proto().computations[0].instructions[2]
                .epsilon
                .to_bits(),
            parsed.as_proto().computations[0].instructions[2]
                .epsilon
                .to_bits()
        );
    }
    for value in ["inf", "-inf", "1e400", "3.4028235e38"] {
        assert!(text::parse_module(&source(value)).is_err(), "{value}");
    }
}
