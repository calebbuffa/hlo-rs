use stablehlo::{
    Attribute, ElementType, ScalarValue,
    text::{parse_attribute, print_attribute},
};

fn bits(value: &str, ty: &str) -> u64 {
    let Attribute::DenseElements { values, .. } =
        parse_attribute(&format!("dense<{value}> : tensor<{ty}>"))
            .unwrap_or_else(|error| panic!("{ty} {value}: {error}"))
    else {
        panic!()
    };
    let [ScalarValue::FloatBits(bits)] = values.as_slice() else {
        panic!()
    };
    *bits as u64
}

fn decimal(value: f64) -> String {
    // Every low-precision test value and midpoint is exactly representable in
    // binary64. Fixed digits spell the exact rational, not a nearby decimal.
    format!("{value:.150}")
}

fn finite_value(code: u64, exponent_bits: u32, mantissa_bits: u32, bias: i32) -> f64 {
    let exponent = ((code >> mantissa_bits) & ((1 << exponent_bits) - 1)) as i32;
    let mantissa = code & ((1 << mantissa_bits) - 1);
    if exponent == 0 {
        mantissa as f64 * 2.0_f64.powi(1 - bias - mantissa_bits as i32)
    } else {
        (1.0 + mantissa as f64 / (1 << mantissa_bits) as f64) * 2.0_f64.powi(exponent - bias)
    }
}

#[test]
fn every_finite_low_precision_encoding_parses_from_exact_decimal() {
    for (ty, exponent, mantissa, bias, maximum, fnuz) in [
        ("f4E2M1FN", 2, 1, 1, 7, false),
        ("f6E2M3FN", 2, 3, 1, 31, false),
        ("f6E3M2FN", 3, 2, 3, 31, false),
        ("f8E3M4", 3, 4, 3, 111, false),
        ("f8E4M3", 4, 3, 7, 119, false),
        ("f8E4M3FN", 4, 3, 7, 126, false),
        ("f8E4M3FNUZ", 4, 3, 8, 127, true),
        ("f8E4M3B11FNUZ", 4, 3, 11, 127, true),
        ("f8E5M2", 5, 2, 15, 123, false),
        ("f8E5M2FNUZ", 5, 2, 16, 127, true),
    ] {
        let sign = 1 << (exponent + mantissa);
        for code in 0..=maximum {
            let value = finite_value(code, exponent, mantissa, bias);
            assert_eq!(bits(&decimal(value), ty), code, "{ty} {code}");
            let expected = if fnuz && code == 0 { 0 } else { sign | code };
            assert_eq!(bits(&decimal(-value), ty), expected, "{ty} negative {code}");
        }
    }
}

#[test]
fn adjacent_low_precision_midpoints_round_directly_ties_to_even() {
    for (ty, exponent, mantissa, bias, maximum) in [
        ("f4E2M1FN", 2, 1, 1, 7),
        ("f6E2M3FN", 2, 3, 1, 31),
        ("f6E3M2FN", 3, 2, 3, 31),
        ("f8E3M4", 3, 4, 3, 111),
        ("f8E4M3FN", 4, 3, 7, 126),
        ("f8E4M3FNUZ", 4, 3, 8, 127),
        ("f8E5M2FNUZ", 5, 2, 16, 127),
    ] {
        for code in 0..maximum {
            let low = finite_value(code, exponent, mantissa, bias);
            let high = finite_value(code + 1, exponent, mantissa, bias);
            let midpoint = (low + high) / 2.0;
            assert_eq!(
                bits(&decimal(midpoint), ty),
                if code % 2 == 0 { code } else { code + 1 },
                "{ty} midpoint after {code}"
            );
            assert_eq!(bits(&decimal(midpoint - (high - low) / 16.0), ty), code);
            assert_eq!(bits(&decimal(midpoint + (high - low) / 16.0), ty), code + 1);
        }
    }
}

#[test]
fn finite_only_overflow_saturates_and_fnuz_overflow_is_nan() {
    for (ty, maximum, sign) in [
        ("f4E2M1FN", 7, 8),
        ("f6E2M3FN", 31, 32),
        ("f6E3M2FN", 31, 32),
    ] {
        assert_eq!(bits("1.0e1000", ty), maximum);
        assert_eq!(bits("-1.0e1000", ty), maximum | sign);
    }
    for ty in ["f8E4M3FNUZ", "f8E4M3B11FNUZ", "f8E5M2FNUZ"] {
        assert_eq!(bits("1.0e1000", ty), 128);
        assert_eq!(bits("-1.0e1000", ty), 128);
        assert_eq!(bits("-0.0", ty), 0);
        assert_eq!(bits("-1.0e-1000", ty), 0);
    }
}

#[test]
fn half_brain_and_tensor_float_use_exact_target_precision() {
    for (ty, value, expected) in [
        ("f16", "1.0", 0x3c00),
        ("bf16", "1.0", 0x3f80),
        ("tf32", "1.0", 0x1fc00),
        ("f16", "1.00048828125", 0x3c00),
        ("f16", "1.00048828125000000000001", 0x3c01),
        ("bf16", "1.00390625", 0x3f80),
        ("bf16", "1.00390625000000000000001", 0x3f81),
        ("f16", "65520.0", 0x7c00),
        ("f16", "-0.0", 0x8000),
        ("f32", "1.000000059604644775390625", 0x3f800000),
        ("f32", "1.00000005960464477539062500000001", 0x3f800001),
    ] {
        assert_eq!(bits(value, ty), expected, "{ty}: {value}");
    }
    assert_eq!(ElementType::Tf32.bit_width(), Some(19));
    assert!(parse_attribute("dense<0x80000> : tensor<tf32>").is_err());
}

#[test]
fn exponent_only_format_has_no_sign_or_zero_and_rounds_power_midpoints_up() {
    assert_eq!(bits("0.0", "f8E8M0FNU"), 0);
    assert_eq!(bits("1.0e-1000", "f8E8M0FNU"), 0);
    assert_eq!(bits("1.0e1000", "f8E8M0FNU"), 255);
    for exponent in -127..=127 {
        let value = 2.0_f64.powi(exponent);
        assert_eq!(bits(&decimal(value), "f8E8M0FNU"), (exponent + 127) as u64);
        if exponent < 127 {
            assert_eq!(
                bits(&decimal(1.5 * value), "f8E8M0FNU"),
                (exponent + 128) as u64,
                "midpoint exponent {exponent}: {}",
                decimal(1.5 * value)
            );
            assert_eq!(
                bits(&decimal(1.375 * value), "f8E8M0FNU"),
                (exponent + 127) as u64
            );
        }
    }
    assert!(parse_attribute("dense<-0.0> : tensor<f8E8M0FNU>").is_err());
    for (value, expected) in [
        ("1.49999999999999999999999999999999999999999999999999", 127),
        ("1.50000000000000000000000000000000000000000000000000", 128),
        ("1.50000000000000000000000000000000000000000000000001", 128),
        (
            "1.0e99999999999999999999999999999999999999999999999999",
            255,
        ),
        ("1.0e-99999999999999999999999999999999999999999999999999", 0),
        ("0.0e99999999999999999999999999999999999999999999999999", 0),
    ] {
        assert_eq!(bits(value, "f8E8M0FNU"), expected, "{value}");
    }
}

#[test]
fn hex_round_trips_do_not_canonicalize_nan_payloads() {
    for (ty, bits) in [("f16", 0x7e23), ("f8E4M3FN", 0xff), ("f8E8M0FNU", 0xff)] {
        let attribute = parse_attribute(&format!("dense<0x{bits:X}> : tensor<{ty}>")).unwrap();
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
}
