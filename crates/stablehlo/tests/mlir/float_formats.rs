use stablehlo::mlir::assembly::{parse_float_bits, print_float_bits};
use stablehlo::types::ElementType;

fn bits(value: &str, ty: ElementType) -> u64 {
    parse_float_bits(value, ty).unwrap_or_else(|error| panic!("{ty:?} {value}: {error}"))
}

fn decimal(value: f64) -> String {
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
    use ElementType::*;
    for (ty, exponent, mantissa, bias, maximum, fnuz) in [
        (F4E2M1Fn, 2, 1, 1, 7, false),
        (F6E2M3Fn, 2, 3, 1, 31, false),
        (F6E3M2Fn, 3, 2, 3, 31, false),
        (F8E3M4, 3, 4, 3, 111, false),
        (F8E4M3, 4, 3, 7, 119, false),
        (F8E4M3Fn, 4, 3, 7, 126, false),
        (F8E4M3FnUz, 4, 3, 8, 127, true),
        (F8E4M3B11FnUz, 4, 3, 11, 127, true),
        (F8E5M2, 5, 2, 15, 123, false),
        (F8E5M2FnUz, 5, 2, 16, 127, true),
    ] {
        let sign = 1 << (exponent + mantissa);
        for code in 0..=maximum {
            let value = finite_value(code, exponent, mantissa, bias);
            assert_eq!(bits(&decimal(value), ty), code, "{ty:?} {code}");
            let expected = if fnuz && code == 0 { 0 } else { sign | code };
            assert_eq!(
                bits(&decimal(-value), ty),
                expected,
                "{ty:?} negative {code}"
            );
        }
    }
}

#[test]
fn adjacent_low_precision_midpoints_round_directly_ties_to_even() {
    use ElementType::*;
    for (ty, exponent, mantissa, bias, maximum) in [
        (F4E2M1Fn, 2, 1, 1, 7),
        (F6E2M3Fn, 2, 3, 1, 31),
        (F6E3M2Fn, 3, 2, 3, 31),
        (F8E3M4, 3, 4, 3, 111),
        (F8E4M3Fn, 4, 3, 7, 126),
        (F8E4M3FnUz, 4, 3, 8, 127),
        (F8E5M2FnUz, 5, 2, 16, 127),
    ] {
        for code in 0..maximum {
            let low = finite_value(code, exponent, mantissa, bias);
            let high = finite_value(code + 1, exponent, mantissa, bias);
            let midpoint = (low + high) / 2.0;
            assert_eq!(
                bits(&decimal(midpoint), ty),
                if code % 2 == 0 { code } else { code + 1 },
                "{ty:?} midpoint after {code}"
            );
            assert_eq!(bits(&decimal(midpoint - (high - low) / 16.0), ty), code);
            assert_eq!(bits(&decimal(midpoint + (high - low) / 16.0), ty), code + 1);
        }
    }
}

#[test]
fn finite_only_overflow_saturates_and_fnuz_overflow_is_nan() {
    use ElementType::*;
    for (ty, maximum, sign) in [(F4E2M1Fn, 7, 8), (F6E2M3Fn, 31, 32), (F6E3M2Fn, 31, 32)] {
        assert_eq!(bits("1.0e1000", ty), maximum);
        assert_eq!(bits("-1.0e1000", ty), maximum | sign);
    }
    for ty in [F8E4M3FnUz, F8E4M3B11FnUz, F8E5M2FnUz] {
        assert_eq!(bits("1.0e1000", ty), 128);
        assert_eq!(bits("-1.0e1000", ty), 128);
        assert_eq!(bits("-0.0", ty), 0);
        assert_eq!(bits("-1.0e-1000", ty), 0);
    }
}

#[test]
fn half_brain_and_tensor_float_use_exact_target_precision() {
    use ElementType::*;
    for (ty, value, expected) in [
        (F16, "1.0", 0x3c00),
        (Bf16, "1.0", 0x3f80),
        (Tf32, "1.0", 0x1fc00),
        (F16, "1.00048828125", 0x3c00),
        (F16, "1.00048828125000000000001", 0x3c01),
        (Bf16, "1.00390625", 0x3f80),
        (Bf16, "1.00390625000000000000001", 0x3f81),
        (F16, "65520.0", 0x7c00),
        (F16, "-0.0", 0x8000),
        (F32, "1.000000059604644775390625", 0x3f800000),
        (F32, "1.00000005960464477539062500000001", 0x3f800001),
    ] {
        assert_eq!(bits(value, ty), expected, "{ty:?}: {value}");
    }
    assert!(parse_float_bits("0x80000", Tf32).is_err());
}

#[test]
fn exponent_only_format_has_no_sign_or_zero_and_rounds_power_midpoints_up() {
    let ty = ElementType::F8E8M0FnUz;
    assert_eq!(bits("0.0", ty), 0);
    assert_eq!(bits("1.0e-1000", ty), 0);
    assert_eq!(bits("1.0e1000", ty), 255);
    for exponent in -127..=127 {
        let value = 2.0_f64.powi(exponent);
        assert_eq!(bits(&decimal(value), ty), (exponent + 127) as u64);
        if exponent < 127 {
            assert_eq!(bits(&decimal(1.5 * value), ty), (exponent + 128) as u64);
            assert_eq!(bits(&decimal(1.375 * value), ty), (exponent + 127) as u64);
        }
    }
    assert!(parse_float_bits("-0.0", ty).is_err());
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
        assert_eq!(bits(value, ty), expected, "{value}");
    }
}

#[test]
fn all_low_precision_raw_payloads_roundtrip_without_nan_canonicalization() {
    use ElementType::*;
    for ty in [
        F4E2M1Fn,
        F6E2M3Fn,
        F6E3M2Fn,
        F8E3M4,
        F8E4M3,
        F8E4M3Fn,
        F8E4M3FnUz,
        F8E4M3B11FnUz,
        F8E5M2,
        F8E5M2FnUz,
        F8E8M0FnUz,
    ] {
        for payload in 0..(1 << ty.bit_width().unwrap()) {
            let text = print_float_bits(payload, ty).unwrap();
            assert_eq!(bits(&text, ty), payload);
        }
    }
    for (ty, payload) in [
        (F16, 0x7d23),
        (Bf16, 0x7f81),
        (F32, 0x7f800001),
        (F64, 0x7ff0000000000001),
    ] {
        assert_eq!(bits(&print_float_bits(payload, ty).unwrap(), ty), payload);
    }
}

#[test]
fn invalid_literals_and_nonfloat_types_are_not_coerced() {
    for literal in [
        "", "1", "+1.0", ".5", "nan", "inf", "0x", "0xG", "1.0x", "1.0e", "1.0e+", "1.0 ", "1.0\n",
    ] {
        assert!(
            parse_float_bits(literal, ElementType::F32).is_err(),
            "{literal:?}"
        );
    }
    assert!(parse_float_bits("1.0", ElementType::I32).is_err());
    assert!(print_float_bits(1, ElementType::C64).is_err());
    assert!(print_float_bits(256, ElementType::F8E3M4).is_err());
}
