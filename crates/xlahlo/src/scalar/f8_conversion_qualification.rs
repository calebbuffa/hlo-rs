//! Regression tests for native F8 conversion semantics.
//!
//! Pinned vectors mirror independent XLA tests named in each test. Exhaustive
//! raw encoding closure is native integration evidence only, not an independent
//! conversion oracle.
use super::{convert, encoding};
use crate::types::ElementType::{self, F8E4M3, F8E4M3FN, F8E5M2, F32};

#[test]
fn f8_conversion_round_trips_all_raw_encodings() {
    for element in [F8E4M3FN, F8E5M2] {
        for raw in 0u128..=u8::MAX.into() {
            let decoded = convert(element, raw, F32)
                .unwrap_or_else(|error| panic!("decode {element:?} 0x{raw:02x}: {error}"))
                .1;
            let encoded = convert(F32, decoded, element)
                .unwrap_or_else(|error| panic!("encode {element:?} 0x{raw:02x}: {error}"))
                .1;
            if encoding::is_nan(element, raw) {
                assert!(
                    encoding::is_nan(element, encoded),
                    "{element:?} 0x{raw:02x} -> 0x{encoded:02x}"
                );
                assert_eq!(encoded >> 7, raw >> 7, "NaN sign {element:?} 0x{raw:02x}");
            } else {
                assert_eq!(encoded, raw, "{element:?} 0x{raw:02x}");
            }
        }
    }
}

#[test]
fn f32_to_f8e4m3fn_matches_pinned_xla_vectors() {
    // Independent reference: extern/xla/xla/codegen/intrinsic/fptrunc_test.cc,
    // F32ToF8e4m3fn and F32ToF8e4m3fn_Vector8. Keep this table to values those
    // tests actually assert; they do not pin negative-zero, infinity, or NaN.
    let vectors = [
        (0.0f32, 0x00),
        (-2.0, 0xc0),
        (0.5, 0x30),
        (-0.5, 0xb0),
        (0.125, 0x20),
        (242.0, 0x77),
        (367.9, 0x7b),
        (368.0, 0x7c),
        (368.1, 0x7c),
        (500.0, 0x7f),
        (f32::MAX, 0x7f),
    ];
    for (value, expected) in vectors {
        assert_eq!(
            convert(F32, value.to_bits().into(), F8E4M3FN).unwrap().1,
            expected
        );
    }
}

#[test]
fn f32_to_f8e5m2_matches_pinned_xla_vectors() {
    // Independent reference: extern/xla/xla/codegen/intrinsic/fptrunc_test.cc,
    // F32ToF8e5m2. The exact half-subnormal tie below is intentionally omitted;
    // the reference checks a value just below it instead.
    let vectors = [
        (0.0f32, 0x00),
        (-0.0, 0x80),
        (16.0, 0x4c),
        (-16.0, 0xcc),
        (0.5, 0x38),
        (-0.5, 0xb8),
        (1.8749, 0x3f),
        (1.8750, 0x40),
        (1.8751, 0x40),
        (2.0f32.powi(-16), 0x01),
        (2.0f32.powi(-15), 0x02),
        (2.0f32.powi(-17) - f32::EPSILON, 0x00),
        (57_344.0, 0x7b),
        (60_000.0, 0x7b),
        (62_000.0, 0x7c),
        (f32::INFINITY, 0x7c),
        (f32::NEG_INFINITY, 0xfc),
        (f32::NAN, 0x7e),
    ];
    for (value, expected) in vectors {
        assert_eq!(
            convert(F32, value.to_bits().into(), F8E5M2).unwrap().1,
            expected
        );
    }
}

#[test]
fn f64_to_f8e5m2_uses_direct_pinned_conversion_vectors() {
    // Independent reference: extern/xla/xla/codegen/intrinsic/fptrunc_test.cc,
    // F64ToF8e5m2. Inputs are converted directly from F64; do not stage through F32.
    let vectors = [
        (1.0f64, 0x3c),
        (1.5, 0x3e),
        (57_344.0, 0x7b),
        (60_000.0, 0x7b),
        (62_000.0, 0x7c),
    ];
    for (value, expected) in vectors {
        assert_eq!(
            convert(ElementType::F64, value.to_bits().into(), F8E5M2)
                .unwrap()
                .1,
            expected
        );
    }
}

#[test]
fn f8e4m3_and_f8e4m3fn_remain_distinct() {
    // Format properties/integration assertions, not independent conversion
    // oracle vectors: IEEE E4M3 reserves exponent 15 for Inf/NaN; E4M3FN uses
    // it for finite values except the all-ones mantissa NaN encoding.
    assert_eq!(
        convert(F8E4M3, 0x78, F32).unwrap().1,
        f32::INFINITY.to_bits().into()
    );
    assert_eq!(
        convert(F8E4M3FN, 0x78, F32).unwrap().1,
        256.0f32.to_bits().into()
    );
    assert!(encoding::is_nan(
        F8E4M3,
        convert(F8E4M3, 0x7c, F32).unwrap().1
    ));
    assert_eq!(
        convert(F8E4M3FN, 0x7c, F32).unwrap().1,
        384.0f32.to_bits().into()
    );
    assert_eq!(
        convert(F32, 256.0f32.to_bits().into(), F8E4M3).unwrap().1,
        0x78
    );
    assert_eq!(
        convert(F32, 256.0f32.to_bits().into(), F8E4M3FN).unwrap().1,
        0x78
    );
    assert_eq!(
        convert(F32, 384.0f32.to_bits().into(), F8E4M3).unwrap().1,
        0x78
    );
    assert_eq!(
        convert(F32, 384.0f32.to_bits().into(), F8E4M3FN).unwrap().1,
        0x7c
    );
}

#[test]
fn f64_f8_midpoint_neighbors_avoid_double_rounding() {
    // Exact mathematical RNE contract: around 1.875, adjacent E5M2 values are
    // 1.75 (odd encoding 0x3f) and 2.0 (even encoding 0x40). Adjacent F64
    // values around the midpoint must round to opposite sides; the exact
    // midpoint rounds to even. This is not an XLA pinned vector.
    let midpoint = 1.875f64;
    let below = f64::from_bits(midpoint.to_bits() - 1);
    let above = f64::from_bits(midpoint.to_bits() + 1);
    for (value, expected) in [(below, 0x3f), (midpoint, 0x40), (above, 0x40)] {
        assert_eq!(
            convert(ElementType::F64, value.to_bits().into(), F8E5M2)
                .unwrap()
                .1,
            expected
        );
    }

    // Both adjacent F64 values collapse to the F32 midpoint before the second
    // rounding, making the two-step conversion observably incorrect below.
    let staged = convert(F32, (below as f32).to_bits().into(), F8E5M2)
        .unwrap()
        .1;
    assert_eq!(staged, 0x40);
    assert_eq!(
        convert(ElementType::F64, below.to_bits().into(), F8E5M2)
            .unwrap()
            .1,
        0x3f
    );
}

#[test]
fn f64_f8e4m3fn_midpoint_neighbors_round_directly() {
    // Exact mathematical RNE boundary between 352 (0x7b) and 384 (0x7c).
    // The pinned XLA F32 fixture establishes the tie result; adjacent F64
    // values exercise direct conversion without F32 staging.
    let midpoint = 368.0f64;
    let below = f64::from_bits(midpoint.to_bits() - 1);
    let above = f64::from_bits(midpoint.to_bits() + 1);
    for (value, expected) in [(below, 0x7b), (midpoint, 0x7c), (above, 0x7c)] {
        assert_eq!(
            convert(ElementType::F64, value.to_bits().into(), F8E4M3FN)
                .unwrap()
                .1,
            expected
        );
    }
    let staged = convert(F32, (below as f32).to_bits().into(), F8E4M3FN)
        .unwrap()
        .1;
    assert_eq!(staged, 0x7c);
    assert_eq!(
        convert(ElementType::F64, below.to_bits().into(), F8E4M3FN)
            .unwrap()
            .1,
        0x7b
    );
}
