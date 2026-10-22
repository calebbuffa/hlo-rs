//! Compatibility checks for F4E2M1FN conversions.
//!
//! The MX specification leaves NaN conversion implementation-defined. The
//! NaN test below records the exact XLA-pinned `ml_dtypes` compatibility
//! behavior; it is not a format-level NaN contract.
use super::convert;
use crate::types::ElementType::{BF16, F4E2M1FN, F16, F32, F64};

#[test]
fn f4_nan_conversion_matches_pinned_ml_dtypes_snapshot() {
    // XLA pins ml_dtypes commit 00d98cd92ade342fef589c0470379abb27baebe9 in
    // extern/xla/third_party/py/ml_dtypes/workspace.bzl. In that snapshot,
    // mxfloat.h's quiet_NaN() is raw 0x8 (negative zero for this format), and
    // float8.h's ConvertImpl preserves the input sign by negating that value.
    // Consequently NaN payload/quietness is discarded: positive NaNs map to
    // negative zero (0x8), while negative NaNs map to positive zero (0x0).
    // The MX specification explicitly permits implementation-defined NaN
    // conversion, so revisit these compatibility expectations if the pin moves.
    let cases = [
        (F16, 0x7e01, 0xfe01, 0x7c01, 0xfc01),
        (BF16, 0x7fc1, 0xffc1, 0x7f81, 0xff81),
        (F32, 0x7fc0_0001, 0xffc0_0001, 0x7f80_0001, 0xff80_0001),
        (
            F64,
            0x7ff8_0000_0000_0001,
            0xfff8_0000_0000_0001,
            0x7ff0_0000_0000_0001,
            0xfff0_0000_0000_0001,
        ),
    ];

    for (source, positive_quiet, negative_quiet, positive_signaling, negative_signaling) in cases {
        for bits in [positive_quiet, positive_signaling] {
            assert_eq!(
                convert(source, bits, F4E2M1FN).unwrap().1,
                0x8,
                "positive NaN from {source:?} bits 0x{bits:x}"
            );
        }
        for bits in [negative_quiet, negative_signaling] {
            assert_eq!(
                convert(source, bits, F4E2M1FN).unwrap().1,
                0x0,
                "negative NaN from {source:?} bits 0x{bits:x}"
            );
        }
    }

    assert!(!super::encoding::is_nan(F4E2M1FN, 0x8));
    assert!(!super::encoding::is_nan(F4E2M1FN, 0x0));
    assert_eq!(
        convert(F4E2M1FN, 0x8, F32).unwrap().1,
        (-0.0f32).to_bits().into()
    );
    assert_eq!(
        convert(F4E2M1FN, 0x0, F32).unwrap().1,
        0.0f32.to_bits().into()
    );
}
