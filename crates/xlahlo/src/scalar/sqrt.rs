// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Based on OpenXLA's square-root evaluator in
// xla/hlo/evaluator/hlo_evaluator_typed_visitor.h.
// The integer restoring-square-root implementation is independent; observed
// host exceptional behavior is a compatibility policy only.
// Full Apache-2.0 terms are in the crate-level LICENSE.

//! Real square root under an explicit simulated floating environment.
//! No ambient floating-point state is read or changed.

use super::{
    RawResult, encoding, invalid, remainder::RemainderContext as SqrtContext,
    rounding::RoundingMode, unsupported, validate,
};
use crate::types::ElementType;

macro_rules! restoring_root {
    ($name:ident, $uint:ty) => {
        fn $name(mut remainder: $uint) -> ($uint, $uint) {
            let mut root = 0;
            let mut bit: $uint = 1 << (<$uint>::BITS - 2);
            while bit > remainder {
                bit >>= 2;
            }
            while bit != 0 {
                let trial = root + bit;
                if remainder >= trial {
                    remainder -= trial;
                    root = (root >> 1) + bit;
                } else {
                    root >>= 1;
                }
                bit >>= 2;
            }
            (root, remainder)
        }
    };
}
restoring_root!(root64, u64);
restoring_root!(root128, u128);

macro_rules! sqrt_bits {
    ($name:ident, $uint:ty, $wide:ty, $root:ident, $mantissa:expr, $bias:expr) => {
        fn $name(original: $uint) -> $uint {
            let sign: $uint = 1 << (<$uint>::BITS - 1);
            let magnitude = original & (sign - 1);
            let infinity: $uint = (2 * $bias + 1) << $mantissa;
            let quiet: $uint = 1 << ($mantissa - 1);
            if magnitude > infinity {
                return original | quiet;
            }
            if magnitude == 0 {
                return original;
            }
            if original & sign != 0 {
                return sign | infinity | quiet;
            }
            if magnitude == infinity {
                return original;
            }
            let hidden: $uint = 1 << $mantissa;
            let mut significand = magnitude & (hidden - 1);
            let mut exponent = (magnitude >> $mantissa) as i32;
            if exponent == 0 {
                exponent = 1;
                while significand < hidden {
                    significand <<= 1;
                    exponent -= 1;
                }
            } else {
                significand |= hidden;
            }
            exponent -= $bias;
            if exponent & 1 != 0 {
                significand <<= 1;
                exponent -= 1;
            }
            let radicand = (significand as $wide) << $mantissa;
            let (floor, remainder) = $root(radicand);
            // The exact midpoint square has fractional part 1/4. An integer
            // radicand cannot tie, and rounds upward iff its residual > floor.
            let rounded = floor + <$wide>::from(remainder > floor);
            let mut result_exponent = exponent / 2 + $bias;
            let mut result_significand = rounded as $uint;
            if result_significand == hidden << 1 {
                result_significand >>= 1;
                result_exponent += 1;
            }
            ((result_exponent as $uint) << $mantissa) | (result_significand & (hidden - 1))
        }
    };
}
sqrt_bits!(sqrt32, u32, u64, root64, 23, 127);
sqrt_bits!(sqrt64, u64, u128, root128, 52, 1023);

pub(crate) fn sqrt(element: ElementType, bits: u128, context: SqrtContext) -> RawResult {
    if context.rounding != RoundingMode::NearestEven
        || context.flush_to_zero
        || context.denormals_are_zero
    {
        return Err(invalid(
            "sqrt requires simulated nearest-even with FTZ and DAZ disabled",
        ));
    }
    validate(element, bits)?;
    if !element.is_float() {
        return Err(unsupported(
            "real sqrt is unimplemented for non-real formats, including complex",
        ));
    }
    let result = if element == ElementType::F64 {
        u128::from(sqrt64(bits as u64))
    } else {
        let compute = encoding::to_f32(element, bits).to_bits();
        encoding::cast(ElementType::F32, u128::from(sqrt32(compute)), element)
    };
    Ok((element, result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Diagnostic;

    const CONTEXT: SqrtContext = SqrtContext {
        rounding: RoundingMode::NearestEven,
        flush_to_zero: false,
        denormals_are_zero: false,
    };

    #[test]
    fn square_root_is_nonnegative_monotone_and_exact_on_representable_squares() {
        let values = [0.0f32, 0.25, 1.0, 2.0, 3.0, 4.0, 9.0, 16.0, 64.0, 256.0];
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            let mut previous = None;
            for value in values {
                let input = encoding::cast(ElementType::F32, u128::from(value.to_bits()), element);
                let x = encoding::to_f32(element, input);
                if !x.is_finite() || x < 0.0 {
                    continue;
                }
                let (_, output) = sqrt(element, input, CONTEXT).unwrap();
                let y = encoding::to_f32(element, output);
                assert!(y.is_finite() && y >= 0.0, "{element:?}: sqrt({x})={y}");
                if let Some(last) = previous {
                    assert!(y >= last, "{element:?}: sqrt not monotone at {x}");
                }
                previous = Some(y);
                let square_root = x.sqrt();
                if square_root.fract() == 0.0 {
                    assert_eq!(y, square_root, "{element:?}: sqrt({x})");
                }
                let error = (y * y - x).abs();
                let tolerance = (x.abs() * 0.5).max(f32::from_bits(1) * 2.0);
                assert!(
                    error <= tolerance,
                    "{element:?}: sqrt({x})={y}, residual={error}"
                );
            }
        }
    }

    #[test]
    fn restoring_integer_roots_satisfy_square_and_remainder_bounds() {
        for value in [0u64, 1, 2, 3, 4, 15, 16, 17, u32::MAX as u64, u64::MAX] {
            let (root, remainder) = root64(value);
            assert_eq!(
                u128::from(root) * u128::from(root) + u128::from(remainder),
                u128::from(value)
            );
            assert!(u128::from(remainder) < 2 * u128::from(root) + 1);
        }
        for value in [0u128, 1, 2, 3, 4, 15, 16, 17, u64::MAX as u128, u128::MAX] {
            let (root, remainder) = root128(value);
            let square = root * root;
            assert_eq!(square + remainder, value);
            assert!(remainder < 2 * root + 1);
        }
        for root in [1u128, 2, 3, 7, 255, 65_535, 1u128 << 32, (1u128 << 63) - 1] {
            let square = root * root;
            for value in [square - 1, square, square + 1] {
                let (actual_root, remainder) = root128(value);
                assert_eq!(actual_root, if value < square { root - 1 } else { root });
                assert_eq!(actual_root * actual_root + remainder, value);
            }
        }
        let mut value = 0x9e37_79b9_7f4a_7c15_d1b5_4a32_d192_ed03u128;
        for _ in 0..512 {
            value = value
                .wrapping_mul(0xda94_2042_e4dd_58b5_9e37_79b9_7f4a_7c15)
                .wrapping_add(0x632b_e59b_d9b4_e019_94d0_49bb_1331_11eb);
            let (root, remainder) = root128(value);
            assert_eq!(root * root + remainder, value);
            assert!(remainder < 2 * root + 1);
            let (root, remainder) = root64(value as u64);
            assert_eq!(
                u128::from(root) * u128::from(root) + u128::from(remainder),
                u128::from(value as u64)
            );
            assert!(u128::from(remainder) < 2 * u128::from(root) + 1);
        }
    }

    #[test]
    fn square_root_special_inputs_follow_classification_and_sign_invariants() {
        use ElementType::*;
        for ty in [F32, F64] {
            let (zero, infinity, nan, negative) = if ty == F32 {
                (0u128, 0x7f80_0000, 0x7fc0_1234, 0xbf80_0000)
            } else {
                (
                    0u128,
                    0x7ff0_0000_0000_0000,
                    0x7ff8_0000_0000_1234,
                    0xbff0_0000_0000_0000,
                )
            };
            assert_eq!(sqrt(ty, zero, CONTEXT).unwrap(), (ty, zero));
            let (_, positive_infinity) = sqrt(ty, infinity, CONTEXT).unwrap();
            assert!(encoding::to_f32(ty, positive_infinity).is_infinite());
            let (_, nan_result) = sqrt(ty, nan, CONTEXT).unwrap();
            assert!(encoding::is_nan(ty, nan_result));
            let (_, negative_result) = sqrt(ty, negative, CONTEXT).unwrap();
            assert!(encoding::is_nan(ty, negative_result));
        }
    }

    #[test]
    fn only_explicit_nearest_with_ftz_daz_disabled_is_admitted() {
        let mut rejected = 0;
        for rounding in [
            RoundingMode::NearestEven,
            RoundingMode::TowardZero,
            RoundingMode::TowardPositive,
            RoundingMode::TowardNegative,
        ] {
            for flush_to_zero in [false, true] {
                for denormals_are_zero in [false, true] {
                    let context = SqrtContext {
                        rounding,
                        flush_to_zero,
                        denormals_are_zero,
                    };
                    if context == CONTEXT {
                        continue;
                    }
                    for &ty in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
                        let error = sqrt(ty, 0, context).unwrap_err();
                        assert!(matches!(
                            error.err.downcast_ref::<Diagnostic>(),
                            Some(Diagnostic::InvalidShape { .. })
                        ));
                        rejected += 1;
                    }
                }
            }
        }
        assert_eq!(rejected, 225);
    }

    #[test]
    fn nonreal_unimplemented_and_excess_significant_bits_are_typed_errors() {
        let mut unimplemented = 0;
        for &ty in ElementType::ALL {
            if ty.is_float() {
                let error = sqrt(ty, 1u128 << ty.bit_width(), CONTEXT).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<Diagnostic>(),
                    Some(Diagnostic::InvalidShape { .. })
                ));
            } else {
                // Complex sqrt exists upstream, but is not implemented in this slice.
                let error = sqrt(ty, 0, CONTEXT).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<Diagnostic>(),
                    Some(Diagnostic::Unsupported { .. })
                ));
                unimplemented += 1;
            }
        }
        assert_eq!(unimplemented, 17);
    }
}
