// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Based on OpenXLA's fmod evaluator in
// xla/hlo/evaluator/hlo_evaluator_typed_visitor.h.
// The integer dyadic implementation is independent; MSVC/UCRT observations
// define its documented exceptional-value compatibility policy only.
// Full Apache-2.0 terms are in the crate-level LICENSE.

//! Real fmod implementation with an explicit simulated host context.
//! No ambient FP state is read or changed; this is not IEEE nearest remainder.

use super::{RawResult, encoding, invalid, rounding::RoundingMode, unsupported, validate};
use crate::types::ElementType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RemainderContext {
    pub(crate) rounding: RoundingMode,
    pub(crate) flush_to_zero: bool,
    pub(crate) denormals_are_zero: bool,
}

macro_rules! finite_remainder {
    ($name:ident, $uint:ty, $mantissa:expr) => {
        fn $name(lhs: $uint, rhs: $uint) -> $uint {
            let sign = lhs & (1 << (<$uint>::BITS - 1));
            let x = lhs & (<$uint>::MAX >> 1);
            let y = rhs & (<$uint>::MAX >> 1);
            if x < y {
                return lhs;
            }
            if x == y {
                return sign;
            }
            let hidden: $uint = 1 << $mantissa;
            let mut ex = (x >> $mantissa) as i32;
            let mut ey = (y >> $mantissa) as i32;
            let mut mx = x & (hidden - 1);
            let mut my = y & (hidden - 1);
            if ex == 0 {
                ex = 1;
                while mx < hidden {
                    mx <<= 1;
                    ex -= 1;
                }
            } else {
                mx |= hidden;
            }
            if ey == 0 {
                ey = 1;
                while my < hidden {
                    my <<= 1;
                    ey -= 1;
                }
            } else {
                my |= hidden;
            }
            while ex > ey {
                if mx >= my {
                    mx -= my;
                    if mx == 0 {
                        return sign;
                    }
                }
                mx <<= 1;
                ex -= 1;
            }
            if mx >= my {
                mx -= my;
            }
            if mx == 0 {
                return sign;
            }
            while mx < hidden {
                mx <<= 1;
                ex -= 1;
            }
            sign | if ex > 0 {
                ((ex as $uint) << $mantissa) | (mx & (hidden - 1))
            } else {
                mx >> (1 - ex)
            }
        }
    };
}
finite_remainder!(finite32, u32, 23);
finite_remainder!(finite64, u64, 52);

fn remainder32(lhs: u32, rhs: u32) -> u32 {
    let x = lhs & 0x7fff_ffff;
    let y = rhs & 0x7fff_ffff;
    if x > 0x7f80_0000 {
        return lhs | 0x0040_0000;
    }
    if x == 0x7f80_0000 {
        return 0xffc0_0000;
    }
    if y > 0x7f80_0000 {
        return rhs | 0x0040_0000;
    }
    if y == 0 {
        return 0xffc0_0000;
    }
    finite32(lhs, rhs)
}

fn remainder64(lhs: u64, rhs: u64) -> u64 {
    let x = lhs & 0x7fff_ffff_ffff_ffff;
    let y = rhs & 0x7fff_ffff_ffff_ffff;
    // Pinned fmod checks rhs NaN first and returns signaling bits unchanged.
    if y > 0x7ff0_0000_0000_0000 {
        return rhs;
    }
    if x > 0x7ff0_0000_0000_0000 {
        return lhs;
    }
    if x == 0x7ff0_0000_0000_0000 {
        return lhs | 0x0008_0000_0000_0000;
    }
    if y == 0 {
        return 0x7ff8_0000_0000_0000;
    }
    finite64(lhs, rhs)
}

pub(crate) fn remainder(
    element: ElementType,
    lhs: u128,
    rhs: u128,
    context: RemainderContext,
) -> RawResult {
    if context.rounding != RoundingMode::NearestEven
        || context.flush_to_zero
        || context.denormals_are_zero
    {
        return Err(invalid(
            "remainder requires simulated nearest-even with FTZ and DAZ disabled",
        ));
    }
    validate(element, lhs)?;
    validate(element, rhs)?;
    if !element.is_float() {
        return Err(unsupported(
            "real remainder requires a real floating format",
        ));
    }
    let result = if element == ElementType::F64 {
        u128::from(remainder64(lhs as u64, rhs as u64))
    } else {
        let lhs = encoding::to_f32(element, lhs).to_bits();
        let rhs = encoding::to_f32(element, rhs).to_bits();
        encoding::cast(ElementType::F32, u128::from(remainder32(lhs, rhs)), element)
    };
    Ok((element, result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Diagnostic;
    use crate::scalar::semantics::{self, EvaluationFloatContext, EvaluationRoundingMode, Scalar};

    const CONTEXT: RemainderContext = RemainderContext {
        rounding: RoundingMode::NearestEven,
        flush_to_zero: false,
        denormals_are_zero: false,
    };

    fn public_context(context: RemainderContext) -> EvaluationFloatContext {
        EvaluationFloatContext {
            rounding: match context.rounding {
                RoundingMode::NearestEven => EvaluationRoundingMode::NearestEven,
                RoundingMode::TowardZero => EvaluationRoundingMode::TowardZero,
                RoundingMode::TowardPositive => EvaluationRoundingMode::TowardPositive,
                RoundingMode::TowardNegative => EvaluationRoundingMode::TowardNegative,
            },
            flush_to_zero: context.flush_to_zero,
            denormals_are_zero: context.denormals_are_zero,
        }
    }

    fn public_remainder(
        element: ElementType,
        lhs: u128,
        rhs: u128,
        context: RemainderContext,
    ) -> RawResult {
        semantics::remainder_with_context(
            Scalar::from_bits(element, lhs)?,
            Scalar::from_bits(element, rhs)?,
            public_context(context),
        )
        .map(|value| (value.element(), value.bits()))
    }

    #[test]
    fn finite_remainder_obeys_magnitude_sign_and_identity_properties() {
        let values = [
            -15.5f32, -7.0, -3.0, -1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0, 3.0, 7.0, 15.5,
        ];
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            let encoded =
                |value: f32| encoding::cast(ElementType::F32, u128::from(value.to_bits()), element);
            let as_f64 = |bits| {
                if element == ElementType::F64 {
                    f64::from_bits(bits as u64)
                } else {
                    f64::from(encoding::to_f32(element, bits))
                }
            };
            for &lhs_value in &values {
                let lhs = encoded(lhs_value);
                let a = as_f64(lhs);
                if !a.is_finite() {
                    continue;
                }
                for &rhs_value in &values {
                    let rhs = encoded(rhs_value);
                    let b = as_f64(rhs);
                    if !b.is_finite() || b == 0.0 {
                        continue;
                    }
                    let (result_element, result_bits) =
                        remainder(element, lhs, rhs, CONTEXT).unwrap();
                    assert_eq!(result_element, element);
                    if element == ElementType::F8E8M0FNU && a % b == 0.0 {
                        assert_eq!(result_bits, 0xff, "zero has no encoding in {element:?}");
                        continue;
                    }
                    let r = as_f64(result_bits);
                    assert!(
                        r.is_finite(),
                        "{element:?}: {a} % {b} -> {result_bits:#x} ({r})"
                    );
                    assert!(r.abs() <= b.abs(), "{element:?}: {a} % {b} = {r}");
                    if r != 0.0 {
                        assert_eq!(r.is_sign_negative(), a.is_sign_negative());
                    }
                    if a.abs() < b.abs() {
                        assert_eq!(result_bits, lhs, "{element:?}: {a} % {b}");
                    }
                    if a == 0.0 || r == 0.0 {
                        assert_eq!(r, 0.0);
                        let negative_zero = encoding::cast(
                            ElementType::F32,
                            u128::from((-0.0f32).to_bits()),
                            element,
                        );
                        if encoding::to_f32(element, negative_zero).is_sign_negative() {
                            assert_eq!(r.is_sign_negative(), a.is_sign_negative());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn simulated_context_rejects_every_unsupported_control_combination_before_math() {
        let mut rejected = 0;
        for mode in [
            RoundingMode::NearestEven,
            RoundingMode::TowardZero,
            RoundingMode::TowardPositive,
            RoundingMode::TowardNegative,
        ] {
            for flush_to_zero in [false, true] {
                for denormals_are_zero in [false, true] {
                    let context = RemainderContext {
                        rounding: mode,
                        flush_to_zero,
                        denormals_are_zero,
                    };
                    if context == CONTEXT {
                        continue;
                    }
                    for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
                        let error = remainder(element, 0, 0, context).unwrap_err();
                        assert!(matches!(
                            error.err.downcast_ref::<Diagnostic>(),
                            Some(Diagnostic::InvalidShape { .. })
                        ));
                        let error = public_remainder(element, 0, 0, context).unwrap_err();
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
    fn invalid_remainder_inputs_are_classified_as_nan_without_payload_assumptions() {
        use ElementType::*;
        for (ty, lhs, rhs) in [
            (F32, 0x7f80_1234, 0x3f80_0000),
            (F32, 0x3f80_0000, 0xffc0_5678),
            (F32, 0x7f80_0000, 0x3f80_0000),
            (F32, 0x3f80_0000, 0),
            (F64, 0x7ff0_0000_0000_1234, 0x3ff0_0000_0000_0000),
            (F64, 0x3ff0_0000_0000_0000, 0xfff8_0000_0000_5678),
            (F64, 0x7ff0_0000_0000_0000, 0x3ff0_0000_0000_0000),
            (F64, 0x3ff0_0000_0000_0000, 0),
        ] {
            let (_, bits) = remainder(ty, lhs, rhs, CONTEXT).unwrap();
            assert!(encoding::is_nan(ty, bits), "{ty:?} {bits:x}");
        }
    }

    #[test]
    fn truncation_quotient_signed_zero_and_subnormal_remainders_are_exact() {
        use ElementType::*;
        for (ty, lhs, rhs, expected) in [
            (F32, 0x40e0_0000, 0x4080_0000, 0x4040_0000),
            (F32, 0xc0e0_0000, 0x4080_0000, 0xc040_0000),
            (F32, 0xc000_0000, 0x3f80_0000, 0x8000_0000),
            (F32, 0x0080_0001, 0x0080_0000, 1),
            (F32, 0x8000_0001, 0x0080_0000, 0x8000_0001),
            (
                F64,
                0x401c_0000_0000_0000,
                0x4010_0000_0000_0000,
                0x4008_0000_0000_0000,
            ),
            (
                F64,
                0xc000_0000_0000_0000,
                0x3ff0_0000_0000_0000,
                0x8000_0000_0000_0000,
            ),
            (F64, 0x0010_0000_0000_0001, 0x0010_0000_0000_0000, 1),
        ] {
            assert_eq!(remainder(ty, lhs, rhs, CONTEXT).unwrap(), (ty, expected));
            assert_eq!(
                public_remainder(ty, lhs, rhs, CONTEXT).unwrap(),
                (ty, expected)
            );
        }
    }

    #[test]
    fn nonreal_types_and_excess_bits_in_either_operand_are_typed_errors() {
        let mut unsupported = 0;
        for &element in ElementType::ALL.iter().filter(|ty| !ty.is_float()) {
            let error = remainder(element, 0, 0, CONTEXT).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::Unsupported { .. })
            ));
            let error = public_remainder(element, 0, 0, CONTEXT).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::Unsupported { .. })
            ));
            unsupported += 1;
        }
        assert_eq!(unsupported, 17);
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            for (lhs, rhs) in [
                (1u128 << element.bit_width(), 0),
                (0, 1u128 << element.bit_width()),
            ] {
                let error = remainder(element, lhs, rhs, CONTEXT).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<Diagnostic>(),
                    Some(Diagnostic::InvalidShape { .. })
                ));
                let invalid_bits = if lhs == 0 { rhs } else { lhs };
                let error = Scalar::from_bits(element, invalid_bits).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<Diagnostic>(),
                    Some(Diagnostic::InvalidShape { .. })
                ));
            }
        }
    }
}
