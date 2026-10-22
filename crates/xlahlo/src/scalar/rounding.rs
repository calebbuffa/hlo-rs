// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Based on OpenXLA's rounding callbacks in
// xla/hlo/evaluator/hlo_evaluator_typed_visitor.h.
// The integer-bit implementation is independent; observed host exceptional
// behavior is a compatibility policy, not copied Microsoft runtime code.
// Full Apache-2.0 terms are in the crate-level LICENSE.
//
//! Rounding implementation. No ambient environment access or change.
//!
//! The caller explicitly supplies its evaluation context's rounding mode.
//! This is not a report of the thread's actual floating-point control register.
//! RoundNearestEven rejects every supplied mode except NearestEven, mirroring
//! the pinned callback's FE_TONEAREST precondition. Integration must establish
//! that context rather than silently manufacture it from an ambient assumption.

use super::{RawResult, encoding, invalid, unsupported, validate};
use crate::{opcode::Opcode, types::ElementType};

/// Explicit numerical context, not an implicit read of ambient process state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RoundingMode {
    NearestEven,
    TowardZero,
    TowardPositive,
    TowardNegative,
}

macro_rules! round_bits {
    ($name:ident, $uint:ty, $mantissa:expr, $exponent:expr, $bias:expr) => {
        fn $name(opcode: Opcode, original: $uint) -> pliron::result::Result<$uint> {
            if !matches!(
                opcode,
                Opcode::RoundNearestAfz | Opcode::RoundNearestEven | Opcode::Ceil | Opcode::Floor
            ) {
                return Err(unsupported("unimplemented rounding operation"));
            }
            let sign = original & (1 << (<$uint>::BITS - 1));
            let magnitude = original & (<$uint>::MAX >> 1);
            let infinity: $uint = ((1 << $exponent) - 1) << $mantissa;
            if magnitude > infinity {
                // Observed MSVC/UCRT: std::round preserves signaling NaNs;
                // nearbyint/ceil/floor quiet them without changing payload/sign.
                return Ok(if opcode == Opcode::RoundNearestAfz {
                    original
                } else {
                    original | (1 << ($mantissa - 1))
                });
            }
            let exponent = (magnitude >> $mantissa) as i32 - $bias;
            if exponent >= $mantissa || magnitude == 0 {
                return Ok(original);
            }
            if exponent < 0 {
                let half: $uint = ($bias - 1) << $mantissa;
                let round_up = match opcode {
                    Opcode::RoundNearestAfz => magnitude >= half,
                    Opcode::RoundNearestEven => magnitude > half,
                    Opcode::Ceil => sign == 0,
                    Opcode::Floor => sign != 0,
                    _ => return Err(unsupported("unimplemented rounding operation")),
                };
                return Ok(sign
                    | if round_up {
                        ($bias as $uint) << $mantissa
                    } else {
                        0
                    });
            }
            let unit: $uint = 1 << ($mantissa - exponent);
            let discarded = magnitude & (unit - 1);
            let truncated = magnitude & !(unit - 1);
            let round_up = match opcode {
                Opcode::RoundNearestAfz => discarded >= unit >> 1,
                Opcode::RoundNearestEven => {
                    discarded > unit >> 1 || (discarded == unit >> 1 && truncated & unit != 0)
                }
                Opcode::Ceil => sign == 0 && discarded != 0,
                Opcode::Floor => sign != 0 && discarded != 0,
                _ => return Err(unsupported("unimplemented rounding operation")),
            };
            Ok(sign
                | if round_up {
                    truncated + unit
                } else {
                    truncated
                })
        }
    };
}
round_bits!(round32, u32, 23, 8, 127);
round_bits!(round64, u64, 52, 11, 1023);

pub(crate) fn unary(
    opcode: Opcode,
    element: ElementType,
    bits: u128,
    mode: RoundingMode,
) -> RawResult {
    validate(element, bits)?;
    if !element.is_float() {
        return Err(unsupported("rounding requires a real floating format"));
    }
    if !matches!(
        opcode,
        Opcode::RoundNearestAfz | Opcode::RoundNearestEven | Opcode::Ceil | Opcode::Floor
    ) {
        return Err(unsupported("unimplemented rounding operation"));
    }
    if opcode == Opcode::RoundNearestEven && mode != RoundingMode::NearestEven {
        return Err(invalid(
            "round-nearest-even requires an explicit nearest-even evaluation context",
        ));
    }
    let rounded = if element == ElementType::F64 {
        u128::from(round64(opcode, bits as u64)?)
    } else {
        let compute = encoding::to_f32(element, bits).to_bits();
        encoding::cast(
            ElementType::F32,
            u128::from(round32(opcode, compute)?),
            element,
        )
    };
    Ok((element, rounded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::semantics::{self, EvaluationRoundingMode, Scalar};

    const OPS: [Opcode; 4] = [
        Opcode::RoundNearestAfz,
        Opcode::RoundNearestEven,
        Opcode::Ceil,
        Opcode::Floor,
    ];

    fn public_rounding(
        opcode: Opcode,
        element: ElementType,
        bits: u128,
        mode: EvaluationRoundingMode,
    ) -> RawResult {
        semantics::rounding_unary(opcode, Scalar::from_bits(element, bits)?, mode)
            .map(|value| (value.element(), value.bits()))
    }

    #[test]
    fn rounding_obeys_integrality_direction_tie_and_idempotence_properties() {
        let values = [
            -3.75f32, -3.5, -2.5, -1.5, -0.5, -0.25, 0.0, 0.25, 0.5, 1.5, 2.5, 3.5, 3.75,
        ];
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            for &value in &values {
                let input = encoding::cast(ElementType::F32, u128::from(value.to_bits()), element);
                let x = f64::from(encoding::to_f32(element, input));
                if !x.is_finite() {
                    continue;
                }
                for op in OPS {
                    let (out_element, output) =
                        unary(op, element, input, RoundingMode::NearestEven).unwrap();
                    assert_eq!(out_element, element);
                    let y = f64::from(encoding::to_f32(element, output));
                    // Rounding can move a finite value beyond a narrow format's
                    // range; exceptional conversion behavior is covered separately.
                    if !y.is_finite() {
                        continue;
                    }
                    assert_eq!(y.fract(), 0.0, "{element:?} {op:?} {x} -> {y}");
                    match op {
                        Opcode::Ceil => assert!(y >= x && y - x < 1.0, "{element:?} {x} -> {y}"),
                        Opcode::Floor => assert!(y <= x && x - y < 1.0, "{element:?} {x} -> {y}"),
                        Opcode::RoundNearestAfz | Opcode::RoundNearestEven => {
                            assert!((y - x).abs() <= 0.5, "{element:?} {op:?} {x} -> {y}");
                            if x.abs() - x.abs().floor() == 0.5 {
                                if op == Opcode::RoundNearestAfz {
                                    assert_eq!(y.abs(), x.abs().ceil(), "{element:?} {x}");
                                } else {
                                    assert_eq!(y.abs() % 2.0, 0.0, "{element:?} {x} -> {y}");
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                    let again = unary(op, element, output, RoundingMode::NearestEven).unwrap();
                    assert_eq!(again, (element, output), "{element:?} {op:?} {x}");
                }
            }
        }
    }

    #[test]
    fn evaluation_context_is_explicit_and_nearest_even_rejects_other_modes() {
        for (mode, public_mode) in [
            (RoundingMode::TowardZero, EvaluationRoundingMode::TowardZero),
            (
                RoundingMode::TowardPositive,
                EvaluationRoundingMode::TowardPositive,
            ),
            (
                RoundingMode::TowardNegative,
                EvaluationRoundingMode::TowardNegative,
            ),
        ] {
            let mut formats = 0;
            for &ty in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
                formats += 1;
                let error = unary(Opcode::RoundNearestEven, ty, 0, mode).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::InvalidShape { .. })
                ));
                let error =
                    public_rounding(Opcode::RoundNearestEven, ty, 0, public_mode).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::InvalidShape { .. })
                ));
                for op in [Opcode::RoundNearestAfz, Opcode::Ceil, Opcode::Floor] {
                    let expected = unary(op, ty, 0, RoundingMode::NearestEven).unwrap();
                    assert_eq!(unary(op, ty, 0, mode).unwrap(), expected);
                    assert_eq!(public_rounding(op, ty, 0, public_mode).unwrap(), expected);
                }
            }
            assert_eq!(formats, 15);
        }
    }

    #[test]
    fn half_ties_even_odd_adjacent_and_signed_zero_are_distinct() {
        use ElementType::*;
        // F32 +/-0.5, +/-1.5, +2.5 and immediate half neighbors.
        for (input, expected) in [
            (0x3f00_0000, [0x3f80_0000, 0, 0x3f80_0000, 0]),
            (
                0xbf00_0000,
                [0xbf80_0000, 0x8000_0000, 0x8000_0000, 0xbf80_0000],
            ),
            (
                0x3fc0_0000,
                [0x4000_0000, 0x4000_0000, 0x4000_0000, 0x3f80_0000],
            ),
            (
                0xbfc0_0000,
                [0xc000_0000, 0xc000_0000, 0xbf80_0000, 0xc000_0000],
            ),
            (
                0x4020_0000,
                [0x4040_0000, 0x4000_0000, 0x4040_0000, 0x4000_0000],
            ),
            (0x3eff_ffff, [0, 0, 0x3f80_0000, 0]),
            (0x3f00_0001, [0x3f80_0000, 0x3f80_0000, 0x3f80_0000, 0]),
        ] {
            for (op, output) in OPS.into_iter().zip(expected) {
                assert_eq!(
                    unary(op, F32, input, RoundingMode::NearestEven).unwrap(),
                    (F32, output)
                );
            }
        }
        for ty in [F32, F64] {
            let zero = 1u128 << (ty.bit_width() - 1);
            for op in OPS {
                assert_eq!(
                    unary(op, ty, zero, RoundingMode::NearestEven).unwrap(),
                    (ty, zero)
                );
            }
        }
    }

    #[test]
    fn rounding_preserves_nan_classification_across_operations_and_formats() {
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            let nan = encoding::cast(ElementType::F32, u128::from(f32::NAN.to_bits()), element);
            if !encoding::is_nan(element, nan) {
                continue;
            }
            for op in OPS {
                let (_, result) = unary(op, element, nan, RoundingMode::NearestEven).unwrap();
                assert!(encoding::is_nan(element, result), "{element:?} {op:?}");
            }
        }
    }

    #[test]
    fn non_real_types_invalid_bits_and_unrelated_opcodes_are_rejected() {
        use ElementType::*;
        for ty in [
            Pred, S1, S2, S4, S8, S16, S32, S64, U1, U2, U4, U8, U16, U32, U64, C64, C128,
        ] {
            for op in OPS {
                let error = unary(op, ty, 0, RoundingMode::NearestEven).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::Unsupported { .. })
                ));
                let error =
                    public_rounding(op, ty, 0, EvaluationRoundingMode::NearestEven).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::Unsupported { .. })
                ));
            }
        }
        for ty in [F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E5M2, F16, F32, F64] {
            for op in OPS {
                let error =
                    unary(op, ty, 1u128 << ty.bit_width(), RoundingMode::NearestEven).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::InvalidShape { .. })
                ));
                let error = Scalar::from_bits(ty, 1u128 << ty.bit_width()).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::InvalidShape { .. })
                ));
            }
        }
        let error = unary(Opcode::Exp, F32, 0, RoundingMode::NearestEven).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<crate::Diagnostic>(),
            Some(crate::Diagnostic::Unsupported { .. })
        ));
        let error =
            public_rounding(Opcode::Exp, F32, 0, EvaluationRoundingMode::NearestEven).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<crate::Diagnostic>(),
            Some(crate::Diagnostic::Unsupported { .. })
        ));
    }
}
