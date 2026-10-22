// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Based on OpenXLA's ReducePrecision evaluator in
// xla/hlo/evaluator/hlo_evaluator_typed_visitor.h. Rewritten in Rust to
// operate on raw bits.
// See the crate-level LICENSE for the full Apache-2.0 terms.
//
//! Real ReducePrecision implementation, not a general math evaluator.
//!
//! Narrow formats compute in F32 before the existing typed conversion back to
//! the original element. Requested widths are not bounded above: guards must
//! precede every attribute-dependent shift. NaN handling inspects the original
//! compute encoding after rounding and exponent clamping have been calculated.

use super::{RawResult, encoding, invalid, unsupported, validate};
use crate::types::ElementType;

macro_rules! reduce_bits {
    ($name:ident, $uint:ty, $mantissa:expr, $exponent:expr) => {
        fn $name(original: $uint, exponent_bits: u32, mantissa_bits: u32) -> $uint {
            let mut value = original;
            if mantissa_bits < $mantissa {
                let shift = $mantissa - mantissa_bits;
                let last_bit: $uint = 1 << shift;
                let bias = ((last_bit >> 1) - 1) + ((value & last_bit) >> shift);
                value = value.wrapping_add(bias) & !(last_bit - 1);
            }
            let exponent_mask: $uint = ((1 << $exponent) - 1) << $mantissa;
            if exponent_bits < $exponent {
                let sign: $uint = value & (1 << (<$uint>::BITS - 1));
                let bias: $uint = (1 << ($exponent - 1)) - 1;
                let reduced_bias: $uint = (1 << (exponent_bits - 1)) - 1;
                let exponent = value & exponent_mask;
                let overflows = exponent > ((bias + reduced_bias) << $mantissa);
                let underflows = exponent <= ((bias - reduced_bias) << $mantissa);
                if overflows {
                    value = sign | exponent_mask;
                }
                if underflows {
                    value = sign;
                }
            }
            let magnitude = original & (<$uint>::MAX >> 1);
            if magnitude > exponent_mask {
                if mantissa_bits > 0 {
                    original
                } else {
                    exponent_mask
                }
            } else {
                value
            }
        }
    };
}
reduce_bits!(reduce32, u32, 23, 8);
reduce_bits!(reduce64, u64, 52, 11);

pub(crate) fn reduce_precision(
    element: ElementType,
    bits: u128,
    exponent_bits: i32,
    mantissa_bits: i32,
) -> RawResult {
    validate(element, bits)?;
    if !element.is_float() {
        return Err(unsupported(
            "reduce-precision requires a real floating format",
        ));
    }
    if exponent_bits < 1 {
        return Err(invalid(
            "reduce-precision exponent_bits must be at least one",
        ));
    }
    if mantissa_bits < 0 {
        return Err(invalid(
            "reduce-precision mantissa_bits must be nonnegative",
        ));
    }
    let exponent_bits = exponent_bits as u32;
    let mantissa_bits = mantissa_bits as u32;
    let result = if element == ElementType::F64 {
        u128::from(reduce64(bits as u64, exponent_bits, mantissa_bits))
    } else {
        let compute = encoding::to_f32(element, bits).to_bits();
        let reduced = reduce32(compute, exponent_bits, mantissa_bits);
        encoding::cast(ElementType::F32, u128::from(reduced), element)
    };
    Ok((element, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn public_precision(ty: ElementType, bits: u128, e: i32, m: i32) -> RawResult {
        crate::scalar::semantics::reduce_precision(
            crate::scalar::semantics::Scalar::from_bits(ty, bits)?,
            &crate::attributes::ReducePrecision {
                exponent_bits: e,
                mantissa_bits: m,
            },
        )
        .map(|value| (value.element(), value.bits()))
    }

    #[test]
    fn precision_reduction_is_idempotent_for_finite_encodings() {
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            let inputs: Vec<u128> = match element.bit_width() {
                width @ 1..=8 => (0..(1u128 << width)).collect(),
                16 => (0..=u16::MAX).map(u128::from).collect(),
                32 => vec![
                    0,
                    0x8000_0000,
                    1,
                    0x8000_0001,
                    0x0080_0000,
                    0x3f80_0000,
                    0x3f80_0001,
                    0x3f7f_ffff,
                    0x7f7f_ffff,
                    0xff7f_ffff,
                ],
                64 => vec![
                    0,
                    0x8000_0000_0000_0000,
                    1,
                    0x8000_0000_0000_0001,
                    0x0010_0000_0000_0000,
                    0x3ff0_0000_0000_0000,
                    0x3ff0_0000_0000_0001,
                    0x3fef_ffff_ffff_ffff,
                    0x7fef_ffff_ffff_ffff,
                    0xffef_ffff_ffff_ffff,
                ],
                width => panic!("unexpected real format width {width}"),
            };
            for bits in inputs {
                if !encoding::is_finite(element, bits) {
                    continue;
                }
                for (exponent_bits, mantissa_bits) in
                    [(1, 0), (5, 10), (8, 23), (11, 52), (i32::MAX, i32::MAX)]
                {
                    let once =
                        reduce_precision(element, bits, exponent_bits, mantissa_bits).unwrap();
                    if !encoding::is_finite(once.0, once.1) {
                        continue;
                    }
                    let twice =
                        reduce_precision(once.0, once.1, exponent_bits, mantissa_bits).unwrap();
                    assert_eq!(
                        twice, once,
                        "{element:?} {bits:x} ({exponent_bits}, {mantissa_bits})"
                    );
                }
            }
        }
    }

    #[test]
    fn attributes_have_lower_bounds_but_no_upper_bound() {
        use ElementType::*;
        for ty in [
            F4E2M1FN,
            F6E2M3FN,
            F6E3M2FN,
            F8E3M4,
            F8E4M3,
            F8E4M3FN,
            F8E4M3B11FNUZ,
            F8E4M3FNUZ,
            F8E5M2,
            F8E5M2FNUZ,
            F8E8M0FNU,
            F16,
            BF16,
            F32,
            F64,
        ] {
            for (e, m) in [(0, 0), (-1, 0), (1, -1), (i32::MIN, 0), (1, i32::MIN)] {
                assert!(reduce_precision(ty, 0, e, m).is_err());
                let error = public_precision(ty, 0, e, m).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::InvalidShape { .. })
                ));
            }
        }
        for (ty, bits) in [
            (ElementType::F32, 0xff80_0001),
            (ElementType::F64, 0xfff0_0000_0000_0001),
        ] {
            assert_eq!(
                reduce_precision(ty, bits, i32::MAX, i32::MAX).unwrap(),
                (ty, bits)
            );
            assert_eq!(
                public_precision(ty, bits, i32::MAX, i32::MAX).unwrap(),
                (ty, bits)
            );
        }
    }

    #[test]
    fn unsupported_types_and_excess_encoding_bits_are_diagnostics() {
        use ElementType::*;
        for ty in [
            Pred, S1, S2, S4, S8, S16, S32, S64, U1, U2, U4, U8, U16, U32, U64, C64, C128,
        ] {
            assert!(reduce_precision(ty, 0, 5, 10).is_err());
            let error = public_precision(ty, 0, 5, 10).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<crate::Diagnostic>(),
                Some(crate::Diagnostic::Unsupported { .. })
            ));
        }
        for ty in [F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E5M2, F16, F32, F64] {
            assert!(reduce_precision(ty, 1u128 << ty.bit_width(), 5, 10).is_err());
            let error = public_precision(ty, 1u128 << ty.bit_width(), 5, 10).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<crate::Diagnostic>(),
                Some(crate::Diagnostic::InvalidShape { .. })
            ));
        }
    }

    #[test]
    fn mantissa_rounding_precedes_exponent_flush_and_preserves_signed_zero() {
        use ElementType::*;
        assert_eq!(
            reduce_precision(F32, 0x387f_f000, 5, 10).unwrap(),
            (F32, 0x3880_0000)
        );
        assert_eq!(reduce_precision(F32, 0x387f_efff, 5, 10).unwrap(), (F32, 0));
        assert_eq!(
            reduce_precision(F32, 0xb87f_efff, 5, 10).unwrap(),
            (F32, 0x8000_0000)
        );
        assert_eq!(reduce_precision(F32, 0x3f80_0000, 1, 23).unwrap(), (F32, 0));
        assert_eq!(reduce_precision(F32, 0x3f80_0001, 1, 23).unwrap(), (F32, 0));
        assert_eq!(
            reduce_precision(F32, 0x4000_0000, 1, 23).unwrap(),
            (F32, 0x7f80_0000)
        );
        assert_eq!(
            reduce_precision(F32, 0x3f80_1000, 8, 10).unwrap(),
            (F32, 0x3f80_0000)
        );
        assert_eq!(
            reduce_precision(F32, 0x3f80_3000, 8, 10).unwrap(),
            (F32, 0x3f80_4000)
        );
        assert_eq!(reduce_precision(F32, 1, 8, 23).unwrap(), (F32, 1));
        assert_eq!(
            reduce_precision(F32, 0x8000_0000, 1, 0).unwrap(),
            (F32, 0x8000_0000)
        );
    }

    #[test]
    fn original_nan_override_and_narrow_typed_codec_are_not_raw_identity() {
        use ElementType::*;
        for raw in [0x7f80_0001, 0xff80_0001, 0x7fff_ffff, 0xffff_ffff] {
            assert_eq!(reduce_precision(F32, raw, 1, 1).unwrap(), (F32, raw));
            assert_eq!(
                reduce_precision(F32, raw, 1, 0).unwrap(),
                (F32, 0x7f80_0000)
            );
        }
        for raw in [0x7ff0_0000_0000_0001, 0xfff0_0000_0000_0001] {
            assert_eq!(reduce_precision(F64, raw, 1, 1).unwrap(), (F64, raw));
            assert_eq!(
                reduce_precision(F64, raw, 1, 0).unwrap(),
                (F64, 0x7ff0_0000_0000_0000)
            );
        }
        assert_eq!(reduce_precision(F16, 0x7c01, 8, 23).unwrap(), (F16, 0x7e00));
        assert_eq!(
            reduce_precision(BF16, 0xff81, 8, 23).unwrap(),
            (BF16, 0xffc0)
        );
        assert_eq!(
            reduce_precision(F8E8M0FNU, 0xff, 1, 0).unwrap(),
            (F8E8M0FNU, 0xff)
        );
        assert_eq!(
            reduce_precision(F8E8M0FNU, 127, 1, 0).unwrap(),
            (F8E8M0FNU, 0xff)
        );
    }
}
