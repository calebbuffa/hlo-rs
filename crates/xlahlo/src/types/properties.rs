//! Exact primitive properties and value-preserving cast analysis.
//!
//! Based on OpenXLA's `xla/primitive_util.cc`, its tests, and zero traits in
//! `xla/types.h`. Small-float limits follow TensorFlow ml_dtypes' `float8.h`
//! and `mxfloat.h`.
//! These are format properties, not storage-width or precision-order estimates.

use super::{ElementType, PrimitiveType};
use crate::Diagnostic;
use pliron::result::Result;

/// Binary floating-point limits and special values used by pinned primitive utilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatProperties {
    /// Precision, including the implicit leading significand bit where present.
    pub significand_width: i32,
    /// Number of exponent encoding bits.
    pub exponent_width: i32,
    /// Bias applied to the encoded exponent.
    pub exponent_bias: i32,
    /// `numeric_limits::min_exponent`, one above the least normal exponent.
    pub underflow_exponent: i32,
    /// `numeric_limits::max_exponent`, one above the greatest finite exponent.
    pub overflow_exponent: i32,
    /// Whether infinities are representable.
    pub has_infinity: bool,
    /// Whether quiet NaNs are representable.
    pub has_nan: bool,
    /// Whether negative zero is representable.
    pub has_negative_zero: bool,
    /// Whether positive zero is representable.
    pub has_positive_zero: bool,
}

/// Return exact limits for a real floating-point primitive.
///
/// Integer, predicate, complex, and structural types produce a typed diagnostic
/// rather than the upstream floating-point type switch's fatal check.
pub fn float_properties(ty: PrimitiveType) -> Result<FloatProperties> {
    let element = ty.array_element()?;
    properties(element).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidShape {
            message: format!(
                "floating-point properties require a real floating-point type: {ty:?}"
            ),
        })
    })
}

fn properties(ty: ElementType) -> Option<FloatProperties> {
    use ElementType::*;
    // Order: significand, exponent bits, bias, min_exponent, max_exponent,
    // infinity, quiet NaN, negative zero, positive zero.
    let (digits, exponent, bias, minimum, maximum, infinity, nan, negative, positive) = match ty {
        F16 => (11, 5, 15, -13, 16, true, true, true, true),
        BF16 => (8, 8, 127, -125, 128, true, true, true, true),
        F32 => (24, 8, 127, -125, 128, true, true, true, true),
        F64 => (53, 11, 1023, -1021, 1024, true, true, true, true),
        F8E3M4 => (5, 3, 3, -1, 4, true, true, true, true),
        F8E4M3 => (4, 4, 7, -5, 8, true, true, true, true),
        F8E4M3FN => (4, 4, 7, -5, 9, false, true, true, true),
        F8E4M3B11FNUZ => (4, 4, 11, -9, 5, false, true, false, true),
        F8E4M3FNUZ => (4, 4, 8, -6, 8, false, true, false, true),
        F8E5M2 => (3, 5, 15, -13, 16, true, true, true, true),
        F8E5M2FNUZ => (3, 5, 16, -14, 16, false, true, false, true),
        F8E8M0FNU => (1, 8, 127, -126, 128, false, true, false, false),
        F4E2M1FN => (2, 2, 1, 1, 3, false, false, true, true),
        F6E3M2FN => (3, 3, 3, -1, 5, false, false, true, true),
        F6E2M3FN => (4, 2, 1, 1, 3, false, false, true, true),
        Pred | S1 | S2 | S4 | S8 | S16 | S32 | S64 | U1 | U2 | U4 | U8 | U16 | U32 | U64 | C64
        | C128 => return None,
    };
    Some(FloatProperties {
        significand_width: digits,
        exponent_width: exponent,
        exponent_bias: bias,
        underflow_exponent: minimum,
        overflow_exponent: maximum,
        has_infinity: infinity,
        has_nan: nan,
        has_negative_zero: negative,
        has_positive_zero: positive,
    })
}

/// Whether the real floating-point type has infinities; false for other types.
pub fn has_infinity(ty: PrimitiveType) -> bool {
    ty.array_element()
        .ok()
        .and_then(properties)
        .is_some_and(|p| p.has_infinity)
}

/// Whether the real floating-point type has quiet NaNs; false for other types.
pub fn has_nan(ty: PrimitiveType) -> bool {
    ty.array_element()
        .ok()
        .and_then(properties)
        .is_some_and(|p| p.has_nan)
}

/// Whether the real floating-point type has negative zero; false for other types.
pub fn has_negative_zero(ty: PrimitiveType) -> bool {
    ty.array_element()
        .ok()
        .and_then(properties)
        .is_some_and(|p| p.has_negative_zero)
}

/// Whether the type has positive zero, including integers and predicates.
///
/// Complex and structural types return false, matching the pinned utility.
pub fn has_positive_zero(ty: PrimitiveType) -> bool {
    ty.array_element().is_ok_and(|element| {
        element.is_integer()
            || element == ElementType::Pred
            || properties(element).is_some_and(|p| p.has_positive_zero)
    })
}

/// Whether the pinned cast contract preserves every source value.
///
/// Supports every numeric array kind, including predicates and complex numbers.
/// Structural or invalid kinds are rejected before identity/predicate shortcuts;
/// upstream's cast matrix is defined over array kinds only.
///
/// This ports `CastPreservesValues`, not a general storage-aliasing guarantee,
/// precision ordering, NaN-payload preservation, or a new mathematical criterion.
pub fn cast_preserves_values(from: PrimitiveType, to: PrimitiveType) -> Result<bool> {
    Ok(element_cast_preserves_values(
        from.array_element()?,
        to.array_element()?,
    ))
}

fn element_cast_preserves_values(from: ElementType, to: ElementType) -> bool {
    use ElementType::*;
    if from == to {
        return true;
    }
    if to == F8E8M0FNU {
        return false;
    }
    if from == Pred {
        return true;
    }
    if to == Pred {
        return false;
    }
    if to.is_complex() {
        let component = |ty| match ty {
            C64 => F32,
            C128 => F64,
            _ => ty,
        };
        return element_cast_preserves_values(component(from), component(to));
    }
    if from.is_complex() {
        return false;
    }
    if let (Some(from), Some(to)) = (properties(from), properties(to)) {
        return from.significand_width <= to.significand_width
            && from.exponent_width <= to.exponent_width
            && (!from.has_infinity || to.has_infinity)
            && (!from.has_nan || to.has_nan)
            && (!from.has_negative_zero || to.has_negative_zero)
            && (!from.has_positive_zero || to.has_positive_zero)
            && from.underflow_exponent - from.significand_width
                >= to.underflow_exponent - to.significand_width
            && from.overflow_exponent <= to.overflow_exponent;
    }
    if !from.is_integer() {
        return false;
    }
    let from_bits = from.bit_width() - u32::from(from.is_signed());
    if let Some(to) = properties(to) {
        return i64::from(from_bits) <= i64::from(to.significand_width)
            && i64::from(from.bit_width()) - 1 < i64::from(to.overflow_exponent);
    }
    if from.is_signed() && !to.is_signed() {
        return false;
    }
    let to_bits = to.bit_width() - u32::from(to.is_signed());
    from_bits <= to_bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use PrimitiveType::*;

    // Independent expected destination sets from pinned primitive_util_test.cc:111-203.
    fn expected_casts() -> Vec<(PrimitiveType, Vec<PrimitiveType>)> {
        vec![
            (
                Pred,
                PrimitiveType::VALUES
                    .iter()
                    .copied()
                    .filter(|t| t.is_array_element() && *t != F8E8M0FNU)
                    .collect(),
            ),
            (
                S1,
                vec![
                    S1,
                    S2,
                    S4,
                    S8,
                    S16,
                    S32,
                    S64,
                    F16,
                    F32,
                    F64,
                    C64,
                    BF16,
                    C128,
                    F8E5M2,
                    F8E4M3,
                    F8E4M3FN,
                    F8E4M3B11FNUZ,
                    F8E5M2FNUZ,
                    F8E4M3FNUZ,
                    F8E3M4,
                    F4E2M1FN,
                    F6E3M2FN,
                    F6E2M3FN,
                ],
            ),
            (
                S2,
                vec![
                    S2,
                    S4,
                    S8,
                    S16,
                    S32,
                    S64,
                    F16,
                    F32,
                    F64,
                    C64,
                    BF16,
                    C128,
                    F8E5M2,
                    F8E4M3,
                    F8E4M3FN,
                    F8E4M3B11FNUZ,
                    F8E5M2FNUZ,
                    F8E4M3FNUZ,
                    F8E3M4,
                    F4E2M1FN,
                    F6E3M2FN,
                    F6E2M3FN,
                ],
            ),
            (
                S4,
                vec![
                    S4,
                    S8,
                    S16,
                    S32,
                    S64,
                    F16,
                    F32,
                    F64,
                    C64,
                    BF16,
                    C128,
                    F8E5M2,
                    F8E4M3,
                    F8E4M3FN,
                    F8E4M3B11FNUZ,
                    F8E5M2FNUZ,
                    F8E4M3FNUZ,
                    F8E3M4,
                    F6E3M2FN,
                ],
            ),
            (S8, vec![S8, S16, S32, S64, F16, F32, F64, C64, BF16, C128]),
            (S16, vec![S16, S32, S64, F32, F64, C64, C128]),
            (S32, vec![S32, S64, F64, C128]),
            (S64, vec![S64]),
            (
                U1,
                vec![
                    S2,
                    S4,
                    S8,
                    S16,
                    S32,
                    S64,
                    U1,
                    U2,
                    U4,
                    U8,
                    U16,
                    U32,
                    U64,
                    F16,
                    F32,
                    F64,
                    C64,
                    BF16,
                    C128,
                    F8E5M2,
                    F8E4M3,
                    F8E4M3FN,
                    F8E4M3B11FNUZ,
                    F8E5M2FNUZ,
                    F8E4M3FNUZ,
                    F8E3M4,
                    F4E2M1FN,
                    F6E3M2FN,
                    F6E2M3FN,
                ],
            ),
            (
                U2,
                vec![
                    S4,
                    S8,
                    S16,
                    S32,
                    S64,
                    U2,
                    U4,
                    U8,
                    U16,
                    U32,
                    U64,
                    F16,
                    F32,
                    F64,
                    C64,
                    BF16,
                    C128,
                    F8E5M2,
                    F8E4M3,
                    F8E4M3FN,
                    F8E4M3B11FNUZ,
                    F8E5M2FNUZ,
                    F8E4M3FNUZ,
                    F8E3M4,
                    F4E2M1FN,
                    F6E3M2FN,
                    F6E2M3FN,
                ],
            ),
            (
                U4,
                vec![
                    S8,
                    S16,
                    S32,
                    S64,
                    U4,
                    U8,
                    U16,
                    U32,
                    U64,
                    F16,
                    F32,
                    F64,
                    C64,
                    BF16,
                    C128,
                    F8E4M3,
                    F8E4M3FN,
                    F8E4M3B11FNUZ,
                    F8E4M3FNUZ,
                    F8E3M4,
                ],
            ),
            (
                U8,
                vec![
                    S16, S32, S64, U8, U16, U32, U64, F16, F32, F64, C64, BF16, C128,
                ],
            ),
            (U16, vec![S32, S64, U16, U32, U64, F32, F64, C64, C128]),
            (U32, vec![S64, U32, U64, F64, C128]),
            (U64, vec![U64]),
            (F16, vec![F16, F32, F64, C64, C128]),
            (F32, vec![F32, F64, C64, C128]),
            (F64, vec![F64, C128]),
            (C64, vec![C64, C128]),
            (BF16, vec![F32, F64, C64, BF16, C128]),
            (C128, vec![C128]),
            (F8E5M2, vec![F16, F32, F64, C64, BF16, C128, F8E5M2]),
            (F8E4M3, vec![F16, F32, F64, C64, BF16, C128, F8E4M3]),
            (F8E4M3FN, vec![F16, F32, F64, C64, BF16, C128, F8E4M3FN]),
            (
                F8E4M3B11FNUZ,
                vec![F16, F32, F64, C64, BF16, C128, F8E4M3B11FNUZ],
            ),
            (F8E5M2FNUZ, vec![F16, F32, F64, C64, BF16, C128, F8E5M2FNUZ]),
            (F8E4M3FNUZ, vec![F16, F32, F64, C64, BF16, C128, F8E4M3FNUZ]),
            (F8E3M4, vec![F16, F32, F64, C64, BF16, C128, F8E3M4]),
            (
                F4E2M1FN,
                vec![
                    F16, F32, F64, C64, BF16, C128, F8E5M2, F8E4M3, F8E4M3FN, F8E3M4, F4E2M1FN,
                    F6E3M2FN, F6E2M3FN,
                ],
            ),
            (
                F6E3M2FN,
                vec![
                    F16, F32, F64, C64, BF16, C128, F8E5M2, F8E4M3, F8E4M3FN, F6E3M2FN,
                ],
            ),
            (
                F6E2M3FN,
                vec![
                    F16, F32, F64, C64, BF16, C128, F8E4M3, F8E4M3FN, F8E3M4, F6E2M3FN,
                ],
            ),
            (F8E8M0FNU, vec![F32, F64, C64, BF16, C128, F8E8M0FNU]),
        ]
    }

    #[test]
    fn pinned_all_numeric_cast_matrix() {
        let rows = expected_casts();
        assert_eq!(rows.len(), ElementType::ALL.len());
        let mut float_pairs = 0;
        let mut pairs = 0;
        for &from in ElementType::ALL {
            let from = PrimitiveType::from(from);
            let expected = &rows.iter().find(|(ty, _)| *ty == from).unwrap().1;
            for &to in ElementType::ALL {
                let to = PrimitiveType::from(to);
                assert_eq!(
                    cast_preserves_values(from, to).unwrap(),
                    expected.contains(&to),
                    "{from:?} -> {to:?}",
                );
                pairs += 1;
                float_pairs += usize::from(from.is_floating() && to.is_floating());
            }
        }
        assert_eq!(pairs, 1024);
        assert_eq!(float_pairs, 225);
    }

    #[test]
    fn exact_float_limits_and_special_values() {
        // Pinned FloatTypeTests plus numeric_limits and zero traits.
        let expected = [
            (F32, 8, 24, 127, -125, 128, true, true, true, true),
            (BF16, 8, 8, 127, -125, 128, true, true, true, true),
            (F4E2M1FN, 2, 2, 1, 1, 3, false, false, true, true),
            (F6E3M2FN, 3, 3, 3, -1, 5, false, false, true, true),
            (F6E2M3FN, 2, 4, 1, 1, 3, false, false, true, true),
            (F8E3M4, 3, 5, 3, -1, 4, true, true, true, true),
            (F8E4M3, 4, 4, 7, -5, 8, true, true, true, true),
            (F8E4M3FN, 4, 4, 7, -5, 9, false, true, true, true),
            (F8E4M3B11FNUZ, 4, 4, 11, -9, 5, false, true, false, true),
            (F8E4M3FNUZ, 4, 4, 8, -6, 8, false, true, false, true),
            (F8E5M2, 5, 3, 15, -13, 16, true, true, true, true),
            (F8E5M2FNUZ, 5, 3, 16, -14, 16, false, true, false, true),
            (F8E8M0FNU, 8, 1, 127, -126, 128, false, true, false, false),
            (F16, 5, 11, 15, -13, 16, true, true, true, true),
            (F64, 11, 53, 1023, -1021, 1024, true, true, true, true),
        ];
        assert_eq!(
            expected.len(),
            ElementType::ALL.iter().filter(|ty| ty.is_float()).count()
        );
        for (
            ty,
            exponent_width,
            significand_width,
            exponent_bias,
            underflow_exponent,
            overflow_exponent,
            has_infinity_expected,
            has_nan_expected,
            has_negative_zero_expected,
            has_positive_zero_expected,
        ) in expected
        {
            assert_eq!(
                float_properties(ty).unwrap(),
                FloatProperties {
                    significand_width,
                    exponent_width,
                    exponent_bias,
                    underflow_exponent,
                    overflow_exponent,
                    has_infinity: has_infinity_expected,
                    has_nan: has_nan_expected,
                    has_negative_zero: has_negative_zero_expected,
                    has_positive_zero: has_positive_zero_expected,
                },
                "{ty:?}"
            );
            assert_eq!(has_infinity(ty), has_infinity_expected, "{ty:?}");
            assert_eq!(has_nan(ty), has_nan_expected, "{ty:?}");
            assert_eq!(has_negative_zero(ty), has_negative_zero_expected, "{ty:?}");
            assert_eq!(has_positive_zero(ty), has_positive_zero_expected, "{ty:?}");
        }
    }

    #[test]
    fn non_float_queries_and_invalid_casts_are_explicit() {
        for &ty in PrimitiveType::VALUES {
            if !ty.is_floating() {
                let error = float_properties(ty).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<Diagnostic>(),
                    Some(Diagnostic::InvalidShape { .. })
                ));
                assert!(!has_infinity(ty));
                assert!(!has_nan(ty));
                assert!(!has_negative_zero(ty));
                assert_eq!(has_positive_zero(ty), ty == Pred || ty.is_integral());
            }
            if !ty.is_array_element() {
                // Invalid identities and predicate shortcuts must not bypass diagnostics.
                for (from, to) in [(ty, ty), (ty, F32), (F32, ty), (Pred, ty), (ty, Pred)] {
                    let error = cast_preserves_values(from, to).unwrap_err();
                    assert!(matches!(
                        error.err.downcast_ref::<Diagnostic>(),
                        Some(Diagnostic::InvalidShape { .. })
                    ));
                }
            }
        }
    }
}
