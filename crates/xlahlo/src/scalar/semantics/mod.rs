// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Based on OpenXLA's HLO evaluator, literal conversions and shape inference.
// Sub-byte integer behavior follows TensorFlow's ml_dtypes conventions.

//! Raw-bit native scalar values and their HLO semantics.
//!
//! Sub-byte integer behavior follows TensorFlow's ml_dtypes conventions for
//! byte storage and signed extension.
//! Floating/complex operations delegate the qualified, bounded raw numerical
//! kernel, including ReducePrecision. Transcendentals remain unsupported.
//! Arithmetic NaN payload/sign routing follows the documented MSVC/UCRT
//! compatibility policy; this is not a universal host ABI claim.
use pliron::{context::Context, result::Result};

mod context;
mod integer;

pub use context::{
    EvaluationFloatContext, EvaluationRoundingMode, reduce_precision, remainder_with_context,
    rounding_unary, sqrt_with_context,
};
use integer::Integer;

#[cfg(test)]
use crate::attributes::{ComparisonOrder, Direction};
use crate::{
    Diagnostic, literal::Literal, literal::semantics, opcode::Opcode, shape, types::ElementType,
};

fn invalid(message: impl Into<String>) -> pliron::result::Error {
    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
        message: message.into(),
    })
}

fn unsupported(opcode: Opcode, element: ElementType) -> pliron::result::Error {
    pliron::verify_error_noloc!(Diagnostic::Unsupported {
        message: format!("native scalar semantics does not support {opcode} on {element:?}"),
    })
}

fn require_format(element: ElementType) -> Result<()> {
    if element.is_integer() || element == ElementType::Pred {
        Ok(())
    } else {
        Err(invalid(
            "integer helper requires an integer or predicate format",
        ))
    }
}

fn mask(element: ElementType) -> u64 {
    u64::MAX >> (64 - element.bit_width())
}

/// An independently owned, canonical significant-bit native scalar.
///
/// All 32 native formats are represented without loss, including C128 and
/// U64 above i64::MAX. Padding bits are not part of a scalar value:
/// use `literal::semantics` for raw storage-preserving operations.
/// Complex encodings place the real component in the low half, imaginary high.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scalar {
    element: ElementType,
    bits: u128,
}

impl Scalar {
    /// Construct from significant encoding bits, rejecting excess bits.
    pub fn from_bits(element: ElementType, bits: u128) -> Result<Self> {
        if element.bit_width() < 128 && bits >> element.bit_width() != 0 {
            return Err(invalid(
                "scalar bits exceed the element's significant width",
            ));
        }
        Ok(Self { element, bits })
    }

    /// Read one active logical element, masking unused sub-byte storage bits.
    pub fn read(ctx: &Context, literal: &Literal, index: &[usize]) -> Result<Self> {
        let element = shape::element_type(ctx, literal.shape)?;
        let bytes = semantics::read_element(ctx, literal, index)?;
        let mut encoding = [0u8; 16];
        encoding[..bytes.len()].copy_from_slice(bytes);
        let bits = u128::from_le_bytes(encoding);
        let bits = if element.bit_width() == 128 {
            bits
        } else {
            bits & ((1u128 << element.bit_width()) - 1)
        };
        Self::from_bits(element, bits)
    }

    /// The native element format, retaining signedness independently of bits.
    pub fn element(self) -> ElementType {
        self.element
    }

    /// Canonical significant bits; signed values use two's complement.
    pub fn bits(self) -> u128 {
        self.bits
    }

    /// Write one element of the same format, leaving the literal unchanged on error.
    ///
    /// Arithmetic values write zero padding; raw storage copy remains a separate
    /// responsibility of `literal::semantics`.
    pub fn write(self, ctx: &Context, literal: &mut Literal, index: &[usize]) -> Result<()> {
        if shape::element_type(ctx, literal.shape)? != self.element {
            return Err(invalid("scalar write requires the same element format"));
        }
        let encoding = self.bits.to_le_bytes();
        semantics::write_element(
            ctx,
            literal,
            index,
            &encoding[..self.element.bit_width().div_ceil(8) as usize],
        )
    }

    /// Construct a rank-zero owned literal without numerical conversion.
    pub fn to_literal(self, ctx: &Context) -> Result<Literal> {
        Literal::from_bits(ctx, shape::scalar(ctx, self.element)?, &[self.bits])
    }

    fn integer(self) -> Result<Integer> {
        require_format(self.element)?;
        let bits = u64::try_from(self.bits)
            .map_err(|_| invalid("integer encoding exceeds its native width"))?;
        Ok(Integer::new(self.element, bits))
    }
}

fn same_format(lhs: Scalar, rhs: Scalar) -> Result<()> {
    if lhs.element == rhs.element {
        Ok(())
    } else {
        Err(invalid("scalar operands require identical element formats"))
    }
}

mod operations;
pub use operations::{binary, bitcast_convert, clamp, compare, convert, select, unary};

fn numerical_result(result: (ElementType, u128), expected: ElementType) -> Result<Scalar> {
    if result.0 != expected {
        return Err(invalid(
            "numerical kernel returned an incompatible element format",
        ));
    }
    Scalar::from_bits(result.0, result.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Dimension;
    use pliron::common_traits::Verify;

    mod numeric {

        use super::*;

        use crate::shape::{Dimension, Layout};

        fn context() -> Context {
            let mut ctx = Context::new();

            crate::register(&mut ctx);

            ctx
        }

        fn mask(element: ElementType) -> u128 {
            u128::MAX >> (128 - element.bit_width())
        }

        fn raw(element: ElementType, bits: u128) -> Scalar {
            Scalar::from_bits(element, bits).unwrap()
        }

        fn inference_error(result: Result<Scalar>) {
            let error = result.unwrap_err();

            assert!(
                matches!(
                    error.err.downcast_ref::<Diagnostic>(),
                    Some(Diagnostic::Inference { .. } | Diagnostic::Unsupported { .. })
                ),
                "{error}"
            );
        }

        #[test]

        fn explicit_sqrt_preserves_context_boundary_and_ordinary_unary_admission() {
            let context = EvaluationFloatContext {
                rounding: EvaluationRoundingMode::NearestEven,

                flush_to_zero: false,

                denormals_are_zero: false,
            };

            for &element in ElementType::ALL {
                let result = sqrt_with_context(raw(element, 0), context);

                assert_eq!(result.is_ok(), element.is_float());

                if !element.is_float() {
                    inference_error(result);
                }

                inference_error(unary(Opcode::Sqrt, raw(element, 0)));
            }

            for (element, input, expected) in [
                (ElementType::F32, 0x8000_0000, 0x8000_0000),
                (ElementType::F32, 0xff80_1234, 0xffc0_1234),
                (
                    ElementType::F64,
                    0xfff0_0000_0000_0000,
                    0xfff8_0000_0000_0000,
                ),
                (
                    ElementType::F64,
                    0x7ff0_0000_0000_1234,
                    0x7ff8_0000_0000_1234,
                ),
            ] {
                assert_eq!(
                    sqrt_with_context(raw(element, input), context).unwrap(),
                    raw(element, expected)
                );
            }

            let mut rejected = 0;

            for rounding in [
                EvaluationRoundingMode::NearestEven,
                EvaluationRoundingMode::TowardZero,
                EvaluationRoundingMode::TowardPositive,
                EvaluationRoundingMode::TowardNegative,
            ] {
                for flush_to_zero in [false, true] {
                    for denormals_are_zero in [false, true] {
                        let invalid_context = EvaluationFloatContext {
                            rounding,

                            flush_to_zero,

                            denormals_are_zero,
                        };

                        if invalid_context == context {
                            continue;
                        }

                        for &element in ElementType::ALL.iter().filter(|element| element.is_float())
                        {
                            let error =
                                sqrt_with_context(raw(element, 0), invalid_context).unwrap_err();

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

        fn explicit_remainder_preserves_context_boundary_and_ordinary_binary_admission() {
            let context = EvaluationFloatContext {
                rounding: EvaluationRoundingMode::NearestEven,

                flush_to_zero: false,

                denormals_are_zero: false,
            };

            for &element in ElementType::ALL {
                let result = remainder_with_context(raw(element, 0), raw(element, 0), context);

                assert_eq!(result.is_ok(), element.is_float());

                if !element.is_float() {
                    inference_error(result);
                }

                if element.is_float() || element.is_complex() {
                    inference_error(binary(Opcode::Remainder, raw(element, 0), raw(element, 0)));
                }
            }

            let error =
                remainder_with_context(raw(ElementType::F32, 0), raw(ElementType::F64, 0), context)
                    .unwrap_err();

            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::InvalidShape { .. })
            ));

            for (element, lhs, rhs, expected) in [
                (ElementType::F32, 0xc000_0000, 0x3f80_0000, 0x8000_0000),
                (ElementType::F32, 0x7f80_0000, 0xff80_5678, 0xffc0_0000),
                (
                    ElementType::F64,
                    0x7ff0_0000_0000_1234,
                    0xfff0_0000_0000_5678,
                    0xfff0_0000_0000_5678,
                ),
                (
                    ElementType::F64,
                    0xc000_0000_0000_0000,
                    0x3ff0_0000_0000_0000,
                    0x8000_0000_0000_0000,
                ),
            ] {
                assert_eq!(
                    remainder_with_context(raw(element, lhs), raw(element, rhs), context).unwrap(),
                    raw(element, expected)
                );
            }

            assert_eq!(
                binary(
                    Opcode::Remainder,
                    raw(ElementType::S32, 7),
                    raw(ElementType::S32, 4)
                )
                .unwrap(),
                raw(ElementType::S32, 3)
            );
        }

        #[test]

        fn all_32_formats_preserve_every_significant_bit_in_owned_scalar_and_literal() {
            let ctx = context();

            assert_eq!(ElementType::ALL.len(), 32);

            for &element in ElementType::ALL {
                for bits in [0, 1, mask(element)] {
                    let scalar = raw(element, bits);

                    let literal = scalar.to_literal(&ctx).unwrap();

                    literal.verify(&ctx).unwrap();

                    assert_eq!(shape::rank(&ctx, literal.shape).unwrap(), 0);

                    assert_eq!(literal.bits(&ctx).unwrap(), [bits]);

                    assert_eq!(Scalar::read(&ctx, &literal, &[]).unwrap(), scalar);
                }
            }
        }

        #[test]

        fn constructor_rejects_excess_bits_for_all_31_narrower_formats() {
            let mut count = 0;

            for &element in ElementType::ALL {
                if element.bit_width() < 128 {
                    let error =
                        Scalar::from_bits(element, 1u128 << element.bit_width()).unwrap_err();

                    assert!(matches!(
                        error.err.downcast_ref::<Diagnostic>(),
                        Some(Diagnostic::InvalidShape { .. })
                    ));

                    count += 1;
                }
            }

            assert_eq!(count, 31);

            assert_eq!(raw(ElementType::C128, u128::MAX).bits(), u128::MAX);
        }

        #[test]

        fn all_32_formats_read_and_write_nondefault_physical_layout_exactly() {
            let ctx = context();

            for &element in ElementType::ALL {
                let shape = shape::array(&ctx, element, vec![2, 2]).unwrap();

                let shape = shape::with_layout(
                    &ctx,
                    shape,
                    Layout {
                        minor_to_major: vec![0, 1],

                        ..Layout::descending(2)
                    },
                )
                .unwrap();

                let mut literal = Literal::from_bits(&ctx, shape, &[0, 1, 0, 1]).unwrap();

                assert_eq!(Scalar::read(&ctx, &literal, &[1, 0]).unwrap().bits(), 1);

                let value = raw(element, mask(element));

                value.write(&ctx, &mut literal, &[0, 1]).unwrap();

                assert_eq!(Scalar::read(&ctx, &literal, &[0, 1]).unwrap(), value);

                assert_eq!(Scalar::read(&ctx, &literal, &[1, 0]).unwrap().bits(), 1);
            }
        }

        #[test]

        fn all_32_formats_failed_write_preserves_destination_atomically() {
            let ctx = context();

            for &element in ElementType::ALL {
                let value = raw(element, mask(element));

                let mut literal = raw(element, 0).to_literal(&ctx).unwrap();

                let before = literal.clone();

                assert!(value.write(&ctx, &mut literal, &[0]).is_err());

                assert_eq!(literal, before);

                let wrong = raw(
                    if element == ElementType::U8 {
                        ElementType::S8
                    } else {
                        ElementType::U8
                    },
                    1,
                );

                assert!(wrong.write(&ctx, &mut literal, &[]).is_err());

                assert_eq!(literal, before);
            }
        }

        #[test]

        fn c128_uses_all_16_bytes_without_component_loss() {
            let ctx = context();

            let bits = 0xfff0_0000_0000_0042_7ff0_0000_0000_0123u128;

            let value = raw(ElementType::C128, bits);

            let mut literal = value.to_literal(&ctx).unwrap();

            assert_eq!(literal.storage.len(), 16);

            assert_eq!(literal.storage.as_slice(), bits.to_le_bytes());

            let changed = raw(ElementType::C128, bits.rotate_left(64));

            changed.write(&ctx, &mut literal, &[]).unwrap();

            assert_eq!(Scalar::read(&ctx, &literal, &[]).unwrap(), changed);
        }

        #[test]

        fn all_subbyte_formats_mask_physical_padding_but_reject_noncanonical_construction() {
            let ctx = context();

            let mut count = 0;

            for &element in ElementType::ALL {
                if element.bit_width() < 8 && element != ElementType::Pred {
                    let shape = shape::scalar(&ctx, element).unwrap();

                    let literal = Literal::from_storage_bytes(&ctx, shape, vec![255]).unwrap();

                    assert!(Scalar::from_bits(element, 255).is_err());

                    let value = Scalar::read(&ctx, &literal, &[]).unwrap();

                    assert_eq!(value.bits(), mask(element));

                    assert_eq!(literal.storage, [255]);

                    let mut written = literal.clone();

                    value.write(&ctx, &mut written, &[]).unwrap();

                    assert_eq!(written.storage, [mask(element) as u8]);

                    count += 1;
                }
            }

            assert_eq!(count, 9);
        }

        #[test]

        fn dynamic_c128_raw_access_checks_active_extent_and_preserves_runtime_size() {
            let ctx = context();

            let shape =
                shape::dynamic_array(&ctx, ElementType::C128, vec![Dimension::Bounded(3)]).unwrap();

            let mut literal =
                Literal::from_dynamic_storage_bytes(&ctx, shape, vec![0; 48], vec![1]).unwrap();

            let before = literal.clone();

            assert!(
                raw(ElementType::C128, u128::MAX)
                    .write(&ctx, &mut literal, &[1])
                    .is_err()
            );

            assert_eq!(literal, before);

            raw(ElementType::C128, u128::MAX)
                .write(&ctx, &mut literal, &[0])
                .unwrap();

            assert_eq!(literal.dynamic_sizes, before.dynamic_sizes);

            assert_eq!(
                Scalar::read(&ctx, &literal, &[0]).unwrap().bits(),
                u128::MAX
            );
        }

        #[test]

        fn all_32_selects_are_raw_and_preserve_signaling_nans_and_c128_components() {
            for &element in ElementType::ALL {
                let on_true = raw(element, mask(element));

                let on_false = raw(element, mask(element) ^ 1);

                assert_eq!(
                    select(raw(ElementType::Pred, 1), on_true, on_false).unwrap(),
                    on_true
                );

                assert_eq!(
                    select(raw(ElementType::Pred, 0), on_true, on_false).unwrap(),
                    on_false
                );

                assert!(select(raw(ElementType::U8, 1), on_true, on_false).is_err());
            }
        }

        #[test]

        fn all_1024_raw_bitcast_pairs_check_storage_width_and_destination_encoding() {
            for &source in ElementType::ALL {
                for &destination in ElementType::ALL {
                    for bits in [0, 1, mask(source)] {
                        let result = bitcast_convert(raw(source, bits), destination);

                        let admitted = source.storage_bit_width()
                            == destination.storage_bit_width()
                            && bits <= mask(destination);

                        assert_eq!(
                            result.is_ok(),
                            admitted,
                            "{source:?}->{destination:?} {bits:x}"
                        );

                        if admitted {
                            let result = result.unwrap();

                            assert_eq!(result.element(), destination);

                            assert_eq!(result.bits(), bits);
                        }
                    }
                }
            }
        }

        #[test]

        fn float_bitcasts_preserve_signaling_nan_and_signed_zero_without_quieting() {
            for (integer, float, bits) in [
                (ElementType::U32, ElementType::F32, 0x7f80_0042),
                (ElementType::U32, ElementType::F32, 0x8000_0000),
                (ElementType::U64, ElementType::F64, 0x7ff0_0000_0000_0042),
                (ElementType::U64, ElementType::F64, 0x8000_0000_0000_0000),
            ] {
                let value = bitcast_convert(raw(integer, bits), float).unwrap();

                assert_eq!(value.bits(), bits);

                assert_eq!(bitcast_convert(value, integer).unwrap().bits(), bits);
            }
        }

        #[test]

        fn c64_and_c128_components_project_and_pack_without_quieting_or_sign_changes() {
            for (component, complex, real, imag, shift) in [
                (
                    ElementType::F32,
                    ElementType::C64,
                    0x7f80_0042u128,
                    0x8000_0000u128,
                    32,
                ),
                (
                    ElementType::F64,
                    ElementType::C128,
                    0x7ff0_0000_0000_0042u128,
                    0x8000_0000_0000_0000u128,
                    64,
                ),
            ] {
                let value =
                    binary(Opcode::Complex, raw(component, real), raw(component, imag)).unwrap();

                assert_eq!(value.element(), complex);

                assert_eq!(value.bits(), real | (imag << shift));

                assert_eq!(unary(Opcode::Real, value).unwrap(), raw(component, real));

                assert_eq!(unary(Opcode::Imag, value).unwrap(), raw(component, imag));
            }
        }

        #[test]

        fn complex_abs_changes_result_type_and_returns_component_magnitude() {
            let c64 = raw(
                ElementType::C64,
                u128::from(3f32.to_bits()) | (u128::from(4f32.to_bits()) << 32),
            );

            let c128 = raw(
                ElementType::C128,
                u128::from(3f64.to_bits()) | (u128::from(4f64.to_bits()) << 64),
            );

            assert_eq!(
                unary(Opcode::Abs, c64).unwrap(),
                raw(ElementType::F32, u128::from(5f32.to_bits()))
            );

            assert_eq!(
                unary(Opcode::Abs, c128).unwrap(),
                raw(ElementType::F64, u128::from(5f64.to_bits()))
            );
        }

        #[test]

        fn real_is_finite_returns_pred_and_complex_is_finite_remains_unsupported() {
            for (element, finite, infinity) in [
                (ElementType::F32, u128::from(1f32.to_bits()), 0x7f80_0000),
                (
                    ElementType::F64,
                    u128::from(1f64.to_bits()),
                    0x7ff0_0000_0000_0000,
                ),
            ] {
                assert_eq!(
                    unary(Opcode::IsFinite, raw(element, finite)).unwrap(),
                    raw(ElementType::Pred, 1)
                );

                assert_eq!(
                    unary(Opcode::IsFinite, raw(element, infinity)).unwrap(),
                    raw(ElementType::Pred, 0)
                );
            }

            inference_error(unary(Opcode::IsFinite, raw(ElementType::C128, 0)));
        }

        #[test]

        fn every_complex_to_noncomplex_convert_is_explicit_inference_error() {
            for source in [ElementType::C64, ElementType::C128] {
                for &destination in ElementType::ALL {
                    if !destination.is_complex() {
                        inference_error(convert(raw(source, 0), destination));
                    }
                }
            }
        }

        #[test]

        fn float_clamp_uses_low_value_high_nan_priority_not_nested_min_max_priority() {
            let low = raw(ElementType::F32, 0x7fc0_0011);

            let value = raw(ElementType::F32, 0x7fc0_0022);

            let high = raw(ElementType::F32, 0x7fc0_0033);

            let zero = raw(ElementType::F32, 0);

            assert_eq!(clamp(low, value, high).unwrap(), low);

            assert_eq!(clamp(zero, value, high).unwrap(), value);

            assert_eq!(clamp(zero, zero, high).unwrap(), high);
        }

        #[test]

        fn real_nan_compare_orders_and_complex_ordered_comparisons_remain_distinct() {
            let nan = raw(ElementType::F32, 0x7fc0_0042);

            let zero = raw(ElementType::F32, 0);

            assert_eq!(
                compare(Direction::Eq, ComparisonOrder::Partial, nan, nan)
                    .unwrap()
                    .bits(),
                0
            );

            assert_eq!(
                compare(Direction::Eq, ComparisonOrder::Weak, nan, nan)
                    .unwrap()
                    .bits(),
                1
            );

            assert_eq!(
                compare(Direction::Gt, ComparisonOrder::Total, nan, zero)
                    .unwrap()
                    .bits(),
                1
            );

            for order in [
                ComparisonOrder::Partial,
                ComparisonOrder::Total,
                ComparisonOrder::Weak,
            ] {
                inference_error(compare(
                    Direction::Lt,
                    order,
                    raw(ElementType::C128, 0),
                    raw(ElementType::C128, 0),
                ));
            }
        }

        #[test]

        fn public_reduce_precision_preserves_original_compute_nan_or_returns_positive_infinity() {
            for (element, bits, infinity) in [
                (ElementType::F32, 0xff80_0042, 0x7f80_0000),
                (
                    ElementType::F64,
                    0xfff0_0000_0000_0042,
                    0x7ff0_0000_0000_0000,
                ),
            ] {
                let input = raw(element, bits);

                for (mantissa_bits, expected) in [(1, bits), (0, infinity)] {
                    let output = reduce_precision(
                        input,
                        &crate::attributes::ReducePrecision {
                            exponent_bits: 1,

                            mantissa_bits,
                        },
                    )
                    .unwrap();

                    assert_eq!(output, raw(element, expected));

                    assert_eq!(input.bits(), bits);
                }
            }
        }

        #[test]

        fn public_reduce_precision_has_no_raw_identity_shortcut_for_narrow_nan() {
            for (element, bits, expected) in [
                (ElementType::F16, 0x7c01, 0x7e00),
                (ElementType::BF16, 0xff81, 0xffc0),
            ] {
                assert_eq!(
                    reduce_precision(
                        raw(element, bits),
                        &crate::attributes::ReducePrecision {
                            exponent_bits: i32::MAX,

                            mantissa_bits: i32::MAX,
                        }
                    )
                    .unwrap(),
                    raw(element, expected),
                );
            }
        }

        #[test]

        fn public_reduce_precision_rounds_before_exponent_flush_and_preserves_signed_zero() {
            let precision = crate::attributes::ReducePrecision {
                exponent_bits: 5,

                mantissa_bits: 10,
            };

            for (bits, expected) in [
                (0x387f_f000, 0x3880_0000),
                (0x387f_efff, 0),
                (0xb87f_efff, 0x8000_0000),
            ] {
                assert_eq!(
                    reduce_precision(raw(ElementType::F32, bits), &precision).unwrap(),
                    raw(ElementType::F32, expected)
                );
            }
        }

        #[test]

        fn public_reduce_precision_all_32_types_have_exact_admission_and_attribute_bounds() {
            for &element in ElementType::ALL {
                let result = reduce_precision(
                    raw(element, 0),
                    &crate::attributes::ReducePrecision {
                        exponent_bits: 1,

                        mantissa_bits: 0,
                    },
                );

                assert_eq!(result.is_ok(), element.is_float(), "{element:?}");

                if !element.is_float() {
                    inference_error(result);
                } else {
                    for (exponent_bits, mantissa_bits) in [(0, 0), (1, -1), (i32::MIN, i32::MIN)] {
                        let error = reduce_precision(
                            raw(element, 0),
                            &crate::attributes::ReducePrecision {
                                exponent_bits,

                                mantissa_bits,
                            },
                        )
                        .unwrap_err();

                        assert!(matches!(
                            error.err.downcast_ref::<Diagnostic>(),
                            Some(Diagnostic::InvalidShape { .. })
                        ));
                    }
                }
            }
        }

        #[test]

        fn public_rounding_distinguishes_half_ties_signed_zero_and_signaling_nans() {
            for (element, half, zero, one, snan, qnan) in [
                (
                    ElementType::F32,
                    0xbf00_0000,
                    0x8000_0000,
                    0xbf80_0000,
                    0xff80_1234,
                    0xffc0_1234,
                ),
                (
                    ElementType::F64,
                    0xbfe0_0000_0000_0000,
                    0x8000_0000_0000_0000,
                    0xbff0_0000_0000_0000,
                    0xfff0_0000_0000_1234,
                    0xfff8_0000_0000_1234,
                ),
            ] {
                for (op, expected) in [
                    (Opcode::RoundNearestAfz, one),
                    (Opcode::RoundNearestEven, zero),
                    (Opcode::Ceil, zero),
                    (Opcode::Floor, one),
                ] {
                    assert_eq!(
                        rounding_unary(op, raw(element, half), EvaluationRoundingMode::NearestEven)
                            .unwrap(),
                        raw(element, expected)
                    );

                    let expected_nan = if op == Opcode::RoundNearestAfz {
                        snan
                    } else {
                        qnan
                    };

                    assert_eq!(
                        rounding_unary(op, raw(element, snan), EvaluationRoundingMode::NearestEven)
                            .unwrap(),
                        raw(element, expected_nan)
                    );

                    if op != Opcode::RoundNearestEven {
                        assert_eq!(
                            unary(op, raw(element, half)).unwrap(),
                            raw(element, expected)
                        );
                    }
                }
            }
        }
    }

    const INTEGERS: [ElementType; 14] = [
        ElementType::S1,
        ElementType::S2,
        ElementType::S4,
        ElementType::S8,
        ElementType::S16,
        ElementType::S32,
        ElementType::S64,
        ElementType::U1,
        ElementType::U2,
        ElementType::U4,
        ElementType::U8,
        ElementType::U16,
        ElementType::U32,
        ElementType::U64,
    ];

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn formats() -> impl Iterator<Item = ElementType> {
        INTEGERS.into_iter().chain([ElementType::Pred])
    }

    fn significant_mask(element: ElementType) -> u64 {
        ((1u128 << element.bit_width()) - 1) as u64
    }

    fn scalar(element: ElementType, value: i128) -> Scalar {
        Scalar::from_bits(
            element,
            u128::from((value as u64) & significant_mask(element)),
        )
        .unwrap()
    }

    fn raw(element: ElementType, bits: u64) -> Scalar {
        Scalar::from_bits(element, u128::from(bits)).unwrap()
    }

    // Keep the qualified integer reference arithmetic and its u64 assertions
    // unchanged while checking that the widened facade never adds high bits.
    trait IntegerEncoding {
        fn integer_bits(self) -> u64;
    }

    impl IntegerEncoding for Scalar {
        fn integer_bits(self) -> u64 {
            u64::try_from(self.bits()).unwrap()
        }
    }

    fn limits(element: ElementType) -> (i128, i128) {
        let width = element.bit_width();
        if element.is_signed() {
            (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
        } else {
            (0, (1i128 << width) - 1)
        }
    }

    fn mathematical(element: ElementType, bits: u64) -> i128 {
        let width = element.bit_width();
        let value = i128::from(bits);
        if element.is_signed() && value >= 1i128 << (width - 1) {
            value - (1i128 << width)
        } else {
            value
        }
    }

    fn samples(element: ElementType) -> Vec<u64> {
        let maximum = significant_mask(element);
        let mut values = vec![0, 1, maximum / 2, maximum / 2 + 1, maximum - 1, maximum];
        values.sort_unstable();
        values.dedup();
        values
    }

    fn assert_binary(op: Opcode, element: ElementType, lhs: i128, rhs: i128, expected: i128) {
        let result = binary(op, scalar(element, lhs), scalar(element, rhs)).unwrap();
        assert_eq!(result, scalar(element, expected), "{op} {element:?}");
    }

    #[test]
    fn u64_full_range_is_not_extracted_through_i64() {
        let ctx = context();
        let value = raw(ElementType::U64, u64::MAX);
        let literal = value.to_literal(&ctx).unwrap();
        assert!(semantics::scalar_integer(&ctx, &literal).is_err());
        assert_eq!(Scalar::read(&ctx, &literal, &[]).unwrap(), value);
    }

    #[test]
    fn read_rejects_malformed_predicate_before_masking() {
        let ctx = context();
        let mut literal = raw(ElementType::Pred, 1).to_literal(&ctx).unwrap();
        literal.storage[0] = 3;
        assert!(Scalar::read(&ctx, &literal, &[]).is_err());
    }

    #[test]
    fn read_rejects_malformed_storage() {
        let ctx = context();
        let mut literal = raw(ElementType::U64, u64::MAX).to_literal(&ctx).unwrap();
        literal.storage.pop();
        assert!(Scalar::read(&ctx, &literal, &[]).is_err());
    }

    #[test]
    fn read_rejects_tuple_and_token_without_panicking() {
        let ctx = context();
        let token = Literal::token(&ctx);
        let tuple = Literal::tuple(&ctx, vec![token.clone()]).unwrap();
        assert!(Scalar::read(&ctx, &token, &[]).is_err());
        assert!(Scalar::read(&ctx, &tuple, &[0]).is_err());
    }

    #[test]
    fn read_checks_rank_and_active_dynamic_extent() {
        let ctx = context();
        let ty = shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(3)]).unwrap();
        let literal =
            Literal::from_dynamic_storage_bytes(&ctx, ty, vec![10, 20, 30], vec![1]).unwrap();
        assert_eq!(
            Scalar::read(&ctx, &literal, &[0]).unwrap().integer_bits(),
            10
        );
        assert!(Scalar::read(&ctx, &literal, &[1]).is_err());
        assert!(Scalar::read(&ctx, &literal, &[]).is_err());
        assert!(Scalar::read(&ctx, &literal, &[0, 0]).is_err());
    }

    #[test]
    fn write_preserves_dynamic_sizes_inactive_data_and_layout() {
        let ctx = context();
        let ty = shape::dynamic_array(&ctx, ElementType::S4, vec![Dimension::Bounded(3)]).unwrap();
        let mut literal =
            Literal::from_dynamic_storage_bytes(&ctx, ty, vec![0xff, 0xaa, 0xbb], vec![1]).unwrap();
        let shape = literal.shape;
        scalar(ElementType::S4, 3)
            .write(&ctx, &mut literal, &[0])
            .unwrap();
        assert_eq!(literal.storage, vec![3, 0xaa, 0xbb]);
        assert_eq!(literal.dynamic_sizes, vec![1]);
        assert_eq!(literal.shape, shape);
        let before = literal.clone();
        assert!(
            scalar(ElementType::S4, 0)
                .write(&ctx, &mut literal, &[1])
                .is_err()
        );
        assert_eq!(literal, before);
    }

    #[test]
    fn binary_operations_reject_mixed_element_types() {
        for opcode in [
            Opcode::Add,
            Opcode::Divide,
            Opcode::And,
            Opcode::ShiftLeft,
            Opcode::Mulhi,
        ] {
            assert!(
                binary(
                    opcode,
                    scalar(ElementType::S32, 1),
                    scalar(ElementType::U32, 1)
                )
                .is_err()
            );
        }
        assert!(
            binary(
                Opcode::Add,
                raw(ElementType::F32, 0),
                raw(ElementType::F64, 0)
            )
            .is_err()
        );
        assert!(
            binary(
                Opcode::Add,
                raw(ElementType::C64, 0),
                raw(ElementType::C128, 0)
            )
            .is_err()
        );
    }

    #[test]
    fn integer_add_subtract_and_multiply_wrap_at_format_width() {
        for element in INTEGERS {
            let (min, max) = limits(element);
            assert_binary(Opcode::Add, element, max, 1, max + 1);
            assert_binary(Opcode::Subtract, element, min, 1, min - 1);
            assert_binary(Opcode::Multiply, element, -1, -1, 1);
        }

        for element in INTEGERS
            .into_iter()
            .filter(|element| element.bit_width() <= 4)
        {
            let maximum = significant_mask(element);
            for lhs in 0..=maximum {
                for rhs in 0..=maximum {
                    for (op, expected) in [
                        (Opcode::Add, lhs + rhs),
                        (Opcode::Subtract, lhs.wrapping_sub(rhs)),
                        (Opcode::Multiply, lhs * rhs),
                    ] {
                        assert_eq!(
                            binary(op, raw(element, lhs), raw(element, rhs))
                                .unwrap()
                                .integer_bits(),
                            expected & maximum,
                            "{op} {element:?} {lhs} {rhs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn negate_wraps_and_is_an_involution_for_all_integer_formats() {
        for element in INTEGERS {
            for bits in samples(element) {
                let value = raw(element, bits);
                let negated = unary(Opcode::Negate, value).unwrap();
                assert_eq!(
                    negated.integer_bits(),
                    0u64.wrapping_sub(bits) & significant_mask(element)
                );
                assert_eq!(unary(Opcode::Negate, negated).unwrap(), value);
            }
        }
    }

    #[test]
    fn abs_of_signed_minimum_wraps_without_host_overflow() {
        for element in INTEGERS.into_iter().filter(|element| element.is_signed()) {
            let (min, max) = limits(element);
            assert_eq!(
                unary(Opcode::Abs, scalar(element, min)).unwrap(),
                scalar(element, min)
            );
            assert_eq!(
                unary(Opcode::Abs, scalar(element, max)).unwrap(),
                scalar(element, max)
            );
            assert_eq!(
                unary(Opcode::Abs, scalar(element, -1)).unwrap(),
                scalar(element, 1)
            );
        }
    }

    #[test]
    fn sign_preserves_zero_and_signed_polarity_all_seven_formats() {
        for element in INTEGERS.into_iter().filter(|element| element.is_signed()) {
            for bits in samples(element) {
                let value = mathematical(element, bits);
                assert_eq!(
                    unary(Opcode::Sign, raw(element, bits)).unwrap(),
                    scalar(element, value.signum())
                );
            }
        }
    }

    #[test]
    fn predicate_not_is_logical_not() {
        for bits in 0..=1 {
            assert_eq!(
                unary(Opcode::Not, raw(ElementType::Pred, bits))
                    .unwrap()
                    .integer_bits(),
                1 - bits
            );
        }
    }

    #[test]
    fn predicate_binary_truth_tables_match_native_bool_casts() {
        for lhs in 0..=1 {
            for rhs in 0..=1 {
                for (op, expected) in [
                    (Opcode::Add, lhs | rhs),
                    (Opcode::Maximum, lhs | rhs),
                    (Opcode::Or, lhs | rhs),
                    (Opcode::Multiply, lhs & rhs),
                    (Opcode::Minimum, lhs & rhs),
                    (Opcode::And, lhs & rhs),
                    (Opcode::Xor, lhs ^ rhs),
                ] {
                    assert_eq!(
                        binary(op, raw(ElementType::Pred, lhs), raw(ElementType::Pred, rhs))
                            .unwrap()
                            .integer_bits(),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn bitwise_not_and_or_xor_cover_every_integer_format() {
        for element in INTEGERS {
            let maximum = significant_mask(element);
            for lhs in samples(element) {
                assert_eq!(
                    unary(Opcode::Not, raw(element, lhs))
                        .unwrap()
                        .integer_bits(),
                    lhs ^ maximum
                );
                for rhs in samples(element) {
                    for (op, expected) in [
                        (Opcode::And, lhs & rhs),
                        (Opcode::Or, lhs | rhs),
                        (Opcode::Xor, lhs ^ rhs),
                    ] {
                        assert_eq!(
                            binary(op, raw(element, lhs), raw(element, rhs))
                                .unwrap()
                                .integer_bits(),
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn division_and_remainder_match_integer_arithmetic_at_every_width() {
        for element in INTEGERS {
            let mut values = if element.bit_width() <= 8 {
                (0..=significant_mask(element)).collect::<Vec<_>>()
            } else {
                samples(element)
            };
            for value in [-7, -3, 3, 7, 9_007_199_254_740_993] {
                let (minimum, maximum) = limits(element);
                if (minimum..=maximum).contains(&value) {
                    values.push(scalar(element, value).integer_bits());
                }
            }
            values.sort_unstable();
            values.dedup();
            for &lhs in &values {
                for &rhs in &values {
                    let a = mathematical(element, lhs);
                    let b = mathematical(element, rhs);
                    let quotient = if b == 0 { -1 } else { a / b };
                    let remainder = if b == 0 { a } else { a % b };
                    assert_eq!(
                        binary(Opcode::Divide, raw(element, lhs), raw(element, rhs)).unwrap(),
                        scalar(element, quotient),
                        "divide {element:?}, {a} / {b}"
                    );
                    assert_eq!(
                        binary(Opcode::Remainder, raw(element, lhs), raw(element, rhs)).unwrap(),
                        scalar(element, remainder),
                        "remainder {element:?}, {a} % {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn shifts_obey_width_limited_bits_and_saturate_out_of_range_counts() {
        for element in INTEGERS {
            let width = element.bit_width();
            let maximum = significant_mask(element);
            let mut values = if width <= 4 {
                (0..=maximum).collect::<Vec<_>>()
            } else {
                samples(element)
            };
            values.extend([0x9234_5678 & maximum, 0x1000_1000 & maximum, 77 & maximum]);
            values.push(maximum.saturating_sub(3));
            values.extend((0..width).map(|bit| 1u64 << bit));
            values.sort_unstable();
            values.dedup();
            let mut counts = (0..=u64::from(width) + 1).collect::<Vec<_>>();
            counts.extend([100, maximum]);
            counts.retain(|&count| count <= maximum);
            counts.sort_unstable();
            counts.dedup();
            for lhs in values {
                for &count in &counts {
                    let sign = lhs & (1u64 << (width - 1)) != 0;
                    let signed = if sign {
                        i128::from(lhs) - (1i128 << width)
                    } else {
                        i128::from(lhs)
                    };
                    let (left, logical, arithmetic) = if count >= u64::from(width) {
                        (0, 0, if sign { maximum } else { 0 })
                    } else {
                        (lhs << count, lhs >> count, (signed >> count) as u64)
                    };
                    for (op, expected) in [
                        (Opcode::ShiftLeft, left),
                        (Opcode::ShiftRightLogical, logical),
                        (Opcode::ShiftRightArithmetic, arithmetic),
                    ] {
                        assert_eq!(
                            binary(op, raw(element, lhs), raw(element, count))
                                .unwrap()
                                .integer_bits(),
                            expected & maximum,
                            "{op} {element:?} {lhs} {count}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn power_zero_exponent_including_zero_to_zero_returns_one_modulo_width() {
        for element in INTEGERS {
            for bits in samples(element) {
                assert_eq!(
                    binary(Opcode::Power, raw(element, bits), raw(element, 0)).unwrap(),
                    scalar(element, 1)
                );
            }
        }
    }

    #[test]
    fn power_negative_integer_exponents_follow_pinned_special_cases() {
        for element in [
            ElementType::S4,
            ElementType::S8,
            ElementType::S16,
            ElementType::S32,
            ElementType::S64,
        ] {
            for exponent in [-1, -2, -3, -4] {
                assert_binary(Opcode::Power, element, 1, exponent, 1);
                assert_binary(
                    Opcode::Power,
                    element,
                    -1,
                    exponent,
                    if exponent % 2 == 0 { 1 } else { -1 },
                );
                assert_binary(Opcode::Power, element, 0, exponent, 0);
                assert_binary(Opcode::Power, element, 2, exponent, 0);
                assert_binary(Opcode::Power, element, -2, exponent, 0);
            }
        }
    }

    #[test]
    fn power_wraps_without_float_conversion() {
        assert_binary(
            Opcode::Power,
            ElementType::U64,
            3,
            40,
            12_157_665_459_056_928_801,
        );
        assert_binary(Opcode::Power, ElementType::U64, 2, 64, 0);
        assert_binary(Opcode::Power, ElementType::S64, 2, 63, i128::from(i64::MIN));
        assert_binary(Opcode::Power, ElementType::S8, -3, 5, 13);
    }

    #[test]
    fn power_large_u64_exponents_are_logarithmic_and_bit_exact() {
        assert_binary(Opcode::Power, ElementType::U64, 1, i128::from(u64::MAX), 1);
        assert_binary(
            Opcode::Power,
            ElementType::U64,
            i128::from(u64::MAX),
            i128::from(u64::MAX),
            -1,
        );
        assert_binary(Opcode::Power, ElementType::S64, -1, i128::from(i64::MIN), 1);
    }

    #[test]
    fn mulhi_uses_128_bit_exact_products_for_64_bit_inputs() {
        assert_binary(
            Opcode::Mulhi,
            ElementType::U64,
            i128::from(u64::MAX),
            i128::from(u64::MAX),
            i128::from(u64::MAX - 1),
        );
        assert_binary(Opcode::Mulhi, ElementType::S64, i128::from(i64::MIN), -1, 0);
        assert_binary(Opcode::Mulhi, ElementType::S64, i128::from(i64::MIN), 2, -1);
        assert_binary(
            Opcode::Mulhi,
            ElementType::S64,
            i128::from(i64::MIN),
            i128::from(i64::MIN),
            1i128 << 62,
        );
    }

    #[test]
    fn subbyte_mulhi_is_explicitly_unsupported_not_approximated() {
        for element in INTEGERS
            .into_iter()
            .filter(|element| element.bit_width() < 8)
        {
            let error = binary(Opcode::Mulhi, raw(element, 1), raw(element, 1)).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::Unsupported { .. })
            ));
        }
    }

    #[test]
    fn mulhi_returns_the_high_half_of_the_widened_product() {
        for element in INTEGERS
            .into_iter()
            .filter(|element| element.bit_width() >= 8)
        {
            let width = element.bit_width();
            for lhs in samples(element) {
                for rhs in samples(element) {
                    let expected = if element.is_signed() {
                        let product = mathematical(element, lhs) * mathematical(element, rhs);
                        product.div_euclid(1i128 << width) as u64
                    } else {
                        ((u128::from(lhs) * u128::from(rhs)) / (1u128 << width)) as u64
                    };
                    assert_eq!(
                        binary(Opcode::Mulhi, raw(element, lhs), raw(element, rhs))
                            .unwrap()
                            .integer_bits(),
                        expected & significant_mask(element)
                    );
                }
            }
        }
    }

    #[test]
    fn bit_counts_match_explicit_iteration_and_subbyte_signed_promotion() {
        for element in INTEGERS {
            let width = element.bit_width();
            let mut values = if width <= 4 {
                (0..=significant_mask(element)).collect::<Vec<_>>()
            } else {
                samples(element)
            };
            values.extend((0..width).map(|bit| 1u64 << bit));
            values.push((significant_mask(element) / 2).saturating_sub(1));
            values.sort_unstable();
            values.dedup();
            for bits in values {
                let mut leading = 0;
                for bit in (0..width).rev() {
                    if bits & (1u64 << bit) != 0 {
                        break;
                    }
                    leading += 1;
                }
                let promoted = if width < 8 {
                    u64::from(mathematical(element, bits) as u8)
                } else {
                    bits
                };
                let population = (0..width.max(8))
                    .filter(|bit| promoted & (1u64 << bit) != 0)
                    .count();
                assert_eq!(
                    unary(Opcode::Clz, raw(element, bits)).unwrap(),
                    scalar(element, leading),
                    "clz {element:?}, bits={bits:#x}"
                );
                assert_eq!(
                    unary(Opcode::PopulationCount, raw(element, bits)).unwrap(),
                    scalar(element, population as i128),
                    "population count {element:?}, bits={bits:#x}"
                );
            }
        }
    }

    #[test]
    fn subbyte_to_all_15_formats_conversion_is_exhaustive() {
        for source in INTEGERS
            .into_iter()
            .filter(|element| element.bit_width() <= 4)
        {
            for bits in 0..=significant_mask(source) {
                for destination in formats() {
                    let value = mathematical(source, bits);
                    let expected = if destination == ElementType::Pred {
                        scalar(destination, i128::from(value != 0))
                    } else {
                        scalar(destination, value)
                    };
                    assert_eq!(convert(raw(source, bits), destination).unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn signed_and_unsigned_min_max_keep_distinct_numeric_orders() {
        for element in INTEGERS {
            let (min, max) = limits(element);
            assert_binary(Opcode::Minimum, element, min, max, min);
            assert_binary(Opcode::Maximum, element, min, max, max);
            assert_binary(Opcode::Minimum, element, max, min, min);
            assert_binary(Opcode::Maximum, element, max, min, max);
        }
    }

    #[test]
    fn compare_all_15_formats_six_directions_and_boundary_pairs() {
        for element in formats() {
            for a in samples(element) {
                for b in samples(element) {
                    let lhs = mathematical(element, a);
                    let rhs = mathematical(element, b);
                    for (direction, expected) in [
                        (Direction::Eq, lhs == rhs),
                        (Direction::Ne, lhs != rhs),
                        (Direction::Lt, lhs < rhs),
                        (Direction::Le, lhs <= rhs),
                        (Direction::Gt, lhs > rhs),
                        (Direction::Ge, lhs >= rhs),
                    ] {
                        assert_eq!(
                            compare(
                                direction,
                                ComparisonOrder::Total,
                                raw(element, a),
                                raw(element, b)
                            )
                            .unwrap(),
                            raw(ElementType::Pred, u64::from(expected)),
                            "{element:?} {direction:?} {a} {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn compare_u64_adjacent_values_above_host_float_precision_are_distinct() {
        for a in [9_007_199_254_740_992, u64::MAX - 1] {
            assert_eq!(
                compare(
                    Direction::Lt,
                    ComparisonOrder::Total,
                    raw(ElementType::U64, a),
                    raw(ElementType::U64, a + 1)
                )
                .unwrap()
                .integer_bits(),
                1
            );
        }
    }

    #[test]
    fn compare_rejects_partial_weak_and_mixed_signedness() {
        let a = scalar(ElementType::S64, -1);
        for order in [ComparisonOrder::Partial, ComparisonOrder::Weak] {
            assert!(compare(Direction::Eq, order, a, a).is_err());
        }
        assert!(
            compare(
                Direction::Eq,
                ComparisonOrder::Total,
                a,
                raw(ElementType::U64, u64::MAX)
            )
            .is_err()
        );
    }

    #[test]
    fn all_225_integer_predicate_conversion_pairs_cover_boundaries() {
        for source in formats() {
            for destination in formats() {
                for bits in samples(source) {
                    let mathematical = mathematical(source, bits);
                    let expected = if destination == ElementType::Pred {
                        u64::from(mathematical != 0)
                    } else {
                        (mathematical as u64) & significant_mask(destination)
                    };
                    assert_eq!(
                        convert(raw(source, bits), destination).unwrap(),
                        raw(destination, expected),
                        "{source:?}->{destination:?} {bits}"
                    );
                }
            }
        }
    }

    #[test]
    fn conversion_narrows_by_truncation_not_saturation() {
        assert_eq!(
            convert(scalar(ElementType::S32, 256), ElementType::U8)
                .unwrap()
                .integer_bits(),
            0
        );
        assert_eq!(
            convert(scalar(ElementType::S32, 255), ElementType::S8).unwrap(),
            scalar(ElementType::S8, -1)
        );
        assert_eq!(
            convert(raw(ElementType::U64, u64::MAX), ElementType::S64).unwrap(),
            scalar(ElementType::S64, -1)
        );
        assert_eq!(
            convert(scalar(ElementType::S64, -9), ElementType::S4).unwrap(),
            scalar(ElementType::S4, 7)
        );
    }

    #[test]
    fn predicate_conversion_tests_nonzero_instead_of_low_bit() {
        for element in INTEGERS
            .into_iter()
            .filter(|element| element.bit_width() >= 2)
        {
            assert_eq!(
                convert(raw(element, 2), ElementType::Pred)
                    .unwrap()
                    .integer_bits(),
                1
            );
        }
        assert_eq!(
            convert(raw(ElementType::Pred, 1), ElementType::S1).unwrap(),
            scalar(ElementType::S1, -1)
        );
    }

    #[test]
    fn integer_conversion_preserves_exact_values_in_standard_float_formats() {
        for (source, value, destination, expected) in [
            (ElementType::S32, 42, ElementType::F32, 0x4228_0000),
            (
                ElementType::S32,
                -42,
                ElementType::F64,
                0xc045_0000_0000_0000,
            ),
            (ElementType::S32, 42, ElementType::C64, 0x4228_0000),
            (
                ElementType::S32,
                -42,
                ElementType::C128,
                0xc045_0000_0000_0000,
            ),
        ] {
            assert_eq!(
                convert(scalar(source, value), destination).unwrap(),
                Scalar::from_bits(destination, expected).unwrap()
            );
        }
    }

    #[test]
    fn predicate_bitcast_is_byte_sized_and_does_not_coerce_invalid_encodings() {
        assert!(bitcast_convert(raw(ElementType::Pred, 1), ElementType::U1).is_err());
        assert_eq!(
            bitcast_convert(raw(ElementType::Pred, 1), ElementType::U8)
                .unwrap()
                .integer_bits(),
            1
        );
        assert_eq!(
            bitcast_convert(raw(ElementType::U8, 1), ElementType::Pred)
                .unwrap()
                .integer_bits(),
            1
        );
        assert!(bitcast_convert(raw(ElementType::U8, 2), ElementType::Pred).is_err());
        assert!(bitcast_convert(scalar(ElementType::S8, -1), ElementType::Pred).is_err());
    }

    #[test]
    fn bitcast_and_numeric_conversion_have_distinct_signed_extension_rules() {
        let value = scalar(ElementType::S8, -1);
        assert_eq!(
            bitcast_convert(value, ElementType::U8)
                .unwrap()
                .integer_bits(),
            255
        );
        assert_eq!(
            convert(value, ElementType::U64).unwrap().integer_bits(),
            u64::MAX
        );
        assert!(bitcast_convert(value, ElementType::U64).is_err());
    }

    #[test]
    fn select_rejects_nonpredicate_conditions_and_mismatched_values() {
        let a = scalar(ElementType::S32, -1);
        assert!(select(scalar(ElementType::U1, 1), a, a).is_err());
        assert!(select(raw(ElementType::Pred, 1), a, scalar(ElementType::U32, -1)).is_err());
    }

    #[test]
    fn clamp_all_15_formats_in_native_numeric_order() {
        for element in formats() {
            let (minimum, maximum) = limits(element);
            let low_value = minimum.max(-1);
            let high_value = maximum.min(1);
            let low = scalar(element, low_value);
            let high = scalar(element, high_value);
            for bits in samples(element) {
                let value = raw(element, bits);
                let expected = mathematical(element, bits).clamp(low_value, high_value);
                assert_eq!(clamp(low, value, high).unwrap(), scalar(element, expected));
            }
        }
    }

    #[test]
    fn reversed_clamp_bounds_return_high_without_rejection() {
        for element in formats() {
            let (min, max) = limits(element);
            let low = scalar(element, max);
            let high = scalar(element, min);
            for bits in samples(element) {
                assert_eq!(clamp(low, raw(element, bits), high).unwrap(), high);
            }
        }
    }

    #[test]
    fn clamp_rejects_either_mismatched_bound() {
        let a = scalar(ElementType::S32, 0);
        let b = scalar(ElementType::U32, 0);
        assert!(clamp(b, a, a).is_err());
        assert!(clamp(a, a, b).is_err());
        assert!(
            clamp(
                raw(ElementType::C64, 0),
                raw(ElementType::C64, 0),
                raw(ElementType::C64, 0)
            )
            .is_err()
        );
    }

    #[test]
    fn signed_64_bit_clamp_obeys_numeric_boundaries() {
        for (low, value, high, expected) in [
            (
                -8_616_761_059_752_331_528,
                -6_780_561_065_411_491_190,
                -6_780_561_065_411_491_180,
                -6_780_561_065_411_491_190,
            ),
            (
                6_780_561_065_411_491_190,
                6_780_561_065_411_491_180,
                8_616_761_065_411_491_190,
                6_780_561_065_411_491_190,
            ),
            (
                -8_616_761_059_752_331_528,
                4_241_131_823_772_864_090,
                3_832_151_243_857_508_051,
                3_832_151_243_857_508_051,
            ),
        ] {
            assert_eq!(
                clamp(
                    scalar(ElementType::S64, low),
                    scalar(ElementType::S64, value),
                    scalar(ElementType::S64, high)
                )
                .unwrap(),
                scalar(ElementType::S64, expected)
            );
        }
    }

    #[test]
    fn scalar_values_outlive_source_literal_storage() {
        let ctx = context();
        let mut literal = scalar(ElementType::S64, -91).to_literal(&ctx).unwrap();
        let value = Scalar::read(&ctx, &literal, &[]).unwrap();
        literal.storage.fill(0);
        assert_eq!(value, scalar(ElementType::S64, -91));
        let mut first = value.to_literal(&ctx).unwrap();
        let second = value.to_literal(&ctx).unwrap();
        first.storage.fill(0);
        assert_ne!(first.storage, second.storage);
    }
}
