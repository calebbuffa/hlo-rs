// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from OpenXLA's HLO evaluator, literal conversion and comparison
// semantics.

//! Exact scalar representations and floating-point and complex semantics.
//!
//! Complex arithmetic uses an independent IEEE-component implementation;
//! complex exceptional results and NaN payloads are not required to match a
//! particular C++ library. ReducePrecision uses its separately qualified
//! raw-bit kernel.
//! Operations outside the documented scalar subset return errors rather than
//! falling back to host arithmetic. This module is not a complete evaluator.
//!
//! Real arithmetic retains a documented host-specific NaN routing policy, not a
//! portable IEEE 754 payload rule.

mod complex;
mod encoding;
mod half;
mod precision;
mod remainder;
mod rounding;
pub mod semantics;
mod sqrt;

use crate::{
    Diagnostic,
    attributes::{ComparisonOrder, Direction},
    opcode::Opcode,
    types::ElementType,
};
use pliron::result::Result;
use std::cmp::Ordering;

use semantics::Scalar;

type RawResult = Result<(ElementType, u128)>;

fn invalid(message: &str) -> pliron::result::Error {
    pliron::verify_error_noloc!(Diagnostic::InvalidShape {
        message: message.into(),
    })
}

fn unsupported(message: &str) -> pliron::result::Error {
    pliron::verify_error_noloc!(Diagnostic::Unsupported {
        message: message.into(),
    })
}

fn validate(element: ElementType, bits: u128) -> Result<()> {
    let width = element.bit_width();
    if width < 128 && bits >> width != 0 {
        return Err(invalid(
            "scalar bits exceed the element's significant width",
        ));
    }
    Ok(())
}

pub(crate) fn convert(element: ElementType, bits: u128, destination: ElementType) -> RawResult {
    validate(element, bits)?;
    if element == destination {
        return Ok((destination, bits));
    }
    if element.is_complex() {
        if !destination.is_complex() {
            return Err(unsupported(
                "complex to noncomplex Convert is not implemented",
            ));
        }
        return Ok((destination, complex::convert(element, bits, destination)));
    }
    if destination.is_complex() {
        let component = if destination == ElementType::C64 {
            ElementType::F32
        } else {
            ElementType::F64
        };
        return Ok((destination, convert(element, bits, component)?.1));
    }
    let integer = |ty: ElementType| ty.is_integer() || ty == ElementType::Pred;
    if integer(element) && integer(destination) {
        let operand = Scalar::from_bits(element, bits)?;
        return Ok((
            destination,
            semantics::convert(operand, destination)?.bits(),
        ));
    }
    if integer(element) {
        return Ok((
            destination,
            encoding::from_integer(element, bits as u64, destination),
        ));
    }
    if integer(destination) {
        return Ok((
            destination,
            encoding::to_integer(element, bits, destination),
        ));
    }
    Ok((destination, encoding::cast(element, bits, destination)))
}

pub(crate) fn decode(element: ElementType, bits: u128) -> RawResult {
    validate(element, bits)?;
    if element.is_complex() || matches!(element, ElementType::F32 | ElementType::F64) {
        return Ok((element, bits));
    }
    if !element.is_float() {
        return Err(unsupported("floating decode requires a floating format"));
    }
    Ok((
        ElementType::F32,
        u128::from(encoding::to_f32(element, bits).to_bits()),
    ))
}

pub(crate) fn unary(opcode: Opcode, element: ElementType, bits: u128) -> RawResult {
    validate(element, bits)?;
    if matches!(opcode, Opcode::Real | Opcode::Imag) && element.is_complex() {
        let (component, width) = if element == ElementType::C64 {
            (ElementType::F32, 32)
        } else {
            (ElementType::F64, 64)
        };
        let result = if opcode == Opcode::Real {
            bits
        } else {
            bits >> width
        };
        return Ok((component, result & ((1u128 << width) - 1)));
    }
    if element.is_complex() {
        return complex::unary(opcode, element, bits);
    }
    if !element.is_float() {
        return Err(unsupported("floating unary requires a floating format"));
    }
    let result = match opcode {
        Opcode::Real => bits,
        Opcode::Imag => encoding::from_integer(ElementType::Pred, 0, element),
        Opcode::IsFinite => {
            return Ok((
                ElementType::Pred,
                u128::from(encoding::is_finite(element, bits)),
            ));
        }
        Opcode::Abs => {
            if element == ElementType::F64 {
                bits & 0x7fff_ffff_ffff_ffff
            } else {
                let x = encoding::to_f32(element, bits);
                // MSVC's float abs overload passes through double; a signaling
                // f32 NaN quiets, unlike double abs and raw Real/Imag.
                let magnitude =
                    (x.to_bits() & 0x7fff_ffff) | if x.is_nan() { 0x0040_0000 } else { 0 };
                encoding::cast(ElementType::F32, u128::from(magnitude), element)
            }
        }
        Opcode::Negate => {
            if element == ElementType::F64 {
                bits ^ (1 << 63)
            } else {
                let x = encoding::to_f32(element, bits);
                encoding::cast(ElementType::F32, u128::from((-x).to_bits()), element)
            }
        }
        Opcode::Sign => {
            if element == ElementType::F64 {
                let x = f64::from_bits(bits as u64);
                if x.is_nan() {
                    bits
                } else {
                    u128::from((if x != 0.0 { 1.0f64 } else { 0.0 }).copysign(x).to_bits())
                }
            } else {
                let x = encoding::to_f32(element, bits);
                let result = if x.is_nan() {
                    // copysign(bool,float) promotes the conditional to double.
                    f32::from_bits(x.to_bits() | 0x0040_0000)
                } else {
                    (if x != 0.0 { 1.0f32 } else { 0.0 }).copysign(x)
                };
                encoding::cast(ElementType::F32, u128::from(result.to_bits()), element)
            }
        }
        _ => return Err(unsupported("unimplemented floating unary operation")),
    };
    Ok((element, result))
}

// Callers order commutative operands to match the instantiated reference SSE
// destination. Selection here is first-NaN, independent of Rust optimization.
// Non-NaN invalid arithmetic yields the reference negative qNaN.
macro_rules! arithmetic {
    ($name:ident, $float:ty, $uint:ty, $quiet:expr, $invalid:expr) => {
        fn $name(opcode: Opcode, a: $float, b: $float) -> Result<$float> {
            if matches!(opcode, Opcode::Minimum | Opcode::Maximum) {
                return Ok(if a.is_nan() {
                    a
                } else if b.is_nan() {
                    b
                } else if (opcode == Opcode::Minimum && b < a)
                    || (opcode == Opcode::Maximum && a < b)
                {
                    b
                } else {
                    a
                });
            }
            if !matches!(
                opcode,
                Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
            ) {
                return Err(unsupported("unimplemented floating binary operation"));
            }
            if a.is_nan() {
                return Ok(<$float>::from_bits(a.to_bits() | $quiet));
            }
            if b.is_nan() {
                return Ok(<$float>::from_bits(b.to_bits() | $quiet));
            }
            let result = match opcode {
                Opcode::Add => a + b,
                Opcode::Subtract => a - b,
                Opcode::Multiply => a * b,
                Opcode::Divide => a / b,
                _ => return Err(unsupported("unimplemented floating binary operation")),
            };
            Ok(if result.is_nan() {
                <$float>::from_bits($invalid as $uint)
            } else {
                result
            })
        }
    };
}
arithmetic!(arithmetic32, f32, u32, 0x0040_0000, 0xffc0_0000u32);
arithmetic!(
    arithmetic64,
    f64,
    u64,
    0x0008_0000_0000_0000,
    0xfff8_0000_0000_0000u64
);

pub(crate) fn binary(opcode: Opcode, element: ElementType, lhs: u128, rhs: u128) -> RawResult {
    validate(element, lhs)?;
    validate(element, rhs)?;
    if opcode == Opcode::Complex {
        return match element {
            ElementType::F32 => Ok((ElementType::C64, lhs | (rhs << 32))),
            ElementType::F64 => Ok((ElementType::C128, lhs | (rhs << 64))),
            _ => Err(unsupported(
                "complex construction requires F32 or F64 components",
            )),
        };
    }
    if element.is_complex() {
        return complex::binary(opcode, element, lhs, rhs);
    }
    if !element.is_float() {
        return Err(unsupported("floating binary requires a floating format"));
    }
    let bits = if element == ElementType::F64 {
        u128::from(
            arithmetic64(
                opcode,
                f64::from_bits(lhs as u64),
                f64::from_bits(rhs as u64),
            )?
            .to_bits(),
        )
    } else {
        let a = encoding::to_f32(element, lhs);
        let b = encoding::to_f32(element, rhs);
        // Narrow real instantiations retain y in the SSE destination for
        // x+y/x*y; the F32 instantiation retains x. Do not change Min/Max.
        let result =
            if element != ElementType::F32 && matches!(opcode, Opcode::Add | Opcode::Multiply) {
                arithmetic32(opcode, b, a)?
            } else {
                arithmetic32(opcode, a, b)?
            };
        encoding::cast(ElementType::F32, u128::from(result.to_bits()), element)
    };
    Ok((element, bits))
}

pub(crate) fn clamp(element: ElementType, low: u128, value: u128, high: u128) -> RawResult {
    for bits in [low, value, high] {
        validate(element, bits)?;
    }
    if !element.is_float() {
        return Err(unsupported(
            "floating clamp requires a real floating format",
        ));
    }
    for bits in [low, value, high] {
        if encoding::is_nan(element, bits) {
            let (compute, widened) = decode(element, bits)?;
            return convert(compute, widened, element);
        }
    }
    binary(
        Opcode::Minimum,
        element,
        high,
        binary(Opcode::Maximum, element, value, low)?.1,
    )
}

fn direction(direction: Direction, ordering: Option<Ordering>) -> bool {
    match direction {
        Direction::Eq => ordering == Some(Ordering::Equal),
        Direction::Ne => ordering != Some(Ordering::Equal),
        Direction::Lt => ordering == Some(Ordering::Less),
        Direction::Le => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        Direction::Gt => ordering == Some(Ordering::Greater),
        Direction::Ge => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
    }
}

pub(crate) fn compare(
    direction_: Direction,
    order: ComparisonOrder,
    element: ElementType,
    lhs: u128,
    rhs: u128,
) -> RawResult {
    validate(element, lhs)?;
    validate(element, rhs)?;
    if element.is_complex() {
        if !matches!(direction_, Direction::Eq | Direction::Ne) {
            return Err(unsupported("ordered complex comparison is not implemented"));
        }
        let equal = complex::equal(element, lhs, rhs);
        return Ok((
            ElementType::Pred,
            u128::from(if direction_ == Direction::Eq {
                equal
            } else {
                !equal
            }),
        ));
    }
    if !element.is_float() {
        return Err(unsupported("floating compare requires a floating format"));
    }
    let ordering = match order {
        ComparisonOrder::Total => {
            Some(encoding::total_key(element, lhs).cmp(&encoding::total_key(element, rhs)))
        }
        ComparisonOrder::Weak
            if encoding::is_nan(element, lhs) || encoding::is_nan(element, rhs) =>
        {
            Some(encoding::is_nan(element, lhs).cmp(&encoding::is_nan(element, rhs)))
        }
        ComparisonOrder::Partial | ComparisonOrder::Weak => {
            if element == ElementType::F64 {
                f64::from_bits(lhs as u64).partial_cmp(&f64::from_bits(rhs as u64))
            } else {
                encoding::to_f32(element, lhs).partial_cmp(&encoding::to_f32(element, rhs))
            }
        }
    };
    Ok((
        ElementType::Pred,
        u128::from(direction(direction_, ordering)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[path = "../f4_conversion_qualification.rs"]
    mod f4_conversion_qualification;
    #[path = "../f8_conversion_qualification.rs"]
    mod f8_conversion_qualification;

    #[test]
    fn real_format_encodings_round_trip_finite_values_and_preserve_classification() {
        for &element in ElementType::ALL.iter().filter(|ty| ty.is_float()) {
            let inputs: Vec<u128> = match element.bit_width() {
                width @ 1..=8 => (0..(1u128 << width)).collect(),
                16 => (0..=u16::MAX).map(u128::from).collect(),
                32 => vec![
                    0,
                    0x8000_0000,
                    1,
                    0x8000_0001,
                    0x3f80_0000,
                    0xbf80_0000,
                    0x7f7f_ffff,
                    0xff7f_ffff,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7fc0_0001,
                    0x7f80_0001,
                ],
                64 => vec![
                    0,
                    0x8000_0000_0000_0000,
                    1,
                    0x8000_0000_0000_0001,
                    0x3ff0_0000_0000_0000,
                    0xbff0_0000_0000_0000,
                    0x7fef_ffff_ffff_ffff,
                    0xffef_ffff_ffff_ffff,
                    0x7ff0_0000_0000_0000,
                    0xfff0_0000_0000_0000,
                    0x7ff8_0000_0000_0001,
                    0x7ff0_0000_0000_0001,
                ],
                width => panic!("unexpected real format width {width}"),
            };
            for bits in inputs {
                assert!(validate(element, bits).is_ok(), "{element:?} {bits:x}");
                let compute = if element == ElementType::F64 {
                    ElementType::F64
                } else {
                    ElementType::F32
                };
                let decoded = encoding::cast(element, bits, compute);
                let round_trip = encoding::cast(compute, decoded, element);
                if encoding::is_finite(element, bits) {
                    assert_eq!(round_trip, bits, "finite {element:?} {bits:x}");
                } else if encoding::is_nan(element, bits) {
                    assert!(
                        encoding::is_nan(element, round_trip),
                        "NaN {element:?} {bits:x}"
                    );
                } else {
                    assert_eq!(
                        encoding::to_f32(element, round_trip).is_infinite(),
                        encoding::to_f32(element, bits).is_infinite(),
                        "infinity {element:?} {bits:x}"
                    );
                    assert_eq!(
                        round_trip >> (element.bit_width() - 1),
                        bits >> (element.bit_width() - 1)
                    );
                }
            }
        }
    }

    #[test]
    fn pinned_constructor_routing_and_nan_encodings() {
        use ElementType::*;
        assert_eq!(
            convert(
                F64,
                (1.0 + 2.0f64.powi(-11) + 2.0f64.powi(-40)).to_bits().into(),
                F16
            )
            .unwrap(),
            (F16, 0x3c00)
        );
        assert_eq!(
            convert(
                F64,
                (1.0 + 2.0f64.powi(-8) + 2.0f64.powi(-40)).to_bits().into(),
                BF16
            )
            .unwrap(),
            (BF16, 0x3f80)
        );
        assert_eq!(convert(F32, 0x7fc1_0001, BF16).unwrap(), (BF16, 0x7fc0));
        assert_eq!(convert(F32, 0xffc1_0001, F16).unwrap(), (F16, 0xfe00));
        assert_eq!(convert(F8E5M2, 0x7d, F16).unwrap(), (F16, 0x7e00));
        assert_eq!(convert(F16, 0x7c01, F8E5M2).unwrap(), (F8E5M2, 0x7e));
        assert_eq!(convert(F32, 0x7fc0_0000, F4E2M1FN).unwrap(), (F4E2M1FN, 8));
        assert_eq!(convert(F32, 0x7f80_0000, F8E5M2).unwrap(), (F8E5M2, 0x7c));
    }

    #[test]
    fn no_zero_unsigned_format_and_fnuz_are_not_ieee() {
        use ElementType::*;
        assert_eq!(convert(F32, 0, F8E8M0FNU).unwrap(), (F8E8M0FNU, 255));
        assert_eq!(
            convert(F32, (-1.0f32).to_bits().into(), F8E8M0FNU).unwrap(),
            (F8E8M0FNU, 255)
        );
        assert_eq!(
            convert(F32, 1.25f32.to_bits().into(), F8E8M0FNU).unwrap(),
            (F8E8M0FNU, 127)
        );
        assert_eq!(
            convert(F32, 1.5f32.to_bits().into(), F8E8M0FNU).unwrap(),
            (F8E8M0FNU, 128)
        );
        assert_eq!(
            unary(Opcode::Negate, F8E4M3FNUZ, 0).unwrap(),
            (F8E4M3FNUZ, 0)
        );
        assert_eq!(
            unary(Opcode::Negate, F8E8M0FNU, 127).unwrap(),
            (F8E8M0FNU, 255)
        );
        assert_eq!(
            unary(Opcode::Imag, F8E8M0FNU, 127).unwrap(),
            (F8E8M0FNU, 255)
        );
    }

    #[test]
    fn min_max_sign_clamp_keep_zero_sign_and_nan_precedence() {
        use ElementType::*;
        for op in [Opcode::Add, Opcode::Multiply] {
            for (element, negative_nan, positive_nan, expected) in [
                (F16, 0xffff, 0x7fff, 0x7e00),
                (BF16, 0xffff, 0x7fff, 0x7fc0),
                (F8E5M2, 0xff, 0x7f, 0x7e),
                (F8E4M3FN, 0xff, 0x7f, 0x7f),
                (F8E4M3, 0xff, 0x7f, 0x7c),
                (F8E3M4, 0xff, 0x7f, 0x78),
            ] {
                assert_eq!(
                    binary(op, element, negative_nan, positive_nan).unwrap(),
                    (element, expected)
                );
            }
            assert_eq!(
                binary(op, F32, 0xffc1_2345, 0x7fc2_3456).unwrap(),
                (F32, 0xffc1_2345)
            );
            assert_eq!(
                binary(op, F64, 0xfff8_0000_0000_1234, 0x7ff8_0000_0000_5678).unwrap(),
                (F64, 0xfff8_0000_0000_1234)
            );
        }
        for op in [Opcode::Minimum, Opcode::Maximum] {
            assert_eq!(binary(op, F32, 0x8000_0000, 0).unwrap().1, 0x8000_0000);
            assert_eq!(binary(op, F32, 0, 0x8000_0000).unwrap().1, 0);
            assert_eq!(
                binary(op, F32, 0x7fc1_2345, 0xffc2_3456).unwrap().1,
                0x7fc1_2345
            );
        }
        assert_eq!(unary(Opcode::Sign, F64, 1 << 63).unwrap(), (F64, 1 << 63));
        assert_eq!(
            clamp(F32, 0x7fc1_2345, 0xffc2_3456, 0x7fc3_4567).unwrap().1,
            0x7fc1_2345
        );
        assert_eq!(
            clamp(F32, 0, 0xffc2_3456, 0x7fc3_4567).unwrap().1,
            0xffc2_3456
        );
        assert_eq!(
            clamp(
                F32,
                2.0f32.to_bits().into(),
                1.0f32.to_bits().into(),
                (-1.0f32).to_bits().into()
            )
            .unwrap()
            .1,
            u128::from((-1.0f32).to_bits())
        );
    }

    #[test]
    fn comparison_orders_keep_original_nan_and_zero_encodings() {
        use ElementType::*;
        assert_eq!(
            compare(Direction::Lt, ComparisonOrder::Total, F32, 0x8000_0000, 0)
                .unwrap()
                .1,
            1
        );
        assert_eq!(
            compare(Direction::Eq, ComparisonOrder::Weak, F32, 0x8000_0000, 0)
                .unwrap()
                .1,
            1
        );
        assert_eq!(
            compare(
                Direction::Eq,
                ComparisonOrder::Weak,
                F32,
                0x7fc1_0000,
                0xffc2_0000
            )
            .unwrap()
            .1,
            1
        );
        assert_eq!(
            compare(
                Direction::Eq,
                ComparisonOrder::Partial,
                F32,
                0x7fc1_0000,
                0x7fc1_0000
            )
            .unwrap()
            .1,
            0
        );
        assert_eq!(
            compare(
                Direction::Lt,
                ComparisonOrder::Total,
                F8E4M3FNUZ,
                0x80,
                0xff
            )
            .unwrap()
            .1,
            1
        );
        assert_eq!(
            compare(Direction::Gt, ComparisonOrder::Total, F8E8M0FNU, 0xff, 0xfe)
                .unwrap()
                .1,
            1
        );
    }

    #[test]
    fn complex_components_project_and_construct_without_numeric_roundtrip() {
        use ElementType::*;
        let real = 0x7ff0_0000_0000_0001u128;
        let imag = 0x8000_0000_0000_0000u128;
        let value = real | (imag << 64);
        assert_eq!(
            binary(Opcode::Complex, F64, real, imag).unwrap(),
            (C128, value)
        );
        assert_eq!(unary(Opcode::Real, C128, value).unwrap(), (F64, real));
        assert_eq!(unary(Opcode::Imag, C128, value).unwrap(), (F64, imag));
        assert_eq!(convert(C128, value, C128).unwrap(), (C128, value));
        assert!(convert(C128, value, F64).is_err());
        assert!(compare(Direction::Lt, ComparisonOrder::Partial, C128, value, value).is_err());
        assert!(binary(Opcode::Maximum, C128, value, value).is_err());
    }

    #[test]
    fn complex_arithmetic_uses_ieee_components_and_scaled_division() {
        use ElementType::*;
        let pair64 = |real: f64, imaginary: f64| {
            u128::from(real.to_bits()) | (u128::from(imaginary.to_bits()) << 64)
        };
        let components64 = |bits: u128| {
            (
                f64::from_bits(bits as u64),
                f64::from_bits((bits >> 64) as u64),
            )
        };
        let pair32 = |real: f32, imaginary: f32| {
            u128::from(real.to_bits()) | (u128::from(imaginary.to_bits()) << 32)
        };
        let components32 = |bits: u128| {
            (
                f32::from_bits(bits as u32),
                f32::from_bits((bits >> 32) as u32),
            )
        };

        assert_eq!(
            components64(
                binary(Opcode::Add, C128, pair64(1.5, 2.0), pair64(3.0, 4.0))
                    .unwrap()
                    .1
            ),
            (4.5, 6.0)
        );
        assert_eq!(
            components64(
                binary(Opcode::Multiply, C128, pair64(1.0, 2.0), pair64(3.0, 4.0))
                    .unwrap()
                    .1
            ),
            (-5.0, 10.0)
        );

        let quotient = components64(
            binary(Opcode::Divide, C128, pair64(1.0, 2.0), pair64(3.0, 4.0))
                .unwrap()
                .1,
        );
        assert!((quotient.0 - 0.44).abs() < 1e-15);
        assert!((quotient.1 - 0.08).abs() < 1e-15);

        let quotient32 = components32(
            binary(Opcode::Divide, C64, pair32(1.0, 2.0), pair32(3.0, 4.0))
                .unwrap()
                .1,
        );
        assert!((quotient32.0 - 0.44).abs() < 1e-6);
        assert!((quotient32.1 - 0.08).abs() < 1e-6);

        for magnitude in [1e308, 1e-308] {
            let value = pair64(magnitude, magnitude);
            let quotient = components64(binary(Opcode::Divide, C128, value, value).unwrap().1);
            assert_eq!(quotient, (1.0, 0.0));
            let abs = unary(Opcode::Abs, C128, value).unwrap();
            assert_eq!(abs.0, F64);
            assert!(f64::from_bits(abs.1 as u64).is_finite());
            assert!(f64::from_bits(abs.1 as u64) > 0.0);
        }
        for magnitude in [1e38f32, 1e-38f32] {
            let value = pair32(magnitude, magnitude);
            assert_eq!(
                components32(binary(Opcode::Divide, C64, value, value).unwrap().1),
                (1.0, 0.0)
            );
        }

        let zero_divisor = components64(
            binary(Opcode::Divide, C128, pair64(1.0, 2.0), pair64(0.0, -0.0))
                .unwrap()
                .1,
        );
        assert!(zero_divisor.0.is_nan() && zero_divisor.1.is_nan());
    }

    #[test]
    fn invalid_bits_and_unimplemented_math_remain_errors() {
        use ElementType::*;
        assert!(convert(F4E2M1FN, 16, F32).is_err());
        assert!(decode(F32, 1 << 32).is_err());
        assert!(unary(Opcode::Exp, F32, 0).is_err());
        assert!(unary(Opcode::ReducePrecision, F32, 0).is_err());
        assert!(unary(Opcode::IsFinite, C64, 0).is_err());
        assert!(binary(Opcode::Complex, BF16, 0, 0).is_err());
        assert!(clamp(C64, 0, 0, 0).is_err());
    }

    #[test]
    fn signaling_nan_quieting_is_operation_and_compute_width_specific() {
        use ElementType::*;
        assert_eq!(
            unary(Opcode::Abs, F32, 0xff80_0001).unwrap(),
            (F32, 0x7fc0_0001)
        );
        assert_eq!(
            unary(Opcode::Abs, F64, 0xfff0_0000_0000_0001).unwrap(),
            (F64, 0x7ff0_0000_0000_0001)
        );
        assert_eq!(
            unary(Opcode::Sign, F32, 0x7f80_0001).unwrap(),
            (F32, 0x7fc0_0001)
        );
        assert_eq!(
            unary(Opcode::Sign, F64, 0x7ff0_0000_0000_0001).unwrap(),
            (F64, 0x7ff0_0000_0000_0001)
        );
        assert_eq!(
            unary(Opcode::Negate, F32, 0x7f80_0001).unwrap(),
            (F32, 0xff80_0001)
        );
        assert_eq!(decode(F16, 0x7c01).unwrap(), (F32, 0x7f80_2000));
        assert_eq!(
            unary(Opcode::Real, F32, 0x7f80_0001).unwrap(),
            (F32, 0x7f80_0001)
        );
        assert_eq!(clamp(F32, 0x7f80_0001, 0, 0).unwrap(), (F32, 0x7f80_0001));
    }
}
