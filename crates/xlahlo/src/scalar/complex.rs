// SPDX-License-Identifier: Apache-2.0

//! Complex arithmetic built from IEEE 754 scalar operations.
//!
//! IEEE 754 specifies the component operations, but not a bit-exact complex
//! division algorithm. Finite division scales both operands before evaluating
//! the quotient to avoid overflow in the intermediate products. NaN payloads
//! and exceptional complex results are not assigned a portable bit pattern.
use super::{RawResult, unsupported};
use crate::{opcode::Opcode, types::ElementType};

macro_rules! component_ops {
    (
        $unary:ident, $binary:ident, $divide:ident, $float:ty, $exponent:ident, $scale:ident,
        $fraction_bits:expr, $exponent_mask:expr, $bias:expr, $fraction_mask:expr,
        $width:expr, $subnormal_exponent:expr, $chunk:expr
    ) => {
        fn $exponent(value: $float) -> i32 {
            let bits = value.abs().to_bits();
            let exponent = ((bits >> $fraction_bits) & $exponent_mask) as i32;
            if exponent != 0 {
                exponent - $bias
            } else {
                let fraction = bits & $fraction_mask;
                let highest_bit = $width - 1 - fraction.leading_zeros() as i32;
                highest_bit + $subnormal_exponent
            }
        }

        fn $scale(mut value: $float, mut power: i32) -> $float {
            // Keep each scale factor representable even at subnormal exponents.
            while power > 0 {
                let step = power.min($chunk);
                value *= (2.0 as $float).powi(step);
                power -= step;
            }
            while power < 0 {
                let step = (-power).min($chunk);
                value *= (2.0 as $float).powi(-step);
                power += step;
            }
            value
        }

        fn $divide(a: [$float; 2], b: [$float; 2]) -> [$float; 2] {
            let [c, d] = b;
            if c == 0.0 && d == 0.0 {
                return [<$float>::NAN, <$float>::NAN];
            }

            if a.iter().chain(b.iter()).all(|value| value.is_finite()) {
                let numerator_power = if a[0] == 0.0 && a[1] == 0.0 {
                    0
                } else {
                    $exponent(a[0].abs().max(a[1].abs()))
                };
                let denominator_power = $exponent(c.abs().max(d.abs()));
                let ar = $scale(a[0], -numerator_power);
                let ai = $scale(a[1], -numerator_power);
                let cr = $scale(c, -denominator_power);
                let ci = $scale(d, -denominator_power);
                let norm = cr * cr + ci * ci;
                let real = (ar * cr + ai * ci) / norm;
                let imaginary = (ai * cr - ar * ci) / norm;
                return [
                    $scale(real, numerator_power - denominator_power),
                    $scale(imaginary, numerator_power - denominator_power),
                ];
            }

            let norm = c * c + d * d;
            [(a[0] * c + a[1] * d) / norm, (a[1] * c - a[0] * d) / norm]
        }

        fn $unary(opcode: Opcode, a: [$float; 2]) -> Result<[$float; 2], pliron::result::Error> {
            match opcode {
                Opcode::Negate => Ok([-a[0], -a[1]]),
                Opcode::Sign => {
                    let magnitude = a[0].hypot(a[1]);
                    if magnitude == 0.0 {
                        Ok([0.0, 0.0])
                    } else {
                        Ok([a[0] / magnitude, a[1] / magnitude])
                    }
                }
                _ => Err(unsupported("unimplemented complex unary operation")),
            }
        }

        fn $binary(
            opcode: Opcode,
            a: [$float; 2],
            b: [$float; 2],
        ) -> Result<[$float; 2], pliron::result::Error> {
            match opcode {
                Opcode::Add => Ok([a[0] + b[0], a[1] + b[1]]),
                Opcode::Subtract => Ok([a[0] - b[0], a[1] - b[1]]),
                Opcode::Multiply => Ok([a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]),
                Opcode::Divide => Ok($divide(a, b)),
                _ => Err(unsupported("unimplemented complex binary operation")),
            }
        }
    };
}

component_ops!(
    unary32,
    binary32,
    divide32,
    f32,
    exponent32,
    scale32,
    23,
    0xff,
    127,
    0x007f_ffff,
    32,
    -149,
    64
);
component_ops!(
    unary64,
    binary64,
    divide64,
    f64,
    exponent64,
    scale64,
    52,
    0x7ff,
    1023,
    0x000f_ffff_ffff_ffff,
    64,
    -1074,
    512
);

fn unpack32(bits: u128) -> [f32; 2] {
    [
        f32::from_bits(bits as u32),
        f32::from_bits((bits >> 32) as u32),
    ]
}

fn unpack64(bits: u128) -> [f64; 2] {
    [
        f64::from_bits(bits as u64),
        f64::from_bits((bits >> 64) as u64),
    ]
}

fn pack32(a: [f32; 2]) -> u128 {
    u128::from(a[0].to_bits()) | (u128::from(a[1].to_bits()) << 32)
}

fn pack64(a: [f64; 2]) -> u128 {
    u128::from(a[0].to_bits()) | (u128::from(a[1].to_bits()) << 64)
}

pub(super) fn convert(src: ElementType, bits: u128, dst: ElementType) -> u128 {
    if src == dst {
        return bits;
    }
    if dst == ElementType::C64 {
        let a = unpack64(bits);
        pack32([a[0] as f32, a[1] as f32])
    } else {
        let a = unpack32(bits);
        pack64([f64::from(a[0]), f64::from(a[1])])
    }
}

pub(super) fn equal(ty: ElementType, lhs: u128, rhs: u128) -> bool {
    if ty == ElementType::C64 {
        unpack32(lhs) == unpack32(rhs)
    } else {
        unpack64(lhs) == unpack64(rhs)
    }
}

pub(super) fn unary(opcode: Opcode, ty: ElementType, bits: u128) -> RawResult {
    if opcode == Opcode::Abs {
        if ty == ElementType::C64 {
            let a = unpack32(bits);
            return Ok((ElementType::F32, u128::from(a[0].hypot(a[1]).to_bits())));
        }
        let a = unpack64(bits);
        return Ok((ElementType::F64, u128::from(a[0].hypot(a[1]).to_bits())));
    }
    Ok((
        ty,
        if ty == ElementType::C64 {
            pack32(unary32(opcode, unpack32(bits))?)
        } else {
            pack64(unary64(opcode, unpack64(bits))?)
        },
    ))
}

pub(super) fn binary(opcode: Opcode, ty: ElementType, lhs: u128, rhs: u128) -> RawResult {
    Ok((
        ty,
        if ty == ElementType::C64 {
            pack32(binary32(opcode, unpack32(lhs), unpack32(rhs))?)
        } else {
            pack64(binary64(opcode, unpack64(lhs), unpack64(rhs))?)
        },
    ))
}
