// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from OpenXLA's HLO evaluator integer scalar operations.

use super::{Scalar, mask, unsupported};
use crate::{opcode::Opcode, types::ElementType};
use pliron::result::Result;

#[derive(Clone, Copy)]
pub(super) struct Integer {
    pub(super) element: ElementType,
    pub(super) bits: u64,
}

impl Integer {
    pub(super) fn new(element: ElementType, bits: u64) -> Self {
        Self { element, bits }
    }

    pub(super) fn scalar(self) -> Result<Scalar> {
        Scalar::from_bits(self.element, u128::from(self.bits))
    }

    pub(super) fn value(self) -> i128 {
        if self.element.is_signed() {
            self.signed_value()
        } else {
            i128::from(self.bits)
        }
    }

    fn signed_value(self) -> i128 {
        let width = self.element.bit_width();
        if self.bits & (1u64 << (width - 1)) != 0 {
            i128::from(self.bits) - (1i128 << width)
        } else {
            i128::from(self.bits)
        }
    }

    fn with_bits(self, bits: u64) -> Self {
        Self {
            element: self.element,
            bits: bits & mask(self.element),
        }
    }
}

pub(super) fn unary(opcode: Opcode, operand: Integer) -> Result<Integer> {
    let width = operand.element.bit_width();
    let bits = match opcode {
        Opcode::Abs => {
            if operand.value() < 0 {
                0u64.wrapping_sub(operand.bits)
            } else {
                operand.bits
            }
        }
        Opcode::Negate => 0u64.wrapping_sub(operand.bits),
        Opcode::Sign => match operand.value().cmp(&0) {
            std::cmp::Ordering::Less => mask(operand.element),
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        },
        Opcode::Not => !operand.bits,
        Opcode::Clz => u64::from(operand.bits.leading_zeros() - (64 - width)),
        Opcode::PopulationCount => {
            let count = if operand.element.is_signed() && width < 8 {
                (operand.value() as u8).count_ones()
            } else {
                operand.bits.count_ones()
            };
            u64::from(count)
        }
        _ => return Err(unsupported(opcode, operand.element)),
    };
    Ok(operand.with_bits(bits))
}

pub(super) fn binary(opcode: Opcode, lhs: Integer, rhs: Integer) -> Result<Integer> {
    if lhs.element == crate::types::ElementType::Pred {
        let bits = match opcode {
            Opcode::Add | Opcode::Maximum | Opcode::Or => lhs.bits | rhs.bits,
            Opcode::Multiply | Opcode::Minimum | Opcode::And => lhs.bits & rhs.bits,
            Opcode::Xor => lhs.bits ^ rhs.bits,
            _ => return Err(unsupported(opcode, lhs.element)),
        };
        return Ok(lhs.with_bits(bits));
    }
    let width = lhs.element.bit_width();
    let bits = match opcode {
        Opcode::Add => lhs.bits.wrapping_add(rhs.bits),
        Opcode::Subtract => lhs.bits.wrapping_sub(rhs.bits),
        Opcode::Multiply => lhs.bits.wrapping_mul(rhs.bits),
        Opcode::Divide => {
            if rhs.bits == 0 {
                mask(lhs.element)
            } else {
                (lhs.value() / rhs.value()) as u64
            }
        }
        Opcode::Remainder => {
            if rhs.bits == 0 {
                lhs.bits
            } else {
                (lhs.value() % rhs.value()) as u64
            }
        }
        Opcode::Minimum => {
            if lhs.value() <= rhs.value() {
                lhs.bits
            } else {
                rhs.bits
            }
        }
        Opcode::Maximum => {
            if lhs.value() >= rhs.value() {
                lhs.bits
            } else {
                rhs.bits
            }
        }
        Opcode::And => lhs.bits & rhs.bits,
        Opcode::Or => lhs.bits | rhs.bits,
        Opcode::Xor => lhs.bits ^ rhs.bits,
        Opcode::ShiftLeft => {
            if rhs.bits >= u64::from(width) {
                0
            } else {
                lhs.bits << rhs.bits
            }
        }
        Opcode::ShiftRightLogical => {
            if rhs.bits >= u64::from(width) {
                0
            } else {
                lhs.bits >> rhs.bits
            }
        }
        Opcode::ShiftRightArithmetic => {
            let signed = lhs.signed_value();
            if rhs.bits >= u64::from(width) {
                if signed < 0 { mask(lhs.element) } else { 0 }
            } else {
                (signed >> rhs.bits) as u64
            }
        }
        Opcode::Power => power(lhs, rhs),
        Opcode::Mulhi => {
            if width < 8 {
                return Err(unsupported(opcode, lhs.element));
            }
            if lhs.element.is_signed() {
                ((lhs.value() * rhs.value()) >> width) as u64
            } else {
                ((u128::from(lhs.bits) * u128::from(rhs.bits)) >> width) as u64
            }
        }
        _ => return Err(unsupported(opcode, lhs.element)),
    };
    Ok(lhs.with_bits(bits))
}

fn power(lhs: Integer, rhs: Integer) -> u64 {
    let exponent = rhs.value();
    if exponent < 0 {
        return match lhs.value() {
            1 => 1,
            -1 => {
                if exponent % 2 == 0 {
                    1
                } else {
                    mask(lhs.element)
                }
            }
            _ => 0,
        };
    }
    let mut exponent = rhs.bits;
    let mut base = lhs.bits;
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        exponent >>= 1;
    }
    result
}
