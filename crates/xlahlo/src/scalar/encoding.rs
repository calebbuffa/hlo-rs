// Copyright 2022 The TensorFlow Authors.
// Copyright 2017 The TensorFlow Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Float8 conversion behavior follows TensorFlow's ml_dtypes and Eigen's
// BFloat16 conventions. The full Apache-2.0 text is in LICENSE.
//! Integer-bit floating-format conversions. Narrow formats use separate
//! conversion paths where their rounding and NaN behavior differ.

use crate::scalar::half::{float_to_half, half_to_float};
use crate::types::ElementType;

#[derive(Clone, Copy)]
struct Format {
    width: u32,
    mantissa: u32,
    exponent: u32,
    bias: i32,
    highest: u64,
    infinity: u64,
    nan: u64,
    kind: Kind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ieee,
    FiniteNan,
    Fnuz,
    Finite,
    Unsigned,
}

fn format(ty: ElementType) -> Format {
    use ElementType::*;
    use Kind::*;
    let (width, mantissa, exponent, bias, highest, infinity, nan, kind) = match ty {
        F16 => (16, 10, 5, 15, 0x7bff, 0x7c00, 0x7e00, Ieee),
        BF16 => (16, 7, 8, 127, 0x7f7f, 0x7f80, 0x7fc0, Ieee),
        F32 => (32, 23, 8, 127, 0x7f7f_ffff, 0x7f80_0000, 0x7fc0_0000, Ieee),
        F64 => (
            64,
            52,
            11,
            1023,
            0x7fef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            Ieee,
        ),
        F8E5M2 => (8, 2, 5, 15, 0x7b, 0x7c, 0x7e, Ieee),
        F8E4M3 => (8, 3, 4, 7, 0x77, 0x78, 0x7c, Ieee),
        F8E3M4 => (8, 4, 3, 3, 0x6f, 0x70, 0x78, Ieee),
        F8E4M3FN => (8, 3, 4, 7, 0x7e, 0x7f, 0x7f, FiniteNan),
        F8E4M3B11FNUZ => (8, 3, 4, 11, 0x7f, 0x80, 0x80, Fnuz),
        F8E4M3FNUZ => (8, 3, 4, 8, 0x7f, 0x80, 0x80, Fnuz),
        F8E5M2FNUZ => (8, 2, 5, 16, 0x7f, 0x80, 0x80, Fnuz),
        F4E2M1FN => (4, 1, 2, 1, 0x7, 0x7, 0x8, Finite),
        F6E3M2FN => (6, 2, 3, 3, 0x1f, 0x1f, 0x20, Finite),
        F6E2M3FN => (6, 3, 2, 1, 0x1f, 0x1f, 0x20, Finite),
        F8E8M0FNU => (8, 0, 8, 127, 0xfe, 0xff, 0xff, Unsigned),
        _ => unreachable!("validated real floating format"),
    };
    Format {
        width,
        mantissa,
        exponent,
        bias,
        highest,
        infinity,
        nan,
        kind,
    }
}

impl Format {
    fn sign_mask(self) -> u64 {
        if self.kind == Kind::Unsigned {
            0
        } else {
            1 << (self.width - 1)
        }
    }
    fn abs(self, bits: u64) -> u64 {
        if self.kind == Kind::Fnuz && bits == self.nan {
            bits
        } else {
            bits & !self.sign_mask()
        }
    }
    fn negate(self, bits: u64) -> u64 {
        if self.kind == Kind::Unsigned {
            self.nan
        } else if self.kind == Kind::Fnuz && bits & 0x7f == 0 {
            bits
        } else {
            bits ^ self.sign_mask()
        }
    }
    fn is_nan(self, bits: u64) -> bool {
        match self.kind {
            Kind::Ieee => self.abs(bits) > self.infinity,
            Kind::FiniteNan => self.abs(bits) == self.nan,
            Kind::Fnuz | Kind::Unsigned => bits == self.nan,
            Kind::Finite => false,
        }
    }
    fn is_infinite(self, bits: u64) -> bool {
        self.kind == Kind::Ieee && self.abs(bits) == self.infinity
    }
    fn max_exponent(self) -> i32 {
        // numeric_limits::max_exponent = greatest finite unbiased exponent + 1.
        ((self.highest >> self.mantissa) as i32) - self.bias + 1
    }
}

pub(super) fn is_nan(ty: ElementType, bits: u128) -> bool {
    format(ty).is_nan(bits as u64)
}

pub(super) fn is_finite(ty: ElementType, bits: u128) -> bool {
    let f = format(ty);
    !f.is_nan(bits as u64) && !f.is_infinite(bits as u64)
}

fn rounded(bits: u128, roundoff: u32, implicit: bool) -> u128 {
    let bias = if implicit {
        1 << (roundoff - 1)
    } else {
        ((bits >> roundoff) & 1) + (1 << (roundoff - 1)) - 1
    };
    bits + bias
}

fn generic(src: ElementType, raw: u64, dst: ElementType) -> u64 {
    if src == dst {
        return raw;
    }
    if src == ElementType::F16 && dst == ElementType::F8E5M2 {
        let magnitude = raw & 0x7fff;
        return if magnitude == 0x7c00 {
            raw >> 8
        } else if magnitude > 0x7c00 {
            (raw >> 8) | 2
        } else {
            (rounded(u128::from(raw), 8, false) >> 8) as u64
        };
    }
    let from = format(src);
    let to = format(dst);
    let sign = raw & from.sign_mask() != 0;
    let from_bits = from.abs(raw);
    let signed = |bits| if sign { to.negate(bits) } else { bits };
    if from.is_infinite(raw) {
        return signed(to.infinity);
    }
    if from.is_nan(raw) {
        return signed(to.nan);
    }
    if from_bits == 0 && from.kind != Kind::Unsigned {
        return if to.kind == Kind::Unsigned {
            to.nan
        } else {
            signed(0)
        };
    }
    if to.kind == Kind::Unsigned && sign {
        return to.nan;
    }
    let offset = to.bias - from.bias;
    let shift = to.mantissa as i32 - from.mantissa as i32;
    let biased = (from_bits >> from.mantissa) as i32;
    let from_minimum = 2 - from.bias - i32::from(from.kind == Kind::Unsigned);
    let to_minimum = 2 - to.bias - i32::from(to.kind == Kind::Unsigned);
    if to_minimum < from_minimum && biased == 0 {
        let normalization = from.mantissa as i32 - (63i32 - from_bits.leading_zeros() as i32);
        let exponent = offset - normalization + 1;
        let mut bits = u128::from(from_bits);
        if exponent <= 0 {
            bits <<= offset as u32;
        } else {
            bits <<= normalization as u32;
            bits &= !(1 << from.mantissa);
            bits |= (exponent as u128) << from.mantissa;
        }
        if shift >= 0 {
            bits <<= shift as u32;
        } else {
            bits =
                rounded(bits, (-shift) as u32, to.mantissa == 0 && offset != 0) >> (-shift) as u32;
        }
        return signed(bits as u64);
    }
    if to_minimum > from_minimum {
        let exponent = biased - from.bias + to.bias;
        if exponent <= 0 {
            let leading = u32::from(biased > 0 || from.kind == Kind::Unsigned);
            let exponent_shift = -shift - exponent + leading as i32;
            let mut bits = u128::from(from_bits & ((1 << from.mantissa) - 1))
                | (u128::from(leading) << from.mantissa);
            if exponent_shift > 0 {
                bits = if exponent_shift <= from.mantissa as i32 + 1 {
                    rounded(bits, exponent_shift as u32, false) >> exponent_shift as u32
                } else {
                    0
                };
            } else {
                bits <<= (-exponent_shift) as u32;
            }
            return signed(bits as u64);
        }
    }
    let mut bits = u128::from(from_bits);
    if shift < 0 {
        bits = rounded(bits, (-shift) as u32, to.mantissa == 0);
        bits &= !((1 << (-shift) as u32) - 1);
    }
    // The upstream unsigned WideBits arithmetic wraps at its actual width.
    let wide_width = (to.mantissa.max(from.mantissa) + to.exponent.max(from.exponent))
        .div_ceil(8)
        .next_power_of_two()
        * 8;
    let wide_mask = (1u128 << wide_width) - 1;
    bits = bits.wrapping_add((offset as i128 as u128).wrapping_shl(from.mantissa)) & wide_mask;
    let mut highest = u128::from(to.highest);
    let result = if shift < 0 {
        highest <<= (-shift) as u32;
        bits >> (-shift) as u32
    } else {
        bits = (bits << shift as u32) & wide_mask;
        bits
    };
    let overflow =
        (to.max_exponent(), to.mantissa) < (from.max_exponent(), from.mantissa) && bits > highest;
    signed(if overflow {
        to.infinity
    } else {
        (result as u64) & ((1u64 << (to.width - 1)).wrapping_mul(2).wrapping_sub(1))
    })
}

fn float_to_bfloat(value: f32) -> u16 {
    let raw = value.to_bits();
    if value.is_nan() {
        ((raw >> 16) & 0x8000) as u16 | 0x7fc0
    } else {
        (raw.wrapping_add(0x7fff + ((raw >> 16) & 1)) >> 16) as u16
    }
}

pub(super) fn to_f32(src: ElementType, bits: u128) -> f32 {
    match src {
        ElementType::F32 => f32::from_bits(bits as u32),
        ElementType::F64 => f64::from_bits(bits as u64) as f32,
        ElementType::F16 => half_to_float(bits as u16),
        ElementType::BF16 => f32::from_bits((bits as u32) << 16),
        _ => f32::from_bits(generic(src, bits as u64, ElementType::F32) as u32),
    }
}

pub(super) fn cast(src: ElementType, bits: u128, dst: ElementType) -> u128 {
    use ElementType::*;
    if src == dst {
        return bits;
    }
    if dst == F16 {
        return u128::from(float_to_half(to_f32(src, bits)));
    }
    if dst == BF16 {
        return u128::from(float_to_bfloat(to_f32(src, bits)));
    }
    if src.bit_width().div_ceil(8) == 1 && dst == F8E3M4 {
        return u128::from(generic(
            F16,
            u64::from(float_to_half(to_f32(src, bits))),
            dst,
        ));
    }
    if matches!(src, F16 | BF16) && matches!(dst, F32 | F64) {
        let value = to_f32(src, bits);
        return if dst == F32 {
            u128::from(value.to_bits())
        } else {
            u128::from((value as f64).to_bits())
        };
    }
    if src == F32 && dst == F64 {
        return u128::from((f32::from_bits(bits as u32) as f64).to_bits());
    }
    if src == F64 && dst == F32 {
        return u128::from((f64::from_bits(bits as u64) as f32).to_bits());
    }
    u128::from(generic(src, bits as u64, dst))
}

pub(super) fn from_integer(src: ElementType, bits: u64, dst: ElementType) -> u128 {
    let width = src.bit_width();
    let signed = if src.is_signed() && bits & (1 << (width - 1)) != 0 {
        i128::from(bits) - (1i128 << width)
    } else {
        i128::from(bits)
    };
    if dst == ElementType::F64 {
        return u128::from(if src.is_signed() {
            (signed as f64).to_bits()
        } else {
            (bits as f64).to_bits()
        });
    }
    let value = if src.is_signed() {
        signed as f32
    } else {
        bits as f32
    };
    if src.storage_bit_width() == 8 && dst == ElementType::F8E3M4 {
        return u128::from(generic(
            ElementType::F16,
            u64::from(float_to_half(value)),
            dst,
        ));
    }
    cast(ElementType::F32, u128::from(value.to_bits()), dst)
}

pub(super) fn to_integer(src: ElementType, bits: u128, dst: ElementType) -> u128 {
    let value = if src == ElementType::F64 {
        f64::from_bits(bits as u64)
    } else {
        f64::from(to_f32(src, bits))
    };
    if dst == ElementType::Pred {
        return u128::from(value != 0.0);
    }
    if value.is_nan() {
        return 0;
    }
    let width = dst.bit_width();
    let mask = (1u128 << width) - 1;
    if dst.is_signed() {
        let low = -(1i128 << (width - 1));
        let high = (1i128 << (width - 1)) - 1;
        let integer = if value >= high as f64 {
            high
        } else if value <= low as f64 {
            low
        } else {
            value as i128
        };
        (integer as u128) & mask
    } else {
        let integer = if value >= mask as f64 {
            mask
        } else if value <= 0.0 {
            0
        } else {
            value as u128
        };
        integer & mask
    }
}

pub(super) fn total_key(ty: ElementType, bits: u128) -> i128 {
    let f = format(ty);
    if f.kind == Kind::Unsigned {
        return bits as i128;
    }
    let raw = bits as u64;
    let negative = raw & f.sign_mask() != 0;
    let mut magnitude = f.abs(raw);
    if f.kind == Kind::Fnuz && negative {
        magnitude = magnitude.wrapping_sub(1);
    }
    if negative {
        !i128::from(magnitude)
    } else {
        i128::from(magnitude)
    }
}
