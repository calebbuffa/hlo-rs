// Adapted from the recovered pre-pliron StableHLO text codec.
use crate::types::ElementType;
use rustc_apfloat::{
    Float, FloatConvert, Round,
    ieee::{BFloat, Double, Float8E4M3FN, Float8E5M2, Half, IeeeFloat, Semantics, Single},
};

use super::TextErrorKind;

macro_rules! float_format {
    ($name:ident, $bits:expr, $exponent:expr, $minimum:expr) => {
        struct $name;
        impl Semantics for $name {
            const BITS: usize = $bits;
            const EXP_BITS: usize = $exponent;
            const MIN_EXP: i32 = $minimum;
        }
    };
}

float_format!(E3M4, 8, 3, -2);
float_format!(E4M3, 8, 4, -6);
float_format!(TensorFloat, 19, 8, -126);
float_format!(ExtendedE2M1, 5, 3, 0);
float_format!(ExtendedE2M3, 7, 3, 0);
float_format!(ExtendedE3M2, 7, 4, -2);
float_format!(ExtendedE4M3FnUz, 9, 5, -7);
float_format!(ExtendedE4M3B11FnUz, 9, 5, -10);
float_format!(ExtendedE5M2FnUz, 9, 6, -15);

// Complex NumberAttr parses double first, then APFloat::convert rounds to the
// component semantics (ComplexAttributes.td at recovered LLVM 2b35fb9283a4).
pub(super) fn narrow_f64(bits: u64, element: ElementType) -> Result<u64, TextErrorKind> {
    fn convert<F: Float>(bits: u64) -> u64
    where
        Double: FloatConvert<F>,
    {
        let value: F = Double::from_bits(u128::from(bits))
            .convert_r(Round::NearestTiesToEven, &mut false)
            .value;
        value.to_bits() as u64
    }
    fn finite<S: Semantics>(bits: u64, width: u32, fnuz: bool) -> u64 {
        let value = convert::<IeeeFloat<S>>(bits);
        let magnitude = value & ((1_u64 << (S::BITS - 1)) - 1);
        let maximum = (1_u64 << (width - 1)) - 1;
        if fnuz && (magnitude > maximum || f64::from_bits(bits).is_nan()) {
            return 1_u64 << (width - 1);
        }
        if fnuz && magnitude == 0 {
            return 0;
        }
        ((value >> (S::BITS - 1)) << (width - 1)) | magnitude.min(maximum)
    }
    let value = f64::from_bits(bits);
    Ok(match element {
        ElementType::F64 => bits,
        ElementType::F16 => convert::<Half>(bits),
        ElementType::Bf16 => convert::<BFloat>(bits),
        ElementType::F32 => convert::<Single>(bits),
        ElementType::Tf32 => convert::<IeeeFloat<TensorFloat>>(bits),
        ElementType::F8E3M4 => convert::<IeeeFloat<E3M4>>(bits),
        ElementType::F8E4M3 => convert::<IeeeFloat<E4M3>>(bits),
        ElementType::F8E4M3Fn => convert::<Float8E4M3FN>(bits),
        ElementType::F8E5M2 => convert::<Float8E5M2>(bits),
        ElementType::F4E2M1Fn if value.is_finite() => finite::<ExtendedE2M1>(bits, 4, false),
        ElementType::F6E2M3Fn if value.is_finite() => finite::<ExtendedE2M3>(bits, 6, false),
        ElementType::F6E3M2Fn if value.is_finite() => finite::<ExtendedE3M2>(bits, 6, false),
        ElementType::F8E4M3FnUz => finite::<ExtendedE4M3FnUz>(bits, 8, true),
        ElementType::F8E4M3B11FnUz => finite::<ExtendedE4M3B11FnUz>(bits, 8, true),
        ElementType::F8E5M2FnUz => finite::<ExtendedE5M2FnUz>(bits, 8, true),
        ElementType::F8E8M0FnUz => {
            if value == 0.0 {
                0
            } else if bits >> 63 != 0 {
                if value.is_finite() && -value <= 2_f64.powi(-128) {
                    0
                } else {
                    return Err(TextErrorKind::Unsupported(
                        "negative exponent-only complex component",
                    ));
                }
            } else if !value.is_finite() {
                255
            } else {
                let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
                let round_up = bits & ((1_u64 << 52) - 1) >= 1_u64 << 51;
                (exponent + 127 + i32::from(round_up)).clamp(0, 255) as u64
            }
        }
        ElementType::F4E2M1Fn | ElementType::F6E2M3Fn | ElementType::F6E3M2Fn => {
            return Err(TextErrorKind::Unsupported(
                "nonfinite complex component in a finite-only format",
            ));
        }
        _ => return Err(TextErrorKind::Expected("floating component type")),
    })
}

pub(super) fn parse(value: &str, element: ElementType) -> Result<u64, TextErrorKind> {
    if let Some(hex) = value.strip_prefix("0x") {
        if hex.is_empty() || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(TextErrorKind::InvalidNumber);
        }
        let bits = u64::from_str_radix(hex, 16).map_err(|_| TextErrorKind::InvalidNumber)?;
        let width = element.bit_width().ok_or(TextErrorKind::InvalidNumber)?;
        if width < 64 && bits >> width != 0 {
            return Err(TextErrorKind::InvalidNumber);
        }
        return Ok(bits);
    }
    decimal_syntax(value)?;
    match element {
        ElementType::F16 => ieee::<Half>(value),
        ElementType::Bf16 => ieee::<BFloat>(value),
        ElementType::F32 => ieee::<Single>(value),
        ElementType::F64 => ieee::<Double>(value),
        ElementType::F8E5M2 => ieee::<Float8E5M2>(value),
        ElementType::F8E5M2FnUz => fnuz::<ExtendedE5M2FnUz>(value),
        ElementType::F8E4M3Fn => ieee::<Float8E4M3FN>(value),
        ElementType::F8E4M3FnUz => fnuz::<ExtendedE4M3FnUz>(value),
        ElementType::F8E4M3B11FnUz => fnuz::<ExtendedE4M3B11FnUz>(value),
        ElementType::F8E3M4 => ieee::<IeeeFloat<E3M4>>(value),
        ElementType::F8E4M3 => ieee::<IeeeFloat<E4M3>>(value),
        ElementType::Tf32 => ieee::<IeeeFloat<TensorFloat>>(value),
        ElementType::F4E2M1Fn => finite::<ExtendedE2M1>(value, 4),
        ElementType::F6E2M3Fn => finite::<ExtendedE2M3>(value, 6),
        ElementType::F6E3M2Fn => finite::<ExtendedE3M2>(value, 6),
        ElementType::F8E8M0FnUz => exponent_only(value),
        _ => Err(TextErrorKind::Expected("floating element type")),
    }
}

fn decimal_syntax(value: &str) -> Result<(), TextErrorKind> {
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    let mantissa_end = unsigned.find(['e', 'E']).unwrap_or(unsigned.len());
    let (mantissa, exponent) = unsigned.split_at(mantissa_end);
    let mut digits = 0;
    let mut dots = 0;
    for byte in mantissa.bytes() {
        if byte.is_ascii_digit() {
            digits += 1;
        } else if byte == b'.' {
            dots += 1;
        } else {
            return Err(TextErrorKind::InvalidNumber);
        }
    }
    if digits == 0 || dots != 1 || !mantissa.as_bytes().first().is_some_and(u8::is_ascii_digit) {
        return Err(TextErrorKind::Expected(
            "floating literal or hexadecimal bits",
        ));
    }
    if !exponent.is_empty() {
        let exponent = exponent[1..]
            .strip_prefix(['-', '+'])
            .unwrap_or(&exponent[1..]);
        if exponent.is_empty() || !exponent.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(TextErrorKind::InvalidNumber);
        }
    }
    Ok(())
}

fn ieee<F: Float>(value: &str) -> Result<u64, TextErrorKind> {
    F::from_str_r(value, Round::NearestTiesToEven)
        .map(|value| value.value.to_bits() as u64)
        .map_err(|_| TextErrorKind::InvalidNumber)
}

fn finite<S: Semantics>(value: &str, width: u32) -> Result<u64, TextErrorKind> {
    let parsed = IeeeFloat::<S>::from_str_r(value, Round::NearestTiesToEven)
        .map_err(|_| TextErrorKind::InvalidNumber)?
        .value;
    let sign = u64::from(parsed.is_negative()) << (width - 1);
    let magnitude = (parsed.to_bits() as u64) & ((1_u64 << (S::BITS - 1)) - 1);
    let maximum = (1_u64 << (width - 1)) - 1;
    Ok(sign | magnitude.min(maximum))
}

fn fnuz<S: Semantics>(value: &str) -> Result<u64, TextErrorKind> {
    let parsed = IeeeFloat::<S>::from_str_r(value, Round::NearestTiesToEven)
        .map_err(|_| TextErrorKind::InvalidNumber)?
        .value;
    if parsed.is_zero() {
        return Ok(0);
    }
    let magnitude = (parsed.to_bits() as u64) & ((1_u64 << (S::BITS - 1)) - 1);
    if magnitude > 127 {
        return Ok(128);
    }
    Ok((u64::from(parsed.is_negative()) << 7) | magnitude)
}

fn exponent_only(value: &str) -> Result<u64, TextErrorKind> {
    if value.starts_with('-') {
        return Err(TextErrorKind::Expected("unsigned floating literal"));
    }
    let value = Decimal::parse(value);
    if value.digits.is_empty() {
        return Ok(0);
    }
    let (mut low, mut high) = (0_u64, 255_u64);
    while low < high {
        let middle = (low + high) / 2;
        let threshold = Decimal::dyadic_midpoint(middle as i32 - 128);
        if value.compare(&threshold).is_lt() {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    Ok(low)
}

struct Decimal {
    digits: Vec<u8>,
    exponent: i128,
}

impl Decimal {
    fn parse(value: &str) -> Self {
        let mantissa_end = value.find(['e', 'E']).unwrap_or(value.len());
        let mantissa = &value[..mantissa_end];
        let exponent = if mantissa_end == value.len() {
            0
        } else {
            let exponent = &value[mantissa_end + 1..];
            let negative = exponent.starts_with('-');
            let magnitude =
                exponent
                    .trim_start_matches(['-', '+'])
                    .bytes()
                    .fold(0_i128, |value, byte| {
                        value
                            .saturating_mul(10)
                            .saturating_add(i128::from(byte - b'0'))
                            .min(i128::MAX / 4)
                    });
            if negative { -magnitude } else { magnitude }
        };
        let fractional = mantissa
            .find('.')
            .map_or(0, |point| mantissa.len() - point - 1);
        let digits = mantissa
            .bytes()
            .filter(u8::is_ascii_digit)
            .map(|byte| byte - b'0')
            .skip_while(|digit| *digit == 0)
            .collect();
        Self {
            digits,
            exponent: exponent - fractional as i128,
        }
    }

    fn dyadic_midpoint(power: i32) -> Self {
        let mut digits = vec![3_u8];
        let factor = if power < 0 { 5 } else { 2 };
        for _ in 0..power.unsigned_abs() {
            let mut carry = 0_u8;
            for digit in digits.iter_mut().rev() {
                let product = *digit * factor + carry;
                *digit = product % 10;
                carry = product / 10;
            }
            if carry != 0 {
                digits.insert(0, carry);
            }
        }
        Self {
            digits,
            exponent: i128::from(power.min(0)),
        }
    }

    fn compare(&self, other: &Self) -> std::cmp::Ordering {
        let order = (self.digits.len() as i128 + self.exponent)
            .cmp(&(other.digits.len() as i128 + other.exponent));
        if !order.is_eq() {
            return order;
        }
        for index in 0..self.digits.len().max(other.digits.len()) {
            let order = self
                .digits
                .get(index)
                .copied()
                .unwrap_or(0)
                .cmp(&other.digits.get(index).copied().unwrap_or(0));
            if !order.is_eq() {
                return order;
            }
        }
        std::cmp::Ordering::Equal
    }
}
