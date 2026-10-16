use super::Parser;
use crate::{Error, PrimitiveType, Result};

pub(super) fn integer(value: &str) -> Result<i64> {
    if value.starts_with('+') {
        return Err(Error::new(
            "literal",
            "leading plus is not an integer token",
        ));
    }
    if value.starts_with('-') {
        value
            .parse::<i64>()
            .map_err(|_| Error::new("literal", "invalid signed integer token"))
    } else {
        value
            .parse::<u64>()
            .map(|value| value as i64)
            .map_err(|_| Error::new("literal", "invalid unsigned integer token"))
    }
}

impl Parser {
    pub(super) fn number(&mut self) -> Result<f64> {
        let start = self.offset();
        let value = self.word()?;
        match value.as_str() {
            "inf" => return Ok(f64::INFINITY),
            "-inf" => return Ok(f64::NEG_INFINITY),
            "nan" | "-nan" => {
                let mut payload = 1_u64 << 51;
                if self.peek() == "(" && self.offset() == start + value.len() {
                    self.expect("(")?;
                    let payload_start = self.offset();
                    let word = self.word()?;
                    let digits = word
                        .strip_prefix("0x")
                        .ok_or_else(|| self.error("expected hexadecimal NaN payload"))?;
                    if payload_start != start + value.len() + 1
                        || self.offset() != payload_start + word.len()
                        || digits.is_empty()
                        || !digits.bytes().all(|byte| byte.is_ascii_hexdigit())
                    {
                        return Err(self.error("malformed NaN payload token"));
                    }
                    payload = u64::from_str_radix(digits, 16)
                        .map_err(|_| self.error("invalid NaN payload"))?;
                    if payload == 0 || payload > 0x000f_ffff_ffff_ffff {
                        return Err(self.error("NaN payload exceeds double significand"));
                    }
                    self.expect(")")?;
                }
                return Ok(f64::from_bits(
                    (u64::from(value.starts_with('-')) << 63) | 0x7ff0_0000_0000_0000 | payload,
                ));
            }
            _ => (),
        }
        if value.starts_with('+') {
            return Err(self.error("leading plus is not a numeric token"));
        }
        if !value.contains(['.', 'e', 'E']) {
            return Ok(integer(&value).map_err(|error| error.at(start))? as f64);
        }
        let result = value
            .parse::<f64>()
            .map_err(|_| self.error("invalid floating-point number"))?;
        if !result.is_finite() {
            return Err(self.error("decimal literal exceeds double range"));
        }
        Ok(result)
    }

    pub(super) fn scalar(&mut self, ty: PrimitiveType) -> Result<u128> {
        if ty.is_complex() {
            self.expect("(")?;
            let real = self.number()?;
            self.expect(",")?;
            let imag = self.number()?;
            self.expect(")")?;
            let component = if ty == PrimitiveType::C64 {
                PrimitiveType::F32
            } else {
                PrimitiveType::F64
            };
            if component == PrimitiveType::F32
                && [real, imag]
                    .iter()
                    .any(|value| value.is_finite() && value.abs() > f64::from(f32::MAX))
            {
                return Err(self.error("complex component exceeds float range"));
            }
            let real_bits = float_bits(real, component)?;
            // The pinned helper passes the real parsed value to both NaN handlers.
            let imag_bits = if real.is_nan() {
                real_bits
            } else if component == PrimitiveType::F32 {
                u128::from((imag as f32).to_bits())
            } else {
                u128::from(imag.to_bits())
            };
            return Ok(real_bits
                | (imag_bits
                    << if component == PrimitiveType::F32 {
                        32
                    } else {
                        64
                    }));
        }
        if ty.is_floating() {
            let value = self.number()?;
            return float_bits(value, ty);
        }
        let value = self.word()?;
        if ty == PrimitiveType::PRED {
            return match value.as_str() {
                "true" => Ok(1),
                "false" => Ok(0),
                _ => Ok(u128::from(integer(&value)? != 0)),
            };
        }
        if !ty.is_integral() {
            return Err(self.error("literal requires numeric primitive type"));
        }
        let value = integer(&value)?;
        let width = ty
            .bit_width()
            .ok_or_else(|| self.error("missing integer bit width"))?;
        if ty.is_signed_integral() {
            if i128::from(value) < -(1_i128 << (width - 1))
                || i128::from(value) >= (1_i128 << (width - 1))
            {
                return Err(self.error("signed literal out of range"));
            }
            Ok((value as u128) & ((1_u128 << width) - 1))
        } else {
            let value = value as u64;
            if width < 64 && value >= (1_u64 << width) {
                return Err(self.error("unsigned literal out of range"));
            }
            Ok(u128::from(value))
        }
    }
}

fn float_bits(value: f64, ty: PrimitiveType) -> Result<u128> {
    match ty {
        PrimitiveType::F32 => {
            if value.is_nan() {
                let mut payload = value.to_bits() & 0x000f_ffff_ffff_ffff;
                if payload == 1 << 51 {
                    payload = 1 << 22;
                }
                if payload > 0x007f_ffff {
                    return Err(Error::new(
                        "literal",
                        "NaN payload exceeds float significand",
                    ));
                }
                Ok(u128::from(
                    ((value.to_bits() >> 32) as u32 & 0x8000_0000) | 0x7f80_0000 | payload as u32,
                ))
            } else {
                let converted = value as f32;
                if value.is_finite() && !converted.is_finite() {
                    return Err(Error::new("literal", "finite literal overflows float"));
                }
                Ok(u128::from(converted.to_bits()))
            }
        }
        PrimitiveType::F64 => Ok(u128::from(value.to_bits())),
        _ => super::floats::parse(value, ty),
    }
}

fn print_float(bits: u128, ty: PrimitiveType) -> Result<String> {
    let (value, negative, payload, quiet) = if ty == PrimitiveType::F32 {
        let bits = bits as u32;
        (
            f64::from(f32::from_bits(bits)),
            bits >> 31 != 0,
            u64::from(bits & 0x007f_ffff),
            1_u64 << 22,
        )
    } else {
        let bits = bits as u64;
        (
            f64::from_bits(bits),
            bits >> 63 != 0,
            bits & 0x000f_ffff_ffff_ffff,
            1_u64 << 51,
        )
    };
    if value.is_nan() {
        let sign = if negative { "-" } else { "" };
        return Ok(if payload == quiet {
            format!("{sign}nan")
        } else {
            format!("{sign}nan(0x{payload:x})")
        });
    }
    let mut result = if ty == PrimitiveType::F32 {
        (value as f32).to_string()
    } else {
        value.to_string()
    };
    if value.is_finite() && !result.contains(['.', 'e', 'E']) {
        result.push_str(".0");
    }
    Ok(result)
}

pub(super) fn print(bits: u128, ty: PrimitiveType) -> Result<String> {
    if matches!(ty, PrimitiveType::F32 | PrimitiveType::F64) {
        return print_float(bits, ty);
    }
    if ty.is_floating() {
        return super::floats::print(bits, ty);
    }
    if ty.is_complex() {
        let (component, shift) = if ty == PrimitiveType::C64 {
            (PrimitiveType::F32, 32)
        } else {
            (PrimitiveType::F64, 64)
        };
        let real = bits & ((1_u128 << shift) - 1);
        let imag = bits >> shift;
        let real_nan = if shift == 32 {
            f32::from_bits(real as u32).is_nan()
        } else {
            f64::from_bits(real as u64).is_nan()
        };
        let imag_nan = if shift == 32 {
            f32::from_bits(imag as u32).is_nan()
        } else {
            f64::from_bits(imag as u64).is_nan()
        };
        if real_nan && real != imag {
            return Err(Error::new(
                "unsupported",
                "pinned complex NaN parser cannot preserve these component bits",
            ));
        }
        let component_text = |bits: u128| {
            if component == PrimitiveType::F32 && !f32::from_bits(bits as u32).is_nan() {
                print_float(
                    u128::from(f64::from(f32::from_bits(bits as u32)).to_bits()),
                    PrimitiveType::F64,
                )
            } else {
                print_float(bits, component)
            }
        };
        let imag_text = if !real_nan && imag_nan && shift == 32 {
            let widened = f64::from(f32::from_bits(imag as u32));
            if (widened as f32).to_bits() != imag as u32 {
                return Err(Error::new(
                    "unsupported",
                    "native complex imaginary NaN conversion cannot preserve these bits",
                ));
            }
            print_float(u128::from(widened.to_bits()), PrimitiveType::F64)?
        } else {
            component_text(imag)?
        };
        return Ok(format!("({},{})", component_text(real)?, imag_text));
    }
    Err(Error::new(
        "unsupported",
        format!("literal text for {ty:?}"),
    ))
}
