use crate::{Error, PrimitiveType, Result};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Special {
    Ieee,
    NanAtMax,
    NanAtNegativeZero,
    FiniteOnly,
    ExponentOnly,
}

struct Format {
    exponent: u32,
    fraction: u32,
    bias: i32,
    special: Special,
    via_f32: bool,
}

impl Format {
    fn for_type(ty: PrimitiveType) -> Result<Self> {
        use PrimitiveType::*;
        use Special::*;
        let (exponent, fraction, bias, special) = match ty {
            F16 => (5, 10, 15, Ieee),
            BF16 => (8, 7, 127, Ieee),
            F8E3M4 => (3, 4, 3, Ieee),
            F8E4M3 => (4, 3, 7, Ieee),
            F8E5M2 => (5, 2, 15, Ieee),
            F8E4M3FN => (4, 3, 7, NanAtMax),
            F8E4M3FNUZ => (4, 3, 8, NanAtNegativeZero),
            F8E4M3B11FNUZ => (4, 3, 11, NanAtNegativeZero),
            F8E5M2FNUZ => (5, 2, 16, NanAtNegativeZero),
            F4E2M1FN => (2, 1, 1, FiniteOnly),
            F6E2M3FN => (2, 3, 1, FiniteOnly),
            F6E3M2FN => (3, 2, 3, FiniteOnly),
            F8E8M0FNU => (8, 0, 127, ExponentOnly),
            _ => return Err(Error::new("literal", "not a low-precision floating type")),
        };
        Ok(Self {
            exponent,
            fraction,
            bias,
            special,
            via_f32: matches!(ty, F16 | BF16),
        })
    }

    fn sign(&self) -> u64 {
        1_u64 << (self.exponent + self.fraction)
    }
    fn magnitude_mask(&self) -> u64 {
        self.sign() - 1
    }
    fn fraction_mask(&self) -> u64 {
        (1_u64 << self.fraction) - 1
    }

    fn max(&self) -> u64 {
        self.magnitude_mask()
            - match self.special {
                Special::Ieee => 1_u64 << self.fraction,
                Special::NanAtMax | Special::ExponentOnly => 1,
                _ => 0,
            }
    }

    fn nan(&self) -> u64 {
        match self.special {
            Special::Ieee => {
                (self.magnitude_mask() & !self.fraction_mask()) | (1_u64 << (self.fraction - 1))
            }
            Special::NanAtMax | Special::ExponentOnly => self.magnitude_mask(),
            Special::NanAtNegativeZero | Special::FiniteOnly => self.sign(),
        }
    }

    fn infinity(&self) -> u64 {
        match self.special {
            Special::Ieee => self.magnitude_mask() & !self.fraction_mask(),
            Special::FiniteOnly => self.max(),
            _ => self.nan(),
        }
    }

    fn negate(&self, bits: u64) -> u64 {
        match self.special {
            Special::ExponentOnly => self.nan(),
            Special::NanAtNegativeZero if bits == 0 || bits == self.nan() => bits,
            _ => bits ^ self.sign(),
        }
    }

    fn decode(&self, bits: u64) -> f64 {
        if self.special == Special::ExponentOnly {
            return if bits == self.nan() {
                f64::NAN
            } else {
                2_f64.powi(bits as i32 - self.bias)
            };
        }
        let magnitude = bits & self.magnitude_mask();
        let negative = bits & self.sign() != 0;
        let exponent = magnitude >> self.fraction;
        let fraction = magnitude & self.fraction_mask();
        let value = if (self.special == Special::NanAtNegativeZero && bits == self.nan())
            || (self.special == Special::NanAtMax && magnitude == self.nan())
        {
            f64::NAN
        } else if self.special == Special::Ieee && exponent == (1_u64 << self.exponent) - 1 {
            if fraction == 0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        } else {
            let significant = if exponent == 0 {
                fraction as f64
            } else {
                ((1_u64 << self.fraction) | fraction) as f64
            };
            significant * 2_f64.powi((exponent.max(1) as i32) - self.bias - self.fraction as i32)
        };
        if negative { -value } else { value }
    }

    fn encode(&self, value: f64) -> u64 {
        let negative = value.is_sign_negative();
        let signed = |bits| if negative { self.negate(bits) } else { bits };
        if value.is_nan() {
            return signed(self.nan());
        }
        if value.is_infinite() {
            return signed(self.infinity());
        }
        if value == 0.0 {
            return if self.special == Special::ExponentOnly {
                self.nan()
            } else {
                signed(0)
            };
        }
        if negative && self.special == Special::ExponentOnly {
            return self.nan();
        }
        let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
        let source_exponent = (bits >> 52) as i32;
        let exponent = source_exponent - 1023 + self.bias;
        let significand =
            (bits & 0x000f_ffff_ffff_ffff) | if source_exponent == 0 { 0 } else { 1_u64 << 52 };
        let magnitude = if exponent <= 0 {
            // This also retains ml_dtypes' special exponent-zero E8M0 boundary.
            let shift = 52 - self.fraction as i32 - exponent + i32::from(source_exponent != 0);
            rounded_shift(significand, shift as u32)
        } else {
            let rounded = rounded_shift(significand, 52 - self.fraction);
            (((exponent - 1) as u64) << self.fraction) + rounded
        };
        signed(if magnitude > self.max() {
            self.infinity()
        } else {
            magnitude
        })
    }
}

fn rounded_shift(value: u64, shift: u32) -> u64 {
    if shift > 53 {
        return 0;
    }
    let quotient = value >> shift;
    let remainder = value & ((1_u64 << shift) - 1);
    let halfway = 1_u64 << (shift - 1);
    quotient + u64::from(remainder > halfway || (remainder == halfway && quotient & 1 != 0))
}

pub(super) fn parse(value: f64, ty: PrimitiveType) -> Result<u128> {
    let format = Format::for_type(ty)?;
    if value.is_nan() {
        let mut payload = value.to_bits() & 0x000f_ffff_ffff_ffff;
        let bits = if format.special == Special::Ieee {
            if payload == 1_u64 << 51 {
                payload = 1_u64 << (format.fraction - 1);
            }
            if payload > format.fraction_mask() {
                return Err(Error::new(
                    "literal",
                    "NaN payload exceeds target significand",
                ));
            }
            format.infinity() | payload
        } else {
            if payload != 1_u64 << 51 {
                return Err(Error::new(
                    "literal",
                    "target format does not support NaN payloads",
                ));
            }
            format.nan()
        };
        return Ok(u128::from(if value.is_sign_negative() {
            format.negate(bits)
        } else {
            bits
        }));
    }
    let converted = if format.via_f32 {
        f64::from(value as f32)
    } else {
        value
    };
    let bits = format.encode(converted);
    if value.is_finite() && !format.decode(bits).is_finite() {
        return Err(Error::new(
            "literal",
            "finite literal is outside target floating range",
        ));
    }
    Ok(u128::from(bits))
}

pub(super) fn print(bits: u128, ty: PrimitiveType) -> Result<String> {
    let format = Format::for_type(ty)?;
    let bits = bits as u64;
    let value = format.decode(bits);
    if value.is_nan() {
        let sign = if format.special != Special::ExponentOnly
            && format.special != Special::NanAtNegativeZero
            && bits & format.sign() != 0
        {
            "-"
        } else {
            ""
        };
        if format.special == Special::Ieee {
            let payload = bits & format.fraction_mask();
            if payload != 1_u64 << (format.fraction - 1) {
                return Ok(format!("{sign}nan(0x{payload:x})"));
            }
        }
        return Ok(format!("{sign}nan"));
    }
    let mut result = value.to_string();
    if value.is_finite() && !result.contains(['.', 'e', 'E']) {
        result.push_str(".0");
    }
    Ok(result)
}
