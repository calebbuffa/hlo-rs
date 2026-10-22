// SPDX-License-Identifier: Apache-2.0
//
// Binary16 conversion implemented from the IEEE 754 field and rounding rules.

pub(super) fn half_to_float(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);

    let result = match (exponent, fraction) {
        (0, 0) => sign,
        (0, fraction) => {
            let shift = fraction.leading_zeros() - 21;
            let significand = fraction << shift;
            let float_exponent = 113 - shift;
            sign | (float_exponent << 23) | ((significand & 0x03ff) << 13)
        }
        (0x1f, fraction) => sign | 0x7f80_0000 | (fraction << 13),
        (exponent, fraction) => {
            let float_exponent = u32::from(exponent) + 112;
            sign | (float_exponent << 23) | (fraction << 13)
        }
    };

    f32::from_bits(result)
}

fn round_right_to_even(value: u32, shift: u32) -> u32 {
    let retained = value >> shift;
    let discarded = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    retained + u32::from(discarded > halfway || (discarded == halfway && retained & 1 != 0))
}

pub(super) fn float_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;

    if exponent == 0xff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }

    if exponent == 0 {
        return sign;
    }

    let unbiased_exponent = exponent as i32 - 127;
    if unbiased_exponent > 15 {
        return sign | 0x7c00;
    }

    let significand = 0x0080_0000 | fraction;
    if unbiased_exponent >= -14 {
        let mut rounded = round_right_to_even(significand, 13);
        let mut half_exponent = unbiased_exponent + 15;
        if rounded == 0x800 {
            rounded = 0x400;
            half_exponent += 1;
        }
        if half_exponent >= 0x1f {
            return sign | 0x7c00;
        }
        return sign | ((half_exponent as u16) << 10) | ((rounded as u16) & 0x03ff);
    }

    let shift = (-unbiased_exponent - 1) as u32;
    if shift > 24 {
        return sign;
    }
    sign | round_right_to_even(significand, shift) as u16
}

#[cfg(test)]
mod tests {
    use super::{float_to_half, half_to_float};

    #[test]
    fn binary16_decoding_matches_exact_field_values() {
        for (bits, expected) in [
            (0x0000, 0x0000_0000),
            (0x8000, 0x8000_0000),
            (0x0001, 0x3380_0000),
            (0x03ff, 0x387f_c000),
            (0x0400, 0x3880_0000),
            (0x3c00, 0x3f80_0000),
            (0x7bff, 0x477f_e000),
            (0x7c00, 0x7f80_0000),
            (0xfc00, 0xff80_0000),
            (0x7c01, 0x7f80_2000),
            (0xfe00, 0xffc0_0000),
        ] {
            assert_eq!(half_to_float(bits).to_bits(), expected, "{bits:#06x}");
        }
    }

    #[test]
    fn binary16_encodings_round_trip_except_canonicalized_nans() {
        for bits in 0..=u16::MAX {
            let exponent = bits & 0x7c00;
            let fraction = bits & 0x03ff;
            let expected = if exponent == 0x7c00 && fraction != 0 {
                (bits & 0x8000) | 0x7e00
            } else {
                bits
            };
            assert_eq!(float_to_half(half_to_float(bits)), expected, "{bits:#06x}");
        }
    }

    #[test]
    fn binary32_to_binary16_rounds_ties_to_even() {
        assert_eq!(float_to_half(f32::from_bits(0x3f80_1000)), 0x3c00);
        assert_eq!(float_to_half(f32::from_bits(0x3f80_3000)), 0x3c02);
        assert_eq!(float_to_half(f32::from_bits(0x3300_0000)), 0x0000);
        assert_eq!(float_to_half(f32::from_bits(0x3380_0000)), 0x0001);
        assert_eq!(float_to_half(f32::from_bits(0x387f_e000)), 0x0400);
        assert_eq!(float_to_half(f32::from_bits(0x477f_e000)), 0x7bff);
        assert_eq!(float_to_half(f32::from_bits(0x477f_f000)), 0x7c00);
        assert_eq!(float_to_half(f32::from_bits(0x3300_0001)), 0x0001);
    }

    #[test]
    fn binary32_to_binary16_handles_sign_overflow_and_nan_policy() {
        assert_eq!(float_to_half(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(float_to_half(-0.0), 0x8000);
        assert_eq!(float_to_half(f32::from_bits(0x477f_efff)), 0x7bff);
        assert_eq!(float_to_half(f32::from_bits(0x477f_f000)), 0x7c00);
        assert_eq!(float_to_half(f32::from_bits(0x7f80_0000)), 0x7c00);
        assert_eq!(float_to_half(f32::from_bits(0x7f80_0001)), 0x7e00);
        assert_eq!(float_to_half(f32::from_bits(0xffc0_0000)), 0xfe00);
        assert_eq!(float_to_half(f32::from_bits(0xff80_0001)), 0xfe00);
    }
}
