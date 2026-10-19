use std::{cmp::Ordering, fmt};

use pliron::builtin::types::Signedness as IntegerSignedness;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum Repr {
    Small(i128),
    Big { negative: bool, words: Vec<u64> },
}

/// An arbitrary-precision integer used for MLIR integer attribute payloads.
///
/// Values use a canonical signed-magnitude representation. Values that fit in
/// `i128` are stored inline; larger magnitudes use little-endian `u64` words.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IntegerValue(Repr);

impl IntegerValue {
    /// The integer zero.
    pub const ZERO: Self = Self(Repr::Small(0));

    /// Parses a base-10 integer, including an optional leading minus sign.
    ///
    /// Returns `None` when the input is empty, malformed, or contains no digits.
    pub fn parse_decimal(value: &str) -> Option<Self> {
        Self::parse_radix(value, 10)
    }

    pub(crate) fn parse_radix(value: &str, radix: u32) -> Option<Self> {
        let (negative, digits) = value
            .strip_prefix('-')
            .map_or((false, value), |s| (true, s));
        if digits.is_empty() || !(2..=36).contains(&radix) {
            return None;
        }
        let mut words = vec![0_u64];
        for digit in digits.bytes() {
            let value = char::from(digit).to_digit(radix)?;
            let mut carry = u128::from(value);
            for word in &mut words {
                let next = u128::from(*word) * u128::from(radix) + carry;
                *word = next as u64; // Keep the low word; `carry` holds the high bits.
                carry = next >> 64;
            }
            if carry != 0 {
                words.push(carry as u64);
            }
        }
        Some(Self::from_magnitude(negative, words))
    }

    fn from_magnitude(negative: bool, mut words: Vec<u64>) -> Self {
        while words.last() == Some(&0) {
            words.pop();
        }
        if words.len() <= 2 {
            let magnitude = u128::from(*words.first().unwrap_or(&0))
                | (u128::from(*words.get(1).unwrap_or(&0)) << 64);
            if negative {
                if magnitude <= (1_u128 << 127) {
                    return Self(Repr::Small(if magnitude == 1_u128 << 127 {
                        i128::MIN
                    } else {
                        -(magnitude as i128)
                    }));
                }
            } else if magnitude <= i128::MAX as u128 {
                return Self(Repr::Small(magnitude as i128));
            }
        }
        if words.is_empty() {
            return Self::ZERO;
        }
        Self(Repr::Big { negative, words })
    }

    fn magnitude_words(&self) -> (bool, Vec<u64>) {
        match &self.0 {
            Repr::Small(value) => {
                let magnitude = value.unsigned_abs();
                let mut words = vec![magnitude as u64, (magnitude >> 64) as u64];
                while words.last() == Some(&0) {
                    words.pop();
                }
                (*value < 0, words)
            }
            Repr::Big { negative, words } => (*negative, words.clone()),
        }
    }

    fn negative(&self) -> bool {
        match &self.0 {
            Repr::Small(value) => *value < 0,
            Repr::Big { negative, .. } => *negative,
        }
    }

    fn magnitude_word_len(&self) -> usize {
        match &self.0 {
            Repr::Small(value) => {
                let magnitude = value.unsigned_abs();
                if magnitude >> 64 != 0 {
                    2
                } else if magnitude != 0 {
                    1
                } else {
                    0
                }
            }
            Repr::Big { words, .. } => words.len(),
        }
    }

    fn magnitude_word(&self, index: usize) -> u64 {
        match &self.0 {
            Repr::Small(value) => {
                let magnitude = value.unsigned_abs();
                match index {
                    0 => magnitude as u64,
                    1 => (magnitude >> 64) as u64,
                    _ => 0,
                }
            }
            Repr::Big { words, .. } => words.get(index).copied().unwrap_or(0),
        }
    }

    /// Returns the value when it is representable as an `i128`.
    pub fn to_i128(&self) -> Option<i128> {
        match self.0 {
            Repr::Small(value) => Some(value),
            Repr::Big { .. } => None,
        }
    }

    pub const fn from_i64(value: i64) -> Self {
        Self(Repr::Small(value as i128))
    }

    pub fn low_u128(&self) -> u128 {
        match &self.0 {
            Repr::Small(value) => *value as u128, // Preserve the low two's-complement bits.
            Repr::Big { negative, words } => {
                let magnitude = u128::from(words.first().copied().unwrap_or(0))
                    | (u128::from(words.get(1).copied().unwrap_or(0)) << 64);
                if *negative {
                    magnitude.wrapping_neg()
                } else {
                    magnitude
                }
            }
        }
    }

    pub fn negated(&self) -> Self {
        let (negative, words) = self.magnitude_words();
        Self::from_magnitude(!negative, words)
    }

    pub(crate) fn is_zero(&self) -> bool {
        matches!(self.0, Repr::Small(0))
    }

    pub(crate) fn to_twos_complement_words(&self, width: u32) -> Vec<u64> {
        let count = width.div_ceil(64) as usize;
        if count == 0 {
            return Vec::new();
        }
        let negative = self.negative();
        let mut words = vec![0_u64; count];
        for (index, target) in words.iter_mut().enumerate() {
            *target = self.magnitude_word(index);
        }
        if negative {
            for word in &mut words {
                *word = !*word;
            }
            let mut carry = true;
            for word in &mut words {
                if carry {
                    let (value, overflow) = word.overflowing_add(1);
                    *word = value;
                    carry = overflow;
                }
            }
        }
        let tail = width % 64;
        if tail != 0
            && let Some(last) = words.last_mut()
        {
            *last &= (1_u64 << tail) - 1;
        }
        words
    }

    pub(crate) fn from_twos_complement_words(words: &[u64], width: u32, signed: bool) -> Self {
        if width == 0 {
            return Self::ZERO;
        }
        let count = width.div_ceil(64) as usize;
        let mut bits = vec![0_u64; count];
        for (target, source) in bits.iter_mut().zip(words.iter()) {
            *target = *source;
        }
        let tail = width % 64;
        if tail != 0
            && let Some(last) = bits.last_mut()
        {
            *last &= (1_u64 << tail) - 1;
        }
        let sign_index = ((width - 1) / 64) as usize;
        let sign_bit = (width - 1) % 64;
        let negative = signed && ((bits[sign_index] >> sign_bit) & 1 != 0);
        if !negative {
            return Self::from_magnitude(false, bits);
        }
        for word in &mut bits {
            *word = !*word;
        }
        if tail != 0
            && let Some(last) = bits.last_mut()
        {
            *last &= (1_u64 << tail) - 1;
        }
        let mut carry = true;
        for word in &mut bits {
            if carry {
                let (value, overflow) = word.overflowing_add(1);
                *word = value;
                carry = overflow;
            }
        }
        Self::from_magnitude(true, bits)
    }

    pub(crate) fn fits_width(&self, width: u32, signedness: IntegerSignedness) -> bool {
        if width == 0 {
            return self.is_zero();
        }
        let bits = self.magnitude_bit_length();
        let negative = self.negative();
        match signedness {
            IntegerSignedness::Unsigned => !negative && bits <= width,
            IntegerSignedness::Signed | IntegerSignedness::Signless => {
                if negative {
                    bits < width || (bits == width && self.is_power_of_two(width - 1))
                } else {
                    bits < width
                }
            }
        }
    }

    fn magnitude_bit_length(&self) -> u32 {
        let count = self.magnitude_word_len();
        if count == 0 {
            0
        } else {
            64 * (count as u32 - 1) + (64 - self.magnitude_word(count - 1).leading_zeros())
        }
    }

    fn is_power_of_two(&self, bit: u32) -> bool {
        let index = (bit / 64) as usize;
        self.magnitude_word(index) == 1_u64 << (bit % 64)
            && (0..self.magnitude_word_len())
                .all(|word_index| word_index == index || self.magnitude_word(word_index) == 0)
    }
}

impl From<i128> for IntegerValue {
    fn from(value: i128) -> Self {
        Self(Repr::Small(value))
    }
}
impl From<i64> for IntegerValue {
    fn from(value: i64) -> Self {
        Self(Repr::Small(i128::from(value)))
    }
}
impl From<i32> for IntegerValue {
    fn from(value: i32) -> Self {
        Self(Repr::Small(i128::from(value)))
    }
}
impl From<u64> for IntegerValue {
    fn from(value: u64) -> Self {
        Self(Repr::Small(i128::from(value)))
    }
}

impl fmt::Display for IntegerValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Repr::Small(value) = self.0 {
            return write!(formatter, "{value}");
        }
        let (negative, mut words) = self.magnitude_words();
        let mut digits = Vec::new();
        while words.iter().any(|word| *word != 0) {
            let mut remainder = 0_u128;
            for word in words.iter_mut().rev() {
                let value = (remainder << 64) | u128::from(*word);
                *word = (value / 10) as u64;
                remainder = value % 10;
            }
            digits.push(b'0' + remainder as u8);
        }
        if negative {
            formatter.write_str("-")?;
        }
        for digit in digits.iter().rev() {
            write!(formatter, "{}", char::from(*digit))?;
        }
        Ok(())
    }
}

impl TryFrom<IntegerValue> for i128 {
    type Error = pliron::result::Error;

    fn try_from(value: IntegerValue) -> Result<Self, Self::Error> {
        value
            .to_i128()
            .ok_or_else(|| crate::Diagnostic::IntegerOutOfRange { target: "i128" }.into_error())
    }
}
impl TryFrom<&IntegerValue> for i128 {
    type Error = pliron::result::Error;

    fn try_from(value: &IntegerValue) -> Result<Self, Self::Error> {
        value
            .to_i128()
            .ok_or_else(|| crate::Diagnostic::IntegerOutOfRange { target: "i128" }.into_error())
    }
}

impl PartialOrd for IntegerValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for IntegerValue {
    fn cmp(&self, other: &Self) -> Ordering {
        if let (Repr::Small(left), Repr::Small(right)) = (&self.0, &other.0) {
            return left.cmp(right);
        }
        match (self.negative(), other.negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (negative, _) => {
                let order = self
                    .magnitude_word_len()
                    .cmp(&other.magnitude_word_len())
                    .then_with(|| {
                        (0..self.magnitude_word_len())
                            .rev()
                            .map(|index| self.magnitude_word(index))
                            .cmp(
                                (0..other.magnitude_word_len())
                                    .rev()
                                    .map(|index| other.magnitude_word(index)),
                            )
                    });
                if negative { order.reverse() } else { order }
            }
        }
    }
}
impl PartialEq<i128> for IntegerValue {
    fn eq(&self, other: &i128) -> bool {
        self.to_i128() == Some(*other)
    }
}
impl PartialOrd<i128> for IntegerValue {
    fn partial_cmp(&self, other: &i128) -> Option<Ordering> {
        Some(self.cmp(&Self::from(*other)))
    }
}
impl PartialEq<IntegerValue> for i128 {
    fn eq(&self, other: &IntegerValue) -> bool {
        other == self
    }
}
impl PartialOrd<IntegerValue> for i128 {
    fn partial_cmp(&self, other: &IntegerValue) -> Option<Ordering> {
        Some(IntegerValue::from(*self).cmp(other))
    }
}

macro_rules! try_into_small {
    ($($ty:ty),* $(,)?) => {$ (
        impl TryFrom<IntegerValue> for $ty {
            type Error = pliron::result::Error;

            fn try_from(value: IntegerValue) -> Result<Self, Self::Error> {
                let value = value.to_i128().ok_or_else(|| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())?;
                <$ty>::try_from(value).map_err(|_| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())
            }
        }
        impl TryFrom<&IntegerValue> for $ty {
            type Error = pliron::result::Error;

            fn try_from(value: &IntegerValue) -> Result<Self, Self::Error> {
                let value = value.to_i128().ok_or_else(|| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())?;
                <$ty>::try_from(value).map_err(|_| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())
            }
        }
    )*};
}
try_into_small!(i64, i32, usize, u64);

#[cfg(test)]
mod tests {
    use super::IntegerValue;
    use pliron::builtin::types::Signedness as IntegerSignedness;

    #[test]
    fn decimal_and_apint_word_conversions_preserve_wide_values() {
        let unsigned = IntegerValue::parse_decimal("340282366920938463463374607431768211455");
        assert!(unsigned.is_some());
        if let Some(unsigned) = unsigned {
            assert_eq!(
                unsigned.to_string(),
                "340282366920938463463374607431768211455"
            );
            assert_eq!(
                u128::from(unsigned.to_twos_complement_words(128)[0]),
                u64::MAX as u128
            );
            assert_eq!(
                IntegerValue::from_twos_complement_words(
                    &unsigned.to_twos_complement_words(128),
                    128,
                    false
                ),
                unsigned
            );
            assert_eq!(unsigned.to_i128(), None);
        }

        let below_i128_min =
            IntegerValue::parse_decimal("-170141183460469231731687303715884105729");
        assert!(below_i128_min.is_some());
        if let Some(below_i128_min) = below_i128_min {
            assert!(below_i128_min.fits_width(129, IntegerSignedness::Signed));
            assert!(!below_i128_min.fits_width(128, IntegerSignedness::Signed));
            let words = below_i128_min.to_twos_complement_words(129);
            assert_eq!(
                IntegerValue::from_twos_complement_words(&words, 129, true),
                below_i128_min
            );
            let (seven, eight) = (IntegerValue::from(7_i128), IntegerValue::from(8_i128));
            let (minus_seven, minus_eight) =
                (IntegerValue::from(-7_i128), IntegerValue::from(-8_i128));
            assert!(eight > seven);
            assert!(minus_eight < minus_seven);
        }
    }
}
