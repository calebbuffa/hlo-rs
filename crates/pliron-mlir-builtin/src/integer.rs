use std::{cmp::Ordering, fmt, ops::Neg, str::FromStr};

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

/// Error returned when an integer literal cannot be parsed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegerParseError;

impl fmt::Display for IntegerParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid arbitrary-precision integer literal")
    }
}

impl std::error::Error for IntegerParseError {}

impl IntegerValue {
    /// The integer zero.
    pub const ZERO: Self = Self(Repr::Small(0));

    /// Parses a base-10 integer, including an optional leading sign.
    ///
    /// Returns `None` when the input is empty, malformed, or contains no digits.
    pub fn parse_decimal(value: &str) -> Option<Self> {
        Self::parse_radix(value, 10)
    }

    /// Parses an integer using a radix from 2 through 36.
    ///
    /// An optional leading `+` or `-` is accepted; digits are case-insensitive for
    /// radices above ten. Returns `None` for an invalid radix, empty digit
    /// sequence, or any digit not valid in the selected radix.
    pub fn parse_radix(value: &str, radix: u32) -> Option<Self> {
        let (negative, digits) = if let Some(digits) = value.strip_prefix('-') {
            (true, digits)
        } else if let Some(digits) = value.strip_prefix('+') {
            (false, digits)
        } else {
            (false, value)
        };
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

    /// Returns the value when it is representable as a nonnegative `u128`.
    pub fn to_u128(&self) -> Option<u128> {
        if self.negative() {
            return None;
        }
        if let Repr::Small(value) = self.0 {
            return u128::try_from(value).ok();
        }
        if self.magnitude_word_len() > 2 {
            return None;
        }
        Some(u128::from(self.magnitude_word(0)) | (u128::from(self.magnitude_word(1)) << 64))
    }

    /// Returns the low 128 bits of the integer in two's-complement form.
    ///
    /// Values wider than 128 bits are truncated; negative values are returned
    /// modulo 2<sup>128</sup>.
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

    /// Returns the additive inverse of this value.
    pub fn negated(&self) -> Self {
        -self.clone()
    }

    pub(crate) fn is_zero(&self) -> bool {
        matches!(self.0, Repr::Small(0))
    }

    /// Encodes the low `width` bits as little-endian two's-complement words.
    ///
    /// The vector contains `ceil(width / 64)` words, and unused high bits in
    /// its final word are zero. Width zero produces an empty vector.
    pub fn to_twos_complement_words(&self, width: u32) -> Vec<u64> {
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

    /// Decodes up to `width` low bits from little-endian words.
    ///
    /// When `signed` is true, the highest selected bit is treated as the sign
    /// bit; otherwise the bits are interpreted as an unsigned magnitude.
    /// Missing input words are treated as zero and excess words/bits are
    /// ignored. A zero width returns [`Self::ZERO`].
    pub fn from_twos_complement_words(words: &[u64], width: u32, signed: bool) -> Self {
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

    /// Returns whether the value is representable in the requested bit width.
    ///
    /// `Unsigned` accepts nonnegative values; `Signed` and `Signless` use the
    /// signed two's-complement range. Only zero fits a zero-bit width.
    pub fn fits_width(&self, width: u32, signedness: IntegerSignedness) -> bool {
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
impl From<u128> for IntegerValue {
    fn from(value: u128) -> Self {
        if value <= i128::MAX as u128 {
            Self(Repr::Small(value as i128))
        } else {
            Self::from_magnitude(false, vec![value as u64, (value >> 64) as u64])
        }
    }
}

impl FromStr for IntegerValue {
    type Err = IntegerParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse_decimal(value).ok_or(IntegerParseError)
    }
}

impl Neg for IntegerValue {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        match &mut self.0 {
            Repr::Small(value) if *value == i128::MIN => {
                Self::from_magnitude(false, vec![0, 1_u64 << 63])
            }
            Repr::Small(value) => {
                *value = -*value;
                self
            }
            Repr::Big { negative, .. } => {
                *negative = !*negative;
                self
            }
        }
    }
}

macro_rules! impl_from_signed {
    ($($ty:ty),* $(,)?) => {$ (
        impl From<$ty> for IntegerValue {
            fn from(value: $ty) -> Self {
                Self(Repr::Small(value as i128))
            }
        }
    )*};
}
impl_from_signed!(i8, i16, i32, i64, isize);

macro_rules! impl_from_unsigned {
    ($($ty:ty),* $(,)?) => {$ (
        impl From<$ty> for IntegerValue {
            fn from(value: $ty) -> Self {
                Self(Repr::Small(value as i128))
            }
        }
    )*};
}
impl_from_unsigned!(u8, u16, u32, u64, usize);

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

macro_rules! try_into_signed {
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
try_into_signed!(i8, i16, i32, i64, i128, isize);

macro_rules! try_into_unsigned {
    ($($ty:ty),* $(,)?) => {$ (
        impl TryFrom<IntegerValue> for $ty {
            type Error = pliron::result::Error;

            fn try_from(value: IntegerValue) -> Result<Self, Self::Error> {
                let value = value.to_u128().ok_or_else(|| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())?;
                <$ty>::try_from(value).map_err(|_| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())
            }
        }
        impl TryFrom<&IntegerValue> for $ty {
            type Error = pliron::result::Error;

            fn try_from(value: &IntegerValue) -> Result<Self, Self::Error> {
                let value = value.to_u128().ok_or_else(|| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())?;
                <$ty>::try_from(value).map_err(|_| crate::Diagnostic::IntegerOutOfRange { target: stringify!($ty) }.into_error())
            }
        }
    )*};
}
try_into_unsigned!(u8, u16, u32, u64, u128, usize);
