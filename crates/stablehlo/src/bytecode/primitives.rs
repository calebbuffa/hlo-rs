use super::{BytecodeError, BytecodeErrorKind};

/// Decodes one MLIR PrefixVarInt and returns its value and byte length.
///
/// Accepts non-minimal encodings, as the reference reader does. This is not
/// LEB128. Bytes following the first integer are left unconsumed.
pub fn decode_varint(source: &[u8]) -> Result<(u64, usize), BytecodeError> {
    let Some(&first) = source.first() else {
        return Err(BytecodeError {
            offset: 0,
            kind: BytecodeErrorKind::UnexpectedEnd,
        });
    };
    let extra = first.trailing_zeros() as usize;
    if source.len() <= extra {
        return Err(BytecodeError {
            offset: source.len(),
            kind: BytecodeErrorKind::UnexpectedEnd,
        });
    }
    if extra == 8 {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&source[1..9]);
        return Ok((u64::from_le_bytes(bytes), 9));
    }
    let mut value = u64::from(first) >> (extra + 1);
    for (index, &byte) in source[1..=extra].iter().enumerate() {
        value |= u64::from(byte) << (7 - extra + index * 8);
    }
    Ok((value, extra + 1))
}

/// Appends the canonical MLIR PrefixVarInt encoding of an unsigned integer.
pub fn encode_varint(value: u64, output: &mut Vec<u8>) {
    let bits = 64 - value.leading_zeros();
    let length = bits.max(1).div_ceil(7) as usize;
    if length > 8 {
        output.push(0);
        output.extend_from_slice(&value.to_le_bytes());
        return;
    }
    let extra = length - 1;
    let first_bits = 7 - extra;
    output.push(((value << (extra + 1)) as u8) | (1 << extra));
    for index in 0..extra {
        output.push((value >> (first_bits + index * 8)) as u8);
    }
}

/// Decodes a zigzag-encoded signed MLIR PrefixVarInt.
pub fn decode_signed_varint(source: &[u8]) -> Result<(i64, usize), BytecodeError> {
    let (value, length) = decode_varint(source)?;
    Ok((((value >> 1) as i64) ^ -((value & 1) as i64), length))
}

/// Appends a zigzag-encoded signed MLIR PrefixVarInt.
pub fn encode_signed_varint(value: i64, output: &mut Vec<u8>) {
    encode_varint(((value as u64) << 1) ^ ((value >> 63) as u64), output);
}

pub(super) struct Cursor<'a> {
    pub data: &'a [u8],
    pub position: usize,
    pub base: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8], base: usize) -> Self {
        Self {
            data,
            position: 0,
            base,
        }
    }

    pub fn error(&self, kind: BytecodeErrorKind) -> BytecodeError {
        BytecodeError {
            offset: self.base + self.position,
            kind,
        }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    pub fn bytes(&mut self, length: usize) -> Result<&'a [u8], BytecodeError> {
        if length > self.remaining() {
            return Err(self.error(BytecodeErrorKind::UnexpectedEnd));
        }
        let start = self.position;
        self.position += length;
        Ok(&self.data[start..self.position])
    }

    pub fn byte(&mut self) -> Result<u8, BytecodeError> {
        Ok(self.bytes(1)?[0])
    }

    pub fn varint(&mut self) -> Result<u64, BytecodeError> {
        let (value, length) = decode_varint(&self.data[self.position..]).map_err(|mut error| {
            error.offset += self.base + self.position;
            error
        })?;
        self.position += length;
        Ok(value)
    }

    pub fn count(&mut self) -> Result<usize, BytecodeError> {
        let value = self.varint()?;
        let count =
            usize::try_from(value).map_err(|_| self.error(BytecodeErrorKind::ResourceLimit))?;
        // Each indexed entry must consume at least one byte.
        if count > self.remaining() {
            return Err(self.error(BytecodeErrorKind::ResourceLimit));
        }
        Ok(count)
    }

    pub fn finish(&self) -> Result<(), BytecodeError> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(self.error(BytecodeErrorKind::TrailingData))
        }
    }
}

pub(super) fn reserve<T>(
    values: &mut Vec<T>,
    count: usize,
    cursor: &Cursor<'_>,
) -> Result<(), BytecodeError> {
    values
        .try_reserve_exact(count)
        .map_err(|_| cursor.error(BytecodeErrorKind::ResourceLimit))
}
