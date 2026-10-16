use crate::{Error, Result};
use std::fmt::Write;

pub(super) fn dictionary_end(source: &str, start: usize) -> Result<usize> {
    let bytes = source.as_bytes();
    let mut index = start;
    let mut depth = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(index + 1);
                }
            }
            b'"' => {
                index += 1;
                while index < bytes.len() && bytes[index] != b'"' {
                    if bytes[index] == b'\\' {
                        index += 1;
                    }
                    index += 1;
                }
                if index >= bytes.len() {
                    return Err(Error::new("syntax", "unterminated dictionary string").at(start));
                }
            }
            _ => (),
        }
        index += 1;
    }
    Err(Error::new("syntax", "unterminated dictionary").at(start))
}

pub(super) fn unquote(source: &str) -> Result<Vec<u8>> {
    let bytes = source.as_bytes();
    if bytes.len() < 2 || !matches!(bytes[0], b'"' | b'\'') || bytes[bytes.len() - 1] != bytes[0] {
        return Err(Error::new("syntax", "expected quoted string"));
    }
    let bytes = &bytes[1..bytes.len() - 1];
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            result.push(bytes[index]);
            index += 1;
            continue;
        }
        index += 1;
        let escape = *bytes
            .get(index)
            .ok_or_else(|| Error::new("syntax", "trailing string escape"))?;
        index += 1;
        match escape {
            b'a' => result.push(7),
            b'b' => result.push(8),
            b'f' => result.push(12),
            b'n' => result.push(b'\n'),
            b'r' => result.push(b'\r'),
            b't' => result.push(b'\t'),
            b'v' => result.push(11),
            b'\\' | b'\'' | b'"' | b'?' => result.push(escape),
            b'0'..=b'7' => {
                let mut value = u16::from(escape - b'0');
                for _ in 0..2 {
                    if let Some(&next @ b'0'..=b'7') = bytes.get(index) {
                        value = value * 8 + u16::from(next - b'0');
                        index += 1;
                    } else {
                        break;
                    }
                }
                result.push(
                    u8::try_from(value).map_err(|_| {
                        Error::new("syntax", "octal string escape exceeds byte range")
                    })?,
                );
            }
            b'x' | b'X' => {
                let start = index;
                let mut value = 0u8;
                while let Some(digit) = bytes
                    .get(index)
                    .and_then(|&byte| (byte as char).to_digit(16))
                {
                    value = value
                        .checked_mul(16)
                        .and_then(|n| n.checked_add(digit as u8))
                        .ok_or_else(|| {
                            Error::new("syntax", "hex string escape exceeds byte range")
                        })?;
                    index += 1;
                }
                if index == start {
                    return Err(Error::new("syntax", "hex string escape requires digits"));
                }
                result.push(value);
            }
            b'u' | b'U' => {
                let count = if escape == b'u' { 4 } else { 8 };
                let mut value = 0u32;
                for _ in 0..count {
                    let digit = bytes
                        .get(index)
                        .and_then(|&byte| (byte as char).to_digit(16))
                        .ok_or_else(|| Error::new("syntax", "incomplete Unicode string escape"))?;
                    value = value
                        .checked_mul(16)
                        .and_then(|n| n.checked_add(digit))
                        .ok_or_else(|| Error::new("syntax", "Unicode string escape overflow"))?;
                    index += 1;
                }
                let character = char::from_u32(value).ok_or_else(|| {
                    Error::new("syntax", "invalid Unicode scalar in string escape")
                })?;
                let mut encoded = [0u8; 4];
                result.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            }
            _ => return Err(Error::new("syntax", "unknown string escape")),
        }
    }
    Ok(result)
}

pub(super) fn quote(bytes: &[u8]) -> Result<String> {
    let mut result = String::from("\"");
    for &byte in bytes {
        match byte {
            b'"' => result.push_str("\\\""),
            b'\\' => result.push_str("\\\\"),
            b'\n' => result.push_str("\\n"),
            b'\r' => result.push_str("\\r"),
            b'\t' => result.push_str("\\t"),
            32..=126 => result.push(byte as char),
            _ => write!(result, "\\{byte:03o}").map_err(|e| Error::new("print", e.to_string()))?,
        }
    }
    result.push('"');
    Ok(result)
}
