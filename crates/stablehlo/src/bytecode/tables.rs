use super::{
    BytecodeError, BytecodeErrorKind as Kind, BytecodeSection, SectionId,
    container::read_section,
    primitives::{Cursor, reserve},
};

/// A dialect namespace and its optional raw version payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeDialect<'a> {
    pub name: &'a [u8],
    pub version: Option<BytecodeSection<'a>>,
}

/// An operation's dialect index and unqualified name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeOperationName<'a> {
    pub dialect: usize,
    pub name: &'a [u8],
    /// Registration was not recorded before bytecode version 5.
    pub registered: Option<bool>,
}

/// An indexed attribute or type encoding, not yet interpreted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedValue<'a> {
    pub dialect: usize,
    pub offset: usize,
    pub custom_encoding: bool,
    pub data: &'a [u8],
}

/// A native operation-property payload; interpretation depends on the op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeProperty<'a> {
    pub offset: usize,
    pub data: &'a [u8],
}

pub(super) fn properties<'a>(
    section: &BytecodeSection<'a>,
) -> Result<Vec<BytecodeProperty<'a>>, BytecodeError> {
    if section.data.is_empty() {
        return Ok(Vec::new());
    }
    let mut cursor = Cursor::new(section.data, section.offset);
    let count = cursor.count()?;
    let mut entries = Vec::new();
    reserve(&mut entries, count, &cursor)?;
    for _ in 0..count {
        let length = cursor.varint()?;
        let length = usize::try_from(length).map_err(|_| cursor.error(Kind::InvalidEntrySize))?;
        let offset = cursor.base + cursor.position;
        entries.push(BytecodeProperty {
            offset,
            data: cursor.bytes(length)?,
        });
    }
    cursor.finish()?;
    Ok(entries)
}

pub(super) fn strings<'a>(section: &BytecodeSection<'a>) -> Result<Vec<&'a [u8]>, BytecodeError> {
    let mut cursor = Cursor::new(section.data, section.offset);
    let count = cursor.count()?;
    let mut lengths = Vec::new();
    reserve(&mut lengths, count, &cursor)?;
    for _ in 0..count {
        let length = cursor.varint()?;
        let length =
            usize::try_from(length).map_err(|_| cursor.error(Kind::InvalidStringLength))?;
        if length == 0 {
            return Err(cursor.error(Kind::InvalidStringLength));
        }
        lengths.push(length);
    }
    let mut strings = Vec::new();
    reserve(&mut strings, count, &cursor)?;
    for length in lengths.into_iter().rev() {
        let value = cursor.bytes(length)?;
        if value.last() != Some(&0) {
            return Err(cursor.error(Kind::InvalidStringTerminator));
        }
        strings.push(&value[..value.len() - 1]);
    }
    cursor.finish()?;
    Ok(strings)
}

fn index(
    cursor: &Cursor<'_>,
    value: u64,
    length: usize,
    table: &'static str,
) -> Result<usize, BytecodeError> {
    let result = usize::try_from(value)
        .ok()
        .filter(|&value| value < length)
        .ok_or_else(|| {
            cursor.error(Kind::InvalidIndex {
                table,
                index: value,
            })
        })?;
    Ok(result)
}

pub(super) fn dialects<'a>(
    section: &BytecodeSection<'a>,
    version: u64,
    strings: &[&'a [u8]],
) -> Result<(Vec<BytecodeDialect<'a>>, Vec<BytecodeOperationName<'a>>), BytecodeError> {
    let mut cursor = Cursor::new(section.data, section.offset);
    let count = cursor.count()?;
    let mut dialects = Vec::new();
    reserve(&mut dialects, count, &cursor)?;
    for _ in 0..count {
        let code = cursor.varint()?;
        let (name, versioned) = if version == 0 {
            (code, false)
        } else {
            (code >> 1, code & 1 != 0)
        };
        let name = strings[index(&cursor, name, strings.len(), "string")?];
        let version = if versioned {
            let section = read_section(&mut cursor)?;
            if section.id != SectionId::DialectVersions {
                return Err(cursor.error(Kind::InvalidDialectVersionSection));
            }
            Some(section)
        } else {
            None
        };
        dialects.push(BytecodeDialect { name, version });
    }
    let expected = (version >= 4).then(|| cursor.count()).transpose()?;
    let mut operations = Vec::new();
    if let Some(count) = expected {
        reserve(&mut operations, count, &cursor)?;
    }
    while cursor.remaining() != 0 {
        let value = cursor.varint()?;
        let dialect = index(&cursor, value, dialects.len(), "dialect")?;
        let count = cursor.count()?;
        if count == 0 || expected.is_some_and(|n| count > n.saturating_sub(operations.len())) {
            return Err(cursor.error(Kind::InvalidGroupSize));
        }
        reserve(&mut operations, count, &cursor)?;
        for _ in 0..count {
            let code = cursor.varint()?;
            let (name, registered) = if version >= 5 {
                (code >> 1, Some(code & 1 != 0))
            } else {
                (code, None)
            };
            let name = strings[index(&cursor, name, strings.len(), "string")?];
            operations.push(BytecodeOperationName {
                dialect,
                name,
                registered,
            });
        }
    }
    if expected.is_some_and(|count| count != operations.len()) {
        return Err(cursor.error(Kind::OperationCountMismatch));
    }
    Ok((dialects, operations))
}

pub(super) fn entries<'a>(
    offsets: &BytecodeSection<'_>,
    data: &BytecodeSection<'a>,
    num_dialects: usize,
) -> Result<(Vec<EncodedValue<'a>>, Vec<EncodedValue<'a>>), BytecodeError> {
    let mut cursor = Cursor::new(offsets.data, offsets.offset);
    let num_attributes = cursor.count()?;
    let num_types = cursor.count()?;
    let mut payload = Cursor::new(data.data, data.offset);
    let mut parse = |count| {
        let mut entries = Vec::new();
        reserve(&mut entries, count, &cursor)?;
        while entries.len() < count {
            let dialect = cursor.varint()?;
            let dialect = index(&cursor, dialect, num_dialects, "dialect")?;
            let group = cursor.count()?;
            if group == 0 || group > count - entries.len() {
                return Err(cursor.error(Kind::InvalidGroupSize));
            }
            for _ in 0..group {
                let code = cursor.varint()?;
                let length =
                    usize::try_from(code >> 1).map_err(|_| cursor.error(Kind::InvalidEntrySize))?;
                if length > payload.remaining() {
                    return Err(cursor.error(Kind::InvalidEntrySize));
                }
                let offset = payload.base + payload.position;
                entries.push(EncodedValue {
                    dialect,
                    offset,
                    custom_encoding: code & 1 != 0,
                    data: payload.bytes(length)?,
                });
            }
        }
        Ok(entries)
    };
    let attributes = parse(num_attributes)?;
    let types = parse(num_types)?;
    cursor.finish()?;
    payload.finish()?;
    Ok((attributes, types))
}
