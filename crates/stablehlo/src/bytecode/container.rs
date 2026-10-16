use super::{
    BytecodeDialect, BytecodeError, BytecodeErrorKind, BytecodeOperationName, BytecodeProperty,
    EncodedValue, primitives::Cursor, tables,
};

/// Section identifiers from the pinned MLIR `Bytecode/Encoding.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SectionId {
    Strings = 0,
    Dialects = 1,
    AttributesAndTypes = 2,
    AttributeAndTypeOffsets = 3,
    Ir = 4,
    Resources = 5,
    ResourceOffsets = 6,
    DialectVersions = 7,
    Properties = 8,
}

impl SectionId {
    fn from_byte(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::Strings,
            1 => Self::Dialects,
            2 => Self::AttributesAndTypes,
            3 => Self::AttributeAndTypeOffsets,
            4 => Self::Ir,
            5 => Self::Resources,
            6 => Self::ResourceOffsets,
            7 => Self::DialectVersions,
            8 => Self::Properties,
            _ => return None,
        })
    }
}

/// A borrowed section body; offsets are relative to the start of the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeSection<'a> {
    pub id: SectionId,
    pub offset: usize,
    /// An explicitly encoded alignment, if present in the section header.
    pub alignment: Option<u32>,
    pub data: &'a [u8],
}

/// An inspected MLIR bytecode container and its indexed tables.
///
/// The producer and table strings are byte strings, not necessarily UTF-8.
/// Section bodies are borrowed without assuming host pointer alignment.
/// IR, properties, resources and custom attribute/type payloads are preserved
/// as bytes; they have not been semantically decoded or verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeContainer<'a> {
    version: u64,
    producer: &'a [u8],
    sections: Vec<BytecodeSection<'a>>,
    strings: Vec<&'a [u8]>,
    dialects: Vec<BytecodeDialect<'a>>,
    operation_names: Vec<BytecodeOperationName<'a>>,
    attributes: Vec<EncodedValue<'a>>,
    types: Vec<EncodedValue<'a>>,
    properties: Vec<BytecodeProperty<'a>>,
}

impl<'a> BytecodeContainer<'a> {
    /// Checks framing, required sections, strings, dialect/name indexes and
    /// attribute/type entry boundaries. Does not deserialize a Rust module.
    pub fn parse(source: &'a [u8]) -> Result<Self, BytecodeError> {
        let mut cursor = Cursor::new(source, 0);
        if cursor.bytes(4)? != b"ML\xefR" {
            return Err(cursor.error(BytecodeErrorKind::InvalidMagic));
        }
        let version = cursor.varint()?;
        if version > 6 {
            return Err(cursor.error(BytecodeErrorKind::UnsupportedVersion(version)));
        }
        let length = source[cursor.position..]
            .iter()
            .position(|&byte| byte == 0)
            .ok_or_else(|| cursor.error(BytecodeErrorKind::MissingTerminator))?;
        let producer = cursor.bytes(length)?;
        cursor.byte()?;
        let mut sections = Vec::with_capacity(9);
        let mut seen = [false; 9];
        while cursor.remaining() > 0 {
            let section = read_section(&mut cursor)?;
            if seen[section.id as usize] {
                return Err(cursor.error(BytecodeErrorKind::DuplicateSection(section.id)));
            }
            seen[section.id as usize] = true;
            sections.push(section);
        }
        for id in [
            SectionId::Strings,
            SectionId::Dialects,
            SectionId::AttributesAndTypes,
            SectionId::AttributeAndTypeOffsets,
            SectionId::Ir,
        ]
        .into_iter()
        .chain((version >= 5).then_some(SectionId::Properties))
        {
            if !seen[id as usize] {
                return Err(cursor.error(BytecodeErrorKind::MissingSection(id)));
            }
        }
        let mut container = Self {
            version,
            producer,
            sections,
            strings: Vec::new(),
            dialects: Vec::new(),
            operation_names: Vec::new(),
            attributes: Vec::new(),
            types: Vec::new(),
            properties: Vec::new(),
        };
        // Required section presence has already been checked.
        container.strings = tables::strings(container.section(SectionId::Strings).unwrap())?;
        (container.dialects, container.operation_names) = tables::dialects(
            container.section(SectionId::Dialects).unwrap(),
            version,
            &container.strings,
        )?;
        (container.attributes, container.types) = tables::entries(
            container
                .section(SectionId::AttributeAndTypeOffsets)
                .unwrap(),
            container.section(SectionId::AttributesAndTypes).unwrap(),
            container.dialects.len(),
        )?;
        if let Some(section) = container.section(SectionId::Properties) {
            container.properties = tables::properties(section)?;
        }
        Ok(container)
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn producer(&self) -> &'a [u8] {
        self.producer
    }

    /// Sections in their original file order.
    pub fn sections(&self) -> &[BytecodeSection<'a>] {
        &self.sections
    }

    pub fn section(&self, id: SectionId) -> Option<&BytecodeSection<'a>> {
        self.sections.iter().find(|section| section.id == id)
    }

    pub fn strings(&self) -> &[&'a [u8]] {
        &self.strings
    }

    pub fn dialects(&self) -> &[BytecodeDialect<'a>] {
        &self.dialects
    }

    pub fn operation_names(&self) -> &[BytecodeOperationName<'a>] {
        &self.operation_names
    }

    pub fn attributes(&self) -> &[EncodedValue<'a>] {
        &self.attributes
    }

    pub fn types(&self) -> &[EncodedValue<'a>] {
        &self.types
    }

    pub fn properties(&self) -> &[BytecodeProperty<'a>] {
        &self.properties
    }
}

pub(super) fn read_section<'a>(
    cursor: &mut Cursor<'a>,
) -> Result<BytecodeSection<'a>, BytecodeError> {
    let code = cursor.byte()?;
    let id = SectionId::from_byte(code & 0x7f)
        .ok_or_else(|| cursor.error(BytecodeErrorKind::InvalidSection(code & 0x7f)))?;
    let length = cursor.varint()?;
    let alignment = if code & 0x80 != 0 {
        let alignment = cursor.varint()?;
        if alignment > u64::from(u32::MAX) || !alignment.is_power_of_two() {
            return Err(cursor.error(BytecodeErrorKind::InvalidAlignment(alignment)));
        }
        let offset = (cursor.base + cursor.position) as u64;
        let padding = (alignment - (offset & (alignment - 1))) & (alignment - 1);
        let padding =
            usize::try_from(padding).map_err(|_| cursor.error(BytecodeErrorKind::ResourceLimit))?;
        for &byte in cursor.bytes(padding)? {
            if byte != 0xcb {
                return Err(cursor.error(BytecodeErrorKind::InvalidPadding(byte)));
            }
        }
        Some(alignment as u32)
    } else {
        None
    };
    let length =
        usize::try_from(length).map_err(|_| cursor.error(BytecodeErrorKind::ResourceLimit))?;
    let offset = cursor.base + cursor.position;
    let data = cursor.bytes(length)?;
    Ok(BytecodeSection {
        id,
        offset,
        alignment,
        data,
    })
}
