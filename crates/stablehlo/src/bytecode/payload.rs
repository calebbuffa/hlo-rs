use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, HashMap},
};

use crate::{Attribute, ElementType, Location, Type, text};

use super::{
    BytecodeContainer, BytecodeError, BytecodeErrorKind as Kind, EncodedValue, primitives::Cursor,
};

mod affine;
mod builtin;
mod quant;
mod stablehlo;

const MAX_EXPANSION: usize = 64 * 1024 * 1024;

pub(super) struct PayloadDecoder<'a, 'b> {
    container: &'b BytecodeContainer<'a>,
    remaining: Cell<usize>,
    attributes: RefCell<HashMap<u64, Attribute>>,
    types: RefCell<HashMap<u64, Type>>,
    pending: Cell<Option<(bool, u64)>>,
    costs: RefCell<HashMap<(bool, u64), usize>>,
}

impl<'a, 'b> PayloadDecoder<'a, 'b> {
    pub fn new(container: &'b BytecodeContainer<'a>) -> Self {
        Self {
            container,
            remaining: Cell::new(MAX_EXPANSION),
            attributes: RefCell::new(HashMap::new()),
            types: RefCell::new(HashMap::new()),
            pending: Cell::new(None),
            costs: RefCell::new(HashMap::new()),
        }
    }

    fn charge(&self, size: usize, cursor: &Cursor<'_>) -> Result<(), BytecodeError> {
        let remaining = self
            .remaining
            .get()
            .checked_sub(size)
            .ok_or_else(|| cursor.error(Kind::ResourceLimit))?;
        self.remaining.set(remaining);
        Ok(())
    }

    fn entry(
        &self,
        id: u64,
        is_type: bool,
        depth: usize,
    ) -> Result<&EncodedValue<'a>, BytecodeError> {
        let table = if is_type {
            self.container.types()
        } else {
            self.container.attributes()
        };
        let entry = usize::try_from(id)
            .ok()
            .and_then(|id| table.get(id))
            .ok_or(BytecodeError {
                offset: 0,
                kind: Kind::InvalidIndex {
                    table: if is_type { "type" } else { "attribute" },
                    index: id,
                },
            })?;
        if depth >= 128 {
            return Err(BytecodeError {
                offset: entry.offset,
                kind: Kind::NestingLimit,
            });
        }
        self.charge(
            std::mem::size_of::<Attribute>(),
            &Cursor::new(entry.data, entry.offset),
        )?;
        Ok(entry)
    }

    fn fallback<'c>(&self, entry: &'c EncodedValue<'_>) -> Result<&'c str, BytecodeError> {
        let data = entry.data.strip_suffix(&[0]).ok_or(BytecodeError {
            offset: entry.offset,
            kind: Kind::InvalidPayload("null-terminated textual attribute/type payload"),
        })?;
        self.charge(data.len(), &Cursor::new(data, entry.offset))?;
        std::str::from_utf8(data).map_err(|_| BytecodeError {
            offset: entry.offset,
            kind: Kind::InvalidUtf8,
        })
    }

    fn unsupported(&self, entry: &EncodedValue<'_>, kind: &'static str) -> BytecodeError {
        BytecodeError {
            offset: entry.offset,
            kind: Kind::UnsupportedEncoding {
                dialect: String::from_utf8_lossy(self.container.dialects()[entry.dialect].name)
                    .into_owned(),
                kind,
            },
        }
    }

    fn resolve(&self, is_type: bool, id: u64) -> Result<(), BytecodeError> {
        let mut stack = vec![(is_type, id)];
        while let Some(&(is_type, id)) = stack.last() {
            if if is_type {
                self.types.borrow().contains_key(&id)
            } else {
                self.attributes.borrow().contains_key(&id)
            } {
                stack.pop();
                continue;
            }
            let entry = self.entry(id, is_type, stack.len() - 1)?;
            self.pending.set(None);
            let before = self.remaining.get();
            let result = if is_type {
                self.decode_type(id, 0).map(|value| {
                    self.types.borrow_mut().insert(id, value);
                })
            } else {
                self.decode_attribute(id, 0).map(|value| {
                    self.attributes.borrow_mut().insert(id, value);
                })
            };
            if let Some(dependency) = self.pending.take() {
                if stack.contains(&dependency) {
                    return Err(BytecodeError {
                        offset: entry.offset,
                        kind: Kind::InvalidPayload("acyclic attribute/type references"),
                    });
                }
                stack.push(dependency);
            } else {
                result?;
                self.costs.borrow_mut().insert(
                    (is_type, id),
                    before - self.remaining.get() + std::mem::size_of::<Attribute>(),
                );
                stack.pop();
            }
        }
        Ok(())
    }

    pub fn attribute(&self, id: u64, _depth: usize) -> Result<Attribute, BytecodeError> {
        self.resolve(false, id)?;
        self.charge(self.costs.borrow()[&(false, id)], &Cursor::new(&[], 0))?;
        Ok(self.attributes.borrow()[&id].clone())
    }

    pub fn ty(&self, id: u64, _depth: usize) -> Result<Type, BytecodeError> {
        self.resolve(true, id)?;
        self.charge(self.costs.borrow()[&(true, id)], &Cursor::new(&[], 0))?;
        Ok(self.types.borrow()[&id].clone())
    }

    fn decode_attribute(&self, id: u64, depth: usize) -> Result<Attribute, BytecodeError> {
        let entry = self.entry(id, false, depth)?;
        if !entry.custom_encoding {
            return text::parse_attribute(self.fallback(entry)?).map_err(|error| BytecodeError {
                offset: entry.offset,
                kind: Kind::Assembly(Box::new(error)),
            });
        }
        let mut cursor = Cursor::new(entry.data, entry.offset);
        let value = match self.container.dialects()[entry.dialect].name {
            b"builtin" => self.builtin_attribute(&mut cursor, depth + 1)?,
            b"stablehlo" => self.stablehlo_attribute(&mut cursor, depth + 1)?,
            _ => return Err(self.unsupported(entry, "attribute")),
        };
        cursor.finish()?;
        let syntax = text::print_attribute(&value)
            .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))?;
        self.charge(syntax.len(), &cursor)?;
        value
            .verify()
            .map_err(|error| cursor.error(Kind::InvalidAttribute(Box::new(error))))?;
        Ok(value)
    }

    fn decode_type(&self, id: u64, depth: usize) -> Result<Type, BytecodeError> {
        let entry = self.entry(id, true, depth)?;
        if !entry.custom_encoding {
            return text::parse_type(self.fallback(entry)?).map_err(|error| BytecodeError {
                offset: entry.offset,
                kind: Kind::Assembly(Box::new(error)),
            });
        }
        let mut cursor = Cursor::new(entry.data, entry.offset);
        let syntax = match self.container.dialects()[entry.dialect].name {
            b"builtin" => self.builtin_type(&mut cursor, depth + 1)?,
            b"quant" => self.quant_type(&mut cursor, depth + 1)?,
            b"stablehlo" => match cursor.varint()? {
                0 => "!stablehlo.token".into(),
                1 => format!(
                    "!stablehlo.future<{}>",
                    self.types(&mut cursor, depth + 1)?.join(", ")
                ),
                _ => return Err(cursor.error(Kind::InvalidPayload("known StableHLO type code"))),
            },
            _ => return Err(self.unsupported(entry, "type")),
        };
        cursor.finish()?;
        self.charge(syntax.len(), &cursor)?;
        text::parse_type(&syntax).map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))
    }

    fn attr_ref(&self, cursor: &mut Cursor<'_>, depth: usize) -> Result<Attribute, BytecodeError> {
        let id = cursor.varint()?;
        self.entry(id, false, depth)?;
        if let Some(value) = self.attributes.borrow().get(&id) {
            self.charge(self.costs.borrow()[&(false, id)], cursor)?;
            return Ok(value.clone());
        }
        self.pending.set(Some((false, id)));
        Err(cursor.error(Kind::InvalidIndex {
            table: "unresolved attribute dependency",
            index: id,
        }))
    }

    fn type_ref(&self, cursor: &mut Cursor<'_>, depth: usize) -> Result<Type, BytecodeError> {
        let id = cursor.varint()?;
        self.entry(id, true, depth)?;
        if let Some(value) = self.types.borrow().get(&id) {
            self.charge(self.costs.borrow()[&(true, id)], cursor)?;
            return Ok(value.clone());
        }
        self.pending.set(Some((true, id)));
        Err(cursor.error(Kind::InvalidIndex {
            table: "unresolved type dependency",
            index: id,
        }))
    }

    fn type_syntax(&self, cursor: &mut Cursor<'_>, depth: usize) -> Result<String, BytecodeError> {
        text::print_type(&self.type_ref(cursor, depth)?)
            .map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))
    }

    fn types(&self, cursor: &mut Cursor<'_>, depth: usize) -> Result<Vec<String>, BytecodeError> {
        (0..cursor.count()?)
            .map(|_| self.type_syntax(cursor, depth))
            .collect()
    }

    fn attrs(
        &self,
        cursor: &mut Cursor<'_>,
        depth: usize,
    ) -> Result<Vec<Attribute>, BytecodeError> {
        (0..cursor.count()?)
            .map(|_| self.attr_ref(cursor, depth))
            .collect()
    }

    fn string(&self, cursor: &mut Cursor<'_>) -> Result<Vec<u8>, BytecodeError> {
        let id = cursor.varint()?;
        let bytes = usize::try_from(id)
            .ok()
            .and_then(|id| self.container.strings().get(id))
            .ok_or_else(|| {
                cursor.error(Kind::InvalidIndex {
                    table: "string",
                    index: id,
                })
            })?;
        self.charge(bytes.len(), cursor)?;
        Ok(bytes.to_vec())
    }

    fn string_attr(&self, cursor: &mut Cursor<'_>, depth: usize) -> Result<Vec<u8>, BytecodeError> {
        match self.attr_ref(cursor, depth)? {
            Attribute::String(value) => Ok(value.into_bytes()),
            Attribute::Bytes(value) => Ok(value),
            _ => Err(cursor.error(Kind::InvalidPayload("untyped string attribute"))),
        }
    }

    fn location(&self, cursor: &mut Cursor<'_>, depth: usize) -> Result<Location, BytecodeError> {
        match self.attr_ref(cursor, depth)? {
            Attribute::Location(value) => Ok(*value),
            _ => Err(cursor.error(Kind::InvalidPayload("location attribute"))),
        }
    }

    fn signed(cursor: &mut Cursor<'_>) -> Result<i64, BytecodeError> {
        let value = cursor.varint()?;
        Ok(((value >> 1) as i64) ^ -((value & 1) as i64))
    }

    fn signed_list(cursor: &mut Cursor<'_>) -> Result<Vec<i64>, BytecodeError> {
        (0..cursor.count()?).map(|_| Self::signed(cursor)).collect()
    }

    fn bits(cursor: &mut Cursor<'_>, width: u32) -> Result<u64, BytecodeError> {
        let bits = if width <= 8 {
            u64::from(cursor.byte()?)
        } else {
            Self::signed(cursor)? as u64
        };
        Ok(if width == 64 {
            bits
        } else {
            bits & ((1_u64 << width) - 1)
        })
    }

    fn element(ty: Type, cursor: &Cursor<'_>) -> Result<ElementType, BytecodeError> {
        match ty {
            Type::Element(element) => Ok(element),
            _ => Err(cursor.error(Kind::InvalidPayload("scalar element type"))),
        }
    }
}
