use crate::{Attribute, Module, text};

use super::{
    BytecodeContainer, BytecodeError, BytecodeErrorKind as Kind, SectionId,
    container::read_section,
    payload::PayloadDecoder,
    primitives::{Cursor, reserve},
};

const MAX_VALUES: usize = 1_000_000;
const MAX_ASSEMBLY_BYTES: usize = 64 * 1024 * 1024;

/// Reads MLIR bytecode into the canonical Rust module and verifies it.
///
/// Supports generic operation/region framing in versions 0 through 6 and
/// textual attribute/type payloads, including the forms emitted by
/// [`super::write_module`], and native builtin/StableHLO/uniform Quant payloads.
/// StableHLO, Func and Module native property schemas are supported.
/// Unmodeled builtin/Quant encodings, resources, successors and custom use-list
/// orderings currently return explicit errors.
/// Limits are 128 nested regions, one million values per scope and 64 MiB
/// of expanded assembly. Decoding never invokes a native tool.
pub fn read_module(source: &[u8]) -> Result<Module, BytecodeError> {
    let container = BytecodeContainer::parse(source)?;
    let resources = container.section(SectionId::Resources);
    let offsets = container.section(SectionId::ResourceOffsets);
    if resources.is_some() != offsets.is_some() {
        return Err(BytecodeError {
            offset: 0,
            kind: Kind::InvalidPayload("paired resource and resource-offset sections"),
        });
    }
    if let (Some(resources), Some(offsets)) = (resources, offsets) {
        let mut cursor = Cursor::new(offsets.data, offsets.offset);
        if cursor.varint()? != 0 || cursor.remaining() != 0 || !resources.data.is_empty() {
            return Err(BytecodeError {
                offset: offsets.offset,
                kind: Kind::Unsupported("resource payloads"),
            });
        }
    }
    let ir = container.section(SectionId::Ir).unwrap();
    let mut reader = Reader {
        container: &container,
        payloads: PayloadDecoder::new(&container),
        output: String::new(),
        next_name: 0,
    };
    let mut cursor = Cursor::new(ir.data, ir.offset);
    if cursor.varint()? != 2 {
        return Err(cursor.error(Kind::InvalidPayload(
            "one top-level operation without arguments",
        )));
    }
    reader.operation(&mut cursor, &mut Scope::default(), 0)?;
    cursor.finish()?;
    text::parse_module(&reader.output).map_err(|error| BytecodeError {
        offset: ir.offset,
        kind: Kind::Assembly(Box::new(error)),
    })
}

#[derive(Clone, Default)]
struct Scope {
    types: Vec<Option<Value>>,
    next: usize,
}

#[derive(Clone)]
struct Value {
    ty: String,
    name: String,
}

struct Reader<'a, 'b> {
    container: &'b BytecodeContainer<'a>,
    payloads: PayloadDecoder<'a, 'b>,
    output: String,
    next_name: usize,
}

fn utf8(bytes: &[u8], offset: usize) -> Result<&str, BytecodeError> {
    std::str::from_utf8(bytes).map_err(|_| BytecodeError {
        offset,
        kind: Kind::InvalidUtf8,
    })
}

impl Reader<'_, '_> {
    fn push(&mut self, text: &str, cursor: &Cursor<'_>) -> Result<(), BytecodeError> {
        if text.len() > MAX_ASSEMBLY_BYTES.saturating_sub(self.output.len()) {
            return Err(cursor.error(Kind::ResourceLimit));
        }
        self.output.push_str(text);
        Ok(())
    }

    fn attribute(&self, cursor: &mut Cursor<'_>) -> Result<Attribute, BytecodeError> {
        self.payloads.attribute(cursor.varint()?, 0)
    }

    fn ty_at(&self, id: u64, cursor: &Cursor<'_>) -> Result<String, BytecodeError> {
        let ty = self.payloads.ty(id, 0)?;
        self.assembly(text::print_type(&ty), cursor)
    }

    fn assembly(
        &self,
        value: Result<String, text::TextError>,
        cursor: &Cursor<'_>,
    ) -> Result<String, BytecodeError> {
        value.map_err(|error| cursor.error(Kind::Assembly(Box::new(error))))
    }

    fn location(&self, cursor: &mut Cursor<'_>) -> Result<String, BytecodeError> {
        let location = self.attribute(cursor)?;
        if !matches!(location, Attribute::Location(_)) {
            return Err(cursor.error(Kind::InvalidPayload("location attribute")));
        }
        self.assembly(text::print_attribute(&location), cursor)
    }

    fn fresh_name(&mut self) -> String {
        let name = format!("%v{}", self.next_name);
        self.next_name += 1;
        name
    }

    fn define(scope: &mut Scope, value: Value, cursor: &Cursor<'_>) -> Result<(), BytecodeError> {
        let index = scope.next;
        let slot = scope
            .types
            .get_mut(index)
            .ok_or_else(|| cursor.error(Kind::ValueCountMismatch))?;
        if slot.is_some() {
            return Err(cursor.error(Kind::ValueCountMismatch));
        }
        *slot = Some(value);
        scope.next += 1;
        Ok(())
    }

    fn operation(
        &mut self,
        cursor: &mut Cursor<'_>,
        scope: &mut Scope,
        depth: usize,
    ) -> Result<(), BytecodeError> {
        if depth >= 128 {
            return Err(cursor.error(Kind::NestingLimit));
        }
        let name_id = cursor.varint()?;
        let name = usize::try_from(name_id)
            .ok()
            .and_then(|id| self.container.operation_names().get(id))
            .ok_or_else(|| {
                cursor.error(Kind::InvalidIndex {
                    table: "operation",
                    index: name_id,
                })
            })?;
        let offset = cursor.base + cursor.position;
        let full_name = format!(
            "{}.{}",
            utf8(self.container.dialects()[name.dialect].name, offset)?,
            utf8(name.name, offset)?
        );
        let quoted_name = self.assembly(
            text::print_attribute(&Attribute::String(full_name.clone())),
            cursor,
        )?;
        let mask = cursor.byte()?;
        if mask & 0x80 != 0
            || (self.container.version() < 3 && mask & 32 != 0)
            || (self.container.version() < 5 && mask & 64 != 0)
        {
            return Err(cursor.error(Kind::InvalidOperationMask(mask)));
        }
        let location = self.location(cursor)?;
        let mut attributes = if mask & 1 != 0 {
            match self.attribute(cursor)? {
                Attribute::Dictionary(values) => values,
                _ => {
                    return Err(
                        cursor.error(Kind::InvalidPayload("operation attribute dictionary"))
                    );
                }
            }
        } else {
            crate::Attributes::new()
        };
        if mask & 64 != 0 {
            for (key, value) in super::properties::read(
                self.container,
                &self.payloads,
                &full_name,
                name.registered == Some(true),
                cursor,
            )? {
                if attributes.insert(key, value).is_some() {
                    return Err(cursor.error(Kind::InvalidPayload(
                        "non-overlapping inherent and discardable attributes",
                    )));
                }
            }
        }
        let attributes = self.assembly(
            text::print_attribute(&Attribute::Dictionary(attributes)),
            cursor,
        )?;
        let count = if mask & 2 != 0 { cursor.count()? } else { 0 };
        let mut results = Vec::new();
        reserve(&mut results, count, cursor)?;
        for _ in 0..count {
            let ty = cursor.varint()?;
            results.push(self.ty_at(ty, cursor)?);
        }
        let count = if mask & 4 != 0 { cursor.count()? } else { 0 };
        let mut operands = Vec::new();
        let mut inputs = Vec::new();
        reserve(&mut operands, count, cursor)?;
        reserve(&mut inputs, count, cursor)?;
        for _ in 0..count {
            let id = cursor.varint()?;
            let value = usize::try_from(id)
                .ok()
                .and_then(|index| scope.types.get(index))
                .and_then(Option::as_ref)
                .ok_or_else(|| cursor.error(Kind::InvalidValue(id)))?;
            operands.push(value.name.clone());
            inputs.push(value.ty.clone());
        }
        if mask & 8 != 0 {
            return Err(cursor.error(Kind::Unsupported("successor blocks")));
        }
        if mask & 32 != 0 {
            return Err(cursor.error(Kind::Unsupported("custom use-list orderings")));
        }
        if results.len() > scope.types.len().saturating_sub(scope.next) {
            return Err(cursor.error(Kind::ValueCountMismatch));
        }
        let result_names = (0..results.len())
            .map(|_| self.fresh_name())
            .collect::<Vec<_>>();
        if !result_names.is_empty() {
            self.push(&format!("{} = ", result_names.join(", ")), cursor)?;
        }
        self.push(&format!("{quoted_name}({})", operands.join(", ")), cursor)?;
        if mask & 16 != 0 {
            let encoding = cursor.varint()?;
            let count =
                usize::try_from(encoding >> 1).map_err(|_| cursor.error(Kind::ResourceLimit))?;
            let isolated = encoding & 1 != 0;
            if count == 0 || count > cursor.remaining() {
                return Err(cursor.error(Kind::InvalidPayload("nonempty bounded region list")));
            }
            self.push(" (", cursor)?;
            if isolated && self.container.version() >= 2 {
                let section = read_section(cursor)?;
                if section.id != SectionId::Ir {
                    return Err(cursor.error(Kind::InvalidPayload("nested IR section")));
                }
                let mut nested = Cursor::new(section.data, section.offset);
                self.regions(&mut nested, &Scope::default(), count, depth + 1)?;
                nested.finish()?;
            } else {
                let outer = if isolated {
                    Scope::default()
                } else {
                    scope.clone()
                };
                self.regions(cursor, &outer, count, depth + 1)?;
            }
            self.push(")", cursor)?;
        }
        self.push(
            &format!(
                " {attributes} : ({}) -> ({}) {location}\n",
                inputs.join(", "),
                results.join(", ")
            ),
            cursor,
        )?;
        for (ty, name) in results.into_iter().zip(result_names) {
            Self::define(scope, Value { ty, name }, cursor)?;
        }
        Ok(())
    }

    fn regions(
        &mut self,
        cursor: &mut Cursor<'_>,
        outer: &Scope,
        count: usize,
        depth: usize,
    ) -> Result<(), BytecodeError> {
        for index in 0..count {
            if index > 0 {
                self.push(", ", cursor)?;
            }
            self.push("{\n", cursor)?;
            let blocks = cursor.count()?;
            if blocks > 0 {
                let values = cursor.count()?;
                if values > MAX_VALUES.saturating_sub(outer.types.len()) {
                    return Err(cursor.error(Kind::ResourceLimit));
                }
                let mut scope = outer.clone();
                scope.next = scope.types.len();
                reserve(&mut scope.types, values, cursor)?;
                scope.types.resize(scope.types.len() + values, None);
                for block in 0..blocks {
                    self.block(cursor, &mut scope, block, depth)?;
                }
                if scope.next != scope.types.len() {
                    return Err(cursor.error(Kind::ValueCountMismatch));
                }
            }
            self.push("}", cursor)?;
        }
        Ok(())
    }

    fn block(
        &mut self,
        cursor: &mut Cursor<'_>,
        scope: &mut Scope,
        index: usize,
        depth: usize,
    ) -> Result<(), BytecodeError> {
        let encoding = cursor.varint()?;
        let count =
            usize::try_from(encoding >> 1).map_err(|_| cursor.error(Kind::ResourceLimit))?;
        if count > cursor.remaining() {
            return Err(cursor.error(Kind::ResourceLimit));
        }
        let mut arguments = Vec::new();
        if encoding & 1 != 0 {
            let count = cursor.count()?;
            reserve(&mut arguments, count, cursor)?;
            for _ in 0..count {
                let code = cursor.varint()?;
                let (ty, location) = if self.container.version() >= 4 {
                    let ty = self.ty_at(code >> 1, cursor)?;
                    let location = if code & 1 != 0 {
                        self.location(cursor)?
                    } else {
                        "loc(unknown)".into()
                    };
                    (ty, location)
                } else {
                    (self.ty_at(code, cursor)?, self.location(cursor)?)
                };
                let name = self.fresh_name();
                Self::define(
                    scope,
                    Value {
                        ty: ty.clone(),
                        name: name.clone(),
                    },
                    cursor,
                )?;
                arguments.push(format!("{name}: {ty} {location}"));
            }
            if self.container.version() >= 3 && cursor.byte()? != 0 {
                return Err(cursor.error(Kind::Unsupported("block-argument use-list orderings")));
            }
        }
        self.push(&format!("^bb{index}({}):\n", arguments.join(", ")), cursor)?;
        for _ in 0..count {
            self.operation(cursor, scope, depth)?;
        }
        Ok(())
    }
}
