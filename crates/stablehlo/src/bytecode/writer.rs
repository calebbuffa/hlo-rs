use std::collections::HashMap;

use crate::{
    Attribute, Attributes, BlockArgument, Computation, Instruction, Location, Module, Operation,
    Region, Type, ValueId, text,
};

use super::{BytecodeError, BytecodeErrorKind as Kind, encode_varint};

/// Serializes a verified Rust module to MLIR bytecode version 0 through 6.
///
/// Attribute/type payloads use MLIR's specified textual fallback, rather than
/// dialect-specific compressed encodings. Inherent attributes are carried in
/// operation dictionaries, which MLIR migrates into registered properties.
/// This writes StableHLO IR, **not a versioned VHLO portable artifact**.
pub fn write_module(module: &Module, version: u64) -> Result<Vec<u8>, BytecodeError> {
    if version > 6 {
        return Err(error(Kind::UnsupportedVersion(version)));
    }
    module
        .verify()
        .map_err(|err| error(Kind::InvalidIr(Box::new(err))))?;
    if module.attributes().contains_key("stablehlo_rust.entry") {
        return Err(error(Kind::InvalidPayload(
            "module attributes without the reserved stablehlo_rust.entry key",
        )));
    }
    let mut writer = Writer {
        version,
        ..Writer::default()
    };
    let mut attributes = module.attributes().clone();
    if !module.name().is_empty() {
        attributes.insert("sym_name".into(), Attribute::String(module.name().into()));
    }
    if let Some(entry) = module.entry() {
        attributes.insert(
            "stablehlo_rust.entry".into(),
            Attribute::SymbolRef(vec![module.computations()[entry.0 as usize].name().into()]),
        );
    }
    let mut ir = Vec::new();
    encode_varint(2, &mut ir); // One top-level operation, no block arguments.
    writer.header(
        &mut ir,
        Header {
            dialect: "builtin",
            name: "module",
            attributes: &attributes,
            location: module.location(),
            results: &[],
            operands: &[],
            regions: 1,
        },
    )?;
    let mut functions = Vec::new();
    encode_varint(1, &mut functions);
    encode_varint(0, &mut functions);
    encode_varint((module.computations().len() as u64) << 1, &mut functions);
    for function in module.computations() {
        writer.function(function, &mut functions)?;
    }
    writer.regions(&mut ir, 1, true, &functions);
    writer.finish(ir)
}

fn error(kind: Kind) -> BytecodeError {
    BytecodeError { offset: 0, kind }
}

fn assembly<T>(result: Result<T, text::TextError>) -> Result<T, BytecodeError> {
    result.map_err(|err| error(Kind::Assembly(Box::new(err))))
}

#[derive(Default)]
struct Intern {
    values: Vec<String>,
    ids: HashMap<String, u64>,
}

impl Intern {
    fn insert(&mut self, value: String) -> u64 {
        if let Some(&id) = self.ids.get(&value) {
            return id;
        }
        let id = self.values.len() as u64;
        self.ids.insert(value.clone(), id);
        self.values.push(value);
        id
    }
}

#[derive(Default)]
struct Writer {
    version: u64,
    strings: Intern,
    dialects: Intern,
    operations: Vec<(u64, u64)>,
    attributes: Intern,
    types: Intern,
}

struct Header<'a> {
    dialect: &'a str,
    name: &'a str,
    attributes: &'a Attributes,
    location: Option<&'a Location>,
    results: &'a [Type],
    operands: &'a [u64],
    regions: usize,
}

#[derive(Clone, Default)]
struct Scope {
    values: HashMap<ValueId, u64>,
    next: u64,
}

impl Scope {
    fn define(&mut self, value: ValueId) {
        self.values.insert(value, self.next);
        self.next += 1;
    }

    fn value(&self, value: ValueId) -> Result<u64, BytecodeError> {
        self.values.get(&value).copied().ok_or_else(|| {
            error(Kind::InvalidIr(Box::new(crate::BuildError::UnknownValue(
                value,
            ))))
        })
    }
}

impl Writer {
    fn operation(&mut self, dialect: &str, name: &str) -> u64 {
        let dialect = self.dialects.insert(dialect.into());
        let name = self.strings.insert(name.into());
        let pair = (dialect, name);
        if let Some(index) = self.operations.iter().position(|&entry| entry == pair) {
            return index as u64;
        }
        let index = self.operations.len() as u64;
        self.operations.push(pair);
        index
    }

    fn attribute(&mut self, attribute: &Attribute) -> Result<u64, BytecodeError> {
        Ok(self
            .attributes
            .insert(assembly(text::print_attribute(attribute))?))
    }

    fn location(&mut self, location: Option<&Location>) -> Result<u64, BytecodeError> {
        match location {
            Some(location) => self.attribute(&Attribute::Location(Box::new(location.clone()))),
            None => Ok(self.attributes.insert("loc(unknown)".into())),
        }
    }

    fn ty(&mut self, ty: &Type) -> Result<u64, BytecodeError> {
        Ok(self.types.insert(assembly(text::print_type(ty))?))
    }

    fn header(&mut self, output: &mut Vec<u8>, header: Header<'_>) -> Result<(), BytecodeError> {
        let name = self.operation(header.dialect, header.name);
        let mut mask = 0;
        if !header.attributes.is_empty() {
            mask |= 1;
        }
        if !header.results.is_empty() {
            mask |= 2;
        }
        if !header.operands.is_empty() {
            mask |= 4;
        }
        if header.regions != 0 {
            mask |= 16;
        }
        encode_varint(name, output);
        output.push(mask);
        let location = self.location(header.location)?;
        encode_varint(location, output);
        if !header.attributes.is_empty() {
            let attribute = self.attribute(&Attribute::Dictionary(header.attributes.clone()))?;
            encode_varint(attribute, output);
        }
        if !header.results.is_empty() {
            encode_varint(header.results.len() as u64, output);
            for ty in header.results {
                let id = self.ty(ty)?;
                encode_varint(id, output);
            }
        }
        if !header.operands.is_empty() {
            encode_varint(header.operands.len() as u64, output);
            for &operand in header.operands {
                encode_varint(operand, output);
            }
        }
        Ok(())
    }

    fn regions(&self, output: &mut Vec<u8>, count: usize, isolated: bool, body: &[u8]) {
        encode_varint(((count as u64) << 1) | u64::from(isolated), output);
        if isolated && self.version >= 2 {
            emit_section(4, body, output);
        } else {
            output.extend_from_slice(body);
        }
    }

    fn function(
        &mut self,
        function: &Computation,
        output: &mut Vec<u8>,
    ) -> Result<(), BytecodeError> {
        if function
            .instructions()
            .iter()
            .any(|instruction| !instruction.called_computations().is_empty())
        {
            return Err(error(Kind::LegacyComputationReferences));
        }
        let mut metadata = function.attributes().clone();
        metadata.insert("sym_name".into(), Attribute::String(function.name().into()));
        metadata.insert(
            "function_type".into(),
            Attribute::Type(Box::new(Type::Function(function.function_type()))),
        );
        self.header(
            output,
            Header {
                dialect: "func",
                name: "func",
                attributes: &metadata,
                location: function.location(),
                results: &[],
                operands: &[],
                regions: 1,
            },
        )?;
        let mut body = Vec::new();
        if function.is_declaration() {
            encode_varint(0, &mut body);
        } else {
            let mut parameters = function
                .instructions()
                .iter()
                .filter_map(|instruction| match instruction.operation() {
                    Operation::Parameter { number, .. } => Some((*number, instruction)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            parameters.sort_unstable_by_key(|&(number, _)| number);
            let arguments = parameters
                .iter()
                .map(|(_, instruction)| {
                    let result = &instruction.results()[0];
                    let argument = BlockArgument::new(result.value(), result.ty().clone());
                    match instruction.location() {
                        Some(location) => argument.with_location(location.clone()),
                        None => argument,
                    }
                })
                .collect::<Vec<_>>();
            let instructions = function
                .instructions()
                .iter()
                .filter(|instruction| !instruction.operation().is_parameter())
                .collect::<Vec<_>>();
            let mut scope = Scope::default();
            for argument in &arguments {
                scope.define(argument.value());
            }
            for instruction in &instructions {
                for result in instruction.results() {
                    scope.define(result.value());
                }
            }
            encode_varint(1, &mut body);
            encode_varint(scope.next, &mut body);
            self.block_header(&mut body, instructions.len() + 1, &arguments)?;
            for instruction in instructions {
                self.instruction(instruction, &scope, &mut body, 0)?;
            }
            let operands = function
                .roots()
                .iter()
                .map(|&value| scope.value(value))
                .collect::<Result<Vec<_>, _>>()?;
            self.header(
                &mut body,
                Header {
                    dialect: "func",
                    name: "return",
                    attributes: &Attributes::new(),
                    location: function.return_location(),
                    results: &[],
                    operands: &operands,
                    regions: 0,
                },
            )?;
        }
        self.regions(output, 1, true, &body);
        Ok(())
    }

    fn block_header(
        &mut self,
        output: &mut Vec<u8>,
        operations: usize,
        arguments: &[BlockArgument],
    ) -> Result<(), BytecodeError> {
        encode_varint(
            ((operations as u64) << 1) | u64::from(!arguments.is_empty()),
            output,
        );
        if !arguments.is_empty() {
            encode_varint(arguments.len() as u64, output);
            for argument in arguments {
                let ty = self.ty(argument.ty())?;
                let known_location = argument
                    .location()
                    .filter(|location| !matches!(location, Location::Unknown));
                if self.version >= 4 {
                    encode_varint((ty << 1) | u64::from(known_location.is_some()), output);
                    if known_location.is_some() {
                        let location = self.location(known_location)?;
                        encode_varint(location, output);
                    }
                } else {
                    encode_varint(ty, output);
                    let location = self.location(argument.location())?;
                    encode_varint(location, output);
                }
            }
            if self.version >= 3 {
                output.push(0);
            }
        }
        Ok(())
    }

    fn instruction(
        &mut self,
        instruction: &Instruction,
        scope: &Scope,
        output: &mut Vec<u8>,
        depth: usize,
    ) -> Result<(), BytecodeError> {
        if depth >= 128 {
            return Err(error(Kind::NestingLimit));
        }
        if !instruction.called_computations().is_empty() {
            return Err(error(Kind::LegacyComputationReferences));
        }
        let (dialect, name) = match instruction.operation() {
            Operation::StableHlo { opcode, .. } => ("stablehlo", opcode.mnemonic()),
            Operation::Call { .. } => ("func", "call"),
            Operation::Parameter { .. } => {
                return Err(error(Kind::InvalidIr(Box::new(
                    crate::BuildError::UnknownValue(instruction.results()[0].value()),
                ))));
            }
        };
        let operands = instruction
            .operands()
            .iter()
            .map(|&value| scope.value(value))
            .collect::<Result<Vec<_>, _>>()?;
        let results = instruction
            .results()
            .iter()
            .map(|result| result.ty().clone())
            .collect::<Vec<_>>();
        let empty = Attributes::new();
        self.header(
            output,
            Header {
                dialect,
                name,
                attributes: instruction.operation().attributes().unwrap_or(&empty),
                location: instruction.location(),
                results: &results,
                operands: &operands,
                regions: instruction.regions().len(),
            },
        )?;
        if !instruction.regions().is_empty() {
            let mut body = Vec::new();
            for region in instruction.regions() {
                self.region(region, scope, &mut body, depth + 1)?;
            }
            self.regions(output, instruction.regions().len(), false, &body);
        }
        Ok(())
    }

    fn region(
        &mut self,
        region: &Region,
        outer: &Scope,
        output: &mut Vec<u8>,
        depth: usize,
    ) -> Result<(), BytecodeError> {
        encode_varint(region.blocks().len() as u64, output);
        if region.blocks().is_empty() {
            return Ok(());
        }
        let mut scope = outer.clone();
        for block in region.blocks() {
            for argument in block.arguments() {
                scope.define(argument.value());
            }
            for instruction in block.instructions() {
                for result in instruction.results() {
                    scope.define(result.value());
                }
            }
        }
        encode_varint(scope.next - outer.next, output);
        for block in region.blocks() {
            self.block_header(output, block.instructions().len(), block.arguments())?;
            for instruction in block.instructions() {
                self.instruction(instruction, &scope, output, depth)?;
            }
        }
        Ok(())
    }

    fn finish(mut self, ir: Vec<u8>) -> Result<Vec<u8>, BytecodeError> {
        let mut payload = Vec::new();
        let mut offsets = Vec::new();
        encode_varint(self.attributes.values.len() as u64, &mut offsets);
        encode_varint(self.types.values.len() as u64, &mut offsets);
        for table in [&self.attributes, &self.types] {
            for value in &table.values {
                let dialect = if value.starts_with("!stablehlo.") {
                    "stablehlo"
                } else if value.starts_with("!quant.") {
                    "quant"
                } else {
                    "builtin"
                };
                let dialect = self.dialects.insert(dialect.into());
                encode_varint(dialect, &mut offsets);
                encode_varint(1, &mut offsets);
                encode_varint(((value.len() as u64) + 1) << 1, &mut offsets);
                payload.extend_from_slice(value.as_bytes());
                payload.push(0);
            }
        }
        let mut dialects = Vec::new();
        encode_varint(self.dialects.values.len() as u64, &mut dialects);
        for name in &self.dialects.values {
            let id = self.strings.insert(name.clone());
            encode_varint(if self.version == 0 { id } else { id << 1 }, &mut dialects);
        }
        if self.version >= 4 {
            encode_varint(self.operations.len() as u64, &mut dialects);
        }
        for &(dialect, name) in &self.operations {
            encode_varint(dialect, &mut dialects);
            encode_varint(1, &mut dialects);
            encode_varint(
                if self.version >= 5 { name << 1 } else { name },
                &mut dialects,
            );
        }
        let mut strings = Vec::new();
        encode_varint(self.strings.values.len() as u64, &mut strings);
        for value in self.strings.values.iter().rev() {
            encode_varint(value.len() as u64 + 1, &mut strings);
        }
        for value in &self.strings.values {
            strings.extend_from_slice(value.as_bytes());
            strings.push(0);
        }
        let mut output = b"ML\xefR".to_vec();
        encode_varint(self.version, &mut output);
        output.extend_from_slice(b"stablehlo-rust\0");
        for (id, body) in [
            (0, strings),
            (1, dialects),
            (2, payload),
            (3, offsets),
            (4, ir),
        ] {
            emit_section(id, &body, &mut output);
        }
        if self.version >= 5 {
            emit_section(8, &[1], &mut output);
        }
        Ok(output)
    }
}

fn emit_section(id: u8, data: &[u8], output: &mut Vec<u8>) {
    output.push(id);
    encode_varint(data.len() as u64, output);
    output.extend_from_slice(data);
}
