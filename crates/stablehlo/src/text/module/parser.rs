use std::collections::HashMap;

use super::{ENTRY_ATTRIBUTE, dictionary, error, ir_error};
use crate::text::locations::ParsedLocation;
use crate::text::{TextError, TextErrorKind, cursor::Cursor};
use crate::{
    Attribute, Attributes, Block, BlockArgument, Computation, FunctionType, Instruction, Module,
    Operation, OperationResult, Region, StableHloOp, Type, ValueId,
};

mod custom;

struct ParsedOperation {
    location: Option<ParsedLocation>,
    offset: usize,
    name: String,
    results: Vec<String>,
    operands: Vec<String>,
    regions: Vec<Vec<ParsedBlock>>,
    attributes: Attributes,
    signature: FunctionType,
}

struct ParsedBlock {
    argument_locations: Vec<Option<ParsedLocation>>,
    arguments: Vec<(String, Type)>,
    operations: Vec<ParsedOperation>,
}

impl ParsedOperation {
    fn error(&self, kind: TextErrorKind) -> TextError {
        TextError {
            offset: self.offset,
            kind,
        }
    }

    fn wrapper(&self, name: &str) -> Result<(), TextError> {
        if self.name != name
            || !self.operands.is_empty()
            || !self.results.is_empty()
            || !self.signature.inputs().is_empty()
            || !self.signature.outputs().is_empty()
            || self.regions.len() != 1
        {
            return Err(self.error(TextErrorKind::Expected(
                "zero-operand, zero-result wrapper with one region",
            )));
        }
        Ok(())
    }
}

pub(in crate::text) fn parse(input: &str) -> Result<Module, TextError> {
    let mut cursor = Cursor::new(input);
    let mut operations = Vec::new();
    loop {
        cursor.alias_definitions()?;
        cursor.space();
        if cursor.position() == input.len() {
            break;
        }
        operations.push(cursor.generic_operation(0)?);
    }
    cursor.finish()?;
    let mut operation = if operations.len() == 1 && operations[0].name == "builtin.module" {
        operations.remove(0)
    } else {
        ParsedOperation {
            location: None,
            offset: 0,
            name: "builtin.module".into(),
            results: vec![],
            operands: vec![],
            regions: vec![vec![ParsedBlock {
                argument_locations: vec![],
                arguments: vec![],
                operations,
            }]],
            attributes: Attributes::new(),
            signature: FunctionType::new(vec![], vec![]),
        }
    };
    operation.wrapper("builtin.module")?;
    let name = match operation.attributes.remove("sym_name") {
        None => String::new(),
        Some(Attribute::String(name)) => name,
        _ => return Err(operation.error(TextErrorKind::Expected("string module name"))),
    };
    let entry = match operation.attributes.remove(ENTRY_ATTRIBUTE) {
        None => None,
        Some(Attribute::SymbolRef(path)) if matches!(path.as_slice(), [name] if !name.is_empty()) => {
            path.into_iter().next()
        }
        _ => return Err(operation.error(TextErrorKind::Expected("flat entry function symbol"))),
    };
    let mut module = Module::new(name);
    if let Some(location) = operation.location.take() {
        module
            .set_location(location.resolve(&mut cursor)?)
            .map_err(ir_error)?;
    }
    module
        .set_attributes(std::mem::take(&mut operation.attributes))
        .map_err(ir_error)?;
    if operation.regions[0].len() != 1
        || operation.regions[0]
            .first()
            .is_some_and(|block| !block.arguments.is_empty())
    {
        return Err(operation.error(TextErrorKind::Expected(
            "one module block without arguments",
        )));
    }
    for block in operation.regions.remove(0) {
        for function in block.operations {
            module.add_computation(build_function(function, &mut cursor)?);
        }
    }
    if let Some(entry) = entry {
        let index = module
            .computations()
            .iter()
            .position(|function| function.name() == entry)
            .ok_or_else(|| {
                operation.error(TextErrorKind::InvalidIr(Box::new(
                    crate::BuildError::UnknownComputationName {
                        operation: "builtin.module",
                        name: entry.clone(),
                    },
                )))
            })?;
        module
            .set_entry(u32::try_from(index).map_err(|_| error(TextErrorKind::ResourceLimit))?)
            .map_err(ir_error)?;
    }
    module.verify().map_err(ir_error)?;
    Ok(module)
}

impl Cursor<'_> {
    fn function_symbol(&mut self) -> Result<String, TextError> {
        self.expect("@")?;
        if self.starts("\"") {
            self.string()
        } else {
            self.word().map(str::to_owned)
        }
    }

    fn custom_module(&mut self, depth: usize) -> Result<ParsedOperation, TextError> {
        self.depth(depth)?;
        self.space();
        let offset = self.position();
        self.expect("module")?;
        let mut attributes = Attributes::new();
        if self.starts("@") {
            attributes.insert(
                "sym_name".into(),
                Attribute::String(self.function_symbol()?),
            );
        }
        if self.take("attributes") {
            for (name, value) in dictionary(self.attribute(depth + 1)?)? {
                if attributes.insert(name.clone(), value).is_some() {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
            }
        }
        self.expect("{")?;
        let mut operations = Vec::new();
        while !self.take("}") {
            operations.push(self.generic_operation(depth + 1)?);
        }
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name: "builtin.module".into(),
            results: vec![],
            operands: vec![],
            regions: vec![vec![ParsedBlock {
                argument_locations: vec![],
                arguments: vec![],
                operations,
            }]],
            attributes,
            signature: FunctionType::new(vec![], vec![]),
        })
    }

    fn custom_function(&mut self, depth: usize) -> Result<ParsedOperation, TextError> {
        self.depth(depth)?;
        self.space();
        let offset = self.position();
        self.expect("func.func")?;
        let visibility = if self.starts("@") {
            None
        } else {
            let visibility = self.word()?;
            if !matches!(visibility, "private" | "public" | "nested") {
                return Err(self.error(TextErrorKind::Expected("function visibility")));
            }
            Some(visibility.to_owned())
        };
        let name = self.function_symbol()?;
        self.expect("(")?;
        let mut arguments = Vec::new();
        let mut argument_locations = Vec::new();
        let mut arg_attrs = Vec::new();
        let mut named = None;
        if !self.take(")") {
            loop {
                let has_name = self.starts("%");
                if named.is_some_and(|named| named != has_name) {
                    return Err(self.error(TextErrorKind::Expected(
                        "consistently named or unnamed function arguments",
                    )));
                }
                named = Some(has_name);
                let name = if has_name {
                    let name = self.prefixed_name("%")?;
                    self.expect(":")?;
                    name
                } else {
                    arguments.len().to_string()
                };
                let ty = self.ty(depth + 1)?;
                let attributes = if self.starts("{") {
                    dictionary(self.attribute(depth + 1)?)?
                } else {
                    Attributes::new()
                };
                arguments.push((name, ty));
                argument_locations.push(self.trailing_location(depth + 1)?);
                arg_attrs.push(Attribute::Dictionary(attributes));
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        let mut outputs = Vec::new();
        let mut res_attrs = Vec::new();
        if self.take("->") {
            if self.take("(") {
                if !self.take(")") {
                    loop {
                        outputs.push(self.ty(depth + 1)?);
                        res_attrs.push(Attribute::Dictionary(if self.starts("{") {
                            dictionary(self.attribute(depth + 1)?)?
                        } else {
                            Attributes::new()
                        }));
                        if self.take(")") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
            } else {
                outputs.push(self.ty(depth + 1)?);
            }
        }
        let signature = FunctionType::new(
            arguments
                .iter()
                .map(|(_, ty)| ty.clone())
                .collect::<Vec<_>>(),
            outputs,
        );
        let mut attributes = Attributes::new();
        attributes.insert("sym_name".into(), Attribute::String(name));
        attributes.insert(
            "function_type".into(),
            Attribute::Type(Box::new(Type::Function(signature))),
        );
        if let Some(visibility) = visibility {
            attributes.insert("sym_visibility".into(), Attribute::String(visibility));
        }
        for (name, values) in [("arg_attrs", arg_attrs), ("res_attrs", res_attrs)] {
            if values.iter().any(|value| matches!(value, Attribute::Dictionary(attributes) if !attributes.is_empty())) {
                attributes.insert(name.into(), Attribute::Array(values));
            }
        }
        if self.take("attributes") {
            for (name, value) in dictionary(self.attribute(depth + 1)?)? {
                if matches!(
                    name.as_str(),
                    "sym_name" | "function_type" | "sym_visibility"
                ) {
                    return Err(self.error(TextErrorKind::Expected(
                        "function identity, type and visibility outside the attribute dictionary",
                    )));
                }
                if attributes.insert(name.clone(), value).is_some() {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
            }
        }
        let blocks = if self.starts("{") {
            if named == Some(false) {
                return Err(self.error(TextErrorKind::Expected(
                    "named arguments in function definition",
                )));
            }
            let mut blocks = self.generic_region(depth + 1)?;
            if blocks.is_empty() {
                return Err(self.error(TextErrorKind::Expected("function body with a return")));
            }
            if !blocks[0].arguments.is_empty() {
                return Err(self.error(TextErrorKind::Expected(
                    "entry arguments only in function signature",
                )));
            }
            blocks[0].arguments = arguments;
            blocks[0].argument_locations = argument_locations;
            blocks
        } else {
            // External functions have no block arguments in MLIR. Their parsed
            // source locations are likewise not retained by FunctionOp's parser.
            for location in argument_locations.into_iter().flatten() {
                location.resolve(self)?;
            }
            vec![]
        };
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name: "func.func".into(),
            results: vec![],
            operands: vec![],
            regions: vec![blocks],
            attributes,
            signature: FunctionType::new(vec![], vec![]),
        })
    }

    fn prefixed_name(&mut self, prefix: &'static str) -> Result<String, TextError> {
        self.expect(prefix)?;
        let start = self.offset;
        while self.input.as_bytes().get(self.offset).is_some_and(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'.' | b'-')
        }) {
            self.offset += 1;
        }
        let name = &self.input[start..self.offset];
        if !super::printer::valid_ssa_name(name) {
            return Err(self.error(TextErrorKind::Expected("SSA or block identifier")));
        }
        Ok(name.into())
    }

    fn value_reference(&mut self) -> Result<String, TextError> {
        let name = self.prefixed_name("%")?;
        let number = if self.take("#") {
            self.result_count()?
        } else {
            0
        };
        Ok(format!("{name}#{number}"))
    }

    fn result_count(&mut self) -> Result<usize, TextError> {
        self.space();
        let start = self.offset;
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_digit)
        {
            self.offset += 1;
        }
        self.input[start..self.offset]
            .parse::<usize>()
            .map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    fn generic_operation(&mut self, depth: usize) -> Result<ParsedOperation, TextError> {
        self.depth(depth)?;
        if self.starts("module") {
            return self.custom_module(depth);
        }
        if self.starts("func.func") {
            return self.custom_function(depth);
        }
        self.space();
        let offset = self.position();
        let mut results = Vec::new();
        if self.starts("%") {
            loop {
                let name = self.prefixed_name("%")?;
                let count = if self.take(":") {
                    self.result_count()?
                } else {
                    1
                };
                if count == 0 || count > 1_048_576 {
                    return Err(self.error(TextErrorKind::ResourceLimit));
                }
                results.extend((0..count).map(|index| format!("{name}#{index}")));
                if !self.take(",") {
                    break;
                }
            }
            self.expect("=")?;
        }
        if self.take("func.call") {
            let callee = self.function_symbol()?;
            self.expect("(")?;
            let operands = self.value_list(")")?;
            let mut attributes = if self.starts("{") {
                dictionary(self.attribute(depth + 1)?)?
            } else {
                Attributes::new()
            };
            if attributes
                .insert("callee".into(), Attribute::SymbolRef(vec![callee]))
                .is_some()
            {
                return Err(self.error(TextErrorKind::DuplicateField("callee".into())));
            }
            self.expect(":")?;
            let Type::Function(signature) = self.ty(depth + 1)? else {
                return Err(self.error(TextErrorKind::Expected("call function signature")));
            };
            if operands.len() != signature.inputs().len()
                || results.len() != signature.outputs().len()
            {
                return Err(self.error(TextErrorKind::Expected(
                    "signature matching operand/result counts",
                )));
            }
            return Ok(ParsedOperation {
                location: self.trailing_location(depth)?,
                offset,
                name: "func.call".into(),
                results,
                operands,
                attributes,
                signature,
                regions: vec![],
            });
        }
        let return_name = if self.take("func.return") || self.take("return") {
            Some("func.return")
        } else if self.take("stablehlo.return") {
            Some("stablehlo.return")
        } else {
            None
        };
        if let Some(name) = return_name {
            if !results.is_empty() {
                return Err(self.error(TextErrorKind::Expected("return without results")));
            }
            let mut operands = Vec::new();
            if self.starts("%") {
                loop {
                    operands.push(self.value_reference()?);
                    if !self.take(",") {
                        break;
                    }
                }
            }
            let attributes = if self.starts("{") {
                dictionary(self.attribute(depth + 1)?)?
            } else {
                Attributes::new()
            };
            let mut inputs = Vec::new();
            if !operands.is_empty() {
                self.expect(":")?;
                loop {
                    inputs.push(self.ty(depth + 1)?);
                    if !self.take(",") {
                        break;
                    }
                }
            }
            if operands.len() != inputs.len() {
                return Err(self.error(TextErrorKind::Expected("return operand types")));
            }
            return Ok(ParsedOperation {
                location: self.trailing_location(depth)?,
                offset,
                name: name.into(),
                results,
                operands,
                attributes,
                signature: FunctionType::new(inputs, vec![]),
                regions: vec![],
            });
        }
        if self.starts("stablehlo.") {
            return self.custom_stablehlo_operation(offset, results, depth);
        }
        let name = self.string()?;
        self.generic_operation_body(offset, name, results, depth)
    }

    fn generic_operation_body(
        &mut self,
        offset: usize,
        name: String,
        results: Vec<String>,
        depth: usize,
    ) -> Result<ParsedOperation, TextError> {
        self.expect("(")?;
        let mut operands = Vec::new();
        if !self.take(")") {
            loop {
                operands.push(self.value_reference()?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        let mut attributes = if self.take("<") {
            let attributes = dictionary(self.attribute(depth + 1)?)?;
            self.expect(">")?;
            attributes
        } else {
            Attributes::new()
        };
        let mut regions = Vec::new();
        if self.take("(") {
            loop {
                regions.push(self.generic_region(depth + 1)?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        if self.starts("{") {
            for (key, value) in dictionary(self.attribute(depth + 1)?)? {
                if attributes.insert(key.clone(), value).is_some() {
                    return Err(self.error(TextErrorKind::DuplicateField(key)));
                }
            }
        }
        self.expect(":")?;
        let Type::Function(signature) = self.ty(depth + 1)? else {
            return Err(self.error(TextErrorKind::Expected("operation function type")));
        };
        if signature.inputs().len() != operands.len() || signature.outputs().len() != results.len()
        {
            return Err(self.error(TextErrorKind::Expected(
                "signature matching operand/result counts",
            )));
        }
        Ok(ParsedOperation {
            location: self.trailing_location(depth)?,
            offset,
            name,
            results,
            operands,
            regions,
            attributes,
            signature,
        })
    }

    fn value_list(&mut self, end: &'static str) -> Result<Vec<String>, TextError> {
        let mut values = Vec::new();
        if !self.take(end) {
            loop {
                values.push(self.value_reference()?);
                if self.take(end) {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }

    fn generic_region(&mut self, depth: usize) -> Result<Vec<ParsedBlock>, TextError> {
        self.depth(depth)?;
        self.expect("{")?;
        let mut blocks = Vec::new();
        while !self.take("}") {
            let mut arguments = Vec::new();
            let mut argument_locations = Vec::new();
            if self.starts("^") {
                self.prefixed_name("^")?;
                if self.take("(") && !self.take(")") {
                    loop {
                        let name = self.prefixed_name("%")?;
                        self.expect(":")?;
                        arguments.push((name, self.ty(depth + 1)?));
                        argument_locations.push(self.trailing_location(depth + 1)?);
                        if self.take(")") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                self.expect(":")?;
            } else if !blocks.is_empty() {
                return Err(self.error(TextErrorKind::Expected("block label")));
            }
            let mut operations = Vec::new();
            while !self.starts("}") && !self.starts("^") {
                operations.push(self.generic_operation(depth + 1)?);
            }
            blocks.push(ParsedBlock {
                argument_locations,
                arguments,
                operations,
            });
        }
        Ok(blocks)
    }
}

fn build_function(
    mut operation: ParsedOperation,
    cursor: &mut Cursor<'_>,
) -> Result<Computation, TextError> {
    operation.wrapper("func.func")?;
    let Some(Attribute::String(name)) = operation.attributes.remove("sym_name") else {
        return Err(operation.error(TextErrorKind::Expected("function sym_name")));
    };
    let Some(Attribute::Type(ty)) = operation.attributes.remove("function_type") else {
        return Err(operation.error(TextErrorKind::Expected("function_type attribute")));
    };
    let Type::Function(signature) = *ty else {
        return Err(operation.error(TextErrorKind::Expected("function type")));
    };
    if operation.regions[0].is_empty() {
        let mut function = Computation::try_declare(name, signature).map_err(ir_error)?;
        function
            .set_attributes(operation.attributes)
            .map_err(ir_error)?;
        if let Some(location) = operation.location {
            function
                .set_location(location.resolve(cursor)?)
                .map_err(ir_error)?;
        }
        return Ok(function);
    }
    if operation.regions[0].len() != 1 {
        return Err(operation.error(TextErrorKind::Unsupported("multi-block functions")));
    }
    let mut block = operation.regions.remove(0).remove(0);
    if !block
        .arguments
        .iter()
        .map(|(_, ty)| ty)
        .eq(signature.inputs())
    {
        return Err(operation.error(TextErrorKind::Expected(
            "block arguments matching function inputs",
        )));
    }
    let mut allocator = Allocator {
        top: 0,
        nested: u32::MAX,
    };
    let mut scope = HashMap::new();
    let mut instructions = Vec::new();
    for (number, (name, ty)) in block.arguments.into_iter().enumerate() {
        let value = allocator.allocate(false)?;
        define(&mut scope, format!("{name}#0"), value, ty.clone())?;
        let mut parameter = Instruction::new(
            Operation::parameter(
                u32::try_from(number).map_err(|_| error(TextErrorKind::ResourceLimit))?,
                name,
            ),
            vec![],
            vec![],
            value,
            ty,
        );
        if let Some(location) = block
            .argument_locations
            .get_mut(number)
            .and_then(Option::take)
        {
            parameter = parameter.with_location(location.resolve(cursor)?);
        }
        instructions.push(parameter);
    }
    let Some(terminator) = block.operations.pop() else {
        return Err(operation.error(TextErrorKind::Expected("func.return terminator")));
    };
    if terminator.name != "func.return"
        || !terminator.results.is_empty()
        || !terminator.regions.is_empty()
        || !terminator.attributes.is_empty()
    {
        return Err(terminator.error(TextErrorKind::Expected("unadorned func.return terminator")));
    }
    for instruction in block.operations {
        instructions.push(build_instruction(
            instruction,
            &mut scope,
            &mut allocator,
            false,
            cursor,
        )?);
    }
    let roots = resolve_operands(&terminator, &scope)?;
    if terminator.signature.inputs() != signature.outputs() {
        return Err(terminator.error(TextErrorKind::Expected(
            "return operands matching function results",
        )));
    }
    let mut function =
        Computation::try_new_with_roots(name, instructions, roots).map_err(ir_error)?;
    if function.function_type() != signature {
        return Err(operation.error(TextErrorKind::Expected(
            "body matching declared function signature",
        )));
    }
    function
        .set_attributes(operation.attributes)
        .map_err(ir_error)?;
    if let Some(location) = operation.location {
        function
            .set_location(location.resolve(cursor)?)
            .map_err(ir_error)?;
    }
    if let Some(location) = terminator.location {
        function
            .set_return_location(location.resolve(cursor)?)
            .map_err(ir_error)?;
    }
    Ok(function)
}

type Scope = HashMap<String, (ValueId, Type)>;

struct Allocator {
    top: u32,
    nested: u32,
}

impl Allocator {
    fn allocate(&mut self, nested: bool) -> Result<ValueId, TextError> {
        if self.top >= self.nested {
            return Err(error(TextErrorKind::ResourceLimit));
        }
        if nested {
            let value = self.nested;
            self.nested -= 1;
            Ok(ValueId(value))
        } else {
            let value = self.top;
            self.top += 1;
            Ok(ValueId(value))
        }
    }
}

fn define(scope: &mut Scope, name: String, value: ValueId, ty: Type) -> Result<(), TextError> {
    if scope.insert(name.clone(), (value, ty)).is_some() {
        return Err(error(TextErrorKind::DuplicateField(format!("%{name}"))));
    }
    Ok(())
}

fn resolve_operands(operation: &ParsedOperation, scope: &Scope) -> Result<Vec<ValueId>, TextError> {
    operation
        .operands
        .iter()
        .zip(operation.signature.inputs())
        .map(|(name, ty)| {
            let (value, actual) = scope
                .get(name)
                .ok_or_else(|| operation.error(TextErrorKind::UnknownValue(name.clone())))?;
            if actual != ty {
                return Err(operation.error(TextErrorKind::Expected(
                    "operand signature matching SSA value type",
                )));
            }
            Ok(*value)
        })
        .collect()
}

fn build_instruction(
    mut parsed: ParsedOperation,
    scope: &mut Scope,
    allocator: &mut Allocator,
    nested: bool,
    cursor: &mut Cursor<'_>,
) -> Result<Instruction, TextError> {
    let operands = resolve_operands(&parsed, scope)?;
    let operation = if parsed.name == "func.call" {
        Operation::Call {
            attributes: std::mem::take(&mut parsed.attributes),
        }
    } else if let Some(name) = parsed.name.strip_prefix("stablehlo.") {
        let opcode = StableHloOp::ALL
            .iter()
            .find(|opcode| opcode.mnemonic() == name)
            .ok_or_else(|| {
                parsed.error(TextErrorKind::Unsupported("unknown StableHLO operation"))
            })?;
        Operation::stablehlo(*opcode, std::mem::take(&mut parsed.attributes))
    } else {
        return Err(parsed.error(TextErrorKind::Unsupported("this operation")));
    };
    let mut results = Vec::new();
    for ty in parsed.signature.outputs() {
        results.push(OperationResult::new(
            allocator.allocate(nested)?,
            ty.clone(),
        ));
    }
    let mut regions = Vec::new();
    for region in parsed.regions {
        let mut blocks = Vec::new();
        for mut block in region {
            let mut child = scope.clone();
            let mut arguments = Vec::new();
            for (index, (name, ty)) in block.arguments.into_iter().enumerate() {
                let value = allocator.allocate(true)?;
                define(&mut child, format!("{name}#0"), value, ty.clone())?;
                let mut argument = BlockArgument::new(value, ty);
                if let Some(location) = block
                    .argument_locations
                    .get_mut(index)
                    .and_then(Option::take)
                {
                    argument = argument.with_location(location.resolve(cursor)?);
                }
                arguments.push(argument);
            }
            let instructions = block
                .operations
                .into_iter()
                .map(|operation| build_instruction(operation, &mut child, allocator, true, cursor))
                .collect::<Result<Vec<_>, _>>()?;
            blocks.push(Block::new(arguments, instructions));
        }
        regions.push(Region::new(blocks));
    }
    for (name, result) in parsed.results.into_iter().zip(&results) {
        define(scope, name, result.value(), result.ty().clone())?;
    }
    let mut instruction =
        Instruction::with_results(operation, operands, vec![], results).with_regions(regions);
    if let Some(location) = parsed.location {
        instruction = instruction.with_location(location.resolve(cursor)?);
    }
    Ok(instruction)
}
