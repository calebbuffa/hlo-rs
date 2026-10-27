use super::{TextError, TextErrorKind, cursor::Cursor, module_attributes, module_ops, printer};
use crate::{
    attributes::{
        DISCARDABLE_ATTRIBUTES, DenseElementsAttr, NamedAttributesAttr, ScalarBitsAttr,
        SparseElementsAttr,
    },
    ops::{CallOp, ConstantOp, FuncOp, ModuleOp, ReturnOp},
    pliron::{
        attribute::{AttrObj, AttributeDict},
        builtin::attributes::{BoolAttr, DictAttr, StringAttr, VecAttr},
        context::{Context, Ptr},
        location::Location,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
    types::{FunctionType, QuantizedElementType, TensorType, primitive_element},
};
use std::collections::{BTreeMap, HashMap, HashSet};

pub(super) type AssemblyAttributes = BTreeMap<String, AttrObj>;

// Parser state carries native types/attributes, not an alternate semantic IR.
struct ParsedOperation {
    offset: usize,
    name: String,
    results: Vec<String>,
    operands: Vec<String>,
    inputs: Vec<TypeHandle>,
    outputs: Vec<TypeHandle>,
    attributes: AssemblyAttributes,
    location: Location,
}

struct ParsedFunction {
    name: String,
    arguments: Vec<String>,
    inputs: Vec<TypeHandle>,
    outputs: Vec<TypeHandle>,
    attributes: AssemblyAttributes,
    operations: Vec<ParsedOperation>,
    location: Location,
}

fn error(kind: TextErrorKind) -> TextError {
    TextError { offset: 0, kind }
}

/// Imports MLIR module/function assembly with straight-line generic StableHLO operations.
///
/// The returned module belongs to `ctx` and uses native StableHLO operation
/// wrappers. This parses MLIR syntax, not pliron's internal textual format.
/// Import supports a bounded subset of MLIR assembly and is not an inverse for
/// every module accepted by [`super::export`], particularly nested-region bodies.
///
/// # Errors
///
/// Returns [`TextError`] for unsupported syntax, malformed assembly, unresolved
/// SSA values, or invalid types, attributes, and operations. Syntax and type
/// parsing finish before operation allocation. A construction or verification
/// failure erases the newly allocated module, never caller-owned operations;
/// interned types and attributes can remain in the context.
pub fn parse_module(ctx: &mut Context, input: &str) -> Result<ModuleOp, TextError> {
    let mut cursor = Cursor::new(input);
    let explicit = cursor.keyword("module");
    let attributes = if explicit && cursor.keyword("attributes") {
        cursor.dictionary(ctx, 0)?
    } else {
        AssemblyAttributes::new()
    };
    if explicit {
        cursor.expect("{")?;
    }
    let mut functions = Vec::new();
    let mut names = HashSet::new();
    loop {
        cursor.space();
        if explicit && cursor.take("}") {
            break;
        }
        if !explicit && cursor.offset == input.len() {
            break;
        }
        let function = cursor.function(ctx)?;
        if !names.insert(function.name.clone()) {
            return Err(cursor.error(TextErrorKind::DuplicateField(function.name)));
        }
        functions.push(function);
    }
    let location = if explicit {
        cursor.optional_location(ctx)?
    } else {
        Location::Unknown
    };
    cursor.finish()?;
    let module = ModuleOp::new(ctx);
    let root = module.get_operation();
    let built = (|| {
        root.deref_mut(ctx).set_loc(location);
        copy_discardable(ctx, root, attributes)?;
        let body = module
            .body_block(ctx)
            .ok_or_else(|| error(TextErrorKind::InvalidIr("module has no body".into())))?;
        for function in functions {
            let signature = FunctionType::intern(ctx, function.inputs, function.outputs);
            let native = FuncOp::new(ctx, function.name, signature)
                .map_err(|failure| error(TextErrorKind::InvalidIr(failure.to_string())))?;
            native.get_operation().insert_at_back(body, ctx);
            native
                .get_operation()
                .deref_mut(ctx)
                .set_loc(function.location);
            copy_discardable(ctx, native.get_operation(), function.attributes)?;
            let mut values = function
                .arguments
                .into_iter()
                .zip(native.arguments(ctx))
                .collect::<HashMap<_, _>>();
            let block = native.entry_block(ctx).ok_or_else(|| {
                error(TextErrorKind::InvalidIr(
                    "function has no entry block".into(),
                ))
            })?;
            for operation in function.operations {
                let fail = |kind| TextError {
                    offset: operation.offset,
                    kind,
                };
                let operands = operation
                    .operands
                    .iter()
                    .map(|name| {
                        values.get(name).copied().ok_or_else(|| {
                            fail(TextErrorKind::UnknownValue(
                                name.strip_suffix("#0").unwrap_or(name).to_owned(),
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if operands.len() != operation.inputs.len()
                    || operands
                        .iter()
                        .zip(&operation.inputs)
                        .any(|(value, ty)| value.get_type(ctx) != *ty)
                {
                    return Err(fail(TextErrorKind::InvalidIr(
                        "operand type annotation mismatch".into(),
                    )));
                }
                if operation.results.len() != operation.outputs.len() {
                    return Err(fail(TextErrorKind::InvalidIr(
                        "result name/type count mismatch".into(),
                    )));
                }
                if operation
                    .results
                    .iter()
                    .any(|name| values.contains_key(name))
                {
                    return Err(fail(TextErrorKind::InvalidIr(
                        "duplicate SSA definition".into(),
                    )));
                }
                let ptr = construct(ctx, &operation, operands).map_err(|mut failure| {
                    failure.offset = operation.offset;
                    failure
                })?;
                ptr.deref_mut(ctx).set_loc(operation.location);
                ptr.insert_at_back(block, ctx);
                for (name, value) in operation.results.into_iter().zip(ptr.deref(ctx).results()) {
                    if values.insert(name.clone(), value).is_some() {
                        return Err(fail(TextErrorKind::DuplicateField(name)));
                    }
                }
            }
        }
        crate::verification::verify_module_fast(ctx, root)
            .map_err(|failure| error(TextErrorKind::InvalidIr(failure.to_string())))
    })();
    if let Err(failure) = built {
        Operation::erase(root, ctx);
        return Err(failure);
    }
    Ok(module)
}

fn copy_discardable(
    ctx: &Context,
    operation: Ptr<Operation>,
    dictionary: AssemblyAttributes,
) -> Result<(), TextError> {
    if dictionary.is_empty() {
        return Ok(());
    }
    for name in dictionary.keys() {
        if !name.contains('.') {
            return Err(error(TextErrorKind::Unsupported(
                "unmapped inherent operation attribute",
            )));
        }
    }
    operation.deref_mut(ctx).attributes.0.insert(
        DISCARDABLE_ATTRIBUTES,
        Box::new(NamedAttributesAttr::new(dictionary)),
    );
    Ok(())
}

fn construct(
    ctx: &mut Context,
    parsed: &ParsedOperation,
    operands: Vec<Value>,
) -> Result<Ptr<Operation>, TextError> {
    let fail = |kind| TextError {
        offset: parsed.offset,
        kind,
    };
    let mut attributes = parsed.attributes.clone();
    let operation = match parsed.name.as_str() {
        "stablehlo.constant" => {
            if !operands.is_empty() || parsed.outputs.len() != 1 {
                return Err(fail(TextErrorKind::InvalidIr(
                    "constant arity mismatch".into(),
                )));
            }
            let value = attributes
                .remove("value")
                .ok_or_else(|| fail(TextErrorKind::Expected("constant value")))?;
            let expected = constant_storage_type(ctx, parsed.outputs[0])?;
            let value_type = if let Some(dense) = value.downcast_ref::<DenseElementsAttr>() {
                dense.ty
            } else if let Some(sparse) = value.downcast_ref::<SparseElementsAttr>() {
                sparse.ty
            } else {
                return Err(fail(TextErrorKind::Expected(
                    "dense or sparse constant elements",
                )));
            };
            if value_type != expected {
                return Err(fail(TextErrorKind::InvalidIr(
                    "constant value/result type mismatch".into(),
                )));
            }
            if let Some(sparse) = value.downcast_ref::<SparseElementsAttr>() {
                ConstantOp::new_sparse(
                    ctx,
                    parsed.outputs[0],
                    sparse
                        .indices
                        .iter()
                        .map(|index| index.values.clone())
                        .collect(),
                    sparse.values.0.clone(),
                )
                .get_operation()
            } else {
                let dense = value
                    .downcast_ref::<DenseElementsAttr>()
                    .ok_or_else(|| fail(TextErrorKind::Expected("dense constant elements")))?;
                let mut bits = Vec::new();
                for value in &dense.values.0 {
                    if let Some(value) = value.downcast_ref::<ScalarBitsAttr>() {
                        bits.extend_from_slice(&value.words);
                    } else if let Some(value) = value.downcast_ref::<BoolAttr>() {
                        bits.push(u64::from(bool::from(value.clone())));
                    } else {
                        return Err(fail(TextErrorKind::Unsupported(
                            "constant element representation",
                        )));
                    }
                }
                ConstantOp::new(ctx, parsed.outputs[0], bits, dense.is_splat).get_operation()
            }
        }
        "stablehlo.return" | "func.return" => {
            if !parsed.outputs.is_empty() {
                return Err(fail(TextErrorKind::InvalidIr(
                    "return cannot produce results".into(),
                )));
            }
            ReturnOp::new_many(ctx, operands).get_operation()
        }
        "func.call" => {
            let callee = attributes
                .remove("callee")
                .ok_or_else(|| fail(TextErrorKind::Expected("callee symbol")))?;
            let symbol = callee
                .downcast_ref::<crate::attributes::SymbolRefAttr>()
                .filter(|symbol| symbol.path.len() == 1)
                .ok_or_else(|| fail(TextErrorKind::Expected("flat callee symbol")))?;
            CallOp::new(
                ctx,
                symbol.path[0].as_str().into(),
                operands,
                parsed.outputs.clone(),
            )
            .get_operation()
        }
        name => {
            if let Some(operation) = module_attributes::build(
                ctx,
                name,
                parsed.outputs.clone(),
                operands.clone(),
                &mut attributes,
            )? {
                operation
            } else {
                module_ops::build(ctx, name, parsed.outputs.clone(), operands).ok_or_else(|| {
                    fail(TextErrorKind::Unsupported("operation construction adapter"))
                })?
            }
        }
    };
    if let Err(failure) = copy_discardable(ctx, operation, attributes) {
        Operation::erase(operation, ctx);
        return Err(failure);
    }
    Ok(operation)
}

impl Cursor<'_> {
    fn dictionary(&mut self, ctx: &Context, depth: usize) -> Result<AssemblyAttributes, TextError> {
        let attribute = self.attribute(ctx, depth)?;
        if let Some(dictionary) = attribute.downcast_ref::<DictAttr>() {
            return Ok(dictionary
                .0
                .0
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect());
        }
        if let Some(dictionary) = attribute.downcast_ref::<NamedAttributesAttr>() {
            return Ok(dictionary
                .entries()
                .map(|(name, value)| (name.to_owned(), value.clone()))
                .collect());
        }
        Err(self.error(TextErrorKind::Expected("attribute dictionary")))
    }

    fn ssa(&mut self) -> Result<String, TextError> {
        self.expect("%")?;
        let start = self.offset;
        while self.input.as_bytes().get(self.offset).is_some_and(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$' | b'-')
        }) {
            self.offset += 1;
        }
        let name = &self.input[start..self.offset];
        if name.is_empty()
            || (name.as_bytes()[0].is_ascii_digit()
                && !name.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(self.error(TextErrorKind::Expected("SSA name")));
        }
        Ok(name.into())
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
            .parse()
            .map_err(|_| self.error(TextErrorKind::InvalidNumber))
    }

    fn value_reference(&mut self) -> Result<String, TextError> {
        let name = self.ssa()?;
        let index = if self.take("#") {
            self.result_count()?
        } else {
            0
        };
        Ok(format!("{name}#{index}"))
    }

    fn signature_list(&mut self, ctx: &Context) -> Result<Vec<TypeHandle>, TextError> {
        if !self.take("(") {
            return Ok(vec![self.ty(ctx, 0)?]);
        }
        let mut types = Vec::new();
        if self.take(")") {
            return Ok(types);
        }
        loop {
            types.push(self.ty(ctx, 0)?);
            if self.take(")") {
                return Ok(types);
            }
            self.expect(",")?;
        }
    }

    fn function(&mut self, ctx: &Context) -> Result<ParsedFunction, TextError> {
        if !self.keyword("func.func") {
            return Err(self.error(TextErrorKind::Expected("func.func")));
        }
        self.expect("@")?;
        let name = self.symbol()?;
        self.expect("(")?;
        let mut arguments = Vec::new();
        let mut inputs = Vec::new();
        let mut seen = HashSet::new();
        if !self.take(")") {
            loop {
                let name = self.ssa()?;
                if !seen.insert(name.clone()) {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
                arguments.push(format!("{name}#0"));
                self.expect(":")?;
                inputs.push(self.ty(ctx, 0)?);
                if self.take(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        let outputs = if self.take("->") {
            self.signature_list(ctx)?
        } else {
            Vec::new()
        };
        let attributes = if self.keyword("attributes") {
            self.dictionary(ctx, 0)?
        } else {
            AssemblyAttributes::new()
        };
        self.expect("{")?;
        let mut operations = Vec::new();
        while !self.take("}") {
            operations.push(self.operation(ctx)?);
        }
        Ok(ParsedFunction {
            name,
            arguments,
            inputs,
            outputs,
            attributes,
            operations,
            location: self.optional_location(ctx)?,
        })
    }

    fn operation(&mut self, ctx: &Context) -> Result<ParsedOperation, TextError> {
        self.space();
        let offset = self.offset;
        let mut results = Vec::new();
        if self.starts("%") {
            loop {
                let name = self.ssa()?;
                if self.take(":") {
                    let count = self.result_count()?;
                    if count == 0 || count > 1_048_576 {
                        return Err(
                            self.error(TextErrorKind::Unsupported("excessive SSA result group"))
                        );
                    }
                    results.extend((0..count).map(|index| format!("{name}#{index}")));
                } else {
                    results.push(format!("{name}#0"));
                }
                if !self.take(",") {
                    break;
                }
            }
            self.expect("=")?;
        }
        let generic = self.starts("\"");
        let name = if generic {
            String::from_utf8(self.string_bytes()?)
                .map_err(|_| self.error(TextErrorKind::Expected("operation name")))?
        } else {
            self.word()?.to_owned()
        };
        if !generic && name == "stablehlo.constant" {
            let value = self.attribute(ctx, 0)?;
            let value_type = if let Some(dense) = value.downcast_ref::<DenseElementsAttr>() {
                dense.ty
            } else if let Some(sparse) = value.downcast_ref::<SparseElementsAttr>() {
                sparse.ty
            } else {
                return Err(
                    self.error(TextErrorKind::Expected("dense or sparse constant elements"))
                );
            };
            let outputs = vec![value_type];
            return Ok(ParsedOperation {
                offset,
                name,
                results,
                operands: Vec::new(),
                inputs: Vec::new(),
                outputs,
                attributes: BTreeMap::from([("value".into(), value)]),
                location: self.optional_location(ctx)?,
            });
        }
        let mut attributes = AssemblyAttributes::new();
        if !generic && name == "func.call" {
            self.expect("@")?;
            let symbol = self.symbol()?;
            attributes.insert(
                "callee".into(),
                Box::new(crate::attributes::SymbolRefAttr {
                    path: vec![StringAttr::from(symbol)],
                }),
            );
        }
        let parenthesized = generic || name == "func.call";
        if parenthesized {
            self.expect("(")?;
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
        if parenthesized {
            self.expect(")")?;
        }
        if self.take("<") {
            let properties = self.dictionary(ctx, 0)?;
            self.expect(">")?;
            attributes.extend(properties);
        }
        if self.starts("{") {
            let extra = self.dictionary(ctx, 0)?;
            for (name, value) in extra {
                if attributes.insert(name.clone(), value).is_some() {
                    return Err(self.error(TextErrorKind::DuplicateField(name)));
                }
            }
        }
        let returns = matches!(name.as_str(), "return" | "func.return" | "stablehlo.return");
        let (inputs, outputs) = if generic || !returns {
            self.expect(":")?;
            let ty = self.ty(ctx, 0)?;
            let borrowed = ty.deref(ctx);
            if let Some(signature) = borrowed.downcast_ref::<FunctionType>() {
                (signature.inputs.clone(), signature.outputs.clone())
            } else if generic || name == "func.call" {
                return Err(self.error(TextErrorKind::Expected("operation function type")));
            } else {
                (vec![ty; operands.len()], vec![ty])
            }
        } else {
            let mut types = Vec::new();
            if self.take(":") {
                loop {
                    types.push(self.ty(ctx, 0)?);
                    if !self.take(",") {
                        break;
                    }
                }
            }
            (types, Vec::new())
        };
        let name = if name == "return" {
            "func.return".into()
        } else {
            name
        };
        Ok(ParsedOperation {
            offset,
            name,
            results,
            operands,
            inputs,
            outputs,
            attributes,
            location: self.optional_location(ctx)?,
        })
    }
}

/// Exports a native StableHLO module as verified MLIR assembly.
///
/// Verifies `module` before translating its operations, attributes, types, and
/// nested regions through the framework's MLIR interfaces. The returned string
/// is MLIR text, not pliron's internal format. Export does not mutate the context.
/// The module and every referenced handle must belong to `ctx`.
///
/// # Errors
///
/// Returns [`TextError`] with [`TextErrorKind::InvalidIr`] if verification fails
/// or a required operation, type, attribute, or location translation is missing
/// or invalid. Unsupported entities are never silently omitted or printed using
/// a fallback representation.
pub fn print_module(ctx: &Context, module: &ModuleOp) -> Result<String, TextError> {
    crate::verification::verify_module_fast(ctx, module.get_operation())
        .map_err(|failure| error(TextErrorKind::InvalidIr(failure.to_string())))?;
    String::try_from(pliron_mlir_interop::MlirPrinter::new(
        ctx,
        &module.get_operation(),
    ))
    .map_err(|failure| error(TextErrorKind::InvalidIr(failure.to_string())))
}

// A quantized constant's attribute carries its storage element type.
// element is replaced by its storage type; dimensions and encodings remain.
fn constant_storage_type(ctx: &Context, ty: TypeHandle) -> Result<TypeHandle, TextError> {
    let object = ty.deref(ctx);
    let tensor = object
        .downcast_ref::<TensorType>()
        .ok_or_else(|| error(TextErrorKind::Expected("ranked constant tensor")))?;
    let element = tensor.element.deref(ctx);
    let Some(quantized) = element.downcast_ref::<QuantizedElementType>() else {
        return Ok(ty);
    };
    let storage = TensorType::get(ctx, tensor.dimensions.clone(), quantized.storage);
    Ok(if let Some(encoding) = tensor.encoding() {
        storage.deref(ctx).with_encoding(ctx, encoding.clone())
    } else {
        storage.into()
    })
}

pub(super) fn constant_attribute(
    ctx: &Context,
    constant: &ConstantOp,
    ty: TypeHandle,
) -> Result<AttrObj, TextError> {
    if let Some(sparse) = constant.sparse_value(ctx) {
        return Ok(Box::new((*sparse).clone()));
    }
    let ty = constant_storage_type(ctx, ty)?;
    let object = ty.deref(ctx);
    let tensor = object
        .downcast_ref::<TensorType>()
        .ok_or_else(|| error(TextErrorKind::Expected("constant tensor")))?;
    let width = primitive_element(ctx, tensor.element)
        .and_then(|element| element.bit_width())
        .ok_or_else(|| error(TextErrorKind::Unsupported("nonprimitive constant text")))?;
    let words = width.div_ceil(64) as usize;
    if words == 0 {
        return Err(error(TextErrorKind::Unsupported(
            "zero-width constant text",
        )));
    }
    let bits = constant
        .bits(ctx)
        .ok_or_else(|| error(TextErrorKind::Expected("constant bits")))?;
    let values = bits
        .chunks(words)
        .map(|words| {
            Box::new(ScalarBitsAttr {
                ty: tensor.element,
                words: words.to_vec(),
            }) as AttrObj
        })
        .collect();
    Ok(Box::new(DenseElementsAttr {
        ty,
        values: VecAttr::new(values),
        is_splat: constant
            .is_splat(ctx)
            .ok_or_else(|| error(TextErrorKind::Expected("constant splat flag")))?,
    }))
}

pub(super) fn operation_dictionary_text(
    ctx: &Context,
    attributes: &AttributeDict,
) -> Result<String, TextError> {
    let mut entries = attributes
        .0
        .iter()
        .filter(|(name, _)| **name != DISCARDABLE_ATTRIBUTES)
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect::<AssemblyAttributes>();
    if let Some(discardable) = attributes.0.get(&DISCARDABLE_ATTRIBUTES) {
        let discardable = discardable
            .downcast_ref::<NamedAttributesAttr>()
            .ok_or_else(|| {
                error(TextErrorKind::InvalidIr(
                    "discardable attributes require string-keyed storage".into(),
                ))
            })?;
        discardable
            .validate_names()
            .map_err(|failure| error(TextErrorKind::InvalidAttribute(failure.to_string())))?;
        for (name, value) in discardable.entries() {
            if !name.contains('.') || entries.insert(name.into(), value.clone()).is_some() {
                return Err(error(TextErrorKind::InvalidIr(
                    "invalid discardable attribute name".into(),
                )));
            }
        }
    }
    printer::mlir_attribute(ctx, &NamedAttributesAttr::new(entries))
        .map_err(|failure| error(TextErrorKind::InvalidAttribute(failure.to_string())))
}

/// Prints a supported operation's inherent and discardable attributes using
/// StableHLO source names and MLIR attribute syntax.
///
/// Returns `None` when the operation has no attributes. Unknown inherent
/// attributes and malformed mapped attributes are rejected rather than
/// omitted.
pub fn print_operation_attributes(
    ctx: &Context,
    name: &str,
    attributes: &AttributeDict,
) -> Result<Option<String>, TextError> {
    let mut attributes = attributes.clone();
    module_attributes::source_attributes(ctx, name, &mut attributes)?;
    module_attributes::check_source_fields(name, &attributes)?;
    if attributes.0.is_empty() {
        return Ok(None);
    }
    operation_dictionary_text(ctx, &attributes).map(Some)
}

pub(super) fn symbol(name: &str) -> String {
    if name
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'.'))
    {
        name.into()
    } else {
        printer::quoted(name.as_bytes())
    }
}
