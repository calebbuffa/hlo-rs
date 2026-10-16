use std::collections::{BTreeMap, HashSet};

use crate::{
    Attribute, ChannelHandle, DotAlgorithm, DotDimensionNumbers, GatherDimensionNumbers,
    OutputOperandAlias, ScatterDimensionNumbers, StableHloAttribute, TypeExtensions,
};

use super::{TextError, TextErrorKind, cursor::Cursor};

impl Cursor<'_> {
    pub(super) fn attribute(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.depth(depth)?;
        enum Frame {
            Array(Vec<Attribute>),
            Dictionary(BTreeMap<String, Attribute>, String),
        }

        let mut frames = Vec::new();
        let mut value: Option<Attribute> = None;
        loop {
            if let Some(attribute) = value.take() {
                let Some(frame) = frames.last_mut() else {
                    attribute
                        .verify()
                        .map_err(|error| self.error(TextErrorKind::InvalidAttribute(error)))?;
                    return Ok(attribute);
                };
                match frame {
                    Frame::Array(attributes) => attributes.push(attribute),
                    Frame::Dictionary(attributes, name) => {
                        if attributes.insert(name.clone(), attribute).is_some() {
                            return Err(self.error(TextErrorKind::DuplicateField(name.clone())));
                        }
                    }
                }
                match frame {
                    Frame::Array(attributes) => {
                        if self.take("]") {
                            value = Some(Attribute::Array(std::mem::take(attributes)));
                            frames.pop();
                        } else {
                            self.expect(",")?;
                        }
                    }
                    Frame::Dictionary(attributes, _) => {
                        if self.take("}") {
                            value = Some(Attribute::Dictionary(std::mem::take(attributes)));
                            frames.pop();
                        } else {
                            self.expect(",")?;
                            let name = if self.starts("\"") {
                                self.string()?
                            } else {
                                self.word()?.to_owned()
                            };
                            if name.is_empty() {
                                return Err(
                                    self.error(TextErrorKind::Expected("nonempty attribute name"))
                                );
                            }
                            if self.take("=") {
                                if let Frame::Dictionary(_, pending) = frame {
                                    *pending = name;
                                }
                                continue;
                            }
                            value = Some(Attribute::Unit);
                            if let Frame::Dictionary(_, pending) = frame {
                                *pending = name;
                            }
                        }
                    }
                }
                continue;
            }

            if self.take("[") {
                if self.take("]") {
                    value = Some(Attribute::Array(Vec::new()));
                } else {
                    frames.push(Frame::Array(Vec::new()));
                    self.depth(depth + frames.len())?;
                    continue;
                }
            } else if self.take("{") {
                if self.take("}") {
                    value = Some(Attribute::Dictionary(BTreeMap::new()));
                } else {
                    let name = if self.starts("\"") {
                        self.string()?
                    } else {
                        self.word()?.to_owned()
                    };
                    if name.is_empty() {
                        return Err(self.error(TextErrorKind::Expected("nonempty attribute name")));
                    }
                    frames.push(Frame::Dictionary(BTreeMap::new(), name.clone()));
                    self.depth(depth + frames.len())?;
                    if !self.take("=") {
                        value = Some(Attribute::Unit);
                    }
                    continue;
                }
            } else {
                value = Some(self.attribute_value_leaf(depth + frames.len())?);
            }
        }
    }

    fn attribute_value_leaf(&mut self, depth: usize) -> Result<Attribute, TextError> {
        if self.take("#complex.number") {
            return self.complex_attribute(depth);
        }
        if self.take("#complex<") {
            self.expect("number")?;
            let value = self.complex_attribute(depth)?;
            self.expect(">")?;
            return Ok(value);
        }
        if self.take("#sparse_tensor.encoding") {
            return self.sparse_encoding(depth);
        }
        if self.take("#sparse_tensor<") {
            if self.take("encoding") {
                let encoding = self.sparse_encoding(depth)?;
                self.expect(">")?;
                return Ok(encoding);
            }
            self.expect("slice")?;
            let slice = self.sparse_slice()?;
            self.expect(">")?;
            return Ok(Attribute::SparseTensorSlice(slice));
        }
        if self.take("affine_map") {
            return self.affine_attribute(false, depth);
        }
        if self.take("affine_set") {
            return self.affine_attribute(true, depth);
        }
        if self.take("strided") {
            self.expect("<")?;
            self.expect("[")?;
            let mut strides = Vec::new();
            if !self.take("]") {
                loop {
                    strides.push(if self.take("?") {
                        None
                    } else {
                        Some(self.i64()?)
                    });
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            let offset = if self.take(",") {
                self.expect("offset")?;
                self.expect(":")?;
                if self.take("?") {
                    None
                } else {
                    Some(self.i64()?)
                }
            } else {
                Some(0)
            };
            self.expect(">")?;
            return Ok(Attribute::StridedLayout { offset, strides });
        }
        if self.starts("loc") {
            return Ok(Attribute::Location(Box::new(self.location(depth)?)));
        }
        if self.take("dense") {
            return self.dense_attribute(depth);
        }
        if self.take("sparse") {
            return self.sparse_attribute(depth);
        }
        if self.take("array") {
            return self.dense_array_attribute(depth);
        }
        let generic_dialect_form = self.take("#stablehlo<");
        if generic_dialect_form || self.take("#stablehlo.") {
            let name = self.word()?;
            if !generic_dialect_form && self.input.as_bytes().get(self.offset) != Some(&b'<') {
                return Err(self.error(TextErrorKind::Expected(
                    "immediately adjacent '<' in a dotted dialect attribute",
                )));
            }
            let attribute = if let Some(values) = crate::attribute::enums::values(name) {
                let delimited = name == "result_accuracy_mode";
                if !delimited && !generic_dialect_form {
                    return Err(self.error(TextErrorKind::Expected(
                        "generic #stablehlo<enum VALUE> assembly",
                    )));
                }
                if delimited {
                    self.expect("<")?;
                }
                let value = self.word()?.to_owned();
                if !values.contains(&value.as_str()) {
                    return Err(self.error(TextErrorKind::Expected(
                        "valid value for this StableHLO enum",
                    )));
                }
                if delimited {
                    self.expect(">")?;
                }
                Attribute::Enum {
                    dialect: "stablehlo".into(),
                    name: name.into(),
                    value,
                }
            } else {
                Attribute::StableHlo(Box::new(self.dialect_attribute(name, depth)?))
            };
            if generic_dialect_form {
                self.expect(">")?;
            }
            return Ok(attribute);
        }
        if self.starts("\"") {
            let value = self.string_bytes()?;
            if self.take(":") {
                return Ok(Attribute::TypedString {
                    value,
                    ty: Box::new(self.ty(depth + 1)?),
                });
            }
            return Ok(match String::from_utf8(value) {
                Ok(value) => Attribute::String(value),
                Err(error) => Attribute::Bytes(error.into_bytes()),
            });
        }
        if self.take("@") {
            let mut path = vec![self.symbol_name()?];
            while self.take("::") {
                self.expect("@")?;
                path.push(self.symbol_name()?);
            }
            return Ok(Attribute::SymbolRef(path));
        }
        for (word, value) in [
            ("unit", Attribute::Unit),
            ("true", Attribute::Bool(true)),
            ("false", Attribute::Bool(false)),
        ] {
            if self.take(word) {
                return Ok(value);
            }
        }
        if self.starts("-")
            || self.starts("+")
            || self
                .input
                .as_bytes()
                .get(self.offset)
                .is_some_and(u8::is_ascii_digit)
        {
            return self.scalar_attribute(depth);
        }
        if self.starts("#") {
            return self.attribute_alias(depth);
        }
        Ok(Attribute::Type(Box::new(self.ty(depth + 1)?)))
    }

    fn symbol_name(&mut self) -> Result<String, TextError> {
        if self.starts("\"") {
            self.string()
        } else {
            self.word().map(str::to_owned)
        }
    }

    pub(super) fn dialect_attribute(
        &mut self,
        name: &str,
        depth: usize,
    ) -> Result<StableHloAttribute, TextError> {
        match name {
            "bounds" | "type_extensions" => {
                self.expect("<")?;
                let generated = name == "type_extensions";
                if generated {
                    self.expect("bounds")?;
                    self.expect("=")?;
                    self.expect("[")?;
                }
                let end = if generated { "]" } else { ">" };
                let mut bounds = Vec::new();
                if !self.take(end) {
                    loop {
                        bounds.push(if self.take("?") { -1 } else { self.i64()? });
                        if self.take(end) {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                if generated {
                    self.expect(">")?;
                }
                Ok(StableHloAttribute::TypeExtensions(TypeExtensions {
                    bounds,
                }))
            }
            "dot" => {
                let mut dimensions = DotDimensionNumbers::default();
                self.fields(|cursor, key| {
                    let target = match key {
                        "lhs_batching_dimensions" => &mut dimensions.lhs_batching_dimensions,
                        "rhs_batching_dimensions" => &mut dimensions.rhs_batching_dimensions,
                        "lhs_contracting_dimensions" => &mut dimensions.lhs_contracting_dimensions,
                        "rhs_contracting_dimensions" => &mut dimensions.rhs_contracting_dimensions,
                        _ => return Err(cursor.error(TextErrorKind::Expected("dot dimension field"))),
                    };
                    *target = cursor.int_list()?;
                    Ok(())
                })?;
                Ok(StableHloAttribute::DotDimensionNumbers(dimensions))
            }
            "gather" => {
                let mut dimensions = GatherDimensionNumbers::default();
                self.fields(|cursor, key| {
                    if key == "index_vector_dim" {
                        dimensions.index_vector_dim = cursor.i64()?;
                        return Ok(());
                    }
                    let target = match key {
                        "offset_dims" => &mut dimensions.offset_dims,
                        "collapsed_slice_dims" => &mut dimensions.collapsed_slice_dims,
                        "operand_batching_dims" => &mut dimensions.operand_batching_dims,
                        "start_indices_batching_dims" => {
                            &mut dimensions.start_indices_batching_dims
                        }
                        "start_index_map" => &mut dimensions.start_index_map,
                        _ => {
                            return Err(
                                cursor.error(TextErrorKind::Expected("gather dimension field"))
                            );
                        }
                    };
                    *target = cursor.int_list()?;
                    Ok(())
                })?;
                Ok(StableHloAttribute::GatherDimensionNumbers(dimensions))
            }
            "scatter" => {
                let mut dimensions = ScatterDimensionNumbers::default();
                self.fields(|cursor, key| {
                    if key == "index_vector_dim" {
                        dimensions.index_vector_dim = cursor.i64()?;
                        return Ok(());
                    }
                    let target = match key {
                        "update_window_dims" => &mut dimensions.update_window_dims,
                        "inserted_window_dims" => &mut dimensions.inserted_window_dims,
                        "input_batching_dims" => &mut dimensions.input_batching_dims,
                        "scatter_indices_batching_dims" => {
                            &mut dimensions.scatter_indices_batching_dims
                        }
                        "scatter_dims_to_operand_dims" => {
                            &mut dimensions.scatter_dims_to_operand_dims
                        }
                        _ => {
                            return Err(
                                cursor.error(TextErrorKind::Expected("scatter dimension field"))
                            );
                        }
                    };
                    *target = cursor.int_list()?;
                    Ok(())
                })?;
                Ok(StableHloAttribute::ScatterDimensionNumbers(dimensions))
            }
            "channel_handle" => {
                let mut channel = ChannelHandle {
                    handle: 0,
                    channel_type: 0,
                };
                let seen = self.fields(|cursor, key| {
                    match key {
                        "handle" => channel.handle = cursor.i64()?,
                        "type" => channel.channel_type = cursor.i64()?,
                        _ => return Err(cursor.error(TextErrorKind::Expected("handle or type"))),
                    }
                    Ok(())
                })?;
                self.required(&seen, &["handle", "type"])?;
                Ok(StableHloAttribute::ChannelHandle(channel))
            }
            "output_operand_alias" => {
                let mut alias = OutputOperandAlias::default();
                // This ODS assembly format is positional, unlike struct(params).
                self.expect("<")?;
                self.expect("output_tuple_indices")?;
                self.expect("=")?;
                alias.output_tuple_indices = self.int_list()?;
                self.expect(",")?;
                self.expect("operand_index")?;
                self.expect("=")?;
                alias.operand_index = self.i64()?;
                self.expect(",")?;
                self.expect("operand_tuple_indices")?;
                self.expect("=")?;
                alias.operand_tuple_indices = self.int_list()?;
                self.expect(">")?;
                Ok(StableHloAttribute::OutputOperandAlias(alias))
            }
            "dot_algorithm" => {
                self.expect("<")?;
                self.expect("lhs_precision_type")?;
                self.expect("=")?;
                let lhs_precision_type = self.ty(depth + 1)?;
                self.expect(",")?;
                self.expect("rhs_precision_type")?;
                self.expect("=")?;
                let rhs_precision_type = self.ty(depth + 1)?;
                self.expect(",")?;
                self.expect("accumulation_type")?;
                self.expect("=")?;
                let accumulation_type = self.ty(depth + 1)?;
                self.expect(",")?;
                self.expect("lhs_component_count")?;
                self.expect("=")?;
                let lhs_component_count = self.i64()?;
                self.expect(",")?;
                self.expect("rhs_component_count")?;
                self.expect("=")?;
                let rhs_component_count = self.i64()?;
                self.expect(",")?;
                self.expect("num_primitive_operations")?;
                self.expect("=")?;
                let num_primitive_operations = self.i64()?;
                self.expect(",")?;
                self.expect("allow_imprecise_accumulation")?;
                self.expect("=")?;
                let allow_imprecise_accumulation = self.boolean()?;
                self.expect(">")?;
                Ok(StableHloAttribute::DotAlgorithm(Box::new(DotAlgorithm {
                    lhs_precision_type,
                    rhs_precision_type,
                    accumulation_type,
                    lhs_component_count,
                    rhs_component_count,
                    num_primitive_operations,
                    allow_imprecise_accumulation,
                })))
            }
            "conv" => self.convolution_attribute(),
            "result_accuracy" => self.accuracy_attribute(depth),
            "sub_axis_info" | "axis_ref" | "mesh_axis" | "mesh" | "replica_group_mesh_axes" => {
                self.mesh_attribute(name, depth)
            }
            _ => Err(self.error(TextErrorKind::Unsupported("this StableHLO attribute"))),
        }
    }

    fn boolean(&mut self) -> Result<bool, TextError> {
        if self.take("true") {
            Ok(true)
        } else if self.take("false") {
            Ok(false)
        } else {
            Err(self.error(TextErrorKind::Expected("true or false")))
        }
    }

    pub(super) fn int_list(&mut self) -> Result<Vec<i64>, TextError> {
        self.expect("[")?;
        let mut values = Vec::new();
        if !self.take("]") {
            loop {
                values.push(self.i64()?);
                if self.take("]") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(values)
    }

    pub(super) fn fields(
        &mut self,
        parse: impl FnMut(&mut Self, &str) -> Result<(), TextError>,
    ) -> Result<HashSet<String>, TextError> {
        self.expect("<")?;
        self.fields_body(parse)
    }

    pub(super) fn fields_body(
        &mut self,
        mut parse: impl FnMut(&mut Self, &str) -> Result<(), TextError>,
    ) -> Result<HashSet<String>, TextError> {
        let mut seen = HashSet::new();
        if !self.take(">") {
            loop {
                let key = self.word()?.to_owned();
                if !seen.insert(key.clone()) {
                    return Err(self.error(TextErrorKind::DuplicateField(key)));
                }
                self.expect("=")?;
                parse(self, &key)?;
                if self.take(">") {
                    break;
                }
                self.expect(",")?;
                if self.take(">") {
                    break;
                }
            }
        }
        Ok(seen)
    }

    pub(super) fn required(
        &self,
        seen: &HashSet<String>,
        fields: &[&'static str],
    ) -> Result<(), TextError> {
        for field in fields {
            if !seen.contains(*field) {
                return Err(self.error(TextErrorKind::Expected(field)));
            }
        }
        Ok(())
    }

    pub(super) fn string(&mut self) -> Result<String, TextError> {
        let bytes = self.string_bytes()?;
        String::from_utf8(bytes)
            .map_err(|_| self.error(TextErrorKind::Unsupported("non-UTF-8 identifier")))
    }

    pub(super) fn string_bytes(&mut self) -> Result<Vec<u8>, TextError> {
        self.expect("\"")?;
        let mut bytes = Vec::new();
        loop {
            match self.input.as_bytes().get(self.offset).copied() {
                Some(b'"') => {
                    self.offset += 1;
                    break;
                }
                Some(b'\\') => {
                    self.offset += 1;
                    let escaped = match self.input.as_bytes().get(self.offset).copied() {
                        Some(b'n') => b'\n',
                        Some(b't') => b'\t',
                        Some(b'\\') => b'\\',
                        Some(b'"') => b'"',
                        Some(byte) if byte.is_ascii_hexdigit() => {
                            let second = self
                                .input
                                .as_bytes()
                                .get(self.offset + 1)
                                .copied()
                                .filter(u8::is_ascii_hexdigit)
                                .ok_or_else(|| {
                                    self.error(TextErrorKind::Expected("two hex escape digits"))
                                })?;
                            self.offset += 1;
                            hex(byte) * 16 + hex(second)
                        }
                        _ => return Err(self.error(TextErrorKind::Expected("MLIR string escape"))),
                    };
                    self.offset += 1;
                    bytes.push(escaped);
                }
                Some(b'\n' | b'\x0b' | b'\x0c') => {
                    return Err(self.error(TextErrorKind::Expected(
                        "printable string character or escape",
                    )));
                }
                Some(_) => {
                    let character = self.input[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.error(TextErrorKind::Expected("string character")))?;
                    let end = self.offset + character.len_utf8();
                    bytes.extend_from_slice(&self.input.as_bytes()[self.offset..end]);
                    self.offset = end;
                }
                None => return Err(self.error(TextErrorKind::Expected("closing quote"))),
            }
        }
        Ok(bytes)
    }
}

fn hex(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => byte - b'A' + 10,
    }
}

pub(super) fn print(attribute: &Attribute, depth: usize) -> Result<String, TextError> {
    enum Event<'a> {
        Attribute(&'a Attribute, usize),
        Text(String),
    }

    let mut output = String::new();
    let mut events = vec![Event::Attribute(attribute, depth)];
    while let Some(event) = events.pop() {
        match event {
            Event::Text(text) => output.push_str(&text),
            Event::Attribute(attribute, depth) => {
                if depth >= 128 {
                    return Err(error(TextErrorKind::NestingLimit));
                }
                match attribute {
                    Attribute::Array(attributes) => {
                        output.push('[');
                        events.push(Event::Text("]".into()));
                        for (index, child) in attributes.iter().enumerate().rev() {
                            events.push(Event::Attribute(child, depth + 1));
                            if index > 0 {
                                events.push(Event::Text(", ".into()));
                            }
                        }
                    }
                    Attribute::Dictionary(attributes) => {
                        output.push('{');
                        events.push(Event::Text("}".into()));
                        for (index, (name, child)) in attributes.iter().enumerate().rev() {
                            events.push(Event::Attribute(child, depth + 1));
                            events.push(Event::Text(" = ".into()));
                            events.push(Event::Text(quote(name)));
                            if index > 0 {
                                events.push(Event::Text(", ".into()));
                            }
                        }
                    }
                    _ => output.push_str(&print_leaf(attribute, depth)?),
                }
            }
        }
    }
    Ok(output)
}

fn print_leaf(attribute: &Attribute, depth: usize) -> Result<String, TextError> {
    if depth >= 128 {
        return Err(error(TextErrorKind::NestingLimit));
    }
    let result = match attribute {
        Attribute::Location(location) => super::locations::print(location, depth)?,
        Attribute::Unit => "unit".into(),
        Attribute::Bool(value) => value.to_string(),
        Attribute::Integer { ty, value } => format!(
            "{value} : {}",
            super::types::print(&crate::Type::Element(*ty), depth + 1)?
        ),
        Attribute::Float { ty, bits } => {
            let literal = if *ty == crate::ElementType::F64 {
                super::floats::print_f64(*bits)
            } else {
                format!("0x{bits:X}")
            };
            format!(
                "{literal} : {}",
                super::types::print(&crate::Type::Element(*ty), depth + 1)?
            )
        }
        Attribute::DenseArray { element, values } => super::arrays::print(*element, values, depth)?,
        Attribute::String(value) => quote(value),
        Attribute::Bytes(value) => quote_bytes(value),
        Attribute::TypedString { value, ty } => {
            format!(
                "{} : {}",
                quote_bytes(value),
                super::types::print(ty, depth + 1)?
            )
        }
        Attribute::SymbolRef(path) if !path.is_empty() => path
            .iter()
            .map(|part| format!("@{}", quote(part)))
            .collect::<Vec<_>>()
            .join("::"),
        Attribute::Array(_) | Attribute::Dictionary(_) => {
            return Err(error(TextErrorKind::Unsupported(
                "attribute container in leaf printer",
            )));
        }
        Attribute::AffineMap(map) => {
            map.verify().map_err(|e| {
                error(TextErrorKind::InvalidAttribute(
                    crate::AttributeValidationError::Affine(e),
                ))
            })?;
            super::affine::print_map(map, depth)?
        }
        Attribute::IntegerSet(set) => {
            set.verify().map_err(|e| {
                error(TextErrorKind::InvalidAttribute(
                    crate::AttributeValidationError::Affine(e),
                ))
            })?;
            super::affine::print_set(set, depth)?
        }
        Attribute::StridedLayout { offset, strides } => format!(
            "strided<[{}], offset: {}>",
            strides
                .iter()
                .map(|stride| stride.map_or_else(|| "?".into(), |stride| stride.to_string()))
                .collect::<Vec<_>>()
                .join(", "),
            offset.map_or_else(|| "?".into(), |offset| offset.to_string())
        ),
        Attribute::Complex {
            ty,
            real,
            imaginary,
        } => super::complex::print(*ty, *real, *imaginary)?,
        Attribute::SparseTensorSlice(slice) => super::sparse::print_slice(slice),
        Attribute::SparseTensorEncoding(encoding) => {
            super::sparse_encoding::print(encoding, depth)?
        }
        Attribute::Type(ty) => super::types::print(ty, depth + 1)?,
        Attribute::Enum {
            dialect,
            name,
            value,
        } => {
            if dialect != "stablehlo" {
                return Err(error(TextErrorKind::Unsupported("this dialect enum")));
            }
            let values = crate::attribute::enums::values(name)
                .ok_or_else(|| error(TextErrorKind::Unsupported("this StableHLO enum")))?;
            if !values.contains(&value.as_str()) {
                return Err(error(TextErrorKind::Expected(
                    "valid value for this StableHLO enum",
                )));
            }
            if name == "result_accuracy_mode" {
                format!("#stablehlo.{name}<{value}>")
            } else {
                format!("#stablehlo<{name} {value}>")
            }
        }
        Attribute::DenseElements { ty, values } => super::literals::print_dense(ty, values, depth)?,
        Attribute::SparseElements {
            ty,
            indices,
            values,
        } => super::literals::print_sparse(ty, indices, values, depth)?,
        Attribute::StableHlo(attribute) => print_dialect(attribute, depth)?,
        _ => {
            return Err(error(TextErrorKind::Unsupported(
                "this attribute representation",
            )));
        }
    };
    attribute
        .verify()
        .map_err(|attribute_error| error(TextErrorKind::InvalidAttribute(attribute_error)))?;
    Ok(result)
}

fn print_dialect(attribute: &StableHloAttribute, depth: usize) -> Result<String, TextError> {
    let (name, fields) = match attribute {
        StableHloAttribute::TypeExtensions(extensions) => {
            let bounds = extensions
                .bounds
                .iter()
                .map(|&bound| {
                    if bound == -1 {
                        "?".into()
                    } else {
                        bound.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Ok(format!("#stablehlo.bounds<{bounds}>"));
        }
        StableHloAttribute::DotDimensionNumbers(dimensions) => (
            "dot",
            vec![
                (
                    "lhs_batching_dimensions",
                    list(&dimensions.lhs_batching_dimensions),
                ),
                (
                    "rhs_batching_dimensions",
                    list(&dimensions.rhs_batching_dimensions),
                ),
                (
                    "lhs_contracting_dimensions",
                    list(&dimensions.lhs_contracting_dimensions),
                ),
                (
                    "rhs_contracting_dimensions",
                    list(&dimensions.rhs_contracting_dimensions),
                ),
            ],
        ),
        StableHloAttribute::GatherDimensionNumbers(dimensions) => (
            "gather",
            vec![
                ("offset_dims", list(&dimensions.offset_dims)),
                (
                    "collapsed_slice_dims",
                    list(&dimensions.collapsed_slice_dims),
                ),
                (
                    "operand_batching_dims",
                    list(&dimensions.operand_batching_dims),
                ),
                (
                    "start_indices_batching_dims",
                    list(&dimensions.start_indices_batching_dims),
                ),
                ("start_index_map", list(&dimensions.start_index_map)),
                ("index_vector_dim", dimensions.index_vector_dim.to_string()),
            ],
        ),
        StableHloAttribute::ScatterDimensionNumbers(dimensions) => (
            "scatter",
            vec![
                ("update_window_dims", list(&dimensions.update_window_dims)),
                (
                    "inserted_window_dims",
                    list(&dimensions.inserted_window_dims),
                ),
                ("input_batching_dims", list(&dimensions.input_batching_dims)),
                (
                    "scatter_indices_batching_dims",
                    list(&dimensions.scatter_indices_batching_dims),
                ),
                (
                    "scatter_dims_to_operand_dims",
                    list(&dimensions.scatter_dims_to_operand_dims),
                ),
                ("index_vector_dim", dimensions.index_vector_dim.to_string()),
            ],
        ),
        StableHloAttribute::ChannelHandle(channel) => (
            "channel_handle",
            vec![
                ("handle", channel.handle.to_string()),
                ("type", channel.channel_type.to_string()),
            ],
        ),
        StableHloAttribute::OutputOperandAlias(alias) => (
            "output_operand_alias",
            vec![
                ("output_tuple_indices", list(&alias.output_tuple_indices)),
                ("operand_index", alias.operand_index.to_string()),
                ("operand_tuple_indices", list(&alias.operand_tuple_indices)),
            ],
        ),
        StableHloAttribute::DotAlgorithm(algorithm) => (
            "dot_algorithm",
            vec![
                (
                    "lhs_precision_type",
                    super::types::print(&algorithm.lhs_precision_type, depth + 1)?,
                ),
                (
                    "rhs_precision_type",
                    super::types::print(&algorithm.rhs_precision_type, depth + 1)?,
                ),
                (
                    "accumulation_type",
                    super::types::print(&algorithm.accumulation_type, depth + 1)?,
                ),
                (
                    "lhs_component_count",
                    algorithm.lhs_component_count.to_string(),
                ),
                (
                    "rhs_component_count",
                    algorithm.rhs_component_count.to_string(),
                ),
                (
                    "num_primitive_operations",
                    algorithm.num_primitive_operations.to_string(),
                ),
                (
                    "allow_imprecise_accumulation",
                    algorithm.allow_imprecise_accumulation.to_string(),
                ),
            ],
        ),
        StableHloAttribute::ConvDimensionNumbers(_)
        | StableHloAttribute::ResultAccuracy(_)
        | StableHloAttribute::SubAxisInfo(_)
        | StableHloAttribute::AxisRef(_)
        | StableHloAttribute::MeshAxis(_)
        | StableHloAttribute::Mesh(_)
        | StableHloAttribute::ReplicaGroupMeshAxes(_) => {
            return super::geometry::print(attribute, depth);
        }
    };
    Ok(format!(
        "#stablehlo.{name}<{}>",
        fields
            .iter()
            .map(|(name, value)| format!("{name} = {value}"))
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

pub(super) fn list(values: &[i64]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn quote(value: &str) -> String {
    quote_bytes(value.as_bytes())
}

pub(super) fn quote_bytes(value: &[u8]) -> String {
    let mut output = String::from("\"");
    for &byte in value {
        match byte {
            b'"' => output.push_str("\\\""),
            b'\\' => output.push_str("\\\\"),
            0x20..=0x7e => output.push(char::from(byte)),
            _ => output.push_str(&format!("\\{byte:02X}")),
        }
    }
    output.push('"');
    output
}

fn error(kind: TextErrorKind) -> TextError {
    TextError { offset: 0, kind }
}
