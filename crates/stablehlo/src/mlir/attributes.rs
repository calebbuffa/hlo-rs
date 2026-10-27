use super::{TextError, TextErrorKind, cursor::Cursor, floats};
use crate::{
    attributes::*,
    integer::IntegerValue,
    pliron::{
        attribute::AttrObj,
        builtin::attributes::{BoolAttr, DictAttr, StringAttr, TypeAttr, UnitAttr, VecAttr},
        builtin::types::Signedness,
        context::Context,
        identifier::Identifier,
        r#type::TypeHandle,
    },
    types::*,
};
use std::collections::HashSet;

/// Parses and verifies one complete MLIR attribute in `ctx`.
///
/// Leading and trailing whitespace are accepted; trailing entities are not.
/// Types referenced by the returned attribute are interned in `ctx`.
///
/// # Errors
///
/// Returns [`TextError`] for unsupported syntax, malformed literals, trailing
/// input, excessive nesting, or attribute verification failures.
pub fn parse_attribute(ctx: &Context, input: &str) -> Result<AttrObj, TextError> {
    let mut cursor = Cursor::new(input);
    let attribute = cursor.attribute(ctx, 0)?;
    cursor.finish()?;
    verify_attribute(ctx, &attribute)
        .map_err(|error| cursor.error(TextErrorKind::InvalidAttribute(error.to_string())))?;
    Ok(attribute)
}

/// Parses a verified MLIR dictionary whose keys are valid pliron identifiers.
///
/// # Errors
///
/// Returns [`TextError`] if attribute parsing or verification fails, the entity
/// is not a dictionary, or its keys require [`NamedAttributesAttr`] storage.
/// Use [`parse_named_attributes`] for arbitrary MLIR dictionary keys.
pub fn parse_dictionary(ctx: &Context, input: &str) -> Result<DictAttr, TextError> {
    let attribute = parse_attribute(ctx, input)?;
    attribute
        .downcast_ref::<DictAttr>()
        .cloned()
        .ok_or(TextError {
            offset: 0,
            kind: TextErrorKind::Expected("attribute dictionary"),
        })
}

/// Parses a verified MLIR dictionary with arbitrary nonempty, unique string keys.
///
/// Unlike [`parse_dictionary`], this accepts keys that are not pliron
/// identifiers. It returns an owned [`NamedAttributesAttr`] container.
///
/// # Errors
///
/// Returns [`TextError`] if attribute parsing or verification fails or the
/// complete input does not represent a dictionary.
pub fn parse_named_attributes(
    ctx: &Context,
    input: &str,
) -> Result<NamedAttributesAttr, TextError> {
    let attribute = parse_attribute(ctx, input)?;
    if let Some(dictionary) = attribute.downcast_ref::<NamedAttributesAttr>() {
        return Ok(dictionary.clone());
    }
    if let Some(dictionary) = attribute.downcast_ref::<DictAttr>() {
        return Ok(NamedAttributesAttr::new(
            dictionary
                .0
                .0
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone())),
        ));
    }
    Err(TextError {
        offset: 0,
        kind: TextErrorKind::Expected("attribute dictionary"),
    })
}

impl Cursor<'_> {
    pub(super) fn attribute(&mut self, ctx: &Context, depth: usize) -> Result<AttrObj, TextError> {
        if depth >= 128 {
            return Err(self.error(TextErrorKind::NestingLimit));
        }
        if self.take("[") {
            let mut values = Vec::new();
            if !self.take("]") {
                loop {
                    values.push(self.attribute(ctx, depth + 1)?);
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            return Ok(Box::new(VecAttr::new(values)));
        }
        if self.take("{") {
            let mut entries = Vec::new();
            let mut names = HashSet::new();
            if !self.take("}") {
                loop {
                    let name = self.symbol()?;
                    if name.is_empty() || !names.insert(name.clone()) {
                        return Err(self.error(TextErrorKind::DuplicateField(name)));
                    }
                    let value: AttrObj = if self.take("=") {
                        self.attribute(ctx, depth + 1)?
                    } else {
                        Box::new(UnitAttr::new())
                    };
                    entries.push((name, value));
                    if self.take("}") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            if entries.iter().all(|(name, _)| Identifier::is_valid(name)) {
                let entries = entries
                    .into_iter()
                    .map(|(name, value)| {
                        Identifier::try_new(name)
                            .map(|name| (name, value))
                            .map_err(|error| {
                                self.error(TextErrorKind::InvalidAttribute(error.to_string()))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Box::new(DictAttr::new(entries)));
            }
            return Ok(Box::new(NamedAttributesAttr::new(entries)));
        }
        if self.starts("\"") {
            let bytes = self.string_bytes()?;
            return if self.take(":") {
                let ty = self.ty(ctx, depth + 1)?;
                Ok(Box::new(TypedStringAttr { value: bytes, ty }))
            } else {
                Ok(Box::new(BytesAttr { value: bytes }))
            };
        }
        if self.take("@") {
            let mut path = vec![StringAttr::from(self.symbol()?)];
            while self.take("::") {
                self.expect("@")?;
                path.push(StringAttr::from(self.symbol()?));
            }
            return Ok(Box::new(SymbolRefAttr { path }));
        }
        if self.keyword("unit") {
            return Ok(Box::new(UnitAttr::new()));
        }
        if self.keyword("affine_map") {
            return self.affine_attribute(false, depth + 1);
        }
        if self.keyword("affine_set") {
            return self.affine_attribute(true, depth + 1);
        }
        if self.keyword("strided") {
            self.expect("<")?;
            self.expect("[")?;
            let mut strides = Vec::new();
            if !self.take("]") {
                loop {
                    strides.push(if self.take("?") {
                        i64::MIN
                    } else {
                        i64::try_from(self.integer()?)
                            .map_err(|_| self.error(TextErrorKind::InvalidNumber))?
                    });
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            let mut offset = vec![0];
            if self.take(",") {
                if !self.keyword("offset") {
                    return Err(self.error(TextErrorKind::Expected("offset")));
                }
                self.expect(":")?;
                offset = if self.take("?") {
                    Vec::new()
                } else {
                    vec![
                        i64::try_from(self.integer()?)
                            .map_err(|_| self.error(TextErrorKind::InvalidNumber))?,
                    ]
                };
            }
            self.expect(">")?;
            return Ok(Box::new(StridedLayoutAttr { offset, strides }));
        }
        for (keyword, value) in [("true", true), ("false", false)] {
            if self.keyword(keyword) {
                return Ok(Box::new(BoolAttr::new(value)));
            }
        }
        if self.keyword("array") {
            self.expect("<")?;
            let element = self.ty(ctx, depth + 1)?;
            let mut values = Vec::new();
            if !self.take(">") {
                self.expect(":")?;
                loop {
                    let token = self.number()?;
                    values.push(self.scalar(ctx, element, &token)?);
                    if self.take(">") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            return Ok(Box::new(DenseArrayAttr {
                element,
                values: VecAttr::new(values),
            }));
        }
        if self.keyword("dense") {
            return self.dense(ctx, depth + 1);
        }
        if self.keyword("sparse") {
            return self.sparse(ctx, depth + 1);
        }
        if self.starts("#") {
            self.expect("#")?;
            let mut name = self.word()?.to_owned();
            if name == "complex.number" || name == "complex" {
                return self.complex_attribute(ctx, name == "complex", depth + 1);
            }
            let generic = name == "stablehlo";
            if generic {
                self.expect("<")?;
                name = format!("stablehlo.{}", self.word()?);
            }
            if name == "stablehlo.type_extensions" {
                self.expect("<")?;
                if !self.keyword("bounds") {
                    return Err(self.error(TextErrorKind::Expected("bounds")));
                }
                self.expect("=")?;
                self.expect("[")?;
                let mut bounds = Vec::new();
                if !self.take("]") {
                    loop {
                        bounds.push(if self.take("?") {
                            -1
                        } else {
                            i64::try_from(self.integer()?)
                                .map_err(|_| self.error(TextErrorKind::InvalidNumber))?
                        });
                        if self.take("]") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                self.expect(">")?;
                if generic {
                    self.expect(">")?;
                }
                return Ok(Box::new(TypeExtensionsAttr { bounds }));
            }
            if let Some(name) = name.strip_prefix("stablehlo.")
                && matches!(
                    name,
                    "precision"
                        | "result_accuracy_mode"
                        | "fft_type"
                        | "comparison_direction"
                        | "comparison_type"
                        | "rng_algorithm"
                        | "rng_distribution"
                )
            {
                let delimited = !generic || (name == "result_accuracy_mode" && self.starts("<"));
                if delimited {
                    self.expect("<")?;
                }
                let spelling = self.word()?;
                self.expect(">")?;
                if generic && delimited {
                    self.expect(">")?;
                }
                macro_rules! parse_enum {
                    ($type:ty) => {
                        spelling
                            .parse::<$type>()
                            .map_err(|_| self.error(TextErrorKind::Expected("enum value")))
                    };
                }
                let parsed: Option<AttrObj> = match name {
                    "precision" => Some(Box::new(parse_enum!(Precision)?)),
                    "comparison_direction" => Some(Box::new(parse_enum!(ComparisonDirection)?)),
                    "comparison_type" => Some(Box::new(parse_enum!(ComparisonKind)?)),
                    "rng_algorithm" => Some(Box::new(parse_enum!(RngAlgorithmAttr)?)),
                    "rng_distribution" => Some(Box::new(parse_enum!(RngDistributionAttr)?)),
                    "result_accuracy_mode" => Some(Box::new(parse_enum!(ResultAccuracyMode)?)),
                    "fft_type" => Some(Box::new(parse_enum!(FftKind)?)),
                    _ => None,
                };
                if let Some(parsed) = parsed {
                    return Ok(parsed);
                }
            }
            if let Some(name) = name.strip_prefix("stablehlo.") {
                let attribute = self.stablehlo_attribute(ctx, name, depth + 1)?;
                if generic {
                    self.expect(">")?;
                }
                return Ok(attribute);
            }
            return Err(self.error(TextErrorKind::Unsupported("dialect attribute")));
        }
        self.space();
        if self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'-')
        {
            let token = self.number()?;
            let ty = if self.take(":") {
                self.ty(ctx, depth + 1)?
            } else {
                if token.contains('.') {
                    ElementType::F64
                } else {
                    ElementType::I64
                }
                .native(ctx)
            };
            return self.scalar(ctx, ty, &token);
        }
        let ty = self.ty(ctx, depth + 1)?;
        Ok(Box::new(TypeAttr::new(ty)))
    }

    fn scalar(&self, ctx: &Context, ty: TypeHandle, token: &str) -> Result<AttrObj, TextError> {
        let element = primitive_element(ctx, ty)
            .ok_or_else(|| self.error(TextErrorKind::Unsupported("nonprimitive scalar")))?;
        if let Some(component) = element.complex_component() {
            let mut literal = Cursor::new(token);
            literal.expect("(")?;
            let real = literal.number()?;
            literal.expect(",")?;
            let imaginary = literal.number()?;
            literal.expect(")")?;
            literal.finish()?;
            let component_type = component.native(ctx);
            let real = self.scalar(ctx, component_type, &real)?;
            let imaginary = self.scalar(ctx, component_type, &imaginary)?;
            let real = real
                .downcast_ref::<ScalarBitsAttr>()
                .ok_or_else(|| self.error(TextErrorKind::Expected("complex component bits")))?;
            let imaginary = imaginary
                .downcast_ref::<ScalarBitsAttr>()
                .ok_or_else(|| self.error(TextErrorKind::Expected("complex component bits")))?;
            let width = component
                .bit_width()
                .ok_or_else(|| self.error(TextErrorKind::Expected("fixed-width component")))?;
            let mut words = vec![0_u64; (2 * width).div_ceil(64) as usize];
            for (offset, component) in [(0, real), (width, imaginary)] {
                for (index, &word) in component.words.iter().enumerate() {
                    let start = offset as usize + index * 64;
                    let shift = start % 64;
                    words[start / 64] |= word << shift;
                    if shift != 0
                        && let Some(next) = words.get_mut(start / 64 + 1)
                    {
                        *next |= word >> (64 - shift);
                    }
                }
            }
            return Ok(Box::new(ScalarBitsAttr { ty, words }));
        }
        if element.is_float() {
            let bits = floats::parse(token, element).map_err(|kind| self.error(kind))?;
            return Ok(Box::new(ScalarBitsAttr {
                ty,
                words: vec![bits],
            }));
        }
        if element == ElementType::Index {
            return token
                .parse::<i64>()
                .map(|value| Box::new(IndexAttr { value }) as AttrObj)
                .map_err(|_| self.error(TextErrorKind::InvalidNumber));
        }
        let (negative, magnitude) = token
            .strip_prefix('-')
            .map_or((false, token), |tail| (true, tail));
        let (digits, radix) = magnitude
            .strip_prefix("0x")
            .map_or((magnitude, 10), |digits| (digits, 16));
        let number = if negative {
            format!("-{digits}")
        } else {
            digits.to_owned()
        };
        let value = IntegerValue::parse_radix(&number, radix)
            .ok_or_else(|| self.error(TextErrorKind::InvalidNumber))?;
        if negative && value == IntegerValue::ZERO {
            return Err(self.error(TextErrorKind::InvalidNumber));
        }
        let (width, signedness) = element
            .integer_parts()
            .ok_or_else(|| self.error(TextErrorKind::Expected("integer scalar")))?;
        let range = if signedness == Signedness::Signless && !negative {
            Signedness::Unsigned
        } else {
            signedness
        };
        if !value.fits_width(width, range) {
            return Err(self.error(TextErrorKind::InvalidNumber));
        }
        Ok(Box::new(ScalarBitsAttr {
            ty,
            words: value.to_twos_complement_words(width),
        }))
    }

    fn dense(&mut self, ctx: &Context, depth: usize) -> Result<AttrObj, TextError> {
        self.expect("<")?;
        let packed = if self.starts("\"") {
            Some(self.string_bytes()?)
        } else {
            None
        };
        let empty = self.starts(">");
        let (tokens, parsed_shape) = if empty || packed.is_some() {
            (Vec::new(), Some(Vec::new()))
        } else {
            self.element_literals(depth)?
        };
        self.expect(">")?;
        self.expect(":")?;
        let ty = self.ty(ctx, depth)?;
        let borrowed = ty.deref(ctx);
        let (shape, element) = if let Some(tensor) = borrowed.downcast_ref::<TensorType>() {
            (&tensor.dimensions, tensor.element)
        } else if let Some(vector) = borrowed.downcast_ref::<VectorType>() {
            (&vector.dimensions, vector.element)
        } else {
            return Err(self.error(TextErrorKind::Expected("ranked tensor or vector type")));
        };
        if let Some(packed) = packed {
            let count = element_count(shape)
                .ok_or_else(|| self.error(TextErrorKind::Expected("static element count")))?;
            let (values, is_splat) = self.packed_elements(ctx, &packed, count, element)?;
            return Ok(Box::new(DenseElementsAttr {
                ty,
                values: VecAttr::new(values),
                is_splat,
            }));
        }
        let splat = parsed_shape.is_none();
        if let Some(parsed_shape) = parsed_shape
            && !(empty && element_count(shape) == Some(0))
            && parsed_shape != *shape
        {
            return Err(self.error(TextErrorKind::InvalidAttribute(
                "dense literal shape does not match its type".into(),
            )));
        }
        let values = self.literal_values(ctx, element, &tokens)?;
        Ok(Box::new(DenseElementsAttr {
            ty,
            values: VecAttr::new(values),
            is_splat: splat,
        }))
    }

    fn literal_values(
        &self,
        ctx: &Context,
        element: TypeHandle,
        tokens: &[String],
    ) -> Result<Vec<AttrObj>, TextError> {
        tokens
            .iter()
            .map(|token| {
                if token == "true" || token == "false" {
                    if primitive_element(ctx, element) != Some(ElementType::I1) {
                        return Err(self.error(TextErrorKind::InvalidAttribute(
                            "boolean elements require i1".into(),
                        )));
                    }
                    Ok(Box::new(BoolAttr::new(token == "true")) as AttrObj)
                } else {
                    self.scalar(ctx, element, token)
                }
            })
            .collect()
    }

    fn sparse(&mut self, ctx: &Context, depth: usize) -> Result<AttrObj, TextError> {
        self.expect("<")?;
        let empty = self.take(">");
        let (coordinates, coordinate_shape, tokens, value_shape, packed) = if empty {
            (Vec::new(), None, Vec::new(), None, None)
        } else {
            let (coordinates, shape) = self.element_literals(depth)?;
            self.expect(",")?;
            let packed = if self.starts("\"") {
                Some(self.string_bytes()?)
            } else {
                None
            };
            let (tokens, value_shape) = if packed.is_some() {
                (Vec::new(), None)
            } else {
                self.element_literals(depth)?
            };
            self.expect(">")?;
            (coordinates, shape, tokens, value_shape, packed)
        };
        self.expect(":")?;
        let ty = self.ty(ctx, depth)?;
        let borrowed = ty.deref(ctx);
        let (shape, element) = if let Some(tensor) = borrowed.downcast_ref::<TensorType>() {
            (&tensor.dimensions, tensor.element)
        } else if let Some(vector) = borrowed.downcast_ref::<VectorType>() {
            (&vector.dimensions, vector.element)
        } else {
            return Err(self.error(TextErrorKind::Expected("ranked tensor or vector type")));
        };
        let rank = shape.len();
        let count = if empty {
            0
        } else {
            match coordinate_shape.as_deref() {
                None => 1,
                Some([count]) if rank == 1 => usize::try_from(*count).map_err(|_| {
                    self.error(TextErrorKind::InvalidAttribute(
                        "sparse coordinate count must be nonnegative and fit usize".into(),
                    ))
                })?,
                Some([count, columns]) if usize::try_from(*columns).ok() == Some(rank) => {
                    usize::try_from(*count).map_err(|_| {
                        self.error(TextErrorKind::InvalidAttribute(
                            "sparse coordinate count must be nonnegative and fit usize".into(),
                        ))
                    })?
                }
                _ => {
                    return Err(self.error(TextErrorKind::Expected(
                        "sparse coordinate shape matching tensor rank",
                    )));
                }
            }
        };
        let mut flat = Vec::new();
        for coordinate in coordinates {
            let value = self.scalar(ctx, ElementType::I64.native(ctx), &coordinate)?;
            let value = value
                .downcast_ref::<ScalarBitsAttr>()
                .and_then(|value| value.integer_value(ctx))
                .and_then(|value| value.to_i128())
                .and_then(|value| i64::try_from(value).ok())
                .ok_or_else(|| self.error(TextErrorKind::InvalidNumber))?;
            flat.push(value);
        }
        if !empty && coordinate_shape.is_none() {
            let Some(&value) = flat.first() else {
                return Err(self.error(TextErrorKind::Expected("sparse coordinate")));
            };
            flat.resize(rank, value);
        }
        if count.checked_mul(rank) != Some(flat.len()) {
            return Err(self.error(TextErrorKind::InvalidAttribute(
                "sparse coordinate cardinality does not match its shape".into(),
            )));
        }
        let mut indices = Vec::new();
        for index in 0..count {
            let start = index.checked_mul(rank).ok_or_else(|| {
                self.error(TextErrorKind::InvalidAttribute(
                    "sparse coordinate offset overflow".into(),
                ))
            })?;
            let values = start
                .checked_add(rank)
                .and_then(|end| flat.get(start..end))
                .ok_or_else(|| {
                    self.error(TextErrorKind::InvalidAttribute(
                        "sparse coordinate range does not match its shape".into(),
                    ))
                })?;
            indices.push(IndexTensorAttr {
                values: values.to_vec(),
            });
        }
        let values = if empty {
            Vec::new()
        } else if let Some(packed) = packed {
            let (mut values, splat) = self.packed_elements(ctx, &packed, count, element)?;
            if splat {
                let value = values
                    .first()
                    .cloned()
                    .ok_or_else(|| self.error(TextErrorKind::Expected("sparse splat value")))?;
                values.resize(count, value);
            }
            values
        } else {
            if value_shape.as_deref().is_some_and(|shape| {
                shape.len() != 1 || usize::try_from(shape[0]).ok() != Some(count)
            }) {
                return Err(self.error(TextErrorKind::Expected(
                    "sparse values matching coordinate count",
                )));
            }
            let mut values = self.literal_values(ctx, element, &tokens)?;
            if value_shape.is_none() {
                let value = values
                    .first()
                    .cloned()
                    .ok_or_else(|| self.error(TextErrorKind::Expected("sparse value")))?;
                values.resize(count, value);
            }
            values
        };
        Ok(Box::new(SparseElementsAttr {
            ty,
            indices,
            values: VecAttr::new(values),
        }))
    }

    fn packed_elements(
        &self,
        ctx: &Context,
        literal: &[u8],
        count: usize,
        element: TypeHandle,
    ) -> Result<(Vec<AttrObj>, bool), TextError> {
        let hex = std::str::from_utf8(literal)
            .ok()
            .and_then(|text| text.strip_prefix("0x"))
            .filter(|text| {
                text.len().is_multiple_of(2) && text.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            .ok_or_else(|| {
                self.error(TextErrorKind::Expected("even-length hexadecimal raw data"))
            })?;
        let primitive = primitive_element(ctx, element)
            .ok_or_else(|| self.error(TextErrorKind::Expected("primitive packed element type")))?;
        let component = primitive.complex_component();
        let width = component
            .unwrap_or(primitive)
            .bit_width()
            .ok_or_else(|| self.error(TextErrorKind::Expected("fixed-width packed element")))?;
        let component_bytes = width.div_ceil(8) as usize;
        let element_bytes = component_bytes * if component.is_some() { 2 } else { 1 };
        let bytes = hex.len() / 2;
        let splat = bytes == element_bytes;
        if !splat && count.checked_mul(element_bytes) != Some(bytes) {
            return Err(self.error(TextErrorKind::Expected(
                "raw byte count matching element count",
            )));
        }
        let size = if count == 0 {
            0
        } else if splat {
            1
        } else {
            count
        };
        let mut values = Vec::with_capacity(size);
        for index in 0..size {
            let hex = &hex[index * element_bytes * 2..(index + 1) * element_bytes * 2];
            let total_width = width * if component.is_some() { 2 } else { 1 };
            let mut words = vec![0_u64; total_width.div_ceil(64) as usize];
            for (byte_index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
                let pair = std::str::from_utf8(pair)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                let byte = u8::from_str_radix(pair, 16)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                let component_index = byte_index / component_bytes;
                let component_bit = (byte_index % component_bytes) as u32 * 8;
                for bit in 0..8 {
                    if component_bit + bit < width && byte & (1 << bit) != 0 {
                        let position = component_index as u32 * width + component_bit + bit;
                        words[position as usize / 64] |= 1 << (position % 64);
                    }
                }
            }
            values.push(Box::new(ScalarBitsAttr { ty: element, words }) as AttrObj);
        }
        Ok((values, splat && count != 0))
    }

    fn element_literals(
        &mut self,
        depth: usize,
    ) -> Result<(Vec<String>, Option<Vec<i64>>), TextError> {
        if depth >= 128 {
            return Err(self.error(TextErrorKind::NestingLimit));
        }
        if !self.take("[") {
            let token = if self.take("(") {
                let real = self.number()?;
                self.expect(",")?;
                let imaginary = self.number()?;
                self.expect(")")?;
                format!("({real},{imaginary})")
            } else if self.keyword("true") {
                "true".into()
            } else if self.keyword("false") {
                "false".into()
            } else {
                self.number()?
            };
            return Ok((vec![token], None));
        }
        let mut tokens = Vec::new();
        let mut shape = None;
        let mut count = 0_i64;
        if !self.take("]") {
            loop {
                let (values, child_shape) = self.element_literals(depth + 1)?;
                let child_shape = child_shape.unwrap_or_default();
                if shape.as_ref().is_some_and(|shape| *shape != child_shape) {
                    return Err(self.error(TextErrorKind::InvalidAttribute(
                        "ragged dense elements".into(),
                    )));
                }
                shape = Some(child_shape);
                tokens.extend(values);
                count = count.checked_add(1).ok_or_else(|| {
                    self.error(TextErrorKind::InvalidAttribute(
                        "element literal dimension overflow".into(),
                    ))
                })?;
                if self.take("]") {
                    break;
                }
                self.expect(",")?;
            }
        }
        let mut dimensions = vec![count];
        dimensions.extend(shape.unwrap_or_default());
        Ok((tokens, Some(dimensions)))
    }
}
