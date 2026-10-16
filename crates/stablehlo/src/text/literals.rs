use crate::{Attribute, ElementType, ScalarValue, Type};

use super::{TextError, TextErrorKind, cursor::Cursor};

// Dense splats may encode enormous allocations in just a few input bytes.
// Explicitly bound expansion, independently of allocator overcommit behavior.
const MAX_EXPANDED_ELEMENTS: usize = 1_048_576;

enum Literal {
    Scalar(String),
    Complex(String, String),
    List(Vec<Literal>),
    Packed(String),
}

impl Cursor<'_> {
    pub(super) fn dense_attribute(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.expect("<")?;
        let literal = if self.take(">") {
            None
        } else {
            let literal = self.literal(depth + 1)?;
            self.expect(">")?;
            Some(literal)
        };
        self.expect(":")?;
        let ty = self.ty(depth + 1)?;
        ty.verify()
            .map_err(|error| self.error(TextErrorKind::InvalidType(error)))?;
        let (dimensions, element) = tensor(&ty).map_err(|kind| self.error(kind))?;
        let count = element_count(&dimensions).map_err(|kind| self.error(kind))?;
        let mut values = Vec::new();
        match literal {
            None if count == 0 => {}
            None => return Err(self.error(TextErrorKind::Expected("dense literal"))),
            Some(Literal::Packed(hex)) => {
                values = packed(&hex, count, element).map_err(|kind| self.error(kind))?;
            }
            Some(Literal::List(items)) => {
                let literal = Literal::List(items);
                if literal_shape(&literal).map_err(|kind| self.error(kind))? != dimensions {
                    return Err(self.error(TextErrorKind::Expected(
                        "dense literal shape matching tensor type",
                    )));
                }
                flatten(literal, &dimensions, element, &mut values)
                    .map_err(|kind| self.error(kind))?;
            }
            Some(literal) => {
                let value = scalar(literal, element).map_err(|kind| self.error(kind))?;
                if count > MAX_EXPANDED_ELEMENTS {
                    return Err(self.error(TextErrorKind::ResourceLimit));
                }
                values
                    .try_reserve_exact(count)
                    .map_err(|_| self.error(TextErrorKind::ResourceLimit))?;
                values.resize(count, value);
            }
        }
        Ok(Attribute::DenseElements {
            ty: Box::new(ty),
            values,
        })
    }

    fn literal(&mut self, depth: usize) -> Result<Literal, TextError> {
        let mut lists: Vec<Vec<Literal>> = Vec::new();
        let mut value: Option<Literal> = None;
        loop {
            if let Some(literal) = value.take() {
                if let Some(items) = lists.last_mut() {
                    items.push(literal);
                    if self.take("]") {
                        value = Some(Literal::List(lists.pop().expect("list frame")));
                    } else {
                        self.expect(",")?;
                    }
                    continue;
                }
                return Ok(literal);
            }

            if self.starts("\"") {
                value = Some(Literal::Packed(self.string()?));
            } else if self.take("[") {
                if self.take("]") {
                    value = Some(Literal::List(Vec::new()));
                } else {
                    self.depth(depth + lists.len())?;
                    lists.push(Vec::new());
                }
            } else if self.take("(") {
                let real = self.literal_token()?;
                self.expect(",")?;
                let imaginary = self.literal_token()?;
                self.expect(")")?;
                value = Some(Literal::Complex(real, imaginary));
            } else {
                value = Some(Literal::Scalar(self.literal_token()?));
            }
            self.depth(depth + lists.len())?;
        }
    }

    fn literal_token(&mut self) -> Result<String, TextError> {
        self.space();
        let start = self.offset;
        while self.input.as_bytes().get(self.offset).is_some_and(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'-' | b'_')
        }) {
            self.offset += 1;
        }
        if start == self.offset {
            return Err(self.error(TextErrorKind::Expected("element literal")));
        }
        Ok(self.input[start..self.offset].to_owned())
    }

    pub(super) fn sparse_attribute(&mut self, depth: usize) -> Result<Attribute, TextError> {
        self.expect("<")?;
        let literals = if self.take(">") {
            None
        } else {
            let indices = self.literal(depth + 1)?;
            self.expect(",")?;
            let literal = self.literal(depth + 1)?;
            self.expect(">")?;
            Some((indices, literal))
        };
        self.expect(":")?;
        let ty = self.ty(depth + 1)?;
        ty.verify()
            .map_err(|error| self.error(TextErrorKind::InvalidType(error)))?;
        let (dimensions, element) = tensor(&ty).map_err(|kind| self.error(kind))?;
        let Some((indices, literal)) = literals else {
            return Ok(Attribute::SparseElements {
                ty: Box::new(ty),
                indices: vec![],
                values: vec![],
            });
        };
        let indices = sparse_indices(indices, dimensions.len()).map_err(|kind| self.error(kind))?;
        let count = indices.len();
        let mut values = Vec::new();
        match literal {
            Literal::List(items) => flatten(Literal::List(items), &[count], element, &mut values)
                .map_err(|kind| self.error(kind))?,
            Literal::Packed(hex) => {
                values = packed(&hex, count, element).map_err(|kind| self.error(kind))?
            }
            literal => {
                let value = scalar(literal, element).map_err(|kind| self.error(kind))?;
                values.resize(count, value);
            }
        }
        Ok(Attribute::SparseElements {
            ty: Box::new(ty),
            indices,
            values,
        })
    }
}

fn sparse_indices(literal: Literal, rank: usize) -> Result<Vec<Vec<i64>>, TextErrorKind> {
    let shape = literal_shape(&literal)?;
    let (count, flattened) = if shape.is_empty() {
        if rank > MAX_EXPANDED_ELEMENTS {
            return Err(TextErrorKind::ResourceLimit);
        }
        let value = scalar(literal, ElementType::I64)?;
        (1, vec![value; rank])
    } else {
        let count = match shape.as_slice() {
            [count] if rank == 1 => *count,
            [count, columns] if *columns == rank => *count,
            _ => {
                return Err(TextErrorKind::Expected(
                    "sparse coordinate shape matching tensor rank",
                ));
            }
        };
        let mut values = Vec::new();
        flatten(literal, &shape, ElementType::I64, &mut values)?;
        (count, values)
    };
    let values = flattened
        .into_iter()
        .map(|value| match value {
            ScalarValue::Integer(value) => {
                i64::try_from(value).map_err(|_| TextErrorKind::InvalidNumber)
            }
            _ => Err(TextErrorKind::Expected("integer coordinate")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((0..count)
        .map(|index| values[index * rank..(index + 1) * rank].to_vec())
        .collect())
}

fn tensor(ty: &Type) -> Result<(Vec<usize>, ElementType), TextErrorKind> {
    let (dimensions, element) = match ty {
        Type::Tensor(tensor) => (tensor.dimensions(), tensor.element()),
        Type::QuantizedTensor(tensor) => (tensor.dimensions(), tensor.storage_type()),
        _ => return Err(TextErrorKind::Expected("statically ranked tensor type")),
    };
    let dimensions = dimensions
        .iter()
        .map(|dimension| {
            dimension
                .and_then(|dimension| usize::try_from(dimension).ok())
                .ok_or(TextErrorKind::Expected("static representable dimension"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((dimensions, element))
}

fn element_count(dimensions: &[usize]) -> Result<usize, TextErrorKind> {
    if dimensions.contains(&0) {
        return Ok(0);
    }
    dimensions.iter().try_fold(1_usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or(TextErrorKind::ResourceLimit)
    })
}

fn literal_shape(literal: &Literal) -> Result<Vec<usize>, TextErrorKind> {
    enum Event<'a> {
        Visit(&'a Literal),
        Finish(usize),
    }
    let mut events = vec![Event::Visit(literal)];
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    while let Some(event) = events.pop() {
        match event {
            Event::Visit(Literal::List(items)) => {
                events.push(Event::Finish(items.len()));
                events.extend(items.iter().rev().map(Event::Visit));
            }
            Event::Visit(_) => shapes.push(Vec::new()),
            Event::Finish(count) => {
                if count == 0 {
                    shapes.push(vec![0]);
                    continue;
                }
                let first = shapes.len() - count;
                let child_shape = shapes[first].clone();
                if shapes[first..].iter().any(|shape| shape != &child_shape) {
                    return Err(TextErrorKind::Expected("rectangular dense literal"));
                }
                shapes.truncate(first);
                let mut shape = Vec::with_capacity(child_shape.len() + 1);
                shape.push(count);
                shape.extend(child_shape);
                shapes.push(shape);
            }
        }
    }
    Ok(shapes.pop().expect("literal has a shape"))
}

fn flatten(
    literal: Literal,
    dimensions: &[usize],
    element: ElementType,
    values: &mut Vec<ScalarValue>,
) -> Result<(), TextErrorKind> {
    let mut pending = vec![(literal, 0usize)];
    while let Some((literal, dimension)) = pending.pop() {
        match (literal, dimensions.get(dimension)) {
            (Literal::List(items), Some(&count)) if items.len() == count => {
                pending.extend(items.into_iter().rev().map(|item| (item, dimension + 1)));
            }
            (literal, None) if !matches!(literal, Literal::List(_)) => {
                values.push(scalar(literal, element)?)
            }
            _ => {
                return Err(TextErrorKind::Expected(
                    "rectangular dense literal matching tensor shape",
                ));
            }
        }
    }
    Ok(())
}

fn scalar(literal: Literal, element: ElementType) -> Result<ScalarValue, TextErrorKind> {
    match literal {
        Literal::Complex(real, imaginary) => {
            let component = match element {
                ElementType::C64 => ElementType::F32,
                ElementType::C128 => ElementType::F64,
                _ => return Err(TextErrorKind::Expected("complex element type")),
            };
            Ok(ScalarValue::ComplexBits {
                real: float_bits(&real, component)?,
                imaginary: float_bits(&imaginary, component)?,
            })
        }
        Literal::Scalar(value)
            if element == ElementType::I1 && matches!(value.as_str(), "true" | "false") =>
        {
            Ok(ScalarValue::Bool(value == "true"))
        }
        Literal::Scalar(value) if element.is_float() => Ok(ScalarValue::FloatBits(u128::from(
            float_bits(&value, element)?,
        ))),
        Literal::Scalar(value) if element.is_integer() || element == ElementType::Index => {
            integer_literal(&value, element, false)
        }
        _ => Err(TextErrorKind::Expected("literal matching element type")),
    }
}

pub(super) fn parse_scalar(
    value: &str,
    element: ElementType,
) -> Result<ScalarValue, TextErrorKind> {
    scalar(Literal::Scalar(value.to_owned()), element)
}

pub(super) fn integer_literal(
    value: &str,
    element: ElementType,
    allow_unsigned_negative: bool,
) -> Result<ScalarValue, TextErrorKind> {
    let negative = value.starts_with('-');
    let unsigned_value = value.strip_prefix('-').unwrap_or(value);
    if unsigned_value.starts_with('+') {
        return Err(TextErrorKind::InvalidNumber);
    }
    let magnitude = if let Some(hex) = unsigned_value.strip_prefix("0x") {
        u128::from_str_radix(hex, 16).map_err(|_| TextErrorKind::InvalidNumber)?
    } else {
        unsigned_value
            .parse::<u128>()
            .map_err(|_| TextErrorKind::InvalidNumber)?
    };
    let width = element.bit_width().unwrap_or(64);
    if magnitude >> width != 0 {
        return Err(TextErrorKind::InvalidNumber);
    }
    if negative {
        if (is_unsigned(element) && !allow_unsigned_negative)
            || magnitude == 0
            || magnitude > 1_u128 << (width - 1)
        {
            return Err(TextErrorKind::InvalidNumber);
        }
        Ok(integer_bits((1_u128 << width) - magnitude, element))
    } else {
        if is_signed(element) && magnitude >= 1_u128 << (width - 1) {
            return Err(TextErrorKind::InvalidNumber);
        }
        Ok(integer_bits(magnitude, element))
    }
}

fn is_unsigned(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::Ui2
            | ElementType::Ui4
            | ElementType::Ui8
            | ElementType::Ui16
            | ElementType::Ui32
            | ElementType::Ui64
    )
}

fn is_signed(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::Index
            | ElementType::Si2
            | ElementType::Si4
            | ElementType::Si8
            | ElementType::Si16
            | ElementType::Si32
            | ElementType::Si64
    )
}

fn integer_bits(bits: u128, element: ElementType) -> ScalarValue {
    if element == ElementType::I1 {
        return ScalarValue::Bool(bits & 1 != 0);
    }
    let width = element.bit_width().unwrap_or(64);
    let value = if !is_unsigned(element) && bits & (1_u128 << (width - 1)) != 0 {
        bits as i128 - (1_i128 << width)
    } else {
        bits as i128
    };
    ScalarValue::Integer(value)
}

fn float_bits(value: &str, element: ElementType) -> Result<u64, TextErrorKind> {
    super::floats::parse(value, element)
}

fn packed(
    hex: &str,
    count: usize,
    element: ElementType,
) -> Result<Vec<ScalarValue>, TextErrorKind> {
    let hex = hex
        .strip_prefix("0x")
        .ok_or(TextErrorKind::Expected("hexadecimal raw dense data"))?;
    if !hex.len().is_multiple_of(2) || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(TextErrorKind::Expected("even number of hexadecimal digits"));
    }
    // Pinned MLIR DenseTypedElementsAttr stores each element in whole bytes,
    // including i1; complex elements store real then imaginary components.
    let width = element.bit_width().unwrap_or(64);
    let bytes_per_element = width.div_ceil(8) as usize;
    let bytes = hex.len() / 2;
    let splat = bytes == bytes_per_element;
    if !splat && count.checked_mul(bytes_per_element) != Some(bytes) {
        return Err(TextErrorKind::Expected(
            "raw dense byte count matching tensor type",
        ));
    }
    if count > MAX_EXPANDED_ELEMENTS {
        return Err(TextErrorKind::ResourceLimit);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| TextErrorKind::ResourceLimit)?;
    for index in 0..count {
        let byte_start = if splat { 0 } else { index * bytes_per_element };
        let encoded = &hex[byte_start * 2..(byte_start + bytes_per_element) * 2];
        let mut bits = 0_u128;
        for offset in 0..bytes_per_element {
            let byte = u8::from_str_radix(&encoded[offset * 2..offset * 2 + 2], 16)
                .map_err(|_| TextErrorKind::InvalidNumber)?;
            bits |= u128::from(byte) << (offset * 8);
        }
        let value = match element {
            ElementType::C64 => ScalarValue::ComplexBits {
                real: (bits & u128::from(u32::MAX)) as u64,
                imaginary: (bits >> 32) as u64,
            },
            ElementType::C128 => ScalarValue::ComplexBits {
                real: (bits & u128::from(u64::MAX)) as u64,
                imaginary: (bits >> 64) as u64,
            },
            ElementType::I1 => ScalarValue::Bool(bits & 1 != 0),
            _ if element.is_float() => ScalarValue::FloatBits(bits & ((1_u128 << width) - 1)),
            _ => {
                // Ignore unused padding bits in sub-byte integer storage.
                let bits = bits & ((1_u128 << width) - 1);
                integer_bits(bits, element)
            }
        };
        values.push(value);
    }
    Ok(values)
}

pub(super) fn print_dense(
    ty: &Type,
    values: &[ScalarValue],
    depth: usize,
) -> Result<String, TextError> {
    let printed_type = super::types::print(ty, depth + 1)?;
    let (dimensions, element) = tensor(ty).map_err(error)?;
    if dimensions.len() + depth >= 128 {
        return Err(error(TextErrorKind::NestingLimit));
    }
    let count = element_count(&dimensions).map_err(error)?;
    if count != values.len() {
        return Err(error(TextErrorKind::InvalidAttribute(
            crate::AttributeValidationError::DenseElementCountMismatch {
                expected: count,
                actual: values.len(),
            },
        )));
    }
    let literal = if values.is_empty() {
        String::new()
    } else if values.len() <= MAX_EXPANDED_ELEMENTS
        && values.iter().all(|value| value == &values[0])
    {
        print_scalar(&values[0], element)?
    } else {
        let mut position = 0;
        print_nested(&dimensions, values, element, &mut position)?
    };
    Ok(format!("dense<{literal}> : {printed_type}"))
}

fn print_nested(
    dimensions: &[usize],
    values: &[ScalarValue],
    element: ElementType,
    position: &mut usize,
) -> Result<String, TextError> {
    enum Event {
        Dimension(usize),
        Text(&'static str),
    }
    let mut output = String::new();
    let mut events = vec![Event::Dimension(0)];
    while let Some(event) = events.pop() {
        match event {
            Event::Text(text) => output.push_str(text),
            Event::Dimension(dimension) if dimension == dimensions.len() => {
                let value = values
                    .get(*position)
                    .ok_or_else(|| error(TextErrorKind::Expected("dense element")))?;
                *position += 1;
                output.push_str(&print_scalar(value, element)?);
            }
            Event::Dimension(dimension) => {
                let count = dimensions[dimension];
                output.push('[');
                events.push(Event::Text("]"));
                for index in (0..count).rev() {
                    events.push(Event::Dimension(dimension + 1));
                    if index > 0 {
                        events.push(Event::Text(", "));
                    }
                }
            }
        }
    }
    Ok(output)
}

pub(super) fn print_sparse(
    ty: &Type,
    indices: &[Vec<i64>],
    values: &[ScalarValue],
    depth: usize,
) -> Result<String, TextError> {
    let printed_type = super::types::print(ty, depth + 1)?;
    let (_, element) = tensor(ty).map_err(error)?;
    if indices.is_empty() {
        return Ok(format!("sparse<> : {printed_type}"));
    }
    let coordinates = indices
        .iter()
        .map(|index| super::attributes::list(index))
        .collect::<Vec<_>>()
        .join(", ");
    let values = values
        .iter()
        .map(|value| print_scalar(value, element))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!(
        "sparse<[{coordinates}], [{values}]> : {printed_type}"
    ))
}

pub(super) fn print_scalar(value: &ScalarValue, element: ElementType) -> Result<String, TextError> {
    match value {
        ScalarValue::Bool(value) => Ok(value.to_string()),
        ScalarValue::Integer(value) => Ok(value.to_string()),
        ScalarValue::FloatBits(bits) => {
            let width = element
                .bit_width()
                .ok_or_else(|| error(TextErrorKind::Expected("float width")))?
                .div_ceil(4) as usize;
            Ok(format!("0x{bits:0width$X}"))
        }
        ScalarValue::ComplexBits { real, imaginary } => {
            let width = if element == ElementType::C64 { 8 } else { 16 };
            Ok(format!("(0x{real:0width$X}, 0x{imaginary:0width$X})"))
        }
        ScalarValue::String(_) => Err(error(TextErrorKind::Unsupported("string tensor elements"))),
    }
}

fn error(kind: TextErrorKind) -> TextError {
    TextError { offset: 0, kind }
}
