use crate::{
    Attribute, BufferLayout, ElementType, FunctionType, QuantizedElementType, QuantizedTensorType,
    TensorType, Type, UnrankedTensorType,
};

use super::{TextError, TextErrorKind, cursor::Cursor};

macro_rules! elements {
    ($($variant:ident => $name:literal),* $(,)?) => {
        fn element_name(element: ElementType) -> &'static str {
            match element { $(ElementType::$variant => $name,)* }
        }
        fn element_from_name(name: &str) -> Option<ElementType> {
            match name { $($name => Some(ElementType::$variant),)* _ => None }
        }
    };
}

elements! {
    Index => "index", I1 => "i1", I2 => "i2", I4 => "i4", I8 => "i8",
    I16 => "i16", I32 => "i32", I64 => "i64",
    Si2 => "si2", Si4 => "si4", Si8 => "si8", Si16 => "si16", Si32 => "si32", Si64 => "si64",
    Ui2 => "ui2", Ui4 => "ui4", Ui8 => "ui8", Ui16 => "ui16", Ui32 => "ui32", Ui64 => "ui64",
    F4E2M1Fn => "f4E2M1FN", F6E2M3Fn => "f6E2M3FN", F6E3M2Fn => "f6E3M2FN",
    F16 => "f16", Bf16 => "bf16", F32 => "f32", F64 => "f64", Tf32 => "tf32",
    F8E3M4 => "f8E3M4", F8E4M3 => "f8E4M3", F8E4M3Fn => "f8E4M3FN",
    F8E4M3FnUz => "f8E4M3FNUZ", F8E4M3B11FnUz => "f8E4M3B11FNUZ",
    F8E5M2 => "f8E5M2", F8E5M2FnUz => "f8E5M2FNUZ", F8E8M0FnUz => "f8E8M0FNU",
    C64 => "complex<f32>", C128 => "complex<f64>",
}

impl Cursor<'_> {
    pub(super) fn ty(&mut self, depth: usize) -> Result<Type, TextError> {
        self.depth(depth)?;
        if self.take("tensor") {
            self.expect("<")?;
            if self.take("*") {
                return self.unranked_tensor_type(depth);
            }
            let dimensions = self.dimensions()?;
            let mut ty = match self.ty(depth + 1)? {
                Type::QuantizedElement(element) => {
                    Type::QuantizedTensor(QuantizedTensorType::from_element(dimensions, element))
                }
                Type::Element(element) => Type::Tensor(TensorType::new(element, dimensions)),
                _ => return Err(self.error(TextErrorKind::Expected("tensor element type"))),
            };
            if self.take(",") {
                let encoding = self.attribute(depth + 1)?;
                ty = match ty {
                    Type::Tensor(tensor) => tensor.with_encoding(encoding).into(),
                    Type::QuantizedTensor(tensor) => {
                        Type::QuantizedTensor(tensor.with_encoding(encoding))
                    }
                    _ => unreachable!("constructed ranked tensor"),
                };
            }
            self.expect(">")?;
            Ok(ty)
        } else if self.take("tuple") {
            self.expect("<")?;
            Ok(Type::Tuple(self.type_list(">", depth + 1)?))
        } else if self.take("!stablehlo.token") {
            Ok(Type::Token)
        } else if self.take("!stablehlo.future") {
            self.expect("<")?;
            Ok(Type::Future(self.type_list(">", depth + 1)?))
        } else if self.take("!quant.uniform") {
            Ok(Type::QuantizedElement(self.quantized(depth + 1)?))
        } else if self.starts("!") {
            self.type_alias(depth)
        } else if self.take("memref") {
            self.buffer_type(depth)
        } else if self.take("(") {
            let inputs = self.type_list(")", depth + 1)?;
            self.expect("->")?;
            let outputs = if self.take("(") {
                self.type_list(")", depth + 1)?
            } else {
                vec![self.ty(depth + 1)?]
            };
            Ok(Type::Function(FunctionType::new(inputs, outputs)))
        } else {
            Ok(Type::Element(self.element(depth)?))
        }
    }

    fn unranked_tensor_type(&mut self, depth: usize) -> Result<Type, TextError> {
        self.expect("x")?;
        let tensor = match self.ty(depth + 1)? {
            Type::Element(element) => UnrankedTensorType::Plain(element),
            Type::QuantizedElement(element) => UnrankedTensorType::Quantized(element),
            _ => return Err(self.error(TextErrorKind::Expected("tensor element type"))),
        };
        self.expect(">")?;
        Ok(tensor.into())
    }

    pub(super) fn type_list(
        &mut self,
        end: &'static str,
        depth: usize,
    ) -> Result<Vec<Type>, TextError> {
        let mut types = Vec::new();
        if !self.take(end) {
            loop {
                types.push(self.ty(depth)?);
                if self.take(end) {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(types)
    }

    pub(super) fn element(&mut self, depth: usize) -> Result<ElementType, TextError> {
        self.depth(depth)?;
        if self.starts("!") {
            return match self.type_alias(depth)? {
                Type::Element(element) => Ok(element),
                _ => Err(self.error(TextErrorKind::Expected("scalar element type alias"))),
            };
        }
        if self.take("complex") {
            self.expect("<")?;
            let element = match self.element(depth + 1)? {
                ElementType::F32 => ElementType::C64,
                ElementType::F64 => ElementType::C128,
                _ => {
                    return Err(self.error(TextErrorKind::Expected("f32 or f64 complex component")));
                }
            };
            self.expect(">")?;
            return Ok(element);
        }
        let name = self.word()?;
        element_from_name(name)
            .ok_or_else(|| self.error(TextErrorKind::Expected("supported element type")))
    }

    pub(super) fn dimensions(&mut self) -> Result<Vec<Option<u64>>, TextError> {
        let mut dimensions = Vec::new();
        loop {
            self.space();
            let byte = self.input.as_bytes().get(self.offset);
            if byte == Some(&b'?') {
                self.offset += 1;
                dimensions.push(None);
            } else if byte.is_some_and(u8::is_ascii_digit) {
                let start = self.offset;
                while self
                    .input
                    .as_bytes()
                    .get(self.offset)
                    .is_some_and(u8::is_ascii_digit)
                {
                    self.offset += 1;
                }
                let size = self.input[start..self.offset]
                    .parse::<u64>()
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?;
                dimensions.push(Some(size));
            } else {
                break;
            }
            self.expect("x")?;
        }
        Ok(dimensions)
    }

    fn quantized(&mut self, depth: usize) -> Result<QuantizedElementType, TextError> {
        self.expect("<")?;
        let storage = self.element(depth)?;
        let (min, max) = if self.take("<") {
            let min = self.integer()?;
            self.expect(":")?;
            let max = self.integer()?;
            self.expect(">")?;
            (Some(min), Some(max))
        } else {
            (None, None)
        };
        self.expect(":")?;
        let expressed = self.element(depth)?;
        let axis = if self.take(":") {
            Some(
                usize::try_from(self.integer()?)
                    .map_err(|_| self.error(TextErrorKind::InvalidNumber))?,
            )
        } else {
            None
        };
        self.expect(",")?;
        let mut scales = Vec::new();
        let mut zeros = Vec::new();
        if axis.is_some() {
            self.expect("{")?;
            if !self.take("}") {
                loop {
                    scales.push(self.float()?);
                    zeros.push(if self.take(":") { self.integer()? } else { 0 });
                    if self.take("}") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
        } else {
            scales.push(self.float()?);
            zeros.push(if self.take(":") { self.integer()? } else { 0 });
        }
        self.expect(">")?;
        Ok(QuantizedElementType::new(
            storage, min, max, expressed, axis, scales, zeros,
        ))
    }
}

pub(super) fn print(ty: &Type, depth: usize) -> Result<String, TextError> {
    let mut events = vec![TypePrintEvent::Type(ty, depth)];
    let mut result = String::new();
    while let Some(event) = events.pop() {
        match event {
            TypePrintEvent::Text(text) => result.push_str(&text),
            TypePrintEvent::Type(ty, depth) => {
                if depth >= 128 {
                    return Err(error(TextErrorKind::NestingLimit));
                }
                match ty {
                    Type::Tuple(elements) => {
                        events.push(TypePrintEvent::Text(">".into()));
                        push_type_list(&mut events, elements, depth + 1);
                        events.push(TypePrintEvent::Text("tuple<".into()));
                    }
                    Type::Future(elements) => {
                        events.push(TypePrintEvent::Text(">".into()));
                        push_type_list(&mut events, elements, depth + 1);
                        events.push(TypePrintEvent::Text("!stablehlo.future<".into()));
                    }
                    Type::Function(function) => {
                        events.push(TypePrintEvent::Text(")".into()));
                        push_type_list(&mut events, function.outputs(), depth + 1);
                        events.push(TypePrintEvent::Text(") -> (".into()));
                        push_type_list(&mut events, function.inputs(), depth + 1);
                        events.push(TypePrintEvent::Text("(".into()));
                    }
                    _ => result.push_str(&print_leaf(ty, depth)?),
                }
            }
        }
    }
    ty.verify()
        .map_err(|type_error| error(TextErrorKind::InvalidType(type_error)))?;
    Ok(result)
}

enum TypePrintEvent<'a> {
    Type(&'a Type, usize),
    Text(String),
}

fn push_type_list<'a>(events: &mut Vec<TypePrintEvent<'a>>, types: &'a [Type], depth: usize) {
    for (index, ty) in types.iter().enumerate().rev() {
        events.push(TypePrintEvent::Type(ty, depth));
        if index > 0 {
            events.push(TypePrintEvent::Text(", ".into()));
        }
    }
}

fn print_leaf(ty: &Type, depth: usize) -> Result<String, TextError> {
    Ok(match ty {
        Type::Element(element) => element_name(*element).into(),
        Type::Token => "!stablehlo.token".into(),
        Type::UnrankedTensor(tensor) => format!(
            "tensor<*x{}>",
            match tensor {
                UnrankedTensorType::Plain(element) => element_name(*element).into(),
                UnrankedTensorType::Quantized(element) => print_quantized(element)?,
            }
        ),
        Type::Tensor(tensor) => format!(
            "tensor<{}{}{}>",
            shape(tensor.dimensions()),
            element_name(tensor.element()),
            encoding(tensor.encoding(), depth + 1)?
        ),
        Type::QuantizedElement(element) => print_quantized(element)?,
        Type::QuantizedTensor(tensor) => format!(
            "tensor<{}{}{}>",
            shape(tensor.dimensions()),
            print_quantized(tensor.element())?,
            encoding(tensor.encoding(), depth + 1)?
        ),
        Type::Buffer(buffer) => {
            let mut output = format!(
                "memref<{}{}",
                shape(buffer.dimensions()),
                element_name(buffer.element())
            );
            match buffer.layout() {
                None | Some(BufferLayout::Identity) => {}
                Some(BufferLayout::Affine(map)) => {
                    map.verify().map_err(|error| {
                        self::error(TextErrorKind::InvalidAttribute(
                            crate::AttributeValidationError::Affine(error),
                        ))
                    })?;
                    output.push_str(", ");
                    output.push_str(&super::affine::print_map(map, depth + 1)?);
                }
                Some(BufferLayout::Strided { offset, strides }) => {
                    let strides = strides
                        .iter()
                        .map(|value| value.map_or("?".into(), |value| value.to_string()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let offset = offset.map_or("?".into(), |value| value.to_string());
                    output.push_str(&format!(", strided<[{strides}], offset: {offset}>"));
                }
                Some(BufferLayout::Opaque(_)) => {
                    return Err(error(TextErrorKind::Unsupported("opaque buffer layout")));
                }
            }
            output.push_str(&encoding(buffer.memory_space(), depth + 1)?);
            output.push('>');
            output
        }
        Type::Tuple(_) | Type::Future(_) | Type::Function(_) => {
            return Err(error(TextErrorKind::Unsupported(
                "nested type printer event",
            )));
        }
        Type::String => return Err(error(TextErrorKind::Unsupported("standalone string type"))),
    })
}

fn print_quantized(element: &QuantizedElementType) -> Result<String, TextError> {
    element
        .verify()
        .map_err(|error| self::error(TextErrorKind::InvalidType(error)))?;
    let storage = element_name(element.storage_type());
    let expressed = element_name(element.expressed_type());
    let (min, max) = element.storage_range();
    let range = if min.is_some() || max.is_some() {
        let (default_min, default_max) =
            element.storage_type().integer_bounds().ok_or_else(|| {
                error(TextErrorKind::InvalidType(
                    crate::TypeError::InvalidQuantizedStorageType(element.storage_type()),
                ))
            })?;
        format!(
            "<{}:{}>",
            min.unwrap_or(default_min),
            max.unwrap_or(default_max)
        )
    } else {
        String::new()
    };
    let axis = element
        .quantization_dimension()
        .map_or(String::new(), |axis| format!(":{axis}"));
    let parameters = element
        .scales()
        .zip(element.zero_points())
        .map(|(scale, zero)| format!("{}:{zero}", super::floats::print_f64(scale.to_bits())))
        .collect::<Vec<_>>()
        .join(", ");
    let parameters = if element.quantization_dimension().is_some() {
        format!("{{{parameters}}}")
    } else {
        parameters
    };
    Ok(format!(
        "!quant.uniform<{storage}{range}:{expressed}{axis}, {parameters}>"
    ))
}

fn error(kind: TextErrorKind) -> TextError {
    TextError { offset: 0, kind }
}

fn shape(dimensions: &[Option<u64>]) -> String {
    dimensions
        .iter()
        .map(|size| size.map_or("?x".into(), |size| format!("{size}x")))
        .collect()
}

fn encoding(attribute: Option<&Attribute>, depth: usize) -> Result<String, TextError> {
    attribute.map_or_else(
        || Ok(String::new()),
        |attribute| {
            super::attributes::print(attribute, depth).map(|attribute| format!(", {attribute}"))
        },
    )
}
