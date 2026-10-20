use super::*;
use crate::types::*;
use pliron::{
    attribute::{Attribute, attr_cast},
    builtin::{
        attr_interfaces::FloatAttr,
        attributes::{BoolAttr, StringAttr, UnitAttr},
        type_interfaces::FunctionTypeInterface,
    },
};

/// Serializes a source dictionary as lossless, deterministic MLIR assembly.
///
/// This does not use pliron's `Printable` syntax. Integer signedness, string
/// bytes, and float payloads are preserved. Floats use hexadecimal assembly,
/// not necessarily MLIR's canonical decimal spelling. Unknown native attribute
/// and type representations return a typed `Unsupported` diagnostic; they are
/// never stringified or silently coerced.
///
/// Supported representations include builtin scalars, strings, units, arrays,
/// dictionaries, symbol references, dense/sparse elements, dense arrays,
/// affine maps/sets, layouts, opaque attributes, StableHLO enums and bounds,
/// and primitive, shaped, tuple, function, and quantized type attributes.
///
/// Remaining serialization prerequisites are explicit: sparse tensor
/// encodings and other dialect-specific metadata need their dialect printers;
/// internal limb-only records have no attribute type; standalone integer
/// complex values have no pinned `complex.number` representation. The pinned
/// numeric elements grammar requires primitive element types, so quantized
/// dense/sparse payloads need a separate storage-typed representation. Its
/// double-based complex/subchannel parsers cannot retain non-f64 signaling
/// NaNs. Subchannel scalar/empty parameter shapes and floating storage zero
/// points also need a representation beyond the pinned parser's braced
/// parameter and integer-zero-point grammar.
pub fn mlir_dictionary(ctx: &Context, dictionary: &DictAttr) -> Result<String> {
    dictionary_text(ctx, dictionary)
}

fn invalid(reason: &'static str) -> pliron::result::Error {
    Diagnostic::InvalidAttribute {
        attribute: "MLIR dictionary serialization",
        reason,
    }
    .into()
}

fn unsupported(entity: String, reason: &'static str) -> pliron::result::Error {
    Diagnostic::UnsupportedSerialization { entity, reason }.into()
}

fn quoted(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut text = String::from("\"");
    for &byte in bytes {
        match byte {
            b'\\' => text.push_str("\\\\"),
            0x20..=0x7e if byte != b'"' => text.push(char::from(byte)),
            _ => {
                text.push('\\');
                text.push(char::from(HEX[usize::from(byte >> 4)]));
                text.push(char::from(HEX[usize::from(byte & 15)]));
            }
        }
    }
    text.push('"');
    text
}

fn keyword(text: &str) -> String {
    let mut bytes = text.bytes();
    if bytes
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        && bytes.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'$' | b'.'))
    {
        text.to_owned()
    } else {
        quoted(text.as_bytes())
    }
}

fn dictionary_text(ctx: &Context, dictionary: &DictAttr) -> Result<String> {
    let mut entries: Vec<_> = dictionary
        .0
        .0
        .iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect();
    entries.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    let mut values = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        if key.is_empty() {
            return Err(invalid("empty attribute name"));
        }
        let key = keyword(&key);
        if value.downcast_ref::<UnitAttr>().is_some() {
            values.push(key);
        } else {
            values.push(format!("{key} = {}", attribute_text(ctx, &**value, false)?));
        }
    }
    Ok(format!("{{{}}}", values.join(", ")))
}

fn integer_text(value: String, element: ElementType, elide: bool) -> String {
    if element == ElementType::I1 {
        return if value == "0" { "false" } else { "true" }.into();
    }
    if elide && element == ElementType::I64 {
        value
    } else {
        format!("{value} : {}", primitive_text(element))
    }
}

fn scalar_integer(ctx: &Context, value: &ScalarBitsAttr) -> Option<crate::integer::IntegerValue> {
    let (width, signedness) = primitive_element(ctx, value.ty)?.integer_parts()?;
    Some(crate::integer::IntegerValue::from_twos_complement_words(
        &value.words,
        width,
        signedness != Signedness::Unsigned && (width > 1 || signedness == Signedness::Signed),
    ))
}

fn attribute_text(ctx: &Context, attr: &dyn Attribute, elide: bool) -> Result<String> {
    if let Some(value) = attr.downcast_ref::<DictAttr>() {
        return dictionary_text(ctx, value);
    }
    if let Some(value) = attr.downcast_ref::<VecAttr>() {
        let values = value
            .0
            .iter()
            .map(|value| attribute_text(ctx, &**value, true))
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!("[{}]", values.join(", ")));
    }
    attr.verify(ctx)?;
    if let Some(value) = attr.downcast_ref::<BoolAttr>() {
        return Ok(bool::from(value.clone()).to_string());
    }
    if let Some(value) = attr.downcast_ref::<IntegerAttr>() {
        let ty = value.get_type().deref(ctx);
        ty.verify(ctx)?;
        let element = ElementType::integer(ty.width(), ty.signedness())
            .ok_or_else(|| invalid("integer width exceeds the MLIR limit"))?;
        return Ok(integer_text(
            value.value().to_string_decimal(!ty.is_unsigned()),
            element,
            elide,
        ));
    }
    if let Some(value) = attr.downcast_ref::<ScalarBitsAttr>() {
        let element = primitive_element(ctx, value.ty)
            .ok_or_else(|| invalid("scalar bits need a primitive element"))?;
        if let Some(integer) = scalar_integer(ctx, value) {
            return Ok(integer_text(integer.to_string(), element, elide));
        }
        if element.is_float() {
            return Ok(format!(
                "0x{:X} : {}",
                value.words[0],
                primitive_text(element)
            ));
        }
        if let Some((real, imaginary)) = complex_bits(ctx, value)?
            && element
                .complex_component()
                .is_some_and(ElementType::is_float)
        {
            return complex_text(ctx, &real, &imaginary);
        }
        return Err(unsupported(
            attr.get_attr_id().to_string(),
            "pinned Complex NumberAttr accepts only floating components; integer complex values require an elements attribute",
        ));
    }
    if let Some(value) = attr_cast::<dyn FloatAttr>(attr) {
        let ty = value.get_type(ctx);
        return Ok(format!("0x{:X} : {}", value.to_bits(), type_text(ctx, ty)?));
    }
    if let Some(value) = attr.downcast_ref::<IndexAttr>() {
        return Ok(format!("{} : index", value.value));
    }
    if let Some(value) = attr.downcast_ref::<StringAttr>() {
        return Ok(quoted(value.as_str().as_bytes()));
    }
    if let Some(value) = attr.downcast_ref::<BytesAttr>() {
        return Ok(quoted(&value.value));
    }
    if let Some(value) = attr.downcast_ref::<TypedStringAttr>() {
        let text = quoted(&value.value);
        return if value.ty.deref(ctx).downcast_ref::<NoneType>().is_some() {
            Ok(text)
        } else {
            Ok(format!("{text} : {}", type_text(ctx, value.ty)?))
        };
    }
    if attr.downcast_ref::<UnitAttr>().is_some() {
        return Ok("unit".into());
    }
    if let Some(value) = attr.downcast_ref::<super::TypeAttr>() {
        return type_text(ctx, value.ty);
    }
    if let Some(value) = attr.downcast_ref::<PlironTypeAttr>() {
        return type_text(ctx, value.get_type(ctx));
    }
    if let Some(value) = attr.downcast_ref::<SymbolRefAttr>() {
        if value.path.is_empty() || value.path.iter().any(|part| part.as_str().is_empty()) {
            return Err(invalid(
                "symbol reference must have a nonempty root and components",
            ));
        }
        return Ok(value
            .path
            .iter()
            .map(|part| format!("@{}", keyword(part.as_str())))
            .collect::<Vec<_>>()
            .join("::"));
    }
    if let Some(value) = attr.downcast_ref::<DenseArrayAttr>() {
        let values = value
            .values
            .0
            .iter()
            .map(|value| element_text(ctx, &**value))
            .collect::<Result<Vec<_>>>()?;
        let suffix = if values.is_empty() {
            String::new()
        } else {
            format!(": {}", values.join(", "))
        };
        return Ok(format!("array<{}{suffix}>", type_text(ctx, value.element)?));
    }
    if let Some(value) = attr.downcast_ref::<DenseElementsAttr>() {
        verify_elements_representation(ctx, value.ty)?;
        let shape = shape(ctx, value.ty)?;
        let values = value
            .values
            .0
            .iter()
            .map(|value| element_text(ctx, &**value))
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!(
            "dense<{}> : {}",
            elements_text(&shape, &values, value.is_splat)?,
            type_text(ctx, value.ty)?
        ));
    }
    if let Some(value) = attr.downcast_ref::<SparseElementsAttr>() {
        verify_elements_representation(ctx, value.ty)?;
        let mut indices = Vec::with_capacity(value.indices.len());
        for index in &value.indices {
            indices.push(format!(
                "[{}]",
                index
                    .values
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let body = if indices.is_empty() {
            String::new()
        } else {
            let values = value
                .values
                .0
                .iter()
                .map(|value| element_text(ctx, &**value))
                .collect::<Result<Vec<_>>>()?;
            format!("[{}], [{}]", indices.join(", "), values.join(", "))
        };
        return Ok(format!("sparse<{body}> : {}", type_text(ctx, value.ty)?));
    }
    if let Some(value) = attr.downcast_ref::<StridedLayoutAttr>() {
        let strides = value
            .strides
            .iter()
            .map(|value| dynamic(*value))
            .collect::<Vec<_>>();
        let offset = value
            .offset
            .first()
            .map_or_else(|| "?".into(), |value| dynamic(*value));
        return Ok(format!(
            "strided<[{}], offset: {offset}>",
            strides.join(", ")
        ));
    }
    if let Some(value) = attr.downcast_ref::<AffineMap>() {
        return affine_map_text(value);
    }
    if let Some(value) = attr.downcast_ref::<IntegerSet>() {
        let constraints = value
            .constraints
            .iter()
            .map(|constraint| {
                Ok(format!(
                    "{} {} 0",
                    affine_expression(&constraint.expression)?,
                    if constraint.equality { "==" } else { ">=" }
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!(
            "affine_set<{} : ({})>",
            affine_arguments(value.dimensions, value.symbols),
            constraints.join(", ")
        ));
    }
    if let Some(value) = attr.downcast_ref::<OpaqueAttr>() {
        return dialect_attribute(value.dialect.as_str(), value.syntax.as_str());
    }
    if let Some(value) = attr.downcast_ref::<EnumAttr>() {
        return dialect_attribute(
            value.dialect.as_str(),
            &format!("{} {}", value.name.as_str(), value.value.as_str()),
        );
    }
    macro_rules! enumeration {
        ($ty:ty, $name:literal) => {
            if let Some(value) = attr.downcast_ref::<$ty>() {
                let spelling = enums::values($name)
                    .and_then(|values| values.get(usize::from(value.value)))
                    .ok_or_else(|| invalid("unknown StableHLO enumeration"))?;
                return Ok(format!("#stablehlo<{} {spelling}>", $name));
            }
        };
    }
    enumeration!(PrecisionAttr, "precision");
    enumeration!(ComparisonDirectionAttr, "comparison_direction");
    enumeration!(ComparisonTypeAttr, "comparison_type");
    enumeration!(FftTypeAttr, "fft_type");
    enumeration!(RngAlgorithmAttr, "rng_algorithm");
    enumeration!(RngDistributionAttr, "rng_distribution");
    enumeration!(ResultAccuracyModeAttr, "result_accuracy_mode");
    if let Some(value) = attr.downcast_ref::<TypeExtensionsAttr>() {
        let bounds = value
            .bounds
            .iter()
            .map(|bound| {
                if *bound == -1 {
                    "?".into()
                } else {
                    bound.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(format!("#stablehlo.type_extensions<bounds = [{bounds}]>"));
    }
    if let Some(value) = attr.downcast_ref::<ComplexAttr>() {
        return complex_text(ctx, &value.real, &value.imaginary);
    }
    if attr.downcast_ref::<Signed128Attr>().is_some() {
        return Err(unsupported(
            attr.get_attr_id().to_string(),
            "this internal limb record does not carry an MLIR integer attribute type",
        ));
    }
    Err(unsupported(
        attr.get_attr_id().to_string(),
        "this native attribute needs an explicit MLIR representation",
    ))
}

fn complex_text(
    ctx: &Context,
    real: &ScalarBitsAttr,
    imaginary: &ScalarBitsAttr,
) -> Result<String> {
    // NumberAttr's custom parser reads doubles, then converts to the
    // declared component type. Its literals are NOT component-width bits.
    Ok(format!(
        "#complex.number<:{} 0x{:X}, 0x{:X}>",
        type_text(ctx, real.ty)?,
        double_bits(ctx, real)?,
        double_bits(ctx, imaginary)?
    ))
}

fn complex_bits(
    ctx: &Context,
    value: &ScalarBitsAttr,
) -> Result<Option<(ScalarBitsAttr, ScalarBitsAttr)>> {
    let Some(component) = primitive_element(ctx, value.ty).and_then(ElementType::complex_component)
    else {
        return Ok(None);
    };
    let width = component
        .bit_width()
        .ok_or_else(|| invalid("complex component has no bit width"))?;
    let extract = |offset: u32| {
        let mut words = Vec::with_capacity(width.div_ceil(64) as usize);
        for bit in (0..width).step_by(64) {
            let start = offset + bit;
            let word = (start / 64) as usize;
            let shift = start % 64;
            let mut payload = value.words[word] >> shift;
            if shift != 0
                && let Some(next) = value.words.get(word + 1)
            {
                payload |= next << (64 - shift);
            }
            let remaining = width - bit;
            if remaining < 64 {
                payload &= (1_u64 << remaining) - 1;
            }
            words.push(payload);
        }
        ScalarBitsAttr {
            ty: component.native(ctx),
            words,
        }
    };
    Ok(Some((extract(0), extract(width))))
}

#[derive(Clone, Copy)]
enum NonFinite {
    Ieee,
    AllOnes,
    NegativeZero,
    Finite,
}

fn double_bits(ctx: &Context, value: &ScalarBitsAttr) -> Result<u64> {
    use ElementType::*;
    let element =
        primitive_element(ctx, value.ty).ok_or_else(|| invalid("expected a floating component"))?;
    let bits = value.words[0];
    if element == F64 {
        return Ok(bits);
    }
    if element == F8E8M0FnUz {
        return Ok(if bits == 255 {
            f64::NAN.to_bits()
        } else {
            2_f64.powi(bits as i32 - 127).to_bits()
        });
    }
    // Pinned LLVM APFloat.cpp fltSemantics, with explicit fraction widths.
    let (exponent_bits, fraction_bits, bias, nonfinite) = match element {
        F4E2M1Fn => (2, 1, 1, NonFinite::Finite),
        F6E2M3Fn => (2, 3, 1, NonFinite::Finite),
        F6E3M2Fn => (3, 2, 3, NonFinite::Finite),
        F8E3M4 => (3, 4, 3, NonFinite::Ieee),
        F8E4M3 => (4, 3, 7, NonFinite::Ieee),
        F8E4M3Fn => (4, 3, 7, NonFinite::AllOnes),
        F8E4M3FnUz => (4, 3, 8, NonFinite::NegativeZero),
        F8E4M3B11FnUz => (4, 3, 11, NonFinite::NegativeZero),
        F8E5M2 => (5, 2, 15, NonFinite::Ieee),
        F8E5M2FnUz => (5, 2, 16, NonFinite::NegativeZero),
        F16 => (5, 10, 15, NonFinite::Ieee),
        Bf16 => (8, 7, 127, NonFinite::Ieee),
        F32 => (8, 23, 127, NonFinite::Ieee),
        Tf32 => (8, 10, 127, NonFinite::Ieee),
        _ => return Err(invalid("expected a floating component")),
    };
    let sign = bits >> (exponent_bits + fraction_bits);
    let fraction_mask = (1_u64 << fraction_bits) - 1;
    let fraction = bits & fraction_mask;
    let exponent_mask = (1_u64 << exponent_bits) - 1;
    let exponent = (bits >> fraction_bits) & exponent_mask;
    let nan = match nonfinite {
        NonFinite::Ieee => exponent == exponent_mask && fraction != 0,
        NonFinite::AllOnes => exponent == exponent_mask && fraction == fraction_mask,
        NonFinite::NegativeZero => bits == 1 << (exponent_bits + fraction_bits),
        NonFinite::Finite => false,
    };
    if nan {
        if matches!(nonfinite, NonFinite::Ieee) {
            if fraction & (1 << (fraction_bits - 1)) == 0 {
                return Err(unsupported(
                    "floating component parsed through double".into(),
                    "pinned dialect parsers use double and APFloat::convert quiets non-f64 signaling NaNs",
                ));
            }
            return Ok((sign << 63) | (0x7ff_u64 << 52) | (fraction << (52 - fraction_bits)));
        }
        let sign = if matches!(nonfinite, NonFinite::NegativeZero) {
            0
        } else {
            sign
        };
        return Ok((sign << 63) | (0x7ff_u64 << 52) | (1 << 51));
    }
    if matches!(nonfinite, NonFinite::Ieee) && exponent == exponent_mask {
        return Ok((sign << 63) | (0x7ff_u64 << 52));
    }
    let significand = if exponent == 0 {
        fraction
    } else {
        (1 << fraction_bits) | fraction
    };
    let power = exponent.max(1) as i32 - bias - fraction_bits;
    let magnitude = (significand as f64) * 2_f64.powi(power);
    Ok((if sign == 0 { magnitude } else { -magnitude }).to_bits())
}

fn dialect_attribute(dialect: &str, syntax: &str) -> Result<String> {
    let mut name = dialect.bytes();
    if !name
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
        || !name.all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'$'))
    {
        return Err(invalid("invalid dialect namespace"));
    }
    // DialectSymbolParser.cpp scans balanced punctuation, strings, and `->`.
    // Validate the wrapper so opaque data cannot terminate its containing attr.
    let mut brackets = Vec::new();
    let mut top_level_groups = 0;
    let bytes = syntax.as_bytes();
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            0 => return Err(invalid("NUL in opaque attribute syntax")),
            b'<' | b'(' | b'[' | b'{' => {
                if brackets.is_empty() {
                    top_level_groups += 1;
                }
                brackets.push(byte);
            }
            b'>' | b')' | b']' | b'}' => {
                let expected = match byte {
                    b'>' => b'<',
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                if brackets.pop() != Some(expected) {
                    return Err(invalid("unbalanced opaque attribute syntax"));
                }
            }
            b'-' if bytes.get(index + 1) == Some(&b'>') => index += 1,
            b'"' => {
                index += 1;
                loop {
                    match bytes.get(index) {
                        None | Some(b'\n' | 0x0b | 0x0c) => {
                            return Err(invalid("unterminated opaque attribute string"));
                        }
                        Some(b'"') => break,
                        Some(b'\\') => {
                            index += 1;
                            match bytes.get(index) {
                                Some(b'"' | b'\\' | b'n' | b't') => (),
                                Some(c)
                                    if c.is_ascii_hexdigit()
                                        && bytes
                                            .get(index + 1)
                                            .is_some_and(u8::is_ascii_hexdigit) =>
                                {
                                    index += 1
                                }
                                _ => return Err(invalid("invalid opaque attribute string escape")),
                            }
                        }
                        _ => (),
                    }
                    index += 1;
                }
            }
            _ => (),
        }
        index += 1;
    }
    if !brackets.is_empty() {
        return Err(invalid("unbalanced opaque attribute syntax"));
    }
    let prefix = syntax
        .bytes()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_'))
        .count();
    let pretty = bytes.first().is_some_and(u8::is_ascii_alphabetic)
        && (prefix == bytes.len()
            || bytes.get(prefix) == Some(&b'<')
                && bytes.last() == Some(&b'>')
                && top_level_groups == 1);
    Ok(if pretty {
        format!("#{dialect}.{syntax}")
    } else {
        format!("#{dialect}<{syntax}>")
    })
}

fn dynamic(value: i64) -> String {
    if value == i64::MIN {
        "?".into()
    } else {
        value.to_string()
    }
}

fn shape(ctx: &Context, ty: TypeHandle) -> Result<Vec<i64>> {
    let ty = ty.deref(ctx);
    if let Some(tensor) = ty.downcast_ref::<TensorType>() {
        return Ok(tensor.dimensions.clone());
    }
    if let Some(vector) = ty.downcast_ref::<VectorType>() {
        return Ok(vector.dimensions.clone());
    }
    Err(invalid("elements require a ranked tensor or vector type"))
}

fn verify_elements_representation(ctx: &Context, ty: TypeHandle) -> Result<()> {
    let ty = ty.deref(ctx);
    let element = if let Some(tensor) = ty.downcast_ref::<TensorType>() {
        tensor.element
    } else if let Some(vector) = ty.downcast_ref::<VectorType>() {
        vector.element
    } else {
        return Err(invalid("elements require a ranked tensor or vector type"));
    };
    if primitive_element(ctx, element).is_some() {
        return Ok(());
    }
    Err(unsupported(
        element.deref(ctx).get_type_id().to_string(),
        "pinned numeric elements grammar accepts only integer, index, float or complex elements; nonprimitive storage needs an explicit MLIR attribute type",
    ))
}

fn element_text(ctx: &Context, value: &dyn Attribute) -> Result<String> {
    if let Some(value) = value.downcast_ref::<ComplexAttr>() {
        return Ok(format!(
            "({},{})",
            element_text(ctx, &value.real)?,
            element_text(ctx, &value.imaginary)?
        ));
    }
    if let Some(value) = value.downcast_ref::<ScalarBitsAttr>() {
        if let Some((real, imaginary)) = complex_bits(ctx, value)? {
            return Ok(format!(
                "({},{})",
                element_text(ctx, &real)?,
                element_text(ctx, &imaginary)?
            ));
        }
        if let Some(integer) = scalar_integer(ctx, value) {
            if primitive_element(ctx, value.ty)
                .and_then(ElementType::integer_parts)
                .is_some_and(|(width, _)| width == 1)
            {
                return Ok(if value.words[0] == 0 { "false" } else { "true" }.into());
            }
            return Ok(integer.to_string());
        }
        if primitive_element(ctx, value.ty).is_some_and(ElementType::is_float) {
            return Ok(format!("0x{:X}", value.words[0]));
        }
    }
    if let Some(value) = value.downcast_ref::<IntegerAttr>() {
        let ty = value.get_type().deref(ctx);
        if ty.width() == 1 {
            return Ok(if value.value().to_string_decimal(false) == "0" {
                "false"
            } else {
                "true"
            }
            .into());
        }
        return Ok(value.value().to_string_decimal(!ty.is_unsigned()));
    }
    if let Some(value) = value.downcast_ref::<IndexAttr>() {
        return Ok(value.value.to_string());
    }
    if let Some(value) = value.downcast_ref::<BoolAttr>() {
        return Ok(bool::from(value.clone()).to_string());
    }
    if let Some(value) = value.downcast_ref::<StringAttr>() {
        return Ok(quoted(value.as_str().as_bytes()));
    }
    Err(unsupported(
        value.get_attr_id().to_string(),
        "dense elements need a primitive literal or a complex pair",
    ))
}

fn elements_text(shape: &[i64], values: &[String], splat: bool) -> Result<String> {
    if splat || shape.is_empty() {
        return values
            .first()
            .cloned()
            .ok_or_else(|| invalid("missing scalar element"));
    }
    if values.is_empty() {
        return Ok(String::new());
    }
    let mut counter = vec![0_i64; shape.len()];
    let mut open = 0;
    let mut text = String::new();
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            text.push_str(", ");
        }
        while open < shape.len() {
            text.push('[');
            open += 1;
        }
        text.push_str(value);
        let last = shape.len() - 1;
        counter[last] += 1;
        for axis in (1..shape.len()).rev() {
            if counter[axis] >= shape[axis] {
                counter[axis] = 0;
                counter[axis - 1] += 1;
                text.push(']');
                open -= 1;
            }
        }
    }
    text.extend(std::iter::repeat_n(']', open));
    Ok(text)
}

fn primitive_text(element: ElementType) -> String {
    if let Some((width, signedness)) = element.integer_parts() {
        return format!(
            "{}i{width}",
            match signedness {
                Signedness::Signed => "s",
                Signedness::Unsigned => "u",
                Signedness::Signless => "",
            }
        );
    }
    if let Some(component) = element.complex_component() {
        return format!("complex<{}>", primitive_text(component));
    }
    match element {
        ElementType::Index => "index",
        ElementType::F4E2M1Fn => "f4E2M1FN",
        ElementType::F6E2M3Fn => "f6E2M3FN",
        ElementType::F6E3M2Fn => "f6E3M2FN",
        ElementType::F16 => "f16",
        ElementType::Bf16 => "bf16",
        ElementType::F32 => "f32",
        ElementType::F64 => "f64",
        ElementType::Tf32 => "tf32",
        ElementType::F8E3M4 => "f8E3M4",
        ElementType::F8E4M3 => "f8E4M3",
        ElementType::F8E4M3Fn => "f8E4M3FN",
        ElementType::F8E4M3FnUz => "f8E4M3FNUZ",
        ElementType::F8E4M3B11FnUz => "f8E4M3B11FNUZ",
        ElementType::F8E5M2 => "f8E5M2",
        ElementType::F8E5M2FnUz => "f8E5M2FNUZ",
        ElementType::F8E8M0FnUz => "f8E8M0FNU",
        _ => unreachable!("integer and complex elements handled above"),
    }
    .into()
}

fn type_text(ctx: &Context, handle: TypeHandle) -> Result<String> {
    let ty = handle.deref(ctx);
    ty.verify(ctx)?;
    if let Some(element) = primitive_element(ctx, handle) {
        return Ok(primitive_text(element));
    }
    if let Some(value) = ty.downcast_ref::<TensorType>() {
        let mut text = format!(
            "tensor<{}{}",
            dimensions(&value.dimensions),
            type_text(ctx, value.element)?
        );
        if let Some(encoding) = value.encoding() {
            text.push_str(", ");
            text.push_str(&attribute_text(ctx, &**encoding, false)?);
        }
        text.push('>');
        return Ok(text);
    }
    if let Some(value) = ty.downcast_ref::<UnrankedTensorType>() {
        return Ok(format!("tensor<*x{}>", type_text(ctx, value.element)?));
    }
    if let Some(value) = ty.downcast_ref::<VectorType>() {
        let mut text = String::from("vector<");
        for (&dim, &scalable) in value.dimensions.iter().zip(&value.scalable) {
            text.push_str(&if scalable {
                format!("[{dim}]x")
            } else {
                format!("{dim}x")
            });
        }
        text.push_str(&type_text(ctx, value.element)?);
        text.push('>');
        return Ok(text);
    }
    if let Some(value) = ty.downcast_ref::<TupleType>() {
        return Ok(format!("tuple<{}>", type_list(ctx, &value.elements)?));
    }
    if let Some(value) = ty.downcast_ref::<FunctionType>() {
        return function_text(ctx, &value.inputs, &value.outputs);
    }
    if let Some(value) = ty.downcast_ref::<pliron::builtin::types::FunctionType>() {
        return function_text(ctx, &value.arg_types(), &value.res_types());
    }
    if ty.downcast_ref::<NoneType>().is_some() {
        return Ok("none".into());
    }
    if ty.downcast_ref::<TokenType>().is_some() {
        return Ok("!stablehlo.token".into());
    }
    if let Some(value) = ty.downcast_ref::<BufferType>() {
        let mut text = format!(
            "memref<{}{}",
            dimensions(&value.dimensions),
            type_text(ctx, value.element)?
        );
        for attribute in value.layout().into_iter().chain(value.memory_space()) {
            text.push_str(", ");
            text.push_str(&attribute_text(ctx, &**attribute, true)?);
        }
        text.push('>');
        return Ok(text);
    }
    if let Some(value) = ty.downcast_ref::<UnrankedBufferType>() {
        let mut text = format!("memref<*x{}", type_text(ctx, value.element)?);
        if let Some(space) = value.memory_space() {
            text.push_str(", ");
            text.push_str(&attribute_text(ctx, &**space, true)?);
        }
        text.push('>');
        return Ok(text);
    }
    if let Some(value) = ty.downcast_ref::<QuantizedElementType>() {
        let storage = quantized_storage(ctx, value.storage, value.storage_range())?;
        let expressed = type_text(ctx, value.expressed)?;
        let parameters = value
            .scale_bits
            .iter()
            .zip(&value.zero_points)
            .map(|(scale, zero)| quantized_parameter(*scale, zero.value()))
            .collect::<Vec<_>>();
        return Ok(if value.axis == -1 {
            format!("!quant.uniform<{storage}:{expressed}, {}>", parameters[0])
        } else {
            format!(
                "!quant.uniform<{storage}:{expressed}:{}, {{{}}}>",
                value.axis,
                parameters.join(", ")
            )
        });
    }
    if let Some(value) = ty.downcast_ref::<AnyQuantizedType>() {
        let storage = quantized_storage(ctx, value.storage, value.storage_range())?;
        let expressed = value
            .expressed
            .first()
            .map(|&ty| type_text(ctx, ty))
            .transpose()?
            .map_or_else(String::new, |text| format!(":{text}"));
        return Ok(format!("!quant.any<{storage}{expressed}>"));
    }
    if let Some(value) = ty.downcast_ref::<CalibratedQuantizedType>() {
        return Ok(format!(
            "!quant.calibrated<{}<0x{:X}:0x{:X}>>",
            type_text(ctx, value.expressed)?,
            value.minimum_bits,
            value.maximum_bits
        ));
    }
    if let Some(value) = ty.downcast_ref::<QuantileType>() {
        let values = value
            .quantile_bits
            .iter()
            .map(|bits| format!("0x{bits:X}"))
            .collect::<Vec<_>>()
            .join(", ");
        let range = value
            .storage_range()
            .map_or_else(String::new, |(min, max)| format!(", <{min}:{max}>"));
        return Ok(format!(
            "!quant.quantile<{}:{}, {{{values}}}{range}>",
            type_text(ctx, value.storage)?,
            type_text(ctx, value.quantile)?
        ));
    }
    if let Some(value) = ty.downcast_ref::<SubchannelQuantizedType>() {
        if value.parameter_shape.is_empty() || value.parameter_shape.contains(&0) {
            return Err(unsupported(
                ty.get_type_id().to_string(),
                "pinned subchannel parser requires nonempty braced parameter dimensions",
            ));
        }
        let storage_metadata = value.storage.deref(ctx);
        let storage_metadata = storage_metadata
            .downcast_ref::<AnyQuantizedType>()
            .ok_or_else(|| invalid("missing subchannel storage metadata"))?;
        let storage = quantized_storage(
            ctx,
            storage_metadata.storage,
            storage_metadata.storage_range(),
        )?;
        let element = storage_metadata
            .storage_type(ctx)
            .ok_or_else(|| invalid("missing subchannel storage element"))?;
        let (width, signedness) = element.integer_parts().ok_or_else(|| unsupported(
            ty.get_type_id().to_string(),
            "pinned subchannel parser constructs zero points from APInt; floating storage zero-point encoding needs a dedicated representation"))?;
        let count = crate::types::element_count(&value.parameter_shape)
            .ok_or_else(|| invalid("invalid subchannel parameter shape"))?;
        let mut parameters = Vec::with_capacity(count);
        for index in 0..count {
            let scale = value.scale_bits[if value.scale_bits.len() == 1 {
                0
            } else {
                index
            }];
            let scale = double_bits(
                ctx,
                &ScalarBitsAttr {
                    ty: value.expressed,
                    words: vec![scale],
                },
            )?;
            let zero = value.zero_point_bits[if value.zero_point_bits.len() == 1 {
                0
            } else {
                index
            }];
            let zero = crate::integer::IntegerValue::from_twos_complement_words(
                &[zero],
                width,
                signedness != Signedness::Unsigned,
            );
            parameters.push(if zero == crate::integer::IntegerValue::ZERO {
                format!("0x{scale:X}")
            } else {
                format!("0x{scale:X}:{zero}")
            });
        }
        let axes = value
            .quantized_dimensions
            .iter()
            .zip(&value.block_sizes)
            .map(|(axis, block)| format!("{axis}:{block}"))
            .collect::<Vec<_>>()
            .join(", ");
        let parameters = elements_text(&value.parameter_shape, &parameters, false)?
            .replace('[', "{")
            .replace(']', "}");
        return Ok(format!(
            "!quant.uniform<{storage}:{}:{{{axes}}}, {parameters}>",
            type_text(ctx, value.expressed)?
        ));
    }
    Err(unsupported(
        ty.get_type_id().to_string(),
        "this native type needs an explicit MLIR representation",
    ))
}

fn quantized_storage(
    ctx: &Context,
    storage: TypeHandle,
    range: (Option<i128>, Option<i128>),
) -> Result<String> {
    let element = primitive_element(ctx, storage)
        .ok_or_else(|| invalid("quantized storage must be primitive"))?;
    let mut text = if let Some((width, signedness)) = element.integer_parts() {
        if signedness == Signedness::Unsigned {
            format!("u{width}")
        } else {
            format!("i{width}")
        }
    } else {
        type_text(ctx, storage)?
    };
    if range.0.is_some() || range.1.is_some() {
        let defaults = match element {
            ElementType::I1 => Some((-1, 0)),
            ElementType::F4E2M1Fn => Some((-6, 6)),
            ElementType::F8E4M3Fn => Some((-448, 448)),
            ElementType::F8E5M2 => Some((-57344, 57344)),
            _ => element.integer_bounds(),
        }
        .ok_or_else(|| invalid("quantized storage has no numeric bounds"))?;
        text.push_str(&format!(
            "<{}:{}>",
            range.0.unwrap_or(defaults.0),
            range.1.unwrap_or(defaults.1)
        ));
    }
    Ok(text)
}

fn quantized_parameter(scale: u64, zero: i128) -> String {
    if zero == 0 {
        format!("0x{scale:X}")
    } else {
        format!("0x{scale:X}:{zero}")
    }
}

fn dimensions(shape: &[i64]) -> String {
    shape
        .iter()
        .map(|dim| {
            if *dim == -1 {
                "?x".into()
            } else {
                format!("{dim}x")
            }
        })
        .collect()
}

fn type_list(ctx: &Context, values: &[TypeHandle]) -> Result<String> {
    Ok(values
        .iter()
        .map(|&value| type_text(ctx, value))
        .collect::<Result<Vec<_>>>()?
        .join(", "))
}

fn function_text(ctx: &Context, inputs: &[TypeHandle], outputs: &[TypeHandle]) -> Result<String> {
    let results = type_list(ctx, outputs)?;
    let single = outputs.first().is_some_and(|ty| {
        let ty = ty.deref(ctx);
        ty.downcast_ref::<FunctionType>().is_none()
            && ty
                .downcast_ref::<pliron::builtin::types::FunctionType>()
                .is_none()
    });
    Ok(format!(
        "({}) -> {}",
        type_list(ctx, inputs)?,
        if outputs.len() == 1 && single {
            results
        } else {
            format!("({results})")
        }
    ))
}

fn affine_arguments(dimensions: usize, symbols: usize) -> String {
    let dims = (0..dimensions)
        .map(|dim| format!("d{dim}"))
        .collect::<Vec<_>>()
        .join(", ");
    if symbols == 0 {
        return format!("({dims})");
    }
    let symbols = (0..symbols)
        .map(|symbol| format!("s{symbol}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("({dims})[{symbols}]")
}

fn affine_expression(expression: &AffineExpr) -> Result<String> {
    let mut stack = Vec::<String>::new();
    for node in &expression.nodes {
        match node.kind {
            0 => stack.push(format!("d{}", node.value)),
            1 => stack.push(format!("s{}", node.value)),
            2 => stack.push(node.value.to_string()),
            3..=7 => {
                let rhs = stack
                    .pop()
                    .ok_or_else(|| invalid("malformed affine expression"))?;
                let lhs = stack
                    .pop()
                    .ok_or_else(|| invalid("malformed affine expression"))?;
                let op = match node.kind {
                    3 => "+",
                    4 => "*",
                    5 => "floordiv",
                    6 => "ceildiv",
                    _ => "mod",
                };
                stack.push(format!("({lhs} {op} {rhs})"));
            }
            _ => return Err(invalid("malformed affine node")),
        }
    }
    if stack.len() != 1 {
        return Err(invalid("malformed affine expression"));
    }
    stack
        .pop()
        .ok_or_else(|| invalid("missing affine expression"))
}

fn affine_map_text(value: &AffineMap) -> Result<String> {
    let results = value
        .results
        .iter()
        .map(affine_expression)
        .collect::<Result<Vec<_>>>()?;
    Ok(format!(
        "affine_map<{} -> ({})>",
        affine_arguments(value.dimensions, value.symbols),
        results.join(", ")
    ))
}

#[cfg(test)]
#[path = "mlir_tests.rs"]
mod tests;
