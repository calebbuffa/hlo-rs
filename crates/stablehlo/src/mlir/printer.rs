use crate::pliron;
use crate::{Diagnostic, attributes::*, types::*};
use pliron::attribute::Attribute;
use pliron::{builtin::types::Signedness, context::Context, result::Result, r#type::TypeHandle};

pub use pliron_mlir_builtin::mlir::{
    print_attribute as mlir_attribute, print_dictionary as mlir_dictionary, print_type as mlir_type,
};

fn attribute_text(ctx: &Context, attr: &dyn Attribute, _elide: bool) -> Result<String> {
    mlir_attribute(ctx, attr)
}
fn type_text(ctx: &Context, ty: TypeHandle) -> Result<String> {
    mlir_type(ctx, ty)
}
fn invalid(reason: &'static str) -> pliron::result::Error {
    Diagnostic::InvalidAttribute {
        attribute: "MLIR dialect serialization",
        reason,
    }
    .into()
}
fn unsupported(entity: String, reason: &'static str) -> pliron::result::Error {
    Diagnostic::UnsupportedSerialization { entity, reason }.into()
}
pub(super) fn dialect_attribute_text(ctx: &Context, attr: &dyn Attribute) -> Result<String> {
    attr.verify(ctx)?;
    if let Some(value) = attr.downcast_ref::<EnumAttr>() {
        if value.dialect.as_str() == "stablehlo" && value.name.as_str() == ResultAccuracyMode::NAME
        {
            return Ok(format!(
                "#stablehlo.result_accuracy_mode<{}>",
                value.value.as_str()
            ));
        }
        return mlir_attribute(
            ctx,
            &OpaqueAttr {
                dialect: value.dialect.clone(),
                syntax: pliron::builtin::attributes::StringAttr::from(format!(
                    "{} {}",
                    value.name.as_str(),
                    value.value.as_str()
                )),
            },
        );
    }
    macro_rules! enumeration {
        ($ty:ty) => {
            if let Some(value) = attr.downcast_ref::<$ty>() {
                return Ok(format!(
                    "#stablehlo<{} {}>",
                    <$ty>::NAME,
                    <$ty>::as_str(*value)
                ));
            }
        };
    }
    enumeration!(Precision);
    enumeration!(ComparisonDirection);
    enumeration!(ComparisonKind);
    if let Some(value) = attr.downcast_ref::<FftKind>() {
        return Ok(format!("#stablehlo<fft_type {}>", FftKind::as_str(*value)));
    }
    enumeration!(RngAlgorithmAttr);
    enumeration!(RngDistributionAttr);
    if let Some(value) = attr.downcast_ref::<ResultAccuracyMode>() {
        return Ok(format!(
            "#stablehlo.result_accuracy_mode<{}>",
            ResultAccuracyMode::as_str(*value)
        ));
    }
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
    if let Some(value) = attr.downcast_ref::<IndexTensorAttr>() {
        return Ok(index_tensor_text(&value.values));
    }
    if let Some(value) = attr.downcast_ref::<LayoutAttr>() {
        return Ok(index_tensor_text(&value.minor_to_major));
    }
    if let Some(value) = attr.downcast_ref::<ArrayOfLayoutAttr>() {
        return Ok(format!(
            "[{}]",
            value
                .layouts
                .iter()
                .map(|layout| index_tensor_text(&layout.minor_to_major))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(value) = attr.downcast_ref::<TilingAttr>() {
        return Ok(format!(
            "[{}]",
            value
                .tiles
                .iter()
                .map(|tile| index_tensor_text(&tile.values))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(value) = attr.downcast_ref::<ArrayOfTilingAttr>() {
        let values = value
            .tilings
            .iter()
            .map(|tiling| attribute_text(ctx, tiling, false))
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!("[{}]", values.join(", ")));
    }
    if let Some(value) = attr.downcast_ref::<SubAxisInfoAttr>() {
        return Ok(format!(
            "#stablehlo<sub_axis_info({}){}>",
            value.pre_size, value.size
        ));
    }
    if let Some(value) = attr.downcast_ref::<AxisRefAttr>() {
        let sub = value
            .sub_axis_info
            .first()
            .map(|sub| attribute_text(ctx, sub, false))
            .transpose()?
            .map_or_else(String::new, |sub| format!(", sub_axis_info = {sub}"));
        return Ok(format!(
            "#stablehlo.axis_ref<name = {}{sub}>",
            quoted(value.name.as_str().as_bytes())
        ));
    }
    if let Some(value) = attr.downcast_ref::<MeshAxisAttr>() {
        return Ok(format!(
            "#stablehlo.mesh_axis<name = {}, size = {}>",
            quoted(value.name.as_str().as_bytes()),
            value.size
        ));
    }
    if let Some(value) = attr.downcast_ref::<MeshAttr>() {
        let axes = value
            .axes
            .iter()
            .map(|axis| attribute_text(ctx, axis, false))
            .collect::<Result<Vec<_>>>()?;
        let devices = value
            .device_ids
            .first()
            .filter(|ids| !ids.values.is_empty())
            .map(|ids| -> Result<String> {
                let shape = value.axes.iter().map(|axis| axis.size).collect::<Vec<_>>();
                let values = ids.values.iter().map(i64::to_string).collect::<Vec<_>>();
                Ok(format!(
                    ", device_ids = dense<{}> : tensor<{}i64>",
                    elements_text(&shape, &values, false)?,
                    dimensions(&shape)
                ))
            })
            .transpose()?
            .unwrap_or_default();
        return Ok(format!(
            "#stablehlo.mesh<axes = [{}]{devices}>",
            axes.join(", ")
        ));
    }
    if let Some(value) = attr.downcast_ref::<MeshReferenceAttr>() {
        return if let Some(mesh) = value.inline.first() {
            attribute_text(ctx, mesh, false)
        } else if let Some(symbol) = value.symbol.first() {
            Ok(format!("@{}", keyword(symbol.as_str())))
        } else {
            Err(invalid("missing mesh reference"))
        };
    }
    if let Some(value) = attr.downcast_ref::<ReplicaGroupMeshAxesAttr>() {
        let axes = value
            .axes
            .iter()
            .map(|axis| attribute_text(ctx, axis, false))
            .collect::<Result<Vec<_>>>()?;
        return Ok(format!(
            "#stablehlo.replica_group_mesh_axes<mesh = {}, axes = [{}]>",
            attribute_text(ctx, &value.mesh, false)?,
            axes.join(", ")
        ));
    }
    if let Some(value) = attr.downcast_ref::<crate::sparse::SparseTensorEncoding>() {
        return sparse_encoding_text(ctx, value);
    }
    if let Some(text) = super::dialect_attributes::print(ctx, attr)? {
        return Ok(text);
    }
    Err(unsupported(
        attr.get_attr_id().to_string(),
        "missing dialect attribute representation",
    ))
}
pub(super) fn dialect_type_text(ctx: &Context, handle: TypeHandle) -> Result<String> {
    let ty = handle.deref(ctx);
    ty.verify(ctx)?;
    if let Some(value) = ty.downcast_ref::<FutureType>() {
        return Ok(format!(
            "!stablehlo.future<{}>",
            type_list(ctx, &value.elements)?
        ));
    }
    if ty.downcast_ref::<StringType>().is_some() {
        return Ok("!tf_type.string".into());
    }
    if ty.downcast_ref::<WitnessType>().is_some() {
        return Ok("!shape.witness".into());
    }
    if ty.downcast_ref::<TokenType>().is_some() {
        return Ok("!stablehlo.token".into());
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
        "missing dialect type representation",
    ))
}
pub(super) fn quoted(bytes: &[u8]) -> String {
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

fn index_tensor_text(values: &[i64]) -> String {
    let values_text = values
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("dense<[{values_text}]> : tensor<{}xindex>", values.len())
}

fn sparse_encoding_text(
    ctx: &Context,
    value: &crate::sparse::SparseTensorEncoding,
) -> Result<String> {
    use crate::sparse::SparseLevelFormat;
    let mut results = Vec::new();
    for (expression, level) in value.dimension_to_level.results.iter().zip(&value.levels) {
        let format = match level
            .level_format()
            .ok_or_else(|| invalid("invalid sparse level"))?
        {
            SparseLevelFormat::Dense => "dense".into(),
            SparseLevelFormat::Batch => "batch".into(),
            SparseLevelFormat::Compressed => "compressed".into(),
            SparseLevelFormat::LooseCompressed => "loose_compressed".into(),
            SparseLevelFormat::Singleton => "singleton".into(),
            SparseLevelFormat::Structured { n, m } => format!("structured[{n}, {m}]"),
        };
        let mut properties = Vec::new();
        if level.nonunique {
            properties.push("nonunique");
        }
        if level.nonordered {
            properties.push("nonordered");
        }
        if level.soa {
            properties.push("soa");
        }
        let properties = if properties.is_empty() {
            String::new()
        } else {
            format!("({})", properties.join(", "))
        };
        results.push(format!(
            "{} : {format}{properties}",
            affine_expression(expression)?
        ));
    }
    let dimensions = (0..value.dimension_to_level.dimensions)
        .map(|dimension| {
            let Some(slice) = value.dimension_slices.get(dimension) else {
                return format!("d{dimension}");
            };
            let field = |value: &[i64]| value.first().map_or_else(|| "?".into(), i64::to_string);
            format!(
                "d{dimension} : #sparse_tensor<slice({}, {}, {})>",
                field(&slice.offset),
                field(&slice.size),
                field(&slice.stride)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let symbols = if value.dimension_to_level.symbols == 0 {
        String::new()
    } else {
        format!(
            "[{}]",
            (0..value.dimension_to_level.symbols)
                .map(|symbol| format!("s{symbol}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut fields = vec![
        format!("map = {symbols}({dimensions}) -> ({})", results.join(", ")),
        format!("posWidth = {}", value.position_width),
        format!("crdWidth = {}", value.coordinate_width),
    ];
    for (name, values) in [
        ("explicitVal", &value.explicit_value),
        ("implicitVal", &value.implicit_value),
    ] {
        if let Some(scalar) = values.0.first() {
            fields.push(format!(
                "{name} = {}",
                attribute_text(ctx, &**scalar, false)?
            ));
        }
    }
    Ok(format!(
        "#sparse_tensor.encoding<{{ {} }}>",
        fields.join(", ")
    ))
}

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

fn affine_expression(expression: &AffineExpr) -> Result<String> {
    let mut stack = Vec::<String>::new();
    for node in &expression.nodes {
        match node.kind {
            0 => stack.push(format!("d{}", node.value)),
            1 => stack.push(format!("s{}", node.value)),
            2 => stack.push(if node.value == i64::MIN {
                // Pinned MLIR parses the unsigned magnitude directly as INT64_MIN.
                "9223372036854775808".into()
            } else {
                node.value.to_string()
            }),
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

#[cfg(test)]
mod tests;

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
