use super::*;
use crate::proto::{
    hlo::{CustomCallApiVersion, CustomCallSchedule},
    xla_data::{FrontendAttributes, OutputOperandAliasing},
};

pub(super) fn parse_aliases(parser: &mut Parser) -> Result<Vec<OutputOperandAliasing>> {
    parser.expect("{")?;
    let mut aliases = Vec::new();
    if !parser.take("}") {
        loop {
            let output_shape_index = parser.integers("{", "}")?;
            parser.expect(":")?;
            parser.expect("(")?;
            let operand_index = parser.integer()?;
            parser.expect(",")?;
            let operand_shape_index = parser.integers("{", "}")?;
            parser.expect(")")?;
            aliases.push(OutputOperandAliasing {
                output_shape_index,
                operand_index,
                operand_shape_index,
                ..Default::default()
            });
            if parser.take("}") {
                break;
            }
            parser.expect(",")?;
            if parser.take("}") {
                break;
            }
        }
    }
    Ok(aliases)
}

pub(super) fn parse_frontend(parser: &mut Parser) -> Result<FrontendAttributes> {
    parser.expect("{")?;
    let mut result = FrontendAttributes::new();
    if !parser.take("}") {
        loop {
            let key = parser.word()?;
            parser.expect("=")?;
            let value = String::from_utf8(parser.string_or_dictionary()?)
                .map_err(|_| parser.error("frontend attribute must be valid UTF-8"))?;
            result.map.insert(key, value);
            if parser.take("}") {
                break;
            }
            parser.expect(",")?;
        }
    }
    Ok(result)
}

pub(super) fn print_frontend(attributes: &FrontendAttributes) -> Result<String> {
    reject_fields(attributes, &["map"])?;
    let mut entries: Vec<_> = attributes.map.iter().collect();
    entries.sort_by_key(|(key, _)| *key);
    let values = entries
        .into_iter()
        .map(|(key, value)| {
            if !valid_name(key) {
                return Err(Error::new("name", "unprintable frontend attribute name"));
            }
            Ok(format!("{key}={}", strings::quote(value.as_bytes())?))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("{{{}}}", values.join(", ")))
}

pub(super) fn parse_layouts(
    parser: &mut Parser,
) -> Result<Vec<crate::proto::xla_data::ShapeProto>> {
    parser.expect("{")?;
    let mut result = Vec::new();
    if !parser.take("}") {
        loop {
            result.push(parser.shape(0)?.into_proto());
            if parser.take("}") {
                break;
            }
            parser.expect(",")?;
        }
    }
    Ok(result)
}

pub(super) fn api_version(parser: &mut Parser) -> Result<CustomCallApiVersion> {
    let value = CustomCallApiVersion::from_str(&parser.word()?)
        .ok_or_else(|| parser.error("unknown custom-call API version"))?;
    if value == CustomCallApiVersion::API_VERSION_UNSPECIFIED {
        return Err(parser.error("unspecified custom-call API version is invalid in text"));
    }
    Ok(value)
}

pub(super) fn schedule(parser: &mut Parser) -> Result<CustomCallSchedule> {
    CustomCallSchedule::from_str(&parser.word()?)
        .ok_or_else(|| parser.error("unknown custom-call schedule"))
}

pub(super) fn validate_constraints(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
) -> Result<()> {
    fn has_layout(shape: &crate::proto::xla_data::ShapeProto) -> bool {
        if shape.element_type.enum_value_or_default() == PrimitiveType::TUPLE {
            shape.tuple_shapes.iter().all(has_layout)
        } else {
            !shape
                .element_type
                .enum_value_or_default()
                .is_array_element()
                || shape.layout.is_some()
        }
    }
    if !instruction.constrain_layout {
        return Ok(());
    }
    if instruction.shape.as_ref().is_none_or(|s| !has_layout(s))
        || instruction.operand_ids.len() != instruction.operand_shapes_with_layout.len()
    {
        return Err(Error::new(
            "custom-call",
            "layout constraint cardinality or result layout mismatch",
        ));
    }
    for (&id, constraint) in instruction
        .operand_ids
        .iter()
        .zip(&instruction.operand_shapes_with_layout)
    {
        let operand = instructions
            .iter()
            .find(|i| i.id == id)
            .and_then(|i| i.shape.as_ref())
            .ok_or_else(|| Error::new("reference", "custom-call operand missing"))?;
        if !has_layout(constraint)
            || !Shape::from_proto(constraint.clone())?
                .compatible(&Shape::from_proto(operand.clone())?)
        {
            return Err(Error::new(
                "custom-call",
                "invalid custom-call operand layout constraint",
            ));
        }
    }
    Ok(())
}

pub(super) fn print_aliases(instruction: &HloInstructionProto) -> Result<String> {
    let mut values = Vec::new();
    for alias in &instruction.output_operand_aliasing {
        reject_fields(
            alias,
            &["output_shape_index", "operand_index", "operand_shape_index"],
        )?;
        let index = |index: &[i64]| {
            format!(
                "{{{}}}",
                index
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        values.push(format!(
            "{}: ({},{})",
            index(&alias.output_shape_index),
            alias.operand_index,
            index(&alias.operand_shape_index)
        ));
    }
    Ok(format!("{{{}}}", values.join(", ")))
}

pub(super) fn print_fields(instruction: &HloInstructionProto, result: &mut String) -> Result<()> {
    let fail = |e: std::fmt::Error| Error::new("print", e.to_string());
    if !instruction.output_operand_aliasing.is_empty() {
        if !matches!(
            instruction.opcode.as_str(),
            "fusion" | "custom-call" | "async-start" | "async-update"
        ) {
            return Err(Error::new(
                "unsupported",
                "opcode output aliases cannot be represented in HLO text",
            ));
        }
        write!(
            result,
            ", output_to_operand_aliasing={}",
            print_aliases(instruction)?
        )
        .map_err(fail)?;
    }
    if let Some(attributes) = instruction.frontend_attributes.as_ref() {
        write!(
            result,
            ", frontend_attributes={}",
            print_frontend(attributes)?
        )
        .map_err(fail)?;
    }
    if !instruction.backend_config.is_empty() {
        write!(
            result,
            ", backend_config={}",
            strings::quote(&instruction.backend_config)?
        )
        .map_err(fail)?;
    }
    if instruction.is_composite {
        result.push_str(", is_composite=true");
    }
    if instruction.opcode == "fusion" {
        if !matches!(
            instruction.fusion_kind.as_str(),
            "kLoop" | "kInput" | "kOutput" | "kCustom"
        ) {
            return Err(Error::new("fusion", "unknown fusion kind"));
        }
        write!(result, ", kind={}", instruction.fusion_kind).map_err(fail)?;
    }
    if instruction.opcode == "custom-call" {
        if let Some(dimensions) = instruction.convolution_dimension_numbers.as_ref() {
            write!(
                result,
                ", dim_labels={}",
                print_convolution_dimensions(dimensions)?
            )
            .map_err(fail)?;
        }
        if let Some(wire) = instruction.literal.as_ref() {
            let literal = Literal::from_proto(wire.clone())?;
            result.push_str(", literal=");
            literals::print_typed(&literal, result)?;
        }
        let version = instruction
            .custom_call_api_version
            .enum_value()
            .map_err(|_| Error::new("custom-call", "unknown custom-call API version"))?;
        if version == CustomCallApiVersion::API_VERSION_UNSPECIFIED {
            return Err(Error::new(
                "unsupported",
                "unspecified custom-call API version cannot be preserved in text",
            ));
        }
        let schedule = instruction
            .custom_call_schedule
            .enum_value()
            .map_err(|_| Error::new("custom-call", "unknown custom-call schedule"))?;
        write!(result,", custom_call_target={}, custom_call_has_side_effect={}, api_version={}, schedule={}, feature_group_count={}, batch_group_count={}",
            strings::quote(instruction.custom_call_target.as_bytes())?,instruction.custom_call_has_side_effect,
            version.descriptor().name(),schedule.descriptor().name(),instruction.feature_group_count,instruction.batch_group_count).map_err(fail)?;
        if instruction.constrain_layout {
            if !instruction.called_computation_ids.is_empty() {
                return Err(Error::new(
                    "unsupported",
                    "layout-constrained custom-call callees are not preserved by the pinned parser",
                ));
            }
            let shapes = instruction
                .operand_shapes_with_layout
                .iter()
                .map(|s| print_shape(&Shape::from_proto(s.clone())?))
                .collect::<Result<Vec<_>>>()?;
            write!(
                result,
                ", operand_layout_constraints={{{}}}",
                shapes.join(", ")
            )
            .map_err(fail)?;
        } else if !instruction.operand_shapes_with_layout.is_empty() {
            return Err(Error::new(
                "unsupported",
                "unconstrained custom-call operand layout records cannot be preserved in text",
            ));
        }
    }
    Ok(())
}
