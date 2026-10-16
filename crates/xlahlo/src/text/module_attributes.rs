use super::*;
use crate::proto::hlo::{
    DebugAttributeTableEntryProto, DebugAttributesProto, HloBufferDonorProto,
    HloInputOutputAliasProto, Kind, debug_attributes_proto::DebugLogModeProto,
    hlo_buffer_donor_proto::BufferDonorEntryProto, hlo_input_output_alias_proto::AliasEntryProto,
};

pub(super) struct RecoveryAttribute {
    pub table: Option<crate::proto::hlo::OriginalValueRecoveryTableProto>,
    pub pending_modules: Vec<(Option<usize>, String)>,
}

pub(super) fn parse_recovery(parser: &mut Parser) -> Result<RecoveryAttribute> {
    use crate::proto::hlo::{
        OriginalValueRecoveryTableProto, original_value_recovery_table_proto::Entry,
    };
    parser.expect("{")?;
    let mut entries = BTreeMap::new();
    let mut nested_sources = Vec::new();
    while !parser.take("}") {
        let old = parser.original_array()?;
        parser.expect(":")?;
        let new = parser.original_array()?;
        let source = if parser.take(",") {
            let source = String::from_utf8(parser.byte_string()?)
                .map_err(|_| parser.error("recovery module text must be UTF-8"))?;
            Some(source)
        } else {
            None
        };
        let key = (old.instruction_name.clone(), old.shape_index.clone());
        let first = !entries.contains_key(&key);
        if first {
            entries.insert(
                key.clone(),
                Entry {
                    old_original_array: Some(old).into(),
                    new_original_array: Some(new).into(),
                    recovery_module: None.into(),
                    ..Default::default()
                },
            );
        }
        if let Some(source) = source {
            nested_sources.push((key, source, first));
        }
    }
    let entry_indices: BTreeMap<_, _> = entries
        .keys()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect();
    let sources = nested_sources
        .into_iter()
        .map(|(key, source, first)| {
            let index = if first {
                Some(
                    *entry_indices
                        .get(&key)
                        .ok_or_else(|| Error::new("module", "missing recovery entry index"))?,
                )
            } else {
                None
            };
            Ok((index, source))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecoveryAttribute {
        table: (!entries.is_empty()).then(|| OriginalValueRecoveryTableProto {
            entries: entries.into_values().collect(),
            ..Default::default()
        }),
        pending_modules: sources,
    })
}

fn print_original_array(array: &crate::proto::xla_data::OriginalArrayProto) -> Result<String> {
    reject_fields(array, &["instruction_name", "shape_index"])?;
    let name = strings::quote(array.instruction_name.as_bytes())?;
    if array.shape_index.is_empty() {
        Ok(format!("{{{name}}}"))
    } else {
        Ok(format!(
            "{{{name} {{{}}}}}",
            array
                .shape_index
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ))
    }
}

pub(super) fn print_recovery(
    wire: &HloModuleProto,
    result: &mut String,
    depth: usize,
    recovery_modules: &[Option<String>],
) -> Result<()> {
    let Some(table) = wire.original_value_recovery_table.as_ref() else {
        return Ok(());
    };
    reject_fields(table, &["entries"])?;
    if table.entries.is_empty() {
        return Err(Error::new("unsupported", "empty recovery table presence"));
    }
    result.push_str(", origin_recovery_table={");
    let mut previous = None;
    for (index, entry) in table.entries.iter().enumerate() {
        reject_fields(
            entry,
            &[
                "old_original_array",
                "new_original_array",
                "recovery_module",
            ],
        )?;
        let old = entry
            .old_original_array
            .as_ref()
            .ok_or_else(|| Error::new("unsupported", "missing old recovery array"))?;
        let new = entry
            .new_original_array
            .as_ref()
            .ok_or_else(|| Error::new("unsupported", "missing new recovery array"))?;
        let key = (&old.instruction_name, &old.shape_index);
        if previous.is_some_and(|previous| previous >= key) {
            return Err(Error::new(
                "unsupported",
                "recovery table keys must be unique and ordered",
            ));
        }
        previous = Some(key);
        write!(
            result,
            "{}:{}",
            print_original_array(old)?,
            print_original_array(new)?
        )
        .map_err(|e| Error::new("print", e.to_string()))?;
        if let Some(nested) = entry.recovery_module.as_ref() {
            if nested.host_program_shape.as_ref().is_some_and(|program| {
                program.parameter_names
                    != (0..program.parameters.len())
                        .map(|index| format!("p{index}"))
                        .collect::<Vec<_>>()
            }) {
                return Err(Error::new(
                    "unsupported",
                    "recovery host parameter names cannot be reconstructed",
                ));
            }
            if depth + 1 >= super::MAX_DEPTH {
                return Err(Error::new(
                    "resource-limit",
                    "recovery module printing nesting limit",
                ));
            }
            let text = recovery_modules
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::new("print", "missing serialized recovery module"))?;
            write!(result, ",{}", strings::quote(text.as_bytes())?)
                .map_err(|e| Error::new("print", e.to_string()))?;
        }
        result.push(' ');
    }
    result.push('}');
    Ok(())
}

pub(super) fn parse_debug_attributes(
    parser: &mut Parser,
) -> Result<Vec<DebugAttributeTableEntryProto>> {
    parser.expect("{")?;
    let mut entries = BTreeMap::new();
    if !parser.take("}") {
        loop {
            let array = parser.original_array()?;
            parser.expect(":")?;
            let list = parser.take("(");
            let mut attributes = Vec::new();
            if !list || !parser.take(")") {
                loop {
                    parser.expect("{")?;
                    let mut value = DebugAttributesProto::new();
                    loop {
                        let field = parser.word()?;
                        parser.expect("=")?;
                        match field.as_str() {
                            "log_mode" => {
                                value.log_mode = match parser.name()?.as_str() {
                                    "default" => DebugLogModeProto::DEFAULT,
                                    "fusion_debugger" => DebugLogModeProto::FUSION_DEBUGGER,
                                    _ => return Err(parser.error("invalid debug log mode")),
                                }
                                .into()
                            }
                            "callback_id" => value.callback_id = parser.integer()?,
                            "partitioned" => value.partitioned = parser.boolean()?,
                            "op_id" => value.op_id = parser.integer()?,
                            _ => return Err(parser.error("unknown debug attribute")),
                        }
                        if parser.take("}") {
                            break;
                        }
                        parser.expect(",")?;
                    }
                    attributes.push(value);
                    if !list || parser.take(")") {
                        break;
                    }
                    parser.expect(",")?;
                }
            }
            entries.insert(
                (array.instruction_name.clone(), array.shape_index.clone()),
                DebugAttributeTableEntryProto {
                    original_array: Some(array).into(),
                    debug_attributes: attributes,
                    ..Default::default()
                },
            );
            if parser.take("}") {
                break;
            }
            parser.expect(",")?;
        }
    }
    Ok(entries
        .into_values()
        .filter(|entry| !entry.debug_attributes.is_empty())
        .collect())
}

fn print_debug_attributes(entries: &[DebugAttributeTableEntryProto]) -> Result<String> {
    let mut keys = BTreeSet::new();
    let mut previous = None;
    let mut values = Vec::new();
    for entry in entries {
        reject_fields(entry, &["original_array", "debug_attributes"])?;
        if entry.debug_attributes.is_empty() {
            return Err(Error::new(
                "unsupported",
                "HLO text cannot preserve empty debug table entries",
            ));
        }
        let array = entry.original_array.as_ref().ok_or_else(|| {
            Error::new(
                "unsupported",
                "debug table original array presence is required",
            )
        })?;
        let key = (&array.instruction_name, &array.shape_index);
        if !keys.insert(key) || previous.is_some_and(|old| old > key) {
            return Err(Error::new(
                "unsupported",
                "debug table keys must be unique and canonically ordered",
            ));
        }
        previous = Some(key);
        let array = print_original_array(array)?;
        let attributes = entry
            .debug_attributes
            .iter()
            .map(|value| {
                reject_fields(value, &["log_mode", "callback_id", "partitioned", "op_id"])?;
                let mode = match value.log_mode.enum_value() {
                    Ok(DebugLogModeProto::NONE) => "",
                    Ok(DebugLogModeProto::DEFAULT) => "log_mode=default,",
                    Ok(DebugLogModeProto::FUSION_DEBUGGER) => "log_mode=fusion_debugger,",
                    _ => {
                        return Err(Error::new(
                            "unsupported",
                            "debug log mode has no HLO text representation",
                        ));
                    }
                };
                Ok(format!(
                    "{{{mode}callback_id={},partitioned={},op_id={}}}",
                    value.callback_id, value.partitioned, value.op_id
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        values.push(format!("{array}:({})", attributes.join(",")));
    }
    Ok(format!("{{{}}}", values.join(",")))
}

pub(super) fn parse_aliases(parser: &mut Parser) -> Result<HloInputOutputAliasProto> {
    parser.expect("{")?;
    let mut aliases = BTreeMap::new();
    if !parser.take("}") {
        loop {
            let output_shape_index = parser.integers("{", "}")?;
            parser.expect(":")?;
            parser.expect("(")?;
            let parameter_number = parser.integer()?;
            parser.expect(",")?;
            let parameter_shape_index = parser.integers("{", "}")?;
            let kind = if parser.take(",") {
                match parser.name()?.as_str() {
                    "may-alias" => Kind::MAY_ALIAS,
                    "must-alias" => Kind::MUST_ALIAS,
                    _ => return Err(parser.error("unknown module alias kind")),
                }
            } else {
                Kind::MAY_ALIAS
            };
            parser.expect(")")?;
            aliases
                .entry(output_shape_index.clone())
                .or_insert(AliasEntryProto {
                    output_shape_index,
                    parameter_number,
                    parameter_shape_index,
                    kind: kind.into(),
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
    Ok(HloInputOutputAliasProto {
        entries: aliases.into_values().collect(),
        ..Default::default()
    })
}

pub(super) fn parse_donors(parser: &mut Parser) -> Result<HloBufferDonorProto> {
    parser.expect("{")?;
    let mut donors = BTreeSet::new();
    if !parser.take("}") {
        loop {
            parser.expect("(")?;
            let number = parser.integer()?;
            parser.expect(",")?;
            let index = parser.integers("{", "}")?;
            parser.expect(")")?;
            donors.insert((number, index));
            if parser.take("}") {
                break;
            }
            parser.expect(",")?;
            if parser.take("}") {
                break;
            }
        }
    }
    Ok(HloBufferDonorProto {
        entries: donors
            .into_iter()
            .map(
                |(parameter_number, parameter_shape_index)| BufferDonorEntryProto {
                    parameter_number,
                    parameter_shape_index,
                    ..Default::default()
                },
            )
            .collect(),
        ..Default::default()
    })
}

pub(super) fn print(wire: &HloModuleProto, result: &mut String) -> Result<()> {
    let fail = |e: std::fmt::Error| Error::new("print", e.to_string());
    if !wire.debug_attributes.is_empty() {
        write!(
            result,
            ", debug_attributes={}",
            print_debug_attributes(&wire.debug_attributes)?
        )
        .map_err(fail)?;
    }
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
    if let Some(config) = wire.input_output_alias.as_ref() {
        reject_fields(config, &["entries"])?;
        let mut outputs = BTreeSet::new();
        let values = config
            .entries
            .iter()
            .map(|alias| {
                reject_fields(
                    alias,
                    &[
                        "output_shape_index",
                        "parameter_number",
                        "parameter_shape_index",
                        "kind",
                    ],
                )?;
                if !outputs.insert(alias.output_shape_index.clone()) {
                    return Err(Error::new(
                        "unsupported",
                        "duplicate module alias outputs cannot be preserved in text",
                    ));
                }
                let kind = match alias.kind.enum_value() {
                    Ok(Kind::MAY_ALIAS) => "may-alias",
                    Ok(Kind::MUST_ALIAS) => "must-alias",
                    _ => {
                        return Err(Error::new(
                            "unsupported",
                            "unknown/undefined module alias kind cannot be preserved in text",
                        ));
                    }
                };
                Ok(format!(
                    "{}: ({},{},{kind})",
                    index(&alias.output_shape_index),
                    alias.parameter_number,
                    index(&alias.parameter_shape_index)
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        write!(result, ", input_output_alias={{{}}}", values.join(", ")).map_err(fail)?;
    }
    if let Some(config) = wire.buffer_donor.as_ref() {
        reject_fields(config, &["entries"])?;
        let mut donors = BTreeSet::new();
        let values = config
            .entries
            .iter()
            .map(|donor| {
                reject_fields(donor, &["parameter_number", "parameter_shape_index"])?;
                if !donors.insert((donor.parameter_number, donor.parameter_shape_index.clone())) {
                    return Err(Error::new(
                        "unsupported",
                        "duplicate buffer donors cannot be preserved in text",
                    ));
                }
                Ok(format!(
                    "({},{})",
                    donor.parameter_number,
                    index(&donor.parameter_shape_index)
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        write!(result, ", buffer_donor={{{}}}", values.join(", ")).map_err(fail)?;
    }
    Ok(())
}
