use super::{Parser, reject_fields, strings};
use crate::{
    Error, Result,
    proto::xla_data::{
        OpMetadata, ParameterReplication, Payload, ProfileType, Statistic, StatisticsViz,
    },
};
use protobuf::Enum;
use std::collections::BTreeSet;

impl Parser {
    pub(super) fn parameter_replication(&mut self) -> Result<ParameterReplication> {
        self.expect("{")?;
        let mut values = Vec::new();
        if !self.take("}") {
            loop {
                values.push(self.boolean()?);
                if self.take("}") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(ParameterReplication {
            replicated_at_leaf_buffers: values,
            ..Default::default()
        })
    }

    pub(super) fn statistics(&mut self) -> Result<StatisticsViz> {
        self.expect("{")?;
        let mut wire = StatisticsViz::new();
        if !self.take("}") {
            let name = self.word()?;
            if !super::valid_name(&name) {
                return Err(self.error("invalid statistics index attribute"));
            }
            self.expect("=")?;
            wire.stat_index_to_visualize = self.integer()?;
            while self.take(",") {
                let name = self.word()?;
                if !super::valid_name(&name) {
                    return Err(self.error("invalid statistic name"));
                }
                self.expect("=")?;
                let value = self.statistic_number()?;
                wire.statistics.push(Statistic {
                    stat_name: name,
                    stat_val: value,
                    ..Default::default()
                });
            }
            self.expect("}")?;
        }
        Ok(wire)
    }

    fn statistic_number(&mut self) -> Result<f64> {
        if matches!(self.peek(), "nan" | "-nan") {
            return self.number();
        }
        let offset = self.offset();
        let token = self.word()?;
        if !token
            .strip_prefix('-')
            .unwrap_or(&token)
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_digit)
        {
            return Err(self.error("expected statistic integer or decimal"));
        }
        if token.contains(['.', 'e', 'E']) {
            // Native statistics consume lexer decimals directly, including overflow.
            token
                .parse()
                .map_err(|_| self.error("invalid statistic decimal"))
        } else {
            Ok(super::numbers::integer(&token).map_err(|e| e.at(offset))? as f64)
        }
    }

    pub(super) fn utf8_string(&mut self) -> Result<String> {
        String::from_utf8(self.byte_string()?).map_err(|_| self.error("expected UTF-8 string"))
    }

    pub(super) fn metadata(&mut self) -> Result<OpMetadata> {
        self.expect("{")?;
        let mut wire = OpMetadata::default();
        let mut seen = BTreeSet::new();
        while !self.take("}") {
            self.take(",");
            let name = self.word()?;
            if !seen.insert(name.clone()) {
                return Err(self.error("duplicate metadata attribute"));
            }
            self.expect("=")?;
            match name.as_str() {
                "op_type" => wire.op_type = self.utf8_string()?,
                "op_name" => wire.op_name = self.utf8_string()?,
                "source_file" => wire.source_file = self.utf8_string()?,
                "deduplicated_name" => wire.deduplicated_name = self.utf8_string()?,
                "scheduling_name" => wire.scheduling_name = self.utf8_string()?,
                "source_line" | "source_end_line" | "source_column" | "source_end_column"
                | "stack_frame_id" => {
                    let value = i32::try_from(self.integer()?)
                        .map_err(|_| self.error("metadata integer exceeds i32"))?;
                    match name.as_str() {
                        "source_line" => wire.source_line = value,
                        "source_end_line" => wire.source_end_line = value,
                        "source_column" => wire.source_column = value,
                        "source_end_column" => wire.source_end_column = value,
                        _ => wire.stack_frame_id = value,
                    }
                }
                "profile_type" => {
                    wire.profile_type = self
                        .integers("{", "}")?
                        .into_iter()
                        .map(|value| {
                            i32::try_from(value)
                                .ok()
                                .and_then(ProfileType::from_i32)
                                .map(Into::into)
                                .ok_or_else(|| self.error("invalid profile type"))
                        })
                        .collect::<Result<_>>()?;
                }
                "metadata_payload" => {
                    let mut payload = Payload::default();
                    payload.set_value(self.byte_string()?);
                    wire.metadata_payload = Some(payload).into();
                }
                _ => return Err(self.error(format!("unknown metadata attribute {name}"))),
            }
        }
        Ok(wire)
    }

    pub(super) fn metadata_list(&mut self) -> Result<Vec<OpMetadata>> {
        if self.peek() == "{"
            && self
                .tokens
                .get(self.index + 1)
                .is_some_and(|token| token.value == "{")
        {
            self.expect("{")?;
            let mut values = vec![self.metadata()?];
            while self.take(",") {
                values.push(self.metadata()?);
            }
            self.expect("}")?;
            Ok(values)
        } else {
            Ok(vec![self.metadata()?])
        }
    }
}

pub(super) fn print_parameter_replication(
    wire: &ParameterReplication,
    shape: &crate::Shape,
) -> Result<String> {
    if wire.replicated_at_leaf_buffers.len() != shape.leaf_count() {
        return Err(Error::new(
            "unsupported",
            "parameter replication count cannot be preserved in text",
        ));
    }
    print_replication(wire)
}

pub(super) fn print_replication(wire: &ParameterReplication) -> Result<String> {
    reject_fields(wire, &["replicated_at_leaf_buffers"])?;
    Ok(format!(
        "{{{}}}",
        wire.replicated_at_leaf_buffers
            .iter()
            .map(bool::to_string)
            .collect::<Vec<_>>()
            .join(",")
    ))
}

pub(super) fn print_statistics(wire: &StatisticsViz) -> Result<String> {
    reject_fields(wire, &["stat_index_to_visualize", "statistics"])?;
    if wire.stat_index_to_visualize == 0 && wire.statistics.is_empty() {
        return Ok("{}".into());
    }
    let mut fields = vec![format!(
        "visualizing_index={}",
        wire.stat_index_to_visualize
    )];
    for statistic in &wire.statistics {
        reject_fields(statistic, &["stat_name", "stat_val"])?;
        if !super::valid_name(&statistic.stat_name) {
            return Err(Error::new("unsupported", "unprintable statistic name"));
        }
        let value = if statistic.stat_val.is_infinite() {
            // Statistics accept overflowing decimals, but not inf tokens.
            if statistic.stat_val.is_sign_negative() {
                "-1e999".into()
            } else {
                "1e999".into()
            }
        } else {
            super::numbers::print(
                u128::from(statistic.stat_val.to_bits()),
                crate::PrimitiveType::F64,
            )?
        };
        fields.push(format!("{}={value}", statistic.stat_name));
    }
    Ok(format!("{{{}}}", fields.join(",")))
}

pub(super) fn same_statistics(a: Option<&StatisticsViz>, b: Option<&StatisticsViz>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.stat_index_to_visualize == b.stat_index_to_visualize
                && a.special_fields == b.special_fields
                && a.statistics.len() == b.statistics.len()
                && a.statistics.iter().zip(&b.statistics).all(|(a, b)| {
                    a.stat_name == b.stat_name
                        && a.stat_val.to_bits() == b.stat_val.to_bits()
                        && a.special_fields == b.special_fields
                })
        }
        _ => false,
    }
}

pub(super) fn print(wire: &OpMetadata) -> Result<String> {
    reject_fields(
        wire,
        &[
            "op_type",
            "op_name",
            "source_file",
            "source_line",
            "source_end_line",
            "source_column",
            "source_end_column",
            "stack_frame_id",
            "profile_type",
            "deduplicated_name",
            "scheduling_name",
            "metadata_payload",
        ],
    )?;
    let mut fields = Vec::new();
    for (name, value) in [
        ("op_type", &wire.op_type),
        ("op_name", &wire.op_name),
        ("source_file", &wire.source_file),
        ("deduplicated_name", &wire.deduplicated_name),
        ("scheduling_name", &wire.scheduling_name),
    ] {
        if !value.is_empty() {
            fields.push(format!("{name}={}", strings::quote(value.as_bytes())?));
        }
    }
    for (name, value) in [
        ("source_line", wire.source_line),
        ("source_end_line", wire.source_end_line),
        ("source_column", wire.source_column),
        ("source_end_column", wire.source_end_column),
        ("stack_frame_id", wire.stack_frame_id),
    ] {
        if value != 0 {
            fields.push(format!("{name}={value}"));
        }
    }
    if !wire.profile_type.is_empty() {
        let values = wire
            .profile_type
            .iter()
            .map(|value| {
                value
                    .enum_value()
                    .map(|value| (value as i32).to_string())
                    .map_err(|_| Error::new("metadata", "unknown profile type"))
            })
            .collect::<Result<Vec<_>>>()?;
        fields.push(format!("profile_type={{{}}}", values.join(",")));
    }
    if let Some(payload) = wire.metadata_payload.as_ref() {
        reject_fields(payload, &["value"])?;
        if !payload.has_value() {
            return Err(Error::new(
                "unsupported",
                "metadata payload without an inline value",
            ));
        }
        fields.push(format!(
            "metadata_payload={}",
            strings::quote(payload.value())?
        ));
    }
    Ok(format!("{{{}}}", fields.join(" ")))
}

pub(super) fn print_list(values: &[OpMetadata]) -> Result<String> {
    let values = values.iter().map(print).collect::<Result<Vec<_>>>()?;
    if values.len() == 1 {
        Ok(values[0].clone())
    } else {
        Ok(format!("{{{}}}", values.join(",")))
    }
}
