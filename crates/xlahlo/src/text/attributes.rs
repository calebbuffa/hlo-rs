use super::{
    ParseOptions, Parser, callable, layout, metadata, original_value, print_shape, reject_fields,
};
use crate::{
    Error, Result, Shape,
    proto::xla_data::{
        FrontendAttributes, LayoutProto, OpMetadata, OriginalValueProto, PaddingConfig,
        ParameterReplication, StatisticsViz, padding_config::PaddingConfigDimension,
    },
};

/// Parse the brace-delimited shape list used by pinned HLO assembly.
pub fn parse_shape_list(source: &str) -> Result<Vec<Shape>> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let values = callable::parse_layouts(&mut parser)?
        .into_iter()
        .map(Shape::from_proto)
        .collect::<Result<_>>()?;
    parser.finish()?;
    Ok(values)
}

pub fn print_shape_list(shapes: &[Shape]) -> Result<String> {
    Ok(format!(
        "{{{}}}",
        shapes
            .iter()
            .map(print_shape)
            .collect::<Result<Vec<_>>>()?
            .join(", ")
    ))
}

/// Parse layout syntax without a shape-dependent rank or physical-layout check.
pub fn parse_layout(source: &str) -> Result<LayoutProto> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let value = parser.layout()?;
    parser.finish()?;
    Ok(value)
}

pub fn print_layout(value: &LayoutProto) -> Result<String> {
    layout::print(value)
}

impl Parser {
    pub(super) fn padding_config(&mut self) -> Result<PaddingConfig> {
        let source = self.word()?;
        let mut config = PaddingConfig::new();
        for dimension in source.split('x') {
            let values = dimension
                .split('_')
                .enumerate()
                .map(|(index, value)| {
                    let digits = if index < 2 {
                        value.strip_prefix('-').unwrap_or(value)
                    } else {
                        value
                    };
                    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                        return Err(self.error("invalid padding decimal component"));
                    }
                    value
                        .parse::<i64>()
                        .map_err(|_| self.error("invalid padding integer"))
                })
                .collect::<Result<Vec<_>>>()?;
            if !(2..=3).contains(&values.len()) {
                return Err(self.error("padding requires low_high or low_high_interior"));
            }
            config.dimensions.push(PaddingConfigDimension {
                edge_padding_low: values[0],
                edge_padding_high: values[1],
                interior_padding: values.get(2).copied().unwrap_or(0),
                ..Default::default()
            });
        }
        Ok(config)
    }
}

pub fn parse_padding_config(source: &str) -> Result<PaddingConfig> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let value = parser.padding_config()?;
    parser.finish()?;
    Ok(value)
}

pub fn print_padding_config(value: &PaddingConfig) -> Result<String> {
    reject_fields(value, &["dimensions"])?;
    if value.dimensions.is_empty() {
        return Err(Error::new(
            "unsupported",
            "empty padding configuration has no standalone HLO syntax",
        ));
    }
    let values = value
        .dimensions
        .iter()
        .map(|dimension| {
            reject_fields(
                dimension,
                &["edge_padding_low", "edge_padding_high", "interior_padding"],
            )?;
            if dimension.interior_padding < 0 {
                return Err(Error::new(
                    "unsupported",
                    "negative interior padding has no HLO text syntax",
                ));
            }
            Ok(format!(
                "{}_{}_{}",
                dimension.edge_padding_low, dimension.edge_padding_high, dimension.interior_padding
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(values.join("x"))
}

pub fn parse_original_value(source: &str) -> Result<OriginalValueProto> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let value = parser.original_value()?;
    parser.finish()?;
    Ok(value)
}

pub fn print_original_value(value: &OriginalValueProto) -> Result<String> {
    original_value::print(value)
}

pub fn parse_frontend_attributes(source: &str) -> Result<FrontendAttributes> {
    let mut parser = Parser::new_with_root_context(
        source,
        ParseOptions::default(),
        Some("frontend_attributes"),
    )?;
    let value = callable::parse_frontend(&mut parser)?;
    parser.finish()?;
    Ok(value)
}

pub fn print_frontend_attributes(value: &FrontendAttributes) -> Result<String> {
    callable::print_frontend(value)
}

pub fn parse_statistics(source: &str) -> Result<StatisticsViz> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let value = parser.statistics()?;
    parser.finish()?;
    Ok(value)
}

pub fn print_statistics(value: &StatisticsViz) -> Result<String> {
    metadata::print_statistics(value)
}

/// Parse replication flags without a parameter-shape cardinality check.
/// Module assembly additionally requires one flag per non-tuple shape leaf.
pub fn parse_parameter_replication(source: &str) -> Result<ParameterReplication> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let value = parser.parameter_replication()?;
    parser.finish()?;
    Ok(value)
}

pub fn print_parameter_replication(value: &ParameterReplication) -> Result<String> {
    metadata::print_replication(value)
}

pub fn parse_metadata(source: &str) -> Result<OpMetadata> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let value = parser.metadata()?;
    parser.finish()?;
    Ok(value)
}

pub fn print_metadata(value: &OpMetadata) -> Result<String> {
    metadata::print(value)
}
