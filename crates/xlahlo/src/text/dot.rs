use super::*;
use crate::proto::xla_data::{BlockScalingConfig, SparsityConfig};

fn integers(parser: &mut Parser) -> Result<Vec<i64>> {
    parser
        .word()?
        .split('x')
        .map(|part| {
            part.parse()
                .map_err(|_| parser.error("expected x-separated integers"))
        })
        .collect()
}

pub(super) fn parse_sparsity(parser: &mut Parser) -> Result<SparsityConfig> {
    let mut config = SparsityConfig::new();
    parser.expect("{")?;
    while !parser.take("}") {
        let side = parser.word()?;
        parser.expect("=")?;
        let side = match side.as_str() {
            "lhs" => config.lhs.mut_or_insert_default(),
            "rhs" => config.rhs.mut_or_insert_default(),
            _ => return Err(parser.error("unknown sparsity side")),
        };
        parser.expect("{")?;
        while !parser.take("}") {
            let key = parser.word()?;
            parser.expect("=")?;
            match key.as_str() {
                "sparsity" => {
                    let values = integers(parser)?;
                    let [nonzero, block] = values.as_slice() else {
                        return Err(parser.error("sparsity requires two x-separated integers"));
                    };
                    side.num_non_zero = *nonzero;
                    side.block_size = *block;
                }
                "dimension" => side.dimension = parser.integer()?,
                "stride" => side.stride = parser.integer()?,
                "idx" => {
                    side.idx = i32::try_from(parser.integer()?)
                        .map_err(|_| parser.error("sparsity operand index exceeds i32"))?
                }
                _ => return Err(parser.error("unknown tensor sparsity field")),
            }
        }
    }
    Ok(config)
}

pub(super) fn parse_block_scaling(parser: &mut Parser) -> Result<BlockScalingConfig> {
    let mut config = BlockScalingConfig::new();
    parser.expect("{")?;
    while !parser.take("}") {
        let side = parser.word()?;
        parser.expect("=")?;
        let side = match side.as_str() {
            "lhs" => config.lhs.mut_or_insert_default(),
            "rhs" => config.rhs.mut_or_insert_default(),
            _ => return Err(parser.error("unknown block scaling side")),
        };
        parser.expect("{")?;
        while !parser.take("}") {
            let key = parser.word()?;
            parser.expect("=")?;
            match key.as_str() {
                "scale_idx" => {
                    side.scale_idx = i32::try_from(parser.integer()?)
                        .map_err(|_| parser.error("scale operand index exceeds i32"))?
                }
                "zero_idx" => {
                    side.zero_idx = Some(
                        i32::try_from(parser.integer()?)
                            .map_err(|_| parser.error("zero point operand index exceeds i32"))?,
                    )
                }
                "strides" => side.strides = integers(parser)?,
                "steps" => side.steps = integers(parser)?,
                _ => return Err(parser.error("unknown tensor block scaling field")),
            }
        }
    }
    Ok(config)
}

pub(super) fn print_fields(instruction: &HloInstructionProto, result: &mut String) -> Result<()> {
    let fail = |e: std::fmt::Error| Error::new("print", e.to_string());
    if instruction.opcode == "ragged-dot" {
        let dimensions = instruction
            .ragged_dot_dimension_numbers
            .as_ref()
            .ok_or_else(|| Error::new("ragged-dot", "missing dimension numbers"))?;
        reject_fields(
            dimensions,
            &[
                "dot_dimension_numbers",
                "lhs_ragged_dimensions",
                "rhs_group_dimensions",
            ],
        )?;
        for (name, axes) in [
            ("lhs_ragged_dims", &dimensions.lhs_ragged_dimensions),
            ("rhs_group_dims", &dimensions.rhs_group_dimensions),
        ] {
            write!(
                result,
                ", {name}={{{}}}",
                axes.iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(fail)?;
        }
    }
    if let Some(config) = instruction.sparsity_config.as_ref() {
        reject_fields(config, &["lhs", "rhs"])?;
        let mut sides = Vec::new();
        for (name, side) in [("lhs", config.lhs.as_ref()), ("rhs", config.rhs.as_ref())] {
            if let Some(side) = side {
                reject_fields(
                    side,
                    &["num_non_zero", "block_size", "dimension", "stride", "idx"],
                )?;
                sides.push(format!(
                    "{name}={{sparsity={}x{} dimension={} stride={} idx={}}}",
                    side.num_non_zero, side.block_size, side.dimension, side.stride, side.idx
                ));
            }
        }
        write!(result, ", sparsity_config={{{}}}", sides.join(" ")).map_err(fail)?;
    }
    if let Some(config) = instruction.block_scaling_config.as_ref() {
        reject_fields(config, &["lhs", "rhs"])?;
        let mut sides = Vec::new();
        for (name, side) in [("lhs", config.lhs.as_ref()), ("rhs", config.rhs.as_ref())] {
            if let Some(side) = side {
                reject_fields(side, &["scale_idx", "zero_idx", "strides", "steps"])?;
                let mut fields = format!("scale_idx={}", side.scale_idx);
                if let Some(zero) = side.zero_idx {
                    write!(fields, " zero_idx={zero}").map_err(fail)?;
                }
                for (name, values) in [("strides", &side.strides), ("steps", &side.steps)] {
                    if !values.is_empty() {
                        write!(
                            fields,
                            " {name}={}",
                            values
                                .iter()
                                .map(i64::to_string)
                                .collect::<Vec<_>>()
                                .join("x")
                        )
                        .map_err(fail)?;
                    }
                }
                sides.push(format!("{name}={{{fields}}}"));
            }
        }
        write!(result, ", block_scaling_config={{{}}}", sides.join(" ")).map_err(fail)?;
    }
    Ok(())
}
