use super::{LayoutProto, Parser, Result, reject_fields};
use crate::{
    COMBINE_DIMENSION, MATCH_DIMENSION,
    proto::xla_data::{SplitConfigProto, TileProto},
};

impl Parser {
    pub(super) fn layout(&mut self) -> Result<LayoutProto> {
        self.expect("{")?;
        let mut layout = LayoutProto {
            tail_padding_alignment_in_elements: 1,
            ..Default::default()
        };
        if super::numbers::integer(self.peek()).is_ok() {
            loop {
                layout.minor_to_major.push(self.integer()?);
                if !self.take(",") {
                    break;
                }
            }
        }
        if self.take(":") {
            if self.take("T") {
                loop {
                    self.expect("(")?;
                    let mut tile = TileProto::default();
                    if !self.take(")") {
                        loop {
                            tile.dimensions.push(if self.take("*") {
                                COMBINE_DIMENSION
                            } else if self.take("d") {
                                MATCH_DIMENSION
                            } else {
                                self.integer()?
                            });
                            if self.take(")") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                    layout.tiles.push(tile);
                    if self.peek() != "(" {
                        break;
                    }
                }
            }
            if self.take("L") {
                layout.tail_padding_alignment_in_elements = self.layout_integer()?;
            }
            if self.take("E") {
                layout.element_size_in_bits = self.layout_integer()?;
            }
            if self.take("S") {
                layout.memory_space = self.layout_integer()?;
            }
            if self.take("SC") {
                loop {
                    self.expect("(")?;
                    let dimension = self.integer()?;
                    let split_indices = self.integers(":", ")")?;
                    layout.split_configs.push(SplitConfigProto {
                        dimension,
                        split_indices,
                        ..Default::default()
                    });
                    if self.peek() != "(" {
                        break;
                    }
                }
            }
            if self.take("M") {
                layout.dynamic_shape_metadata_prefix_bytes = self.layout_integer()?;
            }
        }
        self.expect("}")?;
        Ok(layout)
    }

    fn layout_integer(&mut self) -> Result<i64> {
        self.expect("(")?;
        let value = self.integer()?;
        self.expect(")")?;
        Ok(value)
    }
}

pub(super) fn print(layout: &LayoutProto) -> Result<String> {
    if layout.dynamic_shape_metadata_prefix_bytes < 0 {
        return Err(crate::Error::new(
            "unsupported",
            "upstream layout printing would discard a negative metadata prefix size",
        ));
    }
    reject_fields(
        layout,
        &[
            "minor_to_major",
            "tiles",
            "tail_padding_alignment_in_elements",
            "element_size_in_bits",
            "memory_space",
            "split_configs",
            "dynamic_shape_metadata_prefix_bytes",
        ],
    )?;
    let mut extra = String::new();
    if !layout.tiles.is_empty() {
        extra.push('T');
        for tile in &layout.tiles {
            reject_fields(tile, &["dimensions"])?;
            let dimensions = tile
                .dimensions
                .iter()
                .map(|&dimension| match dimension {
                    COMBINE_DIMENSION => "*".into(),
                    MATCH_DIMENSION => "d".into(),
                    _ => dimension.to_string(),
                })
                .collect::<Vec<_>>()
                .join(",");
            extra.push_str(&format!("({dimensions})"));
        }
    }
    if !matches!(layout.tail_padding_alignment_in_elements, 0 | 1) {
        extra.push_str(&format!("L({})", layout.tail_padding_alignment_in_elements));
    }
    if layout.element_size_in_bits != 0 {
        extra.push_str(&format!("E({})", layout.element_size_in_bits));
    }
    if layout.memory_space != 0 {
        extra.push_str(&format!("S({})", layout.memory_space));
    }
    if !layout.split_configs.is_empty() {
        extra.push_str("SC");
        for split in &layout.split_configs {
            reject_fields(split, &["dimension", "split_indices"])?;
            extra.push_str(&format!(
                "({}:{})",
                split.dimension,
                split
                    .split_indices
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }
    if layout.dynamic_shape_metadata_prefix_bytes != 0 {
        extra.push_str(&format!(
            "M({})",
            layout.dynamic_shape_metadata_prefix_bytes
        ));
    }
    let order = layout
        .minor_to_major
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    Ok(if extra.is_empty() {
        format!("{{{order}}}")
    } else {
        format!("{{{order}:{extra}}}")
    })
}
