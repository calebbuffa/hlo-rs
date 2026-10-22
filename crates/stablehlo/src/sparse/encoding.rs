use crate::Diagnostic;
use crate::{
    attributes::*,
    types::{ElementType, primitive_element},
};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, IntegerAttr, VecAttr},
    common_traits::Verify,
    context::Context,
    derive::pliron_attr,
    result::Result,
};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Storage format used for a single sparse tensor level.
pub enum SparseLevelFormat {
    /// Every coordinate at this level is present.
    Dense,
    /// Batch level, which must appear before non-batch levels.
    Batch,
    /// Compressed sparse level with unique, ordered coordinates by default.
    Compressed,
    /// Compressed level that permits gaps in its position array.
    LooseCompressed,
    /// Singleton sparse level.
    Singleton,
    /// Structured sparse level with an `n`-of-`m` block pattern.
    Structured {
        /// Number of stored entries in each block.
        n: u8,
        /// Total number of entries in each block.
        m: u8,
    },
}
#[pliron_attr(
    name = "stablehlo.sparse_level",
    format = "`<` $format `,` $n `,` $m `,` $nonunique `,` $nonordered `,` $soa `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Sparse level format and its ordering, uniqueness, and layout flags.
pub struct SparseLevelAttr {
    /// Encoded level format discriminant.
    pub format: u8,
    /// Numerator of the structured level's `n`-of-`m` ratio.
    pub n: u8,
    /// Denominator of the structured level's `n`-of-`m` ratio.
    pub m: u8,
    /// Whether coordinates may repeat at this level.
    pub nonunique: bool,
    /// Whether coordinates may be out of order at this level.
    pub nonordered: bool,
    /// Whether singleton values use the structure-of-arrays representation.
    pub soa: bool,
}
impl SparseLevelAttr {
    /// Creates a level descriptor with the format's default properties.
    pub fn new(format: SparseLevelFormat) -> Self {
        let (kind, n, m) = match format {
            SparseLevelFormat::Dense => (0, 0, 0),
            SparseLevelFormat::Batch => (1, 0, 0),
            SparseLevelFormat::Compressed => (2, 0, 0),
            SparseLevelFormat::LooseCompressed => (3, 0, 0),
            SparseLevelFormat::Singleton => (4, 0, 0),
            SparseLevelFormat::Structured { n, m } => (5, n, m),
        };
        Self {
            format: kind,
            n,
            m,
            nonunique: false,
            nonordered: false,
            soa: false,
        }
    }
    /// Decodes the stored discriminant and structured ratio, if valid.
    pub fn level_format(&self) -> Option<SparseLevelFormat> {
        Some(match self.format {
            0 => SparseLevelFormat::Dense,
            1 => SparseLevelFormat::Batch,
            2 => SparseLevelFormat::Compressed,
            3 => SparseLevelFormat::LooseCompressed,
            4 => SparseLevelFormat::Singleton,
            5 => SparseLevelFormat::Structured {
                n: self.n,
                m: self.m,
            },
            _ => return None,
        })
    }
    /// Checks format-specific flags and structured ratio constraints.
    pub fn validate(&self) -> Result<()> {
        if self.format > 5
            || (self.format != 5 && (self.n != 0 || self.m != 0))
            || ((self.format <= 1 || self.format == 5)
                && (self.nonunique || self.nonordered || self.soa))
            || (self.soa && self.format != 4)
        {
            return Err(Diagnostic::InvalidLevelProperties.into());
        }
        if self.format == 5 && self.n > self.m {
            return Err(Diagnostic::InvalidStructuredRatio.into());
        }
        Ok(())
    }
}
impl Verify for SparseLevelAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
#[pliron_attr(
    name = "stablehlo.sparse_slice",
    format = "`<` `[` vec($offset, CharSpace(`,`)) `]` `,` `[` vec($size, CharSpace(`,`)) `]` `,` `[` vec($stride, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Optional offset, extent, and stride metadata for one tensor dimension.
pub struct SparseTensorDimSlice {
    /// Starting coordinate; empty represents a dynamic offset.
    pub offset: Vec<i64>,
    /// Extent of the slice; empty represents a dynamic size.
    pub size: Vec<i64>,
    /// Step between coordinates; empty represents a dynamic stride.
    pub stride: Vec<i64>,
}
impl Default for SparseTensorDimSlice {
    fn default() -> Self {
        Self {
            offset: vec![0],
            size: vec![],
            stride: vec![1],
        }
    }
}
impl SparseTensorDimSlice {
    /// Returns whether all three slice components are unspecified.
    pub fn is_completely_dynamic(&self) -> bool {
        self.offset.is_empty() && self.size.is_empty() && self.stride.is_empty()
    }
    /// Checks the optional scalar fields for valid nonnegative offsets and positive sizes/strides.
    pub fn validate(&self) -> Result<()> {
        if self.offset.len() > 1
            || self.size.len() > 1
            || self.stride.len() > 1
            || self.offset.iter().any(|&v| v < 0)
            || self.size.iter().chain(&self.stride).any(|&v| v <= 0)
        {
            return Err(Diagnostic::InvalidSlice.into());
        }
        Ok(())
    }
}
impl Verify for SparseTensorDimSlice {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
#[pliron_attr(
    name = "stablehlo.sparse_encoding",
    format = "`<` `[` vec($levels, CharSpace(`,`)) `]` `,` $dimension_to_level `,` `[` vec($level_to_dimension, CharSpace(`,`)) `]` `,` $position_width `,` $coordinate_width `,` $explicit_value `,` $implicit_value `,` `[` vec($dimension_slices, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Sparse tensor storage layout, mappings, widths, values, and dimension slices.
pub struct SparseTensorEncoding {
    /// Storage format for each sparse level.
    pub levels: Vec<SparseLevelAttr>,
    /// Mapping from tensor dimensions to sparse levels.
    pub dimension_to_level: AffineMap,
    /// Inverse mapping when it can be inferred from `dimension_to_level`.
    pub level_to_dimension: Vec<AffineMap>,
    /// Bit width of position indices; zero selects the implementation default.
    pub position_width: u32,
    /// Bit width of coordinates; zero selects the implementation default.
    pub coordinate_width: u32,
    /// Optional explicit stored-value default, represented as zero or one numeric attribute.
    pub explicit_value: VecAttr,
    /// Optional implicit-value default, which must be zero for sparse tensors.
    pub implicit_value: VecAttr,
    /// Optional slice metadata for each tensor dimension.
    pub dimension_slices: Vec<SparseTensorDimSlice>,
}
impl SparseTensorEncoding {
    /// Builds an encoding with inferred inverse mapping and default widths and values.
    pub fn new(ctx: &Context, levels: Vec<SparseLevelAttr>, map: AffineMap) -> Result<Self> {
        let inverse = infer_sparse_level_to_dimension(&map)?;
        let encoding = Self {
            levels,
            dimension_to_level: map,
            level_to_dimension: inverse.into_iter().collect(),
            position_width: 0,
            coordinate_width: 0,
            explicit_value: VecAttr::new(vec![]),
            implicit_value: VecAttr::new(vec![]),
            dimension_slices: vec![],
        };
        encoding.validate(ctx)?;
        Ok(encoding)
    }
    /// Validates level sequencing, maps, slices, widths, and optional numeric values.
    pub fn validate(&self, ctx: &Context) -> Result<()> {
        for (field, width) in [
            ("position", self.position_width),
            ("coordinate", self.coordinate_width),
        ] {
            if !matches!(width, 0 | 8 | 16 | 32 | 64) {
                return Err(Diagnostic::InvalidWidth { field, width }.into());
            }
        }
        if self.levels.is_empty() {
            return Err(Diagnostic::EmptyLevels.into());
        }
        let mut saw_nonbatch = false;
        for (i, level) in self.levels.iter().enumerate() {
            level.validate()?;
            if level.format == 1 {
                if saw_nonbatch {
                    return Err(Diagnostic::NonLeadingBatch.into());
                }
            } else {
                saw_nonbatch = true;
            }
            if level.format == 4 {
                let Some(previous_index) = i.checked_sub(1) else {
                    return Err(Diagnostic::InvalidSingletonSequence.into());
                };
                let Some(previous) = self.levels.get(previous_index) else {
                    return Err(Diagnostic::InvalidSingletonSequence.into());
                };
                if !matches!(previous.format, 2..=4) {
                    return Err(Diagnostic::InvalidSingletonSequence.into());
                }
                if previous.format == 4 && previous.soa != level.soa {
                    return Err(Diagnostic::MixedSingletonLayout.into());
                }
            }
            if level.format == 0 && i > 0 && self.levels[i - 1].nonunique {
                return Err(Diagnostic::DenseAfterNonunique.into());
            }
            if level.format == 5 {
                if i != self.levels.len() - 1 || self.levels[..i].iter().any(|l| l.format != 0) {
                    return Err(Diagnostic::InvalidStructuredSequence.into());
                }
                if self.dimension_to_level.dimensions != self.dimension_to_level.results.len() {
                    let mut coefficient = None;
                    for expression in &self.dimension_to_level.results {
                        if expression.nodes.last().is_some_and(|node| node.kind == 7) {
                            let Some(value) = expression
                                .nodes
                                .iter()
                                .rev()
                                .nth(1)
                                .filter(|node| node.kind == 2)
                                .map(|n| n.value)
                            else {
                                return Err(Diagnostic::InvalidStructuredBlock.into());
                            };
                            if value != i64::from(level.m)
                                || coefficient.is_some_and(|old| old != value)
                            {
                                return Err(Diagnostic::InvalidStructuredBlock.into());
                            }
                            coefficient = Some(value);
                        }
                    }
                    if coefficient.is_none() {
                        return Err(Diagnostic::InvalidStructuredBlock.into());
                    }
                }
            }
        }
        let map = &self.dimension_to_level;
        map.validate()?;
        if map.results.len() != self.levels.len() || map.dimensions > self.levels.len() {
            return Err(Diagnostic::MapRankMismatch.into());
        }
        let inferred = infer_sparse_level_to_dimension(map)?;
        if inferred.is_none() && map.symbols == 0 {
            return Err(Diagnostic::InvalidInverse.into());
        }
        if self.level_to_dimension.len() > 1 || self.level_to_dimension.first() != inferred.as_ref()
        {
            return Err(Diagnostic::InvalidInverse.into());
        }
        if !self.dimension_slices.is_empty() {
            if self.dimension_slices.len() != map.dimensions || map.dimensions != self.levels.len()
            {
                return Err(Diagnostic::SliceRankMismatch.into());
            }
            for slice in &self.dimension_slices {
                slice.validate()?;
            }
        }
        for (field, values) in [
            ("explicit", &self.explicit_value),
            ("implicit", &self.implicit_value),
        ] {
            if values.0.len() > 1 {
                return Err(Diagnostic::InvalidNumericValue { field }.into());
            }
            if let Some(value) = values.0.first()
                && (numeric_type(ctx, value).is_none() || value.verify(ctx).is_err())
            {
                return Err(Diagnostic::InvalidNumericValue { field }.into());
            }
        }
        Ok(())
    }
    /// Checks that the encoding is compatible with a ranked tensor and its element type.
    pub fn verify_tensor(&self, ctx: &Context, rank: usize, element: ElementType) -> Result<()> {
        self.validate(ctx)?;
        if rank == 0 || rank != self.dimension_to_level.dimensions {
            return Err(Diagnostic::TensorRankMismatch.into());
        }
        use ElementType::*;
        if !matches!(
            element,
            F16 | Bf16
                | F32
                | F64
                | I8
                | I16
                | I32
                | I64
                | Si8
                | Si16
                | Si32
                | Si64
                | Ui8
                | Ui16
                | Ui32
                | Ui64
                | C64
                | C128
        ) {
            return Err(Diagnostic::InvalidPrimaryType.into());
        }
        for (field, values) in [
            ("explicit", &self.explicit_value),
            ("implicit", &self.implicit_value),
        ] {
            if let Some(value) = values.0.first()
                && numeric_type(ctx, value) != Some(element)
            {
                return Err(Diagnostic::ValueTypeMismatch { field }.into());
            }
        }
        if let Some(value) = self.implicit_value.0.first() {
            let zero = if let Some(v) = value.downcast_ref::<IntegerAttr>() {
                v.value().to_string_decimal(false) == "0"
            } else if let Some(v) = value.downcast_ref::<BoolAttr>() {
                !bool::from(v.clone())
            } else if let Some(v) = value.downcast_ref::<ScalarBitsAttr>() {
                if element.is_float() {
                    let width = element.bit_width().unwrap_or(64);
                    v.words
                        .first()
                        .is_some_and(|bits| bits & ((1u64 << (width - 1)) - 1) == 0)
                } else if let Some(component) = element.complex_component() {
                    let width = component.bit_width().unwrap_or(64);
                    let mask = (1u64 << (width - 1)) - 1;
                    if width == 32 {
                        v.words
                            .first()
                            .is_some_and(|bits| bits & mask == 0 && (bits >> 32) & mask == 0)
                    } else {
                        v.words.iter().all(|bits| bits & mask == 0)
                    }
                } else {
                    v.words.iter().all(|&w| w == 0)
                }
            } else {
                false
            };
            if !zero {
                return Err(Diagnostic::NonzeroImplicitValue.into());
            }
        }
        Ok(())
    }
}
impl Verify for SparseTensorEncoding {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.validate(ctx)
    }
}
fn numeric_type(ctx: &Context, value: &AttrObj) -> Option<ElementType> {
    if let Some(v) = value.downcast_ref::<ScalarBitsAttr>() {
        primitive_element(ctx, v.ty)
    } else if let Some(v) = value.downcast_ref::<IntegerAttr>() {
        primitive_element(ctx, v.get_type().into())
    } else if value.downcast_ref::<BoolAttr>().is_some() {
        Some(ElementType::I1)
    } else {
        None
    }
}
/// Infers the inverse dimension-to-level map for direct or quotient/remainder mappings.
///
/// Returns `None` when the map uses symbols or has a form that is not invertible
/// by these supported patterns; malformed affine expressions return an error.
pub fn infer_sparse_level_to_dimension(map: &AffineMap) -> Result<Option<AffineMap>> {
    map.validate()?;
    if map.symbols != 0 || map.dimensions > map.results.len() {
        return Ok(None);
    }
    let mut direct = vec![None; map.dimensions];
    let mut quotient = vec![None; map.dimensions];
    let mut remainder = vec![None; map.dimensions];
    for (level, expression) in map.results.iter().enumerate() {
        match expression.nodes.as_slice() {
            [
                AffineNode {
                    kind: 0,
                    value: dimension,
                },
            ] => {
                let dimension = *dimension as usize;
                if direct[dimension].is_some() || quotient[dimension].is_some() {
                    return Ok(None);
                }
                direct[dimension] = Some(level);
            }
            [
                AffineNode {
                    kind: 0,
                    value: dimension,
                },
                AffineNode {
                    kind: 2,
                    value: coefficient,
                },
                AffineNode { kind: op, .. },
            ] => {
                let dimension = *dimension as usize;
                if *coefficient <= 0 || direct[dimension].is_some() {
                    return Ok(None);
                }
                match op {
                    5 if quotient[dimension].is_none() => {
                        quotient[dimension] = Some((level, *coefficient))
                    }
                    7 if remainder[dimension].is_none() => {
                        if quotient[dimension].is_none_or(|(_, c)| c != *coefficient) {
                            return Ok(None);
                        }
                        remainder[dimension] = Some(level);
                    }
                    _ => return Ok(None),
                }
            }
            _ => return Ok(None),
        }
    }
    let mut results = Vec::with_capacity(map.dimensions);
    for dimension in 0..map.dimensions {
        if let Some(level) = direct[dimension] {
            results.push(AffineExpr::dimension(level));
        } else if let (Some((q, c)), Some(r)) = (quotient[dimension], remainder[dimension]) {
            results.push(AffineExpr::binary(
                AffineBinaryOp::Add,
                AffineExpr::binary(
                    AffineBinaryOp::Mul,
                    AffineExpr::dimension(q),
                    AffineExpr::constant(c),
                ),
                AffineExpr::dimension(r),
            ));
        } else {
            return Ok(None);
        }
    }
    Ok(Some(AffineMap {
        dimensions: map.results.len(),
        symbols: 0,
        results,
    }))
}
