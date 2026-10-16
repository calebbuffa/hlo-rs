use super::{SparseSliceError, SparseTensorDimSlice};
use crate::{
    AffineBinaryOp as Op, AffineError, AffineExpr as Expr, AffineMap, Attribute, ElementType,
};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SparseLevelFormat {
    Dense,
    Batch,
    Compressed,
    LooseCompressed,
    Singleton,
    Structured { n: u8, m: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SparseLevelType {
    pub format: SparseLevelFormat,
    pub nonunique: bool,
    pub nonordered: bool,
    pub soa: bool,
}

impl SparseLevelType {
    pub const fn new(format: SparseLevelFormat) -> Self {
        Self {
            format,
            nonunique: false,
            nonordered: false,
            soa: false,
        }
    }

    pub fn verify(&self) -> Result<(), SparseEncodingError> {
        if matches!(
            self.format,
            SparseLevelFormat::Dense
                | SparseLevelFormat::Batch
                | SparseLevelFormat::Structured { .. }
        ) && (self.nonunique || self.nonordered || self.soa)
        {
            return Err(SparseEncodingError::InvalidLevelProperties);
        }
        if self.soa && self.format != SparseLevelFormat::Singleton {
            return Err(SparseEncodingError::InvalidLevelProperties);
        }
        if let SparseLevelFormat::Structured { n, m } = self.format
            && n > m
        {
            return Err(SparseEncodingError::InvalidStructuredRatio);
        }
        Ok(())
    }
}

/// Sparse tensor encoding with positional dimension/level maps.
///
/// The inverse is inferred for permutations and blocked floor/mod maps. Maps
/// with symbols may have no inferred inverse, following the reference contract.
/// When an inverse is inferable, `level_to_dimension` must contain that inverse;
/// use [`Self::new`] to initialize both directions consistently.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SparseTensorEncoding {
    pub levels: Vec<SparseLevelType>,
    pub dimension_to_level: AffineMap,
    pub level_to_dimension: Option<AffineMap>,
    pub position_width: u32,
    pub coordinate_width: u32,
    pub explicit_value: Option<Box<Attribute>>,
    pub implicit_value: Option<Box<Attribute>>,
    pub dimension_slices: Vec<SparseTensorDimSlice>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseEncodingError {
    Affine(AffineError),
    Slice(SparseSliceError),
    EmptyLevels,
    InvalidWidth { field: &'static str, width: u32 },
    InvalidLevelProperties,
    InvalidStructuredRatio,
    InvalidSingletonSequence,
    MixedSingletonLayout,
    NonLeadingBatch,
    DenseAfterNonunique,
    InvalidStructuredSequence,
    InvalidStructuredBlock,
    MapRankMismatch,
    InvalidInverse,
    SliceRankMismatch,
    InvalidNumericValue { field: &'static str },
    ValueTypeMismatch { field: &'static str },
    NonzeroImplicitValue,
    InvalidPrimaryType,
    TensorRankMismatch,
}

impl SparseTensorEncoding {
    pub fn new(
        levels: Vec<SparseLevelType>,
        dimension_to_level: AffineMap,
    ) -> Result<Self, SparseEncodingError> {
        let level_to_dimension = infer_inverse(&dimension_to_level)?;
        let encoding = Self {
            levels,
            dimension_to_level,
            level_to_dimension,
            position_width: 0,
            coordinate_width: 0,
            explicit_value: None,
            implicit_value: None,
            dimension_slices: vec![],
        };
        encoding.verify()?;
        Ok(encoding)
    }

    pub fn verify(&self) -> Result<(), SparseEncodingError> {
        use SparseLevelFormat::*;
        for (field, width) in [
            ("position", self.position_width),
            ("coordinate", self.coordinate_width),
        ] {
            if !matches!(width, 0 | 8 | 16 | 32 | 64) {
                return Err(SparseEncodingError::InvalidWidth { field, width });
            }
        }
        if self.levels.is_empty() {
            return Err(SparseEncodingError::EmptyLevels);
        }
        let mut saw_nonbatch = false;
        for (i, level) in self.levels.iter().enumerate() {
            level.verify()?;
            if level.format == Batch {
                if saw_nonbatch {
                    return Err(SparseEncodingError::NonLeadingBatch);
                }
            } else {
                saw_nonbatch = true;
            }
            if level.format == Singleton {
                let Some(previous) = i.checked_sub(1).map(|j| &self.levels[j]) else {
                    return Err(SparseEncodingError::InvalidSingletonSequence);
                };
                if !matches!(previous.format, Compressed | LooseCompressed | Singleton) {
                    return Err(SparseEncodingError::InvalidSingletonSequence);
                }
                if previous.format == Singleton && previous.soa != level.soa {
                    return Err(SparseEncodingError::MixedSingletonLayout);
                }
            }
            if level.format == Dense && i > 0 && self.levels[i - 1].nonunique {
                return Err(SparseEncodingError::DenseAfterNonunique);
            }
            if let Structured { m, .. } = level.format {
                if i != self.levels.len() - 1 || self.levels[..i].iter().any(|l| l.format != Dense)
                {
                    return Err(SparseEncodingError::InvalidStructuredSequence);
                }
                if self.dimension_to_level.dimensions != self.dimension_to_level.results.len() {
                    let mut coefficient = None;
                    for expression in &self.dimension_to_level.results {
                        if let Expr::Binary {
                            op: Op::Mod, rhs, ..
                        } = expression
                        {
                            let Expr::Constant(value) = rhs.as_ref() else {
                                return Err(SparseEncodingError::InvalidStructuredBlock);
                            };
                            if *value != i64::from(m)
                                || coefficient.is_some_and(|old| old != *value)
                            {
                                return Err(SparseEncodingError::InvalidStructuredBlock);
                            }
                            coefficient = Some(*value);
                        }
                    }
                    if coefficient.is_none() {
                        return Err(SparseEncodingError::InvalidStructuredBlock);
                    }
                }
            }
        }
        let map = &self.dimension_to_level;
        map.verify().map_err(SparseEncodingError::Affine)?;
        if map.results.len() != self.levels.len() || map.dimensions > self.levels.len() {
            return Err(SparseEncodingError::MapRankMismatch);
        }
        let inferred = infer_inverse(map)?;
        if inferred.is_none() && map.symbols == 0 {
            return Err(SparseEncodingError::InvalidInverse);
        }
        if let Some(inverse) = &self.level_to_dimension {
            inverse.verify().map_err(SparseEncodingError::Affine)?;
        }
        if self.level_to_dimension != inferred {
            return Err(SparseEncodingError::InvalidInverse);
        }
        if !self.dimension_slices.is_empty() {
            if self.dimension_slices.len() != map.dimensions || map.dimensions != self.levels.len()
            {
                return Err(SparseEncodingError::SliceRankMismatch);
            }
            for slice in &self.dimension_slices {
                slice.verify().map_err(SparseEncodingError::Slice)?;
            }
        }
        for (field, value) in [
            ("explicit", self.explicit_value.as_deref()),
            ("implicit", self.implicit_value.as_deref()),
        ] {
            if let Some(value) = value
                && (numeric_type(value).is_none() || value.verify().is_err())
            {
                return Err(SparseEncodingError::InvalidNumericValue { field });
            }
        }
        Ok(())
    }

    pub(crate) fn verify_tensor(
        &self,
        rank: usize,
        element: ElementType,
    ) -> Result<(), SparseEncodingError> {
        self.verify()?;
        if rank == 0 || rank != self.dimension_to_level.dimensions {
            return Err(SparseEncodingError::TensorRankMismatch);
        }
        if !matches!(
            element,
            ElementType::F16
                | ElementType::Bf16
                | ElementType::F32
                | ElementType::F64
                | ElementType::I8
                | ElementType::I16
                | ElementType::I32
                | ElementType::I64
                | ElementType::Si8
                | ElementType::Si16
                | ElementType::Si32
                | ElementType::Si64
                | ElementType::Ui8
                | ElementType::Ui16
                | ElementType::Ui32
                | ElementType::Ui64
                | ElementType::C64
                | ElementType::C128
        ) {
            return Err(SparseEncodingError::InvalidPrimaryType);
        }
        for (field, value) in [
            ("explicit", self.explicit_value.as_deref()),
            ("implicit", self.implicit_value.as_deref()),
        ] {
            if let Some(value) = value
                && numeric_type(value) != Some(element)
            {
                return Err(SparseEncodingError::ValueTypeMismatch { field });
            }
        }
        if let Some(value) = self.implicit_value.as_deref() {
            let zero = match value {
                Attribute::Bool(value) => !value,
                Attribute::Integer { value, .. } => *value == 0,
                Attribute::Float { ty, bits } => {
                    let width = ty.bit_width().expect("verified float");
                    *bits & ((1_u64 << (width - 1)) - 1) == 0
                }
                Attribute::Complex {
                    ty,
                    real,
                    imaginary,
                } => {
                    let width = if *ty == ElementType::C64 { 32 } else { 64 };
                    let mask = (1_u64 << (width - 1)) - 1;
                    real & mask == 0 && imaginary & mask == 0
                }
                _ => false,
            };
            if !zero {
                return Err(SparseEncodingError::NonzeroImplicitValue);
            }
        }
        Ok(())
    }
}

fn numeric_type(attribute: &Attribute) -> Option<ElementType> {
    match attribute {
        Attribute::Integer { ty, .. }
        | Attribute::Float { ty, .. }
        | Attribute::Complex { ty, .. } => Some(*ty),
        Attribute::Bool(_) => Some(ElementType::I1),
        _ => None,
    }
}

/// Infers the reference-supported permutation or blocked map inverse.
///
/// Invalid expression bindings are errors. Unsupported mappings return `None`;
/// no numerical sampling is used as a substitute for a symbolic inverse.
pub fn infer_sparse_level_to_dimension(
    map: &AffineMap,
) -> Result<Option<AffineMap>, SparseEncodingError> {
    infer_inverse(map)
}

fn infer_inverse(map: &AffineMap) -> Result<Option<AffineMap>, SparseEncodingError> {
    map.verify().map_err(SparseEncodingError::Affine)?;
    if map.symbols != 0 {
        return Ok(None);
    }
    if map.dimensions > map.results.len() {
        return Ok(None);
    }
    let mut direct = vec![None; map.dimensions];
    let mut quotient = vec![None; map.dimensions];
    let mut remainder = vec![None; map.dimensions];
    for (level, expression) in map.results.iter().enumerate() {
        match expression {
            Expr::Dimension(dimension) => {
                if direct[*dimension].is_some() || quotient[*dimension].is_some() {
                    return Ok(None);
                }
                direct[*dimension] = Some(level);
            }
            Expr::Binary { op, lhs, rhs } => {
                let (Expr::Dimension(dimension), Expr::Constant(coefficient)) =
                    (lhs.as_ref(), rhs.as_ref())
                else {
                    return Ok(None);
                };
                if *coefficient <= 0 || direct[*dimension].is_some() {
                    return Ok(None);
                }
                match op {
                    Op::FloorDiv if quotient[*dimension].is_none() => {
                        quotient[*dimension] = Some((level, *coefficient))
                    }
                    Op::Mod if remainder[*dimension].is_none() => {
                        if quotient[*dimension].is_none_or(|(_, c)| c != *coefficient) {
                            return Ok(None);
                        }
                        remainder[*dimension] = Some(level);
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
            results.push(Expr::Dimension(level));
        } else if let (Some((q, c)), Some(r)) = (quotient[dimension], remainder[dimension]) {
            results.push(Expr::binary(
                Op::Add,
                Expr::binary(Op::Mul, Expr::Dimension(q), Expr::Constant(c)),
                Expr::Dimension(r),
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

impl fmt::Display for SparseEncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Affine(error) => write!(f, "invalid sparse affine map: {error}"),
            Self::Slice(error) => write!(f, "invalid sparse slice: {error}"),
            Self::InvalidWidth { field, width } => write!(
                f,
                "invalid sparse {field} width {width}; expected 0, 8, 16, 32 or 64"
            ),
            Self::InvalidNumericValue { field } => {
                write!(f, "sparse {field} value must be a valid numeric attribute")
            }
            Self::ValueTypeMismatch { field } => {
                write!(f, "sparse {field} value type must match the tensor element")
            }
            Self::EmptyLevels => write!(f, "sparse encoding requires at least one level"),
            Self::InvalidLevelProperties => {
                write!(f, "sparse level format does not support its properties")
            }
            Self::InvalidStructuredRatio => write!(f, "structured sparse level requires n <= m"),
            Self::InvalidSingletonSequence => write!(
                f,
                "singleton levels must follow compressed, loose-compressed or singleton levels"
            ),
            Self::MixedSingletonLayout => write!(
                f,
                "consecutive singleton levels must use the same AoS/SoA layout"
            ),
            Self::NonLeadingBatch => write!(f, "batch levels must be leading levels"),
            Self::DenseAfterNonunique => write!(f, "dense level cannot follow a nonunique level"),
            Self::InvalidStructuredSequence => write!(
                f,
                "structured level must be last and preceded only by dense levels"
            ),
            Self::InvalidStructuredBlock => {
                write!(f, "structured level block coefficient must equal m")
            }
            Self::MapRankMismatch => write!(f, "sparse dimension/level map ranks are inconsistent"),
            Self::InvalidInverse => write!(
                f,
                "sparse map has no supported inverse or its supplied inverse differs"
            ),
            Self::SliceRankMismatch => write!(
                f,
                "sparse slices require matching dimension and level ranks"
            ),
            Self::NonzeroImplicitValue => write!(f, "sparse implicit value must be zero"),
            Self::InvalidPrimaryType => write!(f, "unsupported sparse tensor primary element type"),
            Self::TensorRankMismatch => write!(
                f,
                "sparse encoding requires a nonscalar tensor with matching rank"
            ),
        }
    }
}

impl std::error::Error for SparseEncodingError {}
