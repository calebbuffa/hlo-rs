use std::error::Error as StdError;
use thiserror::Error;

/// Recovery guidance associated with a StableHLO diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryCategory {
    /// Correct the StableHLO input or supplied operation attributes.
    InvalidInput,
    /// Use a supported type, operation feature, or shape domain.
    Unsupported,
    /// Report an invariant violation in the dialect implementation.
    InternalInvariant,
}

/// Structured StableHLO diagnostic payload carried by pliron errors.
#[derive(Debug, Error)]
pub enum Diagnostic {
    #[error("unsupported StableHLO type")]
    UnsupportedType,
    #[error("cannot serialize {entity} as MLIR: {reason}")]
    UnsupportedSerialization {
        entity: String,
        reason: &'static str,
    },
    #[error("incompatible tensor shapes")]
    IncompatibleShape,
    #[error("incompatible element types")]
    IncompatibleElement,
    #[error("invalid StableHLO type or element domain")]
    InvalidDomain,
    #[error("invalid {attribute}: {reason}")]
    InvalidAttribute {
        attribute: &'static str,
        reason: &'static str,
    },
    #[error("invalid {ty}: {reason}")]
    InvalidType {
        ty: &'static str,
        reason: &'static str,
    },
    #[error("invalid dimensions")]
    InvalidDimensions,
    #[error("shape arithmetic overflow")]
    Overflow,
    #[error("integer value cannot be represented as {target}")]
    IntegerOutOfRange { target: &'static str },
    #[error("{field}[{index}] must be nonnegative")]
    NegativeDimension { field: &'static str, index: usize },
    #[error("{field} has invalid value {value}")]
    NonPositive { field: &'static str, value: i64 },
    #[error("result accuracy tolerances must be nonnegative")]
    InvalidTolerance,
    #[error("DEFAULT and HIGHEST result accuracy require zero tolerances")]
    InvalidToleranceMode,
    #[error("{0} name must not be empty")]
    EmptyName(&'static str),
    #[error("mesh has duplicate axis {0:?}")]
    DuplicateMeshAxis(String),
    #[error("mesh does not define axis {0:?}")]
    UnknownMeshAxis(String),
    #[error("invalid mesh: {0}")]
    InvalidMesh(&'static str),
    #[error("invalid mesh axis reference {axis:?}: {reason}")]
    InvalidMeshAxisReference { axis: String, reason: &'static str },
    #[error("mesh dimensions or group storage exceed representable limits")]
    MeshTooLarge,
    #[error("dimension bound {index} has invalid value {bound}")]
    InvalidDimensionBound { index: usize, bound: i64 },
    #[error("dot algorithm type/count combination is not specified")]
    UnknownDotAlgorithm,
    #[error("operation {operation} at {location}: {reason}")]
    InvalidOperation {
        operation: String,
        location: String,
        reason: &'static str,
    },
    #[error(
        "operation {operation} at {location} has invalid operand {operand_index}: expected {expected}, got {actual}"
    )]
    InvalidOperand {
        operation: String,
        location: String,
        operand_index: usize,
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "operation {operation} at {location} has invalid dimension {dimension_index}: expected {expected}, got {actual}"
    )]
    DimensionMismatch {
        operation: String,
        location: String,
        dimension_index: usize,
        expected: i64,
        actual: i64,
    },
    #[error(
        "operation {operation} at {location} dimension {dimension_index} has index {value} outside rank {rank}"
    )]
    InvalidDimensionIndex {
        operation: String,
        location: String,
        dimension_index: usize,
        value: i128,
        rank: usize,
    },
    #[error("{operation} requires at least one {role}")]
    EmptyOperands {
        operation: &'static str,
        role: &'static str,
    },
    #[error("internal invariant failed: {context}")]
    InternalInvariant { context: &'static str },
    #[error("affine dimension {position} is outside {dimensions} dimensions")]
    DimensionOutOfBounds { position: usize, dimensions: usize },
    #[error("affine symbol {position} is outside {symbols} symbols")]
    SymbolOutOfBounds { position: usize, symbols: usize },
    #[error("affine product is not affine")]
    NonAffineProduct,
    #[error("affine divisor for {op:?} is not symbolic")]
    NonSymbolicDivisor {
        op: crate::attributes::AffineBinaryOp,
    },
    #[error("affine expression exceeds the nesting limit")]
    NestingLimit,
    #[error(
        "affine argument count mismatch: expected {expected}, got {actual} (symbols: {symbols})"
    )]
    ArgumentCount {
        symbols: bool,
        expected: usize,
        actual: usize,
    },
    #[error("division by zero in affine expression")]
    DivisionByZero,
    #[error("arithmetic overflow in affine expression")]
    ArithmeticOverflow,
    #[error("malformed affine expression")]
    MalformedExpression,
    #[error("sparse encoding contains no levels")]
    EmptyLevels,
    #[error("sparse {field} width {width} is invalid")]
    InvalidWidth { field: &'static str, width: u32 },
    #[error("invalid sparse level properties")]
    InvalidLevelProperties,
    #[error("invalid structured sparse level ratio")]
    InvalidStructuredRatio,
    #[error("invalid singleton level sequence")]
    InvalidSingletonSequence,
    #[error("singleton levels have inconsistent storage layouts")]
    MixedSingletonLayout,
    #[error("batch levels must lead the sparse level sequence")]
    NonLeadingBatch,
    #[error("dense level follows a nonunique level")]
    DenseAfterNonunique,
    #[error("invalid structured level sequence")]
    InvalidStructuredSequence,
    #[error("invalid structured block")]
    InvalidStructuredBlock,
    #[error("sparse map rank does not match level rank")]
    MapRankMismatch,
    #[error("sparse inverse map is invalid")]
    InvalidInverse,
    #[error("sparse slice rank does not match tensor rank")]
    SliceRankMismatch,
    #[error("invalid sparse dimension slice")]
    InvalidSlice,
    #[error("invalid numeric sparse {field} value")]
    InvalidNumericValue { field: &'static str },
    #[error("sparse {field} value has an incompatible type")]
    ValueTypeMismatch { field: &'static str },
    #[error("sparse implicit value must be zero")]
    NonzeroImplicitValue,
    #[error("sparse tensor element type is unsupported")]
    InvalidPrimaryType,
    #[error("sparse tensor rank does not match encoding")]
    TensorRankMismatch,
    #[error("cycle detected in {graph}")]
    CycleDetected { graph: &'static str },
    #[error("failed to build {operation} at {location}: {source}")]
    BuilderFailure {
        operation: &'static str,
        location: String,
        category: RecoveryCategory,
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
    #[error("{context}: {source}")]
    Source {
        category: RecoveryCategory,
        context: &'static str,
        #[source]
        source: Box<dyn StdError + Send + Sync>,
    },
}

impl Diagnostic {
    pub(crate) fn into_error(self) -> pliron::result::Error {
        self.into()
    }

    /// Recovery guidance for handling this diagnostic.
    pub fn recovery_category(&self) -> RecoveryCategory {
        match self {
            Self::UnsupportedType | Self::UnsupportedSerialization { .. } => {
                RecoveryCategory::Unsupported
            }
            Self::Overflow | Self::MeshTooLarge => RecoveryCategory::Unsupported,
            Self::Source { category, .. } | Self::BuilderFailure { category, .. } => *category,
            Self::InternalInvariant { .. } => RecoveryCategory::InternalInvariant,
            _ => RecoveryCategory::InvalidInput,
        }
    }
}

impl From<Diagnostic> for pliron::result::Error {
    fn from(diagnostic: Diagnostic) -> Self {
        match diagnostic.recovery_category() {
            RecoveryCategory::InvalidInput => pliron::arg_error_noloc!(diagnostic),
            RecoveryCategory::Unsupported => pliron::input_error_noloc!(diagnostic),
            RecoveryCategory::InternalInvariant => pliron::verify_error_noloc!(diagnostic),
        }
    }
}

impl Diagnostic {
    pub(crate) fn builder_failure(
        operation: &'static str,
        error: pliron::result::Error,
    ) -> pliron::result::Error {
        let category = error
            .err
            .downcast_ref::<Self>()
            .map_or(RecoveryCategory::InvalidInput, Self::recovery_category);
        let location = format!("{:?}", error.loc);
        Self::BuilderFailure {
            operation,
            location,
            category,
            source: Box::new(error),
        }
        .into()
    }
}
