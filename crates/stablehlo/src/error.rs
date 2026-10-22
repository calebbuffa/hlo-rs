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
    /// The requested type is not supported by the native StableHLO API.
    #[error("unsupported StableHLO type")]
    UnsupportedType,
    /// MLIR serialization is unavailable for the named entity.
    #[error("cannot serialize {entity} as MLIR: {reason}")]
    UnsupportedSerialization {
        /// Kind or name of the entity that could not be serialized.
        entity: String,
        /// Explanation of the missing serialization support.
        reason: &'static str,
    },
    /// Tensor ranks or dimensions cannot be reconciled.
    #[error("incompatible tensor shapes")]
    IncompatibleShape,
    /// Element types cannot be reconciled.
    #[error("incompatible element types")]
    IncompatibleElement,
    /// A type, element, or attribute value is outside the accepted domain.
    #[error("invalid StableHLO type or element domain")]
    InvalidDomain,
    /// An attribute value fails its validation rule.
    #[error("invalid {attribute}: {reason}")]
    InvalidAttribute {
        /// Name of the invalid attribute.
        attribute: &'static str,
        /// Reason the attribute value was rejected.
        reason: &'static str,
    },
    /// A type fails its validation rule.
    #[error("invalid {ty}: {reason}")]
    InvalidType {
        /// Name of the invalid type.
        ty: &'static str,
        /// Reason the type value was rejected.
        reason: &'static str,
    },
    /// A dimension, rank, or dimension-list value is invalid.
    #[error("invalid dimensions")]
    InvalidDimensions,
    /// Shape or size arithmetic exceeded the representable range.
    #[error("shape arithmetic overflow")]
    Overflow,
    /// An integer cannot be represented by the requested target type.
    #[error("integer value cannot be represented as {target}")]
    IntegerOutOfRange {
        /// Name of the target integer type.
        target: &'static str,
    },
    /// A dimension field contains a negative value.
    #[error("{field}[{index}] must be nonnegative")]
    NegativeDimension {
        /// Name of the dimension field.
        field: &'static str,
        /// Position of the negative dimension.
        index: usize,
    },
    /// A field requiring a positive value is nonpositive.
    #[error("{field} has invalid value {value}")]
    NonPositive {
        /// Name of the field with the invalid value.
        field: &'static str,
        /// The rejected field value.
        value: i64,
    },
    /// A result-accuracy tolerance is negative.
    #[error("result accuracy tolerances must be nonnegative")]
    InvalidTolerance,
    /// Tolerances are nonzero for a result-accuracy mode that requires zero.
    #[error("DEFAULT and HIGHEST result accuracy require zero tolerances")]
    InvalidToleranceMode,
    /// A name required by the API is empty.
    #[error("{0} name must not be empty")]
    EmptyName(&'static str),
    /// Two mesh axes have the same name.
    #[error("mesh has duplicate axis {0:?}")]
    DuplicateMeshAxis(String),
    /// A mesh reference names an axis that was not declared.
    #[error("mesh does not define axis {0:?}")]
    UnknownMeshAxis(String),
    /// A mesh definition violates a structural or value constraint.
    #[error("invalid mesh: {0}")]
    InvalidMesh(&'static str),
    /// A mesh axis reference is invalid.
    #[error("invalid mesh axis reference {axis:?}: {reason}")]
    InvalidMeshAxisReference {
        /// Name of the referenced mesh axis.
        axis: String,
        /// Reason the reference is invalid.
        reason: &'static str,
    },
    /// Mesh dimensions or group storage exceed supported representable limits.
    #[error("mesh dimensions or group storage exceed representable limits")]
    MeshTooLarge,
    /// A dimension bound is outside its valid range.
    #[error("dimension bound {index} has invalid value {bound}")]
    InvalidDimensionBound {
        /// Position of the bound in the bound list.
        index: usize,
        /// Rejected bound value.
        bound: i64,
    },
    /// The selected dot algorithm is not defined for the supplied combination.
    #[error("dot algorithm type/count combination is not specified")]
    UnknownDotAlgorithm,
    /// An operation violates a local operand, result, or attribute constraint.
    #[error("operation {operation} at {location}: {reason}")]
    InvalidOperation {
        /// Operation name or identifier.
        operation: String,
        /// Operation location formatted for diagnostics.
        location: String,
        /// Explanation of the failed operation constraint.
        reason: &'static str,
    },
    /// An operand has a type different from the operation's expected type.
    #[error(
        "operation {operation} at {location} has invalid operand {operand_index}: expected {expected}, got {actual}"
    )]
    InvalidOperand {
        /// Operation name or identifier.
        operation: String,
        /// Operation location formatted for diagnostics.
        location: String,
        /// Zero-based operand position.
        operand_index: usize,
        /// Human-readable expected operand type or category.
        expected: &'static str,
        /// Human-readable actual operand type or category.
        actual: &'static str,
    },
    /// A concrete dimension differs from the required dimension.
    #[error(
        "operation {operation} at {location} has invalid dimension {dimension_index}: expected {expected}, got {actual}"
    )]
    DimensionMismatch {
        /// Operation name or identifier.
        operation: String,
        /// Operation location formatted for diagnostics.
        location: String,
        /// Zero-based dimension position.
        dimension_index: usize,
        /// Required dimension value.
        expected: i64,
        /// Actual dimension value.
        actual: i64,
    },
    /// A dimension index is outside the operation's tensor rank.
    #[error(
        "operation {operation} at {location} dimension {dimension_index} has index {value} outside rank {rank}"
    )]
    InvalidDimensionIndex {
        /// Operation name or identifier.
        operation: String,
        /// Operation location formatted for diagnostics.
        location: String,
        /// Position of the invalid dimension-index attribute.
        dimension_index: usize,
        /// Rejected axis value.
        value: i128,
        /// Rank against which the axis was checked.
        rank: usize,
    },
    /// An operation has no values in a required operand or result group.
    #[error("{operation} requires at least one {role}")]
    EmptyOperands {
        /// Operation name or identifier.
        operation: &'static str,
        /// Name of the empty group, such as `operand` or `result`.
        role: &'static str,
    },
    /// An internal implementation invariant was violated.
    #[error("internal invariant failed: {context}")]
    InternalInvariant {
        /// Description of the invariant that failed.
        context: &'static str,
    },
    /// An affine dimension position exceeds the provided dimension count.
    #[error("affine dimension {position} is outside {dimensions} dimensions")]
    DimensionOutOfBounds {
        /// Referenced dimension position.
        position: usize,
        /// Number of dimensions available.
        dimensions: usize,
    },
    /// An affine symbol position exceeds the provided symbol count.
    #[error("affine symbol {position} is outside {symbols} symbols")]
    SymbolOutOfBounds {
        /// Referenced symbol position.
        position: usize,
        /// Number of symbols available.
        symbols: usize,
    },
    /// Multiplication of affine expressions would produce a non-affine expression.
    #[error("affine product is not affine")]
    NonAffineProduct,
    /// The divisor of an affine floor/ceil/modulo expression is not symbolic.
    #[error("affine divisor for {op:?} is not symbolic")]
    NonSymbolicDivisor {
        /// Affine binary operator whose divisor was rejected.
        op: crate::attributes::AffineBinaryOp,
    },
    /// An affine expression exceeded the supported nesting depth.
    #[error("affine expression exceeds the nesting limit")]
    NestingLimit,
    /// Affine evaluation received a dimension or symbol argument count mismatch.
    #[error(
        "affine argument count mismatch: expected {expected}, got {actual} (symbols: {symbols})"
    )]
    ArgumentCount {
        /// `true` when the count refers to symbols, `false` for dimensions.
        symbols: bool,
        /// Number of arguments required by the expression.
        expected: usize,
        /// Number of arguments actually supplied.
        actual: usize,
    },
    /// Affine expression evaluation attempted division by zero.
    #[error("division by zero in affine expression")]
    DivisionByZero,
    /// Affine expression arithmetic overflowed.
    #[error("arithmetic overflow in affine expression")]
    ArithmeticOverflow,
    /// An affine expression has an invalid structure or encoding.
    #[error("malformed affine expression")]
    MalformedExpression,
    /// A sparse encoding contains no level descriptors.
    #[error("sparse encoding contains no levels")]
    EmptyLevels,
    /// A sparse level width is unsupported or inconsistent.
    #[error("sparse {field} width {width} is invalid")]
    InvalidWidth {
        /// Name of the sparse width field.
        field: &'static str,
        /// Rejected width in bits.
        width: u32,
    },
    /// A sparse level's properties are mutually incompatible.
    #[error("invalid sparse level properties")]
    InvalidLevelProperties,
    /// A structured sparse level has an invalid density or ratio.
    #[error("invalid structured sparse level ratio")]
    InvalidStructuredRatio,
    /// A singleton sparse level sequence is not valid.
    #[error("invalid singleton level sequence")]
    InvalidSingletonSequence,
    /// Singleton levels specify inconsistent storage layouts.
    #[error("singleton levels have inconsistent storage layouts")]
    MixedSingletonLayout,
    /// Batch levels do not appear before other sparse levels.
    #[error("batch levels must lead the sparse level sequence")]
    NonLeadingBatch,
    /// A dense level occurs after a level that does not have unique coordinates.
    #[error("dense level follows a nonunique level")]
    DenseAfterNonunique,
    /// A structured sparse level sequence violates its ordering or count rules.
    #[error("invalid structured level sequence")]
    InvalidStructuredSequence,
    /// A structured sparse level contains invalid block metadata.
    #[error("invalid structured block")]
    InvalidStructuredBlock,
    /// The sparse dimension map rank differs from the number of sparse levels.
    #[error("sparse map rank does not match level rank")]
    MapRankMismatch,
    /// The inverse sparse dimension map is not a valid inverse permutation.
    #[error("sparse inverse map is invalid")]
    InvalidInverse,
    /// A sparse slice rank differs from the tensor rank.
    #[error("sparse slice rank does not match tensor rank")]
    SliceRankMismatch,
    /// A sparse dimension slice has invalid bounds or stride.
    #[error("invalid sparse dimension slice")]
    InvalidSlice,
    /// A numeric sparse metadata field contains an invalid value.
    #[error("invalid numeric sparse {field} value")]
    InvalidNumericValue {
        /// Name of the sparse numeric field.
        field: &'static str,
    },
    /// A sparse metadata value has a type incompatible with its field.
    #[error("sparse {field} value has an incompatible type")]
    ValueTypeMismatch {
        /// Name of the sparse field whose value type is invalid.
        field: &'static str,
    },
    /// A sparse tensor's implicit fill value is not zero.
    #[error("sparse implicit value must be zero")]
    NonzeroImplicitValue,
    /// The sparse tensor's primary element type is unsupported.
    #[error("sparse tensor element type is unsupported")]
    InvalidPrimaryType,
    /// The sparse encoding rank differs from the tensor rank.
    #[error("sparse tensor rank does not match encoding")]
    TensorRankMismatch,
    /// A cycle was found while traversing the named graph.
    #[error("cycle detected in {graph}")]
    CycleDetected {
        /// Name of the graph being traversed.
        graph: &'static str,
    },
    /// An operation builder failed while constructing the named operation.
    #[error("failed to build {operation} at {location}: {source}")]
    BuilderFailure {
        /// Operation name or identifier.
        operation: &'static str,
        /// Operation location formatted for diagnostics.
        location: String,
        /// Recovery classification of the source failure.
        category: RecoveryCategory,
        #[source]
        /// Underlying builder error.
        source: Box<dyn StdError + Send + Sync>,
    },
    /// A lower-level error annotated with StableHLO recovery guidance and context.
    #[error("{context}: {source}")]
    Source {
        /// Recovery classification assigned to the source error.
        category: RecoveryCategory,
        /// StableHLO context describing the failed operation.
        context: &'static str,
        #[source]
        /// Underlying error.
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
