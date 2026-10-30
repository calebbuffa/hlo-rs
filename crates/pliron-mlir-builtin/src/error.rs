use std::error::Error as StdError;
use thiserror::Error;

/// Recovery guidance for shared representation errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecoveryCategory {
    /// Correct the input type, attribute, or evaluation arguments.
    InvalidInput,
    /// Use a supported representation or numeric domain.
    Unsupported,
}

/// Structured representation errors carried by the shared pliron envelope.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Diagnostic {
    /// An affine dimension index exceeds the map's dimension count.
    #[error("affine dimension {position} is outside {dimensions} dimensions")]
    DimensionOutOfBounds {
        /// Referenced dimension.
        position: usize,
        /// Available dimensions.
        dimensions: usize,
    },
    /// An affine symbol index exceeds the map's symbol count.
    #[error("affine symbol {position} is outside {symbols} symbols")]
    SymbolOutOfBounds {
        /// Referenced symbol.
        position: usize,
        /// Available symbols.
        symbols: usize,
    },
    /// A product contains two nonsymbolic expressions.
    #[error("affine product is not affine")]
    NonAffineProduct,
    /// An affine divisor is not symbolic.
    #[error("affine divisor for {op:?} is not symbolic")]
    NonSymbolicDivisor {
        /// Operator with an invalid divisor.
        op: crate::attributes::AffineBinaryOp,
    },
    /// An expression exceeds the supported nesting depth.
    #[error("affine expression exceeds the nesting limit")]
    NestingLimit,
    /// The evaluation arguments do not match the map signature.
    #[error(
        "affine argument count mismatch: expected {expected}, got {actual} (symbols: {symbols})"
    )]
    ArgumentCount {
        /// Whether these are symbol rather than dimension arguments.
        symbols: bool,
        /// Required argument count.
        expected: usize,
        /// Supplied argument count.
        actual: usize,
    },
    /// An affine evaluation attempted division by zero.
    #[error("division by zero in affine expression")]
    DivisionByZero,
    /// An affine evaluation overflowed.
    #[error("arithmetic overflow in affine expression")]
    ArithmeticOverflow,
    /// An affine expression has malformed storage.
    #[error("malformed affine expression")]
    MalformedExpression,
    /// A native type violates its representation constraints.
    #[error("invalid {ty}: {reason}")]
    InvalidType {
        /// Type or type family.
        ty: &'static str,
        /// Violated constraint.
        reason: &'static str,
    },
    /// An attribute violates its representation constraints.
    #[error("invalid {attribute}: {reason}")]
    InvalidAttribute {
        /// Attribute or attribute family.
        attribute: &'static str,
        /// Violated constraint.
        reason: &'static str,
    },
    /// The requested integer conversion would lose information.
    #[error("integer value cannot be represented as {target}")]
    IntegerOutOfRange {
        /// Destination integer type.
        target: &'static str,
    },
    /// No lossless MLIR assembly representation is implemented.
    #[error("cannot serialize {entity} as MLIR: {reason}")]
    UnsupportedSerialization {
        /// Entity requiring a dialect-specific representation.
        entity: String,
        /// Missing representation or unsupported syntax.
        reason: &'static str,
    },
}

impl Diagnostic {
    /// Returns recovery guidance without parsing diagnostic text.
    pub fn recovery_category(&self) -> RecoveryCategory {
        match self {
            Self::UnsupportedSerialization { .. } => RecoveryCategory::Unsupported,
            _ => RecoveryCategory::InvalidInput,
        }
    }

    /// Wraps this diagnostic in pliron's error envelope.
    pub fn into_error(self) -> pliron::result::Error {
        self.into()
    }

    /// Finds a builtin diagnostic in an error's source chain.
    pub fn from_error(error: &pliron::result::Error) -> Option<&Self> {
        let mut source: &(dyn StdError + 'static) = error;
        loop {
            if let Some(diagnostic) = source.downcast_ref::<Self>() {
                return Some(diagnostic);
            }
            source = if let Some(envelope) = source.downcast_ref::<pliron::result::Error>() {
                envelope.err.as_ref()
            } else {
                source.source()?
            };
        }
    }
}
impl From<crate::integer::IntegerParseError> for pliron::result::Error {
    fn from(error: crate::integer::IntegerParseError) -> Self {
        pliron::arg_error_noloc!(error)
    }
}

impl From<Diagnostic> for pliron::result::Error {
    fn from(error: Diagnostic) -> Self {
        match error.recovery_category() {
            RecoveryCategory::InvalidInput => pliron::arg_error_noloc!(error),
            RecoveryCategory::Unsupported => pliron::input_error_noloc!(error),
        }
    }
}
