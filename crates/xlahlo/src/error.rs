//! Recoverable dialect errors. Framework traits retain pliron's error boundary.

/// Broad recovery action for an XLAHLO diagnostic or evaluator failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecoveryCategory {
    /// Correct the supplied IR, types, attributes, or request before retrying.
    InvalidInput,
    /// Select a supported operation, representation, or execution strategy.
    Unsupported,
    /// Supply missing values or execution context before retrying.
    MissingInput,
    /// Increase an explicit execution limit or use a different execution strategy.
    ResourceLimit,
    /// Inspect the source error; this crate cannot classify its recovery policy.
    Unknown,
}

/// Typed details carried by [`pliron::result::Error::err`].
///
/// Fallible IR APIs use [`crate::Result`]. Inspect this payload with
/// [`Self::from_error`] rather than matching diagnostic text.
/// The framework envelope retains its category, source location and backtrace.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Diagnostic {
    /// The opcode spelling is not part of the supported catalog.
    #[error("unknown HLO opcode {name:?}; use an opcode from Opcode::ALL")]
    UnknownOpcode {
        /// Unrecognized opcode spelling.
        name: String,
    },
    /// An axis is negative or outside the shape's rank.
    #[error("axis {axis} is outside the valid range 0..{rank}")]
    InvalidAxis {
        /// Rejected logical axis.
        axis: i64,
        /// Number of axes in the operand.
        rank: usize,
    },
    /// A symbolic computation reference cannot be resolved in its module.
    #[error("computation {name:?} does not exist; declare it in the module")]
    UnknownComputation {
        /// Unresolved computation symbol.
        name: String,
    },
    /// Verified editing cannot safely rehearse a context-dependent callback.
    #[error("verified instruction editing cannot rehearse custom verifier callback {field}")]
    UnsupportedCallback {
        /// Verification option whose callback cannot run on staged IR.
        field: &'static str,
    },
    /// A shape, layout, literal, or primitive type violates its contract.
    #[error("invalid HLO shape or value: {message}")]
    InvalidShape {
        /// Explanation of the violated shape or storage constraint.
        message: String,
    },
    /// An instruction attribute is absent, ill-typed, or inconsistent.
    #[error("invalid HLO attribute {attribute}: {message}")]
    InvalidAttribute {
        /// Attribute whose value is invalid or absent.
        attribute: &'static str,
        /// Explanation of the attribute constraint.
        message: String,
    },
    /// Operand types or geometry do not admit the requested instruction.
    #[error("HLO inference failed: {message}")]
    Inference {
        /// Operand or geometry constraint that prevented inference.
        message: String,
    },
    /// A known operation has no native semantic implementation.
    #[error("unsupported HLO operation: {message}")]
    Unsupported {
        /// Known feature or value domain without an implementation.
        message: String,
    },
    /// An operation or graph violates an HLO verification rule.
    #[error("invalid HLO IR: {message}")]
    Verification {
        /// Explanation of the violated graph or instruction rule.
        message: String,
    },
    /// Context identifying the instruction at which verification failed.
    #[error("while verifying {operation}: {source}")]
    Operation {
        /// Instruction identity at the failure site.
        operation: String,
        /// Underlying verification diagnostic, retained for error chaining.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl Diagnostic {
    /// Finds the first XLAHLO diagnostic in an error and its source chain.
    ///
    /// Traverses framework `err` payloads as well as standard error sources.
    /// Returns `None` for leaf conversion errors, evaluator errors, or failures
    /// originating entirely in another dialect or the framework.
    pub fn from_error(error: &pliron::result::Error) -> Option<&Self> {
        let mut source: &dyn std::error::Error = error;
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

    /// Returns the recovery action appropriate for this diagnostic.
    ///
    /// Operation wrappers retain the underlying XLAHLO recovery action, including
    /// evaluator categories. Unrecognized source types return `Unknown` rather
    /// than incorrectly classifying a foreign failure as invalid caller input.
    pub fn recovery_category(&self) -> RecoveryCategory {
        match self {
            Self::Unsupported { .. } | Self::UnsupportedCallback { .. } => {
                RecoveryCategory::Unsupported
            }
            Self::Operation { source, .. } => {
                let mut source: &(dyn std::error::Error + 'static) = source.as_ref();
                loop {
                    if let Some(diagnostic) = source.downcast_ref::<Self>() {
                        return diagnostic.recovery_category();
                    }
                    if let Some(evaluation) =
                        source.downcast_ref::<crate::evaluator::EvaluationError>()
                    {
                        return evaluation.recovery_category();
                    }
                    source = if let Some(envelope) = source.downcast_ref::<pliron::result::Error>()
                    {
                        envelope.err.as_ref()
                    } else if let Some(source) = source.source() {
                        source
                    } else {
                        return RecoveryCategory::Unknown;
                    };
                }
            }
            _ => RecoveryCategory::InvalidInput,
        }
    }
}

impl From<Diagnostic> for pliron::result::Error {
    fn from(error: Diagnostic) -> Self {
        pliron::verify_error_noloc!(error)
    }
}

macro_rules! argument_error {
    ($($ty:ty),+ $(,)?) => {
        $(impl From<$ty> for pliron::result::Error {
            fn from(error: $ty) -> Self {
                pliron::arg_error_noloc!(error)
            }
        })+
    };
}

argument_error!(
    crate::attributes::EnumParseError,
    crate::attributes::ChannelKindValueError,
    crate::types::ElementTypeParseError,
    crate::types::PrimitiveTypeNameError,
    crate::types::PrimitiveTypeValueError,
);
