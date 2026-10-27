use thiserror::Error;

/// Broad recovery strategy for a ClientHLO diagnostic.
///
/// Use this classification to distinguish invalid caller input from missing
/// implementation support or an internal bug, rather than parsing error text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecoveryCategory {
    /// Correct the operand types, attributes, or operation structure and retry.
    InvalidInput,
    /// Choose a supported representation or lower the operation before retrying.
    Unsupported,
    /// Report an implementation bug; changing otherwise valid input is not a fix.
    InternalInvariant,
}

/// Structured failures raised by ClientHLO inference and verification.
///
/// These errors are wrapped in [`pliron::result::Error`]. Inspect its underlying
/// error with `downcast_ref::<Diagnostic>()` to recover the structured details.
/// Inference and verification retain sources when adding operation context.
/// Legalization also distinguishes valid but unsupported representations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Diagnostic {
    /// Valid CHLO for which the selected legalization is not implemented.
    #[error("cannot legalize {operation} at {location}: {reason}")]
    UnsupportedLowering {
        /// Operation that could not be lowered.
        operation: String,
        /// Rendered source location.
        location: String,
        /// Required representation or missing decomposition.
        reason: &'static str,
    },
    /// An operand does not belong to the element-type family accepted by an operation.
    #[error("{operation} operand element type is outside its domain")]
    InvalidDomain {
        /// CHLO operation name whose element-type constraint was violated.
        operation: &'static str,
    },
    /// An operation's attributes, types, or structure violate its contract.
    #[error("operation {operation} at {location}: {reason}")]
    InvalidOperation {
        /// Operation name or attribute description identifying the failed check.
        operation: String,
        /// Rendered source location, or a placeholder when no source is available.
        location: String,
        /// Specific violated constraint; correct the IR before retrying.
        reason: &'static str,
    },
    /// An inference error annotated with the operation being verified.
    #[error("operation {operation} at {location}: {reason}: {source}")]
    OperationFailure {
        /// CHLO operation whose verification requested inference.
        operation: &'static str,
        /// Rendered source location of that operation.
        location: String,
        /// Constraint being checked when inference failed.
        reason: &'static str,
        /// Original inference error, retained for typed inspection.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
impl Diagnostic {
    /// Finds the first ClientHLO diagnostic in an error and its source chain.
    ///
    /// Traverses framework `err` payloads as well as standard error sources.
    /// Returns `None` when the error has no ClientHLO diagnostic.
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

    /// Classifies the action a caller should take before retrying.
    ///
    /// Unsupported lowering requires a supported representation or legalization
    /// path. Other current variants identify invalid input. This classification
    /// does not imply that earlier IR changes have been rolled back.
    pub fn recovery_category(&self) -> RecoveryCategory {
        match self {
            Self::UnsupportedLowering { .. } => RecoveryCategory::Unsupported,
            _ => RecoveryCategory::InvalidInput,
        }
    }
}

impl From<crate::attributes::EnumParseError> for pliron::result::Error {
    fn from(error: crate::attributes::EnumParseError) -> Self {
        pliron::arg_error_noloc!(error)
    }
}

impl From<crate::attributes::EnumValueError> for pliron::result::Error {
    fn from(error: crate::attributes::EnumValueError) -> Self {
        pliron::arg_error_noloc!(error)
    }
}

impl From<Diagnostic> for pliron::result::Error {
    fn from(diagnostic: Diagnostic) -> Self {
        pliron::arg_error_noloc!(diagnostic)
    }
}
