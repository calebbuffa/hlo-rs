use thiserror::Error;

/// Broad recovery strategy for an `arith` diagnostic.
///
/// Use this classification to distinguish invalid caller input from missing
/// implementation support or an internal bug, rather than parsing error text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecoveryCategory {
    /// Correct the operand types, attributes, or operation structure and retry.
    InvalidInput,
    /// Choose a supported representation before retrying.
    Unsupported,
    /// Report an implementation bug; changing otherwise valid input is not a fix.
    InternalInvariant,
}

/// Structured failures raised by `arith` construction and verification.
///
/// These errors are wrapped in [`pliron::result::Error`]. Use
/// [`Diagnostic::from_error`] to recover the structured details.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Diagnostic {
    /// An operation's attributes, types, or structure violate its contract.
    #[error("operation {operation} at {location}: {reason}")]
    InvalidOperation {
        /// Operation name identifying the failed check.
        operation: String,
        /// Rendered source location, or a placeholder when no source is available.
        location: String,
        /// Specific violated constraint; correct the IR before retrying.
        reason: &'static str,
    },
    /// Valid IR that has no representation in the requested output.
    #[error("cannot represent {operation}: {reason}")]
    Unsupported {
        /// Operation or attribute that could not be represented.
        operation: String,
        /// Missing capability.
        reason: &'static str,
    },
}

impl Diagnostic {
    /// Finds the first `arith` diagnostic in an error and its source chain.
    ///
    /// Traverses framework `err` payloads as well as standard error sources.
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
    pub fn recovery_category(&self) -> RecoveryCategory {
        match self {
            Self::InvalidOperation { .. } => RecoveryCategory::InvalidInput,
            Self::Unsupported { .. } => RecoveryCategory::Unsupported,
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
