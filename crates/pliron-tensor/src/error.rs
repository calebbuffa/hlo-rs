use thiserror::Error;

/// Recovery strategy for a tensor diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecoveryCategory {
    /// Correct the operation's types, attributes, or structure.
    InvalidInput,
}

/// Structured tensor verification failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Diagnostic {
    /// An operation violates its contract.
    #[error("operation {operation} at {location}: {reason}")]
    InvalidOperation {
        /// Operation name.
        operation: String,
        /// Source location.
        location: String,
        /// Violated constraint.
        reason: &'static str,
    },
}

impl Diagnostic {
    /// Finds a tensor diagnostic in a framework error or its source chain.
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

    /// Classifies how to recover from this failure.
    pub fn recovery_category(&self) -> RecoveryCategory {
        RecoveryCategory::InvalidInput
    }
}

impl From<Diagnostic> for pliron::result::Error {
    fn from(value: Diagnostic) -> Self {
        pliron::arg_error_noloc!(value)
    }
}
