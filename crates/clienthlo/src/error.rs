use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryCategory {
    InvalidInput,
    Unsupported,
    InternalInvariant,
}

#[derive(Debug, Error)]
pub enum Diagnostic {
    #[error("{operation} operand element type is outside its domain")]
    InvalidDomain { operation: &'static str },
    #[error("operation {operation} at {location}: {reason}")]
    InvalidOperation {
        operation: String,
        location: String,
        reason: &'static str,
    },
}
impl Diagnostic {
    pub fn recovery_category(&self) -> RecoveryCategory {
        RecoveryCategory::InvalidInput
    }
}
impl From<Diagnostic> for pliron::result::Error {
    fn from(diagnostic: Diagnostic) -> Self {
        pliron::arg_error_noloc!(diagnostic)
    }
}
