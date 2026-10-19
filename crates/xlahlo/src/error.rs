//! Recoverable dialect errors. Framework traits retain pliron's error boundary.

/// Typed details carried by [`pliron::result::Error::err`].
///
/// Public fallible APIs use [`pliron::result::Result`]. Inspect this payload with
/// `error.err.downcast_ref::<Diagnostic>()` rather than matching diagnostic text.
/// The framework envelope retains its category, source location and backtrace.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Diagnostic {
    /// The opcode spelling is not part of the supported catalog.
    #[error("unknown HLO opcode {name:?}; use an opcode from Opcode::ALL")]
    UnknownOpcode { name: String },
    /// An axis is negative or outside the shape's rank.
    #[error("axis {axis} is outside the valid range 0..{rank}")]
    InvalidAxis { axis: i64, rank: usize },
    /// A symbolic computation reference cannot be resolved in its module.
    #[error("computation {name:?} does not exist; declare it in the module")]
    UnknownComputation { name: String },
    /// A shape, layout, literal, or primitive type violates its contract.
    #[error("invalid HLO shape or value: {message}")]
    InvalidShape { message: String },
    /// An instruction attribute is absent, ill-typed, or inconsistent.
    #[error("invalid HLO attribute {attribute}: {message}")]
    InvalidAttribute {
        attribute: &'static str,
        message: String,
    },
    /// Operand types or geometry do not admit the requested instruction.
    #[error("HLO inference failed: {message}")]
    Inference { message: String },
    /// An operation or graph violates an HLO verification rule.
    #[error("invalid HLO IR: {message}")]
    Verification { message: String },
    /// Context identifying the instruction at which verification failed.
    #[error("while verifying {operation}: {source}")]
    Operation {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
