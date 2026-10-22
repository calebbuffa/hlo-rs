//! Numerical mode, precision, comparison, and result-accuracy attributes.

use super::*;
use crate::Diagnostic;
use pliron::{common_traits::Verify, context::Context, derive::pliron_attr, result::Result};

#[pliron_attr(name = "xlahlo.random_distribution", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Distribution sampled by the random-number operation.
pub enum RandomDistribution {
    /// Uniform sampling over the operation's configured interval.
    Uniform,
    /// Normal sampling using the operation's mean and standard deviation.
    Normal,
}
#[pliron_attr(name = "xlahlo.random_algorithm", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Counter-based random-number generator selection.
pub enum RandomAlgorithm {
    /// Uses the backend's default generator.
    Default,
    /// Uses the Threefry counter-based generator.
    ThreeFry,
    /// Uses the Philox counter-based generator.
    Philox,
}
#[pliron_attr(name = "xlahlo.transpose_mode", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Matrix transform used by triangular solve.
pub enum Transpose {
    /// Uses the matrix without transposition.
    NoTranspose,
    /// Exchanges the matrix's row and column axes.
    Transpose,
    /// Transposes the matrix and conjugates complex elements.
    Adjoint,
}
record!(TriangularSolveOptions, "xlahlo.triangular_solve_options", "Triangular-system side, triangle, diagonal, and transpose policy.", {
    /// Solves `A * X = B` when true, or `X * A = B` otherwise.
    left_side: bool,
    /// Reads the lower triangle of `A` when true, or the upper triangle otherwise.
    lower: bool,
    /// Treats diagonal entries as one without reading their stored values.
    unit_diagonal: bool,
    /// Transform applied to `A` before solving.
    transpose_a: Transpose,
});
record!(CholeskyOptions, "xlahlo.cholesky_options", "Whether Cholesky returns the lower rather than upper triangle.", {
    /// Produces a lower-triangular factor when true, or an upper-triangular factor otherwise.
    lower: bool,
});
record!(ShuffleMode, "xlahlo.shuffle_mode", "Per-axis cyclic shifts applied by shuffle.", {
    /// Signed cyclic displacement along each logical axis.
    shifts: Axes,
});
record!(CubScanOptions, "xlahlo.cub_scan_options", "Scan tile geometry, reduction kind, and traversal direction.", {
    /// Number of elements in each scan vector.
    vector_length: i64,
    /// Number of elements in each tile row.
    row_length: i64,
    /// Number of elements in each tile column.
    column_length: i64,
    /// Reduction operation accumulated by the scan.
    kind: CubScanKind,
    /// Traverses each scan vector in reverse order.
    is_reverse: bool,
});
#[pliron_attr(name = "xlahlo.cub_scan_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Reduction kind of the specialized scan backend.
pub enum CubScanKind {
    /// Sentinel for an unselected or unsupported scan reduction.
    Invalid,
    /// Accumulates an arithmetic sum.
    Sum,
}
record!(SortOptions, "xlahlo.sort_options", "Whether sort orders values in descending order.", {
    /// Orders larger elements before smaller elements.
    descending: bool,
});

#[pliron_attr(name = "xlahlo.precision_algorithm", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Explicit numerical dot algorithm, including operand formats and accumulation modes.
pub enum Algorithm {
    /// Lets the backend select the dot algorithm.
    Unset,
    /// Multiplies supported 8-bit floating operands with F32 accumulation.
    DotAnyF8AnyF8F32,
    /// Uses a backend fast-accumulation path for 8-bit floating operands.
    DotAnyF8AnyF8F32FastAccum,
    /// Multiplies F16 operands with F16 accumulation.
    DotF16F16F16,
    /// Multiplies F16 operands with F32 accumulation.
    DotF16F16F32,
    /// Multiplies BF16 operands with BF16 accumulation.
    DotBF16BF16BF16,
    /// Multiplies BF16 operands with F32 accumulation.
    DotBF16BF16F32,
    /// Uses a three-component BF16 decomposition with F32 accumulation.
    DotBF16BF16F32X3,
    /// Uses a six-component BF16 decomposition with F32 accumulation.
    DotBF16BF16F32X6,
    /// Multiplies TF32 operands with F32 accumulation.
    DotTF32TF32F32,
    /// Uses a three-component TF32 decomposition with F32 accumulation.
    DotTF32TF32F32X3,
    /// Multiplies F32 operands with F32 accumulation.
    DotF32F32F32,
    /// Multiplies F64 operands with F64 accumulation.
    DotF64F64F64,
    /// Uses a nine-component BF16 decomposition with F32 accumulation.
    DotBF16BF16F32X9,
    /// Selects the backend's three-component BF16/FP8 dot algorithm.
    DotBF16BF16FP8X3,
    /// Selects the backend's four-component BF16/FP8 dot algorithm.
    DotBF16BF16FP8X4,
}
list!(
    OperandPrecision,
    "xlahlo.operand_precision",
    Precision,
    "Requested precision for each numerical operand."
);
record!(PrecisionConfig, "xlahlo.precision_config", "Per-operand precision and an optional explicit dot algorithm.", {
    /// Precision requests in numerical operand order.
    operand_precision: OperandPrecision,
    /// Explicit dot algorithm, or `Algorithm::Unset` for backend selection.
    algorithm: Algorithm,
});
#[pliron_attr(name = "xlahlo.comparison_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Signed, unsigned, partial floating-point, or total floating-point interpretation.
pub enum ComparisonKind {
    /// Floating-point comparison with partial ordering.
    Float,
    /// Floating-point comparison using a total order.
    TotalOrder,
    /// Signed integer interpretation of operand bits.
    Signed,
    /// Unsigned integer interpretation of operand bits.
    Unsigned,
}
record!(Comparison, "xlahlo.comparison", "Comparison direction and numeric interpretation.", {
    /// Relational predicate applied to the operands.
    direction: Direction,
    /// Numeric interpretation and ordering policy.
    comparison_kind: ComparisonKind,
});

#[pliron_attr(name = "xlahlo.result_accuracy", format)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Default, highest, or explicit numerical tolerances for an instruction result.
pub enum ResultAccuracy {
    /// Uses the backend's default accuracy contract.
    Default,
    /// Requests the highest accuracy supported by the backend.
    Highest,
    /// Specifies nonnegative absolute, relative, and ULP tolerances.
    Tolerance {
        /// Exact IEEE 754 binary64 encoding of the absolute tolerance.
        atol_bits: u64,
        /// Exact IEEE 754 binary64 encoding of the relative tolerance.
        rtol_bits: u64,
        /// Maximum permitted error in units in the last place.
        ulps: i64,
    },
}
impl ResultAccuracy {
    /// Construct nonnegative tolerances, rejecting NaNs and negative ULP counts.
    pub fn tolerance(atol: f64, rtol: f64, ulps: i64) -> Result<Self> {
        let value = Self::Tolerance {
            atol_bits: atol.to_bits(),
            rtol_bits: rtol.to_bits(),
            ulps,
        };
        value.validate()?;
        Ok(value)
    }
    /// Compare modes and numeric tolerances; positive and negative zero are equal.
    pub fn equal(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Tolerance {
                    atol_bits: a,
                    rtol_bits: b,
                    ulps: c,
                },
                Self::Tolerance {
                    atol_bits: d,
                    rtol_bits: e,
                    ulps: f,
                },
            ) => {
                f64::from_bits(*a) == f64::from_bits(*d)
                    && f64::from_bits(*b) == f64::from_bits(*e)
                    && c == f
            }
            _ => self == other,
        }
    }
    /// Reject negative or NaN tolerances and negative ULP counts.
    pub fn validate(&self) -> Result<()> {
        if let Self::Tolerance {
            atol_bits,
            rtol_bits,
            ulps,
        } = self
            && !(f64::from_bits(*atol_bits) >= 0.0
                && f64::from_bits(*rtol_bits) >= 0.0
                && *ulps >= 0)
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "result_accuracy",
                message: "accuracy tolerances must be nonnegative".into(),
            }));
        }
        Ok(())
    }
}
impl Verify for ResultAccuracy {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
