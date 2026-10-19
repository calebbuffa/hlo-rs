//! Native numerical HLO operations and their typed construction attributes.
//!
//! Constructors accept explicit result types; use [`crate::inference::numerical`]
//! to infer them and [`crate::verification`] to validate complete computations.
//! Dot, ragged dot, scaled dot, and convolution share [`PrecisionConfig`].
use crate::attributes::{
    Axes, BatchNormOptions, Callees, CholeskyOptions, ConvolutionDimensionNumbers, ConvolutionKind,
    DotDimensionNumbers, FftType, PrecisionConfig, RaggedDotDimensionNumbers, SortOptions,
    TriState, TriangularSolveOptions, Window,
};
use pliron::{attribute::Attribute, context::Context, op::Op};

/// Signed integer metadata for numerical counts and group sizes.
#[pliron::derive::pliron_attr(name = "hlo.numerical_integer", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericalInteger(pub i64);

native_instruction!(DotOp, "hlo.dot", Dot, crate::verification::numerical::instruction,
    dimensions: DotDimensionNumbers, precision: PrecisionConfig);
native_instruction!(ReduceOp, "hlo.reduce", Reduce, crate::verification::numerical::instruction,
    dimensions: Axes, callees: Callees);
native_instruction!(ReduceWindowOp, "hlo.reduce_window", ReduceWindow, crate::verification::numerical::instruction,
    window: Window, callees: Callees);
native_instruction!(SelectAndScatterOp, "hlo.select_and_scatter", SelectAndScatter, crate::verification::numerical::instruction,
    window: Window, callees: Callees);
native_instruction!(ScanOp, "hlo.scan", Scan, crate::verification::numerical::instruction,
    dimensions: Axes, num_carries: NumericalInteger, is_associative: TriState, callees: Callees);
native_instruction!(MapOp, "hlo.map", Map, crate::verification::numerical::instruction, callees: Callees);
native_instruction!(RaggedDotOp, "hlo.ragged_dot", RaggedDot, crate::verification::numerical::instruction,
    dimensions: RaggedDotDimensionNumbers, precision: PrecisionConfig);
native_instruction!(ScaledDotOp, "hlo.scaled_dot", ScaledDot, crate::verification::numerical::instruction,
    dimensions: DotDimensionNumbers, precision: PrecisionConfig);
native_instruction!(ConvolutionOp, "hlo.convolution", Convolution, crate::verification::numerical::instruction,
    dimensions: ConvolutionDimensionNumbers, window: Window,
    feature_groups: NumericalInteger, batch_groups: NumericalInteger, kind: ConvolutionKind,
    precision: PrecisionConfig);
native_instruction!(SortOp, "hlo.sort", Sort, crate::verification::numerical::instruction,
    dimensions: Axes, callees: Callees);
native_instruction!(TopKOp, "hlo.top_k", TopK, crate::verification::numerical::instruction,
    k: NumericalInteger, options: SortOptions);
native_instruction!(FftOp, "hlo.fft", Fft, crate::verification::numerical::instruction,
    kind: FftType, lengths: Axes);
native_instruction!(TriangularSolveOp, "hlo.triangular_solve", TriangularSolve, crate::verification::numerical::instruction,
    options: TriangularSolveOptions);
native_instruction!(CholeskyOp, "hlo.cholesky", Cholesky, crate::verification::numerical::instruction,
    options: CholeskyOptions);
native_instruction!(BatchNormTrainingOp, "hlo.batch_norm_training", BatchNormTraining, crate::verification::numerical::instruction,
    options: BatchNormOptions);
native_instruction!(BatchNormInferenceOp, "hlo.batch_norm_inference", BatchNormInference, crate::verification::numerical::instruction,
    options: BatchNormOptions);
native_instruction!(BatchNormGradOp, "hlo.batch_norm_grad", BatchNormGrad, crate::verification::numerical::instruction,
    options: BatchNormOptions);

/// Register the complete numerical family and its signed integer attribute.
pub(crate) fn register(ctx: &mut Context) {
    NumericalInteger::register(ctx);
    macro_rules! register { ($($op:ty),* $(,)?) => { $(<$op>::register(ctx);)* }; }
    register!(
        DotOp,
        ReduceOp,
        ReduceWindowOp,
        SelectAndScatterOp,
        ScanOp,
        MapOp,
        RaggedDotOp,
        ScaledDotOp,
        ConvolutionOp,
        SortOp,
        TopKOp,
        FftOp,
        TriangularSolveOp,
        CholeskyOp,
        BatchNormTrainingOp,
        BatchNormInferenceOp,
        BatchNormGradOp
    );
}
