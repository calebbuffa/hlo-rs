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
use pliron::{
    attribute::Attribute, builtin::attributes::BoolAttr, context::Context, identifier::Identifier,
    op::Op, result::Result,
};

/// Signed integer metadata for numerical counts and group sizes.
#[pliron::derive::pliron_attr(name = "xlahlo.numerical_integer", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericalInteger(pub i64);

native_instruction!(DotOp, "xlahlo.dot", Dot, crate::verification::numerical::instruction,
    dimensions: DotDimensionNumbers, precision: PrecisionConfig);
native_instruction!(ReduceOp, "xlahlo.reduce", Reduce, crate::verification::numerical::instruction,
    dimensions: Axes, callees: Callees);
native_instruction!(ReduceWindowOp, "xlahlo.reduce_window", ReduceWindow, crate::verification::numerical::instruction,
    window: Window, callees: Callees);
native_instruction!(SelectAndScatterOp, "xlahlo.select_and_scatter", SelectAndScatter, crate::verification::numerical::instruction,
    window: Window, callees: Callees);
native_instruction!(ScanOp, "xlahlo.scan", Scan, crate::verification::numerical::instruction,
    dimensions: Axes, num_carries: NumericalInteger, is_associative: TriState, callees: Callees);

impl ScanOp {
    /// Whether to scan from the last element to the first. Missing metadata
    /// means forward traversal, matching `HloInstructionProto.is_reverse`.
    pub fn is_reverse(&self, ctx: &Context) -> Result<bool> {
        crate::verification::numerical::scan_direction(ctx, self.op)
    }

    /// Set traversal direction without changing carries or associativity.
    pub fn set_is_reverse(&self, ctx: &Context, is_reverse: bool) {
        self.op
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_is_reverse"), BoolAttr::new(is_reverse));
    }
}

native_instruction!(MapOp, "xlahlo.map", Map, crate::verification::numerical::instruction, callees: Callees);
native_instruction!(RaggedDotOp, "xlahlo.ragged_dot", RaggedDot, crate::verification::numerical::instruction,
    dimensions: RaggedDotDimensionNumbers, precision: PrecisionConfig);
native_instruction!(ScaledDotOp, "xlahlo.scaled_dot", ScaledDot, crate::verification::numerical::instruction,
    dimensions: DotDimensionNumbers, precision: PrecisionConfig);
native_instruction!(ConvolutionOp, "xlahlo.convolution", Convolution, crate::verification::numerical::instruction,
    dimensions: ConvolutionDimensionNumbers, window: Window,
    feature_groups: NumericalInteger, batch_groups: NumericalInteger, kind: ConvolutionKind,
    precision: PrecisionConfig);
native_instruction!(SortOp, "xlahlo.sort", Sort, crate::verification::numerical::instruction,
    dimensions: Axes, callees: Callees);

impl SortOp {
    /// Whether equal keys retain their input order. Missing metadata defaults
    /// to false, matching `HloInstructionProto.is_stable`.
    pub fn is_stable(&self, ctx: &Context) -> Result<bool> {
        crate::verification::numerical::sort_stability(ctx, self.op)
    }

    /// Set the sort's stability without changing its comparator or dimension.
    pub fn set_is_stable(&self, ctx: &Context, is_stable: bool) {
        self.op
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_is_stable"), BoolAttr::new(is_stable));
    }
}

native_instruction!(TopKOp, "xlahlo.top_k", TopK, crate::verification::numerical::instruction,
    k: NumericalInteger, options: SortOptions);

impl TopKOp {
    /// Whether equal values retain input order. Absent metadata defaults to
    /// false, matching the proto field rather than C++ `CreateTopK`'s default.
    pub fn is_stable(&self, ctx: &Context) -> Result<bool> {
        crate::verification::numerical::sort_stability(ctx, self.op)
    }

    /// Set stability independently of largest/smallest selection.
    pub fn set_is_stable(&self, ctx: &Context, is_stable: bool) {
        self.op
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_is_stable"), BoolAttr::new(is_stable));
    }
}

native_instruction!(FftOp, "xlahlo.fft", Fft, crate::verification::numerical::instruction,
    kind: FftType, lengths: Axes);
native_instruction!(TriangularSolveOp, "xlahlo.triangular_solve", TriangularSolve, crate::verification::numerical::instruction,
    options: TriangularSolveOptions);
native_instruction!(CholeskyOp, "xlahlo.cholesky", Cholesky, crate::verification::numerical::instruction,
    options: CholeskyOptions);
native_instruction!(BatchNormTrainingOp, "xlahlo.batch_norm_training", BatchNormTraining, crate::verification::numerical::instruction,
    options: BatchNormOptions);
native_instruction!(BatchNormInferenceOp, "xlahlo.batch_norm_inference", BatchNormInference, crate::verification::numerical::instruction,
    options: BatchNormOptions);
native_instruction!(BatchNormGradOp, "xlahlo.batch_norm_grad", BatchNormGrad, crate::verification::numerical::instruction,
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
