//! Typed instruction geometry, numerical options, metadata, and collective attributes.
use crate::Diagnostic;

use pliron::{
    attribute::Attribute, common_traits::Verify, context::Context, derive::pliron_attr,
    identifier::Identifier, result::Result,
};
use std::collections::BTreeMap;

macro_rules! record {
    ($name:ident, $id:literal, $doc:literal, { $($field:ident: $ty:ty),* $(,)? }) => {
        #[pliron_attr(name = $id, format, verifier = "succ")]
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        #[doc = $doc]
        pub struct $name { $(pub $field: $ty),* }
    };
}

macro_rules! list {
    ($name:ident, $id:literal, $ty:ty, $doc:literal) => {
        #[pliron_attr(name = $id, format = "`[` vec($0, CharSpace(`,`)) `]`", verifier = "succ")]
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        #[doc = $doc]
        pub struct $name(pub Vec<$ty>);
    };
}

macro_rules! optional {
    ($name:ident, $id:literal, $ty:ty) => {
        #[pliron_attr(name = $id, format, verifier = "succ")]
        #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
        #[doc = concat!("Explicit absence or presence of a `", stringify!($ty), "` record.")]
        pub enum $name {
            #[default]
            Absent,
            Present($ty),
        }
        impl $name {
            /// Borrow the contained configuration, if present.
            pub fn as_ref(&self) -> Option<&$ty> {
                match self {
                    Self::Absent => None,
                    Self::Present(value) => Some(value),
                }
            }
            /// Whether a configuration is present.
            pub fn is_some(&self) -> bool {
                self.as_ref().is_some()
            }
            /// Whether the configuration is absent.
            pub fn is_none(&self) -> bool {
                self.as_ref().is_none()
            }
        }
        impl From<$ty> for $name {
            fn from(value: $ty) -> Self {
                Self::Present(value)
            }
        }
    };
}

list!(
    Strings,
    "xlahlo.strings",
    String,
    "Ordered string values in native attribute assembly."
);
list!(
    Bytes,
    "xlahlo.bytes",
    u8,
    "Exact uninterpreted bytes for instruction payloads."
);
list!(
    Bools,
    "xlahlo.bools",
    bool,
    "Ordered boolean values in native attribute assembly."
);
record!(StringPair, "xlahlo.string_pair", "One frontend attribute key and string value.", { key: String, value: String });
list!(
    FrontendAttributes,
    "xlahlo.frontend_attributes",
    StringPair,
    "Frontend key/value annotations with last-write-wins lookup."
);

impl FrontendAttributes {
    /// Normalize duplicate keys by their last value and sort keys lexicographically.
    pub fn new(values: impl IntoIterator<Item = (String, String)>) -> Self {
        let map: BTreeMap<_, _> = values.into_iter().collect();
        Self(
            map.into_iter()
                .map(|(key, value)| StringPair { key, value })
                .collect(),
        )
    }
    /// Borrow the last value for a key without normalizing or copying the record.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .rev()
            .find(|pair| pair.key == name)
            .map(|pair| pair.value.as_str())
    }
}

record!(WindowDimension, "xlahlo.window_dimension", "Window size, traversal, padding, dilation, and reversal for one axis.", {
    size: i64, stride: i64, padding_low: i64, padding_high: i64,
    window_dilation: i64, base_dilation: i64, window_reversal: bool,
});
list!(
    Window,
    "xlahlo.window",
    WindowDimension,
    "Per-axis dimensions of a convolution or reduction window."
);
record!(PaddingConfigDimension, "xlahlo.padding_dimension", "Low, high, and interior padding for one logical axis.", {
    edge_padding_low: i64, edge_padding_high: i64, interior_padding: i64,
});
list!(
    PaddingConfig,
    "xlahlo.padding_config",
    PaddingConfigDimension,
    "Per-axis low, high, and interior padding."
);
record!(ConvolutionDimensionNumbers, "xlahlo.convolution_dimensions", "Input, kernel, and output axis roles for convolution.", {
    input_batch_dimension: i64, input_feature_dimension: i64, input_spatial_dimensions: Axes,
    kernel_input_feature_dimension: i64, kernel_output_feature_dimension: i64, kernel_spatial_dimensions: Axes,
    output_batch_dimension: i64, output_feature_dimension: i64, output_spatial_dimensions: Axes,
});
record!(RaggedDotDimensionNumbers, "xlahlo.ragged_dot_dimensions", "Dot axis roles plus ragged and group dimensions.", {
    dot_dimension_numbers: DotDimensionNumbers, lhs_ragged_dimensions: Axes, rhs_group_dimensions: Axes,
});
record!(DotDimensionNumbers, "xlahlo.dot_dimension_numbers", "Contracting and batch axes of both dot operands.", {
    lhs_contracting_dimensions: Axes, rhs_contracting_dimensions: Axes,
    lhs_batch_dimensions: Axes, rhs_batch_dimensions: Axes,
});
record!(GatherDimensionNumbers, "xlahlo.gather_dimension_numbers", "Gather output offsets, collapsed axes, batching, and index mapping.", {
    offset_dims: Axes, collapsed_slice_dims: Axes, start_index_map: Axes,
    index_vector_dim: i64, operand_batching_dims: Axes, start_indices_batching_dims: Axes,
});
record!(ScatterDimensionNumbers, "xlahlo.scatter_dimension_numbers", "Scatter update windows, inserted axes, batching, and operand mapping.", {
    update_window_dims: Axes, inserted_window_dims: Axes, scatter_dims_to_operand_dims: Axes,
    index_vector_dim: i64, input_batching_dims: Axes, scatter_indices_batching_dims: Axes,
});
record!(TensorSparsityConfig, "xlahlo.tensor_sparsity", "Structured sparsity pattern and metadata operand index for one tensor.", {
    num_non_zero: i64, block_size: i64, dimension: i64, stride: i64, idx: i32,
});
optional!(
    OptionalTensorSparsityConfig,
    "xlahlo.optional_tensor_sparsity",
    TensorSparsityConfig
);
record!(SparsityConfig, "xlahlo.sparsity", "Independent optional sparsity configurations for both dot operands.", { lhs: OptionalTensorSparsityConfig, rhs: OptionalTensorSparsityConfig });
record!(TensorBlockScalingConfig, "xlahlo.tensor_block_scaling", "Scale and optional zero operands with block traversal strides and steps.", {
    scale_idx: i32, zero_idx: OptionalInteger, strides: Axes, steps: Axes,
});
optional!(
    OptionalTensorBlockScalingConfig,
    "xlahlo.optional_tensor_block_scaling",
    TensorBlockScalingConfig
);
record!(BlockScalingConfig, "xlahlo.block_scaling", "Independent optional block-scaling configurations for both dot operands.", { lhs: OptionalTensorBlockScalingConfig, rhs: OptionalTensorBlockScalingConfig });
impl Default for SparsityConfig {
    fn default() -> Self {
        Self {
            lhs: OptionalTensorSparsityConfig::Absent,
            rhs: OptionalTensorSparsityConfig::Absent,
        }
    }
}
impl Default for BlockScalingConfig {
    fn default() -> Self {
        Self {
            lhs: OptionalTensorBlockScalingConfig::Absent,
            rhs: OptionalTensorBlockScalingConfig::Absent,
        }
    }
}

#[pliron_attr(name = "xlahlo.fft_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Forward, inverse, real-input, and real-output Fourier transform modes.
pub enum FftType {
    FFT,
    IFFT,
    RFFT,
    IRFFT,
}
#[pliron_attr(name = "xlahlo.padding_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Automatic window padding policy.
pub enum PaddingType {
    Invalid,
    Valid,
    Same,
}
#[pliron_attr(name = "xlahlo.tri_state", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Boolean option with an explicit unspecified state.
pub enum TriState {
    Unspecified,
    True,
    False,
}
#[pliron_attr(name = "xlahlo.convolution_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Forward, input-gradient, weight-gradient, or unselected convolution role.
pub enum ConvolutionKind {
    Unset,
    Fprop,
    Dgrad,
    Wgrad,
}
#[pliron_attr(name = "xlahlo.custom_call_schedule", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Scheduling constraint applied to a custom call.
pub enum CustomCallSchedule {
    None,
    Latest,
    Earliest,
}
#[pliron_attr(name = "xlahlo.custom_call_api_version", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// ABI version used to invoke a custom-call target.
pub enum CustomCallApiVersion {
    Unspecified,
    Original,
    StatusReturning,
    StatusReturningUnified,
    TypedFfi,
}
#[pliron_attr(name = "xlahlo.collective_op_group_mode", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Interpretation of device IDs in collective replica groups.
pub enum CollectiveOpGroupMode {
    CrossReplica,
    CrossPartition,
    CrossReplicaAndPartition,
    FlattenedID,
}
#[pliron_attr(name = "xlahlo.channel_handle", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Communication channel identity and channel kind.
pub struct ChannelHandle {
    pub handle: i64,
    pub kind: i64,
}
#[pliron_attr(name = "xlahlo.payload", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Absent metadata, inline exact bytes, or an index into module metadata payloads.
pub enum Payload {
    Absent,
    Value(Bytes),
    Id(i64),
}
record!(SliceDimensions, "xlahlo.slice_dimensions", "Start, exclusive limit, and stride along one slice axis.", { start: i64, limit: i64, stride: i64 });
record!(ReducePrecision, "xlahlo.reduce_precision", "Exponent and mantissa widths retained by reduce-precision.", { exponent_bits: i32, mantissa_bits: i32 });
record!(BatchNormOptions, "xlahlo.batch_norm_options", "Batch normalization feature axis and exact f32 epsilon encoding.", { epsilon_bits: u32, feature_index: i64 });
record!(KnownInitStep, "xlahlo.known_init_step", "Known initial value and increment of a loop induction variable.", { init: i64, step: i64 });
#[pliron_attr(name = "xlahlo.optional_integer", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// An integer whose absence is distinct from every numeric value.
pub enum OptionalInteger {
    Absent,
    Value(i64),
}
impl OptionalInteger {
    /// Borrow the integer, if present.
    pub fn as_ref(&self) -> Option<&i64> {
        match self {
            Self::Absent => None,
            Self::Value(value) => Some(value),
        }
    }
    /// Whether an integer is present.
    pub fn is_some(&self) -> bool {
        self.as_ref().is_some()
    }
    /// Whether the integer is absent.
    pub fn is_none(&self) -> bool {
        self.as_ref().is_none()
    }
}
record!(DynamicVariable, "xlahlo.dynamic_variable", "A loop-carried tuple member with optional known initial value and step.", {
    tuple_index: i64, init: OptionalInteger, step: OptionalInteger,
});
list!(
    DynamicVariables,
    "xlahlo.dynamic_variables",
    DynamicVariable,
    "Loop-carried variables described by a while-loop backend configuration."
);
list!(
    KnownInitSteps,
    "xlahlo.known_init_steps",
    KnownInitStep,
    "Optional singleton initial-value and step record for a loop."
);
record!(WhileLoopBackendConfig, "xlahlo.while_loop_backend_config", "Known trip-count and induction-variable information for a while loop.", {
    known_trip_count: OptionalInteger, known_init_step: KnownInitSteps,
    known_induction_variable: OptionalInteger, dynamic_variables: DynamicVariables,
});
#[pliron_attr(name = "xlahlo.random_distribution", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Distribution sampled by the random-number operation.
pub enum RandomDistribution {
    Uniform,
    Normal,
}
#[pliron_attr(name = "xlahlo.random_algorithm", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Counter-based random-number generator selection.
pub enum RandomAlgorithm {
    Default,
    ThreeFry,
    Philox,
}
#[pliron_attr(name = "xlahlo.transpose_mode", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Matrix transform used by triangular solve.
pub enum Transpose {
    NoTranspose,
    Transpose,
    Adjoint,
}
record!(TriangularSolveOptions, "xlahlo.triangular_solve_options", "Triangular-system side, triangle, diagonal, and transpose policy.", {
    left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: Transpose,
});
record!(CholeskyOptions, "xlahlo.cholesky_options", "Whether Cholesky returns the lower rather than upper triangle.", { lower: bool });
record!(ShuffleMode, "xlahlo.shuffle_mode", "Per-axis cyclic shifts applied by shuffle.", { shifts: Axes });
record!(CubScanOptions, "xlahlo.cub_scan_options", "Scan tile geometry, reduction kind, and traversal direction.", {
    vector_length: i64, row_length: i64, column_length: i64, kind: CubScanKind, is_reverse: bool,
});
#[pliron_attr(name = "xlahlo.cub_scan_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Reduction kind of the specialized scan backend.
pub enum CubScanKind {
    Invalid,
    Sum,
}
record!(SortOptions, "xlahlo.sort_options", "Whether sort orders values in descending order.", { descending: bool });
record!(SourceTarget, "xlahlo.source_target", "One directed device edge of a collective permute.", { source: i64, target: i64 });
list!(
    SourceTargetPairs,
    "xlahlo.source_target_pairs",
    SourceTarget,
    "Directed device edges of a collective permute."
);
list!(
    ReplicaGroup,
    "xlahlo.replica_group",
    i64,
    "Device identifiers participating in one collective group."
);
list!(
    ReplicaGroups,
    "xlahlo.replica_groups",
    ReplicaGroup,
    "Explicit collective groups, preserving device order."
);
list!(
    ParameterReplication,
    "xlahlo.parameter_replication",
    bool,
    "Replication flags for parameter shape leaves."
);
record!(OutputOperandAliasing, "xlahlo.output_operand_aliasing", "An output subshape aliasing a particular operand subshape.", {
    output_shape_index: Axes, operand_index: i64, operand_shape_index: Axes,
});
list!(
    OutputOperandAliases,
    "xlahlo.output_operand_aliases",
    OutputOperandAliasing,
    "Declared output-to-operand buffer aliases."
);

#[pliron_attr(name = "xlahlo.precision_algorithm", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Explicit numerical dot algorithm, including operand formats and accumulation modes.
pub enum Algorithm {
    Unset,
    DotAnyF8AnyF8F32,
    DotAnyF8AnyF8F32FastAccum,
    DotF16F16F16,
    DotF16F16F32,
    DotBF16BF16BF16,
    DotBF16BF16F32,
    DotBF16BF16F32X3,
    DotBF16BF16F32X6,
    DotTF32TF32F32,
    DotTF32TF32F32X3,
    DotF32F32F32,
    DotF64F64F64,
    DotBF16BF16F32X9,
    DotBF16BF16FP8X3,
    DotBF16BF16FP8X4,
}
list!(
    OperandPrecision,
    "xlahlo.operand_precision",
    Precision,
    "Requested precision for each numerical operand."
);
record!(PrecisionConfig, "xlahlo.precision_config", "Per-operand precision and an optional explicit dot algorithm.", {
    operand_precision: OperandPrecision, algorithm: Algorithm,
});
#[pliron_attr(name = "xlahlo.comparison_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Signed, unsigned, partial floating-point, or total floating-point interpretation.
pub enum ComparisonType {
    Float,
    TotalOrder,
    Signed,
    Unsigned,
}
record!(Comparison, "xlahlo.comparison", "Comparison direction and numeric interpretation.", { direction: Direction, comparison_type: ComparisonType });

impl Direction {
    /// Negate the comparison predicate.
    pub fn inverse(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::Ge => Self::Lt,
        }
    }
    /// Reverse the comparison operands without changing the result.
    pub fn converse(self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Gt => Self::Lt,
            Self::Ge => Self::Le,
            same => same,
        }
    }
}

// Floating metadata uses exact encodings for pliron's Eq/Hash contract.
record!(ProfileInfo, "xlahlo.profile_info", "Instruction profile provenance and exact f64 relative-speedup encoding.", {
    profile_type: Axes, relative_speedup_bits: u64, profile_source: i32,
    compilation_event: i32, profile_generation_strategy: i32,
});
list!(
    ProfileInfos,
    "xlahlo.profile_infos",
    ProfileInfo,
    "Optional singleton instruction profile record."
);
#[pliron_attr(name = "xlahlo.op_metadata", format)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Instruction identity, source location, profiling, scheduling, and payload metadata.
pub struct OpMetadata {
    pub op_type: String,
    pub op_name: String,
    pub source_file: String,
    pub source_line: i32,
    pub source_end_line: i32,
    pub source_column: i32,
    pub source_end_column: i32,
    pub profile_type: Axes,
    pub size_of_generated_code_in_bytes: i64,
    pub size_of_memory_working_set_in_bytes: i64,
    pub profile_info: ProfileInfos,
    pub deduplicated_name: String,
    pub stack_frame_id: i32,
    pub scheduling_name: String,
    pub metadata_payload: Payload,
}
impl Verify for OpMetadata {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.profile_info.0.len() > 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "op_metadata",
                message: "multiple instruction profile records".into(),
            }));
        }
        self.metadata_payload.verify(ctx)
    }
}
list!(
    Metadata,
    "xlahlo.metadata",
    OpMetadata,
    "Instruction metadata records associated with a sharding annotation."
);
impl Default for OpMetadata {
    fn default() -> Self {
        Self {
            op_type: String::new(),
            op_name: String::new(),
            source_file: String::new(),
            source_line: 0,
            source_end_line: 0,
            source_column: 0,
            source_end_column: 0,
            profile_type: Axes(Vec::new()),
            size_of_generated_code_in_bytes: 0,
            size_of_memory_working_set_in_bytes: 0,
            profile_info: ProfileInfos::default(),
            deduplicated_name: String::new(),
            stack_frame_id: 0,
            scheduling_name: String::new(),
            metadata_payload: Payload::Absent,
        }
    }
}
impl OpMetadata {
    /// Construct the named-sharding marker for a non-sum reduction.
    pub(crate) fn reduction(max: bool) -> Self {
        Self {
            op_type: "sdy::reduction_op".into(),
            op_name: if max { "MAX" } else { "MIN" }.into(),
            ..Self::default()
        }
    }
}
record!(Statistic, "xlahlo.statistic", "Named visualization statistic with an exact f64 value encoding.", { stat_name: String, stat_val_bits: u64 });
list!(
    Statistics,
    "xlahlo.statistics",
    Statistic,
    "Named visualization statistics."
);
record!(StatisticsViz, "xlahlo.statistics_viz", "Visualization statistics and the selected statistic index.", { stat_index_to_visualize: i64, statistics: Statistics });

#[pliron_attr(name = "xlahlo.result_accuracy", format)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Default, highest, or explicit numerical tolerances for an instruction result.
pub enum ResultAccuracy {
    Default,
    Highest,
    Tolerance {
        atol_bits: u64,
        rtol_bits: u64,
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

pub(crate) fn register(ctx: &mut Context) {
    macro_rules! attrs { ($($ty:ty),* $(,)?) => { $(<$ty>::register::<$ty>(ctx);)* }; }
    attrs!(
        Axes,
        Index,
        Direction,
        ComparisonOrder,
        Precision,
        FusionKind,
        Slice,
        Callees,
        IndexFlags,
        Strings,
        Bytes,
        Bools,
        StringPair,
        FrontendAttributes,
        WindowDimension,
        Window,
        PaddingConfigDimension,
        PaddingConfig,
        ConvolutionDimensionNumbers,
        RaggedDotDimensionNumbers,
        DotDimensionNumbers,
        GatherDimensionNumbers,
        ScatterDimensionNumbers,
        TensorSparsityConfig,
        OptionalTensorSparsityConfig,
        SparsityConfig,
        TensorBlockScalingConfig,
        OptionalTensorBlockScalingConfig,
        BlockScalingConfig,
        FftType,
        PaddingType,
        TriState,
        ConvolutionKind,
        CustomCallSchedule,
        CustomCallApiVersion,
        CollectiveOpGroupMode,
        ChannelHandle,
        Payload,
        SliceDimensions,
        ReducePrecision,
        BatchNormOptions,
        KnownInitStep,
        OptionalInteger,
        DynamicVariable,
        DynamicVariables,
        KnownInitSteps,
        WhileLoopBackendConfig,
        RandomDistribution,
        RandomAlgorithm,
        Transpose,
        TriangularSolveOptions,
        CholeskyOptions,
        ShuffleMode,
        CubScanOptions,
        CubScanKind,
        SortOptions,
        SourceTarget,
        SourceTargetPairs,
        ReplicaGroup,
        ReplicaGroups,
        ParameterReplication,
        OutputOperandAliasing,
        OutputOperandAliases,
        Algorithm,
        OperandPrecision,
        PrecisionConfig,
        ComparisonType,
        Comparison,
        ProfileInfo,
        ProfileInfos,
        OpMetadata,
        Metadata,
        Statistic,
        Statistics,
        StatisticsViz,
        ResultAccuracy,
    );
}

#[pliron_attr(
    name = "xlahlo.axes",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Ordered dimension indices or shape-index components.
pub struct Axes(pub Vec<i64>);

#[pliron_attr(name = "xlahlo.index", format = "$0", verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Nonnegative instruction index, such as a parameter number or tuple element.
pub struct Index(pub u32);

#[pliron_attr(name = "xlahlo.direction", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Relational predicate applied by a comparison.
pub enum Direction {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[pliron_attr(name = "xlahlo.order", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Partial, total, or weak ordering policy.
pub enum ComparisonOrder {
    Partial,
    Total,
    Weak,
}

#[pliron_attr(name = "xlahlo.precision", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Requested operand precision level.
pub enum Precision {
    Default,
    High,
    Highest,
}

#[pliron_attr(name = "xlahlo.fusion_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Execution organization of a fused computation.
pub enum FusionKind {
    Loop,
    Input,
    Output,
    Custom,
}

#[pliron_attr(
    name = "xlahlo.slice",
    format = "`<` $starts `,` $limits `,` $strides `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Per-axis starts, exclusive limits, and strides for static slicing.
pub struct Slice {
    pub starts: Axes,
    pub limits: Axes,
    pub strides: Axes,
}

#[pliron_attr(
    name = "xlahlo.callees",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Ordered symbols naming computations called by an instruction.
pub struct Callees(pub Vec<Identifier>);

#[pliron_attr(
    name = "xlahlo.flags",
    format = "`<` $sorted `,` $unique `>`",
    verifier = "succ"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Sortedness and uniqueness guarantees for gather or scatter indices.
pub struct IndexFlags {
    pub sorted: bool,
    pub unique: bool,
}
