//! Typed instruction geometry, numerical options, metadata, and collective attributes.
//!
//! These records implement pliron's attribute parsing, printing, equality,
//! and hashing contracts. Floating-point metadata therefore stores exact
//! encodings rather than host floats. Explicit optional-record attributes
//! represent presence in native IR; convert them to or from Rust [`Option`]
//! when working outside the attribute system.
use pliron::{attribute::Attribute, context::Context};

macro_rules! record {
    ($name:ident, $id:literal, $doc:literal, { $($(#[$field_doc:meta])* $field:ident: $ty:ty),* $(,)? }) => {
        #[pliron_attr(name = $id, format, verifier = "succ")]
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        #[doc = $doc]
        pub struct $name { $($(#[$field_doc])* pub $field: $ty),* }
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
            /// No configuration is specified.
            #[default]
            Absent,
            /// The explicitly specified configuration.
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
        impl From<Option<$ty>> for $name {
            fn from(value: Option<$ty>) -> Self {
                match value {
                    Some(value) => Self::Present(value),
                    None => Self::Absent,
                }
            }
        }
        impl From<$name> for Option<$ty> {
            fn from(value: $name) -> Self {
                match value {
                    $name::Present(value) => Some(value),
                    $name::Absent => None,
                }
            }
        }
    };
}

mod common;
mod control;
pub mod enums;
mod geometry;
pub mod keys;
mod metadata;
mod numerical;

pub use common::{
    Axes, Bools, Bytes, Callees, ComparisonOrder, Direction, FrontendAttributes, FusionKind, Index,
    IndexFlags, Precision, Slice, StringPair, Strings,
};
pub use control::{
    BatchNormOptions, ChannelHandle, ChannelKind, ChannelKindValueError, CollectiveOpGroupMode,
    CustomCallApiVersion, CustomCallSchedule, DynamicVariable, DynamicVariables, KnownInitStep,
    KnownInitSteps, OptionalInteger, OutputOperandAliases, OutputOperandAliasing,
    ParameterReplication, Payload, ReducePrecision, ReplicaGroup, ReplicaGroups, SliceDimensions,
    SourceTarget, SourceTargetPairs, WhileLoopBackendConfig,
};
pub use enums::*;
pub use geometry::{
    BlockScalingConfig, ConvolutionDimensionNumbers, ConvolutionKind, DotDimensionNumbers, FftKind,
    GatherDimensionNumbers, OptionalTensorBlockScalingConfig, OptionalTensorSparsityConfig,
    PaddingConfig, PaddingConfigDimension, PaddingKind, RaggedDotDimensionNumbers,
    ScatterDimensionNumbers, SparsityConfig, TensorBlockScalingConfig, TensorSparsityConfig,
    TriState, Window, WindowDimension,
};
pub use metadata::{
    Metadata, OpMetadata, ProfileInfo, ProfileInfos, Statistic, Statistics, StatisticsViz,
};
pub use numerical::{
    Algorithm, CholeskyOptions, Comparison, ComparisonKind, CubScanKind, CubScanOptions,
    OperandPrecision, PrecisionConfig, RandomAlgorithm, RandomDistribution, ResultAccuracy,
    ShuffleMode, SortOptions, Transpose, TriangularSolveOptions,
};

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
        FftKind,
        PaddingKind,
        TriState,
        ConvolutionKind,
        CustomCallSchedule,
        CustomCallApiVersion,
        CollectiveOpGroupMode,
        ChannelKind,
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
        ComparisonKind,
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
