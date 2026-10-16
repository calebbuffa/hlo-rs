#![doc = include_str!("../README.md")]

mod affine;
mod artifact;
mod attribute;
pub mod bytecode;
mod inference;
mod ir;
mod location;
mod operation;
mod sparse;
pub mod text;
mod types;
mod verification;
mod version;

pub use affine::{
    AffineBinaryOp, AffineConstraint, AffineError, AffineExpr, AffineMap, IntegerSet,
};
pub use artifact::{ArtifactVersionError, portable_artifact_version};
pub use attribute::stablehlo::{
    ArrayOfLayout, ArrayOfTiling, AttributeError, AxisRef, ChannelHandle, ConvDimensionNumbers,
    DotAlgorithm, DotDimensionNumbers, GatherDimensionNumbers, IndexTensor, Layout, Mesh, MeshAxis,
    MeshReference, OutputOperandAlias, Precision, ReplicaGroupMeshAxes, ResultAccuracy,
    ResultAccuracyMode, ScatterDimensionNumbers, StableHloAttribute, SubAxisInfo, Tiling,
    TypeExtensions,
};
pub use attribute::{Attribute, AttributeValidationError, Attributes, ScalarValue};
pub use inference::{
    ComparisonType, ConvolutionOptions, FftType, InferenceError, RngAlgorithm, RngDistribution,
    infer_abs_type, infer_all_gather_types, infer_all_reduce_types, infer_all_to_all_types,
    infer_batch_norm_grad_types, infer_batch_norm_inference_type, infer_batch_norm_training_types,
    infer_bitcast_convert_type, infer_broadcast_type, infer_case_types, infer_cholesky_type,
    infer_clamp_type, infer_collective_broadcast_types, infer_collective_permute_type,
    infer_compare_type, infer_complex_type, infer_concatenate_type, infer_convert_type,
    infer_convolution_type, infer_dot_general_type, infer_dot_type, infer_dynamic_gather_type,
    infer_dynamic_slice_type, infer_dynamic_update_slice_type, infer_fft_type, infer_gather_type,
    infer_get_dimension_size_type, infer_get_tuple_element_type, infer_if_types, infer_imag_type,
    infer_is_finite_type, infer_least_specific_type, infer_map_type, infer_most_specific_type,
    infer_optimization_barrier_types, infer_pad_type, infer_real_type, infer_reduce_scatter_type,
    infer_reduce_types, infer_reduce_window_types, infer_reverse_type,
    infer_rng_bit_generator_types, infer_rng_type, infer_scatter_types,
    infer_select_and_scatter_type, infer_select_type, infer_set_dimension_size_type,
    infer_slice_type, infer_sort_types, infer_transpose_type, infer_triangular_solve_type,
    infer_tuple_type, infer_uniform_dequantize_type, infer_uniform_quantize_type,
    infer_while_types,
};
pub use ir::{
    Block, BlockArgument, Computation, ComputationId, Instruction, Module, OperationResult, Region,
    ValueId,
};
pub use location::Location;
pub use operation::{Operation, StableHloOp};
pub use sparse::{
    SparseEncodingError, SparseLevelFormat, SparseLevelType, SparseSliceError,
    SparseTensorDimSlice, SparseTensorEncoding, infer_sparse_level_to_dimension,
};
pub use types::{
    BufferLayout, BufferType, ElementType, FunctionType, QuantizedElementType, QuantizedTensorType,
    TensorType, Type, TypeError, UnrankedTensorType,
};
pub use verification::BuildError;
pub use version::{
    CompatibilityRequirement, STABLEHLO_CURRENT_VERSION, STABLEHLO_MINIMUM_VERSION,
    STABLEHLO_SOURCE_COMMIT, StableHloVersion, VersionError,
};
