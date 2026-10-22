//! Window, padding, dot, gather, scatter, and sparsity attributes.

use super::*;
use pliron::derive::{def_attribute, pliron_attr};

record!(WindowDimension, "xlahlo.window_dimension", "Window size, traversal, padding, dilation, and reversal for one axis.", {
    /// Number of positions in the undilated window.
    size: i64,
    /// Distance between consecutive window placements.
    stride: i64,
    /// Padding before the first base element; negative values crop the base.
    padding_low: i64,
    /// Padding after the last base element; negative values crop the base.
    padding_high: i64,
    /// Spacing between adjacent window elements.
    window_dilation: i64,
    /// Spacing between adjacent base elements.
    base_dilation: i64,
    /// Whether window elements are visited in reverse order along this axis.
    window_reversal: bool,
});
list!(
    Window,
    "xlahlo.window",
    WindowDimension,
    "Per-axis dimensions of a convolution or reduction window."
);
record!(PaddingConfigDimension, "xlahlo.padding_dimension", "Low, high, and interior padding for one logical axis.", {
    /// Padding before the first element; negative values remove edge elements.
    edge_padding_low: i64,
    /// Padding after the last element; negative values remove edge elements.
    edge_padding_high: i64,
    /// Number of padding elements inserted between adjacent input elements.
    interior_padding: i64,
});
list!(
    PaddingConfig,
    "xlahlo.padding_config",
    PaddingConfigDimension,
    "Per-axis low, high, and interior padding."
);
record!(ConvolutionDimensionNumbers, "xlahlo.convolution_dimensions", "Input, kernel, and output axis roles for convolution.", {
    /// Axis containing input batches.
    input_batch_dimension: i64,
    /// Axis containing input feature channels.
    input_feature_dimension: i64,
    /// Input spatial axes, in window order.
    input_spatial_dimensions: Axes,
    /// Kernel axis matching input feature channels.
    kernel_input_feature_dimension: i64,
    /// Kernel axis producing output feature channels.
    kernel_output_feature_dimension: i64,
    /// Kernel spatial axes, in window order.
    kernel_spatial_dimensions: Axes,
    /// Axis containing output batches.
    output_batch_dimension: i64,
    /// Axis containing output feature channels.
    output_feature_dimension: i64,
    /// Output spatial axes, in window order.
    output_spatial_dimensions: Axes,
});
record!(RaggedDotDimensionNumbers, "xlahlo.ragged_dot_dimensions", "Dot axis roles plus ragged and group dimensions.", {
    /// Batch and contraction axes of the underlying dot product.
    dot_dimension_numbers: DotDimensionNumbers,
    /// Left operand axes partitioned into variable-length groups.
    lhs_ragged_dimensions: Axes,
    /// Right operand axes identifying the corresponding groups.
    rhs_group_dimensions: Axes,
});
record!(DotDimensionNumbers, "xlahlo.dot_dimension_numbers", "Contracting and batch axes of both dot operands.", {
    /// Left operand contraction axes, paired with the right operand axes.
    lhs_contracting_dimensions: Axes,
    /// Right operand contraction axes, paired with the left operand axes.
    rhs_contracting_dimensions: Axes,
    /// Left operand batch axes, paired with the right operand batch axes.
    lhs_batch_dimensions: Axes,
    /// Right operand batch axes, paired with the left operand batch axes.
    rhs_batch_dimensions: Axes,
});
record!(GatherDimensionNumbers, "xlahlo.gather_dimension_numbers", "Gather output offsets, collapsed axes, batching, and index mapping.", {
    /// Output axes occupied by the gathered slice.
    offset_dims: Axes,
    /// Operand slice axes removed from the result.
    collapsed_slice_dims: Axes,
    /// Operand axis selected by each start-index vector component.
    start_index_map: Axes,
    /// Start-indices axis containing index-vector components.
    index_vector_dim: i64,
    /// Operand batch axes paired with start-indices batch axes.
    operand_batching_dims: Axes,
    /// Start-indices batch axes paired with operand batch axes.
    start_indices_batching_dims: Axes,
});
record!(ScatterDimensionNumbers, "xlahlo.scatter_dimension_numbers", "Scatter update windows, inserted axes, batching, and operand mapping.", {
    /// Update axes belonging to the scattered window.
    update_window_dims: Axes,
    /// Operand window axes absent from the updates.
    inserted_window_dims: Axes,
    /// Operand axis selected by each scatter-index vector component.
    scatter_dims_to_operand_dims: Axes,
    /// Scatter-indices axis containing index-vector components.
    index_vector_dim: i64,
    /// Input batch axes paired with scatter-indices batch axes.
    input_batching_dims: Axes,
    /// Scatter-indices batch axes paired with input batch axes.
    scatter_indices_batching_dims: Axes,
});
record!(TensorSparsityConfig, "xlahlo.tensor_sparsity", "Structured sparsity pattern and metadata operand index for one tensor.", {
    /// Number of nonzero elements retained per sparsity block.
    num_non_zero: i64,
    /// Number of logical elements represented by one sparsity block.
    block_size: i64,
    /// Tensor axis along which the block pattern is applied.
    dimension: i64,
    /// Distance between sparsity blocks.
    stride: i64,
    /// Operand index containing sparsity metadata.
    idx: i32,
});
optional!(
    OptionalTensorSparsityConfig,
    "xlahlo.optional_tensor_sparsity",
    TensorSparsityConfig
);
record!(SparsityConfig, "xlahlo.sparsity", "Independent optional sparsity configurations for both dot operands.", {
    /// Sparsity configuration for the left operand, if specified.
    lhs: OptionalTensorSparsityConfig,
    /// Sparsity configuration for the right operand, if specified.
    rhs: OptionalTensorSparsityConfig,
});
record!(TensorBlockScalingConfig, "xlahlo.tensor_block_scaling", "Scale and optional zero operands with block traversal strides and steps.", {
    /// Operand index containing block scales.
    scale_idx: i32,
    /// Operand index containing zero points, when explicitly specified.
    zero_idx: OptionalInteger,
    /// Per-axis strides used to locate block-scale values.
    strides: Axes,
    /// Per-axis steps between block-scaled groups.
    steps: Axes,
});
optional!(
    OptionalTensorBlockScalingConfig,
    "xlahlo.optional_tensor_block_scaling",
    TensorBlockScalingConfig
);
record!(BlockScalingConfig, "xlahlo.block_scaling", "Independent optional block-scaling configurations for both dot operands.", {
    /// Block-scaling configuration for the left operand, if specified.
    lhs: OptionalTensorBlockScalingConfig,
    /// Block-scaling configuration for the right operand, if specified.
    rhs: OptionalTensorBlockScalingConfig,
});
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

#[def_attribute("xlahlo.fft_type")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Forward, inverse, real-input, and real-output Fourier transform modes.
pub enum FftKind {
    /// Complex-to-complex forward transform.
    Fft,
    /// Complex-to-complex inverse transform.
    Ifft,
    /// Real-to-complex forward transform.
    Rfft,
    /// Complex-to-real inverse transform.
    Irfft,
}
#[pliron_attr(name = "xlahlo.padding_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Automatic window padding policy.
pub enum PaddingKind {
    /// Sentinel indicating that no valid padding policy was selected.
    Invalid,
    /// Uses only window placements fully contained in the input.
    Valid,
    /// Adds edge padding to preserve the stride-adjusted output extent.
    Same,
}
#[pliron_attr(name = "xlahlo.tri_state", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Boolean option with an explicit unspecified state.
pub enum TriState {
    /// No preference was specified.
    Unspecified,
    /// The option is explicitly enabled.
    True,
    /// The option is explicitly disabled.
    False,
}
#[pliron_attr(name = "xlahlo.convolution_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Forward, input-gradient, weight-gradient, or unselected convolution role.
pub enum ConvolutionKind {
    /// No backend convolution role was selected.
    Unset,
    /// Forward convolution producing output activations.
    Fprop,
    /// Backward convolution producing input gradients.
    Dgrad,
    /// Backward convolution producing weight gradients.
    Wgrad,
}
