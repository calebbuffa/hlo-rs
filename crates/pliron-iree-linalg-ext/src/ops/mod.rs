//! `iree_linalg_ext` operations.
//!
//! All 24 upstream operations are provided. Operands are ranked tensors (or
//! scalars where upstream allows them); buffer (`memref`) forms are not
//! modelled, matching `pliron-linalg`. Every operation follows destination
//! passing style: its results mirror its `outs` operands.
//!
//! Region operations have one single-block region terminated by
//! [`YieldOp`]; their `build` constructors populate it with a
//! [`BodyBuilder`] closure and verify the result.

mod attention;
mod fft;
mod im2col;
mod matmul;
mod quantize;
mod region;
mod scatter;
mod sort;
mod winograd;

pliron::dict_key!(#[doc = "Storage key for attention indexing maps."] ATTENTION_MAPS, "iree_attention_indexing_maps");
pliron::dict_key!(#[doc = "Storage key for attention decomposition hints."] ATTENTION_CONFIG, "iree_attention_decomposition_config");
pliron::dict_key!(#[doc = "Storage key for online-attention indexing maps."] ONLINE_ATTENTION_MAPS, "iree_online_attention_indexing_maps");
pliron::dict_key!(#[doc = "Storage key for online-attention decomposition hints."] ONLINE_ATTENTION_CONFIG, "iree_online_attention_decomposition_config");
pliron::dict_key!(#[doc = "Storage key for the Winograd input output-tile size."] WINOGRAD_INPUT_TILE, "iree_winograd_input_transform_output_tile_size");
pliron::dict_key!(#[doc = "Storage key for the Winograd input kernel size."] WINOGRAD_INPUT_KERNEL, "iree_winograd_input_transform_kernel_size");
pliron::dict_key!(#[doc = "Storage key for the Winograd input image dimensions."] WINOGRAD_INPUT_DIMS, "iree_winograd_input_transform_image_dimensions");
pliron::dict_key!(#[doc = "Storage key for the Winograd filter output-tile size."] WINOGRAD_FILTER_TILE, "iree_winograd_filter_transform_output_tile_size");
pliron::dict_key!(#[doc = "Storage key for the Winograd filter kernel size."] WINOGRAD_FILTER_KERNEL, "iree_winograd_filter_transform_kernel_size");
pliron::dict_key!(#[doc = "Storage key for the Winograd filter kernel dimensions."] WINOGRAD_FILTER_DIMS, "iree_winograd_filter_transform_kernel_dimensions");
pliron::dict_key!(#[doc = "Storage key for the Winograd output output-tile size."] WINOGRAD_OUTPUT_TILE, "iree_winograd_output_transform_output_tile_size");
pliron::dict_key!(#[doc = "Storage key for the Winograd output kernel size."] WINOGRAD_OUTPUT_KERNEL, "iree_winograd_output_transform_kernel_size");
pliron::dict_key!(#[doc = "Storage key for the Winograd output image dimensions."] WINOGRAD_OUTPUT_DIMS, "iree_winograd_output_transform_image_dimensions");

pub use attention::{AttentionOp, CustomOp, ExpReductionOp, OnlineAttentionOp};
pub use fft::FftOp;
pub use im2col::{DYNAMIC_INDEX, Im2colOp, Im2colPadding, Im2colPositions, MixedIndex};
pub use matmul::{GroupMatmulOp, GroupMmt4dOp};
pub use pliron_linalg::ops::BodyBuilder;
pub use quantize::{DequantizeAffineOp, QuantizeAffineOp, QuantizeBounds};
pub use region::{IndexOp, YieldOp};
pub use scatter::{GatherOp, MapLoadOp, MapStoreOp, ScatterOp};
pub use sort::{ArgCompareOp, ScanOp, SortOp, TopkOp, TopkV2Op};
pub use winograd::{
    WinogradFilterTransformOp, WinogradInputTransformOp, WinogradOutputTransformOp,
};
