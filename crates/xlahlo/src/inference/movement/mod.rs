//! Native shape transforms and data-movement inference.

mod broadcast;
mod construction;
mod dimensions;
mod gather_scatter;
mod permutation;
mod reshape;
mod slicing;

pub use broadcast::infer_broadcast;
pub(crate) use broadcast::verify_broadcast;
pub use construction::{infer_concatenate, infer_iota, infer_pad};
pub use dimensions::{
    infer_copy, infer_get_dimension_size, infer_set_dimension_size, infer_shuffle,
};
pub use gather_scatter::{infer_gather, infer_scatter, infer_variadic_scatter};
pub use permutation::{infer_bitcast, infer_reverse, infer_transpose};
pub use reshape::{
    infer_dynamic_reshape, infer_reshape, infer_reshape_with_dimension, infer_static_reshape,
};
pub use slicing::{infer_dynamic_slice, infer_dynamic_update_slice, infer_slice};
