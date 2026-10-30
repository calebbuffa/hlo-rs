//! pliron implementation of the MLIR `arith` dialect.

#![doc = include_str!("../README.md")]
#![deny(missing_docs, unreachable_pub, private_interfaces, private_bounds)]
#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
/// The IR framework used by this dialect, re-exported to keep context and handle types aligned.
pub use pliron;
/// Shared error envelope and result type used by fallible IR APIs.
pub use pliron::result::{Error, Result};
/// The StableHLO crate that supplies the shared scalar, tensor, and vector types.
pub use stablehlo;
pub mod attributes;
mod error;
/// Structured diagnostics and recovery categories for failed dialect operations.
pub use error::{Diagnostic, RecoveryCategory};
#[cfg(feature = "mlir-interop")]
pub mod mlir;
pub mod ops;
pub mod shaped;

/// Generates `get`/`set` accessors on `$op` for an attribute stored under
/// `$key`, a key declared once with [`pliron::dict_key!`].
///
/// pliron rejects a dictionary key declared by more than one operation, so
/// attributes shared by several operations (such as `fastmath`) declare their
/// key once and use this macro instead of `#[pliron_op(attributes = ...)]`.
#[doc(hidden)]
#[macro_export]
macro_rules! shared_attr {
    ($op:ty, $key:path, $ty:ty, $get:ident, $set:ident) => {
        impl $op {
            #[doc = concat!("Returns the `", stringify!($ty), "` attribute, if present.")]
            pub fn $get<'a>(
                &self,
                ctx: &'a $crate::pliron::context::Context,
            ) -> ::core::option::Option<::core::cell::Ref<'a, $ty>> {
                let op = <Self as $crate::pliron::op::Op>::get_operation(self);
                ::core::cell::Ref::filter_map(op.deref(ctx), |op| op.attributes.get::<$ty>(&$key))
                    .ok()
            }
            #[doc = concat!("Sets the `", stringify!($ty), "` attribute.")]
            pub fn $set(&self, ctx: &$crate::pliron::context::Context, value: $ty) {
                let op = <Self as $crate::pliron::op::Op>::get_operation(self);
                op.deref_mut(ctx).attributes.set($key.clone(), value);
            }
        }
    };
}
