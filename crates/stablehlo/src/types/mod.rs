//! StableHLO-specific types and re-exported shared MLIR builtin representations.

use pliron::r#type::{Type as PlironType, TypedHandle};
use pliron::{
    common_traits::Verify, context::Context, derive::pliron_type, result::Result,
    r#type::TypeHandle,
};
use pliron::{location::Location, verify_err};

/// Shared MLIR builtin types, with the same Rust identity as the foundation crate.
pub use pliron_mlir_builtin::types::*;

#[pliron_type(
    name = "stablehlo.token",
    format,
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Marker type for effect tokens.
pub struct TokenType;

mod any_quantized;
mod calibrated;
mod extensions;
mod quantile;
mod quantization;
mod subchannel;
pub use any_quantized::*;
pub use calibrated::*;
pub use extensions::*;
pub use quantile::*;
pub use quantization::*;
pub use subchannel::*;
pub(crate) fn valid_tensor_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.downcast_ref::<TensorType>().is_some_and(|t| {
        t.dimensions.iter().all(|&d| d >= 0) && primitive_element(ctx, t.element).is_some()
    })
}
