//! Native shape/type inference shared by operation builders and verifiers.
use crate::types::TensorType;
use pliron::{context::Context, r#type::TypeHandle};

pub fn compatible_tensor_types(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    let (a, b) = (a.deref(ctx), b.deref(ctx));
    let (Some(a), Some(b)) = (
        a.downcast_ref::<TensorType>(),
        b.downcast_ref::<TensorType>(),
    ) else {
        return false;
    };
    a.element == b.element
        && a.dimensions.len() == b.dimensions.len()
        && a.dimensions
            .iter()
            .zip(&b.dimensions)
            .all(|(&a, &b)| a == b || a == -1 || b == -1)
}

pub(crate) mod elementwise;
pub use compute::*;
pub use control::*;
pub use elementwise::*;
pub use shape::*;
pub(crate) mod compute;
pub(crate) mod control;
pub(crate) mod shape;
