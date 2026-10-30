//! Native attributes, metadata encodings, and verification helpers for StableHLO.
//!
//! Attribute records are public value objects for explicit IR construction.
//! Their constructors do not imply validity; use [`crate::attributes::verify_attribute`] or
//! operation verification before consuming externally supplied metadata.
//! Internal dictionary keys are not part of the public API; prefer the typed
//! accessors on [`crate::ops`] wrappers.
//!
//! ```compile_fail,E0603
//! use stablehlo::attributes::keys::DOT_ALGORITHM;
//! ```

use crate::Diagnostic;
use pliron::{
    attribute::AttrObj,
    builtin::{
        attributes::{IntegerAttr, VecAttr},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    location::Location,
    result::Result,
    r#type::TypeHandle,
    utils::apint::{APInt, bw},
    verify_err,
};

pub(crate) fn integer_attr(ctx: &Context, value: u64) -> IntegerAttr {
    let ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    IntegerAttr::new(ty, APInt::from_u64(value, bw(64)))
}
pub(crate) fn integer_list(ctx: &Context, values: impl IntoIterator<Item = u64>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|v| Box::new(integer_attr(ctx, v)) as AttrObj)
            .collect(),
    )
}
pub(crate) fn read_integer_list(attr: &VecAttr) -> Option<Vec<u64>> {
    attr.0
        .iter()
        .map(|v| {
            v.downcast_ref::<IntegerAttr>()?
                .value()
                .to_string_decimal(false)
                .parse()
                .ok()
        })
        .collect()
}

pub(crate) mod keys;

mod dialect_support;
mod enums;
pub use enums::EnumParseError;
mod stablehlo;
pub use dialect_support::*;
/// Shared builtin attributes, preserving their Rust identity.
pub use pliron_mlir_builtin::attributes::*;
pub use stablehlo::*;
mod mesh;

/// Native operation storage for dialect-qualified discardable attributes.
///
/// Store a [`NamedAttributesAttr`] under this identifier when attaching source
/// metadata that cannot be represented by pliron's identifier-keyed dictionary.
/// This is the native container key, not an attribute name emitted in MLIR.
pub const DISCARDABLE_ATTRIBUTES: pliron::identifier::Identifier =
    pliron::identifier::Identifier::new(crate::attributes::keys::DISCARDABLE_ATTRIBUTES);
