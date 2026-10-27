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
    attribute::{AttrObj, AttributeDict},
    builtin::{
        attributes::{DictAttr, IntegerAttr, TypeAttr as PlironTypeAttr, VecAttr},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    location::Location,
    result::Result,
    r#type::{TypeHandle, Typed},
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

mod affine;
mod builtin;
mod enums;
pub use enums::EnumParseError;
mod named;
mod stablehlo;
pub use affine::*;
pub use builtin::*;
pub use named::*;
pub use stablehlo::*;
mod mesh;

/// Verifies an attribute and recursively checks supported attribute containers.
///
/// Traverses arrays, dictionaries, and named-attribute containers, including
/// nested type attributes. This checks standalone representations, not an
/// enclosing operation's contextual constraints such as dimension ranks or
/// precision-list length.
///
/// # Errors
///
/// Returns the first invalid name, nested type, or attribute diagnostic through
/// the shared error envelope. The input is not modified.
pub fn verify_attribute(ctx: &Context, attribute: &AttrObj) -> Result<()> {
    let mut pending = vec![attribute];
    while let Some(attribute) = pending.pop() {
        if let Some(array) = attribute.downcast_ref::<VecAttr>() {
            pending.extend(array.0.iter());
        } else if let Some(dictionary) = attribute.downcast_ref::<DictAttr>() {
            for (name, value) in &dictionary.0.0 {
                if name.to_string().is_empty() {
                    return verify_err!(
                        Location::Unknown,
                        crate::Diagnostic::InvalidAttribute {
                            attribute: "attribute dictionary",
                            reason: "empty attribute name"
                        }
                    );
                }
                pending.push(value);
            }
        } else if let Some(dictionary) = attribute.downcast_ref::<NamedAttributesAttr>() {
            dictionary.validate_names()?;
            pending.extend(dictionary.values.0.iter());
        } else if let Some(ty) = attribute.downcast_ref::<PlironTypeAttr>() {
            ty.get_type(ctx).deref(ctx).verify(ctx)?;
        } else {
            attribute.verify(ctx)?;
        }
    }
    Ok(())
}

/// Verifies every entry in a heterogeneous, identifier-keyed attribute dictionary.
///
/// Recursively applies [`verify_attribute`] and checks that entry names are
/// nonempty. The dictionary is borrowed without modification; validity does
/// not establish an enclosing operation's attribute constraints.
///
/// # Errors
///
/// Returns the first invalid entry name or nested attribute diagnostic.
pub fn verify_attributes(ctx: &Context, attributes: &AttributeDict) -> Result<()> {
    for (name, value) in &attributes.0 {
        if name.to_string().is_empty() {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "attribute dictionary",
                    reason: "empty attribute name"
                }
            );
        }
        verify_attribute(ctx, value)?;
    }
    Ok(())
}
