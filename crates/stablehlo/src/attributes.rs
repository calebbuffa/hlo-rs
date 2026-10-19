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

mod affine;
mod builtin;
mod enums;
mod stablehlo;
pub use affine::*;
pub use builtin::*;
pub use stablehlo::*;
mod mesh;

/// Checks native attribute containers and builtin type attributes recursively.
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
        } else if let Some(ty) = attribute.downcast_ref::<PlironTypeAttr>() {
            ty.get_type(ctx).deref(ctx).verify(ctx)?;
        } else {
            attribute.verify(ctx)?;
        }
    }
    Ok(())
}

/// Checks every attribute in a heterogeneous attribute dictionary.
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
