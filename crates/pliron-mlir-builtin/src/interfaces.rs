//! Extension points for dialect-owned types and tensor encodings.
//!
//! These interfaces let builtin containers validate foreign elements and
//! encodings without depending on their owning dialect.

use pliron::{
    attribute::Attribute,
    builtin::attributes::VecAttr,
    context::Context,
    derive::{attr_interface, type_interface},
    result::Result,
    r#type::{Type, TypeHandle},
};

/// Dialect-owned element admitted by builtin tensor containers.
#[type_interface]
pub trait TensorElementInterface {
    /// Checks the interface contract for the implementing type.
    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }

    /// Checks constraints that depend on the enclosing ranked tensor shape.
    fn verify_tensor_shape(&self, _ctx: &Context, _shape: &[i64]) -> Result<()> {
        Ok(())
    }

    /// Returns whether tensors with this element may carry an encoding.
    fn allows_encoding(&self) -> bool {
        true
    }

    /// Returns the scalar storage type for dense/sparse payloads, if distinct.
    fn storage_type(&self) -> Option<TypeHandle> {
        None
    }

    /// Checks payload constraints beyond ordinary scalar type matching.
    fn verify_elements(&self, _ctx: &Context, _values: &VecAttr) -> Result<()> {
        Ok(())
    }
}

/// Dialect-owned type admitted by builtin tuple/function containers.
#[type_interface]
pub trait ValueTypeInterface {
    /// Checks the interface contract for the implementing type.
    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Dialect-owned string element represented by string-valued dense payloads.
#[type_interface]
pub trait StringElementInterface {
    /// Checks the interface contract for the implementing type.
    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Tensor encoding whose validation depends on the tensor it annotates.
#[attr_interface]
pub trait TensorEncodingInterface {
    /// Checks the interface contract for the implementing attribute.
    fn verify(_attr: &dyn Attribute, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }

    /// Returns optional dynamic-dimension bounds in tensor dimension order.
    fn bounds(&self) -> Option<&[i64]> {
        None
    }

    /// Checks this encoding against an enclosing tensor's shape and element.
    fn verify_tensor(&self, ctx: &Context, shape: &[i64], element: TypeHandle) -> Result<()>;
}
