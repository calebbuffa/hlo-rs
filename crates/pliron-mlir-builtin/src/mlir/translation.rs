use crate::{attributes::*, types::*};
use pliron::{
    context::Context,
    derive::{attr_interface_impl, type_interface_impl},
    printable::State,
    result::Result,
    r#type::Type,
};
use pliron_mlir_interop::{ToMlirAttr, ToMlirType};
use std::fmt;
macro_rules! types {
    ($($ty:ty),* $(,)?) => {$(
        #[type_interface_impl]
        impl ToMlirType for $ty {
            fn to_mlir(&self, ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                write!(f, "{}", super::print_type(ctx, self.get_self_handle(ctx))?)?;
                Ok(())
            }
        }
    )*};
}

types!(
    IndexType,
    FloatType,
    ComplexType,
    TensorType,
    UnrankedTensorType,
    TupleType,
    FunctionType,
    VectorType,
    BufferType,
    UnrankedBufferType,
    NoneType
);
macro_rules! attributes {
    ($($ty:ty),* $(,)?) => {$(
        #[attr_interface_impl]
        impl ToMlirAttr for $ty {
            fn to_mlir(&self, ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                write!(f, "{}", super::print_attribute(ctx, self)?)?;
                Ok(())
            }
        }
    )*};
}

attributes!(
    ScalarBitsAttr,
    IndexAttr,
    DenseArrayAttr,
    DenseElementsAttr,
    SparseElementsAttr,
    BytesAttr,
    TypedStringAttr,
    SymbolRefAttr,
    OpaqueAttr,
    StridedLayoutAttr,
    crate::attributes::TypeAttr,
    ComplexAttr,
    NamedAttributesAttr,
    AffineMap,
    IntegerSet,
    IndexTensorAttr
);
