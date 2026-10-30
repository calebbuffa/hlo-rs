//! StableHLO-owned constraints on shared builtin containers.

use crate::{attributes::TypeExtensionsAttr, sparse::SparseTensorEncoding, types::*};
use pliron::{
    builtin::attributes::{IntegerAttr, VecAttr},
    context::Context,
    derive::{attr_interface_impl, type_interface_impl},
    result::Result,
    r#type::TypeHandle,
};
use pliron_mlir_builtin::interfaces::{
    StringElementInterface, TensorElementInterface, TensorEncodingInterface, ValueTypeInterface,
};

#[type_interface_impl]
impl TensorElementInterface for TokenType {
    fn allows_encoding(&self) -> bool {
        false
    }
}

macro_rules! tensor_elements {
    ($($ty:ty),* $(,)?) => {$(
        #[type_interface_impl]
        impl TensorElementInterface for $ty {}
    )*};
}

tensor_elements!(
    CalibratedQuantizedType,
    AnyQuantizedType,
    QuantileType,
    SubchannelQuantizedType,
);

macro_rules! value_types {
    ($($ty:ty),* $(,)?) => {$(
        #[type_interface_impl]
        impl ValueTypeInterface for $ty {}
    )*};
}

value_types!(TokenType, FutureType, WitnessType);

#[type_interface_impl]
impl StringElementInterface for StringType {}

#[type_interface_impl]
impl TensorElementInterface for QuantizedElementType {
    fn verify_tensor_shape(&self, _ctx: &Context, shape: &[i64]) -> Result<()> {
        if let Some(axis) = self.quantization_dimension()
            && (axis >= shape.len()
                || (shape[axis] >= 0 && shape[axis] as usize != self.scale_bits.len()))
        {
            return Err(crate::Diagnostic::InvalidType {
                ty: "quantized tensor",
                reason: "per-axis quantization parameter count/rank mismatch",
            }
            .into());
        }
        Ok(())
    }

    fn storage_type(&self) -> Option<TypeHandle> {
        Some(self.storage)
    }

    fn verify_elements(&self, ctx: &Context, values: &VecAttr) -> Result<()> {
        let Some((minimum, maximum)) = self.storage_type(ctx).and_then(ElementType::integer_bounds)
        else {
            return Err(crate::Diagnostic::InvalidAttribute {
                attribute: "quantized elements",
                reason: "dense quantized storage must be integer",
            }
            .into());
        };
        let (lower, upper) = self.storage_range();
        let bounds = lower.unwrap_or(minimum)..=upper.unwrap_or(maximum);
        for value in &values.0 {
            let integer =
                if let Some(bits) = value.downcast_ref::<crate::attributes::ScalarBitsAttr>() {
                    bits.integer_value(ctx).and_then(|value| value.to_i128())
                } else if let Some(value) = value.downcast_ref::<IntegerAttr>() {
                    let signed = !value.get_type().deref(ctx).is_unsigned();
                    value.value().to_string_decimal(signed).parse::<i128>().ok()
                } else {
                    None
                };
            if integer.is_none_or(|value| !bounds.contains(&value)) {
                return Err(crate::Diagnostic::InvalidAttribute {
                    attribute: "quantized elements",
                    reason: "quantized element is outside its declared storage range",
                }
                .into());
            }
        }
        Ok(())
    }
}

#[attr_interface_impl]
impl TensorEncodingInterface for TypeExtensionsAttr {
    fn bounds(&self) -> Option<&[i64]> {
        Some(&self.bounds)
    }

    fn verify_tensor(&self, _ctx: &Context, _shape: &[i64], _element: TypeHandle) -> Result<()> {
        // The builtin tensor checks rank and static/dynamic bounds consistency.
        Ok(())
    }
}

#[attr_interface_impl]
impl TensorEncodingInterface for SparseTensorEncoding {
    fn verify_tensor(&self, ctx: &Context, shape: &[i64], element: TypeHandle) -> Result<()> {
        let Some(element) = primitive_element(ctx, element) else {
            return Err(crate::Diagnostic::InvalidType {
                ty: "sparse tensor",
                reason: "sparse encodings require an ordinary primary scalar",
            }
            .into());
        };
        self.verify_tensor(ctx, shape.len(), element)
            .map_err(|error| {
                crate::Diagnostic::Source {
                    category: crate::RecoveryCategory::InvalidInput,
                    context: "invalid sparse tensor encoding",
                    source: Box::new(error),
                }
                .into()
            })
    }
}
