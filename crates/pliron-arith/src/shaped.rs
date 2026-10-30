//! Classification of the scalar, tensor, and vector types accepted by elementwise operations.
//!
//! Elementwise dialects such as `arith` and `math` accept a scalar element type
//! or a container of one. A [`Shaped`] separates the element from its container
//! so that verifiers can compare shapes and builders can substitute elements,
//! for example to produce the `i1` result of a comparison.

use pliron::{
    builtin::{
        attributes::VecAttr,
        types::{IntegerType, Signedness},
    },
    context::Context,
    r#type::{Type, TypeHandle},
};
use stablehlo::types::{
    ElementType, IndexType, TensorType, UnrankedTensorType, VectorType, primitive_element,
};

/// Container around an elementwise operand's element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Container {
    /// The value is a bare scalar.
    Scalar,
    /// Ranked tensor with dimensions (`-1` is dynamic) and its encoding.
    Tensor {
        /// Tensor extents.
        dimensions: Vec<i64>,
        /// Encoding attribute list, preserved when the element is replaced.
        encoding: VecAttr,
    },
    /// Tensor of unknown rank.
    UnrankedTensor,
    /// Fixed or scalable vector.
    Vector {
        /// Vector extents.
        dimensions: Vec<i64>,
        /// Per-dimension scalability.
        scalable: Vec<bool>,
    },
}

/// An element type together with the container holding it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shaped {
    /// Scalar element type.
    pub element: TypeHandle,
    /// Container around the element.
    pub container: Container,
}

impl Shaped {
    /// Returns whether two shaped types have identical containers, ignoring tensor encodings.
    pub fn same_shape(&self, other: &Self) -> bool {
        match (&self.container, &other.container) {
            (Container::Scalar, Container::Scalar)
            | (Container::UnrankedTensor, Container::UnrankedTensor) => true,
            (Container::Tensor { dimensions: a, .. }, Container::Tensor { dimensions: b, .. }) => {
                a == b
            }
            (
                Container::Vector {
                    dimensions: a,
                    scalable: sa,
                },
                Container::Vector {
                    dimensions: b,
                    scalable: sb,
                },
            ) => a == b && sa == sb,
            _ => false,
        }
    }
}

/// Returns whether `ty` is a scalar integer, index, or floating-point type.
pub fn is_scalar(ctx: &Context, ty: TypeHandle) -> bool {
    primitive_element(ctx, ty)
        .is_some_and(|e| e == ElementType::Index || e.is_integer() || e.is_float())
}

/// Splits `ty` into its element and container.
///
/// Returns `None` when `ty` is neither a scalar nor a tensor or vector of scalars.
pub fn classify(ctx: &Context, ty: TypeHandle) -> Option<Shaped> {
    let (element, container) = {
        let r = ty.deref(ctx);
        if let Some(t) = r.downcast_ref::<TensorType>() {
            (
                t.element,
                Container::Tensor {
                    dimensions: t.dimensions.clone(),
                    encoding: t.encoding.clone(),
                },
            )
        } else if let Some(t) = r.downcast_ref::<UnrankedTensorType>() {
            (t.element, Container::UnrankedTensor)
        } else if let Some(v) = r.downcast_ref::<VectorType>() {
            (
                v.element,
                Container::Vector {
                    dimensions: v.dimensions.clone(),
                    scalable: v.scalable.clone(),
                },
            )
        } else {
            (ty, Container::Scalar)
        }
    };
    is_scalar(ctx, element).then_some(Shaped { element, container })
}

/// Returns the element type of a scalar or shaped type.
pub fn element_type(ctx: &Context, ty: TypeHandle) -> Option<TypeHandle> {
    classify(ctx, ty).map(|s| s.element)
}

/// Rebuilds `shaped`'s container around a different element type.
pub fn with_element(ctx: &Context, shaped: &Shaped, element: TypeHandle) -> TypeHandle {
    match &shaped.container {
        Container::Scalar => element,
        Container::Tensor {
            dimensions,
            encoding,
        } => <TensorType as Type>::instantiate(
            TensorType {
                dimensions: dimensions.clone(),
                element,
                encoding: encoding.clone(),
            },
            ctx,
        )
        .into(),
        Container::UnrankedTensor => UnrankedTensorType::get(ctx, element).into(),
        Container::Vector {
            dimensions,
            scalable,
        } => VectorType::get(ctx, element, dimensions.clone(), scalable.clone()).into(),
    }
}

/// Returns the `i1` type used for predicates.
pub fn i1(ctx: &Context) -> TypeHandle {
    IntegerType::get(ctx, 1, Signedness::Signless).into()
}

/// Returns the builtin `index` type.
pub fn index(ctx: &Context) -> TypeHandle {
    ElementType::Index.native(ctx)
}

/// Returns whether `element` is a floating-point scalar type.
pub fn is_float(ctx: &Context, element: TypeHandle) -> bool {
    primitive_element(ctx, element).is_some_and(ElementType::is_float)
}

/// Returns whether `element` is the builtin `index` type.
pub fn is_index(ctx: &Context, element: TypeHandle) -> bool {
    element.deref(ctx).downcast_ref::<IndexType>().is_some()
}

/// Returns the width of a signless integer element type.
///
/// `arith` and `math` integer operations reject signed and unsigned integers.
pub fn signless_width(ctx: &Context, element: TypeHandle) -> Option<u32> {
    element
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .filter(|i| i.signedness() == Signedness::Signless)
        .map(|i| i.width())
}

/// Returns whether `element` is a signless integer or `index`.
pub fn is_signless_int_or_index(ctx: &Context, element: TypeHandle) -> bool {
    signless_width(ctx, element).is_some() || is_index(ctx, element)
}

/// Returns the bit width of a floating-point element type.
pub fn float_width(ctx: &Context, element: TypeHandle) -> Option<u32> {
    primitive_element(ctx, element)
        .filter(|e| e.is_float())
        .and_then(ElementType::bit_width)
}

/// Returns the bit width of a signless-integer or floating-point element type.
///
/// `index` has no fixed width and returns `None`.
pub fn scalar_width(ctx: &Context, element: TypeHandle) -> Option<u32> {
    signless_width(ctx, element).or_else(|| float_width(ctx, element))
}
