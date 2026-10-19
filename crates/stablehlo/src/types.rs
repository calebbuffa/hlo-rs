//! Native StableHLO types backed by pliron's interned type handles.
//!
//! [`TensorType::get`] constructs a ranked tensor from an element type handle
//! and dimensions (`-1` denotes a dynamic dimension). The [`tensor`] helper
//! accepts an [`ElementType`] and a shape slice. Inspect interned types through
//! `TypeHandle::deref` and downcast to their concrete type to read borrowed
//! dimensions, elements, encodings, and bounds. Integer payload helpers are in
//! [`crate::integer`]; typed metadata is in [`crate::attributes`].
use pliron::builtin::types::{IntegerType, Signedness as IntegerSignedness};
use pliron::r#type::{Type as PlironType, TypedHandle};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, IntegerAttr, VecAttr},
    common_traits::Verify,
    context::Context,
    derive::pliron_type,
    result::Result,
    r#type::TypeHandle,
};
use pliron::{location::Location, verify_err};
use std::fmt;

/// Primitive element types used by tensor values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    Index,
    I1,
    I2,
    I4,
    I8,
    I16,
    I32,
    I64,
    Si1,
    Si2,
    Si4,
    Si8,
    Si16,
    Si32,
    Si64,
    Ui1,
    Ui2,
    Ui4,
    Ui8,
    Ui16,
    Ui32,
    Ui64,
    F4E2M1Fn,
    F6E2M3Fn,
    F6E3M2Fn,
    F16,
    Bf16,
    F32,
    F64,
    Tf32,
    F8E3M4,
    F8E4M3,
    F8E4M3Fn,
    F8E4M3FnUz,
    F8E4M3B11FnUz,
    F8E5M2,
    F8E5M2FnUz,
    F8E8M0FnUz,
    C64,
    C128,
    /// Builtin integer whose width has no named variant. Construct it with
    /// [`ElementType::integer`], which keeps named widths canonical.
    Integer(IntegerElement),
    /// Builtin `complex<T>` whose component is neither `f32` nor `f64`.
    /// Construct it with [`ElementType::complex`], which keeps
    /// [`ElementType::C64`] and [`ElementType::C128`] canonical.
    Complex(ComplexElement),
}

/// Builtin complex type with an integer or floating component other than
/// `f32`/`f64`. Pinned MLIR `ComplexElement::verify` accepts exactly integer and
/// floating components.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComplexElement {
    component: ComplexComponent,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ComplexComponent {
    Named(u8),
    Integer(IntegerElement),
}

const NAMED_COMPLEX_COMPONENTS: [ElementType; 35] = {
    use ElementType::*;
    [
        I1,
        I2,
        I4,
        I8,
        I16,
        I32,
        I64,
        Si1,
        Si2,
        Si4,
        Si8,
        Si16,
        Si32,
        Si64,
        Ui1,
        Ui2,
        Ui4,
        Ui8,
        Ui16,
        Ui32,
        Ui64,
        F4E2M1Fn,
        F6E2M3Fn,
        F6E3M2Fn,
        F16,
        Bf16,
        Tf32,
        F8E3M4,
        F8E4M3,
        F8E4M3Fn,
        F8E4M3FnUz,
        F8E4M3B11FnUz,
        F8E5M2,
        F8E5M2FnUz,
        F8E8M0FnUz,
    ]
};

impl ComplexElement {
    /// Integer or floating component element type.
    pub fn component(self) -> ElementType {
        match self.component {
            ComplexComponent::Named(index) => NAMED_COMPLEX_COMPONENTS[usize::from(index)],
            ComplexComponent::Integer(ty) => ElementType::Integer(ty),
        }
    }
}

impl fmt::Debug for ComplexElement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ComplexElement")
            .field(&self.component())
            .finish()
    }
}

/// Builtin integer type with a width other than 1, 2, 4, 8, 16, 32 or 64.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IntegerElement {
    width: u32,
    signedness: IntegerSignedness,
}

impl IntegerElement {
    /// Pinned MLIR `IntegerElement::kMaxWidth`.
    pub const MAX_WIDTH: u32 = (1 << 24) - 1;

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn signedness(self) -> IntegerSignedness {
        self.signedness
    }
}

impl ElementType {
    /// Builtin integer `iN`, `siN` or `uiN`, using the named variant when one
    /// exists. Returns `None` above [`IntegerElement::MAX_WIDTH`].
    pub fn integer(width: u32, signedness: IntegerSignedness) -> Option<Self> {
        use ElementType::*;
        use IntegerSignedness::*;
        Some(match (width, signedness) {
            (1, Signless) => I1,
            (2, Signless) => I2,
            (4, Signless) => I4,
            (8, Signless) => I8,
            (16, Signless) => I16,
            (32, Signless) => I32,
            (64, Signless) => I64,
            (1, Signed) => Si1,
            (2, Signed) => Si2,
            (4, Signed) => Si4,
            (8, Signed) => Si8,
            (16, Signed) => Si16,
            (32, Signed) => Si32,
            (64, Signed) => Si64,
            (1, Unsigned) => Ui1,
            (2, Unsigned) => Ui2,
            (4, Unsigned) => Ui4,
            (8, Unsigned) => Ui8,
            (16, Unsigned) => Ui16,
            (32, Unsigned) => Ui32,
            (64, Unsigned) => Ui64,
            (width, _) if width > IntegerElement::MAX_WIDTH => return None,
            (width, signedness) => Integer(IntegerElement { width, signedness }),
        })
    }

    /// Builtin `complex<component>`, or `None` unless the component is an
    /// integer or floating type. `f32`/`f64` components return
    /// [`ElementType::C64`]/[`ElementType::C128`].
    pub fn complex(component: Self) -> Option<Self> {
        Some(match component {
            Self::F32 => Self::C64,
            Self::F64 => Self::C128,
            Self::Integer(ty) => Self::Complex(ComplexElement {
                component: ComplexComponent::Integer(ty),
            }),
            component => {
                let index = NAMED_COMPLEX_COMPONENTS
                    .iter()
                    .position(|candidate| *candidate == component)?;
                Self::Complex(ComplexElement {
                    component: ComplexComponent::Named(index as u8),
                })
            }
        })
    }

    /// Component of any builtin complex element type.
    pub fn complex_component(self) -> Option<Self> {
        match self {
            Self::C64 => Some(Self::F32),
            Self::C128 => Some(Self::F64),
            Self::Complex(ty) => Some(ty.component()),
            _ => None,
        }
    }

    /// Width and signedness of any builtin integer element type.
    pub fn integer_parts(self) -> Option<(u32, IntegerSignedness)> {
        use ElementType::*;
        use IntegerSignedness::*;
        Some(match self {
            I1 => (1, Signless),
            I2 => (2, Signless),
            I4 => (4, Signless),
            I8 => (8, Signless),
            I16 => (16, Signless),
            I32 => (32, Signless),
            I64 => (64, Signless),
            Si1 => (1, Signed),
            Si2 => (2, Signed),
            Si4 => (4, Signed),
            Si8 => (8, Signed),
            Si16 => (16, Signed),
            Si32 => (32, Signed),
            Si64 => (64, Signed),
            Ui1 => (1, Unsigned),
            Ui2 => (2, Unsigned),
            Ui4 => (4, Unsigned),
            Ui8 => (8, Unsigned),
            Ui16 => (16, Unsigned),
            Ui32 => (32, Unsigned),
            Ui64 => (64, Unsigned),
            Integer(ty) => (ty.width, ty.signedness),
            _ => return None,
        })
    }

    pub fn is_unsigned_integer(self) -> bool {
        matches!(self.integer_parts(), Some((_, IntegerSignedness::Unsigned)))
    }

    /// Integer payload of the low `min(width, 128)` two's-complement storage
    /// bits. Signed and signless types sign-extend; wider unsigned values that
    /// exceed `i128` are unsupported.
    pub fn integer_from_bits(self, bits: u128) -> Option<i128> {
        let (width, signedness) = self.integer_parts()?;
        let unsigned = signedness == IntegerSignedness::Unsigned;
        Some(match width {
            0 => 0,
            1..=127 => {
                let shift = 128 - width;
                if unsigned {
                    ((bits << shift) >> shift) as i128
                } else {
                    ((bits << shift) as i128) >> shift
                }
            }
            _ if unsigned => i128::try_from(bits).ok()?,
            _ => bits as i128,
        })
    }

    /// Fixed element storage width, or `None` for target-dependent `index`.
    pub fn bit_width(self) -> Option<u32> {
        use ElementType::*;
        Some(match self {
            Index => return None,
            Integer(ty) => ty.width,
            I1 | Si1 | Ui1 => 1,
            I2 | Si2 | Ui2 => 2,
            I4 | Si4 | Ui4 | F4E2M1Fn => 4,
            F6E2M3Fn | F6E3M2Fn => 6,
            I8 | Si8 | Ui8 | F8E3M4 | F8E4M3 | F8E4M3Fn | F8E4M3FnUz | F8E4M3B11FnUz | F8E5M2
            | F8E5M2FnUz | F8E8M0FnUz => 8,
            I16 | Si16 | Ui16 | F16 | Bf16 => 16,
            Tf32 => 19,
            I32 | Si32 | Ui32 | F32 => 32,
            I64 | Si64 | Ui64 | F64 | C64 => 64,
            C128 => 128,
            Complex(ty) => 2 * ty.component().bit_width()?,
        })
    }

    pub fn is_numeric(self) -> bool {
        !matches!(self, Self::Index | Self::I1)
    }

    pub fn is_integer(self) -> bool {
        self.integer_parts().is_some()
    }

    /// Pinned StableHLO `HLO_DimensionValue`: `index` or a signless/unsigned
    /// 2/4/8/16/32/64-bit integer.
    pub fn is_dimension_value(self) -> bool {
        self == Self::Index
            || matches!(
                self.integer_parts(),
                Some((
                    2 | 4 | 8 | 16 | 32 | 64,
                    IntegerSignedness::Signless | IntegerSignedness::Unsigned
                ))
            )
    }

    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::F4E2M1Fn
                | Self::F6E2M3Fn
                | Self::F6E3M2Fn
                | Self::F16
                | Self::Bf16
                | Self::F32
                | Self::F64
                | Self::Tf32
                | Self::F8E3M4
                | Self::F8E4M3
                | Self::F8E4M3Fn
                | Self::F8E4M3FnUz
                | Self::F8E4M3B11FnUz
                | Self::F8E5M2
                | Self::F8E5M2FnUz
                | Self::F8E8M0FnUz
        )
    }

    /// Representable attribute value range. Wide types are clamped to the
    /// `i128` attribute payload.
    pub(crate) fn integer_bounds(self) -> Option<(i128, i128)> {
        let (bits, signedness) = self.integer_parts()?;
        if self == Self::I1 {
            return None;
        }
        Some(match (signedness, bits) {
            (_, 0) => (0, 0),
            (IntegerSignedness::Unsigned, 128..) => (0, i128::MAX),
            (IntegerSignedness::Unsigned, bits) => (0, (1_i128 << bits) - 1),
            (_, 129..) => (i128::MIN, i128::MAX),
            (_, bits) => (i128::MIN >> (128 - bits), i128::MAX >> (128 - bits)),
        })
    }
}

impl ElementType {
    pub fn spelling(self) -> String {
        format!("{self:?}")
    }
    pub fn native(self, ctx: &Context) -> TypeHandle {
        use pliron::builtin::types::{BF16Type, FP16Type, FP32Type, FP64Type};
        if let Some((width, signedness)) = self.integer_parts() {
            return IntegerType::get(ctx, width, signedness).into();
        }
        if let Some(component) = self.complex_component() {
            return ComplexType::get(ctx, component.native(ctx)).into();
        }
        match self {
            Self::Index => IndexType::get(ctx).into(),
            Self::F16 => FP16Type::get(ctx).into(),
            Self::Bf16 => BF16Type::get(ctx).into(),
            Self::F32 => FP32Type::get(ctx).into(),
            Self::F64 => FP64Type::get(ctx).into(),
            element => FloatType::for_format(
                ctx,
                match element {
                    Self::F4E2M1Fn => FloatFormat::F4E2M1Fn,
                    Self::F6E2M3Fn => FloatFormat::F6E2M3Fn,
                    Self::F6E3M2Fn => FloatFormat::F6E3M2Fn,
                    Self::Tf32 => FloatFormat::Tf32,
                    Self::F8E3M4 => FloatFormat::F8E3M4,
                    Self::F8E4M3 => FloatFormat::F8E4M3,
                    Self::F8E4M3Fn => FloatFormat::F8E4M3Fn,
                    Self::F8E4M3FnUz => FloatFormat::F8E4M3FnUz,
                    Self::F8E4M3B11FnUz => FloatFormat::F8E4M3B11FnUz,
                    Self::F8E5M2 => FloatFormat::F8E5M2,
                    Self::F8E5M2FnUz => FloatFormat::F8E5M2FnUz,
                    Self::F8E8M0FnUz => FloatFormat::F8E8M0FnUz,
                    Self::Index | Self::C64 | Self::C128 | Self::Complex(_) => {
                        return match element {
                            Self::Index => IndexType::get(ctx).into(),
                            Self::C64 => ComplexType::get(ctx, FP32Type::get(ctx).into()).into(),
                            Self::C128 => ComplexType::get(ctx, FP64Type::get(ctx).into()).into(),
                            Self::Complex(complex) => {
                                ComplexType::get(ctx, complex.component().native(ctx)).into()
                            }
                            _ => IndexType::get(ctx).into(),
                        };
                    }
                    _ => {
                        let Some((width, signedness)) = element.integer_parts() else {
                            return IndexType::get(ctx).into();
                        };
                        return IntegerType::get(ctx, width, signedness).into();
                    }
                },
            ),
        }
    }
}
#[pliron_type(
    name = "stablehlo.index",
    format,
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IndexType;

#[pliron_type(
    name = "stablehlo.tensor",
    format = "`<` `[` vec($dimensions, CharSpace(`,`)) `]` `,` $element `,` $encoding `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dimensions: Vec<i64>,
    pub element: TypeHandle,
    pub encoding: VecAttr,
}
/// The ranked tensor identity is shared by plain and quantized tensor elements.
impl TensorType {
    pub fn get(ctx: &Context, dimensions: Vec<i64>, element: TypeHandle) -> TypedHandle<Self> {
        Self::instantiate(
            Self {
                dimensions,
                element,
                encoding: VecAttr::new(vec![]),
            },
            ctx,
        )
    }
    pub fn from_element(ctx: &Context, element: TypeHandle, dimensions: Vec<i64>) -> TypeHandle {
        Self::get(ctx, dimensions, element).into()
    }
    pub fn static_shape(
        ctx: &Context,
        element: ElementType,
        dimensions: impl IntoIterator<Item = i64>,
    ) -> TypeHandle {
        Self::from_element(ctx, element.native(ctx), dimensions.into_iter().collect())
    }
    pub fn scalar(ctx: &Context, element: ElementType) -> TypeHandle {
        Self::static_shape(ctx, element, [])
    }
    pub fn ranked(ctx: &Context, shape: impl Into<Vec<i64>>, element: ElementType) -> TypeHandle {
        Self::get(ctx, shape.into(), element.native(ctx)).into()
    }
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
    pub fn element(&self, ctx: &Context) -> Option<ElementType> {
        self.scalar_element(ctx)
    }
    pub fn shape(&self) -> &[i64] {
        &self.dimensions
    }
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }
    pub fn element_type(&self) -> TypeHandle {
        self.element
    }
    pub fn scalar_element(&self, ctx: &Context) -> Option<ElementType> {
        primitive_element(ctx, self.element)
    }
    pub fn element_count(&self) -> Option<usize> {
        element_count(&self.dimensions)
    }
    pub fn encoding(&self) -> Option<&AttrObj> {
        self.encoding.0.first()
    }
    pub fn bounds(&self) -> Option<&[i64]> {
        self.encoding()?
            .downcast_ref::<crate::attributes::TypeExtensionsAttr>()
            .map(|e| e.bounds.as_slice())
    }
    pub fn with_encoding(&self, ctx: &Context, encoding: AttrObj) -> TypeHandle {
        Self::instantiate(
            Self {
                dimensions: self.dimensions.clone(),
                element: self.element,
                encoding: VecAttr::new(vec![encoding]),
            },
            ctx,
        )
        .into()
    }
}
impl Verify for TensorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.dimensions.iter().any(|&d| d < -1) || self.encoding.0.len() > 1 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "invalid ranked tensor shape or encoding count"
                }
            );
        }
        if !is_tensor_element(ctx, self.element) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "invalid ranked tensor element type"
                }
            );
        }
        self.element.deref(ctx).verify(ctx)?;
        for encoding in &self.encoding.0 {
            crate::attributes::verify_attribute(ctx, encoding)?;
        }
        if let Some(encoding) = self
            .encoding()
            .and_then(|a| a.downcast_ref::<crate::sparse::SparseTensorEncoding>())
        {
            let Some(element) = self.scalar_element(ctx) else {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "StableHLO type",
                        reason: "sparse encodings require an ordinary primary scalar"
                    }
                );
            };
            if let Err(error) = encoding.verify_tensor(ctx, self.dimensions.len(), element) {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::Source {
                        category: crate::RecoveryCategory::InvalidInput,
                        context: "invalid sparse tensor encoding",
                        source: Box::new(error),
                    }
                );
            }
        }
        if self.encoding().is_some() && self.scalar_element(ctx).is_none() {
            let element = self.element.deref(ctx);
            if element.downcast_ref::<TokenType>().is_some()
                || element.downcast_ref::<VectorType>().is_some()
            {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "StableHLO type",
                        reason: "nested token/vector tensor elements cannot carry encodings"
                    }
                );
            }
        }
        if let Some(bounds) = self.bounds()
            && (bounds.len() != self.dimensions.len()
                || bounds
                    .iter()
                    .zip(&self.dimensions)
                    .any(|(&bound, &dim)| bound < -1 || (dim >= 0 && bound != -1)))
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "tensor bounds must match rank and apply only to dynamic dimensions"
                }
            );
        }
        let elem = self.element.deref(ctx);
        if let Some(q) = elem.downcast_ref::<QuantizedElementType>()
            && let Some(axis) = q.quantization_dimension()
            && (axis >= self.dimensions.len()
                || (self.dimensions[axis] >= 0
                    && self.dimensions[axis] as usize != q.scale_bits.len()))
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "per-axis quantization parameter count/rank mismatch"
                }
            );
        }
        Ok(())
    }
}
#[pliron_type(
    name = "stablehlo.token",
    format,
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenType;

#[pliron_type(
    name = "stablehlo.tuple",
    format = "`<` vec($elements, CharSpace(`,`)) `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TupleType {
    pub elements: Vec<TypeHandle>,
}

#[pliron_type(
    name = "stablehlo.complex",
    format = "`<` $component `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComplexType {
    pub component: TypeHandle,
}
impl Verify for ComplexType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if primitive_element(ctx, self.component)
            .is_none_or(|element| !element.is_integer() && !element.is_float())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "complex component must be a valid integer or floating-point type"
                }
            );
        }
        self.component.deref(ctx).verify(ctx)
    }
}

impl Verify for TupleType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let mut pending = self.elements.clone();
        let mut seen = std::collections::HashSet::new();
        while let Some(element) = pending.pop() {
            if !seen.insert(element) {
                continue;
            }
            if !is_value_type(ctx, element) {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "StableHLO type",
                        reason: "tuple element must be a value type"
                    }
                );
            }
            let ty = element.deref(ctx);
            if let Some(tuple) = ty.downcast_ref::<TupleType>() {
                pending.extend_from_slice(&tuple.elements);
            } else {
                ty.verify(ctx)?;
            }
        }
        Ok(())
    }
}
#[pliron_type(
    name = "stablehlo.function_type",
    format = "`<` `[` vec($inputs, CharSpace(`,`)) `]` `,` `[` vec($outputs, CharSpace(`,`)) `]` `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub inputs: Vec<TypeHandle>,
    pub outputs: Vec<TypeHandle>,
}
impl FunctionType {
    pub fn intern(ctx: &Context, inputs: Vec<TypeHandle>, outputs: Vec<TypeHandle>) -> TypeHandle {
        Self::get(ctx, inputs, outputs).into()
    }
    pub fn inputs(&self) -> &[TypeHandle] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[TypeHandle] {
        &self.outputs
    }
}
impl Verify for FunctionType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for &ty in self.inputs.iter().chain(&self.outputs) {
            if !is_value_type(ctx, ty) {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidType {
                        ty: "StableHLO type",
                        reason: "function arguments and results must be value types"
                    }
                );
            }
            ty.deref(ctx).verify(ctx)?;
        }
        Ok(())
    }
}
/// Native low-precision float format, including formats absent from pliron builtins.
#[pliron_type(
    name = "stablehlo.float",
    format = "`<` $format `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FloatType {
    pub format: u8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FloatFormat {
    F4E2M1Fn,
    F6E2M3Fn,
    F6E3M2Fn,
    Tf32,
    F8E3M4,
    F8E4M3,
    F8E4M3Fn,
    F8E4M3FnUz,
    F8E4M3B11FnUz,
    F8E5M2,
    F8E5M2FnUz,
    F8E8M0FnUz,
}
impl FloatType {
    pub fn for_format(ctx: &Context, format: FloatFormat) -> TypeHandle {
        Self::get(ctx, format as u8).into()
    }
}
impl Verify for FloatType {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.format > FloatFormat::F8E8M0FnUz as u8 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "unknown floating-point format"
                }
            );
        }
        Ok(())
    }
}

#[pliron_type(
    name = "stablehlo.unranked_tensor",
    format = "`<` $element `>`",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedTensorType {
    pub element: TypeHandle,
}

mod any_quantized;
mod calibrated;
mod quantile;
mod quantization;
mod shaped;
mod subchannel;
pub use any_quantized::*;
pub use calibrated::*;
pub use quantile::*;
pub use quantization::*;
pub use shaped::*;
pub use subchannel::*;
pub fn tensor(ctx: &Context, shape: &[i64], element: ElementType) -> TypeHandle {
    TensorType::ranked(ctx, shape.to_vec(), element)
}
pub fn function_type(
    ctx: &Context,
    arguments: Vec<TypeHandle>,
    results: Vec<TypeHandle>,
) -> TypeHandle {
    FunctionType::get(ctx, arguments, results).into()
}
pub fn primitive_element(ctx: &Context, ty: TypeHandle) -> Option<ElementType> {
    use pliron::builtin::types::{BF16Type, FP16Type, FP32Type, FP64Type};
    let ty = ty.deref(ctx);
    if let Some(i) = ty.downcast_ref::<IntegerType>() {
        return ElementType::integer(i.width(), i.signedness());
    }
    if ty.downcast_ref::<FP16Type>().is_some() {
        return Some(ElementType::F16);
    }
    if ty.downcast_ref::<BF16Type>().is_some() {
        return Some(ElementType::Bf16);
    }
    if ty.downcast_ref::<FP32Type>().is_some() {
        return Some(ElementType::F32);
    }
    if ty.downcast_ref::<FP64Type>().is_some() {
        return Some(ElementType::F64);
    }
    if ty.downcast_ref::<IndexType>().is_some() {
        return Some(ElementType::Index);
    }
    if let Some(c) = ty.downcast_ref::<ComplexType>() {
        return ElementType::complex(primitive_element(ctx, c.component)?);
    }
    if let Some(f) = ty.downcast_ref::<FloatType>() {
        return Some(match f.format {
            0 => ElementType::F4E2M1Fn,
            1 => ElementType::F6E2M3Fn,
            2 => ElementType::F6E3M2Fn,
            3 => ElementType::Tf32,
            4 => ElementType::F8E3M4,
            5 => ElementType::F8E4M3,
            6 => ElementType::F8E4M3Fn,
            7 => ElementType::F8E4M3FnUz,
            8 => ElementType::F8E4M3B11FnUz,
            9 => ElementType::F8E5M2,
            10 => ElementType::F8E5M2FnUz,
            11 => ElementType::F8E8M0FnUz,
            _ => return None,
        });
    }
    None
}
pub(crate) fn element_count(shape: &[i64]) -> Option<usize> {
    if shape.iter().any(|&dimension| dimension < 0) {
        return None;
    }
    if shape.contains(&0) {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| usize::try_from(d).ok()?.checked_mul(n))
}
pub(crate) fn valid_tensor_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.downcast_ref::<TensorType>().is_some_and(|t| {
        t.dimensions.iter().all(|&d| d >= 0) && primitive_element(ctx, t.element).is_some()
    })
}
pub fn tensor_parts(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, ElementType)> {
    let ty = ty.deref(ctx);
    let t = ty.downcast_ref::<TensorType>()?;
    if t.dimensions.iter().any(|&d| d < 0) {
        return None;
    }
    Some((t.dimensions.clone(), primitive_element(ctx, t.element)?))
}

pub fn is_tensor_element(ctx: &Context, ty: TypeHandle) -> bool {
    if let Some(element) = primitive_element(ctx, ty) {
        return element
            .integer_parts()
            .is_none_or(|(width, _)| width <= IntegerElement::MAX_WIDTH);
    }
    let ty = ty.deref(ctx);
    ty.downcast_ref::<TokenType>().is_some()
        || ty.downcast_ref::<VectorType>().is_some()
        || ty.downcast_ref::<QuantizedElementType>().is_some()
        || ty.downcast_ref::<CalibratedQuantizedType>().is_some()
        || ty.downcast_ref::<AnyQuantizedType>().is_some()
        || ty.downcast_ref::<QuantileType>().is_some()
        || ty.downcast_ref::<SubchannelQuantizedType>().is_some()
}

impl Verify for UnrankedTensorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if !is_tensor_element(ctx, self.element) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidType {
                    ty: "StableHLO type",
                    reason: "invalid unranked tensor element"
                }
            );
        }
        self.element.deref(ctx).verify(ctx)
    }
}
pub fn is_value_type(ctx: &Context, ty: TypeHandle) -> bool {
    let ty = ty.deref(ctx);
    ty.downcast_ref::<TensorType>().is_some()
        || ty.downcast_ref::<UnrankedTensorType>().is_some()
        || ty.downcast_ref::<TupleType>().is_some()
        || ty.downcast_ref::<TokenType>().is_some()
        || ty.downcast_ref::<NoneType>().is_some()
        || ty.downcast_ref::<WitnessType>().is_some()
        || ty.downcast_ref::<BufferType>().is_some()
        || ty.downcast_ref::<UnrankedBufferType>().is_some()
        || ty.downcast_ref::<VectorType>().is_some()
        || ty.downcast_ref::<FutureType>().is_some()
}
