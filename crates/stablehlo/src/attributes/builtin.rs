use super::*;
use crate::types::{ElementType, QuantizedElementType, TensorType, VectorType, primitive_element};
use pliron::builtin::types::Signedness as IntegerSignedness;
use pliron::{
    builtin::attributes::{BoolAttr, StringAttr},
    derive::pliron_attr,
    r#type::Typed,
};
use pliron::{location::Location, verify_err};

#[pliron_attr(
    name = "stablehlo.scalar_bits",
    format = "`<` $ty `,` `[` vec($words, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Scalar integer or floating-point payload stored as exact raw bits.
pub struct ScalarBitsAttr {
    /// Primitive scalar type represented by the payload.
    pub ty: TypeHandle,
    /// Little-endian 64-bit limbs containing the scalar's exact bit pattern.
    pub words: Vec<u64>,
}
impl ScalarBitsAttr {
    /// Encodes an integer value after checking that it fits the element type.
    pub fn integer(
        ctx: &Context,
        element: ElementType,
        value: crate::integer::IntegerValue,
    ) -> Result<Self> {
        let Some((width, signedness)) = element.integer_parts() else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "integer attribute requires an integer type"
                }
            );
        };
        let fits = if element == ElementType::I1 {
            value == crate::integer::IntegerValue::ZERO
                || value == crate::integer::IntegerValue::from(1_u8)
        } else {
            value.fits_width(width, signedness)
        };
        if !fits {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "integer attribute is outside its element range"
                }
            );
        }
        Ok(Self {
            ty: element.native(ctx),
            words: value.to_twos_complement_words(width),
        })
    }
    /// Decodes the payload as an integer when its type is an integer element.
    pub fn integer_value(&self, ctx: &Context) -> Option<crate::integer::IntegerValue> {
        let element = primitive_element(ctx, self.ty)?;
        let (width, signedness) = element.integer_parts()?;
        Some(crate::integer::IntegerValue::from_twos_complement_words(
            &self.words,
            width,
            signedness != IntegerSignedness::Unsigned
                && (width > 1 || signedness == IntegerSignedness::Signed),
        ))
    }
    /// Creates an `f64` scalar payload from its exact IEEE-754 bits.
    pub fn float_bits(ctx: &Context, bits: u64) -> Self {
        Self {
            ty: ElementType::F64.native(ctx),
            words: vec![bits],
        }
    }
    /// Creates an `f32` scalar payload from its exact IEEE-754 bits.
    pub fn f32_bits(ctx: &Context, bits: u32) -> Self {
        Self {
            ty: ElementType::F32.native(ctx),
            words: vec![u64::from(bits)],
        }
    }
}
impl Verify for ScalarBitsAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let Some(width) = primitive_element(ctx, self.ty).and_then(ElementType::bit_width) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "scalar bits require a fixed-width scalar element"
                }
            );
        };
        if !words_fit_width(&self.words, width) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "scalar payload word count or high bits do not fit its type"
                }
            );
        }
        Ok(())
    }
}
fn words_fit_width(words: &[u64], width: u32) -> bool {
    words.len() == width.div_ceil(64) as usize
        && (width.is_multiple_of(64) || words.last().is_none_or(|word| word >> (width % 64) == 0))
}
#[pliron_attr(name = "stablehlo.index_value", format = "$value", verifier = "succ")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Signed index constant attribute.
pub struct IndexAttr {
    /// Signed index value.
    pub value: i64,
}

#[pliron_attr(
    name = "stablehlo.dense_array",
    format = "`<` $element `,` $values `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Homogeneous one-dimensional array of primitive scalar values.
pub struct DenseArrayAttr {
    /// Primitive element type of each array entry.
    pub element: TypeHandle,
    /// Values stored as a native vector attribute.
    pub values: VecAttr,
}
impl DenseArrayAttr {
    /// Creates an array of signless 64-bit integer values.
    pub fn integers(ctx: &Context, values: Vec<i64>) -> Self {
        Self {
            element: ElementType::I64.native(ctx),
            values: VecAttr::new(
                values
                    .into_iter()
                    .map(|v| {
                        Box::new(ScalarBitsAttr {
                            ty: ElementType::I64.native(ctx),
                            words: vec![v as u64],
                        }) as AttrObj
                    })
                    .collect(),
            ),
        }
    }
    /// Creates an array of one-bit boolean values.
    pub fn bools(ctx: &Context, values: Vec<bool>) -> Self {
        Self {
            element: ElementType::I1.native(ctx),
            values: VecAttr::new(
                values
                    .into_iter()
                    .map(|v| Box::new(BoolAttr::new(v)) as AttrObj)
                    .collect(),
            ),
        }
    }
    /// Creates an array of `f64` values from their exact bit patterns.
    pub fn floats_bits(ctx: &Context, values: Vec<u64>) -> Self {
        Self {
            element: ElementType::F64.native(ctx),
            values: VecAttr::new(
                values
                    .into_iter()
                    .map(|v| Box::new(ScalarBitsAttr::float_bits(ctx, v)) as AttrObj)
                    .collect(),
            ),
        }
    }
}
impl Verify for DenseArrayAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if !primitive_element(ctx, self.element).is_some_and(dense_array_element_type) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "dense arrays need i1 or byte-aligned integer/float elements"
                }
            );
        }
        verify_values(ctx, self.element, &self.values)
    }
}
/// Returns whether an element type is supported by dense-array attributes.
fn dense_array_element_type(e: ElementType) -> bool {
    e == ElementType::I1
        || (e.is_integer() || e.is_float()) && e.bit_width().is_some_and(|w| w % 8 == 0)
}
fn value_matches(ctx: &Context, ty: TypeHandle, value: &AttrObj) -> bool {
    if let Some(value) = value.downcast_ref::<ComplexAttr>() {
        return value.ty == ty;
    }
    if let Some(v) = value.downcast_ref::<ScalarBitsAttr>() {
        return v.ty == ty;
    }
    if value.downcast_ref::<BoolAttr>().is_some() {
        return ty == ElementType::I1.native(ctx);
    }
    if value.downcast_ref::<IndexAttr>().is_some() {
        return ty == ElementType::Index.native(ctx);
    }
    if let Some(v) = value.downcast_ref::<IntegerAttr>() {
        return TypeHandle::from(v.get_type()) == ty;
    }
    if value.downcast_ref::<StringAttr>().is_some() {
        return ty
            .deref(ctx)
            .downcast_ref::<crate::types::StringType>()
            .is_some();
    }
    false
}
fn verify_values(ctx: &Context, element: TypeHandle, values: &VecAttr) -> Result<()> {
    for value in &values.0 {
        value.verify(ctx)?;
        if !value_matches(ctx, element, value) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "dense value type does not match element type"
                }
            );
        }
    }
    Ok(())
}
fn shaped(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    if let Some(t) = ty.downcast_ref::<TensorType>() {
        let element = t.element.deref(ctx);
        return Some((
            t.dimensions.clone(),
            element
                .downcast_ref::<QuantizedElementType>()
                .map_or(t.element, |q| q.storage),
        ));
    }
    ty.downcast_ref::<VectorType>()
        .map(|v| (v.dimensions.clone(), v.element))
}
#[pliron_attr(
    name = "stablehlo.dense_elements",
    format = "`<` $ty `,` $values `,` $is_splat `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dense values associated with a ranked tensor or vector type.
pub struct DenseElementsAttr {
    /// Ranked tensor or vector type described by these elements.
    pub ty: TypeHandle,
    /// Stored scalar values, one per element unless `is_splat` is true.
    pub values: VecAttr,
    /// Whether the single stored value is broadcast to the entire shaped type.
    pub is_splat: bool,
}
impl DenseElementsAttr {
    /// Constructs a dense payload with one explicit value per element.
    pub fn new(ty: TypeHandle, values: VecAttr) -> Self {
        Self {
            ty,
            values,
            is_splat: false,
        }
    }
    /// Constructs a dense payload by broadcasting one scalar over the shaped type.
    pub fn splat(ty: TypeHandle, value: AttrObj) -> Self {
        Self {
            ty,
            values: VecAttr::new(vec![value]),
            is_splat: true,
        }
    }
}
impl Verify for DenseElementsAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.ty.deref(ctx).verify(ctx)?;
        let Some((shape, element)) = shaped(ctx, self.ty) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "dense elements need a tensor or vector type"
                }
            );
        };
        let Some(count) = crate::types::element_count(&shape) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "dense elements need a static non-overflowing shape"
                }
            );
        };
        if if self.is_splat {
            count == 0 || self.values.0.len() != 1
        } else {
            self.values.0.len() != count
        } {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "dense element count mismatch"
                }
            );
        }
        verify_shaped_values(ctx, self.ty, element, &self.values)
    }
}
#[pliron_attr(
    name = "stablehlo.sparse_elements",
    format = "`<` $ty `,` `[` vec($indices, CharSpace(`,`)) `]` `,` $values `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Coordinate/value representation of a sparse tensor or vector constant.
pub struct SparseElementsAttr {
    /// Ranked tensor or vector type described by the sparse payload.
    pub ty: TypeHandle,
    /// Coordinates of stored values, with one coordinate per shaped dimension.
    pub indices: Vec<IndexTensorAttr>,
    /// Stored scalar values, paired positionally with `indices`.
    pub values: VecAttr,
}
impl Verify for SparseElementsAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.ty.deref(ctx).verify(ctx)?;
        let Some((shape, element)) = shaped(ctx, self.ty) else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "sparse elements need tensor/vector type"
                }
            );
        };
        if crate::types::element_count(&shape).is_none()
            || self.indices.len() != self.values.0.len()
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "sparse indices/value count or static shape mismatch"
                }
            );
        }
        for index in &self.indices {
            if index.values.len() != shape.len()
                || index
                    .values
                    .iter()
                    .zip(&shape)
                    .any(|(&index, &size)| index < 0 || index >= size)
            {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidAttribute {
                        attribute: "builtin attribute",
                        reason: "sparse index rank/bounds mismatch"
                    }
                );
            }
        }
        verify_shaped_values(ctx, self.ty, element, &self.values)
    }
}
fn verify_shaped_values(
    ctx: &Context,
    ty: TypeHandle,
    element: TypeHandle,
    values: &VecAttr,
) -> Result<()> {
    verify_values(ctx, element, values)?;
    let ty = ty.deref(ctx);
    if let Some(tensor) = ty.downcast_ref::<TensorType>() {
        let quantized = tensor.element.deref(ctx);
        if let Some(q) = quantized.downcast_ref::<QuantizedElementType>() {
            let Some((minimum, maximum)) =
                q.storage_type(ctx).and_then(ElementType::integer_bounds)
            else {
                return verify_err!(
                    Location::Unknown,
                    crate::Diagnostic::InvalidAttribute {
                        attribute: "builtin attribute",
                        reason: "dense quantized storage must be integer"
                    }
                );
            };
            let (lower, upper) = q.storage_range();
            let bounds = lower.unwrap_or(minimum)..=upper.unwrap_or(maximum);
            for value in &values.0 {
                let integer = if let Some(bits) = value.downcast_ref::<ScalarBitsAttr>() {
                    bits.integer_value(ctx).and_then(|value| value.to_i128())
                } else if let Some(value) = value.downcast_ref::<IntegerAttr>() {
                    let signed = !value.get_type().deref(ctx).is_unsigned();
                    value.value().to_string_decimal(signed).parse::<i128>().ok()
                } else {
                    None
                };
                if integer.is_none_or(|value| !bounds.contains(&value)) {
                    return verify_err!(
                        Location::Unknown,
                        crate::Diagnostic::InvalidAttribute {
                            attribute: "builtin attribute",
                            reason: "quantized element is outside its declared storage range"
                        }
                    );
                }
            }
        }
    }
    Ok(())
}
#[pliron_attr(
    name = "stablehlo.bytes",
    format = "`[` vec($value, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Opaque sequence of bytes.
pub struct BytesAttr {
    /// Opaque byte sequence.
    pub value: Vec<u8>,
}
#[pliron_attr(
    name = "stablehlo.typed_string",
    format = "`<` $ty `,` `[` vec($value, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Byte string associated with a native type.
pub struct TypedStringAttr {
    /// Type associated with the byte sequence.
    pub ty: TypeHandle,
    /// String contents encoded as bytes.
    pub value: Vec<u8>,
}
impl Verify for TypedStringAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.ty.deref(ctx).verify(ctx)
    }
}
#[pliron_attr(
    name = "stablehlo.symbol_ref",
    format = "`[` vec($path, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Hierarchical symbol reference represented by path components.
pub struct SymbolRefAttr {
    /// Components of the symbol path, from the root symbol to the nested symbol.
    pub path: Vec<StringAttr>,
}
#[pliron_attr(
    name = "stablehlo.opaque",
    format = "`<` $dialect `,` $syntax `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dialect-tagged syntax preserved without interpreting its contents.
pub struct OpaqueAttr {
    /// Dialect that owns the opaque syntax.
    pub dialect: StringAttr,
    /// Dialect-specific textual representation.
    pub syntax: StringAttr,
}
#[pliron_attr(
    name = "stablehlo.enum",
    format = "`<` $dialect `,` $name `,` $value `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Named dialect enum value stored as text.
pub struct EnumAttr {
    /// Dialect that defines the enumeration.
    pub dialect: StringAttr,
    /// Enumeration name.
    pub name: StringAttr,
    /// Selected enumeration spelling.
    pub value: StringAttr,
}
impl Verify for EnumAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.dialect.as_str() == "stablehlo"
            && !enums::is_valid(self.name.as_str(), self.value.as_str())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "unknown StableHLO enum spelling"
                }
            );
        }
        Ok(())
    }
}
#[pliron_attr(
    name = "stablehlo.strided_layout",
    format = "`<` `[` vec($offset, CharSpace(`,`)) `]` `,` `[` vec($strides, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Offset and stride metadata for a strided layout.
pub struct StridedLayoutAttr {
    /// Optional base offset, represented by zero or one signed value.
    pub offset: Vec<i64>,
    /// Per-dimension strides, in element units.
    pub strides: Vec<i64>,
}
impl Verify for StridedLayoutAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.offset.len() > 1 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "strided offset has at most one value"
                }
            );
        }
        Ok(())
    }
}
/// Packages precision values in the vector representation used by operation attributes.
pub fn precision_config(values: Vec<Precision>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|precision| Box::new(precision) as AttrObj)
            .collect(),
    )
}
#[pliron_attr(name = "stablehlo.type_value", format = "`<` $ty `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Attribute whose value is a native type handle.
pub struct TypeAttr {
    /// Type represented by this attribute.
    pub ty: TypeHandle,
}
impl TypeAttr {
    /// Wraps a type handle as a type-valued attribute.
    pub fn new(ty: TypeHandle) -> Self {
        Self { ty }
    }
}
impl Verify for TypeAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.ty.deref(ctx).verify(ctx)
    }
}
impl Typed for TypeAttr {
    fn get_type(&self, _: &Context) -> TypeHandle {
        self.ty
    }
}

/// Exact signed 128-bit metadata, represented by two printable native limbs.
#[pliron_attr(
    name = "stablehlo.signed_128",
    format = "`<` $low `,` $high `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Signed 128-bit value represented by low and high limbs.
pub struct Signed128Attr {
    /// Low 64 bits of the two's-complement representation.
    pub low: u64,
    /// High 64 bits of the two's-complement representation.
    pub high: i64,
}
impl Signed128Attr {
    /// Reconstructs the signed 128-bit value from its two limbs.
    pub fn value(self) -> i128 {
        (i128::from(self.high) << 64) | i128::from(self.low)
    }
}
impl From<i128> for Signed128Attr {
    fn from(value: i128) -> Self {
        Self {
            low: value as u64,
            high: (value >> 64) as i64,
        }
    }
}

/// Standalone complex-number metadata requires floating-point components.
#[pliron_attr(
    name = "stablehlo.complex_value",
    format = "`<` $ty `,` $real `,` $imaginary `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Complex scalar represented by exact real and imaginary component bits.
pub struct ComplexAttr {
    /// Complex scalar type represented by the component payloads.
    pub ty: TypeHandle,
    /// Real component encoded with its exact floating-point bits.
    pub real: ScalarBitsAttr,
    /// Imaginary component encoded with its exact floating-point bits.
    pub imaginary: ScalarBitsAttr,
}
impl Verify for ComplexAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let Some(component) = primitive_element(ctx, self.ty)
            .and_then(ElementType::complex_component)
            .filter(|element| element.is_float())
        else {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "complex-number attributes require a floating component"
                }
            );
        };
        if self.real.ty != component.native(ctx) || self.imaginary.ty != component.native(ctx) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "builtin attribute",
                    reason: "complex component types do not match"
                }
            );
        }
        self.real.verify(ctx)?;
        self.imaginary.verify(ctx)
    }
}
