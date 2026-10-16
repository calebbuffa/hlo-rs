use std::collections::BTreeMap;

use std::fmt;

use crate::{ElementType, Type, TypeError};

pub(crate) mod builtin;
pub(crate) mod enums;
mod mesh;
pub mod stablehlo;

pub type Attributes = BTreeMap<String, Attribute>;

/// Attribute values preserved on operation records.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Attribute {
    SparseTensorEncoding(Box<crate::SparseTensorEncoding>),
    SparseTensorSlice(crate::SparseTensorDimSlice),
    AffineMap(crate::AffineMap),
    IntegerSet(crate::IntegerSet),
    StridedLayout {
        offset: Option<i64>,
        strides: Vec<Option<i64>>,
    },
    Location(Box<crate::Location>),
    Unit,
    Bool(bool),
    Integer {
        ty: ElementType,
        value: i128,
    },
    Float {
        ty: ElementType,
        bits: u64,
    },
    Complex {
        ty: ElementType,
        real: u64,
        imaginary: u64,
    },
    String(String),
    DenseArray {
        element: ElementType,
        values: Vec<ScalarValue>,
    },
    Bytes(Vec<u8>),
    TypedString {
        value: Vec<u8>,
        ty: Box<Type>,
    },
    Type(Box<Type>),
    Array(Vec<Attribute>),
    Dictionary(BTreeMap<String, Attribute>),
    SymbolRef(Vec<String>),
    Enum {
        dialect: String,
        name: String,
        value: String,
    },
    DenseElements {
        ty: Box<Type>,
        values: Vec<ScalarValue>,
    },
    SparseElements {
        ty: Box<Type>,
        indices: Vec<Vec<i64>>,
        values: Vec<ScalarValue>,
    },
    StableHlo(Box<stablehlo::StableHloAttribute>),
    Opaque {
        dialect: Option<String>,
        syntax: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarValue {
    Bool(bool),
    Integer(i128),
    FloatBits(u128),
    ComplexBits { real: u64, imaginary: u64 },
    String(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttributeValidationError {
    SparseEncoding(crate::SparseEncodingError),
    SparseSlice(crate::SparseSliceError),
    Affine(crate::AffineError),
    StableHlo(stablehlo::AttributeError),
    InvalidType(TypeError),
    InvalidScalar {
        ty: ElementType,
    },
    InvalidDenseArrayElementType {
        ty: ElementType,
    },
    DenseElementsRequireStaticTensor,
    DenseElementCountOverflow,
    DenseElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidDenseElement {
        index: usize,
    },
    SparseIndexCountMismatch {
        indices: usize,
        values: usize,
    },
    SparseIndexRankMismatch {
        index: usize,
        expected: usize,
        actual: usize,
    },
    SparseIndexOutOfBounds {
        index: usize,
        dimension: usize,
    },
}

enum VerificationItem<'a> {
    Attribute(&'a Attribute),
    Location(&'a crate::Location),
}

impl Attribute {
    /// Constructs an ordinary signless i64 attribute.
    pub const fn integer(value: i64) -> Self {
        Self::Integer {
            ty: ElementType::I64,
            value: value as i128,
        }
    }

    /// Constructs a signless i32 attribute without discarding its width.
    pub const fn i32(value: i32) -> Self {
        Self::Integer {
            ty: ElementType::I32,
            value: value as i128,
        }
    }

    /// Constructs an f64 attribute without canonicalizing its payload.
    pub const fn float_bits(bits: u64) -> Self {
        Self::Float {
            ty: ElementType::F64,
            bits,
        }
    }

    /// Constructs an f32 attribute without canonicalizing its payload.
    pub const fn f32_bits(bits: u32) -> Self {
        Self::Float {
            ty: ElementType::F32,
            bits: bits as u64,
        }
    }

    /// Constructs a dense i64 array, not an array of scalar attributes.
    pub fn integers(values: Vec<i64>) -> Self {
        Self::DenseArray {
            element: ElementType::I64,
            values: values
                .into_iter()
                .map(|value| ScalarValue::Integer(i128::from(value)))
                .collect(),
        }
    }

    pub fn bools(values: Vec<bool>) -> Self {
        Self::DenseArray {
            element: ElementType::I1,
            values: values.into_iter().map(ScalarValue::Bool).collect(),
        }
    }

    pub fn floats_bits(values: Vec<u64>) -> Self {
        Self::DenseArray {
            element: ElementType::F64,
            values: values
                .into_iter()
                .map(|value| ScalarValue::FloatBits(u128::from(value)))
                .collect(),
        }
    }

    pub fn verify(&self) -> Result<(), AttributeValidationError> {
        Self::verify_pending(vec![VerificationItem::Attribute(self)])
    }

    pub(crate) fn verify_location(
        location: &crate::Location,
    ) -> Result<(), AttributeValidationError> {
        Self::verify_pending(vec![VerificationItem::Location(location)])
    }

    fn verify_pending(
        mut pending: Vec<VerificationItem<'_>>,
    ) -> Result<(), AttributeValidationError> {
        while let Some(item) = pending.pop() {
            let attribute = match item {
                VerificationItem::Attribute(attribute) => attribute,
                VerificationItem::Location(location) => {
                    match location {
                        crate::Location::Name {
                            child: Some(child), ..
                        } => pending.push(VerificationItem::Location(child)),
                        crate::Location::CallSite { callee, caller } => {
                            pending.push(VerificationItem::Location(caller));
                            pending.push(VerificationItem::Location(callee));
                        }
                        crate::Location::Fused {
                            locations,
                            metadata,
                        } => {
                            pending.extend(locations.iter().rev().map(VerificationItem::Location));
                            if let Some(metadata) = metadata {
                                pending.push(VerificationItem::Attribute(metadata));
                            }
                        }
                        _ => (),
                    }
                    continue;
                }
            };
            let result = match attribute {
                Self::SparseTensorEncoding(encoding) => encoding
                    .verify()
                    .map_err(AttributeValidationError::SparseEncoding),
                Self::SparseTensorSlice(slice) => slice
                    .verify()
                    .map_err(AttributeValidationError::SparseSlice),
                Self::AffineMap(map) => map.verify().map_err(AttributeValidationError::Affine),
                Self::IntegerSet(set) => set.verify().map_err(AttributeValidationError::Affine),
                Self::StridedLayout { .. } => Ok(()),
                Self::Location(location) => {
                    pending.push(VerificationItem::Location(location));
                    Ok(())
                }
                Self::DenseArray { element, values } => {
                    if !dense_array_element_type(*element) {
                        return Err(AttributeValidationError::InvalidDenseArrayElementType {
                            ty: *element,
                        });
                    }
                    for (index, value) in values.iter().enumerate() {
                        if !scalar_value_matches(value, *element) {
                            return Err(AttributeValidationError::InvalidDenseElement { index });
                        }
                    }
                    Ok(())
                }
                Self::Integer { ty, value } => {
                    if scalar_value_matches(&ScalarValue::Integer(*value), *ty) {
                        Ok(())
                    } else {
                        Err(AttributeValidationError::InvalidScalar { ty: *ty })
                    }
                }
                Self::Float { ty, bits } => {
                    if scalar_value_matches(&ScalarValue::FloatBits(u128::from(*bits)), *ty) {
                        Ok(())
                    } else {
                        Err(AttributeValidationError::InvalidScalar { ty: *ty })
                    }
                }
                Self::Complex {
                    ty,
                    real,
                    imaginary,
                } => {
                    if scalar_value_matches(
                        &ScalarValue::ComplexBits {
                            real: *real,
                            imaginary: *imaginary,
                        },
                        *ty,
                    ) {
                        Ok(())
                    } else {
                        Err(AttributeValidationError::InvalidScalar { ty: *ty })
                    }
                }
                Self::Type(ty) | Self::TypedString { ty, .. } => {
                    ty.verify().map_err(AttributeValidationError::InvalidType)
                }
                Self::Array(attributes) => {
                    pending.extend(attributes.iter().map(VerificationItem::Attribute));
                    Ok(())
                }
                Self::Dictionary(attributes) => {
                    pending.extend(attributes.values().map(VerificationItem::Attribute));
                    Ok(())
                }
                Self::DenseElements { ty, values } => {
                    let (dimensions, element) = tensor_shape_and_element(ty)?;
                    let expected = dense_element_count(dimensions)?;
                    if values.len() != expected {
                        return Err(AttributeValidationError::DenseElementCountMismatch {
                            expected,
                            actual: values.len(),
                        });
                    }
                    for (index, value) in values.iter().enumerate() {
                        if !scalar_value_matches_type(value, ty, element) {
                            return Err(AttributeValidationError::InvalidDenseElement { index });
                        }
                    }
                    Ok(())
                }
                Self::SparseElements {
                    ty,
                    indices,
                    values,
                } => {
                    let (dimensions, element) = tensor_shape_and_element(ty)?;
                    let dimensions = dimensions
                        .iter()
                        .map(|dimension| {
                            dimension
                                .ok_or(AttributeValidationError::DenseElementsRequireStaticTensor)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if indices.len() != values.len() {
                        return Err(AttributeValidationError::SparseIndexCountMismatch {
                            indices: indices.len(),
                            values: values.len(),
                        });
                    }
                    for (index, coordinate) in indices.iter().enumerate() {
                        if coordinate.len() != dimensions.len() {
                            return Err(AttributeValidationError::SparseIndexRankMismatch {
                                index,
                                expected: dimensions.len(),
                                actual: coordinate.len(),
                            });
                        }
                        for (dimension, (&coordinate, &size)) in
                            coordinate.iter().zip(&dimensions).enumerate()
                        {
                            if coordinate < 0 || coordinate as u64 >= size {
                                return Err(AttributeValidationError::SparseIndexOutOfBounds {
                                    index,
                                    dimension,
                                });
                            }
                        }
                        if !scalar_value_matches_type(&values[index], ty, element) {
                            return Err(AttributeValidationError::InvalidDenseElement { index });
                        }
                    }
                    Ok(())
                }
                Self::StableHlo(attribute) => attribute
                    .verify()
                    .map_err(AttributeValidationError::StableHlo),
                _ => Ok(()),
            };
            result?;
        }
        Ok(())
    }
}

pub(crate) fn dense_array_element_type(element: ElementType) -> bool {
    element == ElementType::I1
        || (element.is_integer() || element.is_float())
            && element.bit_width().is_some_and(|width| width % 8 == 0)
}

fn tensor_shape_and_element(
    ty: &Type,
) -> Result<(&[Option<u64>], ElementType), AttributeValidationError> {
    ty.verify().map_err(AttributeValidationError::InvalidType)?;
    match ty {
        Type::Tensor(tensor) => Ok((tensor.dimensions(), tensor.element())),
        Type::QuantizedTensor(tensor) => Ok((tensor.dimensions(), tensor.storage_type())),
        _ => Err(AttributeValidationError::DenseElementsRequireStaticTensor),
    }
}

fn dense_element_count(dimensions: &[Option<u64>]) -> Result<usize, AttributeValidationError> {
    if dimensions.iter().any(Option::is_none) {
        return Err(AttributeValidationError::DenseElementsRequireStaticTensor);
    }
    if dimensions.contains(&Some(0)) {
        return Ok(0);
    }
    dimensions.iter().try_fold(1_usize, |count, dimension| {
        let dimension =
            dimension.ok_or(AttributeValidationError::DenseElementsRequireStaticTensor)?;
        let dimension = usize::try_from(dimension)
            .map_err(|_| AttributeValidationError::DenseElementCountOverflow)?;
        count
            .checked_mul(dimension)
            .ok_or(AttributeValidationError::DenseElementCountOverflow)
    })
}

fn scalar_value_matches(value: &ScalarValue, element: ElementType) -> bool {
    match value {
        ScalarValue::Bool(_) => element == ElementType::I1,
        ScalarValue::Integer(value) => {
            if element == ElementType::Index {
                (i64::MIN as i128..=i64::MAX as i128).contains(value)
            } else {
                element
                    .integer_bounds()
                    .is_some_and(|(minimum, maximum)| (minimum..=maximum).contains(value))
            }
        }
        ScalarValue::FloatBits(value) => {
            element.is_float()
                && element
                    .bit_width()
                    .is_some_and(|width| *value >> width == 0)
        }
        ScalarValue::ComplexBits { real, imaginary } => match element {
            ElementType::C64 => *real <= u64::from(u32::MAX) && *imaginary <= u64::from(u32::MAX),
            ElementType::C128 => true,
            _ => false,
        },
        ScalarValue::String(_) => false,
    }
}

fn scalar_value_matches_type(value: &ScalarValue, ty: &Type, element: ElementType) -> bool {
    if let (Type::QuantizedTensor(tensor), ScalarValue::Integer(value)) = (ty, value) {
        let Some((type_min, type_max)) = tensor.storage_type().integer_bounds() else {
            return false;
        };
        let (storage_min, storage_max) = tensor.storage_range();
        return (storage_min.unwrap_or(type_min)..=storage_max.unwrap_or(type_max)).contains(value);
    }
    scalar_value_matches(value, element)
}

impl fmt::Display for AttributeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SparseEncoding(error) => write!(f, "invalid sparse encoding: {error}"),
            Self::SparseSlice(error) => write!(f, "invalid sparse slice attribute: {error}"),
            Self::Affine(error) => write!(f, "invalid affine attribute: {error}"),
            Self::StableHlo(error) => write!(f, "invalid StableHLO attribute: {error}"),
            Self::InvalidType(error) => write!(f, "invalid attribute type: {error}"),
            Self::InvalidScalar { ty } => write!(f, "scalar attribute value does not match {ty:?}"),
            Self::InvalidDenseArrayElementType { ty } => write!(
                f,
                "dense arrays require i1 or byte-aligned numeric elements, got {ty:?}"
            ),
            Self::DenseElementsRequireStaticTensor => {
                write!(
                    f,
                    "dense and sparse elements require a statically shaped tensor type"
                )
            }
            Self::DenseElementCountOverflow => write!(f, "element count overflows usize"),
            Self::DenseElementCountMismatch { expected, actual } => write!(
                f,
                "element attribute has {actual} values, expected {expected}"
            ),
            Self::InvalidDenseElement { index } => {
                write!(
                    f,
                    "element attribute value {index} does not match its element type"
                )
            }
            Self::SparseIndexCountMismatch { indices, values } => write!(
                f,
                "sparse attribute has {indices} indices and {values} values"
            ),
            Self::SparseIndexRankMismatch {
                index,
                expected,
                actual,
            } => write!(
                f,
                "sparse index {index} has rank {actual}, expected {expected}"
            ),
            Self::SparseIndexOutOfBounds { index, dimension } => write!(
                f,
                "sparse index {index} is out of bounds in dimension {dimension}"
            ),
        }
    }
}

impl std::error::Error for AttributeValidationError {}
