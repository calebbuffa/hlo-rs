use std::fmt;

use crate::{Attribute, attribute::stablehlo::StableHloAttribute};

mod quantization;
pub use quantization::QuantizedElementType;
mod unranked;
pub use unranked::UnrankedTensorType;

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
    Si2,
    Si4,
    Si8,
    Si16,
    Si32,
    Si64,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Tensor(TensorType),
    UnrankedTensor(UnrankedTensorType),
    QuantizedTensor(QuantizedTensorType),
    QuantizedElement(QuantizedElementType),
    Tuple(Vec<Type>),
    Token,
    Buffer(BufferType),
    Future(Vec<Type>),
    Element(ElementType),
    Function(FunctionType),
    String,
}

impl From<TensorType> for Type {
    fn from(value: TensorType) -> Self {
        Self::Tensor(value)
    }
}

impl Type {
    pub fn is_value_type(&self) -> bool {
        matches!(
            self,
            Self::Tensor(_)
                | Self::UnrankedTensor(_)
                | Self::QuantizedTensor(_)
                | Self::Tuple(_)
                | Self::Token
                | Self::Buffer(_)
                | Self::Future(_)
        )
    }

    /// Verifies recursive StableHLO type constraints represented by this model.
    pub fn verify(&self) -> Result<(), TypeError> {
        #[derive(Clone)]
        enum Context {
            Nested(&'static str, usize),
            Encoding,
        }
        let mut pending = vec![(self, Vec::<Context>::new())];
        while let Some((ty, contexts)) = pending.pop() {
            let mut children = Vec::new();
            let result = (|| match ty {
                Self::UnrankedTensor(tensor) => tensor.verify(),
                Self::QuantizedElement(element) => element.verify(),
                Self::Tensor(ty) => {
                    verify_dimension_sizes(ty.dimensions())?;
                    if let Some(Attribute::SparseTensorEncoding(encoding)) = ty.encoding() {
                        encoding
                            .verify_tensor(ty.rank(), ty.element())
                            .map_err(TypeError::InvalidSparseEncoding)?;
                    }
                    if let Some(encoding) = ty.encoding() {
                        match encoding {
                            Attribute::StableHlo(attribute) => {
                                attribute.verify().map_err(TypeError::InvalidEncoding)?
                            }
                            Attribute::Type(inner) => {
                                let mut child_contexts = contexts.clone();
                                child_contexts.push(Context::Encoding);
                                children.push((inner.as_ref(), child_contexts));
                            }
                            other => other.verify().map_err(|error| {
                                TypeError::InvalidEncodingAttribute(Box::new(error))
                            })?,
                        }
                    }
                    verify_tensor_encoding_metadata(ty.dimensions(), ty.encoding())
                }
                Self::QuantizedTensor(ty) => {
                    ty.verify()?;
                    if matches!(ty.encoding(), Some(Attribute::SparseTensorEncoding(_))) {
                        return Err(TypeError::InvalidSparseEncoding(
                            crate::SparseEncodingError::InvalidPrimaryType,
                        ));
                    }
                    verify_dimension_sizes(ty.dimensions())?;
                    if let Some(encoding) = ty.encoding() {
                        match encoding {
                            Attribute::StableHlo(attribute) => {
                                attribute.verify().map_err(TypeError::InvalidEncoding)?
                            }
                            Attribute::Type(inner) => {
                                let mut child_contexts = contexts.clone();
                                child_contexts.push(Context::Encoding);
                                children.push((inner.as_ref(), child_contexts));
                            }
                            other => other.verify().map_err(|error| {
                                TypeError::InvalidEncodingAttribute(Box::new(error))
                            })?,
                        }
                    }
                    verify_tensor_encoding_metadata(ty.dimensions(), ty.encoding())
                }
                Self::Future(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if !matches!(element, Self::Tensor(_) | Self::QuantizedTensor(_)) {
                            return Err(TypeError::ExpectedRankedTensor {
                                context: "future element",
                                index,
                            });
                        }
                        let mut child_contexts = contexts.clone();
                        child_contexts.push(Context::Nested("future element", index));
                        children.push((element, child_contexts));
                    }
                    Ok(())
                }
                Self::Tuple(elements) => {
                    for (index, element) in elements.iter().enumerate() {
                        if !element.is_value_type() {
                            return Err(TypeError::ExpectedValueType {
                                context: "tuple element",
                                index,
                            });
                        }
                        let mut child_contexts = contexts.clone();
                        child_contexts.push(Context::Nested("tuple element", index));
                        children.push((element, child_contexts));
                    }
                    Ok(())
                }
                Self::Function(function) => {
                    for (context, types) in [
                        ("function input", function.inputs()),
                        ("function output", function.outputs()),
                    ] {
                        for (index, ty) in types.iter().enumerate() {
                            if !ty.is_value_type() {
                                return Err(TypeError::ExpectedValueType { context, index });
                            }
                            let mut child_contexts = contexts.clone();
                            child_contexts.push(Context::Nested(context, index));
                            children.push((ty, child_contexts));
                        }
                    }
                    Ok(())
                }
                Self::Buffer(buffer) => buffer.verify(),
                _ => Ok(()),
            })();
            if let Err(mut error) = result {
                for context in contexts.into_iter().rev() {
                    error = match context {
                        Context::Nested(context, index) => TypeError::Nested {
                            context,
                            index,
                            error: Box::new(error),
                        },
                        Context::Encoding => TypeError::InvalidEncodingAttribute(Box::new(
                            crate::AttributeValidationError::InvalidType(error),
                        )),
                    };
                }
                return Err(error);
            }
            pending.extend(children);
        }
        Ok(())
    }
}

impl Type {
    pub(crate) fn plain_tensor_element(&self) -> Option<ElementType> {
        match self {
            Self::Tensor(tensor) => Some(tensor.element()),
            Self::UnrankedTensor(UnrankedTensorType::Plain(element)) => Some(*element),
            _ => None,
        }
    }

    pub(crate) fn quantized_tensor_element(&self) -> Option<&QuantizedElementType> {
        match self {
            Self::QuantizedTensor(tensor) => Some(tensor.element()),
            Self::UnrankedTensor(UnrankedTensorType::Quantized(element)) => Some(element),
            _ => None,
        }
    }
}

fn verify_dimension_sizes(dimensions: &[Option<u64>]) -> Result<(), TypeError> {
    for (dimension, size) in dimensions.iter().enumerate() {
        if let Some(size) = size
            && *size > i64::MAX as u64
        {
            return Err(TypeError::DimensionTooLarge {
                dimension,
                size: *size,
            });
        }
    }
    Ok(())
}

fn verify_tensor_encoding_metadata(
    dimensions: &[Option<u64>],
    encoding: Option<&Attribute>,
) -> Result<(), TypeError> {
    let Some(Attribute::StableHlo(attribute)) = encoding else {
        return Ok(());
    };
    let StableHloAttribute::TypeExtensions(extensions) = attribute.as_ref() else {
        return Ok(());
    };
    if extensions.bounds.len() != dimensions.len() {
        return Err(TypeError::EncodingBoundsRankMismatch {
            bounds: extensions.bounds.len(),
            rank: dimensions.len(),
        });
    }
    for (index, (dimension, bound)) in dimensions.iter().zip(&extensions.bounds).enumerate() {
        if dimension.is_some() && *bound != -1 {
            return Err(TypeError::StaticDimensionHasBound {
                dimension: index,
                bound: *bound,
            });
        }
    }
    Ok(())
}

impl ElementType {
    /// Fixed element storage width, or `None` for target-dependent `index`.
    pub fn bit_width(self) -> Option<u32> {
        use ElementType::*;
        Some(match self {
            Index => return None,
            I1 => 1,
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
        })
    }

    pub fn is_numeric(self) -> bool {
        !matches!(self, Self::Index | Self::I1)
    }

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I1
                | Self::I2
                | Self::I4
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::Si2
                | Self::Si4
                | Self::Si8
                | Self::Si16
                | Self::Si32
                | Self::Si64
                | Self::Ui2
                | Self::Ui4
                | Self::Ui8
                | Self::Ui16
                | Self::Ui32
                | Self::Ui64
        )
    }

    pub fn is_dimension_value(self) -> bool {
        self == Self::Index || (self.is_integer() && self != Self::I1)
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

    pub(crate) fn integer_bounds(self) -> Option<(i128, i128)> {
        let (bits, signed) = match self {
            Self::I2 => (2, true),
            Self::I4 => (4, true),
            Self::I8 | Self::Si8 => (8, true),
            Self::I16 | Self::Si16 => (16, true),
            Self::I32 | Self::Si32 => (32, true),
            Self::I64 | Self::Si64 => (64, true),
            Self::Si2 => (2, true),
            Self::Si4 => (4, true),
            Self::Ui2 => (2, false),
            Self::Ui4 => (4, false),
            Self::Ui8 => (8, false),
            Self::Ui16 => (16, false),
            Self::Ui32 => (32, false),
            Self::Ui64 => (64, false),
            _ => return None,
        };
        Some(if signed {
            (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
        } else {
            (0, (1_i128 << bits) - 1)
        })
    }
}

/// Ranked tensor type. `None` denotes a dynamic dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    element: ElementType,
    dimensions: Vec<Option<u64>>,
    encoding: Option<Box<Attribute>>,
}

impl TensorType {
    pub fn new(element: ElementType, dimensions: impl Into<Vec<Option<u64>>>) -> Self {
        Self {
            element,
            dimensions: dimensions.into(),
            encoding: None,
        }
    }

    pub fn with_encoding(mut self, encoding: Attribute) -> Self {
        self.encoding = Some(Box::new(encoding));
        self
    }

    pub fn static_shape(element: ElementType, dimensions: impl IntoIterator<Item = u64>) -> Self {
        Self::new(
            element,
            dimensions.into_iter().map(Some).collect::<Vec<_>>(),
        )
    }

    pub fn scalar(element: ElementType) -> Self {
        Self::new(element, Vec::new())
    }

    pub fn element(&self) -> ElementType {
        self.element
    }

    pub fn dimensions(&self) -> &[Option<u64>] {
        &self.dimensions
    }

    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    pub fn encoding(&self) -> Option<&Attribute> {
        self.encoding.as_deref()
    }
}

/// A ranked tensor containing uniform quantized elements.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuantizedTensorType {
    dimensions: Vec<Option<u64>>,
    element: QuantizedElementType,
    encoding: Option<Box<Attribute>>,
}

impl QuantizedTensorType {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dimensions: impl Into<Vec<Option<u64>>>,
        storage_type: ElementType,
        storage_min: Option<i128>,
        storage_max: Option<i128>,
        expressed_type: ElementType,
        quantization_dimension: Option<usize>,
        scales: Vec<f64>,
        zero_points: Vec<i128>,
    ) -> Self {
        Self::from_element(
            dimensions,
            QuantizedElementType::new(
                storage_type,
                storage_min,
                storage_max,
                expressed_type,
                quantization_dimension,
                scales,
                zero_points,
            ),
        )
    }

    pub fn from_element(
        dimensions: impl Into<Vec<Option<u64>>>,
        element: QuantizedElementType,
    ) -> Self {
        Self {
            dimensions: dimensions.into(),
            element,
            encoding: None,
        }
    }

    pub fn element(&self) -> &QuantizedElementType {
        &self.element
    }

    pub fn dimensions(&self) -> &[Option<u64>] {
        &self.dimensions
    }

    pub fn storage_type(&self) -> ElementType {
        self.element.storage_type()
    }

    pub fn storage_range(&self) -> (Option<i128>, Option<i128>) {
        self.element.storage_range()
    }

    pub fn expressed_type(&self) -> ElementType {
        self.element.expressed_type()
    }

    pub fn quantization_dimension(&self) -> Option<usize> {
        self.element.quantization_dimension()
    }

    pub fn scales(&self) -> impl ExactSizeIterator<Item = f64> + '_ {
        self.element.scales()
    }

    pub fn zero_points(&self) -> &[i128] {
        self.element.zero_points()
    }

    /// Checks the quantized tensor constraints defined by the pinned spec.
    pub fn verify(&self) -> Result<(), TypeError> {
        self.element.verify()?;
        if let Some(dimension) = self.quantization_dimension() {
            let Some(size) = self.dimensions.get(dimension) else {
                return Err(TypeError::QuantizationDimensionOutOfRange {
                    dimension,
                    rank: self.dimensions.len(),
                });
            };
            if size.is_some_and(|size| usize::try_from(size).ok() != Some(self.zero_points().len()))
            {
                return Err(TypeError::QuantizationAxisSizeMismatch {
                    dimension,
                    axis_size: *size,
                    parameter_count: self.zero_points().len(),
                });
            }
        }
        Ok(())
    }

    pub fn encoding(&self) -> Option<&Attribute> {
        self.encoding.as_deref()
    }

    pub fn with_encoding(mut self, encoding: Attribute) -> Self {
        self.encoding = Some(Box::new(encoding));
        self
    }
}

/// Buffer type data used by the StableHLO specification's `BufferType`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferType {
    element: ElementType,
    dimensions: Vec<Option<u64>>,
    layout: Option<BufferLayout>,
    memory_space: Option<Box<Attribute>>,
}

impl BufferType {
    pub fn new(element: ElementType, dimensions: impl Into<Vec<Option<u64>>>) -> Self {
        Self {
            element,
            dimensions: dimensions.into(),
            layout: None,
            memory_space: None,
        }
    }

    pub fn element(&self) -> ElementType {
        self.element
    }

    pub fn dimensions(&self) -> &[Option<u64>] {
        &self.dimensions
    }

    pub fn layout(&self) -> Option<&BufferLayout> {
        self.layout.as_ref()
    }

    pub fn memory_space(&self) -> Option<&Attribute> {
        self.memory_space.as_deref()
    }

    pub fn with_layout(mut self, layout: BufferLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn with_memory_space(mut self, memory_space: Attribute) -> Self {
        self.memory_space = Some(Box::new(memory_space));
        self
    }

    fn verify(&self) -> Result<(), TypeError> {
        verify_dimension_sizes(&self.dimensions)?;
        if let Some(BufferLayout::Affine(map)) = &self.layout {
            map.verify().map_err(TypeError::InvalidAffineLayout)?;
            if map.dimensions != self.dimensions.len() {
                return Err(TypeError::AffineLayoutRankMismatch {
                    dimensions: map.dimensions,
                    rank: self.dimensions.len(),
                });
            }
        }
        if let Some(BufferLayout::Strided { strides, .. }) = &self.layout
            && strides.len() != self.dimensions.len()
        {
            return Err(TypeError::BufferLayoutRankMismatch {
                strides: strides.len(),
                rank: self.dimensions.len(),
            });
        }
        if let Some(memory_space) = self.memory_space() {
            memory_space
                .verify()
                .map_err(|error| TypeError::InvalidBufferMemorySpace(Box::new(error)))?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferLayout {
    Identity,
    Affine(crate::AffineMap),
    Strided {
        offset: Option<i64>,
        strides: Vec<Option<i64>>,
    },
    Opaque(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionType {
    inputs: Vec<Type>,
    outputs: Vec<Type>,
}

impl FunctionType {
    pub fn new(inputs: impl Into<Vec<Type>>, outputs: impl Into<Vec<Type>>) -> Self {
        Self {
            inputs: inputs.into(),
            outputs: outputs.into(),
        }
    }

    pub fn inputs(&self) -> &[Type] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Type] {
        &self.outputs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeError {
    InvalidSparseEncoding(crate::SparseEncodingError),
    InvalidEncodingAttribute(Box<crate::attribute::AttributeValidationError>),
    InvalidAffineLayout(crate::AffineError),
    AffineLayoutRankMismatch {
        dimensions: usize,
        rank: usize,
    },
    InvalidEncoding(crate::attribute::stablehlo::AttributeError),
    InvalidBufferMemorySpace(Box<crate::attribute::AttributeValidationError>),
    DimensionTooLarge {
        dimension: usize,
        size: u64,
    },
    BufferLayoutRankMismatch {
        strides: usize,
        rank: usize,
    },
    EncodingBoundsRankMismatch {
        bounds: usize,
        rank: usize,
    },
    StaticDimensionHasBound {
        dimension: usize,
        bound: i64,
    },
    ExpectedValueType {
        context: &'static str,
        index: usize,
    },
    ExpectedRankedTensor {
        context: &'static str,
        index: usize,
    },
    Nested {
        context: &'static str,
        index: usize,
        error: Box<TypeError>,
    },
    InvalidQuantizedStorageType(ElementType),
    InvalidQuantizedExpressedType(ElementType),
    InvalidQuantizedStorageRange,
    MismatchedQuantizationParameters {
        scales: usize,
        zero_points: usize,
    },
    InvalidQuantizationScale,
    QuantizationScaleOutOfRange(ElementType),
    QuantizationZeroPointOutOfRange,
    PerTensorQuantizationRequiresOneScale,
    QuantizationDimensionOutOfRange {
        dimension: usize,
        rank: usize,
    },
    QuantizationAxisSizeMismatch {
        dimension: usize,
        axis_size: Option<u64>,
        parameter_count: usize,
    },
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSparseEncoding(error) => {
                write!(f, "invalid sparse tensor encoding: {error}")
            }
            Self::InvalidEncodingAttribute(error) => {
                write!(f, "invalid tensor encoding attribute: {error}")
            }
            Self::InvalidAffineLayout(error) => write!(f, "invalid affine buffer layout: {error}"),
            Self::AffineLayoutRankMismatch { dimensions, rank } => write!(
                f,
                "affine buffer layout has {dimensions} dimensions, but the buffer rank is {rank}"
            ),
            Self::InvalidEncoding(error) => write!(f, "invalid tensor encoding: {error}"),
            Self::InvalidBufferMemorySpace(error) => {
                write!(f, "invalid buffer memory-space attribute: {error}")
            }
            Self::DimensionTooLarge { dimension, size } => write!(
                f,
                "dimension {dimension} has size {size}, exceeding the StableHLO int64 range"
            ),
            Self::BufferLayoutRankMismatch { strides, rank } => write!(
                f,
                "strided buffer layout has {strides} strides, but the buffer rank is {rank}"
            ),
            Self::EncodingBoundsRankMismatch { bounds, rank } => write!(
                f,
                "tensor encoding has {bounds} bounds, but the tensor rank is {rank}"
            ),
            Self::StaticDimensionHasBound { dimension, bound } => write!(
                f,
                "static tensor dimension {dimension} cannot have bound {bound}"
            ),
            Self::ExpectedValueType { context, index } => {
                write!(f, "{context} {index} is not a StableHLO value type")
            }
            Self::ExpectedRankedTensor { context, index } => {
                write!(f, "{context} {index} must be a ranked tensor")
            }
            Self::Nested {
                context,
                index,
                error,
            } => write!(f, "invalid {context} {index}: {error}"),
            Self::InvalidQuantizedStorageType(element) => {
                write!(f, "{element:?} is not a supported quantized storage type")
            }
            Self::InvalidQuantizedExpressedType(element) => {
                write!(f, "{element:?} is not a floating-point expressed type")
            }
            Self::InvalidQuantizedStorageRange => {
                write!(f, "quantized storage range is outside the storage type")
            }
            Self::MismatchedQuantizationParameters {
                scales,
                zero_points,
            } => write!(
                f,
                "quantized type has {scales} scales and {zero_points} zero points"
            ),
            Self::InvalidQuantizationScale => {
                write!(f, "quantization scales must be positive finite values")
            }
            Self::QuantizationScaleOutOfRange(element) => {
                write!(
                    f,
                    "quantization scale is outside the finite positive range of {element:?}"
                )
            }
            Self::QuantizationZeroPointOutOfRange => {
                write!(f, "quantization zero point is outside the storage range")
            }
            Self::PerTensorQuantizationRequiresOneScale => {
                write!(f, "per-tensor quantization requires exactly one scale")
            }
            Self::QuantizationDimensionOutOfRange { dimension, rank } => {
                write!(
                    f,
                    "quantization dimension {dimension} is out of range for rank {rank}"
                )
            }
            Self::QuantizationAxisSizeMismatch {
                dimension,
                axis_size,
                parameter_count,
            } => write!(
                f,
                "quantization dimension {dimension} has size {axis_size:?}, but there are {parameter_count} parameter sets"
            ),
        }
    }
}

impl std::error::Error for TypeError {}
