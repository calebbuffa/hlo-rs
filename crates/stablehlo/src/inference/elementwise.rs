use super::{InferenceError, dimensions, encoding, infer_most_specific_type};
use crate::{ElementType, StableHloOp, TensorType, Type};

/// The comparison interpretations recognized by the pinned dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComparisonType {
    NoType,
    Float,
    TotalOrder,
    Signed,
    Unsigned,
    WeakOrder,
}

impl ComparisonType {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "NOTYPE" => Self::NoType,
            "FLOAT" => Self::Float,
            "TOTALORDER" => Self::TotalOrder,
            "SIGNED" => Self::Signed,
            "UNSIGNED" => Self::Unsigned,
            "WEAKORDER" => Self::WeakOrder,
            _ => return None,
        })
    }
}

/// Infers predicate elements with lhs shape/encoding, as in `inferCompareOp`.
///
/// Operand types must be compatible. Explicit modes follow compare constraint
/// C3 for ordinary elements. Quantized comparisons dequantize their operands;
/// the reference additionally excludes quantized operands from WEAKORDER.
pub fn infer_compare_type(
    lhs: &Type,
    rhs: &Type,
    comparison: Option<ComparisonType>,
) -> Result<Type, InferenceError> {
    check(StableHloOp::Compare, lhs, 0)?;
    check(StableHloOp::Compare, rhs, 1)?;
    infer_most_specific_type(&[lhs.clone(), rhs.clone()])?;
    let valid_mode = match (lhs, comparison) {
        (_, None) => true,
        (Type::Tensor(tensor), Some(mode)) if tensor.element().is_float() => matches!(
            mode,
            ComparisonType::Float | ComparisonType::TotalOrder | ComparisonType::WeakOrder
        ),
        (Type::Tensor(tensor), Some(mode))
            if matches!(tensor.element(), ElementType::C64 | ElementType::C128) =>
        {
            mode == ComparisonType::Float
        }
        (Type::Tensor(_), Some(mode)) => matches!(
            mode,
            ComparisonType::NoType | ComparisonType::Signed | ComparisonType::Unsigned
        ),
        (Type::QuantizedTensor(_), Some(mode)) => mode != ComparisonType::WeakOrder,
        _ => unreachable!("checked compare operand domain"),
    };
    if !valid_mode {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    Ok(with_element(lhs, ElementType::I1))
}

/// Infers `abs`, preserving shape/encoding and taking the real component type
/// for complex operands. Quantized results inherit the input's parameters.
pub fn infer_abs_type(operand: &Type) -> Result<Type, InferenceError> {
    check(StableHloOp::Abs, operand, 0)?;
    match operand {
        Type::Tensor(tensor) => Ok(with_element(operand, real_element(tensor.element()))),
        _ => Ok(operand.clone()),
    }
}

/// Infers `real`, preserving shape and encoding and extracting complex elements.
pub fn infer_real_type(operand: &Type) -> Result<Type, InferenceError> {
    check(StableHloOp::Real, operand, 0)?;
    let Type::Tensor(tensor) = operand else {
        unreachable!("checked real operand")
    };
    Ok(with_element(operand, real_element(tensor.element())))
}

/// Infers `imag`, which has the same result type rule as `real`.
pub fn infer_imag_type(operand: &Type) -> Result<Type, InferenceError> {
    infer_real_type(operand)
}

/// Infers `is_finite`, replacing elements with i1 and preserving shape/encoding.
pub fn infer_is_finite_type(operand: &Type) -> Result<Type, InferenceError> {
    check(StableHloOp::IsFinite, operand, 0)?;
    Ok(with_element(operand, ElementType::I1))
}

/// Infers `complex` from compatible f32/f64 tensors.
///
/// Matches `inferComplexOp`: shape and encoding come from lhs, not a refinement
/// against rhs. Operand compatibility is checked before producing that type.
pub fn infer_complex_type(lhs: &Type, rhs: &Type) -> Result<Type, InferenceError> {
    check(StableHloOp::Complex, lhs, 0)?;
    check(StableHloOp::Complex, rhs, 1)?;
    infer_most_specific_type(&[lhs.clone(), rhs.clone()])?;
    let Type::Tensor(tensor) = lhs else {
        unreachable!("checked complex operand")
    };
    let element = match tensor.element() {
        ElementType::F32 => ElementType::C64,
        ElementType::F64 => ElementType::C128,
        _ => unreachable!("checked complex element type"),
    };
    Ok(with_element(lhs, element))
}

fn check(opcode: StableHloOp, ty: &Type, index: usize) -> Result<(), InferenceError> {
    ty.verify()
        .map_err(|error| InferenceError::InvalidType { index, error })?;
    if !opcode.accepts_fixed_operand_type(index, ty) {
        return Err(InferenceError::IncompatibleTypes { index });
    }
    Ok(())
}

fn real_element(element: ElementType) -> ElementType {
    match element {
        ElementType::C64 => ElementType::F32,
        ElementType::C128 => ElementType::F64,
        element => element,
    }
}

fn with_element(operand: &Type, element: ElementType) -> Type {
    let mut result = TensorType::new(
        element,
        dimensions(operand).expect("checked tensor").to_vec(),
    );
    if let Some(encoding) = encoding(operand) {
        result = result.with_encoding(encoding.clone());
    }
    result.into()
}
