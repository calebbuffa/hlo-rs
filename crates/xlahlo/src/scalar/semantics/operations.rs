use super::{
    EvaluationRoundingMode, Scalar, integer, invalid, mask, numerical_result, rounding_unary,
    same_format, unsupported,
};
use crate::{
    attributes::{ComparisonOrder, Direction},
    inference::elementwise::unary_result_element,
    opcode::Opcode,
    types::ElementType,
};
use pliron::result::Result;
use std::cmp::Ordering;

/// Apply a supported native unary operation with checked type admission.
///
/// Real formats admit Abs/Negate/Sign/Real/Imag/IsFinite and
/// RoundNearestAfz/Ceil/Floor. RoundNearestEven requires [`rounding_unary`].
/// Complex formats admit
/// Abs/Negate/Sign/Real/Imag; Abs/Real/Imag return the component format.
/// Other floating-point math remains explicitly unsupported.
///
/// Abs and Sign admit signed integers only. Negate, Clz and PopulationCount
/// admit all integers; Not also admits Pred. Results wrap to significant width.
/// Pinned PopulationCount counts `sizeof(ReturnT)*8` bits, so negative S1/S2/S4
/// count sign extension through eight bits before result truncation. Clz instead
/// masks to significant width, following the pinned evaluator's explicit fix.
pub fn unary(opcode: Opcode, operand: Scalar) -> Result<Scalar> {
    if matches!(
        opcode,
        Opcode::RoundNearestAfz | Opcode::Ceil | Opcode::Floor
    ) {
        return rounding_unary(opcode, operand, EvaluationRoundingMode::NearestEven);
    }
    let result_element = unary_result_element(opcode, operand.element)?;
    if operand.element.is_float() || operand.element.is_complex() {
        return numerical_result(
            super::super::unary(opcode, operand.element, operand.bits)?,
            result_element,
        );
    }
    integer::unary(opcode, operand.integer()?)?.scalar()
}

/// Apply a same-format native binary scalar operation.
///
/// All integers admit Add/Subtract/Multiply/Divide/Remainder/Power, min/max,
/// bitwise operations and the three shifts. Divide by zero returns all ones;
/// signed MIN/-1 wraps to MIN. Remainder by zero returns the dividend and
/// MIN%-1 is zero. Negative or oversized shifts produce zero, except arithmetic
/// right shift sign-fills (also for unsigned inputs interpreted as signed bits).
/// Pred admits only Add/Multiply/min/max/And/Or/Xor; Add is OR, not one-bit XOR.
///
/// Mulhi admits 8/16/32/64-bit integers only: the pinned handler's
/// `std::is_integral_v<ReturnT>` rejects custom sub-byte wrappers, and int4 tests
/// explicitly skip it. No substitute significant-width behavior is invented.
///
/// Real formats admit Add/Subtract/Multiply/Divide/Minimum/Maximum. Complex
/// formats admit the four arithmetic operations; Complex constructs C64/C128
/// from F32/F64 components without numerical conversion.
pub fn binary(opcode: Opcode, lhs: Scalar, rhs: Scalar) -> Result<Scalar> {
    same_format(lhs, rhs)?;
    if lhs.element.is_float() || lhs.element.is_complex() {
        let result_element = if opcode == Opcode::Complex {
            match lhs.element {
                ElementType::F32 => ElementType::C64,
                ElementType::F64 => ElementType::C128,
                _ => return Err(unsupported(opcode, lhs.element)),
            }
        } else {
            lhs.element
        };
        return numerical_result(
            super::super::binary(opcode, lhs.element, lhs.bits, rhs.bits)?,
            result_element,
        );
    }
    integer::binary(opcode, lhs.integer()?, rhs.integer()?)?.scalar()
}

/// Compare exact values in their native signed/unsigned interpretation.
///
/// Integral and predicate IR comparisons require Total ordering. Partial/Weak
/// requests are errors. Real formats support all three orders; complex values
/// support only Eq/Ne (independent of the requested order).
pub fn compare(
    direction: Direction,
    order: ComparisonOrder,
    lhs: Scalar,
    rhs: Scalar,
) -> Result<Scalar> {
    same_format(lhs, rhs)?;
    if lhs.element.is_float() || lhs.element.is_complex() {
        return numerical_result(
            super::super::compare(direction, order, lhs.element, lhs.bits, rhs.bits)?,
            ElementType::Pred,
        );
    }
    let lhs = lhs.integer()?;
    let rhs = rhs.integer()?;
    if order != ComparisonOrder::Total {
        return Err(invalid(
            "integer and predicate comparisons require total ordering",
        ));
    }
    let ordering = lhs.value().cmp(&rhs.value());
    let result = match direction {
        Direction::Eq => ordering == Ordering::Equal,
        Direction::Ne => ordering != Ordering::Equal,
        Direction::Lt => ordering == Ordering::Less,
        Direction::Le => ordering != Ordering::Greater,
        Direction::Gt => ordering == Ordering::Greater,
        Direction::Ge => ordering != Ordering::Less,
    };
    Scalar::from_bits(ElementType::Pred, u128::from(result))
}

/// Convert mathematically interpreted integers with native modular narrowing.
///
/// Signed sources sign-extend before truncation, unsigned sources zero-extend.
/// Conversion to Pred tests nonzero; Pred to integer produces zero or one (S1
/// therefore encodes true as -1). This differs intentionally from the checked
/// mathematical-range contract of `literal::semantics::integer_scalar`.
/// Real conversions and C64/C128 conversion use the qualified raw kernel.
/// Complex-to-noncomplex conversion remains unsupported.
pub fn convert(operand: Scalar, destination: ElementType) -> Result<Scalar> {
    if (!operand.element.is_integer() && operand.element != ElementType::Pred)
        || (!destination.is_integer() && destination != ElementType::Pred)
    {
        return numerical_result(
            super::super::convert(operand.element, operand.bits, destination)?,
            destination,
        );
    }
    let operand = operand.integer()?;
    let bits = if destination == ElementType::Pred {
        u64::from(operand.bits != 0)
    } else {
        (operand.value() as u64) & mask(destination)
    };
    Scalar::from_bits(destination, u128::from(bits))
}

/// Reinterpret a single scalar's storage bits without numeric conversion.
///
/// Source and destination storage widths must match, including Pred's eight-bit
/// storage width (not its one significant bit). Splitting/combining multiple
/// elements is a tensor operation and is not implemented here. Invalid predicate
/// encodings return a typed error rather than constructing an invalid literal.
pub fn bitcast_convert(operand: Scalar, destination: ElementType) -> Result<Scalar> {
    if operand.element.storage_bit_width() != destination.storage_bit_width() {
        return Err(invalid("scalar bitcast-convert changes storage bit width"));
    }
    Scalar::from_bits(destination, operand.bits)
}

/// Select one owned same-format value with a predicate; no numeric coercion.
pub fn select(predicate: Scalar, on_true: Scalar, on_false: Scalar) -> Result<Scalar> {
    same_format(on_true, on_false)?;
    if predicate.element != ElementType::Pred {
        return Err(invalid("scalar select requires a predicate condition"));
    }
    Ok(if predicate.bits != 0 {
        on_true
    } else {
        on_false
    })
}

/// Clamp in native numeric order, preserving pinned floating NaN precedence.
///
/// Like the pinned visitor, reversed bounds are not rejected and yield high.
/// Predicate clamp follows the same false-before-true order.
/// Floating NaNs take priority in low/value/high order; complex clamp is
/// unsupported.
pub fn clamp(low: Scalar, value: Scalar, high: Scalar) -> Result<Scalar> {
    same_format(low, value)?;
    same_format(value, high)?;
    if value.element.is_float() || value.element.is_complex() {
        return numerical_result(
            super::super::clamp(value.element, low.bits, value.bits, high.bits)?,
            value.element,
        );
    }
    binary(Opcode::Minimum, high, binary(Opcode::Maximum, value, low)?)
}
