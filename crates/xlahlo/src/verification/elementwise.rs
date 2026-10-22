//! Verification for native elementwise HLO operations.
use crate::{
    Diagnostic,
    attributes::{ComparisonOrder, Direction, ReducePrecision},
    inference::elementwise,
    opcode::Opcode,
    shape,
    types::{ElementType, tensor},
    verification::VerifyOptions,
};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
};
use std::cell::Ref;

use super::{instruction_types, optional_attr};

pub(crate) fn instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let (operands, declared) = instruction_types(ctx, op, opcode)?;
    if opcode != Opcode::ReducePrecision && !options.allow_mixed_precision {
        let mut floating_type = None;
        for &operand in &operands {
            let element = tensor(ctx, operand)?.element;
            if element.is_float() {
                verify_require(
                    floating_type.is_none_or(|previous| previous == element),
                    "mixed floating operand precision is disabled",
                )?;
                floating_type = Some(element);
            }
        }
    }
    if is_unary(opcode) {
        return verify_unary(ctx, opcode, operands[0], declared, options);
    }
    let expected = match opcode {
        Opcode::Add
        | Opcode::And
        | Opcode::Atan2
        | Opcode::Complex
        | Opcode::Compare
        | Opcode::Divide
        | Opcode::Maximum
        | Opcode::Minimum
        | Opcode::Mulhi
        | Opcode::Multiply
        | Opcode::Or
        | Opcode::Power
        | Opcode::Remainder
        | Opcode::ShiftLeft
        | Opcode::ShiftRightArithmetic
        | Opcode::ShiftRightLogical
        | Opcode::Subtract
        | Opcode::Xor => {
            let lhs = tensor(ctx, operands[0])?;
            if opcode == Opcode::Compare {
                let _direction =
                    required_attr::<Direction>(ctx, op, crate::attributes::keys::DIRECTION)?;
                let order =
                    required_attr::<ComparisonOrder>(ctx, op, crate::attributes::keys::ORDER)?;
                if lhs.element.is_integer() || lhs.element == ElementType::Pred {
                    verify_require(
                        *order == ComparisonOrder::Total,
                        "integral and predicate comparisons require total ordering",
                    )?;
                }
            }
            let inferred = elementwise::infer_binary(ctx, opcode, operands[0], operands[1])?;
            let inferred_element = tensor(ctx, inferred)?.element;
            verify_require(
                shape::compatible(
                    ctx,
                    shape::with_element_type(ctx, operands[0], inferred_element)?,
                    inferred,
                ) && shape::compatible(
                    ctx,
                    shape::with_element_type(ctx, operands[1], inferred_element)?,
                    inferred,
                ),
                "implicit elementwise broadcasting is not permitted",
            )?;
            inferred
        }
        Opcode::Clamp => elementwise::infer_clamp(ctx, operands[0], operands[1], operands[2])?,
        Opcode::Select => elementwise::infer_select(ctx, operands[0], operands[1], operands[2])?,
        Opcode::Convert => {
            elementwise::infer_convert(ctx, operands[0], tensor(ctx, declared)?.element)?
        }
        Opcode::BitcastConvert => {
            elementwise::infer_bitcast_convert(ctx, operands[0], tensor(ctx, declared)?.element)?
        }
        Opcode::StochasticConvert => elementwise::infer_stochastic_convert(
            ctx,
            operands[0],
            operands[1],
            tensor(ctx, declared)?.element,
        )?,
        Opcode::ReducePrecision => {
            let precision =
                required_attr::<ReducePrecision>(ctx, op, crate::attributes::keys::PRECISION)?;
            elementwise::infer_reduce_precision(
                ctx,
                operands[0],
                precision.exponent_bits,
                precision.mantissa_bits,
            )?
        }
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Verification {
                message: format!("unsupported elementwise opcode {opcode}"),
            });
        }
    };
    verify_require(
        options.same_shapes(ctx, expected, declared),
        "declared result type differs from inferred elementwise type",
    )
}

pub(super) fn is_unary(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::Abs
            | Opcode::Acos
            | Opcode::Acosh
            | Opcode::Asin
            | Opcode::Asinh
            | Opcode::Atanh
            | Opcode::Cbrt
            | Opcode::Ceil
            | Opcode::Clz
            | Opcode::Cos
            | Opcode::Cosh
            | Opcode::Erf
            | Opcode::Exp
            | Opcode::Exp2
            | Opcode::Expm1
            | Opcode::Floor
            | Opcode::Imag
            | Opcode::IsFinite
            | Opcode::Log
            | Opcode::Log1p
            | Opcode::Log2
            | Opcode::Logistic
            | Opcode::Negate
            | Opcode::Not
            | Opcode::PopulationCount
            | Opcode::Real
            | Opcode::RoundNearestAfz
            | Opcode::RoundNearestEven
            | Opcode::Rsqrt
            | Opcode::Sign
            | Opcode::Sin
            | Opcode::Sinh
            | Opcode::Sqrt
            | Opcode::Tan
            | Opcode::Tanh
    )
}

fn verify_unary(
    ctx: &Context,
    opcode: Opcode,
    operand: TypeHandle,
    declared: TypeHandle,
    options: VerifyOptions,
) -> Result<()> {
    let input_element = tensor(ctx, operand)?.element;
    let result_element = elementwise::unary_result_element(opcode, input_element)?;
    let expected = if result_element == input_element {
        operand
    } else {
        shape::with_element_type(ctx, operand, result_element)?
    };
    verify_require(
        options.same_shapes(ctx, expected, declared),
        "declared result type differs from inferred unary type",
    )
}

fn required_attr<'a, T: Attribute>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    name: &'static str,
) -> Result<Ref<'a, T>> {
    optional_attr(ctx, op, name)?.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: name,
            message: format!(
                "missing or ill-typed {name}; expected {}",
                std::any::type_name::<T>()
            ),
        })
    })
}

fn verify_require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        pliron::verify_err_noloc!(Diagnostic::Verification {
            message: message.to_owned(),
        })
    }
}
