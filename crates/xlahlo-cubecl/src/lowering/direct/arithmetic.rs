use super::*;
use cubecl_ir::dialect::math::{ArcTan2Op, FRemOp, PowfOp};

pub(in crate::lowering) fn check_binary_primitive(
    opcode: Opcode,
    element_type: ElementType,
) -> Result<(), PlanError> {
    if element_type == ElementType::F8E4M3 {
        return Err(PlanError::UnsupportedElementType(element_type));
    }
    if matches!(element_type, ElementType::F8E4M3FN | ElementType::F8E5M2)
        && !matches!(
            opcode,
            Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
        )
    {
        return Err(PlanError::UnsupportedContract {
            kind: crate::UnsupportedKind::Operation,
            feature: format!(
                "F8 arithmetic {opcode}: only Add, Subtract, Multiply and Divide have exact lowering"
            ),
        });
    }
    Ok(())
}

pub(in crate::lowering) fn emit_binary(
    scope: &Scope,
    opcode: Opcode,
    lhs: Value,
    rhs: Value,
    element_type: ElementType,
) -> Result<Value, PlanError> {
    check_binary_primitive(opcode, element_type)?;
    if element_type.is_float() {
        let target_type = lhs.get_type(scope.ctx());
        let widen = !matches!(element_type, ElementType::F32 | ElementType::F64);
        let (lhs, rhs) = if widen {
            let f32_type = ElemType::Float(FloatKind::F32).to_type(scope.ctx());
            (
                crate::float_cast::cast_float(scope, f32_type, lhs),
                crate::float_cast::cast_float(scope, f32_type, rhs),
            )
        } else {
            (lhs, rhs)
        };
        let result = match opcode {
            Opcode::Add => scope.register_with_result(&FAddOp::new(scope.ctx_mut(), lhs, rhs)),
            Opcode::Subtract => scope.register_with_result(&FSubOp::new(scope.ctx_mut(), lhs, rhs)),
            Opcode::Multiply => {
                let working_element = if widen {
                    ElementType::F32
                } else {
                    element_type
                };
                let (operations, product) =
                    make_binary_operations(scope.ctx_mut(), opcode, lhs, rhs, working_element)
                        .map_err(PlanError::UnsupportedFeature)?;
                for operation in operations {
                    scope.register(&*Operation::get_op_dyn(operation, scope.ctx()));
                }
                product
            }
            // Pinned XLA float8 evaluator uses ElementwiseT=float: one F32
            // division, then one narrowing to ReturnT (the original F8 format).
            Opcode::Divide => scope.register_with_result(&FDivOp::new(scope.ctx_mut(), lhs, rhs)),
            Opcode::Power => scope.register_with_result(&PowfOp::new(scope.ctx_mut(), lhs, rhs)),
            Opcode::Remainder => {
                scope.register_with_result(&FRemOp::new(scope.ctx_mut(), lhs, rhs))
            }
            Opcode::Atan2 => scope.register_with_result(&ArcTan2Op::new(scope.ctx_mut(), lhs, rhs)),
            opcode => {
                return Err(PlanError::UnsupportedFeature(format!(
                    "xlahlo operation {opcode}"
                )));
            }
        };
        return if widen {
            Ok(crate::float_cast::cast_float(scope, target_type, result))
        } else {
            Ok(result)
        };
    }
    if element_type.is_integer() {
        if opcode == Opcode::Divide {
            let (operations, result) =
                make_integer_division(scope.ctx_mut(), lhs, rhs, element_type);
            for operation in operations {
                scope.register(&*Operation::get_op_dyn(operation, scope.ctx()));
            }
            return Ok(result);
        }
        if opcode == Opcode::Remainder {
            return Ok(emit_integer_remainder(scope, lhs, rhs, element_type));
        }
        return match opcode {
            Opcode::Add => Ok(scope.register_with_result(&IAddOp::new(scope.ctx_mut(), lhs, rhs))),
            Opcode::Subtract => {
                Ok(scope.register_with_result(&ISubOp::new(scope.ctx_mut(), lhs, rhs)))
            }
            Opcode::Multiply => {
                Ok(scope.register_with_result(&IMulOp::new(scope.ctx_mut(), lhs, rhs)))
            }
            opcode => Err(PlanError::UnsupportedFeature(format!(
                "xlahlo operation {opcode}"
            ))),
        };
    }
    Err(PlanError::UnsupportedFeature(format!(
        "arithmetic for element type {element_type:?}"
    )))
}

fn make_integer_division(
    context: &mut Context,
    lhs: Value,
    rhs: Value,
    element: ElementType,
) -> (Vec<Ptr<Operation>>, Value) {
    let width = element.bit_width();
    let mut operations = Vec::new();
    macro_rules! append {
        ($operation:expr) => {{
            let operation = $operation.get_operation();
            let result = operation.deref(context).get_result(0);
            operations.push(operation);
            result
        }};
    }
    macro_rules! constant {
        ($bits:expr) => {{
            let signedness = if element.is_signed() {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            let ty = IntegerType::get(context, width, signedness);
            let attr = IntegerAttr::new(ty, APInt::from_u64($bits, bw(width as usize)));
            append!(CubeConstantOp::new(context, Box::new(attr)))
        }};
    }
    let zero = constant!(0);
    let one = constant!(1);
    let all_ones = constant!(u64::MAX >> (64 - width));
    let divide_by_zero = append!(IEqualOp::new(context, rhs, zero));
    let (unsafe_divisor, overflow) = if element.is_signed() {
        let minimum = constant!(1u64 << (width - 1));
        let is_minimum = append!(IEqualOp::new(context, lhs, minimum));
        let is_negative_one = append!(IEqualOp::new(context, rhs, all_ones));
        let overflow = append!(BoolAndOp::new(context, is_minimum, is_negative_one));
        (
            append!(BoolOrOp::new(context, divide_by_zero, overflow)),
            Some((overflow, minimum)),
        )
    } else {
        (divide_by_zero, None)
    };
    // Select a safe divisor before division; selecting the result cannot undo UB.
    let divisor = append!(CubeSelectOp::new(context, unsafe_divisor, one, rhs));
    let quotient = if element.is_signed() {
        append!(SDivOp::new(context, lhs, divisor))
    } else {
        append!(UDivOp::new(context, lhs, divisor))
    };
    let quotient = if let Some((overflow, minimum)) = overflow {
        append!(CubeSelectOp::new(context, overflow, minimum, quotient))
    } else {
        quotient
    };
    let result = append!(CubeSelectOp::new(
        context,
        divide_by_zero,
        all_ones,
        quotient
    ));
    (operations, result)
}

fn emit_integer_remainder(scope: &Scope, lhs: Value, rhs: Value, element: ElementType) -> Value {
    let width = element.bit_width();
    let signedness = if element.is_signed() {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    };
    let constant = |bits| {
        let ty = IntegerType::get(scope.ctx(), width, signedness);
        let attr = IntegerAttr::new(ty, APInt::from_u64(bits, bw(width as usize)));
        scope.register_with_result(&CubeConstantOp::new(scope.ctx_mut(), Box::new(attr)))
    };
    let zero = constant(0);
    let one = constant(1);
    let divide_by_zero = scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), rhs, zero));
    let overflow = if element.is_signed() {
        let minimum = constant(1_u64 << (width - 1));
        let minus_one = constant(u64::MAX >> (64 - width));
        let lhs_is_minimum =
            scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), lhs, minimum));
        let rhs_is_minus_one =
            scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), rhs, minus_one));
        Some(scope.register_with_result(&BoolAndOp::new(
            scope.ctx_mut(),
            lhs_is_minimum,
            rhs_is_minus_one,
        )))
    } else {
        None
    };
    let unsafe_divisor = if let Some(overflow) = overflow {
        scope.register_with_result(&BoolOrOp::new(scope.ctx_mut(), divide_by_zero, overflow))
    } else {
        divide_by_zero
    };
    let divisor = scope.register_with_result(&CubeSelectOp::new(
        scope.ctx_mut(),
        unsafe_divisor,
        one,
        rhs,
    ));
    let quotient = if element.is_signed() {
        scope.register_with_result(&SDivOp::new(scope.ctx_mut(), lhs, divisor))
    } else {
        scope.register_with_result(&UDivOp::new(scope.ctx_mut(), lhs, divisor))
    };
    let quotient = if let Some(overflow) = overflow {
        scope.register_with_result(&CubeSelectOp::new(
            scope.ctx_mut(),
            overflow,
            zero,
            quotient,
        ))
    } else {
        quotient
    };
    let product = scope.register_with_result(&IMulOp::new(scope.ctx_mut(), quotient, rhs));
    let remainder = scope.register_with_result(&ISubOp::new(scope.ctx_mut(), lhs, product));
    if let Some(overflow) = overflow {
        scope.register_with_result(&CubeSelectOp::new(
            scope.ctx_mut(),
            overflow,
            zero,
            remainder,
        ))
    } else {
        remainder
    }
}

fn make_binary_operation(
    context: &mut Context,
    opcode: Opcode,
    lhs: Value,
    rhs: Value,
    element_type: ElementType,
) -> Result<Ptr<Operation>, String> {
    if element_type.is_float() {
        return match opcode {
            Opcode::Add => Ok(FAddOp::new(context, lhs, rhs).get_operation()),
            Opcode::Subtract => Ok(FSubOp::new(context, lhs, rhs).get_operation()),
            Opcode::Multiply => Ok(FMulOp::new(context, lhs, rhs).get_operation()),
            Opcode::Divide => Ok(FDivOp::new(context, lhs, rhs).get_operation()),
            opcode => Err(format!("xlahlo operation {opcode}")),
        };
    }
    if element_type.is_integer() {
        return match opcode {
            Opcode::Add => Ok(IAddOp::new(context, lhs, rhs).get_operation()),
            Opcode::Subtract => Ok(ISubOp::new(context, lhs, rhs).get_operation()),
            Opcode::Multiply => Ok(IMulOp::new(context, lhs, rhs).get_operation()),
            Opcode::Divide if element_type.is_signed() => {
                Ok(SDivOp::new(context, lhs, rhs).get_operation())
            }
            Opcode::Divide => Ok(UDivOp::new(context, lhs, rhs).get_operation()),
            opcode => Err(format!("xlahlo operation {opcode}")),
        };
    }
    Err(format!("arithmetic for element type {element_type:?}"))
}

fn make_rounded_float_product(
    context: &mut Context,
    lhs: Value,
    rhs: Value,
    element_type: ElementType,
) -> Result<(Vec<Ptr<Operation>>, Value), String> {
    let (bit_type, width, sign_mask) = match element_type {
        ElementType::F32 => (ElemType::UInt(UIntKind::U32), 32, 0x8000_0000),
        ElementType::F64 => (ElemType::UInt(UIntKind::U64), 64, 0x8000_0000_0000_0000),
        _ => return Err(format!("rounded product for element type {element_type:?}")),
    };
    let magnitude_mask = !sign_mask;
    let bit_type = bit_type.to_type(context);
    let mut operations = Vec::with_capacity(16);
    macro_rules! append {
        ($operation:expr) => {{
            let operation = $operation.get_operation();
            let result = operation.deref(context).get_result(0);
            operations.push(operation);
            result
        }};
    }
    macro_rules! constant {
        ($bits:expr) => {{
            let ty = IntegerType::get(context, width, Signedness::Unsigned);
            let attr = IntegerAttr::new(ty, APInt::from_u64($bits, bw(width as usize)));
            append!(CubeConstantOp::new(context, Box::new(attr)))
        }};
    }
    let lhs_bits = append!(ReinterpretCastOp::new(context, bit_type, lhs));
    let rhs_bits = append!(ReinterpretCastOp::new(context, bit_type, rhs));
    let magnitude_mask = constant!(magnitude_mask);
    let lhs_magnitude = append!(BitwiseAndOp::new(context, lhs_bits, magnitude_mask));
    let rhs_magnitude = append!(BitwiseAndOp::new(context, rhs_bits, magnitude_mask));
    let float_type = lhs.get_type(context);
    let lhs_magnitude = append!(ReinterpretCastOp::new(context, float_type, lhs_magnitude));
    let rhs_magnitude = append!(ReinterpretCastOp::new(context, float_type, rhs_magnitude));
    let product = append!(FMulOp::new(context, lhs_magnitude, rhs_magnitude));
    let product_bits = append!(ReinterpretCastOp::new(context, bit_type, product));
    let product_mask = constant!(!sign_mask);
    let product_magnitude = append!(BitwiseAndOp::new(context, product_bits, product_mask));
    let sign_mask = constant!(sign_mask);
    let lhs_sign = append!(BitwiseAndOp::new(context, lhs_bits, sign_mask));
    let rhs_sign = append!(BitwiseAndOp::new(context, rhs_bits, sign_mask));
    let sign = append!(BitwiseXorOp::new(context, lhs_sign, rhs_sign));
    let result_bits = append!(BitwiseOrOp::new(context, product_magnitude, sign));
    let result = append!(ReinterpretCastOp::new(context, float_type, result_bits));
    Ok((operations, result))
}

pub(super) fn make_add_positive_zero(
    context: &mut Context,
    lhs: Value,
    rhs: Value,
    dynamic_value: Value,
    element_type: ElementType,
) -> Option<(Vec<Ptr<Operation>>, Value)> {
    let (bit_type, width, sign_mask) = match element_type {
        ElementType::F16 | ElementType::BF16 => (ElemType::UInt(UIntKind::U16), 16, 0x8000),
        ElementType::F32 => (ElemType::UInt(UIntKind::U32), 32, 0x8000_0000),
        ElementType::F64 => (ElemType::UInt(UIntKind::U64), 64, 0x8000_0000_0000_0000),
        _ => return None,
    };
    let bit_type = bit_type.to_type(context);
    let float_type = dynamic_value.get_type(context);
    let mut operations = Vec::with_capacity(7);
    macro_rules! append {
        ($operation:expr) => {{
            let operation = $operation.get_operation();
            let result = operation.deref(context).get_result(0);
            operations.push(operation);
            result
        }};
    }
    macro_rules! constant {
        ($bits:expr) => {{
            let ty = IntegerType::get(context, width, Signedness::Unsigned);
            let attr = IntegerAttr::new(ty, APInt::from_u64($bits, bw(width as usize)));
            append!(CubeConstantOp::new(context, Box::new(attr)))
        }};
    }
    let bits = append!(ReinterpretCastOp::new(context, bit_type, dynamic_value));
    let zero_bits = constant!(0);
    let positive_zero = append!(ReinterpretCastOp::new(context, float_type, zero_bits));
    let sum = append!(FAddOp::new(context, lhs, rhs));
    let negative_zero = constant!(sign_mask);
    let is_negative_zero = append!(IEqualOp::new(context, bits, negative_zero));
    let result = append!(CubeSelectOp::new(
        context,
        is_negative_zero,
        positive_zero,
        sum
    ));
    Some((operations, result))
}

pub(in crate::lowering) fn make_binary_operations(
    context: &mut Context,
    opcode: Opcode,
    lhs: Value,
    rhs: Value,
    element_type: ElementType,
) -> Result<(Vec<Ptr<Operation>>, Value), String> {
    if opcode == Opcode::Divide && element_type.is_integer() {
        return Ok(make_integer_division(context, lhs, rhs, element_type));
    }
    if opcode == Opcode::Multiply && matches!(element_type, ElementType::F32 | ElementType::F64) {
        return make_rounded_float_product(context, lhs, rhs, element_type);
    }
    let operation = make_binary_operation(context, opcode, lhs, rhs, element_type)?;
    let result = operation.deref(context).get_result(0);
    Ok((vec![operation], result))
}
