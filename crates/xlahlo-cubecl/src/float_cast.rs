use cubecl_ir::{
    ElemType, FloatKind, Scope, UIntKind,
    dialect::{
        bitwise::{BitwiseAndOp, BitwiseOrOp, LeadingZerosBitsOp, ShiftLeftOp, ShiftRightOp},
        cmp::{IEqualOp, UGreaterThanOp, ULessThanOp},
        general::{CastOp, ReinterpretCastOp, SelectOp as CubeSelectOp},
        math::{IAddOp, ISubOp},
    },
};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};

pub(crate) fn cast_float(scope: &Scope, ty: TypeHandle, input: Value) -> Value {
    let source = input.get_type(scope.ctx());
    if source == ty {
        return input;
    }
    let bf16 = ElemType::Float(FloatKind::BF16).to_type(scope.ctx());
    let f16 = ElemType::Float(FloatKind::F16).to_type(scope.ctx());
    let f32 = ElemType::Float(FloatKind::F32).to_type(scope.ctx());
    let u16 = ElemType::UInt(UIntKind::U16).to_type(scope.ctx());
    let u32 = ElemType::UInt(UIntKind::U32).to_type(scope.ctx());
    if ty == f16 {
        let input = cast_float(scope, f32, input);
        let bits = scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), u32, input));
        let bits = f16_bits(scope, bits, true);
        let bits = scope.register_with_result(&CastOp::new(scope.ctx_mut(), u16, bits));
        return scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), f16, bits));
    }
    if ty == bf16 {
        let input = cast_float(scope, f32, input);
        let bits = scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), u32, input));
        let shift = u32_constant(scope, 16);
        let upper = scope.register_with_result(&ShiftRightOp::new(scope.ctx_mut(), bits, shift));
        let one = u32_constant(scope, 1);
        let parity = scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), upper, one));
        // Explicit integer rounding avoids backend BF16 conversion libcalls.
        let bias = u32_constant(scope, 0x7fff);
        let bias = scope.register_with_result(&IAddOp::new(scope.ctx_mut(), bias, parity));
        let rounded = scope.register_with_result(&IAddOp::new(scope.ctx_mut(), bits, bias));
        let rounded =
            scope.register_with_result(&ShiftRightOp::new(scope.ctx_mut(), rounded, shift));
        let signless = u32_constant(scope, 0x7fff_ffff);
        let signless =
            scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), bits, signless));
        let infinity = u32_constant(scope, 0x7f80_0000);
        let nan =
            scope.register_with_result(&UGreaterThanOp::new(scope.ctx_mut(), signless, infinity));
        let quiet = u32_constant(scope, 0x40);
        let quiet = scope.register_with_result(&BitwiseOrOp::new(scope.ctx_mut(), upper, quiet));
        let selected =
            scope.register_with_result(&CubeSelectOp::new(scope.ctx_mut(), nan, quiet, rounded));
        let bits = scope.register_with_result(&CastOp::new(scope.ctx_mut(), u16, selected));
        return scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), bf16, bits));
    }
    let input = if source == bf16 || source == f16 {
        let bits = scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), u16, input));
        let bits = scope.register_with_result(&CastOp::new(scope.ctx_mut(), u32, bits));
        let bits = if source == f16 {
            f16_bits(scope, bits, false)
        } else {
            let shift = u32_constant(scope, 16);
            scope.register_with_result(&ShiftLeftOp::new(scope.ctx_mut(), bits, shift))
        };
        scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), f32, bits))
    } else {
        input
    };
    if ty == input.get_type(scope.ctx()) {
        input
    } else {
        scope.register_with_result(&CastOp::new(scope.ctx_mut(), ty, input))
    }
}

fn f16_bits(scope: &Scope, bits: Value, narrow: bool) -> Value {
    let c = |value| u32_constant(scope, value);
    macro_rules! bin {
        ($op:ty, $lhs:expr, $rhs:expr) => {{
            let lhs = $lhs;
            let rhs = $rhs;
            scope.register_with_result(&<$op>::new(scope.ctx_mut(), lhs, rhs))
        }};
    }
    let select = |condition, on_true, on_false| {
        scope.register_with_result(&CubeSelectOp::new(
            scope.ctx_mut(),
            condition,
            on_true,
            on_false,
        ))
    };
    if narrow {
        let sign = bin!(ShiftRightOp, bits, c(16));
        let sign = bin!(BitwiseAndOp, sign, c(0x8000));
        let abs = bin!(BitwiseAndOp, bits, c(0x7fff_ffff));
        let upper = bin!(ShiftRightOp, abs, c(13));
        let parity = bin!(BitwiseAndOp, upper, c(1));
        let bias = bin!(IAddOp, c(0xfff), parity);
        let rounded = bin!(IAddOp, abs, bias);
        let normal = bin!(ISubOp, rounded, c(0x3800_0000));
        let normal = bin!(ShiftRightOp, normal, c(13));
        let exponent = bin!(ShiftRightOp, abs, c(23));
        let exponent = select(bin!(ULessThanOp, exponent, c(102)), c(102), exponent);
        let exponent = select(bin!(UGreaterThanOp, exponent, c(112)), c(112), exponent);
        let shift = bin!(ISubOp, c(126), exponent);
        let mantissa = bin!(BitwiseAndOp, abs, c(0x7f_ffff));
        let mantissa = bin!(BitwiseOrOp, mantissa, c(0x80_0000));
        let retained = bin!(ShiftRightOp, mantissa, shift);
        let parity = bin!(BitwiseAndOp, retained, c(1));
        let half_shift = bin!(ISubOp, shift, c(1));
        let half = bin!(ShiftLeftOp, c(1), half_shift);
        let bias = bin!(ISubOp, half, c(1));
        let bias = bin!(IAddOp, bias, parity);
        let rounded = bin!(IAddOp, mantissa, bias);
        let subnormal = bin!(ShiftRightOp, rounded, shift);
        let subnormal = select(bin!(UGreaterThanOp, abs, c(0x3300_0000)), subnormal, c(0));
        let finite = select(bin!(ULessThanOp, abs, c(0x3880_0000)), subnormal, normal);
        let finite = select(bin!(UGreaterThanOp, abs, c(0x477f_efff)), c(0x7c00), finite);
        let payload = bin!(ShiftRightOp, abs, c(13));
        let payload = bin!(BitwiseAndOp, payload, c(0x3ff));
        let nan = bin!(BitwiseOrOp, payload, c(0x7e00));
        let result = select(bin!(UGreaterThanOp, abs, c(0x7f80_0000)), nan, finite);
        bin!(BitwiseOrOp, sign, result)
    } else {
        let sign = bin!(BitwiseAndOp, bits, c(0x8000));
        let sign = bin!(ShiftLeftOp, sign, c(16));
        let exponent = bin!(ShiftRightOp, bits, c(10));
        let exponent = bin!(BitwiseAndOp, exponent, c(0x1f));
        let mantissa = bin!(BitwiseAndOp, bits, c(0x3ff));
        let payload = bin!(ShiftLeftOp, mantissa, c(13));
        let normal_exp = bin!(IAddOp, exponent, c(112));
        let normal_exp = bin!(ShiftLeftOp, normal_exp, c(23));
        let normal = bin!(BitwiseOrOp, normal_exp, payload);
        let safe_mantissa = bin!(BitwiseOrOp, mantissa, c(1));
        let leading =
            scope.register_with_result(&LeadingZerosBitsOp::new(scope.ctx_mut(), safe_mantissa));
        let shift = bin!(ISubOp, leading, c(21));
        let normalized = bin!(ShiftLeftOp, mantissa, shift);
        let normalized = bin!(BitwiseAndOp, normalized, c(0x3ff));
        let normalized = bin!(ShiftLeftOp, normalized, c(13));
        let sub_exp = bin!(ISubOp, c(113), shift);
        let sub_exp = bin!(ShiftLeftOp, sub_exp, c(23));
        let subnormal = bin!(BitwiseOrOp, sub_exp, normalized);
        let subnormal = select(bin!(IEqualOp, mantissa, c(0)), c(0), subnormal);
        let finite = select(bin!(IEqualOp, exponent, c(0)), subnormal, normal);
        let special = bin!(BitwiseOrOp, c(0x7f80_0000), payload);
        let result = select(bin!(IEqualOp, exponent, c(31)), special, finite);
        bin!(BitwiseOrOp, sign, result)
    }
}

fn u32_constant(scope: &Scope, value: u32) -> Value {
    let attr = IntegerAttr::new(
        IntegerType::get(scope.ctx(), 32, Signedness::Unsigned),
        APInt::from_u64(u64::from(value), bw(32)),
    );
    scope.register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(attr)))
}
