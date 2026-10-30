#![cfg(feature = "mlir-interop")]
//! Builds, verifies, and prints representative `math` operations.
#[path = "../../../tests/support/mod.rs"]
mod support;

use pliron::{
    basic_block::BasicBlock,
    builtin::types::FP32Type,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::TypeHandle,
    value::Value,
};
use pliron_math::{
    ops::{AbsFOp, FmaOp},
    pliron_arith::{
        attributes::FastMathFlags,
        stablehlo::{self, types::TensorType},
    },
};
use pliron_mlir_interop::MlirPrinter;

fn tensor(ctx: &Context, dims: &[i64]) -> TypeHandle {
    let f32 = FP32Type::get(ctx).into();
    TensorType::from_element(ctx, f32, dims.to_vec())
}

fn args(ctx: &mut Context, types: Vec<TypeHandle>) -> Vec<Value> {
    let block = BasicBlock::new(ctx, None, types);
    block.deref(ctx).arguments().collect()
}

fn mlir(ctx: &Context, op: Ptr<Operation>) -> String {
    Operation::get_op_dyn(op, ctx)
        .verify(ctx)
        .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    let text =
        String::try_from(MlirPrinter::new(ctx, &op)).unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    support::record(ctx, op, &text);
    text
}

#[test]
fn fastmath_prints() {
    let ctx = &mut Context::new();
    let ty = tensor(ctx, &[4]);
    let v = args(ctx, vec![ty, ty, ty]);
    let abs = AbsFOp::new(ctx, v[0]);
    let text = mlir(ctx, abs.get_operation());
    assert!(text.contains("\"math.absf\"(%"), "{text}");
    assert!(!text.contains("<{"), "{text}");
    let fma = FmaOp::new(ctx, v[0], v[1], v[2]);
    fma.set_fastmath(ctx, FastMathFlags::FAST);
    assert_eq!(fma.fastmath(ctx), FastMathFlags::FAST);
    let text = mlir(ctx, fma.get_operation());
    assert!(text.contains("fastmath = #arith.fastmath<"), "{text}");
}

#[test]
fn complete_operation_catalog() {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron_math::ops::*;
    let ctx = &mut Context::new();
    for (dims, vector) in [
        (None, None),
        (Some(vec![4]), None),
        (Some(vec![-1, 4]), None),
        (Some(vec![4]), Some(false)),
        (Some(vec![4]), Some(true)),
    ] {
        let float = FP32Type::get(ctx).into();
        let integer = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let (float, integer) = match &dims {
            None => (float, integer),
            Some(dims) => match vector {
                None => (
                    TensorType::from_element(ctx, float, dims.clone()),
                    TensorType::from_element(ctx, integer, dims.clone()),
                ),
                Some(scalable) => (
                    stablehlo::types::VectorType::get(
                        ctx,
                        float,
                        dims.clone(),
                        vec![scalable; dims.len()],
                    )
                    .into(),
                    stablehlo::types::VectorType::get(
                        ctx,
                        integer,
                        dims.clone(),
                        vec![scalable; dims.len()],
                    )
                    .into(),
                ),
            },
        };
        let v = args(ctx, vec![float, float, float, integer, integer]);
        macro_rules! unary {
            ($value:expr; $($ty:ident),+ $(,)?) => {$(
                let op = $ty::new(ctx, $value);
                mlir(ctx, op.get_operation());
            )+};
        }

        unary!(v[0]; AbsFOp, CeilOp, FloorOp, RoundOp, RoundEvenOp, TruncOp,
            ExpOp, Exp2Op, ExpM1Op, LogOp, Log2Op, Log10Op, Log1pOp,
            SqrtOp, RsqrtOp, CbrtOp, SinOp, CosOp, TanOp, AsinOp, AcosOp,
            AtanOp, SinhOp, CoshOp, TanhOp, AsinhOp, AcoshOp, AtanhOp,
            ErfOp, ErfcOp, IsNaNOp, IsInfOp, IsFiniteOp, IsNormalOp, SincosOp);
        unary!(v[3]; AbsIOp, CountLeadingZerosOp, CountTrailingZerosOp, CtPopOp);
        macro_rules! binary {
            ($a:expr, $b:expr; $($ty:ident),+) => {$(
                let op = $ty::new(ctx, $a, $b);
                mlir(ctx, op.get_operation());
            )+};
        }
        binary!(v[0], v[1]; Atan2Op, PowFOp, CopySignOp);
        binary!(v[3], v[4]; IPowIOp);
        binary!(v[0], v[3]; FPowIOp);
        let fma = FmaOp::new(ctx, v[0], v[1], v[2]);
        mlir(ctx, fma.get_operation());
        let clamp = ClampFOp::new(ctx, v[0], v[1], v[2]);
        mlir(ctx, clamp.get_operation());
    }
}

#[test]
fn rejects_fastmath_on_integer_operations() {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron_math::ops::{AbsIOp, ExpOp};
    let ctx = &mut Context::new();
    let float = FP32Type::get(ctx).into();
    let integer = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let v = args(ctx, vec![float, integer]);
    let exp = ExpOp::new(ctx, v[0]);
    exp.set_fastmath(ctx, FastMathFlags::NONE);
    let abs = AbsIOp::new(ctx, v[1]);
    let flags = exp.get_operation().deref(ctx).attributes.0.clone();
    abs.get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .extend(flags);
    assert!(String::try_from(MlirPrinter::new(ctx, &abs.get_operation())).is_err());
}
