#![cfg(feature = "mlir-interop")]
//! Builds, verifies, and prints representative `arith` operations.
#[path = "../../../tests/support/mod.rs"]
mod support;

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, IntegerType, Signedness},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::TypeHandle,
    value::Value,
};
use pliron_arith::{
    attributes::{FastMathFlags, RoundingMode},
    ops::{AddFOp, AddIOp, DivSIOp, ExtUIOp},
    pliron_mlir_builtin::types::TensorType,
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
fn shared_attributes_print() {
    let ctx = &mut Context::new();
    let ty = tensor(ctx, &[4]);
    let i32 = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i8 = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let v = args(ctx, vec![ty, ty, i32, i32, i8]);
    let add = AddFOp::new(ctx, v[0], v[1]);
    add.set_fastmath(ctx, FastMathFlags::FAST);
    add.set_rounding_mode(ctx, RoundingMode::Downward);
    let text = mlir(ctx, add.get_operation());
    assert!(text.contains("fastmath = #arith.fastmath<reassoc,nnan,ninf,nsz,arcp,contract,afn>, roundingmode = 1 : i32"), "{text}");
    let addi = AddIOp::new(ctx, v[2], v[3]);
    let text = mlir(ctx, addi.get_operation());
    assert!(!text.contains("<{"), "{text}");
    let div = DivSIOp::new(ctx, v[2], v[3]);
    div.set_exact(ctx, true);
    assert!(div.is_exact(ctx));
    let text = mlir(ctx, div.get_operation());
    assert!(text.contains("<{isExact = unit}>"), "{text}");
    let ext = ExtUIOp::new(ctx, v[4], i32);
    ext.set_non_negative(ctx, true);
    let text = mlir(ctx, ext.get_operation());
    assert!(text.contains("<{nonNeg = unit}>"), "{text}");
}

#[test]
fn complete_operation_catalog() {
    use pliron::builtin::types::{BF16Type, FP16Type};
    use pliron_arith::{attributes::*, ops::*, shaped};
    use pliron_mlir_builtin::attributes::IndexAttr;
    let ctx = &mut Context::new();
    for (dims, vector) in [
        (None, None),
        (Some(vec![4]), None),
        (Some(vec![-1, 4]), None),
        (Some(vec![4]), Some(false)),
        (Some(vec![4]), Some(true)),
    ] {
        let shape = |ty| match &dims {
            None => ty,
            Some(dims) => match vector {
                None => TensorType::from_element(ctx, ty, dims.clone()),
                Some(scalable) => pliron_mlir_builtin::types::VectorType::get(
                    ctx,
                    ty,
                    dims.clone(),
                    vec![scalable; dims.len()],
                )
                .into(),
            },
        };
        let f16 = shape(FP16Type::get(ctx).into());
        let f32 = shape(FP32Type::get(ctx).into());
        let bf16 = shape(BF16Type::get(ctx).into());
        let i8 = shape(IntegerType::get(ctx, 8, Signedness::Signless).into());
        let i32 = shape(IntegerType::get(ctx, 32, Signedness::Signless).into());
        let index = shape(shaped::index(ctx));
        let condition = shape(shaped::i1(ctx));
        let v = args(
            ctx,
            vec![f16, f32, f32, bf16, i8, i32, i32, index, condition],
        );
        macro_rules! binary {
            ($a:expr, $b:expr; $($ty:ident),+ $(,)?) => {$(
                let op = $ty::new(ctx, $a, $b);
                mlir(ctx, op.get_operation());
            )+};
        }

        binary!(v[1], v[2]; AddFOp, SubFOp, MulFOp, DivFOp, RemFOp,
            MaximumFOp, MinimumFOp, MaxNumFOp, MinNumFOp, MaximumNumFOp, MinimumNumFOp);
        binary!(v[5], v[6]; AddIOp, SubIOp, MulIOp, ShLIOp, DivSIOp,
            DivUIOp, CeilDivSIOp, CeilDivUIOp, FloorDivSIOp, RemSIOp,
            RemUIOp, AndIOp, OrIOp, XOrIOp, ShRSIOp, ShRUIOp,
            MaxSIOp, MinSIOp, MaxUIOp, MinUIOp,
            AddUIExtendedOp, SubUIExtendedOp, MulSIExtendedOp, MulUIExtendedOp);
        let neg = NegFOp::new(ctx, v[1]);
        mlir(ctx, neg.get_operation());
        let flush = FlushDenormalsOp::new(ctx, v[1]);
        mlir(ctx, flush.get_operation());
        let cmp = CmpFOp::new(ctx, CmpFPredicate::Olt, v[1], v[2]);
        mlir(ctx, cmp.get_operation());
        let cmp = CmpIOp::new(ctx, CmpIPredicate::Slt, v[5], v[6]);
        mlir(ctx, cmp.get_operation());
        let select = SelectOp::new(ctx, v[8], v[1], v[2]);
        mlir(ctx, select.get_operation());
        macro_rules! cast {
            ($($ty:ident($input:expr, $result:expr)),+ $(,)?) => {$(
                let op = $ty::new(ctx, $input, $result);
                mlir(ctx, op.get_operation());
            )+};
        }
        cast!(
            ExtFOp(v[0], f32),
            TruncFOp(v[1], f16),
            ConvertFOp(v[0], bf16),
            ExtSIOp(v[4], i32),
            ExtUIOp(v[4], i32),
            TruncIOp(v[5], i8),
            SIToFPOp(v[5], f32),
            UIToFPOp(v[5], f32),
            FPToSIOp(v[1], i32),
            FPToUIOp(v[1], i32),
            IndexCastOp(v[5], index),
            IndexCastUIOp(v[7], i32),
            BitcastOp(v[1], i32)
        );
        let ext = ScalingExtFOp::new(ctx, v[0], v[0], f32);
        mlir(ctx, ext.get_operation());
        let trunc = ScalingTruncFOp::new(ctx, v[1], v[1], f16);
        mlir(ctx, trunc.get_operation());
    }
    let index = shaped::index(ctx);
    let constant = ConstantOp::new(ctx, Box::new(IndexAttr { value: 7 }), index);
    mlir(ctx, constant.get_operation());
}

#[test]
fn rejects_wrong_property_owner_type_and_aliases() {
    use pliron::builtin::attributes::BoolAttr;
    use pliron_arith::{attributes::CmpFPredicate, ops::CmpFOp};
    let ctx = &mut Context::new();
    let ty = FP32Type::get(ctx).into();
    let v = args(ctx, vec![ty, ty]);
    let cmp = CmpFOp::new(ctx, CmpFPredicate::Olt, v[0], v[1]);
    let add = AddFOp::new(ctx, v[0], v[1]);
    let predicate = cmp.get_operation().deref(ctx).attributes.0.clone();
    add.get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .extend(predicate);
    assert!(String::try_from(MlirPrinter::new(ctx, &add.get_operation())).is_err());
    *cmp.get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .values_mut()
        .next()
        .unwrap() = Box::new(BoolAttr::from(true));
    assert!(String::try_from(MlirPrinter::new(ctx, &cmp.get_operation())).is_err());
    struct Aliases<'a>(&'a Context, pliron::context::Ptr<Operation>);
    impl std::fmt::Display for Aliases<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            pliron_arith::mlir::print_generic(self.0, self.1, &Default::default(), f, &|_, _, _| {
                Ok(Some(("same", Some("unit".into()))))
            })
            .map_err(|_| std::fmt::Error)
        }
    }
    add.set_fastmath(ctx, pliron_arith::attributes::FastMathFlags::NONE);
    let mut text = String::new();
    assert!(
        std::fmt::write(
            &mut text,
            format_args!("{}", Aliases(ctx, add.get_operation()))
        )
        .is_err()
    );
}
