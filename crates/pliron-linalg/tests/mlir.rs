#![cfg(feature = "mlir-interop")]
//! Builds, verifies, and prints representative `linalg` operations.
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
use pliron_linalg::{
    attributes::TypeFn,
    ops::{BroadcastOp, Conv2DNhwcHwcfOp, MatmulOp, PackOp, TileSize},
    stablehlo::types::TensorType,
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
    let a = tensor(ctx, &[4, 8]);
    let b = tensor(ctx, &[8, 16]);
    let c = tensor(ctx, &[4, 16]);
    let v = args(ctx, vec![a, b, c]);
    let matmul = MatmulOp::new(ctx, v[0], v[1], v[2]);
    matmul.set_cast(ctx, TypeFn::CastUnsigned);
    assert_eq!(matmul.cast(ctx), TypeFn::CastUnsigned);
    mlir(ctx, matmul.get_operation());

    let image = tensor(ctx, &[1, 9, 9, 3]);
    let filter = tensor(ctx, &[3, 3, 3, 8]);
    let out = tensor(ctx, &[1, 4, 4, 8]);
    let v = args(ctx, vec![image, filter, out]);
    let conv = Conv2DNhwcHwcfOp::new(ctx, vec![v[0], v[1]], v[2]);
    conv.set_strides(ctx, vec![2, 2]);
    assert_eq!(conv.strides(ctx), vec![2, 2]);
    assert_eq!(conv.dilations(ctx), vec![1, 1]);
    let text = mlir(ctx, conv.get_operation());
    assert!(
        text.contains("strides = dense<[2, 2]> : tensor<2xi64>"),
        "{text}"
    );

    let src = tensor(ctx, &[16]);
    let dst = tensor(ctx, &[4, 16]);
    let v = args(ctx, vec![src, dst]);
    let broadcast = BroadcastOp::new(ctx, v[0], v[1], vec![0]);
    let text = mlir(ctx, broadcast.get_operation());
    assert!(
        text.contains("dimensions = [0]") || text.contains("dimensions = array<i64: 0>"),
        "{text}"
    );

    let src = tensor(ctx, &[16, 8]);
    let dst = tensor(ctx, &[4, 8, 4]);
    let v = args(ctx, vec![src, dst]);
    let pack = PackOp::new(
        ctx,
        v[0],
        v[1],
        vec![0],
        vec![TileSize::Static(4)],
        vec![],
        None,
    );
    let text = mlir(ctx, pack.get_operation());
    assert!(text.contains("inner_tiles"), "{text}");
}

#[test]
fn rejects_elided_and_misowned_attributes() {
    use pliron::builtin::attributes::BoolAttr;
    use pliron_linalg::{attributes::I64ArrayAttr, ops::CopyOp};
    let ctx = &mut Context::new();
    let ty = tensor(ctx, &[2, 2]);
    let v = args(ctx, vec![ty, ty, ty]);
    let matmul = MatmulOp::new(ctx, v[0], v[1], v[2]);
    matmul.set_indexing_maps(ctx, matmul.indexing_maps(ctx));
    let key = matmul
        .get_operation()
        .deref(ctx)
        .attributes
        .0
        .keys()
        .next()
        .cloned()
        .unwrap();
    matmul
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .insert(key, Box::new(I64ArrayAttr::new(vec![])));
    assert!(String::try_from(MlirPrinter::new(ctx, &matmul.get_operation())).is_err());
    let copy = CopyOp::new(ctx, vec![v[0]], v[1]);
    copy.get_operation().deref_mut(ctx).attributes.0.insert(
        pliron_linalg::ops::ATTR_KEY_LINALG_STRIDES.clone(),
        Box::new(I64ArrayAttr::new(vec![1])),
    );
    assert!(String::try_from(MlirPrinter::new(ctx, &copy.get_operation())).is_err());
    copy.get_operation().deref_mut(ctx).attributes.0.insert(
        pliron_linalg::ops::ATTR_KEY_LINALG_CAST.clone(),
        Box::new(BoolAttr::from(true)),
    );
    assert!(String::try_from(MlirPrinter::new(ctx, &copy.get_operation())).is_err());
}
