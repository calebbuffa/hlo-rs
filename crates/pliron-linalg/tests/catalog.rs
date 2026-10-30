#![cfg(feature = "mlir-interop")]

#[path = "../../../tests/support/mod.rs"]
mod support;

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, FP64Type, IntegerType, Signedness},
    context::Context,
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::TypeHandle,
};
use pliron_linalg::attributes::{ElementwiseKind, IteratorType, WinogradFmr};
use pliron_linalg::{ops::*, pliron_mlir_builtin::types::TensorType};
use pliron_mlir_interop::MlirPrinter;

fn emit(ctx: &Context, op: &impl Op) {
    let ptr = op.get_operation();
    Operation::get_op_dyn(ptr, ctx)
        .verify(ctx)
        .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    let text =
        String::try_from(MlirPrinter::new(ctx, &ptr)).unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    support::record(ctx, ptr, &text);
}

#[test]
fn all_named_operations() {
    let ctx = &mut Context::new();
    let integer: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let dummy = TensorType::from_element(ctx, integer, vec![1, 1]);
    let block = BasicBlock::new(ctx, None, vec![dummy]);
    let dummy = block.deref(ctx).arguments().next().unwrap();
    macro_rules! named {
        ($($ty:ident),+ $(,)?) => {$(
            let probe = $ty::new(ctx, vec![], dummy);
            let maps = probe.indexing_maps(ctx);
            let types = maps.iter().enumerate().map(|(i, m)| {
                if m.results.is_empty() && i + 1 < maps.len() { integer }
                else { TensorType::from_element(ctx, integer, vec![1; m.results.len()]) }
            }).collect();
            let block = BasicBlock::new(ctx, None, types);
            let mut values: Vec<_> = block.deref(ctx).arguments().collect();
            let output = values.pop().unwrap();
            let op = $ty::new(ctx, values, output);
            emit(ctx, &op);
        )+};
    }

    fn all_structured_operations() {
        let ctx = &mut Context::new();
        let f32 = FP32Type::get(ctx).into();
        let tensor = |shape: &[i64]| TensorType::from_element(ctx, f32, shape.to_vec());
        let block = BasicBlock::new(
            ctx,
            None,
            vec![
                tensor(&[2, 4]),
                tensor(&[4, 3]),
                tensor(&[2, 3]),
                tensor(&[2, 3]),
                tensor(&[2]),
            ],
        );
        let v: Vec<_> = block.deref(ctx).arguments().collect();
        let matmul = MatmulOp::new(ctx, v[0], v[1], v[2]);
        let maps = matmul.indexing_maps(ctx);
        let op = ContractOp::new(ctx, v[0], v[1], v[2], maps.clone());
        emit(ctx, &op);
        let op = ScaledContractOp::new(
            ctx,
            [v[0], v[0], v[1], v[1]],
            v[2],
            vec![
                maps[0].clone(),
                maps[0].clone(),
                maps[1].clone(),
                maps[1].clone(),
                maps[2].clone(),
            ],
        );
        emit(ctx, &op);
        let op = ElementwiseOp::new(ctx, ElementwiseKind::Add, vec![v[2], v[3]], v[2]);
        emit(ctx, &op);
        let transposed = TensorType::from_element(ctx, f32, vec![3, 4]);
        let b = BasicBlock::new(ctx, None, vec![transposed]);
        let destination = b.deref(ctx).arguments().next().unwrap();
        let op = TransposeOp::new(ctx, v[1], destination, vec![1, 0]);
        emit(ctx, &op);
        let op = SoftmaxOp::new(ctx, v[2], v[3], 1);
        emit(ctx, &op);
        let op = MapOp::build(
            ctx,
            vec![v[2]],
            v[3],
            Box::new(|_, _, args| Ok(vec![args[0]])),
        )
        .unwrap();
        emit(ctx, &op);
        let op = ReduceOp::build(
            ctx,
            vec![v[2]],
            vec![v[4]],
            vec![1],
            Box::new(|ctx, block, args| {
                let add = pliron_arith::ops::AddFOp::new(ctx, args[0], args[1]);
                add.get_operation().insert_at_back(block, ctx);
                Ok(vec![add.result(ctx)])
            }),
        )
        .unwrap();
        emit(ctx, &op);
        let identity = pliron_linalg::attributes::identity_map(2);
        let op = GenericOp::build(
            ctx,
            vec![v[2]],
            vec![v[3]],
            vec![identity.clone(), identity],
            vec![IteratorType::Parallel; 2],
            Box::new(|ctx, block, args| {
                let index = IndexOp::new(ctx, 0);
                index.get_operation().insert_at_back(block, ctx);
                Ok(vec![args[0]])
            }),
        )
        .unwrap();
        emit(ctx, &op);

        let batch_a = TensorType::from_element(ctx, f32, vec![2, 3, 4]);
        let batch_b = TensorType::from_element(ctx, f32, vec![2, 4, 5]);
        let batch_c = TensorType::from_element(ctx, f32, vec![2, 3, 5]);
        let reduced_c = TensorType::from_element(ctx, f32, vec![3, 5]);
        let b = BasicBlock::new(ctx, None, vec![batch_a, batch_b, batch_c, reduced_c]);
        let a: Vec<_> = b.deref(ctx).arguments().collect();
        let op = BatchMatmulOp::new(ctx, a[0], a[1], a[2]);
        emit(ctx, &op);
        let op = BatchReduceMatmulOp::new(ctx, a[0], a[1], a[3]);
        emit(ctx, &op);

        let b = BasicBlock::new(
            ctx,
            None,
            vec![
                TensorType::from_element(ctx, f32, vec![4, 3, 3, 2]),
                TensorType::from_element(ctx, f32, vec![4, 4, 2, 4]),
                TensorType::from_element(ctx, f32, vec![1, 6, 6, 2]),
                TensorType::from_element(ctx, f32, vec![4, 4, 2, 2, 1, 2]),
                TensorType::from_element(ctx, f32, vec![1, 4, 4, 2]),
            ],
        );
        let a: Vec<_> = b.deref(ctx).arguments().collect();
        let op = WinogradFilterTransformOp::new(ctx, a[0], a[1], WinogradFmr::F2x3);
        emit(ctx, &op);
        let op = WinogradInputTransformOp::new(ctx, a[2], a[3], WinogradFmr::F2x3);
        emit(ctx, &op);
        let op = WinogradOutputTransformOp::new(ctx, a[3], a[4], WinogradFmr::F2x3);
        emit(ctx, &op);
        let b = BasicBlock::new(
            ctx,
            None,
            vec![
                TensorType::from_element(ctx, f32, vec![4, 8, 4]),
                TensorType::from_element(ctx, f32, vec![16, 8]),
            ],
        );
        let a: Vec<_> = b.deref(ctx).arguments().collect();
        let op = UnPackOp::new(ctx, a[0], a[1], vec![0], vec![TileSize::Static(4)], vec![]);
        emit(ctx, &op);
    }
    named!(
        CopyOp,
        QuantizedMatmulOp,
        Mmt4DOp,
        BatchMmt4DOp,
        QuantizedBatchMatmulOp,
        MatvecOp,
        VecmatOp,
        BatchMatvecOp,
        BatchVecmatOp,
        DotOp,
        Conv1DOp,
        Conv2DOp,
        Conv3DOp,
        Conv1DNwcWcfOp,
        Conv1DNcwFcwOp,
        Conv2DNhwcHwcfOp,
        Conv2DNhwcFhwcOp,
        Conv2DNhwcHwcfQOp,
        Conv2DNhwcFhwcQOp,
        Conv2DNchwFchwQOp,
        Conv2DNchwFchwOp,
        Conv2DNgchwFgchwOp,
        Conv2DNgchwGfchwOp,
        Conv2DNhwgcGfhwcOp,
        Conv2DNhwgcGfhwcQOp,
        Conv2DNgchwGfchwQOp,
        Conv3DNdhwcDhwcfOp,
        Conv3DNdhwcDhwcfQOp,
        Conv3DNcdhwFcdhwOp,
        DepthwiseConv1DNwcWcOp,
        DepthwiseConv1DNcwCwOp,
        DepthwiseConv1DNwcWcmOp,
        DepthwiseConv2DNhwcHwcOp,
        DepthwiseConv2DNchwChwOp,
        DepthwiseConv2DNhwcHwcQOp,
        DepthwiseConv2DNhwcHwcmOp,
        DepthwiseConv2DNhwcHwcmQOp,
        DepthwiseConv3DNdhwcDhwcOp,
        DepthwiseConv3DNcdhwCdhwOp,
        DepthwiseConv3DNdhwcDhwcmOp,
        PoolingNhwcSumOp,
        PoolingNchwSumOp,
        PoolingNhwcMaxOp,
        PoolingNhwcMaxUnsignedOp,
        PoolingNchwMaxOp,
        PoolingNhwcMinOp,
        PoolingNhwcMinUnsignedOp,
        PoolingNwcSumOp,
        PoolingNcwSumOp,
        PoolingNwcMaxOp,
        PoolingNwcMaxUnsignedOp,
        PoolingNcwMaxOp,
        PoolingNwcMinOp,
        PoolingNwcMinUnsignedOp,
        PoolingNdhwcSumOp,
        PoolingNdhwcMaxOp,
        PoolingNdhwcMinOp,
        FillOp
    );
    let f64 = FP64Type::get(ctx).into();
    let output = TensorType::from_element(ctx, FP32Type::get(ctx).into(), vec![2, 2]);
    let block = BasicBlock::new(ctx, None, vec![f64, f64, integer, output]);
    let v: Vec<_> = block.deref(ctx).arguments().collect();
    let rng = FillRng2DOp::new(ctx, vec![v[0], v[1], v[2]], v[3]);
    emit(ctx, &rng);
    all_structured_operations();
}
