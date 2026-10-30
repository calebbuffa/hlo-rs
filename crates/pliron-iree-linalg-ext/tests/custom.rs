#![cfg(feature = "mlir-interop")]
#[path = "../../../tests/support/mod.rs"]
mod support;

use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, IntegerType, Signedness},
    context::Context,
    op::Op,
    printable::Printable,
    r#type::TypeHandle,
    value::Value,
};
use pliron_arith::{attributes::CmpFPredicate, ops::CmpFOp, shaped};
use pliron_iree_linalg_ext::ops::*;
use pliron_linalg::attributes::{IteratorType, projection_map};
use pliron_mlir_interop::MlirPrinter;
use stablehlo::types::TensorType;

fn tensor(ctx: &Context, dims: &[i64]) -> TypeHandle {
    TensorType::from_element(ctx, FP32Type::get(ctx).into(), dims.to_vec())
}
fn integer_tensor(ctx: &Context, dims: &[i64], width: u32) -> TypeHandle {
    TensorType::from_element(
        ctx,
        IntegerType::get(ctx, width, Signedness::Signless).into(),
        dims.to_vec(),
    )
}
fn args(ctx: &mut Context, types: Vec<TypeHandle>) -> Vec<Value> {
    let block = BasicBlock::new(ctx, None, types);
    block.deref(ctx).arguments().collect()
}
fn custom(ctx: &Context, op: &impl Op, fragments: &[&str]) -> String {
    let text = String::try_from(MlirPrinter::new(ctx, &op.get_operation()))
        .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    assert!(!text.contains("\"iree_linalg_ext."), "{text}");
    assert!(!text.contains("operandSegmentSizes"), "{text}");
    for fragment in fragments {
        assert!(text.contains(fragment), "missing {fragment:?} in {text}");
    }
    support::record(ctx, op.get_operation(), &text);
    struct Generic<'a>(
        &'a Context,
        pliron::context::Ptr<pliron::operation::Operation>,
    );
    impl std::fmt::Display for Generic<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            pliron_iree_linalg_ext::mlir::print_generic(self.0, self.1, &Default::default(), f)
                .unwrap_or_else(|e| panic!("{}", e.disp(self.0)));
            Ok(())
        }
    }
    let generic = format!("{}", Generic(ctx, op.get_operation()));
    support::record(ctx, op.get_operation(), &generic);
    text
}
fn comparator() -> Box<BodyBuilder<'static>> {
    Box::new(|ctx, block, args| {
        let cmp = CmpFOp::new(ctx, CmpFPredicate::Olt, args[0], args[1]);
        cmp.get_operation().insert_at_back(block, ctx);
        Ok(vec![cmp.get_operation().deref(ctx).get_result(0)])
    })
}

#[test]
fn selection_optional_indices_and_index_base() {
    let ctx = &mut Context::new();
    let input = tensor(ctx, &[8]);
    let out = tensor(ctx, &[2]);
    let in_indices = integer_tensor(ctx, &[8], 32);
    let out_indices = integer_tensor(ctx, &[2], 32);
    let scalar = tensor(ctx, &[]);
    let scalar_index = integer_tensor(ctx, &[], 32);
    let index = shaped::index(ctx);
    let v = args(
        ctx,
        vec![
            input,
            out,
            in_indices,
            out_indices,
            scalar,
            scalar_index,
            index,
        ],
    );
    let sort = SortOp::build(ctx, vec![v[0]], vec![v[0]], 0, comparator()).unwrap();
    custom(
        ctx,
        &sort,
        &["sort dimension(0) ins(", " outs(", "iree_linalg_ext.yield"],
    );
    for indices in [None, Some(v[2])] {
        let topk = TopkOp::build(ctx, v[0], indices, v[1], v[3], 0, comparator()).unwrap();
        custom(
            ctx,
            &topk,
            &[
                "topk dimension(0) ins(",
                " outs(",
                " -> tensor<2xf32>, tensor<2xi32>",
            ],
        );
        let topk2 =
            TopkV2Op::build(ctx, v[0], indices, v[1], Some(v[3]), 0, true, comparator()).unwrap();
        custom(
            ctx,
            &topk2,
            &["topk_v2 dimension(0) is_sorted ins(", " outs("],
        );
    }
    let values_only = TopkV2Op::build(ctx, v[0], None, v[1], None, 0, false, comparator()).unwrap();
    let text = custom(ctx, &values_only, &[" -> tensor<2xf32>"]);
    assert!(!text.contains("is_sorted"), "{text}");
    for input_index in [None, Some(v[2])] {
        let base = input_index.is_none().then_some(v[6]);
        let arg =
            ArgCompareOp::build(ctx, v[0], input_index, v[4], v[5], base, 0, comparator()).unwrap();
        let text = custom(
            ctx,
            &arg,
            &["arg_compare dimension(0)", " -> tensor<f32>, tensor<i32>"],
        );
        if base.is_some() {
            assert!(text.contains(" : index)"), "{text}");
            assert!(
                text.find("index_base(").unwrap() < text.find(" {").unwrap(),
                "{text}"
            );
        } else {
            assert!(!text.contains("index_base("), "{text}");
        }
    }
    let scan = ScanOp::build(
        ctx,
        v[0],
        v[0],
        v[4],
        0,
        true,
        Box::new(|_, _, args| Ok(vec![args[0]])),
    )
    .unwrap();
    custom(
        ctx,
        &scan,
        &["scan dimension(0) inclusive(true) ins(", " outs("],
    );
}

#[test]
fn indexed_data_movement_and_regions() {
    let ctx = &mut Context::new();
    let source = tensor(ctx, &[8, 4]);
    let output = tensor(ctx, &[2, 4]);
    let indices = integer_tensor(ctx, &[2, 1], 32);
    let f32 = FP32Type::get(ctx).into();
    let i1 = shaped::i1(ctx);
    let mask_tensor = integer_tensor(ctx, &[2], 1);
    let v = args(ctx, vec![source, output, indices, f32, i1, mask_tensor]);
    for mask in [None, Some(v[5])] {
        let gather = GatherOp::build(ctx, v[0], v[2], mask, v[1], vec![0]).unwrap();
        custom(ctx, &gather, &["gather dimension_map = [0] ins(", " outs("]);
        let scatter = ScatterOp::build(
            ctx,
            v[1],
            v[2],
            mask,
            v[0],
            vec![0],
            false,
            Box::new(|_, _, args| Ok(vec![args[0]])),
        )
        .unwrap();
        custom(
            ctx,
            &scatter,
            &["scatter dimension_map = [0] unique_indices(false) ins("],
        );
    }
    let pad = v[3];
    let load = MapLoadOp::build(
        ctx,
        v[0],
        v[1],
        Box::new(move |_, _, args| Ok(vec![args[0], args[1], pad])),
    )
    .unwrap();
    custom(
        ctx,
        &load,
        &[
            "map_load %",
            " into %",
            " : tensor<8x4xf32> into tensor<2x4xf32> ->",
        ],
    );
    let mask = v[4];
    let store = MapStoreOp::build(
        ctx,
        v[1],
        v[0],
        Box::new(move |_, _, args| Ok(vec![args[0], args[1], mask])),
    )
    .unwrap();
    custom(
        ctx,
        &store,
        &[
            "map_store %",
            " into %",
            " : tensor<2x4xf32> into tensor<8x4xf32> ->",
        ],
    );
}

#[test]
fn fft_and_grouped_contractions() {
    let ctx = &mut Context::new();
    let t = tensor(ctx, &[8]);
    let index = shaped::index(ctx);
    let v = args(ctx, vec![t, t, index]);
    for coefficients in [None, Some((v[0], v[1]))] {
        let fft = FftOp::build(ctx, v[2], coefficients, v[0], v[1]).unwrap();
        let text = custom(
            ctx,
            &fft,
            &["fft ins(", " outs(", ") : tensor<8xf32>, tensor<8xf32>"],
        );
        assert!(!text.contains(" -> "), "{text}");
    }
    let input = tensor(ctx, &[4, 8]);
    let weights = tensor(ctx, &[2, 8, 16]);
    let offsets = integer_tensor(ctx, &[2], 32);
    let output = tensor(ctx, &[4, 16]);
    let v = args(ctx, vec![input, weights, offsets, index, output]);
    let matmul = GroupMatmulOp::build(ctx, v[0], v[1], v[2], v[3], v[4]).unwrap();
    custom(
        ctx,
        &matmul,
        &["group_matmul ins(", " outs(", " -> tensor<4x16xf32>"],
    );
    let input = tensor(ctx, &[2, 3, 4, 5]);
    let weights = tensor(ctx, &[2, 6, 3, 7, 5]);
    let output = tensor(ctx, &[6, 2, 7, 4]);
    let v = args(ctx, vec![input, weights, offsets, index, index, output]);
    let mmt = GroupMmt4dOp::build(ctx, v[0], v[1], v[2], v[3], v[4], v[5], true).unwrap();
    custom(
        ctx,
        &mmt,
        &["group_mmt4d <transposed = true> ins(", " outs("],
    );
}

#[test]
fn online_attention_custom_and_exponential_regions() {
    let ctx = &mut Context::new();
    let q = tensor(ctx, &[4, 16]);
    let k = tensor(ctx, &[8, 16]);
    let val = tensor(ctx, &[8, 32]);
    let out = tensor(ctx, &[4, 32]);
    let max = tensor(ctx, &[4]);
    let f32 = FP32Type::get(ctx).into();
    let v = args(ctx, vec![q, k, val, f32, out, max, max]);
    let maps = vec![
        projection_map(4, &[0, 1]),
        projection_map(4, &[2, 1]),
        projection_map(4, &[2, 3]),
        projection_map(4, &[]),
        projection_map(4, &[0, 3]),
        projection_map(4, &[0]),
        projection_map(4, &[0]),
    ];
    let online = OnlineAttentionOp::build(
        ctx,
        v[0],
        v[1],
        v[2],
        v[3],
        None,
        v[4],
        v[5],
        v[6],
        maps,
        Box::new(|_, _, args| Ok(vec![args[0]])),
    )
    .unwrap();
    custom(
        ctx,
        &online,
        &[
            "online_attention <indexing_maps = [",
            "> ins(",
            " outs(",
            " -> tensor<4x32xf32>, tensor<4xf32>, tensor<4xf32>",
        ],
    );
    let t = tensor(ctx, &[8]);
    let v = args(ctx, vec![t, t, t]);
    let maps = vec![projection_map(1, &[0]); 3];
    let exp = ExpReductionOp::build(
        ctx,
        vec![v[0]],
        vec![v[1], v[2]],
        maps,
        vec![IteratorType::Parallel],
        vec![1],
        Box::new(|_, _, args| Ok(vec![args[1], args[2]])),
    )
    .unwrap();
    custom(
        ctx,
        &exp,
        &[
            "exp_reduction {indexing_maps = [",
            "iterator_types = [#iree_linalg_ext.iterator_type<parallel>]",
            "exp_reduced_operands = [1]}",
            " outs(",
        ],
    );
    let custom_op = CustomOp::build(
        ctx,
        vec![v[0]],
        vec![v[1]],
        vec![projection_map(1, &[0]); 2],
        vec![IteratorType::Parallel],
        Box::new(|ctx, block, args| {
            let index = IndexOp::new(ctx, 0);
            index.get_operation().insert_at_back(block, ctx);
            Ok(vec![args[1]])
        }),
    )
    .unwrap();
    custom(
        ctx,
        &custom_op,
        &[
            "custom_op {indexing_maps = [",
            "iterator_types = [#iree_linalg_ext.iterator_type<parallel>]}",
            "iree_linalg_ext.index 0 : index",
            "iree_linalg_ext.yield",
        ],
    );
}

#[test]
fn affine_quantization_properties() {
    let ctx = &mut Context::new();
    let input = tensor(ctx, &[8]);
    let output = integer_tensor(ctx, &[8], 8);
    let f32 = FP32Type::get(ctx).into();
    let v = args(ctx, vec![input, f32, output]);
    let maps = vec![
        projection_map(1, &[0]),
        projection_map(1, &[]),
        projection_map(1, &[0]),
    ];
    let bounds = QuantizeBounds { min: 0, max: 255 };
    let quant = QuantizeAffineOp::build(
        ctx,
        v[0],
        v[1],
        None,
        v[2],
        maps.clone(),
        bounds,
        true,
        false,
    )
    .unwrap();
    custom(
        ctx,
        &quant,
        &[
            "quantize_affine <indexing_maps = [",
            "quant_min = 0",
            "quant_max = 255",
            "storage_unsigned> ins(",
        ],
    );
    let dequant =
        DequantizeAffineOp::build(ctx, v[2], v[1], None, v[0], maps, Some(bounds), true, false)
            .unwrap();
    custom(
        ctx,
        &dequant,
        &[
            "dequantize_affine <indexing_maps = [",
            "input_unsigned",
            "> ins(",
        ],
    );
}

#[test]
fn winograd_degenerate_parameters_and_overflow_do_not_panic() {
    let ctx = &mut Context::new();
    let empty = tensor(ctx, &[0, 0]);
    let one = tensor(ctx, &[1, 1]);
    let dynamic = tensor(ctx, &[-1, -1]);
    let v = args(ctx, vec![empty, one, dynamic]);
    // These rank-two cases are admitted by upstream; positivity was an extra restriction.
    let input = WinogradInputTransformOp::build(ctx, v[0], v[0], 0, 1, vec![]).unwrap();
    custom(
        ctx,
        &input,
        &["winograd.input_transform output_tile_size(0) kernel_size(1) image_dimensions([])"],
    );
    let filter = WinogradFilterTransformOp::build(ctx, v[1], v[0], 0, 1, vec![]).unwrap();
    custom(
        ctx,
        &filter,
        &["winograd.filter_transform output_tile_size(0) kernel_size(1) kernel_dimensions([])"],
    );
    let output = WinogradOutputTransformOp::build(ctx, v[0], v[0], 0, 1, vec![]).unwrap();
    custom(
        ctx,
        &output,
        &["winograd.output_transform output_tile_size(0) kernel_size(1) image_dimensions([])"],
    );
    let huge =
        WinogradInputTransformOp::build(ctx, v[2], v[2], i64::MAX, i64::MAX, vec![]).unwrap();
    assert_eq!(huge.input_tile_size(ctx), None);
    custom(ctx, &huge, &["output_tile_size(9223372036854775807)"]);
    let image = tensor(ctx, &[1, 8, 8, 1]);
    let out = tensor(ctx, &[-1; 6]);
    let v = args(ctx, vec![image, out]);
    assert!(WinogradInputTransformOp::build(ctx, v[0], v[1], 0, 1, vec![1, 2]).is_err());
}
