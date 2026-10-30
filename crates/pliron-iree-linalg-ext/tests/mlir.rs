#![cfg(feature = "mlir-interop")]
//! Builds, verifies, and prints representative `iree_linalg_ext` operations.
#[path = "../../../tests/support/mod.rs"]
mod support;

use pliron::{
    attribute::{AttrObj, Attribute},
    basic_block::BasicBlock,
    builtin::{
        attributes::{DictAttr, StringAttr, UnitAttr},
        types::FP32Type,
    },
    context::{Context, Ptr},
    identifier::Identifier,
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::TypeHandle,
    value::Value,
};
use pliron_arith::{attributes::CmpFPredicate, ops::CmpFOp};
use pliron_iree_linalg_ext::{
    attributes::SplitReductionMappingAttr,
    ops::{
        AttentionOp, DYNAMIC_INDEX, Im2colOp, Im2colPadding, Im2colPositions, MixedIndex, SortOp,
        WinogradInputTransformOp,
    },
};
use pliron_linalg::attributes::{I64ArrayAttr, projection_map};
use pliron_mlir_builtin::{attributes::OpaqueAttr, types::TensorType};
use pliron_mlir_interop::MlirPrinter;

fn f32(ctx: &Context) -> TypeHandle {
    FP32Type::get(ctx).into()
}

fn tensor(ctx: &Context, dims: &[i64]) -> TypeHandle {
    let e = f32(ctx);
    TensorType::from_element(ctx, e, dims.to_vec())
}

fn args(ctx: &mut Context, types: Vec<TypeHandle>) -> Vec<Value> {
    let block = BasicBlock::new(ctx, None, types);
    block.deref(ctx).arguments().collect()
}

fn mlir(ctx: &Context, op: Ptr<Operation>) -> String {
    struct Generic<'a>(&'a Context, Ptr<Operation>);
    impl std::fmt::Display for Generic<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            pliron_iree_linalg_ext::mlir::print_generic(self.0, self.1, &Default::default(), f)
                .map_err(|e| panic!("{}", e.disp(self.0)))
        }
    }
    let custom =
        String::try_from(MlirPrinter::new(ctx, &op)).unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    assert!(!custom.contains("\"iree_linalg_ext."), "{custom}");
    support::record(ctx, op, &custom);
    format!("{}", Generic(ctx, op))
}

fn id(name: &str) -> Identifier {
    Identifier::try_from(name).unwrap()
}

fn less_than(ctx: &mut Context, input: Value, output: Value) -> pliron::result::Result<SortOp> {
    SortOp::build(
        ctx,
        vec![input],
        vec![output],
        0,
        Box::new(|ctx, block, args| {
            let cmp = CmpFOp::new(ctx, CmpFPredicate::Olt, args[0], args[1]);
            cmp.get_operation().insert_at_back(block, ctx);
            Ok(vec![cmp.get_operation().deref(ctx).get_result(0)])
        }),
    )
}

#[test]
fn sort_prints_comparator() {
    let ctx = &mut Context::new();
    let ty = tensor(ctx, &[8]);
    let v = args(ctx, vec![ty, ty]);
    let op = less_than(ctx, v[0], v[1]).unwrap();
    let text = mlir(ctx, op.get_operation());
    assert!(text.contains("\"iree_linalg_ext.sort\""), "{text}");
    assert!(text.contains("dimension = 0 : i64"), "{text}");
    assert!(
        text.contains("operandSegmentSizes = array<i32: 1, 1>"),
        "{text}"
    );
    assert!(text.contains("arith.cmpf"), "{text}");
    assert!(text.contains("iree_linalg_ext.yield"), "{text}");
}

#[test]
fn sort_rejects_mismatched_shapes() {
    let ctx = &mut Context::new();
    let a = tensor(ctx, &[8]);
    let b = tensor(ctx, &[4]);
    let v = args(ctx, vec![a, b]);
    assert!(less_than(ctx, v[0], v[1]).is_err());
}

#[test]
fn rejects_wrong_property_schema_and_owner() {
    use pliron::builtin::attributes::BoolAttr;
    use pliron_iree_linalg_ext::{mlir::property, ops::ScanOp};
    let ctx = &mut Context::new();
    assert!(property(ctx, "iree_topk_v2_is_sorted", &BoolAttr::from(true)).is_err());
    assert!(
        property(ctx, "iree_sort_inclusive", &BoolAttr::from(true))
            .unwrap()
            .is_none()
    );
    assert!(
        property(ctx, "foreign_operand_segment_sizes", &UnitAttr)
            .unwrap()
            .is_none()
    );
    let ty = tensor(ctx, &[4]);
    let scalar = tensor(ctx, &[]);
    let v = args(ctx, vec![ty, ty, scalar]);
    let sort = less_than(ctx, v[0], v[1]).unwrap();
    let scan = ScanOp::new(ctx, v[0], v[1], v[2], 0, true);
    let attrs = scan.get_operation().deref(ctx).attributes.0.clone();
    sort.get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .extend(attrs);
    assert!(String::try_from(MlirPrinter::new(ctx, &sort.get_operation())).is_err());
}

#[test]
fn winograd_prints_dotted_upstream_name() {
    let ctx = &mut Context::new();
    let input = tensor(ctx, &[1, 10, 10, 4]);
    let output = tensor(ctx, &[8, 8, 1, 2, 2, 4]);
    let v = args(ctx, vec![input, output]);
    let op = WinogradInputTransformOp::build(ctx, v[0], v[1], 6, 3, vec![1, 2]).unwrap();
    let text = mlir(ctx, op.get_operation());
    assert!(
        text.contains("\"iree_linalg_ext.winograd.input_transform\""),
        "{text}"
    );
    assert!(text.contains("output_tile_size = 6 : i64"), "{text}");
    assert!(
        text.contains("image_dimensions = array<i64: 1, 2>"),
        "{text}"
    );
}

#[test]
fn winograd_rejects_wrong_output_shape() {
    let ctx = &mut Context::new();
    let input = tensor(ctx, &[1, 10, 10, 4]);
    let output = tensor(ctx, &[8, 8, 1, 3, 2, 4]);
    let v = args(ctx, vec![input, output]);
    assert!(WinogradInputTransformOp::build(ctx, v[0], v[1], 6, 3, vec![1, 2]).is_err());
}

#[test]
fn im2col_prints_mixed_offsets() {
    let ctx = &mut Context::new();
    let input = tensor(ctx, &[2, 34, 34, 640]);
    let output = tensor(ctx, &[2, 1024, 5760]);
    let index = pliron_arith::shaped::index(ctx);
    let v = args(ctx, vec![input, output, index]);
    let s = MixedIndex::Static;
    let op = Im2colOp::build(
        ctx,
        v[0],
        v[1],
        vec![1, 1],
        vec![1, 1],
        &[s(3), s(3)],
        &[s(0), MixedIndex::Dynamic(v[2]), s(0)],
        &[vec![s(2)], vec![s(32), s(32)], vec![s(3), s(3), s(640)]],
        Im2colPositions {
            batch: vec![0],
            m: vec![1, 2],
            k: vec![3],
            input_k_perm: vec![0, 1, 2],
            output_perm: vec![0, 1, 2],
        },
        Im2colPadding::default(),
    )
    .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    let text = mlir(ctx, op.get_operation());
    assert!(text.contains("\"iree_linalg_ext.im2col\""), "{text}");
    assert!(
        text.contains(&format!(
            "static_offsets = array<i64: 0, {DYNAMIC_INDEX}, 0>"
        )),
        "{text}"
    );
    assert!(
        text.contains(
            "static_output_sizes = [array<i64: 2>, array<i64: 32, 32>, array<i64: 3, 3, 640>]"
        ),
        "{text}"
    );
    assert_eq!(op.mixed_offsets(ctx).unwrap().len(), 3);
    let custom = String::try_from(MlirPrinter::new(ctx, &op.get_operation())).unwrap();
    assert!(
        custom.contains("im2col strides = [1, 1] dilations = [1, 1] kernel_size = [3, 3]"),
        "{custom}"
    );
    assert!(custom.contains("offsets = [0, %"), "{custom}");
    assert!(
        custom.contains("output_sizes = [[2], [32, 32], [3, 3, 640]]"),
        "{custom}"
    );
    assert!(
        custom.contains("input_k_perm = [0, 1, 2] output_perm = [0, 1, 2] ins("),
        "{custom}"
    );
    op.set_attr_iree_im2col_static_offsets(
        ctx,
        I64ArrayAttr::new(vec![DYNAMIC_INDEX, DYNAMIC_INDEX, 0]),
    );
    assert!(op.mixed_offsets(ctx).is_err());
    assert!(String::try_from(MlirPrinter::new(ctx, &op.get_operation())).is_err());
}

#[test]
fn attention_prints_opaque_decomposition_config() {
    let ctx = &mut Context::new();
    let q = tensor(ctx, &[4, 16]);
    let k = tensor(ctx, &[8, 16]);
    let val = tensor(ctx, &[8, 32]);
    let out = tensor(ctx, &[4, 32]);
    let scale = f32(ctx);
    let v = args(ctx, vec![q, k, val, scale, out]);
    // Iteration space (m, k1, k2, n) = (d0, d1, d2, d3).
    let maps = vec![
        projection_map(4, &[0, 1]),
        projection_map(4, &[2, 1]),
        projection_map(4, &[2, 3]),
        projection_map(4, &[]),
        projection_map(4, &[0, 3]),
    ];
    let op = AttentionOp::build(
        ctx,
        v[0],
        v[1],
        v[2],
        v[3],
        None,
        v[4],
        maps,
        Box::new(|_, _, args| Ok(vec![args[0]])),
    )
    .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    let lowering: AttrObj = Box::new(OpaqueAttr {
        dialect: StringAttr::new("iree_gpu".into()),
        syntax: StringAttr::new("lowering_config<{workgroup = [64, 0, 0, 0]}>".into()),
    });
    let qk = DictAttr::new(vec![
        (id("attention_qk_matmul"), Box::new(UnitAttr::new())),
        (id("lowering_config"), lowering),
    ]);
    op.set_decomposition_config(ctx, DictAttr::new(vec![(id("qk_attrs"), Box::new(qk))]));
    let text = mlir(ctx, op.get_operation());
    assert!(text.contains("\"iree_linalg_ext.attention\""), "{text}");
    assert!(text.contains("decomposition_config = {"), "{text}");
    assert!(
        text.contains("#iree_gpu.lowering_config<{workgroup = [64, 0, 0, 0]}>"),
        "{text}"
    );
    let custom = String::try_from(MlirPrinter::new(ctx, &op.get_operation())).unwrap();
    assert!(custom.contains("attention <indexing_maps = ["), "{custom}");
    assert!(custom.contains("decomposition_config = {"), "{custom}");
    assert!(
        custom.contains("#iree_gpu.lowering_config<{workgroup = [64, 0, 0, 0]}>"),
        "{custom}"
    );
}

#[test]
fn split_reduction_mapping_list() {
    let ok: Vec<AttrObj> = vec![
        Box::new(SplitReductionMappingAttr::new(1)),
        Box::new(SplitReductionMappingAttr::new(0)),
    ];
    assert!(SplitReductionMappingAttr::verify_list(&ok).is_ok());
    let gap: Vec<AttrObj> = vec![Box::new(SplitReductionMappingAttr::new(1))];
    assert!(SplitReductionMappingAttr::verify_list(&gap).is_err());
    let repeat: Vec<AttrObj> = vec![
        Box::new(SplitReductionMappingAttr::new(0)),
        Box::new(SplitReductionMappingAttr::new(0)),
    ];
    assert!(SplitReductionMappingAttr::verify_list(&repeat).is_err());

    let ctx = Context::new();
    let attr: &dyn Attribute = &SplitReductionMappingAttr::new(2);
    let text = String::try_from(MlirPrinter::new(&ctx, attr)).unwrap();
    assert_eq!(text, "#iree_linalg_ext.split_reduction_mapping<2>");
}
