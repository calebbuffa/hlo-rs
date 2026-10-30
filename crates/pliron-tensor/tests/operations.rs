use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{OperandSegmentSizesAttr, UnitAttr},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface_impl, pliron_op},
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::TypeHandle,
    value::Value,
};
use pliron_arith::shaped;
use pliron_mlir_builtin::attributes::I64ArrayAttr;
use pliron_mlir_builtin::types::{TensorType, UnrankedTensorType};
use pliron_tensor::{Diagnostic, ops::*};
#[cfg(feature = "mlir-interop")]
#[path = "../../../tests/support/mod.rs"]
mod support;

fn tensor(ctx: &Context, shape: &[i64]) -> TypeHandle {
    TensorType::from_element(ctx, FP32Type::get(ctx).into(), shape.to_vec())
}
fn integer_tensor(ctx: &Context, shape: &[i64], width: u32) -> TypeHandle {
    let element = IntegerType::get(ctx, width, Signedness::Signless).into();
    TensorType::from_element(ctx, element, shape.to_vec())
}
fn args(ctx: &mut Context, types: Vec<TypeHandle>) -> Vec<Value> {
    let block = BasicBlock::new(ctx, None, types);
    block.deref(ctx).arguments().collect()
}
fn good(ctx: &Context, op: &impl Op) -> String {
    Operation::get_op_dyn(op.get_operation(), ctx)
        .verify(ctx)
        .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
    #[cfg(feature = "mlir-interop")]
    {
        let text = String::try_from(pliron_mlir_interop::MlirPrinter::new(
            ctx,
            &op.get_operation(),
        ))
        .unwrap_or_else(|e| panic!("{}", e.disp(ctx)));
        support::record(ctx, op.get_operation(), &text);
        text
    }
    #[cfg(not(feature = "mlir-interop"))]
    {
        op.disp(ctx).to_string()
    }
}
fn bad(ctx: &Context, op: &impl Op) {
    let error = Operation::get_op_dyn(op.get_operation(), ctx)
        .verify(ctx)
        .expect_err("invalid operation accepted");
    assert!(Diagnostic::from_error(&error).is_some(), "{error:?}");
}
macro_rules! valid {
    ($ctx:expr, $op:expr) => {{
        let op = $op;
        good($ctx, &op)
    }};
}
macro_rules! invalid {
    ($ctx:expr, $op:expr) => {{
        let op = $op;
        bad($ctx, &op)
    }};
}
fn static_slice(offsets: &[i64], sizes: &[i64], strides: &[i64]) -> Slice {
    let list = |v: &[i64]| v.iter().copied().map(MixedIndex::Static).collect();
    Slice {
        offsets: list(offsets),
        sizes: list(sizes),
        strides: list(strides),
    }
}

#[cfg(feature = "mlir-interop")]
#[test]
fn exporter_rejects_properties_from_another_tensor_operation() {
    let ctx = &mut Context::new();
    let ty = tensor(ctx, &[4]);
    let v = args(ctx, vec![ty]);
    let cast = CastOp::new(ctx, v[0], ty);
    cast.get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .insert(pliron_tensor::ops::NOFOLD.clone(), Box::new(UnitAttr));
    assert!(
        String::try_from(pliron_mlir_interop::MlirPrinter::new(
            ctx,
            &cast.get_operation()
        ))
        .is_err()
    );
}

#[test]
fn catalog_matches_upstream() {
    let names = [
        BitcastOp::get_concrete_op_info(),
        CastOp::get_concrete_op_info(),
        ConcatOp::get_concrete_op_info(),
        DimOp::get_concrete_op_info(),
        EmptyOp::get_concrete_op_info(),
        ExtractOp::get_concrete_op_info(),
        ExtractSliceOp::get_concrete_op_info(),
        FromElementsOp::get_concrete_op_info(),
        GatherOp::get_concrete_op_info(),
        GenerateOp::get_concrete_op_info(),
        InsertOp::get_concrete_op_info(),
        InsertSliceOp::get_concrete_op_info(),
        RankOp::get_concrete_op_info(),
        ReshapeOp::get_concrete_op_info(),
        ExpandShapeOp::get_concrete_op_info(),
        CollapseShapeOp::get_concrete_op_info(),
        PadOp::get_concrete_op_info(),
        ParallelInsertSliceOp::get_concrete_op_info(),
        ScatterOp::get_concrete_op_info(),
        SplatOp::get_concrete_op_info(),
        YieldOp::get_concrete_op_info(),
    ];
    assert_eq!(names.len(), 21);
    let ctx = &mut Context::new();
    let mut ids = std::collections::BTreeSet::new();
    for info in names {
        let op = Operation::new(ctx, info, vec![], vec![], vec![], 0);
        ids.insert(Operation::get_op_dyn(op, ctx).get_opid().to_string());
    }
    let expected = [
        "bitcast",
        "cast",
        "concat",
        "dim",
        "empty",
        "extract",
        "extract_slice",
        "from_elements",
        "gather",
        "generate",
        "insert",
        "insert_slice",
        "rank",
        "reshape",
        "expand_shape",
        "collapse_shape",
        "pad",
        "parallel_insert_slice",
        "scatter",
        "splat",
        "yield",
    ];
    assert_eq!(
        ids,
        expected
            .into_iter()
            .map(|n| format!("tensor.{n}"))
            .collect()
    );
}

#[test]
fn scalar_access_creation_and_queries() {
    let ctx = &mut Context::new();
    let f32 = FP32Type::get(ctx).into();
    let index = shaped::index(ctx);
    let dynamic = tensor(ctx, &[-1, 4]);
    let scalar = tensor(ctx, &[]);
    let empty = tensor(ctx, &[0]);
    let four = tensor(ctx, &[4]);
    let v = args(ctx, vec![dynamic, index, f32, scalar]);
    let op = EmptyOp::new(ctx, dynamic, vec![v[1]]);
    good(ctx, &op);
    invalid!(ctx, EmptyOp::new(ctx, dynamic, vec![]));
    valid!(ctx, SplatOp::new(ctx, v[2], dynamic, vec![v[1]]));
    invalid!(ctx, SplatOp::new(ctx, v[1], dynamic, vec![v[1]]));
    valid!(ctx, ExtractOp::new(ctx, v[0], vec![v[1], v[1]], f32));
    invalid!(ctx, ExtractOp::new(ctx, v[0], vec![v[1]], f32));
    valid!(ctx, ExtractOp::new(ctx, v[3], vec![], f32));
    valid!(ctx, InsertOp::new(ctx, v[2], v[0], vec![v[1], v[1]]));
    invalid!(ctx, InsertOp::new(ctx, v[1], v[0], vec![v[1], v[1]]));
    valid!(ctx, FromElementsOp::new(ctx, vec![v[2]; 4], four));
    valid!(ctx, FromElementsOp::new(ctx, vec![], empty));
    valid!(ctx, FromElementsOp::new(ctx, vec![v[2]], scalar));
    invalid!(ctx, FromElementsOp::new(ctx, vec![v[2]; 3], four));
    invalid!(ctx, FromElementsOp::new(ctx, vec![], dynamic));
    valid!(ctx, DimOp::new(ctx, v[0], v[1]));
    invalid!(ctx, DimOp::new(ctx, v[3], v[1]));
    valid!(ctx, RankOp::new(ctx, v[3]));
    invalid!(ctx, RankOp::new(ctx, v[2]));
}

#[test]
fn conversions_support_unranked_tensors() {
    let ctx = &mut Context::new();
    let f32 = FP32Type::get(ctx).into();
    let unranked: TypeHandle = UnrankedTensorType::get(ctx, f32).into();
    let known = tensor(ctx, &[4, 8]);
    let dynamic = tensor(ctx, &[-1, 8]);
    let wrong = tensor(ctx, &[4, 7]);
    let i32 = integer_tensor(ctx, &[4, 8], 32);
    let i16 = integer_tensor(ctx, &[4, 8], 16);
    let v = args(ctx, vec![known, unranked]);
    valid!(ctx, CastOp::new(ctx, v[0], dynamic));
    valid!(ctx, CastOp::new(ctx, v[0], unranked));
    valid!(ctx, CastOp::new(ctx, v[1], known));
    invalid!(ctx, CastOp::new(ctx, v[0], wrong));
    invalid!(ctx, CastOp::new(ctx, v[0], i32));
    valid!(ctx, BitcastOp::new(ctx, v[0], i32));
    invalid!(ctx, BitcastOp::new(ctx, v[0], i16));
}

#[test]
fn concat_checks_all_static_inputs_not_just_dynamic_result() {
    let ctx = &mut Context::new();
    let a = tensor(ctx, &[2, 4]);
    let b = tensor(ctx, &[3, 4]);
    let c = tensor(ctx, &[3, 8]);
    let out = tensor(ctx, &[5, 4]);
    let dyn_out = tensor(ctx, &[-1, -1]);
    let v = args(ctx, vec![a, b, c]);
    let op = ConcatOp::new(ctx, vec![v[0], v[1]], 0, out);
    let text = good(ctx, &op);
    #[cfg(feature = "mlir-interop")]
    assert!(text.contains("dim = 0 : i64"), "{text}");
    #[cfg(not(feature = "mlir-interop"))]
    let _ = text;
    invalid!(ctx, ConcatOp::new(ctx, vec![v[0], v[2]], 0, dyn_out));
    invalid!(ctx, ConcatOp::new(ctx, vec![], 0, out));
    invalid!(ctx, ConcatOp::new(ctx, vec![v[0], v[1]], 2, out));
}

#[test]
fn slices_cover_rank_reduction_dynamic_lists_and_bounds() {
    let ctx = &mut Context::new();
    let source = tensor(ctx, &[4, 8]);
    let reduced = tensor(ctx, &[4]);
    let full = tensor(ctx, &[1, 4]);
    let dynamic = tensor(ctx, &[1, -1]);
    let index = shaped::index(ctx);
    let v = args(ctx, vec![source, reduced, full, index]);
    let slice = static_slice(&[0, 0], &[1, 4], &[1, 2]);
    valid!(ctx, ExtractSliceOp::new(ctx, v[0], reduced, slice.clone()));
    valid!(ctx, ExtractSliceOp::new(ctx, v[0], full, slice.clone()));
    valid!(ctx, InsertSliceOp::new(ctx, v[1], v[0], slice.clone()));
    // Unknown and static extents cannot be interchanged during rank reduction.
    invalid!(ctx, ExtractSliceOp::new(ctx, v[0], dynamic, slice.clone()));
    invalid!(
        ctx,
        ExtractSliceOp::new(ctx, v[0], reduced, static_slice(&[0, 2], &[1, 4], &[1, 2]))
    );
    invalid!(
        ctx,
        ExtractSliceOp::new(ctx, v[0], reduced, static_slice(&[4, 0], &[1, 4], &[1, 1]))
    );
    let mixed = Slice {
        offsets: vec![MixedIndex::Static(0), MixedIndex::Dynamic(v[3])],
        sizes: vec![MixedIndex::Static(1), MixedIndex::Dynamic(v[3])],
        strides: vec![MixedIndex::Static(1), MixedIndex::Static(1)],
    };
    let op = ExtractSliceOp::new(ctx, v[0], dynamic, mixed);
    let text = good(ctx, &op);
    #[cfg(feature = "mlir-interop")]
    {
        assert!(
            text.contains("operandSegmentSizes = array<i32: 1, 1, 1, 0>"),
            "{text}"
        );
        assert!(text.contains("-9223372036854775808"), "{text}");
    }
    #[cfg(not(feature = "mlir-interop"))]
    let _ = text;
    op.get_operation()
        .deref_mut(ctx)
        .attributes
        .set(SEGMENTS.clone(), OperandSegmentSizesAttr(vec![1, 0, 1, 1]));
    bad(ctx, &op);
    let parallel = ParallelInsertSliceOp::new(ctx, v[1], v[0], slice);
    bad(ctx, &parallel);
}

#[pliron_op(
    name = "tensor_test.in_parallel",
    format = "`{` region($0) `}` `:` typesig"
)]
struct ParallelParent;
#[op_interface_impl]
impl InParallelOpInterface for ParallelParent {}
impl Verify for ParallelParent {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        Ok(())
    }
}

#[test]
fn parallel_slice_accepts_an_interface_parent() {
    let ctx = &mut Context::new();
    let source = tensor(ctx, &[4]);
    let dest = tensor(ctx, &[8]);
    let v = args(ctx, vec![source, dest]);
    let parent = Operation::new(
        ctx,
        ParallelParent::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(parent.deref(ctx).get_region(0), ctx);
    let op = ParallelInsertSliceOp::new(ctx, v[0], v[1], static_slice(&[2], &[4], &[1]));
    op.get_operation().insert_at_back(block, ctx);
    good(ctx, &op);
}

#[test]
fn reassociation_and_runtime_reshape() {
    let ctx = &mut Context::new();
    let flat = tensor(ctx, &[8]);
    let matrix = tensor(ctx, &[2, 4]);
    let wrong = tensor(ctx, &[3, 4]);
    let shape = integer_tensor(ctx, &[2], 64);
    let dyn_flat = tensor(ctx, &[-1]);
    let dyn_matrix = tensor(ctx, &[-1, 4]);
    let index = shaped::index(ctx);
    let v = args(ctx, vec![flat, matrix, shape, dyn_flat, index]);
    valid!(ctx, ReshapeOp::new(ctx, v[0], v[2], matrix));
    invalid!(ctx, ReshapeOp::new(ctx, v[0], v[2], wrong));
    let expand = ExpandShapeOp::new(
        ctx,
        v[0],
        matrix,
        vec![vec![0, 1]],
        vec![MixedIndex::Static(2), MixedIndex::Static(4)],
    );
    let text = good(ctx, &expand);
    #[cfg(feature = "mlir-interop")]
    assert!(
        text.contains("reassociation = [[0 : i64, 1 : i64]]"),
        "{text}"
    );
    #[cfg(not(feature = "mlir-interop"))]
    let _ = text;
    valid!(ctx, CollapseShapeOp::new(ctx, v[1], flat, vec![vec![0, 1]]));
    invalid!(ctx, CollapseShapeOp::new(ctx, v[1], flat, vec![vec![1, 0]]));
    valid!(
        ctx,
        ExpandShapeOp::new(
            ctx,
            v[3],
            dyn_matrix,
            vec![vec![0, 1]],
            vec![MixedIndex::Dynamic(v[4]), MixedIndex::Static(4)]
        )
    );
    invalid!(
        ctx,
        ExpandShapeOp::new(
            ctx,
            v[0],
            dyn_matrix,
            vec![vec![0, 1]],
            vec![MixedIndex::Dynamic(v[4]), MixedIndex::Static(4)]
        )
    );
    let scalar = tensor(ctx, &[]);
    let ones = tensor(ctx, &[1, 1]);
    let v = args(ctx, vec![scalar, ones]);
    valid!(
        ctx,
        ExpandShapeOp::new(ctx, v[0], ones, vec![], vec![MixedIndex::Static(1); 2])
    );
    valid!(ctx, CollapseShapeOp::new(ctx, v[1], scalar, vec![]));
}

#[test]
fn gather_and_scatter_enforce_tuple_layout_and_uniqueness() {
    let ctx = &mut Context::new();
    let source = tensor(ctx, &[4, 8]);
    let indices = integer_tensor(ctx, &[3, 1], 32);
    let dynamic_indices = integer_tensor(ctx, &[3, -1], 32);
    let full = tensor(ctx, &[3, 1, 8]);
    let reduced = tensor(ctx, &[3, 8]);
    let v = args(ctx, vec![source, indices, full, reduced, dynamic_indices]);
    valid!(ctx, GatherOp::new(ctx, v[0], v[1], vec![0], false, full));
    valid!(ctx, GatherOp::new(ctx, v[0], v[1], vec![0], true, reduced));
    valid!(ctx, ScatterOp::new(ctx, v[3], v[0], v[1], vec![0], true));
    invalid!(ctx, GatherOp::new(ctx, v[0], v[4], vec![0], false, full));
    invalid!(ctx, GatherOp::new(ctx, v[0], v[1], vec![], false, full));
    invalid!(ctx, GatherOp::new(ctx, v[0], v[1], vec![0, 0], false, full));
    invalid!(ctx, ScatterOp::new(ctx, v[3], v[0], v[1], vec![0], false));
    let scatter = ScatterOp::new(ctx, v[2], v[0], v[1], vec![0], true);
    scatter
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(UNIQUE.clone(), I64ArrayAttr::new(vec![]));
    bad(ctx, &scatter);
    let gather = GatherOp::new(ctx, v[0], v[1], vec![0], false, full);
    gather
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(UNIQUE.clone(), I64ArrayAttr::new(vec![]));
    bad(ctx, &gather);
}

#[test]
fn generated_and_padded_regions_yield_elements() {
    let ctx = &mut Context::new();
    let source = tensor(ctx, &[4]);
    let result = tensor(ctx, &[7]);
    let dynamic = tensor(ctx, &[-1]);
    let f32 = FP32Type::get(ctx).into();
    let index = shaped::index(ctx);
    let v = args(ctx, vec![source, f32, index]);
    let scalar = v[1];
    let generated = GenerateOp::build(
        ctx,
        dynamic,
        vec![v[2]],
        Box::new(move |_, _, args| {
            assert_eq!(args.len(), 1);
            Ok(scalar)
        }),
    )
    .unwrap();
    good(ctx, &generated);
    let padded = PadOp::build(
        ctx,
        v[0],
        result,
        vec![MixedIndex::Static(1)],
        vec![MixedIndex::Static(2)],
        true,
        Box::new(move |_, _, _| Ok(scalar)),
    )
    .unwrap();
    let text = good(ctx, &padded);
    #[cfg(feature = "mlir-interop")]
    {
        assert!(text.contains("\"tensor.yield\""), "{text}");
        assert!(text.contains("nofold = unit"), "{text}");
    }
    #[cfg(not(feature = "mlir-interop"))]
    let _ = text;
    let invalid = PadOp::build(
        ctx,
        v[0],
        dynamic,
        vec![MixedIndex::Static(1)],
        vec![MixedIndex::Static(2)],
        false,
        Box::new(move |_, _, _| Ok(scalar)),
    );
    assert!(invalid.is_err());
    let invalid = GenerateOp::build(ctx, source, vec![], Box::new(move |_, _, args| Ok(args[0])));
    assert!(invalid.is_err());
    invalid!(ctx, YieldOp::new(ctx, scalar));
    // Wrong optional flag storage must fail verification, not just export.
    padded
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(NOFOLD.clone(), I64ArrayAttr::new(vec![]));
    bad(ctx, &padded);
    padded
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(NOFOLD.clone(), UnitAttr::new());
    good(ctx, &padded);
}

#[test]
fn zero_sized_slice_still_checks_offset_like_upstream() {
    let ctx = &mut Context::new();
    let src = tensor(ctx, &[4]);
    let zero = tensor(ctx, &[0]);
    let v = args(ctx, vec![src]);
    valid!(
        ctx,
        ExtractSliceOp::new(ctx, v[0], zero, static_slice(&[3], &[0], &[1]))
    );
    invalid!(
        ctx,
        ExtractSliceOp::new(ctx, v[0], zero, static_slice(&[4], &[0], &[1]))
    );
}

#[test]
fn malformed_segments_do_not_panic() {
    let ctx = &mut Context::new();
    let ty = tensor(ctx, &[4]);
    let v = args(ctx, vec![ty]);
    let op = ExtractSliceOp::new(ctx, v[0], ty, static_slice(&[0], &[4], &[1]));
    op.get_operation().deref_mut(ctx).attributes.set(
        SEGMENTS.clone(),
        OperandSegmentSizesAttr(vec![u32::MAX, 1, 0, 0]),
    );
    bad(ctx, &op);
}
