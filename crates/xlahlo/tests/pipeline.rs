mod support;
use support::{computation as named, context, module, ret};
use xlahlo::{
    attributes::*,
    inference,
    ops::{
        ComputationOp, HloOpInterface, ReturnOp,
        control::{CallOp, ConstantOp, FusionOp, GetTupleElementOp, ParameterOp, TupleOp, WhileOp},
        elementwise::{
            AbsOp, AddOp, AndOp, CeilOp, CompareOp, ConvertOp, CosOp, DivideOp, ExpOp, Expm1Op,
            FloorOp, Log1pOp, LogOp, MaximumOp, MinimumOp, MultiplyOp, NegateOp, NotOp, OrOp,
            PowerOp, RemainderOp, RoundNearestAfzOp, RsqrtOp, SelectOp, ShiftLeftOp,
            ShiftRightArithmeticOp, ShiftRightLogicalOp, SignOp, SinOp, SqrtOp, SubtractOp, TanhOp,
            XorOp,
        },
        movement::{BitcastOp, IotaOp, ReshapeOp},
        numerical::ReduceOp,
    },
    pliron::{
        basic_block::BasicBlock,
        builtin::types::FunctionType,
        common_traits::Verify,
        context::{Context, Ptr},
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        location::{Located, Location},
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
        r#type::TypeHandle,
    },
    types::{ElementType::*, TensorType, TupleType},
};

fn tensor(ctx: &Context, element: xlahlo::types::ElementType, dims: &[i64]) -> TypeHandle {
    TensorType::get(ctx, element, dims.to_vec()).unwrap()
}

fn constant(ctx: &mut Context, shape: TypeHandle, bits: &[u128]) -> ConstantOp {
    let literal = xlahlo::literal::Literal::from_bits(ctx, shape, bits).unwrap();
    ConstantOp::new(ctx, shape, vec![], literal)
}

fn computation(
    ctx: &mut Context,
    args: Vec<TypeHandle>,
    result: TypeHandle,
) -> (ComputationOp, Ptr<BasicBlock>) {
    let signature = FunctionType::get(ctx, args, vec![result]).into();
    let op = ComputationOp::new(ctx, "main".try_into().unwrap(), signature);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(op.body(ctx), ctx);
    (op, block)
}

#[test]
fn static_shapes_and_layouts() {
    let ctx = context();
    assert!(TensorType::get(&ctx, F32, vec![-1]).is_err());
    assert!(TensorType::get(&ctx, F32, vec![i64::MAX, i64::MAX]).is_err());
    assert!(TensorType::with_layout(&ctx, F32, vec![2, 3], vec![0, 0]).is_err());
    let matrix = tensor(&ctx, F32, &[2, 3]);
    assert_eq!(
        inference::movement::infer_reshape(&ctx, matrix, &[6]).unwrap(),
        tensor(&ctx, F32, &[6])
    );
    assert!(inference::movement::infer_reshape(&ctx, matrix, &[5]).is_err());
    assert_eq!(
        inference::movement::infer_transpose(&ctx, matrix, &Axes(vec![1, 0])).unwrap(),
        TensorType::with_layout(&ctx, F32, vec![3, 2], vec![0, 1]).unwrap()
    );
    assert!(inference::movement::infer_transpose(&ctx, matrix, &Axes(vec![1, 1])).is_err());
    assert!(
        inference::movement::infer_broadcast(
            &ctx,
            matrix,
            tensor(&ctx, F32, &[4, 2, 3]),
            &Axes(vec![1, 2])
        )
        .is_ok()
    );
    assert!(
        inference::movement::infer_broadcast(
            &ctx,
            matrix,
            tensor(&ctx, F32, &[4, 2, 3]),
            &Axes(vec![2, 1])
        )
        .is_err()
    );
}

#[test]
fn dot_slice_and_concat_inference() {
    let ctx = context();
    let dims = DotDimensionNumbers {
        lhs_batch_dimensions: Axes(vec![0]),
        rhs_batch_dimensions: Axes(vec![0]),
        lhs_contracting_dimensions: Axes(vec![2]),
        rhs_contracting_dimensions: Axes(vec![1]),
    };
    let lhs = tensor(&ctx, F32, &[2, 3, 4]);
    let rhs = tensor(&ctx, F32, &[2, 4, 5]);
    assert!(xlahlo::shape::Shape::equal_including_layout(
        &ctx,
        inference::numerical::infer_dot(&ctx, lhs, rhs, &dims, Some(F32)).unwrap(),
        tensor(&ctx, F32, &[2, 3, 5])
    ));
    assert!(
        inference::numerical::infer_dot(&ctx, lhs, tensor(&ctx, F32, &[2, 6, 5]), &dims, Some(F32))
            .is_err()
    );
    let mut overlap = dims.clone();
    overlap.lhs_contracting_dimensions = Axes(vec![0]);
    assert!(inference::numerical::infer_dot(&ctx, lhs, rhs, &overlap, Some(F32)).is_err());
    let input = tensor(&ctx, F32, &[10]);
    let geometry = Slice {
        starts: Axes(vec![1]),
        limits: Axes(vec![10]),
        strides: Axes(vec![3]),
    };
    assert!(xlahlo::shape::Shape::equal_including_layout(
        &ctx,
        inference::movement::infer_slice(&ctx, input, &geometry).unwrap(),
        tensor(&ctx, F32, &[3])
    ));
    assert!(
        inference::movement::infer_slice(
            &ctx,
            input,
            &Slice {
                strides: Axes(vec![0]),
                ..geometry
            }
        )
        .is_err()
    );
    assert!(xlahlo::shape::Shape::equal_including_layout(
        &ctx,
        inference::movement::infer_concatenate(&ctx, &[input, input], 0).unwrap(),
        tensor(&ctx, F32, &[20])
    ));
    assert!(inference::movement::infer_concatenate(&ctx, &[], 0).is_err());
}

#[test]
fn gather_and_scatter_inference() {
    let ctx = context();
    let input = tensor(&ctx, F32, &[8, 4]);
    let indices = tensor(&ctx, S32, &[3, 1]);
    let dims = GatherDimensionNumbers {
        offset_dims: Axes(vec![1]),
        collapsed_slice_dims: Axes(vec![0]),
        operand_batching_dims: Axes(vec![]),
        start_indices_batching_dims: Axes(vec![]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: 1,
    };
    assert!(xlahlo::shape::Shape::equal_including_layout(
        &ctx,
        inference::movement::infer_gather(&ctx, input, indices, &dims, &Axes(vec![1, 4])).unwrap(),
        tensor(&ctx, F32, &[3, 4])
    ));
    assert!(
        inference::movement::infer_gather(&ctx, input, indices, &dims, &Axes(vec![2, 4])).is_err()
    );
    assert!(
        inference::movement::infer_gather(
            &ctx,
            input,
            tensor(&ctx, F32, &[3, 1]),
            &dims,
            &Axes(vec![1, 4])
        )
        .is_err()
    );
    let scatter = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![0]),
        input_batching_dims: Axes(vec![]),
        scatter_indices_batching_dims: Axes(vec![]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 1,
    };
    assert_eq!(
        inference::movement::infer_scatter(
            &ctx,
            input,
            indices,
            tensor(&ctx, F32, &[3, 4]),
            &scatter
        )
        .unwrap(),
        input
    );
    assert!(
        inference::movement::infer_scatter(
            &ctx,
            input,
            indices,
            tensor(&ctx, F32, &[3, 5]),
            &scatter
        )
        .is_err()
    );
}

#[test]
fn graph_edits_and_dominance() {
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[4]);
    let (comp, block) = computation(&mut ctx, vec![ty], ty);
    let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
    parameter.get_operation().insert_at_back(block, &ctx);
    let p = parameter.result(&ctx);
    let add = AddOp::new(&mut ctx, ty, vec![p, p]);
    add.get_operation().insert_at_back(block, &ctx);
    let sum = add.result(&ctx);
    let ret = ReturnOp::new(&mut ctx, sum);
    ret.get_operation().insert_at_back(block, &ctx);
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    pliron::op::verify_op(&comp, &ctx).unwrap();
    add.result(&ctx).replace_all_uses_with(&ctx, &p);
    Operation::erase(add.get_operation(), &mut ctx);
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    let late = NegateOp::new(&mut ctx, ty, vec![p]);
    late.get_operation().insert_at_back(block, &ctx);
    Operation::replace_operand(ret.get_operation(), &ctx, 0, late.result(&ctx));
    assert!(xlahlo::verification::verify(&ctx, comp.get_operation()).is_err());
}

#[test]
fn constant_compare_and_tuple_rules() {
    let mut ctx = context();
    let ty = tensor(&ctx, S8, &[]);
    let c = constant(&mut ctx, ty, &[255]);
    c.verify(&ctx).unwrap();
    assert!(xlahlo::literal::Literal::from_bits(&ctx, ty, &[256]).is_err());
    let bad = ConstantOp::new(
        &mut ctx,
        ty,
        vec![],
        xlahlo::literal::Literal {
            shape: ty,
            storage: vec![0, 1],
            tuple_literals: vec![],
            dynamic_sizes: vec![],
        },
    );
    assert!(bad.verify(&ctx).is_err());
    let v = c.result(&ctx);
    let pred = tensor(&ctx, Pred, &[]);
    let cmp = CompareOp::new(
        &mut ctx,
        pred,
        vec![v, v],
        Direction::Lt,
        ComparisonOrder::Total,
    );
    cmp.verify(&ctx).unwrap();
    let bad = CompareOp::new(
        &mut ctx,
        pred,
        vec![v, v],
        Direction::Lt,
        ComparisonOrder::Partial,
    );
    assert!(bad.verify(&ctx).is_err());
    let tuple_ty = TupleType::get(&ctx, vec![ty, pred]).unwrap();
    let comparison = cmp.result(&ctx);
    let tuple = TupleOp::new(&mut ctx, tuple_ty, vec![v, comparison]);
    tuple.verify(&ctx).unwrap();
    let value = tuple.result(&ctx);
    let gte = GetTupleElementOp::new(&mut ctx, pred, vec![value], Index(1));
    gte.verify(&ctx).unwrap();
}

#[test]
fn release_pipeline_budget() {
    if cfg!(debug_assertions) {
        return;
    }
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[16]);
    let start = std::time::Instant::now();
    let (comp, block) = computation(&mut ctx, vec![ty], ty);
    let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
    parameter.get_operation().insert_at_back(block, &ctx);
    let mut value = parameter.result(&ctx);
    let mut erased = Vec::with_capacity(1000);
    for i in 1..10_000 {
        let op = NegateOp::new(&mut ctx, ty, vec![value]);
        op.get_operation().insert_at_back(block, &ctx);
        if i % 10 == 1 {
            erased.push((op.get_operation(), op.result(&ctx), value));
        }
        value = op.result(&ctx);
    }
    let ret = ReturnOp::new(&mut ctx, value);
    ret.get_operation().insert_at_back(block, &ctx);
    let built = start.elapsed();
    for (op, from, to) in erased {
        from.replace_all_uses_with(&ctx, &to);
        Operation::erase(op, &mut ctx);
    }
    let edited = start.elapsed();
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    let total = start.elapsed();
    eprintln!(
        "10k build={built:?}; edits={:?}; verify={:?}; total={total:?}",
        edited - built,
        total - edited
    );
    assert!(
        total <= std::time::Duration::from_millis(15),
        "pipeline budget exceeded: {total:?}"
    );
}

#[test]
fn elementwise_smoke_checks_arity_domains_and_geometry() {
    let mut ctx = context();
    let real = tensor(&ctx, F32, &[2]);
    let integer = tensor(&ctx, S32, &[2]);
    let pred = tensor(&ctx, Pred, &[2]);
    let r = ParameterOp::new(&mut ctx, real, vec![], Index(0)).result(&ctx);
    let n = ParameterOp::new(&mut ctx, integer, vec![], Index(1)).result(&ctx);
    let p = ParameterOp::new(&mut ctx, pred, vec![], Index(2)).result(&ctx);
    let mut ops = Vec::new();
    macro_rules! unary {
        ($($ty:ty),*) => { $(ops.push(<$ty>::new(&mut ctx,real,vec![r]).get_operation());)* };
    }
    unary!(
        NegateOp,
        AbsOp,
        ExpOp,
        Expm1Op,
        LogOp,
        Log1pOp,
        SqrtOp,
        RsqrtOp,
        SinOp,
        CosOp,
        TanhOp,
        CeilOp,
        FloorOp,
        RoundNearestAfzOp,
        SignOp
    );
    macro_rules! binary {
        ($($ty:ty),*) => { $(ops.push(<$ty>::new(&mut ctx,real,vec![r,r]).get_operation());)* };
    }
    binary!(
        AddOp,
        SubtractOp,
        MultiplyOp,
        DivideOp,
        RemainderOp,
        MaximumOp,
        MinimumOp,
        PowerOp
    );
    macro_rules! integral {
        ($($ty:ty),*) => { $(ops.push(<$ty>::new(&mut ctx,integer,vec![n,n]).get_operation());)* };
    }
    integral!(
        AndOp,
        OrOp,
        XorOp,
        ShiftLeftOp,
        ShiftRightArithmeticOp,
        ShiftRightLogicalOp
    );
    ops.push(NotOp::new(&mut ctx, pred, vec![p]).get_operation());
    for op in ops {
        Operation::get_op_dyn(op, &ctx).verify(&ctx).unwrap();
        Operation::pop_operand(op, &ctx);
        assert!(Operation::get_op_dyn(op, &ctx).verify(&ctx).is_err());
    }
    let add = AddOp::new(&mut ctx, real, vec![r, n]);
    assert!(add.verify(&ctx).is_err());
    let exp = ExpOp::new(&mut ctx, integer, vec![n]);
    assert!(exp.verify(&ctx).is_err());
    let logical = NotOp::new(&mut ctx, real, vec![r]);
    assert!(logical.verify(&ctx).is_err());
    let unsigned = tensor(&ctx, U32, &[2]);
    let u = ParameterOp::new(&mut ctx, unsigned, vec![], Index(3)).result(&ctx);
    assert!(
        AbsOp::new(&mut ctx, unsigned, vec![u])
            .verify(&ctx)
            .is_err()
    );
    assert!(
        SignOp::new(&mut ctx, unsigned, vec![u])
            .verify(&ctx)
            .is_err()
    );
    let complex = tensor(&ctx, C64, &[2]);
    let c = ParameterOp::new(&mut ctx, complex, vec![], Index(4)).result(&ctx);
    AbsOp::new(&mut ctx, real, vec![c]).verify(&ctx).unwrap();
    let wrong = tensor(&ctx, F32, &[3]);
    assert!(
        NegateOp::new(&mut ctx, wrong, vec![r])
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn reshape_bitcast_convert_and_select_are_distinct() {
    let mut ctx = context();
    let source = TensorType::with_layout(&ctx, F32, vec![2, 3], vec![1, 0]).unwrap();
    let p = ParameterOp::new(&mut ctx, source, vec![], Index(0)).result(&ctx);
    let target = tensor(&ctx, F32, &[6]);
    ReshapeOp::new(&mut ctx, target, vec![p])
        .verify(&ctx)
        .unwrap();
    let unspecified = TensorType::with_layout(&ctx, F32, vec![6], vec![]).unwrap();
    let bitcast = BitcastOp::new(&mut ctx, unspecified, vec![p]);
    bitcast.verify(&ctx).unwrap();
    bitcast
        .verify_with_options(
            &ctx,
            xlahlo::verification::VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        )
        .unwrap();
    let different_size = tensor(&ctx, U8, &[99]);
    let bitcast = BitcastOp::new(&mut ctx, different_size, vec![p]);
    bitcast.verify(&ctx).unwrap();
    assert!(
        bitcast
            .verify_with_options(
                &ctx,
                xlahlo::verification::VerifyOptions {
                    layout_sensitive: true,
                    ..Default::default()
                }
            )
            .is_err()
    );
    let bytes = TensorType::with_layout(&ctx, U8, vec![24], vec![0]).unwrap();
    BitcastOp::new(&mut ctx, bytes, vec![p])
        .verify(&ctx)
        .unwrap();
    assert!(
        ReshapeOp::new(&mut ctx, bytes, vec![p])
            .verify(&ctx)
            .is_err()
    );
    let converted = tensor(&ctx, S32, &[2, 3]);
    ConvertOp::new(&mut ctx, converted, vec![p])
        .verify(&ctx)
        .unwrap();
    let scalar_pred = tensor(&ctx, Pred, &[]);
    let pred = constant(&mut ctx, scalar_pred, &[1]).result(&ctx);
    SelectOp::new(&mut ctx, source, vec![pred, p, p])
        .verify(&ctx)
        .unwrap();
    assert!(
        SelectOp::new(&mut ctx, source, vec![p, p, p])
            .verify(&ctx)
            .is_err()
    );
    let iota = IotaOp::new(&mut ctx, source, vec![], Index(1));
    iota.verify(&ctx).unwrap();
    assert!(
        IotaOp::new(&mut ctx, source, vec![], Index(2))
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn call_while_reducer_and_fusion_signatures() {
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[4]);
    let scalar = tensor(&ctx, F32, &[]);
    let pred = tensor(&ctx, Pred, &[]);
    let (module, container) = module(&mut ctx);
    let (_, block, p) = named(&mut ctx, container, "condition", vec![ty], pred);
    let false_ = constant(&mut ctx, pred, &[0]);
    false_.get_operation().insert_at_back(block, &ctx);
    let value = false_.result(&ctx);
    ret(&mut ctx, block, value);
    assert_eq!(p.len(), 1);
    let (_, block, p) = named(&mut ctx, container, "body", vec![ty], ty);
    ret(&mut ctx, block, p[0]);
    let (_, block, p) = named(&mut ctx, container, "sum", vec![scalar, scalar], scalar);
    let sum = AddOp::new(&mut ctx, scalar, p);
    sum.get_operation().insert_at_back(block, &ctx);
    let value = sum.result(&ctx);
    ret(&mut ctx, block, value);
    let (_, entry, p) = named(&mut ctx, container, "main", vec![ty], ty);
    let call = CallOp::new(
        &mut ctx,
        ty,
        p.clone(),
        Callees(vec!["body".try_into().unwrap()]),
    );
    call.get_operation().insert_at_back(entry, &ctx);
    let value = call.result(&ctx);
    let loop_ = WhileOp::new(
        &mut ctx,
        ty,
        vec![value],
        Callees(vec![
            "body".try_into().unwrap(),
            "condition".try_into().unwrap(),
        ]),
    );
    loop_.get_operation().insert_at_back(entry, &ctx);
    let value = loop_.result(&ctx);
    let fusion = FusionOp::new(&mut ctx, ty, vec![value], FusionKind::Loop);
    fusion.get_operation().insert_at_back(entry, &ctx);
    let fb = BasicBlock::new(&mut ctx, None, vec![ty]);
    fb.insert_at_back(fusion.body(&ctx), &ctx);
    let arg = fb.deref(&ctx).get_argument(0);
    let neg = NegateOp::new(&mut ctx, ty, vec![arg]);
    neg.get_operation().insert_at_back(fb, &ctx);
    let value = neg.result(&ctx);
    ret(&mut ctx, fb, value);
    let zero = constant(&mut ctx, scalar, &[0]);
    zero.get_operation().insert_at_back(entry, &ctx);
    let initial = zero.result(&ctx);
    let reduction = ReduceOp::new(
        &mut ctx,
        scalar,
        vec![p[0], initial],
        Axes(vec![0]),
        Callees(vec!["sum".try_into().unwrap()]),
    );
    reduction.get_operation().insert_at_back(entry, &ctx);
    let value = fusion.result(&ctx);
    ret(&mut ctx, entry, value);
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    pliron::op::verify_op(&module, &ctx).unwrap();
    Operation::replace_operand(neg.get_operation(), &ctx, 0, p[0]);
    assert!(
        xlahlo::verification::verify(&ctx, module.get_operation()).is_err(),
        "fusion capture must fail"
    );
    Operation::replace_operand(neg.get_operation(), &ctx, 0, arg);
    call.get_operation().deref_mut(&ctx).attributes.set(
        pliron::identifier::Identifier::new("hlo_callees"),
        Callees(vec!["missing".try_into().unwrap()]),
    );
    assert!(
        xlahlo::verification::verify(&ctx, module.get_operation()).is_err(),
        "unresolved callee must fail"
    );
}

#[test]
fn missing_parameters_duplicates_and_recursive_calls_fail() {
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[]);
    let (module, container) = module(&mut ctx);
    let (_, block, p) = named(&mut ctx, container, "main", vec![ty], ty);
    let call = CallOp::new(&mut ctx, ty, p, Callees(vec!["main".try_into().unwrap()]));
    call.get_operation().insert_at_back(block, &ctx);
    let value = call.result(&ctx);
    ret(&mut ctx, block, value);
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    let (comp, block) = computation(&mut ctx, vec![ty], ty);
    let literal = constant(&mut ctx, ty, &[0]);
    literal.get_operation().insert_at_back(block, &ctx);
    let value = literal.result(&ctx);
    ret(&mut ctx, block, value);
    assert!(
        xlahlo::verification::verify(&ctx, comp.get_operation()).is_err(),
        "unused parameters still require declarations"
    );
}

#[test]
fn generic_assembly_preserves_typed_attributes_and_nested_regions() {
    fn snapshot(ctx: &Context, root: Ptr<Operation>) -> Vec<String> {
        let mut rows = Vec::new();
        let mut pending = vec![root];
        while let Some(op) = pending.pop() {
            let data = op.deref(ctx);
            let types: Vec<_> = data
                .result_types()
                .map(|ty| ty.disp(ctx).to_string())
                .collect();
            let attributes: Vec<_> = data
                .attributes
                .0
                .iter()
                .filter(|(name, _)| {
                    **name != pliron::identifier::Identifier::new("builtin_given_names")
                })
                .map(|(name, value)| format!("{name}: {}", value.disp(ctx)))
                .collect();
            rows.push(format!(
                "{} {:?} {:?}",
                Operation::get_op_dyn(op, ctx).get_opid().disp(ctx),
                types,
                attributes,
            ));
            for region in data.regions() {
                for block in region.deref(ctx).iter(ctx) {
                    pending.extend(block.deref(ctx).iter(ctx));
                }
            }
        }
        rows
    }
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[2, 3]);
    let (comp, block) = computation(&mut ctx, vec![ty], ty);
    let p = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
    p.get_operation().insert_at_back(block, &ctx);
    let value = p.result(&ctx);
    let fusion = FusionOp::new(&mut ctx, ty, vec![value], FusionKind::Loop);
    fusion.get_operation().insert_at_back(block, &ctx);
    let body = BasicBlock::new(&mut ctx, None, vec![ty]);
    body.insert_at_back(fusion.body(&ctx), &ctx);
    let arg = body.deref(&ctx).get_argument(0);
    let zero = constant(&mut ctx, ty, &[0; 6]);
    zero.get_operation().insert_at_back(body, &ctx);
    let zero = zero.result(&ctx);
    let add = AddOp::new(&mut ctx, ty, vec![arg, zero]);
    add.get_operation().insert_at_back(body, &ctx);
    let value = add.result(&ctx);
    ret(&mut ctx, body, value);
    let value = fusion.result(&ctx);
    ret(&mut ctx, block, value);
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    let assembly = comp.disp(&ctx).to_string();
    let mut parsed = context();
    let root = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed,
        &assembly,
    )
    .unwrap();
    xlahlo::verification::verify(&parsed, root).unwrap();
    pliron::op::verify_op(Operation::get_op_dyn(root, &parsed).as_ref(), &parsed).unwrap();
    assert_eq!(
        snapshot(&parsed, root),
        snapshot(&ctx, comp.get_operation())
    );
}

#[test]
fn malformed_callable_metadata_returns_error_without_panicking() {
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[]);
    let (comp, block) = computation(&mut ctx, vec![ty], ty);
    let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
    parameter.get_operation().insert_at_back(block, &ctx);
    let value = parameter.result(&ctx);
    ret(&mut ctx, block, value);
    comp.get_operation().deref_mut(&ctx).attributes.set(
        pliron::identifier::Identifier::new("hlo_callees"),
        Callees(vec![]),
    );
    assert!(xlahlo::verification::verify(&ctx, comp.get_operation()).is_err());
}

#[test]
fn verification_diagnostics_preserve_typed_causes_and_allow_repair() {
    let mut ctx = context();
    let ty = tensor(&ctx, F32, &[4]);
    let (comp, block) = computation(&mut ctx, vec![ty], ty);
    let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
    parameter.get_operation().insert_at_back(block, &ctx);
    let input = parameter.result(&ctx);
    let add = AddOp::new(&mut ctx, ty, vec![input]);
    add.get_operation().insert_at_back(block, &ctx);
    let location = Location::Named {
        name: "malformed-add".to_owned(),
        child_loc: Box::new(Location::Unknown),
    };
    add.get_operation()
        .deref_mut(&ctx)
        .set_loc(location.clone());
    let result = add.result(&ctx);
    ret(&mut ctx, block, result);

    let error = xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap_err();
    assert_eq!(error.loc, location);
    assert!(matches!(
        error.kind,
        xlahlo::pliron::result::ErrorKind::VerificationFailed
    ));
    let diagnostic = error.err.downcast_ref::<xlahlo::Diagnostic>().unwrap();
    let xlahlo::Diagnostic::Operation { operation, source } = diagnostic else {
        panic!("expected operation context, got {diagnostic:?}");
    };
    assert!(operation.contains("add"));
    assert!(source.downcast_ref::<xlahlo::Diagnostic>().is_some());
    assert!(std::error::Error::source(diagnostic).is_some());
    assert!(error.to_string().contains("arity"));

    add.result(&ctx).replace_all_uses_with(&ctx, &input);
    Operation::erase(add.get_operation(), &mut ctx);
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
}

#[test]
fn opcode_diagnostics_expose_the_invalid_spelling() {
    let error = "not-an-hlo-op"
        .parse::<xlahlo::opcode::Opcode>()
        .unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<xlahlo::Diagnostic>(),
        Some(xlahlo::Diagnostic::UnknownOpcode { name }) if name == "not-an-hlo-op"
    ));
    assert!("add".parse::<xlahlo::opcode::Opcode>().is_ok());
}

#[test]
fn arithmetic_verification_reuse_checks_each_structure_type_and_attribute() {
    for fault in 0..3 {
        let mut ctx = context();
        let ty = tensor(&ctx, F32, &[4]);
        let (comp, block) = computation(&mut ctx, vec![ty], ty);
        let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
        parameter.get_operation().insert_at_back(block, &ctx);
        let input = parameter.result(&ctx);
        let bad = match fault {
            0 => AddOp::new(&mut ctx, ty, vec![input]),
            1 => {
                let wrong = tensor(&ctx, F32, &[5]);
                AddOp::new(&mut ctx, wrong, vec![input, input])
            }
            _ => {
                let op = AddOp::new(&mut ctx, ty, vec![input, input]);
                op.get_operation().deref_mut(&ctx).attributes.set(
                    xlahlo::pliron::identifier::Identifier::new("hlo_result_accuracy"),
                    Index(0),
                );
                op
            }
        };
        bad.get_operation().insert_at_back(block, &ctx);
        let good = AddOp::new(&mut ctx, ty, vec![input, input]);
        good.get_operation().insert_at_back(block, &ctx);
        let result = good.result(&ctx);
        ret(&mut ctx, block, result);
        assert!(xlahlo::verification::verify(&ctx, comp.get_operation()).is_err());
        let result = bad.result(&ctx);
        result.replace_all_uses_with(&ctx, &input);
        Operation::erase(bad.get_operation(), &mut ctx);
        xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    }
}
