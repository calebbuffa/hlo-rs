use clienthlo::{
    attributes::{
        ComparisonDirection, Precision, RaggedDotDimensionNumbersAttr, RaggedDotDimensions,
        read_precision_config,
    },
    ops::{
        BroadcastCompareOp, BroadcastCompareOptions, ConstantOp as ChloConstantOp, ErfOp,
        RaggedDotOp, RaggedDotOptions, ScanOp, TopKOp, TopKOptions,
    },
    pliron::{
        builtin::ops::ModuleOp, common_traits::Verify, context::Context, identifier::Identifier,
        irfmt::parsers::spaced, linked_list::ContainsLinkedList, op::Op, operation::Operation,
        parsable::parse_from_str, printable::Printable, r#type::Typed,
    },
};
use stablehlo::{
    ops::ReturnOp,
    types::{ElementType, tensor_parts},
};

#[test]
fn mixed_stablehlo_chlo_module_roundtrips_native_assembly() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, Identifier::new("mixed"));
    let function_block = module
        .get_operation()
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_entry_block()
        .unwrap();

    let lhs_constant =
        ChloConstantOp::new_inferred(&mut ctx, vec![2, 3, 4], ElementType::F32, vec![0], true)
            .unwrap();
    let lhs = lhs_constant.get_operation().deref(&ctx).get_result(0);
    lhs_constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let rhs_constant =
        ChloConstantOp::new_inferred(&mut ctx, vec![2, 4, 5], ElementType::F32, vec![0], true)
            .unwrap();
    let rhs = rhs_constant.get_operation().deref(&ctx).get_result(0);
    rhs_constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let groups_constant =
        ChloConstantOp::new_inferred(&mut ctx, vec![2, 2], ElementType::I32, vec![0], true)
            .unwrap();
    let groups = groups_constant.get_operation().deref(&ctx).get_result(0);
    groups_constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let scan_input_constant =
        ChloConstantOp::new_inferred(&mut ctx, vec![2, 3], ElementType::F32, vec![0], true)
            .unwrap();
    let scan_input = scan_input_constant
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    scan_input_constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let init_constant =
        ChloConstantOp::new_inferred(&mut ctx, vec![], ElementType::F32, vec![0], true).unwrap();
    let init = init_constant.get_operation().deref(&ctx).get_result(0);
    init_constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let compare_rhs_constant =
        ChloConstantOp::new_inferred(&mut ctx, vec![3], ElementType::F32, vec![0], true).unwrap();
    let compare_rhs = compare_rhs_constant
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    compare_rhs_constant
        .get_operation()
        .insert_at_back(function_block, &ctx);

    let constant = ChloConstantOp::new_inferred(
        &mut ctx,
        vec![2],
        ElementType::F32,
        vec![0x3f80_0000, 0x4000_0000],
        false,
    )
    .unwrap();
    constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let unary = ErfOp::new_inferred(&mut ctx, lhs).unwrap();
    unary.get_operation().insert_at_back(function_block, &ctx);
    let compare = BroadcastCompareOp::new_inferred(
        &mut ctx,
        scan_input,
        compare_rhs,
        BroadcastCompareOptions {
            broadcast_dimensions: None,
            direction: ComparisonDirection::Gt,
            comparison_kind: None,
        },
    )
    .unwrap();
    compare.get_operation().insert_at_back(function_block, &ctx);
    let top_k = TopKOp::new_inferred(&mut ctx, lhs, TopKOptions::new(2, true)).unwrap();
    top_k.get_operation().insert_at_back(function_block, &ctx);
    let dimensions = RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
        lhs_batching_dimensions: vec![0],
        rhs_batching_dimensions: vec![0],
        lhs_contracting_dimensions: vec![2],
        rhs_contracting_dimensions: vec![1],
        lhs_ragged_dimensions: vec![2],
        rhs_group_dimensions: vec![],
    });
    let ragged_dot = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs,
        rhs,
        groups,
        RaggedDotOptions {
            dimension_numbers: dimensions,
            precision_config: Some(vec![Precision::High, Precision::Highest]),
        },
    )
    .unwrap();
    ragged_dot
        .get_operation()
        .insert_at_back(function_block, &ctx);

    let scan = ScanOp::new_inferred(
        &mut ctx,
        vec![scan_input],
        vec![init],
        0,
        None,
        false,
        None,
        |ctx, block| {
            let returned = block.deref(ctx).arguments().collect();
            ReturnOp::new_many(ctx, returned)
                .get_operation()
                .insert_at_back(block, ctx);
            Ok(())
        },
    )
    .unwrap();
    scan.get_operation().insert_at_back(function_block, &ctx);
    module.verify(&ctx).unwrap();
    constant.verify(&ctx).unwrap();
    unary.verify(&ctx).unwrap();
    compare.verify(&ctx).unwrap();
    top_k.verify(&ctx).unwrap();
    ragged_dot.verify(&ctx).unwrap();
    scan.verify(&ctx).unwrap();
    assert_eq!(top_k.get_operation().deref(&ctx).get_num_results(), 2);
    assert_eq!(
        ragged_dot.precision_config(&ctx),
        Some(vec![Precision::High, Precision::Highest])
    );
    assert_eq!(
        read_precision_config(&ragged_dot.precision_config_attr(&ctx).unwrap()),
        vec![Precision::High, Precision::Highest]
    );
    assert_eq!(scan.scan_dim_size(&ctx), None);
    assert_eq!(scan.is_associative(&ctx), None);
    assert_eq!(scan.is_reverse(&ctx), Some(false));
    assert_eq!(scan.inputs(&ctx).unwrap().len(), 1);
    assert_eq!(scan.inits(&ctx).unwrap().len(), 1);
    assert!(
        constant
            .value(&ctx)
            .unwrap()
            .downcast_ref::<stablehlo::attributes::DenseElementsAttr>()
            .is_some()
    );

    let assembly = module.disp(&ctx).to_string();
    let mut parsed_ctx = Context::new();
    ChloConstantOp::register(&mut parsed_ctx);
    ErfOp::register(&mut parsed_ctx);
    BroadcastCompareOp::register(&mut parsed_ctx);
    TopKOp::register(&mut parsed_ctx);
    RaggedDotOp::register(&mut parsed_ctx);
    ScanOp::register(&mut parsed_ctx);
    ReturnOp::register(&mut parsed_ctx);
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &assembly,
    )
    .unwrap_or_else(|error| panic!("native module failed to parse:\n{assembly}\n{error}"));
    let reparsed_assembly = parsed.disp(&parsed_ctx).to_string();
    assert!(reparsed_assembly.contains("chlo.broadcast_compare"));
    assert!(reparsed_assembly.contains("chlo.ragged_dot"));
    assert!(reparsed_assembly.contains("chlo.precision_config [HIGH, HIGHEST]"));
    assert!(reparsed_assembly.contains("chlo.scan"));
    assert!(reparsed_assembly.contains("stablehlo.return"));

    let reparsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &reparsed_assembly,
    )
    .unwrap_or_else(|error| panic!("reprinted native module failed to parse:\n{error}"));
    let reparsed_module = Operation::get_op_dyn(reparsed, &parsed_ctx);
    reparsed_module
        .as_ref()
        .downcast_ref::<ModuleOp>()
        .unwrap()
        .verify(&parsed_ctx)
        .unwrap();
    let reparsed_block = reparsed
        .deref(&parsed_ctx)
        .get_region(0)
        .deref(&parsed_ctx)
        .iter(&parsed_ctx)
        .next()
        .unwrap();
    let reparsed_operations: Vec<_> = reparsed_block
        .deref(&parsed_ctx)
        .iter(&parsed_ctx)
        .collect();
    assert_eq!(reparsed_operations.len(), 12);
    for operation in &reparsed_operations {
        let dynamic = Operation::get_op_dyn(*operation, &parsed_ctx);
        clienthlo::pliron::op::verify_op(dynamic.as_ref(), &parsed_ctx)
            .unwrap_or_else(|error| panic!("reprinted operation verification failed: {error}"));
    }
    let reparsed_scan = reparsed_operations
        .iter()
        .find_map(|operation| {
            let dynamic = Operation::get_op_dyn(*operation, &parsed_ctx);
            dynamic
                .as_ref()
                .downcast_ref::<ScanOp>()
                .map(|scan| scan.entry_block(&parsed_ctx).unwrap())
        })
        .unwrap();
    assert_eq!(reparsed_scan.deref(&parsed_ctx).arguments().count(), 2);

    let parsed_module = Operation::get_op_dyn(parsed, &parsed_ctx);
    parsed_module
        .as_ref()
        .downcast_ref::<ModuleOp>()
        .unwrap()
        .verify(&parsed_ctx)
        .unwrap();
    let parsed_block = parsed
        .deref(&parsed_ctx)
        .get_region(0)
        .deref(&parsed_ctx)
        .iter(&parsed_ctx)
        .next()
        .unwrap();
    let parsed_ops: Vec<_> = parsed_block.deref(&parsed_ctx).iter(&parsed_ctx).collect();
    let mut saw_typed_constant = false;
    let mut saw_unary_type = false;
    let mut saw_compare = false;
    let mut saw_top_k = false;
    let mut saw_ragged_dot = false;
    let mut saw_scan = false;
    for operation in parsed_ops {
        let dynamic = Operation::get_op_dyn(operation, &parsed_ctx);
        clienthlo::pliron::op::verify_op(dynamic.as_ref(), &parsed_ctx)
            .unwrap_or_else(|error| panic!("parsed operation verification failed: {error}"));
        if let Some(constant) = dynamic.as_ref().downcast_ref::<ChloConstantOp>()
            && let Some(dense) = constant.value(&parsed_ctx).and_then(|value| {
                value
                    .downcast_ref::<stablehlo::attributes::DenseElementsAttr>()
                    .cloned()
            })
            && !dense.is_splat
        {
            let words: Vec<_> = dense
                .values
                .0
                .iter()
                .map(|value| {
                    value
                        .downcast_ref::<stablehlo::attributes::ScalarBitsAttr>()
                        .unwrap()
                        .words
                        .clone()
                })
                .collect();
            assert_eq!(words, vec![vec![0x3f80_0000], vec![0x4000_0000]]);
            saw_typed_constant = true;
        }
        if let Some(unary) = dynamic.as_ref().downcast_ref::<ErfOp>() {
            let result = unary.get_operation().deref(&parsed_ctx).get_result(0);
            assert_eq!(
                tensor_parts(&parsed_ctx, result.get_type(&parsed_ctx)),
                Some((vec![2, 3, 4], ElementType::F32))
            );
            saw_unary_type = true;
        }
        if let Some(compare) = dynamic.as_ref().downcast_ref::<BroadcastCompareOp>() {
            assert_eq!(
                compare.comparison_direction(&parsed_ctx),
                Some(ComparisonDirection::Gt)
            );
            assert_eq!(compare.comparison_kind(&parsed_ctx), None);
            assert_eq!(compare.broadcast_dimensions(&parsed_ctx), None);
            assert_eq!(
                tensor_parts(
                    &parsed_ctx,
                    compare
                        .get_operation()
                        .deref(&parsed_ctx)
                        .get_result(0)
                        .get_type(&parsed_ctx)
                ),
                Some((vec![2, 3], ElementType::I1))
            );
            saw_compare = true;
        }
        if let Some(top_k) = dynamic.as_ref().downcast_ref::<TopKOp>() {
            assert_eq!(top_k.k(&parsed_ctx), Some(2));
            assert!(top_k.is_stable(&parsed_ctx));
            let top_k_ir = top_k.get_operation().deref(&parsed_ctx);
            assert_eq!(top_k_ir.get_num_results(), 2);
            assert_eq!(
                tensor_parts(&parsed_ctx, top_k_ir.get_result(0).get_type(&parsed_ctx)),
                Some((vec![2, 3, 2], ElementType::F32))
            );
            assert_eq!(
                tensor_parts(&parsed_ctx, top_k_ir.get_result(1).get_type(&parsed_ctx)),
                Some((vec![2, 3, 2], ElementType::I32))
            );
            saw_top_k = true;
        }
        if let Some(ragged_dot) = dynamic.as_ref().downcast_ref::<RaggedDotOp>() {
            assert_eq!(
                ragged_dot.precision_config(&parsed_ctx),
                Some(vec![Precision::High, Precision::Highest])
            );
            assert_eq!(
                read_precision_config(&ragged_dot.precision_config_attr(&parsed_ctx).unwrap()),
                vec![Precision::High, Precision::Highest]
            );
            assert_eq!(
                ragged_dot
                    .dimension_numbers(&parsed_ctx)
                    .unwrap()
                    .dimensions(),
                RaggedDotDimensions {
                    lhs_batching_dimensions: vec![0],
                    rhs_batching_dimensions: vec![0],
                    lhs_contracting_dimensions: vec![2],
                    rhs_contracting_dimensions: vec![1],
                    lhs_ragged_dimensions: vec![2],
                    rhs_group_dimensions: vec![],
                }
            );
            assert_eq!(
                tensor_parts(
                    &parsed_ctx,
                    ragged_dot
                        .get_operation()
                        .deref(&parsed_ctx)
                        .get_result(0)
                        .get_type(&parsed_ctx)
                ),
                Some((vec![2, 2, 3, 5], ElementType::F32))
            );
            saw_ragged_dot = true;
        }
        if let Some(scan) = dynamic.as_ref().downcast_ref::<ScanOp>() {
            let scan_body = scan.entry_block(&parsed_ctx).unwrap();
            assert_eq!(scan_body.deref(&parsed_ctx).arguments().count(), 2);
            assert_eq!(scan_body.deref(&parsed_ctx).iter(&parsed_ctx).count(), 1);
            assert_eq!(scan.scan_dim_size(&parsed_ctx), None);
            assert_eq!(scan.is_associative(&parsed_ctx), None);
            assert_eq!(scan.is_reverse(&parsed_ctx), Some(false));
            assert_eq!(scan.inputs(&parsed_ctx).unwrap().len(), 1);
            assert_eq!(scan.inits(&parsed_ctx).unwrap().len(), 1);
            saw_scan = true;
        }
    }
    assert!(
        saw_typed_constant
            && saw_unary_type
            && saw_compare
            && saw_top_k
            && saw_ragged_dot
            && saw_scan
    );
}
