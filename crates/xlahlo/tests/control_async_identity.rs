#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    attributes::{
        Axes, Bytes, ComparisonOrder, Direction, OutputOperandAliases, OutputOperandAliasing,
        Payload, SortOptions, TriState,
    },
    inference::numerical::infer_top_k,
    ir::{self, BackendConfigPayload, ExecutionThread, MetadataPayloads},
    ops::{
        control::*,
        elementwise::CompareOp,
        numerical::{NumericalInteger, ScanOp, SortOp, TopKOp},
    },
    pliron::{
        builtin::attributes::StringAttr,
        common_traits::Verify,
        identifier::Identifier,
        irbuild::{
            cloning::{IrMapping, clone_operation},
            listener::DummyListener,
            rewriter::IRRewriter,
        },
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
    },
    shape::Shape,
    types::ElementType,
    verification,
};

#[test]
fn nested_async_identity_preserves_output_operand_aliases() {
    for use_update in [false, true] {
        let mut ctx = context();
        let data = shape(&ctx, ElementType::F32, &[4]);
        let state = shape(&ctx, ElementType::U32, &[]);
        let state = Shape::tuple(&ctx, vec![state, state]).unwrap();
        let async_context = Shape::tuple(
            &ctx,
            vec![Shape::tuple(&ctx, vec![data]).unwrap(), data, state],
        )
        .unwrap();
        let mut computations = vec![];
        let mut starts = vec![];
        let mut dones = vec![];
        for (body_name, wrapper_name) in [("inner_a", "first"), ("inner_b", "second")] {
            let (body, block, args) = computation(&mut ctx, body_name, vec![data], data);
            ret(&mut ctx, block, args[0]);
            computations.push(body);
            let (wrapper, block, args) = computation(&mut ctx, wrapper_name, vec![data], data);
            attr(
                &ctx,
                wrapper.get_operation(),
                "hlo_execution_thread",
                ExecutionThread("worker".into()),
            );
            let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&[body_name]));
            start.get_operation().insert_at_back(block, &ctx);
            let value = start.result(&ctx);
            let update = AsyncUpdateOp::new(&mut ctx, async_context, vec![value]);
            update.get_operation().insert_at_back(block, &ctx);
            let value = update.result(&ctx);
            let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
            done.get_operation().insert_at_back(block, &ctx);
            dones.push(done.get_operation());
            let value = done.result(&ctx);
            ret(&mut ctx, block, value);
            computations.push(wrapper);
            starts.push(if use_update {
                update.get_operation()
            } else {
                start.get_operation()
            });
        }
        let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
        let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
        attr(
            &ctx,
            start.get_operation(),
            "hlo_async_execution_thread",
            StringAttr::new("worker".into()),
        );
        start.get_operation().insert_at_back(block, &ctx);
        let value = start.result(&ctx);
        let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
        attr(
            &ctx,
            done.get_operation(),
            "hlo_callees",
            callees(&["second"]),
        );
        done.get_operation().insert_at_back(block, &ctx);
        let value = done.result(&ctx);
        ret(&mut ctx, block, value);
        computations.push(main);
        let module = module(&mut ctx, &computations, "main");
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, dones[0], "hlo_aliases", Axes(vec![7]));
        attr(&ctx, dones[1], "hlo_aliases", OutputOperandAliases(vec![]));
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, starts[0], "hlo_aliases", OutputOperandAliases(vec![]));
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, starts[1], "hlo_aliases", Axes(vec![]));
        let invalid = module.get_operation().disp(&ctx).to_string();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
        attr(&ctx, starts[1], "hlo_aliases", OutputOperandAliases(vec![]));
        verification::verify(&ctx, module.get_operation()).unwrap();
        let aliases = OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(if use_update { vec![2, 0] } else { vec![1] }),
            operand_index: 0,
            operand_shape_index: Axes(if use_update { vec![2, 0] } else { vec![] }),
        }]);
        attr(&ctx, starts[1], "hlo_aliases", aliases.clone());
        let invalid = module.get_operation().disp(&ctx).to_string();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
        attr(&ctx, starts[0], "hlo_aliases", aliases);
        verification::verify(&ctx, module.get_operation()).unwrap();
        {
            let output_shape_index = Axes(if use_update { vec![2, 1] } else { vec![0, 0] });
            let operand_shape_index = Axes(if use_update { vec![2, 0] } else { vec![] });
            attr(
                &ctx,
                starts[1],
                "hlo_aliases",
                OutputOperandAliases(vec![OutputOperandAliasing {
                    output_shape_index: output_shape_index.clone(),
                    operand_index: 0,
                    operand_shape_index: operand_shape_index.clone(),
                }]),
            );
            let invalid = module.get_operation().disp(&ctx).to_string();
            assert!(verification::verify(&ctx, module.get_operation()).is_err());
            assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
            attr(
                &ctx,
                starts[0],
                "hlo_aliases",
                OutputOperandAliases(vec![OutputOperandAliasing {
                    output_shape_index: output_shape_index.clone(),
                    operand_index: 0,
                    operand_shape_index: operand_shape_index.clone(),
                }]),
            );
            verification::verify(&ctx, module.get_operation()).unwrap();
            let first = OutputOperandAliasing {
                output_shape_index,
                operand_index: 0,
                operand_shape_index: operand_shape_index.clone(),
            };
            let second = OutputOperandAliasing {
                output_shape_index: Axes(if use_update { vec![2, 0] } else { vec![1] }),
                operand_index: 0,
                operand_shape_index,
            };
            attr(
                &ctx,
                starts[0],
                "hlo_aliases",
                OutputOperandAliases(vec![first.clone(), second.clone()]),
            );
            attr(
                &ctx,
                starts[1],
                "hlo_aliases",
                OutputOperandAliases(vec![second.clone(), first.clone()]),
            );
            let invalid = module.get_operation().disp(&ctx).to_string();
            assert!(verification::verify(&ctx, module.get_operation()).is_err());
            assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
            attr(
                &ctx,
                starts[1],
                "hlo_aliases",
                OutputOperandAliases(vec![first, second]),
            );
            verification::verify(&ctx, module.get_operation()).unwrap();
        }
        let text = module.get_operation().disp(&ctx).to_string();
        let mut fresh = context();
        let parsed =
            parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
        verification::verify(&fresh, parsed).unwrap();
        Operation::erase(parsed, &mut fresh);
        assert!(fresh.is_ir_empty());
        Operation::erase(module.get_operation(), &mut ctx);
        assert!(ctx.is_ir_empty());
    }
}

#[test]
fn async_wrapped_identity_resolves_backend_payloads_before_legacy_configuration() {
    let mut ctx = context();
    let data = shape(&ctx, ElementType::F32, &[4]);
    let (first, block, args) = computation(&mut ctx, "first", vec![data], data);
    let a = CustomCallOp::new(&mut ctx, data, args, StringAttr::new("compute".into()));
    a.get_operation().insert_at_back(block, &ctx);
    let value = a.result(&ctx);
    ret(&mut ctx, block, value);
    let (second, block, args) = computation(&mut ctx, "second", vec![data], data);
    let b = CustomCallOp::new(&mut ctx, data, args, StringAttr::new("compute".into()));
    b.get_operation().insert_at_back(block, &ctx);
    let value = b.result(&ctx);
    ret(&mut ctx, block, value);
    let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
    let state = shape(&ctx, ElementType::U32, &[]);
    let async_context = Shape::tuple(
        &ctx,
        vec![Shape::tuple(&ctx, vec![data]).unwrap(), data, state],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    let module = module(&mut ctx, &[first, second, main], "main");
    ir::set_backend_config_payload(
        &ctx,
        a.get_operation(),
        BackendConfigPayload(Payload::Value(Bytes(vec![0xff, 0, 1]))),
    )
    .unwrap();
    ir::set_backend_config_payload(
        &ctx,
        b.get_operation(),
        BackendConfigPayload(Payload::Value(Bytes(vec![0xff, 0, 2]))),
    )
    .unwrap();
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(
        &ctx,
        module.get_operation(),
        "hlo_metadata_payloads",
        MetadataPayloads(vec![
            Bytes(vec![0xff, 0, 1]),
            Bytes(vec![0xff, 0, 1]),
            Bytes(vec![0xff, 0, 2]),
            Bytes::default(),
        ]),
    );
    attr(
        &ctx,
        a.get_operation(),
        "hlo_backend_config",
        StringAttr::new("old-a".into()),
    );
    attr(
        &ctx,
        b.get_operation(),
        "hlo_backend_config",
        StringAttr::new("old-b".into()),
    );
    ir::set_backend_config_payload(
        &ctx,
        b.get_operation(),
        BackendConfigPayload(Payload::Id(0)),
    )
    .unwrap();
    verification::verify(&ctx, module.get_operation()).unwrap();
    ir::set_backend_config_payload(
        &ctx,
        a.get_operation(),
        BackendConfigPayload(Payload::Id(1)),
    )
    .unwrap();
    verification::verify(&ctx, module.get_operation()).unwrap();
    ir::set_backend_config_payload(
        &ctx,
        b.get_operation(),
        BackendConfigPayload(Payload::Id(2)),
    )
    .unwrap();
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    ir::set_backend_config_payload(
        &ctx,
        a.get_operation(),
        BackendConfigPayload(Payload::Absent),
    )
    .unwrap();
    ir::set_backend_config_payload(
        &ctx,
        b.get_operation(),
        BackendConfigPayload(Payload::Id(3)),
    )
    .unwrap();
    verification::verify(&ctx, module.get_operation()).unwrap();
    ir::set_backend_config_payload(
        &ctx,
        b.get_operation(),
        BackendConfigPayload(Payload::Value(Bytes::default())),
    )
    .unwrap();
    verification::verify(&ctx, module.get_operation()).unwrap();
    for invalid in [-1, 4] {
        ir::set_backend_config_payload(
            &ctx,
            b.get_operation(),
            BackendConfigPayload(Payload::Id(invalid)),
        )
        .unwrap();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
    }
    ir::set_backend_config_payload(
        &ctx,
        b.get_operation(),
        BackendConfigPayload(Payload::Absent),
    )
    .unwrap();
    verification::verify(&ctx, module.get_operation()).unwrap();
    b.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_backend_config_payload"));
    attr(
        &ctx,
        b.get_operation(),
        "hlo_backend_config",
        StringAttr::new(String::new()),
    );
    verification::verify(&ctx, module.get_operation()).unwrap();
    attr(
        &ctx,
        b.get_operation(),
        "hlo_backend_config",
        StringAttr::new("legacy".into()),
    );
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    a.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_backend_config_payload"));
    attr(
        &ctx,
        a.get_operation(),
        "hlo_backend_config",
        StringAttr::new("legacy".into()),
    );
    verification::verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn scan_direction_is_typed_preserved_and_compared_with_forward_as_default() {
    let mut ctx = context();
    let data = shape(&ctx, ElementType::F32, &[4]);
    let scalar = shape(&ctx, ElementType::F32, &[]);
    let (body, block, args) = computation(&mut ctx, "body", vec![scalar], scalar);
    ret(&mut ctx, block, args[0]);
    let (first, block, args) = computation(&mut ctx, "first", vec![data], data);
    let a = ScanOp::new(
        &mut ctx,
        data,
        args,
        Axes(vec![0]),
        NumericalInteger(0),
        TriState::False,
        callees(&["body"]),
    );
    a.get_operation().insert_at_back(block, &ctx);
    let value = a.result(&ctx);
    ret(&mut ctx, block, value);
    let (second, block, args) = computation(&mut ctx, "second", vec![data], data);
    let b = ScanOp::new(
        &mut ctx,
        data,
        args,
        Axes(vec![0]),
        NumericalInteger(0),
        TriState::False,
        callees(&["body"]),
    );
    b.get_operation().insert_at_back(block, &ctx);
    let value = b.result(&ctx);
    ret(&mut ctx, block, value);
    let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
    let state = shape(&ctx, ElementType::U32, &[]);
    let async_context = Shape::tuple(
        &ctx,
        vec![Shape::tuple(&ctx, vec![data]).unwrap(), data, state],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    let module = module(&mut ctx, &[body, first, second, main], "main");
    assert!(!a.is_reverse(&ctx).unwrap());
    b.set_is_reverse(&ctx, false);
    verification::verify(&ctx, module.get_operation()).unwrap();
    b.set_is_reverse(&ctx, true);
    assert!(b.is_reverse(&ctx).unwrap());
    b.verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    a.set_is_reverse(&ctx, true);
    verification::verify(&ctx, module.get_operation()).unwrap();
    let text = module.get_operation().disp(&ctx).to_string();
    let mut parsed_ctx = context();
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &text,
    )
    .unwrap();
    verification::verify(&parsed_ctx, parsed).unwrap();
    let mut pending = vec![parsed];
    let mut parsed_directions = Vec::new();
    while let Some(operation) = pending.pop() {
        if let Some(scan) = Operation::get_op::<ScanOp>(operation, &parsed_ctx) {
            parsed_directions.push(scan.is_reverse(&parsed_ctx).unwrap());
        }
        for region in operation.deref(&parsed_ctx).regions() {
            for block in region.deref(&parsed_ctx).iter(&parsed_ctx) {
                pending.extend(block.deref(&parsed_ctx).iter(&parsed_ctx));
            }
        }
    }
    assert_eq!(parsed_directions, vec![true, true]);
    let mut mapping = IrMapping::new();
    let cloned = clone_operation(
        module.get_operation(),
        &mut ctx,
        &mut IRRewriter::<DummyListener>::default(),
        &mut mapping,
    );
    verification::verify(&ctx, cloned).unwrap();
    let cloned_b =
        Operation::get_op::<ScanOp>(mapping.lookup_op(b.get_operation()).unwrap(), &ctx).unwrap();
    assert!(cloned_b.is_reverse(&ctx).unwrap());
    cloned_b.set_is_reverse(&ctx, false);
    assert!(verification::verify(&ctx, cloned).is_err());
    assert!(b.is_reverse(&ctx).unwrap());
    attr(
        &ctx,
        b.get_operation(),
        "hlo_is_reverse",
        StringAttr::new("true".into()),
    );
    assert!(b.is_reverse(&ctx).is_err());
    assert!(b.verify(&ctx).is_err());
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    b.set_is_reverse(&ctx, true);
    verification::verify(&ctx, module.get_operation()).unwrap();
    assert_eq!(b.num_carries(&ctx).unwrap().0, 0);
    assert_eq!(*b.is_associative(&ctx).unwrap(), TriState::False);
    assert_eq!(b.dimensions(&ctx).unwrap().0, vec![0]);
}

#[test]
fn async_wrapped_topk_identity_preserves_k_largest_and_stability() {
    let mut ctx = context();
    let data = shape(&ctx, ElementType::F32, &[4]);
    let output = infer_top_k(&ctx, data, 2).unwrap();
    let (first, block, args) = computation(&mut ctx, "first", vec![data], output);
    let a = TopKOp::new(
        &mut ctx,
        output,
        args,
        NumericalInteger(2),
        SortOptions { descending: true },
    );
    a.get_operation().insert_at_back(block, &ctx);
    let value = a.result(&ctx);
    ret(&mut ctx, block, value);
    let (second, block, args) = computation(&mut ctx, "second", vec![data], output);
    let b = TopKOp::new(
        &mut ctx,
        output,
        args,
        NumericalInteger(2),
        SortOptions { descending: true },
    );
    b.get_operation().insert_at_back(block, &ctx);
    let value = b.result(&ctx);
    ret(&mut ctx, block, value);
    let (main, block, args) = computation(&mut ctx, "main", vec![data], output);
    let state = shape(&ctx, ElementType::U32, &[]);
    let async_context = Shape::tuple(
        &ctx,
        vec![Shape::tuple(&ctx, vec![data]).unwrap(), output, state],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, output, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    let module = module(&mut ctx, &[first, second, main], "main");
    verification::verify(&ctx, module.get_operation()).unwrap();
    b.set_is_stable(&ctx, true);
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    a.set_is_stable(&ctx, true);
    verification::verify(&ctx, module.get_operation()).unwrap();
    a.set_is_stable(&ctx, false);
    b.set_is_stable(&ctx, false);
    verification::verify(&ctx, module.get_operation()).unwrap();
    attr(
        &ctx,
        b.get_operation(),
        "hlo_options",
        SortOptions { descending: false },
    );
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(
        &ctx,
        b.get_operation(),
        "hlo_options",
        SortOptions { descending: true },
    );
    verification::verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn async_wrapped_sort_identity_includes_stability_with_false_as_default() {
    let mut ctx = context();
    let data = shape(&ctx, ElementType::F32, &[4]);
    let scalar = shape(&ctx, ElementType::F32, &[]);
    let predicate = shape(&ctx, ElementType::Pred, &[]);
    let (compare, block, args) = computation(&mut ctx, "compare", vec![scalar, scalar], predicate);
    let comparison = CompareOp::new(
        &mut ctx,
        predicate,
        args,
        Direction::Lt,
        ComparisonOrder::Total,
    );
    comparison.get_operation().insert_at_back(block, &ctx);
    let comparison_result = comparison.result(&ctx);
    ret(&mut ctx, block, comparison_result);

    let (first, block, args) = computation(&mut ctx, "first", vec![data], data);
    let first_sort = SortOp::new(&mut ctx, data, args, Axes(vec![0]), callees(&["compare"]));
    first_sort.get_operation().insert_at_back(block, &ctx);
    let first_result = first_sort.result(&ctx);
    ret(&mut ctx, block, first_result);

    let (second, block, args) = computation(&mut ctx, "second", vec![data], data);
    let second_sort = SortOp::new(&mut ctx, data, args, Axes(vec![0]), callees(&["compare"]));
    second_sort.get_operation().insert_at_back(block, &ctx);
    let second_result = second_sort.result(&ctx);
    ret(&mut ctx, block, second_result);

    let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
    let state = shape(&ctx, ElementType::U32, &[]);
    let async_context = Shape::tuple(
        &ctx,
        vec![Shape::tuple(&ctx, vec![data]).unwrap(), data, state],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let start_result = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, data, vec![start_result]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let done_result = done.result(&ctx);
    ret(&mut ctx, block, done_result);
    let module = module(&mut ctx, &[compare, first, second, main], "main");

    verification::verify(&ctx, module.get_operation()).unwrap();
    second_sort.set_is_stable(&ctx, true);
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    second_sort.set_is_stable(&ctx, false);
    verification::verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn async_wrapped_custom_calls_require_identical_convolution_group_counts() {
    for key in ["hlo_feature_groups", "hlo_batch_groups"] {
        let mut ctx = context();
        let data = shape(&ctx, ElementType::F32, &[4]);
        let state = shape(&ctx, ElementType::U32, &[]);
        let context = Shape::tuple(
            &ctx,
            vec![Shape::tuple(&ctx, vec![data]).unwrap(), data, state],
        )
        .unwrap();
        let (first, block, args) = computation(&mut ctx, "first", vec![data], data);
        let a = CustomCallOp::new(&mut ctx, data, args, StringAttr::new("convolve".into()));
        a.get_operation().insert_at_back(block, &ctx);
        let value = a.result(&ctx);
        ret(&mut ctx, block, value);
        let (second, block, args) = computation(&mut ctx, "second", vec![data], data);
        let b = CustomCallOp::new(&mut ctx, data, args, StringAttr::new("convolve".into()));
        attr(&ctx, b.get_operation(), key, NumericalInteger(1));
        b.get_operation().insert_at_back(block, &ctx);
        let value = b.result(&ctx);
        ret(&mut ctx, block, value);
        let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
        let start = AsyncStartOp::new(&mut ctx, context, args, callees(&["first"]));
        start.get_operation().insert_at_back(block, &ctx);
        let value = start.result(&ctx);
        let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
        attr(
            &ctx,
            done.get_operation(),
            "hlo_callees",
            callees(&["second"]),
        );
        done.get_operation().insert_at_back(block, &ctx);
        let value = done.result(&ctx);
        ret(&mut ctx, block, value);
        let module = module(&mut ctx, &[first, second, main], "main");
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, b.get_operation(), key, NumericalInteger(2));
        assert!(
            verification::verify(&ctx, module.get_operation()).is_err(),
            "{key}"
        );
        attr(&ctx, b.get_operation(), key, NumericalInteger(1));
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}
