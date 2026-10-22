use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::VecAttr,
    common_traits::Verify,
    context::Context,
    irfmt::parsers::spaced,
    op::Op,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    r#type::{Type, TypeHandle},
};
use stablehlo::{
    attributes::{ArrayOfLayoutAttr, LayoutAttr, OutputOperandAliasAttr},
    ops::{CustomCallOp, FuncOp, ModuleOp, ReturnOp},
    types::{
        BufferType, ElementType, FunctionType, TensorType, TokenType, TupleType,
        UnrankedBufferType, UnrankedTensorType, VectorType, tensor,
    },
    verification::verify_module_fast,
};

fn call(
    ctx: &mut Context,
    target: &str,
    inputs: Vec<TypeHandle>,
    results: Vec<TypeHandle>,
) -> CustomCallOp {
    let block = BasicBlock::new(ctx, None, inputs);
    let inputs = block.deref(ctx).arguments().collect();
    CustomCallOp::new(ctx, target.into(), inputs, results, 1)
}

fn aliases(entries: &[(Vec<i64>, i64, Vec<i64>)]) -> VecAttr {
    VecAttr::new(
        entries
            .iter()
            .map(|(output, operand, input)| {
                OutputOperandAliasAttr {
                    output_tuple_indices: output.clone(),
                    operand_index: *operand,
                    operand_tuple_indices: input.clone(),
                }
                .into()
            })
            .collect(),
    )
}

fn assert_invalid_without_mutation(ctx: &Context, call: CustomCallOp) {
    let before = call.get_operation().disp(ctx).to_string();
    for _ in 0..2 {
        assert!(call.verify(ctx).is_err(), "{before}");
        assert_eq!(call.get_operation().disp(ctx).to_string(), before);
    }
}

#[test]
fn special_buffer_calls_accept_dynamic_geometry_and_unranked_tensors_without_aliases() {
    let mut ctx = Context::new();
    for element in [ElementType::F32, ElementType::I8, ElementType::C128] {
        for (tensor_dims, buffer_dims) in [
            (vec![], vec![]),
            (vec![0, 3], vec![0, 3]),
            (vec![-1, 3], vec![2, 3]),
            (vec![2, 3], vec![-1, -1]),
        ] {
            let tensor = tensor(&ctx, &tensor_dims, element);
            let buffer = BufferType::without_layout(&ctx, element.native(&ctx), buffer_dims);
            call(&mut ctx, "CreateBuffer", vec![], vec![buffer])
                .verify(&ctx)
                .unwrap();
            call(&mut ctx, "Pin", vec![tensor], vec![buffer])
                .verify(&ctx)
                .unwrap();
            call(&mut ctx, "Unpin", vec![buffer], vec![tensor])
                .verify(&ctx)
                .unwrap();
            let unranked = UnrankedTensorType::get(&ctx, element.native(&ctx)).into();
            call(&mut ctx, "Pin", vec![unranked], vec![buffer])
                .verify(&ctx)
                .unwrap();
            call(&mut ctx, "Unpin", vec![buffer], vec![unranked])
                .verify(&ctx)
                .unwrap();
        }
    }
}

#[test]
fn special_buffer_calls_reject_invalid_counts_categories_shapes_and_elements_atomically() {
    let mut ctx = Context::new();
    let t = tensor(&ctx, &[2, 3], ElementType::F32);
    let buffer = BufferType::without_layout(&ctx, ElementType::F32.native(&ctx), vec![2, 3]);
    let unranked = UnrankedBufferType::without_memory_space(&ctx, ElementType::F32.native(&ctx));
    let token = TokenType::get(&ctx).into();
    let tuple = TupleType::get(&ctx, vec![buffer]).into();
    for results in [
        vec![],
        vec![buffer, buffer],
        vec![t],
        vec![unranked],
        vec![tuple],
        vec![token],
    ] {
        let op = call(&mut ctx, "CreateBuffer", vec![], results);
        assert_invalid_without_mutation(&ctx, op);
    }
    let op = call(&mut ctx, "CreateBuffer", vec![t], vec![buffer]);
    assert_invalid_without_mutation(&ctx, op);
    for target in ["Pin", "Unpin"] {
        let (input, output) = if target == "Pin" {
            (t, buffer)
        } else {
            (buffer, t)
        };
        for (inputs, outputs) in [
            (vec![], vec![output]),
            (vec![input, input], vec![output]),
            (vec![input], vec![]),
            (vec![input], vec![output, output]),
            (vec![output], vec![input]),
        ] {
            let op = call(&mut ctx, target, inputs, outputs);
            assert_invalid_without_mutation(&ctx, op);
        }
        for bad in [
            tensor(&ctx, &[2], ElementType::F32),
            tensor(&ctx, &[3, 2], ElementType::F32),
            tensor(&ctx, &[2, 3], ElementType::I32),
            token,
            tuple,
        ] {
            let (input, output) = if target == "Pin" {
                (bad, buffer)
            } else {
                (buffer, bad)
            };
            let op = call(&mut ctx, target, vec![input], vec![output]);
            assert_invalid_without_mutation(&ctx, op);
        }
        let (input, output) = if target == "Pin" {
            (t, unranked)
        } else {
            (unranked, t)
        };
        let op = call(&mut ctx, target, vec![input], vec![output]);
        assert_invalid_without_mutation(&ctx, op);
    }
    let op = call(&mut ctx, "Pin", vec![t], vec![buffer]);
    op.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, aliases(&[(vec![], 0, vec![])]));
    assert_invalid_without_mutation(&ctx, op);
    op.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, aliases(&[]));
    op.verify(&ctx).unwrap();
}

#[test]
fn generic_buffer_aliases_cover_nested_results_but_need_not_cover_every_operand() {
    let mut ctx = Context::new();
    let buffer = BufferType::without_layout(&ctx, ElementType::F32.native(&ctx), vec![2]);
    let t = tensor(&ctx, &[2], ElementType::F32);
    let empty = TupleType::get(&ctx, vec![]).into();
    let inner = TupleType::get(&ctx, vec![t, buffer, empty]).into();
    let nested = TupleType::get(&ctx, vec![inner, buffer]).into();
    for (outputs, entries) in [
        (
            vec![nested],
            vec![(vec![0, 1], 0, vec![0, 1]), (vec![1], 0, vec![1])],
        ),
        (
            vec![t, nested],
            vec![(vec![1, 0, 1], 0, vec![0, 1]), (vec![1, 1], 0, vec![1])],
        ),
    ] {
        let op = call(&mut ctx, "external", vec![nested, buffer], outputs);
        assert_invalid_without_mutation(&ctx, op);
        op.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, aliases(&entries[..1]));
        assert_invalid_without_mutation(&ctx, op);
        op.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, aliases(&entries));
        op.verify(&ctx).unwrap();
    }
    call(&mut ctx, "external", vec![buffer], vec![])
        .verify(&ctx)
        .unwrap();
    let unranked = UnrankedBufferType::without_memory_space(&ctx, ElementType::F32.native(&ctx));
    call(&mut ctx, "external", vec![], vec![unranked])
        .verify(&ctx)
        .unwrap();
}

#[test]
fn generic_buffer_aliases_reject_duplicate_paths_without_restricting_array_aliases() {
    let mut ctx = Context::new();
    let buffer = BufferType::without_layout(&ctx, ElementType::F32.native(&ctx), vec![2]);
    let t = tensor(&ctx, &[2], ElementType::F32);
    for (inputs, outputs, entries) in [
        (
            vec![buffer],
            vec![buffer],
            vec![(vec![], 0, vec![]), (vec![], 0, vec![])],
        ),
        (
            vec![buffer],
            vec![buffer, buffer],
            vec![(vec![0], 0, vec![]), (vec![1], 0, vec![])],
        ),
        (
            vec![buffer, buffer],
            vec![buffer],
            vec![(vec![], 0, vec![]), (vec![], 1, vec![])],
        ),
    ] {
        let op = call(&mut ctx, "external", inputs, outputs);
        op.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, aliases(&entries));
        assert_invalid_without_mutation(&ctx, op);
    }
    let op = call(&mut ctx, "external", vec![t], vec![t, t]);
    op.set_attr_stablehlo_custom_call_output_operand_aliases(
        &ctx,
        aliases(&[
            (vec![0], 0, vec![]),
            (vec![0], 0, vec![]),
            (vec![1], 0, vec![]),
        ]),
    );
    op.verify(&ctx).unwrap();
}

#[test]
fn layouts_treat_unranked_buffers_and_vectors_as_shaped_types() {
    let mut ctx = Context::new();
    let unranked = UnrankedBufferType::without_memory_space(&ctx, ElementType::F32.native(&ctx));
    let vector = VectorType::fixed(&ctx, ElementType::F32, vec![2, 3]);
    for (ty, layout) in [(unranked, vec![9, 9]), (vector, vec![1, 0])] {
        let op = call(&mut ctx, "external", vec![ty], vec![ty]);
        let layouts = ArrayOfLayoutAttr {
            layouts: vec![LayoutAttr {
                minor_to_major: layout,
            }],
        };
        op.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts.clone());
        op.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts);
        op.verify(&ctx).unwrap();
        if ty == vector {
            op.set_attr_stablehlo_custom_call_result_layouts(
                &ctx,
                ArrayOfLayoutAttr {
                    layouts: vec![LayoutAttr {
                        minor_to_major: vec![0, 0],
                    }],
                },
            );
            assert_invalid_without_mutation(&ctx, op);
        }
    }
}

#[test]
fn buffer_custom_calls_roundtrip_through_a_fresh_registered_context() {
    let mut ctx = Context::new();
    let tensor = tensor(&ctx, &[2, 3], ElementType::F32);
    let buffer = BufferType::without_layout(&ctx, ElementType::F32.native(&ctx), vec![2, 3]);
    let signature = FunctionType::intern(&ctx, vec![tensor], vec![tensor]);
    let module = ModuleOp::new(&mut ctx);
    let function = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
    let block = function.entry_block(&ctx).unwrap();
    let input = function.argument(&ctx, 0).unwrap();
    let pin = CustomCallOp::new(&mut ctx, "Pin".into(), vec![input], vec![buffer], 1);
    pin.get_operation().insert_at_back(block, &ctx);
    let pinned = pin.get_operation().deref(&ctx).get_result(0);
    let generic = CustomCallOp::new(&mut ctx, "external".into(), vec![pinned], vec![buffer], 1);
    generic.set_attr_stablehlo_custom_call_output_operand_aliases(
        &ctx,
        aliases(&[(vec![], 0, vec![])]),
    );
    generic.get_operation().insert_at_back(block, &ctx);
    let generic_result = generic.get_operation().deref(&ctx).get_result(0);
    let unpin = CustomCallOp::new(
        &mut ctx,
        "Unpin".into(),
        vec![generic_result],
        vec![tensor],
        1,
    );
    unpin.get_operation().insert_at_back(block, &ctx);
    let unpinned = unpin.get_operation().deref(&ctx).get_result(0);
    ReturnOp::new(&mut ctx, unpinned)
        .get_operation()
        .insert_at_back(block, &ctx);
    function
        .get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    verify_module_fast(&ctx, module.get_operation()).unwrap();
    let text = module.get_operation().disp(&ctx).to_string();
    let mut fresh = Context::new();
    TensorType::register(&mut fresh);
    BufferType::register(&mut fresh);
    FunctionType::register(&mut fresh);
    ModuleOp::register(&mut fresh);
    FuncOp::register(&mut fresh);
    CustomCallOp::register(&mut fresh);
    ReturnOp::register(&mut fresh);
    let parsed = parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
    verify_module_fast(&fresh, parsed).unwrap();
}
