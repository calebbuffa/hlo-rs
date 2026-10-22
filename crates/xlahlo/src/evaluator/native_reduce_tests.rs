use super::*;
use crate::{
    attributes::{Axes, Callees, Index},
    inference::{Signature, numerical::infer_reduce},
    ops::{
        control::{GetTupleElementOp, TupleOp, WhileOp},
        elementwise::{AddOp, MaximumOp, MultiplyOp},
        movement::CopyOp,
        numerical::ReduceOp,
    },
    shape::{Dimension, Layout},
};

#[derive(Clone, Copy)]
enum BinaryReducer {
    Add,
    Maximum,
    Ordered,
}

struct Fixture {
    module: Ptr<Operation>,
    root: Value,
    parameters: Vec<Value>,
}

fn binary_reducer(
    ctx: &mut Context,
    module: Ptr<Operation>,
    name: &'static str,
    scalar: TypeHandle,
    kind: BinaryReducer,
) -> Signature {
    let (_, block, parameters) = computation(ctx, module, name, &[scalar, scalar], scalar);
    let result = match kind {
        BinaryReducer::Add => {
            let operation = AddOp::new(ctx, scalar, parameters.clone()).get_operation();
            append(ctx, block, operation, "combine")
        }
        BinaryReducer::Maximum => {
            let operation = MaximumOp::new(ctx, scalar, parameters.clone()).get_operation();
            append(ctx, block, operation, "combine")
        }
        BinaryReducer::Ordered => {
            let ten = constant(
                ctx,
                block,
                Literal::from_bits(ctx, scalar, &[10]).unwrap(),
                "ten",
            );
            let scaled = MultiplyOp::new(ctx, scalar, vec![parameters[0], ten]);
            let scaled = append(ctx, block, scaled.get_operation(), "scaled");
            let operation = AddOp::new(ctx, scalar, vec![scaled, parameters[1]]);
            append(ctx, block, operation.get_operation(), "combine")
        }
    };
    finish(ctx, block, result);
    (vec![scalar, scalar], scalar)
}

fn fixture(
    ctx: &mut Context,
    input_shape: TypeHandle,
    dimensions: &[i64],
    element: ElementType,
    kind: BinaryReducer,
) -> Fixture {
    let module = module(ctx);
    let scalar = shape::scalar(ctx, element).unwrap();
    let signature = binary_reducer(ctx, module, "reducer", scalar, kind);
    let result = infer_reduce(ctx, &[input_shape, scalar], dimensions, &signature).unwrap();
    let (_, block, parameters) = computation(ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceOp::new(
        ctx,
        result,
        vec![parameters[0], parameters[1]],
        Axes(dimensions.to_vec()),
        Callees(vec!["reducer".into()]),
    );
    let root = append(ctx, block, operation.get_operation(), "reduce");
    finish(ctx, block, root);
    Fixture {
        module,
        root,
        parameters,
    }
}

fn evaluate_fixture(
    ctx: &Context,
    fixture: &Fixture,
    input: Literal,
    initial: Literal,
) -> EvaluatedValue {
    evaluate(
        ctx,
        fixture.module,
        fixture.root,
        &[
            (fixture.parameters[0], known(ctx, input)),
            (fixture.parameters[1], known(ctx, initial)),
        ],
        DependencyMode::Recursive,
    )
    .unwrap()
}

#[test]
fn sum_and_maximum_reduce_static_axes() {
    let mut ctx = context();
    let shape = shape::array(&ctx, ElementType::S32, vec![2, 3]).unwrap();
    let input = Literal::from_bits(&ctx, shape, &[1, 2, 3, 4, 5, 6]).unwrap();

    let sum = fixture(&mut ctx, shape, &[1], ElementType::S32, BinaryReducer::Add);
    let sum_result = evaluate_fixture(
        &ctx,
        &sum,
        input.clone(),
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[10]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(sum_result.bits(&ctx).unwrap(), [16, 25]);

    let maximum = fixture(
        &mut ctx,
        shape,
        &[0],
        ElementType::S32,
        BinaryReducer::Maximum,
    );
    let maximum_result = evaluate_fixture(
        &ctx,
        &maximum,
        input,
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(maximum_result.bits(&ctx).unwrap(), [4, 5, 6]);
}

#[test]
fn layoutless_reduce_result_uses_its_allocated_default_layout() {
    let mut ctx = context();
    let input_shape = shape::array(&ctx, ElementType::S32, vec![2, 3]).unwrap();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let module = module(&mut ctx);
    let signature = binary_reducer(&mut ctx, module, "reducer", scalar, BinaryReducer::Add);
    let inferred = infer_reduce(&ctx, &[input_shape, scalar], &[1], &signature).unwrap();
    let result = shape::without_layout(&ctx, inferred).unwrap();
    let (_, block, parameters) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceOp::new(
        &mut ctx,
        result,
        parameters.clone(),
        Axes(vec![1]),
        Callees(vec!["reducer".into()]),
    );
    let root = append(&ctx, block, operation.get_operation(), "reduce");
    finish(&mut ctx, block, root);

    let output = evaluate(
        &ctx,
        module,
        root,
        &[
            (
                parameters[0],
                known(
                    &ctx,
                    Literal::from_bits(&ctx, input_shape, &[1, 2, 3, 4, 5, 6]).unwrap(),
                ),
            ),
            (
                parameters[1],
                known(&ctx, Literal::from_bits(&ctx, scalar, &[0]).unwrap()),
            ),
        ],
        DependencyMode::Recursive,
    )
    .unwrap()
    .into_literal(&ctx)
    .unwrap();

    assert_eq!(output.bits(&ctx).unwrap(), [6, 15]);
    assert_eq!(
        shape::layout(&ctx, output.shape)
            .unwrap()
            .unwrap()
            .minor_to_major,
        [0]
    );
}

#[test]
fn layoutless_scalar_reduce_result_is_supported() {
    let mut ctx = context();
    let input_shape = shape::array(&ctx, ElementType::S32, vec![2]).unwrap();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let module = module(&mut ctx);
    let signature = binary_reducer(&mut ctx, module, "reducer", scalar, BinaryReducer::Add);
    let inferred = infer_reduce(&ctx, &[input_shape, scalar], &[0], &signature).unwrap();
    let result = shape::without_layout(&ctx, inferred).unwrap();
    let (_, block, parameters) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceOp::new(
        &mut ctx,
        result,
        parameters.clone(),
        Axes(vec![0]),
        Callees(vec!["reducer".into()]),
    );
    let root = append(&ctx, block, operation.get_operation(), "reduce");
    finish(&mut ctx, block, root);

    let output = evaluate(
        &ctx,
        module,
        root,
        &[
            (
                parameters[0],
                known(
                    &ctx,
                    Literal::from_bits(&ctx, input_shape, &[4, 5]).unwrap(),
                ),
            ),
            (
                parameters[1],
                known(&ctx, Literal::from_bits(&ctx, scalar, &[1]).unwrap()),
            ),
        ],
        DependencyMode::Recursive,
    )
    .unwrap()
    .into_literal(&ctx)
    .unwrap();

    assert_eq!(output.bits(&ctx).unwrap(), [10]);
    assert!(
        shape::layout(&ctx, output.shape)
            .unwrap()
            .unwrap()
            .minor_to_major
            .is_empty()
    );
}

#[test]
fn reduction_order_follows_input_layout_and_empty_dimension_list_is_identity() {
    let mut ctx = context();
    let row_major = shape::array(&ctx, ElementType::S32, vec![2, 3]).unwrap();
    let layout = Layout {
        minor_to_major: vec![0, 1],
        ..Layout::descending(2)
    };
    let column_major = shape::with_layout(&ctx, row_major, layout).unwrap();
    let input = Literal::from_bits(&ctx, column_major, &[1, 4, 2, 5, 3, 6]).unwrap();
    let ordered = fixture(
        &mut ctx,
        column_major,
        &[0, 1],
        ElementType::S32,
        BinaryReducer::Ordered,
    );
    let result = evaluate_fixture(
        &ctx,
        &ordered,
        input.clone(),
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(result.bits(&ctx).unwrap(), [142_536]);

    let identity = fixture(
        &mut ctx,
        column_major,
        &[],
        ElementType::S32,
        BinaryReducer::Add,
    );
    let identity_result = evaluate_fixture(
        &ctx,
        &identity,
        input,
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[7]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(identity_result.bits(&ctx).unwrap(), [8, 9, 10, 11, 12, 13]);

    let row_major_3d = shape::array(&ctx, ElementType::S32, vec![2, 2, 2]).unwrap();
    let non_dense_layout = Layout {
        minor_to_major: vec![2, 0, 1],
        ..Layout::descending(3)
    };
    let non_dense_shape = shape::with_layout(&ctx, row_major_3d, non_dense_layout).unwrap();
    let non_dense_input =
        Literal::from_bits(&ctx, non_dense_shape, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let non_dense = fixture(
        &mut ctx,
        non_dense_shape,
        &[0, 2],
        ElementType::S32,
        BinaryReducer::Ordered,
    );
    let non_dense_result = evaluate_fixture(
        &ctx,
        &non_dense,
        non_dense_input,
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(non_dense_result.bits(&ctx).unwrap(), [1234, 5678]);
}

#[test]
fn scalar_and_empty_reduction_domains_keep_the_initial_value_semantics() {
    let mut ctx = context();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let scalar_fixture = fixture(&mut ctx, scalar, &[], ElementType::S32, BinaryReducer::Add);
    let scalar_result = evaluate_fixture(
        &ctx,
        &scalar_fixture,
        Literal::from_bits(&ctx, scalar, &[9]).unwrap(),
        Literal::from_bits(&ctx, scalar, &[3]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(scalar_result.bits(&ctx).unwrap(), [12]);

    let empty_shape = shape::array(&ctx, ElementType::S32, vec![2, 0, 3]).unwrap();
    let empty_fixture = fixture(
        &mut ctx,
        empty_shape,
        &[1],
        ElementType::S32,
        BinaryReducer::Add,
    );
    let empty_result = evaluate_fixture(
        &ctx,
        &empty_fixture,
        Literal::from_bits(&ctx, empty_shape, &[]).unwrap(),
        Literal::from_bits(&ctx, scalar, &[7]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        &*shape::dimensions(&ctx, empty_result.shape).unwrap(),
        &[2, 3]
    );
    assert_eq!(empty_result.bits(&ctx).unwrap(), [7; 6]);
}

#[test]
fn variadic_tuple_reducer_keeps_accumulator_then_input_component_order() {
    let mut ctx = context();
    let module = module(&mut ctx);
    let f32 = shape::scalar(&ctx, ElementType::F32).unwrap();
    let s32 = shape::scalar(&ctx, ElementType::S32).unwrap();
    let f32_input = shape::array(&ctx, ElementType::F32, vec![2]).unwrap();
    let s32_input = shape::array(&ctx, ElementType::S32, vec![2]).unwrap();
    let reducer_result = shape::tuple(&ctx, vec![f32, s32]).unwrap();
    let signature = (vec![f32, s32, f32, s32], reducer_result);
    let (_, reducer_block, parameters) = computation(
        &mut ctx,
        module,
        "variadic_reducer",
        &signature.0,
        reducer_result,
    );
    let add_f32 = AddOp::new(&mut ctx, f32, vec![parameters[0], parameters[2]]);
    let add_f32 = append(&ctx, reducer_block, add_f32.get_operation(), "sum_f32");
    let add_s32 = AddOp::new(&mut ctx, s32, vec![parameters[1], parameters[3]]);
    let add_s32 = append(&ctx, reducer_block, add_s32.get_operation(), "sum_s32");
    let root = TupleOp::new(&mut ctx, reducer_result, vec![add_f32, add_s32]);
    let root = append(&ctx, reducer_block, root.get_operation(), "tuple");
    finish(&mut ctx, reducer_block, root);

    let types = [f32_input, s32_input, f32, s32];
    let inferred = infer_reduce(&ctx, &types, &[0], &signature).unwrap();
    let result = shape::tuple(
        &ctx,
        shape::children(&ctx, inferred)
            .iter()
            .map(|&shape| shape::without_layout(&ctx, shape).unwrap())
            .collect(),
    )
    .unwrap();
    let (_, block, operands) = computation(&mut ctx, module, "main", &types, result);
    let operation = ReduceOp::new(
        &mut ctx,
        result,
        operands.clone(),
        Axes(vec![0]),
        Callees(vec!["variadic_reducer".into()]),
    );
    let reduce = append(&ctx, block, operation.get_operation(), "reduce");
    let f32_output = shape::children(&ctx, result)[0];
    let s32_output = shape::children(&ctx, result)[1];
    let selected_f32 =
        GetTupleElementOp::new(&mut ctx, f32_output, vec![reduce], Index(0)).get_operation();
    let selected_f32 = append(&ctx, block, selected_f32, "selected_f32");
    let copied_f32 = CopyOp::new(&mut ctx, f32_output, vec![selected_f32]).get_operation();
    let copied_f32 = append(&ctx, block, copied_f32, "copied_f32");
    let selected_s32 =
        GetTupleElementOp::new(&mut ctx, s32_output, vec![reduce], Index(1)).get_operation();
    let selected_s32 = append(&ctx, block, selected_s32, "selected_s32");
    let copied_s32 = CopyOp::new(&mut ctx, s32_output, vec![selected_s32]).get_operation();
    let copied_s32 = append(&ctx, block, copied_s32, "copied_s32");
    let tuple = TupleOp::new(&mut ctx, result, vec![copied_f32, copied_s32]).get_operation();
    let root = append(&ctx, block, tuple, "selected_tuple");
    finish(&mut ctx, block, root);
    let substitutions = [
        (
            operands[0],
            known(
                &ctx,
                Literal::from_bits(
                    &ctx,
                    f32_input,
                    &[1.5f32.to_bits() as u128, 2.5f32.to_bits() as u128],
                )
                .unwrap(),
            ),
        ),
        (
            operands[1],
            known(&ctx, Literal::from_bits(&ctx, s32_input, &[1, 2]).unwrap()),
        ),
        (
            operands[2],
            known(
                &ctx,
                Literal::from_bits(&ctx, f32, &[0.5f32.to_bits() as u128]).unwrap(),
            ),
        ),
        (
            operands[3],
            known(&ctx, Literal::from_bits(&ctx, s32, &[10]).unwrap()),
        ),
    ];
    let result = evaluate(
        &ctx,
        module,
        reduce,
        &substitutions,
        DependencyMode::Recursive,
    )
    .unwrap()
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        result.tuple_literals[0].bits(&ctx).unwrap(),
        [4.5f32.to_bits() as u128]
    );
    assert_eq!(result.tuple_literals[1].bits(&ctx).unwrap(), [13]);

    let selected = evaluate(
        &ctx,
        module,
        root,
        &substitutions,
        DependencyMode::Recursive,
    )
    .unwrap()
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        selected.tuple_literals[0].bits(&ctx).unwrap(),
        [4.5f32.to_bits() as u128]
    );
    assert_eq!(selected.tuple_literals[1].bits(&ctx).unwrap(), [13]);

    let unknown_substitutions = [
        (
            operands[0],
            EvaluatedValue::unknown(&ctx, f32_input).unwrap(),
        ),
        substitutions[1].clone(),
        substitutions[2].clone(),
        substitutions[3].clone(),
    ];
    let projected_unknown = evaluate(
        &ctx,
        module,
        root,
        &unknown_substitutions,
        DependencyMode::Recursive,
    )
    .unwrap();
    assert!(matches!(
        projected_unknown.subvalue(&[0]).unwrap().state(),
        ValueState::Unknown
    ));
    assert!(matches!(
        projected_unknown.subvalue(&[1]).unwrap().state(),
        ValueState::Unknown
    ));
}

#[test]
fn odd_packed_subbyte_outputs_allocate_native_literal_storage() {
    let mut ctx = context();
    for (element, input_bits, expected) in [
        (ElementType::F4E2M1FN, vec![0; 6], vec![0, 0, 0]),
        (ElementType::S4, vec![1, 2, 3, 4, 5, 6], vec![3, 7, 11]),
    ] {
        let array = shape::array(&ctx, element, vec![3, 2]).unwrap();
        let packed_layout = Layout {
            minor_to_major: vec![1, 0],
            element_size_in_bits: 4,
            ..Default::default()
        };
        let input_shape = shape::with_layout(&ctx, array, packed_layout.clone()).unwrap();
        let module = module(&mut ctx);
        let scalar = shape::scalar(&ctx, element).unwrap();
        let signature = binary_reducer(&mut ctx, module, "reducer", scalar, BinaryReducer::Add);
        let inferred = infer_reduce(&ctx, &[input_shape, scalar], &[1], &signature).unwrap();
        let result = shape::with_layout(
            &ctx,
            inferred,
            Layout {
                minor_to_major: vec![0],
                ..packed_layout.clone()
            },
        )
        .unwrap();
        let (_, block, parameters) =
            computation(&mut ctx, module, "main", &[input_shape, scalar], result);
        let operation = ReduceOp::new(
            &mut ctx,
            result,
            vec![parameters[0], parameters[1]],
            Axes(vec![1]),
            Callees(vec!["reducer".into()]),
        );
        let root = append(&ctx, block, operation.get_operation(), "reduce");
        finish(&mut ctx, block, root);
        let fixture = Fixture {
            module,
            root,
            parameters,
        };
        let output = evaluate_fixture(
            &ctx,
            &fixture,
            Literal::from_bits(&ctx, input_shape, &input_bits).unwrap(),
            Literal::from_bits(&ctx, scalar, &[0]).unwrap(),
        );
        assert_eq!(
            shape::layout(&ctx, output.shape())
                .unwrap()
                .unwrap()
                .element_size_in_bits,
            4
        );
        let output = output.into_literal(&ctx).unwrap();
        assert_eq!(output.storage.len(), 3, "{element:?}");
        assert_eq!(output.bits(&ctx).unwrap(), expected, "{element:?}");
        assert_eq!(
            shape::layout(&ctx, output.shape)
                .unwrap()
                .unwrap()
                .element_size_in_bits,
            0
        );
    }
}

#[test]
fn bounded_dynamic_reduce_uses_active_input_extent_and_preserves_output_bound() {
    let mut ctx = context();
    let shape = shape::dynamic_array(
        &ctx,
        ElementType::S32,
        vec![
            Dimension::Static(2),
            Dimension::Bounded(4),
            Dimension::Static(2),
        ],
    )
    .unwrap();
    let fixture = fixture(&mut ctx, shape, &[2], ElementType::S32, BinaryReducer::Add);
    let storage = (1_i32..=16).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
    let input = Literal::from_dynamic_storage_bytes(&ctx, shape, storage, vec![2, 2, 2]).unwrap();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let output = evaluate_fixture(
        &ctx,
        &fixture,
        input,
        Literal::from_bits(&ctx, scalar, &[0]).unwrap(),
    )
    .into_literal(&ctx)
    .unwrap();

    assert_eq!(output.dynamic_sizes, [2, 2]);
    assert_eq!(
        crate::literal::semantics::geometry(&ctx, &output)
            .unwrap()
            .dimensions(),
        [2, 2]
    );
    assert_eq!(output.bits(&ctx).unwrap(), [3, 7, 0, 0, 19, 23, 0, 0]);
    assert_eq!(&*shape::dimensions(&ctx, output.shape).unwrap(), &[2, 4]);
}

#[test]
fn unknown_inputs_propagate_without_materializing_zero_data() {
    let mut ctx = context();
    let input_shape = shape::array(&ctx, ElementType::S32, vec![2]).unwrap();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let fixture = fixture(
        &mut ctx,
        input_shape,
        &[0],
        ElementType::S32,
        BinaryReducer::Add,
    );
    let result = evaluate(
        &ctx,
        fixture.module,
        fixture.root,
        &[(
            fixture.parameters[1],
            known(&ctx, Literal::from_bits(&ctx, scalar, &[0]).unwrap()),
        )],
        DependencyMode::Recursive,
    )
    .unwrap();
    assert!(matches!(result.state(), ValueState::Unknown));
}

#[test]
fn reducer_while_obeys_the_request_loop_iteration_budget() {
    let mut ctx = context();
    let module = module(&mut ctx);
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let predicate = shape::scalar(&ctx, ElementType::Pred).unwrap();

    let (_, condition_block, _) = computation(&mut ctx, module, "condition", &[scalar], predicate);
    let true_literal = Literal::from_bits(&ctx, predicate, &[1]).unwrap();
    let always_true = constant(&mut ctx, condition_block, true_literal, "always_true");
    finish(&mut ctx, condition_block, always_true);
    let (_, body_block, body_parameters) = computation(&mut ctx, module, "body", &[scalar], scalar);
    finish(&mut ctx, body_block, body_parameters[0]);

    let (_, reducer_block, reducer_parameters) =
        computation(&mut ctx, module, "reducer", &[scalar, scalar], scalar);
    let loop_value = WhileOp::new(
        &mut ctx,
        scalar,
        vec![reducer_parameters[0]],
        Callees(vec!["body".into(), "condition".into()]),
    );
    let loop_value = append(&ctx, reducer_block, loop_value.get_operation(), "loop");
    finish(&mut ctx, reducer_block, loop_value);

    let input_shape = shape::array(&ctx, ElementType::S32, vec![1]).unwrap();
    let signature = (vec![scalar, scalar], scalar);
    let result = infer_reduce(&ctx, &[input_shape, scalar], &[0], &signature).unwrap();
    let (_, block, operands) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceOp::new(
        &mut ctx,
        result,
        operands.clone(),
        Axes(vec![0]),
        Callees(vec!["reducer".into()]),
    );
    let root = append(&ctx, block, operation.get_operation(), "reduce");
    finish(&mut ctx, block, root);
    let substitutions = [
        (
            operands[0],
            known(&ctx, Literal::from_bits(&ctx, input_shape, &[7]).unwrap()),
        ),
        (
            operands[1],
            known(&ctx, Literal::from_bits(&ctx, scalar, &[0]).unwrap()),
        ),
    ];
    let error = HloEvaluator::new(&ctx, module)
        .unwrap()
        .with_max_loop_iterations(0)
        .evaluate(Request::Instruction {
            value: root,
            substitutions: &substitutions,
            dependency_mode: DependencyMode::Recursive,
        })
        .unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<EvaluationError>(),
        Some(EvaluationError::LoopIterationLimit { limit: 0 })
    ));
}
