use super::*;
use crate::{
    attributes::{Callees, Index, Window, WindowDimension},
    inference::{Signature, numerical::infer_reduce_window},
    ops::{
        control::{GetTupleElementOp, TupleOp, WhileOp},
        elementwise::{AddOp, MaximumOp, MultiplyOp, OrOp},
        movement::CopyOp,
        numerical::ReduceWindowOp,
    },
    shape::{Dimension, Layout},
};

#[derive(Clone, Copy)]
enum BinaryReducer {
    Add,
    Maximum,
}

struct Fixture {
    module: Ptr<Operation>,
    root: Value,
    parameters: Vec<Value>,
}

fn dimension(
    size: i64,
    stride: i64,
    low: i64,
    high: i64,
    base: i64,
    dilation: i64,
    reversed: bool,
) -> WindowDimension {
    WindowDimension {
        size,
        stride,
        padding_low: low,
        padding_high: high,
        base_dilation: base,
        window_dilation: dilation,
        window_reversal: reversed,
    }
}

fn binary_reducer(
    ctx: &mut Context,
    module: Ptr<Operation>,
    name: &'static str,
    element: ElementType,
    kind: BinaryReducer,
) -> Signature {
    let scalar = shape::scalar(ctx, element).unwrap();
    let (_, block, parameters) = computation(ctx, module, name, &[scalar, scalar], scalar);
    let operation = match kind {
        BinaryReducer::Add => AddOp::new(ctx, scalar, parameters.clone()).get_operation(),
        BinaryReducer::Maximum => MaximumOp::new(ctx, scalar, parameters.clone()).get_operation(),
    };
    let root = append(ctx, block, operation, "combine");
    finish(ctx, block, root);
    (vec![scalar, scalar], scalar)
}

fn fixture(
    ctx: &mut Context,
    input_shape: TypeHandle,
    window: Window,
    element: ElementType,
    reducer_kind: BinaryReducer,
) -> Fixture {
    let module = module(ctx);
    let signature = binary_reducer(ctx, module, "reducer", element, reducer_kind);
    let scalar = shape::scalar(ctx, element).unwrap();
    let result = infer_reduce_window(ctx, &[input_shape, scalar], &window, &signature).unwrap();
    let (_, block, parameters) = computation(ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceWindowOp::new(
        ctx,
        result,
        vec![parameters[0], parameters[1]],
        window,
        Callees(vec!["reducer".into()]),
    );
    let root = append(ctx, block, operation.get_operation(), "reduce_window");
    finish(ctx, block, root);
    Fixture {
        module,
        root,
        parameters,
    }
}

fn evaluate_fixture(ctx: &Context, fixture: &Fixture, input: Literal, initial: Literal) -> Literal {
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
    .into_literal(ctx)
    .unwrap()
}

#[test]
fn ordinary_sum_and_maximum_windows_evaluate_through_the_reducer_computation() {
    let mut ctx = context();
    let input_shape = shape::array(&ctx, ElementType::S32, vec![4]).unwrap();
    let input = Literal::from_bits(&ctx, input_shape, &[1, 3, 2, 4]).unwrap();
    let sum = fixture(
        &mut ctx,
        input_shape,
        Window(vec![dimension(3, 1, 0, 0, 1, 1, false)]),
        ElementType::S32,
        BinaryReducer::Add,
    );
    assert_eq!(
        evaluate_fixture(
            &ctx,
            &sum,
            input.clone(),
            Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap()
        )
        .bits(&ctx)
        .unwrap(),
        [6, 9]
    );

    let maximum = fixture(
        &mut ctx,
        input_shape,
        Window(vec![dimension(2, 1, 0, 0, 1, 1, false)]),
        ElementType::S32,
        BinaryReducer::Maximum,
    );
    assert_eq!(
        evaluate_fixture(
            &ctx,
            &maximum,
            input,
            Literal::from_bits(
                &ctx,
                shape::scalar(&ctx, ElementType::S32).unwrap(),
                &[i32::MIN as u32 as u128]
            )
            .unwrap()
        )
        .bits(&ctx)
        .unwrap(),
        [3, 3, 4]
    );
}

#[test]
fn odd_packed_layout_outputs_use_native_literal_bytes_per_element() {
    let mut ctx = context();
    for (element, packed_bits, input_bits) in [
        (ElementType::F4E2M1FN, 4, vec![1, 2, 3]),
        (ElementType::S4, 4, vec![1, 2, 3]),
    ] {
        let array = shape::array(&ctx, element, vec![3]).unwrap();
        let packed_layout = Layout {
            minor_to_major: vec![0],
            element_size_in_bits: packed_bits,
            ..Default::default()
        };
        let input_shape = shape::with_layout(&ctx, array, packed_layout.clone()).unwrap();
        let module = module(&mut ctx);
        let signature = binary_reducer(&mut ctx, module, "reducer", element, BinaryReducer::Add);
        let scalar = shape::scalar(&ctx, element).unwrap();
        let inferred = infer_reduce_window(
            &ctx,
            &[input_shape, scalar],
            &Window(vec![dimension(1, 1, 0, 0, 1, 1, false)]),
            &signature,
        )
        .unwrap();
        let result = shape::with_layout(&ctx, inferred, packed_layout).unwrap();
        let (_, block, parameters) =
            computation(&mut ctx, module, "main", &[input_shape, scalar], result);
        let operation = ReduceWindowOp::new(
            &mut ctx,
            result,
            vec![parameters[0], parameters[1]],
            Window(vec![dimension(1, 1, 0, 0, 1, 1, false)]),
            Callees(vec!["reducer".into()]),
        );
        let root = append(&ctx, block, operation.get_operation(), "reduce_window");
        finish(&mut ctx, block, root);
        let fixture = Fixture {
            module,
            root,
            parameters,
        };
        let output_layout = shape::layout(&ctx, result).unwrap().unwrap();
        assert_eq!(output_layout.element_size_in_bits, packed_bits);
        let output = evaluate_fixture(
            &ctx,
            &fixture,
            Literal::from_bits(&ctx, input_shape, &input_bits).unwrap(),
            Literal::from_bits(&ctx, shape::scalar(&ctx, element).unwrap(), &[0]).unwrap(),
        );
        assert_eq!(output.storage.len(), 3, "{element:?}");
        assert_eq!(output.bits(&ctx).unwrap(), input_bits, "{element:?}");
    }
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
        vec![reducer_parameters[1]],
        Callees(vec!["body".into(), "condition".into()]),
    );
    let loop_value = append(&ctx, reducer_block, loop_value.get_operation(), "loop");
    finish(&mut ctx, reducer_block, loop_value);

    let input_shape = shape::array(&ctx, ElementType::S32, vec![1]).unwrap();
    let window = Window(vec![dimension(1, 1, 0, 0, 1, 1, false)]);
    let result = infer_reduce_window(
        &ctx,
        &[input_shape, scalar],
        &window,
        &(vec![scalar, scalar], scalar),
    )
    .unwrap();
    let (_, block, parameters) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceWindowOp::new(
        &mut ctx,
        result,
        vec![parameters[0], parameters[1]],
        window,
        Callees(vec!["reducer".into()]),
    );
    let root = append(&ctx, block, operation.get_operation(), "reduce_window");
    finish(&mut ctx, block, root);

    let substitutions = [
        (
            parameters[0],
            known(&ctx, Literal::from_bits(&ctx, input_shape, &[7]).unwrap()),
        ),
        (
            parameters[1],
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

#[test]
fn asymmetric_padding_and_non_row_major_input_expose_size_one_axis_shifts() {
    let mut ctx = context();
    let row_major = shape::array(&ctx, ElementType::S32, vec![2, 3]).unwrap();
    let input_shape = shape::with_layout(
        &ctx,
        row_major,
        Layout {
            minor_to_major: vec![0, 1],
            memory_space: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let logical_input = Literal::from_bits(&ctx, row_major, &[1, 2, 3, 4, 5, 6]).unwrap();
    let input = literal_semantics::relayout(&ctx, &logical_input, input_shape).unwrap();
    let window = Window(vec![
        dimension(1, 1, 1, -1, 1, 1, false),
        dimension(5, 1, 2, 2, 1, 1, false),
    ]);
    let fixture = fixture(
        &mut ctx,
        input_shape,
        window,
        ElementType::S32,
        BinaryReducer::Add,
    );
    let result = evaluate_fixture(
        &ctx,
        &fixture,
        input,
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap(),
    );
    assert_eq!(&*shape::dimensions(&ctx, result.shape).unwrap(), &[2, 3]);
    assert_eq!(result.bits(&ctx).unwrap(), [0, 0, 0, 6, 6, 6]);
}

#[test]
fn chained_reduce_windows_consume_the_previous_live_result() {
    let mut ctx = context();
    let module = module(&mut ctx);
    let input_shape = shape::array(&ctx, ElementType::S32, vec![5]).unwrap();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let signature = binary_reducer(
        &mut ctx,
        module,
        "reducer",
        ElementType::S32,
        BinaryReducer::Add,
    );
    let first_window = Window(vec![dimension(2, 1, 0, 0, 1, 1, false)]);
    let first_shape =
        infer_reduce_window(&ctx, &[input_shape, scalar], &first_window, &signature).unwrap();
    let second_window = Window(vec![dimension(2, 1, 0, 0, 1, 1, false)]);
    let result =
        infer_reduce_window(&ctx, &[first_shape, scalar], &second_window, &signature).unwrap();
    let (_, block, parameters) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let first = ReduceWindowOp::new(
        &mut ctx,
        first_shape,
        vec![parameters[0], parameters[1]],
        first_window,
        Callees(vec!["reducer".into()]),
    );
    let first = append(&ctx, block, first.get_operation(), "first");
    let second = ReduceWindowOp::new(
        &mut ctx,
        result,
        vec![first, parameters[1]],
        second_window,
        Callees(vec!["reducer".into()]),
    );
    let root = append(&ctx, block, second.get_operation(), "second");
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
                    Literal::from_bits(&ctx, input_shape, &[1, 2, 3, 4, 5]).unwrap(),
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
    assert_eq!(output.bits(&ctx).unwrap(), [8, 12, 16]);
}

#[test]
fn variadic_tuple_reducer_preserves_accumulator_then_input_order() {
    let mut ctx = context();
    let module = module(&mut ctx);
    let f32_scalar = shape::scalar(&ctx, ElementType::F32).unwrap();
    let s32_scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let reducer_result = shape::tuple(&ctx, vec![f32_scalar, s32_scalar]).unwrap();
    let reducer_signature = (
        vec![f32_scalar, s32_scalar, f32_scalar, s32_scalar],
        reducer_result,
    );
    let (_, reducer_block, reducer_parameters) = computation(
        &mut ctx,
        module,
        "reducer",
        &reducer_signature.0,
        reducer_result,
    );
    let sum_f32 = AddOp::new(
        &mut ctx,
        f32_scalar,
        vec![reducer_parameters[0], reducer_parameters[2]],
    );
    let sum_f32 = append(&ctx, reducer_block, sum_f32.get_operation(), "sum_f32");
    let sum_s32 = AddOp::new(
        &mut ctx,
        s32_scalar,
        vec![reducer_parameters[1], reducer_parameters[3]],
    );
    let sum_s32 = append(&ctx, reducer_block, sum_s32.get_operation(), "sum_s32");
    let reducer_root = TupleOp::new(&mut ctx, reducer_result, vec![sum_f32, sum_s32]);
    let reducer_root = append(&ctx, reducer_block, reducer_root.get_operation(), "result");
    finish(&mut ctx, reducer_block, reducer_root);

    let f32_input = shape::array(&ctx, ElementType::F32, vec![3]).unwrap();
    let s32_input = shape::array(&ctx, ElementType::S32, vec![3]).unwrap();
    let window = Window(vec![dimension(2, 1, 0, 0, 1, 1, false)]);
    let result = infer_reduce_window(
        &ctx,
        &[f32_input, s32_input, f32_scalar, s32_scalar],
        &window,
        &reducer_signature,
    )
    .unwrap();
    let (_, block, parameters) = computation(
        &mut ctx,
        module,
        "main",
        &[f32_input, s32_input, f32_scalar, s32_scalar],
        result,
    );
    let operation = ReduceWindowOp::new(
        &mut ctx,
        result,
        parameters.clone(),
        window,
        Callees(vec!["reducer".into()]),
    );
    let reduce_window = append(&ctx, block, operation.get_operation(), "reduce_window");
    let f32_output = shape::children(&ctx, result)[0];
    let s32_output = shape::children(&ctx, result)[1];
    let selected_f32 =
        GetTupleElementOp::new(&mut ctx, f32_output, vec![reduce_window], Index(0)).get_operation();
    let selected_f32 = append(&ctx, block, selected_f32, "selected_f32");
    let copied_f32 = CopyOp::new(&mut ctx, f32_output, vec![selected_f32]).get_operation();
    let copied_f32 = append(&ctx, block, copied_f32, "copied_f32");
    let selected_s32 =
        GetTupleElementOp::new(&mut ctx, s32_output, vec![reduce_window], Index(1)).get_operation();
    let selected_s32 = append(&ctx, block, selected_s32, "selected_s32");
    let copied_s32 = CopyOp::new(&mut ctx, s32_output, vec![selected_s32]).get_operation();
    let copied_s32 = append(&ctx, block, copied_s32, "copied_s32");
    let tuple = TupleOp::new(&mut ctx, result, vec![copied_f32, copied_s32]).get_operation();
    let root = append(&ctx, block, tuple, "tuple");
    finish(&mut ctx, block, root);
    let substitutions = [
        (
            parameters[0],
            known(
                &ctx,
                Literal::from_bits(&ctx, f32_input, &[0x3f80_0000, 0x4000_0000, 0x4040_0000])
                    .unwrap(),
            ),
        ),
        (
            parameters[1],
            known(
                &ctx,
                Literal::from_bits(&ctx, s32_input, &[10, 20, 30]).unwrap(),
            ),
        ),
        (
            parameters[2],
            known(&ctx, Literal::from_bits(&ctx, f32_scalar, &[0]).unwrap()),
        ),
        (
            parameters[3],
            known(&ctx, Literal::from_bits(&ctx, s32_scalar, &[0]).unwrap()),
        ),
    ];
    let output = evaluate(
        &ctx,
        module,
        reduce_window,
        &substitutions,
        DependencyMode::Recursive,
    )
    .unwrap()
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        output.tuple_literals[0].bits(&ctx).unwrap(),
        [0x4040_0000, 0x40a0_0000]
    );
    assert_eq!(output.tuple_literals[1].bits(&ctx).unwrap(), [30, 50]);

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
        [0x4040_0000, 0x40a0_0000]
    );
    assert_eq!(selected.tuple_literals[1].bits(&ctx).unwrap(), [30, 50]);

    let unknown_substitutions = [
        (
            parameters[0],
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
fn zero_extent_and_scalar_windows_have_no_phantom_reducer_invocations() {
    let mut ctx = context();
    let empty_shape = shape::array(&ctx, ElementType::S32, vec![0]).unwrap();
    let empty_fixture = fixture(
        &mut ctx,
        empty_shape,
        Window(vec![dimension(1, 1, 0, 0, 1, 1, false)]),
        ElementType::S32,
        BinaryReducer::Add,
    );
    let empty = evaluate_fixture(
        &ctx,
        &empty_fixture,
        Literal::from_bits(&ctx, empty_shape, &[]).unwrap(),
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap(),
    );
    assert!(empty.storage.is_empty());

    let padded_empty_fixture = fixture(
        &mut ctx,
        empty_shape,
        Window(vec![dimension(1, 1, 2, 2, 1, 1, false)]),
        ElementType::S32,
        BinaryReducer::Add,
    );
    let padded_empty = evaluate_fixture(
        &ctx,
        &padded_empty_fixture,
        Literal::from_bits(&ctx, empty_shape, &[]).unwrap(),
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[7]).unwrap(),
    );
    assert_eq!(padded_empty.bits(&ctx).unwrap(), [7, 7, 7, 7]);

    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let scalar_fixture = fixture(
        &mut ctx,
        scalar,
        Window(vec![]),
        ElementType::S32,
        BinaryReducer::Add,
    );
    let output = evaluate_fixture(
        &ctx,
        &scalar_fixture,
        Literal::from_bits(&ctx, scalar, &[7]).unwrap(),
        Literal::from_bits(&ctx, scalar, &[5]).unwrap(),
    );
    assert_eq!(output.bits(&ctx).unwrap(), [12]);
}

#[test]
fn base_dilation_matches_and_window_reversal_is_ignored_by_pinned_evaluator() {
    let mut ctx = context();
    let input_shape = shape::array(&ctx, ElementType::S32, vec![3]).unwrap();
    let dilated_fixture = fixture(
        &mut ctx,
        input_shape,
        Window(vec![dimension(2, 2, 0, 0, 2, 2, false)]),
        ElementType::S32,
        BinaryReducer::Add,
    );
    let dilated = evaluate_fixture(
        &ctx,
        &dilated_fixture,
        Literal::from_bits(&ctx, input_shape, &[1, 2, 3]).unwrap(),
        Literal::from_bits(&ctx, shape::scalar(&ctx, ElementType::S32).unwrap(), &[0]).unwrap(),
    );
    assert_eq!(dilated.bits(&ctx).unwrap(), [3, 5]);

    let module = module(&mut ctx);
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let (_, reducer_block, reducer_parameters) = computation(
        &mut ctx,
        module,
        "ordered_reducer",
        &[scalar, scalar],
        scalar,
    );
    let ten_literal = Literal::from_bits(&ctx, scalar, &[10]).unwrap();
    let ten = constant(&mut ctx, reducer_block, ten_literal, "ten");
    let scaled = MultiplyOp::new(&mut ctx, scalar, vec![reducer_parameters[0], ten]);
    let scaled = append(&ctx, reducer_block, scaled.get_operation(), "scaled");
    let appended = AddOp::new(&mut ctx, scalar, vec![scaled, reducer_parameters[1]]);
    let reducer_root = append(&ctx, reducer_block, appended.get_operation(), "appended");
    finish(&mut ctx, reducer_block, reducer_root);
    let window = Window(vec![dimension(3, 1, 0, 0, 1, 1, true)]);
    let result = infer_reduce_window(
        &ctx,
        &[input_shape, scalar],
        &window,
        &(vec![scalar, scalar], scalar),
    )
    .unwrap();
    let (_, block, parameters) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceWindowOp::new(
        &mut ctx,
        result,
        vec![parameters[0], parameters[1]],
        window,
        Callees(vec!["ordered_reducer".into()]),
    );
    let root = append(&ctx, block, operation.get_operation(), "reduce_window");
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
                    Literal::from_bits(&ctx, input_shape, &[1, 2, 3]).unwrap(),
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
    assert_eq!(output.bits(&ctx).unwrap(), [123]);
}

#[test]
fn large_dilation_congruences_seek_directly_and_detect_no_solution() {
    fn evaluate_geometry(
        base_dilation: i64,
        window_size: i64,
        window_dilation: i64,
        input: [u128; 2],
        initial: u128,
    ) -> u128 {
        let mut ctx = context();
        let input_shape = shape::array(&ctx, ElementType::S32, vec![2]).unwrap();
        let effective_window = (window_size - 1) * window_dilation + 1;
        let padded_high = effective_window - base_dilation - 2;
        let fixture = fixture(
            &mut ctx,
            input_shape,
            Window(vec![dimension(
                window_size,
                1,
                1,
                padded_high,
                base_dilation,
                window_dilation,
                false,
            )]),
            ElementType::S32,
            BinaryReducer::Add,
        );
        let input = Literal::from_bits(&ctx, input_shape, &input).unwrap();
        let initial_shape = shape::scalar(&ctx, ElementType::S32).unwrap();
        let initial = Literal::from_bits(&ctx, initial_shape, &[initial]).unwrap();
        evaluate_fixture(&ctx, &fixture, input, initial)
            .bits(&ctx)
            .unwrap()[0]
    }

    let coprime_base = 1_000_000_007;
    assert_eq!(
        evaluate_geometry(coprime_base, coprime_base, 2, [10, 42], 0,),
        42
    );

    let even_base = 1_000_000_006;
    assert_eq!(
        evaluate_geometry(even_base, even_base + 1, even_base + 2, [10, 42], 7,),
        7
    );
}

#[test]
fn bounded_dynamic_reduce_window_uses_runtime_geometry_and_preserves_bound() {
    let mut ctx = context();
    let input_shape = shape::dynamic_array(
        &ctx,
        ElementType::S32,
        vec![Dimension::Static(2), Dimension::Bounded(4)],
    )
    .unwrap();
    let window = Window(vec![
        dimension(1, 1, 0, 0, 1, 1, false),
        dimension(2, 1, 0, 0, 1, 1, false),
    ]);
    let fixture = fixture(
        &mut ctx,
        input_shape,
        window,
        ElementType::S32,
        BinaryReducer::Add,
    );
    let input = Literal::from_dynamic_storage_bytes(
        &ctx,
        input_shape,
        (1_i32..=8).flat_map(i32::to_le_bytes).collect(),
        vec![2, 2],
    )
    .unwrap();
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let output = evaluate_fixture(
        &ctx,
        &fixture,
        input,
        Literal::from_bits(&ctx, scalar, &[0]).unwrap(),
    );
    assert_eq!(output.dynamic_sizes, [2, 1]);
    assert_eq!(
        crate::literal::semantics::geometry(&ctx, &output)
            .unwrap()
            .dimensions(),
        [2, 1]
    );
    assert_eq!(output.bits(&ctx).unwrap(), [3, 0, 0, 11, 0, 0]);
    assert_eq!(&*shape::dimensions(&ctx, output.shape).unwrap(), &[2, 3]);
}

#[test]
fn larger_predicate_identity_window_preserves_each_element() {
    let mut ctx = context();
    let module = module(&mut ctx);
    let input_shape = shape::array(&ctx, ElementType::Pred, vec![4096]).unwrap();
    let scalar = shape::scalar(&ctx, ElementType::Pred).unwrap();
    let signature = (vec![scalar, scalar], scalar);
    let (_, reducer_block, reducer_parameters) =
        computation(&mut ctx, module, "predicate_or", &[scalar, scalar], scalar);
    let combine = OrOp::new(&mut ctx, scalar, reducer_parameters.clone()).get_operation();
    let reducer_root = append(&ctx, reducer_block, combine, "combine");
    finish(&mut ctx, reducer_block, reducer_root);

    let window = Window(vec![dimension(1, 1, 0, 0, 1, 1, false)]);
    let result = infer_reduce_window(&ctx, &[input_shape, scalar], &window, &signature).unwrap();
    let (_, block, parameters) =
        computation(&mut ctx, module, "main", &[input_shape, scalar], result);
    let operation = ReduceWindowOp::new(
        &mut ctx,
        result,
        vec![parameters[0], parameters[1]],
        window,
        Callees(vec!["predicate_or".into()]),
    );
    let root = append(&ctx, block, operation.get_operation(), "reduce_window");
    finish(&mut ctx, block, root);

    let input_bits = (0..4096)
        .map(|index| u128::from(index % 2 == 0))
        .collect::<Vec<_>>();
    let substitutions = [
        (
            parameters[0],
            known(
                &ctx,
                Literal::from_bits(&ctx, input_shape, &input_bits).unwrap(),
            ),
        ),
        (
            parameters[1],
            known(&ctx, Literal::from_bits(&ctx, scalar, &[0]).unwrap()),
        ),
    ];
    let result = evaluate(
        &ctx,
        module,
        root,
        &substitutions,
        DependencyMode::Recursive,
    )
    .unwrap()
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(result.bits(&ctx).unwrap(), input_bits);
}

#[test]
fn unknown_inputs_propagate_and_malformed_window_metadata_is_typed_as_an_error() {
    let mut ctx = context();
    let input_shape = shape::array(&ctx, ElementType::S32, vec![2]).unwrap();
    let fixture = fixture(
        &mut ctx,
        input_shape,
        Window(vec![dimension(1, 1, 0, 0, 1, 1, false)]),
        ElementType::S32,
        BinaryReducer::Add,
    );
    let initial = shape::scalar(&ctx, ElementType::S32).unwrap();
    let partial = evaluate(
        &ctx,
        fixture.module,
        fixture.root,
        &[(
            fixture.parameters[1],
            known(&ctx, Literal::from_bits(&ctx, initial, &[0]).unwrap()),
        )],
        DependencyMode::Recursive,
    )
    .unwrap();
    assert!(matches!(partial.state(), ValueState::Unknown));

    let module = module(&mut ctx);
    let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
    let signature = binary_reducer(
        &mut ctx,
        module,
        "reducer",
        ElementType::S32,
        BinaryReducer::Add,
    );
    let (_, block, parameters) = computation(
        &mut ctx,
        module,
        "main",
        &[input_shape, scalar],
        input_shape,
    );
    let malformed = ReduceWindowOp::new(
        &mut ctx,
        input_shape,
        vec![parameters[0], parameters[1]],
        Window(vec![dimension(0, 1, 0, 0, 1, 1, false)]),
        Callees(vec!["reducer".into()]),
    );
    let root = append(&ctx, block, malformed.get_operation(), "malformed");
    finish(&mut ctx, block, root);
    assert!(
        infer_reduce_window(
            &ctx,
            &[input_shape, scalar],
            &Window(vec![dimension(0, 1, 0, 0, 1, 1, false)]),
            &signature,
        )
        .is_err()
    );
    let error = HloEvaluator::new(&ctx, module)
        .unwrap()
        .evaluate(Request::Instruction {
            value: root,
            substitutions: &[],
            dependency_mode: DependencyMode::Recursive,
        })
        .unwrap_err();
    assert!(error.err.downcast_ref::<crate::Diagnostic>().is_some());
}
