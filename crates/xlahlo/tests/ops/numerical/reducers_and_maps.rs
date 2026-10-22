use super::super::*;

#[test]
fn high_rank_reduction_preserves_free_axis_order_and_dynamic_flags() {
    let ctx = context();
    const RANK: usize = 32_768;
    let input = dynamic(
        &ctx,
        P::F32,
        &(0..RANK)
            .map(|axis| {
                if axis % 3 == 0 {
                    D::Bounded(1)
                } else {
                    D::Static(1)
                }
            })
            .collect::<Vec<_>>(),
    );
    let initial = s(&ctx, P::F32);
    let reducer = (vec![initial; 2], initial);
    for reduced in [
        (0..RANK)
            .step_by(2)
            .rev()
            .map(|axis| axis as i64)
            .collect::<Vec<_>>(),
        (0..RANK as i64).rev().collect(),
        vec![],
    ] {
        let output = inference::infer_reduce(&ctx, &[input, initial], &reduced, &reducer).unwrap();
        let reduced: std::collections::BTreeSet<_> = reduced.into_iter().collect();
        let expected = (0..RANK)
            .filter(|axis| !reduced.contains(&(*axis as i64)))
            .map(|axis| shape::dimension(&ctx, input, axis).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            (0..shape::rank(&ctx, output).unwrap())
                .map(|axis| shape::dimension(&ctx, output, axis).unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            shape::layout(&ctx, output).unwrap().unwrap().minor_to_major,
            (0..expected.len() as i64).rev().collect::<Vec<_>>()
        );
    }
}

#[test]
fn scalar_reduction_supports_partial_total_and_empty_axes_and_round_trips() {
    let mut ctx = context();
    let input = a(&ctx, P::F32, &[2, 3]);
    let initial = s(&ctx, P::F32);
    let reducer = (vec![initial; 2], initial);
    for (axes, expected) in [
        (vec![0], vec![3]),
        (vec![0, 1], vec![]),
        (vec![], vec![2, 3]),
    ] {
        let result = inference::infer_reduce(&ctx, &[input, initial], &axes, &reducer).unwrap();
        assert_eq!(dims(&ctx, result), expected);
        let args = values(&mut ctx, &[input, initial]);
        let op = ReduceOp::new(&mut ctx, result, args, Axes(axes.clone()), callees(1));
        op.verify(&ctx).unwrap();
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Reduce,
            VerifyOptions::default(),
            &[&reducer],
        )
        .unwrap();
        roundtrip(&mut ctx, Axes(axes));
    }
}

#[test]
fn reduction_rejects_malformed_axes_cardinality_and_reducer_contracts() {
    let ctx = context();
    let input = a(&ctx, P::F32, &[2, 3]);
    let initial = s(&ctx, P::F32);
    let reducer = (vec![initial; 2], initial);
    for (args, axes) in [
        (vec![input], vec![0]),
        (vec![input, initial], vec![0, 0]),
        (vec![input, initial], vec![2]),
        (vec![], vec![]),
        (vec![input, a(&ctx, P::F32, &[1])], vec![0]),
    ] {
        assert!(inference::infer_reduce(&ctx, &args, &axes, &reducer).is_err());
    }
    let wrong = signature(&ctx, &[P::F32, P::S32], P::F32);
    assert!(inference::infer_reduce(&ctx, &[input, initial], &[0], &wrong).is_err());
}

#[test]
fn reduction_checks_reducer_before_duplicate_dimensions_but_bounds_before_reducer() {
    let ctx = context();
    let input = a(&ctx, P::F32, &[2]);
    let initial = s(&ctx, P::F32);
    let invalid_reducer = (vec![initial], initial);
    error_contains(
        inference::infer_reduce(&ctx, &[input, initial], &[0, 0], &invalid_reducer),
        "reducer requires two parameters per input",
    );
    error_contains(
        inference::infer_reduce(&ctx, &[input, initial], &[2], &invalid_reducer),
        "reducing out-of-range dimension",
    );
}

#[test]
fn variadic_reduction_has_accumulators_first_and_heterogeneous_tuple_results() {
    let mut ctx = context();
    let f = s(&ctx, P::F32);
    let i = s(&ctx, P::S32);
    let reducer = (vec![f, i, f, i], shape::tuple(&ctx, vec![f, i]).unwrap());
    let args = [
        dynamic(&ctx, P::F32, &[D::Unbounded, D::Static(3)]),
        a(&ctx, P::S32, &[2, 3]),
        f,
        i,
    ];
    let result = inference::infer_reduce(&ctx, &args, &[0], &reducer).unwrap();
    assert!(shape::is_tuple(&ctx, result));
    for &child in shape::children(&ctx, result).iter() {
        assert_eq!(dims(&ctx, child), [3]);
        assert!(shape::layout(&ctx, child).unwrap().is_some());
    }
    let mut wrong = args;
    wrong[1] = a(&ctx, P::S32, &[4, 4]);
    assert!(inference::infer_reduce(&ctx, &wrong, &[0], &reducer).is_err());
    wrong = args;
    wrong.swap(2, 3);
    assert!(inference::infer_reduce(&ctx, &wrong, &[0], &reducer).is_err());
    let vals = values(&mut ctx, &args);
    let op = ReduceOp::new(&mut ctx, result, vals, Axes(vec![0]), callees(1));
    op.verify(&ctx).unwrap();
    verify_callees(
        &ctx,
        op.get_operation(),
        O::Reduce,
        VerifyOptions {
            supported_unbounded_dynamic_op: Some(|_, _| true),
            ..Default::default()
        },
        &[&reducer],
    )
    .unwrap();
}

#[test]
fn reduction_inference_preserves_unreduced_first_operand_bounds() {
    let ctx = context();
    let input = dynamic(&ctx, P::F32, &[D::Bounded(2), D::Static(3)]);
    let initial = s(&ctx, P::F32);
    let result =
        inference::infer_reduce(&ctx, &[input, initial], &[1], &(vec![initial; 2], initial))
            .unwrap();
    assert_eq!(shape::dimension(&ctx, result, 0).unwrap(), D::Bounded(2));
}

#[test]
fn map_supports_scalar_broadcasts_and_heterogeneous_equal_rank_inputs() {
    let mut ctx = context();
    let program = signature(&ctx, &[P::F32, P::F32], P::F32);
    let arguments = [a(&ctx, P::F32, &[2, 3]), s(&ctx, P::F32)];
    let output = inference::infer_map(&ctx, &arguments, &program, &[0, 1]).unwrap();
    assert_eq!(dims(&ctx, output), [2, 3]);
    error_contains(
        inference::infer_map(&ctx, &arguments, &program, &[1, 0]),
        "map requires monotonically increasing dimension numbers",
    );
    error_contains(
        inference::infer_map(&ctx, &arguments, &program, &[0]),
        "map cannot be applied to a subset of dimensions",
    );
    let wrong_arity = (vec![], program.1);
    error_contains(
        inference::infer_map(&ctx, &arguments, &wrong_arity, &[0]),
        "map cannot be applied to a subset of dimensions",
    );
    let wildcard = inference::infer_map(
        &ctx,
        &[
            dynamic(&ctx, P::F32, &[D::Unbounded]),
            a(&ctx, P::F32, &[3]),
        ],
        &program,
        &[0],
    )
    .unwrap();
    assert_eq!(shape::dimension(&ctx, wildcard, 0).unwrap(), D::Unbounded);
    assert!(
        inference::infer_map(
            &ctx,
            &[a(&ctx, P::F32, &[2]), a(&ctx, P::F32, &[3])],
            &program,
            &[0]
        )
        .is_err()
    );
    assert!(
        inference::infer_map(
            &ctx,
            &[
                dynamic(&ctx, P::F32, &[D::Bounded(3)]),
                a(&ctx, P::F32, &[3])
            ],
            &program,
            &[0]
        )
        .is_ok()
    );
    let heterogeneous = signature(&ctx, &[P::F32, P::S32], P::Pred);
    let args = [a(&ctx, P::F32, &[3]), a(&ctx, P::S32, &[3])];
    let result = inference::infer_map(&ctx, &args, &heterogeneous, &[0]).unwrap();
    assert_eq!(shape::element_type(&ctx, result).unwrap(), P::Pred);
    let vals = values(&mut ctx, &args);
    let op = MapOp::new(&mut ctx, result, vals, callees(1));
    op.verify(&ctx).unwrap();
    verify_callees(
        &ctx,
        op.get_operation(),
        O::Map,
        VerifyOptions::default(),
        &[&heterogeneous],
    )
    .unwrap();
    assert!(
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Map,
            VerifyOptions::default(),
            &[]
        )
        .is_err()
    );
}

#[test]
fn map_checks_result_scalar_before_parameter_scalars() {
    let ctx = context();
    let operand = a(&ctx, P::F32, &[2]);
    error_contains(
        inference::infer_map(&ctx, &[operand], &(vec![operand], operand), &[0]),
        "map computation must return a scalar",
    );
    error_contains(
        inference::infer_map(&ctx, &[operand], &(vec![], operand), &[0]),
        "map operand count differs from parameter count",
    );
    let scalar = s(&ctx, P::F32);
    let buffer_scalar = shape::buffer(&ctx, scalar).unwrap();
    for program in [(vec![scalar], buffer_scalar), (vec![buffer_scalar], scalar)] {
        let inferred = inference::infer_map(&ctx, &[operand], &program, &[0]).unwrap();
        assert_eq!(shape::dimension(&ctx, inferred, 0).unwrap(), D::Static(2));
    }
}

#[test]
fn resolved_reducer_precision_respects_verifier_options() {
    let mut ctx = context();
    let input = a(&ctx, P::F32, &[2]);
    let initial = s(&ctx, P::F32);
    let result = s(&ctx, P::F32);
    let args = values(&mut ctx, &[input, initial]);
    let op = ReduceOp::new(&mut ctx, result, args, Axes(vec![0]), callees(1));
    let reducer = signature(&ctx, &[P::F64, P::F64], P::F64);
    assert!(
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Reduce,
            VerifyOptions::default(),
            &[&reducer]
        )
        .is_err()
    );
    verify_callees(
        &ctx,
        op.get_operation(),
        O::Reduce,
        VerifyOptions {
            allow_mixed_precision: true,
            ..Default::default()
        },
        &[&reducer],
    )
    .unwrap();
}

#[test]
fn map_allows_floating_parameter_precision_and_scalar_broadcasts() {
    let ctx = context();
    let operands = [s(&ctx, P::F16), a(&ctx, P::BF16, &[2])];
    let mut program = signature(&ctx, &[P::F32, P::F64], P::S32);
    assert_eq!(
        inference::infer_map(&ctx, &operands, &program, &[0]).unwrap(),
        default_layout(&ctx, a(&ctx, P::S32, &[2]))
    );
    program.0[1] = s(&ctx, P::S16);
    assert!(inference::infer_map(&ctx, &operands, &program, &[0]).is_err());
}

#[test]
fn reducer_inference_allows_different_float_precisions_with_exact_accumulator_result() {
    let ctx = context();
    let mut program = signature(&ctx, &[P::F32, P::F16], P::F32);
    let operands = [a(&ctx, P::BF16, &[3]), s(&ctx, P::F64)];
    let result = inference::infer_reduce(&ctx, &operands, &[0], &program).unwrap();
    assert_eq!(shape::element_type(&ctx, result).unwrap(), P::F32);
    let result = inference::infer_reduce_window(
        &ctx,
        &operands,
        &super::windows::window(2, 1, 0, 0, 1, 1),
        &program,
    )
    .unwrap();
    assert_eq!(shape::element_type(&ctx, result).unwrap(), P::F64);
    assert_eq!(dims(&ctx, result), [2]);
    program.0[0] = s(&ctx, P::BF16);
    assert!(inference::infer_reduce(&ctx, &operands, &[0], &program).is_err());
    program.0[0] = s(&ctx, P::F32);
    program.0[1] = s(&ctx, P::S32);
    assert!(inference::infer_reduce(&ctx, &operands, &[0], &program).is_err());
}

#[test]
fn mixed_policy_relaxes_map_reduce_and_window_parameter_precision() {
    let mut ctx = context();
    let input = a(&ctx, P::F32, &[2]);
    let initial = s(&ctx, P::F32);
    for opcode in [O::Map, O::Reduce, O::ReduceWindow] {
        let result = match opcode {
            O::Map => a(&ctx, P::BF16, &[2]),
            O::Reduce => s(&ctx, P::BF16),
            _ => a(&ctx, P::BF16, &[1]),
        };
        let shapes = if opcode == O::Map {
            vec![input]
        } else {
            vec![input, initial]
        };
        let args = values(&mut ctx, &shapes);
        let op = match opcode {
            O::Map => MapOp::new(&mut ctx, result, args, callees(1)).get_operation(),
            O::Reduce => {
                ReduceOp::new(&mut ctx, result, args, Axes(vec![0]), callees(1)).get_operation()
            }
            _ => ReduceWindowOp::new(
                &mut ctx,
                result,
                args,
                super::windows::window(2, 1, 0, 0, 1, 1),
                callees(1),
            )
            .get_operation(),
        };
        let program = signature(
            &ctx,
            if opcode == O::Map {
                &[P::F64]
            } else {
                &[P::F64, P::F32]
            },
            P::F64,
        );
        assert!(verify_callees(&ctx, op, opcode, VerifyOptions::default(), &[&program]).is_err());
        for layout_sensitive in [false, true] {
            let options = VerifyOptions {
                layout_sensitive,
                allow_mixed_precision: true,
                ..Default::default()
            };
            instruction(&ctx, op, opcode, options).unwrap();
            verify_callees(&ctx, op, opcode, options, &[&program]).unwrap();
        }
    }
}
