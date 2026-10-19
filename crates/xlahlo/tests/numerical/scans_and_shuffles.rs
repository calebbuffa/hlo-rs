use super::super::*;
use super::windows::window;

#[test]
fn select_and_scatter_checks_callees_before_source_geometry() {
    let ctx = context();
    let scalar = s(&ctx, P::F32);
    let program = (vec![scalar; 2], scalar);
    error_contains(
        inference::infer_select_and_scatter(
            &ctx,
            a(&ctx, P::F32, &[5]),
            a(&ctx, P::F32, &[99]),
            scalar,
            &window(2, 1, 0, 0, 1, 1),
            &program,
            &program,
        ),
        "select-and-scatter select must return scalar predicate",
    );
}

#[test]
fn select_and_scatter_checks_predicate_result_before_select_parameters() {
    let ctx = context();
    let select = signature(&ctx, &[P::S32, P::S32], P::F32);
    let scatter = signature(&ctx, &[P::F32, P::F32], P::F32);
    error_contains(
        inference::infer_select_and_scatter(
            &ctx,
            a(&ctx, P::F32, &[4]),
            a(&ctx, P::F32, &[4]),
            s(&ctx, P::F32),
            &Window(vec![]),
            &select,
            &scatter,
        ),
        "select-and-scatter select must return scalar predicate",
    );
}

#[test]
fn scan_checks_sliced_parameters_carries_outputs_and_layout_policy() {
    let mut ctx = context();
    let slice = a(&ctx, P::F32, &[3]);
    let input = a(&ctx, P::F32, &[4, 3]);
    let root = Shape::tuple(&ctx, vec![slice, slice]).unwrap();
    let body = (vec![slice, slice], root);
    let result = inference::infer_scan(&ctx, &[input, slice], &body, 1, 0, None).unwrap();
    assert_eq!(dims(&ctx, Shape::children(&ctx, result)[0]), [4, 3]);
    assert!(
        Shape::layout(&ctx, Shape::children(&ctx, result)[0])
            .unwrap()
            .is_some()
    );
    let vals = values(&mut ctx, &[input, slice]);
    let op = ScanOp::new(
        &mut ctx,
        result,
        vals,
        Axes(vec![0]),
        NumericalInteger(1),
        TriState::False,
        callees(1),
    );
    op.verify(&ctx).unwrap();
    verify_callees(
        &ctx,
        op.get_operation(),
        O::Scan,
        VerifyOptions::default(),
        &[&body],
    )
    .unwrap();
    let strict = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };
    assert!(instruction(&ctx, op.get_operation(), O::Scan, strict).is_err());
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_is_associative"), TriState::True);
    instruction(&ctx, op.get_operation(), O::Scan, strict).unwrap();
    verify_callees(&ctx, op.get_operation(), O::Scan, strict, &[&body]).unwrap();
    for (carries, axis) in [(3, 0), (-1, 0), (1, 2)] {
        assert!(inference::infer_scan(&ctx, &[input, slice], &body, carries, axis, None).is_err());
    }
    let wrong = (vec![a(&ctx, P::F32, &[2]), slice], root);
    assert!(
        inference::verify_scan(&ctx, &[input, slice], result, &wrong, 1, 0, false, false).is_err()
    );
    let wrong_result = Shape::tuple(&ctx, vec![a(&ctx, P::F32, &[5, 3]), slice]).unwrap();
    assert!(
        inference::verify_scan(
            &ctx,
            &[input, slice],
            wrong_result,
            &body,
            1,
            0,
            false,
            false
        )
        .is_err()
    );
    assert!(
        inference::verify_scan(
            &ctx,
            &[input, a(&ctx, P::F32, &[2])],
            result,
            &body,
            1,
            0,
            false,
            false
        )
        .is_err()
    );
    assert!(inference::infer_scan(&ctx, &[input, slice], &body, 1, 0, Some(D::Static(5))).is_err());
}

#[test]
fn scan_layout_sensitive_verification_preserves_dynamic_flags() {
    let ctx = context();
    let slice = a(&ctx, P::F32, &[3]);
    let body = (vec![slice], slice);
    let input = dynamic(&ctx, P::F32, &[D::Static(4), D::Bounded(3)]);
    let result = a(&ctx, P::F32, &[4, 3]);
    inference::verify_scan(&ctx, &[input], result, &body, 0, 0, false, false).unwrap();
    assert!(inference::verify_scan(&ctx, &[input], result, &body, 0, 0, true, false).is_err());
    let input = dynamic(&ctx, P::F32, &[D::Static(4), D::Unbounded]);
    let inferred = inference::infer_scan(&ctx, &[input], &body, 0, 0, None).unwrap();
    assert_eq!(Shape::dimension(&ctx, inferred, 0).unwrap(), D::Static(4));
    assert_eq!(Shape::dimension(&ctx, inferred, 1).unwrap(), D::Static(3));
}

#[test]
fn scans_can_have_no_inputs_or_no_outputs() {
    let ctx = context();
    let scalar = s(&ctx, P::F32);
    let tuple = Shape::tuple(&ctx, vec![scalar, scalar]).unwrap();
    for (args, body, carries) in [
        (vec![scalar], (vec![scalar], tuple), 1),
        (vec![], (vec![], scalar), 0),
    ] {
        let output =
            inference::infer_scan(&ctx, &args, &body, carries, 0, Some(D::Static(5))).unwrap();
        let output = if Shape::is_tuple(&ctx, output) {
            Shape::children(&ctx, output)[0]
        } else {
            output
        };
        assert_eq!(dims(&ctx, output), [5]);
        assert!(inference::infer_scan(&ctx, &args, &body, carries, 0, None).is_err());
    }
    let input = a(&ctx, P::F32, &[5]);
    let body = (vec![scalar, scalar], scalar);
    assert_eq!(
        inference::infer_scan(&ctx, &[input, scalar], &body, 1, 0, None).unwrap(),
        scalar
    );
    assert!(
        inference::infer_scan(
            &ctx,
            &[scalar],
            &(vec![scalar], scalar),
            1,
            0,
            Some(D::Static(5))
        )
        .is_err()
    );
}

#[test]
fn select_and_scatter_checks_both_callees_and_window_source_shape() {
    let mut ctx = context();
    let operand = a(&ctx, P::F32, &[5]);
    let source = a(&ctx, P::F32, &[3]);
    let initial = s(&ctx, P::F32);
    let select = signature(&ctx, &[P::F32, P::F32], P::Pred);
    let scatter = signature(&ctx, &[P::F32, P::F32], P::F32);
    let window = window(3, 1, 0, 0, 1, 1);
    assert_eq!(
        inference::infer_select_and_scatter(
            &ctx, operand, source, initial, &window, &select, &scatter
        )
        .unwrap(),
        operand
    );
    let vals = values(&mut ctx, &[operand, source, initial]);
    let op = SelectAndScatterOp::new(&mut ctx, operand, vals, window.clone(), callees(2));
    op.verify(&ctx).unwrap();
    verify_callees(
        &ctx,
        op.get_operation(),
        O::SelectAndScatter,
        VerifyOptions::default(),
        &[&select, &scatter],
    )
    .unwrap();
    assert!(
        verify_callees(
            &ctx,
            op.get_operation(),
            O::SelectAndScatter,
            VerifyOptions::default(),
            &[&select]
        )
        .is_err()
    );
    assert!(
        inference::infer_select_and_scatter(
            &ctx,
            operand,
            a(&ctx, P::F32, &[4]),
            initial,
            &window,
            &select,
            &scatter
        )
        .is_err()
    );
    for invalid_select in [
        signature(&ctx, &[P::F32], P::Pred),
        signature(&ctx, &[P::S32, P::S32], P::Pred),
        signature(&ctx, &[P::F32, P::F32], P::F32),
    ] {
        assert!(
            inference::infer_select_and_scatter(
                &ctx,
                operand,
                source,
                initial,
                &window,
                &invalid_select,
                &scatter
            )
            .is_err()
        );
    }

    assert!(
        inference::infer_select_and_scatter(
            &ctx,
            operand,
            source,
            initial,
            &window,
            &select,
            &signature(&ctx, &[P::F32, P::S32], P::F32)
        )
        .is_err()
    );
}

#[test]
fn associative_scan_relaxes_body_and_carry_precision_only_under_layout_sensitive_policy() {
    let mut ctx = context();
    let input = a(&ctx, P::F16, &[4, 3]);
    let carry = a(&ctx, P::BF16, &[3]);
    let result = Shape::tuple(&ctx, vec![input, carry]).unwrap();
    let roots = Shape::tuple(&ctx, vec![a(&ctx, P::BF16, &[3]), a(&ctx, P::F32, &[3])]).unwrap();
    let body = (vec![a(&ctx, P::F32, &[3]), a(&ctx, P::F64, &[3])], roots);
    let args = values(&mut ctx, &[input, carry]);
    let op = ScanOp::new(
        &mut ctx,
        result,
        args,
        Axes(vec![0]),
        NumericalInteger(1),
        TriState::True,
        callees(1),
    );
    for (layout_sensitive, allow_mixed_precision, valid) in [
        (false, true, false),
        (true, false, false),
        (true, true, true),
    ] {
        let options = VerifyOptions {
            layout_sensitive,
            allow_mixed_precision,
            ..Default::default()
        };
        assert_eq!(
            verify_callees(&ctx, op.get_operation(), O::Scan, options, &[&body]).is_ok(),
            valid
        );
    }
    let wrong_body = (
        vec![dynamic(&ctx, P::F32, &[D::Bounded(3)]), body.0[1]],
        roots,
    );
    let mixed_strict = VerifyOptions {
        layout_sensitive: true,
        allow_mixed_precision: true,
        ..Default::default()
    };
    assert!(
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Scan,
            mixed_strict,
            &[&wrong_body]
        )
        .is_err()
    );
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_is_associative"), TriState::False);
    assert!(instruction(&ctx, op.get_operation(), O::Scan, mixed_strict).is_err());
}
