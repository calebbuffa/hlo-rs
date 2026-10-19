use super::super::*;
use xlahlo::{Diagnostic, pliron::result::ErrorKind};

fn inference_error<T: std::fmt::Debug>(result: Result<T>, context: &str) {
    let error = result.unwrap_err();
    assert!(matches!(error.kind, ErrorKind::VerificationFailed));
    let Some(Diagnostic::Inference { message }) = error.err.downcast_ref::<Diagnostic>() else {
        panic!("expected typed inference error, got {error:?}");
    };
    assert!(message.contains(context), "{message:?} lacks {context:?}");
}

#[test]
fn malformed_callable_geometry_is_recoverable_with_typed_payloads() {
    let ctx = context();
    let scalar = s(&ctx, P::F32);
    let input = a(&ctx, P::F32, &[2]);
    let reducer = (vec![scalar; 2], scalar);
    let empty = Shape::tuple(&ctx, vec![]).unwrap();
    inference_error(
        inference::infer_reduce(&ctx, &[], &[0], &reducer),
        "operand count",
    );
    inference_error(
        inference::infer_reduce(&ctx, &[input, scalar], &[i64::MAX], &reducer),
        "out-of-range",
    );
    inference_error(
        inference::infer_reduce(&ctx, &[input, scalar], &[0], &(vec![scalar; 2], empty)),
        "cardinality",
    );
    inference_error(
        inference::infer_map(&ctx, &[], &(vec![], scalar), &[]),
        "map",
    );
    inference_error(
        inference::infer_sort(&ctx, &[], 0, &(vec![], s(&ctx, P::Pred))),
        "sort",
    );
    inference_error(
        inference::infer_scan(&ctx, &[input], &(vec![scalar], scalar), i64::MAX, 0, None),
        "carry count",
    );
    inference_error(
        inference::infer_scan(&ctx, &[input], &(vec![scalar], scalar), 0, -1, None),
        "scan axis",
    );
    inference_error(
        inference::verify_scan(&ctx, &[], empty, &(vec![], empty), 0, 0, false, false),
        "at least one input or output",
    );
    assert_eq!(
        dims(
            &ctx,
            inference::infer_reduce(&ctx, &[input, scalar], &[0], &reducer).unwrap()
        ),
        Vec::<i64>::new(),
    );
    assert_eq!(
        dims(
            &ctx,
            inference::infer_scan(&ctx, &[input], &(vec![scalar], scalar), 0, 0, None).unwrap()
        ),
        [2],
    );
}

#[test]
fn extreme_dot_and_sparse_extents_report_geometry_and_recover() {
    let ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[4, 3]);
    inference_error(
        inference::infer_dot(&ctx, lhs, rhs, &dot_dims(&[i64::MAX], &[0], &[], &[]), None),
        "lhs axes [9223372036854775807] for rank 2",
    );
    let huge = a(&ctx, P::F32, &[i64::MAX / 8]);
    let sparse = SparsityConfig {
        lhs: TensorSparsityConfig {
            num_non_zero: 1,
            block_size: i64::MAX,
            dimension: 0,
            stride: 1,
            idx: 2,
        }
        .into(),
        rhs: OptionalTensorSparsityConfig::Absent,
    };
    inference_error(
        inference::infer_dot_with_sparsity(
            &ctx,
            huge,
            huge,
            &dot_dims(&[0], &[0], &[], &[]),
            None,
            Some(&sparse),
        ),
        "exceeds i64",
    );
    assert_eq!(
        dims(
            &ctx,
            inference::infer_dot(&ctx, lhs, rhs, &dot_dims(&[1], &[0], &[], &[]), None).unwrap()
        ),
        [2, 3],
    );
}

#[test]
fn window_and_convolution_overflow_are_typed_and_leave_context_usable() {
    let ctx = context();
    let huge = a(&ctx, P::F32, &[i64::MAX / 8]);
    let window = super::windows::window;
    inference_error(
        inference::infer_window_output(&ctx, huge, &window(1, 1, 0, 0, i64::MAX, 1), P::F32),
        "overflow",
    );
    inference_error(
        inference::infer_window_output(&ctx, huge, &window(1, 0, 0, 0, 1, 1), P::F32),
        "positive",
    );
    let d = ConvolutionDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: Axes(vec![2]),
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 0,
        kernel_spatial_dimensions: Axes(vec![2]),
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: Axes(vec![2]),
    };
    let lhs = a(&ctx, P::F32, &[1, 1, i64::MAX / 8]);
    let rhs = a(&ctx, P::F32, &[1, 1, 1]);
    inference_error(
        inference::infer_convolution(&ctx, lhs, rhs, 0, 1, &window(1, 1, 0, 0, 1, 1), &d, None),
        "got 0",
    );
    inference_error(
        inference::infer_convolution_kind(
            &ctx,
            lhs,
            rhs,
            1,
            1,
            &window(1, i64::MAX, 0, 0, i64::MAX, 1),
            &d,
            None,
            None,
            ConvolutionKind::Dgrad,
        ),
        "overflows",
    );
    let input = a(&ctx, P::F32, &[1, 1, 3]);
    assert_eq!(
        dims(
            &ctx,
            inference::infer_convolution(
                &ctx,
                input,
                rhs,
                1,
                1,
                &window(1, 1, 0, 0, 1, 1),
                &d,
                None
            )
            .unwrap()
        ),
        [1, 1, 3],
    );
}

#[test]
fn numerical_attribute_errors_identify_the_attribute_and_allow_repair() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[4, 3]);
    let result = a(&ctx, P::F32, &[2, 3]);
    let args = values(&mut ctx, &[lhs, rhs]);
    let op = DotOp::new(
        &mut ctx,
        result,
        args,
        dot_dims(&[1], &[0], &[], &[]),
        precision(),
    );
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_precision"));
    let error = op.verify(&ctx).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_precision",
            ..
        }),
    ));
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_precision"), precision());
    op.verify(&ctx).unwrap();
}
