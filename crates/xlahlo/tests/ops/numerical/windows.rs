use super::super::*;

pub(super) fn window(
    size: i64,
    stride: i64,
    low: i64,
    high: i64,
    base: i64,
    kernel: i64,
) -> Window {
    Window(vec![WindowDimension {
        size,
        stride,
        padding_low: low,
        padding_high: high,
        base_dilation: base,
        window_dilation: kernel,
        window_reversal: false,
    }])
}

#[test]
fn window_output_inference_uses_default_layout_instead_of_copying_operand_layout() {
    let ctx = context();
    let input = shape::with_layout(
        &ctx,
        a(&ctx, P::F32, &[8, 8]),
        Layout {
            minor_to_major: vec![0, 1],
            ..Default::default()
        },
    )
    .unwrap();
    let windows = Window(vec![window(3, 1, 0, 0, 1, 1).0[0].clone(); 2]);
    let output = inference::infer_window_output(&ctx, input, &windows, P::F32).unwrap();
    assert_eq!(dims(&ctx, output), [6, 6]);
    assert_eq!(
        shape::layout(&ctx, output).unwrap().unwrap().minor_to_major,
        [1, 0]
    );
}

#[test]
fn window_construction_defaults_optional_lists_without_validating_geometry() {
    let mut ctx = context();
    let defaults =
        inference::infer_window_from_dimensions(&[2, 3], &[], &[], &[], &[], None).unwrap();
    assert_eq!(
        defaults
            .0
            .iter()
            .map(|w| (w.size, w.stride, w.base_dilation, w.window_dilation))
            .collect::<Vec<_>>(),
        [(2, 1, 1, 1), (3, 1, 1, 1)]
    );
    assert_eq!(
        inference::infer_window_from_dimensions(&[2, 3], &[], &[], &[], &[], Some(&[])).unwrap(),
        defaults
    );
    let explicit = inference::infer_window_from_dimensions(
        &[2, 3],
        &[4, 5],
        &[(-1, 2), (3, -4)],
        &[6, 7],
        &[8, 9],
        Some(&[true, false]),
    )
    .unwrap();
    assert_eq!(
        explicit.0[0],
        WindowDimension {
            size: 2,
            stride: 4,
            padding_low: -1,
            padding_high: 2,
            base_dilation: 6,
            window_dilation: 8,
            window_reversal: true
        }
    );
    roundtrip(&mut ctx, explicit);
    assert!(
        inference::infer_window_from_dimensions(&[-2, 0], &[0, -1], &[], &[], &[], None).is_ok()
    );
    assert!(
        inference::infer_window_from_dimensions(&[], &[], &[], &[], &[], None)
            .unwrap()
            .0
            .is_empty()
    );
    for (sizes, strides, padding, base, kernel, reversal) in [
        (vec![2, 3], vec![1], vec![], vec![], vec![], vec![]),
        (vec![2, 3], vec![], vec![(0, 0)], vec![], vec![], vec![]),
        (vec![2, 3], vec![], vec![], vec![1], vec![], vec![]),
        (vec![2, 3], vec![], vec![], vec![], vec![1], vec![]),
        (vec![2, 3], vec![], vec![], vec![], vec![], vec![false]),
        (vec![], vec![1], vec![], vec![], vec![], vec![]),
    ] {
        assert!(
            inference::infer_window_from_dimensions(
                &sizes,
                &strides,
                &padding,
                &base,
                &kernel,
                Some(&reversal)
            )
            .is_err()
        );
    }
}

#[test]
fn window_geometry_matches_enumerated_small_sliding_windows() {
    let ctx = context();
    for size in 0..8 {
        let input = a(&ctx, P::F32, &[size]);
        for kernel in 1..5 {
            for low in -1..3 {
                for high in -1..3 {
                    for base_dilation in 1..3 {
                        for kernel_dilation in 1..3 {
                            for stride in 1..4 {
                                let config = window(
                                    kernel,
                                    stride,
                                    low,
                                    high,
                                    base_dilation,
                                    kernel_dilation,
                                );
                                let base = if size == 0 {
                                    0
                                } else {
                                    (size - 1) * base_dilation + 1
                                };
                                let padded = base + low + high;
                                let kernel = (kernel - 1) * kernel_dilation + 1;
                                let result =
                                    inference::infer_window_output(&ctx, input, &config, P::F32);
                                if padded < 0 {
                                    assert!(result.is_err());
                                } else {
                                    let count = (0..=padded)
                                        .filter(|position| {
                                            position % stride == 0 && position + kernel <= padded
                                        })
                                        .count()
                                        as i64;
                                    assert_eq!(dims(&ctx, result.unwrap()), [count]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn window_inference_preserves_dynamic_bounds_and_rejects_overflow_and_bad_factors() {
    let ctx = context();
    let input = dynamic(&ctx, P::F32, &[D::Bounded(5)]);
    assert_eq!(
        shape::dimension(
            &ctx,
            inference::infer_window_output(&ctx, input, &window(3, 1, 0, 0, 1, 1), P::F32).unwrap(),
            0
        )
        .unwrap(),
        D::Bounded(3)
    );
    let unbounded = dynamic(&ctx, P::F32, &[D::Unbounded]);
    assert_eq!(
        shape::dimension(
            &ctx,
            inference::infer_window_output(&ctx, unbounded, &window(3, 1, 0, 0, 1, 1), P::F32)
                .unwrap(),
            0
        )
        .unwrap(),
        D::Unbounded
    );
    for config in [
        window(0, 1, 0, 0, 1, 1),
        window(1, 0, 0, 0, 1, 1),
        window(1, 1, 0, 0, 0, 1),
        window(1, 1, 0, 0, 1, 0),
        window(i64::MAX, 1, 0, 0, 1, 2),
        window(1, 1, i64::MAX, 0, 1, 1),
    ] {
        assert!(inference::infer_window_output(&ctx, input, &config, P::F32).is_err());
    }
}

#[test]
fn reduce_window_shares_reducer_checks_and_round_trips_complete_window_attributes() {
    let mut ctx = context();
    let input = a(&ctx, P::F32, &[5]);
    let initial = s(&ctx, P::F32);
    let mut reducer = signature(&ctx, &[P::F32, P::F32], P::F32);
    let window = window(3, 1, 0, 0, 1, 1);
    let result =
        inference::infer_reduce_window(&ctx, &[input, initial], &window, &reducer).unwrap();
    assert_eq!(dims(&ctx, result), [3]);
    let args = values(&mut ctx, &[input, initial]);
    let op = ReduceWindowOp::new(&mut ctx, result, args, window.clone(), callees(1));
    op.verify(&ctx).unwrap();
    verify_callees(
        &ctx,
        op.get_operation(),
        O::ReduceWindow,
        VerifyOptions::default(),
        &[&reducer],
    )
    .unwrap();
    assert!(inference::infer_reduce_window(&ctx, &[input], &window, &reducer).is_err());
    reducer.1 = shape::tuple(&ctx, vec![initial]).unwrap();
    let tuple = inference::infer_reduce_window(&ctx, &[input, initial], &window, &reducer).unwrap();
    assert!(shape::is_tuple(&ctx, tuple));
    assert_eq!(dims(&ctx, shape::children(&ctx, tuple)[0]), [3]);
    for config in [window.clone(), Window(vec![])] {
        if config.0.is_empty() {
            assert!(
                inference::infer_reduce_window(&ctx, &[input, initial], &config, &reducer).is_err()
            );
        } else {
            roundtrip(&mut ctx, config);
        }
    }
}
