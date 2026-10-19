use super::super::*;
use super::windows::window;

fn standard() -> ConvolutionDimensionNumbers {
    ConvolutionDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: Axes(vec![2]),
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 0,
        kernel_spatial_dimensions: Axes(vec![2]),
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: Axes(vec![2]),
    }
}

#[test]
fn convolution_inference_uses_default_layout_instead_of_copying_operand_layouts() {
    let ctx = context();
    let lhs = Shape::with_layout(
        &ctx,
        a(&ctx, P::F32, &[2, 3, 8]),
        Layout {
            minor_to_major: vec![0, 1, 2],
            ..Default::default()
        },
    )
    .unwrap();
    let rhs = Shape::with_layout(
        &ctx,
        a(&ctx, P::F32, &[4, 3, 3]),
        Layout {
            minor_to_major: vec![0, 1, 2],
            ..Default::default()
        },
    )
    .unwrap();
    let result = inference::infer_convolution(
        &ctx,
        lhs,
        rhs,
        1,
        1,
        &window(3, 1, 0, 0, 1, 1),
        &standard(),
        None,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [2, 4, 6]);
    assert_eq!(
        Shape::layout(&ctx, result).unwrap().unwrap().minor_to_major,
        [2, 1, 0]
    );
}

#[test]
fn one_dimensional_convolution_matches_window_geometry_exhaustively() {
    let ctx = context();
    for input_size in 0..6 {
        for kernel_size in 1..5 {
            for stride in 1..3 {
                for padding in 0..3 {
                    let lhs = a(&ctx, P::F32, &[2, 3, input_size]);
                    let rhs = a(&ctx, P::F32, &[4, 3, kernel_size]);
                    let result = inference::infer_convolution(
                        &ctx,
                        lhs,
                        rhs,
                        1,
                        1,
                        &window(kernel_size, stride, padding, padding, 1, 1),
                        &standard(),
                        None,
                    )
                    .unwrap();
                    let padded = input_size + 2 * padding;
                    let expected = if padded == 0 || kernel_size > padded {
                        0
                    } else {
                        (padded - kernel_size) / stride + 1
                    };
                    assert_eq!(dims(&ctx, result), [2, 4, expected]);
                }
            }
        }
    }
}

#[test]
fn convolution_permutations_groups_dilation_and_dynamic_propagation() {
    let ctx = context();
    let d = ConvolutionDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 2,
        input_spatial_dimensions: Axes(vec![1]),
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 2,
        kernel_spatial_dimensions: Axes(vec![0]),
        output_batch_dimension: 1,
        output_feature_dimension: 2,
        output_spatial_dimensions: Axes(vec![0]),
    };
    let lhs = dynamic(&ctx, P::F32, &[D::Bounded(4), D::Bounded(8), D::Static(6)]);
    let w = window(3, 2, 1, 2, 2, 2);
    let result = inference::infer_convolution(
        &ctx,
        lhs,
        a(&ctx, P::F32, &[3, 2, 9]),
        3,
        1,
        &w,
        &d,
        Some(P::F64),
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [7, 4, 9]);
    assert_eq!(Shape::dimension(&ctx, result, 0).unwrap(), D::Bounded(7));
    assert_eq!(Shape::dimension(&ctx, result, 1).unwrap(), D::Bounded(4));
    assert_eq!(Shape::element_type(&ctx, result).unwrap(), P::F64);
    let result =
        inference::infer_convolution(&ctx, lhs, a(&ctx, P::F32, &[3, 6, 8]), 1, 2, &w, &d, None)
            .unwrap();
    assert_eq!(dims(&ctx, result), [7, 2, 8]);
    let d = ConvolutionDimensionNumbers {
        input_spatial_dimensions: Axes(vec![]),
        kernel_input_feature_dimension: 0,
        kernel_output_feature_dimension: 1,
        kernel_spatial_dimensions: Axes(vec![]),
        output_spatial_dimensions: Axes(vec![]),
        ..standard()
    };
    let result = inference::infer_convolution(
        &ctx,
        a(&ctx, P::F32, &[2, 3]),
        a(&ctx, P::F32, &[3, 5]),
        1,
        1,
        &Window(vec![]),
        &d,
        None,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [2, 5]);
}

#[test]
fn convolution_rejects_invalid_groups_windows_and_dimension_numbers() {
    let ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 3, 8]);
    let rhs = a(&ctx, P::F32, &[4, 3, 3]);
    let w = window(3, 1, 0, 0, 1, 1);
    let d = standard();
    for (fg, bg) in [(0, 1), (1, 0), (-1, 1), (2, 2), (2, 1), (1, 3)] {
        assert!(inference::infer_convolution(&ctx, lhs, rhs, fg, bg, &w, &d, None).is_err());
    }
    assert!(
        inference::infer_convolution(
            &ctx,
            Shape::buffer(&ctx, lhs).unwrap(),
            rhs,
            1,
            1,
            &w,
            &d,
            None
        )
        .is_err()
    );
    assert!(
        inference::infer_convolution(
            &ctx,
            lhs,
            Shape::buffer(&ctx, rhs).unwrap(),
            1,
            1,
            &w,
            &d,
            None
        )
        .is_err()
    );
    assert!(
        inference::infer_convolution(&ctx, lhs, rhs, 1, 1, &window(2, 1, 0, 0, 1, 1), &d, None)
            .is_err()
    );
    for malformed in [
        ConvolutionDimensionNumbers {
            input_batch_dimension: 1,
            ..d.clone()
        },
        ConvolutionDimensionNumbers {
            input_feature_dimension: 0,
            kernel_input_feature_dimension: 3,
            ..d.clone()
        },
        ConvolutionDimensionNumbers {
            output_spatial_dimensions: Axes(vec![99]),
            ..d.clone()
        },
        ConvolutionDimensionNumbers {
            kernel_spatial_dimensions: Axes(vec![]),
            ..d.clone()
        },
    ] {
        assert!(inference::infer_convolution(&ctx, lhs, rhs, 1, 1, &w, &malformed, None).is_err());
    }
    let dynamic_kernel = dynamic(&ctx, P::F32, &[D::Bounded(4), D::Static(3), D::Static(3)]);
    assert!(inference::infer_convolution(&ctx, lhs, dynamic_kernel, 1, 1, &w, &d, None).is_err());
    error_contains(
        inference::infer_convolution(
            &ctx,
            Shape::buffer(&ctx, lhs).unwrap(),
            rhs,
            0,
            1,
            &w,
            &d,
            None,
        ),
        "lhs must be an array",
    );
    error_contains(
        inference::infer_convolution(
            &ctx,
            lhs,
            Shape::buffer(&ctx, rhs).unwrap(),
            1,
            1,
            &w,
            &d,
            None,
        ),
        "rhs must be an array",
    );
    error_contains(
        inference::infer_convolution(
            &ctx,
            a(&ctx, P::F32, &[2, 3]),
            rhs,
            1,
            1,
            &w,
            &ConvolutionDimensionNumbers {
                kernel_spatial_dimensions: Axes(vec![]),
                ..d.clone()
            },
            None,
        ),
        "input and kernel spatial ranks differ",
    );
    error_contains(
        inference::infer_convolution(&ctx, lhs, rhs, 1, 3, &w, &d, None),
        "kernel output features must be divisible by batch groups",
    );
    error_contains(
        inference::infer_convolution(
            &ctx,
            lhs,
            rhs,
            1,
            1,
            &w,
            &ConvolutionDimensionNumbers {
                input_feature_dimension: 0,
                kernel_input_feature_dimension: 3,
                ..d
            },
            None,
        ),
        "dimension number out of range",
    );
}

#[test]
fn data_gradient_uses_reverse_window_geometry() {
    let ctx = context();
    let lhs = dynamic(&ctx, P::F32, &[D::Bounded(2), D::Static(4), D::Bounded(3)]);
    let rhs = a(&ctx, P::F32, &[4, 8, 3]);
    let w = window(3, 2, 1, 2, 2, 2);
    let result = inference::infer_convolution_kind(
        &ctx,
        lhs,
        rhs,
        1,
        1,
        &w,
        &standard(),
        Some(P::F64),
        None,
        ConvolutionKind::Dgrad,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [2, 8, 10]);
    assert_eq!(Shape::dimension(&ctx, result, 0).unwrap(), D::Bounded(2));
    assert_eq!(Shape::dimension(&ctx, result, 2).unwrap(), D::Bounded(10));
    assert_eq!(Shape::element_type(&ctx, result).unwrap(), P::F64);
    assert!(inference::infer_convolution(&ctx, lhs, rhs, 1, 1, &w, &standard(), None).is_err());
}

#[test]
fn small_data_gradient_geometries_match_reverse_window_formula() {
    let ctx = context();
    for input in 0..5 {
        for kernel in 1..5 {
            for stride in 1..3 {
                for padding in 0..3 {
                    let expected = (input - 1) * stride + kernel - 2 * padding;
                    let result = inference::infer_convolution_kind(
                        &ctx,
                        a(&ctx, P::F32, &[2, 4, input]),
                        a(&ctx, P::F32, &[4, 8, kernel]),
                        1,
                        1,
                        &window(kernel, stride, padding, padding, 1, 1),
                        &standard(),
                        None,
                        None,
                        ConvolutionKind::Dgrad,
                    );
                    if expected < 0 {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(dims(&ctx, result.unwrap()), [2, 8, expected]);
                    }
                }
            }
        }
    }
}

#[test]
fn data_gradient_rejects_geometry_and_propagates_unbounded_spatial_axes() {
    let ctx = context();
    let lhs = dynamic(&ctx, P::F32, &[D::Static(2), D::Static(4), D::Unbounded]);
    let w = window(3, 1, 0, 0, 1, 1);
    let result = inference::infer_convolution_kind(
        &ctx,
        lhs,
        a(&ctx, P::F32, &[4, 8, 3]),
        1,
        1,
        &w,
        &standard(),
        None,
        None,
        ConvolutionKind::Dgrad,
    )
    .unwrap();
    assert_eq!(Shape::dimension(&ctx, result, 2).unwrap(), D::Unbounded);
    for rhs in [
        a(&ctx, P::F32, &[5, 8, 3]),
        a(&ctx, P::F32, &[4, 8, 2]),
        dynamic(&ctx, P::F32, &[D::Bounded(4), D::Static(8), D::Static(3)]),
    ] {
        assert!(
            inference::infer_convolution_kind(
                &ctx,
                lhs,
                rhs,
                1,
                1,
                &w,
                &standard(),
                None,
                None,
                ConvolutionKind::Dgrad
            )
            .is_err()
        );
    }
}

#[test]
fn weight_gradient_uses_kernel_axis_positions_and_output_dynamic_mapping() {
    let ctx = context();
    let lhs = dynamic(&ctx, P::F32, &[D::Bounded(2), D::Static(8), D::Bounded(7)]);
    let rhs = a(&ctx, P::F32, &[2, 4, 5]);
    let w = window(3, 1, 0, 0, 1, 1);
    let result = inference::infer_convolution_kind(
        &ctx,
        lhs,
        rhs,
        1,
        1,
        &w,
        &standard(),
        None,
        None,
        ConvolutionKind::Wgrad,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [4, 8, 3]);
    assert_eq!(
        Shape::layout(&ctx, result).unwrap().unwrap().minor_to_major,
        [2, 1, 0]
    );
    assert_eq!(Shape::dimension(&ctx, result, 0).unwrap(), D::Bounded(4));
    assert_eq!(Shape::dimension(&ctx, result, 2).unwrap(), D::Bounded(3));
    let lhs = a(&ctx, P::F32, &[2, 9, 7]);
    for rhs in [
        rhs,
        dynamic(&ctx, P::F32, &[D::Bounded(2), D::Static(4), D::Static(5)]),
    ] {
        let result = inference::infer_convolution_kind(
            &ctx,
            lhs,
            rhs,
            2,
            1,
            &w,
            &standard(),
            None,
            None,
            ConvolutionKind::Wgrad,
        )
        .unwrap();
        assert_eq!(dims(&ctx, result), [4, 4, 3]);
        assert_eq!(Shape::dimension(&ctx, result, 0).unwrap(), D::Static(4));
    }
}

#[test]
fn weight_gradient_requires_batch_equality_but_does_not_check_rhs_window_size() {
    let ctx = context();
    let d = ConvolutionDimensionNumbers {
        input_spatial_dimensions: Axes(vec![]),
        kernel_input_feature_dimension: 0,
        kernel_output_feature_dimension: 1,
        kernel_spatial_dimensions: Axes(vec![]),
        output_spatial_dimensions: Axes(vec![]),
        ..standard()
    };
    let lhs = a(&ctx, P::F32, &[2, 8]);
    let result = inference::infer_convolution_kind(
        &ctx,
        lhs,
        a(&ctx, P::F32, &[2, 4]),
        1,
        1,
        &Window(vec![]),
        &d,
        None,
        None,
        ConvolutionKind::Wgrad,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [8, 4]);
    assert!(
        inference::infer_convolution_kind(
            &ctx,
            lhs,
            a(&ctx, P::F32, &[3, 4]),
            1,
            1,
            &Window(vec![]),
            &d,
            None,
            None,
            ConvolutionKind::Wgrad
        )
        .is_err()
    );
}

#[test]
fn gradient_kind_metadata_does_not_change_default_hlo_verifier_inference() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4, 3]);
    let rhs = a(&ctx, P::F32, &[4, 8, 3]);
    let output = a(&ctx, P::F32, &[2, 8, 5]);
    let vals = values(&mut ctx, &[lhs, rhs]);
    let op = ConvolutionOp::new(
        &mut ctx,
        output,
        vals,
        standard(),
        window(3, 1, 0, 0, 1, 1),
        NumericalInteger(1),
        NumericalInteger(1),
        ConvolutionKind::Dgrad,
        precision(),
    );
    assert!(op.verify(&ctx).is_err());
}

#[test]
fn convolution_labels_and_module_assembly_round_trip() {
    let mut ctx = context();
    roundtrip(&mut ctx, standard());
    let lhs = a(&ctx, P::F32, &[2, 3, 8]);
    let rhs = a(&ctx, P::F32, &[4, 3, 3]);
    let result = a(&ctx, P::F32, &[2, 4, 6]);
    let vals = values(&mut ctx, &[lhs, rhs]);
    ConvolutionOp::new(
        &mut ctx,
        result,
        vals,
        standard(),
        window(3, 1, 0, 0, 1, 1),
        NumericalInteger(1),
        NumericalInteger(1),
        ConvolutionKind::Unset,
        precision(),
    )
    .verify(&ctx)
    .unwrap();
}
