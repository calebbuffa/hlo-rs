use xlahlo::proto::hlo::ConvolutionKind as K;
use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

#[test]
fn data_gradient_uses_reverse_window_geometry() {
    let lhs =
        Shape::dynamic_array(P::F32, vec![D::Bounded(2), D::Static(4), D::Bounded(3)]).unwrap();
    let rhs = Shape::array(P::F32, vec![4, 8, 3]).unwrap();
    let dims = text::parse_convolution_dimensions("bf0_oi0->bf0").unwrap();
    let window = text::parse_window("size=3 stride=2 lhs_dilate=2 rhs_dilate=2 pad=1_2").unwrap();
    let result = inference::infer_convolution_kind(
        &lhs,
        &rhs,
        1,
        1,
        &window,
        &dims,
        Some(P::F64),
        &Default::default(),
        K::CONVOLUTION_KIND_DGRAD,
    )
    .unwrap();
    assert_eq!(result.dimensions(), [2, 8, 10]);
    assert_eq!(result.dimension(0).unwrap(), D::Bounded(2));
    assert_eq!(result.dimension(2).unwrap(), D::Bounded(10));
    assert_eq!(result.element_type(), P::F64);
    assert!(inference::infer_convolution(&lhs, &rhs, 1, 1, &window, &dims, None).is_err());
}

#[test]
fn small_data_gradient_geometries_match_reverse_window_formula() {
    let dims = text::parse_convolution_dimensions("bf0_oi0->bf0").unwrap();
    for input in 0..5 {
        for kernel in 1..5 {
            for stride in 1..3 {
                for padding in 0..3 {
                    let lhs = Shape::array(P::F32, vec![2, 4, input]).unwrap();
                    let rhs = Shape::array(P::F32, vec![4, 8, kernel]).unwrap();
                    let window = text::parse_window(&format!(
                        "size={kernel} stride={stride} pad={padding}_{padding}"
                    ))
                    .unwrap();
                    let expected = (input - 1) * stride + kernel - 2 * padding;
                    let result = inference::infer_convolution_kind(
                        &lhs,
                        &rhs,
                        1,
                        1,
                        &window,
                        &dims,
                        None,
                        &Default::default(),
                        K::CONVOLUTION_KIND_DGRAD,
                    );
                    if expected < 0 {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(result.unwrap().dimensions(), [2, 8, expected]);
                    }
                }
            }
        }
    }
}

#[test]
fn data_gradient_rejects_geometry_and_propagates_unbounded_spatial_axes() {
    let dims = text::parse_convolution_dimensions("bf0_oi0->bf0").unwrap();
    let window = text::parse_window("size=3").unwrap();
    let lhs = Shape::dynamic_array(P::F32, vec![D::Static(2), D::Static(4), D::Unbounded]).unwrap();
    let rhs = Shape::array(P::F32, vec![4, 8, 3]).unwrap();
    let result = inference::infer_convolution_kind(
        &lhs,
        &rhs,
        1,
        1,
        &window,
        &dims,
        None,
        &Default::default(),
        K::CONVOLUTION_KIND_DGRAD,
    )
    .unwrap();
    assert_eq!(result.dimension(2).unwrap(), D::Unbounded);
    for rhs in [
        Shape::array(P::F32, vec![5, 8, 3]).unwrap(),
        Shape::array(P::F32, vec![4, 8, 2]).unwrap(),
        Shape::dynamic_array(P::F32, vec![D::Bounded(4), D::Static(8), D::Static(3)]).unwrap(),
    ] {
        assert!(
            inference::infer_convolution_kind(
                &lhs,
                &rhs,
                1,
                1,
                &window,
                &dims,
                None,
                &Default::default(),
                K::CONVOLUTION_KIND_DGRAD
            )
            .is_err()
        );
    }
}

#[test]
fn weight_gradient_uses_kernel_axis_positions_and_pinned_dynamic_mapping() {
    let dims = text::parse_convolution_dimensions("bf0_oi0->bf0").unwrap();
    let window = text::parse_window("size=3").unwrap();
    let lhs =
        Shape::dynamic_array(P::F32, vec![D::Bounded(2), D::Static(8), D::Bounded(7)]).unwrap();
    let rhs = Shape::array(P::F32, vec![2, 4, 5]).unwrap();
    let result = inference::infer_convolution_kind(
        &lhs,
        &rhs,
        1,
        1,
        &window,
        &dims,
        None,
        &Default::default(),
        K::CONVOLUTION_KIND_WGRAD,
    )
    .unwrap();
    assert_eq!(result.dimensions(), [4, 8, 3]);
    assert_eq!(result.dimension(0).unwrap(), D::Bounded(4));
    assert_eq!(result.dimension(2).unwrap(), D::Bounded(3));
    let lhs = Shape::array(P::F32, vec![2, 9, 7]).unwrap();
    let result = inference::infer_convolution_kind(
        &lhs,
        &rhs,
        2,
        1,
        &window,
        &dims,
        None,
        &Default::default(),
        K::CONVOLUTION_KIND_WGRAD,
    )
    .unwrap();
    assert_eq!(result.dimensions(), [4, 4, 3]);
}

#[test]
fn weight_gradient_requires_batch_equality_but_does_not_check_rhs_window_size() {
    let dims = text::parse_convolution_dimensions("bf_io->bf").unwrap();
    let lhs = Shape::array(P::F32, vec![2, 8]).unwrap();
    let rhs = Shape::array(P::F32, vec![2, 4]).unwrap();
    assert_eq!(
        inference::infer_convolution_kind(
            &lhs,
            &rhs,
            1,
            1,
            &Default::default(),
            &dims,
            None,
            &Default::default(),
            K::CONVOLUTION_KIND_WGRAD
        )
        .unwrap()
        .dimensions(),
        [8, 4]
    );
    let rhs = Shape::array(P::F32, vec![3, 4]).unwrap();
    assert!(
        inference::infer_convolution_kind(
            &lhs,
            &rhs,
            1,
            1,
            &Default::default(),
            &dims,
            None,
            &Default::default(),
            K::CONVOLUTION_KIND_WGRAD
        )
        .is_err()
    );
}

#[test]
fn gradient_kind_metadata_does_not_change_default_hlo_verifier_inference() {
    let source = "HloModule m ENTRY main { l = f32[2,4,3] parameter(0) r = f32[4,8,3] parameter(1) ROOT c = f32[2,8,5] convolution(l,r), dim_labels=bf0_oi0->bf0, window={size=3}, convolution_kind=dgrad }";
    assert!(
        text::parse_module(source)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}
