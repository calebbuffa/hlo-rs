use xlahlo::{Dimension as D, PrimitiveType as P, Shape, inference, text};

#[test]
fn one_dimensional_convolution_matches_window_geometry_exhaustively() {
    let dimensions = text::parse_convolution_dimensions("bf0_oi0->bf0").unwrap();
    for input_size in 0..6 {
        for kernel_size in 1..5 {
            for stride in 1..3 {
                for padding in 0..3 {
                    let lhs = Shape::array(P::F32, vec![2, 3, input_size]).unwrap();
                    let rhs = Shape::array(P::F32, vec![4, 3, kernel_size]).unwrap();
                    let window = text::parse_window(&format!(
                        "size={kernel_size} stride={stride} pad={padding}_{padding}"
                    ))
                    .unwrap();
                    let result =
                        inference::infer_convolution(&lhs, &rhs, 1, 1, &window, &dimensions, None)
                            .unwrap();
                    let padded = input_size + 2 * padding;
                    let expected = if padded == 0 || kernel_size > padded {
                        0
                    } else {
                        (padded - kernel_size) / stride + 1
                    };
                    assert_eq!(
                        result,
                        Shape::array(P::F32, vec![2, 4, expected])
                            .unwrap()
                            .with_default_layout()
                            .unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn convolution_permutations_groups_dilation_and_dynamic_propagation() {
    let dimensions = text::parse_convolution_dimensions("b0f_0io->0bf").unwrap();
    let lhs =
        Shape::dynamic_array(P::F32, vec![D::Bounded(4), D::Bounded(8), D::Static(6)]).unwrap();
    let rhs = Shape::array(P::F32, vec![3, 2, 9]).unwrap();
    let window = text::parse_window("size=3 stride=2 lhs_dilate=2 rhs_dilate=2 pad=1_2").unwrap();
    let output =
        inference::infer_convolution(&lhs, &rhs, 3, 1, &window, &dimensions, Some(P::F64)).unwrap();
    assert_eq!(output.dimensions(), [7, 4, 9]);
    assert_eq!(output.dimension(0).unwrap(), D::Bounded(7));
    assert_eq!(output.dimension(1).unwrap(), D::Bounded(4));
    assert_eq!(output.element_type(), P::F64);
    let batch_rhs = Shape::array(P::F32, vec![3, 6, 8]).unwrap();
    let output =
        inference::infer_convolution(&lhs, &batch_rhs, 1, 2, &window, &dimensions, None).unwrap();
    assert_eq!(output.dimensions(), [7, 2, 8]);
    let dims = text::parse_convolution_dimensions("bf_io->bf").unwrap();
    let output = inference::infer_convolution(
        &Shape::array(P::F32, vec![2, 3]).unwrap(),
        &Shape::array(P::F32, vec![3, 5]).unwrap(),
        1,
        1,
        &Default::default(),
        &dims,
        None,
    )
    .unwrap();
    assert_eq!(output.dimensions(), [2, 5]);
}

#[test]
fn convolution_rejects_invalid_groups_windows_and_dimension_numbers() {
    let dimensions = text::parse_convolution_dimensions("bf0_oi0->bf0").unwrap();
    let lhs = Shape::array(P::F32, vec![2, 3, 8]).unwrap();
    let rhs = Shape::array(P::F32, vec![4, 3, 3]).unwrap();
    let window = text::parse_window("size=3").unwrap();
    for (features, batches) in [(0, 1), (1, 0), (-1, 1), (2, 2), (2, 1), (1, 3)] {
        assert!(
            inference::infer_convolution(&lhs, &rhs, features, batches, &window, &dimensions, None)
                .is_err()
        );
    }
    let bad_window = text::parse_window("size=2").unwrap();
    assert!(
        inference::infer_convolution(&lhs, &rhs, 1, 1, &bad_window, &dimensions, None).is_err()
    );
    let mut dims = dimensions.clone();
    dims.input_batch_dimension = dims.input_feature_dimension;
    assert!(inference::infer_convolution(&lhs, &rhs, 1, 1, &window, &dims, None).is_err());
    let mut dims = dimensions.clone();
    dims.output_spatial_dimensions = vec![99];
    assert!(inference::infer_convolution(&lhs, &rhs, 1, 1, &window, &dims, None).is_err());
    let dynamic_kernel =
        Shape::dynamic_array(P::F32, vec![D::Bounded(4), D::Static(3), D::Static(3)]).unwrap();
    assert!(
        inference::infer_convolution(&lhs, &dynamic_kernel, 1, 1, &window, &dimensions, None)
            .is_err()
    );
}

#[test]
fn convolution_labels_and_module_assembly_round_trip() {
    for labels in [
        "b01f_01io->b01f",
        "01bf_01io->01bf",
        "bf_io->bf",
        "?b0f_0i?o->b0f?",
    ] {
        let dimensions = text::parse_convolution_dimensions(labels).unwrap();
        let printed = text::print_convolution_dimensions(&dimensions).unwrap();
        assert_eq!(
            text::parse_convolution_dimensions(&printed).unwrap(),
            dimensions
        );
    }
    for invalid in [
        "bb0_oi0->bf0",
        "bf1_oi0->bf0",
        "bf0_oi01->bf0",
        "bf0_oi0_bf0",
        "bf0_oi0->bf00",
    ] {
        assert!(text::parse_convolution_dimensions(invalid).is_err());
    }
    let source = "HloModule m ENTRY main { a = f32[2,3,8] parameter(0) b = f32[4,3,3] parameter(1) ROOT c = f32[2,4,6] convolution(a,b), window={size=3}, dim_labels=bf0_oi0->bf0 }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("feature_group_count=1"));
    assert!(printed.contains("operand_precision={default,default}"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for (input, kernel, output, attrs) in [
        ("f32[2,3]", "f32[4,3]", "f32[2,4]", "dim_labels=bf_oi->bf"),
        (
            "f32[2,6,8]",
            "f32[9,2,3]",
            "f32[2,9,6]",
            "window={size=3}, dim_labels=bf0_oi0->bf0, feature_group_count=3",
        ),
        (
            "f32[4,3,8]",
            "f32[8,3,3]",
            "f32[2,8,6]",
            "window={size=3}, dim_labels=bf0_oi0->bf0, batch_group_count=2",
        ),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ a = {input} parameter(0) b = {kernel} parameter(1) ROOT c = {output} convolution(a,b), {attrs} }}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        text::parse_module(&text::print_module(&module).unwrap())
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    assert!(text::parse_module(&source.replace(", dim_labels=bf0_oi0->bf0", "")).is_err());
}
