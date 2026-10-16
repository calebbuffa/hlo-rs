use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, ConvDimensionNumbers, ConvolutionOptions, ElementType,
    StableHloAttribute, StableHloOp, TensorType, Type, infer_convolution_type,
};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn dimensions() -> ConvDimensionNumbers {
    ConvDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 2,
        input_spatial_dimensions: vec![1],
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 2,
        kernel_spatial_dimensions: vec![0],
        output_batch_dimension: 0,
        output_feature_dimension: 2,
        output_spatial_dimensions: vec![1],
    }
}

fn attributes(d: &ConvDimensionNumbers, feature: i64, batch: i64) -> Attributes {
    [
        (
            "dimension_numbers".into(),
            Attribute::StableHlo(Box::new(StableHloAttribute::ConvDimensionNumbers(
                d.clone(),
            ))),
        ),
        ("feature_group_count".into(), Attribute::integer(feature)),
        ("batch_group_count".into(), Attribute::integer(batch)),
    ]
    .into()
}

#[test]
fn convolution_inference_uses_selected_elements_and_discards_operand_and_template_encodings() {
    let d = dimensions();
    let options = ConvolutionOptions::new(&d);
    assert_eq!(
        infer_convolution_type(
            &ty("tensor<?x?x4xf32, #stablehlo.bounds<4, 7, ?>>"),
            &ty("tensor<3x4x6xf32>"),
            &ty("tensor<?xf32, #stablehlo.bounds<8>>"),
            options,
        )
        .unwrap(),
        ty("tensor<?x?x6xf32>"),
    );
    let encoded: Type = TensorType::static_shape(ElementType::F32, [2, 7, 4])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_convolution_type(
            &encoded,
            &ty("tensor<3x4x6xf32>"),
            &ty("tensor<f32>"),
            options
        )
        .unwrap(),
        ty("tensor<2x5x6xf32>"),
    );
}

#[test]
fn convolution_all_independent_rank_three_permutations_have_reference_shapes() {
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let tensor = |axes: [usize; 3], sizes: [u64; 3]| {
        let mut shape = [0; 3];
        for (axis, size) in axes.into_iter().zip(sizes) {
            shape[axis] = size;
        }
        TensorType::static_shape(ElementType::F32, shape).into()
    };
    for a in permutations {
        for b in permutations {
            for c in permutations {
                let d = ConvDimensionNumbers {
                    input_batch_dimension: a[0] as i64,
                    input_feature_dimension: a[1] as i64,
                    input_spatial_dimensions: vec![a[2] as i64],
                    kernel_input_feature_dimension: b[0] as i64,
                    kernel_output_feature_dimension: b[1] as i64,
                    kernel_spatial_dimensions: vec![b[2] as i64],
                    output_batch_dimension: c[0] as i64,
                    output_feature_dimension: c[1] as i64,
                    output_spatial_dimensions: vec![c[2] as i64],
                };
                let lhs = tensor(a, [2, 4, 7]);
                let rhs = tensor(b, [4, 6, 3]);
                let expected: Type = tensor(c, [2, 6, 5]);
                assert_eq!(
                    infer_convolution_type(
                        &lhs,
                        &rhs,
                        &ty("tensor<f32>"),
                        ConvolutionOptions::new(&d)
                    )
                    .unwrap(),
                    expected,
                    "{d:?}",
                );
                common::computation(
                    StableHloOp::Convolution,
                    vec![lhs, rhs],
                    vec![expected],
                    attributes(&d, 1, 1),
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn convolution_grouped_feature_bounds_are_checked_without_refining_dynamic_output_dimensions() {
    let d = dimensions();
    let options = ConvolutionOptions {
        feature_group_count: 2,
        ..ConvolutionOptions::new(&d)
    };
    for (lhs, rhs, accepted) in [
        (
            "tensor<2x7x?xf32, #stablehlo.bounds<?, ?, 3>>",
            "tensor<3x2x6xf32>",
            false,
        ),
        (
            "tensor<2x7x?xf32, #stablehlo.bounds<?, ?, 4>>",
            "tensor<3x2x6xf32>",
            true,
        ),
        (
            "tensor<2x7x4xf32>",
            "tensor<3x?x6xf32, #stablehlo.bounds<?, 1, ?>>",
            false,
        ),
        (
            "tensor<2x7x4xf32>",
            "tensor<3x?x6xf32, #stablehlo.bounds<?, 2, ?>>",
            true,
        ),
    ] {
        assert_eq!(
            infer_convolution_type(&ty(lhs), &ty(rhs), &ty("tensor<f32>"), options).is_ok(),
            accepted,
            "{lhs}, {rhs}"
        );
    }
    let result = infer_convolution_type(
        &ty("tensor<?x7x4xf32, #stablehlo.bounds<5, ?, ?>>"),
        &ty("tensor<3x4x6xf32>"),
        &ty("tensor<f32>"),
        ConvolutionOptions {
            batch_group_count: 2,
            ..ConvolutionOptions::new(&d)
        },
    )
    .unwrap();
    assert_eq!(result, ty("tensor<?x5x6xf32>"));
}

#[test]
fn convolution_result_batch_and_feature_witnesses_obey_bounds_and_group_divisibility() {
    let d = dimensions();
    for opcode in [StableHloOp::Convolution, StableHloOp::DynamicConv] {
        for (lhs, rhs, output, feature, batch, accepted) in [
            (
                "tensor<?x7x4xf32, #stablehlo.bounds<5, ?, ?>>",
                "tensor<3x4x6xf32>",
                "tensor<3x5x6xf32>",
                1,
                2,
                false,
            ),
            (
                "tensor<?x7x4xf32, #stablehlo.bounds<5, ?, ?>>",
                "tensor<3x4x6xf32>",
                "tensor<2x5x6xf32>",
                1,
                2,
                true,
            ),
            (
                "tensor<4x7x4xf32>",
                "tensor<3x4x6xf32>",
                "tensor<?x5x6xf32, #stablehlo.bounds<1, ?, ?>>",
                1,
                2,
                false,
            ),
            (
                "tensor<2x7x4xf32>",
                "tensor<3x2x?xf32>",
                "tensor<2x5x5xf32>",
                2,
                1,
                false,
            ),
            (
                "tensor<2x7x4xf32>",
                "tensor<3x2x?xf32>",
                "tensor<2x5x6xf32>",
                2,
                1,
                true,
            ),
            (
                "tensor<2x7x4xf32>",
                "tensor<3x4x?xf32, #stablehlo.bounds<?, ?, 5>>",
                "tensor<2x5x6xf32>",
                1,
                1,
                false,
            ),
            (
                "tensor<2x7x4xf32>",
                "tensor<3x4x6xf32>",
                "tensor<2x5x?xf32, #stablehlo.bounds<?, ?, 5>>",
                1,
                1,
                false,
            ),
        ] {
            let mut inputs = vec![ty(lhs), ty(rhs)];
            if opcode == StableHloOp::DynamicConv {
                inputs.push(ty("tensor<1x2xi64>"));
            }
            assert_eq!(
                common::computation(
                    opcode,
                    inputs,
                    vec![ty(output)],
                    attributes(&d, feature, batch)
                )
                .is_ok(),
                accepted,
                "{opcode:?}: {lhs}, {rhs}, {output}, {feature}, {batch}",
            );
        }
    }
}

#[test]
fn convolution_rejects_invalid_ranks_dimensions_groups_and_all_window_vectors() {
    let d = dimensions();
    let lhs = ty("tensor<2x7x4xf32>");
    let rhs = ty("tensor<3x4x6xf32>");
    for (feature, batch) in [(0, 1), (-1, 1), (1, 0), (1, -1), (2, 2), (3, 1), (1, 3)] {
        assert!(
            infer_convolution_type(
                &lhs,
                &rhs,
                &ty("tensor<f32>"),
                ConvolutionOptions {
                    feature_group_count: feature,
                    batch_group_count: batch,
                    ..ConvolutionOptions::new(&d)
                },
            )
            .is_err()
        );
    }
    for bad in [
        ConvDimensionNumbers {
            input_batch_dimension: 2,
            ..d.clone()
        },
        ConvDimensionNumbers {
            input_spatial_dimensions: vec![],
            ..d.clone()
        },
        ConvDimensionNumbers {
            kernel_output_feature_dimension: -1,
            ..d.clone()
        },
        ConvDimensionNumbers {
            output_spatial_dimensions: vec![3],
            ..d.clone()
        },
    ] {
        assert!(
            infer_convolution_type(
                &lhs,
                &rhs,
                &ty("tensor<f32>"),
                ConvolutionOptions::new(&bad)
            )
            .is_err()
        );
    }
    for (a, b) in [
        ("tensor<f32>", "tensor<f32>"),
        ("tensor<2xf32>", "tensor<3xf32>"),
        ("tensor<2x7x4xf32>", "tensor<3x4xf32>"),
    ] {
        assert!(
            infer_convolution_type(
                &ty(a),
                &ty(b),
                &ty("tensor<f32>"),
                ConvolutionOptions::new(&d)
            )
            .is_err()
        );
    }
    for vector in [vec![], vec![0], vec![-1], vec![1, 1]] {
        for field in 0..3 {
            let mut options = ConvolutionOptions::new(&d);
            match field {
                0 => options.window_strides = Some(&vector),
                1 => options.lhs_dilation = Some(&vector),
                _ => options.rhs_dilation = Some(&vector),
            }
            assert!(infer_convolution_type(&lhs, &rhs, &ty("tensor<f32>"), options).is_err());
        }
    }
    for reversal in [vec![], vec![false, true]] {
        assert!(
            infer_convolution_type(
                &lhs,
                &rhs,
                &ty("tensor<f32>"),
                ConvolutionOptions {
                    window_reversal: Some(&reversal),
                    ..ConvolutionOptions::new(&d)
                }
            )
            .is_err()
        );
    }
    for padding in [vec![], vec![[0, 0], [0, 0]]] {
        assert!(
            infer_convolution_type(
                &lhs,
                &rhs,
                &ty("tensor<f32>"),
                ConvolutionOptions {
                    padding: Some(&padding),
                    ..ConvolutionOptions::new(&d)
                }
            )
            .is_err()
        );
    }
}

#[test]
fn convolution_supports_rank_two_zero_kernel_extents_and_checked_overflow() {
    let rank_two = ConvDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        kernel_input_feature_dimension: 0,
        kernel_output_feature_dimension: 1,
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        ..Default::default()
    };
    assert_eq!(
        infer_convolution_type(
            &ty("tensor<2x4xf32>"),
            &ty("tensor<4x6xf32>"),
            &ty("tensor<f32>"),
            ConvolutionOptions::new(&rank_two)
        )
        .unwrap(),
        ty("tensor<2x6xf32>"),
    );
    let d = dimensions();
    for (input, kernel, output) in [(0, 3, 0), (7, 0, 8), (0, 0, 0)] {
        assert_eq!(
            infer_convolution_type(
                &ty(&format!("tensor<2x{input}x4xf32>")),
                &ty(&format!("tensor<{kernel}x4x6xf32>")),
                &ty("tensor<f32>"),
                ConvolutionOptions::new(&d)
            )
            .unwrap(),
            ty(&format!("tensor<2x{output}x6xf32>")),
        );
    }
    assert!(
        infer_convolution_type(
            &ty(&format!("tensor<2x{}x4xf32>", i64::MAX)),
            &ty("tensor<1x4x6xf32>"),
            &ty("tensor<f32>"),
            ConvolutionOptions {
                lhs_dilation: Some(&[i64::MAX]),
                ..ConvolutionOptions::new(&d)
            },
        )
        .is_err()
    );
}

#[test]
fn convolution_quantized_template_is_revalidated_after_shape_and_axis_selection() {
    let d = dimensions();
    assert_eq!(
        infer_convolution_type(
            &ty("tensor<2x7x4x!quant.uniform<i8:f32, 0.25:1>>"),
            &ty("tensor<3x4x6x!quant.uniform<i8:f32, 0.5:2>>"),
            &ty("tensor<!quant.uniform<i16:f32, 0.125:0>>"),
            ConvolutionOptions::new(&d),
        )
        .unwrap(),
        ty("tensor<2x5x6x!quant.uniform<i16:f32, 0.125:0>>"),
    );
    let per_axis = ty("tensor<1x1x2x!quant.uniform<i8:f32:2, {0.25,0.5}>>");
    assert!(
        infer_convolution_type(
            &ty("tensor<2x7x4xf32>"),
            &ty("tensor<3x4x6xf32>"),
            &per_axis,
            ConvolutionOptions::new(&d),
        )
        .is_err()
    );
    for source in [
        "tensor<2x7x4xindex>",
        "tensor<2x7x4xtf32>",
        "tensor<*xf32>",
        "!stablehlo.token",
    ] {
        assert!(
            infer_convolution_type(
                &ty(source),
                &ty("tensor<3x4x6xf32>"),
                &ty("tensor<f32>"),
                ConvolutionOptions::new(&d)
            )
            .is_err()
        );
    }
}

#[test]
fn convolution_small_window_geometry_matches_enumerated_strided_positions() {
    let d = dimensions();
    for size in 0..=4 {
        for kernel in 0..=3 {
            for stride in 1..=2 {
                for base in 1..=2 {
                    for dilation in 1..=2 {
                        for low in -1..=1 {
                            for high in -1..=1 {
                                let padded =
                                    if size == 0 { 0 } else { (size - 1) * base + 1 } + low + high;
                                let window = if kernel == 0 {
                                    0
                                } else {
                                    (kernel - 1) * dilation + 1
                                };
                                let expected = if padded <= 0 {
                                    0
                                } else {
                                    (0..=padded)
                                        .filter(|&start| {
                                            start % stride == 0 && start + window <= padded
                                        })
                                        .count()
                                };
                                let result = infer_convolution_type(
                                    &ty(&format!("tensor<2x{size}x4xf32>")),
                                    &ty(&format!("tensor<{kernel}x4x6xf32>")),
                                    &ty("tensor<f32>"),
                                    ConvolutionOptions {
                                        window_strides: Some(&[stride]),
                                        lhs_dilation: Some(&[base]),
                                        rhs_dilation: Some(&[dilation]),
                                        padding: Some(&[[low, high]]),
                                        ..ConvolutionOptions::new(&d)
                                    },
                                )
                                .unwrap();
                                assert_eq!(
                                    result,
                                    ty(&format!("tensor<2x{expected}x6xf32>")),
                                    "{size},{kernel},{stride},{base},{dilation},{low},{high}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn dynamic_convolution_unknown_padding_preserves_batch_and_feature_constraints() {
    let d = dimensions();
    for output in [
        "tensor<2x?x6xf32>",
        "tensor<2x0x6xf32>",
        "tensor<2x100x6xf32>",
    ] {
        common::computation(
            StableHloOp::DynamicConv,
            vec![
                ty("tensor<2x7x4xf32>"),
                ty("tensor<3x4x6xf32>"),
                ty("tensor<1x2xi32>"),
            ],
            vec![ty(output)],
            attributes(&d, 1, 1),
        )
        .unwrap();
    }
    for padding in [
        "tensor<2xi64>",
        "tensor<2x2xi64>",
        "tensor<1x3xi64>",
        "tensor<?x2xi64>",
        "tensor<1x?xi64>",
        "tensor<1x2xf32>",
    ] {
        assert!(
            common::computation(
                StableHloOp::DynamicConv,
                vec![
                    ty("tensor<2x7x4xf32>"),
                    ty("tensor<3x4x6xf32>"),
                    ty(padding)
                ],
                vec![ty("tensor<2x?x6xf32>")],
                attributes(&d, 1, 1),
            )
            .is_err(),
            "{padding}"
        );
    }
    for (name, value) in [
        ("window_strides", Attribute::integers(vec![0])),
        ("lhs_dilation", Attribute::integers(vec![-1])),
        ("rhs_dilation", Attribute::integers(vec![1, 1])),
        ("window_reversal", Attribute::bools(vec![])),
    ] {
        let mut attrs = attributes(&d, 1, 1);
        attrs.insert(name.into(), value);
        assert!(
            common::computation(
                StableHloOp::DynamicConv,
                vec![
                    ty("tensor<2x7x4xf32>"),
                    ty("tensor<3x4x6xf32>"),
                    ty("tensor<1x2xi64>")
                ],
                vec![ty("tensor<2x?x6xf32>")],
                attrs,
            )
            .is_err(),
            "{name}"
        );
    }
}

#[test]
fn convolution_small_grouped_shape_witnesses_match_exact_extent_feasibility() {
    let d = dimensions();
    let witnesses = [
        (Some(0), None),
        (Some(2), None),
        (Some(3), None),
        (Some(4), None),
        (None, None),
        (None, Some(1)),
        (None, Some(3)),
    ];
    let admits = |w: (Option<u64>, Option<u64>), value| {
        w.0.is_none_or(|size| size == value) && w.1.is_none_or(|upper| value <= upper)
    };
    let with_witness = |shape: &[Option<u64>], axis, w: (Option<u64>, Option<u64>)| -> Type {
        let mut shape = shape.to_vec();
        shape[axis] = w.0;
        let mut t = TensorType::new(ElementType::F32, shape);
        if let Some(upper) = w.1 {
            let mut bounds = vec![-1; 3];
            bounds[axis] = upper as i64;
            t = t.with_encoding(Attribute::StableHlo(Box::new(
                StableHloAttribute::TypeExtensions(stablehlo::TypeExtensions { bounds }),
            )));
        }
        t.into()
    };
    for group in 1..=3 {
        for a in witnesses {
            for b in witnesses {
                let expected = (0..=12).any(|v| admits(a, v * group) && admits(b, v));
                let lhs = with_witness(&[Some(2), Some(7), None], 2, a);
                let rhs = with_witness(&[Some(3), None, Some(6)], 1, b);
                assert_eq!(
                    infer_convolution_type(
                        &lhs,
                        &rhs,
                        &ty("tensor<f32>"),
                        ConvolutionOptions {
                            feature_group_count: group as i64,
                            ..ConvolutionOptions::new(&d)
                        }
                    )
                    .is_ok(),
                    expected,
                    "feature witnesses {a:?}, {b:?}, group {group}",
                );
                for c in witnesses {
                    let expected = (0..=12).any(|v| admits(a, v * group))
                        && (0..=12).any(|v| v % group == 0 && admits(b, v) && admits(c, v));
                    let lhs = with_witness(&[None, Some(7), Some(4)], 0, a);
                    let rhs = with_witness(&[Some(3), Some(4), None], 2, b);
                    let result = with_witness(&[None, Some(5), None], 2, c);
                    assert_eq!(
                        common::computation(
                            StableHloOp::Convolution,
                            vec![lhs, rhs],
                            vec![result],
                            attributes(&d, 1, group as i64)
                        )
                        .is_ok(),
                        expected,
                        "output witnesses {a:?}, {b:?}, {c:?}, group {group}",
                    );
                }
            }
        }
    }
}
