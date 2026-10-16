use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, ElementType, InferenceError, ScalarValue, StableHloOp, TensorType, Type,
    infer_all_to_all_types,
};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn attributes(split: i64, concat: i64, count: i64) -> Attributes {
    [
        ("split_dimension".into(), Attribute::integer(split)),
        ("concat_dimension".into(), Attribute::integer(concat)),
        ("split_count".into(), Attribute::integer(count)),
        (
            "replica_groups".into(),
            Attribute::DenseElements {
                ty: Box::new(TensorType::static_shape(ElementType::I64, [1, count as u64]).into()),
                values: (0..count)
                    .map(|id| ScalarValue::Integer(id.into()))
                    .collect(),
            },
        ),
    ]
    .into()
}

#[test]
fn all_to_all_infers_heterogeneous_variadic_results_and_each_operands_own_bounds() {
    let inputs = [
        ty("tensor<?x3xf32, #stablehlo.bounds<7, ?>>"),
        ty("tensor<4x?xi16, #stablehlo.bounds<?, 5>>"),
        ty("tensor<0x3x!quant.uniform<i8:f32, 0.25:1>>"),
    ];
    let expected = [
        ty("tensor<?x6xf32, #stablehlo.bounds<3, ?>>"),
        ty("tensor<2x?xi16, #stablehlo.bounds<?, 10>>"),
        ty("tensor<0x6x!quant.uniform<i8:f32, 0.25:1>>"),
    ];
    assert_eq!(infer_all_to_all_types(&inputs, 0, 1, 2).unwrap(), expected);
    common::computation(
        StableHloOp::AllToAll,
        inputs.to_vec(),
        expected.to_vec(),
        attributes(0, 1, 2),
    )
    .unwrap();
    assert!(infer_all_to_all_types(&[], 0, 0, 1).unwrap().is_empty());
}

#[test]
fn all_to_all_same_axis_preserves_extents_and_tightens_dynamic_bounds_to_multiples() {
    assert_eq!(
        infer_all_to_all_types(
            &[
                ty("tensor<6x3xf32>"),
                ty("tensor<?x3xf32, #stablehlo.bounds<7, ?>>"),
            ],
            0,
            0,
            2
        )
        .unwrap(),
        [
            ty("tensor<6x3xf32>"),
            ty("tensor<?x3xf32, #stablehlo.bounds<6, ?>>"),
        ],
    );
    assert!(infer_all_to_all_types(&[ty("tensor<7xf32>")], 0, 0, 2).is_err());
    assert_eq!(
        infer_all_to_all_types(&[ty("tensor<?xf32>")], 0, 0, 2).unwrap(),
        [ty("tensor<?xf32>")]
    );
}

#[test]
fn all_to_all_retains_unrelated_encoding_and_exact_quantized_elements() {
    let input: Type = TensorType::static_shape(ElementType::F32, [4, 3])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    let expected: Type = TensorType::static_shape(ElementType::F32, [2, 6])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_all_to_all_types(&[input], 0, 1, 2).unwrap(),
        [expected]
    );
    assert!(
        common::computation(
            StableHloOp::AllToAll,
            vec![ty("tensor<4x3x!quant.uniform<i8:f32, 0.25:1>>")],
            vec![ty("tensor<2x6x!quant.uniform<i8:f32, 0.5:1>>")],
            attributes(0, 1, 2),
        )
        .is_err()
    );
}

#[test]
fn all_to_all_rejects_bad_domains_axes_counts_and_nondivisible_extents() {
    assert!(infer_all_to_all_types(&[], -1, 0, 1).is_err());
    assert!(infer_all_to_all_types(&[], 0, -1, 1).is_err());
    for input in [
        "tensor<f32>",
        "tensor<*xf32>",
        "tensor<4x3xindex>",
        "!stablehlo.token",
        "tensor<4x3x!quant.uniform<i8:f32:1, {0.25,0.5,0.75}>>",
    ] {
        assert!(
            infer_all_to_all_types(&[ty(input)], 0, 1, 2).is_err(),
            "{input}"
        );
    }
    let input = ty("tensor<4x3xf32>");
    for (split, concat, count) in [
        (-1, 0, 2),
        (0, -1, 2),
        (2, 0, 2),
        (0, 2, 2),
        (0, 1, 0),
        (0, 1, -1),
        (1, 0, 2),
    ] {
        assert!(
            infer_all_to_all_types(std::slice::from_ref(&input), split, concat, count).is_err()
        );
    }
    assert!(matches!(
        infer_all_to_all_types(&[input], 1, 0, 2),
        Err(InferenceError::NonDivisibleSplit {
            dimension: 1,
            size: 3,
            count: 2
        })
    ));
}

#[test]
fn all_to_all_checks_extent_and_transformed_bound_overflow() {
    for input in [
        format!("tensor<2x{}xf32>", i64::MAX),
        format!("tensor<2x?xf32, #stablehlo.bounds<?, {}>>", i64::MAX),
    ] {
        assert!(matches!(
            infer_all_to_all_types(&[ty(&input)], 0, 1, 2),
            Err(InferenceError::DimensionOverflow { dimension: 1 })
        ));
    }
    assert_eq!(
        infer_all_to_all_types(&[ty("tensor<0x0xf32>")], 0, 1, i64::MAX).unwrap(),
        [ty("tensor<0x0xf32>")]
    );
}

#[test]
fn all_to_all_verifier_checks_transformed_bounds_and_output_concat_divisibility() {
    for (input, output, split, concat, accepted) in [
        (
            "tensor<?x3xf32, #stablehlo.bounds<7, ?>>",
            "tensor<3x6xf32>",
            0,
            1,
            true,
        ),
        (
            "tensor<?x3xf32, #stablehlo.bounds<7, ?>>",
            "tensor<4x6xf32>",
            0,
            1,
            false,
        ),
        (
            "tensor<4x?xf32, #stablehlo.bounds<?, 3>>",
            "tensor<2x6xf32>",
            0,
            1,
            true,
        ),
        (
            "tensor<4x?xf32, #stablehlo.bounds<?, 3>>",
            "tensor<2x8xf32>",
            0,
            1,
            false,
        ),
        ("tensor<4x?xf32>", "tensor<2x5xf32>", 0, 1, false),
        (
            "tensor<4x3xf32>",
            "tensor<?x6xf32, #stablehlo.bounds<1, ?>>",
            0,
            1,
            false,
        ),
        ("tensor<?x3xf32>", "tensor<5x3xf32>", 0, 0, false),
        (
            "tensor<?x3xf32, #stablehlo.bounds<7, ?>>",
            "tensor<6x3xf32>",
            0,
            0,
            true,
        ),
        (
            "tensor<?x3xf32, #stablehlo.bounds<7, ?>>",
            "tensor<8x3xf32>",
            0,
            0,
            false,
        ),
    ] {
        assert_eq!(
            common::computation(
                StableHloOp::AllToAll,
                vec![ty(input)],
                vec![ty(output)],
                attributes(split, concat, 2)
            )
            .is_ok(),
            accepted,
            "{input}, {output}, {split}, {concat}"
        );
    }
}

#[test]
fn all_to_all_small_shapes_and_axis_combinations_follow_split_then_concat() {
    for rows in 0..=6 {
        for columns in 0..=6 {
            for split in 0..2 {
                for concat in 0..2 {
                    for count in 1..=3 {
                        let shape = [rows, columns];
                        let input: Type = TensorType::static_shape(ElementType::F32, shape).into();
                        let result = infer_all_to_all_types(
                            &[input],
                            split as i64,
                            concat as i64,
                            count as i64,
                        );
                        if shape[split] % count != 0 {
                            assert!(result.is_err());
                        } else {
                            let mut expected = shape;
                            expected[split] /= count;
                            expected[concat] *= count;
                            assert_eq!(
                                result.unwrap(),
                                [TensorType::static_shape(ElementType::F32, expected).into()]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn all_to_all_bounded_witnesses_match_exact_divisible_extent_feasibility() {
    let witnesses = [
        (Some(0), None),
        (Some(2), None),
        (Some(3), None),
        (Some(4), None),
        (None, None),
        (None, Some(1)),
        (None, Some(3)),
    ];
    let admits =
        |w: (Option<u64>, Option<u64>), v| w.0.is_none_or(|s| s == v) && w.1.is_none_or(|b| v <= b);
    let tensor = |w: (Option<u64>, Option<u64>)| -> Type {
        let extent = w.0.map_or("?".into(), |n| n.to_string());
        ty(&match w.1 {
            Some(bound) => format!("tensor<{extent}xf32, #stablehlo.bounds<{bound}>>"),
            None => format!("tensor<{extent}xf32>"),
        })
    };
    for count in 1..=3 {
        for input in witnesses {
            for output in witnesses {
                let feasible =
                    (0..=12).any(|v| v % count == 0 && admits(input, v) && admits(output, v));
                assert_eq!(
                    common::computation(
                        StableHloOp::AllToAll,
                        vec![tensor(input)],
                        vec![tensor(output)],
                        attributes(0, 0, count as i64)
                    )
                    .is_ok(),
                    feasible,
                    "{input:?}, {output:?}, {count}",
                );
            }
        }
    }
}
