use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, Instruction, Operation, Region,
    ScatterDimensionNumbers, StableHloAttribute, StableHloOp, Type, ValueId, infer_scatter_types,
};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn row() -> ScatterDimensionNumbers {
    ScatterDimensionNumbers {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    }
}

fn batched() -> ScatterDimensionNumbers {
    ScatterDimensionNumbers {
        input_batching_dims: vec![0],
        scatter_indices_batching_dims: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    }
}

fn verify(
    inputs: Vec<Type>,
    indices: Type,
    updates: Vec<Type>,
    results: Vec<Type>,
    accumulators: &[Type],
    d: &ScatterDimensionNumbers,
) -> Result<stablehlo::Computation, stablehlo::BuildError> {
    let arguments = accumulators
        .iter()
        .chain(accumulators)
        .enumerate()
        .map(|(i, t)| BlockArgument::new(ValueId(100 + i as u32), t.clone()))
        .collect();
    let region = Region::new(vec![Block::new(
        arguments,
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            (0..accumulators.len())
                .map(|i| ValueId(100 + i as u32))
                .collect(),
            vec![],
            vec![],
        )],
    )]);
    let attributes = [(
        "scatter_dimension_numbers".into(),
        Attribute::StableHlo(Box::new(StableHloAttribute::ScatterDimensionNumbers(
            d.clone(),
        ))),
    )]
    .into();
    let operands = inputs.into_iter().chain([indices]).chain(updates).collect();
    common::computation_with_regions(
        StableHloOp::Scatter,
        operands,
        results,
        attributes,
        vec![region],
    )
}

#[test]
fn scatter_retains_each_input_shape_encoding_and_corresponding_scalar_return_elements() {
    let inputs = [
        ty("tensor<?x4xf32, #stablehlo.bounds<5, ?>>"),
        ty("tensor<5x?xi8, #stablehlo.bounds<?, 4>>"),
    ];
    let updates = [ty("tensor<2x4xf32>"), ty("tensor<2x4xi8>")];
    let accumulators = [ty("tensor<f64>"), ty("tensor<i32>")];
    let expected = [
        ty("tensor<?x4xf64, #stablehlo.bounds<5, ?>>"),
        ty("tensor<5x?xi32, #stablehlo.bounds<?, 4>>"),
    ];
    assert_eq!(
        infer_scatter_types(
            &inputs,
            &ty("tensor<2x1xi64>"),
            &updates,
            &row(),
            &accumulators
        )
        .unwrap(),
        expected
    );
    verify(
        inputs.to_vec(),
        ty("tensor<2x1xi64>"),
        updates.to_vec(),
        expected.to_vec(),
        &accumulators,
        &row(),
    )
    .unwrap();
    let encoded: Type = stablehlo::TensorType::static_shape(stablehlo::ElementType::F32, [5, 4])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_scatter_types(
            std::slice::from_ref(&encoded),
            &ty("tensor<2x1xi64>"),
            &[ty("tensor<2x4xf32>")],
            &row(),
            &[ty("tensor<f32>")]
        )
        .unwrap(),
        [encoded]
    );
}

#[test]
fn scatter_retains_exact_quantized_metadata_and_rejects_mismatched_update_parameters() {
    let input = ty("tensor<5x4x!quant.uniform<i8:f32, 0.25:1>>");
    let update = ty("tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>");
    let accumulator = ty("tensor<!quant.uniform<i8:f32, 0.5:2>>");
    let expected = ty("tensor<5x4x!quant.uniform<i8:f32, 0.5:2>>");
    assert_eq!(
        infer_scatter_types(
            std::slice::from_ref(&input),
            &ty("tensor<2x1xi32>"),
            std::slice::from_ref(&update),
            &row(),
            std::slice::from_ref(&accumulator)
        )
        .unwrap(),
        std::slice::from_ref(&expected)
    );
    verify(
        vec![input.clone()],
        ty("tensor<2x1xi32>"),
        vec![update],
        vec![expected],
        &[accumulator],
        &row(),
    )
    .unwrap();
    assert!(
        infer_scatter_types(
            &[input],
            &ty("tensor<2x1xi32>"),
            &[ty("tensor<2x4x!quant.uniform<i8:f32, 0.5:1>>")],
            &row(),
            &[ty("tensor<f32>")]
        )
        .is_err()
    );
}

#[test]
fn scatter_handles_scalar_operands_empty_index_vectors_and_implicit_vectors() {
    for (input, indices, update, d) in [
        (
            "tensor<f32>",
            "tensor<2x0xi64>",
            "tensor<2xf32>",
            ScatterDimensionNumbers {
                index_vector_dim: 1,
                ..Default::default()
            },
        ),
        ("tensor<5x4xf32>", "tensor<2xi64>", "tensor<2x4xf32>", row()),
        (
            "tensor<5xf32>",
            "tensor<0xi64>",
            "tensor<0xf32>",
            ScatterDimensionNumbers {
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..Default::default()
            },
        ),
    ] {
        assert_eq!(
            infer_scatter_types(
                &[ty(input)],
                &ty(indices),
                &[ty(update)],
                &d,
                &[ty("tensor<f32>")]
            )
            .unwrap(),
            [ty(input)]
        );
        verify(
            vec![ty(input)],
            ty(indices),
            vec![ty(update)],
            vec![ty(input)],
            &[ty("tensor<f32>")],
            &d,
        )
        .unwrap();
    }
}

#[test]
fn scatter_checks_index_vector_cardinality_against_dynamic_bounds() {
    for (indices, accepted) in [
        ("tensor<2x?xi64, #stablehlo.bounds<?, 0>>", false),
        ("tensor<2x?xi64, #stablehlo.bounds<?, 1>>", true),
        ("tensor<2x2xi64>", false),
    ] {
        assert_eq!(
            infer_scatter_types(
                &[ty("tensor<5x4xf32>")],
                &ty(indices),
                &[ty("tensor<2x4xf32>")],
                &row(),
                &[ty("tensor<f32>")]
            )
            .is_ok(),
            accepted,
            "{indices}"
        );
    }
}

#[test]
fn scatter_checks_all_input_update_and_declared_result_bounds_in_both_directions() {
    for (input, indices, update, output, accepted) in [
        (
            "tensor<5x?xf32, #stablehlo.bounds<?, 3>>",
            "tensor<2x1xi64>",
            "tensor<2x4xf32>",
            "tensor<5x?xf32>",
            false,
        ),
        (
            "tensor<5x?xf32>",
            "tensor<2x1xi64>",
            "tensor<2x4xf32>",
            "tensor<5x?xf32, #stablehlo.bounds<?, 3>>",
            false,
        ),
        (
            "tensor<5x?xf32>",
            "tensor<2x1xi64>",
            "tensor<2x4xf32>",
            "tensor<5x?xf32, #stablehlo.bounds<?, 4>>",
            true,
        ),
        (
            "tensor<5x4xf32>",
            "tensor<?x1xi64, #stablehlo.bounds<1, ?>>",
            "tensor<2x4xf32>",
            "tensor<5x4xf32>",
            false,
        ),
        (
            "tensor<5x4xf32>",
            "tensor<2x1xi64>",
            "tensor<?x4xf32, #stablehlo.bounds<1, ?>>",
            "tensor<5x4xf32>",
            false,
        ),
        (
            "tensor<5x4xf32>",
            "tensor<2x1xi64>",
            "tensor<?x?xf32, #stablehlo.bounds<2, 8>>",
            "tensor<5x4xf32>",
            true,
        ),
        (
            "tensor<5x?xf32, #stablehlo.bounds<?, 3>>",
            "tensor<2x1xi64>",
            "tensor<2x?xf32, #stablehlo.bounds<?, 8>>",
            "tensor<5x?xf32>",
            true,
        ),
    ] {
        assert_eq!(
            verify(
                vec![ty(input)],
                ty(indices),
                vec![ty(update)],
                vec![ty(output)],
                &[ty("tensor<f32>")],
                &row()
            )
            .is_ok(),
            accepted,
            "{input}, {indices}, {update}, {output}"
        );
    }
}

#[test]
fn scatter_rejects_bad_counts_domains_accumulators_and_dimension_maps() {
    let input = ty("tensor<5x4xf32>");
    let indices = ty("tensor<2x1xi64>");
    let update = ty("tensor<2x4xf32>");
    let scalar = ty("tensor<f32>");
    assert!(infer_scatter_types(&[], &indices, &[], &row(), &[]).is_err());
    assert!(
        infer_scatter_types(
            std::slice::from_ref(&input),
            &indices,
            &[],
            &row(),
            std::slice::from_ref(&scalar)
        )
        .is_err()
    );
    assert!(
        infer_scatter_types(
            std::slice::from_ref(&input),
            &indices,
            std::slice::from_ref(&update),
            &row(),
            &[]
        )
        .is_err()
    );
    for accumulator in [
        "tensor<1xf32>",
        "tensor<index>",
        "tensor<tf32>",
        "!stablehlo.token",
    ] {
        assert!(
            infer_scatter_types(
                std::slice::from_ref(&input),
                &indices,
                std::slice::from_ref(&update),
                &row(),
                &[ty(accumulator)]
            )
            .is_err()
        );
    }
    for bad in [
        "tensor<2x1xf32>",
        "tensor<*xi64>",
        "!stablehlo.token",
        "tensor<2x1x!quant.uniform<i8:f32, 0.5>>",
    ] {
        assert!(
            infer_scatter_types(
                std::slice::from_ref(&input),
                &ty(bad),
                std::slice::from_ref(&update),
                &row(),
                std::slice::from_ref(&scalar)
            )
            .is_err()
        );
    }
    for d in [
        ScatterDimensionNumbers {
            update_window_dims: vec![1, 1],
            ..row()
        },
        ScatterDimensionNumbers {
            update_window_dims: vec![-1],
            ..row()
        },
        ScatterDimensionNumbers {
            inserted_window_dims: vec![2],
            ..row()
        },
        ScatterDimensionNumbers {
            scatter_dims_to_operand_dims: vec![0, 0],
            ..row()
        },
        ScatterDimensionNumbers {
            index_vector_dim: -1,
            ..row()
        },
        ScatterDimensionNumbers {
            index_vector_dim: 3,
            ..row()
        },
        ScatterDimensionNumbers {
            input_batching_dims: vec![0],
            scatter_indices_batching_dims: vec![0],
            ..row()
        },
    ] {
        assert!(
            infer_scatter_types(
                std::slice::from_ref(&input),
                &indices,
                std::slice::from_ref(&update),
                &d,
                std::slice::from_ref(&scalar)
            )
            .is_err(),
            "{d:?}"
        );
    }
}

#[test]
fn scatter_dimension_maps_cover_each_inserted_vector_and_window_position() {
    let mut cases = 0;
    for inserted in 0..3 {
        for vector in 0..3 {
            for window in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] {
                let input = [5, 6, 7];
                let mut index = [2, 3, 0];
                let mut index_shape = vec![2, 3];
                index_shape.insert(vector, 1);
                index.copy_from_slice(&index_shape);
                let retained = (0..3).filter(|&i| i != inserted).collect::<Vec<_>>();
                let mut update = [0; 4];
                update[window[0]] = input[retained[0]] - 1;
                update[window[1]] = input[retained[1]] - 1;
                let scatter_axes = (0..4).filter(|i| !window.contains(i)).collect::<Vec<_>>();
                update[scatter_axes[0]] = 2;
                update[scatter_axes[1]] = 3;
                let tensor = |shape: &[u64], element| -> Type {
                    stablehlo::TensorType::static_shape(element, shape.iter().copied()).into()
                };
                let d = ScatterDimensionNumbers {
                    inserted_window_dims: vec![inserted as i64],
                    update_window_dims: window.into_iter().map(|a| a as i64).collect(),
                    scatter_dims_to_operand_dims: vec![inserted as i64],
                    index_vector_dim: vector as i64,
                    ..Default::default()
                };
                let input = tensor(&input, stablehlo::ElementType::F32);
                let indices = tensor(&index, stablehlo::ElementType::I64);
                let updates = tensor(&update, stablehlo::ElementType::F32);
                assert_eq!(
                    infer_scatter_types(
                        std::slice::from_ref(&input),
                        &indices,
                        std::slice::from_ref(&updates),
                        &d,
                        &[ty("tensor<f32>")]
                    )
                    .unwrap(),
                    std::slice::from_ref(&input)
                );
                verify(
                    vec![input.clone()],
                    indices,
                    vec![updates],
                    vec![input],
                    &[ty("tensor<f32>")],
                    &d,
                )
                .unwrap();
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 54);
}

type Witness = (Option<u64>, Option<u64>);
const WITNESSES: [Witness; 7] = [
    (Some(0), None),
    (Some(1), None),
    (Some(2), None),
    (Some(3), None),
    (None, None),
    (None, Some(1)),
    (None, Some(2)),
];

fn admits(w: Witness, v: u64) -> bool {
    w.0.is_none_or(|s| s == v) && w.1.is_none_or(|b| v <= b)
}

#[test]
fn variadic_scatter_rejects_jointly_inconsistent_input_and_update_witnesses() {
    for (inputs, updates) in [
        (
            ["tensor<?x4xf32>", "tensor<2x4xf32>", "tensor<3x4xf32>"],
            ["tensor<2x4xf32>", "tensor<2x4xf32>", "tensor<2x4xf32>"],
        ),
        (
            [
                "tensor<?x4xf32>",
                "tensor<?x4xf32, #stablehlo.bounds<2, ?>>",
                "tensor<3x4xf32>",
            ],
            ["tensor<2x4xf32>", "tensor<2x4xf32>", "tensor<2x4xf32>"],
        ),
        (
            ["tensor<5x4xf32>", "tensor<5x4xf32>", "tensor<5x4xf32>"],
            ["tensor<2x?xf32>", "tensor<2x3xf32>", "tensor<2x4xf32>"],
        ),
        (
            ["tensor<5x4xf32>", "tensor<5x4xf32>", "tensor<5x4xf32>"],
            [
                "tensor<2x?xf32>",
                "tensor<2x?xf32, #stablehlo.bounds<?, 2>>",
                "tensor<2x3xf32>",
            ],
        ),
    ] {
        assert!(
            infer_scatter_types(
                &inputs.map(ty),
                &ty("tensor<2x1xi64>"),
                &updates.map(ty),
                &row(),
                &[ty("tensor<f32>"), ty("tensor<f32>"), ty("tensor<f32>")]
            )
            .is_err()
        );
    }
    let input = ty("tensor<5x4xf32>");
    assert!(
        infer_scatter_types(
            &[input],
            &ty("tensor<2x1xi64>"),
            &[ty("tensor<2x4xi32>")],
            &row(),
            &[ty("tensor<f32>")]
        )
        .is_err()
    );
}

fn witness_tensor(w: Witness, suffix: &str) -> Type {
    let extent = w.0.map_or("?".into(), |n| n.to_string());
    ty(&match w.1 {
        Some(bound) => format!(
            "tensor<{extent}{suffix}, #stablehlo.bounds<{bound}{}>>",
            if suffix == "x0xi64" { ", ?" } else { "" }
        ),
        None => format!("tensor<{extent}{suffix}>"),
    })
}

#[test]
fn scatter_batched_input_index_update_and_result_witnesses_match_exact_feasibility() {
    for input in WITNESSES {
        for index in WITNESSES {
            for update in WITNESSES {
                let inputs = [witness_tensor(input, "xf32")];
                let indices = witness_tensor(index, "x0xi64");
                let updates = [witness_tensor(update, "xf32")];
                let expected =
                    (0..=4).any(|v| admits(input, v) && admits(index, v) && admits(update, v));
                assert_eq!(
                    infer_scatter_types(
                        &inputs,
                        &indices,
                        &updates,
                        &batched(),
                        &[ty("tensor<f32>")]
                    )
                    .is_ok(),
                    expected,
                    "{input:?}, {index:?}, {update:?}"
                );
                for result in WITNESSES {
                    let expected = (0..=4).any(|v| {
                        admits(input, v)
                            && admits(index, v)
                            && admits(update, v)
                            && admits(result, v)
                    });
                    assert_eq!(
                        verify(
                            inputs.to_vec(),
                            indices.clone(),
                            updates.to_vec(),
                            vec![witness_tensor(result, "xf32")],
                            &[ty("tensor<f32>")],
                            &batched()
                        )
                        .is_ok(),
                        expected,
                        "{input:?}, {index:?}, {update:?}, {result:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn scatter_window_inequality_combines_all_declared_input_and_result_witnesses() {
    let d = ScatterDimensionNumbers {
        update_window_dims: vec![1],
        index_vector_dim: 1,
        ..Default::default()
    };
    for input in WITNESSES {
        for result in WITNESSES {
            for update in WITNESSES {
                let update_type = match update.1 {
                    Some(b) => ty(&format!("tensor<2x?xf32, #stablehlo.bounds<?, {b}>>")),
                    None => ty(&format!(
                        "tensor<2x{}xf32>",
                        update.0.map_or("?".into(), |n| n.to_string())
                    )),
                };
                let expected = (0..=4).any(|v| {
                    admits(input, v) && admits(result, v) && (0..=v).any(|u| admits(update, u))
                });
                assert_eq!(
                    verify(
                        vec![witness_tensor(input, "xf32")],
                        ty("tensor<2x0xi64>"),
                        vec![update_type],
                        vec![witness_tensor(result, "xf32")],
                        &[ty("tensor<f32>")],
                        &d
                    )
                    .is_ok(),
                    expected,
                    "{input:?}, {result:?}, {update:?}"
                );
            }
        }
    }
}
