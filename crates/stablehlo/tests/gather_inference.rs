use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{
    Attribute, Attributes, GatherDimensionNumbers, StableHloAttribute, StableHloOp, Type,
    infer_dynamic_gather_type, infer_gather_type,
};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn row() -> GatherDimensionNumbers {
    GatherDimensionNumbers {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    }
}

fn attributes(d: GatherDimensionNumbers, sizes: Option<Vec<i64>>) -> Attributes {
    let mut a: Attributes = [(
        "dimension_numbers".into(),
        Attribute::StableHlo(Box::new(StableHloAttribute::GatherDimensionNumbers(d))),
    )]
    .into();
    if let Some(sizes) = sizes {
        a.insert("slice_sizes".into(), Attribute::integers(sizes));
    }
    a
}

#[test]
fn static_gather_propagates_only_index_batch_bounds_and_operand_elements() {
    let operand = ty("tensor<?x4xf32, #stablehlo.bounds<5, ?>>");
    let indices = ty("tensor<?x1xi64, #stablehlo.bounds<7, ?>>");
    assert_eq!(
        infer_gather_type(&operand, &indices, &row(), &[1, 4]).unwrap(),
        ty("tensor<?x4xf32, #stablehlo.bounds<7, ?>>"),
    );
    assert_eq!(
        infer_gather_type(
            &operand,
            &ty("tensor<?xi32, #stablehlo.bounds<7>>"),
            &row(),
            &[1, 4]
        )
        .unwrap(),
        ty("tensor<?x4xf32, #stablehlo.bounds<7, ?>>"),
    );
    let quantized = ty("tensor<5x4x!quant.uniform<i8:f32, 0.25:1>>");
    assert_eq!(
        infer_gather_type(&quantized, &indices, &row(), &[1, 4]).unwrap(),
        ty("tensor<?x4x!quant.uniform<i8:f32, 0.25:1>, #stablehlo.bounds<7, ?>>"),
    );
}

#[test]
fn dynamic_gather_has_unknown_offset_extents_and_preserves_index_bounds() {
    assert_eq!(
        infer_dynamic_gather_type(
            &ty("tensor<5x4xf32>"),
            &ty("tensor<?x1xi64, #stablehlo.bounds<7, ?>>"),
            &ty("tensor<2xi32>"),
            &row(),
        )
        .unwrap(),
        ty("tensor<?x?xf32, #stablehlo.bounds<7, ?>>"),
    );
    for sizes in [
        "tensor<1xi64>",
        "tensor<?xi64>",
        "tensor<2xf32>",
        "tensor<1x2xi64>",
        "!stablehlo.token",
    ] {
        assert!(
            infer_dynamic_gather_type(
                &ty("tensor<5x4xf32>"),
                &ty("tensor<2x1xi64>"),
                &ty(sizes),
                &row(),
            )
            .is_err(),
            "{sizes}"
        );
    }
}

#[test]
fn gather_checks_dynamic_operand_and_index_vector_bounds() {
    for (operand, indices, sizes) in [
        (
            "tensor<?x4xf32, #stablehlo.bounds<0, ?>>",
            "tensor<2x1xi64>",
            vec![1, 4],
        ),
        (
            "tensor<5x?xf32, #stablehlo.bounds<?, 3>>",
            "tensor<2x1xi64>",
            vec![1, 4],
        ),
        (
            "tensor<5x4xf32>",
            "tensor<2x?xi64, #stablehlo.bounds<?, 0>>",
            vec![1, 4],
        ),
        ("tensor<5x4xf32>", "tensor<2x1xi64>", vec![-1, 4]),
    ] {
        assert!(infer_gather_type(&ty(operand), &ty(indices), &row(), &sizes).is_err());
    }
    assert!(
        infer_gather_type(
            &ty("tensor<0x4xf32>"),
            &ty("tensor<0x1xi64>"),
            &row(),
            &[0, 4],
        )
        .is_ok()
    );
}

#[test]
fn gather_bounds_constrain_declared_results_for_static_and_dynamic_forms() {
    for opcode in [StableHloOp::Gather, StableHloOp::DynamicGather] {
        for (size, accepted) in [(0, true), (3, true), (4, false)] {
            let mut inputs = vec![
                ty("tensor<5x4xf32>"),
                ty("tensor<?x1xi64, #stablehlo.bounds<3, ?>>"),
            ];
            if opcode == StableHloOp::DynamicGather {
                inputs.push(ty("tensor<2xi64>"));
            }
            let output = ty(&format!("tensor<{size}x4xf32>"));
            assert_eq!(
                common::computation(
                    opcode,
                    inputs,
                    vec![output],
                    attributes(row(), (opcode == StableHloOp::Gather).then(|| vec![1, 4])),
                )
                .is_ok(),
                accepted,
                "{opcode:?}, {size}",
            );
        }
    }
}

#[test]
fn gather_batching_checks_input_indices_and_declared_result_together() {
    let d = GatherDimensionNumbers {
        offset_dims: vec![2],
        collapsed_slice_dims: vec![1],
        operand_batching_dims: vec![0],
        start_indices_batching_dims: vec![0],
        start_index_map: vec![1],
        index_vector_dim: 2,
    };
    for (input, indices, output, accepted) in [
        (
            "tensor<?x5x4xf32, #stablehlo.bounds<3, ?, ?>>",
            "tensor<4x2x1xi64>",
            "tensor<4x2x4xf32>",
            false,
        ),
        (
            "tensor<4x5x4xf32>",
            "tensor<?x2x1xi64, #stablehlo.bounds<3, ?, ?>>",
            "tensor<4x2x4xf32>",
            false,
        ),
        (
            "tensor<?x5x4xf32, #stablehlo.bounds<3, ?, ?>>",
            "tensor<?x2x1xi64>",
            "tensor<4x2x4xf32>",
            false,
        ),
        (
            "tensor<4x5x4xf32>",
            "tensor<?x2x1xi64>",
            "tensor<3x2x4xf32>",
            false,
        ),
        (
            "tensor<4x5x4xf32>",
            "tensor<?x2x1xi64>",
            "tensor<?x2x4xf32, #stablehlo.bounds<3, ?, ?>>",
            false,
        ),
        (
            "tensor<?x5x4xf32, #stablehlo.bounds<3, ?, ?>>",
            "tensor<?x2x1xi64>",
            "tensor<3x2x4xf32>",
            true,
        ),
    ] {
        for opcode in [StableHloOp::Gather, StableHloOp::DynamicGather] {
            let mut inputs = vec![ty(input), ty(indices)];
            if opcode == StableHloOp::DynamicGather {
                inputs.push(ty("tensor<3xi64>"));
            }
            assert_eq!(
                common::computation(
                    opcode,
                    inputs,
                    vec![ty(output)],
                    attributes(
                        d.clone(),
                        (opcode == StableHloOp::Gather).then(|| vec![1, 1, 4])
                    ),
                )
                .is_ok(),
                accepted,
                "{opcode:?}: {input}, {indices}, {output}",
            );
        }
    }
}

#[test]
fn gather_requires_supported_operand_index_and_exact_quantized_result_elements() {
    for source in [
        "tensor<5x4xindex>",
        "tensor<5x4xtf32>",
        "tensor<*xf32>",
        "!stablehlo.token",
    ] {
        assert!(infer_gather_type(&ty(source), &ty("tensor<2x1xi64>"), &row(), &[1, 4]).is_err());
    }
    for source in [
        "tensor<2x1xf32>",
        "tensor<2x1xi1>",
        "tensor<*xi64>",
        "tensor<2x1xindex>",
    ] {
        assert!(infer_gather_type(&ty("tensor<5x4xf32>"), &ty(source), &row(), &[1, 4]).is_err());
    }
    for (scale, accepted) in [("0.25", true), ("0.5", false)] {
        assert_eq!(
            common::computation(
                StableHloOp::Gather,
                vec![
                    ty("tensor<5x4x!quant.uniform<i8:f32, 0.25:1>>"),
                    ty("tensor<2x1xi64>")
                ],
                vec![ty(&format!(
                    "tensor<2x4x!quant.uniform<i8:f32, {scale}:1>>"
                ))],
                attributes(row(), Some(vec![1, 4])),
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn constant_dynamic_gather_uses_shared_slice_bound_checks_and_roundtrips() {
    for (slice, accepted) in [(3, true), (4, false)] {
        let source = format!(
            r#"
            func.func @g(%input: tensor<5x?xf32, #stablehlo.bounds<?, 3>>,
                         %indices: tensor<2x1xi64>) -> tensor<2x{slice}xf32> {{
              %sizes = stablehlo.constant dense<[1, {slice}]> : tensor<2xi64>
              %r = "stablehlo.dynamic_gather"(%input, %indices, %sizes) {{
                dimension_numbers = #stablehlo.gather<offset_dims = [1],
                  collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
              }} : (tensor<5x?xf32, #stablehlo.bounds<?, 3>>, tensor<2x1xi64>, tensor<2xi64>)
                   -> tensor<2x{slice}xf32>
              return %r : tensor<2x{slice}xf32>
            }}
        "#
        );
        let parsed = parse_module(&source);
        assert_eq!(parsed.is_ok(), accepted, "{parsed:?}");
        if let Ok(module) = parsed {
            assert_eq!(
                parse_module(&print_module(&module).unwrap()).unwrap(),
                module
            );
        }
    }
}

#[test]
fn dynamic_gather_declared_offset_extents_must_fit_operand_sizes_and_bounds() {
    for operand in [
        "tensor<5x3xf32>",
        "tensor<5x?xf32, #stablehlo.bounds<?, 3>>",
    ] {
        for (extent, accepted) in [(0, true), (3, true), (4, false)] {
            assert_eq!(
                common::computation(
                    StableHloOp::DynamicGather,
                    vec![ty(operand), ty("tensor<2x1xi64>"), ty("tensor<2xi64>")],
                    vec![ty(&format!("tensor<2x{extent}xf32>"))],
                    attributes(row(), None),
                )
                .is_ok(),
                accepted,
                "{operand}, {extent}"
            );
        }
    }
}

#[test]
fn gather_all_offset_positions_and_explicit_vector_positions_reconstruct_shapes_and_bounds() {
    let operand = ty("tensor<5x7x9xf32>");
    for collapsed in 0..3 {
        for vector in 0..=2 {
            for offset0 in 0..4 {
                for offset1 in offset0 + 1..4 {
                    let mut index_shape = vec!["?".to_owned(), "3".to_owned()];
                    let mut index_bounds = vec!["6".to_owned(), "?".to_owned()];
                    index_shape.insert(vector, "1".into());
                    index_bounds.insert(vector, "?".into());
                    let indices = ty(&format!(
                        "tensor<{}xi64, #stablehlo.bounds<{}>>",
                        index_shape.join("x"),
                        index_bounds.join(", "),
                    ));
                    let d = GatherDimensionNumbers {
                        offset_dims: vec![offset0 as i64, offset1 as i64],
                        collapsed_slice_dims: vec![collapsed as i64],
                        start_index_map: vec![collapsed as i64],
                        index_vector_dim: vector as i64,
                        ..Default::default()
                    };
                    let mut sizes = vec![2, 4, 6];
                    sizes[collapsed] = 1;
                    let slice_shape = sizes
                        .iter()
                        .enumerate()
                        .filter(|&(axis, _)| axis != collapsed)
                        .map(|(_, size)| size.to_string())
                        .collect::<Vec<_>>();
                    let mut expected_shape = vec!["?".into(), "3".into()];
                    let mut expected_bounds = vec!["6".to_owned(), "?".to_owned()];
                    for (axis, size) in [offset0, offset1].into_iter().zip(slice_shape) {
                        expected_shape.insert(axis, size);
                        expected_bounds.insert(axis, "?".into());
                    }
                    let expected = ty(&format!(
                        "tensor<{}xf32, #stablehlo.bounds<{}>>",
                        expected_shape.join("x"),
                        expected_bounds.join(", "),
                    ));
                    assert_eq!(
                        infer_gather_type(&operand, &indices, &d, &sizes).unwrap(),
                        expected,
                        "{d:?}",
                    );
                }
            }
        }
    }
}

#[test]
fn gather_small_batching_witnesses_match_exact_three_way_extent_feasibility() {
    let witnesses = [
        (Some(0), None),
        (Some(1), None),
        (Some(2), None),
        (None, None),
        (None, Some(0)),
        (None, Some(1)),
        (None, Some(2)),
    ];
    let tensor = |(size, bound): (Option<u64>, Option<u64>), suffix: &str| {
        let size = size.map_or_else(|| "?".into(), |value| value.to_string());
        let encoding = bound.map_or_else(String::new, |value| {
            format!(", #stablehlo.bounds<{value}, ?, ?>")
        });
        ty(&format!("tensor<{size}x{suffix}{encoding}>"))
    };
    let d = GatherDimensionNumbers {
        offset_dims: vec![2],
        collapsed_slice_dims: vec![1],
        operand_batching_dims: vec![0],
        start_indices_batching_dims: vec![0],
        start_index_map: vec![1],
        index_vector_dim: 2,
    };
    for input in witnesses {
        for indices in witnesses {
            for output in witnesses {
                let expected = (0..=2).any(|extent| {
                    [input, indices, output].iter().all(|&(size, bound)| {
                        size.is_none_or(|size| size == extent)
                            && bound.is_none_or(|bound| extent <= bound)
                    })
                });
                for opcode in [StableHloOp::Gather, StableHloOp::DynamicGather] {
                    let mut inputs = vec![tensor(input, "5x4xf32"), tensor(indices, "2x1xi64")];
                    if opcode == StableHloOp::DynamicGather {
                        inputs.push(ty("tensor<3xi64>"));
                    }
                    assert_eq!(
                        common::computation(
                            opcode,
                            inputs,
                            vec![tensor(output, "2x4xf32")],
                            attributes(
                                d.clone(),
                                (opcode == StableHloOp::Gather).then(|| vec![0, 1, 4])
                            ),
                        )
                        .is_ok(),
                        expected,
                        "{opcode:?}, {input:?}, {indices:?}, {output:?}",
                    );
                }
            }
        }
    }
}

#[test]
fn gather_supports_scalar_operands_empty_index_vectors_and_scalar_results() {
    let d = GatherDimensionNumbers {
        index_vector_dim: 0,
        ..Default::default()
    };
    assert_eq!(
        infer_gather_type(&ty("tensor<f32>"), &ty("tensor<0xi64>"), &d, &[]).unwrap(),
        ty("tensor<f32>"),
    );
    assert_eq!(
        infer_dynamic_gather_type(
            &ty("tensor<f32>"),
            &ty("tensor<0xi64>"),
            &ty("tensor<0xi64>"),
            &d,
        )
        .unwrap(),
        ty("tensor<f32>"),
    );
    let d = GatherDimensionNumbers {
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 0,
        ..Default::default()
    };
    assert_eq!(
        infer_gather_type(&ty("tensor<5xf32>"), &ty("tensor<i64>"), &d, &[1]).unwrap(),
        ty("tensor<f32>"),
    );
    assert!(
        infer_gather_type(
            &ty("tensor<5xf32>"),
            &ty("tensor<i64>"),
            &GatherDimensionNumbers {
                start_index_map: vec![],
                ..d
            },
            &[1],
        )
        .is_err()
    );
}

#[test]
fn gather_drops_bounds_removed_with_the_index_vector_axis() {
    let operand = ty("tensor<5x4xf32>");
    assert_eq!(
        infer_gather_type(
            &operand,
            &ty("tensor<2x?xi64, #stablehlo.bounds<?, 1>>"),
            &row(),
            &[1, 4],
        )
        .unwrap(),
        ty("tensor<2x4xf32>"),
    );
    let scalar_indices = stablehlo::TensorType::static_shape(stablehlo::ElementType::I64, [0])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "encoding".into(),
        });
    let d = GatherDimensionNumbers {
        index_vector_dim: 0,
        ..Default::default()
    };
    let result = infer_gather_type(&ty("tensor<f32>"), &scalar_indices.into(), &d, &[]).unwrap();
    assert_eq!(
        result,
        stablehlo::TensorType::scalar(stablehlo::ElementType::F32)
            .with_encoding(Attribute::Opaque {
                dialect: Some("consumer".into()),
                syntax: "encoding".into(),
            })
            .into(),
    );
}

#[test]
fn gather_rejects_every_invalid_dimension_map_category() {
    let base = GatherDimensionNumbers {
        offset_dims: vec![2],
        collapsed_slice_dims: vec![1],
        operand_batching_dims: vec![0],
        start_indices_batching_dims: vec![0],
        start_index_map: vec![1],
        index_vector_dim: 2,
    };
    let bad = [
        GatherDimensionNumbers {
            offset_dims: vec![3],
            ..base.clone()
        },
        GatherDimensionNumbers {
            offset_dims: vec![-1],
            ..base.clone()
        },
        GatherDimensionNumbers {
            collapsed_slice_dims: vec![0],
            ..base.clone()
        },
        GatherDimensionNumbers {
            collapsed_slice_dims: vec![3],
            ..base.clone()
        },
        GatherDimensionNumbers {
            operand_batching_dims: vec![1],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_indices_batching_dims: vec![2],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_indices_batching_dims: vec![3],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_indices_batching_dims: vec![],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_index_map: vec![0],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_index_map: vec![3],
            ..base.clone()
        },
        GatherDimensionNumbers {
            index_vector_dim: -1,
            ..base.clone()
        },
        GatherDimensionNumbers {
            index_vector_dim: 4,
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_index_map: vec![1, 1],
            ..base.clone()
        },
        GatherDimensionNumbers {
            offset_dims: vec![2, 2],
            ..base.clone()
        },
    ];
    let operand = ty("tensor<2x5x4xf32>");
    let indices = ty("tensor<2x3x1xi64>");
    assert!(infer_gather_type(&operand, &indices, &base, &[1, 1, 4]).is_ok());
    for d in bad {
        assert!(
            infer_gather_type(&operand, &indices, &d, &[1, 1, 4]).is_err(),
            "{d:?}"
        );
    }
    for sizes in [
        vec![1, 1],
        vec![1, 1, 4, 1],
        vec![2, 1, 4],
        vec![1, 2, 4],
        vec![1, 1, 5],
    ] {
        assert!(
            infer_gather_type(&operand, &indices, &base, &sizes).is_err(),
            "{sizes:?}"
        );
    }
}
