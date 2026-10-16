use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, BuildError, ElementType, InferenceError,
    Instruction, Operation, QuantizedTensorType, Region, StableHloAttribute, StableHloOp,
    TensorType, Type, TypeExtensions, ValueId, infer_reduce_types,
};

mod common;

fn tensor(element: ElementType, shape: Vec<Option<u64>>, bounds: Option<Vec<i64>>) -> Type {
    let mut tensor = TensorType::new(element, shape);
    if let Some(bounds) = bounds {
        tensor = tensor.with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
        )));
    }
    tensor.into()
}

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

#[test]
fn reduction_checks_all_input_shape_and_bound_witnesses_even_when_reduced_away() {
    for inputs in [
        vec![
            tensor(ElementType::F32, vec![None], None),
            tensor(ElementType::I32, vec![Some(3)], None),
            tensor(ElementType::I8, vec![Some(4)], None),
        ],
        vec![
            tensor(ElementType::F32, vec![None], Some(vec![3])),
            tensor(ElementType::I32, vec![Some(4)], None),
        ],
        vec![
            tensor(ElementType::F32, vec![Some(4)], None),
            tensor(ElementType::I32, vec![None], Some(vec![3])),
        ],
    ] {
        let accumulators = vec![scalar(ElementType::F32); inputs.len()];
        for dimensions in [vec![], vec![0]] {
            assert!(
                infer_reduce_types(&inputs, &dimensions, &accumulators).is_err(),
                "{inputs:?}, {dimensions:?}",
            );
        }
    }
}

#[test]
fn reduction_input_and_accumulator_domains_follow_the_operation_schema() {
    for source in [
        "tensor<2xindex>",
        "tensor<2xtf32>",
        "tensor<2xsi32>",
        "tensor<*xf32>",
    ] {
        let input = stablehlo::text::parse_type(source).unwrap();
        assert!(
            infer_reduce_types(&[input], &[0], &[scalar(ElementType::F32)]).is_err(),
            "{source}"
        );
    }
    for source in ["tensor<index>", "tensor<tf32>", "tensor<si32>"] {
        let accumulator = stablehlo::text::parse_type(source).unwrap();
        assert!(
            infer_reduce_types(
                &[tensor(ElementType::F32, vec![Some(2)], None)],
                &[0],
                &[accumulator],
            )
            .is_err(),
            "{source}",
        );
    }
}

#[test]
fn reduction_small_bounded_shape_witnesses_match_exact_feasibility_without_refinement() {
    let witnesses = [
        (Some(0), None),
        (Some(1), None),
        (Some(2), None),
        (None, None),
        (None, Some(0)),
        (None, Some(1)),
        (None, Some(2)),
    ];
    let input = |(size, bound): (Option<u64>, Option<i64>), element| {
        tensor(element, vec![size], bound.map(|bound| vec![bound]))
    };
    for lhs in witnesses {
        for rhs in witnesses {
            for third in witnesses {
                let expected = (0..=2).any(|extent| {
                    [lhs, rhs, third].iter().all(|&(size, bound)| {
                        size.is_none_or(|size| size == extent)
                            && bound.is_none_or(|bound| extent <= bound as u64)
                    })
                });
                let inputs = [
                    input(lhs, ElementType::F32),
                    input(rhs, ElementType::I32),
                    input(third, ElementType::I8),
                ];
                let accumulators = [
                    scalar(ElementType::F32),
                    scalar(ElementType::I32),
                    scalar(ElementType::I8),
                ];
                for dimensions in [vec![], vec![0]] {
                    let result = infer_reduce_types(&inputs, &dimensions, &accumulators);
                    assert_eq!(
                        result.is_ok(),
                        expected,
                        "{lhs:?}, {rhs:?}, {third:?}, {dimensions:?}",
                    );
                    if let Ok(results) = result {
                        assert_eq!(
                            results[0],
                            if dimensions.is_empty() {
                                inputs[0].clone()
                            } else {
                                scalar(ElementType::F32)
                            },
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn every_reduction_subset_removes_matching_dimensions_and_bounds() {
    let shape = [None, Some(3), None];
    let bounds = [5, -1, 7];
    let input = tensor(ElementType::F16, shape.to_vec(), Some(bounds.to_vec()));
    for mask in 0..8 {
        let mut dimensions = (0..3)
            .filter(|&axis| mask & (1 << axis) != 0)
            .collect::<Vec<i64>>();
        dimensions.reverse();
        let retained = (0..3)
            .filter(|&axis| mask & (1 << axis) == 0)
            .collect::<Vec<_>>();
        let expected_bounds = retained
            .iter()
            .map(|&axis| bounds[axis])
            .collect::<Vec<_>>();
        let expected_bounds = expected_bounds
            .iter()
            .any(|&bound| bound >= 0)
            .then_some(expected_bounds);
        assert_eq!(
            infer_reduce_types(
                std::slice::from_ref(&input),
                &dimensions,
                &[scalar(ElementType::F32)]
            )
            .unwrap(),
            vec![tensor(
                ElementType::F32,
                retained.iter().map(|&axis| shape[axis]).collect(),
                expected_bounds,
            )],
        );
    }
}

#[test]
fn multi_result_inference_uses_the_first_input_witness_and_each_return_element_type() {
    let inputs = vec![
        tensor(ElementType::F16, vec![None, Some(3)], Some(vec![8, -1])),
        tensor(ElementType::I8, vec![Some(4), Some(3)], None),
    ];
    assert_eq!(
        infer_reduce_types(
            &inputs,
            &[1],
            &[scalar(ElementType::F32), scalar(ElementType::I32)]
        )
        .unwrap(),
        vec![
            tensor(ElementType::F32, vec![None], Some(vec![8])),
            tensor(ElementType::I32, vec![None], Some(vec![8])),
        ],
    );
}

#[test]
fn reductions_do_not_propagate_unrelated_encodings_or_accumulator_encodings() {
    let encoding = Attribute::Opaque {
        dialect: Some("consumer".into()),
        syntax: "layout".into(),
    };
    let input: Type = TensorType::static_shape(ElementType::F32, [2, 3])
        .with_encoding(encoding.clone())
        .into();
    let accumulator: Type = TensorType::scalar(ElementType::F32)
        .with_encoding(encoding)
        .into();
    assert_eq!(
        infer_reduce_types(&[input], &[1], &[accumulator]).unwrap(),
        vec![tensor(ElementType::F32, vec![Some(2)], None)],
    );
    assert_eq!(
        infer_reduce_types(
            &[scalar(ElementType::I32)],
            &[],
            &[scalar(ElementType::I64)]
        )
        .unwrap(),
        vec![scalar(ElementType::I64)],
    );
}

#[test]
fn malformed_inputs_accumulator_counts_and_dimensions_are_rejected() {
    assert_eq!(
        infer_reduce_types(&[], &[], &[]),
        Err(InferenceError::EmptyInputs)
    );
    let input = tensor(ElementType::F32, vec![Some(2), Some(3)], None);
    assert_eq!(
        infer_reduce_types(std::slice::from_ref(&input), &[], &[]),
        Err(InferenceError::AccumulatorCountMismatch {
            expected: 1,
            actual: 0
        }),
    );
    for dimensions in [vec![0, 0], vec![-1], vec![2], vec![i64::MAX]] {
        assert_eq!(
            infer_reduce_types(
                std::slice::from_ref(&input),
                &dimensions,
                &[scalar(ElementType::F32)]
            ),
            Err(InferenceError::InvalidDimensions {
                rank: 2,
                dimensions
            }),
        );
    }
    for (bad, expected) in [
        (
            tensor(ElementType::I32, vec![Some(2)], None),
            InferenceError::RankMismatch {
                expected: 2,
                actual: 1,
            },
        ),
        (
            tensor(ElementType::I32, vec![Some(2), Some(4)], None),
            InferenceError::DimensionMismatch {
                dimension: 1,
                lhs: 3,
                rhs: 4,
            },
        ),
    ] {
        assert_eq!(
            infer_reduce_types(
                &[input.clone(), bad],
                &[1],
                &[scalar(ElementType::F32), scalar(ElementType::I32)]
            ),
            Err(expected),
        );
    }
    assert_eq!(
        infer_reduce_types(
            std::slice::from_ref(&input),
            &[1],
            std::slice::from_ref(&input)
        ),
        Err(InferenceError::NonScalarAccumulator { index: 0, rank: 2 }),
    );
    assert!(matches!(
        infer_reduce_types(&[Type::Token], &[], &[scalar(ElementType::F32)]),
        Err(InferenceError::IncompatibleTypes { index: 0 }),
    ));
    assert!(matches!(
        infer_reduce_types(std::slice::from_ref(&input), &[], &[Type::Token]),
        Err(InferenceError::IncompatibleTypes { index: 1 }),
    ));
    assert!(matches!(
        infer_reduce_types(
            &[tensor(ElementType::F32, vec![None], Some(vec![-2]))],
            &[],
            &[scalar(ElementType::F32)]
        ),
        Err(InferenceError::InvalidType { index: 0, .. }),
    ));
}

#[test]
fn per_tensor_quantization_is_retained_but_per_axis_quantization_is_not_a_reduce_type() {
    let input = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::I8,
        Some(-127),
        Some(127),
        ElementType::F32,
        None,
        vec![0.25],
        vec![1],
    ));
    let accumulator = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![],
        ElementType::I16,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.5],
        vec![2],
    ));
    let expected = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2)],
        ElementType::I16,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.5],
        vec![2],
    ));
    assert_eq!(
        infer_reduce_types(&[input], &[1], std::slice::from_ref(&accumulator)).unwrap(),
        vec![expected]
    );
    let input = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2)],
        ElementType::I8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.25, 0.5],
        vec![0, 0],
    ));
    assert_eq!(
        infer_reduce_types(&[input], &[0], &[accumulator]),
        Err(InferenceError::IncompatibleTypes { index: 0 }),
    );
}

fn reducer_body() -> Region {
    Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(100), scalar(ElementType::F32)),
            BlockArgument::new(ValueId(101), scalar(ElementType::F32)),
        ],
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            vec![ValueId(100)],
            vec![],
            vec![],
        )],
    )])
}

#[test]
fn verifier_rejects_static_results_above_retained_input_bounds() {
    for (size, accepted) in [(0, true), (5, true), (6, false)] {
        let result = common::computation_with_regions(
            StableHloOp::Reduce,
            vec![
                tensor(ElementType::F32, vec![None, Some(3)], Some(vec![5, -1])),
                scalar(ElementType::F32),
            ],
            vec![tensor(ElementType::F32, vec![Some(size)], None)],
            [("dimensions".into(), Attribute::integers(vec![1]))].into(),
            vec![reducer_body()],
        );
        if accepted {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(BuildError::InvalidResultType {
                    operation: "reduce",
                    ..
                })
            ));
        }
    }
}

#[test]
fn every_reduction_input_bound_constrains_every_static_result_witness() {
    let body = Region::new(vec![Block::new(
        (100..104)
            .map(|id| BlockArgument::new(ValueId(id), scalar(ElementType::F32)))
            .collect(),
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            vec![ValueId(100), ValueId(101)],
            vec![],
            vec![],
        )],
    )]);
    for bounds in [[5, 8], [8, 5]] {
        for static_result in 0..2 {
            for (size, accepted) in [(5, true), (6, false)] {
                let mut outputs = vec![tensor(ElementType::F32, vec![None], None); 2];
                outputs[static_result] = tensor(ElementType::F32, vec![Some(size)], None);
                let result = common::computation_with_regions(
                    StableHloOp::Reduce,
                    vec![
                        tensor(
                            ElementType::F32,
                            vec![None, Some(3)],
                            Some(vec![bounds[0], -1]),
                        ),
                        tensor(
                            ElementType::F32,
                            vec![None, Some(3)],
                            Some(vec![bounds[1], -1]),
                        ),
                        scalar(ElementType::F32),
                        scalar(ElementType::F32),
                    ],
                    outputs,
                    [("dimensions".into(), Attribute::integers(vec![1]))].into(),
                    vec![body.clone()],
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "{bounds:?}, result {static_result}, size {size}"
                );
            }
        }
    }
}
