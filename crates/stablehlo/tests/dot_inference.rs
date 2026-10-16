use stablehlo::{
    Attribute, DotDimensionNumbers, ElementType, InferenceError, QuantizedTensorType,
    StableHloAttribute, StableHloOp, TensorType, Type, TypeExtensions, infer_dot_general_type,
    infer_dot_type,
};

mod common;

fn tensor(shape: Vec<Option<u64>>, bounds: Option<Vec<i64>>) -> Type {
    let mut ty = TensorType::new(ElementType::F32, shape);
    if let Some(bounds) = bounds {
        ty = ty.with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
        )));
    }
    ty.into()
}

fn dims(lb: Vec<i64>, rb: Vec<i64>, lc: Vec<i64>, rc: Vec<i64>) -> DotDimensionNumbers {
    DotDimensionNumbers {
        lhs_batching_dimensions: lb,
        rhs_batching_dimensions: rb,
        lhs_contracting_dimensions: lc,
        rhs_contracting_dimensions: rc,
    }
}

#[test]
fn legacy_dot_covers_every_vector_matrix_combination_and_retained_bounds() {
    let vector = tensor(vec![Some(3)], None);
    let left = tensor(vec![None, Some(3)], Some(vec![5, -1]));
    let right = tensor(vec![Some(3), None], Some(vec![-1, 7]));
    let template = tensor(vec![], None);
    for (lhs, rhs, expected) in [
        (&vector, &vector, tensor(vec![], None)),
        (&left, &vector, tensor(vec![None], Some(vec![5]))),
        (&vector, &right, tensor(vec![None], Some(vec![7]))),
        (&left, &right, tensor(vec![None, None], Some(vec![5, 7]))),
    ] {
        assert_eq!(infer_dot_type(lhs, rhs, &template).unwrap(), expected);
    }
    let bounded = tensor(vec![None], Some(vec![5]));
    assert_eq!(
        infer_dot_type(&bounded, &bounded, &template).unwrap(),
        template
    );
}

#[test]
fn batch_axes_use_attribute_order_and_free_axes_use_operand_order() {
    let lhs = tensor(vec![Some(2), Some(3), Some(5), Some(7)], None);
    let rhs = tensor(vec![Some(5), Some(11), Some(2), Some(7)], None);
    assert_eq!(
        infer_dot_general_type(
            &lhs,
            &rhs,
            &dims(vec![2, 0], vec![0, 2], vec![3], vec![3]),
            &tensor(vec![], None)
        )
        .unwrap(),
        tensor(vec![Some(5), Some(2), Some(3), Some(11)], None),
    );
}

#[test]
fn batching_refines_dynamic_dimensions_with_the_tightest_compatible_bound() {
    for (left_size, left_bound, right_size, right_bound, result_size, result_bound) in [
        (None, Some(8), None, Some(5), None, Some(5)),
        (None, Some(8), Some(5), None, Some(5), None),
        (Some(5), None, None, Some(8), Some(5), None),
        (None, None, None, Some(8), None, Some(8)),
        (Some(0), None, Some(0), None, Some(0), None),
    ] {
        let lhs = tensor(
            vec![left_size, Some(3)],
            left_bound.map(|bound| vec![bound, -1]),
        );
        let rhs = tensor(
            vec![right_size, Some(3)],
            right_bound.map(|bound| vec![bound, -1]),
        );
        assert_eq!(
            infer_dot_general_type(
                &lhs,
                &rhs,
                &dims(vec![0], vec![0], vec![1], vec![1]),
                &tensor(vec![], None)
            )
            .unwrap(),
            tensor(vec![result_size], result_bound.map(|bound| vec![bound])),
        );
    }
}

#[test]
fn invalid_axis_lists_pair_counts_shapes_and_bound_contradictions_are_errors() {
    let input = tensor(vec![Some(2), Some(3)], None);
    for dimensions in [
        dims(vec![0], vec![], vec![], vec![]),
        dims(vec![], vec![], vec![1], vec![]),
        dims(vec![0, 0], vec![0, 1], vec![], vec![]),
        dims(vec![], vec![], vec![-1], vec![1]),
        dims(vec![], vec![], vec![i64::MAX], vec![1]),
        dims(vec![0], vec![0], vec![0], vec![1]),
        dims(vec![], vec![], vec![0], vec![1]),
    ] {
        assert!(
            infer_dot_general_type(&input, &input, &dimensions, &tensor(vec![], None)).is_err()
        );
    }
    let small = tensor(vec![None, Some(3)], Some(vec![1, -1]));
    assert!(matches!(
        infer_dot_general_type(
            &small,
            &input,
            &dims(vec![0], vec![0], vec![1], vec![1]),
            &tensor(vec![], None)
        ),
        Err(InferenceError::BoundExceeded {
            dimension: 0,
            size: 2,
            bound: 1
        }),
    ));
    assert!(matches!(
        infer_dot_type(
            &tensor(vec![Some(2), Some(4)], None),
            &input,
            &tensor(vec![], None)
        ),
        Err(InferenceError::DimensionMismatch { .. }),
    ));
    for bad in [
        Type::Token,
        tensor(vec![], None),
        tensor(vec![Some(2); 3], None),
    ] {
        assert!(infer_dot_type(&bad, &input, &tensor(vec![], None)).is_err());
    }
    assert!(
        infer_dot_general_type(
            &input,
            &input,
            &dims(vec![], vec![], vec![1], vec![1]),
            &Type::Token
        )
        .is_err()
    );
}

#[test]
fn result_template_supplies_element_metadata_not_shape_or_encoding() {
    let lhs = tensor(vec![Some(2), Some(3)], None);
    let rhs = tensor(vec![Some(3), Some(4)], None);
    let template: Type = TensorType::scalar(ElementType::F64).into();
    assert_eq!(
        infer_dot_type(&lhs, &rhs, &template).unwrap(),
        Type::from(TensorType::static_shape(ElementType::F64, [2, 4])),
    );
    let template = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(4)],
        ElementType::I8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.25; 4],
        vec![1; 4],
    ));
    let dimensions = dims(vec![], vec![], vec![0], vec![0]);
    let lhs = tensor(vec![Some(3)], None);
    assert_eq!(
        infer_dot_general_type(&lhs, &rhs, &dimensions, &template).unwrap(),
        template
    );
}

#[test]
fn verifier_rejects_free_and_batch_result_dimensions_exceeding_inferred_bounds() {
    let dimensions = dims(vec![0], vec![0], vec![2], vec![1]);
    for (size, accepted) in [(5, true), (6, false)] {
        for shape in [
            vec![Some(4), Some(size), Some(7)],
            vec![Some(size), Some(5), Some(7)],
        ] {
            let result = common::computation(
                StableHloOp::DotGeneral,
                vec![
                    tensor(vec![None, None, Some(3)], Some(vec![5, 5, -1])),
                    tensor(vec![None, Some(3), Some(7)], Some(vec![8, -1, -1])),
                ],
                vec![tensor(shape, None)],
                [(
                    "dot_dimension_numbers".into(),
                    Attribute::StableHlo(Box::new(StableHloAttribute::DotDimensionNumbers(
                        dimensions.clone(),
                    ))),
                )]
                .into(),
            );
            assert_eq!(result.is_ok(), accepted);
        }
    }
    for (size, accepted) in [(5, true), (6, false)] {
        let result = common::computation(
            StableHloOp::Dot,
            vec![
                tensor(vec![None, Some(3)], Some(vec![5, -1])),
                tensor(vec![Some(3)], None),
            ],
            vec![tensor(vec![Some(size)], None)],
            Default::default(),
        );
        assert_eq!(result.is_ok(), accepted);
    }
}
