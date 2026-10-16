use stablehlo::{
    Attribute, Computation, ElementType, InferenceError, Instruction, Operation,
    QuantizedTensorType, ScalarValue, StableHloAttribute, StableHloOp, TensorType, Type,
    TypeExtensions, ValueId, infer_get_dimension_size_type, infer_set_dimension_size_type,
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

fn size_type() -> Type {
    TensorType::scalar(ElementType::I32).into()
}

#[test]
fn get_supports_per_axis_quantization_but_set_requires_per_tensor() {
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
        infer_get_dimension_size_type(&input, 0).unwrap(),
        size_type()
    );
    common::computation(
        StableHloOp::GetDimensionSize,
        vec![input.clone()],
        vec![size_type()],
        [("dimension".into(), Attribute::integer(0))].into(),
    )
    .unwrap();
    assert_eq!(
        infer_set_dimension_size_type(&input, &size_type(), 0, None),
        Err(InferenceError::IncompatibleTypes { index: 0 }),
    );
    let input = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2)],
        ElementType::I8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.25],
        vec![1],
    ));
    let result = infer_set_dimension_size_type(&input, &size_type(), 0, None).unwrap();
    let Type::QuantizedTensor(result) = result else {
        panic!("lost quantization");
    };
    assert_eq!(result.dimensions(), &[None]);
    assert_eq!(result.quantization_dimension(), None);
    assert_eq!(result.scales().collect::<Vec<_>>(), vec![0.25]);
    assert_eq!(result.zero_points(), &[1]);
}

#[test]
fn setting_each_static_axis_converts_its_extent_to_a_bound() {
    let input = tensor(vec![Some(2), Some(3), Some(4)], None);
    for axis in 0..3 {
        let mut shape = vec![Some(2), Some(3), Some(4)];
        let mut bounds = vec![-1; 3];
        bounds[axis] = shape[axis].unwrap() as i64;
        shape[axis] = None;
        assert_eq!(
            infer_set_dimension_size_type(&input, &size_type(), axis, None).unwrap(),
            tensor(shape, Some(bounds)),
        );
        assert_eq!(
            infer_get_dimension_size_type(&input, axis).unwrap(),
            size_type()
        );
    }
}

#[test]
fn pinned_reference_bound_propagation_cases_match_expected_results() {
    let cases = [
        (
            tensor(vec![Some(3), Some(5)], None),
            0,
            None,
            tensor(vec![None, Some(5)], Some(vec![3, -1])),
        ),
        (
            tensor(vec![None, Some(5)], Some(vec![8, -1])),
            0,
            Some(8),
            tensor(vec![Some(8), Some(5)], None),
        ),
        (
            tensor(vec![None, Some(5)], Some(vec![3, -1])),
            1,
            None,
            tensor(vec![None, None], Some(vec![3, 5])),
        ),
        (
            tensor(vec![None, Some(5)], Some(vec![3, -1])),
            0,
            None,
            tensor(vec![None, Some(5)], Some(vec![3, -1])),
        ),
        (
            tensor(vec![None, None], Some(vec![3, -1])),
            1,
            None,
            tensor(vec![None, None], Some(vec![3, -1])),
        ),
        (tensor(vec![None], None), 0, None, tensor(vec![None], None)),
        (
            tensor(vec![Some(0)], None),
            0,
            Some(0),
            tensor(vec![Some(0)], None),
        ),
    ];
    for (input, axis, constant, expected) in cases {
        assert_eq!(
            infer_set_dimension_size_type(&input, &size_type(), axis, constant).unwrap(),
            expected
        );
    }
}

#[test]
fn only_constants_equal_to_the_bound_restore_static_dimensions() {
    for extent in 0..=8 {
        let input = tensor(vec![Some(extent)], None);
        for constant in -1..=9 {
            let expected = if constant == extent as i32 {
                input.clone()
            } else {
                tensor(vec![None], Some(vec![extent as i64]))
            };
            assert_eq!(
                infer_set_dimension_size_type(&input, &size_type(), 0, Some(constant)).unwrap(),
                expected
            );
        }
    }
    let input = tensor(vec![Some(i64::MAX as u64)], None);
    assert_eq!(
        infer_set_dimension_size_type(&input, &size_type(), 0, None).unwrap(),
        tensor(vec![None], Some(vec![i64::MAX])),
    );
}

#[test]
fn unrecognized_encodings_are_dropped_and_invalid_inputs_are_reported() {
    let input: Type = TensorType::new(ElementType::F32, vec![None])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_set_dimension_size_type(&input, &size_type(), 0, None).unwrap(),
        tensor(vec![None], None)
    );
    for bad_size in [
        Type::Token,
        TensorType::scalar(ElementType::I64).into(),
        TensorType::static_shape(ElementType::I32, [1]).into(),
    ] {
        assert_eq!(
            infer_set_dimension_size_type(&input, &bad_size, 0, None),
            Err(InferenceError::IncompatibleTypes { index: 1 }),
        );
    }
    for axis in [1, usize::MAX] {
        assert_eq!(
            infer_set_dimension_size_type(&input, &size_type(), axis, None),
            Err(InferenceError::DimensionOutOfBounds {
                dimension: axis,
                rank: 1
            }),
        );
        assert!(infer_get_dimension_size_type(&input, axis).is_err());
    }
    assert!(infer_get_dimension_size_type(&tensor(vec![], None), 0).is_err());
    assert_eq!(
        infer_get_dimension_size_type(&Type::Token, 0),
        Err(InferenceError::IncompatibleTypes { index: 0 }),
    );
    assert!(matches!(
        infer_get_dimension_size_type(&tensor(vec![None], Some(vec![-2])), 0),
        Err(InferenceError::InvalidType { index: 0, .. }),
    ));
}

#[test]
fn verifier_rejects_static_results_exceeding_new_or_preserved_bounds() {
    for input in [
        tensor(vec![Some(5)], None),
        tensor(vec![None], Some(vec![5])),
    ] {
        for (extent, accepted) in [(0, true), (5, true), (6, false)] {
            let result = common::computation(
                StableHloOp::SetDimensionSize,
                vec![input.clone(), size_type()],
                vec![tensor(vec![Some(extent)], None)],
                [("dimension".into(), Attribute::integer(0))].into(),
            );
            assert_eq!(result.is_ok(), accepted);
        }
    }
}

#[test]
fn dominating_constant_equal_to_bound_checks_static_result_compatibility() {
    for (extent, accepted) in [(4, false), (5, true)] {
        let parameter = Instruction::new(
            Operation::parameter(0, "input"),
            vec![],
            vec![],
            ValueId(0),
            tensor(vec![None], Some(vec![5])),
        );
        let constant = Instruction::new(
            Operation::stablehlo(
                StableHloOp::Constant,
                [(
                    "value".into(),
                    Attribute::DenseElements {
                        ty: Box::new(size_type()),
                        values: vec![ScalarValue::Integer(5)],
                    },
                )]
                .into(),
            ),
            vec![],
            vec![],
            ValueId(1),
            size_type(),
        );
        let set = Instruction::new(
            Operation::stablehlo(
                StableHloOp::SetDimensionSize,
                [("dimension".into(), Attribute::integer(0))].into(),
            ),
            vec![ValueId(0), ValueId(1)],
            vec![],
            ValueId(2),
            tensor(vec![Some(extent)], None),
        );
        assert_eq!(
            Computation::try_new("set", vec![parameter, constant, set], ValueId(2)).is_ok(),
            accepted
        );
    }
}

#[test]
fn constant_dimension_size_refinement_applies_inside_capturing_regions() {
    for (extent, accepted) in [(4, false), (5, true)] {
        let source = format!(
            r#"module {{
                  func.func @main(%input: tensor<?xf32, #stablehlo.bounds<5>>) {{
                    %size = stablehlo.constant dense<5> : tensor<i32>
                    stablehlo.while () cond {{
                      %false = stablehlo.constant dense<false> : tensor<i1>
                      stablehlo.return %false : tensor<i1>
                    }} do {{
                      %set = "stablehlo.set_dimension_size"(%input, %size)
                        {{dimension = 0 : i64}}
                        : (tensor<?xf32, #stablehlo.bounds<5>>, tensor<i32>) -> tensor<{extent}xf32>
                      stablehlo.return
                    }}
                    return
                  }}
                }}"#
        );
        let result = stablehlo::text::parse_module(&source);
        assert_eq!(result.is_ok(), accepted, "{result:?}");
    }
}
