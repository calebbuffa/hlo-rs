use stablehlo::{
    Attribute, ElementType, InferenceError, QuantizedTensorType, StableHloAttribute, TensorType,
    Type, TypeExtensions, infer_least_specific_type, infer_most_specific_type,
};

fn tensor(size: Option<u64>, bound: Option<i64>) -> Type {
    let mut tensor = TensorType::new(ElementType::F32, vec![size]);
    if let Some(bound) = bound {
        tensor = tensor.with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions {
                bounds: vec![bound],
            }),
        )));
    }
    tensor.into()
}

#[test]
fn most_specific_dimension_rules_cover_the_reference_table_symmetrically() {
    for (lhs, rhs, expected) in [
        (
            tensor(Some(3), None),
            tensor(Some(3), None),
            tensor(Some(3), None),
        ),
        (
            tensor(Some(3), None),
            tensor(None, None),
            tensor(Some(3), None),
        ),
        (
            tensor(Some(3), None),
            tensor(None, Some(5)),
            tensor(Some(3), None),
        ),
        (tensor(None, None), tensor(None, None), tensor(None, None)),
        (
            tensor(None, None),
            tensor(None, Some(5)),
            tensor(None, Some(5)),
        ),
        (
            tensor(None, Some(5)),
            tensor(None, Some(7)),
            tensor(None, Some(5)),
        ),
        (
            tensor(Some(0), None),
            tensor(None, Some(0)),
            tensor(Some(0), None),
        ),
    ] {
        for inputs in [[lhs.clone(), rhs.clone()], [rhs, lhs]] {
            assert_eq!(infer_most_specific_type(&inputs), Ok(expected.clone()));
        }
    }
}

#[test]
fn least_specific_dimension_rules_cover_the_reference_table_symmetrically() {
    for (lhs, rhs, expected) in [
        (
            tensor(Some(3), None),
            tensor(Some(3), None),
            tensor(Some(3), None),
        ),
        (
            tensor(Some(3), None),
            tensor(None, None),
            tensor(None, None),
        ),
        (
            tensor(Some(3), None),
            tensor(None, Some(5)),
            tensor(None, Some(5)),
        ),
        (
            tensor(Some(7), None),
            tensor(None, Some(5)),
            tensor(None, Some(7)),
        ),
        (tensor(None, None), tensor(None, None), tensor(None, None)),
        (
            tensor(None, None),
            tensor(None, Some(5)),
            tensor(None, None),
        ),
        (
            tensor(None, Some(5)),
            tensor(None, Some(7)),
            tensor(None, Some(7)),
        ),
    ] {
        for inputs in [[lhs.clone(), rhs.clone()], [rhs, lhs]] {
            assert_eq!(infer_least_specific_type(&inputs), Ok(expected.clone()));
        }
    }
}

#[test]
fn inference_rejects_empty_invalid_or_contradictory_inputs_without_panicking() {
    for infer in [infer_most_specific_type, infer_least_specific_type] {
        assert_eq!(infer(&[]), Err(InferenceError::EmptyInputs));
        assert!(matches!(
            infer(&[tensor(Some(3), None), tensor(Some(4), None)]),
            Err(InferenceError::DimensionMismatch { dimension: 0, .. })
        ));
        assert!(matches!(
            infer(&[
                tensor(None, None),
                TensorType::scalar(ElementType::F32).into()
            ]),
            Err(InferenceError::RankMismatch { .. })
        ));
        assert!(matches!(
            infer(&[tensor(None, Some(-2))]),
            Err(InferenceError::InvalidType { .. })
        ));
        assert!(matches!(
            infer(&[tensor(Some(3), Some(5))]),
            Err(InferenceError::InvalidType { .. })
        ));
        assert!(matches!(
            infer(&[tensor(Some(i64::MAX as u64 + 1), None)]),
            Err(InferenceError::InvalidType { .. })
        ));
        assert!(matches!(
            infer(&[tensor(None, None), Type::Token]),
            Err(InferenceError::IncompatibleTypes { .. })
        ));
        assert!(matches!(
            infer(&[
                tensor(None, None),
                TensorType::new(ElementType::I32, vec![None]).into()
            ]),
            Err(InferenceError::IncompatibleTypes { .. })
        ));
    }
    for inputs in [
        [tensor(Some(7), None), tensor(None, Some(5))],
        [tensor(None, Some(5)), tensor(Some(7), None)],
    ] {
        assert_eq!(
            infer_most_specific_type(&inputs),
            Err(InferenceError::BoundExceeded {
                dimension: 0,
                size: 7,
                bound: 5,
            })
        );
    }
}

#[test]
fn nested_tuple_joins_preserve_tokens_and_reject_mixed_kinds_and_lengths() {
    let lhs = Type::Tuple(vec![Type::Token, Type::Tuple(vec![tensor(None, Some(5))])]);
    let rhs = Type::Tuple(vec![Type::Token, Type::Tuple(vec![tensor(Some(3), None)])]);
    assert_eq!(
        infer_most_specific_type(&[lhs.clone(), rhs.clone()]),
        Ok(Type::Tuple(vec![
            Type::Token,
            Type::Tuple(vec![tensor(Some(3), None)])
        ]))
    );
    assert_eq!(
        infer_least_specific_type(&[lhs.clone(), rhs]),
        Ok(lhs.clone())
    );
    for infer in [infer_most_specific_type, infer_least_specific_type] {
        assert!(infer(&[lhs.clone(), Type::Token]).is_err());
        assert!(infer(&[lhs.clone(), Type::Tuple(vec![])]).is_err());
        assert_eq!(
            infer(&[Type::Tuple(vec![]), Type::Tuple(vec![])]),
            Ok(Type::Tuple(vec![]))
        );
        assert_eq!(infer(&[Type::Token, Type::Token]), Ok(Type::Token));
    }
}

#[test]
fn refinement_merges_every_input_without_a_dynamic_witness_hiding_conflicts() {
    for inputs in [
        vec![
            tensor(None, None),
            tensor(Some(3), None),
            tensor(Some(4), None),
        ],
        vec![
            tensor(None, Some(5)),
            tensor(None, None),
            tensor(Some(7), None),
        ],
    ] {
        assert!(infer_most_specific_type(&inputs).is_err());
    }
    assert_eq!(
        infer_most_specific_type(&[
            tensor(None, None),
            tensor(None, Some(8)),
            tensor(None, Some(5)),
        ]),
        Ok(tensor(None, Some(5)))
    );
    assert_eq!(
        infer_least_specific_type(&[
            tensor(None, Some(3)),
            tensor(None, Some(8)),
            tensor(None, Some(5)),
        ]),
        Ok(tensor(None, Some(8)))
    );
}

#[test]
fn nonbounded_encoding_and_first_quantized_element_metadata_are_preserved() {
    let encoding = Attribute::Opaque {
        dialect: Some("sparse_tensor".into()),
        syntax: "encoding".into(),
    };
    let lhs =
        Type::from(TensorType::new(ElementType::F32, vec![None]).with_encoding(encoding.clone()));
    let rhs = tensor(Some(3), None);
    assert_eq!(
        infer_most_specific_type(&[lhs, rhs]),
        Ok(TensorType::static_shape(ElementType::F32, [3])
            .with_encoding(encoding)
            .into())
    );
    let quantized = |shape, scale, zero| {
        Type::QuantizedTensor(QuantizedTensorType::new(
            vec![shape],
            ElementType::I8,
            None,
            None,
            ElementType::F32,
            None,
            vec![scale],
            vec![zero],
        ))
    };
    assert_eq!(
        infer_most_specific_type(&[quantized(None, 0.5, 1), quantized(Some(3), 0.25, 2),]),
        Ok(quantized(Some(3), 0.5, 1))
    );
}
