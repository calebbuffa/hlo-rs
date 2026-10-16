use stablehlo::{
    Attribute, BuildError, ElementType, InferenceError, QuantizedTensorType, StableHloAttribute,
    StableHloOp, TensorType, Type, TypeExtensions, infer_slice_type,
};

mod common;

fn bounded(sizes: &[Option<u64>], bounds: Vec<i64>) -> Type {
    TensorType::new(ElementType::F32, sizes.to_vec())
        .with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
        )))
        .into()
}

#[test]
fn exhaustive_small_static_ranges_match_direct_index_enumeration() {
    for size in 0..12_i64 {
        let operand: Type = TensorType::static_shape(ElementType::F32, [size as u64]).into();
        for start in 0..=size {
            for limit in start..=size {
                for stride in 1..=15 {
                    let expected = (start..limit).step_by(stride as usize).count() as u64;
                    assert_eq!(
                        infer_slice_type(&operand, &[start], &[limit], &[stride]).unwrap(),
                        Type::from(TensorType::static_shape(ElementType::F32, [expected])),
                    );
                }
            }
        }
    }
}

#[test]
fn dynamic_sizes_and_upper_bounds_are_checked_before_static_result_inference() {
    let operand = bounded(&[None, Some(7), None], vec![8, -1, -1]);
    assert_eq!(
        infer_slice_type(&operand, &[1, 2, 0], &[8, 7, 11], &[2, 3, 4]).unwrap(),
        Type::from(TensorType::static_shape(ElementType::F32, [4, 2, 3])),
    );
    assert!(matches!(
        infer_slice_type(&operand, &[0, 0, 0], &[9, 7, 0], &[1, 1, 1]),
        Err(InferenceError::SliceLimitExceeded {
            dimension: 0,
            limit: 9,
            size_or_bound: 8
        }),
    ));
    assert!(matches!(
        infer_slice_type(&operand, &[0, 0, 0], &[8, 8, 0], &[1, 1, 1]),
        Err(InferenceError::SliceLimitExceeded {
            dimension: 1,
            limit: 8,
            size_or_bound: 7
        }),
    ));
}

#[test]
fn scalar_empty_and_large_integer_ranges_have_exact_result_shapes() {
    let scalar: Type = TensorType::scalar(ElementType::I32).into();
    assert_eq!(infer_slice_type(&scalar, &[], &[], &[]).unwrap(), scalar);
    let operand: Type = TensorType::new(ElementType::F32, vec![None]).into();
    for (start, limit, stride, expected) in [
        (0, i64::MAX, 1, i64::MAX as u64),
        (0, i64::MAX, 2, (i64::MAX as u64).div_ceil(2)),
        (i64::MAX, i64::MAX, 1, 0),
        (0, 1, i64::MAX, 1),
    ] {
        assert_eq!(
            infer_slice_type(&operand, &[start], &[limit], &[stride]).unwrap(),
            Type::from(TensorType::static_shape(ElementType::F32, [expected])),
        );
    }
}

#[test]
fn malformed_types_lengths_and_ranges_return_structured_diagnostics() {
    let operand: Type = TensorType::static_shape(ElementType::F32, [4]).into();
    for (starts, limits, strides, attribute) in [
        (vec![], vec![4], vec![1], "start_indices"),
        (vec![0], vec![], vec![1], "limit_indices"),
        (vec![0], vec![4], vec![], "strides"),
        (vec![0, 0], vec![4], vec![1], "start_indices"),
    ] {
        assert!(matches!(
            infer_slice_type(&operand, &starts, &limits, &strides),
            Err(InferenceError::AttributeLengthMismatch { attribute: actual, .. }) if actual == attribute
        ));
    }
    for (start, limit, stride) in [(-1, 4, 1), (3, 2, 1), (0, 4, 0), (0, 4, -1)] {
        assert!(matches!(
            infer_slice_type(&operand, &[start], &[limit], &[stride]),
            Err(InferenceError::InvalidSliceRange { dimension: 0, .. }),
        ));
    }
    assert!(matches!(
        infer_slice_type(&Type::Token, &[], &[], &[]),
        Err(InferenceError::IncompatibleTypes { index: 0 })
    ));
    assert!(matches!(
        infer_slice_type(&bounded(&[None], vec![0, 0]), &[0], &[0], &[1]),
        Err(InferenceError::InvalidType { index: 0, .. })
    ));
}

#[test]
fn inference_preserves_element_metadata_and_unrelated_encodings() {
    let encoding = Attribute::Opaque {
        dialect: Some("consumer".into()),
        syntax: "layout".into(),
    };
    let operand: Type = TensorType::static_shape(ElementType::C128, [4])
        .with_encoding(encoding.clone())
        .into();
    assert_eq!(
        infer_slice_type(&operand, &[1], &[4], &[2]).unwrap(),
        Type::from(TensorType::static_shape(ElementType::C128, [2]).with_encoding(encoding)),
    );
    let operand = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(4)],
        ElementType::I8,
        Some(-127),
        Some(127),
        ElementType::F32,
        None,
        vec![0.25],
        vec![3],
    ));
    assert_eq!(
        infer_slice_type(&operand, &[1], &[4], &[2]).unwrap(),
        Type::QuantizedTensor(QuantizedTensorType::new(
            vec![Some(2)],
            ElementType::I8,
            Some(-127),
            Some(127),
            ElementType::F32,
            None,
            vec![0.25],
            vec![3],
        )),
    );
}

#[test]
fn verifier_checks_declared_dynamic_result_bounds_against_the_inferred_static_size() {
    for (bound, accepted) in [(1, false), (2, true), (3, true)] {
        let result = common::computation(
            StableHloOp::Slice,
            vec![TensorType::static_shape(ElementType::F32, [4]).into()],
            vec![bounded(&[None], vec![bound])],
            [
                ("start_indices".into(), Attribute::integers(vec![0])),
                ("limit_indices".into(), Attribute::integers(vec![4])),
                ("strides".into(), Attribute::integers(vec![2])),
            ]
            .into(),
        );
        if accepted {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(BuildError::InvalidResultType {
                    operation: "slice",
                    ..
                })
            ));
        }
    }
}

#[test]
fn verifier_uses_shared_inference_and_rejects_limits_exceeding_dynamic_bounds() {
    let operand = bounded(&[None], vec![4]);
    for (limit, accepted) in [(4, true), (5, false)] {
        let result = common::computation(
            StableHloOp::Slice,
            vec![operand.clone()],
            vec![TensorType::static_shape(ElementType::F32, [limit as u64]).into()],
            [
                ("start_indices".into(), Attribute::integers(vec![0])),
                ("limit_indices".into(), Attribute::integers(vec![limit])),
                ("strides".into(), Attribute::integers(vec![1])),
            ]
            .into(),
        );
        if accepted {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(BuildError::TypeInference {
                    operation: "slice",
                    error: InferenceError::SliceLimitExceeded { dimension: 0, .. },
                })
            ));
        }
    }
}
