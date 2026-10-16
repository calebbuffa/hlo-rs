use stablehlo::{
    Attribute, BuildError, ElementType, InferenceError, StableHloAttribute, StableHloOp,
    TensorType, Type, TypeExtensions, infer_pad_type,
};

mod common;

fn tensor(shape: Vec<Option<u64>>, bounds: Option<Vec<i64>>) -> Type {
    let tensor = TensorType::new(ElementType::F32, shape);
    match bounds {
        Some(bounds) => tensor
            .with_encoding(Attribute::StableHlo(Box::new(
                StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
            )))
            .into(),
        None => tensor.into(),
    }
}

fn scalar() -> Type {
    TensorType::scalar(ElementType::F32).into()
}

#[test]
fn exhaustive_small_padding_sizes_include_cropping_and_empty_inputs() {
    for size in 0..9_i64 {
        for low in -5..=5 {
            for high in -5..=5 {
                for interior in 0..=4 {
                    let operand = tensor(vec![Some(size as u64)], None);
                    let result = infer_pad_type(&operand, &scalar(), &[low], &[high], &[interior]);
                    let expected = size + low + high + (size - 1).max(0) * interior;
                    if expected < 0 {
                        assert_eq!(
                            result,
                            Err(InferenceError::NegativePaddingResult { dimension: 0 })
                        );
                    } else {
                        assert_eq!(result.unwrap(), tensor(vec![Some(expected as u64)], None));
                    }
                }
            }
        }
    }
}

#[test]
fn bounded_dynamic_padding_transforms_bounds_without_inventing_static_sizes() {
    let operand = tensor(vec![None, Some(3), None], Some(vec![8, -1, -1]));
    assert_eq!(
        infer_pad_type(&operand, &scalar(), &[1, -1, 0], &[2, 2, 1], &[1, 2, 0]).unwrap(),
        tensor(vec![None, Some(8), None], Some(vec![18, -1, -1])),
    );
    let operand = tensor(vec![None], Some(vec![0]));
    assert_eq!(
        infer_pad_type(&operand, &scalar(), &[0], &[0], &[100]).unwrap(),
        tensor(vec![None], Some(vec![0])),
    );
    assert_eq!(
        infer_pad_type(&operand, &scalar(), &[-1], &[0], &[0]),
        Err(InferenceError::NegativePaddingResult { dimension: 0 }),
    );
}

#[test]
fn padding_overflow_is_detected_for_static_sizes_and_dynamic_bounds() {
    for operand in [
        tensor(vec![Some(i64::MAX as u64)], None),
        tensor(vec![None], Some(vec![i64::MAX])),
    ] {
        assert_eq!(
            infer_pad_type(&operand, &scalar(), &[1], &[0], &[0]),
            Err(InferenceError::DimensionOverflow { dimension: 0 }),
        );
        assert_eq!(
            infer_pad_type(&operand, &scalar(), &[0], &[0], &[i64::MAX]),
            Err(InferenceError::DimensionOverflow { dimension: 0 }),
        );
    }
    let operand = tensor(vec![Some(1)], None);
    assert_eq!(
        infer_pad_type(&operand, &scalar(), &[i64::MIN], &[i64::MAX], &[0]).unwrap(),
        tensor(vec![Some(0)], None),
    );
}

#[test]
fn padding_requires_scalar_matching_elements_valid_lengths_and_nonnegative_interior() {
    let operand = tensor(vec![Some(2)], None);
    for padding in [
        Type::Token,
        operand.clone(),
        TensorType::scalar(ElementType::I32).into(),
    ] {
        assert_eq!(
            infer_pad_type(&operand, &padding, &[0], &[0], &[0]),
            Err(InferenceError::IncompatibleTypes { index: 1 }),
        );
    }
    for (low, high, interior, attribute) in [
        (vec![], vec![0], vec![0], "edge_padding_low"),
        (vec![0], vec![], vec![0], "edge_padding_high"),
        (vec![0], vec![0], vec![], "interior_padding"),
    ] {
        assert!(matches!(
            infer_pad_type(&operand, &scalar(), &low, &high, &interior),
            Err(InferenceError::AttributeLengthMismatch { attribute: actual, .. }) if actual == attribute
        ));
    }
    assert_eq!(
        infer_pad_type(&operand, &scalar(), &[0], &[0], &[-1]),
        Err(InferenceError::NegativeInteriorPadding {
            dimension: 0,
            value: -1
        }),
    );
    assert_eq!(
        infer_pad_type(&scalar(), &scalar(), &[], &[], &[]).unwrap(),
        scalar()
    );
}

#[test]
fn verifier_rejects_result_sizes_that_exceed_the_inferred_dynamic_bound() {
    let operand = tensor(vec![None], Some(vec![4]));
    for (output, accepted) in [
        (tensor(vec![Some(9)], None), true),
        (tensor(vec![Some(10)], None), false),
        (tensor(vec![None], Some(vec![9])), true),
    ] {
        let result = common::computation(
            StableHloOp::Pad,
            vec![operand.clone(), scalar()],
            vec![output],
            [
                ("edge_padding_low".into(), Attribute::integers(vec![1])),
                ("edge_padding_high".into(), Attribute::integers(vec![1])),
                ("interior_padding".into(), Attribute::integers(vec![1])),
            ]
            .into(),
        );
        if accepted {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(BuildError::InvalidResultType {
                    operation: "pad",
                    ..
                })
            ));
        }
    }
}

#[test]
fn verifier_rejects_negative_padding_of_a_bounded_dynamic_dimension() {
    let result = common::computation(
        StableHloOp::Pad,
        vec![tensor(vec![None], Some(vec![2])), scalar()],
        vec![tensor(vec![None], None)],
        [
            ("edge_padding_low".into(), Attribute::integers(vec![-3])),
            ("edge_padding_high".into(), Attribute::integers(vec![0])),
            ("interior_padding".into(), Attribute::integers(vec![0])),
        ]
        .into(),
    );
    assert!(matches!(
        result,
        Err(BuildError::TypeInference {
            operation: "pad",
            error: InferenceError::NegativePaddingResult { dimension: 0 },
        })
    ));
}
