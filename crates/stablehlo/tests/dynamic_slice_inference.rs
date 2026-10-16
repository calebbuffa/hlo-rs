use stablehlo::{
    Attribute, Attributes, BuildError, ElementType, InferenceError, QuantizedTensorType,
    StableHloAttribute, StableHloOp, TensorType, Type, TypeExtensions, infer_dynamic_slice_type,
    infer_dynamic_update_slice_type,
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

fn index(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

#[test]
fn exhaustive_small_slices_accept_exactly_feasible_sizes() {
    for extent in 0..=8 {
        for size in -1..=9 {
            for input in [
                tensor(vec![Some(extent)], None),
                tensor(vec![None], Some(vec![extent as i64])),
            ] {
                let inferred =
                    infer_dynamic_slice_type(&input, &[index(ElementType::I64)], &[size]);
                if size >= 0 && size as u64 <= extent {
                    assert_eq!(inferred.unwrap(), tensor(vec![Some(size as u64)], None));
                } else {
                    assert!(inferred.is_err(), "{input:?}, size {size}");
                }
            }
        }
    }
}

#[test]
fn scalar_unknown_and_extreme_slices_have_static_unencoded_results() {
    let scalar = tensor(vec![], None);
    assert_eq!(infer_dynamic_slice_type(&scalar, &[], &[]).unwrap(), scalar);
    let input = tensor(vec![None, Some(i64::MAX as u64)], None);
    assert_eq!(
        infer_dynamic_slice_type(&input, &vec![index(ElementType::I32); 2], &[i64::MAX, 0])
            .unwrap(),
        tensor(vec![Some(i64::MAX as u64), Some(0)], None),
    );
    let input: Type = TensorType::static_shape(ElementType::F32, [3])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_dynamic_slice_type(&input, &[index(ElementType::I16)], &[2]).unwrap(),
        tensor(vec![Some(2)], None),
    );
}

#[test]
fn index_types_counts_and_invalid_metadata_are_checked() {
    let input = tensor(vec![Some(2), Some(3)], None);
    for indices in [
        vec![],
        vec![index(ElementType::I32)],
        vec![index(ElementType::I32); 3],
        vec![index(ElementType::I32), index(ElementType::I64)],
        vec![index(ElementType::I1); 2],
        vec![index(ElementType::Index); 2],
        vec![index(ElementType::Si32); 2],
        vec![index(ElementType::F32); 2],
        vec![Type::Token; 2],
        vec![TensorType::static_shape(ElementType::I32, [1]).into(); 2],
    ] {
        assert!(
            infer_dynamic_slice_type(&input, &indices, &[1, 1]).is_err(),
            "{indices:?}"
        );
        assert!(
            infer_dynamic_update_slice_type(&input, &input, &indices).is_err(),
            "{indices:?}"
        );
    }
    assert_eq!(
        infer_dynamic_slice_type(&input, &vec![index(ElementType::I32); 2], &[1]),
        Err(InferenceError::AttributeLengthMismatch {
            attribute: "slice_sizes",
            expected: 2,
            actual: 1,
        }),
    );
    assert!(matches!(
        infer_dynamic_slice_type(
            &tensor(vec![None], Some(vec![-2])),
            &[index(ElementType::I64)],
            &[1]
        ),
        Err(InferenceError::InvalidType { index: 0, .. }),
    ));
    assert_eq!(
        infer_dynamic_slice_type(&Type::Token, &[], &[]),
        Err(InferenceError::IncompatibleTypes { index: 0 }),
    );
    for element in [
        ElementType::I2,
        ElementType::I4,
        ElementType::I8,
        ElementType::I16,
        ElementType::I32,
        ElementType::I64,
        ElementType::Ui2,
        ElementType::Ui4,
        ElementType::Ui8,
        ElementType::Ui16,
        ElementType::Ui32,
        ElementType::Ui64,
    ] {
        infer_dynamic_slice_type(&input, &vec![index(element); 2], &[1, 1]).unwrap();
    }
}

#[test]
fn update_sizes_are_feasible_and_preserve_operand_encodings() {
    for extent in 0..=8 {
        for size in 0..=9 {
            for input in [
                tensor(vec![Some(extent)], None),
                tensor(vec![None], Some(vec![extent as i64])),
            ] {
                let update = tensor(vec![Some(size)], None);
                let inferred =
                    infer_dynamic_update_slice_type(&input, &update, &[index(ElementType::I32)]);
                if size <= extent {
                    assert_eq!(inferred.unwrap(), input);
                } else {
                    assert!(inferred.is_err());
                }
            }
        }
    }
    let input: Type = TensorType::static_shape(ElementType::F32, [4])
        .with_encoding(Attribute::Opaque {
            dialect: None,
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_dynamic_update_slice_type(
            &input,
            &tensor(vec![None], Some(vec![9])),
            &[index(ElementType::I64)]
        )
        .unwrap(),
        input,
    );
    assert_eq!(
        infer_dynamic_update_slice_type(&tensor(vec![], None), &tensor(vec![], None), &[]).unwrap(),
        tensor(vec![], None),
    );
    assert!(matches!(
        infer_dynamic_update_slice_type(
            &input,
            &tensor(vec![Some(1), Some(2)], None),
            &[index(ElementType::I64)]
        ),
        Err(InferenceError::RankMismatch {
            expected: 1,
            actual: 2
        }),
    ));
    assert!(matches!(
        infer_dynamic_update_slice_type(
            &input,
            &index(ElementType::I32),
            &[index(ElementType::I64)]
        ),
        Err(InferenceError::RankMismatch { .. }),
    ));
    assert!(matches!(
        infer_dynamic_update_slice_type(
            &input,
            &TensorType::static_shape(ElementType::I32, [2]).into(),
            &[index(ElementType::I64)]
        ),
        Err(InferenceError::IncompatibleTypes { index: 1 }),
    ));
}

#[test]
fn per_tensor_quantization_is_preserved_exactly_and_per_axis_is_rejected() {
    let quantized = |shape, scale, axis| {
        Type::QuantizedTensor(QuantizedTensorType::new(
            shape,
            ElementType::I8,
            Some(-127),
            Some(127),
            ElementType::F32,
            axis,
            vec![scale],
            vec![1],
        ))
    };
    let input = quantized(vec![Some(3)], 0.25, None);
    let sliced = quantized(vec![Some(1)], 0.25, None);
    assert_eq!(
        infer_dynamic_slice_type(&input, &[index(ElementType::I32)], &[1]).unwrap(),
        sliced
    );
    assert_eq!(
        infer_dynamic_update_slice_type(&input, &sliced, &[index(ElementType::I32)]).unwrap(),
        input
    );
    assert!(
        infer_dynamic_update_slice_type(
            &input,
            &quantized(vec![Some(1)], 0.5, None),
            &[index(ElementType::I32)]
        )
        .is_err()
    );
    let per_axis = quantized(vec![Some(1)], 0.25, Some(0));
    assert!(infer_dynamic_slice_type(&per_axis, &[index(ElementType::I32)], &[1]).is_err());
    assert!(
        infer_dynamic_update_slice_type(&input, &per_axis, &[index(ElementType::I32)]).is_err()
    );
}

#[test]
fn verifier_enforces_dynamic_input_bounds_and_update_result_bounds() {
    for (size, accepted) in [(5, true), (6, false)] {
        let result = common::computation(
            StableHloOp::DynamicSlice,
            vec![tensor(vec![None], Some(vec![5])), index(ElementType::I32)],
            vec![tensor(vec![Some(size)], None)],
            [("slice_sizes".into(), Attribute::integers(vec![size as i64]))].into(),
        );
        assert_eq!(result.is_ok(), accepted);
        let result = common::computation(
            StableHloOp::DynamicUpdateSlice,
            vec![
                tensor(vec![None], Some(vec![5])),
                tensor(vec![Some(1)], None),
                index(ElementType::I32),
            ],
            vec![tensor(vec![Some(size)], None)],
            Attributes::new(),
        );
        if accepted {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(BuildError::InvalidResultType {
                    operation: "dynamic_update_slice",
                    ..
                })
            ));
        }
    }
}
