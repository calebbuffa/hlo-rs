use stablehlo::{
    Attribute, ElementType, InferenceError, QuantizedTensorType, StableHloAttribute, StableHloOp,
    TensorType, Type, TypeExtensions, infer_broadcast_type, infer_reverse_type,
};

mod common;

fn bounded() -> Type {
    TensorType::new(ElementType::F32, vec![None, Some(3)])
        .with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions {
                bounds: vec![5, -1],
            }),
        )))
        .into()
}

#[test]
fn reverse_preserves_all_metadata_for_every_axis_subset_and_order() {
    for input in [
        bounded(),
        TensorType::static_shape(ElementType::F32, [2, 3])
            .with_encoding(Attribute::Opaque {
                dialect: None,
                syntax: "layout".into(),
            })
            .into(),
    ] {
        for axes in [vec![], vec![0], vec![1], vec![0, 1], vec![1, 0]] {
            assert_eq!(infer_reverse_type(&input, &axes).unwrap(), input);
        }
    }
    let scalar: Type = TensorType::scalar(ElementType::I32).into();
    assert_eq!(infer_reverse_type(&scalar, &[]).unwrap(), scalar);
}

#[test]
fn reverse_rejects_duplicate_negative_and_out_of_range_axes() {
    for dimensions in [vec![0, 0], vec![-1], vec![2], vec![i64::MAX]] {
        assert_eq!(
            infer_reverse_type(&bounded(), &dimensions),
            Err(InferenceError::InvalidDimensions {
                rank: 2,
                dimensions
            }),
        );
    }
    assert_eq!(
        infer_reverse_type(&Type::Token, &[]),
        Err(InferenceError::IncompatibleTypes { index: 0 })
    );
}

#[test]
fn broadcasts_prepend_zero_and_extreme_extents_without_propagating_encoding() {
    let input = bounded();
    for leading in [vec![], vec![0], vec![2, 0], vec![i64::MAX, 3]] {
        let expected: Type = TensorType::new(
            ElementType::F32,
            leading
                .iter()
                .map(|&size| Some(size as u64))
                .chain([None, Some(3)])
                .collect::<Vec<_>>(),
        )
        .into();
        assert_eq!(infer_broadcast_type(&input, &leading).unwrap(), expected);
    }
    assert_eq!(
        infer_broadcast_type(&input, &[2, -1]),
        Err(InferenceError::InvalidBroadcastSize {
            dimension: 1,
            size: -1
        }),
    );
    assert!(infer_broadcast_type(&Type::Token, &[]).is_err());
}

#[test]
fn per_tensor_quantization_is_preserved_but_per_axis_is_not_in_the_operand_domain() {
    let quantized = |shape, axis| {
        Type::QuantizedTensor(QuantizedTensorType::new(
            shape,
            ElementType::I8,
            Some(-127),
            Some(127),
            ElementType::F32,
            axis,
            vec![0.25],
            vec![1],
        ))
    };
    let input = quantized(vec![Some(1)], None);
    assert_eq!(infer_reverse_type(&input, &[0]).unwrap(), input);
    assert_eq!(
        infer_broadcast_type(&input, &[2]).unwrap(),
        quantized(vec![Some(2), Some(1)], None)
    );
    let input = quantized(vec![Some(1)], Some(0));
    assert!(infer_reverse_type(&input, &[0]).is_err());
    assert!(infer_broadcast_type(&input, &[2]).is_err());
}

#[test]
fn verifier_rejects_changed_quantization_parameters_on_identity_element_transforms() {
    let quantized = |shape, scale, zero| {
        Type::QuantizedTensor(QuantizedTensorType::new(
            shape,
            ElementType::I8,
            None,
            None,
            ElementType::F32,
            None,
            vec![scale],
            vec![zero],
        ))
    };
    for (scale, zero, accepted) in [(0.25, 1, true), (0.5, 1, false), (0.25, 2, false)] {
        for (op, shape, attr) in [
            (StableHloOp::Reverse, vec![Some(3)], ("dimensions", vec![0])),
            (
                StableHloOp::Broadcast,
                vec![Some(2), Some(3)],
                ("broadcast_sizes", vec![2]),
            ),
        ] {
            let result = common::computation(
                op,
                vec![quantized(vec![Some(3)], 0.25, 1)],
                vec![quantized(shape, scale, zero)],
                [(attr.0.into(), Attribute::integers(attr.1))].into(),
            );
            assert_eq!(
                result.is_ok(),
                accepted,
                "{op:?}, scale {scale}, zero {zero}"
            );
        }
    }
}

#[test]
fn verifier_checks_input_bounds_on_reverse_and_broadcast_result_refinements() {
    for (size, accepted) in [(0, true), (5, true), (6, false)] {
        for (op, shape, attr) in [
            (StableHloOp::Reverse, vec![size, 3], ("dimensions", vec![0])),
            (
                StableHloOp::Broadcast,
                vec![2, size, 3],
                ("broadcast_sizes", vec![2]),
            ),
        ] {
            let result = common::computation(
                op,
                vec![bounded()],
                vec![TensorType::static_shape(ElementType::F32, shape).into()],
                [(attr.0.into(), Attribute::integers(attr.1))].into(),
            );
            assert_eq!(result.is_ok(), accepted, "{op:?} with size {size}");
        }
    }
}
