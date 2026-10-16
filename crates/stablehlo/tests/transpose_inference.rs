use stablehlo::{
    Attribute, BuildError, ElementType, InferenceError, QuantizedTensorType, StableHloAttribute,
    StableHloOp, TensorType, Type, TypeExtensions, infer_transpose_type,
};

mod common;

fn bounded(shape: Vec<Option<u64>>, bounds: Vec<i64>) -> Type {
    TensorType::new(ElementType::F32, shape)
        .with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
        )))
        .into()
}

#[test]
fn every_rank_three_permutation_moves_shapes_and_bounds_together() {
    let shape = [None, Some(3), None];
    let bounds = [5, -1, 7];
    let operand = bounded(shape.to_vec(), bounds.to_vec());
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        assert_eq!(
            infer_transpose_type(&operand, &permutation).unwrap(),
            bounded(
                permutation
                    .iter()
                    .map(|&axis| shape[axis as usize])
                    .collect(),
                permutation
                    .iter()
                    .map(|&axis| bounds[axis as usize])
                    .collect(),
            ),
        );
    }
}

#[test]
fn invalid_permutations_and_invalid_operands_produce_explicit_errors() {
    let operand: Type = TensorType::static_shape(ElementType::F32, [2, 3]).into();
    for permutation in [
        vec![],
        vec![0],
        vec![0, 1, 2],
        vec![-1, 0],
        vec![0, 0],
        vec![0, i64::MAX],
    ] {
        assert_eq!(
            infer_transpose_type(&operand, &permutation),
            Err(InferenceError::InvalidPermutation {
                rank: 2,
                permutation
            }),
        );
    }
    assert_eq!(
        infer_transpose_type(&Type::Token, &[]),
        Err(InferenceError::IncompatibleTypes { index: 0 }),
    );
    assert!(matches!(
        infer_transpose_type(&bounded(vec![None], vec![1, 2]), &[0]),
        Err(InferenceError::InvalidType { index: 0, .. }),
    ));
}

#[test]
fn per_axis_quantization_uses_inverse_permutation_without_changing_parameters() {
    for input_axis in 0..3 {
        let operand = Type::QuantizedTensor(QuantizedTensorType::new(
            vec![Some(2), Some(2), Some(2)],
            ElementType::I8,
            Some(-127),
            Some(127),
            ElementType::F32,
            Some(input_axis),
            vec![0.25, 0.5],
            vec![1, 2],
        ));
        for permutation in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
            let inferred = infer_transpose_type(&operand, &permutation).unwrap();
            let expected = Type::QuantizedTensor(QuantizedTensorType::new(
                vec![Some(2), Some(2), Some(2)],
                ElementType::I8,
                Some(-127),
                Some(127),
                ElementType::F32,
                permutation
                    .iter()
                    .position(|&axis| axis as usize == input_axis),
                vec![0.25, 0.5],
                vec![1, 2],
            ));
            assert_eq!(inferred, expected);
            common::computation(
                StableHloOp::Transpose,
                vec![operand.clone()],
                vec![inferred],
                [(
                    "permutation".into(),
                    Attribute::integers(permutation.to_vec()),
                )]
                .into(),
            )
            .unwrap();
        }
    }
}

#[test]
fn quantization_axis_remapping_handles_nonuniform_axis_sizes_and_dynamic_bounds() {
    let operand = Type::QuantizedTensor(
        QuantizedTensorType::new(
            vec![Some(2), Some(3), None],
            ElementType::I8,
            None,
            None,
            ElementType::F32,
            Some(1),
            vec![0.25, 0.5, 0.75],
            vec![1, 2, 3],
        )
        .with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions {
                bounds: vec![-1, -1, 8],
            }),
        ))),
    );
    let result = infer_transpose_type(&operand, &[2, 0, 1]).unwrap();
    let Type::QuantizedTensor(tensor) = &result else {
        panic!("quantization was lost");
    };
    assert_eq!(tensor.dimensions(), &[None, Some(2), Some(3)]);
    assert_eq!(tensor.quantization_dimension(), Some(2));
    assert_eq!(tensor.zero_points(), &[1, 2, 3]);
    common::computation(
        StableHloOp::Transpose,
        vec![operand],
        vec![result],
        [("permutation".into(), Attribute::integers(vec![2, 0, 1]))].into(),
    )
    .unwrap();
}

#[test]
fn scalar_plain_and_nonbounds_encoded_tensors_keep_their_metadata() {
    let scalar: Type = TensorType::scalar(ElementType::C128).into();
    assert_eq!(infer_transpose_type(&scalar, &[]).unwrap(), scalar);
    let encoding = Attribute::Opaque {
        dialect: Some("consumer".into()),
        syntax: "layout".into(),
    };
    let operand: Type = TensorType::static_shape(ElementType::F32, [2, 3])
        .with_encoding(encoding.clone())
        .into();
    let expected: Type = TensorType::static_shape(ElementType::F32, [3, 2])
        .with_encoding(encoding)
        .into();
    assert_eq!(infer_transpose_type(&operand, &[1, 0]).unwrap(), expected);
    assert_eq!(
        infer_transpose_type(&bounded(vec![Some(2), None], vec![-1, -1]), &[1, 0]).unwrap(),
        Type::from(TensorType::new(ElementType::F32, vec![None, Some(2)])),
    );
    let operand = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::I8,
        Some(-127),
        Some(127),
        ElementType::F32,
        None,
        vec![0.25],
        vec![1],
    ));
    let expected = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(3), Some(2)],
        ElementType::I8,
        Some(-127),
        Some(127),
        ElementType::F32,
        None,
        vec![0.25],
        vec![1],
    ));
    assert_eq!(infer_transpose_type(&operand, &[1, 0]).unwrap(), expected);
}

#[test]
fn verifier_checks_transposed_upper_bounds_against_static_result_refinements() {
    for (size, accepted) in [(5, true), (6, false)] {
        let result = common::computation(
            StableHloOp::Transpose,
            vec![bounded(vec![Some(3), None], vec![-1, 5])],
            vec![TensorType::static_shape(ElementType::F32, [size, 3]).into()],
            [("permutation".into(), Attribute::integers(vec![1, 0]))].into(),
        );
        if accepted {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(BuildError::InvalidResultType {
                    operation: "transpose",
                    ..
                })
            ));
        }
    }
}
