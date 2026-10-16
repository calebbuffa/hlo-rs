use stablehlo::{
    Attribute, ElementType, FftType, InferenceError, StableHloAttribute, StableHloOp, TensorType,
    Type, TypeExtensions, infer_fft_type,
};

mod common;

fn tensor(element: ElementType, shape: Vec<Option<u64>>, bounds: Option<Vec<i64>>) -> Type {
    let mut ty = TensorType::new(element, shape);
    if let Some(bounds) = bounds {
        ty = ty.with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions { bounds }),
        )));
    }
    ty.into()
}

#[test]
fn real_and_inverse_transforms_match_every_small_length_at_both_precisions() {
    for (real, complex) in [
        (ElementType::F32, ElementType::C64),
        (ElementType::F64, ElementType::C128),
    ] {
        for length in 0..=16 {
            let compressed = if length == 0 { 0 } else { length / 2 + 1 };
            for rank in 1..=3 {
                let mut lengths = vec![3; rank];
                lengths[rank - 1] = length as i64;
                let mut real_shape = vec![Some(2)];
                real_shape.extend(lengths.iter().map(|&size| Some(size as u64)));
                let mut complex_shape = real_shape.clone();
                *complex_shape.last_mut().unwrap() = Some(compressed);
                let real = tensor(real, real_shape, None);
                let complex = tensor(complex, complex_shape, None);
                assert_eq!(
                    infer_fft_type(&real, FftType::Rfft, &lengths).unwrap(),
                    complex
                );
                assert_eq!(
                    infer_fft_type(&complex, FftType::Irfft, &lengths).unwrap(),
                    real
                );
            }
        }
    }
}

#[test]
fn complex_transforms_preserve_all_shapes_and_metadata() {
    for element in [ElementType::C64, ElementType::C128] {
        for input in [
            tensor(element, vec![None, Some(4)], Some(vec![8, -1])),
            Type::from(TensorType::static_shape(element, [3, 4]).with_encoding(
                Attribute::Opaque {
                    dialect: Some("consumer".into()),
                    syntax: "layout".into(),
                },
            )),
        ] {
            for mode in [FftType::Fft, FftType::Ifft] {
                for lengths in [vec![0], vec![2, 8], vec![i64::MAX]] {
                    assert_eq!(infer_fft_type(&input, mode, &lengths).unwrap(), input);
                }
            }
        }
    }
}

#[test]
fn real_conversion_preserves_batch_bounds_and_clears_the_transformed_bound() {
    let input = tensor(
        ElementType::F32,
        vec![None, None, None],
        Some(vec![8, 5, 10]),
    );
    assert_eq!(
        infer_fft_type(&input, FftType::Rfft, &[5, 10]).unwrap(),
        tensor(
            ElementType::C64,
            vec![None, None, Some(6)],
            Some(vec![8, 5, -1])
        ),
    );
    let input = tensor(ElementType::C128, vec![None, None], Some(vec![8, 6]));
    assert_eq!(
        infer_fft_type(&input, FftType::Irfft, &[10]).unwrap(),
        tensor(ElementType::F64, vec![None, Some(10)], Some(vec![8, -1])),
    );
    assert_eq!(
        infer_fft_type(
            &tensor(ElementType::F32, vec![None], Some(vec![10])),
            FftType::Rfft,
            &[10]
        )
        .unwrap(),
        tensor(ElementType::C64, vec![Some(6)], None),
    );
    assert!(matches!(
        infer_fft_type(
            &tensor(ElementType::F32, vec![None], Some(vec![9])),
            FftType::Rfft,
            &[10]
        ),
        Err(InferenceError::BoundExceeded {
            dimension: 0,
            size: 10,
            bound: 9
        }),
    ));
}

#[test]
fn invalid_fft_ranks_types_lengths_shapes_and_metadata_are_rejected() {
    let input = tensor(ElementType::C64, vec![Some(3)], None);
    for lengths in [vec![], vec![3, 3], vec![3, 3, 3, 3]] {
        assert!(matches!(
            infer_fft_type(&input, FftType::Fft, &lengths),
            Err(InferenceError::InvalidFftRank { .. })
        ));
    }
    assert!(matches!(
        infer_fft_type(&input, FftType::Fft, &[-1]),
        Err(InferenceError::InvalidFftLength {
            dimension: 0,
            length: -1
        })
    ));
    for (element, mode) in [
        (ElementType::F16, FftType::Rfft),
        (ElementType::I32, FftType::Rfft),
        (ElementType::F32, FftType::Fft),
        (ElementType::F64, FftType::Ifft),
        (ElementType::F32, FftType::Irfft),
        (ElementType::C64, FftType::Rfft),
    ] {
        assert!(matches!(
            infer_fft_type(&tensor(element, vec![Some(3)], None), mode, &[3]),
            Err(InferenceError::IncompatibleTypes { index: 0 })
        ));
    }
    for (element, mode, lengths) in [
        (ElementType::F32, FftType::Rfft, vec![4]),
        (ElementType::C64, FftType::Irfft, vec![8]),
    ] {
        assert!(matches!(
            infer_fft_type(&tensor(element, vec![Some(3)], None), mode, &lengths),
            Err(InferenceError::DimensionMismatch { .. })
        ));
    }
    assert!(infer_fft_type(&Type::Token, FftType::Fft, &[3]).is_err());
    assert!(matches!(
        infer_fft_type(
            &tensor(ElementType::C64, vec![None], Some(vec![-2])),
            FftType::Fft,
            &[3]
        ),
        Err(InferenceError::InvalidType { index: 0, .. }),
    ));
    assert!(infer_fft_type(&tensor(ElementType::C64, vec![], None), FftType::Fft, &[3]).is_err());
}

#[test]
fn zero_length_irfft_uses_specification_zero_extent_rule() {
    infer_fft_type(
        &tensor(ElementType::C64, vec![Some(0)], None),
        FftType::Irfft,
        &[0],
    )
    .unwrap();
    assert!(
        infer_fft_type(
            &tensor(ElementType::C64, vec![Some(1)], None),
            FftType::Irfft,
            &[0]
        )
        .is_err()
    );
    assert_eq!(
        infer_fft_type(
            &tensor(ElementType::C64, vec![None], None),
            FftType::Irfft,
            &[0]
        )
        .unwrap(),
        tensor(ElementType::F32, vec![Some(0)], None),
    );
}

#[test]
fn verifier_rejects_static_batch_refinement_above_operand_bound() {
    for (batch, accepted) in [(8, true), (9, false)] {
        let result = common::computation(
            StableHloOp::Fft,
            vec![tensor(
                ElementType::F32,
                vec![None, Some(10)],
                Some(vec![8, -1]),
            )],
            vec![tensor(ElementType::C64, vec![Some(batch), Some(6)], None)],
            [
                (
                    "fft_type".into(),
                    Attribute::Enum {
                        dialect: "stablehlo".into(),
                        name: "fft_type".into(),
                        value: "RFFT".into(),
                    },
                ),
                ("fft_length".into(), Attribute::integers(vec![10])),
            ]
            .into(),
        );
        assert_eq!(result.is_ok(), accepted);
    }
}
