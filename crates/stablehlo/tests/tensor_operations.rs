use stablehlo::{
    Attribute, Attributes, BuildError, DotDimensionNumbers, ElementType, Precision,
    QuantizedTensorType, StableHloAttribute, StableHloOp, TensorType, Type,
};

mod common;
use common::computation;

fn tensor(element: ElementType, shape: &[u64]) -> Type {
    TensorType::static_shape(element, shape.iter().copied()).into()
}

fn enumeration(name: &str, value: &str) -> Attribute {
    Attribute::Enum {
        dialect: "stablehlo".to_owned(),
        name: name.to_owned(),
        value: value.to_owned(),
    }
}

#[test]
fn real_dynamic_slice_checks_index_vector_types_lengths_and_result_rank() {
    for (start, limit, stride, output, accepted) in [
        (
            tensor(ElementType::Index, &[2]),
            tensor(ElementType::Index, &[2]),
            tensor(ElementType::Index, &[2]),
            tensor(ElementType::F32, &[2, 3]),
            true,
        ),
        (
            tensor(ElementType::I64, &[2]),
            tensor(ElementType::I32, &[2]),
            tensor(ElementType::I64, &[2]),
            tensor(ElementType::F32, &[2, 3]),
            false,
        ),
        (
            tensor(ElementType::I64, &[1]),
            tensor(ElementType::I64, &[1]),
            tensor(ElementType::I64, &[1]),
            tensor(ElementType::F32, &[2, 3]),
            false,
        ),
        (
            tensor(ElementType::I64, &[2]),
            tensor(ElementType::I64, &[2]),
            tensor(ElementType::I64, &[2]),
            tensor(ElementType::F32, &[6]),
            false,
        ),
    ] {
        assert_eq!(
            computation(
                StableHloOp::RealDynamicSlice,
                vec![tensor(ElementType::F32, &[4, 5]), start, limit, stride],
                vec![output],
                Attributes::new()
            )
            .is_ok(),
            accepted
        );
    }
}
#[test]
fn dot_covers_vector_matrix_forms_and_rejects_contracting_mismatches() {
    let f32 = |shape: &[u64]| tensor(ElementType::F32, shape);
    for (lhs, rhs, output) in [
        (vec![3], vec![3], vec![]),
        (vec![2, 3], vec![3], vec![2]),
        (vec![3], vec![3, 4], vec![4]),
        (vec![2, 3], vec![3, 4], vec![2, 4]),
    ] {
        assert!(
            computation(
                StableHloOp::Dot,
                vec![f32(&lhs), f32(&rhs)],
                vec![f32(&output)],
                Attributes::new()
            )
            .is_ok()
        );
    }
    for (lhs, rhs, output) in [
        (vec![], vec![3], vec![3]),
        (vec![2, 3], vec![4, 5], vec![2, 5]),
        (vec![2, 3], vec![3, 4], vec![4, 2]),
        (vec![2, 3, 4], vec![4], vec![2, 3]),
    ] {
        assert!(
            computation(
                StableHloOp::Dot,
                vec![f32(&lhs), f32(&rhs)],
                vec![f32(&output)],
                Attributes::new()
            )
            .is_err()
        );
    }
}

#[test]
fn broadcast_inserts_leading_dimensions_and_preserves_element_type() {
    let attributes = |sizes| [("broadcast_sizes".to_owned(), Attribute::integers(sizes))].into();
    let input = tensor(ElementType::F32, &[2, 3]);
    assert!(
        computation(
            StableHloOp::Broadcast,
            vec![input.clone()],
            vec![tensor(ElementType::F32, &[4, 5, 2, 3])],
            attributes(vec![4, 5])
        )
        .is_ok()
    );
    for (shape, element, sizes) in [
        (vec![4, 2, 4], ElementType::F32, vec![4]),
        (vec![4, 2, 3], ElementType::I32, vec![4]),
        (vec![4, 2, 3], ElementType::F32, vec![-4]),
    ] {
        assert!(
            computation(
                StableHloOp::Broadcast,
                vec![input.clone()],
                vec![tensor(element, &shape)],
                attributes(sizes)
            )
            .is_err()
        );
    }
}

#[test]
fn cholesky_requires_square_matrices_and_preserves_result_type() {
    assert!(
        computation(
            StableHloOp::Cholesky,
            vec![tensor(ElementType::C64, &[2, 3, 3])],
            vec![tensor(ElementType::C64, &[2, 3, 3])],
            Attributes::new()
        )
        .is_ok()
    );
    for (input, output) in [
        (vec![3], vec![3]),
        (vec![2, 3], vec![2, 3]),
        (vec![3, 3], vec![4, 4]),
    ] {
        assert!(
            computation(
                StableHloOp::Cholesky,
                vec![tensor(ElementType::F32, &input)],
                vec![tensor(ElementType::F32, &output)],
                Attributes::new()
            )
            .is_err()
        );
    }
}

fn triangular_attributes(left: bool, transpose: &str) -> Attributes {
    [
        ("left_side".to_owned(), Attribute::Bool(left)),
        ("lower".to_owned(), Attribute::Bool(true)),
        ("unit_diagonal".to_owned(), Attribute::Bool(false)),
        (
            "transpose_a".to_owned(),
            enumeration("transpose", transpose),
        ),
    ]
    .into()
}

#[test]
fn triangular_solve_checks_left_right_batching_and_transpose() {
    for (left, rhs) in [(true, vec![2, 3, 4]), (false, vec![2, 4, 3])] {
        for transpose in ["NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"] {
            assert!(
                computation(
                    StableHloOp::TriangularSolve,
                    vec![
                        tensor(ElementType::F32, &[2, 3, 3]),
                        tensor(ElementType::F32, &rhs)
                    ],
                    vec![tensor(ElementType::F32, &rhs)],
                    triangular_attributes(left, transpose)
                )
                .is_ok()
            );
        }
    }
    for (lhs, rhs, transpose) in [
        (vec![2, 3, 4], vec![2, 3, 4], "NO_TRANSPOSE"),
        (vec![2, 3, 3], vec![4, 3, 4], "NO_TRANSPOSE"),
        (vec![2, 3, 3], vec![2, 4, 4], "NO_TRANSPOSE"),
        (vec![2, 3, 3], vec![2, 3, 4], "TRANSPOSE_INVALID"),
        (vec![3], vec![3], "NO_TRANSPOSE"),
    ] {
        assert!(
            computation(
                StableHloOp::TriangularSolve,
                vec![
                    tensor(ElementType::F32, &lhs),
                    tensor(ElementType::F32, &rhs)
                ],
                vec![tensor(ElementType::F32, &rhs)],
                triangular_attributes(true, transpose)
            )
            .is_err()
        );
    }
}

fn fft_attributes(mode: &str, lengths: &[i64]) -> Attributes {
    [
        ("fft_type".to_owned(), enumeration("fft_type", mode)),
        (
            "fft_length".to_owned(),
            Attribute::integers(lengths.to_vec()),
        ),
    ]
    .into()
}

#[test]
fn fft_checks_all_modes_element_types_and_real_transform_shapes() {
    for (mode, input_element, output_element, input, output, lengths) in [
        (
            "FFT",
            ElementType::C64,
            ElementType::C64,
            vec![2, 8],
            vec![2, 8],
            vec![8],
        ),
        (
            "IFFT",
            ElementType::C128,
            ElementType::C128,
            vec![2, 8],
            vec![2, 8],
            vec![8],
        ),
        (
            "RFFT",
            ElementType::F32,
            ElementType::C64,
            vec![2, 8],
            vec![2, 5],
            vec![8],
        ),
        (
            "IRFFT",
            ElementType::C128,
            ElementType::F64,
            vec![2, 5],
            vec![2, 8],
            vec![8],
        ),
        (
            "RFFT",
            ElementType::F64,
            ElementType::C128,
            vec![2, 4, 8],
            vec![2, 4, 5],
            vec![4, 8],
        ),
        (
            "IRFFT",
            ElementType::C64,
            ElementType::F32,
            vec![2, 4, 5],
            vec![2, 4, 8],
            vec![4, 8],
        ),
        (
            "RFFT",
            ElementType::F32,
            ElementType::C64,
            vec![0],
            vec![0],
            vec![0],
        ),
        (
            "IRFFT",
            ElementType::C64,
            ElementType::F32,
            vec![0],
            vec![0],
            vec![0],
        ),
    ] {
        assert!(
            computation(
                StableHloOp::Fft,
                vec![tensor(input_element, &input)],
                vec![tensor(output_element, &output)],
                fft_attributes(mode, &lengths)
            )
            .is_ok(),
            "{mode} {input:?}"
        );
    }
    for (mode, input_element, output_element, input, output, lengths) in [
        (
            "FFT",
            ElementType::F32,
            ElementType::F32,
            vec![8],
            vec![8],
            vec![8],
        ),
        (
            "RFFT",
            ElementType::F16,
            ElementType::C64,
            vec![8],
            vec![5],
            vec![8],
        ),
        (
            "RFFT",
            ElementType::F32,
            ElementType::C64,
            vec![8],
            vec![5],
            vec![9],
        ),
        (
            "IRFFT",
            ElementType::C64,
            ElementType::F32,
            vec![4],
            vec![8],
            vec![8],
        ),
        (
            "FFT",
            ElementType::C64,
            ElementType::C64,
            vec![8],
            vec![8],
            vec![],
        ),
        (
            "FFT",
            ElementType::C64,
            ElementType::C64,
            vec![8],
            vec![8],
            vec![-1],
        ),
        (
            "FFT",
            ElementType::C64,
            ElementType::C64,
            vec![8],
            vec![8],
            vec![1, 1, 1, 8],
        ),
    ] {
        assert!(
            computation(
                StableHloOp::Fft,
                vec![tensor(input_element, &input)],
                vec![tensor(output_element, &output)],
                fft_attributes(mode, &lengths)
            )
            .is_err(),
            "{mode} {lengths:?}"
        );
    }
}

#[test]
fn bitcasts_preserve_bits_and_add_or_remove_only_the_minor_dimension() {
    for (input_element, input, output_element, output) in [
        (ElementType::F32, vec![2], ElementType::I8, vec![2, 4]),
        (ElementType::I8, vec![2, 4], ElementType::F32, vec![2]),
        (ElementType::F32, vec![], ElementType::I32, vec![]),
        (ElementType::C128, vec![2], ElementType::C64, vec![2, 2]),
    ] {
        assert!(
            computation(
                StableHloOp::BitcastConvert,
                vec![tensor(input_element, &input)],
                vec![tensor(output_element, &output)],
                Attributes::new()
            )
            .is_ok()
        );
    }
    for (input_element, input, output_element, output) in [
        (ElementType::F32, vec![2], ElementType::I8, vec![2, 3]),
        (ElementType::I8, vec![2], ElementType::F32, vec![2]),
        (ElementType::F32, vec![2], ElementType::I32, vec![2, 1]),
        (ElementType::C64, vec![2], ElementType::F64, vec![2]),
        (ElementType::F6E2M3Fn, vec![2], ElementType::I4, vec![2, 1]),
    ] {
        assert!(
            computation(
                StableHloOp::BitcastConvert,
                vec![tensor(input_element, &input)],
                vec![tensor(output_element, &output)],
                Attributes::new()
            )
            .is_err()
        );
    }
}

#[test]
fn batch_normalization_checks_feature_size_and_all_operand_result_shapes() {
    for opcode in [
        StableHloOp::BatchNormTraining,
        StableHloOp::BatchNormInference,
        StableHloOp::BatchNormGrad,
    ] {
        let data = tensor(ElementType::F32, &[2, 3, 4]);
        let channel = tensor(ElementType::F32, &[3]);
        let mut operands = match opcode {
            StableHloOp::BatchNormTraining => vec![data.clone(), channel.clone(), channel.clone()],
            StableHloOp::BatchNormInference => vec![
                data.clone(),
                channel.clone(),
                channel.clone(),
                channel.clone(),
                channel.clone(),
            ],
            _ => vec![
                data.clone(),
                channel.clone(),
                channel.clone(),
                channel.clone(),
                data.clone(),
            ],
        };
        let results = if opcode == StableHloOp::BatchNormInference {
            vec![data]
        } else {
            vec![data, channel.clone(), channel]
        };
        let attributes = || {
            [
                ("epsilon".to_owned(), Attribute::f32_bits(0)),
                ("feature_index".to_owned(), Attribute::integer(1)),
            ]
            .into()
        };
        assert!(computation(opcode, operands.clone(), results.clone(), attributes()).is_ok());
        operands[1] = tensor(ElementType::F32, &[4]);
        assert!(computation(opcode, operands, results, attributes()).is_err());
    }
}

fn quantized(shape: Vec<Option<u64>>, expressed: ElementType) -> Type {
    Type::QuantizedTensor(QuantizedTensorType::new(
        shape,
        ElementType::I8,
        None,
        None,
        expressed,
        None,
        vec![0.5],
        vec![0],
    ))
}

#[test]
fn quantize_and_dequantize_preserve_shape_and_expressed_type() {
    let float = tensor(ElementType::F32, &[2, 3]);
    let quant = quantized(vec![Some(2), Some(3)], ElementType::F32);
    assert!(
        computation(
            StableHloOp::UniformQuantize,
            vec![float.clone()],
            vec![quant.clone()],
            Attributes::new()
        )
        .is_ok()
    );
    assert!(
        computation(
            StableHloOp::UniformDequantize,
            vec![quant.clone()],
            vec![float.clone()],
            Attributes::new()
        )
        .is_ok()
    );
    assert!(
        computation(
            StableHloOp::UniformQuantize,
            vec![float],
            vec![quantized(vec![Some(2), Some(3)], ElementType::F64)],
            Attributes::new()
        )
        .is_err()
    );
    assert!(
        computation(
            StableHloOp::UniformDequantize,
            vec![quant],
            vec![tensor(ElementType::F32, &[2, 4])],
            Attributes::new()
        )
        .is_err()
    );
}

#[test]
fn infeed_requires_a_final_token_and_valid_layout_arrays() {
    assert!(
        computation(
            StableHloOp::Infeed,
            vec![Type::Token],
            vec![tensor(ElementType::F32, &[2]), Type::Token],
            [(
                "layout".to_owned(),
                Attribute::Array(vec![Attribute::Array(vec![Attribute::integer(0)])])
            )]
            .into()
        )
        .is_ok()
    );
    for results in [
        vec![],
        vec![tensor(ElementType::F32, &[2])],
        vec![Type::Token, Type::Token],
    ] {
        assert!(
            computation(
                StableHloOp::Infeed,
                vec![Type::Token],
                results,
                Attributes::new()
            )
            .is_err()
        );
    }
    assert!(
        computation(
            StableHloOp::Infeed,
            vec![Type::Token],
            vec![tensor(ElementType::F32, &[2]), Type::Token],
            [(
                "layout".to_owned(),
                Attribute::Array(vec![Attribute::Array(vec![Attribute::String(
                    "0".to_owned()
                )])])
            )]
            .into()
        )
        .is_err()
    );
}

#[test]
fn precision_configuration_length_is_at_most_two() {
    for length in 0..=3 {
        let attributes = [(
            "precision_config".to_owned(),
            Attribute::precision_config(vec![Precision::Default; length]),
        )]
        .into();
        let result = computation(
            StableHloOp::Dot,
            vec![
                tensor(ElementType::F32, &[2]),
                tensor(ElementType::F32, &[2]),
            ],
            vec![tensor(ElementType::F32, &[])],
            attributes,
        );
        assert_eq!(result.is_ok(), length <= 2);
    }
}

#[test]
fn dot_general_supports_hybrid_float_and_quantized_weights() {
    let attributes = || {
        [(
            "dot_dimension_numbers".to_owned(),
            Attribute::StableHlo(Box::new(StableHloAttribute::DotDimensionNumbers(
                DotDimensionNumbers {
                    lhs_contracting_dimensions: vec![1],
                    rhs_contracting_dimensions: vec![0],
                    ..Default::default()
                },
            ))),
        )]
        .into()
    };
    assert!(
        computation(
            StableHloOp::DotGeneral,
            vec![
                tensor(ElementType::F32, &[2, 3]),
                quantized(vec![Some(3), Some(4)], ElementType::F32)
            ],
            vec![tensor(ElementType::F32, &[2, 4])],
            attributes()
        )
        .is_ok()
    );
    assert!(matches!(
        computation(
            StableHloOp::DotGeneral,
            vec![
                tensor(ElementType::F32, &[2, 3]),
                quantized(vec![Some(3), Some(4)], ElementType::F64)
            ],
            vec![tensor(ElementType::F32, &[2, 4])],
            attributes()
        ),
        Err(BuildError::InvalidDotQuantization { .. })
    ));
}
