use stablehlo::{
    Attribute, Attributes, ConvDimensionNumbers, ElementType, QuantizedTensorType, ScalarValue,
    StableHloAttribute, StableHloOp, TensorType, Type,
};

mod common;
use common::computation;

fn tensor(element: ElementType, dimensions: &[u64]) -> Type {
    TensorType::static_shape(element, dimensions.iter().copied()).into()
}

fn dimensions() -> ConvDimensionNumbers {
    ConvDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 2,
        input_spatial_dimensions: vec![1],
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 2,
        kernel_spatial_dimensions: vec![0],
        output_batch_dimension: 0,
        output_feature_dimension: 2,
        output_spatial_dimensions: vec![1],
    }
}

fn attributes(dimensions: ConvDimensionNumbers, feature: i64, batch: i64) -> Attributes {
    [
        (
            "dimension_numbers".into(),
            Attribute::StableHlo(Box::new(StableHloAttribute::ConvDimensionNumbers(
                dimensions,
            ))),
        ),
        ("feature_group_count".into(), Attribute::integer(feature)),
        ("batch_group_count".into(), Attribute::integer(batch)),
    ]
    .into()
}

fn convolution(lhs: &[u64], rhs: &[u64], output: &[u64], attributes: Attributes) -> bool {
    computation(
        StableHloOp::Convolution,
        vec![tensor(ElementType::F32, lhs), tensor(ElementType::F32, rhs)],
        vec![tensor(ElementType::F32, output)],
        attributes,
    )
    .is_ok()
}

#[test]
fn convolution_infers_general_dimension_permutations() {
    assert!(convolution(
        &[2, 7, 4],
        &[3, 4, 6],
        &[2, 5, 6],
        attributes(dimensions(), 1, 1)
    ));
    let permuted = ConvDimensionNumbers {
        output_batch_dimension: 2,
        output_feature_dimension: 0,
        ..dimensions()
    };
    assert!(convolution(
        &[2, 7, 4],
        &[3, 4, 6],
        &[6, 5, 2],
        attributes(permuted, 1, 1)
    ));
    assert!(!convolution(
        &[2, 7, 4],
        &[3, 4, 6],
        &[2, 6, 6],
        attributes(dimensions(), 1, 1)
    ));
}

#[test]
fn convolution_checks_group_divisibility_and_exclusivity() {
    assert!(convolution(
        &[2, 7, 4],
        &[3, 2, 6],
        &[2, 5, 6],
        attributes(dimensions(), 2, 1)
    ));
    assert!(convolution(
        &[4, 7, 4],
        &[3, 4, 6],
        &[2, 5, 6],
        attributes(dimensions(), 1, 2)
    ));
    for (lhs, rhs, feature, batch) in [
        ([2, 7, 4], [3, 2, 6], 2, 2),
        ([3, 7, 4], [3, 4, 6], 1, 2),
        ([2, 7, 3], [3, 2, 6], 2, 1),
        ([2, 7, 4], [3, 3, 6], 2, 1),
        ([2, 7, 4], [3, 2, 5], 2, 1),
        ([2, 7, 4], [3, 4, 5], 1, 2),
    ] {
        assert!(!convolution(
            &lhs,
            &rhs,
            &[2, 5, 6],
            attributes(dimensions(), feature, batch)
        ));
    }
}

#[test]
fn convolution_requires_complete_in_range_permutations_and_equal_ranks() {
    for dimensions in [
        ConvDimensionNumbers {
            input_feature_dimension: 0,
            ..dimensions()
        },
        ConvDimensionNumbers {
            kernel_spatial_dimensions: vec![3],
            ..dimensions()
        },
        ConvDimensionNumbers {
            output_spatial_dimensions: vec![],
            ..dimensions()
        },
        ConvDimensionNumbers {
            output_batch_dimension: -1,
            ..dimensions()
        },
    ] {
        assert!(!convolution(
            &[2, 7, 4],
            &[3, 4, 6],
            &[2, 5, 6],
            attributes(dimensions, 1, 1)
        ));
    }
    assert!(!convolution(
        &[2, 7, 4],
        &[3, 4],
        &[2, 5, 6],
        attributes(dimensions(), 1, 1)
    ));
}

#[test]
fn convolution_validates_window_vectors_and_infers_padding_and_dilations() {
    let mut configuration = attributes(dimensions(), 1, 1);
    configuration.extend([
        ("window_strides".into(), Attribute::integers(vec![4])),
        ("lhs_dilation".into(), Attribute::integers(vec![2])),
        ("rhs_dilation".into(), Attribute::integers(vec![2])),
        (
            "padding".into(),
            Attribute::DenseElements {
                ty: Box::new(tensor(ElementType::I64, &[1, 2])),
                values: vec![ScalarValue::Integer(2), ScalarValue::Integer(1)],
            },
        ),
        ("window_reversal".into(), Attribute::bools(vec![true])),
    ]);
    assert!(convolution(
        &[2, 7, 4],
        &[3, 4, 6],
        &[2, 3, 6],
        configuration
    ));
    for name in ["window_strides", "lhs_dilation", "rhs_dilation"] {
        for vector in [vec![0], vec![-1], vec![], vec![1, 1]] {
            let mut configuration = attributes(dimensions(), 1, 1);
            configuration.insert(name.into(), Attribute::integers(vector));
            assert!(!convolution(
                &[2, 7, 4],
                &[3, 4, 6],
                &[2, 5, 6],
                configuration
            ));
        }
    }
    let mut configuration = attributes(dimensions(), 1, 1);
    configuration.insert("window_reversal".into(), Attribute::bools(vec![]));
    assert!(!convolution(
        &[2, 7, 4],
        &[3, 4, 6],
        &[2, 5, 6],
        configuration
    ));
}

#[test]
fn convolution_handles_zero_input_and_kernel_extents() {
    assert!(convolution(
        &[2, 0, 4],
        &[3, 4, 6],
        &[2, 0, 6],
        attributes(dimensions(), 1, 1)
    ));
    assert!(convolution(
        &[2, 7, 4],
        &[0, 4, 6],
        &[2, 8, 6],
        attributes(dimensions(), 1, 1)
    ));
    assert!(convolution(
        &[2, 0, 4],
        &[0, 4, 6],
        &[2, 0, 6],
        attributes(dimensions(), 1, 1)
    ));
}

#[test]
fn convolution_supports_mixed_fp8_formats_but_keeps_lhs_result_format() {
    let inputs = vec![
        tensor(ElementType::F8E4M3Fn, &[2, 7, 4]),
        tensor(ElementType::F8E5M2, &[3, 4, 6]),
    ];
    for (element, accepted) in [
        (ElementType::F8E4M3Fn, true),
        (ElementType::F8E5M2, false),
        (ElementType::F32, false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::Convolution,
                inputs.clone(),
                vec![tensor(element, &[2, 5, 6])],
                attributes(dimensions(), 1, 1)
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn hybrid_quantized_convolution_enforces_output_channel_axis_and_expressed_type() {
    for (axis, expressed, accepted) in [
        (2, ElementType::F32, true),
        (0, ElementType::F32, false),
        (2, ElementType::F64, false),
    ] {
        let size = if axis == 0 { 3 } else { 6 };
        let rhs = Type::QuantizedTensor(QuantizedTensorType::new(
            vec![Some(3), Some(4), Some(6)],
            ElementType::I8,
            None,
            None,
            expressed,
            Some(axis),
            vec![0.5; size],
            vec![1; size],
        ));
        assert_eq!(
            computation(
                StableHloOp::Convolution,
                vec![tensor(ElementType::F32, &[2, 7, 4]), rhs],
                vec![tensor(ElementType::F32, &[2, 5, 6])],
                attributes(dimensions(), 1, 1)
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn dynamic_convolution_checks_padding_shape_and_non_spatial_results() {
    for (padding, output, accepted) in [
        (vec![1, 2], vec![2, 8, 6], true),
        (vec![2, 2], vec![2, 8, 6], false),
        (vec![1, 2], vec![3, 8, 6], false),
        (vec![1, 2], vec![2, 8, 7], false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::DynamicConv,
                vec![
                    tensor(ElementType::F32, &[2, 7, 4]),
                    tensor(ElementType::F32, &[3, 4, 6]),
                    tensor(ElementType::I64, &padding)
                ],
                vec![tensor(ElementType::F32, &output)],
                attributes(dimensions(), 1, 1)
            )
            .is_ok(),
            accepted
        );
    }
}
