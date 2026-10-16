use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, StableHloOp, Type, infer_batch_norm_grad_types,
    infer_batch_norm_inference_type, infer_batch_norm_training_types,
};

mod common;
use common::computation;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn attrs(feature: usize) -> Attributes {
    [
        ("epsilon".into(), Attribute::f32_bits(0)),
        ("feature_index".into(), Attribute::integer(feature as i64)),
    ]
    .into()
}

#[test]
fn batch_norm_projects_only_operand_feature_bounds_to_vector_results() {
    let data = ty("tensor<?x?x4xf32, #stablehlo.bounds<8, 5, ?>>");
    let scale = ty("tensor<3xf32>");
    let offset = ty("tensor<?xf32, #stablehlo.bounds<4>>");
    let expected = vec![
        data.clone(),
        ty("tensor<?xf32, #stablehlo.bounds<5>>"),
        ty("tensor<?xf32, #stablehlo.bounds<5>>"),
    ];
    assert_eq!(
        infer_batch_norm_training_types(&data, &scale, &offset, 1).unwrap(),
        expected
    );
    assert_eq!(
        infer_batch_norm_grad_types(&data, &scale, &offset, &scale, &data, 1).unwrap(),
        expected
    );
    assert_eq!(
        infer_batch_norm_inference_type(&data, &scale, &offset, &scale, &offset, 1).unwrap(),
        data
    );
    let data = ty("tensor<2x3x4xf32>");
    assert_eq!(
        infer_batch_norm_training_types(&data, &scale, &offset, 1).unwrap(),
        vec![data, scale.clone(), scale]
    );
}

#[test]
fn batch_norm_quantized_results_inherit_data_parameters_not_scale_parameters() {
    let data = ty("tensor<2x3x!quant.uniform<i8:f32, 0.5:1>>");
    let channel = ty("tensor<3x!quant.uniform<i8:f32, 0.25:2>>");
    let inferred = infer_batch_norm_training_types(&data, &channel, &channel, 1).unwrap();
    assert_eq!(
        inferred,
        vec![
            data.clone(),
            ty("tensor<3x!quant.uniform<i8:f32, 0.5:1>>"),
            ty("tensor<3x!quant.uniform<i8:f32, 0.5:1>>")
        ]
    );
    assert!(
        computation(
            StableHloOp::BatchNormTraining,
            vec![data, channel.clone(), channel],
            inferred,
            attrs(1)
        )
        .is_ok()
    );
}

#[test]
fn batch_norm_allows_zero_features_and_drops_empty_projected_bounds() {
    let data = ty("tensor<?x0xf64, #stablehlo.bounds<8, ?>>");
    let channel = ty("tensor<0xf64>");
    let inferred = infer_batch_norm_training_types(&data, &channel, &channel, 1).unwrap();
    assert_eq!(
        inferred,
        vec![data.clone(), channel.clone(), channel.clone()]
    );
    assert!(
        computation(
            StableHloOp::BatchNormTraining,
            vec![data, channel.clone(), channel],
            inferred,
            attrs(1)
        )
        .is_ok()
    );
}

#[test]
fn batch_norm_rejects_invalid_feature_indices_ranks_and_baseline_elements() {
    let data = ty("tensor<2x3xf32>");
    let channel = ty("tensor<3xf32>");
    for feature in [2, usize::MAX] {
        assert!(infer_batch_norm_training_types(&data, &channel, &channel, feature).is_err());
    }
    for source in [
        "tensor<f32>",
        "tensor<1x3xf32>",
        "tensor<3xf64>",
        "tensor<3xi32>",
        "tensor<*xf32>",
        "tensor<3x!quant.uniform<i8:f32, 0.5>>",
    ] {
        assert!(
            infer_batch_norm_training_types(&data, &ty(source), &channel, 1).is_err(),
            "{source}"
        );
    }
    assert!(infer_batch_norm_training_types(&ty("tensor<f32>"), &channel, &channel, 0).is_err());
    assert!(
        infer_batch_norm_grad_types(
            &data,
            &channel,
            &channel,
            &channel,
            &ty("tensor<2x3xf64>"),
            1
        )
        .is_err()
    );
}

#[test]
fn batch_norm_shared_dynamic_feature_cannot_conceal_conflicting_operand_extents() {
    let data = ty("tensor<2x?xf32>");
    let dynamic = ty("tensor<?xf32>");
    let three = ty("tensor<3xf32>");
    let four = ty("tensor<4xf32>");
    assert!(infer_batch_norm_training_types(&data, &three, &four, 1).is_err());
    assert!(infer_batch_norm_inference_type(&data, &dynamic, &three, &four, &dynamic, 1).is_err());
    assert!(
        infer_batch_norm_grad_types(&data, &dynamic, &three, &dynamic, &ty("tensor<2x4xf32>"), 1)
            .is_err()
    );
    let bounded = ty("tensor<?xf32, #stablehlo.bounds<2>>");
    assert!(infer_batch_norm_training_types(&data, &three, &bounded, 1).is_err());
    assert!(
        infer_batch_norm_grad_types(
            &ty("tensor<?x?xf32, #stablehlo.bounds<2, 4>>"),
            &dynamic,
            &dynamic,
            &dynamic,
            &ty("tensor<3x3xf32>"),
            1
        )
        .is_err()
    );
}

#[test]
fn batch_norm_checks_all_result_witnesses_in_addition_to_all_inputs() {
    let data = ty("tensor<2x?xf32>");
    let dynamic = ty("tensor<?xf32>");
    for opcode in [
        StableHloOp::BatchNormTraining,
        StableHloOp::BatchNormGrad,
        StableHloOp::BatchNormInference,
    ] {
        let inputs = match opcode {
            StableHloOp::BatchNormTraining => vec![data.clone(), dynamic.clone(), dynamic.clone()],
            StableHloOp::BatchNormGrad => vec![
                data.clone(),
                dynamic.clone(),
                dynamic.clone(),
                dynamic.clone(),
                data.clone(),
            ],
            _ => vec![
                data.clone(),
                dynamic.clone(),
                dynamic.clone(),
                dynamic.clone(),
                ty("tensor<3xf32>"),
            ],
        };
        let outputs = if opcode == StableHloOp::BatchNormInference {
            vec![ty("tensor<2x4xf32>")]
        } else {
            vec![data.clone(), ty("tensor<3xf32>"), ty("tensor<4xf32>")]
        };
        assert!(
            computation(opcode, inputs, outputs, attrs(1)).is_err(),
            "{opcode:?}"
        );
    }
    let inputs = vec![data.clone(), ty("tensor<3xf32>"), dynamic.clone()];
    assert!(
        computation(
            StableHloOp::BatchNormTraining,
            inputs.clone(),
            vec![ty("tensor<2x4xf32>"), dynamic.clone(), dynamic.clone()],
            attrs(1)
        )
        .is_err()
    );
    assert!(
        computation(
            StableHloOp::BatchNormTraining,
            inputs.clone(),
            vec![
                data.clone(),
                ty("tensor<?xf32, #stablehlo.bounds<2>>"),
                dynamic.clone()
            ],
            attrs(1)
        )
        .is_err()
    );
    assert!(
        computation(
            StableHloOp::BatchNormTraining,
            inputs,
            vec![data, ty("tensor<3xf32>"), ty("tensor<3xf32>")],
            attrs(1)
        )
        .is_ok()
    );
}
