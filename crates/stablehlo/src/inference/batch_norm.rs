use super::{
    InferenceError, Specificity, bound, bounds_encoding, dimensions, merge_dimension,
    with_shape_and_encoding,
};
use crate::{StableHloOp, Type, verification::tensor_elements_compatible};

/// Infers batch-normalization inference's operand-shaped result.
pub fn infer_batch_norm_inference_type(
    operand: &Type,
    scale: &Type,
    offset: &Type,
    mean: &Type,
    variance: &Type,
    feature_index: usize,
) -> Result<Type, InferenceError> {
    let inputs = [operand, scale, offset, mean, variance];
    validate(StableHloOp::BatchNormInference, &inputs, feature_index, &[])?;
    Ok(operand.clone())
}

/// Infers training's data result, mean and variance, in that order.
///
/// Follows `inferBatchNormOp`: result metadata comes from the data operand,
/// and feature-vector bounds are projected from that operand's feature axis.
/// Scale/offset witnesses validate feasibility without refining the output.
pub fn infer_batch_norm_training_types(
    operand: &Type,
    scale: &Type,
    offset: &Type,
    feature_index: usize,
) -> Result<Vec<Type>, InferenceError> {
    let inputs = [operand, scale, offset];
    validate(StableHloOp::BatchNormTraining, &inputs, feature_index, &[])?;
    Ok(training_results(operand, feature_index))
}

/// Infers the operand, scale and offset gradients, preserving input metadata.
pub fn infer_batch_norm_grad_types(
    operand: &Type,
    scale: &Type,
    mean: &Type,
    variance: &Type,
    grad_output: &Type,
    feature_index: usize,
) -> Result<Vec<Type>, InferenceError> {
    let inputs = [operand, scale, mean, variance, grad_output];
    validate(StableHloOp::BatchNormGrad, &inputs, feature_index, &[])?;
    Ok(training_results(operand, feature_index))
}

fn training_results(operand: &Type, feature: usize) -> Vec<Type> {
    let size = dimensions(operand).expect("validated data operand")[feature];
    let feature_type = with_shape_and_encoding(
        operand,
        vec![size],
        bounds_encoding(vec![bound(operand, feature)]),
    );
    vec![operand.clone(), feature_type.clone(), feature_type]
}

pub(crate) fn validate(
    opcode: StableHloOp,
    inputs: &[&Type],
    feature: usize,
    results: &[&Type],
) -> Result<(), InferenceError> {
    for (index, ty) in inputs.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !opcode.accepts_fixed_operand_type(index, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let data = inputs[0];
    let data_shape = dimensions(data).expect("checked operand domains");
    if feature >= data_shape.len() {
        return Err(InferenceError::DimensionOutOfBounds {
            dimension: feature,
            rank: data_shape.len(),
        });
    }
    let mut sizes = data_shape.to_vec();
    let mut bounds = (0..sizes.len())
        .map(|axis| bound(data, axis))
        .collect::<Vec<_>>();
    for (index, ty, multi) in inputs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, ty)| {
            (
                index,
                *ty,
                opcode == StableHloOp::BatchNormGrad && index == 4,
            )
        })
        .chain(
            results
                .iter()
                .enumerate()
                .map(|(index, ty)| (inputs.len() + index, *ty, index == 0)),
        )
    {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !tensor_elements_compatible(data, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
        let shape = dimensions(ty).ok_or(InferenceError::IncompatibleTypes { index })?;
        let expected_rank = if multi { sizes.len() } else { 1 };
        if shape.len() != expected_rank {
            return Err(InferenceError::RankMismatch {
                expected: expected_rank,
                actual: shape.len(),
            });
        }
        for (axis, &size) in shape.iter().enumerate() {
            let target = if multi { axis } else { feature };
            (sizes[target], bounds[target]) = merge_dimension(
                target,
                sizes[target],
                size,
                bounds[target],
                bound(ty, axis),
                Specificity::Most,
            )?;
        }
    }
    Ok(())
}
