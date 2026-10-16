use super::InferenceError;
use crate::{ElementType, StableHloOp, TensorType, Type};

/// The distribution selected by `stablehlo.rng`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngDistribution {
    Uniform,
    Normal,
}

/// The algorithm selected by `stablehlo.rng_bit_generator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngAlgorithm {
    Default,
    ThreeFry,
    Philox,
}

/// Infers bit-generator state and data results from an explicit output type.
///
/// The data shape cannot be inferred from the state: it is supplied by the
/// caller and must be static. State results preserve the initial state's exact
/// metadata. State element/rank and algorithm-specific lengths follow the
/// written specification's input domain and C2.
pub fn infer_rng_bit_generator_types(
    initial_state: &Type,
    output: &Type,
    algorithm: RngAlgorithm,
) -> Result<Vec<Type>, InferenceError> {
    validate_rng_state(initial_state, algorithm)?;
    output
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 1, error })?;
    if !StableHloOp::RngBitGenerator.accepts_fixed_result_type(1, output) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    Ok(vec![initial_state.clone(), output.clone()])
}

pub(crate) fn validate_rng_state(
    state: &Type,
    algorithm: RngAlgorithm,
) -> Result<(), InferenceError> {
    state
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    let Type::Tensor(tensor) = state else {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    };
    if tensor.element() != ElementType::Ui64 || tensor.rank() != 1 {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    let permitted: &[u64] = match algorithm {
        RngAlgorithm::Default => return Ok(()),
        RngAlgorithm::ThreeFry => &[2],
        RngAlgorithm::Philox => &[2, 3],
    };
    if !permitted.iter().any(|&size| {
        tensor.dimensions()[0].is_none_or(|extent| extent == size)
            && super::bound(state, 0).is_none_or(|bound| size <= bound)
    }) {
        return Err(InferenceError::IncompatibleTypes { index: 0 });
    }
    Ok(())
}

/// Infers RNG output dimensions from a constant shape, or its vector length.
///
/// Runtime shape values give dynamic dimensions. Constant values must fit the
/// shape operand's signed element representation and be nonnegative. The
/// result uses `b`'s elements and no encoding, following `inferRngOp`.
pub fn infer_rng_type(
    a: &Type,
    b: &Type,
    shape: &Type,
    shape_values: Option<&[i64]>,
    distribution: RngDistribution,
) -> Result<Type, InferenceError> {
    for (index, ty) in [a, b, shape].into_iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::Rng.accepts_fixed_operand_type(index, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let Type::Tensor(a) = a else {
        unreachable!("RNG schema requires nonquantized tensors");
    };
    let Type::Tensor(b) = b else {
        unreachable!("RNG schema requires nonquantized tensors");
    };
    let Type::Tensor(shape) = shape else {
        unreachable!("RNG schema requires a nonquantized shape tensor");
    };
    if a.element() != b.element()
        || (distribution == RngDistribution::Normal && !b.element().is_float())
    {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let rank = shape.dimensions()[0].expect("RNG schema requires a static vector");
    if let Some(values) = shape_values
        && values.len() as u64 != rank
    {
        return Err(InferenceError::AttributeLengthMismatch {
            attribute: "shape",
            expected: usize::try_from(rank)
                .map_err(|_| InferenceError::UnrepresentableRank { rank })?,
            actual: values.len(),
        });
    }
    let mut dimensions = Vec::new();
    dimensions
        .try_reserve_exact(
            usize::try_from(rank).map_err(|_| InferenceError::UnrepresentableRank { rank })?,
        )
        .map_err(|_| InferenceError::UnrepresentableRank { rank })?;
    if let Some(values) = shape_values {
        let width = shape.element().bit_width().unwrap_or(64);
        let maximum = ((1_u64 << (width - 1)) - 1) as i64;
        for (dimension, &size) in values.iter().enumerate() {
            if size < 0 || size > maximum {
                return Err(InferenceError::InvalidShapeExtent { dimension, size });
            }
            dimensions.push(Some(size as u64));
        }
    } else {
        dimensions.resize(rank as usize, None);
    }
    let result = Type::from(TensorType::new(b.element(), dimensions));
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 3, error })?;
    Ok(result)
}
