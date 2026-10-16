use super::{
    InferenceError, bound, bounds, bounds_encoding, dimensions as shape,
    shape_witnesses::common_shape, with_shape_and_encoding,
};
use crate::{StableHloOp, Type};

/// Infers reduction results from input types and scalar reducer return types.
///
/// Follows `TypeInference.cpp::inferReduceOp`: dimensions and bounds come from
/// the first input, while each result's element type comes from its reducer
/// return value. Unrelated input encodings are not propagated. This does not
/// verify the reducer body, initialization values or accumulator promotion.
/// All input shapes and bounds are checked together without refining results
/// from later inputs, including when contradictory axes are reduced away.
pub fn infer_reduce_types(
    inputs: &[Type],
    dimensions: &[i64],
    accumulators: &[Type],
) -> Result<Vec<Type>, InferenceError> {
    let witness = inputs.first().ok_or(InferenceError::EmptyInputs)?;
    if inputs.len() != accumulators.len() {
        return Err(InferenceError::AccumulatorCountMismatch {
            expected: inputs.len(),
            actual: accumulators.len(),
        });
    }
    for (index, ty) in inputs.iter().chain(accumulators).enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::Reduce.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let witness_shape = shape(witness).expect("validated tensor input");
    let rank = witness_shape.len();
    common_shape(&inputs.iter().collect::<Vec<_>>())?;
    let mut reduced = vec![false; rank];
    for &dimension in dimensions {
        let entry = usize::try_from(dimension)
            .ok()
            .and_then(|dimension| reduced.get_mut(dimension))
            .ok_or_else(|| InferenceError::InvalidDimensions {
                rank,
                dimensions: dimensions.to_vec(),
            })?;
        if std::mem::replace(entry, true) {
            return Err(InferenceError::InvalidDimensions {
                rank,
                dimensions: dimensions.to_vec(),
            });
        }
    }
    let retained = (0..rank).filter(|&axis| !reduced[axis]).collect::<Vec<_>>();
    let sizes = retained
        .iter()
        .map(|&axis| witness_shape[axis])
        .collect::<Vec<_>>();
    let encoding = bounds(witness)
        .filter(|bounds| !bounds.is_empty())
        .and_then(|_| bounds_encoding(retained.iter().map(|&axis| bound(witness, axis)).collect()));
    accumulators
        .iter()
        .enumerate()
        .map(|(index, accumulator)| {
            let accumulator_shape = shape(accumulator).expect("validated tensor accumulator");
            if !accumulator_shape.is_empty() {
                return Err(InferenceError::NonScalarAccumulator {
                    index,
                    rank: accumulator_shape.len(),
                });
            }
            let result = with_shape_and_encoding(accumulator, sizes.clone(), encoding.clone());
            result
                .verify()
                .map_err(|error| InferenceError::InvalidType {
                    index: inputs.len() + index,
                    error,
                })?;
            Ok(result)
        })
        .collect()
}
