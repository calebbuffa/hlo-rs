use super::{InferenceError, dimensions, shape_witnesses::common_shape, with_shape_and_encoding};
use crate::{StableHloOp, Type};

/// Infers map results from inputs and the scalar returned by its computation.
///
/// Follows `inferMapOp`: shape comes from the final input, element metadata
/// from the computation's scalar result, and the result has no encoding.
/// Input witnesses must have compatible common shapes. Region argument and
/// terminator contracts remain part of operation verification.
pub fn infer_map_type(
    inputs: &[Type],
    computation_result: &Type,
    axes: &[i64],
) -> Result<Type, InferenceError> {
    for (index, ty) in inputs.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if !StableHloOp::Map.accepts_variadic_operand_type(0, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let shape = common_shape(&inputs.iter().collect::<Vec<_>>())?;
    if axes.len() != shape.len()
        || axes
            .iter()
            .enumerate()
            .any(|(axis, &value)| usize::try_from(value) != Ok(axis))
    {
        return Err(InferenceError::InvalidMapDimensions {
            rank: shape.len(),
            dimensions: axes.to_vec(),
        });
    }
    let index = inputs.len();
    computation_result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index, error })?;
    if !StableHloOp::Map.accepts_fixed_result_type(0, computation_result)
        || dimensions(computation_result) != Some(&[])
    {
        return Err(InferenceError::IncompatibleTypes { index });
    }
    let last = inputs.last().expect("common shape checked nonempty inputs");
    Ok(with_shape_and_encoding(
        computation_result,
        dimensions(last).expect("checked ranked inputs").to_vec(),
        None,
    ))
}
