use super::{InferenceError, dimensions, encoding, with_shape_and_encoding};
use crate::{StableHloOp, Type};

/// Infers all-reduce results, also used by collective-reduce.
///
/// Follows `inferAllReduceOp` and `inferCollectiveReduceOp`: every result keeps
/// its input's shape and encoding, with elements selected by the common scalar
/// reducer return. Shapes and ranks of different inputs need not match.
/// Reducer arguments, accumulator promotion, roots, groups and channels remain
/// operation-verifier responsibilities.
pub fn infer_all_reduce_types(
    inputs: &[Type],
    accumulator: &Type,
) -> Result<Vec<Type>, InferenceError> {
    accumulator
        .verify()
        .map_err(|error| InferenceError::InvalidType {
            index: inputs.len(),
            error,
        })?;
    if !StableHloOp::AllReduce.accepts_variadic_operand_type(0, accumulator) {
        return Err(InferenceError::IncompatibleTypes {
            index: inputs.len(),
        });
    }
    let shape = dimensions(accumulator).expect("checked scalar domain");
    if !shape.is_empty() {
        return Err(InferenceError::NonScalarAccumulator {
            index: 0,
            rank: shape.len(),
        });
    }
    inputs
        .iter()
        .enumerate()
        .map(|(index, input)| {
            input
                .verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            if !StableHloOp::AllReduce.accepts_variadic_operand_type(0, input) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
            let result = with_shape_and_encoding(
                accumulator,
                dimensions(input).expect("checked input domain").to_vec(),
                encoding(input).cloned(),
            );
            result
                .verify()
                .map_err(|error| InferenceError::InvalidType { index, error })?;
            Ok(result)
        })
        .collect()
}
