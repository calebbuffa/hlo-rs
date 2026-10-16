use super::{
    InferenceError, dimensions, encoding, infer_most_specific_type, with_shape_and_encoding,
};
use crate::verification::tensor_elements_compatible;
use crate::{StableHloOp, Type};

/// Infers `select` using the most-specific value types, as in `inferSelectOp`.
///
/// A scalar predicate broadcasts; a nonscalar predicate must have a compatible
/// shape but does not refine the returned type. Quantization parameters come
/// from `on_true`; the operands may use different parameters with the same
/// baseline element type.
pub fn infer_select_type(
    predicate: &Type,
    on_true: &Type,
    on_false: &Type,
) -> Result<Type, InferenceError> {
    let predicate_shape = checked_shape(predicate, 0, StableHloOp::Select)?;
    checked_shape(on_true, 1, StableHloOp::Select)?;
    checked_shape(on_false, 2, StableHloOp::Select)?;
    let inferred = infer_most_specific_type(&[on_true.clone(), on_false.clone()])?;
    if !predicate_shape.is_empty() {
        let witness = with_shape_and_encoding(
            on_true,
            predicate_shape.to_vec(),
            encoding(predicate).cloned(),
        );
        infer_most_specific_type(&[inferred.clone(), witness])?;
    }
    Ok(inferred)
}

/// Infers `clamp`, retaining the operand type exactly, as in `inferClampOp`.
///
/// Bounds may be scalars or shape-compatible tensors. All operands must share
/// a baseline element type, not necessarily quantization scales or zero points.
pub fn infer_clamp_type(min: &Type, operand: &Type, max: &Type) -> Result<Type, InferenceError> {
    let min_shape = checked_shape(min, 0, StableHloOp::Clamp)?;
    checked_shape(operand, 1, StableHloOp::Clamp)?;
    let max_shape = checked_shape(max, 2, StableHloOp::Clamp)?;
    let mut witnesses = vec![operand.clone()];
    for (index, bound, shape) in [(0, min, min_shape), (2, max, max_shape)] {
        if !tensor_elements_compatible(bound, operand) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
        if !shape.is_empty() {
            witnesses.push(bound.clone());
        }
    }
    infer_most_specific_type(&witnesses)?;
    Ok(operand.clone())
}

fn checked_shape(
    ty: &Type,
    index: usize,
    opcode: StableHloOp,
) -> Result<&[Option<u64>], InferenceError> {
    ty.verify()
        .map_err(|error| InferenceError::InvalidType { index, error })?;
    if !opcode.accepts_fixed_operand_type(index, ty) {
        return Err(InferenceError::IncompatibleTypes { index });
    }
    dimensions(ty).ok_or(InferenceError::IncompatibleTypes { index })
}
