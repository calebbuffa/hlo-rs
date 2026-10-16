use super::{InferenceError, Specificity, bound, dimensions, merge_dimension};
use crate::{StableHloOp, Type, verification::tensor_elements_compatible};

/// Infers Cholesky's unchanged operand type, including encoding and bounds.
///
/// Follows `inferCholeskyOp`. The matrix's two minor dimensions must admit
/// an equal runtime extent; inference does not make either dimension static.
pub fn infer_cholesky_type(operand: &Type) -> Result<Type, InferenceError> {
    let shape = matrix(StableHloOp::Cholesky, operand, 0)?;
    square(operand, shape.len())?;
    Ok(operand.clone())
}

/// Infers triangular-solve's unchanged rhs type, as in `inferTriangularSolveOp`.
///
/// Checks operand domains, equal ranks and elements, batching dimensions and
/// the common matrix extent. Transpose, lower-triangle and unit-diagonal
/// attributes do not affect result inference and are verified on the operation.
pub fn infer_triangular_solve_type(
    a: &Type,
    b: &Type,
    left_side: bool,
) -> Result<Type, InferenceError> {
    let a_shape = matrix(StableHloOp::TriangularSolve, a, 0)?;
    let b_shape = matrix(StableHloOp::TriangularSolve, b, 1)?;
    if !tensor_elements_compatible(a, b) {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let rank = a_shape.len();
    if rank != b_shape.len() {
        return Err(InferenceError::RankMismatch {
            expected: rank,
            actual: b_shape.len(),
        });
    }
    let (size, upper) = square(a, rank)?;
    let b_axis = rank - if left_side { 2 } else { 1 };
    merge_dimension(
        b_axis,
        size,
        b_shape[b_axis],
        upper,
        bound(b, b_axis),
        Specificity::Most,
    )?;
    for axis in 0..rank - 2 {
        merge_dimension(
            axis,
            a_shape[axis],
            b_shape[axis],
            bound(a, axis),
            bound(b, axis),
            Specificity::Most,
        )?;
    }
    Ok(b.clone())
}

fn matrix(opcode: StableHloOp, ty: &Type, index: usize) -> Result<&[Option<u64>], InferenceError> {
    ty.verify()
        .map_err(|error| InferenceError::InvalidType { index, error })?;
    if !opcode.accepts_fixed_operand_type(index, ty) {
        return Err(InferenceError::IncompatibleTypes { index });
    }
    let shape = dimensions(ty).expect("checked ranked operand domain");
    if shape.len() < 2 {
        return Err(InferenceError::InvalidMatrixRank {
            index,
            rank: shape.len(),
        });
    }
    Ok(shape)
}

fn square(ty: &Type, rank: usize) -> Result<(Option<u64>, Option<u64>), InferenceError> {
    let shape = dimensions(ty).expect("checked matrix");
    merge_dimension(
        rank - 1,
        shape[rank - 2],
        shape[rank - 1],
        bound(ty, rank - 2),
        bound(ty, rank - 1),
        Specificity::Most,
    )
}
