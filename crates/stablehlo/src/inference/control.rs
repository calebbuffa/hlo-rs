//! Type inference for StableHLO control-flow and collective operations.
use crate::Diagnostic;
use crate::types::{ElementType, TensorType, TupleType};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

fn ranked_parts(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let object = ty.deref(ctx);
    let tensor = object.downcast_ref::<TensorType>()?;
    Some((tensor.dimensions.clone(), tensor.element))
}

/// Infers a tuple type containing the supplied element types in order.
///
/// Every element type is verified; invalid or unsupported types return an
/// error. An empty input produces the empty tuple type.
pub fn infer_tuple_type(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    for &ty in operands {
        ty.deref(ctx)
            .verify(ctx)
            .map_err(|_| Diagnostic::UnsupportedType)?;
    }
    Ok(TupleType::get(ctx, operands.to_vec()).into())
}
/// Returns the type at `index` in a tuple operand.
///
/// Negative indices, indices outside the tuple, and non-tuple operands return
/// an error.
pub fn infer_get_tuple_element_type(
    ctx: &Context,
    operand: TypeHandle,
    index: i32,
) -> Result<TypeHandle> {
    let object = operand.deref(ctx);
    let tuple = object
        .downcast_ref::<TupleType>()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    usize::try_from(index)
        .ok()
        .and_then(|i| tuple.elements.get(i).copied())
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())
}
/// Returns the operand types unchanged after checking that each is a valid
/// StableHLO value type.
pub fn infer_optimization_barrier_types(
    ctx: &Context,
    operands: &[TypeHandle],
) -> Result<Vec<TypeHandle>> {
    check_values(ctx, operands)?;
    Ok(operands.to_vec())
}
/// Returns the loop-carried types unchanged after validating them as value
/// types.
pub fn infer_while_types(ctx: &Context, operands: &[TypeHandle]) -> Result<Vec<TypeHandle>> {
    check_values(ctx, operands)?;
    Ok(operands.to_vec())
}
fn check_values(ctx: &Context, types: &[TypeHandle]) -> Result<()> {
    for &ty in types {
        ty.deref(ctx)
            .verify(ctx)
            .map_err(|_| Diagnostic::UnsupportedType)?;
        if !crate::types::is_value_type(ctx, ty) {
            return Err(Diagnostic::UnsupportedType.into());
        }
    }
    Ok(())
}
/// Joins the result types of the true and false branches.
///
/// The predicate must be a scalar `i1` tensor. Each branch must return the
/// same number of results; corresponding ranked tensors must have equal element
/// types and compatible ranks/dimensions. Dynamic dimensions remain dynamic.
pub fn infer_if_types(
    ctx: &Context,
    predicate: TypeHandle,
    true_results: &[TypeHandle],
    false_results: &[TypeHandle],
) -> Result<Vec<TypeHandle>> {
    if !is_scalar(ctx, predicate, ElementType::I1) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    join(ctx, &[true_results, false_results])
}
/// Joins the result types returned by a case operation's branches.
///
/// The index must be a scalar `i32` tensor. Branch result counts and
/// corresponding types must be compatible; at least one branch is required.
pub fn infer_case_types(
    ctx: &Context,
    index: TypeHandle,
    branches: &[Vec<TypeHandle>],
) -> Result<Vec<TypeHandle>> {
    if !is_scalar(ctx, index, ElementType::I32) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let refs: Vec<_> = branches.iter().map(Vec::as_slice).collect();
    join(ctx, &refs)
}
fn is_scalar(ctx: &Context, ty: TypeHandle, e: ElementType) -> bool {
    crate::types::tensor_parts(ctx, ty)
        .is_some_and(|(shape, element)| shape.is_empty() && element == e)
}
fn join(ctx: &Context, branches: &[&[TypeHandle]]) -> Result<Vec<TypeHandle>> {
    let first = *branches
        .first()
        .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?;
    if branches.iter().any(|branch| branch.len() != first.len()) {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    let mut outputs = Vec::with_capacity(first.len());
    for (index, &first_ty) in first.iter().enumerate() {
        let Some((mut shape, element)) = ranked_parts(ctx, first_ty) else {
            if branches
                .iter()
                .skip(1)
                .all(|branch| branch[index] == first_ty)
            {
                outputs.push(first_ty);
                continue;
            }
            return Err(Diagnostic::IncompatibleShape.into());
        };
        for branch in branches.iter().skip(1) {
            let Some((other, other_element)) = ranked_parts(ctx, branch[index]) else {
                return Err(Diagnostic::IncompatibleShape.into());
            };
            if other_element != element || other.len() != shape.len() {
                return Err(Diagnostic::IncompatibleShape.into());
            }
            for (current, &candidate) in shape.iter_mut().zip(&other) {
                if *current >= 0 && candidate >= 0 && *current != candidate {
                    return Err(Diagnostic::IncompatibleShape.into());
                }
                if *current < 0 || candidate < 0 {
                    *current = -1
                }
            }
        }
        outputs.push(TensorType::get(ctx, shape, element).into());
    }
    Ok(outputs)
}

/// Returns the ranked tensor operand type for a collective permute.
///
/// Non-ranked-tensor operands are rejected; otherwise the type is unchanged.
pub fn infer_collective_permute_type(ctx: &Context, operand: TypeHandle) -> Result<TypeHandle> {
    if ranked_parts(ctx, operand).is_none() {
        Err(Diagnostic::UnsupportedType.into())
    } else {
        Ok(operand)
    }
}
/// Infers result types for a collective broadcast.
///
/// With `dynamic_root == false`, every operand is data and must be a ranked
/// tensor. With `dynamic_root == true`, the final operand is a rank-one `i32`
/// root-index tensor whose length equals the number of data operands; it is
/// omitted from the returned types. At least one data operand is required.
pub fn infer_collective_broadcast_types(
    ctx: &Context,
    operands: &[TypeHandle],
    dynamic_root: bool,
) -> Result<Vec<TypeHandle>> {
    let data = if dynamic_root {
        let (root, data) = operands
            .split_last()
            .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        let Some((shape, element)) = ranked_parts(ctx, *root) else {
            return Err(Diagnostic::InvalidDomain.into());
        };
        if shape != [data.len() as i64]
            || crate::types::primitive_element(ctx, element) != Some(ElementType::I32)
        {
            return Err(Diagnostic::InvalidDomain.into());
        };
        data
    } else {
        operands
    };
    if data.is_empty() || data.iter().any(|&ty| ranked_parts(ctx, ty).is_none()) {
        return Err(Diagnostic::UnsupportedType.into());
    }
    Ok(data.to_vec())
}

/// Infers all-reduce results using a scalar accumulator type.
///
/// Each operand must be a ranked tensor. The accumulator element must support
/// the same reduction domain: integer signedness must match and its width may
/// not be narrower, floating-point width may not be narrower, and complex
/// widening is limited to `C64` to `C64`/`C128` or `C128` to `C128`.
pub fn infer_all_reduce_types(
    ctx: &Context,
    operands: &[TypeHandle],
    accumulator: TypeHandle,
) -> Result<Vec<TypeHandle>> {
    let (acc_shape, acc_element) =
        ranked_parts(ctx, accumulator).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if !acc_shape.is_empty() {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let acc = crate::types::primitive_element(ctx, acc_element)
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    operands
        .iter()
        .map(|&ty| {
            let (shape, input_element) =
                ranked_parts(ctx, ty).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
            let input = crate::types::primitive_element(ctx, input_element)
                .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
            let integer = match (input.integer_parts(), acc.integer_parts()) {
                (Some((iw, is)), Some((aw, asgn))) => is == asgn && iw <= aw,
                _ => false,
            };
            let float = input.is_float()
                && acc.is_float()
                && input
                    .bit_width()
                    .zip(acc.bit_width())
                    .is_some_and(|(i, a)| i <= a);
            let complex = matches!(
                (input, acc),
                (ElementType::C64, ElementType::C64 | ElementType::C128)
                    | (ElementType::C128, ElementType::C128)
            );
            if !(integer || float || complex) {
                return Err(Diagnostic::InvalidDomain.into());
            }
            Ok(TensorType::get(ctx, shape, acc_element).into())
        })
        .collect()
}
/// Infers a single reduce-scatter result by applying accumulator conversion
/// followed by a scatter along `dimension`.
///
/// `group_size` must be positive and the known extent on that axis must divide
/// evenly by it. Invalid axes, incompatible accumulator types, or overflow
/// return an error.
pub fn infer_reduce_scatter_type(
    ctx: &Context,
    operand: TypeHandle,
    accumulator: TypeHandle,
    dimension: i64,
    group_size: i64,
) -> Result<TypeHandle> {
    let reduced = infer_all_reduce_types(ctx, &[operand], accumulator)?.remove(0);
    infer_reduce_scatter_types(ctx, &[reduced], dimension, group_size)?
        .into_iter()
        .next()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())
}
/// Infers all-gather result tensors by multiplying `dimension` by `group_size`.
///
/// The group size must be positive and the axis must exist. Dynamic extents
/// remain dynamic; a known-extent multiplication that overflows is rejected.
pub fn infer_all_gather_types(
    ctx: &Context,
    operands: &[TypeHandle],
    dimension: i64,
    group_size: i64,
) -> Result<Vec<TypeHandle>> {
    if group_size <= 0 {
        return Err(Diagnostic::InvalidDomain.into());
    }
    operands
        .iter()
        .map(|&ty| {
            let (shape, element) =
                ranked_parts(ctx, ty).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
            let axis = usize::try_from(dimension)
                .ok()
                .filter(|&d| d < shape.len())
                .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
            let mut out = shape;
            if out[axis] >= 0 {
                out[axis] = out[axis]
                    .checked_mul(group_size)
                    .ok_or_else(|| Diagnostic::Overflow.into_error())?;
            }
            Ok(TensorType::get(ctx, out, element).into())
        })
        .collect()
}
/// Infers reduce-scatter result tensors by dividing `dimension` by
/// `group_size`.
///
/// The group size must be positive, the axis must exist, and a known extent
/// must be evenly divisible. Dynamic extents remain dynamic.
pub fn infer_reduce_scatter_types(
    ctx: &Context,
    operands: &[TypeHandle],
    dimension: i64,
    group_size: i64,
) -> Result<Vec<TypeHandle>> {
    if group_size <= 0 {
        return Err(Diagnostic::InvalidDomain.into());
    }
    operands
        .iter()
        .map(|&ty| {
            let (shape, element) =
                ranked_parts(ctx, ty).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
            let axis = usize::try_from(dimension)
                .ok()
                .filter(|&d| d < shape.len())
                .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
            if shape[axis] >= 0 && shape[axis] % group_size != 0 {
                return Err(Diagnostic::InvalidDimensions.into());
            };
            let mut out = shape;
            out[axis] = if out[axis] < 0 {
                -1
            } else {
                out[axis] / group_size
            };
            Ok(TensorType::get(ctx, out, element).into())
        })
        .collect()
}
/// Infers all-to-all results by dividing each tensor's `split` extent and
/// multiplying its `concat` extent by `count`.
///
/// `count` must be positive and both axes must be in range. A known split
/// extent must be evenly divisible by `count`; dynamic dimensions remain
/// dynamic and concat overflow is reported.
pub fn infer_all_to_all_types(
    ctx: &Context,
    operands: &[TypeHandle],
    split: i64,
    concat: i64,
    count: i64,
) -> Result<Vec<TypeHandle>> {
    if count <= 0 {
        return Err(Diagnostic::InvalidDomain.into());
    }
    operands
        .iter()
        .map(|&ty| {
            let (mut shape, element) =
                ranked_parts(ctx, ty).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
            let s = usize::try_from(split)
                .ok()
                .filter(|&i| i < shape.len())
                .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
            let c = usize::try_from(concat)
                .ok()
                .filter(|&i| i < shape.len())
                .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
            if shape[s] >= 0 && shape[s] % count != 0 {
                return Err(Diagnostic::InvalidDimensions.into());
            };
            shape[s] = if shape[s] < 0 { -1 } else { shape[s] / count };
            if shape[c] >= 0 {
                shape[c] = shape[c]
                    .checked_mul(count)
                    .ok_or_else(|| Diagnostic::Overflow.into_error())?;
            }
            Ok(TensorType::get(ctx, shape, element).into())
        })
        .collect()
}
