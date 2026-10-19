//! Local verification for shape-related operations.

use pliron::{context::Context, r#type::TypeHandle};

pub(crate) fn ranked_tensor(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    let tensor = ty.downcast_ref::<crate::types::TensorType>()?;
    (tensor.dimensions.iter().all(|&dimension| dimension >= -1))
        .then(|| (tensor.dimensions.clone(), tensor.element))
}

pub(crate) fn dimension_bound(ctx: &Context, ty: TypeHandle, axis: usize) -> Option<u64> {
    let ty = ty.deref(ctx);
    let tensor = ty.downcast_ref::<crate::types::TensorType>()?;
    let dimension = *tensor.dimensions.get(axis)?;
    if dimension >= 0 {
        return Some(dimension as u64);
    }
    let bound = *tensor.bounds()?.get(axis)?;
    u64::try_from(bound).ok()
}

pub(crate) fn compatible_shape(ctx: &Context, expected: TypeHandle, actual: TypeHandle) -> bool {
    let (Some((expected_shape, expected_element)), Some((actual_shape, actual_element))) =
        (ranked_tensor(ctx, expected), ranked_tensor(ctx, actual))
    else {
        return false;
    };
    if expected_element != actual_element || expected_shape.len() != actual_shape.len() {
        return false;
    }
    let expected_type = expected.deref(ctx);
    let actual_type = actual.deref(ctx);
    let Some(expected_tensor) = expected_type.downcast_ref::<crate::types::TensorType>() else {
        return false;
    };
    let Some(actual_tensor) = actual_type.downcast_ref::<crate::types::TensorType>() else {
        return false;
    };
    expected_shape
        .iter()
        .zip(actual_shape)
        .enumerate()
        .all(|(axis, (&expected, actual))| {
            if expected >= 0 && actual >= 0 {
                expected == actual
            } else if expected >= 0 {
                actual_tensor
                    .bounds()
                    .and_then(|bounds| bounds.get(axis))
                    .copied()
                    .and_then(|bound| u64::try_from(bound).ok())
                    .is_none_or(|bound| expected as u64 <= bound)
            } else if actual >= 0 {
                expected_tensor
                    .bounds()
                    .and_then(|bounds| bounds.get(axis))
                    .copied()
                    .and_then(|bound| u64::try_from(bound).ok())
                    .is_none_or(|bound| actual as u64 <= bound)
            } else {
                true
            }
        })
}
