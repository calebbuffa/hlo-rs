use pliron::{attribute::AttrObj, builtin::types::IntegerType, result::Result};
// Native shape-family inference.

// Inference for native StableHLO shape and indexing operations.
use crate::Diagnostic;
use crate::types::TensorType;
use pliron::{context::Context, r#type::TypeHandle};

fn tensor(ctx: &Context, ty: TypeHandle) -> Result<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    let t = ty
        .downcast_ref::<TensorType>()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if t.dimensions.iter().any(|&d| d < -1) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok((t.dimensions.clone(), t.element))
}
fn make(ctx: &Context, shape: Vec<i64>, elem: TypeHandle) -> TypeHandle {
    TensorType::get(ctx, shape, elem).into()
}
fn dimension_bound(ctx: &Context, ty: TypeHandle, axis: usize) -> Option<i64> {
    let object = ty.deref(ctx);
    let tensor = object.downcast_ref::<TensorType>()?;
    let dimension = *tensor.dimensions.get(axis)?;
    if dimension >= 0 {
        Some(dimension)
    } else {
        tensor
            .bounds()?
            .get(axis)
            .copied()
            .filter(|&bound| bound >= 0)
    }
}
fn make_with_bounds(
    ctx: &Context,
    shape: Vec<i64>,
    elem: TypeHandle,
    bounds: Vec<Option<i64>>,
) -> TypeHandle {
    if !bounds.iter().any(Option::is_some) {
        return make(ctx, shape, elem);
    }
    let encoded = bounds
        .into_iter()
        .zip(&shape)
        .map(|(bound, &dimension)| {
            if dimension >= 0 {
                -1
            } else {
                bound.unwrap_or(-1)
            }
        })
        .collect();
    let result = TensorType::get(ctx, shape, elem);
    let ranked = result.deref(ctx);
    ranked.with_encoding(
        ctx,
        Box::new(crate::attributes::TypeExtensionsAttr { bounds: encoded }) as AttrObj,
    )
}

pub fn infer_transpose(
    ctx: &Context,
    input: TypeHandle,
    permutation: &[usize],
) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    if permutation.len() != shape.len() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut seen = vec![false; shape.len()];
    let mut out = Vec::with_capacity(shape.len());
    let mut out_bounds = Vec::with_capacity(shape.len());
    for &d in permutation {
        if d >= shape.len() || seen[d] {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        seen[d] = true;
        out.push(shape[d]);
        out_bounds.push(
            (shape[d] < 0)
                .then(|| dimension_bound(ctx, input, d))
                .flatten(),
        );
    }
    Ok(make_with_bounds(ctx, out, elem, out_bounds))
}
pub fn infer_broadcast(ctx: &Context, input: TypeHandle, sizes: &[i64]) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    if sizes.iter().any(|&d| d < 0) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(make(
        ctx,
        sizes.iter().copied().chain(shape).collect(),
        elem,
    ))
}
pub fn infer_broadcast_in_dim(
    ctx: &Context,
    input: TypeHandle,
    result_shape: &[i64],
    dimensions: &[usize],
) -> Result<TypeHandle> {
    let (src, elem) = tensor(ctx, input)?;
    if dimensions.len() != src.len() || result_shape.iter().any(|&d| d < -1) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut seen = vec![false; result_shape.len()];
    for (&axis, &size) in dimensions.iter().zip(&src) {
        if axis >= seen.len() || seen[axis] {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        seen[axis] = true;
        let dst = result_shape[axis];
        if size >= 0 && dst >= 0 && size != 1 && size != dst {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    Ok(make(ctx, result_shape.to_vec(), elem))
}
pub fn infer_reshape(ctx: &Context, input: TypeHandle, result_shape: &[i64]) -> Result<TypeHandle> {
    let (src, elem) = tensor(ctx, input)?;
    if result_shape.iter().any(|&d| d < -1)
        || src.iter().any(|&d| d < 0)
        || result_shape.iter().any(|&d| d < 0)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let count = |shape: &[i64]| {
        if shape.contains(&0) {
            Some(0)
        } else {
            shape
                .iter()
                .try_fold(1_i64, |product, &dimension| product.checked_mul(dimension))
        }
    };
    let source_count = count(&src).ok_or_else(|| Diagnostic::Overflow.into_error())?;
    let result_count = count(result_shape).ok_or_else(|| Diagnostic::Overflow.into_error())?;
    if source_count != result_count {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    Ok(make(ctx, result_shape.to_vec(), elem))
}
pub fn infer_slice(
    ctx: &Context,
    input: TypeHandle,
    starts: &[i64],
    limits: &[i64],
    strides: &[i64],
) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    if starts.len() != shape.len() || limits.len() != shape.len() || strides.len() != shape.len() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut out = Vec::with_capacity(shape.len());
    for i in 0..shape.len() {
        let (a, b, s) = (starts[i], limits[i], strides[i]);
        if a < 0 || b < a || s <= 0 || dimension_bound(ctx, input, i).is_some_and(|bound| b > bound)
        {
            return Err(Diagnostic::InvalidDomain.into());
        }
        let x = b - a;
        out.push(x / s + i64::from(x % s != 0));
    }
    Ok(make(ctx, out, elem))
}
pub fn infer_dynamic_slice(ctx: &Context, input: TypeHandle, sizes: &[i64]) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    if sizes.len() != shape.len()
        || sizes.iter().any(|&s| s < 0)
        || sizes.iter().enumerate().any(|(axis, &size)| {
            dimension_bound(ctx, input, axis).is_some_and(|bound| size > bound)
        })
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(make(ctx, sizes.to_vec(), elem))
}
pub fn infer_dynamic_update_slice(
    ctx: &Context,
    input: TypeHandle,
    update: TypeHandle,
) -> Result<TypeHandle> {
    let (a, ae) = tensor(ctx, input)?;
    let (b, be) = tensor(ctx, update)?;
    if ae != be
        || a.len() != b.len()
        || b.iter().enumerate().any(|(axis, &size)| {
            (size >= 0 && dimension_bound(ctx, input, axis).is_some_and(|bound| size > bound))
                || dimension_bound(ctx, update, axis)
                    .zip(dimension_bound(ctx, input, axis))
                    .is_some_and(|(update_bound, input_bound)| update_bound > input_bound)
        })
    {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    Ok(input)
}
pub fn infer_reverse(ctx: &Context, input: TypeHandle, axes: &[usize]) -> Result<TypeHandle> {
    let (shape, _) = tensor(ctx, input)?;
    let mut seen = vec![false; shape.len()];
    for &d in axes {
        if d >= shape.len() || seen[d] {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        seen[d] = true;
    }
    Ok(input)
}
pub fn infer_concatenate(
    ctx: &Context,
    inputs: &[TypeHandle],
    dimension: usize,
) -> Result<TypeHandle> {
    let first = *inputs
        .first()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let (first_shape, element) = tensor(ctx, first)?;
    let rank = first_shape.len();
    if dimension >= rank {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut output = first_shape.clone();
    let mut upper = (0..rank)
        .map(|axis| dimension_bound(ctx, first, axis))
        .collect::<Vec<_>>();
    output[dimension] = 0;
    upper[dimension] = Some(0);
    for &ty in inputs {
        let (shape, other_element) = tensor(ctx, ty)?;
        if other_element != element || shape.len() != rank {
            return Err(Diagnostic::IncompatibleShape.into());
        }
        for axis in 0..rank {
            let current = output[axis];
            let incoming = shape[axis];
            let current_bound = upper[axis];
            let incoming_bound = dimension_bound(ctx, ty, axis);
            if axis == dimension {
                if current >= 0 && incoming >= 0 {
                    output[axis] = current
                        .checked_add(incoming)
                        .ok_or_else(|| Diagnostic::Overflow.into_error())?;
                    upper[axis] = None;
                } else {
                    output[axis] = -1;
                    upper[axis] = match (current_bound, incoming_bound) {
                        (Some(a), Some(b)) => Some(
                            a.checked_add(b)
                                .ok_or_else(|| Diagnostic::Overflow.into_error())?,
                        ),
                        _ => None,
                    };
                }
            } else if current >= 0 && incoming >= 0 {
                if current != incoming {
                    return Err(Diagnostic::IncompatibleShape.into());
                }
                output[axis] = current;
                upper[axis] = None;
            } else if current >= 0 {
                if incoming_bound.is_some_and(|bound| current > bound) {
                    return Err(Diagnostic::IncompatibleShape.into());
                }
                output[axis] = current;
                upper[axis] = None;
            } else if incoming >= 0 {
                if current_bound.is_some_and(|bound| incoming > bound) {
                    return Err(Diagnostic::IncompatibleShape.into());
                }
                output[axis] = incoming;
                upper[axis] = None;
            } else {
                output[axis] = -1;
                upper[axis] = match (current_bound, incoming_bound) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
            }
        }
    }
    Ok(make_with_bounds(ctx, output, element, upper))
}

pub fn infer_pad(
    ctx: &Context,
    input: TypeHandle,
    padding_value: TypeHandle,
    low: &[i64],
    high: &[i64],
    interior: &[i64],
) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    let (pad_shape, pad_elem) = tensor(ctx, padding_value)?;
    if !pad_shape.is_empty()
        || elem != pad_elem
        || low.len() != shape.len()
        || high.len() != shape.len()
        || interior.len() != shape.len()
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut out = Vec::with_capacity(shape.len());
    let mut out_bounds = vec![None; shape.len()];
    for axis in 0..shape.len() {
        let padding = interior[axis];
        if padding < 0 {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        let Some(size) = dimension_bound(ctx, input, axis) else {
            out.push(-1);
            continue;
        };
        let padded = i128::from(size)
            + i128::from(low[axis])
            + i128::from(high[axis])
            + (i128::from(size) - 1).max(0) * i128::from(padding);
        if padded < 0 {
            return Err(Diagnostic::InvalidDomain.into());
        }
        if padded > i128::from(i64::MAX) {
            return Err(Diagnostic::Overflow.into());
        }
        if shape[axis] >= 0 {
            out.push(padded as i64)
        } else {
            out.push(-1);
            out_bounds[axis] = Some(padded as i64)
        }
    }
    Ok(make_with_bounds(ctx, out, elem, out_bounds))
}

pub fn infer_transpose_type(
    ctx: &Context,
    input: TypeHandle,
    permutation: &[usize],
) -> Result<TypeHandle> {
    infer_transpose(ctx, input, permutation)
}
pub fn infer_broadcast_type(ctx: &Context, input: TypeHandle, sizes: &[i64]) -> Result<TypeHandle> {
    infer_broadcast(ctx, input, sizes)
}
pub fn infer_reverse_type(ctx: &Context, input: TypeHandle, axes: &[usize]) -> Result<TypeHandle> {
    infer_reverse(ctx, input, axes)
}
pub fn infer_slice_type(
    ctx: &Context,
    input: TypeHandle,
    starts: &[i64],
    limits: &[i64],
    strides: &[i64],
) -> Result<TypeHandle> {
    infer_slice(ctx, input, starts, limits, strides)
}
pub fn infer_dynamic_slice_type(
    ctx: &Context,
    input: TypeHandle,
    sizes: &[i64],
) -> Result<TypeHandle> {
    infer_dynamic_slice(ctx, input, sizes)
}
pub fn infer_dynamic_update_slice_type(
    ctx: &Context,
    input: TypeHandle,
    update: TypeHandle,
) -> Result<TypeHandle> {
    infer_dynamic_update_slice(ctx, input, update)
}
pub fn infer_concatenate_type(
    ctx: &Context,
    inputs: &[TypeHandle],
    dimension: usize,
) -> Result<TypeHandle> {
    infer_concatenate(ctx, inputs, dimension)
}
pub fn infer_pad_type(
    ctx: &Context,
    input: TypeHandle,
    padding: TypeHandle,
    low: &[i64],
    high: &[i64],
    interior: &[i64],
) -> Result<TypeHandle> {
    infer_pad(ctx, input, padding, low, high, interior)
}
pub fn infer_dynamic_reshape_type(
    ctx: &Context,
    input: TypeHandle,
    output_shape: TypeHandle,
) -> Result<TypeHandle> {
    let (_, elem) = tensor(ctx, input)?;
    let (shape, _) = tensor(ctx, output_shape)?;
    if shape.len() != 1 || shape[0] < 0 {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(make(
        ctx,
        vec![-1; usize::try_from(shape[0]).map_err(|_| Diagnostic::InvalidDimensions)?],
        elem,
    ))
}
pub fn infer_real_dynamic_slice_type(
    ctx: &Context,
    input: TypeHandle,
    start: TypeHandle,
    limit: TypeHandle,
    stride: TypeHandle,
) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    for ty in [start, limit, stride] {
        let (s, _) = tensor(ctx, ty)?;
        if s != vec![shape.len() as i64] {
            return Err(Diagnostic::InvalidDimensions.into());
        }
    }
    Ok(make(ctx, vec![-1; shape.len()], elem))
}
pub fn infer_dynamic_pad_type(
    ctx: &Context,
    input: TypeHandle,
    padding: TypeHandle,
) -> Result<TypeHandle> {
    let (shape, elem) = tensor(ctx, input)?;
    let (p, pe) = tensor(ctx, padding)?;
    if !p.is_empty() || elem != pe {
        return Err(Diagnostic::IncompatibleElement.into());
    }
    Ok(make(ctx, vec![-1; shape.len()], elem))
}
pub fn infer_dynamic_broadcast_in_dim_type(
    ctx: &Context,
    input: TypeHandle,
    output_dimensions: TypeHandle,
    dimensions: &[usize],
) -> Result<TypeHandle> {
    let (src, elem) = tensor(ctx, input)?;
    let (output, _) = tensor(ctx, output_dimensions)?;
    if output.len() != 1
        || output[0] < 0
        || dimensions.len() != src.len()
        || dimensions.iter().any(|&x| x >= output[0] as usize)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(make(ctx, vec![-1; output[0] as usize], elem))
}
pub fn infer_get_dimension_size_type(
    ctx: &Context,
    input: TypeHandle,
    dimension: usize,
) -> Result<TypeHandle> {
    let (shape, _) = tensor(ctx, input)?;
    if dimension >= shape.len() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(TensorType::scalar(ctx, crate::types::ElementType::I32))
}
pub fn infer_set_dimension_size_type(
    ctx: &Context,
    input: TypeHandle,
    size: TypeHandle,
    dimension: usize,
) -> Result<TypeHandle> {
    let (mut shape, elem) = tensor(ctx, input)?;
    let (ss, se) = tensor(ctx, size)?;
    if dimension >= shape.len()
        || !ss.is_empty()
        || se != crate::types::ElementType::I32.native(ctx)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut bounds = vec![None; shape.len()];
    bounds[dimension] = dimension_bound(ctx, input, dimension);
    shape[dimension] = -1;
    Ok(make_with_bounds(ctx, shape, elem, bounds))
}

pub fn infer_gather_type(
    ctx: &Context,
    operand: TypeHandle,
    indices: TypeHandle,
    d: &crate::attributes::GatherDimensionNumbersAttr,
    slice_sizes: &[i64],
) -> Result<TypeHandle> {
    let (input, elem) = tensor(ctx, operand)?;
    let (index, _) = tensor(ctx, indices)?;
    if slice_sizes.len() != input.len()
        || input.len()
            != d.offset_dims.len() + d.collapsed_slice_dims.len() + d.operand_batching_dims.len()
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let vector = usize::try_from(d.index_vector_dim).map_err(|_| Diagnostic::InvalidDimensions)?;
    if vector > index.len() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let v = index.get(vector).copied().unwrap_or(1);
    if v >= 0 && v != d.start_index_map.len() as i64 {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let axes = |dims: &[i64], rank: usize, sorted: bool| -> Option<Vec<usize>> {
        let mut seen = vec![false; rank];
        let mut out = Vec::new();
        for &x in dims {
            let x = usize::try_from(x).ok()?;
            if x >= rank || std::mem::replace(&mut seen[x], true) {
                return None;
            }
            out.push(x)
        }
        if sorted && !out.windows(2).all(|w| w[0] < w[1]) {
            return None;
        }
        Some(out)
    };
    let offsets = axes(
        &d.offset_dims,
        index.len() - usize::from(vector < index.len()) + d.offset_dims.len(),
        true,
    )
    .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    let _ = offsets;
    let collapsed = axes(&d.collapsed_slice_dims, input.len(), true)
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    let batches = axes(&d.operand_batching_dims, input.len(), true)
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    let index_batches = axes(&d.start_indices_batching_dims, index.len(), false)
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    let map = axes(&d.start_index_map, input.len(), false)
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    if batches.len() != index_batches.len()
        || batches
            .iter()
            .any(|x| collapsed.binary_search(x).is_ok() || map.contains(x))
        || index_batches.contains(&vector)
        || slice_sizes.iter().enumerate().any(|(i, &s)| {
            s < 0
                || dimension_bound(ctx, operand, i).is_some_and(|bound| s > bound)
                || s > 1 && (collapsed.contains(&i) || batches.contains(&i))
        })
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    for (&a, &b) in batches.iter().zip(&index_batches) {
        if input[a] >= 0 && index[b] >= 0 && input[a] != index[b] {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    let batch_axes = (0..index.len())
        .filter(|&a| a != vector)
        .collect::<Vec<_>>();
    let rank = batch_axes.len() + d.offset_dims.len();
    let offsets = axes(&d.offset_dims, rank, true)
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    let slice_axes = (0..input.len())
        .filter(|a| !collapsed.contains(a) && !batches.contains(a))
        .collect::<Vec<_>>();
    if slice_axes.len() != d.offset_dims.len() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut out = Vec::with_capacity(rank);
    let mut out_bounds = Vec::with_capacity(rank);
    let (mut si, mut bi) = (0, 0);
    for axis in 0..rank {
        if offsets.contains(&axis) {
            out.push(slice_sizes[slice_axes[si]]);
            out_bounds.push(None);
            si += 1
        } else {
            let batch_axis = batch_axes[bi];
            out.push(index[batch_axis]);
            out_bounds.push(
                (index[batch_axis] < 0)
                    .then(|| dimension_bound(ctx, indices, batch_axis))
                    .flatten(),
            );
            bi += 1
        }
    }
    Ok(make_with_bounds(ctx, out, elem, out_bounds))
}
pub fn infer_dynamic_gather_type(
    ctx: &Context,
    operand: TypeHandle,
    indices: TypeHandle,
    slice_sizes: TypeHandle,
    d: &crate::attributes::GatherDimensionNumbersAttr,
) -> Result<TypeHandle> {
    let (input_shape, element) = tensor(ctx, operand)?;
    let (size_shape, size_element) = tensor(ctx, slice_sizes)?;
    if size_shape != vec![input_shape.len() as i64]
        || size_element
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_none()
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let _ = infer_gather_type(ctx, operand, indices, d, &vec![0; input_shape.len()])?;
    let (index_shape, _) = tensor(ctx, indices)?;
    let vector = usize::try_from(d.index_vector_dim).map_err(|_| Diagnostic::InvalidDimensions)?;
    let batch_axes = (0..index_shape.len())
        .filter(|&axis| axis != vector)
        .collect::<Vec<_>>();
    let rank = batch_axes.len() + d.offset_dims.len();
    let offsets = d
        .offset_dims
        .iter()
        .map(|&axis| usize::try_from(axis).ok())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Diagnostic::InvalidDimensions.into_error())?;
    let mut bounds = Vec::with_capacity(rank);
    let mut batch = 0;
    for axis in 0..rank {
        if offsets.contains(&axis) {
            bounds.push(None)
        } else {
            let batch_axis = batch_axes[batch];
            bounds.push(
                (index_shape[batch_axis] < 0)
                    .then(|| dimension_bound(ctx, indices, batch_axis))
                    .flatten(),
            );
            batch += 1
        }
    }
    Ok(make_with_bounds(ctx, vec![-1; rank], element, bounds))
}

pub fn infer_scatter_types(
    inputs: &[TypeHandle],
    _indices: TypeHandle,
    updates: &[TypeHandle],
    d: &crate::attributes::ScatterDimensionNumbersAttr,
) -> Result<Vec<TypeHandle>> {
    if inputs.is_empty() || inputs.len() != updates.len() || d.index_vector_dim < 0 {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(inputs.to_vec())
}
