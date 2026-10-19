//! Shape inference for the compute and reduction families.
use crate::Diagnostic;
use crate::{
    attributes::{ConvDimensionNumbersAttr, DotDimensionNumbersAttr, FftTypeAttr},
    types::{ElementType, TensorType},
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

fn ranked(ctx: &Context, ty: TypeHandle) -> Option<TensorType> {
    ty.deref(ctx).downcast_ref::<TensorType>().cloned()
}
fn merge(a: i64, b: i64) -> Option<i64> {
    if a >= 0 && b >= 0 && a != b {
        None
    } else if a < 0 {
        Some(b)
    } else {
        Some(a)
    }
}
fn result(ctx: &Context, shape: Vec<i64>, element: TypeHandle) -> TypeHandle {
    TensorType::get(ctx, shape, element).into()
}
fn primitive(ctx: &Context, ty: TypeHandle) -> Option<ElementType> {
    crate::types::primitive_element(ctx, ty)
}

/// Infer the result shape for `dot_general`; result element metadata is supplied by the caller.
pub fn infer_dot_general_type(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    dims: &DotDimensionNumbersAttr,
    result_element: TypeHandle,
) -> Result<TypeHandle> {
    let l = ranked(ctx, lhs).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let r = ranked(ctx, rhs).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if dims.lhs_batching_dimensions.len() != dims.rhs_batching_dimensions.len()
        || dims.lhs_contracting_dimensions.len() != dims.rhs_contracting_dimensions.len()
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut la = vec![false; l.rank()];
    let mut ra = vec![false; r.rank()];
    let check = |axes: &[i64], used: &mut Vec<bool>| -> Result<()> {
        for &d in axes {
            let i = usize::try_from(d).map_err(|_| Diagnostic::InvalidDimensions)?;
            let Some(slot) = used.get_mut(i) else {
                return Err(Diagnostic::InvalidDimensionIndex {
                    operation: "dot_general".to_owned(),
                    location: "unknown".to_owned(),
                    dimension_index: i,
                    value: i128::from(d),
                    rank: used.len(),
                }
                .into());
            };
            if *slot {
                return Err(Diagnostic::InvalidDimensions.into());
            }
            *slot = true;
        }
        Ok(())
    };
    check(&dims.lhs_batching_dimensions, &mut la)?;
    check(&dims.lhs_contracting_dimensions, &mut la)?;
    check(&dims.rhs_batching_dimensions, &mut ra)?;
    check(&dims.rhs_contracting_dimensions, &mut ra)?;
    let mut shape = Vec::new();
    for (&a, &b) in dims
        .lhs_batching_dimensions
        .iter()
        .zip(&dims.rhs_batching_dimensions)
    {
        let a = l.dimensions[a as usize];
        let b = r.dimensions[b as usize];
        shape.push(merge(a, b).ok_or_else(|| Diagnostic::IncompatibleShape.into_error())?)
    }
    for (&a, &b) in dims
        .lhs_contracting_dimensions
        .iter()
        .zip(&dims.rhs_contracting_dimensions)
    {
        if merge(l.dimensions[a as usize], r.dimensions[b as usize]).is_none() {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    shape.extend(
        l.dimensions
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (!la[i]).then_some(d)),
    );
    shape.extend(
        r.dimensions
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (!ra[i]).then_some(d)),
    );
    Ok(result(ctx, shape, result_element))
}
pub fn infer_dot_type(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    result_element: TypeHandle,
) -> Result<TypeHandle> {
    let l = ranked(ctx, lhs).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let r = ranked(ctx, rhs).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if !(1..=2).contains(&l.rank()) || !(1..=2).contains(&r.rank()) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let d = DotDimensionNumbersAttr {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![(l.rank() - 1) as i64],
        rhs_contracting_dimensions: vec![0],
    };
    infer_dot_general_type(ctx, lhs, rhs, &d, result_element)
}

#[derive(Clone, Copy, Debug)]
pub struct ConvolutionOptions<'a> {
    pub dimensions: &'a ConvDimensionNumbersAttr,
    pub strides: Option<&'a [i64]>,
    pub padding: Option<&'a [[i64; 2]]>,
    pub lhs_dilation: Option<&'a [i64]>,
    pub rhs_dilation: Option<&'a [i64]>,
    pub feature_group_count: i64,
    pub batch_group_count: i64,
}
pub fn infer_convolution_type(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    result_element: TypeHandle,
    o: ConvolutionOptions<'_>,
) -> Result<TypeHandle> {
    let a = ranked(ctx, lhs).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let b = ranked(ctx, rhs).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let d = o.dimensions;
    let rank = a.rank();
    if rank < 2
        || b.rank() != rank
        || o.feature_group_count <= 0
        || o.batch_group_count <= 0
        || (o.feature_group_count > 1 && o.batch_group_count > 1)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let axes = [
        (
            d.input_batch_dimension,
            d.input_feature_dimension,
            &d.input_spatial_dimensions,
        ),
        (
            d.kernel_input_feature_dimension,
            d.kernel_output_feature_dimension,
            &d.kernel_spatial_dimensions,
        ),
        (
            d.output_batch_dimension,
            d.output_feature_dimension,
            &d.output_spatial_dimensions,
        ),
    ];
    let mut normalized = Vec::new();
    for (x, y, sp) in axes {
        if sp.len() + 2 != rank {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        let v = [x, y]
            .into_iter()
            .chain(sp.iter().copied())
            .collect::<Vec<_>>();
        let mut seen = vec![false; rank];
        for d in &v {
            let i = usize::try_from(*d).map_err(|_| Diagnostic::InvalidDimensions)?;
            if i >= rank || seen[i] {
                return Err(Diagnostic::InvalidDimensions.into());
            }
            seen[i] = true;
        }
        normalized.push(v.into_iter().map(|x| x as usize).collect::<Vec<_>>())
    }
    let (input, kernel, output) = (&normalized[0], &normalized[1], &normalized[2]);
    let feature_groups = o.feature_group_count;
    let batch_groups = o.batch_group_count;
    let divide = |size: i64, groups: i64| -> Result<i64> {
        if size < 0 {
            Ok(-1)
        } else if size % groups == 0 {
            Ok(size / groups)
        } else {
            Err(Diagnostic::InvalidDimensions.into())
        }
    };
    let input_features = divide(a.dimensions[input[1]], feature_groups)?;
    let kernel_input_features = b.dimensions[kernel[0]];
    if merge(input_features, kernel_input_features).is_none() {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    divide(b.dimensions[kernel[1]], feature_groups)?;
    divide(b.dimensions[kernel[1]], batch_groups)?;
    let mut shape = vec![-1; rank];
    shape[output[0]] = divide(a.dimensions[input[0]], batch_groups)?;
    shape[output[1]] = b.dimensions[kernel[1]];
    let strides = o.strides.unwrap_or(&[]);
    let ld = o.lhs_dilation.unwrap_or(&[]);
    let rd = o.rhs_dilation.unwrap_or(&[]);
    let padding = o.padding.unwrap_or(&[]);
    if !strides.is_empty() && strides.len() != rank - 2
        || !ld.is_empty() && ld.len() != rank - 2
        || !rd.is_empty() && rd.len() != rank - 2
        || !padding.is_empty() && padding.len() != rank - 2
        || strides.iter().any(|&value| value <= 0)
        || ld.iter().any(|&value| value <= 0)
        || rd.iter().any(|&value| value <= 0)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    for i in 0..rank - 2 {
        let x = a.dimensions[input[i + 2]];
        let k = b.dimensions[kernel[i + 2]];
        if x < 0 || k < 0 {
            shape[output[i + 2]] = -1;
            continue;
        }
        let stride = *strides.get(i).unwrap_or(&1);
        let lhs_d = *ld.get(i).unwrap_or(&1);
        let rhs_d = *rd.get(i).unwrap_or(&1);
        if stride <= 0 || lhs_d <= 0 || rhs_d <= 0 {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        let p = padding.get(i).copied().unwrap_or([0, 0]);
        let lhs = x
            .saturating_sub(1)
            .checked_mul(lhs_d)
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(p[0]))
            .and_then(|value| value.checked_add(p[1]))
            .ok_or_else(|| Diagnostic::Overflow.into_error())?;
        let kernel = k
            .saturating_sub(1)
            .checked_mul(rhs_d)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Diagnostic::Overflow.into_error())?;
        let output_size = if lhs < kernel {
            0
        } else {
            (lhs - kernel)
                .checked_div(stride)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Diagnostic::Overflow.into_error())?
        };
        let Some(output_axis) = output.get(i + 2).copied() else {
            return Err(Diagnostic::InvalidDimensions.into());
        };
        let Some(output_dimension) = shape.get_mut(output_axis) else {
            return Err(Diagnostic::InvalidDimensions.into());
        };
        *output_dimension = output_size;
    }
    Ok(result(ctx, shape, result_element))
}

pub fn infer_fft_type(
    ctx: &Context,
    input: TypeHandle,
    fft_type: FftTypeAttr,
    length: &[i64],
) -> Result<TypeHandle> {
    let t = ranked(ctx, input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if !(1..=3).contains(&length.len()) || length.len() > t.rank() || length.iter().any(|&x| x <= 0)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let e = primitive(ctx, t.element);
    let element = match (fft_type.value, e) {
        (0, Some(ElementType::C64)) | (1, Some(ElementType::C64)) => t.element,
        (0, Some(ElementType::C128)) | (1, Some(ElementType::C128)) => t.element,
        (2, Some(ElementType::F32)) => ElementType::C64.native(ctx),
        (2, Some(ElementType::F64)) => ElementType::C128.native(ctx),
        (3, Some(ElementType::C64)) => ElementType::F32.native(ctx),
        (3, Some(ElementType::C128)) => ElementType::F64.native(ctx),
        _ => return Err(Diagnostic::IncompatibleElement.into()),
    };
    let mut shape = t.dimensions.clone();
    if matches!(fft_type.value, 2 | 3) {
        let start = shape.len() - length.len();
        for (i, &len) in length.iter().enumerate() {
            let axis = start + i;
            let expected = if fft_type.value == 3 && i + 1 == length.len() {
                len / 2 + 1
            } else {
                len
            };
            if shape[axis] >= 0 && shape[axis] != expected {
                return Err(Diagnostic::IncompatibleShape.into());
            }
            if i + 1 == length.len() {
                shape[axis] = if fft_type.value == 2 {
                    len / 2 + 1
                } else {
                    len
                };
            }
        }
    }
    Ok(result(ctx, shape, element))
}

pub fn infer_cholesky_type(ctx: &Context, input: TypeHandle) -> Result<TypeHandle> {
    let t = ranked(ctx, input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let n = t.rank();
    if n < 2 || merge(t.dimensions[n - 2], t.dimensions[n - 1]).is_none() {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    Ok(input)
}
pub fn infer_triangular_solve_type(
    ctx: &Context,
    a: TypeHandle,
    b: TypeHandle,
    left_side: bool,
) -> Result<TypeHandle> {
    let x = ranked(ctx, a).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let y = ranked(ctx, b).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if x.rank() < 2 || x.rank() != y.rank() || x.element != y.element {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    let n = x.rank();
    let xdim = merge(x.dimensions[n - 2], x.dimensions[n - 1])
        .ok_or_else(|| Diagnostic::IncompatibleShape.into_error())?;
    let axis = n - if left_side { 2 } else { 1 };
    if merge(xdim, y.dimensions[axis]).is_none() {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    for i in 0..n - 2 {
        if merge(x.dimensions[i], y.dimensions[i]).is_none() {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    Ok(b)
}

pub fn infer_batch_norm_inference_type(
    ctx: &Context,
    operand: TypeHandle,
    scale: TypeHandle,
    offset: TypeHandle,
    mean: TypeHandle,
    variance: TypeHandle,
    feature: usize,
) -> Result<TypeHandle> {
    validate_batch_norm(ctx, operand, &[scale, offset, mean, variance], feature)?;
    Ok(operand)
}
pub fn infer_batch_norm_training_types(
    ctx: &Context,
    operand: TypeHandle,
    scale: TypeHandle,
    offset: TypeHandle,
    feature: usize,
) -> Result<Vec<TypeHandle>> {
    validate_batch_norm(ctx, operand, &[scale, offset], feature)?;
    let t = ranked(ctx, operand).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let Some(&feature_size) = t.dimensions.get(feature) else {
        return Err(Diagnostic::InvalidDimensionIndex {
            operation: "batch_norm".to_owned(),
            location: "unknown".to_owned(),
            dimension_index: feature,
            value: feature as i128,
            rank: t.rank(),
        }
        .into());
    };
    let f = TensorType::get(ctx, vec![feature_size], t.element).into();
    Ok(vec![operand, f, f])
}
pub fn infer_batch_norm_grad_types(
    ctx: &Context,
    operand: TypeHandle,
    scale: TypeHandle,
    mean: TypeHandle,
    variance: TypeHandle,
    grad: TypeHandle,
    feature: usize,
) -> Result<Vec<TypeHandle>> {
    validate_batch_norm(ctx, operand, &[scale, mean, variance, grad], feature)?;
    let t = ranked(ctx, operand).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let Some(&feature_size) = t.dimensions.get(feature) else {
        return Err(Diagnostic::InvalidDimensionIndex {
            operation: "batch_norm".to_owned(),
            location: "unknown".to_owned(),
            dimension_index: feature,
            value: feature as i128,
            rank: t.rank(),
        }
        .into());
    };
    let f = TensorType::get(ctx, vec![feature_size], t.element).into();
    Ok(vec![operand, f, f])
}
fn validate_batch_norm(
    ctx: &Context,
    operand: TypeHandle,
    others: &[TypeHandle],
    feature: usize,
) -> Result<()> {
    let x = ranked(ctx, operand).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if feature >= x.rank() {
        return Err(Diagnostic::InvalidDimensionIndex {
            operation: "batch_norm".to_owned(),
            location: "unknown".to_owned(),
            dimension_index: feature,
            value: feature as i128,
            rank: x.rank(),
        }
        .into());
    }
    for (i, &other) in others.iter().enumerate() {
        let y = ranked(ctx, other).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        if x.element != y.element
            || y.rank() != 1 && !(i == others.len() - 1 && y.rank() == x.rank())
            || y.rank() == 1 && merge(x.dimensions[feature], y.dimensions[0]).is_none()
        {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    Ok(())
}

pub fn infer_reduce_types(
    ctx: &Context,
    inputs: &[TypeHandle],
    dimensions: &[i64],
    accumulators: &[TypeHandle],
) -> Result<Vec<TypeHandle>> {
    let first = *inputs.first().ok_or_else(|| {
        Diagnostic::EmptyOperands {
            operation: "reduce",
            role: "input",
        }
        .into_error()
    })?;
    let t = ranked(ctx, first).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if inputs.len() != accumulators.len() {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    let mut seen = vec![false; t.rank()];
    for &d in dimensions {
        let i = usize::try_from(d).map_err(|_| Diagnostic::InvalidDimensions)?;
        if i >= seen.len() || seen[i] {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        seen[i] = true;
    }
    let mut shape = t
        .dimensions
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| (!seen[i]).then_some(d))
        .collect::<Vec<_>>();
    for &input in inputs {
        let x = ranked(ctx, input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        if x.rank() != t.rank() {
            return Err(Diagnostic::IncompatibleShape.into());
        }
        for (a, &b) in x.dimensions.iter().enumerate() {
            if merge(t.dimensions[a], b).is_none() {
                return Err(Diagnostic::IncompatibleShape.into());
            }
        }
    }
    let mut output = Vec::new();
    for &acc in accumulators {
        let a = ranked(ctx, acc).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        if a.rank() != 0 {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        output.push(result(ctx, shape.clone(), a.element));
    }
    shape.clear();
    Ok(output)
}

pub fn infer_sort_types(
    ctx: &Context,
    inputs: &[TypeHandle],
    dimension: i64,
) -> Result<Vec<TypeHandle>> {
    let t = ranked(
        ctx,
        *inputs
            .first()
            .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?,
    )
    .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let d = dimension as i128;
    if d < -(t.rank() as i128) || d >= t.rank() as i128 {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    for &x in &inputs[1..] {
        let a = ranked(ctx, x).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        if a.rank() != t.rank()
            || a.dimensions
                .iter()
                .zip(&t.dimensions)
                .any(|(&x, &y)| merge(x, y).is_none())
        {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    Ok(inputs.to_vec())
}
pub fn infer_map_type(
    ctx: &Context,
    inputs: &[TypeHandle],
    computation_result: TypeHandle,
    axes: &[i64],
) -> Result<TypeHandle> {
    let last_input = *inputs.last().ok_or_else(|| {
        Diagnostic::EmptyOperands {
            operation: "map",
            role: "input",
        }
        .into_error()
    })?;
    let t = ranked(ctx, last_input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if axes.len() != t.rank() || axes.iter().enumerate().any(|(i, &x)| x != i as i64) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    for &x in inputs {
        let a = ranked(ctx, x).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        if a.rank() != t.rank()
            || a.dimensions
                .iter()
                .zip(&t.dimensions)
                .any(|(&x, &y)| merge(x, y).is_none())
        {
            return Err(Diagnostic::IncompatibleShape.into());
        }
    }
    let scalar =
        ranked(ctx, computation_result).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if scalar.rank() != 0 {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(result(ctx, t.dimensions.clone(), scalar.element))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngDistributionAttr {
    Uniform,
    Normal,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngAlgorithmAttr {
    Default,
    ThreeFry,
    Philox,
}
pub fn infer_rng_bit_generator_types(
    ctx: &Context,
    state: TypeHandle,
    output: TypeHandle,
    algorithm: RngAlgorithmAttr,
) -> Result<Vec<TypeHandle>> {
    let s = ranked(ctx, state).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let o = ranked(ctx, output).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if primitive(ctx, s.element) != Some(ElementType::Ui64)
        || s.rank() != 1
        || primitive(ctx, o.element).is_none_or(|e| !e.is_integer())
    {
        return Err(Diagnostic::IncompatibleElement.into());
    }
    let allowed = match algorithm {
        RngAlgorithmAttr::Default => true,
        RngAlgorithmAttr::ThreeFry => s
            .dimensions
            .first()
            .is_some_and(|&size| size < 0 || size == 2),
        RngAlgorithmAttr::Philox => s
            .dimensions
            .first()
            .is_some_and(|&size| size < 0 || size == 2 || size == 3),
    };
    if !allowed {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(vec![state, output])
}
pub fn infer_rng_type(
    ctx: &Context,
    lower: TypeHandle,
    upper: TypeHandle,
    shape: &[i64],
    distribution: RngDistributionAttr,
) -> Result<TypeHandle> {
    let a = ranked(ctx, lower).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let b = ranked(ctx, upper).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if a.element != b.element
        || a.rank() != b.rank()
        || a.dimensions
            .iter()
            .zip(&b.dimensions)
            .any(|(&x, &y)| merge(x, y).is_none())
        || distribution == RngDistributionAttr::Normal
            && primitive(ctx, b.element).is_none_or(|e| !e.is_float())
    {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    if shape.iter().any(|&x| x < 0) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    Ok(result(ctx, shape.to_vec(), b.element))
}

#[allow(clippy::too_many_arguments)]
pub fn infer_reduce_window_types(
    ctx: &Context,
    inputs: &[TypeHandle],
    accumulators: &[TypeHandle],
    window_dimensions: &[i64],
    window_strides: Option<&[i64]>,
    base_dilations: Option<&[i64]>,
    window_dilations: Option<&[i64]>,
    padding: Option<&[[i64; 2]]>,
) -> Result<Vec<TypeHandle>> {
    let first_input = *inputs.first().ok_or_else(|| {
        Diagnostic::EmptyOperands {
            operation: "reduce_window",
            role: "input",
        }
        .into_error()
    })?;
    if inputs.len() != accumulators.len() {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    let first = ranked(ctx, first_input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let rank = first.rank();
    if window_dimensions.len() != rank || window_dimensions.iter().any(|&x| x <= 0) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let strides = window_strides.unwrap_or(&[]);
    let base = base_dilations.unwrap_or(&[]);
    let dilation = window_dilations.unwrap_or(&[]);
    if !strides.is_empty() && strides.len() != rank
        || !base.is_empty() && base.len() != rank
        || !dilation.is_empty() && dilation.len() != rank
        || padding.is_some_and(|p| p.len() != rank)
    {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let mut shape = Vec::new();
    for (i, (&size, &window_dimension)) in
        first.dimensions.iter().zip(window_dimensions).enumerate()
    {
        if size < 0 {
            shape.push(-1);
            continue;
        }
        let b = *base.get(i).unwrap_or(&1);
        let d = *dilation.get(i).unwrap_or(&1);
        let s = *strides.get(i).unwrap_or(&1);
        if b <= 0 || d <= 0 || s <= 0 {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        let p = padding
            .and_then(|values| values.get(i))
            .copied()
            .unwrap_or([0, 0]);
        let lhs = (size - 1)
            .checked_mul(b)
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(p[0]))
            .and_then(|value| value.checked_add(p[1]))
            .ok_or_else(|| Diagnostic::Overflow.into_error())?;
        let window = (window_dimension - 1)
            .checked_mul(d)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Diagnostic::Overflow.into_error())?;
        let output_size = if lhs < window {
            0
        } else {
            (lhs - window)
                .checked_div(s)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Diagnostic::Overflow.into_error())?
        };
        shape.push(output_size)
    }
    let mut outputs = Vec::new();
    for (&input, &acc) in inputs.iter().zip(accumulators) {
        let x = ranked(ctx, input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        let a = ranked(ctx, acc).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
        if x.rank() != rank || a.rank() != 0 {
            return Err(Diagnostic::IncompatibleShape.into());
        }
        outputs.push(result(ctx, shape.clone(), a.element));
    }
    Ok(outputs)
}
pub fn infer_select_and_scatter_type(
    ctx: &Context,
    input: TypeHandle,
    source: TypeHandle,
    accumulator: TypeHandle,
    window_dimensions: &[i64],
    strides: Option<&[i64]>,
    padding: Option<&[[i64; 2]]>,
) -> Result<TypeHandle> {
    let x = ranked(ctx, input).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let s = ranked(ctx, source).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let a = ranked(ctx, accumulator).ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if s.element != x.element
        || a.rank() != 0
        || s.rank() != x.rank()
        || window_dimensions.len() != x.rank()
    {
        return Err(Diagnostic::IncompatibleShape.into());
    }
    if window_dimensions.iter().any(|&dimension| dimension <= 0) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let ones = vec![1; x.rank()];
    let padding = padding.map_or_else(|| vec![[0, 0]; x.rank()], |values| values.to_vec());
    if padding.len() != x.rank() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    let _source_window = infer_reduce_window_types(
        ctx,
        &[input],
        &[result(ctx, vec![], x.element)],
        window_dimensions,
        strides,
        None,
        None,
        Some(&padding),
    )?;
    let stride_values = strides.filter(|values| !values.is_empty()).unwrap_or(&ones);
    if stride_values.len() != x.rank() {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    for axis in 0..x.rank() {
        if stride_values[axis] <= 0 {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        let input_size = x.dimensions[axis];
        if input_size >= 0 {
            let padded = input_size
                .saturating_sub(1)
                .checked_add(1)
                .and_then(|value| value.checked_add(padding[axis][0]))
                .and_then(|value| value.checked_add(padding[axis][1]))
                .ok_or_else(|| Diagnostic::Overflow.into_error())?;
            let window = window_dimensions[axis];
            let expected_source = if padded < window {
                0
            } else {
                (padded - window)
                    .checked_div(stride_values[axis])
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| Diagnostic::Overflow.into_error())?
            };
            if s.dimensions[axis] >= 0 && s.dimensions[axis] != expected_source {
                return Err(Diagnostic::IncompatibleShape.into());
            }
        }
    }
    Ok(result(ctx, x.dimensions, a.element))
}
