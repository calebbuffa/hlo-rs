//! Native shape and data-movement inference.
use crate::{
    Diagnostic,
    attributes::{
        Axes, GatherDimensionNumbers, PaddingConfig, ScatterDimensionNumbers, ShuffleMode, Slice,
    },
    shape::{Dimension, Layout, Shape, UNBOUNDED_SIZE},
    types::{self, ElementType, TensorType, tensor},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};

use super::{axes, require};

fn element_count(ctx: &Context, ty: TypeHandle) -> Result<i64> {
    let ty = tensor(ctx, ty)?;
    ty.dimensions.iter().try_fold(1i64, |count, &dimension| {
        require(
            dimension != UNBOUNDED_SIZE,
            "unbounded shape has no fixed element count",
        )?;
        count.checked_mul(dimension).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!(
                    "element count overflows multiplying {count} by dimension {dimension}"
                ),
            })
        })
    })
}

fn expanded_indices(ctx: &Context, indices: TypeHandle, vector: usize) -> Result<Vec<Dimension>> {
    let indices = tensor(ctx, indices)?;
    require(indices.element.is_integer(), "indices must be integral")?;
    require(
        vector <= indices.dimensions.len(),
        &format!(
            "index vector dimension {vector} out of range for indices rank {}",
            indices.dimensions.len()
        ),
    )?;
    let mut output = indices
        .dimensions
        .iter()
        .zip(&indices.dynamic_dimensions)
        .map(|(&size, &dynamic)| {
            if size == UNBOUNDED_SIZE {
                Dimension::Unbounded
            } else if dynamic {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            }
        })
        .collect::<Vec<_>>();
    if vector == output.len() {
        output.push(Dimension::Static(1));
    }
    Ok(output)
}

fn bound(dimension: Dimension) -> i64 {
    match dimension {
        Dimension::Static(size) | Dimension::Bounded(size) => size,
        Dimension::Unbounded => UNBOUNDED_SIZE,
    }
}

fn compatible_size(left: i64, right: i64) -> bool {
    left == right || left == UNBOUNDED_SIZE || right == UNBOUNDED_SIZE
}

fn validate_index_shapes(ctx: &Context, operand_rank: usize, indices: &[TypeHandle]) -> Result<()> {
    if let Some(&index) = indices.first() {
        let index = Shape::array_view(ctx, index)?;
        let index_ty = tensor(ctx, index)?;
        if index_ty.dimensions.len() == 1 {
            require(
                indices.len() == 1,
                "index vector must be the only index operand",
            )?;
            require(
                index_ty.element.is_integer() && index_ty.dimensions == [operand_rank as i64],
                "index vector must be integral and match the operand rank",
            )?;
            return Ok(());
        }
    }
    require(
        indices.len() == operand_rank,
        &format!(
            "index count {} must match operand rank {operand_rank}",
            indices.len()
        ),
    )?;
    if let Some(&first) = indices.first() {
        let first_array = Shape::array_view(ctx, first)?;
        let first_ty = tensor(ctx, first_array)?;
        require(
            first_ty.dimensions.is_empty() && first_ty.element.is_integer(),
            "indices must be integral scalars",
        )?;
        for &index in &indices[1..] {
            let index = Shape::array_view(ctx, index)?;
            require(
                Shape::compatible(ctx, first_array, index),
                "indices must have the same scalar type and shape",
            )?;
        }
    }
    Ok(())
}

/// Validate a broadcast mapping and return the declared output type.
pub fn infer_broadcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    dimensions: &Axes,
) -> Result<TypeHandle> {
    check_broadcast(ctx, input, output, dimensions, true, true)?;
    Ok(output)
}

pub(crate) fn verify_broadcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    dimensions: &Axes,
    require_order: bool,
) -> Result<()> {
    check_broadcast(ctx, input, output, dimensions, require_order, false)
}

fn check_broadcast(
    ctx: &Context,
    input: TypeHandle,
    output: TypeHandle,
    dimensions: &Axes,
    require_order: bool,
    reject_unbounded_output: bool,
) -> Result<()> {
    let input = tensor(ctx, input)?;
    let output = tensor(ctx, output)?;
    axes(output.dimensions.len(), &dimensions.0, require_order)?;
    require(
        dimensions.0.len() == input.dimensions.len(),
        "broadcast mapping rank mismatch",
    )?;
    if reject_unbounded_output {
        require(
            !output.dimensions.contains(&UNBOUNDED_SIZE),
            "unbounded output is not supported by broadcast inference",
        )?;
    }
    for (axis, &target) in dimensions.0.iter().enumerate() {
        let target = target as usize;
        if input.dimensions[axis] != UNBOUNDED_SIZE {
            require(
                input.dimensions[axis] == 1 || input.dimensions[axis] == output.dimensions[target],
                "broadcast mapped dimensions differ",
            )?;
            require(
                input.dynamic_dimensions[axis] == output.dynamic_dimensions[target],
                "broadcast mapped dynamism differs",
            )?;
        }
    }
    Ok(())
}

fn factor_product(dimensions: &[i64]) -> Result<i64> {
    dimensions.iter().try_fold(1i64, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!(
                    "reshape factor product overflows multiplying {product} by {dimension}"
                ),
            })
        })
    })
}

fn common_factors(left: &[i64], right: &[i64]) -> Result<Vec<(usize, usize)>> {
    if left == right {
        return Ok((0..=left.len()).map(|index| (index, index)).collect());
    }
    let mut boundaries = Vec::new();
    let (mut left_index, mut right_index) = (0, 0);
    let mut previous = None;
    while left_index < left.len()
        && right_index < right.len()
        && left[left_index] == right[right_index]
    {
        boundaries.push((left_index, right_index));
        previous = Some((left_index, right_index));
        left_index += 1;
        right_index += 1;
    }
    let left_tail = factor_product(&left[left_index..])?;
    let right_tail = factor_product(&right[right_index..])?;
    if left_tail != right_tail {
        return Ok(vec![(0, 0), (left.len(), right.len())]);
    }
    if left_tail == 0 {
        boundaries.extend([(left_index, right_index), (left.len(), right.len())]);
        return Ok(boundaries);
    }
    let (mut left_size, mut right_size) = (1i64, 1i64);
    loop {
        if left_size == right_size && previous != Some((left_index, right_index)) {
            boundaries.push((left_index, right_index));
            previous = Some((left_index, right_index));
            continue;
        }
        if left_index == left.len() && right_index == right.len() {
            break;
        }
        let advance_left = left_size < right_size
            || (left_index < left.len()
                && (right_index == right.len()
                    || left_size == right_size && left[left_index] <= right[right_index]));
        let advance_right = right_size < left_size
            || (right_index < right.len()
                && (left_index == left.len()
                    || right_size == left_size && right[right_index] <= left[left_index]));
        if advance_left {
            left_size = left_size.checked_mul(left[left_index]).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("reshape input factor product overflows at axis {left_index}"),
                })
            })?;
            left_index += 1;
        }
        if advance_right {
            right_size = right_size.checked_mul(right[right_index]).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!(
                        "reshape output factor product overflows at axis {right_index}"
                    ),
                })
            })?;
            right_index += 1;
        }
    }
    Ok(boundaries)
}

/// Infer a reshape, propagating bounded dynamic dimensions through factor groups.
pub fn infer_reshape(ctx: &Context, operand: TypeHandle, dimensions: &[i64]) -> Result<TypeHandle> {
    infer_reshape_with_dimension(ctx, operand, dimensions, None)
}

/// Infer the static-dimension reshape form, including its unbounded-input rules.
pub fn infer_static_reshape(
    ctx: &Context,
    operand: TypeHandle,
    dimensions: &[i64],
) -> Result<TypeHandle> {
    infer_reshape_with_dimension(ctx, operand, dimensions, None)
}

/// Infer a reshape while assigning a split dynamic factor to a chosen output axis.
pub fn infer_reshape_with_dimension(
    ctx: &Context,
    input: TypeHandle,
    dimensions: &[i64],
    inferred_dimension: Option<usize>,
) -> Result<TypeHandle> {
    if let Some(axis) = inferred_dimension {
        require(
            axis < dimensions.len(),
            &format!(
                "inferred reshape axis {axis} out of range for output rank {}",
                dimensions.len()
            ),
        )?;
    }
    let input = tensor(ctx, input)?;
    let output = TensorType::get(ctx, input.element, dimensions.to_vec())?;
    let mut result = tensor(ctx, output)?.clone();
    if input.dimensions.contains(&UNBOUNDED_SIZE) {
        require(
            !input
                .dimensions
                .iter()
                .zip(&input.dynamic_dimensions)
                .any(|(&size, &dynamic)| dynamic && size != UNBOUNDED_SIZE),
            "reshape cannot mix bounded and unbounded dynamic dimensions",
        )?;
        return Shape::with_default_layout(ctx, output);
    }
    let input_count = factor_product(&input.dimensions)?;
    let output_count = factor_product(&result.dimensions)?;
    require(input_count == output_count, "reshape changes element count")?;
    let factors = common_factors(&input.dimensions, &result.dimensions)?;
    let mut output_bounds = vec![(usize::MAX, 0); input.dimensions.len() + 1];
    for &(input_boundary, output_boundary) in &factors {
        let bounds = &mut output_bounds[input_boundary];
        bounds.0 = bounds.0.min(output_boundary);
        bounds.1 = bounds.1.max(output_boundary);
    }
    let mut next_nonunit = vec![result.dimensions.len(); result.dimensions.len() + 1];
    for axis in (0..result.dimensions.len()).rev() {
        next_nonunit[axis] = if result.dimensions[axis] != 1 {
            axis
        } else {
            next_nonunit[axis + 1]
        };
    }
    for (axis, dimension) in input
        .dynamic_dimensions
        .iter()
        .enumerate()
        .filter(|(_, dynamic)| **dynamic)
        .map(|(axis, _)| (axis, input.dimensions[axis]))
    {
        let mut group_index = 0;
        while group_index + 1 < factors.len() && factors[group_index + 1].0 <= axis {
            group_index += 1;
        }
        let group = factors
            .get(group_index..group_index + 2)
            .filter(|group| axis >= group[0].0 && axis < group[1].0)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("dynamic input axis {axis} has no reshape factor group"),
                })
            })?;
        let ((input_start, output_start), (input_end, output_end)) = (group[0], group[1]);
        require(
            input_end - input_start <= 1 || output_end - output_start <= 1,
            "dynamic reshape dimension is both combined and split",
        )?;
        let start = output_bounds[input_start].0;
        let end = output_bounds[input_end].1;
        let group_extent = end.checked_sub(start).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!(
                    "reshape input axis {axis} has invalid output factor bounds {start}..{end}"
                ),
            })
        })?;
        let mut target = None;
        if dimension == 1 && !result.dimensions.is_empty() {
            if axis == 0 {
                target = Some(0);
            } else if axis + 1 == input.dimensions.len() {
                target = Some(result.dimensions.len() - 1);
            } else {
                return pliron::verify_err_noloc!(Diagnostic::Inference {
                    message: format!(
                        "interior dynamic unit reshape dimension at input axis {axis}"
                    ),
                });
            }
        }
        if target.is_none() {
            target = if group_extent == 1 {
                Some(start)
            } else if group_extent > 1 {
                inferred_dimension
            } else {
                None
            };
        }
        if target.is_none() {
            let next = *next_nonunit.get(start).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!(
                        "reshape factor start {start} exceeds output rank {}",
                        result.dimensions.len()
                    ),
                })
            })?;
            target = (next < end).then_some(next);
        }
        if let Some(target) = target {
            let dynamic = result.dynamic_dimensions.get_mut(target).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("inferred reshape dimension {target} out of range"),
                })
            })?;
            *dynamic = true;
        }
    }
    result.verify(ctx)?;
    Shape::with_default_layout(ctx, TensorType::instantiate(result, ctx).into())
}

/// Infer a reshape whose runtime sizes correspond to dynamic output dimensions.
pub fn infer_dynamic_reshape(
    ctx: &Context,
    input: TypeHandle,
    sizes: &[TypeHandle],
    bounds: Vec<Dimension>,
) -> Result<TypeHandle> {
    let input = Shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input)?;
    for &size in sizes {
        if let Ok(size_array) = Shape::array_view(ctx, size) {
            let size_ty = tensor(ctx, size_array)?;
            require(
                size_ty.dimensions.is_empty() || size_ty.element == ElementType::S32,
                "dynamic reshape vector sizes must be S32",
            )?;
        } else {
            require(
                types::is_hlo_type(ctx, size),
                "dynamic reshape size must be an HLO value",
            )?;
        }
    }
    require(
        sizes.len()
            == bounds
                .iter()
                .filter(|dimension| {
                    matches!(dimension, Dimension::Bounded(_) | Dimension::Unbounded)
                })
                .count(),
        "dynamic reshape requires one size operand per dynamic result dimension",
    )?;
    let output = Shape::dynamic_array(ctx, input_ty.element, bounds)?;
    require(
        element_count(ctx, input)? == element_count(ctx, output)?,
        "dynamic reshape input and output bounds have different element counts",
    )?;
    Shape::with_default_layout(ctx, output)
}

/// Validate array operands and return the declared bitcast result.
pub fn infer_bitcast(ctx: &Context, input: TypeHandle, output: TypeHandle) -> Result<TypeHandle> {
    tensor(ctx, Shape::array_view(ctx, input)?)?;
    tensor(ctx, Shape::array_view(ctx, output)?)?;
    Ok(output)
}

/// Infer dimensions and layout after a permutation.
pub fn infer_transpose(ctx: &Context, input: TypeHandle, permutation: &Axes) -> Result<TypeHandle> {
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    axes(rank, &permutation.0, false)?;
    require(
        permutation.0.len() == rank,
        "transpose requires a full permutation",
    )?;
    let mut inverse = vec![0usize; rank];
    for (target, &source) in permutation.0.iter().enumerate() {
        inverse[source as usize] = target;
    }
    let mut output = input_ty.clone();
    output.dimensions = permutation
        .0
        .iter()
        .map(|&axis| input_ty.dimensions[axis as usize])
        .collect();
    output.dynamic_dimensions = permutation
        .0
        .iter()
        .map(|&axis| input_ty.dynamic_dimensions[axis as usize])
        .collect();
    for layout in &mut output.layout {
        for axis in &mut layout.minor_to_major {
            let source = usize::try_from(*axis).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("negative transpose layout axis {axis} for rank {rank}"),
                })
            })?;
            *axis = i64::try_from(*inverse.get(source).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("transpose layout axis {source} out of range for rank {rank}"),
                })
            })?)
            .map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: format!("transpose layout axis {source} exceeds i64"),
                })
            })?;
        }
    }
    output.verify(ctx)?;
    Ok(TensorType::instantiate(output, ctx).into())
}

/// Validate reversed axes and return the unchanged type.
pub fn infer_reverse(ctx: &Context, input: TypeHandle, dimensions: &Axes) -> Result<TypeHandle> {
    let rank = Shape::rank(ctx, input)?;
    require(
        dimensions
            .0
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == dimensions.0.len(),
        "a reverse dimension is repeated",
    )?;
    axes(rank, &dimensions.0, false)?;
    Ok(input)
}

/// Infer the shape of a strided slice.
pub fn infer_slice(ctx: &Context, input: TypeHandle, geometry: &Slice) -> Result<TypeHandle> {
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    require(
        geometry.starts.0.len() == rank
            && geometry.limits.0.len() == rank
            && geometry.strides.0.len() == rank,
        "slice geometry rank mismatch",
    )?;
    let mut dimensions = Vec::with_capacity(rank);
    let mut dynamic = Vec::with_capacity(rank);
    for axis in 0..rank {
        let (start, limit, stride) = (
            geometry.starts.0[axis],
            geometry.limits.0[axis],
            geometry.strides.0[axis],
        );
        require(start >= 0, "negative slice start")?;
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE || limit <= input_ty.dimensions[axis],
            "slice limit exceeds input bound",
        )?;
        require(
            limit >= start && stride > 0,
            "invalid slice bounds or stride",
        )?;
        let extent = limit.checked_sub(start).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!(
                    "slice extent overflows at axis {axis}: start {start}, limit {limit}"
                ),
            })
        })?;
        let size = extent / stride + i64::from(extent % stride != 0);
        dimensions.push(size);
        dynamic.push(input_ty.dynamic_dimensions[axis] && size != 1);
    }
    let mut output = input_ty.clone();
    output.dimensions = dimensions;
    output.dynamic_dimensions = dynamic;
    output.layout = vec![Layout::descending(rank)];
    output.verify(ctx)?;
    Ok(TensorType::instantiate(output, ctx).into())
}

/// Infer the bounded result of a runtime-indexed slice.
pub fn infer_dynamic_slice(
    ctx: &Context,
    input: TypeHandle,
    indices: &[TypeHandle],
    sizes: &Axes,
) -> Result<TypeHandle> {
    let input = Shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    validate_index_shapes(ctx, rank, indices)?;
    require(sizes.0.len() == rank, "dynamic slice sizes must match rank")?;
    let mut dimensions = Vec::with_capacity(rank);
    for (axis, &size) in sizes.0.iter().enumerate() {
        require(size >= 0, "dynamic slice size must be nonnegative")?;
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE || size <= input_ty.dimensions[axis],
            "dynamic slice size exceeds input bound",
        )?;
        dimensions.push(
            if input_ty.dynamic_dimensions[axis] && size > 1 && size == input_ty.dimensions[axis] {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            },
        );
    }
    let output = Shape::dynamic_array(ctx, input_ty.element, dimensions)?;
    Shape::with_default_layout(ctx, output)
}

/// Infer the result of updating an array at runtime indices.
pub fn infer_dynamic_update_slice(
    ctx: &Context,
    input: TypeHandle,
    update: TypeHandle,
    indices: &[TypeHandle],
) -> Result<TypeHandle> {
    let input_array = Shape::array_view(ctx, input)?;
    let update = Shape::array_view(ctx, update)?;
    let input_ty = tensor(ctx, input_array)?;
    let update_ty = tensor(ctx, update)?;
    let rank = input_ty.dimensions.len();
    validate_index_shapes(ctx, rank, indices)?;
    require(
        update_ty.dimensions.len() == rank,
        "dynamic update rank mismatch",
    )?;
    require(
        update_ty
            .element
            .same_ignoring_fp_precision(input_ty.element),
        "dynamic update element type mismatch",
    )?;
    for axis in 0..rank {
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE
                || update_ty.dimensions[axis] <= input_ty.dimensions[axis],
            "dynamic update exceeds input bound",
        )?;
    }
    let mut output = input_ty.clone();
    for axis in 0..rank {
        output.dynamic_dimensions[axis] = input_ty.dynamic_dimensions[axis]
            || update_ty.dynamic_dimensions[axis]
                && update_ty.dimensions[axis] == input_ty.dimensions[axis];
    }
    output.verify(ctx)?;
    let output: TypeHandle = TensorType::instantiate(output, ctx).into();
    if input.deref(ctx).is::<types::BufferType>() {
        Shape::buffer(ctx, output)
    } else {
        Ok(output)
    }
}

/// Infer the merged dimensions of arrays concatenated along one axis.
pub fn infer_concatenate(ctx: &Context, inputs: &[TypeHandle], axis: usize) -> Result<TypeHandle> {
    require(!inputs.is_empty(), "concatenate needs an operand")?;
    let first = tensor(ctx, inputs[0])?;
    let rank = first.dimensions.len();
    require(
        axis < rank,
        &format!("concatenate axis {axis} out of bounds for rank {rank}"),
    )?;
    let mut dimensions = (0..rank)
        .map(|axis| Shape::dimension(ctx, inputs[0], axis))
        .collect::<Result<Vec<_>>>()?;
    let mut element = first.element;
    for &input in &inputs[1..] {
        let input = tensor(ctx, input)?;
        require(
            input.dimensions.len() == rank,
            "concatenate operand ranks differ",
        )?;
        require(
            input.element.same_ignoring_fp_precision(first.element),
            "incompatible concatenate element types",
        )?;
        element = element.higher_precision(first.element)?;
        for (dimension, current) in dimensions.iter_mut().enumerate() {
            let next = if input.dimensions[dimension] == UNBOUNDED_SIZE {
                Dimension::Unbounded
            } else if input.dynamic_dimensions[dimension] {
                Dimension::Bounded(input.dimensions[dimension])
            } else {
                Dimension::Static(input.dimensions[dimension])
            };
            if dimension == axis {
                *current = match (*current, next) {
                    (Dimension::Unbounded, _) | (_, Dimension::Unbounded) => Dimension::Unbounded,
                    (Dimension::Static(a), Dimension::Static(b)) => {
                        Dimension::Static(a.checked_add(b).ok_or_else(|| {
                            pliron::verify_error_noloc!(Diagnostic::Inference {
                                message: format!(
                                    "concatenate dimension {axis} overflows adding {a} and {b}"
                                ),
                            })
                        })?)
                    }
                    (
                        Dimension::Static(a) | Dimension::Bounded(a),
                        Dimension::Static(b) | Dimension::Bounded(b),
                    ) => Dimension::Bounded(a.checked_add(b).ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: format!(
                                "concatenate dimension {axis} overflows adding {a} and {b}"
                            ),
                        })
                    })?),
                };
            } else {
                *current = match (*current, next) {
                    (Dimension::Static(a), Dimension::Static(b)) if a == b => Dimension::Static(a),
                    (Dimension::Static(a), Dimension::Bounded(b))
                    | (Dimension::Bounded(a), Dimension::Static(b))
                    | (Dimension::Bounded(a), Dimension::Bounded(b))
                        if a == b =>
                    {
                        Dimension::Bounded(a)
                    }
                    (Dimension::Unbounded, Dimension::Unbounded) => Dimension::Unbounded,
                    (Dimension::Unbounded, Dimension::Static(b))
                    | (Dimension::Static(b), Dimension::Unbounded) => Dimension::Static(b),
                    (Dimension::Unbounded, Dimension::Bounded(b))
                    | (Dimension::Bounded(b), Dimension::Unbounded) => Dimension::Bounded(b),
                    _ => {
                        return pliron::verify_err_noloc!(Diagnostic::Inference {
                            message: format!(
                                "concatenate non-axis dimension {dimension} differs: {current:?} versus {next:?}"
                            ),
                        });
                    }
                };
            }
        }
    }
    Shape::with_default_layout(ctx, Shape::dynamic_array(ctx, element, dimensions)?)
}

/// Infer dimensions after applying edge and interior padding.
pub fn infer_pad(
    ctx: &Context,
    input: TypeHandle,
    value: TypeHandle,
    config: &PaddingConfig,
) -> Result<TypeHandle> {
    let input = tensor(ctx, input)?;
    let value = tensor(ctx, Shape::array_view(ctx, value)?)?;
    require(value.dimensions.is_empty(), "padding value must be scalar")?;
    require(
        !value.dynamic_dimensions.iter().any(|&dynamic| dynamic),
        "dynamic padding scalar is unsupported",
    )?;
    require(
        input.element.same_ignoring_fp_precision(value.element),
        "incompatible pad element types",
    )?;
    require(
        config.0.len() == input.dimensions.len(),
        "pad requires one config per input axis",
    )?;
    require(
        config.0.iter().all(|padding| padding.interior_padding >= 0),
        "negative interior padding",
    )?;
    let mut dimensions = Vec::with_capacity(config.0.len());
    for (axis, padding) in config.0.iter().enumerate() {
        if input.dimensions[axis] == UNBOUNDED_SIZE {
            dimensions.push(Dimension::Unbounded);
            continue;
        }
        let extent = (input.dimensions[axis] - 1)
            .max(0)
            .checked_mul(padding.interior_padding)
            .and_then(|size| size.checked_add(input.dimensions[axis]))
            .and_then(|size| size.checked_add(padding.edge_padding_low))
            .and_then(|size| size.checked_add(padding.edge_padding_high))
            .ok_or_else(|| pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!("padding result overflows at axis {axis}: extent {}, low {}, high {}, interior {}", input.dimensions[axis], padding.edge_padding_low, padding.edge_padding_high, padding.interior_padding),
            }))?;
        require(extent >= 0, "padding results in negative extent")?;
        dimensions.push(if input.dynamic_dimensions[axis] {
            Dimension::Bounded(extent)
        } else {
            Dimension::Static(extent)
        });
    }
    let element = input.element.higher_precision(value.element)?;
    Shape::with_default_layout(ctx, Shape::dynamic_array(ctx, element, dimensions)?)
}

/// Validate an iota result type and its varying dimension.
pub fn infer_iota(ctx: &Context, result: TypeHandle, axis: usize) -> Result<TypeHandle> {
    let output = tensor(ctx, result)?;
    require(
        !output.dimensions.is_empty(),
        "iota requires non-scalar result",
    )?;
    require(
        axis < output.dimensions.len(),
        &format!(
            "iota dimension {axis} out of range for rank {}",
            output.dimensions.len()
        ),
    )?;
    require(
        output.element.is_integer() || output.element.is_float() || output.element.is_complex(),
        "iota requires numerical element type",
    )?;
    Ok(result)
}

/// Infer the output shape for gather index and window partitions.
pub fn infer_gather(
    ctx: &Context,
    input: TypeHandle,
    indices: TypeHandle,
    dims: &GatherDimensionNumbers,
    sizes: &Axes,
) -> Result<TypeHandle> {
    let input_ty = tensor(ctx, input)?;
    let rank = input_ty.dimensions.len();
    let vector = usize::try_from(dims.index_vector_dim).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "index_vector_dim",
            message: format!(
                "gather index vector dimension {} must be nonnegative",
                dims.index_vector_dim
            ),
        })
    })?;
    let expanded = expanded_indices(ctx, indices, vector)?;
    require(
        vector < expanded.len(),
        "gather index vector dimension out of range",
    )?;
    let output_rank = dims
        .offset_dims
        .0
        .len()
        .checked_add(expanded.len() - 1)
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: "gather result rank overflow".into(),
            })
        })?;
    let offsets = super::axes(output_rank, &dims.offset_dims.0, true)?;
    let collapsed = super::axes(rank, &dims.collapsed_slice_dims.0, true)?;
    let batching = super::axes(rank, &dims.operand_batching_dims.0, true)?;
    let index_batching = super::axes(expanded.len(), &dims.start_indices_batching_dims.0, false)?;
    let start_map = super::axes(rank, &dims.start_index_map.0, false)?;
    let vector_size = bound(expanded[vector]);
    require(
        vector_size == UNBOUNDED_SIZE || vector_size == dims.start_index_map.0.len() as i64,
        "gather index vector and start map cardinalities differ",
    )?;
    require(
        sizes.0.len() == rank,
        "gather requires one slice size per input axis",
    )?;
    require(
        rank == dims.offset_dims.0.len()
            + dims.collapsed_slice_dims.0.len()
            + dims.operand_batching_dims.0.len(),
        "gather input partition mismatch",
    )?;
    require(
        dims.operand_batching_dims.0.len() == dims.start_indices_batching_dims.0.len()
            && !index_batching[vector],
        "gather batching cardinality or vector overlap",
    )?;
    for (&operand_axis, &index_axis) in dims
        .operand_batching_dims
        .0
        .iter()
        .zip(&dims.start_indices_batching_dims.0)
    {
        require(
            compatible_size(
                input_ty.dimensions[operand_axis as usize],
                bound(expanded[index_axis as usize]),
            ),
            "gather batching sizes differ",
        )?;
    }
    for axis in 0..rank {
        require(
            !(batching[axis] && (collapsed[axis] || start_map[axis])),
            "gather dimensions overlap",
        )?;
        let size = sizes.0[axis];
        require(size >= 0, "negative gather slice size")?;
        require(
            input_ty.dimensions[axis] == UNBOUNDED_SIZE || size <= input_ty.dimensions[axis],
            "gather slice size exceeds input bound",
        )?;
        require(
            !(collapsed[axis] || batching[axis]) || size <= 1,
            "collapsed or batching gather slice exceeds one",
        )?;
    }
    let mut windows = (0..rank).filter(|&axis| !collapsed[axis] && !batching[axis]);
    let mut batches = expanded
        .iter()
        .enumerate()
        .filter(|(axis, _)| *axis != vector);
    let mut dimensions = Vec::with_capacity(output_rank);
    for is_offset in offsets {
        if is_offset {
            let source = windows.next().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: "gather output offset has no remaining input window dimension".into(),
                })
            })?;
            dimensions.push(if sizes.0[source] == input_ty.dimensions[source] {
                Shape::dimension(ctx, input, source)?
            } else {
                Dimension::Static(sizes.0[source])
            });
        } else {
            dimensions.push(
                *batches
                    .next()
                    .ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: "gather output batch has no remaining index dimension".into(),
                        })
                    })?
                    .1,
            );
        }
    }
    require(
        windows.next().is_none() && batches.next().is_none(),
        "gather output partition mismatch",
    )?;
    Shape::with_default_layout(
        ctx,
        Shape::dynamic_array(ctx, input_ty.element, dimensions)?,
    )
}

/// Infer the output shape for a single-input scatter.
pub fn infer_scatter(
    ctx: &Context,
    input: TypeHandle,
    indices: TypeHandle,
    update: TypeHandle,
    dims: &ScatterDimensionNumbers,
) -> Result<TypeHandle> {
    infer_variadic_scatter(ctx, &[input], indices, &[update], dims)
}

/// Infer the result type for variadic scatter inputs and updates.
///
/// A single input retains its type; multiple inputs produce a tuple.
pub fn infer_variadic_scatter(
    ctx: &Context,
    inputs: &[TypeHandle],
    indices: TypeHandle,
    updates: &[TypeHandle],
    dims: &ScatterDimensionNumbers,
) -> Result<TypeHandle> {
    require(
        !inputs.is_empty() && inputs.len() == updates.len(),
        "scatter requires equally many inputs and updates",
    )?;
    let vector = usize::try_from(dims.index_vector_dim).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "index_vector_dim",
            message: format!(
                "scatter index vector dimension {} must be nonnegative",
                dims.index_vector_dim
            ),
        })
    })?;
    let expanded = expanded_indices(ctx, indices, vector)?;
    require(
        vector < expanded.len(),
        "scatter index vector dimension out of range",
    )?;
    let mut results = Vec::with_capacity(inputs.len());
    for (&input, &update) in inputs.iter().zip(updates) {
        let input_ty = tensor(ctx, input)?;
        let update_ty = tensor(ctx, update)?;
        let rank = input_ty.dimensions.len();
        let update_rank = update_ty.dimensions.len();
        let update_windows = super::axes(update_rank, &dims.update_window_dims.0, true)?;
        let inserted = super::axes(rank, &dims.inserted_window_dims.0, true)?;
        let batching = super::axes(rank, &dims.input_batching_dims.0, true)?;
        let index_batching =
            super::axes(expanded.len(), &dims.scatter_indices_batching_dims.0, false)?;
        let scatter_map = super::axes(rank, &dims.scatter_dims_to_operand_dims.0, false)?;
        require(
            update_rank == expanded.len() - 1 + dims.update_window_dims.0.len(),
            "scatter update rank differs from index and window dimensions",
        )?;
        require(
            rank == dims.update_window_dims.0.len()
                + dims.inserted_window_dims.0.len()
                + dims.input_batching_dims.0.len(),
            "scatter window partition mismatch",
        )?;
        let vector_size = bound(expanded[vector]);
        require(
            vector_size == UNBOUNDED_SIZE
                || vector_size == dims.scatter_dims_to_operand_dims.0.len() as i64,
            "scatter index vector and map cardinalities differ",
        )?;
        require(
            dims.input_batching_dims.0.len() == dims.scatter_indices_batching_dims.0.len()
                && !index_batching[vector],
            "scatter batching cardinality or vector overlap",
        )?;
        require(
            dims.scatter_dims_to_operand_dims
                .0
                .iter()
                .all(|&axis| !batching[axis as usize])
                && scatter_map
                    .iter()
                    .enumerate()
                    .all(|(axis, &mapped)| !mapped || !batching[axis]),
            "scatter batching and map dimensions overlap",
        )?;
        require(
            dims.inserted_window_dims
                .0
                .iter()
                .all(|&axis| !batching[axis as usize]),
            "scatter batching and inserted dimensions overlap",
        )?;
        for (&operand_axis, &indices_axis) in dims
            .input_batching_dims
            .0
            .iter()
            .zip(&dims.scatter_indices_batching_dims.0)
        {
            require(
                compatible_size(
                    input_ty.dimensions[operand_axis as usize],
                    bound(expanded[indices_axis as usize]),
                ),
                "scatter batching sizes differ",
            )?;
        }
        let mut input_windows = (0..rank).filter(|&axis| !inserted[axis] && !batching[axis]);
        let mut index_dimensions = expanded
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != vector);
        for (axis, &is_window) in update_windows.iter().enumerate() {
            if is_window {
                let source = input_windows.next().ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Inference {
                        message: format!(
                            "scatter update axis {axis} has no input window dimension"
                        ),
                    })
                })?;
                require(
                    input_ty.dimensions[source] == UNBOUNDED_SIZE
                        || update_ty.dimensions[axis] <= input_ty.dimensions[source],
                    "scatter update window exceeds input",
                )?;
            } else {
                let index_size = *index_dimensions
                    .next()
                    .ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: format!(
                                "scatter update axis {axis} has no index batch dimension"
                            ),
                        })
                    })?
                    .1;
                require(
                    compatible_size(update_ty.dimensions[axis], bound(index_size)),
                    "scatter update batch size differs from indices",
                )?;
            }
        }
        require(
            input_windows.next().is_none() && index_dimensions.next().is_none(),
            "scatter update partition mismatch",
        )?;
        require(
            input_ty
                .element
                .same_ignoring_fp_precision(update_ty.element),
            "scatter input and update element types differ",
        )?;
        results.push(input);
    }
    if results.len() == 1 {
        Ok(results[0])
    } else {
        Shape::tuple(ctx, results)
    }
}

/// Validate and return the input type of a copy.
pub fn infer_copy(ctx: &Context, input: TypeHandle) -> Result<TypeHandle> {
    require(
        crate::types::is_hlo_type(ctx, input),
        "copy operand must be an HLO type",
    )?;
    Ok(input)
}

/// Infer the S32 scalar returned by get-dimension-size.
pub fn infer_get_dimension_size(
    ctx: &Context,
    input: TypeHandle,
    axis: usize,
) -> Result<TypeHandle> {
    let input = Shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input)?;
    require(
        axis < input_ty.dimensions.len(),
        &format!(
            "get-dimension-size axis {axis} out of range for rank {}",
            input_ty.dimensions.len()
        ),
    )?;
    require(
        input_ty.dimensions[axis] <= i64::from(i32::MAX),
        "dimension exceeds S32 limit",
    )?;
    let scalar = Shape::scalar(ctx, ElementType::S32)?;
    Shape::with_default_layout(ctx, scalar)
}

/// Infer an input type with the selected dimension marked dynamic.
pub fn infer_set_dimension_size(
    ctx: &Context,
    input: TypeHandle,
    size: TypeHandle,
    axis: usize,
) -> Result<TypeHandle> {
    let input_array = Shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input_array)?;
    require(
        axis < input_ty.dimensions.len(),
        &format!(
            "set-dimension-size axis {axis} out of range for rank {}",
            input_ty.dimensions.len()
        ),
    )?;
    require(
        input_ty.dimensions[axis] <= i64::from(i32::MAX),
        "dimension exceeds S32 limit",
    )?;
    let size = Shape::array_view(ctx, size)?;
    let size_ty = tensor(ctx, size)?;
    require(
        size_ty.dimensions.is_empty() && size_ty.element == ElementType::S32,
        "dimension size must be S32 scalar",
    )?;
    let mut output = input_ty.clone();
    output.dynamic_dimensions[axis] = true;
    output.verify(ctx)?;
    let output = TensorType::instantiate(output, ctx).into();
    if input.deref(ctx).is::<types::BufferType>() {
        Shape::buffer(ctx, output)
    } else {
        Ok(output)
    }
}

/// Validate shuffle axes and shifts, preserving the input type.
pub fn infer_shuffle(
    ctx: &Context,
    input: TypeHandle,
    dimensions: &Axes,
    mode: &ShuffleMode,
) -> Result<TypeHandle> {
    let rank = Shape::rank(ctx, input)?;
    require(
        !dimensions.0.is_empty(),
        "shuffle must include at least one dimension",
    )?;
    require(
        dimensions
            .0
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == dimensions.0.len(),
        "a shuffle dimension is repeated",
    )?;
    axes(rank, &dimensions.0, false)?;
    require(
        mode.shifts.0.len() == dimensions.0.len(),
        "shuffle dimensions and shifts must have the same size",
    )?;
    Ok(input)
}
