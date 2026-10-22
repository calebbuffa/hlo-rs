use crate::{
    Diagnostic,
    shape::{self, Dimension, UNBOUNDED_SIZE},
    types::{self, ElementType, TensorType, tensor},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};

use super::super::require;

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
    require(
        !dimensions.contains(&UNBOUNDED_SIZE),
        "reshaping with an unbounded result shape is not supported",
    )?;
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
        return shape::with_default_layout(ctx, output);
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
            }
            if axis + 1 == input.dimensions.len() {
                target = Some(result.dimensions.len() - 1);
            }
            if target.is_none() {
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
    shape::with_default_layout(ctx, TensorType::instantiate(result, ctx).into())
}
/// Infer reshape bounds without imposing instruction-level size-operand arity.
///
/// Size shapes must be scalar or S32 arrays. Instruction verification separately
/// requires one S32 size operand per result dimension, including static ones.
pub fn infer_dynamic_reshape(
    ctx: &Context,
    input: TypeHandle,
    sizes: &[TypeHandle],
    bounds: Vec<Dimension>,
) -> Result<TypeHandle> {
    let input = shape::array_view(ctx, input)?;
    let input_ty = tensor(ctx, input)?;
    for &size in sizes {
        if let Ok(size_array) = shape::array_view(ctx, size) {
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
    let output = shape::dynamic_array(ctx, input_ty.element, bounds)?;
    require(
        element_count(ctx, input)? == element_count(ctx, output)?,
        "dynamic reshape input and output bounds have different element counts",
    )?;
    shape::with_default_layout(ctx, output)
}
