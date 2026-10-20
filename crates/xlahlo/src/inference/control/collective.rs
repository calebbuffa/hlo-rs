//! Collective result geometry and start/done context inference.
use super::infer_get_tuple_element;
use crate::Diagnostic;
use crate::{
    inference::require,
    shape::{Shape, UNBOUNDED_SIZE},
    types::{ElementType, TensorType, tensor},
    verification::control::same_ignoring_memory,
};
use pliron::result::Result;
use pliron::{
    common_traits::Verify,
    context::Context,
    r#type::{Type, TypeHandle},
};

fn one_or_tuple(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    if let [operand] = operands {
        Ok(*operand)
    } else {
        Shape::tuple(ctx, operands.to_vec())
    }
}

fn axis(ctx: &Context, operand: TypeHandle, axis: i64) -> Result<usize> {
    tensor(ctx, Shape::array_view(ctx, operand)?)?.verify(ctx)?;
    let axis = usize::try_from(axis).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "negative collective dimension".into()
        })
    })?;
    require(
        axis < Shape::rank(ctx, Shape::array_view(ctx, operand)?)?,
        "collective dimension out of range",
    )?;
    Ok(axis)
}

/// Expand the gather dimension by `shards`, preserving each operand's layout.
pub fn infer_all_gather(
    ctx: &Context,
    operands: &[TypeHandle],
    dimension: i64,
    shards: i64,
) -> Result<TypeHandle> {
    require(
        dimension >= 0 && shards > 0,
        "all-gather requires nonnegative axis and positive shard count",
    )?;
    let outputs = operands
        .iter()
        .map(|&operand| {
            let axis = axis(ctx, operand, dimension)?;
            let mut output = tensor(ctx, operand)?.clone();
            let bound = output.dimensions[axis];
            output.dimensions[axis] = if bound == UNBOUNDED_SIZE {
                bound
            } else {
                bound.checked_mul(shards).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Inference {
                        message: "all-gather dimension overflow".into()
                    })
                })?
            };
            output.dynamic_dimensions[axis] = bound == UNBOUNDED_SIZE;
            Ok(TensorType::instantiate(output, ctx).into())
        })
        .collect::<Result<Vec<_>>>()?;
    one_or_tuple(ctx, &outputs)
}

/// Infer the paired input/output context for asynchronous all-gather.
pub fn infer_all_gather_start(
    ctx: &Context,
    operands: &[TypeHandle],
    dimension: i64,
    shards: i64,
) -> Result<TypeHandle> {
    Shape::tuple(
        ctx,
        vec![
            one_or_tuple(ctx, operands)?,
            infer_all_gather(ctx, operands, dimension, shards)?,
        ],
    )
}
/// Extract all-gather's output from its asynchronous context.
pub fn infer_all_gather_done(ctx: &Context, context: TypeHandle) -> Result<TypeHandle> {
    infer_get_tuple_element(ctx, context, 1)
}
/// Retain one operand's type or form a tuple for a variadic reduction.
pub fn infer_all_reduce(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    for &operand in operands {
        Shape::rank(ctx, operand)?;
    }
    one_or_tuple(ctx, operands)
}
/// Infer an asynchronous all-reduce context with unchanged result geometry.
pub fn infer_all_reduce_start(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    infer_all_reduce(ctx, operands)
}
/// Retain the all-reduce context type as its completed result.
pub fn infer_all_reduce_done(_: &Context, context: TypeHandle) -> Result<TypeHandle> {
    Ok(context)
}

/// Divide the scatter dimension by `shards`, checking divisibility.
pub fn infer_reduce_scatter(
    ctx: &Context,
    operands: &[TypeHandle],
    dimension: i64,
    shards: i64,
) -> Result<TypeHandle> {
    require(
        dimension >= 0 && shards > 0,
        "reduce-scatter requires nonnegative axis and positive shard count",
    )?;
    let outputs = operands
        .iter()
        .map(|&operand| {
            let axis = axis(ctx, operand, dimension)?;
            let mut output = tensor(ctx, operand)?.clone();
            require(
                output.dimensions[axis] % shards == 0,
                "reduce-scatter dimension must be divisible by shard count",
            )?;
            output.dimensions[axis] = if output.dynamic_dimensions[axis] {
                UNBOUNDED_SIZE
            } else {
                output.dimensions[axis] / shards
            };
            Ok(TensorType::instantiate(output, ctx).into())
        })
        .collect::<Result<Vec<_>>>()?;
    one_or_tuple(ctx, &outputs)
}

/// Exchange split and concatenation dimensions with a default result layout.
pub fn infer_all_to_all(
    ctx: &Context,
    operand: TypeHandle,
    split_axis: i64,
    concat_axis: i64,
    count: i64,
) -> Result<TypeHandle> {
    require(count > 0, "all-to-all requires positive split count")?;
    let operand = Shape::array_view(ctx, operand)?;
    let mut output = tensor(ctx, operand)?.clone();
    require(
        !output
            .dimensions
            .iter()
            .zip(&output.dynamic_dimensions)
            .any(|(&bound, &dynamic)| dynamic && bound != UNBOUNDED_SIZE),
        "all-to-all does not support bounded dynamic shapes",
    )?;
    let split = axis(ctx, operand, split_axis)?;
    let concat = axis(ctx, operand, concat_axis)?;
    if output.dimensions[split] != UNBOUNDED_SIZE {
        require(
            output.dimensions[split] % count == 0,
            "all-to-all split dimension must be divisible by split count",
        )?;
        output.dimensions[split] /= count;
    }
    if output.dimensions[concat] != UNBOUNDED_SIZE {
        output.dimensions[concat] =
            output.dimensions[concat]
                .checked_mul(count)
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Inference {
                        message: "all-to-all concatenation overflow".into()
                    })
                })?;
    }
    Shape::with_default_layout(ctx, TensorType::instantiate(output, ctx).into())
}

/// Form a tuple after checking operand equivalence apart from memory space.
pub fn infer_all_to_all_tuple(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    let first = operands.first().ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "tuple all-to-all requires operands".into()
        })
    })?;
    for &operand in operands {
        require(
            !Shape::has_unbounded(ctx, operand),
            "tuple all-to-all does not support unbounded dynamics",
        )?;
        require(
            same_ignoring_memory(ctx, *first, operand, false)?,
            "tuple all-to-all operands differ apart from memory space",
        )?;
    }
    Shape::tuple(ctx, operands.to_vec())
}

fn dynamic_root<'a>(
    ctx: &Context,
    operands: &'a [TypeHandle],
    enabled: bool,
    exact: bool,
) -> Result<&'a [TypeHandle]> {
    if !enabled {
        return Ok(operands);
    }
    let (root, data) = operands
        .split_last()
        .filter(|(_, data)| !data.is_empty())
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: "dynamic-root collective requires data and root operands".into()
            })
        })?;
    let count = i64::try_from(data.len()).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "too many operands".into()
        })
    })?;
    let valid = if exact {
        let expected =
            Shape::with_default_layout(ctx, Shape::array(ctx, ElementType::S32, vec![count])?)?;
        Shape::equal_including_layout(ctx, *root, expected)
    } else {
        let root = tensor(ctx, *root)?;
        root.element == ElementType::S32 && root.dimensions == [count]
    };
    require(
        valid,
        "dynamic-root collective requires S32 vector with one root per data operand",
    )?;
    Ok(data)
}
/// Infer broadcast data, optionally removing the trailing dynamic-root vector.
///
/// The root must be an S32 vector with a bound equal to the number of data
/// operands. Bounded-dynamic roots and arbitrary layouts are accepted, but an
/// unbounded root cannot establish the required number of indices.
pub fn infer_collective_broadcast(
    ctx: &Context,
    operands: &[TypeHandle],
    has_dynamic_root: bool,
) -> Result<TypeHandle> {
    let data = dynamic_root(ctx, operands, has_dynamic_root, false)?;
    if has_dynamic_root {
        return one_or_tuple(ctx, data);
    }
    let first = data.first().ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "broadcast requires an operand".into()
        })
    })?;
    Shape::rank(ctx, *first)?;
    Ok(*first)
}
/// Infer reduction data and validate an optional default-layout S32 root vector.
pub fn infer_collective_reduce(
    ctx: &Context,
    operands: &[TypeHandle],
    has_dynamic_root: bool,
) -> Result<TypeHandle> {
    infer_all_reduce(ctx, dynamic_root(ctx, operands, has_dynamic_root, true)?)
}
/// Return the preallocated output operand's array type.
pub fn infer_ragged_all_to_all(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    let output = operands.get(1).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "ragged all-to-all requires an output operand".into()
        })
    })?;
    Shape::rank(ctx, *output)?;
    Ok(*output)
}
/// Infer permutation results, using the output buffer for in-place operations.
pub fn infer_collective_permute(
    ctx: &Context,
    operands: &[TypeHandle],
    inplace: bool,
) -> Result<TypeHandle> {
    if inplace {
        require(
            operands.len() == 4,
            "in-place permute requires four operands",
        )?;
        Ok(operands[1])
    } else {
        infer_all_reduce(ctx, operands)
    }
}
/// Build a permutation context from input, output, and device context types.
pub fn infer_collective_permute_start(
    ctx: &Context,
    operands: &[TypeHandle],
    contexts: &[TypeHandle],
    inplace: bool,
) -> Result<TypeHandle> {
    let mut shapes = if inplace {
        require(
            operands.len() == 4,
            "in-place permute start requires four operands",
        )?;
        vec![operands[0], operands[1]]
    } else {
        if operands.len() == 1 {
            Shape::rank(ctx, operands[0])?;
        }
        let shape = one_or_tuple(ctx, operands)?;
        vec![shape, shape]
    };
    shapes.extend_from_slice(contexts);
    Shape::tuple(ctx, shapes)
}
/// Extract the output component of a permutation context.
pub fn infer_collective_permute_done(ctx: &Context, context: TypeHandle) -> Result<TypeHandle> {
    infer_get_tuple_element(ctx, context, 1)
}
