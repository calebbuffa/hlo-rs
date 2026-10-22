//! Native control, communication, and stateful operation inference.
use super::{Signature, require};
use crate::Diagnostic;
use crate::{
    attributes::RandomDistribution,
    shape,
    types::{ElementType, TensorType, tensor},
};
use pliron::result::Result;
use pliron::{
    common_traits::Verify,
    context::Context,
    r#type::{Type, TypeHandle},
};
use std::cell::Ref;

/// Collective shape inference with explicit group cardinalities.
pub mod collective;

/// Validate scalar distribution parameters and retain the declared random array shape.
///
/// Uniform distributions accept real floating-point, integer, and predicate
/// parameters; normal distributions require real floating-point parameters.
/// Mixed precision permits a different real floating-point result format, never
/// different parameter formats. Generator/backend capabilities are not checked.
pub fn infer_rng(
    ctx: &Context,
    operands: &[TypeHandle],
    result: TypeHandle,
    distribution: RandomDistribution,
    allow_mixed_precision: bool,
) -> Result<TypeHandle> {
    require(operands.len() == 2, "rng requires two scalar operands")?;
    let a = tensor(ctx, shape::array_view(ctx, operands[0])?)?;
    let b = tensor(ctx, shape::array_view(ctx, operands[1])?)?;
    let out = tensor(ctx, shape::array_view(ctx, result)?)?;
    require(
        a.dimensions.is_empty() && b.dimensions.is_empty(),
        "rng requires two scalar operands",
    )?;
    require(
        a.element == b.element
            && (a.element == out.element
                || allow_mixed_precision && a.element.same_ignoring_fp_precision(out.element)),
        "rng operand and result element types differ",
    )?;
    require(
        match distribution {
            RandomDistribution::Uniform => {
                a.element.is_float() || a.element.is_integer() || a.element == ElementType::Pred
            }
            RandomDistribution::Normal => a.element.is_float(),
        },
        "invalid random distribution or element type",
    )?;
    Ok(result)
}

/// Check an optional `(state, generated values)` result and retain its shape.
///
/// Non-tuple results are accepted. Tuple results require exactly two children
/// and a first child compatible with the input state; algorithm-specific state
/// sizes and generated-value formats belong to lowering, not this shape rule.
pub fn infer_rng_bit_generator(
    ctx: &Context,
    state: TypeHandle,
    result: TypeHandle,
) -> Result<TypeHandle> {
    if shape::is_tuple(ctx, result) {
        let parts = shape::children(ctx, result);
        require(
            parts.len() == 2 && shape::compatible(ctx, parts[0], state),
            "rng-bit-generator requires two tuple components and matching state shapes",
        )?;
    }
    Ok(result)
}

/// Infer the fixed `U64[2]` state returned by the stateful RNG counter operation.
pub fn infer_rng_get_and_update_state(ctx: &Context) -> Result<TypeHandle> {
    shape::array(ctx, ElementType::U64, vec![2])
}

/// Infer a tuple containing the operand types in order.
pub fn infer_tuple(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    shape::tuple(ctx, operands.to_vec())
}

/// Select a tuple subshape, rejecting negative and out-of-range indices.
pub fn infer_get_tuple_element(
    ctx: &Context,
    operand: TypeHandle,
    index: i64,
) -> Result<TypeHandle> {
    require(
        shape::is_tuple(ctx, operand),
        "get-tuple-element requires a tuple",
    )?;
    let index = usize::try_from(index).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "negative tuple index".into()
        })
    })?;
    shape::subshape(ctx, operand, &[index])
}

/// Check argument compatibility and return the callee's result type.
pub fn infer_call(
    ctx: &Context,
    operands: &[TypeHandle],
    callee: &Signature,
) -> Result<TypeHandle> {
    require(
        operands.len() == callee.0.len(),
        "call operand count differs from parameter count",
    )?;
    require(
        operands
            .iter()
            .zip(&callee.0)
            .all(|(&a, &b)| shape::compatible(ctx, a, b)),
        "call operand and parameter shapes differ",
    )?;
    Ok(callee.1)
}

/// Check loop state and predicate types. Callee ordering is body, then condition.
pub fn infer_while(
    ctx: &Context,
    initial: TypeHandle,
    body: &Signature,
    condition: &Signature,
) -> Result<TypeHandle> {
    require(
        condition.0.len() == 1,
        "while condition must take one state parameter",
    )?;
    require(
        body.0.len() == 1,
        "while body must take one state parameter",
    )?;
    require(
        shape::compatible(ctx, condition.1, shape::scalar(ctx, ElementType::Pred)?),
        "while condition must return PRED scalar",
    )?;
    require(
        [body.0[0], condition.0[0], initial]
            .iter()
            .all(|&ty| shape::compatible(ctx, body.1, ty)),
        "while body result and state shapes differ",
    )?;
    Ok(initial)
}

/// Validate a scalar PRED or S32 selector and its branch/operand cardinalities.
pub fn validate_conditional_selector(
    ctx: &Context,
    selector: TypeHandle,
    operand_count: usize,
    branch_count: usize,
) -> Result<()> {
    let selector = tensor(ctx, shape::array_view(ctx, selector)?)?;
    require(
        selector.dimensions.is_empty()
            && matches!(selector.element, ElementType::Pred | ElementType::S32)
            && branch_count > 0
            && (selector.element != ElementType::Pred || branch_count == 2)
            && operand_count == branch_count,
        "invalid conditional selector or branch cardinality",
    )
}

fn merge_dynamic(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<TypeHandle> {
    if shape::is_tuple(ctx, a) {
        let elements = shape::children(ctx, a)
            .iter()
            .copied()
            .zip(shape::children(ctx, b).iter().copied())
            .map(|(a, b)| merge_dynamic(ctx, a, b))
            .collect::<Result<Vec<_>>>()?;
        return shape::tuple(ctx, elements);
    }
    if shape::is_array(ctx, a) {
        let array = tensor(ctx, a)?;
        array.verify(ctx)?;
        let mut merged = array.clone();
        let other = tensor(ctx, b)?;
        other.verify(ctx)?;
        for (a, b) in merged
            .dynamic_dimensions
            .iter_mut()
            .zip(&other.dynamic_dimensions)
        {
            *a |= *b;
        }
        return Ok(TensorType::instantiate(merged, ctx).into());
    }
    Ok(a)
}

/// Merge compatible branch results, retaining bounded-dynamic dimensions.
pub fn infer_conditional(
    ctx: &Context,
    selector: TypeHandle,
    operands: &[TypeHandle],
    branches: &[Signature],
) -> Result<TypeHandle> {
    validate_conditional_selector(ctx, selector, operands.len(), branches.len())?;
    let mut result = infer_call(ctx, &operands[..1], &branches[0])?;
    for (&operand, branch) in operands[1..].iter().zip(&branches[1..]) {
        let other = infer_call(ctx, &[operand], branch)?;
        require(
            shape::compatible(ctx, result, other),
            "conditional branch result shapes differ",
        )?;
        result = merge_dynamic(ctx, result, other)?;
    }
    Ok(result)
}

/// Infer the `(data, U32 context, token)` send context.
pub fn infer_send(ctx: &Context, data: TypeHandle) -> Result<TypeHandle> {
    shape::tuple(
        ctx,
        vec![
            data,
            shape::with_default_layout(ctx, shape::scalar(ctx, ElementType::U32)?)?,
            shape::token(ctx),
        ],
    )
}
/// Infer the `(data, U32 context, token)` receive context.
pub fn infer_recv(ctx: &Context, data: TypeHandle) -> Result<TypeHandle> {
    infer_send(ctx, data)
}
/// Extract received data and its completion token.
pub fn infer_recv_done(ctx: &Context, context: TypeHandle) -> Result<TypeHandle> {
    shape::tuple(
        ctx,
        vec![infer_get_tuple_element(ctx, context, 0)?, shape::token(ctx)],
    )
}
/// Infer the `(destination, source, U32 context)` asynchronous copy context.
pub fn infer_copy_start(ctx: &Context, data: TypeHandle) -> Result<TypeHandle> {
    shape::tuple(
        ctx,
        vec![
            data,
            data,
            shape::with_default_layout(ctx, shape::scalar(ctx, ElementType::U32)?)?,
        ],
    )
}
/// Extract a copy destination after checking source/destination compatibility.
pub fn infer_copy_done(ctx: &Context, context: TypeHandle) -> Result<TypeHandle> {
    let destination = infer_get_tuple_element(ctx, context, 0)?;
    let source = infer_get_tuple_element(ctx, context, 1)?;
    require(
        shape::compatible(ctx, destination, source),
        "copy context source and destination differ",
    )?;
    Ok(destination)
}

/// Borrow bound parameter types and extract the bound result of an async context.
pub fn async_context(
    ctx: &Context,
    shape: TypeHandle,
) -> Result<(Ref<'_, [TypeHandle]>, TypeHandle)> {
    require(shape::is_tuple(ctx, shape), "async context must be a tuple")?;
    let children = shape::children(ctx, shape);
    require(
        children.len() >= 2,
        "context requires ((bound operands...), bound result, states...)",
    )?;
    require(
        shape::is_tuple(ctx, children[0]),
        "context child zero must be a tuple",
    )?;
    Ok((shape::children(ctx, children[0]), children[1]))
}
