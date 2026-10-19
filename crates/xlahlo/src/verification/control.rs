//! Native control, communication, and stateful operation verification.
use super::{VerifyOptions, instruction_types, optional_attr, require, same};
use crate::Diagnostic;
use crate::{
    attributes::*,
    inference::{Signature, control as infer},
    literal::Literal,
    opcode::{Opcode, Opcode::*},
    ops::{
        ReturnOp,
        control::{FusionOp, ShapeList},
    },
    shape::{Shape, layouts_equal},
    types::{BufferType, ElementType, TokenType, tensor},
};
use pliron::result::Result;
use pliron::{
    attribute::Attribute,
    builtin::attributes::StringAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
};
use std::{cell::Ref, collections::HashSet};

/// Async ownership, producer tracing, and channel-pair validation.
pub(crate) mod asynchronous;
/// Collective group, reducer, and module consistency validation.
pub(crate) mod collective;
mod deadlocks;
#[cfg(test)]
mod tests;

pub(super) fn required<'a, T: Attribute>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    name: &'static str,
) -> Result<Ref<'a, T>> {
    optional_attr(ctx, op, name)?.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: name,
            message: "required attribute is missing".into(),
        })
    })
}

pub(crate) fn flag(ctx: &Context, op: Ptr<Operation>, name: &'static str) -> Result<bool> {
    match optional_attr::<Bools>(ctx, op, name)? {
        None => Ok(false),
        Some(value) => {
            if value.0.len() != 1 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: name,
                    message: "boolean attribute requires exactly one value".into(),
                }));
            }
            Ok(value.0[0])
        }
    }
}

pub(crate) fn channel(ctx: &Context, op: Ptr<Operation>) -> Result<Option<i64>> {
    Ok(
        match optional_attr::<OptionalInteger>(ctx, op, "hlo_channel")?.as_deref() {
            None | Some(OptionalInteger::Absent) => None,
            Some(OptionalInteger::Value(value)) => Some(*value),
        },
    )
}

pub(crate) fn same_ignoring_memory(
    ctx: &Context,
    left: TypeHandle,
    right: TypeHandle,
    splits: bool,
) -> Result<bool> {
    if !Shape::equal_ignoring_layout(ctx, left, right) {
        return Ok(false);
    }
    let mut pending = vec![(left, right)];
    while let Some((a, b)) = pending.pop() {
        if !Shape::is_array_or_buffer(ctx, a) {
            pending.extend(
                Shape::children(ctx, a)
                    .iter()
                    .copied()
                    .zip(Shape::children(ctx, b).iter().copied()),
            );
            continue;
        }
        let a = if a.deref(ctx).is::<BufferType>() {
            Shape::array_view(ctx, a)?
        } else {
            a
        };
        let b = if b.deref(ctx).is::<BufferType>() {
            Shape::array_view(ctx, b)?
        } else {
            b
        };
        let (mut a_layout, mut b_layout) = (
            Shape::layout(ctx, a)?.as_deref().cloned(),
            Shape::layout(ctx, b)?.as_deref().cloned(),
        );
        for layout in [&mut a_layout, &mut b_layout].into_iter().flatten() {
            layout.memory_space = 0;
            if splits {
                layout.split_configs.clear();
            }
        }
        if match (&a_layout, &b_layout) {
            (Some(a), Some(b)) => !layouts_equal(ctx, a, b),
            (None, None) => false,
            _ => true,
        } {
            return Ok(false);
        }
        pending.extend(
            Shape::children(ctx, a)
                .iter()
                .copied()
                .zip(Shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}

fn compare(
    ctx: &Context,
    actual: TypeHandle,
    expected: TypeHandle,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let exact = matches!(
        opcode,
        Call | Conditional
            | Constant
            | CopyDone
            | CopyStart
            | CustomCall
            | GetTupleElement
            | Infeed
            | Outfeed
            | OptimizationBarrier
            | Parameter
            | Recv
            | RecvDone
            | Send
            | SendDone
            | Tuple
            | While
            | Fusion
    );
    let equal = if exact && options.layout_sensitive {
        if matches!(
            opcode,
            Opcode::Constant | Opcode::Infeed | Opcode::Send | Opcode::Recv | Opcode::CopyStart
        ) {
            same_minor_layout(ctx, actual, expected)?
        } else {
            options.same_shapes(ctx, actual, expected)
        }
    } else if !exact && options.allow_mixed_precision {
        Shape::compatible_ignoring_fp_precision(ctx, actual, expected)
    } else {
        same(ctx, actual, expected)
    };
    require(equal, "declared result differs from inferred shape")
}

fn same_minor_layout(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    if !Shape::equal_ignoring_layout(ctx, a, b) {
        return Ok(false);
    }
    let mut pending = vec![(a, b)];
    while let Some((a, b)) = pending.pop() {
        if !Shape::is_array_or_buffer(ctx, a) {
            pending.extend(
                Shape::children(ctx, a)
                    .iter()
                    .copied()
                    .zip(Shape::children(ctx, b).iter().copied()),
            );
            continue;
        }
        let (a, b) = (
            if a.deref(ctx).is::<BufferType>() {
                Shape::array_view(ctx, a)?
            } else {
                a
            },
            if b.deref(ctx).is::<BufferType>() {
                Shape::array_view(ctx, b)?
            } else {
                b
            },
        );
        match (Shape::layout(ctx, a)?, Shape::layout(ctx, b)?) {
            (Some(a), Some(b)) => {
                if a.minor_to_major != b.minor_to_major
                    || a.physical_shape.len() != b.physical_shape.len()
                    || !a
                        .physical_shape
                        .iter()
                        .zip(&b.physical_shape)
                        .all(|(&a, &b)| Shape::equal_including_layout(ctx, a, b))
                {
                    return Ok(false);
                }
            }
            (None, None) => {}
            _ => return Ok(false),
        }
        pending.extend(
            Shape::children(ctx, a)
                .iter()
                .copied()
                .zip(Shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}
pub(crate) fn instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    let expected = match opcode {
        Parameter => {
            let _: Ref<'_, Index> = required(ctx, op, "hlo_index")?;
            result
        }
        Constant => {
            let literal: Ref<'_, Literal> = required(ctx, op, "hlo_literal")?;
            literal.verify(ctx)?;
            literal.shape()
        }
        Tuple => infer::infer_tuple(ctx, &operands)?,
        GetTupleElement => infer::infer_get_tuple_element(
            ctx,
            operands[0],
            i64::from(required::<Index>(ctx, op, "hlo_index")?.0),
        )?,
        Domain => {
            // Sharding-domain consistency is checked with the enclosing graph.
            let _: Option<Ref<'_, crate::sharding::Sharding>> =
                optional_attr(ctx, op, "hlo_domain_entry_sharding")?;
            let _: Option<Ref<'_, crate::sharding::Sharding>> =
                optional_attr(ctx, op, "hlo_domain_exit_sharding")?;
            operands[0]
        }
        OptimizationBarrier => operands[0],
        AfterAll => Shape::token(ctx),
        AddDependency => {
            require(
                operands[1].deref(ctx).is::<TokenType>(),
                "add-dependency requires a token second operand",
            )?;
            operands[0]
        }
        ReplicaId | PartitionId => Shape::scalar(ctx, ElementType::U32)?,
        Call | While | Conditional => {
            let callees: Ref<'_, Callees> = required(ctx, op, "hlo_callees")?;
            require(
                match opcode {
                    Call => callees.0.len() == 1,
                    While => callees.0.len() == 2,
                    _ => !callees.0.is_empty(),
                },
                "invalid callee cardinality",
            )?;
            if opcode == Conditional {
                let (selector, branches) = operands.split_first().ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "conditional requires a selector".into()
                    })
                })?;
                infer::validate_conditional_selector(
                    ctx,
                    *selector,
                    branches.len(),
                    callees.0.len(),
                )?;
            }
            if opcode == Call && flag(ctx, op, "hlo_is_composite")? {
                let attrs: Ref<'_, FrontendAttributes> =
                    required(ctx, op, "hlo_frontend_attributes")?;
                require(
                    attrs.get("composite.name").is_some_and(|s| !s.is_empty())
                        && attrs
                            .get("composite.attributes")
                            .is_none_or(|s| !s.is_empty())
                        && attrs
                            .get("composite.version")
                            .is_none_or(|s| s.trim().parse::<i64>().is_ok_and(|n| n >= 0)),
                    "invalid composite name, attributes or version",
                )?;
            }
            result
        }
        CustomCall => {
            verify_custom_call(ctx, op, result, &operands, options)?;
            result
        }
        Rng => {
            require(operands.len() == 2, "rng requires two scalar operands")?;
            let a = tensor(ctx, Shape::array_view(ctx, operands[0])?)?;
            let b = tensor(ctx, Shape::array_view(ctx, operands[1])?)?;
            let out = tensor(ctx, Shape::array_view(ctx, result)?)?;
            require(
                a.dimensions.is_empty() && b.dimensions.is_empty(),
                "rng requires two scalar operands",
            )?;
            require(
                a.element == b.element
                    && (a.element == out.element
                        || options.allow_mixed_precision
                            && a.element.same_ignoring_fp_precision(out.element)),
                "rng operand and result element types differ",
            )?;
            let distribution: Ref<'_, RandomDistribution> = required(ctx, op, "hlo_distribution")?;
            require(
                match *distribution {
                    RandomDistribution::Uniform => {
                        a.element.is_float()
                            || a.element.is_integer()
                            || a.element == ElementType::Pred
                    }
                    RandomDistribution::Normal => a.element.is_float(),
                },
                "invalid random distribution or element type",
            )?;
            result
        }
        RngBitGenerator => {
            let _: Ref<'_, RandomAlgorithm> = required(ctx, op, "hlo_algorithm")?;
            if Shape::is_tuple(ctx, result) {
                let parts = Shape::children(ctx, result);
                require(
                    parts.len() == 2 && same(ctx, parts[0], operands[0]),
                    "rng-bit-generator requires two tuple components and matching state shapes",
                )?;
            }
            result
        }
        RngGetAndUpdateState => Shape::array(ctx, ElementType::U64, vec![2])?,
        AllGather
        | AllGatherStart
        | AllGatherDone
        | AllReduce
        | AllReduceStart
        | AllReduceDone
        | AllToAll
        | ReduceScatter
        | RaggedAllToAll
        | CollectiveBroadcast
        | CollectiveReduce
        | CollectivePermute
        | CollectivePermuteStart
        | CollectivePermuteDone => collective::infer(ctx, op, opcode, result, &operands, options)?,
        AsyncStart | AsyncUpdate | AsyncDone | CopyStart | CopyDone | Send | SendDone | Recv
        | RecvDone | Infeed | Outfeed => {
            asynchronous::infer(ctx, op, opcode, result, &operands, options)?
        }
        _ => {
            return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "opcode is not a control-family instruction".into()
            }));
        }
    };
    if !matches!(opcode, AsyncStart | AsyncUpdate) {
        verify_aliases(ctx, op, opcode, result, &operands, options)?;
    }
    compare(ctx, result, expected, opcode, options)
}

/// Validate local callee signatures resolved by the module's symbol index.
///
/// The slice must include every callee in attribute order, even when empty.
/// While uses body then condition; inline Fusion requires no external callee.
pub(crate) fn verify_callees(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
    resolved: &[&Signature],
) -> Result<()> {
    if opcode == Fusion {
        return require(
            resolved.is_empty(),
            "inline fusion cannot call an external computation",
        );
    }
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    let parameters = |operands: &[TypeHandle], callee: &Signature| -> Result<()> {
        require(
            operands.len() == callee.0.len(),
            "operand/parameter cardinality mismatch",
        )?;
        require(
            operands
                .iter()
                .zip(&callee.0)
                .all(|(&a, &b)| options.same_shapes(ctx, a, b)),
            "operand and parameter shapes differ",
        )
    };
    match opcode {
        Call => {
            require(resolved.len() == 1, "call requires exactly one computation")?;
            parameters(&operands, resolved[0])?;
            compare(ctx, result, resolved[0].1, opcode, options)
        }
        While => {
            require(
                resolved.len() == 2,
                "while requires body and condition computations",
            )?;
            for callee in resolved {
                parameters(&operands, callee)?;
            }
            infer::infer_while(ctx, operands[0], resolved[0], resolved[1])?;
            compare(ctx, result, resolved[0].1, opcode, options)
        }
        Conditional => {
            let (selector, branches) = operands.split_first().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "conditional requires a selector".into()
                })
            })?;
            infer::validate_conditional_selector(ctx, *selector, branches.len(), resolved.len())?;
            for (&operand, branch) in branches.iter().zip(resolved) {
                parameters(&[operand], branch)?;
                compare(ctx, result, branch.1, opcode, options)?;
            }
            Ok(())
        }
        AllReduce | AllReduceStart | ReduceScatter | CollectiveReduce => {
            collective::verify_reducer(ctx, op, &operands, resolved, options)
        }
        AsyncStart | AsyncUpdate | AsyncDone => {
            asynchronous::verify_callees(ctx, op, opcode, result, &operands, resolved, options)
        }
        CustomCall => Ok(()),
        _ => require(resolved.is_empty(), "unexpected called computations"),
    }
}

fn verify_custom_call(
    ctx: &Context,
    op: Ptr<Operation>,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<()> {
    let target: Ref<'_, StringAttr> = required(ctx, op, "hlo_target")?;
    let target = target.as_str();
    if flag(ctx, op, "hlo_constrain_layout")?
        && !matches!(target, "LayoutConstraint" | "control_dep")
    {
        require(
            Shape::has_complete_layout(ctx, result)?,
            "layout-constrained custom call requires result layout",
        )?;
        let constraints: Ref<'_, ShapeList> = required(ctx, op, "hlo_operand_layouts")?;
        require(
            constraints.0.len() == operands.len(),
            "layout-constrained custom call requires operand layouts",
        )?;
        for (&operand, &constraint) in operands.iter().zip(&constraints.0) {
            require(
                Shape::has_complete_layout(ctx, constraint)? && same(ctx, operand, constraint),
                "operand layout constraint is missing layout or has incompatible geometry",
            )?;
        }
    }
    if target == "GetRngSeed" {
        require(
            operands.is_empty() && same(ctx, result, Shape::scalar(ctx, ElementType::U64)?),
            "GetRngSeed requires no operands and a U64 scalar result",
        )?;
    }
    if matches!(target, "Pin" | "Unpin") {
        require(operands.len() == 1, "Pin/Unpin requires one operand")?;
        let (array, buffer) = if target == "Pin" {
            (operands[0], result)
        } else {
            (result, operands[0])
        };
        require(
            Shape::is_array(ctx, array) && buffer.deref(ctx).is::<BufferType>(),
            "Pin/Unpin requires an array and a buffer",
        )?;
        let inner = Shape::array_view(ctx, buffer)?;
        require(
            if options.layout_sensitive {
                options.same_shapes(ctx, array, inner)
            } else {
                Shape::equal_ignoring_layout(ctx, array, inner)
            },
            "Pin/Unpin requires equal array shapes",
        )?;
        let aliases: Ref<'_, OutputOperandAliases> = required(ctx, op, "hlo_aliases")?;
        require(aliases.0.len() == 1, "Pin/Unpin requires exactly one alias")?;
    } else if target == "CreateBuffer" {
        require(
            operands.is_empty() && result.deref(ctx).is::<BufferType>(),
            "CreateBuffer requires no operands and a buffer result",
        )?;
    }
    let _: Option<Ref<'_, CustomCallSchedule>> = optional_attr(ctx, op, "hlo_schedule")?;
    let _: Option<Ref<'_, CustomCallApiVersion>> = optional_attr(ctx, op, "hlo_api_version")?;
    flag(ctx, op, "hlo_has_side_effect")?;
    Ok(())
}

fn indexed(ctx: &Context, shape: TypeHandle, index: &[i64]) -> Result<TypeHandle> {
    let index = index
        .iter()
        .map(|&i| {
            usize::try_from(i).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "negative shape index".into()
                })
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Shape::subshape(ctx, shape, &index)
}

fn verify_aliases(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<()> {
    if !matches!(
        opcode,
        Opcode::Call
            | Opcode::Fusion
            | Opcode::CustomCall
            | Opcode::AsyncStart
            | Opcode::AsyncUpdate
    ) {
        return Ok(());
    }
    let Some(aliases) = optional_attr::<OutputOperandAliases>(ctx, op, "hlo_aliases")? else {
        return Ok(());
    };
    let target = optional_attr::<StringAttr>(ctx, op, "hlo_target")?;
    let target = target.as_ref().map(|t| t.as_str());
    let skip = optional_attr::<FrontendAttributes>(ctx, op, "hlo_frontend_attributes")?
        .is_some_and(|a| a.get("xla_skip_custom_call_alias_shape_check").is_some());
    for alias in &aliases.0 {
        let operand = usize::try_from(alias.operand_index)
            .ok()
            .and_then(|i| operands.get(i))
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "alias operand index out of range".into()
                })
            })?;
        let mut operand = indexed(ctx, *operand, &alias.operand_shape_index.0)?;
        let mut output = indexed(ctx, result, &alias.output_shape_index.0)?;
        if opcode == Opcode::CustomCall && skip {
            return Ok(());
        }
        if opcode == Opcode::CustomCall && target.is_some_and(|t| matches!(t, "Pin" | "Unpin")) {
            if operand.deref(ctx).is::<BufferType>() {
                operand = Shape::array_view(ctx, operand)?;
            }
            if output.deref(ctx).is::<BufferType>() {
                output = Shape::array_view(ctx, output)?;
            }
        }
        let equal = if opcode == Opcode::Fusion {
            same(ctx, operand, output)
        } else if options.layout_sensitive
            && opcode == Opcode::CustomCall
            && Shape::is_array_or_buffer(ctx, operand)
            && Shape::dimensions(ctx, Shape::array_view(ctx, operand)?)?
                .iter()
                .all(|&d| d == 1)
        {
            same_ignoring_memory(ctx, operand, output, false)?
        } else if options.layout_sensitive && opcode == Opcode::Call {
            call_alias_equal(ctx, operand, output)?
        } else if options.layout_sensitive {
            options.same_shapes(ctx, operand, output)
        } else {
            same(ctx, operand, output)
        };
        require(equal, "aliased output and operand subshapes differ")?;
    }
    Ok(())
}

/// Complete fusion alias layout checks after the module caller index is built.
///
/// Set `relaxed` for a non-entry computation with no callers. An inline enclosing
/// Fusion also enables relaxation. Only memory spaces and tiles are ignored;
/// alias indices, geometry, and all other layout fields remain checked.
pub(crate) fn verify_fusion_aliases(
    ctx: &Context,
    op: Ptr<Operation>,
    options: VerifyOptions,
    relaxed: bool,
) -> Result<()> {
    require(
        Operation::is_op::<FusionOp>(op, ctx),
        "fusion alias hook requires a fusion",
    )?;
    let data = op.deref(ctx);
    require(data.get_num_results() == 1, "fusion requires one result")?;
    let Some(aliases) = optional_attr::<OutputOperandAliases>(ctx, op, "hlo_aliases")? else {
        return Ok(());
    };
    let relaxed = relaxed
        || data
            .get_parent_op(ctx)
            .is_some_and(|parent| Operation::is_op::<FusionOp>(parent, ctx));
    for alias in &aliases.0 {
        let operand = usize::try_from(alias.operand_index).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "negative alias operand index".into()
            })
        })?;
        require(
            operand < data.get_num_operands(),
            "alias operand index out of range",
        )?;
        let operand = indexed(
            ctx,
            data.get_operand(operand).get_type(ctx),
            &alias.operand_shape_index.0,
        )?;
        let result = indexed(ctx, data.get_type(0), &alias.output_shape_index.0)?;
        let equal = if !options.layout_sensitive {
            same(ctx, operand, result)
        } else if relaxed {
            fusion_alias_equal(ctx, operand, result)?
        } else {
            options.same_shapes(ctx, operand, result)
        };
        require(equal, "aliased output and operand subshapes differ")?;
    }
    Ok(())
}

fn call_alias_equal(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    if !same(ctx, a, b) {
        return Ok(false);
    }
    let mut pending = vec![(a, b)];
    while let Some((a, b)) = pending.pop() {
        if !Shape::is_array_or_buffer(ctx, a) {
            pending.extend(
                Shape::children(ctx, a)
                    .iter()
                    .copied()
                    .zip(Shape::children(ctx, b).iter().copied()),
            );
            continue;
        }
        match (Shape::layout(ctx, a)?, Shape::layout(ctx, b)?) {
            (Some(a), Some(b)) if !layouts_equal(ctx, &a, &b) => return Ok(false),
            (Some(_), None) | (None, Some(_)) => return Ok(false),
            _ => {}
        }
        pending.extend(
            Shape::children(ctx, a)
                .iter()
                .copied()
                .zip(Shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}

fn fusion_alias_equal(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    if !Shape::equal_ignoring_layout(ctx, a, b) {
        return Ok(false);
    }
    let mut pending = vec![(a, b)];
    while let Some((a, b)) = pending.pop() {
        if !Shape::is_array_or_buffer(ctx, a) {
            pending.extend(
                Shape::children(ctx, a)
                    .iter()
                    .copied()
                    .zip(Shape::children(ctx, b).iter().copied()),
            );
            continue;
        }
        match (Shape::layout(ctx, a)?, Shape::layout(ctx, b)?) {
            (Some(a), Some(b)) => {
                let (mut a, mut b) = ((*a).clone(), (*b).clone());
                a.memory_space = 0;
                b.memory_space = 0;
                a.tiles.clear();
                b.tiles.clear();
                if !layouts_equal(ctx, &a, &b) {
                    return Ok(false);
                }
            }
            (Some(_), None) | (None, Some(_)) => return Ok(false),
            _ => {}
        }
        pending.extend(
            Shape::children(ctx, a)
                .iter()
                .copied()
                .zip(Shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}

pub(crate) fn fusion(ctx: &Context, fusion: &FusionOp, options: VerifyOptions) -> Result<()> {
    let data = fusion.get_operation().deref(ctx);
    require(
        data.get_num_results() == 1
            && data.num_regions() == 1
            && data.get_num_successors() == 0
            && fusion.get_attr_hlo_fusion_kind(ctx).is_some(),
        "invalid fusion structure",
    )?;
    let block = super::single_block(ctx, fusion.body(ctx))?;
    require(
        block.deref(ctx).get_num_arguments() <= data.get_num_operands(),
        "fusion has more parameters than operands",
    )?;
    require(
        block
            .deref(ctx)
            .arguments()
            .map(|arg| arg.get_type(ctx))
            .zip(data.operand_types(ctx))
            .all(|(a, b)| options.same_shapes(ctx, a, b)),
        "fusion arguments mismatch",
    )?;
    drop(data);
    let result_type = fusion.result(ctx).get_type(ctx);
    require(
        options.same_shapes(ctx, result_type, super::returned(ctx, block)?),
        "fusion return shape mismatch",
    )?;
    let instructions: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let mut used = HashSet::new();
    let mut root = None;
    for &instruction in &instructions {
        let data = instruction.deref(ctx);
        for operand in data.operands() {
            used.insert(operand);
        }
        if Operation::is_op::<ReturnOp>(instruction, ctx) {
            require(
                data.get_num_operands() == 1,
                "fusion return must have one operand",
            )?;
            root = Some(data.get_operand(0));
        }
    }
    let root = root.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "fusion lacks root".into()
        })
    })?;
    for argument in block.deref(ctx).arguments() {
        require(
            argument == root || used.contains(&argument),
            "unused non-root parameter in fusion",
        )?;
    }
    let root_users = instructions
        .iter()
        .filter(|&&op| !Operation::is_op::<ReturnOp>(op, ctx))
        .any(|&op| op.deref(ctx).operands().any(|operand| operand == root));
    require(!root_users, "fusion root cannot have users")?;
    for &op in &instructions {
        if Operation::is_op::<ReturnOp>(op, ctx) {
            continue;
        }
        let data = op.deref(ctx);
        let live = data
            .results()
            .any(|value| value == root || used.contains(&value));
        require(
            live || has_local_side_effect(ctx, op)?,
            "unused non-root instruction in fusion",
        )?;
    }
    let data = fusion.get_operation().deref(ctx);
    verify_aliases(
        ctx,
        fusion.get_operation(),
        Opcode::Fusion,
        result_type,
        &data.operand_types(ctx).collect::<Vec<_>>(),
        options,
    )
}

/// Query direct effects with channel-sensitive semantics, without visiting callees.
fn has_local_side_effect(ctx: &Context, op: Ptr<Operation>) -> Result<bool> {
    has_local_side_effect_with_context(ctx, op, true)
}

/// Query direct effects, optionally treating channel IDs as semantically relevant.
fn has_local_side_effect_with_context(
    ctx: &Context,
    op: Ptr<Operation>,
    channel_id_sensitive: bool,
) -> Result<bool> {
    Ok(match asynchronous::opcode(ctx, op) {
        Some(
            Send
            | SendDone
            | Recv
            | RecvDone
            | Rng
            | RngGetAndUpdateState
            | Infeed
            | Outfeed
            | AllReduceStart
            | AllReduceDone
            | AllGatherStart
            | AllGatherDone
            | CollectiveBroadcast
            | CollectivePermuteStart
            | CollectivePermuteDone,
        ) => true,
        Some(AllToAll | AllGather | AllReduce | ReduceScatter | CollectiveReduce) => {
            flag(ctx, op, "hlo_constrain_layout")?
                || channel(ctx, op)?.is_some() && channel_id_sensitive
        }
        Some(CollectivePermute) => channel(ctx, op)?.is_some() && channel_id_sensitive,
        Some(CustomCall) => flag(ctx, op, "hlo_has_side_effect")?,
        _ => false,
    })
}
