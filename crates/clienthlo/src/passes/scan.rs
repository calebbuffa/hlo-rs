use super::{
    builder::{Builder, ranked},
    unsupported,
};
use crate::{Result, ops::ScanOp};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    irbuild::{inserter::OpInsertionPoint, rewriter::Rewriter},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::Typed,
    value::Value,
};
use stablehlo::{
    attributes::ComparisonDirection as D,
    ops as hlo,
    types::{ElementType, TensorType},
};

pub(super) fn preflight(ctx: &Context, source: Ptr<Operation>) -> Result<()> {
    let scan = ScanOp::from_operation(source);
    let inputs = scan
        .inputs(ctx)
        .ok_or_else(|| unsupported(ctx, source, "missing scan inputs"))?;
    let carries = scan
        .inits(ctx)
        .ok_or_else(|| unsupported(ctx, source, "missing scan carries"))?;
    let axis = scan
        .dimension(ctx)
        .ok_or_else(|| unsupported(ctx, source, "missing scan dimension"))? as usize;
    let results: Vec<_> = source.deref(ctx).result_types().collect();
    if inputs.is_empty()
        && scan.scan_dim_size(ctx).is_none()
        && results
            .first()
            .and_then(|&ty| ranked(ctx, ty))
            .and_then(|(shape, _)| shape.get(axis).copied())
            .is_none_or(|size| size < 0)
    {
        return Err(unsupported(
            ctx,
            source,
            "cannot infer scan length without inputs",
        ));
    }
    for (index, &ty) in results[..results.len() - carries.len()].iter().enumerate() {
        let (shape, _) = ranked(ctx, ty)
            .ok_or_else(|| unsupported(ctx, source, "scan outputs must be ranked"))?;
        for (dim, &extent) in shape.iter().enumerate() {
            if dim != axis && extent < 0 {
                let reference = inputs
                    .get(index)
                    .or_else(|| inputs.first())
                    .and_then(|v| ranked(ctx, v.get_type(ctx)));
                if reference.is_none_or(|(shape, _)| dim >= shape.len()) {
                    return Err(unsupported(
                        ctx,
                        source,
                        "cannot determine dynamic scan output extent from inputs",
                    ));
                }
            }
        }
    }
    Ok(())
}

pub(super) fn lower(b: &mut Builder<'_>) -> Result<Vec<Value>> {
    let scan = ScanOp::from_operation(b.source);
    let inputs = scan
        .inputs(b.ctx)
        .ok_or_else(|| unsupported(b.ctx, b.source, "missing scan inputs"))?;
    let inits = scan
        .inits(b.ctx)
        .ok_or_else(|| unsupported(b.ctx, b.source, "missing scan carries"))?;
    let axis = scan
        .dimension(b.ctx)
        .ok_or_else(|| unsupported(b.ctx, b.source, "missing scan dimension"))?
        as usize;
    let reverse = scan
        .is_reverse(b.ctx)
        .ok_or_else(|| unsupported(b.ctx, b.source, "invalid scan direction"))?;
    let results: Vec<_> = b.source.deref(b.ctx).result_types().collect();
    let outputs = results.len() - inits.len();
    let count = if let Some(&input) = inputs.first() {
        let (shape, _) = ranked(b.ctx, input.get_type(b.ctx))
            .ok_or_else(|| unsupported(b.ctx, b.source, "scan inputs must be ranked"))?;
        b.dimension(input, axis, shape[axis])?
    } else {
        let count = scan
            .scan_dim_size(b.ctx)
            .or_else(|| {
                results
                    .first()
                    .and_then(|&ty| ranked(b.ctx, ty))
                    .and_then(|(shape, _)| shape.get(axis).copied())
                    .and_then(|size| u64::try_from(size).ok())
            })
            .ok_or_else(|| {
                unsupported(b.ctx, b.source, "cannot infer scan length without inputs")
            })?;
        b.integer(count as i64, ElementType::I64)?
    };
    let zero = b.integer(0, ElementType::I64)?;
    let one = b.integer(1, ElementType::I64)?;
    let mut initial = vec![zero];
    initial.extend(inits.iter().copied());
    for (index, &ty) in results[..outputs].iter().enumerate() {
        let (shape, _) = ranked(b.ctx, ty)
            .ok_or_else(|| unsupported(b.ctx, b.source, "scan outputs must be ranked"))?;
        let sizes = if shape.iter().any(|&size| size < 0) {
            let mut sizes = Vec::new();
            for (dim, &extent) in shape.iter().enumerate() {
                sizes.push(if dim == axis {
                    count
                } else if extent >= 0 {
                    b.integer(extent, ElementType::I64)?
                } else {
                    let reference = inputs
                        .get(index)
                        .or_else(|| inputs.first())
                        .copied()
                        .ok_or_else(|| {
                            unsupported(
                                b.ctx,
                                b.source,
                                "cannot determine dynamic scan output extent without inputs",
                            )
                        })?;
                    b.dimension(reference, dim, -1)?
                });
            }
            Some(b.shape_vector(sizes)?)
        } else {
            None
        };
        initial.push(b.zero(ty, sizes)?);
    }
    let statically_empty = scan.scan_dim_size(b.ctx) == Some(0)
        || inputs.first().is_some_and(|input| {
            ranked(b.ctx, input.get_type(b.ctx)).is_some_and(|(shape, _)| shape[axis] == 0)
        });
    if statically_empty {
        let mut replacements = initial[1 + inits.len()..].to_vec();
        replacements.extend(inits);
        return Ok(replacements);
    }
    let types: Vec<_> = initial.iter().map(|v| v.get_type(b.ctx)).collect();
    let loop_op = hlo::WhileOp::new(b.ctx, initial, types.clone());
    let condition = BasicBlock::new(b.ctx, None, types.clone());
    condition.insert_at_back(loop_op.condition_region(b.ctx), b.ctx);
    b.in_block(condition, |b| {
        let index = condition.deref(b.ctx).get_argument(0);
        let predicate = b.compare(index, count, D::Lt)?;
        let ret = hlo::ReturnOp::new(b.ctx, predicate);
        b.insert(ret)
    })?;
    let body = BasicBlock::new(b.ctx, None, types);
    body.insert_at_back(loop_op.body_region(b.ctx), b.ctx);
    b.in_block(body, |b| {
        let arguments: Vec<_> = body.deref(b.ctx).arguments().collect();
        let index = arguments[0];
        let position = if reverse {
            let last = b.sub(count, one)?;
            b.sub(last, index)?
        } else {
            index
        };
        let mut mapped = Vec::new();
        for &input in &inputs {
            mapped.push(b.slice_axis(input, axis, position)?);
        }
        mapped.extend(arguments[1..1 + inits.len()].iter().copied());
        let original = scan
            .entry_block(b.ctx)
            .ok_or_else(|| unsupported(b.ctx, b.source, "scan body is missing"))?;
        let original_args: Vec<_> = original.deref(b.ctx).arguments().collect();
        for (old, new) in original_args.into_iter().zip(mapped) {
            b.rewriter.replace_value_uses_with(b.ctx, old, new);
        }
        let operations: Vec<_> = original.deref(b.ctx).iter(b.ctx).collect();
        let (&terminator, operations) = operations
            .split_last()
            .ok_or_else(|| unsupported(b.ctx, b.source, "scan body is empty"))?;
        for &operation in operations {
            b.rewriter
                .move_operation(b.ctx, operation, OpInsertionPoint::AtBlockEnd(body));
        }
        let returned: Vec<_> = terminator.deref(b.ctx).operands().collect();
        b.rewriter.erase_operation(b.ctx, terminator);
        let mut updated = Vec::new();
        for (i, &value) in returned[..outputs].iter().enumerate() {
            let output = arguments[1 + inits.len() + i];
            let (mut shape, element) = ranked(b.ctx, output.get_type(b.ctx))
                .ok_or_else(|| unsupported(b.ctx, b.source, "scan outputs must be ranked"))?;
            let mut sizes = Vec::new();
            for (dim, &extent) in shape.iter().enumerate() {
                sizes.push(if dim == axis {
                    one
                } else {
                    b.dimension(output, dim, extent)?
                });
            }
            shape[axis] = 1;
            let ty = TensorType::get(b.ctx, shape.clone(), element).into();
            let update = b.reshape(value, ty, sizes)?;
            let mut starts = vec![zero; shape.len()];
            starts[axis] = position;
            let update = hlo::DynamicUpdateSliceOp::new(
                b.ctx,
                output,
                update,
                starts,
                output.get_type(b.ctx),
            );
            updated.push(b.emit(update)?);
        }
        let next = b.add(index, one)?;
        let mut yield_values = vec![next];
        yield_values.extend(returned[outputs..].iter().copied());
        yield_values.extend(updated);
        let ret = hlo::ReturnOp::new_many(b.ctx, yield_values);
        b.insert(ret)
    })?;
    let operation = loop_op.get_operation();
    b.insert(loop_op)?;
    let values: Vec<_> = operation.deref(b.ctx).results().collect();
    let mut replacements = values[1 + inits.len()..].to_vec();
    replacements.extend(values[1..1 + inits.len()].iter().copied());
    Ok(replacements)
}
