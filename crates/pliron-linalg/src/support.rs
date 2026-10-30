//! Verification and construction helpers shared by `linalg` and IREE operations.

use crate::Diagnostic;
use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::OperandSegmentSizesAttr,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use stablehlo::{
    attributes::{AffineExpr, AffineMap},
    types::TensorType,
};

/// Verification error carrying the operation's name and source location.
pub(crate) fn fail(
    ctx: &Context,
    op: Ptr<Operation>,
    reason: &'static str,
) -> pliron::result::Error {
    let location = op.deref(ctx).loc().disp(ctx).to_string();
    Diagnostic::InvalidOperation {
        operation: Operation::get_op_dyn(op, ctx).get_opid().to_string(),
        location,
        reason,
    }
    .into()
}

/// Returns the dimensions and element of a ranked tensor type.
pub(crate) fn ranked(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    ty.deref(ctx)
        .downcast_ref::<TensorType>()
        .map(|t| (t.dimensions.clone(), t.element))
}

/// Returns the dimensions and element of a ranked tensor, or `([], ty)` for a scalar.
pub(crate) fn ranked_or_scalar(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    ranked(ctx, ty).or_else(|| shaped::is_scalar(ctx, ty).then(|| (Vec::new(), ty)))
}

/// Returns whether `element` is a float, signless integer, or `index`.
pub(crate) fn numeric(ctx: &Context, element: TypeHandle) -> bool {
    shaped::is_float(ctx, element) || shaped::is_signless_int_or_index(ctx, element)
}

/// Returns the position of a bare dimension expression `dN`.
pub(crate) fn as_dim(expr: &AffineExpr) -> Option<usize> {
    match expr.nodes.as_slice() {
        [node] if node.kind == 0 => usize::try_from(node.value).ok(),
        _ => None,
    }
}

/// Returns the positions of every dimension referenced by `expr`.
pub(crate) fn dims_of(expr: &AffineExpr) -> impl Iterator<Item = usize> + '_ {
    expr.nodes
        .iter()
        .filter(|node| node.kind == 0)
        .filter_map(|node| usize::try_from(node.value).ok())
}

/// Returns whether every result of `map` is a distinct bare dimension.
pub(crate) fn is_projected_permutation(map: &AffineMap) -> bool {
    let mut seen = vec![false; map.dimensions];
    map.results.iter().all(|expr| match as_dim(expr) {
        Some(d) if d < seen.len() && !seen[d] => {
            seen[d] = true;
            true
        }
        _ => false,
    })
}

/// Verifies indexing maps against operand shapes, as MLIR's structured-op
/// interface verifier does.
///
/// Every map must take `loops` dimensions and no symbols and produce one
/// result per operand dimension (scalars have rank 0). Static extents indexed
/// by a bare `dN` must agree; compound expressions evaluated at the largest
/// static loop index must stay inside the operand.
///
/// Returns the inferred static loop extents, `-1` where unknown.
pub(crate) fn verify_indexing(
    ctx: &Context,
    op: Ptr<Operation>,
    maps: &[AffineMap],
    shapes: &[Vec<i64>],
    loops: usize,
) -> Result<Vec<i64>> {
    if maps.len() != shapes.len() {
        return Err(fail(ctx, op, "expected one indexing map per operand"));
    }
    for (map, shape) in maps.iter().zip(shapes) {
        map.validate()?;
        if map.dimensions != loops || map.symbols != 0 {
            return Err(fail(
                ctx,
                op,
                "indexing maps must take one dimension per loop and no symbols",
            ));
        }
        if map.results.len() != shape.len() {
            return Err(fail(
                ctx,
                op,
                "indexing map result count must equal the operand rank",
            ));
        }
    }
    let mut extents = vec![-1i64; loops];
    for (map, shape) in maps.iter().zip(shapes) {
        for (expr, &size) in map.results.iter().zip(shape) {
            if let Some(d) = as_dim(expr)
                && size >= 0
            {
                if extents[d] >= 0 && extents[d] != size {
                    return Err(fail(
                        ctx,
                        op,
                        "inconsistent static extents for a loop dimension",
                    ));
                }
                extents[d] = size;
            }
        }
    }
    for (map, shape) in maps.iter().zip(shapes) {
        for (expr, &size) in map.results.iter().zip(shape) {
            if as_dim(expr).is_some() || size < 0 {
                continue;
            }
            if dims_of(expr).any(|d| extents[d] <= 0) {
                continue;
            }
            let point: Vec<i64> = extents.iter().map(|&e| (e - 1).max(0)).collect();
            if expr.evaluate_at(&point, &[])? + 1 > i128::from(size) {
                return Err(fail(
                    ctx,
                    op,
                    "operand dimension is smaller than the range its indexing map accesses",
                ));
            }
        }
    }
    Ok(extents)
}

/// Verifies a region-less destination-passing operation with `inputs` `ins`
/// operands and one ranked-tensor `outs` operand mirrored by its result.
///
/// Returns each operand's shape (empty for scalars) and element type.
pub(crate) fn verify_dps(
    ctx: &Context,
    op: Ptr<Operation>,
    inputs: usize,
) -> Result<Vec<(Vec<i64>, TypeHandle)>> {
    {
        let ir = op.deref(ctx);
        if ir.get_num_operands() != inputs + 1 || ir.get_num_results() != 1 || ir.num_regions() != 0
        {
            return Err(fail(ctx, op, "operand, result, or region count mismatch"));
        }
    }
    verify_tensor_outputs(ctx, op, inputs..inputs + 1)?;
    operand_types(ctx, op)
        .into_iter()
        .map(|ty| {
            ranked_or_scalar(ctx, ty)
                .ok_or_else(|| fail(ctx, op, "operands must be ranked tensors or scalars"))
        })
        .collect()
}

/// Returns whether two static-or-dynamic (`-1`) extents can be equal.
pub(crate) fn compatible(a: i64, b: i64) -> bool {
    a < 0 || b < 0 || a == b
}

/// Returns whether two shapes have equal rank and compatible extents.
pub(crate) fn compatible_shapes(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| compatible(x, y))
}

pub(crate) fn operand_types(ctx: &Context, op: Ptr<Operation>) -> Vec<TypeHandle> {
    op.deref(ctx).operands().map(|v| v.get_type(ctx)).collect()
}

pub(crate) fn result_types(ctx: &Context, op: Ptr<Operation>) -> Vec<TypeHandle> {
    op.deref(ctx).results().map(|v| v.get_type(ctx)).collect()
}

/// Allocates an operation with one region holding an entry block of `args`.
pub(crate) fn new_with_body<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
    args: Vec<TypeHandle>,
) -> T {
    let info = T::get_concrete_op_info();
    let op = Operation::new(ctx, info, results, operands, vec![], 1);
    let block = BasicBlock::new(ctx, None, args);
    let region = op.deref(ctx).get_region(0);
    block.insert_at_back(region, ctx);
    make(op)
}

/// Returns the only block of the only region, if the operation has that structure.
pub(crate) fn entry_block(ctx: &Context, op: Ptr<Operation>) -> Option<Ptr<BasicBlock>> {
    let ir = op.deref(ctx);
    if ir.num_regions() != 1 {
        return None;
    }
    let region = ir.get_region(0);
    let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
    (blocks.len() == 1).then(|| blocks[0])
}

/// Returns the entry block, failing when the operation lacks exactly one single-block region.
pub(crate) fn body(ctx: &Context, op: Ptr<Operation>) -> Result<Ptr<BasicBlock>> {
    entry_block(ctx, op).ok_or_else(|| fail(ctx, op, "expected one region with one block"))
}

pub(crate) fn block_arg_types(ctx: &Context, block: Ptr<BasicBlock>) -> Vec<TypeHandle> {
    block
        .deref(ctx)
        .arguments()
        .map(|v| v.get_type(ctx))
        .collect()
}

pub(crate) fn block_args(ctx: &Context, block: Ptr<BasicBlock>) -> Vec<Value> {
    block.deref(ctx).arguments().collect()
}

/// Returns the operand types of the block's terminator if it is a `T`.
pub(crate) fn terminator_operands<T: Op>(
    ctx: &Context,
    owner: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    reason: &'static str,
) -> Result<Vec<TypeHandle>> {
    let last = block
        .deref(ctx)
        .iter(ctx)
        .last()
        .ok_or_else(|| fail(ctx, owner, reason))?;
    if Operation::get_op_dyn(last, ctx)
        .downcast_ref::<T>()
        .is_none()
    {
        return Err(fail(ctx, owner, reason));
    }
    Ok(operand_types(ctx, last))
}

/// Inserts `op` at the end of `block`.
pub(crate) fn append(ctx: &Context, op: Ptr<Operation>, block: Ptr<BasicBlock>) {
    op.insert_at_back(block, ctx);
}

/// Returns the operation that owns the region containing `op`, if any.
pub(crate) fn parent_op(ctx: &Context, op: Ptr<Operation>) -> Option<Ptr<Operation>> {
    let block = op.deref(ctx).get_parent_block()?;
    let region = block.deref(ctx).get_parent_region()?;
    Some(region.deref(ctx).get_parent_op())
}

/// Verifies stored `[inputs, outputs]` segment sizes against the operand count.
///
/// Returns the input and output counts.
pub(crate) fn segments(
    ctx: &Context,
    op: Ptr<Operation>,
    sizes: Option<&OperandSegmentSizesAttr>,
) -> Result<(usize, usize)> {
    let sizes = sizes.ok_or_else(|| fail(ctx, op, "missing operand segment sizes"))?;
    let [inputs, outputs] = sizes.0[..] else {
        return Err(fail(ctx, op, "expected two operand segments (ins, outs)"));
    };
    let (inputs, outputs) = (inputs as usize, outputs as usize);
    if inputs + outputs != op.deref(ctx).get_num_operands() {
        return Err(fail(
            ctx,
            op,
            "operand segment sizes do not match operand count",
        ));
    }
    Ok((inputs, outputs))
}

/// Verifies that results mirror the ranked-tensor `outs` operands at `outputs`.
pub(crate) fn verify_tensor_outputs(
    ctx: &Context,
    op: Ptr<Operation>,
    outputs: std::ops::Range<usize>,
) -> Result<()> {
    let operands = operand_types(ctx, op);
    let outs = &operands[outputs];
    if outs.iter().any(|&ty| ranked(ctx, ty).is_none()) {
        return Err(fail(ctx, op, "outs operands must be ranked tensors"));
    }
    if result_types(ctx, op) != outs {
        return Err(fail(
            ctx,
            op,
            "result types must equal the outs operand types",
        ));
    }
    if op.deref(ctx).get_num_successors() != 0 {
        return Err(fail(ctx, op, "operation must not have successors"));
    }
    Ok(())
}
