//! Verification and construction helpers shared by `iree_linalg_ext` operations.

use crate::{Diagnostic, ops::YieldOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{attributes::OperandSegmentSizesAttr, types::IntegerType},
    common_traits::Verify,
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
use pliron_linalg::ops::BodyBuilder;
use pliron_mlir_builtin::{
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

/// Fails with `reason` unless `ok`.
pub(crate) fn ensure(
    ctx: &Context,
    op: Ptr<Operation>,
    ok: bool,
    reason: &'static str,
) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(fail(ctx, op, reason))
    }
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

/// Returns whether `ty` is a non-tensor scalar.
pub(crate) fn is_plain_scalar(ctx: &Context, ty: TypeHandle) -> bool {
    ranked(ctx, ty).is_none() && shaped::is_scalar(ctx, ty)
}

/// Returns the shape and element of a ranked-tensor operand, failing otherwise.
pub(crate) fn ranked_operand(
    ctx: &Context,
    op: Ptr<Operation>,
    ty: TypeHandle,
) -> Result<(Vec<i64>, TypeHandle)> {
    ranked(ctx, ty).ok_or_else(|| fail(ctx, op, "operand must be a ranked tensor"))
}

/// Returns the element type of a tensor value, or the value's type for a scalar.
pub(crate) fn element(ctx: &Context, value: Value) -> TypeHandle {
    let ty = value.get_type(ctx);
    ranked(ctx, ty).map_or(ty, |(_, e)| e)
}

/// Returns the width of an integer type of any signedness.
pub(crate) fn int_width(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
}

/// Returns whether `ty` is an integer of any signedness or `index`.
pub(crate) fn is_int_or_index(ctx: &Context, ty: TypeHandle) -> bool {
    int_width(ctx, ty).is_some() || shaped::is_index(ctx, ty)
}

/// Returns whether `ty` is an integer of any signedness or a float.
pub(crate) fn is_int_or_float(ctx: &Context, ty: TypeHandle) -> bool {
    int_width(ctx, ty).is_some() || shaped::is_float(ctx, ty)
}

/// Returns the position of a bare dimension expression `dN`.
pub(crate) fn as_dim(expr: &AffineExpr) -> Option<usize> {
    match expr.nodes.as_slice() {
        [node] if node.kind == 0 => usize::try_from(node.value).ok(),
        _ => None,
    }
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

/// Returns whether `values` is a permutation of `0..values.len()`.
pub(crate) fn is_permutation(values: &[i64]) -> bool {
    let mut seen = vec![false; values.len()];
    values.iter().all(|&v| match usize::try_from(v) {
        Ok(i) if i < seen.len() && !seen[i] => {
            seen[i] = true;
            true
        }
        _ => false,
    })
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

pub(crate) fn num_operands(ctx: &Context, op: Ptr<Operation>) -> usize {
    op.deref(ctx).get_num_operands()
}

/// Allocates a region-less operation.
pub(crate) fn new_plain<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> T {
    make(Operation::new(
        ctx,
        T::get_concrete_op_info(),
        results,
        operands,
        vec![],
        0,
    ))
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

/// Returns the operand types of the block's `iree_linalg_ext.yield` terminator.
pub(crate) fn yielded(
    ctx: &Context,
    owner: Ptr<Operation>,
    block: Ptr<BasicBlock>,
) -> Result<Vec<TypeHandle>> {
    let reason = "region must terminate with iree_linalg_ext.yield";
    let last = block
        .deref(ctx)
        .iter(ctx)
        .last()
        .ok_or_else(|| fail(ctx, owner, reason))?;
    if Operation::get_op_dyn(last, ctx)
        .downcast_ref::<YieldOp>()
        .is_none()
    {
        return Err(fail(ctx, owner, reason));
    }
    Ok(operand_types(ctx, last))
}

/// Checks that the region's block arguments are exactly `args` and returns the yielded types.
pub(crate) fn region_signature(
    ctx: &Context,
    op: Ptr<Operation>,
    args: &[TypeHandle],
) -> Result<Vec<TypeHandle>> {
    let block = body(ctx, op)?;
    ensure(
        ctx,
        op,
        block_arg_types(ctx, block) == args,
        "region block arguments have the wrong types",
    )?;
    yielded(ctx, op, block)
}

/// Runs `body` on `op`'s entry block, terminates it with `iree_linalg_ext.yield`, and verifies `op`.
pub(crate) fn populate<T: Op + Verify>(
    ctx: &mut Context,
    op: T,
    body: Box<BodyBuilder<'_>>,
) -> Result<T> {
    let block = self::body(ctx, op.get_operation())?;
    let args = block_args(ctx, block);
    let values = body(ctx, block, &args)?;
    let yield_op = YieldOp::new(ctx, values);
    yield_op.get_operation().insert_at_back(block, ctx);
    op.verify(ctx)?;
    Ok(op)
}

/// Returns the operation that owns the region containing `op`, if any.
pub(crate) fn parent_op(ctx: &Context, op: Ptr<Operation>) -> Option<Ptr<Operation>> {
    let block = op.deref(ctx).get_parent_block()?;
    let region = block.deref(ctx).get_parent_region()?;
    Some(region.deref(ctx).get_parent_op())
}

/// Builds operand segment sizes from per-segment counts.
pub(crate) fn segment_sizes(counts: &[usize]) -> OperandSegmentSizesAttr {
    OperandSegmentSizesAttr(counts.iter().map(|&c| c as u32).collect())
}

/// Verifies `N` stored operand segment sizes against the operand count and
/// returns the start offset and length of each segment.
pub(crate) fn segments<const N: usize>(
    ctx: &Context,
    op: Ptr<Operation>,
    sizes: Option<&OperandSegmentSizesAttr>,
) -> Result<[(usize, usize); N]> {
    let sizes = sizes.ok_or_else(|| fail(ctx, op, "missing operand segment sizes"))?;
    ensure(
        ctx,
        op,
        sizes.0.len() == N,
        "wrong number of operand segments",
    )?;
    let mut out = [(0, 0); N];
    let mut start = 0;
    for (slot, &len) in out.iter_mut().zip(&sizes.0) {
        *slot = (start, len as usize);
        start += len as usize;
    }
    ensure(
        ctx,
        op,
        start == num_operands(ctx, op),
        "operand segment sizes do not match operand count",
    )?;
    Ok(out)
}

/// Verifies that results mirror the ranked-tensor `outs` operands at `outputs`
/// and that the operation has no successors.
pub(crate) fn verify_outputs(
    ctx: &Context,
    op: Ptr<Operation>,
    outputs: std::ops::Range<usize>,
) -> Result<()> {
    let operands = operand_types(ctx, op);
    let outs = operands
        .get(outputs)
        .ok_or_else(|| fail(ctx, op, "missing outs operands"))?;
    ensure(
        ctx,
        op,
        outs.iter().all(|&ty| ranked(ctx, ty).is_some()),
        "outs operands must be ranked tensors",
    )?;
    ensure(
        ctx,
        op,
        result_types(ctx, op) == outs,
        "result types must equal the outs operand types",
    )?;
    ensure(
        ctx,
        op,
        op.deref(ctx).get_num_successors() == 0,
        "operation must not have successors",
    )
}

/// Verifies the operation has no regions.
pub(crate) fn no_regions(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    ensure(
        ctx,
        op,
        op.deref(ctx).num_regions() == 0,
        "operation must not have regions",
    )
}

/// Returns `dim` as an index below `rank`.
pub(crate) fn dimension_in(
    ctx: &Context,
    op: Ptr<Operation>,
    dim: Option<i64>,
    rank: usize,
) -> Result<usize> {
    let dim = dim.ok_or_else(|| fail(ctx, op, "missing dimension"))?;
    usize::try_from(dim)
        .ok()
        .filter(|&d| d < rank)
        .ok_or_else(|| fail(ctx, op, "dimension is out of range"))
}
