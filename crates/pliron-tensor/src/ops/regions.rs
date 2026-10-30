use super::*;
use crate::support::*;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{OperandSegmentSizesAttr, UnitAttr},
        op_interfaces::IsTerminatorInterface,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::op_interface_impl,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use pliron_mlir_builtin::attributes::I64ArrayAttr;

/// Populates a region using its block and index arguments, returning one element.
pub type BodyBuilder<'a> =
    dyn FnOnce(&mut Context, Ptr<BasicBlock>, &[Value]) -> Result<Value> + 'a;

fn allocate<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    operands: Vec<Value>,
    result: TypeHandle,
) -> Result<T> {
    let op = Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![result],
        operands,
        vec![],
        1,
    );
    let (shape, _) = ranked(ctx, op, result)?;
    let index = shaped::index(ctx);
    let block = BasicBlock::new(ctx, None, vec![index; shape.len()]);
    block.insert_at_back(op.deref(ctx).get_region(0), ctx);
    Ok(make(op))
}

fn body(ctx: &Context, op: Ptr<Operation>) -> Result<Ptr<BasicBlock>> {
    let raw = op.deref(ctx);
    ensure(
        ctx,
        op,
        raw.num_regions() == 1,
        "expected exactly one region",
    )?;
    let blocks: Vec<_> = raw.get_region(0).deref(ctx).iter(ctx).collect();
    ensure(ctx, op, blocks.len() == 1, "expected a single-block region")?;
    Ok(blocks[0])
}

fn populate(ctx: &mut Context, op: Ptr<Operation>, build: Box<BodyBuilder<'_>>) -> Result<()> {
    let block = body(ctx, op)?;
    let args: Vec<_> = block.deref(ctx).arguments().collect();
    let yielded = build(ctx, block, &args)?;
    YieldOp::new(ctx, yielded)
        .get_operation()
        .insert_at_back(block, ctx);
    Ok(())
}

fn verify_body(
    ctx: &Context,
    op: Ptr<Operation>,
    shape: &[i64],
    element: TypeHandle,
) -> Result<()> {
    let block = body(ctx, op)?;
    let arg_types: Vec<_> = block
        .deref(ctx)
        .arguments()
        .map(|v| v.get_type(ctx))
        .collect();
    ensure(
        ctx,
        op,
        arg_types.len() == shape.len(),
        "body must have one index argument per tensor dimension",
    )?;
    indices(ctx, op, &arg_types)?;
    let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let last = operations
        .last()
        .copied()
        .ok_or_else(|| fail(ctx, op, "body must terminate with tensor.yield"))?;
    ensure(
        ctx,
        op,
        Operation::get_op::<YieldOp>(last, ctx).is_some(),
        "body must terminate with tensor.yield",
    )?;
    for (i, &nested) in operations.iter().enumerate() {
        if Operation::get_op::<YieldOp>(nested, ctx).is_some() {
            ensure(
                ctx,
                op,
                i == operations.len() - 1,
                "tensor.yield must be last in its block",
            )?;
        }
        Operation::get_op_dyn(nested, ctx).verify(ctx)?;
    }
    let (yielded, _) = types(ctx, last);
    ensure(
        ctx,
        op,
        yielded == [element],
        "body must yield exactly one tensor element",
    )
}

operation!(
    GenerateOp,
    "tensor.generate",
    "Generates tensor elements using a scalar region.",
    "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig"
);
impl GenerateOp {
    /// Allocates a generator with an empty body and one index argument per axis.
    ///
    /// Returns an error for a non-ranked result type. Allocation is not transactional.
    pub fn new(ctx: &mut Context, result: TypeHandle, dynamic_sizes: Vec<Value>) -> Result<Self> {
        allocate(ctx, |op| Self { op }, dynamic_sizes, result)
    }
    /// Populates and verifies a generator. Failed builds may leave allocated IR.
    pub fn build(
        ctx: &mut Context,
        result: TypeHandle,
        dynamic_sizes: Vec<Value>,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, result, dynamic_sizes)?;
        populate(ctx, op.get_operation(), body)?;
        op.verify(ctx)?;
        Ok(op)
    }
    /// Returns the single body block, or an error for malformed regions.
    pub fn body(&self, ctx: &Context) -> Result<Ptr<BasicBlock>> {
        body(ctx, self.get_operation())
    }
}
impl Verify for GenerateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 0..=usize::MAX, 1, 1)?;
        let (args, results) = types(ctx, op);
        let (shape, element) = ranked(ctx, op, results[0])?;
        dynamic_count(ctx, op, &shape, &args)?;
        verify_body(ctx, op, &shape, element)
    }
}

operation!(
    PadOp,
    "tensor.pad",
    "Pads a tensor using a region to compute padding elements.",
    "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig"
);
impl PadOp {
    /// Allocates padding with an empty region. Low/high lists have one entry per axis.
    ///
    /// Returns an error for a non-ranked result type. Allocation is not transactional.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        result: TypeHandle,
        low: Vec<MixedIndex>,
        high: Vec<MixedIndex>,
        nofold: bool,
    ) -> Result<Self> {
        let (low, low_dynamic) = split(&low);
        let (high, high_dynamic) = split(&high);
        let segments =
            OperandSegmentSizesAttr(vec![1, low_dynamic.len() as u32, high_dynamic.len() as u32]);
        let operands = std::iter::once(source)
            .chain(low_dynamic)
            .chain(high_dynamic)
            .collect();
        let op = allocate(ctx, |op| Self { op }, operands, result)?;
        set(ctx, op.get_operation(), &LOW, I64ArrayAttr::new(low));
        set(ctx, op.get_operation(), &HIGH, I64ArrayAttr::new(high));
        set(ctx, op.get_operation(), &SEGMENTS, segments);
        if nofold {
            set(ctx, op.get_operation(), &NOFOLD, UnitAttr::new());
        }
        Ok(op)
    }
    /// Populates and verifies padding. Failed builds may leave allocated IR.
    pub fn build(
        ctx: &mut Context,
        source: Value,
        result: TypeHandle,
        low: Vec<MixedIndex>,
        high: Vec<MixedIndex>,
        nofold: bool,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, source, result, low, high, nofold)?;
        populate(ctx, op.get_operation(), body)?;
        op.verify(ctx)?;
        Ok(op)
    }
    /// Returns the single padding region block, or an error for malformed regions.
    pub fn body(&self, ctx: &Context) -> Result<Ptr<BasicBlock>> {
        body(ctx, self.get_operation())
    }
}
impl Verify for PadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 1..=usize::MAX, 1, 1)?;
        let (args, results) = types(ctx, op);
        let (source, element) = ranked(ctx, op, args[0])?;
        let (result, e) = ranked(ctx, op, results[0])?;
        ensure(
            ctx,
            op,
            element == e && source.len() == result.len(),
            "pad must preserve rank and element type",
        )?;
        let segments = segments(ctx, op)?;
        ensure(
            ctx,
            op,
            segments.len() == 3 && segments[0] == (0, 1),
            "pad requires source, low, and high operand segments",
        )?;
        let low = array(ctx, op, &LOW)?;
        let high = array(ctx, op, &HIGH)?;
        ensure(
            ctx,
            op,
            low.len() == source.len() && high.len() == source.len(),
            "pad low/high lists must match tensor rank",
        )?;
        for (i, list) in [&low, &high].into_iter().enumerate() {
            let (start, count) = segments[i + 1];
            mixed_types(ctx, op, list, &args[start..start + count], false)?;
        }
        for i in 0..source.len() {
            if source[i] < 0 || low[i] == i64::MIN || high[i] == i64::MIN {
                continue;
            }
            let inferred = source[i]
                .checked_add(low[i])
                .and_then(|n| n.checked_add(high[i]))
                .ok_or_else(|| fail(ctx, op, "padded dimension exceeds i64"))?;
            ensure(
                ctx,
                op,
                result[i] == inferred,
                "pad result static dimension does not equal source plus padding",
            )?;
        }
        if op.deref(ctx).attributes.0.contains_key(&NOFOLD) {
            ensure(
                ctx,
                op,
                op.deref(ctx).attributes.get::<UnitAttr>(&NOFOLD).is_some(),
                "nofold must be a unit attribute",
            )?;
        }
        verify_body(ctx, op, &source, element)
    }
}

operation!(
    YieldOp,
    "tensor.yield",
    "Yields one element from a generate or pad region."
);
impl YieldOp {
    /// Allocates a region terminator yielding one tensor element.
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        new(ctx, |op| Self { op }, vec![value], vec![])
    }
}
#[op_interface_impl]
impl IsTerminatorInterface for YieldOp {}
impl Verify for YieldOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 1..=1, 0, 0)?;
        let block = op
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| fail(ctx, op, "yield must belong to a generate or pad region"))?;
        let region = block
            .deref(ctx)
            .get_parent_region()
            .ok_or_else(|| fail(ctx, op, "yield has no parent region"))?;
        let parent = region.deref(ctx).get_parent_op();
        ensure(
            ctx,
            op,
            Operation::get_op::<GenerateOp>(parent, ctx).is_some()
                || Operation::get_op::<PadOp>(parent, ctx).is_some(),
            "yield parent must be tensor.generate or tensor.pad",
        )?;
        ensure(
            ctx,
            op,
            block.deref(ctx).iter(ctx).last() == Some(op),
            "yield must terminate its block",
        )?;
        let (_, results) = types(ctx, parent);
        ensure(
            ctx,
            op,
            results.len() == 1,
            "yield parent must have one result",
        )?;
        let (_, element) = ranked(ctx, op, results[0])?;
        ensure(
            ctx,
            op,
            types(ctx, op).0 == [element],
            "yield must produce exactly one tensor element",
        )
    }
}
