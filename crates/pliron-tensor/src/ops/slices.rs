use super::*;
use crate::support::*;
use pliron::{
    builtin::attributes::OperandSegmentSizesAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_linalg::attributes::I64ArrayAttr;

/// Mixed offsets, sizes, and strides, one entry per source/destination axis.
#[derive(Clone)]
pub struct Slice {
    /// Starting indices.
    pub offsets: Vec<MixedIndex>,
    /// Slice extents.
    pub sizes: Vec<MixedIndex>,
    /// Step between successive indices.
    pub strides: Vec<MixedIndex>,
}

fn allocate<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    mut operands: Vec<Value>,
    results: Vec<TypeHandle>,
    slice: Slice,
) -> T {
    let prefix = operands.len() as u32;
    let mut counts = vec![1; prefix as usize];
    let mut attrs = Vec::new();
    for (key, list) in [
        (&OFFSETS, slice.offsets),
        (&SIZES, slice.sizes),
        (&STRIDES, slice.strides),
    ] {
        let (statics, dynamic) = split(&list);
        counts.push(dynamic.len() as u32);
        operands.extend(dynamic);
        attrs.push((key, statics));
    }
    let op = new(ctx, make, operands, results);
    for (key, statics) in attrs {
        set(ctx, op.get_operation(), key, I64ArrayAttr::new(statics));
    }
    set(
        ctx,
        op.get_operation(),
        &SEGMENTS,
        OperandSegmentSizesAttr(counts),
    );
    op
}

/// Matches upstream's greedy rank reduction: only static unit extents drop.
fn rank_reduced(full: &[i64], reduced: &[i64]) -> bool {
    let mut i = 0;
    for &d in full {
        if reduced.get(i) == Some(&d) {
            i += 1;
        } else if d != 1 {
            return false;
        }
    }
    i == reduced.len()
}

fn verify_slice(ctx: &Context, op: Ptr<Operation>, insert: bool, parallel: bool) -> Result<()> {
    let prefix = if insert { 2 } else { 1 };
    structure(ctx, op, prefix..=usize::MAX, usize::from(!parallel), 0)?;
    let (args, results) = types(ctx, op);
    let (base, element) = ranked(ctx, op, args[usize::from(insert)])?;
    let slice_ty = if insert { args[0] } else { results[0] };
    let (slice_shape, e) = ranked(ctx, op, slice_ty)?;
    ensure(ctx, op, element == e, "slice element types differ")?;
    if insert && !parallel {
        ensure(
            ctx,
            op,
            results[0] == args[1],
            "insert_slice result must equal destination type",
        )?;
    }
    let segments = segments(ctx, op)?;
    ensure(
        ctx,
        op,
        segments.len() == prefix + 3 && segments[..prefix].iter().all(|&(_, n)| n == 1),
        "invalid slice operand segments",
    )?;
    let mut lists = Vec::new();
    for (i, key) in [&OFFSETS, &SIZES, &STRIDES].into_iter().enumerate() {
        let values = array(ctx, op, key)?;
        ensure(
            ctx,
            op,
            values.len() == base.len(),
            "slice mixed lists must match tensor rank",
        )?;
        let (start, count) = segments[prefix + i];
        let dynamic = &args[start..start + count];
        indices(ctx, op, dynamic)?;
        ensure(
            ctx,
            op,
            values.iter().filter(|&&n| n == i64::MIN).count() == count,
            "slice mixed-list operand count mismatch",
        )?;
        if i < 2 {
            ensure(
                ctx,
                op,
                values.iter().all(|&n| n == i64::MIN || n >= 0),
                "slice offsets and sizes must be nonnegative",
            )?;
        }
        lists.push(values);
    }
    let sizes: Vec<_> = lists[1]
        .iter()
        .map(|&n| if n == i64::MIN { -1 } else { n })
        .collect();
    ensure(
        ctx,
        op,
        rank_reduced(&sizes, &slice_shape),
        "slice result/source does not match static sizes or a valid unit-dimension rank reduction",
    )?;
    for (i, &extent) in base.iter().enumerate() {
        let offset = lists[0][i];
        let size = lists[1][i];
        let stride = lists[2][i];
        if extent < 0 || offset == i64::MIN {
            continue;
        }
        ensure(ctx, op, offset < extent, "slice offset out of bounds")?;
        if size == i64::MIN || stride == i64::MIN {
            continue;
        }
        // Compute in i128 to preserve the bound check without i64 overflow.
        let last = i128::from(offset) + (i128::from(size) - 1) * i128::from(stride);
        ensure(
            ctx,
            op,
            last < i128::from(extent),
            "slice extends beyond tensor bounds",
        )?;
    }
    if parallel {
        let parent = op
            .deref(ctx)
            .get_parent_block()
            .and_then(|b| b.deref(ctx).get_parent_region())
            .map(|r| r.deref(ctx).get_parent_op())
            .ok_or_else(|| {
                fail(
                    ctx,
                    op,
                    "parallel_insert_slice requires an in-parallel parent",
                )
            })?;
        let parent = Operation::get_op_dyn(parent, ctx);
        ensure(
            ctx,
            op,
            op_cast::<dyn InParallelOpInterface>(&*parent).is_some(),
            "parallel_insert_slice parent must implement InParallelOpInterface",
        )?;
    }
    Ok(())
}

operation!(
    ExtractSliceOp,
    "tensor.extract_slice",
    "Extracts a possibly rank-reduced tensor slice."
);
impl ExtractSliceOp {
    /// Allocates a slice with an explicit result type.
    pub fn new(ctx: &mut Context, source: Value, result: TypeHandle, slice: Slice) -> Self {
        allocate(ctx, |op| Self { op }, vec![source], vec![result], slice)
    }
}
impl Verify for ExtractSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_slice(ctx, self.get_operation(), false, false)
    }
}

operation!(
    InsertSliceOp,
    "tensor.insert_slice",
    "Returns a destination with a tensor slice replaced."
);
impl InsertSliceOp {
    /// Allocates a slice insertion with a result of the destination's type.
    pub fn new(ctx: &mut Context, source: Value, dest: Value, slice: Slice) -> Self {
        let result = dest.get_type(ctx);
        allocate(
            ctx,
            |op| Self { op },
            vec![source, dest],
            vec![result],
            slice,
        )
    }
}
impl Verify for InsertSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_slice(ctx, self.get_operation(), true, false)
    }
}

operation!(
    ParallelInsertSliceOp,
    "tensor.parallel_insert_slice",
    "Combines a slice into a parallel operation's destination."
);
impl ParallelInsertSliceOp {
    /// Allocates a result-less insertion to be attached to an in-parallel parent.
    pub fn new(ctx: &mut Context, source: Value, dest: Value, slice: Slice) -> Self {
        allocate(ctx, |op| Self { op }, vec![source, dest], vec![], slice)
    }
}
impl Verify for ParallelInsertSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_slice(ctx, self.get_operation(), true, true)
    }
}
