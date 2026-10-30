use crate::Diagnostic;
use pliron::{
    attribute::Attribute,
    builtin::attributes::{OperandSegmentSizesAttr, VecAttr},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use pliron_linalg::attributes::I64ArrayAttr;
use stablehlo::types::{TensorType, UnrankedTensorType};

pub(crate) fn fail(
    ctx: &Context,
    op: Ptr<Operation>,
    reason: &'static str,
) -> pliron::result::Error {
    Diagnostic::InvalidOperation {
        operation: Operation::get_op_dyn(op, ctx).get_opid().to_string(),
        location: op.deref(ctx).loc().disp(ctx).to_string(),
        reason,
    }
    .into()
}

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

pub(crate) fn new<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    operands: Vec<Value>,
    results: Vec<TypeHandle>,
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

pub(crate) fn structure(
    ctx: &Context,
    op: Ptr<Operation>,
    operands: std::ops::RangeInclusive<usize>,
    results: usize,
    regions: usize,
) -> Result<()> {
    let raw = op.deref(ctx);
    ensure(
        ctx,
        op,
        operands.contains(&raw.get_num_operands())
            && raw.get_num_results() == results
            && raw.num_regions() == regions
            && raw.get_num_successors() == 0,
        "invalid operand, result, region, or successor count",
    )
}

pub(crate) fn types(ctx: &Context, op: Ptr<Operation>) -> (Vec<TypeHandle>, Vec<TypeHandle>) {
    let raw = op.deref(ctx);
    (
        raw.operands().map(|v| v.get_type(ctx)).collect(),
        raw.results().map(|v| v.get_type(ctx)).collect(),
    )
}

pub(crate) fn ranked(
    ctx: &Context,
    op: Ptr<Operation>,
    ty: TypeHandle,
) -> Result<(Vec<i64>, TypeHandle)> {
    ty.deref(ctx)
        .downcast_ref::<TensorType>()
        .map(|t| (t.dimensions.clone(), t.element))
        .ok_or_else(|| fail(ctx, op, "expected a ranked tensor"))
}

pub(crate) fn tensor(
    ctx: &Context,
    op: Ptr<Operation>,
    ty: TypeHandle,
) -> Result<(Option<Vec<i64>>, TypeHandle)> {
    let ty = ty.deref(ctx);
    if let Some(t) = ty.downcast_ref::<TensorType>() {
        Ok((Some(t.dimensions.clone()), t.element))
    } else if let Some(t) = ty.downcast_ref::<UnrankedTensorType>() {
        Ok((None, t.element))
    } else {
        Err(fail(ctx, op, "expected a tensor"))
    }
}

pub(crate) fn compatible(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| x < 0 || y < 0 || x == y)
}

pub(crate) fn indices(ctx: &Context, op: Ptr<Operation>, types: &[TypeHandle]) -> Result<()> {
    ensure(
        ctx,
        op,
        types.iter().all(|&t| shaped::is_index(ctx, t)),
        "dynamic sizes and indices must have index type",
    )
}

pub(crate) fn elements(ctx: &Context, op: Ptr<Operation>, dims: &[i64]) -> Result<Option<i64>> {
    if dims.iter().any(|&d| d < 0) {
        return Ok(None);
    }
    // A zero dimension gives an empty tensor even if other extents overflow.
    if dims.contains(&0) {
        return Ok(Some(0));
    }
    dims.iter()
        .try_fold(1_i64, |n, &d| n.checked_mul(d))
        .map(Some)
        .ok_or_else(|| fail(ctx, op, "number of elements exceeds i64"))
}

pub(crate) fn dynamic_count(
    ctx: &Context,
    op: Ptr<Operation>,
    shape: &[i64],
    dynamic: &[TypeHandle],
) -> Result<()> {
    indices(ctx, op, dynamic)?;
    ensure(
        ctx,
        op,
        shape.iter().filter(|&&d| d == -1).count() == dynamic.len(),
        "dynamic size count must match dynamic tensor dimensions",
    )
}

pub(crate) fn set<A: Attribute>(
    ctx: &Context,
    op: Ptr<Operation>,
    key: &pliron::identifier::Identifier,
    attr: A,
) {
    op.deref_mut(ctx).attributes.set(key.clone(), attr);
}

pub(crate) fn array(
    ctx: &Context,
    op: Ptr<Operation>,
    key: &pliron::identifier::Identifier,
) -> Result<Vec<i64>> {
    op.deref(ctx)
        .attributes
        .get::<I64ArrayAttr>(key)
        .map(|a| a.values.clone())
        .ok_or_else(|| {
            fail(
                ctx,
                op,
                "missing or incorrectly typed integer-array attribute",
            )
        })
}

pub(crate) fn split(list: &[crate::ops::MixedIndex]) -> (Vec<i64>, Vec<Value>) {
    let mut statics = Vec::new();
    let mut dynamic = Vec::new();
    for &entry in list {
        match entry {
            crate::ops::MixedIndex::Static(n) => statics.push(n),
            crate::ops::MixedIndex::Dynamic(v) => {
                statics.push(i64::MIN);
                dynamic.push(v);
            }
        }
    }
    (statics, dynamic)
}

pub(crate) fn segments(ctx: &Context, op: Ptr<Operation>) -> Result<Vec<(usize, usize)>> {
    let raw = op.deref(ctx);
    let a = raw
        .attributes
        .get::<OperandSegmentSizesAttr>(&crate::ops::SEGMENTS)
        .ok_or_else(|| fail(ctx, op, "missing operand segment sizes"))?;
    let mut start = 0_usize;
    let mut segments = Vec::new();
    for &n in &a.0 {
        let n = n as usize;
        segments.push((start, n));
        start = start
            .checked_add(n)
            .ok_or_else(|| fail(ctx, op, "operand count overflow"))?;
    }
    ensure(
        ctx,
        op,
        start == raw.get_num_operands(),
        "operand segments do not match operands",
    )?;
    Ok(segments)
}

pub(crate) fn mixed_types(
    ctx: &Context,
    op: Ptr<Operation>,
    statics: &[i64],
    dynamic: &[TypeHandle],
    positive: bool,
) -> Result<()> {
    indices(ctx, op, dynamic)?;
    ensure(
        ctx,
        op,
        statics.iter().filter(|&&n| n == i64::MIN).count() == dynamic.len(),
        "mixed-list dynamic operand count mismatch",
    )?;
    ensure(
        ctx,
        op,
        statics
            .iter()
            .all(|&n| n == i64::MIN || if positive { n > 0 } else { n >= 0 }),
        "invalid static size, offset, stride, or padding",
    )
}

pub(crate) fn reassociation(ctx: &Context, op: Ptr<Operation>) -> Result<Vec<Vec<i64>>> {
    let raw = op.deref(ctx);
    let groups = raw
        .attributes
        .get::<VecAttr>(&crate::ops::REASSOCIATION)
        .ok_or_else(|| fail(ctx, op, "missing reassociation"))?;
    groups
        .0
        .iter()
        .map(|a| {
            a.downcast_ref::<I64ArrayAttr>()
                .map(|a| a.values.clone())
                .ok_or_else(|| fail(ctx, op, "reassociation must contain integer arrays"))
        })
        .collect()
}

pub(crate) fn set_reassociation(ctx: &Context, op: Ptr<Operation>, groups: Vec<Vec<i64>>) {
    let groups = groups
        .into_iter()
        .map(|g| -> pliron::attribute::AttrObj { Box::new(I64ArrayAttr::new(g)) })
        .collect();
    set(ctx, op, &crate::ops::REASSOCIATION, VecAttr::new(groups));
}
