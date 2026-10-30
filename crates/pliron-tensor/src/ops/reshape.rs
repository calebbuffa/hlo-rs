use super::*;
use crate::support::*;
use pliron::{
    common_traits::Verify, context::Context, op::Op, result::Result, r#type::TypeHandle,
    value::Value,
};
use pliron_arith::shaped;
use pliron_linalg::attributes::I64ArrayAttr;

operation!(
    ReshapeOp,
    "tensor.reshape",
    "Reshapes a ranked or unranked tensor using a shape tensor."
);
impl ReshapeOp {
    /// Allocates a reshape with explicit source, shape vector, and result type.
    pub fn new(ctx: &mut Context, source: Value, shape: Value, result: TypeHandle) -> Self {
        new(ctx, |op| Self { op }, vec![source, shape], vec![result])
    }
}
impl Verify for ReshapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 2..=2, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (source, element) = tensor(ctx, op, args[0])?;
        let (result, e) = tensor(ctx, op, results[0])?;
        let (shape, shape_element) = ranked(ctx, op, args[1])?;
        ensure(
            ctx,
            op,
            element == e
                && shape.len() == 1
                && shaped::is_signless_int_or_index(ctx, shape_element),
            "reshape requires equal tensor elements and a rank-one integer/index shape",
        )?;
        if let Some(result) = &result {
            ensure(
                ctx,
                op,
                usize::try_from(shape[0]).ok() == Some(result.len()),
                "ranked reshape result requires a statically sized shape vector matching its rank",
            )?;
        }
        if let Some((source, result)) = source.as_ref().zip(result.as_ref())
            && let Some((a, b)) = elements(ctx, op, source)?.zip(elements(ctx, op, result)?)
        {
            ensure(ctx, op, a == b, "reshape changes the number of elements")?;
        }
        Ok(())
    }
}

operation!(
    ExpandShapeOp,
    "tensor.expand_shape",
    "Expands tensor axes according to contiguous reassociation groups."
);
impl ExpandShapeOp {
    /// Allocates an expansion. `output_shape` includes one entry per result axis.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        result: TypeHandle,
        reassociation: Vec<Vec<i64>>,
        output_shape: Vec<MixedIndex>,
    ) -> Self {
        let (statics, dynamic) = split(&output_shape);
        let op = new(
            ctx,
            |op| Self { op },
            std::iter::once(source).chain(dynamic).collect(),
            vec![result],
        );
        set_reassociation(ctx, op.get_operation(), reassociation);
        set(
            ctx,
            op.get_operation(),
            &OUTPUT_SHAPE,
            I64ArrayAttr::new(statics),
        );
        op
    }
}

operation!(
    CollapseShapeOp,
    "tensor.collapse_shape",
    "Collapses contiguous tensor axes into single dimensions."
);
impl CollapseShapeOp {
    /// Allocates a collapse using contiguous reassociation groups.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        result: TypeHandle,
        reassociation: Vec<Vec<i64>>,
    ) -> Self {
        let op = new(ctx, |op| Self { op }, vec![source], vec![result]);
        set_reassociation(ctx, op.get_operation(), reassociation);
        op
    }
}

fn reassociate(
    ctx: &Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
    expand: bool,
) -> Result<()> {
    structure(ctx, op, 1..=if expand { usize::MAX } else { 1 }, 1, 0)?;
    let (args, results) = types(ctx, op);
    let (source, se) = ranked(ctx, op, args[0])?;
    let (result, re) = ranked(ctx, op, results[0])?;
    ensure(
        ctx,
        op,
        se == re,
        "reassociation must preserve tensor element type",
    )?;
    let (expanded, collapsed) = if expand {
        (&result, &source)
    } else {
        (&source, &result)
    };
    let groups = reassociation(ctx, op)?;
    ensure(
        ctx,
        op,
        expanded.len() >= collapsed.len() && groups.len() == collapsed.len(),
        "reassociation count/ranks disagree",
    )?;
    let flattened: Vec<_> = groups.iter().flatten().copied().collect();
    if collapsed.is_empty() {
        ensure(
            ctx,
            op,
            expanded.iter().all(|&d| d == 1),
            "scalar reassociation requires all expanded dimensions to be one",
        )?;
    } else {
        ensure(
            ctx,
            op,
            groups.iter().all(|g| !g.is_empty())
                && flattened.len() == expanded.len()
                && flattened
                    .iter()
                    .enumerate()
                    .all(|(i, &d)| usize::try_from(d).ok() == Some(i)),
            "reassociation must partition all expanded axes into nonempty contiguous groups",
        )?;
        let mut start = 0;
        for (i, group) in groups.iter().enumerate() {
            let shape = &expanded[start..start + group.len()];
            let inferred = elements(ctx, op, shape)?.unwrap_or(-1);
            ensure(
                ctx,
                op,
                collapsed[i] == inferred,
                "collapsed dimensions must equal the corresponding expanded products",
            )?;
            start += group.len();
        }
    }
    if let Some((a, b)) = elements(ctx, op, &source)?.zip(elements(ctx, op, &result)?) {
        ensure(
            ctx,
            op,
            a == b,
            "reassociation changes the number of elements",
        )?;
    }
    if expand {
        let statics = array(ctx, op, &OUTPUT_SHAPE)?;
        ensure(
            ctx,
            op,
            statics.len() == result.len(),
            "output_shape must match result rank",
        )?;
        mixed_types(ctx, op, &statics, &args[1..], false)?;
        dynamic_count(ctx, op, &result, &args[1..])?;
        ensure(
            ctx,
            op,
            result.iter().zip(&statics).all(|(&d, &s)| d < 0 || d == s),
            "static output shape disagrees with result type",
        )?;
    }
    Ok(())
}
impl Verify for ExpandShapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        reassociate(ctx, self.get_operation(), true)
    }
}
impl Verify for CollapseShapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        reassociate(ctx, self.get_operation(), false)
    }
}
