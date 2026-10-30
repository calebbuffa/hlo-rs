use super::*;
use crate::support::*;
use pliron::{
    builtin::attributes::UnitAttr,
    common_traits::Verify,
    context::Context,
    op::Op,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use pliron_mlir_builtin::attributes::I64ArrayAttr;
use pliron_mlir_builtin::attributes::IndexAttr;

macro_rules! conversion {
    ($ty:ident, $name:literal, $bitcast:literal) => {
        operation!(
            $ty,
            $name,
            "Changes a tensor's static shape information or element bit interpretation."
        );
        impl $ty {
            /// Allocates a conversion to `result`.
            pub fn new(ctx: &mut Context, source: Value, result: TypeHandle) -> Self {
                new(ctx, |op| Self { op }, vec![source], vec![result])
            }
        }
        impl Verify for $ty {
            fn verify(&self, ctx: &Context) -> Result<()> {
                let op = self.get_operation();
                structure(ctx, op, 1..=1, 1, 0)?;
                let (args, results) = types(ctx, op);
                let (a, ae) = tensor(ctx, op, args[0])?;
                let (b, be) = tensor(ctx, op, results[0])?;
                if $bitcast {
                    let width = |ty: TypeHandle| {
                        ty.deref(ctx)
                            .downcast_ref::<pliron::builtin::types::IntegerType>()
                            .map(|t| t.width())
                            .or_else(|| shaped::float_width(ctx, ty))
                    };
                    ensure(
                        ctx,
                        op,
                        width(ae).is_some() && width(ae) == width(be),
                        "bitcast requires integer/float elements of equal width",
                    )?;
                } else {
                    ensure(ctx, op, ae == be, "cast must preserve element type")?;
                }
                ensure(
                    ctx,
                    op,
                    a.as_ref()
                        .zip(b.as_ref())
                        .is_none_or(|(a, b)| compatible(a, b)),
                    "tensor shapes are incompatible",
                )
            }
        }
    };
}
conversion!(CastOp, "tensor.cast", false);
conversion!(BitcastOp, "tensor.bitcast", true);

operation!(EmptyOp, "tensor.empty", "Creates an uninitialized tensor.");
impl EmptyOp {
    /// Allocates a tensor with one index operand per dynamic dimension.
    pub fn new(ctx: &mut Context, result: TypeHandle, dynamic_sizes: Vec<Value>) -> Self {
        new(ctx, |op| Self { op }, dynamic_sizes, vec![result])
    }
}
impl Verify for EmptyOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 0..=usize::MAX, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (shape, _) = ranked(ctx, op, results[0])?;
        dynamic_count(ctx, op, &shape, &args)
    }
}

operation!(
    SplatOp,
    "tensor.splat",
    "Fills a tensor with one scalar or vector element."
);
impl SplatOp {
    /// Allocates a splat with dynamic extents in result-dimension order.
    pub fn new(
        ctx: &mut Context,
        value: Value,
        result: TypeHandle,
        dynamic_sizes: Vec<Value>,
    ) -> Self {
        new(
            ctx,
            |op| Self { op },
            std::iter::once(value).chain(dynamic_sizes).collect(),
            vec![result],
        )
    }
}
impl Verify for SplatOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 1..=usize::MAX, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (shape, element) = ranked(ctx, op, results[0])?;
        ensure(
            ctx,
            op,
            args[0] == element,
            "splat input must match the tensor element type",
        )?;
        dynamic_count(ctx, op, &shape, &args[1..])
    }
}

operation!(
    FromElementsOp,
    "tensor.from_elements",
    "Creates a static tensor from flattened elements."
);
impl FromElementsOp {
    /// Allocates a static tensor whose elements are supplied in row-major order.
    pub fn new(ctx: &mut Context, values: Vec<Value>, result: TypeHandle) -> Self {
        new(ctx, |op| Self { op }, values, vec![result])
    }
}
impl Verify for FromElementsOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 0..=usize::MAX, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (shape, element) = ranked(ctx, op, results[0])?;
        ensure(
            ctx,
            op,
            elements(ctx, op, &shape)? == i64::try_from(args.len()).ok(),
            "from_elements requires a static shape with exactly one operand per element",
        )?;
        ensure(
            ctx,
            op,
            args.iter().all(|&t| t == element),
            "operand element types must match result",
        )
    }
}

operation!(
    RankOp,
    "tensor.rank",
    "Returns the rank of a ranked or unranked tensor."
);
impl RankOp {
    /// Allocates a tensor rank query with an index result.
    pub fn new(ctx: &mut Context, source: Value) -> Self {
        let index = shaped::index(ctx);
        new(ctx, |op| Self { op }, vec![source], vec![index])
    }
}
impl Verify for RankOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 1..=1, 1, 0)?;
        let (args, results) = types(ctx, op);
        tensor(ctx, op, args[0])?;
        indices(ctx, op, &results)
    }
}

operation!(
    DimOp,
    "tensor.dim",
    "Returns the size of one tensor dimension."
);
impl DimOp {
    /// Allocates a dimension query. Out-of-range runtime indices are undefined,
    /// as upstream; the verifier does not invent a static bounds restriction.
    pub fn new(ctx: &mut Context, source: Value, dimension: Value) -> Self {
        let index = shaped::index(ctx);
        new(ctx, |op| Self { op }, vec![source, dimension], vec![index])
    }
}
impl Verify for DimOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 2..=2, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (shape, _) = tensor(ctx, op, args[0])?;
        ensure(
            ctx,
            op,
            shape.is_none_or(|s| !s.is_empty()),
            "dim requires a nonzero-rank or unranked tensor",
        )?;
        indices(ctx, op, &[args[1], results[0]])
    }
}

operation!(
    ExtractOp,
    "tensor.extract",
    "Reads one element of a ranked tensor."
);
impl ExtractOp {
    /// Allocates an element read with an explicit result type.
    pub fn new(ctx: &mut Context, source: Value, indices: Vec<Value>, result: TypeHandle) -> Self {
        new(
            ctx,
            |op| Self { op },
            std::iter::once(source).chain(indices).collect(),
            vec![result],
        )
    }
}
impl Verify for ExtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 1..=usize::MAX, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (shape, element) = ranked(ctx, op, args[0])?;
        ensure(
            ctx,
            op,
            shape.len() == args.len() - 1 && element == results[0],
            "extract index count or result element type mismatch",
        )?;
        indices(ctx, op, &args[1..])
    }
}

operation!(
    InsertOp,
    "tensor.insert",
    "Returns a tensor with one element replaced."
);
impl InsertOp {
    /// Allocates an element insertion; the result has `dest`'s type.
    pub fn new(ctx: &mut Context, value: Value, dest: Value, indices: Vec<Value>) -> Self {
        let result = dest.get_type(ctx);
        new(
            ctx,
            |op| Self { op },
            [value, dest].into_iter().chain(indices).collect(),
            vec![result],
        )
    }
}
impl Verify for InsertOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 2..=usize::MAX, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (shape, element) = ranked(ctx, op, args[1])?;
        ensure(
            ctx,
            op,
            shape.len() == args.len() - 2 && element == args[0] && results[0] == args[1],
            "insert index count, element type, or destination type mismatch",
        )?;
        indices(ctx, op, &args[2..])
    }
}

operation!(
    ConcatOp,
    "tensor.concat",
    "Concatenates tensors along a static dimension."
);
impl ConcatOp {
    /// Allocates a concatenation with an explicit output type.
    pub fn new(ctx: &mut Context, inputs: Vec<Value>, dim: i64, result: TypeHandle) -> Self {
        let op = new(ctx, |op| Self { op }, inputs, vec![result]);
        set(ctx, op.get_operation(), &DIM, IndexAttr { value: dim });
        op
    }
}
impl Verify for ConcatOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        structure(ctx, op, 1..=usize::MAX, 1, 0)?;
        let (args, results) = types(ctx, op);
        let (out, element) = ranked(ctx, op, results[0])?;
        let dim = op
            .deref(ctx)
            .attributes
            .get::<IndexAttr>(&DIM)
            .and_then(|a| usize::try_from(a.value).ok())
            .filter(|&d| d < out.len())
            .ok_or_else(|| fail(ctx, op, "concatenation dimension out of range"))?;
        let mut inferred = vec![-1; out.len()];
        let mut total = Some(0_i64);
        for arg in args {
            let (shape, e) = ranked(ctx, op, arg)?;
            ensure(
                ctx,
                op,
                e == element && shape.len() == out.len(),
                "concat rank or element type mismatch",
            )?;
            for (i, &d) in shape.iter().enumerate() {
                if i == dim {
                    continue;
                }
                ensure(
                    ctx,
                    op,
                    inferred[i] < 0 || d < 0 || inferred[i] == d,
                    "non-concatenated static dimensions disagree",
                )?;
                if d >= 0 {
                    inferred[i] = d;
                }
            }
            total = if shape[dim] < 0 {
                None
            } else {
                total
                    .map(|n| {
                        n.checked_add(shape[dim])
                            .ok_or_else(|| fail(ctx, op, "concatenated extent exceeds i64"))
                    })
                    .transpose()?
            };
        }
        inferred[dim] = total.unwrap_or(-1);
        ensure(
            ctx,
            op,
            compatible(&out, &inferred),
            "concat result shape mismatch",
        )
    }
}

operation!(
    GatherOp,
    "tensor.gather",
    "Gathers slices from a tensor using an index tensor."
);
impl GatherOp {
    /// Allocates a gather; only indexed unit dimensions may be rank-reduced.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        indices: Value,
        dims: Vec<i64>,
        unique: bool,
        result: TypeHandle,
    ) -> Self {
        let op = new(ctx, |op| Self { op }, vec![source, indices], vec![result]);
        set(
            ctx,
            op.get_operation(),
            &GATHER_DIMS,
            I64ArrayAttr::new(dims),
        );
        if unique {
            set(ctx, op.get_operation(), &UNIQUE, UnitAttr::new());
        }
        op
    }
}

operation!(
    ScatterOp,
    "tensor.scatter",
    "Scatters slices into a destination tensor."
);
impl ScatterOp {
    /// Allocates a scatter. Upstream requires the uniqueness promise.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        dest: Value,
        indices: Value,
        dims: Vec<i64>,
        unique: bool,
    ) -> Self {
        let result = dest.get_type(ctx);
        let op = new(
            ctx,
            |op| Self { op },
            vec![source, dest, indices],
            vec![result],
        );
        set(
            ctx,
            op.get_operation(),
            &SCATTER_DIMS,
            I64ArrayAttr::new(dims),
        );
        if unique {
            set(ctx, op.get_operation(), &UNIQUE, UnitAttr::new());
        }
        op
    }
}

fn gather_scatter(
    ctx: &Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
    scatter: bool,
) -> Result<()> {
    let count = if scatter { 3 } else { 2 };
    structure(ctx, op, count..=count, 1, 0)?;
    let (args, results) = types(ctx, op);
    let base_ty = args[usize::from(scatter)];
    let slice_ty = if scatter { args[0] } else { results[0] };
    let (base, element) = ranked(ctx, op, base_ty)?;
    let (slice, slice_element) = ranked(ctx, op, slice_ty)?;
    let (index_shape, index_element) = ranked(ctx, op, args[count - 1])?;
    let dims = array(ctx, op, if scatter { &SCATTER_DIMS } else { &GATHER_DIMS })?;
    ensure(
        ctx,
        op,
        shaped::is_signless_int_or_index(ctx, index_element),
        "indices must contain signless integers or index",
    )?;
    ensure(
        ctx,
        op,
        !dims.is_empty()
            && dims.windows(2).all(|w| w[0] < w[1])
            && dims.iter().all(|&d| d >= 0 && (d as usize) < base.len())
            && index_shape.last().copied() == i64::try_from(dims.len()).ok(),
        "invalid gather/scatter dimensions or index tuple length",
    )?;
    ensure(
        ctx,
        op,
        element == slice_element,
        "gather/scatter element type mismatch",
    )?;
    let mut full = index_shape[..index_shape.len() - 1].to_vec();
    let mut reduced = full.clone();
    for (i, &size) in base.iter().enumerate() {
        if dims.contains(&(i as i64)) {
            full.push(1);
        } else {
            full.push(size);
            reduced.push(size);
        }
    }
    ensure(
        ctx,
        op,
        slice == full || slice == reduced,
        "gather/scatter slice type mismatch",
    )?;
    let raw = op.deref(ctx);
    if raw.attributes.0.contains_key(&UNIQUE) {
        ensure(
            ctx,
            op,
            raw.attributes.get::<UnitAttr>(&UNIQUE).is_some(),
            "unique must be a unit attribute",
        )?;
    }
    if scatter {
        ensure(
            ctx,
            op,
            results[0] == base_ty && raw.attributes.get::<UnitAttr>(&UNIQUE).is_some(),
            "scatter requires matching destination/result types and unique",
        )?;
    }
    let base = base_ty.deref(ctx);
    let slice = slice_ty.deref(ctx);
    let base = base
        .downcast_ref::<pliron_mlir_builtin::types::TensorType>()
        .ok_or_else(|| fail(ctx, op, "expected tensor"))?;
    let slice = slice
        .downcast_ref::<pliron_mlir_builtin::types::TensorType>()
        .ok_or_else(|| fail(ctx, op, "expected tensor"))?;
    ensure(
        ctx,
        op,
        base.encoding == slice.encoding,
        "gather/scatter must preserve tensor encoding",
    )
}
impl Verify for GatherOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        gather_scatter(ctx, self.get_operation(), false)
    }
}
impl Verify for ScatterOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        gather_scatter(ctx, self.get_operation(), true)
    }
}
