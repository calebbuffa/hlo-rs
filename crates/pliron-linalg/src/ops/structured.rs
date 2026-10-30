//! Contraction and elementwise structured operations defined in MLIR's
//! `LinalgStructuredOps.td`: the `matmul` family, `contract`,
//! `scaled_contract`, and `elementwise`.
//!
//! These operations carry no body region; the scalar computation is implied
//! by the operation and its attributes, as in MLIR's custom form.

use crate::{
    attributes::{
        ElementwiseKind, ElementwiseKindAttr, IndexingMapsAttr, IteratorType, TypeFn, TypeFnAttr,
        identity_map, projection_map,
    },
    support::*,
};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
};
use pliron_arith::shaped;
use stablehlo::attributes::{AffineExpr, AffineMap};

fn allocate<T: Op>(ctx: &mut Context, make: fn(Ptr<Operation>) -> T, operands: Vec<Value>) -> T {
    let result = operands.last().map(|v| v.get_type(ctx));
    let results = result.into_iter().collect();
    make(Operation::new(
        ctx,
        T::get_concrete_op_info(),
        results,
        operands,
        vec![],
        0,
    ))
}

fn operand(ctx: &Context, op: Ptr<Operation>, i: usize) -> Value {
    op.deref(ctx).get_operand(i)
}

/// Which member of the matmul family an operation is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MatmulKind {
    Matmul,
    Batch,
    BatchReduce,
}

impl MatmulKind {
    const fn loops(self) -> usize {
        match self {
            Self::Matmul => 3,
            Self::Batch | Self::BatchReduce => 4,
        }
    }

    const fn iterators(self) -> &'static [IteratorType] {
        use IteratorType::{Parallel as P, Reduction as R};
        match self {
            Self::Matmul => &[P, P, R],
            Self::Batch => &[P, P, P, R],
            Self::BatchReduce => &[R, P, P, R],
        }
    }

    fn defaults(self) -> Vec<AffineMap> {
        let n = self.loops();
        match self {
            Self::Matmul => vec![
                projection_map(n, &[0, 2]),
                projection_map(n, &[2, 1]),
                projection_map(n, &[0, 1]),
            ],
            Self::Batch => vec![
                projection_map(n, &[0, 1, 3]),
                projection_map(n, &[0, 3, 2]),
                projection_map(n, &[0, 1, 2]),
            ],
            Self::BatchReduce => vec![
                projection_map(n, &[0, 1, 3]),
                projection_map(n, &[0, 3, 2]),
                projection_map(n, &[1, 2]),
            ],
        }
    }

    /// Checks user-supplied maps against MLIR's extended matmul semantics.
    fn check(self, maps: &[AffineMap]) -> std::result::Result<(), &'static str> {
        let defaults = self.defaults();
        if maps.len() != 3 {
            return Err("expected three indexing maps");
        }
        if maps
            .iter()
            .any(|m| m.dimensions != self.loops() || m.symbols != 0)
        {
            return Err("indexing maps must take one dimension per loop and no symbols");
        }
        for (i, (map, default)) in maps[..2].iter().zip(&defaults).enumerate() {
            if !map.results.iter().all(|e| default.results.contains(e)) {
                return Err("unexpected dimension expression in an input map");
            }
            let dims: Vec<Option<usize>> = map.results.iter().map(as_dim).collect();
            let full = default.results.len();
            match self {
                Self::Matmul => {
                    if map.results.len() < full && !(map.results.len() == 1 && dims[0] == Some(2)) {
                        return Err(
                            "a broadcast matmul input must index only the reduction dimension",
                        );
                    }
                }
                Self::Batch | Self::BatchReduce => match dims.as_slice() {
                    [Some(0), _, _] => {}
                    [_, _, _] => {
                        return Err("a batch matmul input must index the batch dimension first");
                    }
                    [Some(3)] => {}
                    [Some(0 | 1), Some(3)] if i == 0 => {}
                    [Some(0), Some(3)] | [Some(3), Some(2)] if i == 1 => {}
                    _ => return Err("invalid broadcast map for a batch matmul input"),
                },
            }
        }
        if self != Self::Matmul && maps[2] != defaults[2] {
            return Err("unexpected output map for a batch matmul");
        }
        Ok(())
    }
}

fn verify_matmul(
    ctx: &Context,
    op: Ptr<Operation>,
    kind: MatmulKind,
    maps: Option<Vec<AffineMap>>,
) -> Result<()> {
    let operands = verify_dps(ctx, op, 2)?;
    if operands.iter().any(|(_, e)| !numeric(ctx, *e)) {
        return Err(fail(ctx, op, "operand elements must be numeric"));
    }
    let maps = match maps {
        Some(maps) => {
            kind.check(&maps).map_err(|reason| fail(ctx, op, reason))?;
            maps
        }
        None => kind.defaults(),
    };
    let shapes: Vec<Vec<i64>> = operands.into_iter().map(|(s, _)| s).collect();
    verify_indexing(ctx, op, &maps, &shapes, kind.loops())?;
    Ok(())
}

macro_rules! matmul_op {
    ($struct:ident, $name:literal, $kind:expr, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Operands are `lhs`, `rhs` (`ins`) and the accumulator (`outs`).
        /// Inputs are converted with `cast` (default [`TypeFn::CastSigned`])
        /// to the accumulator element type before multiplying. Optional
        /// `indexing_maps` express transposed or broadcast inputs.
        #[pliron_op(
                            name = $name,
                            format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        crate::pliron_arith::shared_attr!(
            $struct,
            super::ATTR_KEY_LINALG_INDEXING_MAPS,
            IndexingMapsAttr,
            get_attr_linalg_indexing_maps,
            set_attr_linalg_indexing_maps
        );
        crate::pliron_arith::shared_attr!(
            $struct,
            super::ATTR_KEY_LINALG_CAST,
            TypeFnAttr,
            get_attr_linalg_cast,
            set_attr_linalg_cast
        );

        impl $struct {
            /// Allocates the operation with default maps; the result type is `init`'s type.
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, init: Value) -> Self {
                allocate(ctx, |op| Self { op }, vec![lhs, rhs, init])
            }
            /// Returns the left operand.
            pub fn lhs(&self, ctx: &Context) -> Value {
                operand(ctx, self.get_operation(), 0)
            }
            /// Returns the right operand.
            pub fn rhs(&self, ctx: &Context) -> Value {
                operand(ctx, self.get_operation(), 1)
            }
            /// Returns the accumulator (`outs`) operand.
            pub fn init(&self, ctx: &Context) -> Value {
                operand(ctx, self.get_operation(), 2)
            }
            /// Sets explicit indexing maps (`lhs`, `rhs`, `init`).
            pub fn set_indexing_maps(&self, ctx: &Context, maps: Vec<AffineMap>) {
                self.set_attr_linalg_indexing_maps(ctx, IndexingMapsAttr::new(maps));
            }
            /// Returns the explicit maps, or the defaults when absent.
            pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
                self.get_attr_linalg_indexing_maps(ctx)
                    .map_or_else(|| $kind.defaults(), |a| a.maps.clone())
            }
            /// Returns whether the maps differ from the defaults.
            pub fn has_user_maps(&self, ctx: &Context) -> bool {
                self.get_attr_linalg_indexing_maps(ctx)
                    .is_some_and(|a| a.maps != $kind.defaults())
            }
            /// Returns the default maps of this operation.
            pub fn default_indexing_maps() -> Vec<AffineMap> {
                $kind.defaults()
            }
            /// Returns the iterator types.
            pub fn iterator_types() -> Vec<IteratorType> {
                $kind.iterators().to_vec()
            }
            /// Sets the input conversion.
            pub fn set_cast(&self, ctx: &Context, cast: TypeFn) {
                self.set_attr_linalg_cast(ctx, TypeFnAttr::new(cast));
            }
            /// Returns the input conversion.
            pub fn cast(&self, ctx: &Context) -> TypeFn {
                self.get_attr_linalg_cast(ctx)
                    .map_or(TypeFn::CastSigned, |a| a.value)
            }
        }

        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                let maps = self
                    .get_attr_linalg_indexing_maps(ctx)
                    .map(|a| a.maps.clone());
                verify_matmul(ctx, self.get_operation(), $kind, maps)
            }
        }
    };
}

matmul_op!(
    MatmulOp,
    "linalg.matmul",
    MatmulKind::Matmul,
    "Matrix multiplication `C[m, n] += A[m, k] * B[k, n]`."
);
matmul_op!(
    BatchMatmulOp,
    "linalg.batch_matmul",
    MatmulKind::Batch,
    "Batched matrix multiplication `C[b, m, n] += A[b, m, k] * B[b, k, n]`."
);
matmul_op!(
    BatchReduceMatmulOp,
    "linalg.batch_reduce_matmul",
    MatmulKind::BatchReduce,
    "Batch-reduced matrix multiplication `C[m, n] += sum_b A[b, m, k] * B[b, k, n]`."
);

/// Counts, per loop dimension, its uses in the input maps and the output map.
fn occurrences(loops: usize, inputs: &[&AffineMap], output: &AffineMap) -> (Vec<u32>, Vec<u32>) {
    let mut ins = vec![0u32; loops];
    let mut outs = vec![0u32; loops];
    for map in inputs {
        for d in map.results.iter().flat_map(dims_of) {
            ins[d] += 1;
        }
    }
    for d in output.results.iter().flat_map(dims_of) {
        outs[d] += 1;
    }
    (ins, outs)
}

/// Checks MLIR's contraction-dimension rules over the `lhs`, `rhs`, and output maps.
fn check_contraction(
    loops: usize,
    lhs: &AffineMap,
    rhs: &AffineMap,
    out: &AffineMap,
) -> std::result::Result<(), &'static str> {
    let (ins, outs) = occurrences(loops, &[lhs, rhs], out);
    let mut contracting = false;
    for (i, o) in ins.into_iter().zip(outs) {
        match (i, o) {
            (0, 0) => return Err("every loop dimension must index an operand"),
            (1, o) if o != 1 => {
                return Err("a dimension indexing one input must index the output exactly once");
            }
            (2, 0) => contracting = true,
            _ => {}
        }
    }
    if contracting {
        Ok(())
    } else {
        Err("expected at least one contracting dimension")
    }
}

fn loops_of(ctx: &Context, op: Ptr<Operation>, maps: &[AffineMap]) -> Result<usize> {
    let loops = maps
        .first()
        .ok_or_else(|| fail(ctx, op, "missing indexing maps"))?
        .dimensions;
    if maps.iter().any(|m| m.dimensions != loops || m.symbols != 0) {
        return Err(fail(
            ctx,
            op,
            "indexing maps must share their dimensions and take no symbols",
        ));
    }
    Ok(loops)
}

/// General tensor contraction, as described by its required indexing maps.
///
/// Operands are `lhs`, `rhs` (`ins`) and the accumulator (`outs`). Each map
/// must be a projected permutation; dimensions shared by both inputs and
/// absent from the output are contracted.
#[pliron_op(
    name = "linalg.contract",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct ContractOp;
crate::pliron_arith::shared_attr!(
    ContractOp,
    super::ATTR_KEY_LINALG_INDEXING_MAPS,
    IndexingMapsAttr,
    get_attr_linalg_indexing_maps,
    set_attr_linalg_indexing_maps
);
crate::pliron_arith::shared_attr!(
    ContractOp,
    super::ATTR_KEY_LINALG_CAST,
    TypeFnAttr,
    get_attr_linalg_cast,
    set_attr_linalg_cast
);

impl ContractOp {
    /// Allocates the operation; the result type is `init`'s type.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        init: Value,
        maps: Vec<AffineMap>,
    ) -> Self {
        let op = allocate(ctx, |op| Self { op }, vec![lhs, rhs, init]);
        op.set_attr_linalg_indexing_maps(ctx, IndexingMapsAttr::new(maps));
        op
    }
    /// Returns the left operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 0)
    }
    /// Returns the right operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 1)
    }
    /// Returns the accumulator (`outs`) operand.
    pub fn init(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 2)
    }
    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_linalg_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }
    /// Sets the input conversion.
    pub fn set_cast(&self, ctx: &Context, cast: TypeFn) {
        self.set_attr_linalg_cast(ctx, TypeFnAttr::new(cast));
    }
    /// Returns the input conversion.
    pub fn cast(&self, ctx: &Context) -> TypeFn {
        self.get_attr_linalg_cast(ctx)
            .map_or(TypeFn::CastSigned, |a| a.value)
    }
}

impl Verify for ContractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let operands = verify_dps(ctx, op, 2)?;
        if operands.iter().any(|(_, e)| !numeric(ctx, *e)) {
            return Err(fail(ctx, op, "operand elements must be numeric"));
        }
        let maps = self.indexing_maps(ctx);
        if maps.len() != 3 {
            return Err(fail(ctx, op, "expected three indexing maps"));
        }
        let loops = loops_of(ctx, op, &maps)?;
        if !maps.iter().all(is_projected_permutation) {
            return Err(fail(
                ctx,
                op,
                "indexing maps must be projected permutations",
            ));
        }
        check_contraction(loops, &maps[0], &maps[1], &maps[2]).map_err(|r| fail(ctx, op, r))?;
        let shapes: Vec<Vec<i64>> = operands.into_iter().map(|(s, _)| s).collect();
        verify_indexing(ctx, op, &maps, &shapes, loops)?;
        Ok(())
    }
}

/// Returns `(dim, divisor)` for a scale-map result `dN` or `dN floordiv c`.
fn scale_term(expr: &AffineExpr) -> Option<(usize, i64)> {
    if let Some(d) = as_dim(expr) {
        return Some((d, 1));
    }
    match expr.nodes.as_slice() {
        [dim, c, op] if dim.kind == 0 && c.kind == 2 && op.kind == 5 && c.value > 0 => {
            Some((usize::try_from(dim.value).ok()?, c.value))
        }
        _ => None,
    }
}

/// Contraction of block-scaled inputs: `C += (A * scale_A) * (B * scale_B)`.
///
/// Operands are `A`, `scale_A`, `B`, `scale_B` (`ins`) and the accumulator
/// (`outs`), with five required indexing maps in that order. Scale maps are
/// projected permutations whose results may be divided by a positive block
/// size (`dN floordiv c`).
#[pliron_op(
    name = "linalg.scaled_contract",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct ScaledContractOp;
crate::pliron_arith::shared_attr!(
    ScaledContractOp,
    super::ATTR_KEY_LINALG_INDEXING_MAPS,
    IndexingMapsAttr,
    get_attr_linalg_indexing_maps,
    set_attr_linalg_indexing_maps
);

impl ScaledContractOp {
    /// Allocates the operation; the result type is `init`'s type.
    pub fn new(
        ctx: &mut Context,
        [a, scale_a, b, scale_b]: [Value; 4],
        init: Value,
        maps: Vec<AffineMap>,
    ) -> Self {
        let op = allocate(ctx, |op| Self { op }, vec![a, scale_a, b, scale_b, init]);
        op.set_attr_linalg_indexing_maps(ctx, IndexingMapsAttr::new(maps));
        op
    }
    /// Returns the `ins` operands `A`, `scale_A`, `B`, `scale_B`.
    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().take(4).collect()
    }
    /// Returns the accumulator (`outs`) operand.
    pub fn init(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 4)
    }
    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_linalg_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }
}

impl Verify for ScaledContractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let operands = verify_dps(ctx, op, 4)?;
        let out_width = match shaped::float_width(ctx, operands[4].1) {
            Some(w) => w,
            None => return Err(fail(ctx, op, "output element must be a float")),
        };
        for (_, element) in &operands[..4] {
            let ok =
                shaped::is_float(ctx, *element) || shaped::signless_width(ctx, *element).is_some();
            if !ok || shaped::scalar_width(ctx, *element).is_none_or(|w| w > out_width) {
                return Err(fail(
                    ctx,
                    op,
                    "input elements must be integers or floats no wider than the output element",
                ));
            }
        }
        let maps = self.indexing_maps(ctx);
        if maps.len() != 5 {
            return Err(fail(ctx, op, "expected five indexing maps"));
        }
        let loops = loops_of(ctx, op, &maps)?;
        for i in [0, 2, 4] {
            if !is_projected_permutation(&maps[i]) {
                return Err(fail(ctx, op, "operand maps must be projected permutations"));
            }
        }
        check_contraction(loops, &maps[0], &maps[2], &maps[4]).map_err(|r| fail(ctx, op, r))?;
        for (scale, input) in [(1, 0), (3, 2)] {
            let (scale_shape, input_shape) = (&operands[scale].0, &operands[input].0);
            if scale_shape.len() != input_shape.len() {
                return Err(fail(ctx, op, "scale rank must equal its input rank"));
            }
            let mut seen = vec![false; loops];
            for (k, expr) in maps[scale].results.iter().enumerate() {
                let (d, block) = scale_term(expr).ok_or_else(|| {
                    fail(ctx, op, "scale map results must be dN or dN floordiv c")
                })?;
                if d >= loops || std::mem::replace(&mut seen[d], true) {
                    return Err(fail(
                        ctx,
                        op,
                        "scale map dimensions must be distinct loop dimensions",
                    ));
                }
                let position = maps[input]
                    .results
                    .iter()
                    .position(|e| as_dim(e) == Some(d))
                    .ok_or_else(|| {
                        fail(ctx, op, "scale map dimensions must index the scaled input")
                    })?;
                let (extent, scales) = (input_shape[position], scale_shape[k]);
                if extent >= 0 && scales >= 0 && (extent + block - 1) / block != scales {
                    return Err(fail(
                        ctx,
                        op,
                        "scale extent must be the input extent divided by the block size",
                    ));
                }
            }
        }
        for i in [0, 2, 4] {
            if maps[i].results.len() != operands[i].0.len() {
                return Err(fail(
                    ctx,
                    op,
                    "indexing map result count must equal the operand rank",
                ));
            }
        }
        let picked = [maps[0].clone(), maps[2].clone(), maps[4].clone()];
        let shapes = [
            operands[0].0.clone(),
            operands[2].0.clone(),
            operands[4].0.clone(),
        ];
        verify_indexing(ctx, op, &picked, &shapes, loops)?;
        Ok(())
    }
}

/// Elementwise computation selected by `kind` over the output's iteration space.
///
/// Operands are one `ins` operand per arity of `kind` and one `outs` tensor.
/// Optional `indexing_maps` default to the identity over the output rank.
#[pliron_op(
    name = "linalg.elementwise",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (linalg_kind: ElementwiseKindAttr)
)]
pub struct ElementwiseOp;
crate::pliron_arith::shared_attr!(
    ElementwiseOp,
    super::ATTR_KEY_LINALG_INDEXING_MAPS,
    IndexingMapsAttr,
    get_attr_linalg_indexing_maps,
    set_attr_linalg_indexing_maps
);

impl ElementwiseOp {
    /// Allocates the operation; the result type is `init`'s type.
    pub fn new(ctx: &mut Context, kind: ElementwiseKind, inputs: Vec<Value>, init: Value) -> Self {
        let operands = inputs.into_iter().chain([init]).collect();
        let op = allocate(ctx, |op| Self { op }, operands);
        op.set_attr_linalg_kind(ctx, ElementwiseKindAttr::new(kind));
        op
    }
    /// Returns the computation, or `None` when the attribute is absent.
    pub fn kind(&self, ctx: &Context) -> Option<ElementwiseKind> {
        self.get_attr_linalg_kind(ctx).map(|a| a.value)
    }
    /// Returns the `ins` operands.
    pub fn inputs(&self, ctx: &Context) -> Vec<Value> {
        let ir = self.get_operation().deref(ctx);
        let n = ir.get_num_operands().saturating_sub(1);
        ir.operands().take(n).collect()
    }
    /// Returns the `outs` operand.
    pub fn init(&self, ctx: &Context) -> Option<Value> {
        self.get_operation().deref(ctx).operands().last()
    }
    /// Sets explicit indexing maps, one per operand.
    pub fn set_indexing_maps(&self, ctx: &Context, maps: Vec<AffineMap>) {
        self.set_attr_linalg_indexing_maps(ctx, IndexingMapsAttr::new(maps));
    }
    /// Returns the default maps: the identity over the output rank, per operand.
    pub fn default_indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        let rank = self
            .init(ctx)
            .and_then(|v| ranked(ctx, v.get_type(ctx)))
            .map_or(0, |(dims, _)| dims.len());
        let operands = self.get_operation().deref(ctx).get_num_operands();
        vec![identity_map(rank); operands]
    }
    /// Returns the explicit maps, or the defaults when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_linalg_indexing_maps(ctx)
            .map_or_else(|| self.default_indexing_maps(ctx), |a| a.maps.clone())
    }
    /// Returns whether the maps differ from the defaults.
    pub fn has_user_maps(&self, ctx: &Context) -> bool {
        self.get_attr_linalg_indexing_maps(ctx)
            .is_some_and(|a| a.maps != self.default_indexing_maps(ctx))
    }
}

impl Verify for ElementwiseOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let kind = self
            .kind(ctx)
            .ok_or_else(|| fail(ctx, op, "missing elementwise kind"))?;
        let inputs = kind.arity().inputs();
        let operands = verify_dps(ctx, op, inputs)?;
        if operands.iter().any(|(_, e)| !numeric(ctx, *e)) {
            return Err(fail(ctx, op, "operand elements must be numeric"));
        }
        let rank = operands[inputs].0.len();
        let shapes: Vec<Vec<i64>> = operands.into_iter().map(|(s, _)| s).collect();
        verify_indexing(ctx, op, &self.indexing_maps(ctx), &shapes, rank)?;
        Ok(())
    }
}
