//! Attention, exponential reduction, and prescriptively tiled custom operations.

use super::BodyBuilder;
use crate::support::*;
use pliron::{
    builtin::attributes::{DictAttr, OperandSegmentSizesAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{Type, TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use pliron_linalg::attributes::{I64ArrayAttr, IndexingMapsAttr, IteratorType, IteratorTypesAttr};
use pliron_mlir_builtin::{attributes::AffineMap, types::TensorType};

/// Verifies attention operands, indexing maps, and the score region.
///
/// Mirrors upstream `AttentionOp::verify` / `OnlineAttentionOp::verify`:
/// every non-scale operand's map has one bare-dimension result per tensor
/// dimension, static extents agree per iteration dimension, and every map
/// has the same number of inputs (dims + symbols) as the query map.
fn verify_attention(
    ctx: &Context,
    op: Ptr<Operation>,
    maps: Option<&IndexingMapsAttr>,
    outs: usize,
) -> Result<()> {
    let count = num_operands(ctx, op);
    let fixed = 4 + outs;
    ensure(
        ctx,
        op,
        count == fixed || count == fixed + 1,
        "unexpected attention operand count",
    )?;
    verify_outputs(ctx, op, count - outs..count)?;
    let types = operand_types(ctx, op);
    ensure(
        ctx,
        op,
        shaped::is_float(ctx, types[3]),
        "expected scale to be of floating point type",
    )?;
    let maps = maps.ok_or_else(|| fail(ctx, op, "missing indexing maps"))?;
    ensure(
        ctx,
        op,
        maps.maps.len() == count,
        "expected an indexing map for each operand",
    )?;
    for map in &maps.maps {
        map.validate()?;
    }
    let query_map = &maps.maps[0];
    let domain = query_map.dimensions;
    let inputs = query_map.dimensions + query_map.symbols;
    for (i, map) in maps.maps.iter().enumerate() {
        ensure(
            ctx,
            op,
            map.dimensions + map.symbols == inputs,
            "mismatched map domain",
        )?;
        if i == 3 {
            continue;
        }
        ensure(
            ctx,
            op,
            is_projected_permutation(map),
            "failed to verify op's indexing maps",
        )?;
    }
    let mut shape: Vec<Option<i64>> = vec![None; domain];
    for (i, (map, &ty)) in maps.maps.iter().zip(&types).enumerate() {
        if i == 3 {
            continue;
        }
        let (dims, _) = ranked_operand(ctx, op, ty)?;
        ensure(
            ctx,
            op,
            map.results.len() == dims.len(),
            "rank mismatch between operand and indexing map",
        )?;
        for (expr, &extent) in map.results.iter().zip(&dims) {
            let pos = as_dim(expr)
                .filter(|&d| d < domain)
                .ok_or_else(|| fail(ctx, op, "failed to verify op's indexing maps"))?;
            if extent < 0 {
                continue;
            }
            let known = *shape[pos].get_or_insert(extent);
            ensure(ctx, op, known == extent, "shape mismatch between operands")?;
        }
    }
    let block = body(ctx, op)?;
    let args = block_arg_types(ctx, block);
    ensure(
        ctx,
        op,
        args.len() == 1,
        "expects single block argument for score",
    )?;
    ensure(
        ctx,
        op,
        shaped::is_float(ctx, args[0]),
        "block argument 0 should be float",
    )?;
    let yielded = yielded(ctx, op, block)?;
    ensure(ctx, op, yielded.len() == 1, "expected only one return")
}

macro_rules! attention_op {
    (
        $struct:ident, $name:literal,
        $maps:ident, $get_maps:ident, $set_maps:ident,
        $config:ident, $get_config:ident, $set_config:ident,
        $doc:literal, [$($out:ident),+]
    ) => {
        #[doc = $doc]
        ///
        /// Operands are `query`, `key`, `value`, `scale` (a float scalar), an
        /// optional `mask`, then the outs. `indexing_maps` holds one map per
        /// operand over a shared iteration domain; the scale map has no
        /// results. The region receives the score and yields the (possibly
        /// modified) score. `decomposition_config` optionally carries
        /// lowering hints. Only tensor semantics are modeled.
        #[pliron_op(
            name = $name,
            format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig"
        )]
        pub struct $struct;
        pliron_arith::shared_attr!($struct, super::$maps, IndexingMapsAttr, $get_maps, $set_maps);
        pliron_arith::shared_attr!($struct, super::$config, DictAttr, $get_config, $set_config);

        impl $struct {
            /// Allocates the operation with an empty score block of `scale`'s type.
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                ctx: &mut Context,
                query: Value,
                key: Value,
                value: Value,
                scale: Value,
                mask: Option<Value>,
                $($out: Value,)+
                indexing_maps: Vec<AffineMap>,
            ) -> Self {
                let outs = [$($out),+];
                let results = outs.iter().map(|v| v.get_type(ctx)).collect();
                let score = scale.get_type(ctx);
                let operands = [query, key, value, scale].into_iter().chain(mask).chain(outs).collect();
                let op = new_with_body(ctx, |op| Self { op }, results, operands, vec![score]);
                op.$set_maps(ctx, IndexingMapsAttr::new(indexing_maps));
                op
            }

            /// Builds the operation with a score modifier from `body` and verifies it.
            ///
            /// # Errors
            ///
            /// Propagates errors from `body` and verification.
            #[allow(clippy::too_many_arguments)]
            pub fn build(
                ctx: &mut Context,
                query: Value,
                key: Value,
                value: Value,
                scale: Value,
                mask: Option<Value>,
                $($out: Value,)+
                indexing_maps: Vec<AffineMap>,
                body: Box<BodyBuilder<'_>>,
            ) -> Result<Self> {
                let op = Self::new(ctx, query, key, value, scale, mask, $($out,)+ indexing_maps);
                populate(ctx, op, body)
            }

            /// Returns whether a mask operand is present.
            pub fn has_mask(&self, ctx: &Context) -> bool {
                num_operands(ctx, self.get_operation()) == 5 + [$(stringify!($out)),+].len()
            }

            /// Returns the indexing maps, or an empty list when absent.
            pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
                self.$get_maps(ctx).map(|a| a.maps.clone()).unwrap_or_default()
            }

            /// Returns the decomposition configuration, if any.
            pub fn decomposition_config(&self, ctx: &Context) -> Option<DictAttr> {
                self.$get_config(ctx).map(|a| (*a).clone())
            }

            /// Sets the decomposition configuration.
            pub fn set_decomposition_config(&self, ctx: &Context, config: DictAttr) {
                self.$set_config(ctx, config);
            }
        }

        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                let maps = self.$get_maps(ctx);
                verify_attention(ctx, self.get_operation(), maps.as_deref(), [$(stringify!($out)),+].len())
            }
        }
    };
}

attention_op!(
    AttentionOp,
    "iree_linalg_ext.attention",
    ATTENTION_MAPS,
    get_attr_iree_attention_indexing_maps,
    set_attr_iree_attention_indexing_maps,
    ATTENTION_CONFIG,
    get_attr_iree_attention_decomposition_config,
    set_attr_iree_attention_decomposition_config,
    "Fused scaled dot-product attention `softmax(scale * Q * K^T + mask) * V`.",
    [output]
);

attention_op!(
    OnlineAttentionOp,
    "iree_linalg_ext.online_attention",
    ONLINE_ATTENTION_MAPS,
    get_attr_iree_online_attention_indexing_maps,
    set_attr_iree_online_attention_indexing_maps,
    ONLINE_ATTENTION_CONFIG,
    get_attr_iree_online_attention_decomposition_config,
    set_attr_iree_online_attention_decomposition_config,
    "Attention with a running maximum and sum, for tiled (flash) attention.",
    [output, max, sum]
);

/// Checks one map per operand, each over `loops` dimensions with one result
/// per operand dimension.
fn verify_maps(
    ctx: &Context,
    op: Ptr<Operation>,
    maps: &[AffineMap],
    ranks: &[usize],
    loops: usize,
) -> Result<()> {
    ensure(
        ctx,
        op,
        maps.len() == ranks.len(),
        "expected one indexing map per operand",
    )?;
    for (map, &rank) in maps.iter().zip(ranks) {
        map.validate()?;
        ensure(
            ctx,
            op,
            map.dimensions == loops,
            "indexing map dimensions must match the iterator count",
        )?;
        ensure(
            ctx,
            op,
            map.results.len() == rank,
            "indexing map results must match the operand rank",
        )?;
    }
    Ok(())
}

/// Element types of `values`, used as linalg-style block arguments.
fn elements(ctx: &Context, values: &[Value]) -> Vec<TypeHandle> {
    values.iter().map(|&v| element(ctx, v)).collect()
}

/// A `linalg.generic`-like reduction over `e^x` that keeps a running maximum
/// in the first `outs` operand for numerically stable online normalization.
///
/// The `outs` listed in `exp_reduced_operands` (never 0, which holds the
/// maximum) are rescaled by `2^(new_max - previous_max)` before the body
/// runs. The region receives one element per operand and yields one value
/// per `outs` operand.
#[pliron_op(
    name = "iree_linalg_ext.exp_reduction",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (
        iree_exp_reduction_indexing_maps: IndexingMapsAttr,
        iree_exp_reduction_iterator_types: IteratorTypesAttr,
        iree_exp_reduction_exp_reduced_operands: I64ArrayAttr,
        iree_exp_reduction_operand_segment_sizes: OperandSegmentSizesAttr
    )
)]
pub struct ExpReductionOp;

impl ExpReductionOp {
    /// Allocates the operation with an empty body taking one element per operand.
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        indexing_maps: Vec<AffineMap>,
        iterator_types: Vec<IteratorType>,
        exp_reduced_operands: Vec<i64>,
    ) -> Self {
        let results = outputs.iter().map(|v| v.get_type(ctx)).collect();
        let sizes = segment_sizes(&[inputs.len(), outputs.len()]);
        let operands: Vec<Value> = inputs.into_iter().chain(outputs).collect();
        let args = elements(ctx, &operands);
        let op = new_with_body(ctx, |op| Self { op }, results, operands, args);
        op.set_attr_iree_exp_reduction_indexing_maps(ctx, IndexingMapsAttr::new(indexing_maps));
        op.set_attr_iree_exp_reduction_iterator_types(ctx, IteratorTypesAttr::new(iterator_types));
        op.set_attr_iree_exp_reduction_exp_reduced_operands(
            ctx,
            I64ArrayAttr::new(exp_reduced_operands),
        );
        op.set_attr_iree_exp_reduction_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Builds the operation with a body from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        indexing_maps: Vec<AffineMap>,
        iterator_types: Vec<IteratorType>,
        exp_reduced_operands: Vec<i64>,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            inputs,
            outputs,
            indexing_maps,
            iterator_types,
            exp_reduced_operands,
        );
        populate(ctx, op, body)
    }

    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_iree_exp_reduction_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }

    /// Returns the iterator types, or an empty list when absent.
    pub fn iterator_types(&self, ctx: &Context) -> Vec<IteratorType> {
        self.get_attr_iree_exp_reduction_iterator_types(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }

    /// Returns the `outs` indices rescaled by the running-maximum norm.
    pub fn exp_reduced_operands(&self, ctx: &Context) -> Vec<i64> {
        self.get_attr_iree_exp_reduction_exp_reduced_operands(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }
}

impl Verify for ExpReductionOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let sizes = self.get_attr_iree_exp_reduction_operand_segment_sizes(ctx);
        let [(_, inputs), (_, outputs)] = segments::<2>(ctx, op, sizes.as_deref())?;
        verify_outputs(ctx, op, inputs..inputs + outputs)?;
        let types = operand_types(ctx, op);
        let ranks = types
            .iter()
            .map(|&t| ranked_operand(ctx, op, t).map(|(d, _)| d.len()))
            .collect::<Result<Vec<_>>>()?;
        let reduced = self
            .get_attr_iree_exp_reduction_exp_reduced_operands(ctx)
            .ok_or_else(|| fail(ctx, op, "missing exp_reduced_operands"))?
            .values
            .clone();
        for r in reduced {
            ensure(
                ctx,
                op,
                r != 0,
                "operand index in exp_reduced_operands cannot be the 0th operand, it always contains the maximum value",
            )?;
            ensure(
                ctx,
                op,
                r > 0 && (r as usize) < outputs,
                "operand index in exp_reduced_operands must index the outs operands",
            )?;
        }
        let loops = self
            .get_attr_iree_exp_reduction_iterator_types(ctx)
            .ok_or_else(|| fail(ctx, op, "missing iterator types"))?
            .values
            .len();
        let maps = self
            .get_attr_iree_exp_reduction_indexing_maps(ctx)
            .ok_or_else(|| fail(ctx, op, "missing indexing maps"))?
            .maps
            .clone();
        verify_maps(ctx, op, &maps, &ranks, loops)?;
        ensure(
            ctx,
            op,
            maps.iter().all(is_projected_permutation),
            "all indexing maps must be projected permutations",
        )?;
        let block = body(ctx, op)?;
        let has_linalg_index = block.deref(ctx).iter(ctx).any(|o| {
            Operation::get_op_dyn(o, ctx)
                .downcast_ref::<pliron_linalg::ops::IndexOp>()
                .is_some()
        });
        ensure(
            ctx,
            op,
            !has_linalg_index,
            "linalg.index is not supported in body",
        )?;
        yielded(ctx, op, block).map(drop)
    }
}

/// Returns `ty` as a ranked tensor with every extent dynamic, keeping its
/// element and encoding, or `ty` unchanged for a scalar.
fn dynamic_slice_type(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    let tensor = ty.deref(ctx).downcast_ref::<TensorType>().cloned();
    match tensor {
        Some(mut t) => {
            t.dimensions = vec![-1; t.dimensions.len()];
            <TensorType as Type>::instantiate(t, ctx).into()
        }
        None => ty,
    }
}

/// Prescriptively tiled computation with an explicit inter-tile loop nest.
///
/// `indexing_maps` give, per operand, the slice accessed by each loop
/// iteration; symbols denote whole (untiled) dimensions and an empty map
/// leaves the operand unindexed. The region receives one argument per
/// operand: the scalar itself, or a fully dynamic tensor slice of the same
/// rank and element. It yields one value per `outs` operand, typed as that
/// operand's block argument.
#[pliron_op(
    name = "iree_linalg_ext.custom_op",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (
        iree_custom_op_indexing_maps: IndexingMapsAttr,
        iree_custom_op_iterator_types: IteratorTypesAttr,
        iree_custom_op_operand_segment_sizes: OperandSegmentSizesAttr
    )
)]
pub struct CustomOp;

impl CustomOp {
    /// Allocates the operation with an empty body whose arguments are derived from the operands.
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        indexing_maps: Vec<AffineMap>,
        iterator_types: Vec<IteratorType>,
    ) -> Self {
        let results = outputs.iter().map(|v| v.get_type(ctx)).collect();
        let sizes = segment_sizes(&[inputs.len(), outputs.len()]);
        let operands: Vec<Value> = inputs.into_iter().chain(outputs).collect();
        let args = operands
            .iter()
            .map(|v| {
                let ty = v.get_type(ctx);
                dynamic_slice_type(ctx, ty)
            })
            .collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, args);
        op.set_attr_iree_custom_op_indexing_maps(ctx, IndexingMapsAttr::new(indexing_maps));
        op.set_attr_iree_custom_op_iterator_types(ctx, IteratorTypesAttr::new(iterator_types));
        op.set_attr_iree_custom_op_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Builds the operation with a body from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    pub fn build(
        ctx: &mut Context,
        inputs: Vec<Value>,
        outputs: Vec<Value>,
        indexing_maps: Vec<AffineMap>,
        iterator_types: Vec<IteratorType>,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, inputs, outputs, indexing_maps, iterator_types);
        populate(ctx, op, body)
    }

    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_iree_custom_op_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }

    /// Returns the iterator types (one per loop), or an empty list when absent.
    pub fn iterator_types(&self, ctx: &Context) -> Vec<IteratorType> {
        self.get_attr_iree_custom_op_iterator_types(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }
}

impl Verify for CustomOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let sizes = self.get_attr_iree_custom_op_operand_segment_sizes(ctx);
        let [(_, inputs), (_, outputs)] = segments::<2>(ctx, op, sizes.as_deref())?;
        verify_outputs(ctx, op, inputs..inputs + outputs)?;
        let types = operand_types(ctx, op);
        let ranks = types
            .iter()
            .map(|&t| {
                ranked_or_scalar(ctx, t)
                    .map(|(d, _)| d.len())
                    .ok_or_else(|| fail(ctx, op, "operands must be ranked tensors or scalars"))
            })
            .collect::<Result<Vec<_>>>()?;
        let loops = self
            .get_attr_iree_custom_op_iterator_types(ctx)
            .ok_or_else(|| fail(ctx, op, "missing iterator types"))?
            .values
            .len();
        let maps = self
            .get_attr_iree_custom_op_indexing_maps(ctx)
            .ok_or_else(|| fail(ctx, op, "missing indexing maps"))?
            .maps
            .clone();
        ensure(
            ctx,
            op,
            maps.len() == types.len(),
            "expected number of indexing maps to be same as the number of input/output operands",
        )?;
        let mut symbols = None;
        for (map, &rank) in maps.iter().zip(&ranks) {
            if map.dimensions == 0 && map.symbols == 0 && map.results.is_empty() {
                continue;
            }
            map.validate()?;
            ensure(
                ctx,
                op,
                map.dimensions == loops,
                "indexing map must have one dimension per loop",
            )?;
            let expected = *symbols.get_or_insert(map.symbols);
            ensure(
                ctx,
                op,
                map.symbols == expected,
                "inconsistent number of symbol dimensions in indexing maps",
            )?;
            ensure(
                ctx,
                op,
                map.results.len() == rank,
                "operand rank must match the indexing map result count",
            )?;
        }
        let block = body(ctx, op)?;
        let args = block_arg_types(ctx, block);
        ensure(
            ctx,
            op,
            args.len() == types.len(),
            "expected as many basic block arguments as the number of operands",
        )?;
        for (&arg, &ty) in args.iter().zip(&types) {
            if is_plain_scalar(ctx, ty) {
                ensure(
                    ctx,
                    op,
                    arg == ty,
                    "scalar operand requires a block argument of the same type",
                )?;
            } else {
                ensure(
                    ctx,
                    op,
                    arg == dynamic_slice_type(ctx, ty),
                    "tensor operand requires a fully dynamic tensor block argument of the same rank and element",
                )?;
            }
        }
        let yielded = yielded(ctx, op, block)?;
        ensure(
            ctx,
            op,
            yielded == args[inputs..],
            "expected one yield per outs operand, typed as its block argument",
        )
    }
}
