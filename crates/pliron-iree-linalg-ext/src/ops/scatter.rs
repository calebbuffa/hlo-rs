//! Indexed and remapped data movement: scatter, gather, map_load, map_store.

use super::BodyBuilder;
use crate::support::*;
use pliron::{
    builtin::attributes::BoolAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use pliron_mlir_builtin::attributes::I64ArrayAttr;

/// Verifies the shared contract of scatter and gather.
///
/// `original` is the indexed tensor (scatter destination, gather source) and
/// `update` is the batched slice tensor (scatter updates, gather output).
fn verify_gather_scatter(
    ctx: &Context,
    op: Ptr<Operation>,
    original: &[i64],
    update: &[i64],
    indices: (Vec<i64>, TypeHandle),
    mask: Option<(Vec<i64>, TypeHandle)>,
    dimension_map: &[i64],
) -> Result<()> {
    let (indices, index_element) = indices;
    ensure(
        ctx,
        op,
        !indices.is_empty() && is_int_or_index(ctx, index_element),
        "expected indices to be of rank 1 or greater and of integer or index element type",
    )?;
    ensure(
        ctx,
        op,
        is_permutation(dimension_map),
        "dimension map is invalid",
    )?;
    ensure(
        ctx,
        op,
        !dimension_map.is_empty(),
        "dimension map must have at least one element",
    )?;
    let index_depth = dimension_map.len();
    ensure(
        ctx,
        op,
        original.len() >= index_depth,
        "expected original rank to be greater or equal to index depth",
    )?;
    let slice_rank = original.len() - index_depth;
    ensure(
        ctx,
        op,
        update.len() >= slice_rank,
        "expected update to be at least the rank of non indexed original dims",
    )?;
    let batch_rank = update.len() - slice_rank;
    ensure(
        ctx,
        op,
        (indices.len() == batch_rank && index_depth == 1) || indices.len() == batch_rank + 1,
        "expected indices to be equal to batch rank or batch rank + 1",
    )?;
    ensure(
        ctx,
        op,
        indices
            .iter()
            .take(batch_rank)
            .eq(update.iter().take(batch_rank)),
        "mismatch in shape of indices and update value",
    )?;
    if batch_rank + 1 < indices.len() {
        ensure(
            ctx,
            op,
            indices.last().copied() == Some(index_depth as i64),
            "size of dimension map must match the last dimension of indices",
        )?;
    }
    if let Some((mask, mask_element)) = mask {
        ensure(
            ctx,
            op,
            matches!(int_width(ctx, mask_element), Some(1 | 8)),
            "expected mask to have i1 or storage-legalized i8 element type",
        )?;
        ensure(
            ctx,
            op,
            mask.len() == batch_rank,
            "expected mask rank to match batch rank",
        )?;
        ensure(
            ctx,
            op,
            mask.iter().zip(update).all(|(&m, &u)| compatible(m, u)),
            "mask shape must match batch dimensions",
        )?;
    }
    let slices_match =
        (0..slice_rank).all(|i| compatible(original[i + index_depth], update[i + batch_rank]));
    ensure(
        ctx,
        op,
        slices_match,
        "shape of update value slice must match original value",
    )
}

/// Splits `[a, b, (mask), out]` operands into their parts.
fn split(ctx: &Context, op: Ptr<Operation>) -> Result<(Vec<TypeHandle>, bool)> {
    let count = num_operands(ctx, op);
    ensure(
        ctx,
        op,
        (3..=4).contains(&count),
        "expected two or three ins operands and one outs operand",
    )?;
    verify_outputs(ctx, op, count - 1..count)?;
    Ok((operand_types(ctx, op), count == 4))
}

fn dimension_map(
    ctx: &Context,
    op: Ptr<Operation>,
    attr: Option<impl std::ops::Deref<Target = I64ArrayAttr>>,
) -> Result<Vec<i64>> {
    attr.map(|a| a.values.clone())
        .ok_or_else(|| fail(ctx, op, "missing dimension map"))
}

/// Scatters `updates` slices into `original` at `indices`, combining with a region.
///
/// Operands are `updates`, `indices`, optional `mask` (ins), then `original`
/// (outs). `dimension_map` lists which `original` dimensions each index
/// coordinate addresses. The region receives `(update, original)` elements
/// and yields the stored element.
#[pliron_op(
    name = "iree_linalg_ext.scatter",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",
    attributes = (iree_scatter_dimension_map: I64ArrayAttr, iree_scatter_unique_indices: BoolAttr)
)]
pub struct ScatterOp;

impl ScatterOp {
    /// Allocates the operation with an empty combiner block.
    pub fn new(
        ctx: &mut Context,
        updates: Value,
        indices: Value,
        mask: Option<Value>,
        original: Value,
        dimension_map: Vec<i64>,
        unique_indices: bool,
    ) -> Self {
        let results = vec![original.get_type(ctx)];
        let args = vec![element(ctx, updates), element(ctx, original)];
        let operands = [updates, indices]
            .into_iter()
            .chain(mask)
            .chain([original])
            .collect();
        let op = new_with_body(ctx, |op| Self { op }, results, operands, args);
        op.set_attr_iree_scatter_dimension_map(ctx, I64ArrayAttr::new(dimension_map));
        op.set_attr_iree_scatter_unique_indices(ctx, BoolAttr::new(unique_indices));
        op
    }

    /// Builds the operation with a combiner from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        updates: Value,
        indices: Value,
        mask: Option<Value>,
        original: Value,
        dimension_map: Vec<i64>,
        unique_indices: bool,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            updates,
            indices,
            mask,
            original,
            dimension_map,
            unique_indices,
        );
        populate(ctx, op, body)
    }

    /// Returns the dimension map, or an empty list when absent.
    pub fn dimension_map(&self, ctx: &Context) -> Vec<i64> {
        self.get_attr_iree_scatter_dimension_map(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }

    /// Returns whether indices are unique; upstream defaults to `true`.
    pub fn unique_indices(&self, ctx: &Context) -> bool {
        self.get_attr_iree_scatter_unique_indices(ctx)
            .is_none_or(|a| bool::from((*a).clone()))
    }
}

impl Verify for ScatterOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let (types, has_mask) = split(ctx, op)?;
        let (update, update_element) = ranked_operand(ctx, op, types[0])?;
        let indices = ranked_operand(ctx, op, types[1])?;
        let mask = has_mask
            .then(|| ranked_operand(ctx, op, types[2]))
            .transpose()?;
        let (original, original_element) = ranked_operand(ctx, op, types[types.len() - 1])?;
        let map = dimension_map(ctx, op, self.get_attr_iree_scatter_dimension_map(ctx))?;
        verify_gather_scatter(ctx, op, &original, &update, indices, mask, &map)?;
        let block = body(ctx, op)?;
        let args = block_arg_types(ctx, block);
        ensure(
            ctx,
            op,
            args.len() == 2,
            "expected region to have two arguments",
        )?;
        ensure(
            ctx,
            op,
            args.iter().all(|&a| is_int_or_float(ctx, a)),
            "expected region to have scalar argument of integer or float types",
        )?;
        ensure(
            ctx,
            op,
            args[0] == update_element,
            "mismatch in argument 0 of region and element type of update value",
        )?;
        ensure(
            ctx,
            op,
            args[1] == original_element,
            "mismatch in argument 1 of region and element type of original value",
        )?;
        ensure(
            ctx,
            op,
            args[0] == args[1],
            "mismatch in region argument types",
        )?;
        let yielded = yielded(ctx, op, block)?;
        ensure(
            ctx,
            op,
            yielded == [args[0]],
            "region must yield a single value of the argument type",
        )
    }
}

/// Gathers slices of `source` at `indices` into `output`.
///
/// Operands are `source`, `indices`, optional `mask` (ins), then `output`
/// (outs). `dimension_map` lists which `source` dimensions each index
/// coordinate addresses.
#[pliron_op(
    name = "iree_linalg_ext.gather",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    attributes = (iree_gather_dimension_map: I64ArrayAttr)
)]
pub struct GatherOp;

impl GatherOp {
    /// Allocates the operation without verifying it.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        indices: Value,
        mask: Option<Value>,
        output: Value,
        dimension_map: Vec<i64>,
    ) -> Self {
        let results = vec![output.get_type(ctx)];
        let operands = [source, indices]
            .into_iter()
            .chain(mask)
            .chain([output])
            .collect();
        let op = new_plain(ctx, |op| Self { op }, results, operands);
        op.set_attr_iree_gather_dimension_map(ctx, I64ArrayAttr::new(dimension_map));
        op
    }

    /// Builds and verifies the operation.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    pub fn build(
        ctx: &mut Context,
        source: Value,
        indices: Value,
        mask: Option<Value>,
        output: Value,
        dimension_map: Vec<i64>,
    ) -> Result<Self> {
        let op = Self::new(ctx, source, indices, mask, output, dimension_map);
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns the dimension map, or an empty list when absent.
    pub fn dimension_map(&self, ctx: &Context) -> Vec<i64> {
        self.get_attr_iree_gather_dimension_map(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }
}

impl Verify for GatherOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        no_regions(ctx, op)?;
        let (types, has_mask) = split(ctx, op)?;
        let (source, _) = ranked_operand(ctx, op, types[0])?;
        let indices = ranked_operand(ctx, op, types[1])?;
        let mask = has_mask
            .then(|| ranked_operand(ctx, op, types[2]))
            .transpose()?;
        let (output, _) = ranked_operand(ctx, op, types[types.len() - 1])?;
        let map = dimension_map(ctx, op, self.get_attr_iree_gather_dimension_map(ctx))?;
        verify_gather_scatter(ctx, op, &source, &output, indices, mask, &map)
    }
}

/// Loads `output` from `source` through an index transformation.
///
/// The region receives one `index` per output dimension and yields one
/// `index` per source dimension followed by a padding value of the source
/// element type for out-of-bounds reads.
#[pliron_op(
    name = "iree_linalg_ext.map_load",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig"
)]
pub struct MapLoadOp;

impl MapLoadOp {
    /// Allocates the operation with an empty transformation block.
    pub fn new(ctx: &mut Context, source: Value, output: Value) -> Self {
        let output_ty = output.get_type(ctx);
        let rank = ranked(ctx, output_ty).map_or(0, |(d, _)| d.len());
        let index = shaped::index(ctx);
        new_with_body(
            ctx,
            |op| Self { op },
            vec![output_ty],
            vec![source, output],
            vec![index; rank],
        )
    }

    /// Builds the operation with a transformation from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    pub fn build(
        ctx: &mut Context,
        source: Value,
        output: Value,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, source, output);
        populate(ctx, op, body)
    }
}

impl Verify for MapLoadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        ensure(
            ctx,
            op,
            num_operands(ctx, op) == 2,
            "expected source and output",
        )?;
        verify_outputs(ctx, op, 1..2)?;
        let types = operand_types(ctx, op);
        let (source, source_element) = ranked_operand(ctx, op, types[0])?;
        let (output, output_element) = ranked_operand(ctx, op, types[1])?;
        ensure(
            ctx,
            op,
            source_element == output_element,
            "expected source and output element types to match",
        )?;
        let block = body(ctx, op)?;
        let args = block_arg_types(ctx, block);
        ensure(
            ctx,
            op,
            args.len() == output.len(),
            "expected number of block arguments to be equal to the output rank",
        )?;
        ensure(
            ctx,
            op,
            args.iter().all(|&a| shaped::is_index(ctx, a)),
            "expected block arguments to be index types",
        )?;
        let yielded = yielded(ctx, op, block)?;
        ensure(
            ctx,
            op,
            yielded.len() == source.len() + 1,
            "expected transformation_region to yield a value for each source dimension and a padding value",
        )?;
        ensure(
            ctx,
            op,
            yielded[..source.len()]
                .iter()
                .all(|&t| shaped::is_index(ctx, t)),
            "expected yielded indices to be index types",
        )?;
        ensure(
            ctx,
            op,
            yielded[source.len()] == source_element,
            "expected yielded padding value type to match source element type",
        )
    }
}

/// Stores `input` into `output` through an index transformation.
///
/// The region receives one `index` per input dimension and yields one
/// `index` per output dimension followed by an `i1` mask; masked-off
/// elements are not stored.
#[pliron_op(
    name = "iree_linalg_ext.map_store",
    format = "operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig"
)]
pub struct MapStoreOp;

impl MapStoreOp {
    /// Allocates the operation with an empty transformation block.
    pub fn new(ctx: &mut Context, input: Value, output: Value) -> Self {
        let input_ty = input.get_type(ctx);
        let rank = ranked(ctx, input_ty).map_or(0, |(d, _)| d.len());
        let index = shaped::index(ctx);
        let results = vec![output.get_type(ctx)];
        new_with_body(
            ctx,
            |op| Self { op },
            results,
            vec![input, output],
            vec![index; rank],
        )
    }

    /// Builds the operation with a transformation from `body` and verifies it.
    ///
    /// # Errors
    ///
    /// Propagates errors from `body` and verification.
    pub fn build(
        ctx: &mut Context,
        input: Value,
        output: Value,
        body: Box<BodyBuilder<'_>>,
    ) -> Result<Self> {
        let op = Self::new(ctx, input, output);
        populate(ctx, op, body)
    }
}

impl Verify for MapStoreOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        ensure(
            ctx,
            op,
            num_operands(ctx, op) == 2,
            "expected input and output",
        )?;
        verify_outputs(ctx, op, 1..2)?;
        let types = operand_types(ctx, op);
        let (input, input_element) = ranked_operand(ctx, op, types[0])?;
        let (output, output_element) = ranked_operand(ctx, op, types[1])?;
        ensure(
            ctx,
            op,
            input_element == output_element,
            "expected input and output element types to match",
        )?;
        ensure(
            ctx,
            op,
            !input.is_empty(),
            "expected input type to have non-zero rank",
        )?;
        let block = body(ctx, op)?;
        let args = block_arg_types(ctx, block);
        ensure(
            ctx,
            op,
            args.len() == input.len(),
            "expected number of block arguments to be equal to the input rank",
        )?;
        ensure(
            ctx,
            op,
            args.iter().all(|&a| shaped::is_index(ctx, a)),
            "expected block arguments to be index types",
        )?;
        let yielded = yielded(ctx, op, block)?;
        ensure(
            ctx,
            op,
            yielded.len() == output.len() + 1,
            "expected transformation_region to yield a value for each output dimension and a mask",
        )?;
        ensure(
            ctx,
            op,
            yielded[..output.len()]
                .iter()
                .all(|&t| shaped::is_index(ctx, t)),
            "expected yielded indices to be index types",
        )?;
        ensure(
            ctx,
            op,
            int_width(ctx, yielded[output.len()]) == Some(1),
            "expected yielded mask to be i1 type",
        )
    }
}
