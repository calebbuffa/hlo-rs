//! Image-to-column rearrangement for convolution lowering.

use crate::support::*;
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{OperandSegmentSizesAttr, VecAttr},
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
use pliron_mlir_builtin::attributes::I64ArrayAttr;

/// Static marker for a dynamic entry in a mixed static/dynamic index list.
pub const DYNAMIC_INDEX: i64 = i64::MIN;

/// One entry of a mixed static/dynamic index list.
#[derive(Clone, Copy)]
pub enum MixedIndex {
    /// A constant index.
    Static(i64),
    /// An `index`-typed SSA value.
    Dynamic(Value),
}

/// Dimension roles of the input and the output dimension order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Im2colPositions {
    /// Input dimensions carried through as batch dimensions.
    pub batch: Vec<i64>,
    /// Input dimensions that form the M (output spatial) side.
    pub m: Vec<i64>,
    /// Input dimensions that form the K (reduction) side.
    pub k: Vec<i64>,
    /// Order in which the shared `m + k` coordinates index the input.
    pub input_k_perm: Vec<i64>,
    /// Permutation from canonical `[batch, M, K]` to the output layout.
    pub output_perm: Vec<i64>,
}

/// Optional input and output padding.
#[derive(Clone, Default)]
pub struct Im2colPadding {
    /// Low input padding, one entry per input dimension or empty.
    pub input_low: Vec<MixedIndex>,
    /// High input padding, one entry per input dimension or empty.
    pub input_high: Vec<MixedIndex>,
    /// Low output padding, one entry per output dimension or empty.
    pub output_low: Vec<MixedIndex>,
    /// High output padding, one entry per output dimension or empty.
    pub output_high: Vec<MixedIndex>,
    /// Value produced for out-of-bounds positions; required when padding is set.
    pub value: Option<Value>,
}

fn split(list: &[MixedIndex]) -> (Vec<i64>, Vec<Value>) {
    let mut statics = Vec::with_capacity(list.len());
    let mut dynamics = Vec::new();
    for entry in list {
        match *entry {
            MixedIndex::Static(v) => statics.push(v),
            MixedIndex::Dynamic(v) => {
                statics.push(DYNAMIC_INDEX);
                dynamics.push(v);
            }
        }
    }
    (statics, dynamics)
}

fn array(values: Vec<i64>) -> I64ArrayAttr {
    I64ArrayAttr::new(values)
}

fn values(attr: Option<impl std::ops::Deref<Target = I64ArrayAttr>>) -> Vec<i64> {
    attr.map(|a| a.values.clone()).unwrap_or_default()
}

/// Rebuilds a mixed list from its static entries and the dynamic operands at `segment`.
fn mixed(
    ctx: &Context,
    op: Ptr<Operation>,
    statics: &[i64],
    segment: (usize, usize),
) -> Result<Vec<MixedIndex>> {
    ensure(
        ctx,
        op,
        statics.iter().filter(|&&v| v == DYNAMIC_INDEX).count() == segment.1,
        "dynamic entries must match the dynamic operand counts",
    )?;
    let mut next = segment.0;
    Ok(statics
        .iter()
        .map(|&v| {
            if v == DYNAMIC_INDEX {
                next += 1;
                MixedIndex::Dynamic(op.deref(ctx).get_operand(next - 1))
            } else {
                MixedIndex::Static(v)
            }
        })
        .collect())
}

/// Rearranges convolution input windows into a matrix (im2col).
///
/// The input's dimensions are partitioned into `batch`, `m`, and `k`
/// positions. The output, in canonical `[batch..., M..., K...]` order
/// permuted by `output_perm`, is described by per-dimension `offsets` and
/// nested `output_sizes` (the inner sizes of each output dimension
/// delinearize it into input coordinates). `strides`, `dilations`, and
/// `kernel_size` give one entry per `m` position. Optional input/output
/// padding reads `pad_value` out of bounds.
///
/// Operand segments are `input`, `output` (outs), `kernel_size`, `offsets`,
/// `output_sizes`, `input_pad_low`, `input_pad_high`, `output_pad_low`,
/// `output_pad_high`, and optional `pad_value`. Dynamic entries of the
/// static lists are [`DYNAMIC_INDEX`] and are taken from those segments in order.
#[pliron_op(
    name = "iree_linalg_ext.im2col",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    attributes = (
        iree_im2col_strides: I64ArrayAttr,
        iree_im2col_dilations: I64ArrayAttr,
        iree_im2col_static_kernel_size: I64ArrayAttr,
        iree_im2col_static_offsets: I64ArrayAttr,
        iree_im2col_static_output_sizes: VecAttr,
        iree_im2col_batch_pos: I64ArrayAttr,
        iree_im2col_m_pos: I64ArrayAttr,
        iree_im2col_k_pos: I64ArrayAttr,
        iree_im2col_input_k_perm: I64ArrayAttr,
        iree_im2col_output_perm: I64ArrayAttr,
        iree_im2col_static_input_pad_low: I64ArrayAttr,
        iree_im2col_static_input_pad_high: I64ArrayAttr,
        iree_im2col_static_output_pad_low: I64ArrayAttr,
        iree_im2col_static_output_pad_high: I64ArrayAttr,
        iree_im2col_operand_segment_sizes: OperandSegmentSizesAttr
    )
)]
pub struct Im2colOp;

impl Im2colOp {
    /// Allocates the operation without verifying it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        input: Value,
        output: Value,
        strides: Vec<i64>,
        dilations: Vec<i64>,
        kernel_size: &[MixedIndex],
        offsets: &[MixedIndex],
        output_sizes: &[Vec<MixedIndex>],
        positions: Im2colPositions,
        padding: Im2colPadding,
    ) -> Self {
        let (kernel_static, kernel_dynamic) = split(kernel_size);
        let (offsets_static, offsets_dynamic) = split(offsets);
        let mut sizes_static = Vec::with_capacity(output_sizes.len());
        let mut sizes_dynamic = Vec::new();
        for inner in output_sizes {
            let (s, d) = split(inner);
            sizes_static.push(s);
            sizes_dynamic.extend(d);
        }
        let pads = [
            &padding.input_low,
            &padding.input_high,
            &padding.output_low,
            &padding.output_high,
        ]
        .map(|p| split(p));
        let mut counts = vec![
            1,
            1,
            kernel_dynamic.len(),
            offsets_dynamic.len(),
            sizes_dynamic.len(),
        ];
        counts.extend(pads.iter().map(|(_, d)| d.len()));
        counts.push(usize::from(padding.value.is_some()));
        let operands = [input, output]
            .into_iter()
            .chain(kernel_dynamic)
            .chain(offsets_dynamic)
            .chain(sizes_dynamic)
            .chain(pads.iter().flat_map(|(_, d)| d.iter().copied()))
            .chain(padding.value)
            .collect();
        let results = vec![output.get_type(ctx)];
        let op = new_plain(ctx, |op| Self { op }, results, operands);
        let [input_low, input_high, output_low, output_high] = pads.map(|(s, _)| s);
        let nested: Vec<AttrObj> = sizes_static
            .into_iter()
            .map(|s| -> AttrObj { Box::new(array(s)) })
            .collect();
        op.set_attr_iree_im2col_strides(ctx, array(strides));
        op.set_attr_iree_im2col_dilations(ctx, array(dilations));
        op.set_attr_iree_im2col_static_kernel_size(ctx, array(kernel_static));
        op.set_attr_iree_im2col_static_offsets(ctx, array(offsets_static));
        op.set_attr_iree_im2col_static_output_sizes(ctx, VecAttr::new(nested));
        op.set_attr_iree_im2col_batch_pos(ctx, array(positions.batch));
        op.set_attr_iree_im2col_m_pos(ctx, array(positions.m));
        op.set_attr_iree_im2col_k_pos(ctx, array(positions.k));
        op.set_attr_iree_im2col_input_k_perm(ctx, array(positions.input_k_perm));
        op.set_attr_iree_im2col_output_perm(ctx, array(positions.output_perm));
        op.set_attr_iree_im2col_static_input_pad_low(ctx, array(input_low));
        op.set_attr_iree_im2col_static_input_pad_high(ctx, array(input_high));
        op.set_attr_iree_im2col_static_output_pad_low(ctx, array(output_low));
        op.set_attr_iree_im2col_static_output_pad_high(ctx, array(output_high));
        op.set_attr_iree_im2col_operand_segment_sizes(ctx, segment_sizes(&counts));
        op
    }

    /// Builds and verifies the operation.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        input: Value,
        output: Value,
        strides: Vec<i64>,
        dilations: Vec<i64>,
        kernel_size: &[MixedIndex],
        offsets: &[MixedIndex],
        output_sizes: &[Vec<MixedIndex>],
        positions: Im2colPositions,
        padding: Im2colPadding,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            input,
            output,
            strides,
            dilations,
            kernel_size,
            offsets,
            output_sizes,
            positions,
            padding,
        );
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns the dimension roles and permutations.
    pub fn positions(&self, ctx: &Context) -> Im2colPositions {
        Im2colPositions {
            batch: values(self.get_attr_iree_im2col_batch_pos(ctx)),
            m: values(self.get_attr_iree_im2col_m_pos(ctx)),
            k: values(self.get_attr_iree_im2col_k_pos(ctx)),
            input_k_perm: values(self.get_attr_iree_im2col_input_k_perm(ctx)),
            output_perm: values(self.get_attr_iree_im2col_output_perm(ctx)),
        }
    }

    /// Returns the convolution strides.
    pub fn strides(&self, ctx: &Context) -> Vec<i64> {
        values(self.get_attr_iree_im2col_strides(ctx))
    }

    /// Returns the convolution dilations.
    pub fn dilations(&self, ctx: &Context) -> Vec<i64> {
        values(self.get_attr_iree_im2col_dilations(ctx))
    }

    fn statics(&self, ctx: &Context) -> [Vec<i64>; 6] {
        [
            values(self.get_attr_iree_im2col_static_kernel_size(ctx)),
            values(self.get_attr_iree_im2col_static_offsets(ctx)),
            values(self.get_attr_iree_im2col_static_input_pad_low(ctx)),
            values(self.get_attr_iree_im2col_static_input_pad_high(ctx)),
            values(self.get_attr_iree_im2col_static_output_pad_low(ctx)),
            values(self.get_attr_iree_im2col_static_output_pad_high(ctx)),
        ]
    }

    /// Returns the static inner output sizes, `None` when an entry is not an
    /// `i64` array.
    pub fn static_output_sizes(&self, ctx: &Context) -> Option<Vec<Vec<i64>>> {
        let attr = self.get_attr_iree_im2col_static_output_sizes(ctx)?;
        attr.0
            .iter()
            .map(|a| a.downcast_ref::<I64ArrayAttr>().map(|a| a.values.clone()))
            .collect()
    }

    fn segments(&self, ctx: &Context) -> Result<[(usize, usize); 10]> {
        let sizes = self.get_attr_iree_im2col_operand_segment_sizes(ctx);
        segments::<10>(ctx, self.get_operation(), sizes.as_deref())
    }

    /// Returns the kernel sizes as a mixed list.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand segments are malformed.
    pub fn mixed_kernel_size(&self, ctx: &Context) -> Result<Vec<MixedIndex>> {
        let s = self.segments(ctx)?;
        mixed(ctx, self.get_operation(), &self.statics(ctx)[0], s[2])
    }

    /// Returns the output offsets as a mixed list.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand segments are malformed.
    pub fn mixed_offsets(&self, ctx: &Context) -> Result<Vec<MixedIndex>> {
        let s = self.segments(ctx)?;
        mixed(ctx, self.get_operation(), &self.statics(ctx)[1], s[3])
    }

    /// Returns the nested output sizes as mixed lists.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand segments or sizes are malformed.
    pub fn mixed_output_sizes(&self, ctx: &Context) -> Result<Vec<Vec<MixedIndex>>> {
        let op = self.get_operation();
        let (mut start, len) = self.segments(ctx)?[4];
        let end = start + len;
        let sizes = self
            .static_output_sizes(ctx)
            .ok_or_else(|| fail(ctx, op, "static_output_sizes must hold i64 arrays"))?;
        let lists = sizes
            .iter()
            .map(|inner| {
                let dynamic = inner.iter().filter(|&&v| v == DYNAMIC_INDEX).count();
                ensure(
                    ctx,
                    op,
                    dynamic <= end - start,
                    "not enough dynamic output-size operands",
                )?;
                let list = mixed(ctx, op, inner, (start, dynamic))?;
                start += dynamic;
                Ok(list)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure(ctx, op, start == end, "unused dynamic output-size operands")?;
        Ok(lists)
    }

    /// Returns the padding lists and pad value.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand segments are malformed.
    pub fn padding(&self, ctx: &Context) -> Result<Im2colPadding> {
        let op = self.get_operation();
        let s = self.segments(ctx)?;
        ensure(ctx, op, s[9].1 <= 1, "expected at most one pad_value")?;
        let [_, _, input_low, input_high, output_low, output_high] = self.statics(ctx);
        Ok(Im2colPadding {
            input_low: mixed(ctx, op, &input_low, s[5])?,
            input_high: mixed(ctx, op, &input_high, s[6])?,
            output_low: mixed(ctx, op, &output_low, s[7])?,
            output_high: mixed(ctx, op, &output_high, s[8])?,
            value: (s[9].1 == 1).then(|| op.deref(ctx).get_operand(s[9].0)),
        })
    }
}

impl Verify for Im2colOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        no_regions(ctx, op)?;
        let s = self.segments(ctx)?;
        ensure(ctx, op, s[0].1 == 1, "expected only one ShapedType operand")?;
        ensure(ctx, op, s[1].1 == 1, "expected one output operand")?;
        ensure(ctx, op, s[9].1 <= 1, "expected at most one pad_value")?;
        verify_outputs(ctx, op, s[1].0..s[1].0 + 1)?;
        let types = operand_types(ctx, op);
        let (input, _) = ranked_operand(ctx, op, types[0])?;
        let (output, _) = ranked_operand(ctx, op, types[1])?;
        if s[9].1 == 1 {
            ensure(
                ctx,
                op,
                ranked(ctx, types[s[9].0]).is_none(),
                "expected only one ShapedType operand",
            )?;
        }
        ensure(
            ctx,
            op,
            types[s[2].0..s[8].0 + s[8].1]
                .iter()
                .all(|&t| shaped::is_index(ctx, t)),
            "dynamic sizes, offsets, and padding must be index",
        )?;
        let statics = self.statics(ctx);
        let sizes = self
            .static_output_sizes(ctx)
            .ok_or_else(|| fail(ctx, op, "static_output_sizes must hold i64 arrays"))?;
        let dynamic = |list: &[i64]| list.iter().filter(|&&v| v == DYNAMIC_INDEX).count();
        let segment_for = [s[2], s[3], s[5], s[6], s[7], s[8]];
        let consistent = statics
            .iter()
            .zip(segment_for)
            .all(|(list, seg)| dynamic(list) == seg.1)
            && sizes.iter().map(|inner| dynamic(inner)).sum::<usize>() == s[4].1;
        ensure(
            ctx,
            op,
            consistent,
            "dynamic entries must match the dynamic operand counts",
        )?;

        let p = self.positions(ctx);
        let input_rank = input.len();
        let output_rank = output.len();
        ensure(
            ctx,
            op,
            input_rank == p.batch.len() + p.m.len() + p.k.len(),
            "expected input rank to be the sum of batch, m, and k ranks",
        )?;
        ensure(
            ctx,
            op,
            statics[1].len() == output_rank,
            "expected offsets size to match output rank",
        )?;
        ensure(
            ctx,
            op,
            sizes.len() == output_rank,
            "expected output_sizes outer size to match output rank",
        )?;

        let batch_dims = p.batch.len();
        ensure(
            ctx,
            op,
            batch_dims <= sizes.len(),
            "expected batch_pos size to not exceed output rank",
        )?;
        let batch_inner: usize = sizes[..batch_dims].iter().map(Vec::len).sum();
        ensure(
            ctx,
            op,
            batch_inner == p.batch.len(),
            "expected sum of batch output_sizes inner dimensions to equal batch_pos size",
        )?;
        let mut m_inner = 0;
        let mut m_dims = 0;
        for (i, inner) in sizes.iter().enumerate().skip(batch_dims) {
            m_inner += inner.len();
            if m_inner == p.m.len() {
                m_dims = i - batch_dims + 1;
                break;
            }
        }
        ensure(
            ctx,
            op,
            m_dims != 0,
            "M/K boundary not found: accumulated output_sizes inner dimensions must equal m_pos size at some output dim boundary",
        )?;
        let k_inner: usize = sizes[batch_dims + m_dims..].iter().map(Vec::len).sum();
        ensure(
            ctx,
            op,
            k_inner == p.m.len() + p.k.len(),
            "expected sum of K output_sizes inner dimensions to equal m_pos.size() + k_pos.size()",
        )?;

        let kernel_rank = statics[0].len();
        ensure(
            ctx,
            op,
            kernel_rank == p.m.len(),
            "expected kernel rank to be equal to the m_pos rank",
        )?;
        ensure(
            ctx,
            op,
            self.strides(ctx).len() == kernel_rank,
            "expected strides rank to be equal to the kernel rank",
        )?;
        ensure(
            ctx,
            op,
            self.dilations(ctx).len() == kernel_rank,
            "expected dilations rank to be equal to the kernel rank",
        )?;
        ensure(
            ctx,
            op,
            p.input_k_perm.len() == p.m.len() + p.k.len() && is_permutation(&p.input_k_perm),
            "expected input_k_perm to be a permutation of the shared m_pos + k_pos dimensions",
        )?;
        ensure(
            ctx,
            op,
            is_permutation(&p.output_perm),
            "expected output_perm to be a permutation",
        )?;
        ensure(
            ctx,
            op,
            p.output_perm.len() == output_rank,
            "expected output_perm to have the same rank as the result",
        )?;

        let has_pad_value = s[9].1 == 1;
        let [_, _, input_low, input_high, output_low, output_high] = &statics;
        ensure(
            ctx,
            op,
            input_low.len() == input_high.len(),
            "expected input_pad_low and input_pad_high to have the same size",
        )?;
        ensure(
            ctx,
            op,
            input_low.is_empty() || input_low.len() == input_rank,
            "expected input_pad_low size to match input rank",
        )?;
        ensure(
            ctx,
            op,
            input_low.is_empty() || has_pad_value,
            "expected pad_value when input_pad_low/input_pad_high are specified",
        )?;
        ensure(
            ctx,
            op,
            output_low.len() == output_high.len(),
            "expected output_pad_low and output_pad_high to have the same size",
        )?;
        ensure(
            ctx,
            op,
            output_low.is_empty() || output_low.len() == output_rank,
            "expected output_pad_low size to match output rank",
        )?;
        ensure(
            ctx,
            op,
            output_low.is_empty() || has_pad_value,
            "expected pad_value when output_pad_low/output_pad_high are specified",
        )
    }
}
