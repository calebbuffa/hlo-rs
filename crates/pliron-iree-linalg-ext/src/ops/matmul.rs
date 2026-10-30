//! Grouped (mixture-of-experts) matrix multiplication.

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
use pliron_arith::{ops::ConstantOp, shaped};
use pliron_mlir_builtin::attributes::IndexAttr;

/// Checks a ranked float tensor operand of exactly `rank` dimensions.
fn float_tensor(
    ctx: &Context,
    op: Ptr<Operation>,
    ty: TypeHandle,
    rank: usize,
) -> Result<(Vec<i64>, TypeHandle)> {
    let (dims, element) = ranked_operand(ctx, op, ty)?;
    ensure(
        ctx,
        op,
        dims.len() == rank && shaped::is_float(ctx, element),
        "operand must be a float tensor of the expected rank",
    )?;
    Ok((dims, element))
}

/// Checks the shared group operands: offsets, `index` scalars, and element types.
fn verify_group(
    ctx: &Context,
    op: Ptr<Operation>,
    types: &[TypeHandle],
    scalars: std::ops::Range<usize>,
    elements: [TypeHandle; 3],
    expert_count: i64,
) -> Result<()> {
    let (offsets, offsets_element) = ranked_operand(ctx, op, types[2])?;
    ensure(
        ctx,
        op,
        offsets.len() == 1 && shaped::is_signless_int_or_index(ctx, offsets_element),
        "expert_offsets must be a rank-1 tensor of signless integer or index",
    )?;
    ensure(
        ctx,
        op,
        types[scalars].iter().all(|&t| shaped::is_index(ctx, t)),
        "row_base and row_stride must be index",
    )?;
    ensure(
        ctx,
        op,
        expert_count >= 0 && offsets[0] >= 0,
        "expert count must be static",
    )?;
    ensure(
        ctx,
        op,
        offsets[0] == expert_count,
        "expert_offsets' E dimension must match expert_weights' E",
    )?;
    ensure(
        ctx,
        op,
        elements[0] == elements[1] && elements[0] == elements[2],
        "input, expert_weights, and output element types must match",
    )
}

/// Returns the value of `value` when it is an `arith.constant` of `index` type.
fn constant_index(ctx: &Context, value: Value) -> Option<i64> {
    let def = value.defining_op()?;
    let constant = Operation::get_op::<ConstantOp>(def, ctx)?;
    constant
        .value(ctx)?
        .downcast_ref::<IndexAttr>()
        .map(|a| a.value)
}

/// Grouped matmul: rows of `input` (`[T, N]`) are multiplied by the expert
/// weight matrix (`[N, M]`) selected from `expert_weights` (`[E, N, M]`).
///
/// Operands are `input`, `expert_weights`, `expert_offsets` (`[E]`),
/// `row_base` (`index`), and `output` (`[T, M]`, outs). `expert_offsets` holds
/// absolute exclusive row ends; global row `row_base + r` uses expert `e`
/// when `start[e] <= g < expert_offsets[e]`. Uncovered rows keep their
/// `output` values.
#[pliron_op(
    name = "iree_linalg_ext.group_matmul",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig"
)]
pub struct GroupMatmulOp;

impl GroupMatmulOp {
    /// Allocates the operation without verifying it.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        expert_weights: Value,
        expert_offsets: Value,
        row_base: Value,
        output: Value,
    ) -> Self {
        let results = vec![output.get_type(ctx)];
        let operands = vec![input, expert_weights, expert_offsets, row_base, output];
        new_plain(ctx, |op| Self { op }, results, operands)
    }

    /// Builds and verifies the operation.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    pub fn build(
        ctx: &mut Context,
        input: Value,
        expert_weights: Value,
        expert_offsets: Value,
        row_base: Value,
        output: Value,
    ) -> Result<Self> {
        let op = Self::new(ctx, input, expert_weights, expert_offsets, row_base, output);
        op.verify(ctx)?;
        Ok(op)
    }
}

impl Verify for GroupMatmulOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        no_regions(ctx, op)?;
        ensure(
            ctx,
            op,
            num_operands(ctx, op) == 5,
            "expected five operands",
        )?;
        verify_outputs(ctx, op, 4..5)?;
        let types = operand_types(ctx, op);
        let (input, input_element) = float_tensor(ctx, op, types[0], 2)?;
        let (weights, weights_element) = float_tensor(ctx, op, types[1], 3)?;
        let (output, output_element) = float_tensor(ctx, op, types[4], 2)?;
        ensure(ctx, op, weights[0] >= 0, "expert count must be static")?;
        ensure(
            ctx,
            op,
            compatible(input[1], weights[1]),
            "input's N dimension must match expert_weights' N",
        )?;
        ensure(
            ctx,
            op,
            compatible(output[1], weights[2]),
            "output's M dimension must match expert_weights' M",
        )?;
        ensure(
            ctx,
            op,
            compatible(input[0], output[0]),
            "output's T dimension must match input's T",
        )?;
        verify_group(
            ctx,
            op,
            &types,
            3..4,
            [input_element, weights_element, output_element],
            weights[0],
        )
    }
}

/// Data-tiled form of [`GroupMatmulOp`], analogous to `linalg.mmt4d`.
///
/// Operands are `input` [M, K, Mi, Ki], `expert_weights` [E, N, K, Ni, Ki],
/// `expert_offsets` [E], `row_base` and `row_stride` (`index`), and `output`
/// (outs) laid out [M, N, Mi, Ni], or [N, M, Ni, Mi] when `transposed`.
/// Input element `[m, k, mi, ki]` belongs to global row
/// `row_base + m * row_stride + mi`.
#[pliron_op(
    name = "iree_linalg_ext.group_mmt4d",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    attributes = (iree_group_mmt4d_transposed: BoolAttr)
)]
pub struct GroupMmt4dOp;

impl GroupMmt4dOp {
    /// Allocates the operation without verifying it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        input: Value,
        expert_weights: Value,
        expert_offsets: Value,
        row_base: Value,
        row_stride: Value,
        output: Value,
        transposed: bool,
    ) -> Self {
        let results = vec![output.get_type(ctx)];
        let operands = vec![
            input,
            expert_weights,
            expert_offsets,
            row_base,
            row_stride,
            output,
        ];
        let op = new_plain(ctx, |op| Self { op }, results, operands);
        op.set_attr_iree_group_mmt4d_transposed(ctx, BoolAttr::new(transposed));
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
        expert_weights: Value,
        expert_offsets: Value,
        row_base: Value,
        row_stride: Value,
        output: Value,
        transposed: bool,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            input,
            expert_weights,
            expert_offsets,
            row_base,
            row_stride,
            output,
            transposed,
        );
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns whether the output is transposed; upstream defaults to `false`.
    pub fn transposed(&self, ctx: &Context) -> bool {
        self.get_attr_iree_group_mmt4d_transposed(ctx)
            .is_some_and(|a| bool::from((*a).clone()))
    }

    /// Returns the output-to-canonical `[M, N, Mi, Ni]` permutation.
    pub fn output_permutation(&self, ctx: &Context) -> [usize; 4] {
        if self.transposed(ctx) {
            [1, 0, 3, 2]
        } else {
            [0, 1, 2, 3]
        }
    }

    /// Returns the loop permutation over `[M, N, K, Mi, Ni, Ki]`.
    pub fn loop_permutation(&self, ctx: &Context) -> [usize; 6] {
        if self.transposed(ctx) {
            [1, 0, 2, 4, 3, 5]
        } else {
            [0, 1, 2, 3, 4, 5]
        }
    }
}

impl Verify for GroupMmt4dOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        no_regions(ctx, op)?;
        ensure(ctx, op, num_operands(ctx, op) == 6, "expected six operands")?;
        verify_outputs(ctx, op, 5..6)?;
        let types = operand_types(ctx, op);
        let (input, input_element) = float_tensor(ctx, op, types[0], 4)?;
        let (weights, weights_element) = float_tensor(ctx, op, types[1], 5)?;
        let (output, output_element) = float_tensor(ctx, op, types[5], 4)?;
        ensure(ctx, op, weights[0] >= 0, "expert count must be static")?;
        let p = self.output_permutation(ctx);
        let dims_match = compatible(input[0], output[p[0]])
            && compatible(input[1], weights[2])
            && compatible(output[p[1]], weights[1])
            && compatible(input[2], output[p[2]])
            && compatible(output[p[3]], weights[3])
            && compatible(input[3], weights[4]);
        ensure(ctx, op, dims_match, "operand dimensions must match mmt4d")?;
        verify_group(
            ctx,
            op,
            &types,
            3..5,
            [input_element, weights_element, output_element],
            weights[0],
        )?;
        let row_stride = op.deref(ctx).get_operand(4);
        if let Some(stride) = constant_index(ctx, row_stride) {
            ensure(
                ctx,
                op,
                stride >= 0 && (input[2] < 0 || stride >= input[2]),
                "row_stride must be nonnegative and at least the inner row dimension",
            )?;
        }
        Ok(())
    }
}
