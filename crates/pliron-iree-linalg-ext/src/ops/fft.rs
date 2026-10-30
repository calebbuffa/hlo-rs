//! Iterative radix-2 FFT stage.

use crate::support::*;
use pliron::{
    builtin::attributes::OperandSegmentSizesAttr, common_traits::Verify, context::Context,
    derive::pliron_op, op::Op, result::Result, r#type::Typed, value::Value,
};

/// One stage of an iterative 1-D FFT over the innermost dimension.
///
/// Operands are `stage` (a scalar), optional `real_coeff` and `imag_coeff`
/// tensors (present together), then `real` and `imag` (outs). Bit reversal
/// is assumed to have been applied to the input already. When the innermost
/// extent is static it must be a power of two.
#[pliron_op(
    name = "iree_linalg_ext.fft",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    attributes = (iree_fft_operand_segment_sizes: OperandSegmentSizesAttr)
)]
pub struct FftOp;

impl FftOp {
    /// Allocates the operation without verifying it.
    pub fn new(
        ctx: &mut Context,
        stage: Value,
        coefficients: Option<(Value, Value)>,
        real: Value,
        imag: Value,
    ) -> Self {
        let results = vec![real.get_type(ctx), imag.get_type(ctx)];
        let coeff = usize::from(coefficients.is_some());
        let operands = std::iter::once(stage)
            .chain(coefficients.into_iter().flat_map(|(r, i)| [r, i]))
            .chain([real, imag])
            .collect();
        let op = new_plain(ctx, |op| Self { op }, results, operands);
        op.set_attr_iree_fft_operand_segment_sizes(ctx, segment_sizes(&[1, coeff, coeff, 1, 1]));
        op
    }

    /// Builds and verifies the operation.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    pub fn build(
        ctx: &mut Context,
        stage: Value,
        coefficients: Option<(Value, Value)>,
        real: Value,
        imag: Value,
    ) -> Result<Self> {
        let op = Self::new(ctx, stage, coefficients, real, imag);
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns whether coefficient tensors are carried as inputs.
    pub fn has_coeff(&self, ctx: &Context) -> bool {
        num_operands(ctx, self.get_operation()) == 5
    }

    /// Returns the innermost extent of `real`, negative when dynamic.
    pub fn fft_length(&self, ctx: &Context) -> Option<i64> {
        let op = self.get_operation();
        let real = op
            .deref(ctx)
            .get_operand(num_operands(ctx, op).checked_sub(2)?);
        ranked(ctx, real.get_type(ctx)).and_then(|(dims, _)| dims.last().copied())
    }
}

impl Verify for FftOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        no_regions(ctx, op)?;
        let sizes = self.get_attr_iree_fft_operand_segment_sizes(ctx);
        let [stage, real_coeff, imag_coeff, real, imag] = segments::<5>(ctx, op, sizes.as_deref())?;
        ensure(
            ctx,
            op,
            stage.1 == 1 && real.1 == 1 && imag.1 == 1,
            "expected one stage, real, and imag operand",
        )?;
        ensure(
            ctx,
            op,
            real_coeff.1 <= 1 && real_coeff.1 == imag_coeff.1,
            "real_coeff and imag_coeff must be present together",
        )?;
        verify_outputs(ctx, op, real.0..imag.0 + 1)?;
        let types = operand_types(ctx, op);
        for coeff in [real_coeff, imag_coeff].into_iter().filter(|s| s.1 == 1) {
            ranked_operand(ctx, op, types[coeff.0])?;
        }
        let (dims, _) = ranked_operand(ctx, op, types[real.0])?;
        let length = *dims
            .last()
            .ok_or_else(|| fail(ctx, op, "real must have rank at least 1"))?;
        if length < 0 {
            return Ok(());
        }
        ensure(
            ctx,
            op,
            length & (length - 1) == 0,
            "only powers of 2 are handled currently",
        )?;
        ensure(
            ctx,
            op,
            is_plain_scalar(ctx, types[stage.0]),
            "expected to carry `stage` input",
        )
    }
}
