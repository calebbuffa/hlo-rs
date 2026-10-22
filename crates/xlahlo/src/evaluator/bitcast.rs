//! Physical byte-copy semantics for native HLO Bitcast.
//!
//! This follows pinned `HloEvaluator::HandleBitcast`: result array storage is
//! initialized from the beginning of the operand's physical bytes. No logical
//! indexing, element conversion, or dynamic-size propagation is performed.

use super::*;
use crate::shape::Dimension;
use pliron::common_traits::Verify;

impl Session<'_> {
    pub(super) fn bitcast(
        &mut self,
        frame: &mut Frame,
        operands: &[Value],
        target: TypeHandle,
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        if operands.len() != 1 {
            return Err(malformed("Bitcast requires exactly one operand"));
        }
        let input = self.operand(frame, operands[0], &[], mode)?;
        if !input.is_known() {
            return EvaluatedValue::unknown(self.ctx, target);
        }
        let input = input.into_literal(self.ctx)?;
        let output = physical_bitcast(self.ctx, &input, target)?;
        EvaluatedValue::known(self.ctx, output)
    }
}

fn physical_bitcast(ctx: &Context, input: &Literal, target: TypeHandle) -> Result<Literal> {
    input.verify(ctx)?;
    target.deref(ctx).verify(ctx)?;
    if !shape::is_array(ctx, input.shape) || !shape::is_array(ctx, target) {
        return Err(unsupported_value(
            Opcode::Bitcast,
            "native bitcast requires array literals",
        ));
    }
    if shape::is_dynamic(ctx, target) {
        return Err(unsupported_value(
            Opcode::Bitcast,
            "pinned HandleBitcast does not define bounded-dynamic result extents",
        ));
    }

    let mut target = target;
    if shape::layout(ctx, target)?.is_none() {
        let scalar_like = shape::dimensions(ctx, target)?
            .iter()
            .all(|&dimension| dimension == 1);
        if !scalar_like {
            return Err(unsupported_value(
                Opcode::Bitcast,
                "non-scalar output requires an explicit physical layout",
            ));
        }
        target = shape::with_default_layout(ctx, target)?;
    }

    let mut count = 1usize;
    for axis in 0..shape::rank(ctx, target)? {
        let bound = match shape::dimension(ctx, target, axis)? {
            Dimension::Static(bound) | Dimension::Bounded(bound) => {
                usize::try_from(bound).map_err(|_| malformed("negative Bitcast output bound"))?
            }
            Dimension::Unbounded => {
                return Err(unsupported_value(
                    Opcode::Bitcast,
                    "unbounded output storage is not implemented",
                ));
            }
        };
        count = count
            .checked_mul(bound)
            .ok_or_else(|| malformed("Bitcast output element count overflow"))?;
    }
    let width = shape::element_type(ctx, target)?.bit_width().div_ceil(8) as usize;
    let output_bytes = count
        .checked_mul(width)
        .ok_or_else(|| malformed("Bitcast output byte size overflow"))?;
    let source_bytes = input
        .storage
        .get(..output_bytes)
        .ok_or_else(|| malformed("Bitcast output is larger than its operand storage"))?
        .to_vec();

    Literal::from_dynamic_storage_bytes(ctx, target, source_bytes, Vec::new())
}
