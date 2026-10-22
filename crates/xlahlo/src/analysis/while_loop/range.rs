//! Pinned, empty-known-map source range matcher, not an execution proof.
use super::*;
use crate::types::ElementType;

/// Source-matcher metadata, not a proof of finite execution or a trip count.
///
/// Endpoints retain their source integer formats. The inclusive end may precede
/// the start. GT/GE and swapped comparisons retain the pinned matcher's quirks:
/// consumers must independently prove direction, termination and invariance
/// before using this metadata to rewrite IR or infer executed values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceLoopRange {
    /// Initial value, preserving the source integer element type.
    pub start: Scalar,
    /// Aligned inclusive endpoint, preserving the bound's source element type.
    /// It may be less than `start` when the source pattern does not execute.
    pub inclusive_end: Scalar,
    /// Positive S64 scalar.
    pub step: Scalar,
}

impl Analysis<'_> {
    pub(super) fn source_range(
        &mut self,
        parts: &Parts,
        index: usize,
    ) -> Result<Option<SourceLoopRange>> {
        let ctx = self.view.ctx;
        let initial = trace(ctx, operand(ctx, parts.input, index)?)?;
        let Some(initial) = self.instruction(initial, &[], EvaluationStage::InitialValue)? else {
            return Ok(None);
        };
        let Some(start) = integer(ctx, &initial)? else {
            return Ok(None);
        };
        // Range matching deliberately does not trace the update or its operand.
        let update = operand(ctx, parts.body_root, index)?;
        let Some(iv) = nonconstant_operand(ctx, update)? else {
            return Ok(None);
        };
        if unique_gte(ctx, parts.body_parameter, index)? != Some(iv)
            || !is_opcode(ctx, update, Opcode::Add)?
        {
            return Ok(None);
        }
        let operands = inputs(ctx, defining(update)?);
        let step_value = if operands[0] == iv {
            operands[1]
        } else {
            operands[0]
        };
        if !is_opcode(ctx, step_value, Opcode::Constant)? {
            return Ok(None);
        }
        let Some(step_value) = self.instruction(step_value, &[], EvaluationStage::Update)? else {
            return Ok(None);
        };
        let Some(step) = integer(ctx, &step_value)?.filter(|value| *value > 0) else {
            return Ok(None);
        };
        let Some(iv) = unique_gte(ctx, parts.condition_parameter, index)? else {
            return Ok(None);
        };
        let condition = parts.condition_root;
        if !is_opcode(ctx, condition, Opcode::Compare)? {
            return Ok(None);
        }
        let operation = defining(condition)?;
        let operands = inputs(ctx, operation);
        let bound = if operands[0] == iv {
            operands[1]
        } else if operands[1] == iv {
            operands[0]
        } else {
            return Ok(None);
        };
        if !is_opcode(ctx, bound, Opcode::Constant)? {
            return Ok(None);
        }
        let Some(bound_value) = self.instruction(bound, &[], EvaluationStage::Condition)? else {
            return Ok(None);
        };
        let Some(mut end) = integer(ctx, &bound_value)? else {
            return Ok(None);
        };
        match attr::<Direction>(ctx, operation, "hlo_direction")? {
            Direction::Lt | Direction::Gt => {
                let Some(decremented) = end.checked_sub(1) else {
                    return Ok(None);
                };
                end = decremented;
            }
            Direction::Le | Direction::Ge => {}
            _ => return Ok(None),
        }
        // Widen only the remainder difference, which can overflow signed i64.
        // All endpoint arithmetic still declines on the source's i64 bounds.
        let step_wide = i128::from(step);
        let mut remainder = (i128::from(end % step) - i128::from(start % step)) % step_wide;
        if remainder < 0 {
            remainder += step_wide;
        }
        let remainder = i64::try_from(remainder)
            .map_err(|_| invalid("range alignment remainder exceeds i64"))?;
        let Some(end) = end.checked_sub(remainder) else {
            return Ok(None);
        };
        let initial_element = shape::element_type(ctx, initial.shape())?;
        if end >= start {
            let Some(post_loop) = end.checked_add(step) else {
                return Ok(None);
            };
            let width = initial_element.bit_width();
            let (minimum, maximum) = if initial_element.is_signed() {
                (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
            } else {
                (0, (1i128 << width) - 1)
            };
            if !(minimum..=maximum).contains(&i128::from(post_loop)) {
                return Ok(None);
            }
        }
        Ok(Some(SourceLoopRange {
            start: endpoint(initial_element, start)?,
            inclusive_end: endpoint(shape::element_type(ctx, bound_value.shape())?, end)?,
            step: endpoint(ElementType::S64, step)?,
        }))
    }
}

fn endpoint(element: ElementType, value: i64) -> Result<Scalar> {
    let mask = (1u128 << element.bit_width()) - 1;
    Scalar::from_bits(element, (value as u128) & mask)
}
