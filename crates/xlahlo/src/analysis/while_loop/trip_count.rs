//! Evaluator-backed exact and upper trip-count queries and checked count arithmetic.

use super::*;

impl Analysis<'_> {
    fn evaluate(
        &mut self,
        request: Request<'_>,
        stage: EvaluationStage,
    ) -> Result<Option<EvaluatedValue>> {
        match self.evaluator.evaluate(request) {
            Ok(value) if value.is_known() => Ok(Some(value)),
            Ok(_) => {
                self.declines.push(EvaluationDecline {
                    stage,
                    reason: DeclineReason::UnknownValue,
                });
                Ok(None)
            }
            Err(error) => {
                let reason = match error.err.downcast_ref::<EvaluationError>() {
                    Some(EvaluationError::UnknownValue) => DeclineReason::UnknownValue,
                    Some(EvaluationError::ShallowDependency) => DeclineReason::ShallowDependency,
                    Some(EvaluationError::Unsupported { opcode }) => {
                        DeclineReason::Unsupported { opcode: *opcode }
                    }
                    Some(EvaluationError::UnsupportedValue { opcode, message }) => {
                        DeclineReason::UnsupportedValue {
                            opcode: *opcode,
                            message: message.clone(),
                        }
                    }
                    _ => return Err(error),
                };
                self.declines.push(EvaluationDecline { stage, reason });
                Ok(None)
            }
        }
    }

    pub(super) fn instruction(
        &mut self,
        value: Value,
        substitutions: &[(Value, EvaluatedValue)],
        stage: EvaluationStage,
    ) -> Result<Option<EvaluatedValue>> {
        self.evaluate(
            Request::Instruction {
                value,
                substitutions,
                dependency_mode: DependencyMode::Shallow,
            },
            stage,
        )
    }

    pub(super) fn exact(
        &mut self,
        parts: &Parts,
        index: usize,
        budget: i64,
    ) -> Result<Option<i64>> {
        let ctx = self.view.ctx;
        let initial = trace(ctx, operand(ctx, parts.input, index)?)?;
        let Some(mut current) = self.instruction(initial, &[], EvaluationStage::InitialValue)?
        else {
            return Ok(None);
        };
        if let Some(count) = self.trivial_count(parts, index, &current)? {
            return Ok(Some(count));
        }
        let update = operand(ctx, parts.body_root, index)?;
        let Some(body_gte) = nonconstant_operand(ctx, update)? else {
            return Ok(None);
        };
        let Some(cond_gte) = nonconstant_operand(ctx, parts.condition_root)? else {
            return Ok(None);
        };
        if unique_gte(ctx, parts.body_parameter, index)? != Some(body_gte)
            || unique_gte(ctx, parts.condition_parameter, index)? != Some(cond_gte)
        {
            return Ok(None);
        }
        for count in 0..=budget {
            let Some(condition) = self.instruction(
                parts.condition_root,
                &[(cond_gte, current.clone())],
                EvaluationStage::Condition,
            )?
            else {
                return Ok(None);
            };
            if !literal_semantics::scalar_predicate(ctx, &condition.into_literal(ctx)?)? {
                return Ok(Some(count));
            }
            // Upstream evaluates this even after a true condition at budget.
            let Some(next) =
                self.instruction(update, &[(body_gte, current)], EvaluationStage::Update)?
            else {
                return Ok(None);
            };
            current = next;
        }
        Ok(None)
    }

    fn trivial_count(
        &mut self,
        parts: &Parts,
        index: usize,
        initial: &EvaluatedValue,
    ) -> Result<Option<i64>> {
        let ctx = self.view.ctx;
        let Some(init) = integer(ctx, initial)? else {
            return Ok(None);
        };
        let update = operand(ctx, parts.body_root, index)?;
        let update = trace(ctx, update)?;
        let Some(gte) = nonconstant_operand(ctx, update)? else {
            return Ok(None);
        };
        if Some(trace(ctx, gte)?) != unique_gte(ctx, parts.body_parameter, index)?
            || !is_opcode(ctx, update, Opcode::Add)?
        {
            return Ok(None);
        }
        let operands = inputs(ctx, defining(update)?);
        let constant = if operands[0] == gte {
            operands[1]
        } else {
            operands[0]
        };
        if !is_opcode(ctx, constant, Opcode::Constant)? {
            return Ok(None);
        }
        let Some(step) = self.instruction(constant, &[], EvaluationStage::Update)? else {
            return Ok(None);
        };
        let Some(step) = integer(ctx, &step)?.filter(|step| *step > 0) else {
            return Ok(None);
        };
        let condition = parts.condition_root;
        let Some(gte) = nonconstant_operand(ctx, condition)? else {
            return Ok(None);
        };
        if Some(gte) != unique_gte(ctx, parts.condition_parameter, index)?
            || !is_opcode(ctx, condition, Opcode::Compare)?
        {
            return Ok(None);
        }
        let operation = defining(condition)?;
        let operands = inputs(ctx, operation);
        if operands[0] != gte || !is_opcode(ctx, operands[1], Opcode::Constant)? {
            return Ok(None);
        }
        let direction = attr::<Direction>(ctx, operation, "hlo_direction")?;
        let Some(bound) = self.instruction(operands[1], &[], EvaluationStage::Condition)? else {
            return Ok(None);
        };
        let Some(bound) = integer(ctx, &bound)? else {
            return Ok(None);
        };
        let element = shape::element_type(ctx, initial.shape())?;
        Ok(checked_count(init, bound, step, direction, element))
    }

    pub(super) fn upper(&mut self, parts: &Parts) -> Result<Option<i64>> {
        let ctx = self.view.ctx;
        let users: HashSet<_> = parts
            .condition_parameter
            .uses(ctx)
            .iter()
            .map(|use_| use_.user_op())
            .collect();
        if users.len() != 1 {
            return Ok(None);
        }
        let Some(&gte) = users.iter().next() else {
            return Ok(None);
        };
        if Operation::is_op::<ReturnOp>(gte, ctx)
            || opcode(ctx, gte)? != Opcode::GetTupleElement
            || !is_opcode(ctx, parts.body_root, Opcode::Tuple)?
        {
            return Ok(None);
        }
        let index = attr::<Index>(ctx, gte, "hlo_index")?.0 as usize;
        let constant = operand(ctx, parts.body_root, index)?;
        if !is_opcode(ctx, constant, Opcode::Constant)? {
            return Ok(None);
        }
        let literal = attr::<crate::literal::Literal>(ctx, defining(constant)?, "hlo_literal")?;
        let shape = parts.condition_parameter.get_type(ctx);
        let mut children = Vec::new();
        for (position, &ty) in shape::children(ctx, shape).iter().enumerate() {
            children.push(if position == index {
                EvaluatedValue::known(ctx, literal.clone())?
            } else {
                EvaluatedValue::unknown(ctx, ty)?
            });
        }
        let argument = EvaluatedValue::tuple(ctx, shape, children)?;
        let Some(condition) = self.evaluate(
            Request::Computation {
                callee: parts.condition,
                args: &[argument],
            },
            EvaluationStage::UpperBound,
        )?
        else {
            return Ok(None);
        };
        Ok(
            (!literal_semantics::scalar_predicate(ctx, &condition.into_literal(ctx)?)?)
                .then_some(1),
        )
    }
}

pub(super) fn checked_count(
    init: i64,
    bound: i64,
    step: i64,
    direction: Direction,
    element: crate::types::ElementType,
) -> Option<i64> {
    if step <= 0 || !element.is_integer() {
        return None;
    }
    let distance = match direction {
        Direction::Lt if bound <= init => return Some(0),
        Direction::Le if bound < init => return Some(0),
        Direction::Lt | Direction::Le => bound.checked_sub(init)?,
        _ => return None,
    };
    let count = if direction == Direction::Lt {
        (distance - 1) / step + 1
    } else {
        (distance / step).checked_add(1)?
    };
    let final_value =
        i128::from(init).checked_add(i128::from(count).checked_mul(i128::from(step))?)?;
    let width = element.bit_width();
    let (minimum, maximum) = if element.is_signed() {
        (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
    } else {
        (0, (1i128 << width) - 1)
    };
    if final_value < minimum || final_value > maximum || i64::try_from(final_value).is_err() {
        return None;
    }
    Some(count)
}
