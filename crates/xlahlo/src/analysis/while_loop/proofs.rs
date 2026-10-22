//! Checked recurrence proofs, separate from source metadata.
use super::*;
use crate::{
    analysis::effects::SideEffects,
    ir::instruction_order::{control_predecessors, control_successors},
    types::ElementType,
};
use pliron::pass::{Analysis as _, AnalysisManager};

/// A finite, nonnegative, direct constant-bound induction proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvenPrimaryInduction {
    tuple_index: usize,
    init: i64,
    step: i64,
    trip_count: i64,
    final_value: Scalar,
}

impl ProvenPrimaryInduction {
    /// Return the carried tuple element proved to control the loop condition.
    pub fn tuple_index(self) -> usize {
        self.tuple_index
    }
    /// Return the checked initial scalar value.
    pub fn init(self) -> i64 {
        self.init
    }
    /// Return the positive per-iteration increment.
    pub fn step(self) -> i64 {
        self.step
    }
    /// Return the exact positive number of body executions.
    pub fn trip_count(self) -> i64 {
        self.trip_count
    }
    /// Return the representable carried value after the final body execution.
    pub fn final_value(self) -> Scalar {
        self.final_value
    }
}

/// Checked coefficients for a genuine loop-carried scalar over the exact count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvenDynamicVariable {
    tuple_index: usize,
    init: i64,
    step: i64,
}

impl ProvenDynamicVariable {
    /// Return the carried tuple element whose recurrence was proved.
    pub fn tuple_index(self) -> usize {
        self.tuple_index
    }
    /// Return the checked initial scalar value.
    pub fn init(self) -> i64 {
        self.init
    }
    /// Return the checked per-iteration additive offset.
    pub fn step(self) -> i64 {
        self.step
    }
}

fn scalar_integer_type(ctx: &Context, value: Value) -> Result<Option<ElementType>> {
    let shape = value.get_type(ctx);
    if !shape::is_array(ctx, shape)
        || shape::is_dynamic(ctx, shape)
        || shape::rank(ctx, shape)? != 0
    {
        return Ok(None);
    }
    let element = shape::element_type(ctx, shape)?;
    Ok(element.is_integer().then_some(element))
}

fn number(value: Scalar) -> Option<i64> {
    let width = value.element().bit_width();
    let bits = u64::try_from(value.bits()).ok()?;
    if value.element().is_signed() {
        Some(((bits << (64 - width)) as i64) >> (64 - width))
    } else {
        i64::try_from(bits).ok()
    }
}

fn fits(element: ElementType, value: i128) -> bool {
    let width = element.bit_width();
    let (min, max) = if element.is_signed() {
        (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
    } else {
        (0, (1i128 << width) - 1)
    };
    (min..=max).contains(&value)
}

fn encoded(element: ElementType, value: i64) -> Result<Scalar> {
    Scalar::from_bits(
        element,
        (value as u128) & ((1u128 << element.bit_width()) - 1),
    )
}

fn direct_constant(ctx: &Context, value: Value, element: ElementType) -> Result<Option<i64>> {
    if scalar_integer_type(ctx, value)? != Some(element)
        || !is_opcode(ctx, value, Opcode::Constant)?
    {
        return Ok(None);
    }
    let literal = attr::<crate::literal::Literal>(ctx, defining(value)?, "hlo_literal")?;
    Ok(number(Scalar::read(ctx, &literal, &[])?))
}

fn controlled(ctx: &Context, value: Value) -> Result<bool> {
    Ok(
        !control_predecessors(ctx, value)?.is_empty()
            || !control_successors(ctx, value)?.is_empty(),
    )
}

fn pure_uncontrolled(view: &ModuleView<'_>, parts: &Parts, effects: &SideEffects) -> Result<bool> {
    let ctx = view.ctx;
    for parameter in [parts.body_parameter, parts.condition_parameter] {
        let owner = defining(parameter)?
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| invalid("proof parameter has no computation"))?;
        if effects.has_side_effect(owner)? {
            return Ok(false);
        }
        let mut pending = vec![owner];
        let mut visited = HashSet::new();
        while let Some(operation) = pending.pop() {
            if !visited.insert(operation) {
                continue;
            }
            if !Operation::is_op::<ComputationOp>(operation, ctx)
                && !Operation::is_op::<ReturnOp>(operation, ctx)
                && controlled(ctx, operation.deref(ctx).get_result(0))?
            {
                return Ok(false);
            }
            if let Some(names) = optional_attr::<Callees>(ctx, operation, "hlo_callees")? {
                for name in &names.0 {
                    pending.push(view.resolve(name)?);
                }
            }
            for region in operation.deref(ctx).regions() {
                for bb in region.deref(ctx).iter(ctx) {
                    pending.extend(bb.deref(ctx).iter(ctx));
                }
            }
        }
    }
    Ok(true)
}

fn genuine_indices(view: &ModuleView<'_>, parts: &Parts, index: usize) -> Result<bool> {
    let ctx = view.ctx;
    let owner = defining(parts.body_root)?
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| invalid("proof body root has no computation"))?;
    let mut pending = Vec::new();
    for operation in block(ctx, owner)?.deref(ctx).iter(ctx) {
        if Operation::is_op::<ReturnOp>(operation, ctx) {
            continue;
        }
        let first = match opcode(ctx, operation)? {
            Opcode::DynamicSlice => 1,
            Opcode::DynamicUpdateSlice => 2,
            _ => continue,
        };
        pending.extend(inputs(ctx, operation).into_iter().skip(first));
    }
    let mut found = false;
    let mut visited = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        let Some(operation) = value.defining_op() else {
            continue;
        };
        match opcode(ctx, operation)? {
            Opcode::GetTupleElement
                if attr::<Index>(ctx, operation, "hlo_index")?.0 as usize == index =>
            {
                if operation.deref(ctx).get_operand(0) != parts.body_parameter {
                    return Ok(false);
                }
                found = true;
            }
            Opcode::Copy | Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide => {
                pending.extend(inputs(ctx, operation))
            }
            _ => {}
        }
    }
    Ok(found)
}

pub(super) fn primary(
    view: &ModuleView<'_>,
    parts: &Parts,
    index: Option<usize>,
    exact: Option<i64>,
    range: Option<SourceLoopRange>,
) -> Result<Option<ProvenPrimaryInduction>> {
    let (Some(index), Some(count), Some(range)) = (index, exact, range) else {
        return Ok(None);
    };
    if count <= 0 {
        return Ok(None);
    }
    let ctx = view.ctx;
    let update = operand(ctx, parts.body_root, index)?;
    let Some(element) = scalar_integer_type(ctx, update)? else {
        return Ok(None);
    };
    let Some(body_iv) = unique_gte(ctx, parts.body_parameter, index)? else {
        return Ok(None);
    };
    let Some(cond_iv) = unique_gte(ctx, parts.condition_parameter, index)? else {
        return Ok(None);
    };
    if scalar_integer_type(ctx, body_iv)? != Some(element)
        || scalar_integer_type(ctx, cond_iv)? != Some(element)
        || !is_opcode(ctx, update, Opcode::Add)?
        || !is_opcode(ctx, parts.condition_root, Opcode::Compare)?
    {
        return Ok(None);
    }
    let update_args = inputs(ctx, defining(update)?);
    let increment = if update_args[0] == body_iv {
        update_args[1]
    } else if update_args[1] == body_iv {
        update_args[0]
    } else {
        return Ok(None);
    };
    let Some(step) = direct_constant(ctx, increment, element)?.filter(|value| *value > 0) else {
        return Ok(None);
    };
    let cond = defining(parts.condition_root)?;
    let condition_args = inputs(ctx, cond);
    if condition_args[0] != cond_iv
        || !matches!(
            attr::<Direction>(ctx, cond, "hlo_direction")?,
            Direction::Lt | Direction::Le
        )
        || direct_constant(ctx, condition_args[1], element)?.is_none()
    {
        return Ok(None);
    }
    if range.start.element() != element
        || range.inclusive_end.element() != element
        || range.step.element() != ElementType::S64
    {
        return Ok(None);
    }
    let (Some(init), Some(end), Some(range_step)) = (
        number(range.start),
        number(range.inclusive_end),
        number(range.step),
    ) else {
        return Ok(None);
    };
    if init < 0 || end < init || step != range_step {
        return Ok(None);
    }
    let Some(expected) = end
        .checked_sub(init)
        .and_then(|distance| (distance / step).checked_add(1))
    else {
        return Ok(None);
    };
    if count != expected {
        return Ok(None);
    }
    let final_wide = i128::from(init) + i128::from(step) * i128::from(count);
    if !fits(element, final_wide) {
        return Ok(None);
    }
    let Ok(final_value) = i64::try_from(final_wide) else {
        return Ok(None);
    };
    let initial = trace(ctx, operand(ctx, parts.input, index)?)?;
    if scalar_integer_type(ctx, initial)? != Some(element) || controlled(ctx, initial)? {
        return Ok(None);
    }
    let effects = SideEffects::compute(view.module, ctx, &mut AnalysisManager::default())?;
    if !pure_uncontrolled(view, parts, &effects)? {
        return Ok(None);
    }
    Ok(Some(ProvenPrimaryInduction {
        tuple_index: index,
        init,
        step,
        trip_count: count,
        final_value: encoded(element, final_value)?,
    }))
}

// Same-format Add/Sub are modular-ring operations. Internal wrap/cancellation
// preserves the affine result; only the carried states must fit without wrap.
fn recurrence_source(
    ctx: &Context,
    expression: Value,
    parameter: Value,
    element: ElementType,
) -> Result<Option<usize>> {
    let mut pending = vec![expression];
    let mut source = None;
    while let Some(value) = pending.pop() {
        if scalar_integer_type(ctx, value)? != Some(element) {
            return Ok(None);
        }
        let Some(operation) = value.defining_op() else {
            return Ok(None);
        };
        match opcode(ctx, operation)? {
            Opcode::GetTupleElement => {
                if operation.deref(ctx).get_operand(0) != parameter || source.is_some() {
                    return Ok(None);
                }
                source = Some(attr::<Index>(ctx, operation, "hlo_index")?.0 as usize);
            }
            Opcode::Constant => {
                if direct_constant(ctx, value, element)?.is_none() {
                    return Ok(None);
                }
            }
            Opcode::Add | Opcode::Subtract => pending.extend(inputs(ctx, operation)),
            _ => return Ok(None),
        }
    }
    Ok(source)
}

pub(super) fn dynamics(
    view: &ModuleView<'_>,
    parts: &Parts,
    exact: Option<i64>,
    candidates: &[SourceDynamicVariable],
    primary: Option<ProvenPrimaryInduction>,
) -> Result<Vec<ProvenDynamicVariable>> {
    let Some(count) = exact.filter(|count| *count >= 0) else {
        return Ok(Vec::new());
    };
    let ctx = view.ctx;
    let effects = SideEffects::compute(view.module, ctx, &mut AnalysisManager::default())?;
    if !pure_uncontrolled(view, parts, &effects)? {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for candidate in candidates {
        let index = candidate.tuple_index;
        if !genuine_indices(view, parts, index)? {
            continue;
        }
        let Some(update) = inputs(ctx, defining(parts.body_root)?).get(index).copied() else {
            continue;
        };
        let Some(element) = scalar_integer_type(ctx, update)? else {
            continue;
        };
        let Some(source) = recurrence_source(ctx, update, parts.body_parameter, element)? else {
            continue;
        };
        if source != index && primary.is_none_or(|primary| primary.tuple_index != source) {
            continue;
        }
        let trusted_primary = primary.map(|primary| super::affine::PrimaryAffine {
            tuple_index: primary.tuple_index,
            init: primary.init,
            step: primary.step,
        });
        let Some(coefficients) = super::affine::resolve(view, parts, index, trusted_primary)?
        else {
            continue;
        };
        let initial = operand(ctx, parts.input, index)?;
        if direct_constant(ctx, initial, element)? != Some(coefficients.init)
            || controlled(ctx, initial)?
        {
            continue;
        }
        if source != index && primary.is_none_or(|primary| primary.final_value.element() != element)
        {
            continue;
        }
        let final_value =
            i128::from(coefficients.init) + i128::from(coefficients.step) * i128::from(count);
        if !fits(element, i128::from(coefficients.init)) || !fits(element, final_value) {
            continue;
        }
        result.push(ProvenDynamicVariable {
            tuple_index: index,
            init: coefficients.init,
            step: coefficients.step,
        });
    }
    Ok(result)
}
