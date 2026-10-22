//! Structural affine metadata with checked host-integer offsets.
use super::*;

/// Source-pattern coefficients, not a typed recurrence or termination proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceAffine {
    /// Integer initial value matched in the While input tuple.
    pub init: i64,
    /// Checked source-pattern offset per iteration. This coefficient is
    /// structural metadata and does not establish runtime integer semantics.
    pub step: i64,
}

/// Trusted helper input only; never populated from source-range metadata.
#[derive(Clone, Copy)]
pub(super) struct PrimaryAffine {
    pub tuple_index: usize,
    pub init: i64,
    pub step: i64,
}

#[derive(Clone, Copy)]
struct LinearTerm {
    source: Option<usize>,
    offset: i64,
}

fn constant_integer(ctx: &Context, value: Value) -> Result<Option<i64>> {
    if !is_opcode(ctx, value, Opcode::Constant)? {
        return Ok(None);
    }
    let literal = attr::<crate::literal::Literal>(ctx, defining(value)?, "hlo_literal")?;
    let shape = literal.shape;
    if !shape::is_array(ctx, shape)
        || shape::is_dynamic(ctx, shape)
        || shape::element_count(ctx, shape)? != 1
        || !shape::element_type(ctx, shape)?.is_integer()
    {
        return Ok(None);
    }
    let scalar = Scalar::read(ctx, &literal, &vec![0; shape::rank(ctx, shape)?])?;
    if !scalar.element().is_signed() && scalar.bits() > i64::MAX as u128 {
        return Ok(None);
    }
    literal_semantics::scalar_integer(ctx, &literal).map(Some)
}

fn linear(ctx: &Context, expression: Value, parameter: Value) -> Result<Option<LinearTerm>> {
    let mut pending = vec![(expression, false)];
    let mut active = HashSet::new();
    let mut terms: Vec<Option<LinearTerm>> = Vec::new();
    while let Some((value, expanded)) = pending.pop() {
        let Some(operation) = value.defining_op() else {
            terms.push(None);
            continue;
        };
        let code = opcode(ctx, operation)?;
        let operands = inputs(ctx, operation);
        if !expanded {
            let term = match code {
                Opcode::GetTupleElement if operands.first() == Some(&parameter) => {
                    Some(LinearTerm {
                        source: Some(attr::<Index>(ctx, operation, "hlo_index")?.0 as usize),
                        offset: 0,
                    })
                }
                Opcode::Constant => constant_integer(ctx, value)?.map(|offset| LinearTerm {
                    source: None,
                    offset,
                }),
                Opcode::Add | Opcode::Subtract => {
                    if !active.insert(value) {
                        return Err(invalid("cycle in affine source expression"));
                    }
                    pending.push((value, true));
                    for &operand in operands.iter().rev() {
                        pending.push((operand, false));
                    }
                    continue;
                }
                _ => None,
            };
            terms.push(term);
            continue;
        }
        active.remove(&value);
        let rhs = terms
            .pop()
            .ok_or_else(|| invalid("affine source walk lost its right operand"))?;
        let lhs = terms
            .pop()
            .ok_or_else(|| invalid("affine source walk lost its left operand"))?;
        let term = match (lhs, rhs) {
            (Some(lhs), Some(rhs))
                if !(rhs.source.is_some()
                    && (lhs.source.is_some() || code == Opcode::Subtract)) =>
            {
                let offset = if code == Opcode::Subtract {
                    lhs.offset.checked_sub(rhs.offset)
                } else {
                    lhs.offset.checked_add(rhs.offset)
                };
                offset.map(|offset| LinearTerm {
                    source: lhs.source.or(rhs.source),
                    offset,
                })
            }
            _ => None,
        };
        terms.push(term);
    }
    if terms.len() != 1 {
        return Err(invalid("affine source walk has an invalid result count"));
    }
    Ok(terms.pop().flatten())
}

pub(super) fn resolve(
    view: &ModuleView<'_>,
    parts: &Parts,
    index: usize,
    primary: Option<PrimaryAffine>,
) -> Result<Option<SourceAffine>> {
    let ctx = view.ctx;
    if !is_opcode(ctx, parts.input, Opcode::Tuple)?
        || !is_opcode(ctx, parts.body_root, Opcode::Tuple)?
    {
        return Ok(None);
    }
    let Some(update) = inputs(ctx, defining(parts.body_root)?).get(index).copied() else {
        return Ok(None);
    };
    let Some(term) = linear(ctx, update, parts.body_parameter)? else {
        return Ok(None);
    };
    let Some(source) = term.source else {
        return Ok(None);
    };
    let Some(initial) = inputs(ctx, defining(parts.input)?).get(index).copied() else {
        return Ok(None);
    };
    let Some(init) = constant_integer(ctx, initial)? else {
        return Ok(None);
    };
    if source == index {
        return Ok(Some(SourceAffine {
            init,
            step: term.offset,
        }));
    }
    let Some(primary) = primary.filter(|primary| primary.tuple_index == source) else {
        return Ok(None);
    };
    let Some(derived) = primary
        .init
        .checked_sub(primary.step)
        .and_then(|value| value.checked_add(term.offset))
    else {
        return Ok(None);
    };
    Ok((derived == init).then_some(SourceAffine {
        init,
        step: primary.step,
    }))
}
