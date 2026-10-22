//! Pinned auxiliary-induction source matching, not affine analysis.
use super::*;
use std::collections::BTreeMap;

/// A source-body extraction candidate, not a scalar or execution proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceAuxiliaryInduction {
    /// Tuple element whose body output depends on this extraction.
    pub tuple_index: usize,
    /// Body-parameter `GetTupleElement` that anchors the matched dependency
    /// chain; this is a source-shape candidate, not a proof of recurrence.
    pub extraction: Value,
}

pub(super) fn candidates(
    view: &ModuleView<'_>,
    parts: &Parts,
) -> Result<Vec<SourceAuxiliaryInduction>> {
    let ctx = view.ctx;
    if !is_opcode(ctx, parts.body_root, Opcode::Tuple)? {
        return Ok(Vec::new());
    }
    let users: HashSet<_> = parts
        .body_parameter
        .uses(ctx)
        .iter()
        .map(|use_| use_.user_op())
        .collect();
    let mut extractions = BTreeMap::new();
    for user in users {
        if Operation::is_op::<ReturnOp>(user, ctx) || opcode(ctx, user)? != Opcode::GetTupleElement
        {
            continue;
        }
        let index = attr::<Index>(ctx, user, "hlo_index")?.0 as usize;
        extractions
            .entry(index)
            .and_modify(|value| *value = None)
            .or_insert(Some(user.deref(ctx).get_result(0)));
    }
    let outputs = inputs(ctx, defining(parts.body_root)?);
    let mut result = Vec::new();
    for (index, extraction) in extractions {
        let (Some(extraction), Some(&output)) = (extraction, outputs.get(index)) else {
            continue;
        };
        if output == extraction || is_opcode(ctx, output, Opcode::Constant)? {
            continue;
        }
        let mut current = output;
        let mut visited = HashSet::new();
        while visited.insert(current) {
            if current == extraction {
                result.push(SourceAuxiliaryInduction {
                    tuple_index: index,
                    extraction,
                });
                break;
            }
            let Some(operation) = current.defining_op() else {
                break;
            };
            if !matches!(
                opcode(ctx, operation)?,
                Opcode::GetTupleElement
                    | Opcode::Add
                    | Opcode::Subtract
                    | Opcode::Multiply
                    | Opcode::Divide
            ) {
                break;
            }
            let mut next = None;
            let mut occurrences = 0;
            for value in inputs(ctx, operation) {
                if !is_opcode(ctx, value, Opcode::Constant)? {
                    occurrences += 1;
                    next = Some(value);
                }
            }
            if occurrences != 1 {
                break;
            }
            let Some(value) = next else {
                break;
            };
            current = value;
        }
    }
    Ok(result)
}
