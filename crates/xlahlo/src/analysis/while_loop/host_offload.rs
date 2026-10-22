//! Host-offload source candidates, not origin or execution proofs.
use super::affine::SourceAffine;
use super::*;
use pliron::builtin::attributes::StringAttr;

/// A source slice-index candidate; the GTE need not extract the body parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceDynamicVariable {
    /// Loop-state tuple element referenced by dynamic slice indices in the
    /// host-offload pattern.
    pub tuple_index: usize,
    /// Initial integration resolves self-affine patterns only.
    pub affine: Option<SourceAffine>,
}

fn gate(ctx: &Context, instruction: Ptr<Operation>) -> Result<bool> {
    if opcode(ctx, instruction)? != Opcode::CustomCall {
        return Ok(false);
    }
    let target = attr::<StringAttr>(ctx, instruction, "hlo_target")?;
    let (forward, slice) = match target.as_str() {
        "MoveToHost" => (true, Opcode::DynamicUpdateSlice),
        "MoveToDevice" => (false, Opcode::DynamicSlice),
        _ => return Ok(false),
    };
    let mut pending = vec![instruction.deref(ctx).get_result(0)];
    let mut visited = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        let neighbors: Vec<Value> = if forward {
            value
                .uses(ctx)
                .iter()
                .map(|use_| use_.user_op())
                .filter(|&user| !Operation::is_op::<ReturnOp>(user, ctx))
                .map(|user| user.deref(ctx).get_result(0))
                .collect()
        } else {
            inputs(ctx, defining(value)?)
        };
        for neighbor in neighbors {
            let Some(operation) = neighbor.defining_op() else {
                continue;
            };
            match opcode(ctx, operation)? {
                code if code == slice => return Ok(true),
                Opcode::Reshape | Opcode::Broadcast | Opcode::Transpose => {
                    pending.push(neighbor);
                }
                _ => {}
            }
        }
    }
    Ok(false)
}

fn collect_indices(ctx: &Context, value: Value, indices: &mut HashSet<usize>) -> Result<()> {
    let mut pending = vec![value];
    let mut visited = HashSet::new();
    while let Some(value) = pending.pop() {
        if !visited.insert(value) {
            continue;
        }
        let Some(operation) = value.defining_op() else {
            continue;
        };
        let operands = inputs(ctx, operation);
        match opcode(ctx, operation)? {
            Opcode::GetTupleElement => {
                indices.insert(attr::<Index>(ctx, operation, "hlo_index")?.0 as usize);
            }
            Opcode::Copy if operands.len() == 1 => pending.extend(operands),
            Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide => {
                pending.extend(operands);
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn dynamic_variables(
    view: &ModuleView<'_>,
    parts: &Parts,
) -> Result<Vec<SourceDynamicVariable>> {
    let ctx = view.ctx;
    let owner = defining(parts.body_root)?
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| invalid("While body root has no computation"))?;
    let instructions: Vec<_> = block(ctx, owner)?.deref(ctx).iter(ctx).collect();
    let mut enabled = false;
    for &instruction in &instructions {
        if !Operation::is_op::<ReturnOp>(instruction, ctx) && gate(ctx, instruction)? {
            enabled = true;
            break;
        }
    }
    if !enabled {
        return Ok(Vec::new());
    }
    let mut indices = HashSet::new();
    for instruction in instructions {
        if Operation::is_op::<ReturnOp>(instruction, ctx) {
            continue;
        }
        let first_index = match opcode(ctx, instruction)? {
            Opcode::DynamicSlice => 1,
            Opcode::DynamicUpdateSlice => 2,
            _ => continue,
        };
        for value in inputs(ctx, instruction).into_iter().skip(first_index) {
            collect_indices(ctx, value, &mut indices)?;
        }
    }
    let mut indices: Vec<_> = indices.into_iter().collect();
    indices.sort_unstable();
    indices
        .into_iter()
        .map(|tuple_index| {
            Ok(SourceDynamicVariable {
                tuple_index,
                affine: super::affine::resolve(view, parts, tuple_index, None)?,
            })
        })
        .collect()
}
