//! Existing replacement policies on the creation-capable common edit plan.
//!
//! Raw editing, alias policy, schedule repair and callback diagnostics stay in
//! their existing owner; this only shares rehearsal/publication machinery.
use super::{
    Edit, InstructionReplacement, alias_owners, containing_module, endpoint_block, operation_tree,
    remove_instruction_impl, replace_instruction_impl, require_rehearsable_policy,
    restore_alias_owners, result,
};
use crate::verification::{VerifyOptions, error};
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
};
use std::collections::HashMap;

pub(super) fn verified_edit(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    edit: Edit,
    verification: VerifyOptions,
) -> Result<InstructionReplacement> {
    require_rehearsable_policy(verification)?;
    let module = containing_module(ctx, endpoint_block(ctx, result(ctx, instruction)?)?)?;
    super::call_inlining::edit_staging::staged_edit_if(
        ctx,
        module,
        Some(verification),
        |ctx, scratch, mapping| {
            let reverse: HashMap<_, _> = operation_tree(ctx, module)
                .into_iter()
                .map(|original| {
                    mapping
                        .lookup_op(original)
                        .map(|copy| (copy, original))
                        .ok_or_else(|| error("edit lost an original operation"))
                })
                .collect::<Result<_>>()?;
            let mut aliases = alias_owners(ctx, &operation_tree(ctx, scratch))?;
            let instruction = mapping
                .lookup_op(instruction)
                .ok_or_else(|| error("edit lost the instruction"))?;
            let mut outcome = match edit {
                Edit::Replace(value, options) => replace_instruction_impl(
                    ctx,
                    instruction,
                    mapping
                        .lookup_value(value)
                        .ok_or_else(|| error("replacement is outside the module"))?,
                    options,
                )?,
                Edit::Forward(value) => {
                    let value = mapping
                        .lookup_value(value)
                        .ok_or_else(|| error("replacement is outside the module"))?;
                    let original = result(ctx, instruction)?;
                    for snapshot in &mut aliases {
                        for owner in &mut snapshot.owners {
                            if *owner == original {
                                *owner = value;
                            }
                        }
                    }
                    super::forwarding::replace_all_uses_impl(ctx, instruction, value)?
                }
                Edit::Remove(ignore_controls) => InstructionReplacement {
                    changed: true,
                    removed: remove_instruction_impl(ctx, instruction, ignore_controls)?,
                },
            };
            if outcome.changed {
                restore_alias_owners(ctx, aliases, &outcome.removed)?;
                crate::ir::schedule_update::update_schedule(ctx, scratch)?;
            }
            outcome.removed = outcome
                .removed
                .iter()
                .map(|operation| {
                    reverse
                        .get(operation)
                        .copied()
                        .ok_or_else(|| error("removed instruction has no original identity"))
                })
                .collect::<Result<_>>()?;
            Ok(outcome)
        },
        |result| result.changed,
        |result, _| Ok(result),
    )
}
