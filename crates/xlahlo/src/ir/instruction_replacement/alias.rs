//! Fusion output/operand alias owner capture and restoration around edits.

use super::*;

pub(super) struct AliasOwners {
    pub(super) fusion: Ptr<Operation>,
    pub(super) aliases: OutputOperandAliases,
    pub(super) owners: Vec<Value>,
}

pub(super) fn alias_owners(
    ctx: &Context,
    operations: &[Ptr<Operation>],
) -> Result<Vec<AliasOwners>> {
    let mut snapshots = Vec::new();
    for &fusion in operations {
        if !Operation::is_op::<FusionOp>(fusion, ctx) {
            continue;
        }
        let Some(aliases) = optional_attr::<OutputOperandAliases>(ctx, fusion, "hlo_aliases")?
            .as_deref()
            .cloned()
        else {
            continue;
        };
        let owners = aliases
            .0
            .iter()
            .map(|alias| {
                let index = usize::try_from(alias.operand_index)
                    .map_err(|_| error("negative fusion alias operand"))?;
                require(
                    index < fusion.deref(ctx).get_num_operands(),
                    "fusion alias operand is outside caller",
                )?;
                Ok(fusion.deref(ctx).get_operand(index))
            })
            .collect::<Result<Vec<_>>>()?;
        snapshots.push(AliasOwners {
            fusion,
            aliases,
            owners,
        });
    }
    Ok(snapshots)
}

pub(super) fn restore_alias_owners(
    ctx: &Context,
    snapshots: Vec<AliasOwners>,
    removed: &HashSet<Ptr<Operation>>,
) -> Result<()> {
    for mut snapshot in snapshots {
        if removed.contains(&snapshot.fusion) {
            continue;
        }
        for (alias, owner) in snapshot.aliases.0.iter_mut().zip(snapshot.owners) {
            let fusion = snapshot.fusion.deref(ctx);
            let previous = usize::try_from(alias.operand_index)
                .map_err(|_| error("negative fusion alias operand"))?;
            // Custom fusions retain duplicate operands and their alias slots.
            let index =
                if previous < fusion.get_num_operands() && fusion.get_operand(previous) == owner {
                    Some(previous)
                } else {
                    fusion.operands().position(|operand| operand == owner)
                }
                .ok_or_else(|| error("verified edit would remove a fusion alias owner"))?;
            alias.operand_index =
                i64::try_from(index).map_err(|_| error("fusion alias operand index overflow"))?;
        }
        snapshot
            .fusion
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_aliases"), snapshot.aliases);
    }
    Ok(())
}

pub(super) fn verified_edit(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    edit: Edit,
    verification: VerifyOptions,
) -> Result<InstructionReplacement> {
    creation_staging::verified_edit(ctx, instruction, edit, verification)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{alias_fixture, forwarding_options, has_order, rich_order};
    use super::*;
    use crate::verification::verify_with_options;
    use pliron::printable::Printable;

    fn alias_owner(ctx: &Context, fusion: Ptr<Operation>) -> Value {
        let aliases = optional_attr::<OutputOperandAliases>(ctx, fusion, "hlo_aliases")
            .unwrap()
            .unwrap();
        fusion
            .deref(ctx)
            .get_operand(aliases.0[0].operand_index as usize)
    }

    #[test]
    fn raw_direct_alias_owner_removal_succeeds_but_final_verification_fails() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 0, 2);
        let owner = rich_order(ctx, f.body);
        let edit =
            replace_instruction(ctx, f.projection, f.selected, forwarding_options(), None).unwrap();
        assert!(edit.changed);
        assert!(!has_order(ctx, owner));
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 1);
        assert_eq!(
            optional_attr::<OutputOperandAliases>(ctx, f.fusion, "hlo_aliases")
                .unwrap()
                .unwrap()
                .0[0]
                .operand_index,
            1
        );
        assert!(verify_with_options(ctx, f.outer.module, VerifyOptions::default()).is_err());
    }

    #[test]
    fn raw_shifted_alias_can_silently_rebind_to_a_different_same_shaped_owner() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 1, 3);
        let owner = alias_owner(ctx, f.fusion);
        replace_instruction(ctx, f.projection, f.selected, forwarding_options(), None).unwrap();
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 2);
        assert_ne!(alias_owner(ctx, f.fusion), owner);
        assert_eq!(alias_owner(ctx, f.fusion), f.outer.parameters[2]);
        verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_shifted_alias_preserves_owner_and_surviving_native_handles() {
        for inputs in [2, 3] {
            let ctx = &mut Context::new();
            let f = alias_fixture(ctx, 1, inputs);
            let owner = alias_owner(ctx, f.fusion);
            let edit = replace_instruction(
                ctx,
                f.projection,
                f.selected,
                forwarding_options(),
                Some(VerifyOptions::default()),
            )
            .unwrap();
            assert!(edit.changed);
            assert!(edit.removed.contains(&f.projection));
            assert_eq!(alias_owner(ctx, f.fusion), owner);
            assert_eq!(f.fusion.deref(ctx).get_num_operands(), inputs - 1);
            assert_eq!(f.body.deref(ctx).get_argument(0), f.selected);
            assert_eq!(
                fusion_parameters(ctx, f.body).unwrap().0[0].name.as_deref(),
                Some("arg1")
            );
            assert_eq!(f.fusion.deref(ctx).get_parent_block(), Some(f.outer.block));
            verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
        }
    }

    #[test]
    fn verified_alias_owner_loss_repeatedly_rejects_without_mutation_and_recovers() {
        for _ in 0..2 {
            let ctx = &mut Context::new();
            let f = alias_fixture(ctx, 0, 2);
            let owner = rich_order(ctx, f.body);
            let before = f.outer.module.disp(ctx).to_string();
            let values: Vec<_> = f
                .outer
                .parameters
                .iter()
                .copied()
                .chain(f.body.deref(ctx).arguments())
                .chain(std::iter::once(result(ctx, f.projection).unwrap()))
                .collect();
            let uses: Vec<_> = values.iter().map(|value| value.uses(ctx)).collect();
            for _ in 0..3 {
                let failure = replace_instruction(
                    ctx,
                    f.projection,
                    f.selected,
                    forwarding_options(),
                    Some(VerifyOptions::default()),
                )
                .err()
                .expect("lost alias owner must be rejected");
                assert!(failure.to_string().contains("fusion alias owner"));
                assert_eq!(f.outer.module.disp(ctx).to_string(), before);
                assert!(has_order(ctx, owner));
                assert!(
                    values
                        .iter()
                        .map(|value| value.uses(ctx))
                        .collect::<Vec<_>>()
                        == uses
                );
                assert_eq!(f.projection.deref(ctx).get_parent_block(), Some(f.body));
                verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
            }
            // The caller changes its contract explicitly; the edit never drops it.
            f.fusion
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&Identifier::new("hlo_aliases"));
            replace_instruction(
                ctx,
                f.projection,
                f.selected,
                forwarding_options(),
                Some(VerifyOptions::default()),
            )
            .unwrap();
            verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
        }
    }

    #[test]
    fn verified_alias_owner_survives_a_duplicate_caller_occurrence_removal() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 1, 2);
        Operation::replace_operand(f.fusion, ctx, 1, f.outer.parameters[0]);
        let owner = alias_owner(ctx, f.fusion);
        replace_instruction(
            ctx,
            f.projection,
            f.selected,
            forwarding_options(),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 1);
        assert_eq!(alias_owner(ctx, f.fusion), owner);
        verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
    }
}
