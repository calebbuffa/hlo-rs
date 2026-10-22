use crate::control_support::*;
use xlahlo::{
    attributes::{
        Axes, Bools, OptionalInteger, ReplicaGroup, ReplicaGroups, SourceTarget, SourceTargetPairs,
    },
    opcode::Opcode,
    ops::control::{
        AllGatherOp, AllToAllOp, AsyncDoneOp, AsyncStartOp, CollectiveBroadcastOp,
        CollectivePermuteOp, RaggedAllToAllOp,
    },
    pliron::{
        builtin::ops::ModuleOp,
        context::{Context, Ptr},
        irfmt::parsers::spaced,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
    },
    shape,
    sharding::{IotaReplicaGroupList, MeshAxesReplicaGroupList},
    sharding::{Mesh, MeshAxis},
    types::ElementType,
    verification,
};

fn pair(ctx: &mut Context, kind: Opcode) -> (ModuleOp, Ptr<Operation>, Ptr<Operation>) {
    let array = shape(ctx, ElementType::F32, &[2, 2]);
    let array = shape::with_default_layout(ctx, array).unwrap();
    let inputs = if kind == Opcode::RaggedAllToAll {
        vec![array, array, shape(ctx, ElementType::S64, &[1])]
    } else {
        vec![array]
    };
    let mut computations = vec![];
    let mut instructions = vec![];
    for name in ["first", "second"] {
        let (computation, block, args) = computation(ctx, name, inputs.clone(), array);
        let instruction = match kind {
            Opcode::AllGather => {
                AllGatherOp::new(ctx, array, args, Axes(vec![0]), groups(1)).get_operation()
            }
            Opcode::AllToAll => {
                AllToAllOp::new(ctx, array, args, Axes(vec![0]), groups(1)).get_operation()
            }
            Opcode::CollectiveBroadcast => {
                CollectiveBroadcastOp::new(ctx, array, args, groups(1)).get_operation()
            }
            Opcode::CollectivePermute => {
                CollectivePermuteOp::new(ctx, array, args, SourceTargetPairs(vec![]))
                    .get_operation()
            }
            Opcode::RaggedAllToAll => RaggedAllToAllOp::new(
                ctx,
                array,
                vec![args[0], args[1], args[2], args[2], args[2], args[2]],
                groups(1),
            )
            .get_operation(),
            _ => unreachable!(),
        };
        instruction.insert_at_back(block, ctx);
        let result = instruction.deref(ctx).get_result(0);
        ret(ctx, block, result);
        computations.push(computation);
        instructions.push(instruction);
    }
    let (main, block, args) = computation(ctx, "main", inputs.clone(), array);
    let state = shape(ctx, ElementType::U32, &[]);
    let inputs = shape::tuple(ctx, inputs).unwrap();
    let context = shape::tuple(ctx, vec![inputs, array, state]).unwrap();
    let start = AsyncStartOp::new(ctx, context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, ctx);
    let input = start.result(ctx);
    let done = AsyncDoneOp::new(ctx, array, vec![input]);
    attr(
        ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, ctx);
    let result = done.result(ctx);
    ret(ctx, block, result);
    computations.push(main);
    (
        module(ctx, &computations, "main"),
        instructions[0],
        instructions[1],
    )
}

#[test]
fn collective_identity_compares_layout_constraints_only_for_owning_opcodes() {
    for kind in [
        Opcode::AllGather,
        Opcode::AllToAll,
        Opcode::CollectiveBroadcast,
        Opcode::CollectivePermute,
        Opcode::RaggedAllToAll,
    ] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, kind);
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, second, "hlo_constrain_layout", Bools(vec![false]));
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, second, "hlo_constrain_layout", Bools(vec![true]));
        Operation::get_op_dyn(second, &ctx).verify(&ctx).unwrap();
        assert_eq!(
            verification::verify(&ctx, module.get_operation()).is_ok(),
            matches!(kind, Opcode::CollectivePermute | Opcode::RaggedAllToAll),
            "{kind:?}"
        );
        attr(&ctx, first, "hlo_constrain_layout", Bools(vec![true]));
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn collective_identity_preserves_owned_channels() {
    for kind in [
        Opcode::AllGather,
        Opcode::AllToAll,
        Opcode::CollectiveBroadcast,
        Opcode::CollectivePermute,
        Opcode::RaggedAllToAll,
    ] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, kind);
        for operation in [first, second] {
            attr(&ctx, operation, "hlo_channel", OptionalInteger::Value(17));
        }
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, second, "hlo_channel", OptionalInteger::Value(23));
        Operation::get_op_dyn(second, &ctx).verify(&ctx).unwrap();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        attr(&ctx, second, "hlo_channel", OptionalInteger::Value(17));
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn collective_identity_does_not_inherit_other_opcodes_geometry_or_flags() {
    for kind in [
        Opcode::AllGather,
        Opcode::AllToAll,
        Opcode::CollectiveBroadcast,
        Opcode::CollectivePermute,
        Opcode::RaggedAllToAll,
    ] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, kind);
        for key in ["hlo_has_dynamic_root", "hlo_use_global_device_ids"] {
            if kind == Opcode::AllGather && key == "hlo_use_global_device_ids" {
                continue;
            }
            attr(&ctx, first, key, Bools(vec![false]));
            attr(&ctx, second, key, Bools(vec![true]));
        }
        if kind != Opcode::CollectivePermute {
            attr(&ctx, first, "hlo_pairs", SourceTargetPairs(vec![]));
            attr(
                &ctx,
                second,
                "hlo_pairs",
                SourceTargetPairs(vec![SourceTarget {
                    source: 0,
                    target: 0,
                }]),
            );
            attr(&ctx, first, "hlo_dynamic_slice_sizes", Axes(vec![]));
            attr(&ctx, second, "hlo_dynamic_slice_sizes", Axes(vec![99]));
        } else {
            attr(&ctx, first, "hlo_groups", groups(1));
            attr(&ctx, second, "hlo_groups", groups(3));
        }
        if matches!(
            kind,
            Opcode::CollectiveBroadcast | Opcode::CollectivePermute | Opcode::RaggedAllToAll
        ) {
            attr(&ctx, first, "hlo_dimensions", Axes(vec![]));
            attr(&ctx, second, "hlo_dimensions", Axes(vec![99]));
        }
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn collective_identity_still_compares_owned_geometry_and_pairs() {
    for kind in [
        Opcode::AllGather,
        Opcode::AllToAll,
        Opcode::CollectivePermute,
    ] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, kind);
        if kind == Opcode::CollectivePermute {
            attr(
                &ctx,
                second,
                "hlo_pairs",
                SourceTargetPairs(vec![SourceTarget {
                    source: 0,
                    target: 0,
                }]),
            );
        } else {
            attr(&ctx, second, "hlo_dimensions", Axes(vec![1]));
        }
        Operation::get_op_dyn(second, &ctx).verify(&ctx).unwrap();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        if kind == Opcode::CollectivePermute {
            attr(
                &ctx,
                first,
                "hlo_pairs",
                SourceTargetPairs(vec![SourceTarget {
                    source: 0,
                    target: 0,
                }]),
            );
        } else {
            attr(&ctx, first, "hlo_dimensions", Axes(vec![1]));
        }
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

fn device_list(ctx: &Context, op: Ptr<Operation>, representation: usize) {
    match representation {
        0 => attr(
            ctx,
            op,
            "hlo_groups",
            ReplicaGroups([0, 2, 1, 3].map(|id| ReplicaGroup(vec![id])).to_vec()),
        ),
        1 => attr(
            ctx,
            op,
            "hlo_iota_groups",
            IotaReplicaGroupList {
                num_replica_groups: 4,
                num_devices_per_group: 1,
                iota_reshape_dims: vec![2, 2],
                iota_transpose_perm: vec![1, 0],
            },
        ),
        2 => {
            let mesh = Mesh::new(
                vec![MeshAxis {
                    name: "x".into(),
                    size: 4,
                }],
                vec![0, 2, 1, 3],
                vec![],
            )
            .unwrap();
            attr(
                ctx,
                op,
                "hlo_mesh_groups",
                MeshAxesReplicaGroupList::new(mesh, vec![]).unwrap(),
            );
        }
        _ => unreachable!(),
    }
}

#[test]
fn collective_identity_resolves_equivalent_explicit_iota_and_mesh_groups_in_order() {
    for kind in [
        Opcode::AllGather,
        Opcode::AllToAll,
        Opcode::CollectiveBroadcast,
        Opcode::RaggedAllToAll,
    ] {
        for first_representation in 0..3 {
            for second_representation in 0..3 {
                let mut ctx = context();
                let (module, first, second) = pair(&mut ctx, kind);
                device_list(&ctx, first, first_representation);
                device_list(&ctx, second, second_representation);
                verification::verify(&ctx, module.get_operation()).unwrap();
                if first_representation == 1 && second_representation == 2 {
                    let text = module.get_operation().disp(&ctx).to_string();
                    let mut fresh = context();
                    let parsed =
                        parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text)
                            .unwrap();
                    verification::verify(&fresh, parsed).unwrap();
                    assert_eq!(module.get_operation().disp(&ctx).to_string(), text);
                    Operation::erase(parsed, &mut fresh);
                    assert!(fresh.is_ir_empty());
                }
                if second_representation == 0 {
                    attr(
                        &ctx,
                        second,
                        "hlo_groups",
                        ReplicaGroups((0..4).map(|id| ReplicaGroup(vec![id])).collect()),
                    );
                    Operation::get_op_dyn(second, &ctx).verify(&ctx).unwrap();
                    let text = module.get_operation().disp(&ctx).to_string();
                    assert!(verification::verify(&ctx, module.get_operation()).is_err());
                    assert_eq!(module.get_operation().disp(&ctx).to_string(), text);
                    device_list(&ctx, second, 0);
                    verification::verify(&ctx, module.get_operation()).unwrap();
                }
                Operation::erase(module.get_operation(), &mut ctx);
                assert!(ctx.is_ir_empty());
            }
        }
    }
}

#[test]
fn collective_identity_preserves_compact_group_precedence_over_unused_encodings() {
    let mut ctx = context();
    let (module, first, second) = pair(&mut ctx, Opcode::AllGather);
    device_list(&ctx, first, 0);
    device_list(&ctx, second, 2);
    attr(&ctx, second, "hlo_groups", groups(3));
    verification::verify(&ctx, module.get_operation()).unwrap();
    device_list(&ctx, second, 1);
    verification::verify(&ctx, module.get_operation()).unwrap();
    let different = IotaReplicaGroupList {
        num_replica_groups: 4,
        num_devices_per_group: 1,
        iota_reshape_dims: vec![4],
        iota_transpose_perm: vec![0],
    };
    attr(&ctx, second, "hlo_iota_groups", different);
    Operation::get_op_dyn(second, &ctx).verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn collective_identity_keeps_equal_large_compact_assignments_compact() {
    for representation in 0..2 {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, Opcode::AllGather);
        for instruction in [first, second] {
            if representation == 0 {
                attr(
                    &ctx,
                    instruction,
                    "hlo_iota_groups",
                    IotaReplicaGroupList {
                        num_replica_groups: 2_000_000_000,
                        num_devices_per_group: 1,
                        iota_reshape_dims: vec![2_000_000_000],
                        iota_transpose_perm: vec![0],
                    },
                );
            } else {
                let mesh = Mesh::new(
                    vec![MeshAxis {
                        name: "devices".into(),
                        size: 2_000_000_000,
                    }],
                    vec![],
                    vec![],
                )
                .unwrap();
                attr(
                    &ctx,
                    instruction,
                    "hlo_mesh_groups",
                    MeshAxesReplicaGroupList::new(mesh, vec![]).unwrap(),
                );
            }
        }

        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn collective_identity_preserves_group_boundaries_not_just_flat_device_order() {
    for kind in [Opcode::CollectiveBroadcast, Opcode::RaggedAllToAll] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, kind);
        let split = ReplicaGroups(vec![ReplicaGroup(vec![0, 1]), ReplicaGroup(vec![2, 3])]);
        attr(&ctx, first, "hlo_groups", split.clone());
        attr(&ctx, second, "hlo_groups", groups(4));
        Operation::get_op_dyn(first, &ctx).verify(&ctx).unwrap();
        Operation::get_op_dyn(second, &ctx).verify(&ctx).unwrap();
        let text = module.get_operation().disp(&ctx).to_string();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        assert_eq!(module.get_operation().disp(&ctx).to_string(), text);
        attr(&ctx, second, "hlo_groups", split);
        verification::verify(&ctx, module.get_operation()).unwrap();
        Operation::erase(module.get_operation(), &mut ctx);
        assert!(ctx.is_ir_empty());
    }
}
