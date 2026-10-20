#[path = "control_support/mod.rs"]
mod support;
use support::*;
use xlahlo::{
    Diagnostic,
    attributes::*,
    inference::control::collective as i,
    ops::{HloOpInterface, control::*, elementwise::AddOp},
    pliron::{
        common_traits::Verify, op::Op, operation::Operation, printable::Printable, r#type::Type,
    },
    replica_groups::{IotaReplicaGroupList, MeshAxesReplicaGroupList},
    shape::{Dimension as D, Shape},
    sharding::{AxisRef, Mesh, MeshAxis},
    types::{ElementType, ElementType::*, TensorType},
    verification::{self, CollectiveContext, VerifyOptions},
};

#[test]
fn ragged_offsets_preserve_native_rank_type_and_memory_space_rules() {
    for &element in ElementType::ALL {
        for dimensions in [vec![2], vec![2, 3]] {
            let mut ctx = context();
            let array = shape(&ctx, F32, &[4]);
            let offsets = shape(&ctx, element, &dimensions);
            let mut layout = Shape::layout(&ctx, offsets).unwrap().unwrap().clone();
            layout.memory_space = 7;
            let alternate = Shape::with_layout(&ctx, offsets, layout).unwrap();
            let input = param(&mut ctx, array, 0);
            let offsets = param(&mut ctx, offsets, 1);
            let alternate = param(&mut ctx, alternate, 2);
            let values = vec![
                input.result(&ctx),
                input.result(&ctx),
                offsets.result(&ctx),
                alternate.result(&ctx),
                offsets.result(&ctx),
                alternate.result(&ctx),
            ];
            let instruction = RaggedAllToAllOp::new(&mut ctx, array, values, groups(1));
            instruction.verify(&ctx).unwrap();
            Operation::erase(instruction.get_operation(), &mut ctx);
            for parameter in [input, offsets, alternate] {
                Operation::erase(parameter.get_operation(), &mut ctx);
            }
            assert!(ctx.is_ir_empty());
        }
    }
}

#[test]
fn ragged_offsets_reject_non_memory_shape_differences_without_mutation() {
    let mut ctx = context();
    let array = shape(&ctx, F32, &[4]);
    let offsets = shape(&ctx, S64, &[2, 3]);
    let mut layout = Shape::layout(&ctx, offsets).unwrap().unwrap().clone();
    layout.minor_to_major.reverse();
    let transposed_layout = Shape::with_layout(&ctx, offsets, layout).unwrap();
    let invalid = [
        shape(&ctx, S64, &[]),
        shape(&ctx, S64, &[2, 3, 1]),
        shape(&ctx, S64, &[2, 4]),
        shape(&ctx, F64, &[2, 3]),
        transposed_layout,
    ];
    let input = param(&mut ctx, array, 0);
    let offsets = param(&mut ctx, offsets, 1);
    let input_value = input.result(&ctx);
    let offset_value = offsets.result(&ctx);
    let instruction = RaggedAllToAllOp::new(
        &mut ctx,
        array,
        vec![
            input_value,
            input_value,
            offset_value,
            offset_value,
            offset_value,
            offset_value,
        ],
        groups(1),
    );
    instruction.verify(&ctx).unwrap();
    for ty in invalid {
        let invalid = param(&mut ctx, ty, 2);
        for index in 2..6 {
            Operation::replace_operand(
                instruction.get_operation(),
                &ctx,
                index,
                invalid.result(&ctx),
            );
            let before = instruction.get_operation().disp(&ctx).to_string();
            assert!(instruction.verify(&ctx).is_err(), "operand {index}");
            assert_eq!(instruction.get_operation().disp(&ctx).to_string(), before);
            Operation::replace_operand(instruction.get_operation(), &ctx, index, offset_value);
            instruction.verify(&ctx).unwrap();
        }
        Operation::erase(invalid.get_operation(), &mut ctx);
    }
    Operation::erase(instruction.get_operation(), &mut ctx);
    Operation::erase(input.get_operation(), &mut ctx);
    Operation::erase(offsets.get_operation(), &mut ctx);
    assert!(ctx.is_ir_empty());
}

#[test]
fn ragged_result_tracks_output_operand_without_restricting_input_to_its_shape() {
    let mut ctx = context();
    let output = shape(&ctx, F32, &[4]);
    let offsets = shape(&ctx, S64, &[1]);
    let token = Shape::token(&ctx);
    let input = param(&mut ctx, token, 0);
    let output_parameter = param(&mut ctx, output, 1);
    let offsets = param(&mut ctx, offsets, 2);
    let values = vec![
        input.result(&ctx),
        output_parameter.result(&ctx),
        offsets.result(&ctx),
        offsets.result(&ctx),
        offsets.result(&ctx),
        offsets.result(&ctx),
    ];
    for result in [output, shape(&ctx, F32, &[5]), shape(&ctx, F64, &[4])] {
        let instruction = RaggedAllToAllOp::new(&mut ctx, result, values.clone(), groups(1));
        assert_eq!(instruction.verify(&ctx).is_ok(), result == output);
        Operation::erase(instruction.get_operation(), &mut ctx);
    }
    for parameter in [input, output_parameter, offsets] {
        Operation::erase(parameter.get_operation(), &mut ctx);
    }
    assert!(ctx.is_ir_empty());
}

#[test]
fn malformed_collective_shapes_and_overflows_return_typed_errors() {
    let ctx = context();
    let malformed: xlahlo::pliron::r#type::TypeHandle = TensorType::instantiate(
        TensorType {
            element: F32,
            dimensions: vec![4],
            dynamic_dimensions: vec![],
            layout: vec![],
        },
        &ctx,
    )
    .into();
    for error in [
        i::infer_all_gather(&ctx, &[malformed], 0, 2).unwrap_err(),
        i::infer_reduce_scatter(&ctx, &[malformed], 0, 2).unwrap_err(),
    ] {
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidShape { .. })
        ));
    }
    let input = shape(&ctx, Pred, &[i64::MAX]);
    let error = i::infer_all_gather(&ctx, &[input], 0, 2).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::Inference { .. })
    ));
    assert!(i::infer_all_gather(&ctx, &[shape(&ctx, F32, &[4])], 0, 2).is_ok());
}

#[test]
fn gather_and_scatter_preserve_layout_but_follow_distinct_dynamic_axis_rules() {
    let ctx = context();
    let input = Shape::dynamic_array(&ctx, F32, vec![D::Bounded(6), D::Bounded(2)]).unwrap();
    let gathered = i::infer_all_gather(&ctx, &[input], 0, 3).unwrap();
    assert_eq!(Shape::dimension(&ctx, gathered, 0).unwrap(), D::Static(18));
    assert_eq!(Shape::dimension(&ctx, gathered, 1).unwrap(), D::Bounded(2));
    assert_eq!(
        Shape::layout(&ctx, gathered).unwrap().as_deref(),
        Shape::layout(&ctx, input).unwrap().as_deref()
    );
    let scattered = i::infer_reduce_scatter(&ctx, &[input], 0, 3).unwrap();
    assert_eq!(Shape::dimension(&ctx, scattered, 0).unwrap(), D::Unbounded);
    assert_eq!(Shape::dimension(&ctx, scattered, 1).unwrap(), D::Bounded(2));
    assert!(i::infer_reduce_scatter(&ctx, &[input], 0, 4).is_err());
    let unbounded = Shape::dynamic_array(&ctx, F32, vec![D::Unbounded]).unwrap();
    assert_eq!(
        i::infer_all_gather(&ctx, &[unbounded], 0, 2).unwrap(),
        unbounded
    );
    assert!(i::infer_all_gather(&ctx, &[shape(&ctx, Pred, &[i64::MAX])], 0, 2).is_err());
    assert!(i::infer_all_gather(&ctx, &[Shape::token(&ctx)], 0, 1).is_err());
    let buffer = Shape::buffer(&ctx, shape(&ctx, F32, &[2])).unwrap();
    assert!(i::infer_all_gather(&ctx, &[buffer], 0, 2).is_err());
}
#[test]
fn variadic_context_shapes_and_empty_inference_match_source() {
    let ctx = context();
    let a = shape(&ctx, F32, &[2]);
    let b = shape(&ctx, S32, &[]);
    let gathered = i::infer_all_gather_start(&ctx, &[a], 0, 2).unwrap();
    assert_eq!(Shape::subshape(&ctx, gathered, &[0]).unwrap(), a);
    assert_eq!(
        *Shape::dimensions(&ctx, i::infer_all_gather_done(&ctx, gathered).unwrap()).unwrap(),
        [4]
    );
    let reduced = i::infer_all_reduce_start(&ctx, &[a, b]).unwrap();
    assert_eq!(reduced, Shape::tuple(&ctx, vec![a, b]).unwrap());
    assert_eq!(i::infer_all_reduce_done(&ctx, reduced).unwrap(), reduced);
    assert_eq!(
        i::infer_all_gather(&ctx, &[], 0, 2).unwrap(),
        Shape::tuple(&ctx, vec![]).unwrap()
    );
    assert_eq!(
        i::infer_all_reduce(&ctx, &[]).unwrap(),
        Shape::tuple(&ctx, vec![]).unwrap()
    );
    assert!(i::infer_all_gather_done(&ctx, Shape::token(&ctx)).is_err());
}
#[test]
fn static_gather_scatter_small_geometry_is_exact() {
    let ctx = context();
    for size in 1..=8 {
        for shards in 1..=4 {
            let input = shape(&ctx, F32, &[size]);
            let gathered = i::infer_all_gather(&ctx, &[input], 0, shards).unwrap();
            assert_eq!(*Shape::dimensions(&ctx, gathered).unwrap(), [size * shards]);
            assert_eq!(
                i::infer_reduce_scatter(&ctx, &[gathered], 0, shards).unwrap(),
                input
            );
            assert_eq!(
                i::infer_reduce_scatter(&ctx, &[input], 0, shards).is_ok(),
                size % shards == 0
            );
        }
    }
}
#[test]
fn all_to_all_axis_exchange_checks_divisibility_dynamics_and_overflow() {
    let ctx = context();
    let input = shape(&ctx, F32, &[6, 4]);
    assert_eq!(
        *Shape::dimensions(&ctx, i::infer_all_to_all(&ctx, input, 0, 1, 3).unwrap()).unwrap(),
        [2, 12]
    );
    let buffer = Shape::buffer(&ctx, input).unwrap();
    assert!(Shape::equal_including_layout(
        &ctx,
        i::infer_all_to_all(&ctx, buffer, 0, 1, 3).unwrap(),
        shape(&ctx, F32, &[2, 12])
    ));
    assert!(Shape::equal_including_layout(
        &ctx,
        i::infer_all_to_all(&ctx, input, 0, 0, 3).unwrap(),
        input
    ));
    assert!(i::infer_all_to_all(&ctx, input, 0, 1, 4).is_err());
    assert!(i::infer_all_to_all(&ctx, input, -1, 1, 3).is_err());
    let bounded = Shape::dynamic_array(&ctx, F32, vec![D::Bounded(6)]).unwrap();
    assert!(i::infer_all_to_all(&ctx, bounded, 0, 0, 3).is_err());
    let unbounded = Shape::dynamic_array(&ctx, F32, vec![D::Unbounded, D::Static(2)]).unwrap();
    assert_eq!(
        *Shape::dimensions(&ctx, i::infer_all_to_all(&ctx, unbounded, 0, 1, 3).unwrap()).unwrap(),
        [i64::MIN, 6]
    );
    assert!(i::infer_all_to_all(&ctx, shape(&ctx, F32, &[0, i64::MAX]), 0, 1, 2).is_err());
}
#[test]
fn tuple_all_to_all_compares_full_shapes_except_memory_and_retains_each_layout() {
    let ctx = context();
    let first = Shape::with_default_layout(
        &ctx,
        Shape::dynamic_array(&ctx, F32, vec![D::Bounded(3), D::Static(2)]).unwrap(),
    )
    .unwrap();
    let mut layout = Shape::layout(&ctx, first).unwrap().unwrap().clone();
    layout.memory_space = 5;
    let second = Shape::with_layout(&ctx, first, layout.clone()).unwrap();
    assert_eq!(
        i::infer_all_to_all_tuple(&ctx, &[first, second]).unwrap(),
        Shape::tuple(&ctx, vec![first, second]).unwrap()
    );
    layout.minor_to_major.reverse();
    let transposed = Shape::with_layout(&ctx, first, layout).unwrap();
    assert!(i::infer_all_to_all_tuple(&ctx, &[first, transposed]).is_err());
    let unbounded = Shape::dynamic_array(&ctx, F32, vec![D::Unbounded]).unwrap();
    assert!(i::infer_all_to_all_tuple(&ctx, &[unbounded]).is_err());
    assert!(i::infer_all_to_all_tuple(&ctx, &[]).is_err());
}
#[test]
fn dynamic_root_broadcast_and_reduce_have_different_type_and_layout_constraints() {
    let ctx = context();
    let input = shape(&ctx, F32, &[2]);
    let root = shape(&ctx, S32, &[1]);
    assert_eq!(
        i::infer_collective_broadcast(&ctx, &[input, root], true).unwrap(),
        input
    );
    assert_eq!(
        i::infer_collective_reduce(&ctx, &[input, root], true).unwrap(),
        input
    );
    let mut layout = Shape::layout(&ctx, root).unwrap().unwrap().clone();
    layout.memory_space = 5;
    let root = Shape::with_layout(&ctx, root, layout).unwrap();
    assert!(i::infer_collective_broadcast(&ctx, &[input, root], true).is_ok());
    assert!(i::infer_collective_reduce(&ctx, &[input, root], true).is_err());
    assert!(i::infer_collective_reduce(&ctx, &[Shape::token(&ctx), root], true).is_err());
    assert_eq!(
        i::infer_ragged_all_to_all(&ctx, &[Shape::token(&ctx), input]).unwrap(),
        input
    );
}
#[test]
fn gather_checks_group_partitions_shape_ratios_and_variadic_geometry() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2, 3]);
    let out = shape(&ctx, F32, &[4, 3]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let gather = AllGatherOp::new(&mut ctx, out, vec![pv], Axes(vec![0]), groups(2));
    gather.verify(&ctx).unwrap();
    for bad in [
        vec![vec![0, 0]],
        vec![vec![0, 2]],
        vec![vec![]],
        vec![vec![0], vec![1, 2]],
    ] {
        attr(
            &ctx,
            gather.get_operation(),
            "hlo_groups",
            ReplicaGroups(bad.into_iter().map(ReplicaGroup).collect()),
        );
        assert!(gather.verify(&ctx).is_err());
    }
    attr(&ctx, gather.get_operation(), "hlo_groups", groups(2));
    let bad = shape(&ctx, F32, &[3, 3]);
    assert!(
        AllGatherOp::new(&mut ctx, bad, vec![pv], Axes(vec![0]), groups(2))
            .verify(&ctx)
            .is_err()
    );
}
#[test]
fn replica_group_checks_are_configurable_without_skipping_shape_inference() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let out = shape(&ctx, F32, &[4]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let gather = AllGatherOp::new(
        &mut ctx,
        out,
        vec![pv],
        Axes(vec![0]),
        ReplicaGroups(vec![ReplicaGroup(vec![0, 0])]),
    );
    assert!(gather.verify(&ctx).is_err());
    gather
        .verify_with_options(
            &ctx,
            VerifyOptions {
                check_replica_groups: false,
                collective_context: CollectiveContext {
                    replica_count: 3,
                    partition_count: 1,
                },
                ..Default::default()
            },
        )
        .unwrap();
    attr(
        &ctx,
        gather.get_operation(),
        "hlo_dimensions",
        Axes(vec![1]),
    );
    assert!(
        gather
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    check_replica_groups: false,
                    ..Default::default()
                }
            )
            .is_err()
    );
}
#[test]
fn module_counts_and_channel_global_modes_select_the_right_subgroups() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let out = shape(&ctx, F32, &[12]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let gather = AllGatherOp::new(&mut ctx, out, vec![pv], Axes(vec![0]), groups(2));
    attr(
        &ctx,
        gather.get_operation(),
        "hlo_channel",
        OptionalInteger::Value(1),
    );
    let options = VerifyOptions {
        collective_context: CollectiveContext {
            replica_count: 2,
            partition_count: 3,
        },
        ..Default::default()
    };
    gather.verify_with_options(&ctx, options).unwrap();
    assert!(
        gather
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    collective_context: CollectiveContext {
                        replica_count: 2,
                        partition_count: 2
                    },
                    ..Default::default()
                }
            )
            .is_err()
    );
    attr(
        &ctx,
        gather.get_operation(),
        "hlo_use_global_device_ids",
        Bools(vec![true]),
    );
    assert!(gather.verify_with_options(&ctx, options).is_err());
}
#[test]
fn collective_permute_rejects_duplicate_devices_and_invalid_inplace_offsets() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let pairs = SourceTargetPairs(vec![
        SourceTarget {
            source: 0,
            target: 1,
        },
        SourceTarget {
            source: 1,
            target: 0,
        },
    ]);
    let permute = CollectivePermuteOp::new(&mut ctx, f, vec![pv], pairs);
    permute.verify(&ctx).unwrap();
    attr(
        &ctx,
        permute.get_operation(),
        "hlo_pairs",
        SourceTargetPairs(vec![
            SourceTarget {
                source: 0,
                target: 1,
            },
            SourceTarget {
                source: 0,
                target: 2,
            },
        ]),
    );
    assert!(permute.verify(&ctx).is_err());
    attr(
        &ctx,
        permute.get_operation(),
        "hlo_dynamic_slice_sizes",
        DynamicSliceSizes(vec![Axes(vec![1])]),
    );
    assert!(permute.verify(&ctx).is_err());
}
#[test]
fn reduction_reducers_and_layout_constraint_policy_are_checked() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let scalar = shape(&ctx, F32, &[]);
    let (sum, block, args) = computation(&mut ctx, "sum", vec![scalar, scalar], scalar);
    let add = AddOp::new(&mut ctx, scalar, args);
    add.get_operation().insert_at_back(block, &ctx);
    let root = add.result(&ctx);
    ret(&mut ctx, block, root);
    let (main, block, args) = computation(&mut ctx, "main", vec![f], f);
    let pv = args[0];
    let first = AllReduceOp::new(&mut ctx, f, vec![pv], callees(&["sum"]), groups(1));
    first.get_operation().insert_at_back(block, &ctx);
    first.verify(&ctx).unwrap();
    let second = AllReduceOp::new(&mut ctx, f, vec![pv], callees(&["sum"]), groups(1));
    second.get_operation().insert_at_back(block, &ctx);
    let root = second.result(&ctx);
    ret(&mut ctx, block, root);
    let module = module(&mut ctx, &[main, sum], "main");
    verification::verify(&ctx, module.get_operation()).unwrap();
    attr(&ctx, first.get_operation(), "hlo_callees", Callees(vec![]));
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(
        &ctx,
        first.get_operation(),
        "hlo_callees",
        callees(&["sum"]),
    );
    attr(
        &ctx,
        second.get_operation(),
        "hlo_constrain_layout",
        Bools(vec![true]),
    );
    assert!(
        verification::verify(&ctx, module.get_operation())
            .unwrap_err()
            .to_string()
            .contains("constrain")
    );
    attr(
        &ctx,
        first.get_operation(),
        "hlo_channel",
        OptionalInteger::Value(0),
    );
    assert!(first.verify(&ctx).is_err());
}
#[test]
fn all_collective_opcodes_have_concrete_local_verifiers() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let g = shape(&ctx, F32, &[4]);
    let offsets = shape(&ctx, S32, &[2]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let o = param(&mut ctx, offsets, 1);
    let ov = o.result(&ctx);
    let start_ty = i::infer_all_gather_start(&ctx, &[f], 0, 2).unwrap();
    let start = AllGatherStartOp::new(&mut ctx, start_ty, vec![pv], Axes(vec![0]), groups(2));
    start.verify(&ctx).unwrap();
    let start_value = start.result(&ctx);
    AllGatherDoneOp::new(&mut ctx, g, vec![start_value])
        .verify(&ctx)
        .unwrap();
    let reduce = AllReduceStartOp::new(&mut ctx, f, vec![pv], callees(&["sum"]), groups(2));
    reduce.verify(&ctx).unwrap();
    let rv = reduce.result(&ctx);
    AllReduceDoneOp::new(&mut ctx, f, vec![rv])
        .verify(&ctx)
        .unwrap();
    let one = shape(&ctx, F32, &[1]);
    ReduceScatterOp::new(
        &mut ctx,
        one,
        vec![pv],
        Axes(vec![0]),
        callees(&["sum"]),
        groups(2),
    )
    .verify(&ctx)
    .unwrap();
    AllToAllOp::new(&mut ctx, f, vec![pv], Axes(vec![0]), groups(2))
        .verify(&ctx)
        .unwrap();
    RaggedAllToAllOp::new(&mut ctx, f, vec![pv, pv, ov, ov, ov, ov], groups(2))
        .verify(&ctx)
        .unwrap();
    CollectiveBroadcastOp::new(&mut ctx, f, vec![pv], groups(2))
        .verify(&ctx)
        .unwrap();
    CollectiveReduceOp::new(&mut ctx, f, vec![pv], callees(&["sum"]), groups(2))
        .verify(&ctx)
        .unwrap();
    let context = Shape::tuple(&ctx, vec![f, f]).unwrap();
    let permute =
        CollectivePermuteStartOp::new(&mut ctx, context, vec![pv], SourceTargetPairs(vec![]));
    permute.verify(&ctx).unwrap();
    let value = permute.result(&ctx);
    CollectivePermuteDoneOp::new(&mut ctx, f, vec![value])
        .verify(&ctx)
        .unwrap();
}

#[test]
fn compact_iota_and_mesh_collective_groups_validate_without_materializing_devices() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let out = shape(&ctx, F32, &[4]);
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    let gather = AllGatherOp::new(
        &mut ctx,
        out,
        vec![pv],
        Axes(vec![0]),
        ReplicaGroups(vec![]),
    );
    let iota = IotaReplicaGroupList {
        num_replica_groups: 1,
        num_devices_per_group: 2,
        iota_reshape_dims: vec![2],
        iota_transpose_perm: vec![0],
    };
    attr(
        &ctx,
        gather.get_operation(),
        "hlo_iota_groups",
        iota.clone(),
    );
    gather.verify(&ctx).unwrap();
    let mut bad = iota;
    bad.iota_transpose_perm = vec![1];
    attr(&ctx, gather.get_operation(), "hlo_iota_groups", bad);
    assert!(gather.verify(&ctx).is_err());
    let mesh = Mesh::new(
        vec![
            MeshAxis {
                name: "x".into(),
                size: 1_000_000_000,
            },
            MeshAxis {
                name: "y".into(),
                size: 2,
            },
        ],
        vec![],
        vec![],
    )
    .unwrap();
    let mesh_groups =
        MeshAxesReplicaGroupList::new(mesh, vec![AxisRef::new(1, vec![]).unwrap()]).unwrap();
    let gather = AllGatherOp::new(
        &mut ctx,
        out,
        vec![pv],
        Axes(vec![0]),
        ReplicaGroups(vec![]),
    );
    attr(&ctx, gather.get_operation(), "hlo_mesh_groups", mesh_groups);
    gather.verify(&ctx).unwrap();
}
