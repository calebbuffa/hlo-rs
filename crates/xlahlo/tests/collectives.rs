use xlahlo::{CollectiveContext, Module, VerifyOptions, text};

fn options(replicas: i64, partitions: i64) -> VerifyOptions {
    VerifyOptions {
        collective_context: CollectiveContext {
            replica_count: replicas,
            partition_count: partitions,
        },
        ..Default::default()
    }
}
fn verify(source: &str) -> Module {
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    module
}
fn round_trip(module: &Module) {
    let printed = text::print_module(module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    parsed.verify(Default::default()).unwrap();
    assert_eq!(printed, text::print_module(&parsed).unwrap());
}
const GATHER: &str = "HloModule m ENTRY main { p = f32[2,3] parameter(0) ROOT g = f32[4,3] all-gather(p), dimensions={0}, replica_groups={{0,1}} }";
const REDUCER: &str =
    "sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT add = f32[] add(a,b) }";

#[test]
fn gather_checks_group_partitions_shape_ratios_and_variadic_geometry() {
    round_trip(&verify(GATHER));
    let variadic = "HloModule m ENTRY main { p = f32[2,3] parameter(0) q = s32[4] parameter(1) ROOT g = (f32[4,3],s32[8]) all-gather(p,q), dimensions={0}, replica_groups={{0,1}} }";
    round_trip(&verify(variadic));
    for changed in [
        GATHER.replace("{{0,1}}", "{{0,0}}"),
        GATHER.replace("{{0,1}}", "{{0,2}}"),
        GATHER.replace("{{0,1}}", "{{}}"),
        GATHER.replace("{{0,1}}", "{{0},{1,2}}"),
        GATHER.replace("g = f32[4,3]", "g = f32[3,3]"),
        GATHER.replace("dimensions={0}", "dimensions={2}"),
        variadic.replace("s32[8]", "s32[4]"),
    ] {
        assert!(
            text::parse_module(&changed)
                .unwrap()
                .verify(Default::default())
                .is_err(),
            "{changed}"
        );
    }
    assert!(text::parse_module(&GATHER.replace("dimensions={0}", "dimensions={0,1}")).is_err());
}

#[test]
fn replica_group_checks_are_configurable_without_skipping_shape_inference() {
    let gather = text::parse_module(&GATHER.replace("{{0,1}}", "{{0,0}}")).unwrap();
    assert!(gather.verify(Default::default()).is_err());
    gather
        .verify(VerifyOptions {
            check_replica_groups: false,
            collective_context: CollectiveContext {
                replica_count: 3,
                partition_count: 1,
            },
            ..Default::default()
        })
        .unwrap();

    let nonuniform = text::parse_module(&GATHER.replace("{{0,1}}", "{{0,1},{2}}")).unwrap();
    assert!(nonuniform.verify(Default::default()).is_err());
    nonuniform
        .verify(VerifyOptions {
            check_replica_groups: false,
            collective_context: CollectiveContext {
                replica_count: 3,
                partition_count: 1,
            },
            ..Default::default()
        })
        .unwrap();

    let all_reduce = text::parse_module(&format!(
        "HloModule m {REDUCER} ENTRY main {{ p = f32[2] parameter(0) ROOT r = f32[2] all-reduce(p), to_apply=sum, replica_groups={{{{0,0}}}} }}"
    ))
    .unwrap();
    assert!(all_reduce.verify(Default::default()).is_err());
    all_reduce
        .verify(VerifyOptions {
            check_replica_groups: false,
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn module_counts_and_channel_global_modes_select_the_right_subgroups() {
    let module = text::parse_module(GATHER).unwrap();
    module.verify(options(2, 7)).unwrap();
    assert!(module.verify(options(3, 1)).is_err());
    let cross = GATHER
        .replace("g = f32[4,3]", "g = f32[12,3]")
        .replace("dimensions={0}", "dimensions={0}, channel_id=1");
    text::parse_module(&cross)
        .unwrap()
        .verify(options(2, 3))
        .unwrap();
    assert!(
        text::parse_module(&cross)
            .unwrap()
            .verify(options(2, 2))
            .is_err()
    );
    let flattened = GATHER
        .replace("g = f32[4,3]", "g = f32[6,3]")
        .replace("{{0,1}}", "{{0,1,2},{3,4,5}}")
        .replace(
            "dimensions={0}",
            "dimensions={0}, channel_id=1, use_global_device_ids=true",
        );
    text::parse_module(&flattened)
        .unwrap()
        .verify(options(2, 3))
        .unwrap();
    assert!(
        text::parse_module(&flattened)
            .unwrap()
            .verify(options(2, 2))
            .is_err()
    );
    assert!(
        text::parse_module(&GATHER.replace(
            "dimensions={0}",
            "dimensions={0}, use_global_device_ids=true"
        ))
        .unwrap()
        .verify(Default::default())
        .is_err()
    );
    assert!(
        text::parse_module(&flattened.replace("{{0,1,2},{3,4,5}}", "{}"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    let partition = "HloModule m ENTRY main { p = f32[6] parameter(0) ROOT a = f32[6] all-to-all(p), dimensions={0}, channel_id=1, replica_groups={{0,1,2}} }";
    text::parse_module(partition)
        .unwrap()
        .verify(options(99, 3))
        .unwrap();
    assert!(
        text::parse_module(partition)
            .unwrap()
            .verify(options(99, 4))
            .is_err()
    );
}

#[test]
fn all_reduce_accepts_nonuniform_groups_and_layout_constraint_policy_is_module_wide() {
    let source = format!(
        "HloModule m {REDUCER} ENTRY main {{ p = f32[2] parameter(0) r = f32[2] all-reduce(p), to_apply=sum, replica_groups={{{{0}},{{1,2}}}} ROOT s = f32[2] all-reduce(r), to_apply=sum }}"
    );
    round_trip(&verify(&source));
    assert!(
        text::parse_module(
            &source.replace("all-reduce(r),", "all-reduce(r), constrain_layout=true,")
        )
        .unwrap()
        .verify(Default::default())
        .is_err()
    );
    round_trip(&verify(
        &source
            .replace("all-reduce(p),", "all-reduce(p), constrain_layout=true,")
            .replace("all-reduce(r),", "all-reduce(r), constrain_layout=true,"),
    ));
    let legacy = source.replace("all-reduce(p),", "all-reduce(p), mode=cross_partition,");
    round_trip(&verify(&legacy));
    let empty =
        format!("HloModule m {REDUCER} ENTRY main {{ ROOT r = () all-reduce(), to_apply=sum }}");
    round_trip(&verify(&empty));
}

#[test]
fn reduce_scatter_checks_divisibility_shape_ratios_and_variadic_results() {
    let source = format!(
        "HloModule m {REDUCER} ENTRY main {{ p = f32[6] parameter(0) ROOT r = f32[2] reduce-scatter(p), dimensions={{0}}, to_apply=sum, replica_groups={{{{0,1,2}}}} }}"
    );
    round_trip(&verify(&source));
    assert!(
        text::parse_module(&source.replace("r = f32[2]", "r = f32[4]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace("p = f32[6]", "p = f32[5]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    let variadic = source.replace(
        "ROOT r = f32[2] reduce-scatter(p)",
        "q = s32[9] parameter(1) ROOT r = (f32[2],s32[3]) reduce-scatter(p,q)",
    );
    round_trip(&verify(&variadic));
}

#[test]
fn tuple_all_to_all_requires_one_operand_per_subgroup_member_and_equal_layouts() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) q = f32[2] parameter(1) ROOT a = (f32[2],f32[2]) all-to-all(p,q), replica_groups={{0,1}} }";
    round_trip(&verify(source));
    assert!(
        text::parse_module(&source.replace("{{0,1}}", "{}"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace("q = f32[2]", "q = f32[3]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn compact_iota_groups_verify_without_materialization_and_preserve_transpose() {
    let source = GATHER.replace("{{0,1}}", "[2,2]<=[2,2]T(1,0)");
    let module = verify(&source);
    module.verify(options(4, 1)).unwrap();
    round_trip(&module);
    round_trip(&verify(&GATHER.replace("{{0,1}}", "[1,2]<=[2]")));
    for groups in [
        "[1,2]<=[3]",
        "[2,2]<=[2,2]T(0,0)",
        "[0,2]<=[0]",
        "[2,2,2]<=[8]",
    ] {
        assert!(text::parse_module(&GATHER.replace("{{0,1}}", groups)).is_err());
    }
    let mut wire = text::parse_module(GATHER).unwrap().to_proto().unwrap();
    let root = wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "all-gather")
        .unwrap();
    root.replica_groups = root.collective_device_list().replica_groups.clone();
    root.clear_collective_device_list();
    let legacy = Module::from_proto(wire).unwrap();
    legacy.verify(Default::default()).unwrap();
    round_trip(&legacy);
}

#[test]
fn ragged_offsets_sizes_follow_shape_checks_without_invented_integer_domains() {
    let source = "HloModule m ENTRY main { input = f32[6,2] parameter(0) output = f32[8,2] parameter(1) offsets = f32[2] parameter(2) ROOT a = f32[8,2] ragged-all-to-all(input,output,offsets,offsets,offsets,offsets), replica_groups={{0,1}} }";
    round_trip(&verify(source));
    assert!(
        text::parse_module(&source.replace("offsets = f32[2]", "offsets = f32[]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn collective_reduce_and_broadcast_retain_pinned_dynamic_root_verifier_difference() {
    let source = format!(
        "HloModule m {REDUCER} ENTRY main {{ p = f32[2] parameter(0) roots = s32[1] parameter(1) ROOT r = f32[2] collective-reduce(p,roots), to_apply=sum, has_dynamic_root=true }}"
    );
    round_trip(&verify(&source));
    assert!(
        text::parse_module(&source.replace("roots = s32[1]", "roots = s32[2]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    let broadcast = "HloModule m ENTRY main { p = f32[2] parameter(0) roots = s32[] parameter(1) ROOT r = f32[2] collective-broadcast(p,roots), has_dynamic_root=true, replica_groups={{99}} }";
    // Unlike public inference, the pinned handler ignores dynamic-root and groups.
    round_trip(&verify(broadcast));
}
