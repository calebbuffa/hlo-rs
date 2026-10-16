use xlahlo::{
    CollectiveContext, Module, ShardingContext, VerifyOptions,
    proto::{
        hlo::HloModuleProto,
        xla_data::{OpSharding, op_sharding::Type},
    },
    text,
};

fn device(id: i64) -> OpSharding {
    OpSharding {
        type_: Type::MAXIMAL.into(),
        tile_assignment_devices: vec![id],
        ..Default::default()
    }
}
fn verify(wire: HloModuleProto, spmd: bool, check: bool, partitions: i64) -> xlahlo::Result<()> {
    Module::from_proto(wire)?.verify(VerifyOptions {
        sharding_context: ShardingContext {
            use_spmd_partitioning: spmd,
            verify_device_numbers: check,
        },
        collective_context: CollectiveContext {
            partition_count: partitions,
            ..Default::default()
        },
        ..Default::default()
    })
}
fn assign(wire: &mut HloModuleProto, name: &str, sharding: OpSharding) {
    wire.computations
        .iter_mut()
        .flat_map(|computation| &mut computation.instructions)
        .find(|instruction| instruction.name == name)
        .unwrap()
        .sharding = Some(sharding).into();
}

#[test]
fn configured_device_policy_requires_spmd_enabled_check_and_multiple_partitions() {
    let mut wire = text::parse_module("HloModule m ENTRY main { ROOT p = f32[2] parameter(0) }")
        .unwrap()
        .to_proto()
        .unwrap();
    assign(&mut wire, "p", device(7));
    for (spmd, check, partitions, valid) in [
        (false, true, 4, true),
        (true, false, 4, true),
        (true, true, 1, true),
        (true, true, 4, false),
        (true, true, 8, true),
    ] {
        assert_eq!(verify(wire.clone(), spmd, check, partitions).is_ok(), valid);
    }
    assign(&mut wire, "p", device(-1));
    assert!(verify(wire, false, false, 1).is_err());
}

#[test]
fn imported_shardings_preserve_wire_and_validate_flat_leaf_shapes() {
    let mut wire = text::parse_module(
        "HloModule m ENTRY main { ROOT p = (f32[2],(s32[3],token[])) parameter(0) }",
    )
    .unwrap()
    .to_proto()
    .unwrap();
    assign(&mut wire, "p", device(1));
    let module = Module::from_proto(wire.clone()).unwrap();
    module.verify(Default::default()).unwrap();
    assert_eq!(module.to_proto().unwrap(), wire);
    assign(
        &mut wire,
        "p",
        OpSharding {
            type_: Type::TUPLE.into(),
            tuple_shardings: vec![device(1), OpSharding::default(), device(-1)],
            ..Default::default()
        },
    );
    assert!(verify(wire.clone(), false, false, 1).is_err());
    wire.computations[0].instructions[0]
        .sharding
        .as_mut()
        .unwrap()
        .tuple_shardings
        .pop();
    assert!(verify(wire, false, false, 1).is_err());
}

#[test]
fn while_compares_all_present_related_shardings_but_not_its_init_or_condition_root() {
    let source = "HloModule m body { b = f32[2] parameter(0) ROOT br = f32[2] copy(b) } cond { c = f32[2] parameter(0) ROOT cr = pred[] constant(false) } ENTRY main { init = f32[2] parameter(0) ROOT w = f32[2] while(init), body=body, condition=cond }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    assign(&mut wire, "w", device(1));
    assign(&mut wire, "init", device(9));
    assign(&mut wire, "cr", device(10));
    verify(wire.clone(), false, true, 1).unwrap();
    for name in ["br", "b", "c"] {
        let mut changed = wire.clone();
        assign(&mut changed, name, device(2));
        assert!(verify(changed, false, true, 1).is_err(), "{name}");
        assign(&mut wire, name, device(1));
    }
    verify(wire, false, true, 1).unwrap();
}

#[test]
fn conditional_compares_branch_roots_ignoring_metadata_and_missing_annotations() {
    let source = "HloModule m left { l = f32[2] parameter(0) ROOT lr = f32[2] copy(l) } right { r = f32[2] parameter(0) ROOT rr = f32[2] copy(r) } ENTRY main { p = pred[] parameter(0) a = f32[2] parameter(1) ROOT select = f32[2] conditional(p,a,a), true_computation=left, false_computation=right }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    assign(&mut wire, "lr", device(1));
    assign(&mut wire, "r", device(7));
    verify(wire.clone(), false, true, 1).unwrap();
    let mut annotated = device(1);
    annotated.metadata.push(Default::default());
    assign(&mut wire, "rr", annotated);
    verify(wire.clone(), false, true, 1).unwrap();
    assign(&mut wire, "select", device(2));
    assert!(verify(wire, false, true, 1).is_err());
}
