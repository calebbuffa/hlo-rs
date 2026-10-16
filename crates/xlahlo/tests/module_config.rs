use xlahlo::{Module, ModuleConfig, VerifyOptions, text};

#[test]
fn counts_round_trip_outside_the_graph_wire_message() {
    let module = text::parse_module(
        "HloModule m, replica_count=2, num_partitions=3 ENTRY main { ROOT p = f32[] parameter(0) }",
    )
    .unwrap();
    let config = ModuleConfig {
        replica_count: 2,
        num_partitions: 3,
        use_spmd_partitioning: true,
        ..Default::default()
    };
    assert_eq!(module.config(), config);
    assert_eq!(module.fork().unwrap().config(), config);
    let printed = text::print_module(&module).unwrap();
    assert_eq!(text::parse_module(&printed).unwrap().config(), config);
    let wire = module.to_proto().unwrap();
    assert_eq!(
        Module::from_proto(wire.clone()).unwrap().config(),
        ModuleConfig::default()
    );
    assert_eq!(
        Module::from_proto_with_config(wire, config.clone())
            .unwrap()
            .config(),
        config
    );
}

#[test]
fn transactions_commit_or_roll_back_owned_configuration() {
    let mut module =
        text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
    let config = ModuleConfig {
        replica_count: 2,
        ..Default::default()
    };
    module
        .edit(|staged| {
            staged.set_config(config.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(module.config(), config);
    assert!(
        module
            .edit(|staged| {
                staged.set_config(ModuleConfig::default());
                Err(xlahlo::Error::new("test", "rollback"))
            })
            .is_err()
    );
    assert_eq!(module.config(), config);
}

#[test]
fn verified_configuration_changes_roll_back_when_participants_no_longer_match() {
    let mut module = text::parse_module(
        "HloModule m ENTRY main { p = f32[1] parameter(0)
 ROOT g = f32[3] all-gather(p), dimensions={0}, replica_groups={{0,1,2}} }",
    )
    .unwrap();
    assert!(
        module
            .edit_verified(Default::default(), |staged| {
                staged.set_config(ModuleConfig {
                    replica_count: 2,
                    ..Default::default()
                });
                Ok(())
            })
            .is_err()
    );
    assert_eq!(module.config(), ModuleConfig::default());
    module.verify(Default::default()).unwrap();
}

#[test]
fn owned_counts_drive_collective_verification_unless_explicitly_overridden() {
    let module = text::parse_module(
        "HloModule m, replica_count=2 ENTRY main {
 p = f32[1] parameter(0)
 ROOT g = f32[3] all-gather(p), dimensions={0}, replica_groups={{0,1,2}}
}",
    )
    .unwrap();
    assert!(module.verify(Default::default()).is_err());
    module
        .verify(VerifyOptions {
            ignore_module_config: true,
            ..Default::default()
        })
        .unwrap();
    let config = module.config();
    let restored = Module::from_proto_with_config(module.to_proto().unwrap(), config).unwrap();
    assert!(restored.verify(Default::default()).is_err());
}

#[test]
fn partition_counts_enable_spmd_sharding_checks() {
    let module = text::parse_module(
        "HloModule m, num_partitions=2 ENTRY main {
 ROOT p = f32[] parameter(0), sharding={maximal device=3}
}",
    )
    .unwrap();
    assert!(module.verify(Default::default()).is_err());
    module
        .verify(VerifyOptions {
            ignore_module_config: true,
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn text_rejects_configuration_modes_not_represented_by_headers() {
    let mut module =
        text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
    module.set_config(ModuleConfig {
        num_partitions: 2,
        ..Default::default()
    });
    assert!(text::print_module(&module).is_err());
    module.set_config(ModuleConfig {
        use_spmd_partitioning: true,
        ..Default::default()
    });
    assert!(text::print_module(&module).is_err());
}

#[test]
fn header_flags_preserve_lists_defaults_and_owned_lifecycle() {
    let module = text::parse_module(
        "HloModule m, alias_passthrough_params=true, allow_spmd_sharding_propagation_to_parameters={true,false,true}, allow_spmd_sharding_propagation_to_output=true ENTRY main { ROOT p = f32[] parameter(0) }",
    ).unwrap();
    let config = module.config();
    assert!(config.alias_passthrough_params);
    assert_eq!(
        config.allow_spmd_sharding_propagation_to_parameters,
        [true, false, true]
    );
    assert_eq!(config.allow_spmd_sharding_propagation_to_output, [true]);
    assert_eq!(module.fork().unwrap().config(), config);
    assert_eq!(
        text::parse_module(&text::print_module(&module).unwrap())
            .unwrap()
            .config(),
        config
    );
    assert_eq!(
        Module::from_proto_with_config(module.to_proto().unwrap(), config.clone())
            .unwrap()
            .config(),
        config
    );
    let defaults = text::parse_module(
        "HloModule m, alias_passthrough_params=false, allow_spmd_sharding_propagation_to_parameters={}, allow_spmd_sharding_propagation_to_output={false} ENTRY main { ROOT p = f32[] parameter(0) }",
    ).unwrap();
    assert_eq!(defaults.config(), ModuleConfig::default());
}

#[test]
fn empty_owned_lists_fail_printing_instead_of_becoming_defaults() {
    for parameters in [true, false] {
        let mut module =
            text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
        let mut config = module.config();
        if parameters {
            config.allow_spmd_sharding_propagation_to_parameters.clear();
        } else {
            config.allow_spmd_sharding_propagation_to_output.clear();
        }
        module.set_config(config);
        assert!(text::print_module(&module).is_err());
    }
}

#[test]
fn flags_do_not_override_explicit_compatibility_count_contexts() {
    let module = text::parse_module(
        "HloModule m, alias_passthrough_params=true ENTRY main { p = f32[1] parameter(0) ROOT g = f32[3] all-gather(p), dimensions={0}, replica_groups={{0,1,2}} }",
    ).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(
        module
            .verify(VerifyOptions {
                collective_context: xlahlo::CollectiveContext {
                    replica_count: 2,
                    partition_count: 1
                },
                ..Default::default()
            })
            .is_err()
    );
}

#[test]
fn flag_edits_commit_and_roll_back_without_sharing_mutable_lists() {
    let mut module =
        text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
    module
        .edit(|staged| {
            let mut config = staged.config();
            config.alias_passthrough_params = true;
            config.allow_spmd_sharding_propagation_to_output = vec![true, false];
            staged.set_config(config);
            Ok(())
        })
        .unwrap();
    let expected = module.config();
    assert!(
        module
            .edit(|staged| {
                let mut config = staged.config();
                config.allow_spmd_sharding_propagation_to_output.push(true);
                staged.set_config(config);
                Err(xlahlo::Error::new("test", "rollback"))
            })
            .is_err()
    );
    assert_eq!(module.config(), expected);
}
