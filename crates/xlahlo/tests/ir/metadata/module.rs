use crate::support::{computation, context, module, ret};
use xlahlo::{
    attributes::{Bytes, OpMetadata, Payload},
    ir::OriginalArray,
    ir::{self, *},
    pliron::{
        builtin::attributes::StringAttr,
        common_traits::Verify,
        context::{Context, Ptr},
        identifier::Identifier,
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
        value::DefiningEntity,
    },
    sharding::Sharding,
    types::{ElementType, TensorType, TupleType},
    verification::verify,
};

fn fixture(ctx: &mut Context) -> (Ptr<Operation>, Ptr<Operation>) {
    let ty = TensorType::get(ctx, ElementType::F32, vec![4]).unwrap();
    let (module, container) = module(ctx);
    let (_, block, parameters) = computation(ctx, container, "main", vec![ty], ty);
    let DefiningEntity::Op(parameter) = parameters[0].defining_entity() else {
        unreachable!()
    };
    ret(ctx, block, parameters[0]);
    (module.get_operation(), parameter)
}

fn original(name: &str) -> OriginalArray {
    OriginalArray {
        instruction_name: name.into(),
        shape_index: vec![1, 2],
    }
}

fn stack_index() -> StackFrameIndex {
    StackFrameIndex {
        file_names: vec!["file.rs".into()],
        function_names: vec!["function".into()],
        file_locations: vec![FileLocation {
            file_name_id: 1,
            function_name_id: 1,
            line: 7,
            end_line: 9,
            column: 2,
            end_column: 12,
        }],
        stack_frames: vec![
            StackFrame {
                file_location_id: 1,
                parent_frame_id: 0,
            },
            StackFrame {
                file_location_id: 1,
                parent_frame_id: 1,
            },
        ],
    }
}

fn full_metadata() -> ModuleMetadata {
    ModuleMetadata {
        cross_program_prefetches: vec![CrossProgramPrefetch {
            parameter: 0,
            index: vec![],
            offset: 12345,
        }],
        spmd_output_sharding: Some(Sharding::replicated()),
        profile_info: vec![
            ModuleProfileInfo {
                profile_type: 7,
                relative_speedup_bits: 1.25f64.to_bits(),
                profile_source: 2,
                compilation_event: 1,
                fingerprint: "fingerprint\0\"".into(),
                profile_generation_strategy: 6,
                original_changelist: 123,
                changelist: 456,
            },
            ModuleProfileInfo {
                profile_type: 1001,
                relative_speedup_bits: 0x7ff8_0000_0000_0042,
                profile_source: -1002,
                compilation_event: 1003,
                fingerprint: "unknown profile".into(),
                profile_generation_strategy: 1004,
                original_changelist: -1,
                changelist: i64::MAX,
            },
        ],
        spmd_parameters_shardings: vec![Sharding::maximal(0)],
        device_assignment: Some(DeviceAssignment {
            replica_count: 1,
            computation_count: 1,
            computation_devices: vec![ComputationDevice {
                replica_device_ids: vec![99],
            }],
        }),
        use_auto_spmd_partitioning: true,
        stack_frame_index: Some(stack_index()),
        original_value_recovery_table: Some(OriginalValueRecoveryTable(vec![
            OriginalValueRecoveryEntry {
                old_original_array: Some(original("old")),
                new_original_array: Some(original("new")),
                recovery_module: None,
            },
        ])),
        device_type: "unrecognized-device".into(),
        debug_attributes: vec![DebugAttributeTableEntry {
            original_array: Some(original("debug")),
            debug_attributes: vec![DebugAttributes {
                log_mode: 4,
                callback_id: i64::MAX,
                op_id: 3,
                partitioned: true,
                operands_sharding: "{opaque sharding}".into(),
                x64_config: Some(X64Config {
                    xprecision_64bit_fragment: "fragment".into(),
                    original_log_instruction_name: "log".into(),
                    xprecision_64bit_original_type: "f64".into(),
                }),
            }],
        }],
        backend_config: Bytes(vec![0, 255, 128, b'"', b'\\', b'\n']),
    }
}

fn roundtrip(ctx: &mut Context, root: Ptr<Operation>) -> Ptr<Operation> {
    let text = root.disp(ctx).to_string();
    let parsed = parse_from_str(spaced(Operation::top_level_parser()), ctx, &text).unwrap();
    verify(ctx, parsed).unwrap();
    parsed
}

fn instruction_payloads(ctx: &Context, root: Ptr<Operation>) -> Vec<BackendConfigPayload> {
    let mut pending = vec![root];
    let mut payloads = Vec::new();
    while let Some(op) = pending.pop() {
        if let Some(payload) = ir::backend_config_payload(ctx, op).unwrap() {
            payloads.push(payload);
        }
        for region in op.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    payloads
}

#[test]
fn every_module_proto_field_roundtrips_exactly_without_changing_config() {
    let mut ctx = context();
    let (root, instruction) = fixture(&mut ctx);
    let config = ModuleConfig {
        debug_options: DebugOptions {
            xla_backend_extra_options: [("unknown".into(), "preserved".into())].into(),
        },
        ..Default::default()
    };
    ir::set_config(&ctx, root, config.clone()).unwrap();
    let metadata = full_metadata();
    ir::set_metadata(&ctx, root, metadata.clone()).unwrap();
    instruction.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_metadata"),
        OpMetadata {
            stack_frame_id: 2,
            ..Default::default()
        },
    );
    verify(&ctx, root).unwrap();
    let parsed = roundtrip(&mut ctx, root);
    assert_eq!(ir::metadata(&ctx, parsed).unwrap(), metadata);
    assert_eq!(ir::config(&ctx, parsed).unwrap(), config);
    assert!(config.channel_id_sensitive());
}

#[test]
fn defaults_and_optional_message_presence_are_not_conflated() {
    let mut ctx = context();
    let (root, _) = fixture(&mut ctx);
    assert_eq!(ir::metadata(&ctx, root).unwrap(), ModuleMetadata::default());
    for metadata in [
        ModuleMetadata::default(),
        ModuleMetadata {
            spmd_output_sharding: Some(Sharding::default()),
            stack_frame_index: Some(StackFrameIndex::default()),
            original_value_recovery_table: Some(OriginalValueRecoveryTable::default()),
            debug_attributes: vec![
                DebugAttributeTableEntry::default(),
                DebugAttributeTableEntry {
                    original_array: Some(original("empty-debug")),
                    debug_attributes: vec![DebugAttributes {
                        x64_config: Some(X64Config::default()),
                        ..Default::default()
                    }],
                },
            ],
            ..Default::default()
        },
    ] {
        ir::set_metadata(&ctx, root, metadata.clone()).unwrap();
        let parsed = roundtrip(&mut ctx, root);
        assert_eq!(ir::metadata(&ctx, parsed).unwrap(), metadata);
    }
}

#[test]
fn each_profile_type_and_debug_log_mode_including_unknown_are_preserved() {
    let mut ctx = context();
    let (root, _) = fixture(&mut ctx);
    let metadata = ModuleMetadata {
        profile_info: (0..=8)
            .chain([-1, i32::MAX])
            .map(|profile_type| ModuleProfileInfo {
                profile_type,
                relative_speedup_bits: (-0.0f64).to_bits(),
                ..Default::default()
            })
            .collect(),
        debug_attributes: vec![DebugAttributeTableEntry {
            original_array: None,
            debug_attributes: (0..=5)
                .chain([-1, i32::MAX])
                .map(|log_mode| DebugAttributes {
                    log_mode,
                    ..Default::default()
                })
                .collect(),
        }],
        ..Default::default()
    };
    ir::set_metadata(&ctx, root, metadata.clone()).unwrap();
    let parsed = roundtrip(&mut ctx, root);
    assert_eq!(ir::metadata(&ctx, parsed).unwrap(), metadata);
}

#[test]
fn invalid_prefetch_numbers_paths_and_tuple_targets_are_rejected() {
    let mut ctx = context();
    let (root, _) = fixture(&mut ctx);
    for (parameter, index) in [(-1, vec![]), (1, vec![]), (0, vec![-1]), (0, vec![0])] {
        root.deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_module_metadata"),
            ModuleMetadata {
                cross_program_prefetches: vec![CrossProgramPrefetch {
                    parameter,
                    index,
                    offset: 0,
                }],
                ..Default::default()
            },
        );
        assert!(verify(&ctx, root).is_err());
    }
    let ty = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
    let tuple = TupleType::get(&ctx, vec![ty]).unwrap();
    let (root, container) = module(&mut ctx);
    let (_, block, parameters) = computation(&mut ctx, container, "main", vec![tuple], tuple);
    ret(&mut ctx, block, parameters[0]);
    ir::set_metadata(
        &ctx,
        root.get_operation(),
        ModuleMetadata {
            cross_program_prefetches: vec![CrossProgramPrefetch::default()],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(verify(&ctx, root.get_operation()).is_err());
    ir::set_metadata(
        &ctx,
        root.get_operation(),
        ModuleMetadata {
            cross_program_prefetches: vec![CrossProgramPrefetch {
                index: vec![0],
                offset: -1,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    verify(&ctx, root.get_operation()).unwrap();
}

#[test]
fn spmd_shardings_check_geometry_parameter_count_and_device_policy() {
    let mut ctx = context();
    let (root, _) = fixture(&mut ctx);
    let invalid_tuple = Sharding::tuple(vec![Sharding::replicated(), Sharding::replicated()]);
    for metadata in [
        ModuleMetadata {
            spmd_output_sharding: Some(invalid_tuple.clone()),
            ..Default::default()
        },
        ModuleMetadata {
            spmd_parameters_shardings: vec![invalid_tuple],
            ..Default::default()
        },
        ModuleMetadata {
            spmd_parameters_shardings: vec![Sharding::replicated(); 2],
            ..Default::default()
        },
    ] {
        root.deref_mut(&ctx)
            .attributes
            .set(Identifier::new("hlo_module_metadata"), metadata);
        assert!(verify(&ctx, root).is_err());
    }
    ir::set_config(
        &ctx,
        root,
        ModuleConfig {
            num_partitions: 2,
            use_spmd_partitioning: true,
            ..Default::default()
        },
    )
    .unwrap();
    for output in [true, false] {
        let metadata = if output {
            ModuleMetadata {
                spmd_output_sharding: Some(Sharding::maximal(2)),
                ..Default::default()
            }
        } else {
            ModuleMetadata {
                spmd_parameters_shardings: vec![Sharding::maximal(2)],
                ..Default::default()
            }
        };
        ir::set_metadata(&ctx, root, metadata).unwrap();
        assert!(verify(&ctx, root).is_err());
    }
}

#[test]
fn device_assignment_matches_pinned_deserializer_geometry() {
    let ctx = context();
    for replica_count in [-1, 0, 1, 2] {
        for computation_count in [-1, 0, 1, 2] {
            let assignment = DeviceAssignment {
                replica_count,
                computation_count,
                computation_devices: vec![ComputationDevice {
                    replica_device_ids: vec![123],
                }],
            };
            assert_eq!(
                assignment.verify(&ctx).is_ok(),
                replica_count == 1 && computation_count == 1
            );
        }
    }
    let valid = DeviceAssignment {
        replica_count: 2,
        computation_count: 2,
        computation_devices: vec![
            ComputationDevice {
                replica_device_ids: vec![-1, -1]
            };
            2
        ],
    };
    // Upstream Deserialize checks geometry, not device uniqueness or sign.
    valid.verify(&ctx).unwrap();
    let mut invalid = valid;
    invalid.computation_devices[1].replica_device_ids.pop();
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn stack_index_references_duplicates_and_cycles_are_rejected() {
    let ctx = context();
    stack_index().verify(&ctx).unwrap();
    for id in [-1, 0, 2] {
        for field in 0..3 {
            let mut invalid = stack_index();
            match field {
                0 => invalid.file_locations[0].file_name_id = id,
                1 => invalid.file_locations[0].function_name_id = id,
                _ => invalid.stack_frames[0].file_location_id = id,
            }
            assert!(invalid.verify(&ctx).is_err());
        }
    }
    for id in [-1, 3] {
        let mut invalid = stack_index();
        invalid.stack_frames[0].parent_frame_id = id;
        assert!(invalid.verify(&ctx).is_err());
    }
    for field in 0..4 {
        let mut invalid = stack_index();
        match field {
            0 => invalid.file_names.push(invalid.file_names[0].clone()),
            1 => invalid
                .function_names
                .push(invalid.function_names[0].clone()),
            2 => invalid
                .file_locations
                .push(invalid.file_locations[0].clone()),
            _ => invalid.stack_frames.push(invalid.stack_frames[0].clone()),
        }
        assert!(invalid.verify(&ctx).is_err());
    }
    let mut cyclic = stack_index();
    cyclic.stack_frames[0].parent_frame_id = 2;
    assert!(cyclic.verify(&ctx).is_err());
    cyclic.stack_frames[0].parent_frame_id = 1;
    assert!(cyclic.verify(&ctx).is_err());
}

#[test]
fn instruction_stack_frame_ids_use_module_table_and_zero_sentinel() {
    let mut ctx = context();
    let (root, instruction) = fixture(&mut ctx);
    for table in [None, Some(stack_index())] {
        ir::set_metadata(
            &ctx,
            root,
            ModuleMetadata {
                stack_frame_index: table.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        for stack_frame_id in [-1, 0, 1, 2, 3] {
            instruction.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_metadata"),
                OpMetadata {
                    stack_frame_id,
                    ..Default::default()
                },
            );
            let valid =
                stack_frame_id == 0 || (table.is_some() && (1..=2).contains(&stack_frame_id));
            assert_eq!(verify(&ctx, root).is_ok(), valid);
        }
    }
}

#[test]
fn recovery_snapshot_deep_clones_nested_modules_and_all_metadata() {
    let mut ctx = context();
    let (inner, instruction) = fixture(&mut ctx);
    let expected = full_metadata();
    ir::set_metadata(&ctx, inner, expected.clone()).unwrap();
    inner.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_metadata_payloads"),
        MetadataPayloads(vec![Bytes(vec![0, 255])]),
    );
    ir::set_backend_config_payload(&ctx, instruction, BackendConfigPayload(Payload::Id(0)))
        .unwrap();
    let snapshot = RecoveryModule::capture(&ctx, inner).unwrap();
    let (outer, _) = fixture(&mut ctx);
    let metadata = ModuleMetadata {
        original_value_recovery_table: Some(OriginalValueRecoveryTable(vec![
            OriginalValueRecoveryEntry {
                old_original_array: Some(original("before")),
                new_original_array: Some(original("after")),
                recovery_module: Some(snapshot.clone()),
            },
        ])),
        ..Default::default()
    };
    ir::set_metadata(&ctx, outer, metadata.clone()).unwrap();
    let parsed = roundtrip(&mut ctx, outer);
    assert_eq!(ir::metadata(&ctx, parsed).unwrap(), metadata);
    ir::set_metadata(&ctx, inner, ModuleMetadata::default()).unwrap();
    let first = snapshot.materialize(&mut ctx).unwrap();
    let second = snapshot.clone().materialize(&mut ctx).unwrap();
    assert_ne!(first, second);
    assert_eq!(ir::metadata(&ctx, first).unwrap(), expected);
    for copy in [first, second] {
        assert_eq!(
            instruction_payloads(&ctx, copy),
            vec![BackendConfigPayload(Payload::Id(0))]
        );
        assert_eq!(
            copy.deref(&ctx)
                .attributes
                .get::<MetadataPayloads>(&Identifier::new("hlo_metadata_payloads"))
                .unwrap()
                .0,
            vec![Bytes(vec![0, 255])]
        );
    }
    ir::set_metadata(&ctx, first, ModuleMetadata::default()).unwrap();
    assert_eq!(ir::metadata(&ctx, second).unwrap(), expected);
    let nested = RecoveryModule::capture(&ctx, outer).unwrap();
    let mut fresh = context();
    let recovered = nested.materialize(&mut fresh).unwrap();
    assert_eq!(ir::metadata(&fresh, recovered).unwrap(), metadata);
}

#[test]
fn malformed_recovery_modules_and_original_indices_fail_explicitly() {
    let mut ctx = context();
    for text in ["not assembly", "xlahlo.unknown", ""] {
        assert!(RecoveryModule(text.into()).verify(&ctx).is_err());
    }
    let (root, instruction) = fixture(&mut ctx);
    assert!(RecoveryModule::capture(&ctx, instruction).is_err());
    let mut bad_array = original("bad");
    bad_array.shape_index = vec![-1];
    for old in [true, false] {
        let entry = OriginalValueRecoveryEntry {
            old_original_array: old.then_some(bad_array.clone()),
            new_original_array: (!old).then_some(bad_array.clone()),
            recovery_module: None,
        };
        assert!(entry.verify(&ctx).is_err());
    }
    assert!(
        DebugAttributeTableEntry {
            original_array: Some(bad_array),
            debug_attributes: vec![],
        }
        .verify(&ctx)
        .is_err()
    );
    let bad = ModuleMetadata {
        original_value_recovery_table: Some(OriginalValueRecoveryTable(vec![
            OriginalValueRecoveryEntry {
                recovery_module: Some(RecoveryModule("invalid".into())),
                ..Default::default()
            },
        ])),
        ..Default::default()
    };
    assert!(ir::set_metadata(&ctx, root, bad).is_err());
}

#[test]
fn backend_payload_precedence_exact_bytes_and_present_empty_message() {
    let payloads = MetadataPayloads(vec![Bytes(vec![0, 255, 128, b'\n'])]);
    let legacy = Bytes(vec![1, 2, 3]);
    for (payload, expected) in [
        (None, legacy.clone()),
        (
            Some(BackendConfigPayload(Payload::Absent)),
            Bytes::default(),
        ),
        (
            Some(BackendConfigPayload(Payload::Value(Bytes::default()))),
            Bytes::default(),
        ),
        (
            Some(BackendConfigPayload(Payload::Value(Bytes(vec![255, 0])))),
            Bytes(vec![255, 0]),
        ),
        (
            Some(BackendConfigPayload(Payload::Id(0))),
            payloads.0[0].clone(),
        ),
    ] {
        assert_eq!(
            ir::resolve_backend_config(payload.as_ref(), &legacy, &payloads).unwrap(),
            expected
        );
    }
    for id in [-1, 1, i64::MAX] {
        assert!(
            ir::resolve_backend_config(
                Some(&BackendConfigPayload(Payload::Id(id))),
                &legacy,
                &payloads
            )
            .is_err()
        );
    }
}

#[test]
fn backend_payload_ids_and_inline_values_roundtrip_without_normalization() {
    let mut ctx = context();
    let (root, instruction) = fixture(&mut ctx);
    root.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_metadata_payloads"),
        MetadataPayloads(vec![Bytes(vec![255, 0])]),
    );
    assert_eq!(ir::backend_config_payload(&ctx, instruction).unwrap(), None);
    for payload in [
        Payload::Absent,
        Payload::Value(Bytes::default()),
        Payload::Value(Bytes(vec![0, 255, 128])),
        Payload::Id(0),
    ] {
        ir::set_backend_config_payload(&ctx, instruction, BackendConfigPayload(payload.clone()))
            .unwrap();
        verify(&ctx, root).unwrap();
        assert_eq!(
            ir::backend_config_payload(&ctx, instruction).unwrap(),
            Some(BackendConfigPayload(payload.clone()))
        );
        let parsed = roundtrip(&mut ctx, root);
        assert_eq!(
            instruction_payloads(&ctx, parsed),
            vec![BackendConfigPayload(payload)]
        );
    }
    for id in [-1, 1, i64::MAX] {
        ir::set_backend_config_payload(&ctx, instruction, BackendConfigPayload(Payload::Id(id)))
            .unwrap();
        assert!(verify(&ctx, root).is_err());
    }
    assert!(
        ir::set_backend_config_payload(&ctx, root, BackendConfigPayload(Payload::Absent)).is_err()
    );
}

#[test]
fn unknown_attribute_encodings_are_errors_not_discarded_payloads() {
    let mut ctx = context();
    let (root, instruction) = fixture(&mut ctx);
    ir::set_backend_config_payload(
        &ctx,
        instruction,
        BackendConfigPayload(Payload::Value(Bytes(vec![255, 0]))),
    )
    .unwrap();
    let text = root.disp(&ctx).to_string();
    let unsupported = text.replace(
        "xlahlo.backend_config_payload",
        "xlahlo.future_backend_payload",
    );
    assert_ne!(text, unsupported);
    assert!(
        parse_from_str(
            spaced(Operation::top_level_parser()),
            &mut ctx,
            &unsupported
        )
        .is_err()
    );
}

#[test]
fn wrong_attribute_types_and_nonmodule_metadata_access_fail() {
    let mut ctx = context();
    let (root, instruction) = fixture(&mut ctx);
    assert!(ir::set_metadata(&ctx, instruction, ModuleMetadata::default()).is_err());
    assert!(ir::metadata(&ctx, instruction).is_err());
    root.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_module_metadata"),
        StringAttr::new("wrong".into()),
    );
    assert!(ir::metadata(&ctx, root).is_err());
    assert!(verify(&ctx, root).is_err());
    ir::set_metadata(&ctx, root, ModuleMetadata::default()).unwrap();
    instruction.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_backend_config_payload"),
        StringAttr::new("wrong".into()),
    );
    assert!(ir::backend_config_payload(&ctx, instruction).is_err());
    assert!(verify(&ctx, root).is_err());
}
