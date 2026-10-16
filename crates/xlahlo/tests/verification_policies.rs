use xlahlo::{
    InstructionId, Module, VerifyOptions,
    proto::{
        hlo::HloModuleProto,
        xla_data::{OpMetadata, TileProto},
    },
    text,
};

fn mutate(source: &str, update: impl FnOnce(&mut HloModuleProto)) -> Module {
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    update(&mut wire);
    Module::from_proto(wire).unwrap()
}

fn disallow(_: &Module, _: InstructionId) -> bool {
    false
}

fn layout_policy() -> VerifyOptions {
    VerifyOptions {
        instruction_can_change_layout: Some(disallow),
        ..Default::default()
    }
}

#[test]
fn scheduling_names_are_checked_with_pinned_substring_exemptions() {
    for (name, valid) in [
        ("old", true),
        ("new", false),
        ("new.clone", true),
        ("new.remat", true),
        ("new.clone_suffix", true),
        ("new.Clone", false),
    ] {
        let source = format!("HloModule m ENTRY main {{ ROOT {name} = f32[] parameter(0) }}");
        let module = mutate(&source, |wire| {
            wire.computations[0].instructions[0].metadata = Some(OpMetadata {
                scheduling_name: "old".into(),
                ..Default::default()
            })
            .into();
        });
        module.verify(Default::default()).unwrap();
        assert_eq!(
            module
                .verify(VerifyOptions {
                    verify_instruction_name_unchanged: true,
                    ..Default::default()
                })
                .is_ok(),
            valid,
            "{name}"
        );
    }
    text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }")
        .unwrap()
        .verify(VerifyOptions {
            verify_instruction_name_unchanged: true,
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn host_memory_prohibition_visits_nested_tuple_and_buffer_shapes() {
    for source in [
        "HloModule m ENTRY main { ROOT p = (f32[2],(s32[3],token[])) parameter(0) }",
        "HloModule m ENTRY main { ROOT p = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" }",
    ] {
        let module = mutate(source, |wire| {
            let shape = wire.computations[0].instructions[0].shape.as_mut().unwrap();
            shape.tuple_shapes[0].layout.as_mut().unwrap().memory_space = 5;
        });
        module.verify(Default::default()).unwrap();
        assert_eq!(
            module
                .verify(VerifyOptions {
                    verify_no_host_memory_space: true,
                    ..Default::default()
                })
                .unwrap_err()
                .category,
            "memory-space"
        );
    }
}

#[test]
fn explicit_layout_predicate_applies_independently_of_layout_sensitive_mode() {
    let source = "HloModule m ENTRY main { p = f32[2,3] parameter(0) ROOT r = f32[2,3] negate(p) }";
    let module = mutate(source, |wire| {
        wire.computations[0].instructions[1]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .minor_to_major
            .reverse();
    });
    module.verify(Default::default()).unwrap();
    for strict in [false, true] {
        assert_eq!(
            module
                .verify(VerifyOptions {
                    layout_sensitive: strict,
                    ..layout_policy()
                })
                .unwrap_err()
                .category,
            "layout"
        );
    }

    fn permit_negate(module: &Module, id: InstructionId) -> bool {
        module.instruction(id).unwrap().opcode == "negate"
    }
    module
        .verify(VerifyOptions {
            instruction_can_change_layout: Some(permit_negate),
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn entry_computation_layout_is_checked_only_in_layout_sensitive_mode() {
    let input_layout_mismatch = text::parse_module(
        "HloModule m, entry_computation_layout={(f32[2,3]{1,0})->f32[2,3]{1,0}} ENTRY main { p = f32[2,3]{0,1} parameter(0) ROOT r = f32[2,3]{0,1} negate(p) }",
    )
    .unwrap();
    input_layout_mismatch.verify(Default::default()).unwrap();
    assert!(
        input_layout_mismatch
            .verify(VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            })
            .is_err()
    );

    let output_layout_mismatch = text::parse_module(
        "HloModule m, entry_computation_layout={(f32[2,3]{1,0})->f32[2,3]{1,0}} ENTRY main { p = f32[2,3]{1,0} parameter(0) ROOT r = f32[2,3]{0,1} negate(p) }",
    )
    .unwrap();
    output_layout_mismatch.verify(Default::default()).unwrap();
    assert!(
        output_layout_mismatch
            .verify(VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            })
            .is_err()
    );
}

#[test]
fn entry_layout_ignores_pinned_memory_tile_and_tail_padding_differences() {
    let module = text::parse_module(
        "HloModule m, entry_computation_layout={(f32[2,3]{1,0:T(2,2)S(5)M(8)})->f32[2,3]{1,0:T(2,2)S(5)M(8)}} ENTRY main { p = f32[2,3]{1,0:T(4,4)S(1)M(16)} parameter(0) ROOT r = f32[2,3]{1,0:T(4,4)S(1)M(16)} negate(p) }",
    )
    .unwrap();
    module
        .verify(VerifyOptions {
            layout_sensitive: true,
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn layout_predicate_ignores_tiles_and_memory_space_but_not_other_layout_fields() {
    let source = "HloModule m ENTRY main { p = f32[2,3] parameter(0) ROOT r = f32[2,3] negate(p) }";
    let module = mutate(source, |wire| {
        let layout = wire.computations[0].instructions[1]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap();
        layout.memory_space = 17;
        layout.tiles = vec![TileProto {
            dimensions: vec![2, 2],
            ..Default::default()
        }];
    });
    module.verify(layout_policy()).unwrap();
    let changed = mutate(source, |wire| {
        wire.computations[0].instructions[1]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .tail_padding_alignment_in_elements = 8;
    });
    assert_eq!(
        changed.verify(layout_policy()).unwrap_err().category,
        "layout"
    );
}

#[test]
fn representation_changing_operations_ignore_element_storage_bits() {
    for source in [
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = bf16[2] convert(p) }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = pred[2] compare(p,p), direction=EQ }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = pred[2] is-finite(p) }",
    ] {
        let module = mutate(source, |wire| {
            wire.computations[0].instructions[0]
                .shape
                .as_mut()
                .unwrap()
                .layout
                .as_mut()
                .unwrap()
                .element_size_in_bits = 32;
        });
        module.verify(layout_policy()).unwrap();
    }
    let source = "HloModule m ENTRY main { p = pred[2] parameter(0) x = f32[2] parameter(1) ROOT r = f32[2] select(p,x,x) }";
    let module = mutate(source, |wire| {
        wire.computations[0].instructions[0]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .element_size_in_bits = 1;
    });
    module.verify(layout_policy()).unwrap();
    let module = mutate(source, |wire| {
        wire.computations[0].instructions[1]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .element_size_in_bits = 32;
    });
    assert_eq!(
        module.verify(layout_policy()).unwrap_err().category,
        "layout"
    );
}

#[test]
fn host_offload_layout_transitions_match_source_directional_exclusions() {
    for (opcode, source) in [
        (
            "copy",
            "HloModule m ENTRY main { p = f32[3] parameter(0) ROOT r = f32[3] copy(p) }",
        ),
        (
            "dynamic-slice",
            "HloModule m ENTRY main { p = f32[3] parameter(0) i = s32[] constant(0) ROOT r = f32[2] dynamic-slice(p,i), dynamic_slice_sizes={2} }",
        ),
        (
            "dynamic-update-slice",
            "HloModule m ENTRY main { p = f32[3] parameter(0) u = f32[2] parameter(1) i = s32[] constant(0) ROOT r = f32[3] dynamic-update-slice(p,u,i) }",
        ),
    ] {
        for from in [0, 1, 5, 17] {
            for to in [0, 1, 5, 17] {
                let module = mutate(source, |wire| {
                    for i in &mut wire.computations[0].instructions {
                        let memory = if i.name == "p" {
                            from
                        } else if i.name == "r" {
                            to
                        } else {
                            0
                        };
                        i.shape
                            .as_mut()
                            .unwrap()
                            .layout
                            .as_mut()
                            .unwrap()
                            .memory_space = memory;
                    }
                });
                module.verify(Default::default()).unwrap();
                let invalid = (from == 1 && to != 1)
                    || (opcode == "dynamic-slice" && from == 0 && to == 5)
                    || (opcode == "dynamic-update-slice" && from == 5 && to == 0);
                assert_eq!(
                    module.verify(layout_policy()).is_ok(),
                    !invalid,
                    "{opcode} {from}->{to}"
                );
            }
        }
    }
}

#[test]
fn layout_comparisons_unwrap_buffers_and_skip_different_operand_ranks() {
    let source = "HloModule m ENTRY main { b = b(f32[2,3]) custom-call(), custom_call_target=\"CreateBuffer\" ROOT u = f32[2,3] custom-call(b), custom_call_target=\"Unpin\", output_to_operand_aliasing={{}: (0,{})} }";
    let module = mutate(source, |wire| {
        wire.computations[0].instructions[1]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .minor_to_major
            .reverse();
    });
    module.verify(Default::default()).unwrap();
    assert_eq!(
        module.verify(layout_policy()).unwrap_err().category,
        "layout"
    );
    let source = "HloModule m ENTRY main { p = f32[3] parameter(0) ROOT r = f32[2,3] broadcast(p), dimensions={1} }";
    let module = mutate(source, |wire| {
        wire.computations[0].instructions[1]
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .minor_to_major
            .reverse();
    });
    module.verify(layout_policy()).unwrap();
}

#[test]
fn optional_thread_policy_checks_every_to_apply_kind_not_arbitrary_custom_call_callees() {
    for source in [
        "HloModule m body { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[2] parameter(0) ROOT r = f32[2] map(p), to_apply=body }",
        "HloModule m body { p = f32[] parameter(0) q = f32[] parameter(1) ROOT r = f32[] add(p,q) } ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0) ROOT r = f32[] reduce(p,z), dimensions={0}, to_apply=body }",
        "HloModule m body { p = f32[] parameter(0) q = f32[] parameter(1) ROOT r = f32[] add(p,q) } ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0) ROOT r = f32[1] reduce-window(p,z), window={size=2}, to_apply=body }",
        "HloModule m body { p = f32[] parameter(0) q = f32[] parameter(1) ROOT r = pred[] compare(p,q), direction=LT } ENTRY main { p = f32[2] parameter(0) ROOT r = f32[2] sort(p), dimensions={0}, to_apply=body }",
        "HloModule m body { ROOT p = f32[] parameter(0) } ENTRY main { p = f32[2] parameter(0) ROOT r = f32[2] scan(p), dimensions={0}, num_carries=0, to_apply=body, is_associative=true }",
        "HloModule m body { p = f32[] parameter(0) q = f32[] parameter(1) ROOT r = f32[] add(p,q) } ENTRY main { p = f32[2] parameter(0) ROOT r = f32[2] all-reduce(p), to_apply=body }",
    ] {
        let module = mutate(source, |wire| {
            wire.computations
                .iter_mut()
                .find(|c| c.name == "body")
                .unwrap()
                .execution_thread = "worker".into();
        });
        module.verify(Default::default()).unwrap();
        assert_eq!(
            module
                .verify(VerifyOptions {
                    verify_call_nested_computation_thread_name: true,
                    ..Default::default()
                })
                .unwrap_err()
                .category,
            "execution-thread",
            "{source}"
        );
    }
    for (callees, valid) in [("body", false), ("body,other", true)] {
        let source = format!(
            "HloModule m body {{ ROOT p = f32[] parameter(0) }} other {{ ROOT p = f32[] parameter(0) }} ENTRY main {{ ROOT r = f32[] custom-call(), custom_call_target=\"backend\", called_computations={{{callees}}} }}"
        );
        let module = mutate(&source, |wire| {
            wire.computations
                .iter_mut()
                .find(|c| c.name == "body")
                .unwrap()
                .execution_thread = "worker".into();
        });
        module.verify(Default::default()).unwrap();
        assert_eq!(
            module
                .verify(VerifyOptions {
                    verify_call_nested_computation_thread_name: true,
                    ..Default::default()
                })
                .is_ok(),
            valid,
            "{callees}"
        );
    }
}
