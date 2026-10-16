use xlahlo::{Module, VerifyOptions, text};

const SIMPLE: &str = "HloModule m work { p = f32[] parameter(0) ROOT n = f32[] negate(p) } ENTRY main { p = f32[] parameter(0) s = ((f32[]),f32[],s32[]) async-start(p), calls=work ROOT d = f32[] async-done(s) }";
const PARTIAL: &str = "HloModule m work { a = f32[] parameter(0) b = f32[] parameter(1) ROOT r = f32[] add(a,b) } ENTRY main { p = f32[] parameter(0) q = f32[] parameter(1) s = ((f32[]),(),s32[]) async-start(p), calls=work u = ((f32[],f32[]),f32[],s64[]) async-update(s,q) ROOT d = f32[] async-done(u) }";

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}

fn verify(source: &str) -> Module {
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    module
}

#[test]
fn generic_async_round_trips_context_and_inherited_computation() {
    for source in [SIMPLE, PARTIAL] {
        let module = verify(source);
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        reparsed.verify(strict()).unwrap();
        let wire = module.to_proto().unwrap();
        Module::from_proto(
            xlahlo::interchange::decode(
                &xlahlo::interchange::encode(&wire).unwrap(),
                Default::default(),
            )
            .unwrap(),
        )
        .unwrap()
        .verify(strict())
        .unwrap();
        assert_eq!(wire.computations.len(), 2);
    }
}

#[test]
fn output_binding_may_be_empty_when_done_parameters_are_fully_bound() {
    verify(&SIMPLE.replace("((f32[]),f32[],s32[])", "((f32[]),(),s32[])"));
    let source = PARTIAL.replace(
        "ROOT d = f32[] async-done(u)",
        "ROOT d = f32[] async-done(s)",
    );
    assert!(
        text::parse_module(&source)
            .unwrap()
            .verify(strict())
            .is_err()
    );
}

#[test]
fn context_prefixes_and_new_binding_counts_are_checked() {
    for source in [
        PARTIAL.replace("async-update(s,q)", "async-update(s)"),
        PARTIAL.replace("async-update(s,q)", "async-update(s,q,q)"),
        PARTIAL.replace(
            "u = ((f32[],f32[]),f32[],s64[])",
            "u = ((f32[]),f32[],s64[])",
        ),
        PARTIAL.replace("s = ((f32[]),(),s32[])", "s = ((f32[],f32[]),(),s32[])"),
        SIMPLE.replace("s = ((f32[]),f32[],s32[])", "s = (f32[],f32[],s32[])"),
        SIMPLE.replace("s = ((f32[]),f32[],s32[])", "s = ((f32[]))"),
        SIMPLE.replace("s = ((f32[]),f32[],s32[])", "s = ((f32[]),s32[],s32[])"),
        SIMPLE.replace("d = f32[]", "d = s32[]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(strict())
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn async_chain_pairing_rejects_duplicate_done_orphan_and_illegal_consumers() {
    for source in [
        SIMPLE.replace("ROOT d =", "d2 = f32[] async-done(s) ROOT d ="),
        SIMPLE.replace(
            "ROOT d = f32[] async-done(s)",
            "ROOT d = ((f32[]),f32[],s32[]) copy(s)",
        ),
        SIMPLE.replace(
            "ROOT d = f32[] async-done(s)",
            "ROOT d = ((f32[]),f32[],s32[]) add(s,s)",
        ),
        PARTIAL.replace(
            "ROOT d = f32[] async-done(u)",
            "ROOT d = f32[] async-done(s)",
        ),
        PARTIAL.replace("async-update(s,q)", "async-update(s,s)"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(strict())
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn async_aliases_use_full_computation_shape_and_update_context_two_only() {
    verify(
        &PARTIAL
            .replace(
                "calls=work",
                "calls=work, output_to_operand_aliasing={{0,1}: (1,{})}",
            )
            .replace(
                "async-update(s,q)",
                "async-update(s,q), output_to_operand_aliasing={{2}: (0,{2})}",
            )
            .replace("s64[]", "s32[]"),
    );
    for source in [
        PARTIAL.replace(
            "calls=work",
            "calls=work, output_to_operand_aliasing={{}: (0,{})}",
        ),
        PARTIAL.replace(
            "calls=work",
            "calls=work, output_to_operand_aliasing={{1}: (2,{})}",
        ),
        PARTIAL.replace(
            "calls=work",
            "calls=work, output_to_operand_aliasing={{0,2}: (0,{})}",
        ),
        PARTIAL.replace(
            "async-update(s,q)",
            "async-update(s,q), output_to_operand_aliasing={{0,0}: (0,{0,0})}",
        ),
        PARTIAL.replace(
            "async-update(s,q)",
            "async-update(s,q), output_to_operand_aliasing={{2}: (1,{2})}",
        ),
        PARTIAL.replace(
            "async-update(s,q)",
            "async-update(s,q), output_to_operand_aliasing={{2}: (0,{2})}",
        ),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(strict())
                .is_err(),
            "{source}"
        );
    }
    assert!(
        text::parse_module(&SIMPLE.replace(
            "async-done(s)",
            "async-done(s), output_to_operand_aliasing={}"
        ))
        .is_err()
    );
}

#[test]
fn generic_async_shape_comparison_obeys_layout_policy() {
    let source = SIMPLE
        .replace("f32[]", "f32[2,3]{1,0}")
        .replace("s = ((f32[2,3]{1,0})", "s = ((f32[2,3]{0,1})");
    let module = text::parse_module(&source).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(module.verify(strict()).is_err());
}

#[test]
fn async_execution_thread_is_inherited_and_propagated_to_nested_callees() {
    let source = SIMPLE.replace(
        "calls=work",
        "calls=work, async_execution_thread=\"worker\"",
    );
    let module = verify(&source);
    let work = module
        .as_proto()
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap();
    assert_eq!(work.execution_thread, "worker");
    verify(&text::print_module(&module).unwrap());
    verify(&source.replace(
        "async-done(s)",
        "async-done(s), calls=work, async_execution_thread=\"worker\"",
    ));
    assert!(
        text::parse_module(&source.replace(
            "async-done(s)",
            "async-done(s), async_execution_thread=\"different\""
        ))
        .is_err()
    );
    let mut wire = module.to_proto().unwrap();
    wire.computations
        .iter_mut()
        .find(|c| c.name == "work")
        .unwrap()
        .execution_thread = "wrong".into();
    assert!(Module::from_proto(wire).unwrap().verify(strict()).is_err());
}

#[test]
fn async_computations_cannot_also_have_non_async_callers() {
    let source = SIMPLE.replace("ROOT d =", "c = f32[] call(p), to_apply=work ROOT d =");
    assert!(text::parse_module(&source).is_err());
    let module = verify(SIMPLE);
    let mut wire = module.to_proto().unwrap();
    let work = wire
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap()
        .id;
    let entry = wire
        .computations
        .iter_mut()
        .find(|c| c.name == "main")
        .unwrap();
    let mut caller = entry
        .instructions
        .iter()
        .find(|i| i.name == "d")
        .unwrap()
        .clone();
    caller.id = 100;
    caller.name = "call".into();
    caller.opcode = "call".into();
    caller.operand_ids = vec![
        entry
            .instructions
            .iter()
            .find(|i| i.opcode == "parameter")
            .unwrap()
            .id,
    ];
    caller.called_computation_ids = vec![work];
    entry.instructions.push(caller);
    assert!(Module::from_proto(wire).unwrap().verify(strict()).is_err());
}

#[test]
fn backward_async_dataflow_can_pass_through_tuple_and_barrier() {
    let source=SIMPLE.replace("ROOT d = f32[] async-done(s)",
        "t = (((f32[]),f32[],s32[])) tuple(s) g = ((f32[]),f32[],s32[]) get-tuple-element(t), index=0 b = ((f32[]),f32[],s32[]) opt-barrier(g) ROOT d = f32[] async-done(b)");
    verify(&source);
}

#[test]
fn generic_async_context_can_be_carried_through_while_state() {
    let source = r#"HloModule m
    work { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
    body {
      p = (((f32[]),f32[],s32[])) parameter(0)
      g = ((f32[]),f32[],s32[]) get-tuple-element(p), index=0
      u = ((f32[]),f32[],s32[]) async-update(g)
      ROOT t = (((f32[]),f32[],s32[])) tuple(u)
    }
    cond { p = (((f32[]),f32[],s32[])) parameter(0) ROOT f = pred[] constant(false) }
    ENTRY main {
      p = f32[] parameter(0)
      s = ((f32[]),f32[],s32[]) async-start(p), calls=work
      t = (((f32[]),f32[],s32[])) tuple(s)
      w = (((f32[]),f32[],s32[])) while(t), body=body, condition=cond
      g = ((f32[]),f32[],s32[]) get-tuple-element(w), index=0
      ROOT d = f32[] async-done(g)
    }"#;
    verify(source);
    verify(&source.replace(
        "ROOT d = f32[] async-done(g)",
        "ROOT d = ((f32[]),f32[],s32[]) opt-barrier(g)",
    ));
}

#[test]
fn nested_result_prefixes_extend_recursively() {
    let source = r#"HloModule m
    work {
      p = f32[] parameter(0)
      t = (f32[],f32[]) tuple(p,p)
      ROOT r = ((f32[],f32[]),f32[]) tuple(t,p)
    }
    ENTRY main {
      p = f32[] parameter(0)
      s = ((f32[]),((f32[])),s32[]) async-start(p), calls=work, async_execution_thread="worker"
      u = ((f32[]),((f32[],f32[]),f32[])) async-update(s)
      ROOT d = ((f32[],f32[]),f32[]) async-done(u)
    }"#;
    verify(source);
    let main_thread = source.replace(", async_execution_thread=\"worker\"", "");
    assert!(
        text::parse_module(&main_thread)
            .unwrap()
            .verify(strict())
            .is_err()
    );
    verify(&main_thread.replace(
        "calls=work",
        "calls=work, frontend_attributes={_collectives_group=\"\"}",
    ));
    assert!(
        text::parse_module(&source.replace("((f32[])),s32[]", "((s32[])),s32[]"))
            .unwrap()
            .verify(strict())
            .is_err()
    );
}

#[test]
fn async_thread_propagation_reaches_calls_and_nested_async_computations() {
    let source = r#"HloModule m
    leaf { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
    inner {
      p = f32[] parameter(0)
      s = ((f32[]),f32[]) async-start(p), calls=leaf, async_execution_thread="original"
      ROOT d = f32[] async-done(s), async_execution_thread="original"
    }
    work { p = f32[] parameter(0) ROOT c = f32[] call(p), to_apply=inner }
    ENTRY main {
      p = f32[] parameter(0)
      s = ((f32[]),f32[]) async-start(p), calls=work, async_execution_thread="worker"
      ROOT d = f32[] async-done(s)
    }"#;
    let module = verify(source);
    for computation in &module.as_proto().computations {
        assert_eq!(
            computation.execution_thread,
            if computation.name == "main" {
                ""
            } else {
                "worker"
            }
        );
    }
    verify(&text::print_module(&module).unwrap());
}

#[test]
fn shared_async_attributes_are_also_applied_to_the_wrapped_root() {
    let source = SIMPLE
        .replace(
            "calls=work",
            "calls=work, backend_config=\"first\", frontend_attributes={key=\"first\"}",
        )
        .replace(
            "async-done(s)",
            "async-done(s), backend_config=\"\", frontend_attributes={key=\"last\"}",
        );
    let module = verify(&source);
    let root = module
        .as_proto()
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap()
        .instructions
        .iter()
        .find(|i| i.opcode == "negate")
        .unwrap();
    assert!(root.backend_config.is_empty());
    assert_eq!(
        root.frontend_attributes.as_ref().unwrap().map["key"],
        "last"
    );
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("backend_config=\"\""));
    let reparsed = verify(&printed);
    let root = reparsed
        .as_proto()
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap()
        .instructions
        .iter()
        .find(|i| i.opcode == "negate")
        .unwrap();
    assert!(root.backend_config.is_empty());
}

#[test]
fn async_metadata_and_original_values_propagate_in_source_order() {
    let source = SIMPLE
        .replace("calls=work", "calls=work, metadata={op_name=\"start\"}")
        .replace(
            "async-done(s)",
            "async-done(s), metadata={op_name=\"done\"}, origin={{\"last\"}}, statistics={ignored=1,time=nan(0x1),cost=-0.0}",
        );
    let module = verify(&source);
    let wrapped = module
        .as_proto()
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap();
    let root = wrapped
        .instructions
        .iter()
        .find(|i| i.id == wrapped.root_id)
        .unwrap();
    assert_eq!(root.metadata.as_ref().unwrap().op_name, "done");
    let done = module
        .instruction(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert_eq!(root.original_value, done.original_value);
    assert_eq!(
        root.statistics_viz.as_ref().unwrap().statistics[0]
            .stat_val
            .to_bits(),
        done.statistics_viz.as_ref().unwrap().statistics[0]
            .stat_val
            .to_bits()
    );
    let printed = text::print_module(&module).unwrap();
    let reparsed = verify(&printed);
    assert_eq!(text::print_module(&reparsed).unwrap(), printed);
    let cleared = verify(&source.replace("op_name=\"done\"", ""));
    let wrapped = cleared
        .as_proto()
        .computations
        .iter()
        .find(|c| c.name == "work")
        .unwrap();
    let root = wrapped
        .instructions
        .iter()
        .find(|i| i.id == wrapped.root_id)
        .unwrap();
    assert!(root.metadata.as_ref().unwrap().op_name.is_empty());
    verify(&text::print_module(&cleared).unwrap());
    for original in [false, true] {
        let mut wire = module.to_proto().unwrap();
        let root = wire
            .computations
            .iter_mut()
            .find(|c| c.name == "work")
            .unwrap()
            .instructions
            .iter_mut()
            .find(|i| i.opcode == "negate")
            .unwrap();
        if original {
            root.original_value.clear();
        } else {
            root.metadata.mut_or_insert_default().op_name = "different".into();
        }
        assert_eq!(
            text::print_module(&Module::from_proto(wire).unwrap())
                .unwrap_err()
                .category,
            "unsupported"
        );
    }
    let mut wire = module.to_proto().unwrap();
    wire.computations
        .iter_mut()
        .find(|c| c.name == "work")
        .unwrap()
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "negate")
        .unwrap()
        .statistics_viz
        .clear();
    assert_eq!(
        text::print_module(&Module::from_proto(wire).unwrap())
            .unwrap_err()
            .category,
        "unsupported"
    );
}

#[test]
fn printer_rejects_raw_async_metadata_that_constructor_propagation_would_change() {
    let source = SIMPLE.replace(
        "calls=work",
        "calls=work, backend_config=\"shared\", frontend_attributes={key=\"shared\"}",
    );
    let module = verify(&source);
    for frontend in [false, true] {
        let mut wire = module.to_proto().unwrap();
        let root = wire
            .computations
            .iter_mut()
            .find(|c| c.name == "work")
            .unwrap()
            .instructions
            .iter_mut()
            .find(|i| i.opcode == "negate")
            .unwrap();
        if frontend {
            root.frontend_attributes
                .mut_or_insert_default()
                .map
                .insert("key".into(), "different".into());
        } else {
            root.backend_config = b"different".to_vec();
        }
        let module = Module::from_proto(wire).unwrap();
        module.verify(strict()).unwrap();
        assert!(text::print_module(&module).is_err());
    }
}
