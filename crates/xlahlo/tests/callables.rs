use xlahlo::{Instruction, Module, Opcode, PrimitiveType as P, Shape, VerifyOptions, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}

#[test]
fn fusion_owns_its_body_and_checks_shapes_kind_and_internal_liveness() {
    let source = "HloModule m body { p = f32[2]{0} parameter(0) ROOT n = f32[2]{0} negate(p) } ENTRY main { x = f32[2]{0} parameter(0) ROOT f = f32[2]{0} fusion(x), kind=kLoop, calls=body }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    assert!(
        module
            .as_proto()
            .computations
            .iter()
            .find(|c| c.name == "body")
            .unwrap()
            .is_fusion_computation
    );
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(strict())
        .unwrap();
    for source in [
        source.replace("ROOT n =", "unused = f32[2]{0} negate(p) ROOT n ="),
        source
            .replace("ROOT n =", "n =")
            .replace("p = f32", "ROOT p = f32"),
        source.replace("f = f32[2]", "f = f32[3]"),
        source.replace(
            "ROOT f =",
            "g = f32[2]{0} fusion(x), kind=kLoop, calls=body ROOT f =",
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
    assert!(text::parse_module(&source.replace("kind=kLoop", "kind=kUnknown")).is_err());
    assert!(text::parse_module(&source.replace(", kind=kLoop", "")).is_err());
}

#[test]
fn fusion_allows_extra_operands_but_not_unused_body_parameters() {
    let source = "HloModule m body { ROOT p = s32[] parameter(0) } ENTRY main { x = s32[] parameter(0) y = s32[] parameter(1) ROOT f = s32[] fusion(x,y), kind=kInput, calls=body }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let mut module = Module::new("builder");
    let body = module.add_computation("body").unwrap();
    let scalar = Shape::scalar(P::S32).unwrap();
    let p = module
        .add_instruction(
            body,
            Instruction::new("p", Opcode::Parameter, scalar.clone()),
            &[],
            &[],
        )
        .unwrap();
    module.set_root(body, p).unwrap();
    let entry = module.add_computation("main").unwrap();
    let x = module
        .add_instruction(
            entry,
            Instruction::new("x", Opcode::Parameter, scalar.clone()),
            &[],
            &[],
        )
        .unwrap();
    let f = module
        .add_instruction(
            entry,
            Instruction::new("f", Opcode::Fusion, scalar)
                .with_fields(|i| i.fusion_kind = "kLoop".into())
                .unwrap(),
            &[x],
            &[body],
        )
        .unwrap();
    module.set_root(entry, f).unwrap();
    module.set_entry(entry).unwrap();
    module.verify(strict()).unwrap();
    assert!(module.computation(body).unwrap().is_fusion_computation);
    module.set_called_computations(f, &[]).unwrap();
    assert!(!module.computation(body).unwrap().is_fusion_computation);
    assert!(module.verify(strict()).is_err());
}

#[test]
fn fusion_keeps_unused_local_effects_but_not_unused_pure_custom_calls() {
    let source = "HloModule m body { effect = () custom-call(), custom_call_target=\"effect\", custom_call_has_side_effect=true ROOT p = s32[] parameter(0) } ENTRY main { x = s32[] parameter(0) ROOT f = s32[] fusion(x), kind=kCustom, calls=body }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    assert!(
        text::parse_module(&source.replace("has_side_effect=true", "has_side_effect=false"))
            .unwrap()
            .verify(strict())
            .is_err()
    );
}

#[test]
fn custom_call_preserves_opaque_payloads_frontend_attributes_and_callees() {
    let source = r#"HloModule m callee { ROOT p = s32[] parameter(0) } ENTRY main { x = f32[2] parameter(0) ROOT c = f32[2] custom-call(x), custom_call_target="backend", called_computations={callee}, api_version=API_VERSION_TYPED_FFI, schedule=SCHEDULE_LATEST, custom_call_has_side_effect=true, backend_config="\000\377\"payload", frontend_attributes={alpha="a\nb", composite.attributes="{}"} }"#;
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    reparsed.verify(strict()).unwrap();
    let call = |module: &Module| {
        module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .find(|i| i.opcode == "custom-call")
            .unwrap()
            .clone()
    };
    assert_eq!(call(&module).backend_config, call(&reparsed).backend_config);
    assert_eq!(
        call(&module).frontend_attributes,
        call(&reparsed).frontend_attributes
    );
    assert_eq!(
        call(&module).custom_call_api_version,
        call(&reparsed).custom_call_api_version
    );
    assert!(
        text::parse_module(&source.replace("API_VERSION_TYPED_FFI", "API_VERSION_UNSPECIFIED"))
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace(
            "called_computations={callee}",
            "called_computations={callee}, to_apply=callee"
        ))
        .is_err()
    );
}

#[test]
fn custom_call_layout_constraints_are_logical_not_operand_layout_equalities() {
    let source = "HloModule m ENTRY main { p = f32[2,3]{1,0} parameter(0) ROOT c = f32[2,3]{1,0} custom-call(p), custom_call_target=\"backend\", operand_layout_constraints={f32[2,3]{0,1}} }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(strict())
        .unwrap();
    assert!(
        text::parse_module(&source.replace("constraints={f32[2,3]", "constraints={f32[3,2]"))
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace("constraints={f32[2,3]{0,1}}", "constraints={}"))
            .is_err()
    );
    assert!(
        text::parse_module_with_options(
            &source.replace("c = f32[2,3]{1,0}", "c = f32[2,3]"),
            text::ParseOptions {
                fill_missing_layouts: false,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn aliases_validate_nested_indices_shapes_and_the_pinned_skip_attribute() {
    let source = "HloModule m ENTRY main { p = (f32[2],s32[]) parameter(0) ROOT c = (f32[2],s32[]) custom-call(p), custom_call_target=\"backend\", output_to_operand_aliasing={{0}: (0,{0}),{1}: (0,{1})} }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(strict())
        .unwrap();
    for replacement in [
        "{{0}: (1,{0})}",
        "{{2}: (0,{0})}",
        "{{0}: (0,{-1})}",
        "{{0}: (0,{1})}",
    ] {
        let changed = source.replace("{{0}: (0,{0}),{1}: (0,{1})}", replacement);
        assert!(
            text::parse_module(&changed)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let source=source.replace("{1}: (0,{1})","{99}: (99,{99})").replace("custom_call_target=\"backend\"","custom_call_target=\"backend\", frontend_attributes={xla_skip_custom_call_alias_shape_check=\"true\"}");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    assert!(
        text::parse_module(&source.replace("{0}: (0,{0})", "{99}: (0,{0})"))
            .unwrap()
            .verify(strict())
            .is_err()
    );
}

#[test]
fn call_aliases_ignore_dynamic_flags_and_get_rng_seed_is_not_arbitrary_backend_data() {
    let source = "HloModule m body { ROOT p = f32[2] parameter(0) } ENTRY main { x = f32[<=2] parameter(0) ROOT c = f32[2] call(x), to_apply=body }";
    let with_alias = |source: &str| {
        let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
        let entry = wire
            .computations
            .iter_mut()
            .find(|c| c.id == wire.entry_computation_id)
            .unwrap();
        entry
            .instructions
            .iter_mut()
            .find(|i| i.id == entry.root_id)
            .unwrap()
            .output_operand_aliasing
            .push(xlahlo::proto::xla_data::OutputOperandAliasing::default());
        Module::from_proto(wire).unwrap()
    };
    let module = with_alias(source);
    module.verify(Default::default()).unwrap();
    assert!(module.verify(strict()).is_err());
    let source = source.replace("f32[<=2]", "f32[2]");
    let module = with_alias(&source);
    module.verify(strict()).unwrap();
    assert!(text::print_module(&module).is_err());
    assert!(
        text::parse_module(&source.replace(
            "to_apply=body",
            "to_apply=body, output_to_operand_aliasing={{}: (0,{})}"
        ))
        .is_err()
    );
    let source = "HloModule m ENTRY main { ROOT c = u64[] custom-call(), custom_call_target=\"GetRngSeed\" }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    for source in [
        source.replace("u64[]", "s64[]"),
        source.replace("u64[]", "u64[1]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}

#[test]
fn call_thread_policy_is_opt_in_while_fusion_thread_policy_is_not() {
    let source = "HloModule m body { ROOT p = s32[] parameter(0) } ENTRY main { x = s32[] parameter(0) ROOT c = s32[] call(x), to_apply=body }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    wire.computations
        .iter_mut()
        .find(|c| c.name == "body")
        .unwrap()
        .execution_thread = "worker".into();
    let module = Module::from_proto(wire.clone()).unwrap();
    module.verify(strict()).unwrap();
    assert_eq!(
        module
            .verify(VerifyOptions {
                verify_call_nested_computation_thread_name: true,
                ..strict()
            })
            .unwrap_err()
            .category,
        "execution-thread"
    );
    let instruction = wire
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
        .find(|i| i.opcode == "call")
        .unwrap();
    instruction.opcode = "fusion".into();
    instruction.fusion_kind = "kLoop".into();
    assert_eq!(
        Module::from_proto(wire)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "execution-thread"
    );
}
