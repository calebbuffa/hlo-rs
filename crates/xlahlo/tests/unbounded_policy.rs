use xlahlo::{Error, InstructionId, Module, Result, Shape, VerifyOptions, text};

fn allow(_: &Module, _: InstructionId) -> bool {
    true
}

fn options() -> VerifyOptions {
    VerifyOptions {
        supported_unbounded_dynamic_op: Some(allow),
        ..Default::default()
    }
}

#[test]
fn unbounded_support_is_a_per_instruction_predicate_and_never_called_for_static_shapes() {
    let module = text::parse_module(
        "HloModule m ENTRY main { p = f32[?] parameter(0) ROOT r = f32[?] negate(p) }",
    )
    .unwrap();
    assert_eq!(
        module.verify(Default::default()).unwrap_err().category,
        "unbounded"
    );
    fn parameters_only(module: &Module, id: InstructionId) -> bool {
        module.instruction(id).unwrap().opcode == "parameter"
    }
    assert_eq!(
        module
            .verify(VerifyOptions {
                supported_unbounded_dynamic_op: Some(parameters_only),
                ..Default::default()
            })
            .unwrap_err()
            .category,
        "unbounded"
    );
    module.verify(options()).unwrap();
    fn must_not_be_called(_: &Module, _: InstructionId) -> bool {
        panic!("static instruction reached unbounded policy")
    }
    text::parse_module("HloModule m ENTRY main { ROOT p = f32[<=3] parameter(0) }")
        .unwrap()
        .verify(VerifyOptions {
            supported_unbounded_dynamic_op: Some(must_not_be_called),
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn predicate_has_graph_context_and_checks_nested_shapes() {
    fn context(module: &Module, id: InstructionId) -> bool {
        let instruction = module.instruction(id).unwrap();
        instruction.opcode == "parameter"
            || module
                .operands(id)
                .unwrap()
                .iter()
                .all(|&operand| module.instruction(operand).unwrap().opcode == "parameter")
    }
    let module=text::parse_module("HloModule m ENTRY main { p = (f32[?],(s32[],bf16[?])) parameter(0) ROOT r = (f32[?],(s32[],bf16[?])) opt-barrier(p) }").unwrap();
    module
        .verify(VerifyOptions {
            supported_unbounded_dynamic_op: Some(context),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        module.verify(Default::default()).unwrap_err().category,
        "unbounded"
    );
}

#[test]
fn approval_does_not_bypass_shape_or_operation_contracts() {
    for source in [
        "HloModule m ENTRY main { p = f32[?] parameter(0) ROOT r = f32[2,3] negate(p) }",
        "HloModule m ENTRY main { p = s32[?] parameter(0) ROOT r = s32[?] sine(p) }",
    ] {
        assert!(
            text::parse_module(source)
                .unwrap()
                .verify(options())
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn default_bitcast_size_uses_unbounded_sentinel_but_respects_explicit_callbacks() {
    let module = text::parse_module(
        "HloModule m ENTRY main { p = u8[?] parameter(0) ROOT r = f64[?,2] bitcast(p) }",
    )
    .unwrap();
    let options = VerifyOptions {
        layout_sensitive: true,
        ..options()
    };
    module.verify(options).unwrap();
    fn fail(_: &Shape) -> Result<i64> {
        Err(Error::new("test-size", "explicit callback reached"))
    }
    assert_eq!(
        module
            .verify(VerifyOptions {
                shape_size: Some(fail),
                ..options
            })
            .unwrap_err()
            .category,
        "test-size"
    );
    let bounded = text::parse_module(
        "HloModule m ENTRY main { p = u8[?] parameter(0) ROOT r = f64[2] bitcast(p) }",
    )
    .unwrap();
    assert_eq!(bounded.verify(options).unwrap_err().category, "bitcast");
    fn constant_size(_: &Shape) -> Result<i64> {
        Ok(64)
    }
    bounded
        .verify(VerifyOptions {
            shape_size: Some(constant_size),
            ..options
        })
        .unwrap();
}

#[test]
fn module_alias_sizing_shares_the_unbounded_default_and_callback_override() {
    let module=text::parse_module("HloModule m, input_output_alias={{}: (0,{})} ENTRY main { p = f32[?] parameter(0) ROOT r = f64[?] convert(p) }").unwrap();
    let options = VerifyOptions {
        layout_sensitive: true,
        ..options()
    };
    module.verify(options).unwrap();
    fn fail(_: &Shape) -> Result<i64> {
        Err(Error::new(
            "test-size",
            "explicit alias size callback reached",
        ))
    }
    assert_eq!(
        module
            .verify(VerifyOptions {
                shape_size: Some(fail),
                ..options
            })
            .unwrap_err()
            .category,
        "test-size"
    );
}
