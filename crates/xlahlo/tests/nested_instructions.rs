use xlahlo::{PrimitiveType, text};

#[test]
fn nested_expressions_use_ordinary_inference_and_scheduled_instruction_order() {
    let source = "HloModule m, is_scheduled=true ENTRY main {
 p = f32[2] parameter(0)
 ROOT r = add(sqrt(p), f32[2] negate(copy(p)))
}";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let computation = &module.as_proto().computations[0];
    assert_eq!(computation.instructions.len(), 5);
    assert_eq!(computation.instructions.last().unwrap().name, "r");
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn nested_literals_parameters_and_compact_operands_share_constructors() {
    for source in [
        "HloModule m ENTRY main { ROOT r = add(f32[] constant(2), f32[] constant(3)) }",
        "HloModule m ENTRY main { ROOT r = copy(f32[] parameter(0)) }",
        "HloModule m ENTRY main { p = (f32[],f32[]) parameter(0) ROOT r = add(copy(p#0), negate(p#1)) }",
        "HloModule m ENTRY main { p = f32[] parameter(0) a = copy(copy(p)) ROOT r = add(a,copy.anon) }",
        "HloModule m ENTRY main { p = f32[] parameter(0) copy.anon = copy(p) ROOT r = add(copy(p),copy.anon) }",
        "HloModule m helper { ROOT r = copy(f32[] parameter(0)) } ENTRY main { ROOT r = copy(f32[] parameter(0)) }",
        "HloModule m ENTRY main { ROOT r = f32[2] broadcast(f32[] constant(1)) }",
        "HloModule m ENTRY main { t = after-all() ROOT r = token[] outfeed(negate(f32[] constant(1)),t) }",
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        text::parse_module(&text::print_module(&module).unwrap())
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    let module = text::parse_module(
        "HloModule m ENTRY main { ROOT r = tuple(s32[] constant(1), after-all()) }",
    )
    .unwrap();
    assert_eq!(
        module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap()
            .element_type(),
        PrimitiveType::TUPLE
    );
}

#[test]
fn nested_instructions_reject_attributes_and_bound_recursion() {
    for source in [
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = copy(transpose(p), dimensions={0}) }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = copy(get-tuple-element(p), index=0) }",
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = add(copy(p),) }",
        "HloModule m ENTRY main { ROOT r = copy(constant(1)) }",
        "HloModule m ENTRY main { p = f32[] parameter(0) a = copy(copy(p)) copy.anon = negate(p) ROOT r = add(a,copy.anon) }",
    ] {
        assert!(text::parse_module(source).is_err(), "{source}");
    }
    let source = |count| {
        format!(
            "HloModule m ENTRY main {{ p = f32[] parameter(0) ROOT r = {}p{} }}",
            "copy(".repeat(count),
            ")".repeat(count)
        )
    };
    text::parse_module(&source(100))
        .unwrap()
        .verify(Default::default())
        .unwrap();
    let error = text::parse_module(&source(101)).unwrap_err();
    assert!(
        error.message.contains("nested instruction depth limit"),
        "{error}"
    );
}
