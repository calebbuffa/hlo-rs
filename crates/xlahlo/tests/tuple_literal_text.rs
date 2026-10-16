use xlahlo::{interchange, proto::xla_data::LiteralProto, text};

fn payload(source: &str) -> LiteralProto {
    text::parse_module(source).unwrap().as_proto().computations[0].instructions[0]
        .literal
        .as_ref()
        .unwrap()
        .clone()
}

#[test]
fn nested_tuple_literals_and_leaf_layouts_round_trip() {
    for source in [
        "HloModule m ENTRY main { ROOT c = () constant(()) }",
        "HloModule m ENTRY main { ROOT c = (f32[],s32[2]) constant((1.5,{2,3})) }",
        "HloModule m ENTRY main { ROOT c = ((),(u8[0],f32[2,2]{0,1}),pred[]) constant(((),({},{ {1,2},{3,4} }),true)) }",
    ] {
        let module = text::parse_module(source).unwrap_or_else(|error| panic!("{source}: {error}"));
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let after = payload(&printed);
        assert_eq!(
            interchange::encode(&payload(source)).unwrap(),
            interchange::encode(&after).unwrap()
        );
    }
}

#[test]
fn tuple_literal_arity_and_aggregate_element_budget_are_enforced() {
    for source in [
        "HloModule m ENTRY main { ROOT c = (f32[],s32[]) constant((1)) }",
        "HloModule m ENTRY main { ROOT c = (f32[]) constant((1,2)) }",
        "HloModule m ENTRY main { ROOT c = () constant((1)) }",
    ] {
        assert!(text::parse_module(source).is_err());
    }
    let source = "HloModule m ENTRY main { ROOT c = (f32[2],f32[2]) constant(({1,2},{3,4})) }";
    assert!(
        text::parse_module_with_options(
            source,
            text::ParseOptions {
                max_literal_elements: 3,
                ..Default::default()
            }
        )
        .is_err()
    );
    text::parse_module_with_options(
        source,
        text::ParseOptions {
            max_literal_elements: 4,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn elided_tuple_leaves_use_existing_generation_and_relayout() {
    let source = "HloModule m ENTRY main { ROOT c = (f32[2],s32[2]) constant(({...},{...})) }";
    let module = text::parse_module_with_options(
        source,
        text::ParseOptions {
            fill_shortform_constants_with_random_values: false,
            ..Default::default()
        },
    )
    .unwrap();
    let literal = module.as_proto().computations[0].instructions[0]
        .literal
        .as_ref()
        .unwrap();
    assert_eq!(literal.tuple_literals[0].f32s, [0.0, 0.0]);
    assert_eq!(literal.tuple_literals[1].s32s, [0, 0]);
    text::print_module(&module).unwrap();
}
