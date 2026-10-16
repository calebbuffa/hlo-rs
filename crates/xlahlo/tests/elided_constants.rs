use xlahlo::{Literal, PrimitiveType as P, Shape, text};

fn root_literal(module: &xlahlo::Module) -> Literal {
    Literal::from_proto(
        module
            .instruction(module.root(module.entry().unwrap()).unwrap())
            .unwrap()
            .literal
            .as_ref()
            .unwrap()
            .clone(),
    )
    .unwrap()
}

#[test]
fn parser_defaults_synthesize_process_wide_counter_data_and_relayout_it() {
    let source = "HloModule m ENTRY main { ROOT c = u32[2,2]{0,1} constant({...}) }";
    let first = text::parse_module(source).unwrap();
    first.verify(Default::default()).unwrap();
    let bits = root_literal(&first).bits().unwrap();
    assert_eq!(bits[2], bits[0].wrapping_add(1) & u128::from(u32::MAX));
    assert_eq!(bits[1], bits[0].wrapping_add(2) & u128::from(u32::MAX));
    assert_eq!(bits[3], bits[0].wrapping_add(3) & u128::from(u32::MAX));
    let second = text::parse_module(source).unwrap();
    assert_eq!(
        root_literal(&second).bits().unwrap()[0],
        bits[0].wrapping_add(4) & u128::from(u32::MAX)
    );
    let printed = text::print_module(&first).unwrap();
    assert!(!printed.contains("..."));
    let again = text::parse_module(&printed).unwrap();
    assert_eq!(root_literal(&again).bits().unwrap(), bits);
}

#[test]
fn disabling_shortform_fill_matches_upstream_zero_initialized_literals() {
    for shape in ["f32[3]", "pred[7]", "s4[5]", "c128[2]", "f16[2,3]", "u8[0]"] {
        let source = format!("HloModule m ENTRY main {{ ROOT c = {shape} constant({{...}}) }}");
        let module = text::parse_module_with_options(
            &source,
            text::ParseOptions {
                fill_shortform_constants_with_random_values: false,
                ..Default::default()
            },
        )
        .unwrap();
        module.verify(Default::default()).unwrap();
        assert!(
            root_literal(&module)
                .bits()
                .unwrap()
                .iter()
                .all(|&bits| bits == 0)
        );
    }
    assert!(
        text::parse_module("HloModule m ENTRY main { ROOT c = f32[] constant({...}) }").is_err()
    );
    assert!(
        text::parse_module("HloModule m ENTRY main { ROOT c = f32[2] constant(...) }").is_err()
    );
    assert!(
        text::parse_module("HloModule m ENTRY main { ROOT c = f32[2,2] constant({{...},{...}}) }")
            .is_err()
    );
}

#[test]
fn native_literal_storage_retains_unused_bits_without_claiming_lossless_text() {
    let shape = Shape::array(P::U4, vec![2]).unwrap();
    let literal = Literal::from_storage_bytes(shape, &[0xf1, 0xa2]).unwrap();
    assert_eq!(literal.as_proto().u4s, [0xf1, 0xa2]);
    assert_eq!(literal.bits().unwrap(), [1, 2]);
    assert!(Literal::from_storage_bytes(Shape::scalar(P::PRED).unwrap(), &[2]).is_err());
    assert!(Literal::from_storage_bytes(Shape::scalar(P::F32).unwrap(), &[0]).is_err());
    let module =
        text::parse_module("HloModule m ENTRY main { ROOT c = u4[2] constant({1,2}) }").unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.computations[0].instructions[0].literal = Some(literal.into_proto()).into();
    assert_eq!(
        text::print_module(&xlahlo::Module::from_proto(wire).unwrap())
            .unwrap_err()
            .category,
        "unsupported"
    );
}
