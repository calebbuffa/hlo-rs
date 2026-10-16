use xlahlo::{Instruction, Module, Opcode, PrimitiveType as P, Shape, text};

#[test]
fn every_opcode_handles_malformed_type_domains_without_panicking() {
    let shapes = [
        Shape::token(),
        Shape::opaque(),
        Shape::tuple(vec![]).unwrap(),
        Shape::scalar(P::PRED).unwrap(),
        Shape::scalar(P::F32).unwrap(),
        Shape::array(P::S32, vec![0, 2]).unwrap(),
        Shape::scalar(P::C128).unwrap(),
    ];
    for &opcode in Opcode::ALL {
        for shape in &shapes {
            let result = std::panic::catch_unwind(|| {
                let mut module = Module::new("robustness");
                let c = module.add_computation("main").unwrap();
                let mut operands = Vec::new();
                for index in 0..opcode.arity().unwrap_or(0) {
                    let prototype =
                        Instruction::new(format!("p{index}"), Opcode::Parameter, shape.clone())
                            .with_fields(|i| i.parameter_number = index as i64)
                            .unwrap();
                    operands.push(module.add_instruction(c, prototype, &[], &[]).unwrap());
                }
                let root = module
                    .add_instruction(
                        c,
                        Instruction::new("root", opcode, shape.clone()),
                        &operands,
                        &[],
                    )
                    .unwrap();
                module.set_root(c, root).unwrap();
                module.set_entry(c).unwrap();
                let _ = module.verify(Default::default());
            });
            assert!(result.is_ok(), "{opcode:?} and {shape:?}");
        }
    }
}

#[test]
fn small_arbitrary_text_is_rejected_without_parser_panics() {
    let alphabet = [
        '(', ')', '[', ']', '{', '}', '=', ',', ':', '%', '?', '0', 'a', ' ', '\u{03bb}',
    ];
    for &a in &alphabet {
        for &b in &alphabet {
            for &c in &alphabet {
                let source = format!("{a}{b}{c}");
                assert!(std::panic::catch_unwind(|| text::parse_shape(&source)).is_ok());
                assert!(std::panic::catch_unwind(|| text::parse_module(&source)).is_ok());
            }
        }
    }
}

#[test]
fn literal_resource_limit_is_checked_before_allocation() {
    let source = "HloModule m ENTRY main { ROOT c = u8[1000001] constant({...}) }";
    assert_eq!(
        text::parse_module(source).unwrap_err().category,
        "resource-limit"
    );
}

#[test]
fn unknown_protobuf_fields_are_retained_but_not_silently_printed() {
    let mut wire = text::parse_module("HloModule m ENTRY main { ROOT c = u8[] constant(2) }")
        .unwrap()
        .to_proto()
        .unwrap();
    wire.computations[0].instructions[0]
        .literal
        .as_mut()
        .unwrap()
        .special_fields
        .mut_unknown_fields()
        .add_varint(999, 1);
    let module = Module::from_proto(wire).unwrap();
    assert_eq!(
        text::print_module(&module).unwrap_err().category,
        "unsupported"
    );
}
