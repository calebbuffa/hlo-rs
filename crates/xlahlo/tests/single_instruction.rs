use xlahlo::text;

#[test]
fn parses_shape_only_operands_into_entry_parameters() {
    let module = text::parse_instruction("f32[2] add(f32[2], f32[2])").unwrap();
    module.verify(Default::default()).unwrap();
    let entry = module.entry().unwrap();
    let computation = module.computation(entry).unwrap();
    let parameters = computation
        .instructions
        .iter()
        .filter(|instruction| instruction.opcode == "parameter")
        .collect::<Vec<_>>();
    assert_eq!(parameters.len(), 2);
    assert_eq!(parameters[0].name, "_0");
    assert_eq!(parameters[1].name, "_1");
    assert!(module.as_proto().schedule.is_some());
}

#[test]
fn parses_named_instruction_and_reuses_typed_implicit_parameters() {
    let module = text::parse_instruction("sum = f32[2] add(f32[2] lhs, f32[2] lhs)").unwrap();
    module.verify(Default::default()).unwrap();
    let computation = module.computation(module.entry().unwrap()).unwrap();
    assert_eq!(
        computation
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode == "parameter")
            .count(),
        1
    );
}

#[test]
fn parses_nested_instruction_operands() {
    let module = text::parse_instruction("f32[2] add(f32[2] negate(f32[2]), f32[2])").unwrap();
    module.verify(Default::default()).unwrap();
    let computation = module.computation(module.entry().unwrap()).unwrap();
    assert_eq!(
        computation
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode == "parameter")
            .count(),
        3
    );
    assert_eq!(
        computation
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode == "negate")
            .count(),
        1
    );
}

#[test]
fn rejects_unshaped_missing_operands_and_trailing_instructions() {
    assert!(text::parse_instruction("f32[2] add(lhs, rhs)").is_err());
    assert!(text::parse_instruction("same = f32[2] add(f32[2] same, f32[2])").is_err());
    assert!(text::parse_instruction("f32[] negate(f32[]) extra").is_err());
}
