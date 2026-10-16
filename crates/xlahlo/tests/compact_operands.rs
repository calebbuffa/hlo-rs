use xlahlo::{PrimitiveType, text};

const SOURCE: &str = "HloModule m, is_scheduled=true ENTRY main {
 p = (f32[2],(s32[],f32[])) parameter(0)
 a = copy(f32[2] p#0)
 b = add(p#0,p#00)
 c = negate(p#1#0)
 ROOT r = tuple(a,b,c)
}";

#[test]
fn compact_projections_are_shared_and_print_as_ordinary_instructions() {
    let module = text::parse_module(SOURCE).unwrap();
    module.verify(Default::default()).unwrap();
    let computation = module
        .as_proto()
        .computations
        .iter()
        .find(|c| c.id == module.as_proto().entry_computation_id)
        .unwrap();
    let projections = computation
        .instructions
        .iter()
        .filter(|instruction| instruction.opcode == "get-tuple-element")
        .collect::<Vec<_>>();
    assert_eq!(projections.len(), 3);
    assert!(
        projections
            .iter()
            .any(|instruction| instruction.name == "p_1_0")
    );
    let printed = text::print_module(&module).unwrap();
    assert!(!printed.contains('#'));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn typed_projection_annotations_describe_the_selected_leaf() {
    let module = text::parse_module(
        "HloModule m ENTRY main {
 p = ((f32[2]{0}),token[]) parameter(0)
 ROOT r = copy(f32[2]{0} p#0#0)
}",
    )
    .unwrap();
    let shape = module
        .shape(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert_eq!(shape.element_type(), PrimitiveType::F32);
    assert_eq!(shape.dimensions(), [2]);
}

#[test]
fn compact_projection_names_do_not_capture_user_instruction_names() {
    let module = text::parse_module(
        "HloModule m ENTRY main {
 p = (f32[]) parameter(0)
 p_0 = f32[] constant(1)
 ROOT r = add(p#0,p_0)
}",
    )
    .unwrap();
    module.verify(Default::default()).unwrap();
    let computation = &module.as_proto().computations[0];
    let names = computation
        .instructions
        .iter()
        .map(|i| &i.name)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names.len(), computation.instructions.len());
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn malformed_compact_operands_return_checked_errors() {
    for body in [
        "p = (f32[]) parameter(0) ROOT r = copy(p#-1)",
        "p = (f32[]) parameter(0) ROOT r = copy(p#1)",
        "p = (f32[]) parameter(0) ROOT r = copy(p#0#0)",
        "p = (f32[]) parameter(0) ROOT r = copy(f64[] p#0)",
        "ROOT r = f32[] copy(p#0) p = (f32[]) parameter(0)",
        "p = (f32[]) parameter(0) ROOT r = copy(p#)",
    ] {
        assert!(
            text::parse_module(&format!("HloModule m ENTRY main {{ {body} }}")).is_err(),
            "{body}"
        );
    }
}
