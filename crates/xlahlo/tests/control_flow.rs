use xlahlo::{Dimension, Instruction, Module, Opcode, PrimitiveType as P, Shape, inference, text};

const WHILE: &str = r#"
HloModule loop
body {
  p = f32[] parameter(0)
  ROOT n = f32[] negate(p)
}
condition {
  p = f32[] parameter(0)
  ROOT stop = pred[] constant(false)
}
ENTRY main {
  initial = f32[] parameter(0)
  ROOT loop = f32[] while(initial), condition=condition, body=body
}
"#;

#[test]
fn while_wire_order_is_body_then_condition_and_round_trips() {
    let module = text::parse_module(WHILE).unwrap();
    module.verify(Default::default()).unwrap();
    let root = module
        .instruction(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    let names: Vec<_> = root
        .called_computation_ids
        .iter()
        .map(|id| {
            module
                .as_proto()
                .computations
                .iter()
                .find(|c| c.id == *id)
                .unwrap()
                .name
                .as_str()
        })
        .collect();
    assert_eq!(names, ["body", "condition"]);
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("body=%body, condition=%condition"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn while_rejects_invalid_condition_body_and_callee_cardinality() {
    for invalid in [
        WHILE.replace("pred[] constant(false)", "s32[] constant(0)"),
        WHILE.replace("f32[] negate(p)", "s32[] convert(p)"),
        WHILE.replace("parameter(0)\n  ROOT stop", "parameter(1)\n  ROOT stop"),
        WHILE.replace("condition=condition, body=body", "to_apply=body"),
        WHILE.replace(
            "condition=condition, body=body",
            "condition=condition, body=condition",
        ),
    ] {
        assert!(
            text::parse_module(&invalid)
                .and_then(|m| m.verify(Default::default()))
                .is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn calls_check_signature_result_and_composite_frontend_attributes() {
    let source = r#"
HloModule call
callee { p = f32[2] parameter(0) ROOT n = f32[2] negate(p) }
ENTRY main { p = f32[2] parameter(0) ROOT c = f32[2] call(p), to_apply=callee }
"#;
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    for invalid in [
        source.replace("call(p)", "call(p,p)"),
        source.replace("c = f32[2]", "c = f32[3]"),
        source.replace("ENTRY main { p = f32[2]", "ENTRY main { p = s32[2]"),
    ] {
        assert!(
            text::parse_module(&invalid)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let mut wire = module.to_proto().unwrap();
    let call = wire
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
        .find(|i| i.opcode == "call")
        .unwrap();
    call.is_composite = true;
    assert_eq!(
        Module::from_proto(wire.clone())
            .unwrap()
            .verify(Default::default())
            .unwrap_err()
            .category,
        "composite"
    );
    let call = wire
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
        .find(|i| i.opcode == "call")
        .unwrap();
    let mut attrs = xlahlo::proto::xla_data::FrontendAttributes::new();
    attrs.map.insert("composite.name".into(), "example".into());
    attrs.map.insert("composite.attributes".into(), "{}".into());
    attrs.map.insert("composite.version".into(), "2".into());
    call.frontend_attributes = Some(attrs).into();
    Module::from_proto(wire.clone())
        .unwrap()
        .verify(Default::default())
        .unwrap();
    let call = wire
        .computations
        .iter_mut()
        .flat_map(|c| &mut c.instructions)
        .find(|i| i.opcode == "call")
        .unwrap();
    call.frontend_attributes
        .as_mut()
        .unwrap()
        .map
        .insert("composite.version".into(), "-1".into());
    assert!(
        Module::from_proto(wire)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn predicate_and_indexed_conditionals_validate_all_branches_and_round_trip() {
    let predicate = r#"
HloModule conditional
yes { p = f32[] parameter(0) ROOT n = f32[] negate(p) }
no { ROOT p = f32[] parameter(0) }
ENTRY main {
  selector = pred[] parameter(0)
  p = f32[] parameter(1)
  ROOT c = f32[] conditional(selector, p, p), false_computation=no, true_computation=yes
}
"#;
    let indexed = predicate
        .replace("pred[] parameter", "s32[] parameter")
        .replace(
            "false_computation=no, true_computation=yes",
            "branch_computations={yes,no}",
        );
    for source in [predicate.to_owned(), indexed] {
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
        let invalid = source.replace("conditional(selector, p, p)", "conditional(selector, p)");
        assert!(
            text::parse_module(&invalid)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
    let one = predicate.replace(
        "false_computation=no, true_computation=yes",
        "branch_computations={yes}",
    );
    assert!(
        text::parse_module(&one)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn conditional_inference_unions_dynamic_flags_recursively() {
    use xlahlo::proto::xla_data::ProgramShapeProto;
    let parameter = Shape::scalar(P::F32).unwrap();
    let branch = |dimension| ProgramShapeProto {
        parameters: vec![parameter.as_proto().clone()],
        result: Some(
            Shape::tuple(vec![
                Shape::tuple(vec![Shape::dynamic_array(P::F32, vec![dimension]).unwrap()]).unwrap(),
            ])
            .unwrap()
            .into_proto(),
        )
        .into(),
        ..Default::default()
    };
    let result = inference::infer_conditional(
        &Shape::scalar(P::PRED).unwrap(),
        &[parameter.clone(), parameter.clone()],
        &[branch(Dimension::Static(4)), branch(Dimension::Bounded(4))],
    )
    .unwrap();
    assert_eq!(
        result.subshape(&[0, 0]).unwrap().dimension(0).unwrap(),
        Dimension::Bounded(4)
    );
}

#[test]
fn callee_changes_are_atomic_and_cannot_form_recursion() {
    let mut m = Module::new("calls");
    let a = m.add_computation("a").unwrap();
    let b = m.add_computation("b").unwrap();
    let shape = Shape::scalar(P::F32).unwrap();
    let p = m
        .add_instruction(
            b,
            Instruction::new("p", Opcode::Parameter, shape.clone()),
            &[],
            &[],
        )
        .unwrap();
    m.set_root(b, p).unwrap();
    let input = m
        .add_instruction(
            a,
            Instruction::new("p", Opcode::Parameter, shape.clone()),
            &[],
            &[],
        )
        .unwrap();
    let call = m
        .add_instruction(
            a,
            Instruction::new("c", Opcode::Call, shape),
            &[input],
            &[b],
        )
        .unwrap();
    m.set_root(a, call).unwrap();
    m.set_entry(a).unwrap();
    m.verify(Default::default()).unwrap();
    let before = m.to_proto().unwrap();
    assert!(m.set_called_computations(call, &[a]).is_err());
    assert_eq!(m.to_proto().unwrap(), before);
}

#[test]
fn explicit_control_edges_round_trip_and_reject_unknown_or_cyclic_edges() {
    let source = "HloModule m ENTRY main { p = f32[] parameter(0) ROOT n = f32[] negate(p), control-predecessors={p} }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("control-predecessors={%p}"));
    let again = text::parse_module(&printed).unwrap();
    assert_eq!(
        again
            .instruction(again.root(again.entry().unwrap()).unwrap())
            .unwrap()
            .control_predecessor_ids
            .len(),
        1
    );
    assert!(
        text::parse_module(&source.replace("predecessors={p}", "predecessors={missing}")).is_err()
    );
    assert!(text::parse_module(&source.replace("predecessors={p}", "predecessors={n}")).is_err());
}
