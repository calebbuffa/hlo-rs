use xlahlo::{PrimitiveType as P, text};

const NEGATE: &str = "HloModule m ENTRY main {
 p = f32[2] parameter(0)
 s = ((f32[2]),f32[2],u32[]) negate-start(p)
 u = ((f32[2]),f32[2],u32[]) negate-update(s)
 ROOT d = f32[2] negate-done(u)
}";

#[test]
fn sugar_builds_real_wrapper_parameters_and_chain() {
    let module = text::parse_module(NEGATE).unwrap();
    module.verify(Default::default()).unwrap();
    let entry = module.entry().unwrap();
    let start = module
        .instructions(entry)
        .unwrap()
        .into_iter()
        .find(|&id| module.instruction(id).unwrap().opcode == "async-start")
        .unwrap();
    let called = module.instruction(start).unwrap().called_computation_ids[0];
    let wrapped = module
        .computations()
        .find(|&c| module.computation(c).unwrap().id == called)
        .unwrap();
    let root = module.root(wrapped).unwrap();
    assert_eq!(module.instruction(root).unwrap().opcode, "negate");
    assert_eq!(module.operands(root).unwrap().len(), 1);
    assert_eq!(module.shape(root).unwrap().element_type(), P::F32);
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("async-start"));
    assert!(!printed.contains("negate-start"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn start_parameters_follow_written_context_not_actual_argument_shapes() {
    let source = NEGATE
        .replace("((f32[2]),f32[2]", "((f32[3]),f32[3]")
        .replace("ROOT d = f32[2]", "ROOT d = f32[3]");
    let module = text::parse_module(&source).unwrap();
    assert!(module.verify(Default::default()).is_err());
    let body = module
        .computations()
        .find(|&c| {
            module
                .computation(c)
                .unwrap()
                .name
                .starts_with("async_wrapped")
        })
        .unwrap();
    let root = module.root(body).unwrap();
    let parameter = module.operands(root).unwrap()[0];
    assert_eq!(module.shape(parameter).unwrap().dimensions(), [3]);
}

#[test]
fn chain_opcode_consistency_and_dedicated_families_are_checked() {
    assert!(text::parse_module(&NEGATE.replace("negate-done", "abs-done")).is_err());
    assert!(text::parse_module(&NEGATE.replace("negate-update", "abs-update")).is_err());
    assert!(text::parse_module(&NEGATE.replace("negate-update", "copy-update")).is_err());
    assert!(text::parse_module(&NEGATE.replace("negate-update", "send-update")).is_err());
    assert!(text::parse_module(&NEGATE.replace("negate-start", "not-an-op-start")).is_err());
}

#[test]
fn wrapper_names_do_not_collide_with_user_computations_or_instructions() {
    let source = NEGATE.replace(
        "ENTRY main",
        "async_wrapped { ROOT async_op = f32[] parameter(0) } ENTRY main",
    );
    let module = text::parse_module(&source).unwrap();
    module.verify(Default::default()).unwrap();
    let names = module
        .computations()
        .map(|c| module.computation(c).unwrap().name.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names.len(), 3);
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn shared_metadata_is_outer_and_wrapped_but_sharding_is_only_outer() {
    let source = NEGATE.replace("negate-start(p)", "negate-start(p), async_execution_thread=\"worker\", metadata={op_name=\"start\"}, sharding={replicated}")
        .replace("negate-done(u)", "negate-done(u), metadata={op_name=\"done\"}, statistics={index=3,a=-0.0}");
    let module = text::parse_module(&source).unwrap();
    module.verify(Default::default()).unwrap();
    let body = module
        .computations()
        .find(|&c| {
            module
                .computation(c)
                .unwrap()
                .name
                .starts_with("async_wrapped")
        })
        .unwrap();
    assert_eq!(module.computation(body).unwrap().execution_thread, "worker");
    let root = module.instruction(module.root(body).unwrap()).unwrap();
    assert_eq!(root.metadata.get_or_default().op_name, "done");
    assert!(root.sharding.is_none());
    assert_eq!(
        root.statistics_viz.get_or_default().stat_index_to_visualize,
        3
    );
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn late_binding_updates_dummy_parameters_and_result_shapes() {
    let source = NEGATE.replace("s = ((f32[2]),f32[2],u32[])", "s = ((),(),u32[])");
    let module = text::parse_module(&source).unwrap();
    let body = module
        .computations()
        .find(|&c| {
            module
                .computation(c)
                .unwrap()
                .name
                .starts_with("async_wrapped")
        })
        .unwrap();
    let root = module.root(body).unwrap();
    assert_eq!(module.shape(root).unwrap().dimensions(), [2]);
    let parameter = module.operands(root).unwrap()[0];
    assert_eq!(module.shape(parameter).unwrap().dimensions(), [2]);
    assert!(module.verify(Default::default()).is_err());
    let printed = text::print_module(&module).unwrap();
    assert!(
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn call_and_fusion_wrappers_bind_to_callee_signatures() {
    for (opcode, attrs) in [("call", "to_apply=f"), ("fusion", "kind=kLoop, calls=f")] {
        let source = format!(
            "HloModule m
f {{ ROOT p = f32[2] parameter(0) }}
ENTRY main {{
 s = ((),(),u32[]) {opcode}-start(), {attrs}
 ROOT d = f32[2] {opcode}-done(s)
}}"
        );
        let module = text::parse_module(&source).unwrap();
        let body = module
            .computations()
            .find(|&c| {
                module
                    .computation(c)
                    .unwrap()
                    .name
                    .starts_with("async_wrapped")
            })
            .unwrap();
        let root = module.root(body).unwrap();
        assert_eq!(module.instruction(root).unwrap().opcode, opcode);
        assert_eq!(module.shape(root).unwrap().dimensions(), [2]);
        let parameter = module.operands(root).unwrap()[0];
        assert_eq!(module.shape(parameter).unwrap().dimensions(), [2]);
        assert!(module.verify(Default::default()).is_err());
    }
}

#[test]
fn constant_sugar_has_separate_outer_operands_and_literal_constructor() {
    let module = text::parse_module(
        "HloModule m ENTRY main {
 s = ((),f32[],u32[]) constant-start()(1)
 ROOT d = f32[] constant-done(s)
}",
    )
    .unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    let constant = module
        .as_proto()
        .computations
        .iter()
        .flat_map(|c| &c.instructions)
        .find(|i| i.opcode == "constant")
        .unwrap();
    assert_eq!(
        xlahlo::Literal::from_proto(constant.literal.get_or_default().clone())
            .unwrap()
            .bits()
            .unwrap(),
        vec![1.0_f32.to_bits() as u128]
    );
}

#[test]
fn update_and_done_do_not_accept_wrapped_root_attributes() {
    for source in [
        NEGATE.replace("negate-done(u)", "negate-done(u), dimensions={0}"),
        NEGATE.replace("negate-update(s)", "negate-update(s), calls=main"),
        NEGATE.replace(
            "negate-done(u)",
            "negate-done(u), result_accuracy={mode=highest}",
        ),
        NEGATE.replace("u = ((f32[2]),f32[2],u32[])", "u = f32[2]"),
    ] {
        assert!(text::parse_module(&source).is_err(), "{source}");
    }
}
