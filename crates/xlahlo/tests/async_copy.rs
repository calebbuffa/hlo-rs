use xlahlo::{Module, VerifyOptions, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}
const SOURCE: &str = "HloModule m ENTRY main { p = f32[2,3]{1,0} parameter(0) s = (f32[2,3]{1,0},f32[2,3]{1,0},u32[]) copy-start(p), cross_program_prefetch_index=0 ROOT d = f32[2,3]{1,0} copy-done(s) }";

#[test]
fn copy_context_and_prefetch_presence_round_trip() {
    let module = text::parse_module(SOURCE).unwrap();
    module.verify(strict()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("cross_program_prefetch_index=0"));
    text::parse_module(&printed)
        .unwrap()
        .verify(strict())
        .unwrap();
    let module =
        text::parse_module(&SOURCE.replace(", cross_program_prefetch_index=0", "")).unwrap();
    module.verify(strict()).unwrap();
    assert!(
        !text::print_module(&module)
            .unwrap()
            .contains("cross_program_prefetch_index")
    );
    text::parse_module(&SOURCE.replace("index=0", "index=-1"))
        .unwrap()
        .verify(strict())
        .unwrap();
    assert!(text::parse_module(&SOURCE.replace("index=0", "index=2147483648")).is_err());
}

#[test]
fn copy_pair_requires_one_real_user_and_the_done_requires_a_direct_start() {
    for source in [
        SOURCE.replace("ROOT d =", "extra = f32[2,3]{1,0} copy-done(s) ROOT d ="),
        SOURCE.replace(
            "ROOT d = f32[2,3]{1,0} copy-done(s)",
            "ROOT d = (f32[2,3]{1,0},f32[2,3]{1,0},u32[]) opt-barrier(s)",
        ),
        SOURCE.replace(
            "ROOT d = f32[2,3]{1,0} copy-done(s)",
            "ROOT d = f32[2,3]{1,0} copy-done(p)",
        ),
    ] {
        let module = text::parse_module(&source).unwrap();
        if source.contains("opt-barrier(s)") {
            module.verify(strict()).unwrap();
        } else {
            assert!(module.verify(strict()).is_err());
        }
    }
    let module = text::parse_module(&SOURCE.replace(
        "ROOT d =",
        "control = () custom-call(s), custom_call_target=\"control_dep\" ROOT d =",
    ))
    .unwrap();
    module.verify(strict()).unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.computations[0].root_id = wire.computations[0]
        .instructions
        .iter()
        .find(|i| i.opcode == "copy-start")
        .unwrap()
        .id;
    wire.computations[0]
        .instructions
        .retain(|i| matches!(i.opcode.as_str(), "parameter" | "copy-start"));
    assert_eq!(
        Module::from_proto(wire)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "async-pair"
    );
}

#[test]
fn copy_context_allows_memory_space_changes_but_not_physical_transposes() {
    let mut wire = text::parse_module(SOURCE).unwrap().to_proto().unwrap();
    let computation = &mut wire.computations[0];
    let start = computation
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "copy-start")
        .unwrap();
    start.shape.as_mut().unwrap().tuple_shapes[0]
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 5;
    computation
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "copy-done")
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 5;
    let module = Module::from_proto(wire.clone()).unwrap();
    module.verify(strict()).unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "copy-start")
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .tuple_shapes[0]
        .layout
        .as_mut()
        .unwrap()
        .minor_to_major
        .reverse();
    let module = Module::from_proto(wire).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(module.verify(strict()).is_err());
}

#[test]
fn invalid_copy_result_shapes_are_not_accepted_by_pairing_alone() {
    for source in [
        SOURCE.replace("u32[]", "s32[]"),
        SOURCE.replace("d = f32[2,3]", "d = f32[3,2]"),
        SOURCE.replace(
            "(f32[2,3]{1,0},f32[2,3]{1,0},u32[])",
            "(f32[2,3]{1,0},f32[3,2]{1,0},u32[])",
        ),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err()
        );
    }
}
