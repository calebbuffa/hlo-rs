use xlahlo::{VerifyOptions, text};
fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}
const GATHER: &str = "HloModule m ENTRY main { p = f32[2] parameter(0) s = (f32[2],f32[4]) all-gather-start(p), dimensions={0}, replica_groups={{0,1}} ROOT d = f32[4] all-gather-done(s) }";
const REDUCE: &str = "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT add = f32[] add(a,b) } ENTRY main { p = f32[2] parameter(0) s = f32[2] all-reduce-start(p), to_apply=sum ROOT d = f32[2] all-reduce-done(s) }";

#[test]
fn dedicated_collective_contexts_and_attributes_round_trip() {
    for source in [GATHER, REDUCE] {
        let module = text::parse_module(source).unwrap();
        module.verify(strict()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let parsed = text::parse_module(&printed).unwrap();
        parsed.verify(strict()).unwrap();
        assert_eq!(printed, text::print_module(&parsed).unwrap());
    }
    for changed in [
        GATHER.replace("s = (f32[2],f32[4])", "s = (f32[3],f32[4])"),
        GATHER.replace("s = (f32[2],f32[4])", "s = (f32[2],f32[4],u32[])"),
        GATHER.replace("d = f32[4]", "d = f32[2]"),
        REDUCE.replace("s = f32[2]", "s = f32[3]"),
    ] {
        assert!(
            text::parse_module(&changed)
                .unwrap()
                .verify(strict())
                .is_err()
        );
    }
}

#[test]
fn starts_have_one_real_user_but_intermediaries_do_not_require_a_terminal_done() {
    let source = GATHER.replace(
        "ROOT d =",
        "control = () custom-call(s), custom_call_target=\"control_dep\" ROOT d =",
    );
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let changed = GATHER.replace("ROOT d =", "extra = f32[4] all-gather-done(s) ROOT d =");
    assert_eq!(
        text::parse_module(&changed)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "async-pair"
    );
    let changed = GATHER.replace(
        "ROOT d = f32[4] all-gather-done(s)",
        "ROOT t = ((f32[2],f32[4])) tuple(s)",
    );
    text::parse_module(&changed)
        .unwrap()
        .verify(strict())
        .unwrap();
}

#[test]
fn collective_done_traces_tuple_indices_copies_barriers_and_allowed_custom_calls() {
    let intermediaries = "t = ((f32[2],f32[4])) tuple(s) g = (f32[2],f32[4]) get-tuple-element(t), index=0 c = (f32[2],f32[4]) copy(g) b = (f32[2],f32[4]) opt-barrier(c) a = (f32[2],f32[4]) custom-call(b), custom_call_target=\"Sharding\"";
    let source = GATHER
        .replace("ROOT d =", &format!("{intermediaries} ROOT d ="))
        .replace("all-gather-done(s)", "all-gather-done(a)");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    assert!(
        text::parse_module(&source.replace(
            "custom_call_target=\"Sharding\"",
            "custom_call_target=\"not-an-intermediary\""
        ))
        .unwrap()
        .verify(strict())
        .is_err()
    );
    let wrong=GATHER.replace("ROOT d = f32[4] all-gather-done(s)","t = (f32[2],(f32[2],f32[4])) tuple(p,s) g = f32[2] get-tuple-element(t), index=0 ROOT d = f32[2] all-reduce-done(g)");
    assert_eq!(
        text::parse_module(&wrong)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "async-pair"
    );
}

#[test]
fn async_dataflow_walks_while_body_and_initial_state_without_cycling() {
    let bodies = "body { ROOT p = (f32[2],f32[4]) parameter(0) } cond { p = (f32[2],f32[4]) parameter(0) ROOT f = pred[] constant(false) }";
    let source = GATHER
        .replace("ENTRY main", &format!("{bodies} ENTRY main"))
        .replace(
            "ROOT d =",
            "w = (f32[2],f32[4]) while(s), body=body, condition=cond ROOT d =",
        )
        .replace("all-gather-done(s)", "all-gather-done(w)");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let bodies = "body { p = (f32[2],f32[4]) parameter(0) a = f32[2] get-tuple-element(p), index=0 s = (f32[2],f32[4]) all-gather-start(a), dimensions={0}, replica_groups={{0,1}} g = (f32[2],f32[4]) opt-barrier(s) ROOT b = (f32[2],f32[4]) copy(g) } cond { p = (f32[2],f32[4]) parameter(0) ROOT f = pred[] constant(false) }";
    let source = format!(
        "HloModule m {bodies} ENTRY main {{ p = f32[2] parameter(0) q = f32[4] parameter(1) t = (f32[2],f32[4]) tuple(p,q) w = (f32[2],f32[4]) while(t), body=body, condition=cond ROOT d = f32[4] all-gather-done(w) }}"
    );
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
}

#[test]
fn intermediary_operand_matching_does_not_relax_dedicated_copy_done() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) s = (f32[2],f32[2],u32[]) copy-start(p) t = ((f32[2],f32[2],u32[])) tuple(s) g = (f32[2],f32[2],u32[]) get-tuple-element(t), index=0 ROOT d = f32[2] copy-done(g) }";
    assert_eq!(
        text::parse_module(source)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "async-pair"
    );
}

#[test]
fn one_while_is_one_caller_even_if_body_and_condition_share_a_computation() {
    let source = "HloModule m reducer { a = pred[] parameter(0) b = pred[] parameter(1) ROOT c = pred[] or(a,b) } shared { p = pred[] parameter(0) d = pred[] all-reduce-done(p) s = pred[] all-reduce-start(p), to_apply=reducer ROOT b = pred[] opt-barrier(s) } ENTRY main { p = pred[] parameter(0) ROOT w = pred[] while(p), body=shared, condition=shared }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
}
