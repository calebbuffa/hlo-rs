use xlahlo::{CollectiveContext, Module, VerifyOptions, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}
fn round_trip(source: &str) {
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    reparsed.verify(strict()).unwrap();
    assert_eq!(printed, text::print_module(&reparsed).unwrap());
}
const SIMPLE: &str = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT c = f32[2] collective-permute(p), source_target_pairs={{0,1},{1,0}} }";
const INPLACE: &str = "HloModule m ENTRY main { p = f32[2,3] parameter(0) q = f32[4,5] parameter(1) z = f32[] constant(0) offsets = (f32[],f32[]) tuple(z,z) ROOT c = f32[4,5] collective-permute(p,q,offsets,offsets), source_target_pairs={{0,1},{0,1}}, slice_sizes={{1,2}} }";

#[test]
fn regular_permute_checks_distinct_ids_and_explicit_participant_bounds() {
    round_trip(SIMPLE);
    let module = text::parse_module(SIMPLE).unwrap();
    module
        .verify(VerifyOptions {
            collective_context: CollectiveContext {
                replica_count: 2,
                partition_count: 7,
            },
            ..strict()
        })
        .unwrap();
    for pairs in ["{{0,1},{0,2}}", "{{0,1},{2,1}}", "{{-1,0}}"] {
        assert!(
            text::parse_module(&SIMPLE.replace("{{0,1},{1,0}}", pairs))
                .unwrap()
                .verify(strict())
                .is_err()
        );
    }
    let source = SIMPLE.replace("{{0,1},{1,0}}", "{{3,4}}");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    assert!(
        text::parse_module(&source)
            .unwrap()
            .verify(VerifyOptions {
                collective_context: CollectiveContext {
                    replica_count: 3,
                    partition_count: 1
                },
                ..strict()
            })
            .is_err()
    );
    text::parse_module(
        &source.replace("source_target_pairs=", "channel_id=1, source_target_pairs="),
    )
    .unwrap()
    .verify(VerifyOptions {
        collective_context: CollectiveContext {
            replica_count: 2,
            partition_count: 5,
        },
        ..strict()
    })
    .unwrap();
    assert!(text::parse_module(&SIMPLE.replace("{{0,1},{1,0}}", "{{0,1,2}}")).is_err());
}

#[test]
fn in_place_permute_uses_output_shape_and_offset_tuple_contracts() {
    round_trip(INPLACE);
    // Pinned shape verification does not require integer/scalar offset leaves,
    // equal input/output dimensions, or checks on the slice-size values.
    round_trip(&INPLACE.replace("slice_sizes={{1,2}}", "slice_sizes={{-99,999}}"));
    for source in [
        INPLACE.replace(
            "offsets = (f32[],f32[]) tuple(z,z)",
            "offsets = (f32[]) tuple(z)",
        ),
        INPLACE.replace("{{0,1},{0,1}}", "{{0,1},{0,1},{0,1}}"),
        INPLACE.replace("c = f32[4,5]", "c = f32[2,3]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(strict())
                .is_err()
        );
    }
    assert!(
        text::parse_module(&INPLACE.replace("slice_sizes={{1,2}}", "slice_sizes={{}}")).is_err()
    );
    assert!(text::parse_module(&INPLACE.replace("p,q,offsets,offsets", "p,q,offsets")).is_err());
}

#[test]
fn tuple_and_multi_slice_offset_layouts_recover_slice_size_rows() {
    let tuple = "HloModule m ENTRY main { a = f32[2] parameter(0) b = s32[2,3] parameter(1) p = (f32[2],s32[2,3]) tuple(a,b) z = s32[] constant(0) x = (s32[]) tuple(z) y = (s32[],s32[]) tuple(z,z) offsets = ((s32[]),(s32[],s32[])) tuple(x,y) ROOT c = (f32[2],s32[2,3]) collective-permute(p,p,offsets,offsets), source_target_pairs={{0,1}}, slice_sizes={{1},{1,2}} }";
    round_trip(tuple);
    let multi = "HloModule m ENTRY main { p = f32[4] parameter(0) z = s32[] constant(0) x = (s32[]) tuple(z) offsets = ((s32[]),(s32[])) tuple(x,x) ROOT c = f32[4] collective-permute(p,p,offsets,offsets), source_target_pairs={{0,1},{0,1}}, slice_sizes={{2},{2}} }";
    round_trip(multi);
    let mut wire = text::parse_module(multi).unwrap().to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "collective-permute")
        .unwrap()
        .dynamic_slice_sizes
        .push(99);
    let raw = Module::from_proto(wire).unwrap();
    raw.verify(strict()).unwrap();
    assert_eq!(
        text::print_module(&raw).unwrap_err().category,
        "unsupported"
    );
}

#[test]
fn asynchronous_permute_keeps_arbitrary_context_shapes_and_traces_intermediaries() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) s = (f32[2],f32[2],u32[],token[]) collective-permute-start(p), source_target_pairs={{0,1}} t = ((f32[2],f32[2],u32[],token[])) tuple(s) g = (f32[2],f32[2],u32[],token[]) get-tuple-element(t), index=0 ROOT d = f32[2] collective-permute-done(g) }";
    round_trip(source);
    for source in [
        source.replace(
            "s = (f32[2],f32[2],u32[],token[])",
            "s = (f32[3],f32[2],u32[],token[])",
        ),
        source.replace("d = f32[2]", "d = f32[3]"),
        source.replace(
            "ROOT d =",
            "extra = f32[2] collective-permute-done(s) ROOT d =",
        ),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(strict())
                .is_err()
        );
    }
    let source = INPLACE
        .replace(
            "ROOT c = f32[4,5] collective-permute",
            "s = (f32[2,3],f32[4,5],u32[]) collective-permute-start",
        )
        .replace(
            "slice_sizes={{1,2}} }",
            "slice_sizes={{1,2}} ROOT d = f32[4,5] collective-permute-done(s) }",
        );
    round_trip(&source);
}

#[test]
fn variadic_start_inference_is_more_permissive_than_synchronous_permute() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) t = token[] after-all() s = ((f32[2],token[]),(f32[2],token[])) collective-permute-start(p,t), source_target_pairs={{0,1}} ROOT d = (f32[2],token[]) collective-permute-done(s) }";
    round_trip(source);
    let source =
        "HloModule m ENTRY main { ROOT c = () collective-permute(), source_target_pairs={} }";
    round_trip(source);
}
