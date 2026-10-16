use xlahlo::{VerifyOptions, text};

fn mixed(layout_sensitive: bool) -> VerifyOptions {
    VerifyOptions {
        allow_mixed_precision: true,
        layout_sensitive,
        ..Default::default()
    }
}

#[test]
fn defining_operations_can_change_float_precision_without_crossing_type_domains() {
    for source in [
        "HloModule m ENTRY main { a = f64[2] parameter(0) b = f32[2] parameter(1) ROOT r = bf16[2] add(a,b) }",
        "HloModule m ENTRY main { a = f64[2] parameter(0) ROOT r = f16[2] negate(a) }",
        "HloModule m ENTRY main { a = f64[2] parameter(0) ROOT r = f16[2] copy(a) }",
        "HloModule m ENTRY main { a = f64[2] parameter(0) ROOT r = bf16[2] transpose(a), dimensions={0} }",
    ] {
        let module = text::parse_module(source).unwrap();
        assert!(module.verify(Default::default()).is_err(), "{source}");
        module.verify(mixed(false)).unwrap();
        module.verify(mixed(true)).unwrap();
    }
    for source in [
        "HloModule m ENTRY main { a = s32[2] parameter(0) ROOT r = u32[2] negate(a) }",
        "HloModule m ENTRY main { a = c64[2] parameter(0) ROOT r = c128[2] negate(a) }",
        "HloModule m ENTRY main { a = f32[2] parameter(0) ROOT r = f64[2] is-finite(a) }",
    ] {
        assert!(
            text::parse_module(source)
                .unwrap()
                .verify(mixed(false))
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn pass_throughs_reshapes_broadcasts_and_host_copies_keep_exact_element_types() {
    for source in [
        "HloModule m ENTRY main { a = f32[2] parameter(0) ROOT r = (bf16[2]) tuple(a) }",
        "HloModule m ENTRY main { a = (f32[2]) parameter(0) ROOT r = bf16[2] get-tuple-element(a), index=0 }",
        "HloModule m f { ROOT a = f32[2] parameter(0) } ENTRY main { a = f32[2] parameter(0) ROOT r = bf16[2] call(a), to_apply=f }",
        "HloModule m ENTRY main { a = f32[2] parameter(0) ROOT r = bf16[1,2] reshape(a) }",
        "HloModule m ENTRY main { a = f32[2] parameter(0) size = s32[] constant(2) ROOT r = bf16[2] dynamic-reshape(a,size) }",
        "HloModule m ENTRY main { a = f32[2] parameter(0) ROOT r = bf16[2,3] broadcast(a), dimensions={0} }",
    ] {
        let module = text::parse_module(source).unwrap();
        for strict in [false, true] {
            assert!(module.verify(mixed(strict)).is_err(), "{source}");
        }
    }
    for host_opcode in ["parameter", "copy"] {
        let mut wire = text::parse_module(
            "HloModule m ENTRY main { a = f32[2] parameter(0) ROOT r = bf16[2] copy(a) }",
        )
        .unwrap()
        .to_proto()
        .unwrap();
        for instruction in &mut wire.computations[0].instructions {
            if instruction.opcode == host_opcode {
                instruction
                    .shape
                    .as_mut()
                    .unwrap()
                    .layout
                    .as_mut()
                    .unwrap()
                    .memory_space = 5;
            }
        }
        let module = xlahlo::Module::from_proto(wire).unwrap();
        module.verify(mixed(false)).unwrap();
        assert!(module.verify(mixed(true)).is_err(), "{host_opcode}");
    }
}

#[test]
fn rng_requires_identical_operand_types_even_when_result_precision_is_relaxed() {
    let source = "HloModule m ENTRY main { a = f32[] parameter(0) b = f32[] parameter(1) ROOT r = bf16[2] rng(a,b), distribution=rng_uniform }";
    let module = text::parse_module(source).unwrap();
    assert!(module.verify(Default::default()).is_err());
    module.verify(mixed(false)).unwrap();
    for source in [
        source.replace("b = f32[]", "b = f64[]"),
        source.replace("bf16[2]", "s32[2]"),
        source.replace("f32[]", "s32[]"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(mixed(false))
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn mixed_policy_relaxes_map_reduce_and_window_parameter_precision() {
    for source in [
        r#"HloModule m
f { ROOT p = f64[] parameter(0) }
ENTRY main { p = f32[2] parameter(0) ROOT m = bf16[2] map(p), to_apply=f }"#,
        r#"HloModule m
f { p = f64[] parameter(0) q = f32[] parameter(1) ROOT r = f64[] copy(p) }
ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0)
ROOT r = bf16[] reduce(p,z), dimensions={0}, to_apply=f }"#,
        r#"HloModule m
f { p = f64[] parameter(0) q = f32[] parameter(1) ROOT r = f64[] copy(p) }
ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0)
ROOT r = bf16[1] reduce-window(p,z), window={size=2}, to_apply=f }"#,
    ] {
        let module = text::parse_module(source).unwrap();
        assert!(module.verify(Default::default()).is_err());
        module.verify(mixed(false)).unwrap();
        module.verify(mixed(true)).unwrap();
    }
}

#[test]
fn associative_scan_relaxes_body_and_carry_precision_only_under_layout_sensitive_policy() {
    let source = r#"HloModule m
body {
  x = f32[3] parameter(0)
  c = f64[3] parameter(1)
  y = bf16[3] convert(x)
  ROOT r = (bf16[3],f32[3]) tuple(y,x)
}
ENTRY main {
  x = f16[4,3] parameter(0)
  c = bf16[3] parameter(1)
  ROOT r = (f16[4,3],bf16[3]) scan(x,c), dimensions={0}, num_carries=1, to_apply=body, is_associative=true
}"#;
    let module = text::parse_module(source).unwrap();
    assert!(module.verify(mixed(false)).is_err());
    assert!(
        module
            .verify(VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            })
            .is_err()
    );
    module.verify(mixed(true)).unwrap();
    for source in [
        source.replace("x = f32[3] parameter(0)", "x = f32[<=3] parameter(0)"),
        source.replace("x = f16[4,3] parameter(0)", "x = c64[4,3] parameter(0)"),
        source.replace("is_associative=true", "is_associative=false"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(mixed(true))
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn dynamic_update_slice_accepts_floating_update_precision_but_retains_operand_precision() {
    use xlahlo::{PrimitiveType as P, Shape, inference};
    let data = Shape::array(P::F16, vec![3]).unwrap();
    let update = Shape::array(P::BF16, vec![2]).unwrap();
    assert_eq!(
        inference::infer_dynamic_update_slice(&data, &update, &[Shape::scalar(P::S32).unwrap()])
            .unwrap(),
        data
    );
    for source in [
        "HloModule m ENTRY main { p = f16[3] parameter(0) u = bf16[2] parameter(1) i = s32[] constant(0) ROOT r = f16[3] dynamic-update-slice(p,u,i) }",
        "HloModule m ENTRY main { p = b(f16[3]) custom-call(), custom_call_target=\"CreateBuffer\" u = bf16[2] parameter(0) i = s32[] constant(0) d = b(f16[3]) dynamic-update-slice(p,u,i) ROOT r = f16[3] custom-call(d), custom_call_target=\"Unpin\", output_to_operand_aliasing={{}: (0,{})} }",
    ] {
        let module = text::parse_module(source).unwrap();
        assert!(module.verify(Default::default()).is_err());
        module.verify(mixed(false)).unwrap();
        module.verify(mixed(true)).unwrap();
    }
    let source = "HloModule m ENTRY main { p = f16[3] parameter(0) u = bf16[2] parameter(1) i = s32[] constant(0) ROOT r = bf16[3] dynamic-update-slice(p,u,i) }";
    assert!(
        text::parse_module(source)
            .unwrap()
            .verify(mixed(false))
            .is_err()
    );
}

#[test]
fn handlers_without_check_shape_do_not_gain_its_global_precision_restriction() {
    let scan = "HloModule m body { p = f32[] parameter(0) c = bf16[] parameter(1) ROOT r = (f32[],bf16[]) tuple(p,c) } ENTRY main { p = f32[3] parameter(0) c = bf16[] parameter(1) ROOT r = (f32[3],bf16[]) scan(p,c), dimensions={0}, num_carries=1, to_apply=body, is_associative=true }";
    let module = text::parse_module(scan).unwrap();
    module.verify(Default::default()).unwrap();
    module
        .verify(VerifyOptions {
            layout_sensitive: true,
            ..Default::default()
        })
        .unwrap();
    for output in ["u32[2]", "((f32[],bf16[]),u32[2])"] {
        let source = format!(
            "HloModule m ENTRY main {{ p = (f32[],bf16[]) parameter(0) ROOT r = {output} rng-bit-generator(p), algorithm=rng_philox }}"
        );
        text::parse_module(&source)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    let source = "HloModule m ENTRY main { p = (f32[],bf16[]) parameter(0) ROOT r = (f32[],bf16[]) bitcast(p) }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    module
        .verify(VerifyOptions {
            layout_sensitive: true,
            shape_size: Some(|shape| shape.byte_size(8)),
            ..Default::default()
        })
        .unwrap();
}
