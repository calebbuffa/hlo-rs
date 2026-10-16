use xlahlo::{PrimitiveType, text};

#[test]
fn dependent_inference_observes_canonical_constructor_shapes() {
    let module = text::parse_module(
        "HloModule m ENTRY main {
 p = f32[2] parameter(0)
 t = after-all()
 s = (f32[2],s32[],token[]) send(p,t), channel_id=1
 g = get-tuple-element(s), index=1
 ROOT c = copy(u32[] g)
}",
    )
    .unwrap();
    assert_eq!(
        module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap()
            .element_type(),
        PrimitiveType::U32
    );
    let module = text::parse_module(
        "HloModule m ENTRY main {
 p = f32[2] parameter(0)
 n = negate(p)
 t = after-all()
 ROOT o = token[] outfeed(n,t)
}",
    )
    .unwrap();
    let outfeed = module
        .instruction(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert_eq!(
        outfeed
            .outfeed_shape
            .as_ref()
            .unwrap()
            .element_type
            .enum_value()
            .unwrap(),
        PrimitiveType::F32
    );
    module.verify(Default::default()).unwrap();
}

#[test]
fn broadcast_sort_and_scan_omission_follow_native_constructor_shapes() {
    for (source, expected) in [
        (
            "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT r = broadcast(p), dimensions={1} }",
            "f32[1,2]{1,0}",
        ),
        (
            "HloModule m cmp { a = f32[] parameter(0) b = f32[] parameter(1) ROOT c = compare(a,b), direction=LT } ENTRY main { p = f32[3] parameter(0) ROOT r = sort(p), dimensions={0}, to_apply=cmp }",
            "f32[3]{0}",
        ),
        (
            "HloModule m body { p = f32[2,3]{0,1} parameter(0) ROOT n = copy(p) } ENTRY main { p = f32[2,4,3] parameter(0) ROOT r = scan(p), dimensions={1}, num_carries=0, to_apply=body }",
            "f32[2,4,3]{0,2,1}",
        ),
        (
            "HloModule m body { p = f32[<=3] parameter(0) ROOT n = copy(p) } ENTRY main { p = f32[<=4,3] parameter(0) ROOT r = scan(p), dimensions={0}, num_carries=0, to_apply=body }",
            "f32[4,3]{1,0}",
        ),
    ] {
        let module = text::parse_module(source).unwrap();
        let shape = module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        assert!(
            shape.equal_including_layout(&text::parse_shape(expected).unwrap()),
            "{source}: {shape:?}"
        );
        assert_eq!(
            shape.layout().unwrap().tail_padding_alignment_in_elements,
            1
        );
        module.verify(Default::default()).unwrap();
    }
}

#[test]
fn scaled_dot_omission_retains_native_parser_and_verifier_operand_pair_difference() {
    let source = "HloModule m ENTRY main { l = f32[8,8] parameter(0) r = f32[8,6] parameter(1) a = f32[8,4] parameter(2) b = f32[8,6] parameter(3) ROOT d = scaled-dot(l,r,a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0} }";
    let module = text::parse_module(source).unwrap();
    assert_eq!(
        module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap()
            .dimensions(),
        [8, 4]
    );
    assert!(module.verify(Default::default()).is_err());
    text::parse_module(&source.replace("d = scaled-dot", "d = f32[8,6] scaled-dot"))
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn inferred_callee_results_use_current_instruction_signatures() {
    for (source, expected) in [
        (
            "HloModule m neg { p = f32[] parameter(0) ROOT n = negate(p) } ENTRY main { p = f32[] parameter(0) ROOT c = call(p), to_apply=neg }",
            "f32[]",
        ),
        (
            "HloModule m neg { p = f32[] parameter(0) ROOT n = negate(p) } ENTRY main { p = f32[2] parameter(0) ROOT c = map(p), dimensions={0}, to_apply=neg }",
            "f32[2]",
        ),
        (
            "HloModule m sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT c = add(a,b) } ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0) ROOT c = reduce(p,z), dimensions={0}, to_apply=sum }",
            "f32[]",
        ),
    ] {
        let module = text::parse_module(source).unwrap();
        module.verify(Default::default()).unwrap();
        let shape = module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        assert!(shape.equal_including_layout(&text::parse_shape(expected).unwrap()));
        text::parse_module(&text::print_module(&module).unwrap())
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}

#[test]
fn inferred_elementwise_results_reuse_checked_shape_inference() {
    for (expression, expected) in [
        ("abs(p)", "f32[2,3]{1,0}"),
        ("add(p,p)", "f32[2,3]{1,0}"),
        ("is-finite(p)", "pred[2,3]{1,0}"),
        ("compare(p,p), direction=LT", "pred[2,3]{1,0}"),
        ("complex(p,p)", "c64[2,3]{1,0}"),
        ("clamp(p,p,p)", "f32[2,3]{1,0}"),
    ] {
        let source =
            format!("HloModule m ENTRY main {{ p = f32[2,3] parameter(0) ROOT r = {expression} }}");
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let shape = module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        assert!(
            shape.equal_including_layout(&text::parse_shape(expected).unwrap()),
            "{expression}: {shape:?}"
        );
        let printed = text::print_module(&module).unwrap();
        assert!(printed.contains("ROOT %r = "), "{printed}");
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
}

#[test]
fn inferred_tuple_geometry_and_typed_operands_propagate_in_source_order() {
    let module = text::parse_module(
        "HloModule m ENTRY main {
 p = f32[2,3]{0,1} parameter(0)
 n = negate(f32[2,3] p)
 t = tuple(n,p)
 g = get-tuple-element(t), index=0
 ROOT r = transpose(g), dimensions={1,0}
}",
    )
    .unwrap();
    module.verify(Default::default()).unwrap();
    let shape = module
        .shape(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert!(shape.equal_including_layout(&text::parse_shape("f32[3,2]{1,0}").unwrap()));
    let token = text::parse_module(
        "HloModule m ENTRY main { a = after-all() ROOT b = add-dependency(a,a) }",
    )
    .unwrap();
    token.verify(Default::default()).unwrap();
    assert_eq!(
        token
            .shape(token.root(token.entry().unwrap()).unwrap())
            .unwrap()
            .element_type(),
        PrimitiveType::TOKEN
    );
    let module = text::parse_module(
        "HloModule m ENTRY main { p = b(f32[2]{0}) custom-call(), custom_call_target=\"CreateBuffer\" ROOT n = copy(b(f32[2]) p) }",
    ).unwrap();
    assert_eq!(
        module
            .shape(module.root(module.entry().unwrap()).unwrap())
            .unwrap()
            .element_type(),
        PrimitiveType::BUFFER
    );
}

#[test]
fn inference_preserves_dynamic_flags_and_validates_omission_policy() {
    let module = text::parse_module(
        "HloModule m ENTRY main { p = f32[<=2,3] parameter(0) ROOT n = negate(p) }",
    )
    .unwrap();
    let shape = module
        .shape(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert!(shape.is_dynamic());
    for body in [
        "ROOT p = parameter(0)",
        "ROOT c = constant(1)",
        "p = f32[2] parameter(0) ROOT n = negate()",
        "p = pred[2] parameter(0) ROOT n = negate(p)",
        "p = f32[2] parameter(0) ROOT n = transpose(p), dimensions={1}",
        "p = f32[2] parameter(0) ROOT n = reshape(p)",
        "p = f32[2] parameter(0) ROOT n = opt-barrier(p)",
        "p = f32[2] parameter(0) t = token[] after-all() ROOT n = outfeed(p,t)",
    ] {
        assert!(
            text::parse_module(&format!("HloModule m ENTRY main {{ {body} }}")).is_err(),
            "{body}"
        );
    }
}
