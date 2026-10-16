use xlahlo::{Dimension as D, Module, PrimitiveType as P, Shape, VerifyOptions, inference, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}

#[test]
fn layout_sensitive_preprocess_requires_full_minor_order_in_nondefault_memory() {
    let mut wire = text::parse_module("HloModule m ENTRY main { ROOT p = f32[2,3] parameter(0) }")
        .unwrap()
        .to_proto()
        .unwrap();
    let layout = wire.computations[0].instructions[0]
        .shape
        .as_mut()
        .unwrap()
        .layout
        .as_mut()
        .unwrap();
    layout.memory_space = 1;
    layout.minor_to_major.clear();
    let module = Module::from_proto(wire).unwrap();
    module.verify(Default::default()).unwrap();
    assert_eq!(module.verify(strict()).unwrap_err().category, "layout");
}

fn copy_with_layouts(input_memory: i64, output_memory: i64, transpose: bool) -> Module {
    let source = "HloModule m ENTRY main { p = f32[2,3] parameter(0) ROOT c = f32[2,3] copy(p) }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    let instructions = &mut wire.computations[0].instructions;
    for instruction in instructions {
        let output = instruction.opcode == "copy";
        let shape = instruction.shape.as_mut().unwrap();
        let layout = shape.layout.as_mut().unwrap();
        layout.memory_space = if output { output_memory } else { input_memory };
        if output && transpose {
            layout.minor_to_major.reverse();
        }
    }
    Module::from_proto(wire).unwrap()
}

#[test]
fn host_copy_layout_policy_matches_pinned_directional_rules() {
    copy_with_layouts(0, 0, true).verify(strict()).unwrap();
    copy_with_layouts(0, 5, true).verify(strict()).unwrap();
    copy_with_layouts(5, 0, false).verify(strict()).unwrap();
    assert!(copy_with_layouts(5, 0, true).verify(strict()).is_err());
    copy_with_layouts(1, 2, true).verify(strict()).unwrap();
    copy_with_layouts(5, 0, true)
        .verify(Default::default())
        .unwrap();
    let mut wire = copy_with_layouts(0, 5, true).to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "copy")
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .dimensions = vec![3, 2];
    // The pinned device-to-host branch checks the element type, not geometry.
    Module::from_proto(wire).unwrap().verify(strict()).unwrap();
}

#[test]
fn layout_sensitive_calls_check_callee_result_layouts_and_dynamic_flags() {
    let source = "HloModule m f { ROOT p = f32[2,3]{0,1} parameter(0) } ENTRY main { p = f32[2,3]{0,1} parameter(0) ROOT c = f32[2,3]{1,0} call(p), to_apply=f }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(module.verify(strict()).is_err());
    let source = source.replace("c = f32[2,3]{1,0}", "c = f32[2,3]{0,1}");
    text::parse_module(&source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let module = text::parse_module(&source.replace("c = f32[2,3]", "c = f32[<=2,3]")).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(module.verify(strict()).is_err());
}

#[test]
fn dynamic_reshape_distinguishes_upstream_inference_and_verifier_size_contracts() {
    let input = Shape::array(P::F32, vec![6]).unwrap();
    let bounds = vec![D::Bounded(2), D::Static(3)];
    let scalar = Shape::scalar(P::S32).unwrap();
    let expected = Shape::dynamic_array(P::F32, bounds.clone())
        .unwrap()
        .with_default_layout()
        .unwrap();
    assert_eq!(
        inference::infer_dynamic_reshape(&input, &[scalar.clone(), scalar], bounds.clone())
            .unwrap(),
        expected
    );
    let float_scalar = Shape::scalar(P::F64).unwrap();
    assert!(inference::infer_dynamic_reshape(&input, &[float_scalar], bounds.clone()).is_ok());
    let float_vector = Shape::array(P::F64, vec![1]).unwrap();
    assert!(inference::infer_dynamic_reshape(&input, &[float_vector], bounds).is_err());
    let source = "HloModule m ENTRY main { a = f32[6] parameter(0) n = s32[1] parameter(1) ROOT r = f32[<=2,3] dynamic-reshape(a,n,n) }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(strict())
        .unwrap();
    for source in [
        source.replace("dynamic-reshape(a,n,n)", "dynamic-reshape(a,n)"),
        source.replace("f32[<=2,3]", "f32[<=3,3]"),
    ] {
        assert!(text::parse_module(&source).is_err());
    }
    assert!(
        text::parse_module(&source.replace("n = s32[1]", "n = f64[]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn layout_insensitive_bitcast_does_not_invent_shape_size_constraints() {
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT b = u8[99] bitcast(p) }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    assert_eq!(module.verify(strict()).unwrap_err().category, "bitcast");
    text::parse_module(&text::print_module(&module).unwrap())
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn exact_logical_equality_normalizes_omitted_false_dynamic_flags() {
    let shape = Shape::array(P::F32, vec![2]).unwrap();
    let mut proto = shape.as_proto().clone();
    proto.is_dynamic_dimension.clear();
    proto.layout.clear();
    let absent = Shape::from_proto(proto).unwrap();
    assert!(shape.equal_ignoring_layout(&absent));
    assert!(
        !shape.equal_ignoring_layout(&Shape::dynamic_array(P::F32, vec![D::Bounded(2)]).unwrap())
    );
}

#[test]
fn shape_size_uses_storage_width_packing_and_explicit_pointer_sizes() {
    let shape = Shape::array(P::U4, vec![3]).unwrap();
    assert_eq!(shape.byte_size(-1).unwrap(), 3);
    let mut layout = shape
        .clone()
        .with_default_layout()
        .unwrap()
        .layout()
        .unwrap()
        .clone();
    layout.element_size_in_bits = 4;
    layout.tail_padding_alignment_in_elements = 64;
    assert_eq!(
        shape
            .clone()
            .with_layout(layout)
            .unwrap()
            .byte_size(-1)
            .unwrap(),
        2
    );
    let tuple = Shape::tuple(vec![shape, Shape::token()]).unwrap();
    assert_eq!(tuple.byte_size(8).unwrap(), 16);
    assert!(tuple.byte_size(-1).is_err());
    assert_eq!(Shape::opaque().byte_size(4).unwrap(), 4);
    assert_eq!(Shape::token().byte_size(-1).unwrap(), 0);
}

#[test]
fn bitcast_supports_explicit_target_sizing_and_constant_memory_space_exemption() {
    fn custom_size(_: &Shape) -> xlahlo::Result<i64> {
        Ok(64)
    }
    let source = "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT b = u8[99] bitcast(p) }";
    let module = text::parse_module(source).unwrap();
    module
        .verify(VerifyOptions {
            shape_size: Some(custom_size),
            ..strict()
        })
        .unwrap();
    let module = text::parse_module(&source.replace("u8[99]", "u8[8]")).unwrap();
    module.verify(strict()).unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "bitcast")
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 5;
    assert_eq!(
        Module::from_proto(wire)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "bitcast"
    );
    let source = "HloModule m ENTRY main { c = f32[2] constant({1,2}) p = f32[2] copy(c) ROOT b = u8[8] bitcast(p) }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "bitcast")
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 5;
    Module::from_proto(wire).unwrap().verify(strict()).unwrap();
}

#[test]
fn raw_hlo_does_not_inherit_builder_elementwise_or_broadcast_conveniences() {
    let source = "HloModule m ENTRY main { a = f32[] parameter(0) b = f32[2] parameter(1) ROOT c = f32[2] add(a,b) }";
    assert_eq!(
        text::parse_module(source)
            .unwrap()
            .verify(Default::default())
            .unwrap_err()
            .category,
        "broadcast"
    );
    let source = "HloModule m ENTRY main { a = f32[2,3] parameter(0) ROOT b = f32[3,2] broadcast(a), dimensions={1,0} }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    assert!(
        module
            .verify(VerifyOptions {
                verify_broadcast_dimensions_order: true,
                ..Default::default()
            })
            .is_err()
    );
    assert!(
        text::parse_module(&source.replace("f32[2,3]", "f32[1,3]"))
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    let source = "HloModule m ENTRY main { a = f32[2,2] parameter(0) ROOT b = f32[2,3] broadcast(a), dimensions={0,0} }";
    text::parse_module(source)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}
