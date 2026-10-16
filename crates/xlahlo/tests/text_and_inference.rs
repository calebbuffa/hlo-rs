use xlahlo::{Dimension, Opcode, PrimitiveType as P, Shape, inference, text};

#[test]
fn comparison_constructor_defaults_and_legacy_types_are_materialized() {
    for (ty, attribute, expected) in [
        ("f32", "", "PARTIAL"),
        ("c64", "", "PARTIAL"),
        ("s32", "", "TOTAL"),
        ("pred", "", "TOTAL"),
        ("f32", ", type=SIGNED", "TOTAL"),
        ("s32", ", type=FLOAT", "PARTIAL"),
        ("f32", ", order=WEAK", "WEAK"),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ p = {ty}[] parameter(0) ROOT c = pred[] compare(p,p), direction=EQ{attribute} }}"
        );
        let module = text::parse_module(&source).unwrap();
        let comparison = module.as_proto().computations[0]
            .instructions
            .iter()
            .find(|i| i.opcode == "compare")
            .unwrap();
        assert_eq!(comparison.comparison_order, expected);
        assert!(comparison.comparison_type.is_empty());
        assert_eq!(
            module.verify(Default::default()).is_ok(),
            ty != "s32" || expected == "TOTAL"
        );
        for attribute in [
            ", type=INVALID",
            ", order=INVALID",
            ", type=FLOAT, order=PARTIAL",
        ] {
            assert!(
                text::parse_module(
                    &source.replace(", direction=EQ", &format!(", direction=EQ{attribute}"))
                )
                .is_err()
            );
        }
    }
}

#[test]
fn map_constructor_uses_every_result_dimension_even_if_text_supplies_another_list() {
    for dimensions in ["", ", dimensions={}", ", dimensions={1}"] {
        let source = format!(
            "HloModule m body {{ ROOT p = f32[] parameter(0) }} ENTRY main {{ p = f32[2,3] parameter(0) ROOT r = f32[2,3] map(p), to_apply=body{dimensions} }}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let map = module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .find(|i| i.opcode == "map")
            .unwrap();
        assert_eq!(map.dimensions, [0, 1]);
        text::parse_module(&text::print_module(&module).unwrap()).unwrap();
    }
}

#[test]
fn instruction_names_follow_native_lexer_and_sanitizer_rules() {
    for name in ["pred", "s1", "token", "__reserved", "bad$name", "bad\\name"] {
        for prefix in ["", "%"] {
            let source =
                format!("HloModule m ENTRY main {{ ROOT {prefix}{name} = f32[] parameter(0) }}");
            assert!(text::parse_module(&source).is_err(), "{source}");
        }
    }
    for name in [
        "ROOT",
        "nan",
        "true",
        "replicated",
        "__xla_internal",
        "tuple",
        "buffer",
    ] {
        let source = format!("HloModule m ENTRY main {{ ROOT %{name} = f32[] parameter(0) }}");
        let module = text::parse_module(&source).unwrap();
        text::parse_module(&text::print_module(&module).unwrap()).unwrap();
    }
}

#[test]
fn shape_assembly_preserves_dynamics_tuples_and_default_layout_behavior() {
    for source in [
        "f32[]",
        "s4[2,0]",
        "f32[<=4,?]{1,0}",
        "(token[], (u32[2], f64[]))",
    ] {
        let shape = text::parse_shape(source).unwrap();
        let canonical = text::print_shape(&shape).unwrap();
        assert_eq!(text::parse_shape(&canonical).unwrap(), shape);
    }
    assert!(text::parse_shape("f32[] trailing").is_err());
    assert!(text::parse_shape("f32[1,]").is_err());
    assert!(text::parse_shape("f32[-1]").is_err());
    let absent = text::parse_shape_with_options(
        "f32[2]",
        text::ParseOptions {
            fill_missing_layouts: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(absent.layout().is_none());
    assert!(text::parse_shape("f32[2]").unwrap().layout().is_some());
    let nested = format!("{}f32[]{}", "(".repeat(101), ")".repeat(101));
    assert_eq!(
        text::parse_shape(&nested).unwrap_err().category,
        "resource-limit"
    );
}

#[test]
fn bounded_dynamic_constant_text_uses_native_static_literal_shape() {
    for (source, expected_shape) in [
        (
            "HloModule m ENTRY main { ROOT c = s32[<=3] constant({1,2,3}) }",
            "s32[3]",
        ),
        (
            "HloModule m ENTRY main { ROOT c = (s32[<=2],f32[<=1,3]) constant(({1,2},{{3.0,4.0,5.0}})) }",
            "(s32[2], f32[1,3])",
        ),
    ] {
        let module = text::parse_module(source).unwrap();
        let constant = &module.as_proto().computations[0].instructions[0];
        let shape = Shape::from_proto(constant.shape.as_ref().unwrap().clone()).unwrap();
        assert_eq!(shape, text::parse_shape(expected_shape).unwrap());
        let literal = constant.literal.as_ref().unwrap();
        assert_eq!(literal.shape.as_ref(), Some(shape.as_proto()));
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        assert!(!printed.contains("<="));
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    let error =
        text::parse_module("HloModule m ENTRY main { ROOT c = s32[?] constant({1}) }").unwrap_err();
    assert_eq!(error.category, "unsupported");
    assert_eq!(error.message, "unbounded dimensions in text constants");
}

#[test]
fn scalar_layout_defaults_and_empty_layout_syntax_follow_native_shape_parser() {
    let shape = text::parse_shape_with_options(
        "f32[]",
        text::ParseOptions {
            fill_missing_layouts: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(shape.layout().is_some());
    assert_eq!(text::print_shape(&shape).unwrap(), "f32[]");
    assert!(text::parse_shape("f32[]{}").is_err());
    let shape = text::parse_shape("f32[]{:S(5)}").unwrap();
    assert_eq!(text::print_shape(&shape).unwrap(), "f32[]{:S(5)}");
    assert_eq!(
        text::parse_shape(&text::print_shape(&shape).unwrap()).unwrap(),
        shape
    );
}

#[test]
fn axpy_from_upstream_parser_fixture_round_trips_and_verifies() {
    // Adapted from hlo_parser_test.cc::CreateTestCases, AxpyParam.
    let source = r#"
HloModule axpy_module, entry_computation_layout={(f32[], f32[2,4]{1,0}, f32[2,4]{1,0})->f32[2,4]{1,0}}
ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}
"#;
    let module = text::parse_module(source).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert_eq!(
        text::print_module(&text::parse_module(&printed).unwrap()).unwrap(),
        printed
    );
    module.verify(Default::default()).unwrap();
}

#[test]
fn constant_text_respects_logical_and_physical_layout_order() {
    let source =
        "HloModule literal\nENTRY main {\n ROOT c = s32[2,3]{0,1} constant({{1,2,3},{4,5,6}})\n}";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let computation = module.entry().unwrap();
    let root = module
        .instruction(module.root(computation).unwrap())
        .unwrap();
    assert_eq!(root.literal.as_ref().unwrap().s32s, [1, 4, 2, 5, 3, 6]);
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("{{1, 2, 3}, {4, 5, 6}}"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn parser_errors_include_offsets_and_never_invent_constant_values() {
    for source in [
        "HloModule m\nENTRY main { ROOT c = s32[] constant(2147483648) }",
        "HloModule m\nENTRY main { ROOT c = pred[] constant(2.0) }",
        "HloModule m\nENTRY main { ROOT c = s32[2] constant({1}) }",
        "HloModule m\nENTRY main { ROOT c = s32[] constant(...) }",
        "HloModule m\nENTRY main { ROOT a = f32[] negate(missing) }",
    ] {
        assert!(text::parse_module(source).is_err(), "{source}");
    }
    assert!(
        text::parse_module("HloModule m\nENTRY main { ROOT a = ")
            .unwrap_err()
            .offset
            .is_some()
    );
}

#[test]
fn printing_preserves_backend_payloads_instead_of_discarding_them() {
    let mut wire = text::parse_module("HloModule m\nENTRY main { ROOT p = f32[] parameter(0) }")
        .unwrap()
        .to_proto()
        .unwrap();
    wire.computations[0].instructions[0].backend_config = vec![0, 255];
    let module = xlahlo::Module::from_proto(wire).unwrap();
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    assert_eq!(
        reparsed.to_proto().unwrap().computations[0].instructions[0].backend_config,
        vec![0, 255]
    );
}

#[test]
fn unary_type_domains_cover_every_primitive_without_panics() {
    for &ty in <P as protobuf::Enum>::VALUES {
        if !ty.is_array_element() {
            continue;
        }
        let shape = Shape::array(ty, vec![2]).unwrap();
        for &opcode in Opcode::ALL {
            let outcome = std::panic::catch_unwind(|| inference::infer_unary(opcode, &shape));
            assert!(outcome.is_ok(), "{opcode:?}, {ty:?}");
        }
        assert_eq!(
            inference::infer_unary(Opcode::Not, &shape).is_ok(),
            ty == P::PRED || ty.is_integral()
        );
        assert_eq!(
            inference::infer_unary(Opcode::Abs, &shape).is_ok(),
            ty.is_signed_integral() || ty.is_floating() || ty.is_complex()
        );
    }
    let complex = Shape::scalar(P::C128).unwrap();
    assert_eq!(
        inference::infer_unary(Opcode::Abs, &complex)
            .unwrap()
            .element_type(),
        P::F64
    );
}

#[test]
fn binary_inference_preserves_dynamic_flags_and_element_changes() {
    let a = Shape::dynamic_array(P::F32, vec![Dimension::Bounded(4)]).unwrap();
    let b = Shape::array(P::F32, vec![4]).unwrap();
    assert_eq!(
        inference::infer_binary(Opcode::Add, &a, &b)
            .unwrap()
            .dimension(0)
            .unwrap(),
        Dimension::Bounded(4)
    );
    assert_eq!(
        inference::infer_binary(Opcode::Complex, &a, &b)
            .unwrap()
            .element_type(),
        P::C64
    );
    assert_eq!(
        inference::infer_binary(Opcode::Compare, &a, &b)
            .unwrap()
            .element_type(),
        P::PRED
    );
}

#[test]
fn exhaustive_small_slice_geometry_matches_independent_integer_enumeration() {
    for size in 0..8 {
        let shape = Shape::array(P::S32, vec![size]).unwrap();
        for start in 0..=size {
            for limit in start..=size {
                for stride in 1..5 {
                    let inferred =
                        inference::infer_slice(&shape, &[start], &[limit], &[stride]).unwrap();
                    let expected = (start..limit).step_by(stride as usize).count() as i64;
                    assert_eq!(inferred.dimensions(), [expected]);
                }
            }
        }
    }
    assert!(
        inference::infer_slice(&Shape::array(P::S32, vec![4]).unwrap(), &[0], &[4], &[0]).is_err()
    );
}

#[test]
fn transpose_and_reshape_laws_use_checked_geometry() {
    let shape = Shape::array(P::F32, vec![2, 3, 4])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let transposed = inference::infer_transpose(&shape, &[2, 0, 1]).unwrap();
    assert_eq!(transposed.dimensions(), [4, 2, 3]);
    assert_eq!(
        inference::infer_transpose(&transposed, &[1, 2, 0]).unwrap(),
        shape
    );
    assert!(inference::infer_transpose(&shape, &[0, 0, 1]).is_err());
    assert_eq!(
        inference::infer_static_reshape(&shape, &[6, 4])
            .unwrap()
            .element_count()
            .unwrap(),
        24
    );
    assert!(inference::infer_static_reshape(&shape, &[25]).is_err());
}
