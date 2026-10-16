use xlahlo::{Literal, text};

#[test]
fn headerless_computations_infer_the_module_name_and_entry() {
    for (source, entry) in [
        ("ENTRY main { ROOT p = f32[] parameter(0) }", "main"),
        ("main { ROOT p = f32[] parameter(0) }", "main"),
        (
            "helper { ROOT p = f32[] parameter(0) } main { x = f32[] parameter(0) ROOT r = call(x), to_apply=helper }",
            "main",
        ),
        (
            "ENTRY first { ROOT x = f32[] constant(0) } last { ROOT x = f32[] constant(1) }",
            "first",
        ),
    ] {
        let module = text::parse_module(source).unwrap();
        assert_eq!(module.as_proto().name, format!("module_{entry}"));
        assert_eq!(module.as_proto().entry_computation_name, entry);
        assert_eq!(module.config(), Default::default());
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        assert!(printed.starts_with(&format!("HloModule module_{entry}")));
        text::parse_module(&printed)
            .unwrap()
            .verify(Default::default())
            .unwrap();
    }
    for source in [
        "main, replica_count=2 { ROOT p = f32[] parameter(0) }",
        "HloModule",
        "",
        "ENTRY main { ROOT x = f32[] constant(0) } ENTRY last { ROOT x = f32[] constant(1) }",
    ] {
        assert!(text::parse_module(source).is_err(), "{source}");
    }
}

#[test]
fn block_comments_work_between_and_attached_to_shape_and_literal_tokens() {
    let source = "/* module */HloModule m ENTRY main {
 ROOT c = (f32[2,0], /*index=1*/ s32[]) constant(
  ({/*i0=0*/{}, /*i0=1*/{}}, /*scalar*/ 7/*value*/))
}/*eof*/";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    assert!(text::parse_shape("f32/*axis*/[2/*size*/]").is_ok());
    for source in ["/*", "/*missing terminator", "f32[2] /* trailing"] {
        assert!(text::parse_shape(source).is_err(), "{source}");
    }
}

#[test]
fn comment_markers_inside_quoted_attributes_and_raw_dictionaries_are_preserved() {
    let source = "HloModule m ENTRY main {
 ROOT c = f32[] custom-call(), custom_call_target=\"/*literal*/\",
 backend_config={\"text\":\"/*not a comment*/\"},
 frontend_attributes={a=\"//also literal\"}
}";
    let module = text::parse_module(source).unwrap();
    let root = module
        .instruction(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert_eq!(root.custom_call_target, "/*literal*/");
    assert_eq!(root.backend_config, b"{\"text\":\"/*not a comment*/\"}");
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn custom_call_literal_attributes_have_their_own_shape_and_payload() {
    for literal in [
        "f32[] 0.1",
        "s32[2]{0} {1,2}",
        "(s32[] 7,f32[1]{0} {2})",
        "()",
        "((s32[] 1),f32[] 2)",
    ] {
        let source = format!(
            "HloModule m ENTRY main {{
 ROOT c = f32[3] custom-call(), custom_call_target=\"literal\", literal={literal}
}}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let root = module
            .instruction(module.root(module.entry().unwrap()).unwrap())
            .unwrap();
        let value = Literal::from_proto(root.literal.get_or_default().clone()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        let root = reparsed
            .instruction(reparsed.root(reparsed.entry().unwrap()).unwrap())
            .unwrap();
        assert_eq!(root.literal.as_ref(), Some(value.as_proto()));
    }
    assert!(
        text::parse_module(
            "HloModule m ENTRY main {
 ROOT p = f32[] parameter(0), literal=f32[] 1
}"
        )
        .is_err()
    );
}

#[test]
fn scalar_reduce_window_materializes_empty_window_defaults() {
    let source = "HloModule m
 sum { a = f32[] parameter(0) b = f32[] parameter(1) ROOT s = f32[] add(a,b) }
 ENTRY main { x = f32[] constant(3) z = f32[] constant(0)
 ROOT r = f32[] reduce-window(x,z), to_apply=sum }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let root = module
        .instruction(module.root(module.entry().unwrap()).unwrap())
        .unwrap();
    assert!(root.window.is_some());
    assert!(root.window.get_or_default().dimensions.is_empty());
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn debug_tables_preserve_defaults_lists_keys_and_last_write_semantics() {
    let source = "HloModule m, debug_attributes={
 {\"z\" {1}}:(),
 {\"c\"}:({log_mode=default,callback_id=3,partitioned=true},{op_id=7}),
 {\"a\"}:{callback_id=1,callback_id=2}
} ENTRY main { ROOT c = f32[] constant(1) }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let table = &module.as_proto().debug_attributes;
    assert_eq!(table.len(), 2);
    assert_eq!(
        table[0].original_array.get_or_default().instruction_name,
        "a"
    );
    assert_eq!(table[0].debug_attributes[0].callback_id, 2);
    assert_eq!(table[1].debug_attributes.len(), 2);
    let printed = text::print_module(&module).unwrap();
    assert_eq!(
        text::parse_module(&printed)
            .unwrap()
            .as_proto()
            .debug_attributes,
        *table
    );
    for attributes in [
        "{{\"c\"}:{}}",
        "{{\"c\"}:{log_mode=none}}",
        "{{\"c\"}:{unknown=1}}",
    ] {
        let source = format!(
            "HloModule m, debug_attributes={attributes} ENTRY main {{ ROOT c = f32[] constant(1) }}"
        );
        assert!(text::parse_module(&source).is_err(), "{source}");
    }
    let module = text::parse_module(
        "HloModule m, debug_attributes={
 {\"c\"}:{callback_id=1},{\"c\"}:()
} ENTRY main { ROOT c = f32[] constant(1) }",
    )
    .unwrap();
    assert!(module.as_proto().debug_attributes.is_empty());
}

#[test]
fn imported_debug_fields_and_noncanonical_order_are_not_discarded() {
    let module = text::parse_module(
        "HloModule m, debug_attributes={
 {\"a\"}:{callback_id=1},{\"b\"}:{op_id=2}
} ENTRY main { ROOT c = f32[] constant(1) }",
    )
    .unwrap();
    for change in 0..4 {
        let mut wire = module.to_proto_with_program_shapes().unwrap();
        match change {
            0 => wire.debug_attributes.reverse(),
            1 => wire.debug_attributes[0].debug_attributes[0].operands_sharding = "opaque".into(),
            2 => wire.debug_attributes[0].debug_attributes.clear(),
            _ => wire.debug_attributes[0]
                .special_fields
                .mut_unknown_fields()
                .add_varint(1234, 1),
        }
        let imported = xlahlo::Module::from_proto(wire).unwrap();
        assert!(text::print_module(&imported).is_err());
    }
}

#[test]
fn scalar_slice_keeps_its_required_empty_slice_attribute() {
    let module = text::parse_module(
        "HloModule m ENTRY main { c = s32[] constant(1) ROOT s = s32[] slice(c), slice={} }",
    )
    .unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("slice={}"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}

#[test]
fn computation_annotations_do_not_override_actual_program_shapes() {
    let module = text::parse_module(
        "HloModule m ENTRY main (ignored: s32[7]) -> f32[2,3]{1,0} {
 ROOT p = f32[2,3]{0,1} parameter(0)
}",
    )
    .unwrap();
    module.verify(Default::default()).unwrap();
    let wire = module.to_proto_with_program_shapes().unwrap();
    let program = wire.computations[0].program_shape.as_ref().unwrap();
    assert_eq!(program.parameter_names, ["p"]);
    assert_eq!(program.parameters[0], *program.result.get_or_default());
    assert_eq!(
        program
            .result
            .get_or_default()
            .layout
            .get_or_default()
            .minor_to_major,
        [0, 1]
    );
    let mut imported = wire;
    imported.computations[0]
        .program_shape
        .mut_or_insert_default()
        .parameters[0]
        .layout
        .mut_or_insert_default()
        .minor_to_major = vec![1, 0];
    let imported = xlahlo::Module::from_proto(imported).unwrap();
    assert!(text::print_module(&imported).is_err());
    assert!(
        text::parse_module("HloModule m ENTRY main () -> s32[2] { ROOT p = s32[3] parameter(0) }",)
            .is_err()
    );
}

#[test]
fn legacy_convolution_attributes_on_custom_calls_are_printed() {
    let source = "HloModule m ENTRY main {
 p = f32[1,2,1] parameter(0) k = f32[1,1,1] parameter(1)
 ROOT c = f32[1,2,1] custom-call(p,k), custom_call_target=\"DynamicConvolutionForward\",
 window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}
}";
    let module = text::parse_module(source).unwrap();
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    let root = |module: &xlahlo::Module| {
        module
            .instruction(module.root(module.entry().unwrap()).unwrap())
            .unwrap()
            .clone()
    };
    assert_eq!(root(&module).window, root(&reparsed).window);
    assert_eq!(
        root(&module).convolution_dimension_numbers,
        root(&reparsed).convolution_dimension_numbers
    );
}
