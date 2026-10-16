use xlahlo::{
    Module, OriginalValue,
    proto::xla_data::{OriginalArrayProto, OriginalValueElementProto, OriginalValueProto},
    text,
};

fn element(index: &[i64], name: Option<&str>) -> OriginalValueElementProto {
    OriginalValueElementProto {
        shape_index: index.to_vec(),
        original_array: name
            .map(|name| OriginalArrayProto {
                instruction_name: name.into(),
                ..Default::default()
            })
            .into(),
        ..Default::default()
    }
}

fn value(elements: Vec<OriginalValueElementProto>) -> OriginalValue {
    OriginalValue::from_proto(OriginalValueProto {
        elements,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn original_tuple_trees_and_call_hierarchies_round_trip() {
    for (shape, original) in [
        ("f32[]", r#"{{"unoptimized" {2,1}}}"#),
        ("f32[]", r#"{{}}"#),
        ("f32[]", r#"{{"escaped\"name"},["call#$"]}"#),
        (
            "(f32[],(s32[],f64[]))",
            r#"{({"a"},({"b" {1}},{})),["outer.inner#$"]}"#,
        ),
        ("((),f32[])", r#"{((),{"a"})}"#),
        ("()", "{()}"),
        ("f32[]", "{[synthetic_call]}"),
        ("(f32[],s32[])", "{[synthetic_call]}"),
        ("f32[]", r#"{{"discarded"},[""]}"#),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = {shape} parameter(0), origin={original} }}"
        );
        let module = text::parse_module(&source).unwrap();
        module.verify(Default::default()).unwrap();
        let printed = text::print_module(&module).unwrap();
        let reparsed = text::parse_module(&printed).unwrap();
        assert_eq!(
            module.as_proto().computations[0].instructions[0].original_value,
            reparsed.as_proto().computations[0].instructions[0].original_value,
            "{source} -> {printed}"
        );
    }
}

#[test]
fn verifier_checks_leaf_indices_not_reference_names_or_missing_values() {
    for (shape, original, valid) in [
        ("f32[]", r#"{{"not.a.live.instruction" {500}}}"#, true),
        ("f32[]", "{{}}", true),
        ("f32[]", "{()}", false),
        ("f32[]", r#"{({"a"})}"#, false),
        ("(f32[],f32[])", r#"{{"a"}}"#, false),
        ("(f32[],f32[])", r#"{({"a"},{})}"#, true),
        ("()", "{()}", true),
        ("()", "{{}}", false),
        ("((),f32[])", r#"{((),{"a"})}"#, true),
        ("(f32[],())", r#"{({"a"},())}"#, true),
    ] {
        let source = format!(
            "HloModule m ENTRY main {{ ROOT p = {shape} parameter(0), origin={original} }}"
        );
        let module = text::parse_module(&source).unwrap();
        assert_eq!(module.verify(Default::default()).is_ok(), valid, "{source}");
    }
    for shape in ["token[]", "opaque[]", "b(f32[2])"] {
        value(vec![element(&[], None)])
            .verify_shape(&text::parse_shape(shape).unwrap())
            .unwrap();
    }
}

#[test]
fn tuple_tree_import_is_last_write_wins_in_both_ancestor_directions() {
    let original = value(vec![
        element(&[0], Some("parent")),
        element(&[0, 1], Some("child")),
    ]);
    assert_eq!(
        original
            .leaves()
            .map(|(index, _)| index.to_vec())
            .collect::<Vec<_>>(),
        [vec![0, 1]]
    );
    original
        .verify_shape(&text::parse_shape("(((),f32[]))").unwrap())
        .unwrap();
    let original = value(vec![
        element(&[0, 1], Some("child")),
        element(&[0], Some("parent")),
    ]);
    assert_eq!(
        original
            .leaves()
            .next()
            .unwrap()
            .1
            .unwrap()
            .instruction_name,
        "parent"
    );
    assert_eq!(original.leaves().count(), 1);
    original
        .verify_shape(&text::parse_shape("(f32[])").unwrap())
        .unwrap();
    let original = value(vec![element(&[], Some("old")), element(&[], Some("new"))]);
    assert_eq!(
        original
            .leaves()
            .next()
            .unwrap()
            .1
            .unwrap()
            .instruction_name,
        "new"
    );
}

#[test]
fn synthetic_presence_and_legacy_import_follow_the_pinned_constructor() {
    let legacy = OriginalValueProto {
        elements: vec![element(&[-1], Some("ignored"))],
        is_synthetic_call: true,
        ..Default::default()
    };
    let original = OriginalValue::from_proto(legacy.clone()).unwrap();
    assert!(!original.is_synthetic_call());
    assert!(
        original
            .verify_shape(&text::parse_shape("f32[]").unwrap())
            .is_err()
    );
    original
        .verify_shape(&text::parse_shape("()").unwrap())
        .unwrap();
    assert!(!original.normalized_proto().is_synthetic_call);
    assert_eq!(original.as_proto(), &legacy);

    let original = OriginalValue::from_proto(OriginalValueProto {
        call_hierarchy: Some(String::new()),
        ..legacy
    })
    .unwrap();
    assert!(original.is_synthetic_call());
    original
        .verify_shape(&text::parse_shape("(f32[],s32[])").unwrap())
        .unwrap();
    assert!(original.normalized_proto().elements.is_empty());

    let original = value(vec![element(&[], Some("x"))]);
    let mut present = original.as_proto().clone();
    present.call_hierarchy = Some(String::new());
    assert!(!original.equal(&OriginalValue::from_proto(present).unwrap()));
}

#[test]
fn negative_metadata_indices_are_checked_instead_of_dropping_the_annotation() {
    assert!(
        OriginalValue::from_proto(OriginalValueProto {
            elements: vec![element(&[-1], Some("x"))],
            ..Default::default()
        })
        .is_err()
    );
    let mut array = element(&[], Some("x"));
    array.original_array.as_mut().unwrap().shape_index = vec![-1];
    assert!(
        OriginalValue::from_proto(OriginalValueProto {
            elements: vec![array],
            ..Default::default()
        })
        .is_err()
    );
}

#[test]
fn text_refuses_noncanonical_unknown_and_unbounded_width_metadata() {
    let module =
        text::parse_module("HloModule m ENTRY main { ROOT p = f32[] parameter(0) }").unwrap();
    let mut unknown = value(vec![element(&[], Some("x"))]).into_proto();
    unknown.elements[0]
        .original_array
        .as_mut()
        .unwrap()
        .special_fields
        .mut_unknown_fields()
        .add_varint(199, 7);
    for original in [
        value(vec![element(&[i64::MAX], Some("x"))]).into_proto(),
        value(vec![element(&[], Some("old")), element(&[], Some("new"))]).into_proto(),
        value(vec![element(&[], Some(""))]).into_proto(),
        unknown,
    ] {
        let mut wire = module.to_proto().unwrap();
        wire.computations[0].instructions[0].original_value = Some(original).into();
        assert!(text::print_module(&Module::from_proto(wire).unwrap()).is_err());
    }
}

#[test]
fn provenance_assembly_uses_the_native_origin_attribute_name() {
    let source = r#"HloModule m ENTRY main { ROOT p = f32[] parameter(0), origin={{"x"}} }"#;
    let module = text::parse_module(source).unwrap();
    assert!(text::print_module(&module).unwrap().contains(", origin="));
    assert!(text::parse_module(&source.replace("origin=", "original_value=")).is_err());
}

#[test]
fn malformed_original_value_text_fails_without_panicking() {
    for original in [
        r#"{[unknown]}"#,
        r#"{{"x"},["a"],["b"]}"#,
        r#"{)}"#,
        r#"{{"x" {-1}}}"#,
        r#"{{"x"}"#,
    ] {
        let source =
            format!("HloModule m ENTRY main {{ ROOT p = f32[] parameter(0), origin={original} }}");
        assert!(text::parse_module(&source).is_err(), "{source}");
    }
}

#[test]
fn verified_edits_roll_back_incompatible_original_value_payloads() {
    let mut module = text::parse_module(
        r#"HloModule m ENTRY main { ROOT p = f32[] parameter(0), origin={{"x"}} }"#,
    )
    .unwrap();
    let before = module.to_proto().unwrap();
    let root = module.root(module.entry().unwrap()).unwrap();
    let invalid = value(vec![element(&[0], Some("x"))]).normalized_proto();
    assert!(
        module
            .edit_verified(Default::default(), |staged| {
                staged.edit_instruction_fields(root, |instruction| {
                    instruction.original_value = Some(invalid).into();
                    Ok(())
                })
            })
            .is_err()
    );
    assert_eq!(module.to_proto().unwrap(), before);
}
