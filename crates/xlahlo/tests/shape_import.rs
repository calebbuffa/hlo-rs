use xlahlo::{
    Literal, Module, PrimitiveType as P, Shape, VerifyOptions, inference, interchange,
    proto::xla_data::ShapeProto, text,
};

fn array() -> Shape {
    Shape::array(P::F32, vec![2])
        .unwrap()
        .with_default_layout()
        .unwrap()
}

#[test]
fn array_tuple_payload_is_retained_but_not_a_logical_child() {
    let expected = array();
    let mut wire = expected.as_proto().clone();
    let mut ignored = ShapeProto::new();
    ignored.is_dynamic_dimension = vec![true];
    ignored.dimensions = vec![i64::MIN];
    ignored.layout = Some(xlahlo::proto::xla_data::LayoutProto {
        memory_space: 5,
        ..Default::default()
    })
    .into();
    wire.tuple_shapes.push(ignored);
    let encoded = interchange::encode(&wire).unwrap();
    let shape = Shape::from_proto(wire).unwrap();
    assert_eq!(shape.dimensions(), [2]);
    assert_eq!(shape.leaf_count(), 1);
    assert_eq!(shape.element_count().unwrap(), 2);
    assert!(!shape.is_dynamic());
    assert!(shape.compatible(&expected));
    assert!(shape.equal_ignoring_layout(&expected));
    assert!(shape.equal_including_layout(&expected));
    assert_ne!(shape, expected);
    assert_eq!(interchange::encode(shape.as_proto()).unwrap(), encoded);
    assert_eq!(
        shape.as_proto().tuple_shapes[0]
            .layout
            .as_ref()
            .unwrap()
            .tail_padding_alignment_in_elements,
        0
    );
    assert_eq!(
        shape.without_layout().as_proto().tuple_shapes,
        shape.as_proto().tuple_shapes
    );
    assert!(text::print_shape(&shape).is_err());
}

#[test]
fn nonarray_dimensions_and_dynamic_flags_are_semantically_ignored() {
    for expected in [
        Shape::tuple(vec![array(), Shape::token()]).unwrap(),
        Shape::token(),
        Shape::opaque(),
        Shape::buffer(array()).unwrap(),
    ] {
        let mut wire = expected.as_proto().clone();
        wire.dimensions = vec![-7, i64::MIN];
        wire.is_dynamic_dimension = vec![true, false, true];
        if matches!(expected.element_type(), P::TOKEN | P::OPAQUE_TYPE) {
            wire.tuple_shapes = vec![ShapeProto::new()];
        }
        let encoded = interchange::encode(&wire).unwrap();
        let shape = Shape::from_proto(wire).unwrap();
        assert!(shape.dimensions().is_empty());
        assert!(!shape.is_dynamic());
        assert_eq!(shape.leaf_count(), expected.leaf_count());
        assert!(shape.compatible(&expected));
        assert!(shape.equal_ignoring_layout(&expected));
        assert!(shape.equal_including_layout(&expected));
        assert_eq!(interchange::encode(shape.as_proto()).unwrap(), encoded);
        assert!(text::print_shape(&shape).is_err());
    }
}

#[test]
fn applicable_layout_dimension_and_tuple_constraints_are_not_relaxed() {
    let mut wire = array().into_proto();
    wire.is_dynamic_dimension = vec![true, false];
    assert!(Shape::from_proto(wire).is_err());
    let mut wire = array().into_proto();
    wire.dimensions[0] = -1;
    assert!(Shape::from_proto(wire).is_err());
    for expected in [
        Shape::tuple(vec![]).unwrap(),
        Shape::token(),
        Shape::opaque(),
    ] {
        let mut wire = expected.into_proto();
        wire.layout = Some(Default::default()).into();
        assert!(Shape::from_proto(wire).is_err());
    }

    let mut wire = Shape::tuple(vec![]).unwrap().into_proto();
    wire.tuple_shapes.push(ShapeProto::new());
    assert!(Shape::from_proto(wire).is_err());
    let mut wire = Shape::buffer(array()).unwrap().into_proto();
    wire.tuple_shapes.push(array().into_proto());
    assert!(Shape::from_proto(wire).is_err());
}

#[test]
fn buffer_outer_layout_overrides_storage_without_losing_wire_payloads() {
    let inner = text::parse_shape("f32[2,3]{1,0:S(5)}").unwrap();
    let expected = text::parse_shape("b(f32[2,3]{0,1})").unwrap();
    let mut wire = Shape::buffer(inner).unwrap().into_proto();
    wire.layout = text::parse_shape("f32[2,3]{0,1}")
        .unwrap()
        .as_proto()
        .layout
        .clone();
    let bytes = interchange::encode(&wire).unwrap();
    let buffer = Shape::from_proto(wire).unwrap();
    assert_eq!(buffer.layout().unwrap().minor_to_major, [0, 1]);
    assert_eq!(buffer.layout().unwrap().memory_space, 0);
    assert_eq!(
        buffer.array_view().unwrap().layout(),
        expected.array_view().unwrap().layout()
    );
    assert!(buffer.equal_including_layout(&expected));
    assert_ne!(buffer, expected);
    assert_eq!(buffer.byte_size(8).unwrap(), 24);
    assert_eq!(buffer.array_data_size().unwrap(), 24);
    assert_eq!(interchange::encode(buffer.as_proto()).unwrap(), bytes);
    assert!(text::print_shape(&buffer).is_err());
    let tuple = Shape::tuple(vec![buffer.clone()]).unwrap();
    assert!(tuple.equal_including_layout(&Shape::tuple(vec![expected]).unwrap()));
    assert!(buffer.without_layout().equal_including_layout(
        &Shape::buffer(text::parse_shape("f32[2,3]").unwrap().without_layout()).unwrap()
    ));
}

#[test]
fn buffer_override_keeps_import_checks_but_replaces_layout_geometry() {
    let mut wire = text::parse_shape("b(f32[2,3]{1,0})").unwrap().into_proto();
    wire.layout = text::parse_shape("f32[2,3]{0,1}")
        .unwrap()
        .into_proto()
        .layout;
    wire.tuple_shapes[0]
        .layout
        .mut_or_insert_default()
        .minor_to_major = vec![0, 0];
    assert_eq!(
        Shape::from_proto(wire.clone())
            .unwrap()
            .layout()
            .unwrap()
            .minor_to_major,
        [0, 1]
    );
    wire.tuple_shapes[0]
        .layout
        .mut_or_insert_default()
        .tail_padding_alignment_in_elements = -1;
    assert!(Shape::from_proto(wire).is_err());
}

#[test]
fn zero_tail_alignment_imports_as_the_pinned_default() {
    let mut array_wire = array().into_proto();
    array_wire
        .layout
        .mut_or_insert_default()
        .tail_padding_alignment_in_elements = 0;
    assert_eq!(
        Shape::from_proto(array_wire)
            .unwrap()
            .layout()
            .unwrap()
            .tail_padding_alignment_in_elements,
        1
    );

    let mut tuple = Shape::tuple(vec![array()]).unwrap().into_proto();
    tuple.tuple_shapes[0]
        .layout
        .mut_or_insert_default()
        .tail_padding_alignment_in_elements = 0;
    assert_eq!(
        Shape::from_proto(tuple)
            .unwrap()
            .subshape(&[0])
            .unwrap()
            .layout()
            .unwrap()
            .tail_padding_alignment_in_elements,
        1
    );

    let mut buffer = Shape::buffer(array()).unwrap().into_proto();
    buffer.layout = array().as_proto().layout.clone();
    buffer
        .layout
        .mut_or_insert_default()
        .tail_padding_alignment_in_elements = 0;
    buffer.tuple_shapes[0]
        .layout
        .mut_or_insert_default()
        .tail_padding_alignment_in_elements = 0;
    let buffer = Shape::from_proto(buffer).unwrap();
    assert_eq!(
        buffer.layout().unwrap().tail_padding_alignment_in_elements,
        1
    );
    assert_eq!(
        buffer
            .array_view()
            .unwrap()
            .layout()
            .unwrap()
            .tail_padding_alignment_in_elements,
        1
    );
}

#[test]
fn cpp_ignored_sparse_dimension_fields_are_retained_as_wire_data() {
    let dense = array();
    let mut wire = dense.as_proto().clone();
    let layout = wire.layout.mut_or_insert_default();
    layout.dim_level_types = vec![xlahlo::proto::xla_data::DimLevelType::DIM_COMPRESSED.into()];
    layout.dim_unique = vec![false];
    layout.dim_ordered = vec![false];

    let shape = Shape::from_proto(wire.clone()).unwrap();
    assert_ne!(shape, dense);
    assert!(shape.equal_including_layout(&dense));
    assert_eq!(
        shape.as_proto().layout.as_ref().unwrap().dim_level_types,
        wire.layout.as_ref().unwrap().dim_level_types
    );
    assert_eq!(
        shape.as_proto().layout.as_ref().unwrap().dim_unique,
        [false]
    );
    assert_eq!(
        shape.as_proto().layout.as_ref().unwrap().dim_ordered,
        [false]
    );
}

#[test]
fn buffer_outer_layout_drives_host_and_layout_sensitive_verification() {
    let mut wire =
        text::parse_module("HloModule m ENTRY main { ROOT p = b(f32[2]{0:S(5)}) custom-call(), custom_call_target=\"CreateBuffer\" }")
            .unwrap()
            .to_proto()
            .unwrap();
    wire.computations[0].instructions[0]
        .shape
        .mut_or_insert_default()
        .layout = text::parse_shape("f32[2]{0}")
        .unwrap()
        .as_proto()
        .layout
        .clone();
    let module = Module::from_proto(wire).unwrap();
    module
        .verify(VerifyOptions {
            verify_no_host_memory_space: true,
            layout_sensitive: true,
            ..Default::default()
        })
        .unwrap();
    let mut wire = module.to_proto().unwrap();
    wire.computations[0].instructions[0]
        .shape
        .mut_or_insert_default()
        .layout
        .mut_or_insert_default()
        .memory_space = 5;
    assert!(
        Module::from_proto(wire)
            .unwrap()
            .verify(VerifyOptions {
                verify_no_host_memory_space: true,
                ..Default::default()
            })
            .is_err()
    );
}

#[test]
fn ignored_shape_payloads_do_not_trigger_unbounded_host_or_precision_checks() {
    let mut wire = text::parse_module(
        "HloModule m ENTRY main { p = f32[2] parameter(0) ROOT n = f32[2] negate(p) }",
    )
    .unwrap()
    .to_proto_with_program_shapes()
    .unwrap();
    let ignored = text::parse_shape("f64[?]{0:S(5)}").unwrap().into_proto();
    for instruction in &mut wire.computations[0].instructions {
        instruction
            .shape
            .as_mut()
            .unwrap()
            .tuple_shapes
            .push(ignored.clone());
    }
    let module = Module::from_proto(wire).unwrap();
    module
        .verify(VerifyOptions {
            verify_no_host_memory_space: true,
            layout_sensitive: true,
            ..Default::default()
        })
        .unwrap();
    assert!(text::print_module(&module).is_err());
}

#[test]
fn collective_shape_comparison_ignores_nonsemantic_payloads() {
    let expected = array();
    let mut wire = expected.as_proto().clone();
    wire.tuple_shapes.push(ShapeProto {
        dimensions: vec![i64::MIN],
        ..Default::default()
    });
    let shape = Shape::from_proto(wire).unwrap();
    let result = inference::infer_all_to_all_tuple(&[expected, shape.clone()]).unwrap();
    assert!(result.is_tuple());
    assert!(
        result
            .subshape(&[1])
            .unwrap()
            .equal_including_layout(&shape)
    );
}

#[test]
fn literal_shape_ignored_payloads_do_not_introduce_opaque_or_dynamic_leaves() {
    let mut wire = Literal::from_bits(array(), &[1, 2]).unwrap().into_proto();
    wire.shape
        .as_mut()
        .unwrap()
        .tuple_shapes
        .push(Shape::opaque().into_proto());
    let literal = Literal::from_proto(wire).unwrap();
    assert_eq!(literal.bits().unwrap(), [1, 2]);
    assert!(!literal.shape().is_dynamic());
}
