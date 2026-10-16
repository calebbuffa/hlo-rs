use xlahlo::{
    Dimension, Instruction, Literal, Module, Opcode, PrimitiveType as P, Shape, interchange,
    proto::xla_data::{LiteralProto, ShapeProto},
    text,
};

fn array() -> Literal {
    Literal::from_bits(Shape::array(P::F32, vec![2, 2]).unwrap(), &[0, 1, 2, 3]).unwrap()
}

fn nested() -> LiteralProto {
    Literal::tuple(vec![
        array(),
        Literal::tuple(vec![array(), Literal::token()]).unwrap(),
    ])
    .unwrap()
    .into_proto()
}

#[test]
fn import_uses_root_context_for_tuple_and_token_pieces() {
    for ignored_shape in [
        None,
        Some(Shape::token().into_proto()),
        Some(ShapeProto::new()),
    ] {
        let mut wire = nested();
        wire.tuple_literals[1].shape = ignored_shape.into();
        let token = &mut wire.tuple_literals[1].tuple_literals[1];
        token.shape.clear();
        token.f32s.push(17.0);
        token.tuple_literals.push(LiteralProto::new());
        let before = interchange::encode(&wire).unwrap();
        let literal = Literal::from_proto(wire).unwrap();
        assert!(literal.shape().subshape(&[1]).unwrap().is_tuple());
        assert_eq!(literal.shape().subshape(&[1, 1]).unwrap(), Shape::token());
        assert_eq!(interchange::encode(literal.as_proto()).unwrap(), before);
        assert_eq!(
            literal.sub_literal(&[1, 0]).unwrap().bits().unwrap(),
            [0, 1, 2, 3]
        );
        let subtree = literal.sub_literal(&[1]).unwrap();
        assert!(subtree.shape().is_tuple());
        assert_eq!(subtree.sub_literal(&[1]).unwrap().shape(), &Shape::token());
        assert!(literal.sub_literal(&[2]).is_err());
        assert!(literal.sub_literal(&[0, 0]).is_err());
    }
}

#[test]
fn token_literals_preserve_ignored_fields_and_have_no_array_bits() {
    let mut wire = Literal::token().into_proto();
    wire.sparse_indices.push(9);
    wire.f32s.push(17.0);
    wire.tuple_literals.push(LiteralProto::new());
    wire.special_fields.mut_unknown_fields().add_varint(999, 7);
    let before = interchange::encode(&wire).unwrap();
    let literal = Literal::from_proto(wire).unwrap();
    assert_eq!(literal.shape(), &Shape::token());
    assert!(literal.bits().is_err());
    assert_eq!(interchange::encode(literal.as_proto()).unwrap(), before);
}

#[test]
fn array_literals_preserve_unselected_payloads() {
    let mut wire = array().into_proto();
    wire.tuple_literals.push(LiteralProto::new());
    wire.sparse_indices.push(7);
    wire.f64s.push(19.0);
    wire.dynamic_sizes.push(-1);
    wire.special_fields.mut_unknown_fields().add_varint(999, 7);
    let before = interchange::encode(&wire).unwrap();
    let literal = Literal::from_proto(wire).unwrap();
    assert_eq!(literal.bits().unwrap(), [0, 1, 2, 3]);
    assert_eq!(interchange::encode(literal.as_proto()).unwrap(), before);
    let mut malformed = literal.into_proto();
    malformed.f32s.pop();
    assert!(Literal::from_proto(malformed).is_err());
}

#[test]
fn array_leaf_shapes_require_exact_contextual_layout_and_dynamic_flags() {
    for change in 0..6 {
        let mut wire = nested();
        let child = &mut wire.tuple_literals[0];
        match change {
            0 => child.shape.clear(),
            1 => child.shape.as_mut().unwrap().layout.clear(),
            2 => {
                child
                    .shape
                    .as_mut()
                    .unwrap()
                    .layout
                    .as_mut()
                    .unwrap()
                    .minor_to_major = vec![0, 1];
            }
            3 => {
                child
                    .shape
                    .as_mut()
                    .unwrap()
                    .layout
                    .as_mut()
                    .unwrap()
                    .memory_space = 5
            }
            4 => {
                child.shape.as_mut().unwrap().is_dynamic_dimension = vec![true, false];
                child.dynamic_sizes = vec![2, 2];
            }
            5 => wire.shape.as_mut().unwrap().tuple_shapes[0].layout.clear(),
            _ => unreachable!(),
        }
        assert!(Literal::from_proto(wire).is_err(), "change {change}");
    }
}

#[test]
fn contextual_tuple_arity_and_root_shape_are_still_required() {
    let mut wire = nested();
    wire.tuple_literals[1].shape.clear();
    wire.tuple_literals[1].tuple_literals.pop();
    assert!(Literal::from_proto(wire).is_err());
    let mut wire = nested();
    wire.shape.clear();
    assert!(Literal::from_proto(wire).is_err());
    let wire = LiteralProto {
        shape: Some(Shape::opaque().into_proto()).into(),
        ..Default::default()
    };
    assert!(Literal::from_proto(wire).is_err());
    assert!(Literal::tuple(Vec::new()).unwrap().shape().is_tuple());
}

#[test]
fn dynamic_sizes_are_checked_against_contextual_bounds() {
    let shape = Shape::dynamic_array(P::S32, vec![Dimension::Bounded(3)])
        .unwrap()
        .with_default_layout()
        .unwrap();
    for (sizes, accepted) in [
        (vec![], false),
        (vec![-1], false),
        (vec![0], true),
        (vec![3], true),
        (vec![4], false),
        (vec![1, 2], false),
    ] {
        let leaf = LiteralProto {
            shape: Some(shape.as_proto().clone()).into(),
            s32s: vec![1, 2, 3],
            dynamic_sizes: sizes,
            ..Default::default()
        };
        let wire = LiteralProto {
            shape: Some(Shape::tuple(vec![shape.clone()]).unwrap().into_proto()).into(),
            tuple_literals: vec![leaf],
            ..Default::default()
        };
        assert_eq!(Literal::from_proto(wire).is_ok(), accepted);
    }
}

#[test]
fn mixed_dynamic_literal_axes_are_imported_without_upstream_assertion() {
    let shape = Shape::dynamic_array(P::S32, vec![Dimension::Bounded(3), Dimension::Static(2)])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let wire = LiteralProto {
        shape: Some(shape.into_proto()).into(),
        s32s: vec![1, 2, 3, 4, 5, 6],
        dynamic_sizes: vec![3, 2],
        ..Default::default()
    };
    let literal = Literal::from_proto(wire.clone()).unwrap();
    assert_eq!(literal.bits().unwrap(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(literal.as_proto(), &wire);

    let mut invalid = wire;
    invalid.dynamic_sizes = vec![4, 2];
    assert!(Literal::from_proto(invalid).is_err());
}

fn constant(literal: Literal) -> Module {
    let mut module = Module::new("m");
    let main = module.add_computation("main").unwrap();
    let instruction = Instruction::new("c", Opcode::Constant, literal.shape().clone())
        .with_fields(|wire| wire.literal = Some(literal.into_proto()).into())
        .unwrap();
    let root = module.add_instruction(main, instruction, &[], &[]).unwrap();
    module.set_root(main, root).unwrap();
    module.set_entry(main).unwrap();
    module
}

#[test]
fn text_refuses_ignored_payloads_and_contextual_shape_materialization() {
    for mut wire in [array().into_proto(), Literal::token().into_proto()] {
        wire.tuple_literals.push(LiteralProto::new());
        let module = constant(Literal::from_proto(wire).unwrap());
        module.verify(Default::default()).unwrap();
        assert!(text::print_module(&module).is_err());
    }
    for ignored_shape in [None, Some(Shape::token().into_proto())] {
        let mut wire = Literal::tuple(vec![array(), Literal::tuple(vec![array()]).unwrap()])
            .unwrap()
            .into_proto();
        wire.tuple_literals[1].shape = ignored_shape.into();
        let module = constant(Literal::from_proto(wire).unwrap());
        module.verify(Default::default()).unwrap();
        let error = text::print_module(&module).unwrap_err();
        assert_eq!(error.category, "unsupported");
        assert!(error.message.contains("tuple literal payload shape"));
    }
}

#[test]
fn ordinary_nested_array_literals_still_roundtrip_through_text() {
    let literal = Literal::tuple(vec![array(), Literal::tuple(vec![array()]).unwrap()]).unwrap();
    let module = constant(literal);
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    parsed.verify(Default::default()).unwrap();
    let before = &module.as_proto().computations[0].instructions[0].literal;
    let after = &parsed.as_proto().computations[0].instructions[0].literal;
    assert_eq!(before, after);
}
