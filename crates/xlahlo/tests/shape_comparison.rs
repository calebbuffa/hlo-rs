use xlahlo::{Dimension as D, PrimitiveType as P, Shape};

#[test]
fn unbounded_extents_are_wildcards_but_concrete_bounds_still_must_match() {
    let unknown = Shape::dynamic_array(P::F32, vec![D::Unbounded]).unwrap();
    let two = Shape::array(P::F32, vec![2]).unwrap();
    let three = Shape::dynamic_array(P::F32, vec![D::Bounded(3)]).unwrap();
    assert!(unknown.compatible(&two));
    assert!(two.compatible(&unknown));
    assert!(unknown.compatible(&three));
    assert!(!two.compatible(&three));
    assert!(!unknown.compatible(&Shape::scalar(P::F32).unwrap()));
    assert!(!unknown.compatible(&Shape::array(P::F64, vec![2]).unwrap()));
    let left = Shape::tuple(vec![Shape::tuple(vec![unknown]).unwrap(), Shape::token()]).unwrap();
    let right = Shape::tuple(vec![Shape::tuple(vec![two]).unwrap(), Shape::token()]).unwrap();
    assert!(left.compatible(&right));
    assert_ne!(left, right);
}

#[test]
fn ignoring_float_precision_is_recursive_and_does_not_ignore_other_domains() {
    for a in [P::F16, P::BF16, P::F32, P::F64, P::F4E2M1FN, P::F8E5M2] {
        for b in [P::F16, P::BF16, P::F32, P::F64, P::F4E2M1FN, P::F8E5M2] {
            let left = Shape::tuple(vec![Shape::array(a, vec![2]).unwrap()]).unwrap();
            let right =
                Shape::tuple(vec![Shape::dynamic_array(b, vec![D::Bounded(2)]).unwrap()]).unwrap();
            assert!(left.compatible_ignoring_fp_precision(&right));
            assert_eq!(left.compatible(&right), a == b);
        }
    }
    for (a, b) in [
        (P::S32, P::U32),
        (P::F32, P::S32),
        (P::C64, P::C128),
        (P::PRED, P::U1),
    ] {
        assert!(
            !Shape::scalar(a)
                .unwrap()
                .compatible_ignoring_fp_precision(&Shape::scalar(b).unwrap())
        );
    }
}

#[test]
fn exact_float_ignoring_equality_recurses_but_preserves_bounds_and_dynamic_flags() {
    fn nested(element: P, dimension: D) -> Shape {
        Shape::tuple(vec![
            Shape::tuple(vec![
                Shape::dynamic_array(element, vec![dimension]).unwrap(),
            ])
            .unwrap(),
            Shape::token(),
        ])
        .unwrap()
    }
    let left = nested(P::F32, D::Bounded(3));
    let right = nested(P::BF16, D::Bounded(3));
    assert!(left.equal_ignoring_layout_and_fp_precision(&right));
    assert!(!left.equal_ignoring_layout(&right));
    for other in [
        nested(P::BF16, D::Static(3)),
        nested(P::BF16, D::Bounded(2)),
        nested(P::BF16, D::Unbounded),
        nested(P::S32, D::Bounded(3)),
    ] {
        assert!(!left.equal_ignoring_layout_and_fp_precision(&other));
    }
    let absent = Shape::array(P::F32, vec![3]).unwrap();
    let present = Shape::array(P::BF16, vec![3])
        .unwrap()
        .with_default_layout()
        .unwrap();
    assert!(absent.equal_ignoring_layout_and_fp_precision(&present));
}

#[test]
fn semantic_layout_equality_is_not_wire_equality() {
    let left = xlahlo::text::parse_shape("f32[4,6]{1,0:T(2,2)L(64)E(4)S(5)SC(0:1,2)M(8)}").unwrap();
    let mut wire = left.as_proto().clone();
    wire.special_fields.mut_unknown_fields().add_varint(900, 42);
    wire.is_dynamic_dimension.clear();
    let layout = wire.layout.as_mut().unwrap();
    layout
        .special_fields
        .mut_unknown_fields()
        .add_varint(901, 43);
    layout.tiles[0]
        .special_fields
        .mut_unknown_fields()
        .add_varint(902, 44);
    layout.split_configs[0]
        .special_fields
        .mut_unknown_fields()
        .add_varint(903, 45);
    layout.dynamic_shape_metadata_prefix_bytes = 16;
    layout.dim_unique = vec![false, true];
    layout.dim_ordered = vec![true, false];
    let right = Shape::from_proto(wire).unwrap();
    assert_ne!(left, right);
    assert!(left.equal_including_layout(&right));
    for change in [
        (|layout: &mut xlahlo::proto::xla_data::LayoutProto| layout.minor_to_major.reverse())
            as fn(&mut xlahlo::proto::xla_data::LayoutProto),
        |layout| layout.tiles[0].dimensions[0] = 3,
        |layout| layout.tail_padding_alignment_in_elements = 32,
        |layout| layout.element_size_in_bits = 8,
        |layout| layout.memory_space = 0,
        |layout| layout.split_configs[0].split_indices.push(3),
    ] {
        let mut layout = left.layout().unwrap().clone();
        change(&mut layout);
        let changed = left.clone().with_layout(layout).unwrap();
        assert!(!left.equal_including_layout(&changed));
    }
    let nested = |shape| Shape::tuple(vec![Shape::buffer(shape).unwrap(), Shape::token()]).unwrap();
    assert!(nested(left).equal_including_layout(&nested(right)));
}

#[test]
fn verifier_shape_and_layout_policies_use_semantic_comparisons() {
    let sources = [
        "HloModule m f { ROOT p = f32[2]{0:M(8)} parameter(0) } ENTRY main { p = f32[2]{0:M(16)} parameter(0) ROOT c = f32[2]{0:M(32)} call(p), to_apply=f }",
        "HloModule m ENTRY main { p = f32[2]{0:M(8)} parameter(0) ROOT c = f32[2]{0:M(16)} negate(p) }",
    ];
    for source in sources {
        let mut wire = xlahlo::text::parse_module(source)
            .unwrap()
            .to_proto()
            .unwrap();
        wire.computations
            .last_mut()
            .unwrap()
            .instructions
            .last_mut()
            .unwrap()
            .shape
            .as_mut()
            .unwrap()
            .layout
            .as_mut()
            .unwrap()
            .special_fields
            .mut_unknown_fields()
            .add_varint(900, 42);
        let module = xlahlo::Module::from_proto(wire).unwrap();
        module
            .verify(xlahlo::VerifyOptions {
                layout_sensitive: true,
                instruction_can_change_layout: Some(|_, _| false),
                ..Default::default()
            })
            .unwrap();
    }
}

#[test]
fn stochastic_and_collective_root_inference_use_layout_semantics() {
    let operand = xlahlo::text::parse_shape("f32[2]{0:M(8)}").unwrap();
    let random = xlahlo::text::parse_shape("u32[2]{0:M(16)}").unwrap();
    let output = xlahlo::inference::infer_stochastic_convert(&operand, &random, P::F16).unwrap();
    assert_eq!(output.element_type(), P::F16);
    let mut wire = xlahlo::text::parse_shape("s32[1]{0:M(8)}")
        .unwrap()
        .as_proto()
        .clone();
    wire.layout
        .as_mut()
        .unwrap()
        .special_fields
        .mut_unknown_fields()
        .add_varint(900, 42);
    xlahlo::inference::infer_collective_reduce(&[operand, Shape::from_proto(wire).unwrap()], true)
        .unwrap();
}
