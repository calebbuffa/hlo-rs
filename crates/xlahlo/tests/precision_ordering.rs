use xlahlo::{PrimitiveType as P, Shape, inference, text};

#[test]
fn floating_precision_prioritizes_range_then_significand_then_storage() {
    let types = [
        P::F4E2M1FN,
        P::F6E2M3FN,
        P::F8E3M4,
        P::F6E3M2FN,
        P::F8E4M3B11FNUZ,
        P::F8E4M3,
        P::F8E4M3FN,
        P::F8E5M2,
        P::F16,
        P::F8E8M0FNU,
        P::BF16,
        P::F32,
        P::F64,
    ];
    for (i, &a) in types.iter().enumerate() {
        for (j, &b) in types.iter().enumerate() {
            assert_eq!(a.higher_precision(b).unwrap(), types[i.max(j)]);
        }
    }
    for (a, b) in [
        (P::F8E4M3, P::F8E4M3FNUZ),
        (P::F8E5M2, P::F8E5M2FNUZ),
        (P::PRED, P::U1),
    ] {
        assert!(a.higher_precision(b).is_err());
        assert!(b.higher_precision(a).is_err());
        assert_eq!(a.higher_precision(a).unwrap(), a);
    }
}

#[test]
fn complex_signedness_and_integer_width_use_upstream_lexicographic_priority() {
    for (a, b, expected) in [
        (P::C64, P::F64, P::C64),
        (P::C128, P::C64, P::C128),
        (P::S8, P::U8, P::S8),
        (P::S8, P::U16, P::U16),
        (P::S1, P::PRED, P::S1),
        (P::F4E2M1FN, P::S64, P::F4E2M1FN),
        (P::F8E8M0FNU, P::F16, P::F8E8M0FNU),
    ] {
        assert_eq!(a.higher_precision(b).unwrap(), expected);
        assert_eq!(b.higher_precision(a).unwrap(), expected);
    }
    for invalid in [
        P::PRIMITIVE_TYPE_INVALID,
        P::TUPLE,
        P::TOKEN,
        P::OPAQUE_TYPE,
        P::BUFFER,
    ] {
        assert!(invalid.higher_precision(P::F32).is_err());
        assert!(P::F32.higher_precision(invalid).is_err());
    }
}

#[test]
fn all_array_primitive_pairs_are_symmetric_and_idempotent() {
    use protobuf::Enum;
    for &a in P::VALUES.iter().filter(|ty| ty.is_array_element()) {
        assert_eq!(a.higher_precision(a).unwrap(), a);
        for &b in P::VALUES.iter().filter(|ty| ty.is_array_element()) {
            assert_eq!(a.higher_precision(b), b.higher_precision(a));
        }
    }
}

#[test]
fn dot_and_convolution_default_precision_share_audited_ordering() {
    for (left, right, expected) in [
        (P::F16, P::BF16, P::BF16),
        (P::C64, P::F64, P::C64),
        (P::F8E4M3FN, P::F8E4M3, P::F8E4M3FN),
        (P::S8, P::U16, P::U16),
    ] {
        let lhs = Shape::array(left, vec![2, 3]).unwrap();
        let rhs = Shape::array(right, vec![3, 4]).unwrap();
        let dimensions = xlahlo::proto::xla_data::DotDimensionNumbers {
            lhs_contracting_dimensions: vec![1],
            rhs_contracting_dimensions: vec![0],
            ..Default::default()
        };
        assert_eq!(
            inference::infer_dot(&lhs, &rhs, &dimensions, None)
                .unwrap()
                .element_type(),
            expected
        );
        let dimensions = text::parse_convolution_dimensions("bf_io->bf").unwrap();
        assert_eq!(
            inference::infer_convolution(&lhs, &rhs, 1, 1, &Default::default(), &dimensions, None)
                .unwrap()
                .element_type(),
            expected
        );
    }
}

#[test]
fn array_inference_promotes_floating_types_but_not_integer_domains() {
    use xlahlo::Opcode as O;
    let a = Shape::array(P::F16, vec![2, 3]).unwrap();
    let b = Shape::array(P::BF16, vec![2, 3]).unwrap();
    assert_eq!(
        inference::infer_binary(O::Add, &a, &b)
            .unwrap()
            .element_type(),
        P::BF16
    );
    assert_eq!(
        inference::infer_binary(O::Compare, &a, &b)
            .unwrap()
            .element_type(),
        P::PRED
    );
    let scalar = Shape::scalar(P::BF16).unwrap();
    assert_eq!(
        inference::infer_binary(O::Add, &scalar, &a)
            .unwrap()
            .element_type(),
        P::BF16
    );
    let vector = Shape::array(P::BF16, vec![3]).unwrap();
    assert_eq!(
        inference::infer_binary_in_dim(O::Add, &a, &vector, &[1])
            .unwrap()
            .element_type(),
        P::BF16
    );
    assert_eq!(
        inference::infer_pad(
            &a,
            &scalar,
            &xlahlo::proto::xla_data::PaddingConfig {
                dimensions: vec![Default::default(); 2],
                ..Default::default()
            }
        )
        .unwrap()
        .element_type(),
        P::BF16
    );
    assert_eq!(
        inference::infer_select(&Shape::scalar(P::PRED).unwrap(), &a, &b)
            .unwrap()
            .element_type(),
        P::BF16
    );
    assert_eq!(inference::infer_clamp(&scalar, &a, &b).unwrap(), a);
    assert!(inference::infer_binary(O::Complex, &a, &b).is_err());
    let ints = Shape::array(P::S16, vec![2, 3]).unwrap();
    assert!(inference::infer_binary(O::Add, &a, &ints).is_err());
    assert!(inference::infer_concatenate(&[a.clone(), ints.clone()], 0).is_err());
    assert!(inference::infer_clamp(&scalar, &ints, &ints).is_err());
    assert!(inference::infer_select(&Shape::scalar(P::PRED).unwrap(), &a, &ints).is_err());
}

#[test]
fn floating_binary_pairs_match_precision_selection_including_ambiguous_comparisons() {
    use protobuf::Enum;
    for &a in P::VALUES.iter().filter(|ty| ty.is_floating()) {
        for &b in P::VALUES.iter().filter(|ty| ty.is_floating()) {
            let left = Shape::array(a, vec![2]).unwrap();
            let right = Shape::array(b, vec![2]).unwrap();
            match a.higher_precision(b) {
                Ok(element) => assert_eq!(
                    inference::infer_binary(xlahlo::Opcode::Add, &left, &right)
                        .unwrap()
                        .element_type(),
                    element
                ),
                Err(_) => {
                    assert!(inference::infer_binary(xlahlo::Opcode::Add, &left, &right).is_err());
                    assert!(
                        inference::infer_binary(xlahlo::Opcode::Compare, &left, &right).is_err()
                    );
                }
            }
        }
    }
}

#[test]
fn concatenation_compares_precision_with_first_not_accumulated_result() {
    let operands = [P::F16, P::F64, P::F32].map(|ty| Shape::array(ty, vec![2]).unwrap());
    let result = inference::infer_concatenate(&operands, 0).unwrap();
    assert_eq!(result.element_type(), P::F32);
    assert_eq!(result.dimensions(), [6]);
    let operands = [P::F16, P::F32, P::BF16].map(|ty| Shape::array(ty, vec![2]).unwrap());
    assert_eq!(
        inference::infer_concatenate(&operands, 0)
            .unwrap()
            .element_type(),
        P::BF16
    );
}

#[test]
fn mixed_precision_clamp_preserves_operand_even_for_ambiguous_precision_keys() {
    let a = Shape::array(P::F8E4M3, vec![2]).unwrap();
    let b = Shape::array(P::F8E4M3FNUZ, vec![2]).unwrap();
    assert_eq!(inference::infer_clamp(&a, &b, &a).unwrap(), b);
}

#[test]
fn map_allows_floating_parameter_precision_and_scalar_broadcasts() {
    use xlahlo::proto::xla_data::ProgramShapeProto;
    let operands = [
        Shape::scalar(P::F16).unwrap(),
        Shape::array(P::BF16, vec![2]).unwrap(),
    ];
    let program = ProgramShapeProto {
        parameters: vec![
            Shape::scalar(P::F32).unwrap().into_proto(),
            Shape::scalar(P::F64).unwrap().into_proto(),
        ],
        result: Some(Shape::scalar(P::S32).unwrap().into_proto()).into(),
        ..Default::default()
    };
    assert_eq!(
        inference::infer_map(&operands, &program, &[0]).unwrap(),
        text::parse_shape("s32[2]").unwrap()
    );
    let mut bad = program;
    bad.parameters[1] = Shape::scalar(P::S16).unwrap().into_proto();
    assert!(inference::infer_map(&operands, &bad, &[0]).is_err());
}

#[test]
fn select_unbounded_values_merge_bounds_and_validate_both_predicate_relations() {
    use xlahlo::Dimension as D;
    let unbounded = Shape::dynamic_array(P::F16, vec![D::Unbounded]).unwrap();
    for dim in [D::Static(3), D::Bounded(3), D::Unbounded] {
        let rhs = Shape::dynamic_array(P::BF16, vec![dim]).unwrap();
        for pred in [
            Shape::scalar(P::PRED).unwrap(),
            Shape::dynamic_array(P::PRED, vec![D::Unbounded]).unwrap(),
        ] {
            let result = inference::infer_select(&pred, &unbounded, &rhs).unwrap();
            assert_eq!(result.dimension(0).unwrap(), dim);
            assert_eq!(result.element_type(), P::BF16);
            let result = inference::infer_select(&pred, &rhs, &unbounded).unwrap();
            assert_eq!(result.dimension(0).unwrap(), dim);
        }
    }
    let rhs = Shape::array(P::BF16, vec![3]).unwrap();
    assert!(
        inference::infer_select(&Shape::array(P::PRED, vec![2]).unwrap(), &unbounded, &rhs)
            .is_err()
    );
    let low = Shape::array(P::BF16, vec![2]).unwrap();
    assert!(inference::infer_clamp(&low, &unbounded, &rhs).is_err());
}

#[test]
fn reducer_inference_allows_different_float_precisions_with_exact_accumulator_result() {
    use xlahlo::proto::xla_data::ProgramShapeProto;
    let program = ProgramShapeProto {
        parameters: vec![
            Shape::scalar(P::F32).unwrap().into_proto(),
            Shape::scalar(P::F16).unwrap().into_proto(),
        ],
        result: Some(Shape::scalar(P::F32).unwrap().into_proto()).into(),
        ..Default::default()
    };
    let operands = [
        Shape::array(P::BF16, vec![3]).unwrap(),
        Shape::scalar(P::F64).unwrap(),
    ];
    assert_eq!(
        inference::infer_reduce(&operands, &[0], &program)
            .unwrap()
            .element_type(),
        P::F32
    );
    let window = text::parse_window("size=2").unwrap();
    let result = inference::infer_reduce_window(&operands, &window, &program).unwrap();
    assert_eq!(result.element_type(), P::F64);
    assert_eq!(result.dimensions(), [2]);
    let mut bad = program;
    bad.parameters[0] = Shape::scalar(P::BF16).unwrap().into_proto();
    assert!(inference::infer_reduce(&operands, &[0], &bad).is_err());
    bad.parameters[0] = Shape::scalar(P::F32).unwrap().into_proto();
    bad.parameters[1] = Shape::scalar(P::S32).unwrap().into_proto();
    assert!(inference::infer_reduce(&operands, &[0], &bad).is_err());
}

#[test]
fn default_verifier_still_requires_exact_map_reduce_and_window_parameter_types() {
    for source in [
        r#"HloModule m
f { ROOT p = f64[] parameter(0) }
ENTRY main { p = f32[2] parameter(0) ROOT m = f64[2] map(p), to_apply=f }"#,
        r#"HloModule m
f { p = f64[] parameter(0) q = f32[] parameter(1) ROOT r = f64[] copy(p) }
ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0)
ROOT r = f64[] reduce(p,z), dimensions={0}, to_apply=f }"#,
        r#"HloModule m
f { p = f64[] parameter(0) q = f32[] parameter(1) ROOT r = f64[] copy(p) }
ENTRY main { p = f32[2] parameter(0) z = f32[] constant(0)
ROOT r = f32[1] reduce-window(p,z), window={size=2}, to_apply=f }"#,
    ] {
        let module = text::parse_module(source).unwrap();
        assert_eq!(
            module.verify(Default::default()).unwrap_err().category,
            "mixed-precision",
            "{source}"
        );
    }
}
