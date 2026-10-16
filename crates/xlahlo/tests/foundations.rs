use protobuf::{Enum, EnumOrUnknown, Message};
use xlahlo::{
    Dimension, Literal, Opcode, PrimitiveType as P, Shape, UNBOUNDED_SIZE, interchange,
    proto::{
        hlo::HloModuleProto,
        xla_data::{LayoutProto, ShapeProto, TileProto},
    },
};

#[test]
fn complete_opcode_catalog_is_unique_and_round_trips() {
    let mut names = std::collections::BTreeSet::new();
    assert_eq!(Opcode::ALL.len(), 137);
    for &opcode in Opcode::ALL {
        assert!(names.insert(opcode.name()));
        assert_eq!(opcode.name().parse::<Opcode>().unwrap(), opcode);
        assert_eq!(opcode.to_string(), opcode.name());
    }
    assert_eq!(Opcode::Convolution.arity(), None);
    assert_eq!(Opcode::While.arity(), Some(1));
    assert_eq!(Opcode::RaggedAllToAll.arity(), Some(6));
    assert!("stablehlo.add".parse::<Opcode>().is_err());
    assert!("Add".parse::<Opcode>().is_err());
}

#[test]
fn every_primitive_has_the_correct_category_and_wire_number() {
    for &ty in P::VALUES {
        assert_eq!(P::from_i32(ty.value()), Some(ty));
        if ty == P::PRIMITIVE_TYPE_INVALID {
            continue;
        }
        assert_eq!(P::from_hlo_name(&ty.hlo_name()).unwrap(), ty);
        assert_eq!(ty.is_array_element(), ty.bit_width().is_some());
        if ty.is_array_element() {
            let shape = Shape::array(ty, vec![2, 0, 3]).unwrap();
            assert_eq!(shape.element_count().unwrap(), 0);
        } else {
            assert!(Shape::scalar(ty).is_err());
        }
    }
    assert_eq!(P::BUFFER.value(), 34);
    assert_eq!(P::F6E2M3FN.value(), 36);
    assert_eq!(P::C128.bit_width(), Some(128));
}

#[test]
fn dynamic_dimensions_preserve_bounds_and_the_upstream_sentinel() {
    let shape = Shape::dynamic_array(
        P::F32,
        vec![
            Dimension::Static(2),
            Dimension::Bounded(7),
            Dimension::Unbounded,
        ],
    )
    .unwrap();
    assert_eq!(shape.dimensions(), [2, 7, i64::MIN]);
    assert_eq!(UNBOUNDED_SIZE, i64::MIN);
    assert_eq!(shape.dimension(1).unwrap(), Dimension::Bounded(7));
    assert_eq!(shape.dimension(2).unwrap(), Dimension::Unbounded);
    assert!(shape.is_dynamic());
    assert!(shape.element_count().is_err());
    assert!(Shape::array(P::F32, vec![UNBOUNDED_SIZE]).is_err());
    assert!(Shape::dynamic_array(P::F32, vec![Dimension::Bounded(-1)]).is_err());
    let static_shape = Shape::array(P::F32, vec![7]).unwrap();
    let bounded = Shape::dynamic_array(P::F32, vec![Dimension::Bounded(7)]).unwrap();
    assert!(static_shape.compatible(&bounded));
    assert!(!static_shape.compatible(&Shape::array(P::F32, vec![6]).unwrap()));
}

#[test]
fn malformed_shapes_are_errors_instead_of_invalid_canonical_values() {
    for element in [0, 999] {
        assert!(
            Shape::from_proto(ShapeProto {
                element_type: EnumOrUnknown::from_i32(element),
                ..Default::default()
            })
            .is_err()
        );
    }
    assert!(
        Shape::from_proto(ShapeProto {
            element_type: P::F32.into(),
            dimensions: vec![2, 3],
            is_dynamic_dimension: vec![true],
            ..Default::default()
        })
        .is_err()
    );
    assert!(Shape::array(P::U8, vec![i64::MAX, 2]).is_err());
    assert!(Shape::array(P::F64, vec![i64::MAX]).is_err());
    assert!(Shape::array(P::F32, vec![-1]).is_err());
    let tuple = Shape::tuple(vec![
        Shape::token(),
        Shape::tuple(vec![Shape::scalar(P::F64).unwrap()]).unwrap(),
    ])
    .unwrap();
    assert_eq!(tuple.subshape(&[1, 0]).unwrap().element_type(), P::F64);
    assert!(tuple.subshape(&[0, 0]).is_err());
    assert!(tuple.subshape(&[2]).is_err());
    assert!(Shape::buffer(tuple).is_err());
    assert_eq!(
        Shape::buffer(Shape::scalar(P::F32).unwrap())
            .unwrap()
            .element_type(),
        P::BUFFER
    );
}

#[test]
fn layout_validation_checks_permutation_tiles_and_physical_splits() {
    let shape = Shape::array(P::F32, vec![4, 8]).unwrap();
    for permutation in [vec![1, 0], vec![0, 1]] {
        assert!(
            shape
                .clone()
                .with_layout(LayoutProto {
                    minor_to_major: permutation,
                    ..Default::default()
                })
                .is_ok()
        );
    }
    for permutation in [vec![0, 0], vec![2, 1], vec![-1, 0], vec![1]] {
        assert!(
            shape
                .clone()
                .with_layout(LayoutProto {
                    minor_to_major: permutation,
                    ..Default::default()
                })
                .is_err()
        );
    }
    for tile in [vec![0], vec![-2], vec![]] {
        let layout = LayoutProto {
            minor_to_major: vec![1, 0],
            tiles: vec![TileProto {
                dimensions: tile,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(shape.clone().with_layout(layout).is_err());
    }
    assert!(
        shape
            .with_layout(LayoutProto {
                minor_to_major: vec![1, 0],
                tiles: vec![TileProto {
                    dimensions: vec![xlahlo::COMBINE_DIMENSION, 8],
                    ..Default::default()
                }],
                ..Default::default()
            })
            .is_ok()
    );
}

#[test]
fn bit_literals_preserve_every_array_primitive_including_nan_and_signed_zero() {
    for &ty in P::VALUES {
        let Some(width) = ty.bit_width() else {
            continue;
        };
        let patterns = if width == 128 {
            vec![0, 1, u128::MAX, 1_u128 << 127]
        } else {
            vec![0, 1, (1_u128 << width) - 1, 1_u128 << (width - 1)]
        };
        let shape = Shape::array(ty, vec![patterns.len() as i64]).unwrap();
        let literal = Literal::from_bits(shape, &patterns).unwrap();
        assert_eq!(literal.bits().unwrap(), patterns, "{ty:?}");
        let encoded = interchange::encode(literal.as_proto()).unwrap();
        let decoded =
            Literal::from_proto(interchange::decode(&encoded, Default::default()).unwrap())
                .unwrap();
        assert_eq!(decoded.bits().unwrap(), patterns, "{ty:?}");
    }
    for ty in [
        P::S1,
        P::S2,
        P::S4,
        P::U1,
        P::U2,
        P::U4,
        P::F4E2M1FN,
        P::F6E2M3FN,
        P::F6E3M2FN,
    ] {
        let width = ty.bit_width().unwrap();
        let values: Vec<_> = (0..1_u128 << width).collect();
        let literal = Literal::from_bits(
            Shape::array(ty, vec![values.len() as i64]).unwrap(),
            &values,
        )
        .unwrap();
        assert_eq!(literal.bits().unwrap(), values);
    }
    assert!(Literal::from_bits(Shape::scalar(P::U8).unwrap(), &[256]).is_err());
    assert!(Literal::from_bits(Shape::scalar(P::F32).unwrap(), &[]).is_err());
}

#[test]
fn wire_interchange_retains_unknown_fields_and_proto3_optional_presence() {
    let mut wire = HloModuleProto {
        name: "wire".into(),
        pjrt_id: Some(0),
        backend_config: vec![0, 255, 3],
        ..Default::default()
    };
    wire.special_fields
        .mut_unknown_fields()
        .add_varint(900, 1234);
    let encoded = wire.write_to_bytes().unwrap();
    let decoded: HloModuleProto = interchange::decode(&encoded, Default::default()).unwrap();
    assert_eq!(decoded, wire);
    assert!(decoded.pjrt_id.is_some());
    assert!(
        interchange::decode::<HloModuleProto>(
            &encoded,
            interchange::DecodeOptions {
                max_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        interchange::decode::<HloModuleProto>(&[0x0a, 0x7f, 0x41], Default::default()).is_err()
    );
    let json = interchange::print_json(&wire).unwrap();
    let from_json: HloModuleProto = interchange::parse_json(&json).unwrap();
    assert_eq!(from_json.pjrt_id, Some(0));
    assert_eq!(from_json.backend_config, wire.backend_config);
    let text = interchange::print_textproto(&wire);
    // Textproto/JSON are not unknown-field-preserving formats.
    let known = HloModuleProto {
        name: "wire".into(),
        pjrt_id: Some(0),
        backend_config: vec![0, 255, 3],
        ..Default::default()
    };
    let known_text = interchange::print_textproto(&known);
    assert_eq!(
        interchange::parse_textproto::<HloModuleProto>(&known_text).unwrap(),
        known
    );
    assert!(!text.is_empty());
}
