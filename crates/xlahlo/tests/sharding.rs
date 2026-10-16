use xlahlo::{
    Shape, Sharding,
    proto::xla_data::{OpSharding, op_sharding::Type},
    text,
};

fn shape(source: &str) -> Shape {
    text::parse_shape(source).unwrap()
}
fn tiled(dimensions: &[i64], devices: &[i64]) -> OpSharding {
    OpSharding {
        type_: Type::OTHER.into(),
        tile_assignment_dimensions: dimensions.into(),
        tile_assignment_devices: devices.into(),
        ..Default::default()
    }
}
fn validate(wire: OpSharding, source: &str, devices: Option<i64>) -> bool {
    Sharding::from_proto(wire)
        .and_then(|sharding| sharding.validate(&shape(source), devices))
        .is_ok()
}

#[test]
fn leaf_modes_normalize_unused_assignments_and_tuple_import() {
    for ty in [
        Type::REPLICATED,
        Type::MANUAL,
        Type::UNKNOWN,
        Type::UNREDUCED,
    ] {
        let wire = OpSharding {
            type_: ty.into(),
            tile_assignment_dimensions: vec![-1],
            tile_assignment_devices: vec![-7, 300],
            ..Default::default()
        };
        for source in ["f32[2]", "(f32[],(s32[2],token[]))", "()", "opaque[]"] {
            assert!(validate(wire.clone(), source, Some(4)));
        }
    }
    assert!(validate(tiled(&[], &[-9]), "f32[2]", Some(4)));
    let maximal = OpSharding {
        type_: Type::MAXIMAL.into(),
        tile_assignment_devices: vec![3],
        ..Default::default()
    };
    assert!(validate(maximal.clone(), "(f32[],f32[2])", Some(4)));
    assert!(!validate(maximal, "f32[2]", Some(3)));
    assert!(!validate(
        OpSharding {
            type_: Type::MAXIMAL.into(),
            ..Default::default()
        },
        "f32[]",
        None
    ));
}

#[test]
fn explicit_assignments_require_unique_in_range_devices_and_exact_data_rank() {
    let wire = tiled(&[2, 2], &[0, 2, 1, 3]);
    assert!(validate(wire.clone(), "f32[3,7]", Some(4)));
    assert!(validate(wire.clone(), "b(f32[3,7])", Some(4)));
    assert!(!validate(wire.clone(), "f32[3]", None));
    assert!(!validate(wire.clone(), "f32[3,7]", Some(8)));
    assert!(validate(tiled(&[2], &[7, 19]), "f32[3]", None));
    assert!(!validate(tiled(&[2], &[7, 19]), "f32[3]", Some(2)));
    assert!(!validate(tiled(&[2], &[1, 1]), "f32[3]", None));
    assert!(!validate(tiled(&[2], &[-1, 1]), "f32[3]", None));
    // Top-level tokens skip shape/device validation after successful import.
    assert!(validate(tiled(&[2], &[-1, -1]), "token[]", Some(2)));
    for bad in [
        tiled(&[2], &[0, 1, 2]),
        tiled(&[0, 2], &[0, 1]),
        tiled(&[i64::MAX, 2], &[0, 1]),
    ] {
        assert!(Sharding::from_proto(bad).is_err());
    }
}

#[test]
fn compact_iota_assignments_are_checked_without_materialization() {
    let wire = OpSharding {
        type_: Type::OTHER.into(),
        tile_assignment_dimensions: vec![2, 3],
        iota_reshape_dims: vec![3, 2],
        iota_transpose_perm: vec![1, 0],
        ..Default::default()
    };
    assert!(validate(wire.clone(), "f32[3,9]", Some(6)));
    let mut huge = wire.clone();
    huge.tile_assignment_dimensions = vec![1_000_000_000, 2];
    huge.iota_reshape_dims = vec![2, 1_000_000_000];
    assert!(validate(huge, "f32[1,1]", Some(2_000_000_000)));
    for permutation in [vec![0, 0], vec![-1, 0], vec![0, 2], vec![0]] {
        let mut bad = wire.clone();
        bad.iota_transpose_perm = permutation;
        assert!(Sharding::from_proto(bad).is_err());
    }
    let mut bad = wire.clone();
    bad.iota_reshape_dims = vec![2, 2];
    assert!(Sharding::from_proto(bad).is_err());
    let mut bad = wire;
    bad.tile_assignment_devices = vec![0];
    assert!(Sharding::from_proto(bad).is_err());
}

#[test]
fn partial_and_subgroup_constructors_preserve_upstream_leaf_normalization() {
    let mut wire = tiled(&[2, 2], &[3, 1, 2, 0]);
    wire.replicate_on_last_tile_dim = true;
    assert!(validate(wire.clone(), "f32[3]", Some(4)));
    assert!(!validate(wire.clone(), "f32[3,3]", Some(4)));
    wire.last_tile_dims = vec![Type::REPLICATED.into()];
    assert!(Sharding::from_proto(wire.clone()).is_err());
    wire.replicate_on_last_tile_dim = false;
    assert!(validate(wire.clone(), "f32[3]", Some(4)));
    for ty in [Type::MANUAL, Type::REPLICATED, Type::UNREDUCED] {
        wire.tile_assignment_dimensions = vec![1, 4];
        wire.last_tile_dims = vec![ty.into()];
        // Normalized leaf modes do not retain device bounds or rank checks.
        assert!(validate(wire.clone(), "f32[3,7,9]", Some(1)));
    }
    wire.tile_assignment_dimensions = vec![2, 2];
    wire.last_tile_dims = vec![
        Type::REPLICATED.into(),
        Type::MANUAL.into(),
        Type::UNREDUCED.into(),
    ];
    assert!(Sharding::from_proto(wire).is_err());
}

#[test]
fn tuple_leaf_cardinality_empty_tuples_and_metadata_follow_import_rules() {
    let leaf = OpSharding::default();
    let tuple = |elements| OpSharding {
        type_: Type::TUPLE.into(),
        tuple_shardings: elements,
        ..Default::default()
    };
    assert!(validate(
        tuple(vec![leaf.clone(); 3]),
        "(f32[],(s32[],token[]))",
        None
    ));
    assert!(!validate(
        tuple(vec![leaf.clone(); 2]),
        "(f32[],(s32[],token[]))",
        None
    ));
    assert!(validate(tuple(vec![]), "()", None));
    assert!(validate(tuple(vec![leaf.clone()]), "()", None));
    assert!(!validate(tuple(vec![leaf.clone(); 2]), "()", None));
    assert!(!validate(
        tuple(vec![tuple(vec![leaf.clone()])]),
        "(f32[])",
        None
    ));
    let mut metadata = tuple(vec![leaf]);
    metadata.metadata.push(xlahlo::proto::xla_data::OpMetadata {
        op_type: "source".into(),
        ..Default::default()
    });
    assert!(Sharding::from_proto(metadata.clone()).is_err());
    metadata.metadata[0].op_type = "sdy::reduction_op".into();
    assert!(Sharding::from_proto(metadata).is_ok());
}

#[test]
fn sharding_wire_retains_unknown_fields() {
    let mut wire = tiled(&[2], &[1, 0]);
    wire.special_fields.mut_unknown_fields().add_varint(900, 42);
    let sharding = Sharding::from_proto(wire.clone()).unwrap();
    sharding.validate(&shape("f32[3]"), None).unwrap();
    assert_eq!(sharding.as_proto(), &wire);
    assert_eq!(sharding.into_proto(), wire);
}
