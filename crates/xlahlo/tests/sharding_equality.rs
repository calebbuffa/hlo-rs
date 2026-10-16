use xlahlo::{
    Sharding,
    proto::xla_data::{OpMetadata, OpSharding, op_sharding::Type},
    text,
};

fn tiled(dimensions: &[i64], devices: &[i64], types: &[Type]) -> OpSharding {
    OpSharding {
        type_: Type::OTHER.into(),
        tile_assignment_dimensions: dimensions.to_vec(),
        tile_assignment_devices: devices.to_vec(),
        last_tile_dims: types.iter().map(|&ty| ty.into()).collect(),
        ..Default::default()
    }
}
fn equal(left: OpSharding, right: OpSharding) -> bool {
    Sharding::from_proto(left)
        .unwrap()
        .equal_semantics(&Sharding::from_proto(right).unwrap())
        .unwrap()
}

#[test]
fn constructor_normalization_ignores_unused_fields_but_distinguishes_leaf_modes() {
    let replicated = OpSharding::default();
    let mut irrelevant = replicated.clone();
    irrelevant.tile_assignment_devices = vec![-99];
    irrelevant.tile_assignment_dimensions = vec![3, 4];
    irrelevant.metadata.push(OpMetadata {
        op_name: "location".into(),
        ..Default::default()
    });
    irrelevant
        .special_fields
        .mut_unknown_fields()
        .add_varint(7000, 1);
    assert!(equal(replicated.clone(), irrelevant));
    assert!(equal(replicated.clone(), tiled(&[-4], &[-7], &[])));
    for ty in [Type::MANUAL, Type::UNREDUCED, Type::UNKNOWN] {
        assert!(!equal(
            replicated.clone(),
            OpSharding {
                type_: ty.into(),
                ..Default::default()
            }
        ));
    }
    let manual = OpSharding {
        type_: Type::MANUAL.into(),
        ..Default::default()
    };
    assert!(equal(
        manual.clone(),
        tiled(&[1, 2, 1], &[1, 0], &[Type::MANUAL, Type::REPLICATED])
    ));
    assert!(!equal(manual, tiled(&[1, 2], &[0, 1], &[Type::UNREDUCED])));
}

#[test]
fn partial_replication_sorts_devices_and_removes_unit_subgroups() {
    let mut legacy = tiled(&[2, 2], &[1, 0, 3, 2], &[]);
    legacy.replicate_on_last_tile_dim = true;
    assert!(equal(
        legacy.clone(),
        tiled(&[2, 2], &[0, 1, 2, 3], &[Type::REPLICATED])
    ));
    assert!(!equal(
        legacy,
        tiled(&[2, 2], &[0, 2, 1, 3], &[Type::REPLICATED])
    ));
    let mut legacy = tiled(&[2, 1], &[1, 0], &[]);
    legacy.replicate_on_last_tile_dim = true;
    assert!(equal(legacy, tiled(&[2], &[1, 0], &[])));
    assert!(equal(
        tiled(&[2, 1, 1], &[1, 0], &[Type::MANUAL, Type::UNKNOWN]),
        tiled(&[2], &[1, 0], &[])
    ));
}

#[test]
fn subgroup_merging_and_sorting_transpose_assignment_axes() {
    let left = tiled(
        &[2, 2, 2],
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[Type::REPLICATED, Type::MANUAL],
    );
    let right = tiled(
        &[2, 2, 2],
        &[0, 2, 1, 3, 4, 6, 5, 7],
        &[Type::MANUAL, Type::REPLICATED],
    );
    assert!(equal(left.clone(), right.clone()));
    assert!(!equal(
        left,
        tiled(
            &[2, 2, 2],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[Type::MANUAL, Type::REPLICATED]
        )
    ));
    assert!(equal(
        tiled(
            &[2, 2, 2],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[Type::MANUAL, Type::MANUAL]
        ),
        tiled(&[2, 4], &[0, 1, 2, 3, 4, 5, 6, 7], &[Type::MANUAL])
    ));
}

#[test]
fn compact_assignments_compare_with_explicit_and_equivalent_large_transforms() {
    let mut compact = tiled(&[2, 3], &[], &[]);
    compact.iota_reshape_dims = vec![3, 2];
    compact.iota_transpose_perm = vec![1, 0];
    assert!(equal(compact, tiled(&[2, 3], &[0, 2, 4, 1, 3, 5], &[])));
    let mut left = tiled(&[2_000_000_000], &[], &[]);
    left.iota_reshape_dims = vec![2_000_000_000];
    left.iota_transpose_perm = vec![0];
    let mut right = left.clone();
    right.iota_reshape_dims = vec![2, 1_000_000_000];
    right.iota_transpose_perm = vec![0, 1];
    assert!(equal(left.clone(), right.clone()));
    right.iota_transpose_perm = vec![1, 0];
    assert!(!equal(left, right));
}

#[test]
fn reductions_shard_groups_and_flat_tuple_entries_are_semantic() {
    let mut left = OpSharding::default();
    left.metadata.push(OpMetadata {
        op_type: "sdy::reduction_op".into(),
        op_name: "MAX".into(),
        ..Default::default()
    });
    assert!(!equal(left.clone(), OpSharding::default()));
    let mut right = left.clone();
    right.metadata.push(OpMetadata {
        op_type: "sdy::reduction_op".into(),
        op_name: "SUM".into(),
        ..Default::default()
    });
    assert!(equal(left.clone(), right));
    left.is_shard_group = true;
    left.shard_group_id = 3;
    let mut right = left.clone();
    right.shard_group_id = 4;
    assert!(!equal(left.clone(), right));
    let tuple = Sharding::from_proto(left)
        .unwrap()
        .normalize_tuple(&text::parse_shape("(f32[],(s32[],token[]))").unwrap())
        .unwrap();
    assert_eq!(tuple.as_proto().tuple_shardings.len(), 3);
    let mut other = tuple.as_proto().clone();
    other.tuple_shardings[1].type_ = Type::MANUAL.into();
    assert!(
        !tuple
            .equal_semantics(&Sharding::from_proto(other).unwrap())
            .unwrap()
    );
    let empty = Sharding::from_proto(OpSharding::default())
        .unwrap()
        .normalize_tuple(&text::parse_shape("(())").unwrap())
        .unwrap();
    assert_eq!(empty.as_proto().tuple_shardings.len(), 1);
}
