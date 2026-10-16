use xlahlo::{
    AxisRef, Mesh, NamedSharding, Sharding,
    proto::xla_data::{
        AxisRefProto, MeshProto, NamedShardingProto, OpSharding,
        axis_ref_proto::SubAxis,
        mesh_proto::{IotaTransform, MeshAxis},
        named_sharding_proto::DimensionSharding,
    },
    text,
};

fn mesh(sizes: &[i64]) -> MeshProto {
    MeshProto {
        axes: sizes
            .iter()
            .enumerate()
            .map(|(index, &size)| MeshAxis {
                name: format!("a{index}"),
                size,
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }
}
fn axis(index: i64, sub: Option<(i64, i64)>) -> AxisRefProto {
    AxisRefProto {
        mesh_axis_index: index,
        sub_axis_info: sub
            .map(|(pre_size, size)| SubAxis {
                pre_size,
                size,
                ..Default::default()
            })
            .into(),
        ..Default::default()
    }
}
fn named() -> NamedShardingProto {
    NamedShardingProto {
        mesh: Some(mesh(&[4, 2])).into(),
        dim_shardings: vec![DimensionSharding {
            axes: vec![axis(0, None)],
            is_closed: true,
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn mesh_assignments_validate_and_query_transposed_iota_without_allocating_devices() {
    let mut wire = mesh(&[2, 3]);
    wire.iota_transform = Some(IotaTransform {
        reshape_dims: vec![3, 2],
        transpose_perm: vec![1, 0],
        ..Default::default()
    })
    .into();
    let checked = Mesh::from_proto(wire.clone()).unwrap();
    assert_eq!(
        (0..6)
            .map(|index| checked.device_at(index).unwrap())
            .collect::<Vec<_>>(),
        [0, 2, 4, 1, 3, 5]
    );
    assert!(checked.device_at(-1).is_err());
    assert!(checked.device_at(6).is_err());
    wire.device_ids = vec![0, 2, 4, 1, 3, 5];
    assert!(Mesh::from_proto(wire.clone()).is_err());
    wire.iota_transform.clear();
    assert!(Mesh::from_proto(wire.clone()).is_ok());
    wire.device_ids = vec![0, 1, 2, 3, 4, 5];
    assert!(Mesh::from_proto(wire.clone()).is_err());
    wire.device_ids = vec![0, 2, 4, 1, 3, 3];
    assert!(Mesh::from_proto(wire.clone()).is_err());
    wire.device_ids = vec![-1, 2, 4, 1, 3, 5];
    assert!(Mesh::from_proto(wire).is_err());
    let huge = Mesh::from_proto(mesh(&[1_000_000_000, 2])).unwrap();
    assert_eq!(huge.device_at(1_999_999_999).unwrap(), 1_999_999_999);
    let mut duplicate = mesh(&[2, 3]);
    duplicate.axes[1].name = "a0".into();
    assert!(Mesh::from_proto(duplicate).is_err());
    assert!(Mesh::from_proto(mesh(&[0])).is_err());
    assert!(Mesh::from_proto(mesh(&[i64::MAX, 2])).is_err());
    assert_eq!(
        Mesh::from_proto(MeshProto::default())
            .unwrap()
            .device_count(),
        0
    );
}

#[test]
fn named_axes_require_valid_subdivisions_and_global_nonoverlap() {
    let mesh = Mesh::from_proto(mesh(&[24, 2])).unwrap();
    for (pre, size, valid) in [
        (1, 2, true),
        (2, 3, true),
        (6, 4, true),
        (1, 24, false),
        (5, 2, false),
        (0, 2, false),
        (1, 1, false),
        (i64::MAX, 2, false),
    ] {
        assert_eq!(
            AxisRef::from_proto(axis(0, Some((pre, size))))
                .and_then(|axis| axis.validate(&mesh))
                .is_ok(),
            valid
        );
    }
    let a = AxisRef::from_proto(axis(0, Some((1, 2)))).unwrap();
    let b = AxisRef::from_proto(axis(0, Some((2, 3)))).unwrap();
    let c = AxisRef::from_proto(axis(0, Some((3, 2)))).unwrap();
    assert!(a.can_merge(&b));
    assert!(a.can_coexist_without_overlap(&b));
    assert!(!a.can_coexist_without_overlap(&c));
    assert!(!a.can_coexist_without_overlap(&a));
    assert!(
        !AxisRef::from_proto(axis(0, None))
            .unwrap()
            .can_coexist_without_overlap(&a)
    );
    assert!(a.can_coexist_without_overlap(&AxisRef::from_proto(axis(1, None)).unwrap()));
    assert!(AxisRef::from_proto(axis(-1, None)).is_err());
    assert!(
        AxisRef::from_proto(axis(2, None))
            .unwrap()
            .validate(&mesh)
            .is_err()
    );
}

#[test]
fn dimension_axes_need_not_be_sorted_but_other_axes_must_be() {
    let mut wire = named();
    wire.dim_shardings[0].axes = vec![axis(1, None), axis(0, None)];
    assert!(NamedSharding::from_proto(wire.clone()).is_ok());
    wire.dim_shardings.clear();
    wire.replicated_axes = vec![axis(1, None), axis(0, None)];
    assert!(NamedSharding::from_proto(wire.clone()).is_err());
    wire.replicated_axes.reverse();
    assert!(NamedSharding::from_proto(wire.clone()).is_ok());
    wire.manual_axes = vec![axis(0, None)];
    assert!(NamedSharding::from_proto(wire).is_err());
    let mut wire = named();
    wire.dim_shardings[0].axes = vec![axis(0, Some((1, 2))), axis(0, Some((2, 2)))];
    assert!(NamedSharding::from_proto(wire).is_err());
}

#[test]
fn named_leaf_modes_and_shape_constraints_follow_cpp() {
    let mut wire = named();
    let checked = NamedSharding::from_proto(wire.clone()).unwrap();
    assert!(!checked.is_replicated());
    let wrapped = |named| OpSharding {
        named_sharding: Some(named).into(),
        ..Default::default()
    };
    let shape = text::parse_shape("f32[3]").unwrap();
    Sharding::from_proto(wrapped(wire.clone()))
        .unwrap()
        .validate(&shape, Some(8))
        .unwrap();
    assert!(
        Sharding::from_proto(wrapped(wire.clone()))
            .unwrap()
            .validate(&shape, Some(4))
            .is_err()
    );
    assert!(
        Sharding::from_proto(wrapped(wire.clone()))
            .unwrap()
            .validate(&text::parse_shape("f32[3,7]").unwrap(), None)
            .is_err()
    );
    wire.dim_shardings.clear();
    assert!(
        NamedSharding::from_proto(wire.clone())
            .unwrap()
            .is_replicated()
    );
    wire.manual_axes = vec![axis(0, None), axis(1, None)];
    assert!(NamedSharding::from_proto(wire.clone()).unwrap().is_manual());
    Sharding::from_proto(wrapped(wire.clone()))
        .unwrap()
        .validate(&text::parse_shape("f32[3,7,9]").unwrap(), Some(1))
        .unwrap();
    wire.manual_axes.clear();
    wire.unreduced_axes = vec![axis(0, None), axis(1, None)];
    assert!(NamedSharding::from_proto(wire).unwrap().is_unreduced());
    let single = NamedShardingProto {
        mesh: Some(MeshProto {
            device_ids: vec![7],
            ..Default::default()
        })
        .into(),
        ..Default::default()
    };
    assert!(
        NamedSharding::from_proto(single.clone())
            .unwrap()
            .is_single_device()
    );
    assert!(
        Sharding::from_proto(wrapped(single))
            .unwrap()
            .validate(&shape, Some(2))
            .is_err()
    );
}

#[test]
fn named_equality_ignores_explicit_replication_metadata_and_mesh_names() {
    let original = named();
    let mut changed = original.clone();
    changed.mesh.as_mut().unwrap().axes[0].name = "renamed".into();
    changed.replicated_axes.push(axis(1, None));
    changed.metadata.push(Default::default());
    changed
        .special_fields
        .mut_unknown_fields()
        .add_varint(4000, 7);
    assert!(
        NamedSharding::from_proto(original.clone())
            .unwrap()
            .equal_semantics(&NamedSharding::from_proto(changed.clone()).unwrap())
    );
    changed.dim_shardings[0].is_closed = false;
    assert!(
        !NamedSharding::from_proto(original.clone())
            .unwrap()
            .equal_semantics(&NamedSharding::from_proto(changed).unwrap())
    );
    let mut reduced = original.clone();
    reduced.reduction_op = protobuf::EnumOrUnknown::from_i32(1);
    assert!(
        !NamedSharding::from_proto(original.clone())
            .unwrap()
            .equal_semantics(&NamedSharding::from_proto(reduced.clone()).unwrap())
    );
    reduced.reduction_op = protobuf::EnumOrUnknown::from_i32(999);
    assert!(
        NamedSharding::from_proto(original)
            .unwrap()
            .equal_semantics(&NamedSharding::from_proto(reduced).unwrap())
    );
    let replicated = NamedSharding::from_proto(NamedShardingProto::default()).unwrap();
    let mut other = named();
    other.dim_shardings[0].axes.clear();
    assert!(replicated.equal_semantics(&NamedSharding::from_proto(other).unwrap()));
}

#[test]
fn compact_mesh_equality_matches_enumerated_small_transforms_and_scales() {
    let mut candidates = Vec::new();
    for dimensions in [
        vec![12],
        vec![2, 6],
        vec![3, 4],
        vec![2, 2, 3],
        vec![1, 3, 4],
    ] {
        fn permutations(values: &mut [i64], start: usize, output: &mut Vec<Vec<i64>>) {
            if start == values.len() {
                output.push(values.to_vec());
                return;
            }

            for index in start..values.len() {
                values.swap(start, index);
                permutations(values, start + 1, output);
                values.swap(start, index);
            }
        }
        let mut perms = Vec::new();
        permutations(
            &mut (0..dimensions.len() as i64).collect::<Vec<_>>(),
            0,
            &mut perms,
        );
        for transpose_perm in perms {
            let mut wire = mesh(&[3, 4]);
            wire.iota_transform = Some(IotaTransform {
                reshape_dims: dimensions.clone(),
                transpose_perm,
                ..Default::default()
            })
            .into();
            candidates.push(Mesh::from_proto(wire).unwrap());
        }
    }
    for left in &candidates {
        for right in &candidates {
            assert_eq!(
                left.device_assignment_equal(right),
                (0..12).all(|index| left.device_at(index) == right.device_at(index))
            );
        }
    }
    let plain = Mesh::from_proto(mesh(&[1_000_000_000, 2])).unwrap();
    let mut transformed = plain.as_proto().clone();
    transformed.iota_transform = Some(IotaTransform {
        reshape_dims: vec![1, 2, 1_000_000_000],
        transpose_perm: vec![0, 1, 2],
        ..Default::default()
    })
    .into();
    assert!(plain.device_assignment_equal(&Mesh::from_proto(transformed.clone()).unwrap()));
    transformed.iota_transform.as_mut().unwrap().transpose_perm = vec![0, 2, 1];
    assert!(!plain.device_assignment_equal(&Mesh::from_proto(transformed).unwrap()));
}

#[test]
fn mixed_named_and_legacy_equality_converts_axis_order_and_implicit_replication() {
    use xlahlo::proto::xla_data::op_sharding::Type;
    let named_wire = |named| OpSharding {
        named_sharding: Some(named).into(),
        ..Default::default()
    };
    let equal = |left, right| {
        Sharding::from_proto(left)
            .unwrap()
            .equal_semantics(&Sharding::from_proto(right).unwrap())
            .unwrap()
    };
    let legacy = OpSharding {
        type_: Type::OTHER.into(),
        tile_assignment_dimensions: vec![4, 2],
        iota_reshape_dims: vec![8],
        iota_transpose_perm: vec![0],
        replicate_on_last_tile_dim: true,
        ..Default::default()
    };
    assert!(equal(named_wire(named()), legacy.clone()));
    let mut swapped = named();
    swapped.dim_shardings[0].axes = vec![axis(1, None)];
    let mut swapped_legacy = legacy.clone();
    swapped_legacy.tile_assignment_dimensions = vec![2, 4];
    swapped_legacy.iota_reshape_dims = vec![4, 2];
    swapped_legacy.iota_transpose_perm = vec![1, 0];
    assert!(equal(named_wire(swapped), swapped_legacy));
    let mut subset = named();
    subset.dim_shardings[0].axes = vec![axis(0, Some((1, 2)))];
    let mut subset_legacy = legacy;
    subset_legacy.tile_assignment_dimensions = vec![2, 4];
    assert!(equal(named_wire(subset.clone()), subset_legacy.clone()));
    subset.mesh.as_mut().unwrap().device_ids = vec![7, 6, 5, 4, 3, 2, 1, 0];
    subset_legacy.iota_reshape_dims.clear();
    subset_legacy.iota_transpose_perm.clear();
    subset_legacy.tile_assignment_devices = vec![7, 6, 5, 4, 3, 2, 1, 0];
    assert!(equal(named_wire(subset.clone()), subset_legacy));
    subset.dim_shardings.clear();
    subset.manual_axes = vec![axis(0, None), axis(1, None)];
    assert!(equal(
        named_wire(subset),
        OpSharding {
            type_: Type::MANUAL.into(),
            ..Default::default()
        }
    ));
}
