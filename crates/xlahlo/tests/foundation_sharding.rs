use xlahlo::{
    Diagnostic,
    pliron::{common_traits::Verify, context::Context},
    replica_groups::MeshAxesReplicaGroupList,
    shape::Shape,
    sharding::{
        AxisRef, DimensionSharding, IotaTransform, Mesh, MeshAxis, NamedSharding, Sharding,
        ShardingType as T, SubAxis,
    },
    types::ElementType as P,
};

#[test]
fn malformed_mesh_and_axis_queries_return_typed_errors_instead_of_panicking() {
    let mut invalid_meshes = vec![];
    for sizes in [[i64::MAX, 2], [-1, 2], [0, 2]] {
        invalid_meshes.push(Mesh {
            axes: sizes
                .iter()
                .enumerate()
                .map(|(i, &size)| MeshAxis {
                    name: format!("a{i}"),
                    size,
                })
                .collect(),
            ..Mesh::default()
        });
    }
    for mesh in invalid_meshes {
        let error = mesh.device_count().unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                ..
            })
        ));
        assert!(mesh.device_at(0).is_err());
        assert!(mesh.device_assignment_equal(&mesh).is_err());
        let named = NamedSharding {
            mesh,
            ..NamedSharding::default()
        };
        assert!(named.equal_semantics(&named).is_err());
        assert!(named.is_single_device().is_err());
        assert!(named.is_replicated().is_err());
        assert!(named.is_manual().is_err());
        assert!(named.is_unreduced().is_err());
    }
    for (pre_size, size) in [(0, 2), (1, 0), (i64::MAX, 2), (i64::MIN, -1)] {
        let bad = AxisRef {
            mesh_axis_index: 0,
            sub_axis_info: vec![SubAxis { pre_size, size }],
        };
        assert!(bad.can_merge(&bad).is_err());
        assert!(bad.can_coexist_without_overlap(&bad).is_err());
        assert!(bad.size(&mesh(&[8])).is_err());
    }
    assert_eq!(mesh(&[2, 4]).device_count().unwrap(), 8);
    assert!(
        axis(0, Some((1, 2)))
            .can_merge(&axis(0, Some((2, 2))))
            .unwrap()
    );
}

#[test]
fn semantic_equality_rejects_excessively_nested_raw_shardings() {
    let mut sharding = Sharding::default();
    for _ in 0..100 {
        sharding = Sharding {
            kind: T::Tuple,
            tuple_shardings: vec![sharding],
            ..Sharding::default()
        };
    }
    let error = sharding.equal_semantics(&sharding).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "sharding",
            ..
        })
    ));
}

fn tiled(dimensions: &[i64], devices: &[i64], types: &[T]) -> Sharding {
    Sharding {
        kind: T::Other,
        tile_assignment_dimensions: dimensions.into(),
        tile_assignment_devices: devices.into(),
        last_tile_dims: types.into(),
        ..Sharding::default()
    }
}
fn mesh(sizes: &[i64]) -> Mesh {
    Mesh::new(
        sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| MeshAxis {
                name: format!("a{i}"),
                size,
            })
            .collect(),
        vec![],
        vec![],
    )
    .unwrap()
}

#[test]
fn mesh_group_materialization_preserves_compact_and_explicit_device_partitions() {
    let compact = Mesh::new(
        mesh(&[2, 3]).axes,
        vec![],
        vec![IotaTransform {
            reshape_dims: vec![3, 2],
            transpose_perm: vec![1, 0],
        }],
    )
    .unwrap();
    let explicit = Mesh::new(
        compact.axes.clone(),
        (0..6)
            .map(|index| compact.device_at(index).unwrap())
            .collect(),
        vec![],
    )
    .unwrap();
    for axes in [
        vec![],
        vec![axis(0, None)],
        vec![axis(1, None)],
        vec![axis(1, None), axis(0, None)],
    ] {
        let a = MeshAxesReplicaGroupList::new(compact.clone(), axes.clone()).unwrap();
        let b = MeshAxesReplicaGroupList::new(explicit.clone(), axes).unwrap();
        let materialized = a.replica_groups(6).unwrap();
        assert_eq!(materialized, b.replica_groups(6).unwrap());
        let mut devices = Vec::new();
        for (group, members) in materialized.0.iter().enumerate() {
            for (member, &device) in members.0.iter().enumerate() {
                assert_eq!(a.device_at(group as i64, member as i64).unwrap(), device);
                assert_eq!(b.device_at(group as i64, member as i64).unwrap(), device);
                devices.push(device);
            }
        }
        devices.sort_unstable();
        assert_eq!(devices, [0, 1, 2, 3, 4, 5]);
        assert!(a.replica_groups(5).is_err());
    }
}

#[test]
fn large_explicit_mesh_materializes_without_per_device_assignment_validation() {
    let count = 32_768;
    let assignment: Vec<_> = (0..count).rev().collect();
    let groups = MeshAxesReplicaGroupList::new(
        Mesh::new(mesh(&[count]).axes, assignment.clone(), vec![]).unwrap(),
        vec![axis(0, None)],
    )
    .unwrap();
    let materialized = groups.replica_groups(count as usize).unwrap();
    assert_eq!(materialized.0.len(), 1);
    assert_eq!(materialized.0[0].0, assignment);
    assert!(groups.replica_groups(count as usize - 1).is_err());

    let empty = MeshAxesReplicaGroupList::new(Mesh::default(), vec![]).unwrap();
    assert!(empty.replica_groups(0).unwrap().0.is_empty());
    let mut malformed = groups;
    malformed.mesh.device_ids[1] = malformed.mesh.device_ids[0];
    assert!(malformed.replica_groups(count as usize).is_err());
}

fn axis(index: i64, sub: Option<(i64, i64)>) -> AxisRef {
    AxisRef::new(
        index,
        sub.map(|(pre_size, size)| SubAxis { pre_size, size })
            .into_iter()
            .collect(),
    )
    .unwrap()
}
fn equal(ctx: &Context, a: Sharding, b: Sharding) -> bool {
    a.verify(ctx).unwrap();
    b.verify(ctx).unwrap();
    a.equal_semantics(&b).unwrap()
}

#[test]
fn leaf_modes_normalize_unused_assignments_and_tuple_import() {
    let ctx = Context::new();
    let array = Shape::array(&ctx, P::F32, vec![2]).unwrap();
    let tuple = Shape::tuple(&ctx, vec![array, Shape::token(&ctx)]).unwrap();
    for kind in [T::Replicated, T::Manual, T::Unknown, T::Unreduced] {
        let sharding = Sharding {
            kind,
            tile_assignment_dimensions: vec![-1],
            tile_assignment_devices: vec![-7, 300],
            ..Sharding::default()
        };
        for shape in [
            array,
            tuple,
            Shape::tuple(&ctx, vec![]).unwrap(),
            Shape::opaque(&ctx),
        ] {
            sharding.validate(&ctx, shape, Some(4)).unwrap();
        }
    }
    tiled(&[], &[-9], &[])
        .validate(&ctx, array, Some(4))
        .unwrap();
    Sharding::maximal(3).validate(&ctx, tuple, Some(4)).unwrap();
    assert!(Sharding::maximal(3).validate(&ctx, array, Some(3)).is_err());
    assert!(
        Sharding {
            kind: T::Maximal,
            ..Sharding::default()
        }
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn explicit_assignments_require_unique_in_range_devices_and_exact_data_rank() {
    let ctx = Context::new();
    let shape = Shape::array(&ctx, P::F32, vec![3, 7]).unwrap();
    let one = Shape::array(&ctx, P::F32, vec![3]).unwrap();
    let sharding = tiled(&[2, 2], &[0, 2, 1, 3], &[]);
    sharding.validate(&ctx, shape, Some(4)).unwrap();
    sharding
        .validate(&ctx, Shape::buffer(&ctx, shape).unwrap(), Some(4))
        .unwrap();
    assert!(sharding.validate(&ctx, one, None).is_err());
    assert!(sharding.validate(&ctx, shape, Some(8)).is_err());
    tiled(&[2], &[7, 19], &[])
        .validate(&ctx, one, None)
        .unwrap();
    for devices in [&[7, 19][..], &[1, 1], &[-1, 1]] {
        assert!(
            tiled(&[2], devices, &[])
                .validate(&ctx, one, Some(2))
                .is_err()
        );
    }
    tiled(&[2], &[-1, -1], &[])
        .validate(&ctx, Shape::token(&ctx), Some(2))
        .unwrap();
    for bad in [
        tiled(&[2], &[0, 1, 2], &[]),
        tiled(&[0, 2], &[0, 1], &[]),
        tiled(&[i64::MAX, 2], &[0, 1], &[]),
    ] {
        assert!(bad.verify(&ctx).is_err());
    }
}

#[test]
fn compact_iota_assignments_are_checked_without_materialization() {
    let ctx = Context::new();
    let shape = Shape::array(&ctx, P::F32, vec![3, 9]).unwrap();
    let sharding = Sharding {
        iota_reshape_dims: vec![3, 2],
        iota_transpose_perm: vec![1, 0],
        ..tiled(&[2, 3], &[], &[])
    };
    sharding.validate(&ctx, shape, Some(6)).unwrap();
    let huge = Sharding {
        tile_assignment_dimensions: vec![1_000_000_000, 2],
        iota_reshape_dims: vec![2, 1_000_000_000],
        ..sharding.clone()
    };
    huge.validate(&ctx, shape, Some(2_000_000_000)).unwrap();
    for permutation in [vec![0, 0], vec![-1, 0], vec![0, 2], vec![0]] {
        assert!(
            Sharding {
                iota_transpose_perm: permutation,
                ..sharding.clone()
            }
            .verify(&ctx)
            .is_err()
        );
    }
    assert!(
        Sharding {
            iota_reshape_dims: vec![2, 2],
            ..sharding.clone()
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        Sharding {
            tile_assignment_devices: vec![0],
            ..sharding
        }
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn partial_replication_sorts_devices_and_removes_unit_subgroups() {
    let ctx = Context::new();
    let legacy = Sharding {
        replicate_on_last_tile_dim: true,
        ..tiled(&[2, 2], &[1, 0, 3, 2], &[])
    };
    assert!(equal(
        &ctx,
        legacy.clone(),
        tiled(&[2, 2], &[0, 1, 2, 3], &[T::Replicated])
    ));
    assert!(!equal(
        &ctx,
        legacy,
        tiled(&[2, 2], &[0, 2, 1, 3], &[T::Replicated])
    ));
    let legacy = Sharding {
        replicate_on_last_tile_dim: true,
        ..tiled(&[2, 1], &[1, 0], &[])
    };
    assert!(equal(&ctx, legacy, tiled(&[2], &[1, 0], &[])));
    assert!(equal(
        &ctx,
        tiled(&[2, 1, 1], &[1, 0], &[T::Manual, T::Unknown]),
        tiled(&[2], &[1, 0], &[])
    ));
}

#[test]
fn subgroup_merging_and_sorting_transpose_assignment_axes() {
    let ctx = Context::new();
    let left = tiled(
        &[2, 2, 2],
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[T::Replicated, T::Manual],
    );
    let right = tiled(
        &[2, 2, 2],
        &[0, 2, 1, 3, 4, 6, 5, 7],
        &[T::Manual, T::Replicated],
    );
    assert!(equal(&ctx, left.clone(), right));
    assert!(!equal(
        &ctx,
        left,
        tiled(
            &[2, 2, 2],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[T::Manual, T::Replicated]
        )
    ));
    assert!(equal(
        &ctx,
        tiled(
            &[2, 2, 2],
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[T::Manual, T::Manual]
        ),
        tiled(&[2, 4], &[0, 1, 2, 3, 4, 5, 6, 7], &[T::Manual])
    ));
}

#[test]
fn compact_assignments_compare_with_explicit_and_equivalent_large_transforms() {
    let ctx = Context::new();
    let compact = Sharding {
        iota_reshape_dims: vec![3, 2],
        iota_transpose_perm: vec![1, 0],
        ..tiled(&[2, 3], &[], &[])
    };
    assert!(equal(
        &ctx,
        compact,
        tiled(&[2, 3], &[0, 2, 4, 1, 3, 5], &[])
    ));
    let left = Sharding {
        iota_reshape_dims: vec![2_000_000_000],
        iota_transpose_perm: vec![0],
        ..tiled(&[2_000_000_000], &[], &[])
    };
    let right = Sharding {
        iota_reshape_dims: vec![2, 1_000_000_000],
        iota_transpose_perm: vec![0, 1],
        ..left.clone()
    };
    assert!(equal(&ctx, left.clone(), right.clone()));
    assert!(!equal(
        &ctx,
        left,
        Sharding {
            iota_transpose_perm: vec![1, 0],
            ..right
        }
    ));
}

#[test]
fn tuple_leaf_cardinality_empty_tuples_and_metadata_follow_import_rules() {
    let ctx = Context::new();
    let a = Shape::scalar(&ctx, P::F32).unwrap();
    let tuple = Shape::tuple(
        &ctx,
        vec![a, Shape::tuple(&ctx, vec![a, Shape::token(&ctx)]).unwrap()],
    )
    .unwrap();
    Sharding::tuple(vec![Sharding::default(); 3])
        .validate(&ctx, tuple, None)
        .unwrap();
    assert!(
        Sharding::tuple(vec![Sharding::default(); 2])
            .validate(&ctx, tuple, None)
            .is_err()
    );
    let empty = Shape::tuple(&ctx, vec![]).unwrap();
    for count in [0, 1] {
        Sharding::tuple(vec![Sharding::default(); count])
            .validate(&ctx, empty, None)
            .unwrap();
    }
    assert!(
        Sharding::tuple(vec![Sharding::default(); 2])
            .validate(&ctx, empty, None)
            .is_err()
    );
    assert_eq!(
        Sharding::default()
            .normalize_tuple(&ctx, tuple)
            .unwrap()
            .tuple_shardings
            .len(),
        3
    );
    assert_eq!(
        Sharding::default()
            .normalize_tuple(&ctx, empty)
            .unwrap()
            .tuple_shardings
            .len(),
        1
    );
}

#[test]
fn mesh_assignments_validate_and_query_transposed_iota_without_allocating_devices() {
    let mut m = mesh(&[2, 3]);
    m.iota_transform = vec![IotaTransform {
        reshape_dims: vec![3, 2],
        transpose_perm: vec![1, 0],
    }];
    assert_eq!(
        (0..6).map(|i| m.device_at(i).unwrap()).collect::<Vec<_>>(),
        [0, 2, 4, 1, 3, 5]
    );
    assert!(m.device_at(-1).is_err());
    assert!(m.device_at(6).is_err());
    m.device_ids = vec![0, 2, 4, 1, 3, 5];
    assert!(m.validate().is_err());
    m.iota_transform.clear();
    m.validate().unwrap();
    m.device_ids = vec![0, 1, 2, 3, 4, 5];
    assert!(m.validate().is_err());
    m.device_ids = vec![0, 2, 4, 1, 3, 3];
    assert!(m.validate().is_err());
    m.device_ids = vec![-1, 2, 4, 1, 3, 5];
    assert!(m.validate().is_err());
    assert_eq!(
        mesh(&[1_000_000_000, 2]).device_at(1_999_999_999).unwrap(),
        1_999_999_999
    );
    let mut duplicate = mesh(&[2, 3]);
    duplicate.axes[1].name = "a0".into();
    assert!(duplicate.validate().is_err());
    assert_eq!(Mesh::default().device_count().unwrap(), 0);
}

#[test]
fn named_axes_require_valid_subdivisions_and_global_nonoverlap() {
    let m = mesh(&[24, 2]);
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
            AxisRef::new(
                0,
                vec![SubAxis {
                    pre_size: pre,
                    size
                }]
            )
            .and_then(|a| a.validate(&m))
            .is_ok(),
            valid
        );
    }
    let a = axis(0, Some((1, 2)));
    let b = axis(0, Some((2, 3)));
    let c = axis(0, Some((3, 2)));
    assert!(a.can_merge(&b).unwrap());
    assert!(a.can_coexist_without_overlap(&b).unwrap());
    assert!(!a.can_coexist_without_overlap(&c).unwrap());
    assert!(!a.can_coexist_without_overlap(&a).unwrap());
    assert!(!axis(0, None).can_coexist_without_overlap(&a).unwrap());
    assert!(a.can_coexist_without_overlap(&axis(1, None)).unwrap());
    assert!(AxisRef::new(-1, vec![]).is_err());
    assert!(axis(2, None).validate(&m).is_err());
}

#[test]
fn dimension_axes_need_not_be_sorted_but_other_axes_must_be() {
    let mut n = NamedSharding {
        mesh: mesh(&[4, 2]),
        dim_shardings: vec![DimensionSharding {
            axes: vec![axis(1, None), axis(0, None)],
            is_closed: true,
        }],
        ..NamedSharding::default()
    };
    n.validate().unwrap();
    n.dim_shardings.clear();
    n.replicated_axes = vec![axis(1, None), axis(0, None)];
    assert!(n.validate().is_err());
    n.replicated_axes.reverse();
    n.validate().unwrap();
    n.manual_axes = vec![axis(0, None)];
    assert!(n.validate().is_err());
}

#[test]
fn named_sharding_compares_with_tile_assignment_and_preserves_shape_constraints() {
    let ctx = Context::new();
    let n = NamedSharding {
        mesh: mesh(&[4, 2]),
        dim_shardings: vec![DimensionSharding {
            axes: vec![axis(0, None)],
            is_closed: true,
        }],
        ..NamedSharding::default()
    };
    n.validate().unwrap();
    let shape = Shape::array(&ctx, P::F32, vec![3]).unwrap();
    let s = Sharding {
        named_sharding: vec![n.clone()],
        ..Sharding::default()
    };
    s.validate(&ctx, shape, Some(8)).unwrap();
    assert!(s.validate(&ctx, shape, Some(4)).is_err());
    assert!(
        s.validate(&ctx, Shape::array(&ctx, P::F32, vec![3, 7]).unwrap(), None)
            .is_err()
    );
    assert!(equal(&ctx, s, n.to_tile_assignment().unwrap()));
}

#[test]
fn axes_and_sub_axes_group_devices_in_written_order() {
    for (m, axes, expected) in [
        (
            mesh(&[2, 3]),
            vec![axis(0, None)],
            vec![vec![0, 3], vec![1, 4], vec![2, 5]],
        ),
        (
            mesh(&[2, 3]),
            vec![axis(1, None), axis(0, None)],
            vec![vec![0, 3, 1, 4, 2, 5]],
        ),
        (
            mesh(&[8]),
            vec![axis(0, Some((1, 2)))],
            vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        ),
        (
            mesh(&[16]),
            vec![axis(0, Some((8, 2))), axis(0, Some((1, 2)))],
            vec![
                vec![0, 8, 1, 9],
                vec![2, 10, 3, 11],
                vec![4, 12, 5, 13],
                vec![6, 14, 7, 15],
            ],
        ),
        (
            Mesh::new(vec![], vec![0], vec![]).unwrap(),
            vec![],
            vec![vec![0]],
        ),
    ] {
        let groups = MeshAxesReplicaGroupList::new(m, axes).unwrap();
        assert_eq!(groups.num_replica_groups().unwrap(), expected.len() as i64);
        assert_eq!(
            groups.num_devices_per_group().unwrap(),
            expected[0].len() as i64
        );
        assert_eq!(
            groups
                .replica_groups(100)
                .unwrap()
                .0
                .into_iter()
                .map(|g| g.0)
                .collect::<Vec<_>>(),
            expected
        );
        assert!(groups.replica_groups(0).is_err());
        assert!(groups.device_at(-1, 0).is_err());
        assert!(
            groups
                .device_at(0, groups.num_devices_per_group().unwrap())
                .is_err()
        );
    }
    let huge =
        MeshAxesReplicaGroupList::new(mesh(&[1_000_000_000, 2]), vec![axis(1, None)]).unwrap();
    assert_eq!(huge.num_replica_groups().unwrap(), 1_000_000_000);
    assert_eq!(huge.device_at(999_999_999, 1).unwrap(), 1_999_999_999);
    assert!(huge.replica_groups(1_000_000).is_err());
}
