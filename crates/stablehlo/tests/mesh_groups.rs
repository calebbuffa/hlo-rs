use stablehlo::{
    Attribute, Attributes, AxisRef, ElementType, Mesh, MeshAxis, MeshReference,
    ReplicaGroupMeshAxes, StableHloAttribute, StableHloOp, SubAxisInfo, TensorType, Type,
};

mod common;
use common::computation;

fn mesh(sizes: &[(&str, i64)]) -> Mesh {
    Mesh {
        axes: sizes
            .iter()
            .map(|&(name, size)| MeshAxis {
                name: name.into(),
                size,
            })
            .collect(),
        device_ids: None,
    }
}

fn axis(name: &str, sub: Option<(i64, i64)>) -> AxisRef {
    AxisRef {
        name: name.into(),
        sub_axis_info: sub.map(|(pre_size, size)| SubAxisInfo { pre_size, size }),
    }
}

#[test]
fn full_axes_define_group_membership_and_reference_order() {
    let mesh = mesh(&[("x", 2), ("y", 3)]);
    assert_eq!(
        mesh.replica_groups(&[axis("y", None)]).unwrap(),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
    assert_eq!(
        mesh.replica_groups(&[axis("x", None)]).unwrap(),
        vec![vec![0, 3], vec![1, 4], vec![2, 5]]
    );
    assert_eq!(
        mesh.replica_groups(&[axis("y", None), axis("x", None)])
            .unwrap(),
        vec![vec![0, 3, 1, 4, 2, 5]]
    );
    assert_eq!(
        mesh.replica_groups(&[]).unwrap(),
        vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]
    );
}

#[test]
fn sub_axes_factor_mesh_dimensions_before_transposition() {
    let mesh = mesh(&[("x", 8)]);
    assert_eq!(
        mesh.replica_groups(&[axis("x", Some((1, 2)))]).unwrap(),
        vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]]
    );
    assert_eq!(
        mesh.replica_groups(&[axis("x", Some((2, 2)))]).unwrap(),
        vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    assert_eq!(
        mesh.replica_groups(&[axis("x", Some((2, 2))), axis("x", Some((4, 2)))])
            .unwrap(),
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]
    );
}

#[test]
fn explicit_device_ids_are_applied_after_coordinate_permutation() {
    let mut mesh = mesh(&[("x", 2), ("y", 2)]);
    mesh.device_ids = Some(vec![7, 2, 8, 4]);
    assert_eq!(
        mesh.replica_groups(&[axis("x", None)]).unwrap(),
        vec![vec![7, 8], vec![2, 4]]
    );
    mesh.device_ids = Some(vec![]);
    assert_eq!(
        mesh.replica_groups(&[axis("x", None)]).unwrap(),
        vec![vec![0, 2], vec![1, 3]]
    );
}

#[test]
fn invalid_meshes_sub_axes_and_overlaps_return_errors() {
    let mesh = mesh(&[("x", 8)]);
    for axes in [
        vec![axis("missing", None)],
        vec![axis("x", Some((0, 2)))],
        vec![axis("x", Some((1, 0)))],
        vec![axis("x", Some((3, 2)))],
        vec![axis("x", Some((4, 4)))],
        vec![axis("x", None), axis("x", None)],
        vec![axis("x", None), axis("x", Some((2, 2)))],
        vec![axis("x", Some((1, 4))), axis("x", Some((2, 4)))],
    ] {
        assert!(mesh.replica_groups(&axes).is_err(), "{axes:?}");
        assert!(mesh.replica_group_size(&axes).is_err(), "{axes:?}");
    }
    for ids in [
        vec![0, 1],
        vec![0, 1, 2, 3, 4, 5, 6, 6],
        vec![-1, 1, 2, 3, 4, 5, 6, 7],
    ] {
        let mut invalid = mesh.clone();
        invalid.device_ids = Some(ids);
        assert!(invalid.replica_groups(&[]).is_err());
    }
}

#[test]
fn scalar_unit_and_unrepresentable_meshes_have_defined_behavior() {
    assert_eq!(mesh(&[]).replica_groups(&[]).unwrap(), vec![vec![0]]);
    assert_eq!(
        mesh(&[("x", 1)])
            .replica_groups(&[axis("x", None)])
            .unwrap(),
        vec![vec![0]]
    );
    assert!(mesh(&[("x", 0)]).replica_groups(&[]).is_err());
    assert!(
        mesh(&[("x", i64::MAX), ("y", 2)])
            .replica_group_size(&[])
            .is_err()
    );
    assert!(mesh(&[("x", i64::MAX)]).replica_groups(&[]).is_err());
}

#[test]
fn inline_mesh_cardinality_constrains_collective_geometry() {
    for (output, accepted) in [(6, true), (9, false)] {
        let attributes: Attributes = [
            ("all_gather_dim".into(), Attribute::integer(0)),
            (
                "replica_groups".into(),
                Attribute::StableHlo(Box::new(StableHloAttribute::ReplicaGroupMeshAxes(
                    ReplicaGroupMeshAxes {
                        mesh: MeshReference::Inline(mesh(&[("x", 2), ("y", 2)])),
                        axes: vec![axis("x", None)],
                    },
                ))),
            ),
        ]
        .into();
        let ty = |size| Type::from(TensorType::static_shape(ElementType::F32, [size]));
        assert_eq!(
            computation(
                StableHloOp::AllGather,
                vec![ty(3)],
                vec![ty(output)],
                attributes
            )
            .is_ok(),
            accepted
        );
    }
    let attributes: Attributes = [
        ("split_dimension".into(), Attribute::integer(0)),
        ("concat_dimension".into(), Attribute::integer(0)),
        ("split_count".into(), Attribute::integer(2)),
        (
            "replica_groups".into(),
            Attribute::StableHlo(Box::new(StableHloAttribute::ReplicaGroupMeshAxes(
                ReplicaGroupMeshAxes {
                    mesh: MeshReference::Inline(mesh(&[("x", 4)])),
                    axes: vec![axis("x", None)],
                },
            ))),
        ),
    ]
    .into();
    let ty: Type = TensorType::static_shape(ElementType::F32, [8]).into();
    assert!(
        computation(
            StableHloOp::AllToAll,
            vec![ty.clone()],
            vec![ty],
            attributes
        )
        .is_err()
    );
}

#[test]
fn symbolic_axis_references_still_validate_local_sub_axis_metadata() {
    let attribute = StableHloAttribute::ReplicaGroupMeshAxes(ReplicaGroupMeshAxes {
        mesh: MeshReference::Symbol("mesh".into()),
        axes: vec![axis("x", Some((0, 2)))],
    });
    assert!(attribute.verify().is_err());
}
