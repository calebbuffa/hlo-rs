use xlahlo::{MeshAxesReplicaGroupList, text};

#[test]
fn axes_and_sub_axes_group_devices_in_written_order() {
    for (source, expected) in [
        (
            "mesh[\"x\"=2,\"y\"=3] {\"x\"}",
            vec![vec![0, 3], vec![1, 4], vec![2, 5]],
        ),
        (
            "mesh[\"x\"=2,\"y\"=3] {\"y\",\"x\"}",
            vec![vec![0, 3, 1, 4, 2, 5]],
        ),
        (
            "mesh[\"x\"=8] {\"x\":(1)2}",
            vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        ),
        (
            "mesh[\"x\"=16] {\"x\":(8)2,\"x\":(1)2}",
            vec![
                vec![0, 8, 1, 9],
                vec![2, 10, 3, 11],
                vec![4, 12, 5, 13],
                vec![6, 14, 7, 15],
            ],
        ),
        (
            "mesh[\"x\"=2,\"y\"=2], device_ids=(3,2,1,0) {\"x\"}",
            vec![vec![3, 1], vec![2, 0]],
        ),
        ("single_device[device_id=0] {}", vec![vec![0]]),
        ("mesh[\"x\"=2] {}", vec![vec![0], vec![1]]),
    ] {
        let list = text::parse_collective_device_list(source).unwrap();
        let text::CollectiveDeviceList::MeshAxesReplicaGroupList(wire) = list.clone() else {
            panic!()
        };
        let groups = MeshAxesReplicaGroupList::from_proto(wire).unwrap();
        assert_eq!(groups.num_replica_groups(), expected.len() as i64);
        assert_eq!(groups.num_devices_per_group(), expected[0].len() as i64);
        let materialized = groups.replica_groups(100).unwrap();
        assert_eq!(
            materialized
                .iter()
                .map(|g| g.replica_ids.clone())
                .collect::<Vec<_>>(),
            expected,
            "{source}"
        );
        assert_eq!(text::parse_replica_groups(source).unwrap(), materialized);
        assert!(groups.replica_groups(0).is_err());
        assert!(groups.device_at(-1, 0).is_err());
        assert!(groups.device_at(0, groups.num_devices_per_group()).is_err());
        let printed = text::print_collective_device_list(&list).unwrap();
        assert_eq!(text::parse_collective_device_list(&printed).unwrap(), list);
    }
}

#[test]
fn invalid_mesh_axis_spans_return_errors_instead_of_native_assertions() {
    for source in [
        "mesh[\"x\"=4] {\"x\",\"x\"}",
        "mesh[\"x\"=4] {\"missing\"}",
        "mesh[\"x\"=8] {\"x\":(1)2,\"x\":(2)2}",
        "mesh[\"x\"=8] {\"x\":(1)4,\"x\":(2)2}",
        "mesh[\"x\"=8] {\"x\":(3)2}",
        "mesh[\"x\"=8] {\"x\":(1)8}",
        "mesh[\"x\"=8] {\"x\",?}",
    ] {
        assert!(
            text::parse_collective_device_list(source).is_err(),
            "{source}"
        );
    }
}

#[test]
fn mesh_groups_flow_through_module_text_binary_and_verification() {
    let source = "HloModule m ENTRY main {
 p = f32[2] parameter(0)
 ROOT g = f32[4] all-gather(p), dimensions={0}, replica_groups=mesh[\"x\"=2] {\"x\"}
}";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
    let imported =
        xlahlo::Module::from_proto(module.to_proto_with_program_shapes().unwrap()).unwrap();
    imported.verify(Default::default()).unwrap();
    let wrong =
        text::parse_module(&source.replace("f32[4] all-gather", "f32[6] all-gather")).unwrap();
    assert!(wrong.verify(Default::default()).is_err());
}

#[test]
fn large_meshes_remain_indexable_without_materialization() {
    let text::CollectiveDeviceList::MeshAxesReplicaGroupList(wire) =
        text::parse_collective_device_list("mesh[\"x\"=1000000000,\"y\"=2] {\"y\"}").unwrap()
    else {
        panic!()
    };
    let groups = MeshAxesReplicaGroupList::from_proto(wire).unwrap();
    assert_eq!(groups.num_replica_groups(), 1_000_000_000);
    assert_eq!(groups.device_at(999_999_999, 1).unwrap(), 1_999_999_999);
    assert!(groups.replica_groups(1_000_000).is_err());
}
