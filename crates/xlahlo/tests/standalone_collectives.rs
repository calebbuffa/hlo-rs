use xlahlo::{proto::xla_data::ReplicaGroup, text};

#[test]
fn explicit_groups_preserve_order_empty_groups_and_duplicates() {
    for source in [
        "{}",
        "{{}}",
        "{{0,1},{2,3}}",
        "{{-1,4,4},{7}}",
        "{{18446744073709551615}}",
    ] {
        let groups = text::parse_replica_groups(source).unwrap();
        let printed = text::print_replica_groups(&groups).unwrap();
        assert_eq!(text::parse_replica_groups(&printed).unwrap(), groups);
        let compact = text::parse_collective_device_list(source).unwrap();
        assert_eq!(
            text::parse_collective_device_list(
                &text::print_collective_device_list(&compact).unwrap()
            )
            .unwrap(),
            compact
        );
    }
}

#[test]
fn compact_lists_stay_compact_or_materialize_in_transposed_order() {
    let source = "[2,3]<=[3,2]T(1,0)";
    let compact = text::parse_collective_device_list(source).unwrap();
    assert!(matches!(
        compact,
        text::CollectiveDeviceList::IotaCollectiveDeviceList(_)
    ));
    let groups = text::parse_replica_groups(source).unwrap();
    assert_eq!(
        groups
            .iter()
            .map(|g| g.replica_ids.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 2, 4], vec![1, 3, 5]]
    );
    assert_eq!(
        text::parse_collective_device_list(&text::print_collective_device_list(&compact).unwrap())
            .unwrap(),
        compact
    );
    assert!(text::parse_collective_device_list("[1000001,1]<=[1000001]").is_ok());
    assert!(text::parse_replica_groups("[1000001,1]<=[1000001]").is_err());
}

#[test]
fn incomplete_invalid_or_trailing_group_syntax_is_rejected() {
    for source in [
        "",
        "{0,1}",
        "{{1,}}",
        "{{1}} extra",
        "[2,3]<=[5]",
        "[2,3]<=[3,2]T(0,0)",
        "[2,3]<=[3,2]",
        "[2,3]<=[6]T(0)",
    ] {
        assert!(
            text::parse_collective_device_list(source).is_err(),
            "{source}"
        );
        assert!(text::parse_replica_groups(source).is_err(), "{source}");
    }
}

#[test]
fn unrepresentable_group_fields_are_not_silently_dropped() {
    let mut group = ReplicaGroup::new();
    group
        .special_fields
        .mut_unknown_fields()
        .add_varint(1234, 1);
    assert!(text::print_replica_groups(&[group]).is_err());
}

#[test]
fn module_printing_uses_native_single_axis_iota_syntax() {
    let source = "HloModule m ENTRY main {
 p = f32[2] parameter(0)
 ROOT g = f32[2] all-gather(p), dimensions={0}, replica_groups=[1,1]<=[1]
}";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let printed = text::print_module(&module).unwrap();
    assert!(printed.contains("[1,1]<=[1]"));
    assert!(!printed.contains("T(0)"));
    text::parse_module(&printed)
        .unwrap()
        .verify(Default::default())
        .unwrap();
}
