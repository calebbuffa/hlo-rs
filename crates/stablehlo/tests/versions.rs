use stablehlo::{
    CompatibilityRequirement, STABLEHLO_CURRENT_VERSION, STABLEHLO_MINIMUM_VERSION,
    StableHloVersion as Version, VersionError,
};

fn version(source: &str) -> Version {
    source.parse().unwrap()
}

#[test]
fn versions_parse_normalize_and_roundtrip_all_numeric_boundaries() {
    assert_eq!(Version::CURRENT.to_string(), STABLEHLO_CURRENT_VERSION);
    assert_eq!(Version::MINIMUM.to_string(), STABLEHLO_MINIMUM_VERSION);
    for major in [0, 1, 9, 10, i64::MAX as u64] {
        for minor in [0, 22, i64::MAX as u64] {
            for patch in [0, 3, i64::MAX as u64] {
                let expected = Version::new(major, minor, patch).unwrap();
                assert_eq!(version(&expected.to_string()), expected);
                assert_eq!(
                    (expected.major(), expected.minor(), expected.patch()),
                    (major, minor, patch)
                );
                assert_eq!(version(&format!("000{major}.00{minor}.0{patch}")), expected);
            }
        }
    }
    assert_eq!(
        version(&format!("{}.0.0", "0".repeat(1000))),
        version("0.0.0")
    );
}

#[test]
fn malformed_or_out_of_range_versions_fail_without_panicking() {
    for input in [
        "",
        "1",
        "1.2",
        "1.2.3.4",
        "1..3",
        ".1.2",
        "1.2.",
        "-1.2.3",
        "+1.2.3",
        " 1.2.3",
        "1.2.3\n",
        "v1.2.3",
        "１.2.3",
        "1.2.0x3",
        "1.2.3-beta",
    ] {
        assert!(
            matches!(input.parse::<Version>(), Err(VersionError::InvalidSyntax)),
            "{input:?}"
        );
    }
    for input in [
        "9223372036854775808.0.0",
        "0.9223372036854775808.0",
        "0.0.18446744073709551616",
    ] {
        assert!(
            matches!(
                input.parse::<Version>(),
                Err(VersionError::ComponentOutOfRange)
            ),
            "{input}"
        );
    }
    for components in [(u64::MAX, 0, 0), (0, u64::MAX, 0), (0, 0, u64::MAX)] {
        assert!(Version::new(components.0, components.1, components.2).is_err());
    }
}

#[test]
fn numeric_order_is_lexicographic_not_string_order() {
    let ordered = [
        "0.9.0", "0.9.1", "0.10.0", "0.10.9", "0.12.0", "1.0.0", "1.2.0", "1.10.0", "1.22.0",
        "2.0.0",
    ];
    for (left_index, left) in ordered.iter().enumerate() {
        for (right_index, right) in ordered.iter().enumerate() {
            assert_eq!(
                version(left).cmp(&version(right)),
                left_index.cmp(&right_index)
            );
        }
    }
}

#[test]
fn compatibility_targets_match_the_pinned_reference_policy() {
    for (requirement, expected) in [
        (CompatibilityRequirement::None, "1.22.0"),
        (CompatibilityRequirement::FourWeeks, "1.18.0"),
        (CompatibilityRequirement::TwelveWeeks, "1.16.3"),
        (CompatibilityRequirement::Maximum, "0.9.0"),
    ] {
        assert_eq!(Version::for_compatibility(requirement), version(expected));
    }
}

#[test]
fn every_bytecode_policy_transition_and_compatibility_boundary_is_checked() {
    for (input, expected) in [
        ("0.9.0", 0),
        ("0.9.999", 0),
        ("0.10.0", 1),
        ("0.11.999", 1),
        ("0.12.0", 3),
        ("0.13.999", 3),
        ("0.14.0", 4),
        ("0.14.999", 4),
        ("0.15.0", 6),
        ("1.0.0", 6),
        ("1.16.3", 6),
        ("1.21.999", 6),
        ("1.22.0", 6),
    ] {
        assert_eq!(
            version(input).bytecode_version().unwrap(),
            expected,
            "{input}"
        );
    }
    for input in ["0.0.0", "0.8.999", "1.22.1", "1.23.0", "2.0.0"] {
        assert!(
            matches!(
                version(input).bytecode_version(),
                Err(VersionError::UnsupportedTarget { .. })
            ),
            "{input}"
        );
        assert!(version(input).target_opset().is_err(), "{input}");
    }
    for input in ["0.9.3", "0.14.999", "1.16.3", "1.21.999", "1.22.0"] {
        let target = version(input).target_opset().unwrap();
        assert_eq!(target.major(), version(input).major());
        assert_eq!(target.minor(), version(input).minor());
        assert_eq!(target.patch(), 0);
    }
}
