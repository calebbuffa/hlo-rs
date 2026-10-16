use stablehlo::{ArtifactVersionError, StableHloVersion, portable_artifact_version};

mod upstream_support;

fn header(version: &str) -> Vec<u8> {
    let mut bytes = b"ML\xefR\x0dStableHLO_v".to_vec();
    bytes.extend_from_slice(version.as_bytes());
    bytes.push(0);
    bytes
}

#[test]
fn discovery_returns_version_without_claiming_body_validity_or_compatibility() {
    for version in [
        "0.1.0",
        "0.9.0",
        "1.16.3",
        "1.22.0",
        "2.0.0",
        "9223372036854775807.0.0",
        "0001.022.00",
    ] {
        let bytes = header(version);
        assert_eq!(
            portable_artifact_version(&bytes).unwrap(),
            version.parse::<StableHloVersion>().unwrap()
        );
        let mut with_body = bytes;
        with_body.extend_from_slice(&[0xff, 0, 0x80, 0xcb]);
        assert_eq!(
            portable_artifact_version(&with_body).unwrap(),
            version.parse::<StableHloVersion>().unwrap()
        );
    }
}

#[test]
fn truncated_malformed_and_wrong_producer_headers_return_explicit_errors() {
    let valid = header("1.22.0");
    for length in 0..19 {
        assert_eq!(
            portable_artifact_version(&valid[..length]),
            Err(ArtifactVersionError::HeaderTooShort)
        );
    }
    let mut missing_terminator = valid.clone();
    missing_terminator.pop();
    assert_eq!(
        portable_artifact_version(&missing_terminator),
        Err(ArtifactVersionError::MissingProducerTerminator)
    );
    for index in 0..4 {
        let mut wrong_magic = valid.clone();
        wrong_magic[index] ^= 1;
        assert_eq!(
            portable_artifact_version(&wrong_magic),
            Err(ArtifactVersionError::InvalidMagic)
        );
    }
    let mut wrong_producer = valid.clone();
    wrong_producer[5] = b'x';
    assert_eq!(
        portable_artifact_version(&wrong_producer),
        Err(ArtifactVersionError::MissingStableHloProducer)
    );
    for version in ["1.2", "1.2.3suffix", "1.-2.0", "1.2.9223372036854775808"] {
        assert!(matches!(
            portable_artifact_version(&header(version)),
            Err(ArtifactVersionError::InvalidVersion(_))
        ));
    }
    let mut invalid_utf8 = valid;
    invalid_utf8[16] = 0xff;
    assert!(matches!(
        portable_artifact_version(&invalid_utf8),
        Err(ArtifactVersionError::InvalidVersion(_))
    ));
}

#[test]
fn discovery_matches_reference_header_scanning_including_its_permissive_prefix() {
    let mut bytes = b"ML\xefR\x0dprovider/StableHLO_v1.16.3\0".to_vec();
    assert_eq!(
        portable_artifact_version(&bytes).unwrap(),
        "1.16.3".parse::<StableHloVersion>().unwrap()
    );
    bytes[5] = 0;
    assert_eq!(
        portable_artifact_version(&bytes),
        Err(ArtifactVersionError::MissingStableHloProducer)
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn versions_are_discovered_in_independently_serialized_portable_artifacts() {
    use upstream_support::UpstreamResult;
    let versions = [
        "0.9.0", "0.10.0", "0.12.0", "0.14.0", "0.15.0", "1.0.0", "1.16.3", "1.20.0",
    ];
    let module = "module { func.func @add(%a:tensor<2xf32>,%b:tensor<2xf32>)->tensor<2xf32> { \
        %r=stablehlo.add %a,%b:tensor<2xf32> return %r:tensor<2xf32> } }";
    let requests = versions
        .iter()
        .map(|version| (version.to_string(), module.to_string()))
        .collect::<Vec<_>>();
    for (version, result) in versions.iter().zip(upstream_support::serialize(&requests)) {
        let hex = match result {
            UpstreamResult::Accepted(hex) => hex,
            UpstreamResult::Rejected(error) => {
                panic!("upstream serialization to {version} failed: {error}")
            }
        };
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        assert!(hex.len().is_multiple_of(2));
        assert_eq!(
            portable_artifact_version(&bytes).unwrap(),
            version.parse::<StableHloVersion>().unwrap()
        );
        let expected_bytecode = version
            .parse::<StableHloVersion>()
            .unwrap()
            .bytecode_version()
            .unwrap();
        assert_eq!(bytes[4] as u64, (expected_bytecode << 1) | 1);
    }
    // Exercise the original assembly-verification path through the same adapter.
    assert!(matches!(
        &upstream_support::verify(&[module.to_string()])[0],
        UpstreamResult::Accepted(_)
    ));
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn matching_native_artifacts_cover_current_and_historical_versions_and_roundtrip() {
    use upstream_support::UpstreamResult;
    let versions = [
        "0.9.0", "0.10.0", "0.12.0", "0.14.0", "0.15.0", "1.16.3", "1.20.0", "1.22.0",
    ];
    let module = "module { func.func @add(%a:tensor<2xf32>,%b:tensor<2xf32>)->tensor<2xf32> { \
        %r=stablehlo.add %a,%b:tensor<2xf32> return %r:tensor<2xf32> } }";
    let requests = versions
        .iter()
        .map(|version| (version.to_string(), module.to_string()))
        .collect::<Vec<_>>();
    let artifacts = upstream_support::serialize_native(&requests)
        .into_iter()
        .zip(versions)
        .map(|(result, version)| {
            let hex = match result {
                UpstreamResult::Accepted(hex) => hex,
                UpstreamResult::Rejected(error) => panic!("{version}: {error}"),
            };
            assert!(hex.len().is_multiple_of(2));
            let bytes = hex
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect::<Vec<_>>();
            let expected: StableHloVersion = version.parse().unwrap();
            assert_eq!(portable_artifact_version(&bytes).unwrap(), expected);
            assert_eq!(
                bytes[4] as u64,
                (expected.bytecode_version().unwrap() << 1) | 1
            );
            bytes
        })
        .collect::<Vec<_>>();
    for (version, result) in versions
        .iter()
        .zip(upstream_support::deserialize_native(&artifacts))
    {
        match result {
            UpstreamResult::Accepted(text) => {
                stablehlo::text::parse_module(&text)
                    .unwrap_or_else(|error| panic!("{version}: {error}\n{text}"));
            }
            UpstreamResult::Rejected(error) => panic!("{version}: {error}"),
        }
    }
    let invalid = [
        ("0.8.0".to_owned(), module.to_owned()),
        ("1.23.0".to_owned(), module.to_owned()),
    ];
    for result in upstream_support::serialize_native(&invalid) {
        assert!(matches!(result, UpstreamResult::Rejected(_)));
    }
    for result in upstream_support::deserialize_native(&[b"not bytecode".to_vec()]) {
        assert!(matches!(result, UpstreamResult::Rejected(_)));
    }
}
