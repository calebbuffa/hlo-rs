use stablehlo::bytecode::{
    BytecodeContainer, BytecodeErrorKind as Kind, SectionId, decode_signed_varint, decode_varint,
    encode_signed_varint, encode_varint,
};

mod upstream_support;

fn encoded(value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    encode_varint(value, &mut bytes);
    bytes
}

fn section(id: u8, data: &[u8]) -> Vec<u8> {
    let mut bytes = vec![id];
    encode_varint(data.len() as u64, &mut bytes);
    bytes.extend_from_slice(data);
    bytes
}

fn container(version: u64, sections: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut bytes = b"ML\xefR".to_vec();
    encode_varint(version, &mut bytes);
    bytes.extend_from_slice(b"Rust-test\0");
    for (id, body) in sections {
        bytes.extend(section(*id, body));
    }
    bytes
}

// These are framing/table fixtures, not semantically valid MLIR modules.
fn empty_sections(version: u64) -> Vec<(u8, Vec<u8>)> {
    let mut sections = vec![
        (0, vec![1]),
        (1, if version >= 4 { vec![1, 1] } else { vec![1] }),
        (2, vec![]),
        (3, vec![1, 1]),
        (4, vec![1]),
    ];
    if version >= 5 {
        sections.push((8, vec![1]));
    }
    sections
}

#[test]
fn unsigned_prefix_varints_match_exact_reference_format_vectors() {
    for (value, bytes) in [
        (0, vec![0x01]),
        (1, vec![0x03]),
        (63, vec![0x7f]),
        (127, vec![0xff]),
        (128, vec![0x02, 0x02]),
        (16383, vec![0xfe, 0xff]),
        (16384, vec![0x04, 0x00, 0x02]),
        (1 << 21, vec![0x08, 0x00, 0x00, 0x02]),
        (
            (1 << 56) - 1,
            vec![0x80, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ),
        (1 << 56, vec![0, 0, 0, 0, 0, 0, 0, 0, 1]),
        (
            u64::MAX,
            vec![0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ),
    ] {
        assert_eq!(encoded(value), bytes, "{value}");
        let mut with_suffix = bytes.clone();
        with_suffix.extend([0xee, 0xdd]);
        assert_eq!(decode_varint(&with_suffix).unwrap(), (value, bytes.len()));
        for length in 0..bytes.len() {
            let error = decode_varint(&bytes[..length]).unwrap_err();
            assert_eq!(error.kind, Kind::UnexpectedEnd);
            assert_eq!(error.offset, length);
        }
    }
}

#[test]
fn unsigned_encodings_cover_every_width_boundary_and_generated_bit_pattern() {
    let mut witnesses = vec![0, u64::MAX];
    for bit in 0..64 {
        let value = 1_u64 << bit;
        witnesses.extend([value - 1, value, value.saturating_add(1)]);
    }
    let mut state = 0x1234_5678_9abc_def0_u64;
    for _ in 0..10_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        witnesses.push(state);
    }
    for value in witnesses {
        let bytes = encoded(value);
        let expected_length = ((64 - value.leading_zeros()).max(1).div_ceil(7) as usize).min(9);
        assert_eq!(bytes.len(), expected_length);
        assert_eq!(decode_varint(&bytes).unwrap(), (value, bytes.len()));
    }
}

#[test]
fn signed_varints_preserve_all_sign_and_width_extremes() {
    let mut witnesses = vec![i64::MIN, i64::MAX, -1, 0, 1];
    for bit in 0..63 {
        let value = 1_i64 << bit;
        witnesses.extend([value - 1, value, -value, -(value - 1)]);
    }
    for value in witnesses {
        let mut bytes = Vec::new();
        encode_signed_varint(value, &mut bytes);
        assert_eq!(decode_signed_varint(&bytes).unwrap(), (value, bytes.len()));
    }
    for (value, bytes) in [
        (0, vec![1]),
        (-1, vec![3]),
        (1, vec![5]),
        (-64, vec![0xff]),
        (64, vec![2, 2]),
    ] {
        let mut actual = Vec::new();
        encode_signed_varint(value, &mut actual);
        assert_eq!(actual, bytes);
    }
}

#[test]
fn reader_accepts_nonminimal_varints_like_the_reference() {
    assert_eq!(decode_varint(&[2, 0]).unwrap(), (0, 2));
    assert_eq!(decode_varint(&[4, 0, 0]).unwrap(), (0, 3));
    assert_eq!(decode_varint(&[0, 1, 0, 0, 0, 0, 0, 0, 0]).unwrap(), (1, 9));
    let original = container(6, &empty_sections(6));
    let mut bytes = original[..4].to_vec();
    bytes.extend([0, 6, 0, 0, 0, 0, 0, 0, 0]);
    bytes.extend_from_slice(&original[5..]);
    assert_eq!(BytecodeContainer::parse(&bytes).unwrap().version(), 6);
}

#[test]
fn container_reads_all_format_versions_and_required_section_transitions() {
    for version in 0..=6 {
        let bytes = container(version, &empty_sections(version));
        let parsed = BytecodeContainer::parse(&bytes).unwrap();
        assert_eq!(parsed.version(), version);
        assert_eq!(parsed.producer(), b"Rust-test");
        assert!(parsed.strings().is_empty());
        assert!(parsed.dialects().is_empty());
        assert!(parsed.operation_names().is_empty());
        assert!(parsed.attributes().is_empty());
        assert!(parsed.types().is_empty());
        assert_eq!(parsed.sections().len(), if version >= 5 { 6 } else { 5 });
        for section in parsed.sections() {
            assert_eq!(
                section.data,
                &bytes[section.offset..section.offset + section.data.len()]
            );
        }
        for id in [0, 1, 2, 3, 4, 8] {
            if id == 8 && version < 5 {
                continue;
            }
            let missing = empty_sections(version)
                .into_iter()
                .filter(|(current, _)| *current != id)
                .collect::<Vec<_>>();
            assert!(matches!(
                BytecodeContainer::parse(&container(version, &missing))
                    .unwrap_err()
                    .kind,
                Kind::MissingSection(_)
            ));
        }
    }
}

#[test]
fn malformed_headers_sections_lengths_duplicates_and_alignment_reject() {
    let valid = container(6, &empty_sections(6));
    for length in 0..valid.len() {
        assert!(
            BytecodeContainer::parse(&valid[..length]).is_err(),
            "{length}"
        );
    }
    let mut wrong_magic = valid.clone();
    wrong_magic[0] = b'X';
    assert_eq!(
        BytecodeContainer::parse(&wrong_magic).unwrap_err().kind,
        Kind::InvalidMagic
    );
    assert_eq!(
        BytecodeContainer::parse(b"ML\xefR\x0dproducer")
            .unwrap_err()
            .kind,
        Kind::MissingTerminator
    );
    for version in [7, 128, u64::MAX] {
        assert_eq!(
            BytecodeContainer::parse(&container(version, &[]))
                .unwrap_err()
                .kind,
            Kind::UnsupportedVersion(version)
        );
    }
    for id in 9..=127 {
        let mut bytes = valid.clone();
        bytes.extend([id, 1]);
        assert_eq!(
            BytecodeContainer::parse(&bytes).unwrap_err().kind,
            Kind::InvalidSection(id)
        );
    }
    for id in [0, 1, 2, 3, 4, 8] {
        let mut bytes = valid.clone();
        bytes.extend(section(id, &[]));
        assert!(matches!(
            BytecodeContainer::parse(&bytes).unwrap_err().kind,
            Kind::DuplicateSection(_)
        ));
    }
    for alignment in [0, 3, 5, 1 << 32, u64::MAX] {
        let mut bytes = valid.clone();
        bytes.extend([0x85, 1]);
        encode_varint(alignment, &mut bytes);
        assert_eq!(
            BytecodeContainer::parse(&bytes).unwrap_err().kind,
            Kind::InvalidAlignment(alignment)
        );
    }
    let mut enormous = valid;
    enormous.push(5);
    enormous.extend(encoded(u64::MAX));
    assert!(BytecodeContainer::parse(&enormous).is_err());
}

#[test]
fn aligned_sections_are_checked_by_file_offset_not_allocation_address() {
    for alignment in [1, 2, 4, 8, 16, 32, 64, 128, 4096] {
        let mut bytes = container(6, &empty_sections(6));
        bytes.push(0x85);
        bytes.extend(encoded(3));
        bytes.extend(encoded(alignment));
        let padding_start = bytes.len();
        while !(bytes.len() as u64).is_multiple_of(alignment) {
            bytes.push(0xcb);
        }
        let offset = bytes.len();
        bytes.extend([0xff, 0x00, 0xee]);
        let parsed = BytecodeContainer::parse(&bytes).unwrap();
        let resource = parsed.section(SectionId::Resources).unwrap();
        assert_eq!(resource.offset, offset);
        assert_eq!(resource.alignment, Some(alignment as u32));
        assert_eq!(resource.data, [0xff, 0, 0xee]);
        let mut shifted = vec![0x11];
        shifted.extend_from_slice(&bytes);
        assert_eq!(BytecodeContainer::parse(&shifted[1..]).unwrap(), parsed);
        if padding_start < offset {
            bytes[padding_start] = 0xcc;
            assert_eq!(
                BytecodeContainer::parse(&bytes).unwrap_err().kind,
                Kind::InvalidPadding(0xcc)
            );
        }
    }
}

fn string_table(strings: &[&[u8]]) -> Vec<u8> {
    let mut bytes = encoded(strings.len() as u64);
    for string in strings.iter().rev() {
        bytes.extend(encoded(string.len() as u64 + 1));
    }
    for string in strings {
        bytes.extend_from_slice(string);
        bytes.push(0);
    }
    bytes
}

#[test]
fn strings_preserve_empty_non_utf8_and_embedded_nul_bytes() {
    let strings: &[&[u8]] = &[b"", b"abc", &[0xff, 0, 0xfe], b"z"];
    let mut sections = empty_sections(6);
    sections[0].1 = string_table(strings);
    let bytes = container(6, &sections);
    assert_eq!(BytecodeContainer::parse(&bytes).unwrap().strings(), strings);
    for data in [
        vec![3, 1],
        vec![3, 3, b'x'],
        vec![3, 5, 0],
        vec![1, 0],
        encoded(u64::MAX),
    ] {
        sections[0].1 = data;
        assert!(BytecodeContainer::parse(&container(6, &sections)).is_err());
    }
}

#[test]
fn dialect_names_versions_registration_and_group_order_follow_format_versions() {
    for version in 0..=6 {
        let mut sections = empty_sections(version);
        sections[0].1 = string_table(&[b"builtin", b"module", b"stablehlo", b"add", b"return"]);
        let mut dialects = encoded(2);
        if version == 0 {
            dialects.extend([1, 5]);
        } else {
            dialects.extend(encoded(0));
            dialects.extend(encoded((2 << 1) | 1));
            dialects.extend(section(7, &[0x22, 0x44]));
        }
        if version >= 4 {
            dialects.extend(encoded(3));
        }
        for (dialect, names) in [(1, vec![3, 4]), (0, vec![1])] {
            dialects.extend(encoded(dialect));
            dialects.extend(encoded(names.len() as u64));
            for name in names {
                dialects.extend(encoded(if version >= 5 { (name << 1) | 1 } else { name }));
            }
        }
        sections[1].1 = dialects;
        let bytes = container(version, &sections);
        let parsed = BytecodeContainer::parse(&bytes).unwrap();
        assert_eq!(parsed.dialects()[0].name, b"builtin");
        assert_eq!(parsed.dialects()[1].name, b"stablehlo");
        if version > 0 {
            assert_eq!(
                parsed.dialects()[1].version.as_ref().unwrap().data,
                [0x22, 0x44]
            );
        } else {
            assert!(parsed.dialects()[1].version.is_none());
        }
        let operations = parsed.operation_names();
        assert_eq!(
            operations.iter().map(|op| op.name).collect::<Vec<_>>(),
            [b"add".as_slice(), b"return", b"module"]
        );
        assert_eq!(
            operations.iter().map(|op| op.dialect).collect::<Vec<_>>(),
            [1, 1, 0]
        );
        assert_eq!(operations[0].registered, (version >= 5).then_some(true));
    }
}

#[test]
fn invalid_dialect_string_indexes_versions_counts_and_groups_reject() {
    let mut sections = empty_sections(6);
    sections[0].1 = string_table(&[b"builtin", b"module"]);
    for body in [
        vec![3, 9, 1],             // bad dialect string index
        vec![3, 3, 6, 1, 1],       // wrong nested version section
        vec![3, 1, 3],             // promises an operation but no group
        vec![3, 1, 3, 5, 3, 7],    // bad group dialect index
        vec![3, 1, 3, 1, 1],       // empty group
        vec![3, 1, 3, 1, 3, 0xff], // bad op string index
        vec![3, 1, 1, 1, 3, 7],    // more operations than promised
        encoded(u64::MAX),
    ] {
        sections[1].1 = body;
        assert!(BytecodeContainer::parse(&container(6, &sections)).is_err());
    }
}

#[test]
fn attribute_and_type_entries_preserve_dialect_groups_flags_and_absolute_offsets() {
    let mut sections = empty_sections(6);
    sections[0].1 = string_table(&[b"builtin", b"stablehlo"]);
    sections[1].1 = [encoded(2), encoded(0), encoded(2), encoded(0)].concat();
    sections[2].1 = b"unit\0\x03\xff".to_vec();
    sections[3].1 = [
        encoded(1),
        encoded(1),
        encoded(0),
        encoded(1),
        encoded(5 << 1),
        encoded(1),
        encoded(1),
        encoded((2 << 1) | 1),
    ]
    .concat();
    let bytes = container(6, &sections);
    let parsed = BytecodeContainer::parse(&bytes).unwrap();
    assert_eq!(parsed.attributes().len(), 1);
    assert_eq!(parsed.types().len(), 1);
    let attribute = &parsed.attributes()[0];
    let ty = &parsed.types()[0];
    assert_eq!(attribute.dialect, 0);
    assert!(!attribute.custom_encoding);
    assert_eq!(attribute.data, b"unit\0");
    assert_eq!(ty.dialect, 1);
    assert!(ty.custom_encoding);
    assert_eq!(ty.data, [3, 0xff]);
    assert_eq!(attribute.offset + attribute.data.len(), ty.offset);
    assert_eq!(&bytes[ty.offset..ty.offset + 2], ty.data);
    for offsets in [
        vec![3, 1, 1, 1],       // zero group
        vec![3, 1, 5, 3, 3],    // bad dialect
        vec![3, 1, 1, 5, 3, 3], // group larger than table
        vec![3, 1, 1, 3, 0xff], // entry longer than payload
        vec![1, 1],             // unreferenced payload
        vec![1, 1, 1],          // trailing offset data
    ] {
        sections[3].1 = offsets;
        assert!(BytecodeContainer::parse(&container(6, &sections)).is_err());
    }
}

#[test]
fn arbitrary_truncated_and_mutated_input_never_panics_or_allocates_by_untrusted_count() {
    let seeds = [
        container(0, &empty_sections(0)),
        container(6, &empty_sections(6)),
    ];
    for seed in seeds {
        for index in 0..seed.len() {
            for value in 0..=255 {
                let mut bytes = seed.clone();
                bytes[index] = value;
                let _ = BytecodeContainer::parse(&bytes);
            }
        }
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn native_historical_artifacts_have_checked_rust_containers_and_indexed_payloads() {
    use upstream_support::UpstreamResult;
    let versions = [
        "0.9.0", "0.10.0", "0.12.0", "0.14.0", "0.15.0", "1.16.3", "1.20.0", "1.22.0",
    ];
    let source = "module { func.func @main(%x:tensor<2xf32>)->tensor<2xf32> { \
        %y=stablehlo.add %x,%x:tensor<2xf32> return %y:tensor<2xf32> } }";
    let requests = versions
        .iter()
        .map(|version| (version.to_string(), source.to_owned()))
        .collect::<Vec<_>>();
    for (version, result) in versions
        .iter()
        .zip(upstream_support::serialize_native(&requests))
    {
        let hex = match result {
            UpstreamResult::Accepted(hex) => hex,
            UpstreamResult::Rejected(error) => panic!("{version}: {error}"),
        };
        let bytes = hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let parsed =
            BytecodeContainer::parse(&bytes).unwrap_or_else(|error| panic!("{version}: {error}"));
        assert_eq!(
            parsed.version(),
            version
                .parse::<stablehlo::StableHloVersion>()
                .unwrap()
                .bytecode_version()
                .unwrap()
        );
        assert_eq!(
            parsed.producer(),
            format!("StableHLO_v{version}").as_bytes()
        );
        assert!(
            parsed
                .dialects()
                .iter()
                .any(|dialect| dialect.name == b"vhlo")
        );
        assert!(
            parsed
                .operation_names()
                .iter()
                .any(|operation| operation.name == b"add_v1")
        );
        assert!(!parsed.attributes().is_empty());
        assert!(!parsed.types().is_empty());
        let mut wide_version = bytes[..4].to_vec();
        wide_version.push(0);
        wide_version.extend(parsed.version().to_le_bytes());
        wide_version.extend_from_slice(&bytes[5..]);
        // No section in this fixture has alignment requirements.
        assert!(
            parsed
                .sections()
                .iter()
                .all(|section| section.alignment.is_none())
        );
        assert_eq!(
            BytecodeContainer::parse(&wide_version)
                .unwrap()
                .operation_names(),
            parsed.operation_names()
        );
        assert!(matches!(
            &upstream_support::deserialize_native(&[wide_version])[0],
            UpstreamResult::Accepted(_)
        ));
    }
}

#[test]
fn property_table_boundaries_and_zero_length_entries_are_preserved() {
    let mut sections = empty_sections(6);
    sections[5].1 = vec![7, 1, 5, 0xaa, 0xbb, 3, 0xcc];
    let bytes = container(6, &sections);
    let parsed = BytecodeContainer::parse(&bytes).unwrap();
    assert_eq!(parsed.properties().len(), 3);
    assert_eq!(parsed.properties()[0].data, []);
    assert_eq!(parsed.properties()[1].data, [0xaa, 0xbb]);
    assert_eq!(parsed.properties()[2].data, [0xcc]);
    for property in parsed.properties() {
        assert_eq!(
            &bytes[property.offset..property.offset + property.data.len()],
            property.data
        );
    }
    for empty in [vec![], vec![1]] {
        sections[5].1 = empty;
        let bytes = container(6, &sections);
        assert!(
            BytecodeContainer::parse(&bytes)
                .unwrap()
                .properties()
                .is_empty()
        );
    }
}

#[test]
fn malformed_property_counts_lengths_and_trailing_bytes_reject() {
    let mut sections = empty_sections(6);
    for invalid in [
        vec![3],
        vec![3, 5, 0xaa],
        vec![1, 1],
        encoded(u64::MAX),
        [encoded(1), encoded(u64::MAX)].concat(),
    ] {
        sections[5].1 = invalid;
        assert!(BytecodeContainer::parse(&container(6, &sections)).is_err());
    }
}
