use stablehlo::{
    Attribute, ElementType, ScalarValue,
    bytecode::{BytecodeErrorKind as Kind, encode_signed_varint, encode_varint, read_module},
    text::{parse_attribute, parse_type},
};

fn unsigned(values: &[u64]) -> Vec<u8> {
    let mut output = Vec::new();
    for &value in values {
        encode_varint(value, &mut output);
    }
    output
}

fn signed(values: &[i64]) -> Vec<u8> {
    let mut output = Vec::new();
    for &value in values {
        encode_signed_varint(value, &mut output);
    }
    output
}

fn payload(tag: u64, values: &[i64]) -> Vec<u8> {
    [unsigned(&[tag]), signed(values)].concat()
}

fn builtin(value: Vec<u8>) -> (u64, Vec<u8>) {
    (0, value)
}
fn hlo(value: Vec<u8>) -> (u64, Vec<u8>) {
    (1, value)
}

fn fixture(attributes: &[(u64, Vec<u8>)], types: &[(u64, Vec<u8>)]) -> Vec<u8> {
    fixture_with_properties(attributes, types, None)
}

fn fixture_with_properties(
    attributes: &[(u64, Vec<u8>)],
    types: &[(u64, Vec<u8>)],
    properties: Option<&[u8]>,
) -> Vec<u8> {
    let strings = [
        "builtin",
        "stablehlo",
        "quant",
        "module",
        "test.value",
        "hello",
    ];
    let mut string_table = unsigned(&[strings.len() as u64]);
    for value in strings.iter().rev() {
        string_table.extend(unsigned(&[value.len() as u64 + 1]));
    }
    for value in strings {
        string_table.extend(value.as_bytes());
        string_table.push(0);
    }
    let mut attrs = vec![
        builtin(unsigned(&[1, 1, 1, 3])),
        builtin(unsigned(&[2, 4])),
        builtin(unsigned(&[15])),
    ];
    attrs.extend_from_slice(attributes);
    let mut entries = unsigned(&[attrs.len() as u64, types.len() as u64]);
    let mut data = Vec::new();
    for (dialect, bytes) in attrs.iter().chain(types) {
        entries.extend(unsigned(&[*dialect, 1, ((bytes.len() as u64) << 1) | 1]));
        data.extend(bytes);
    }
    // One module, with metadata and an empty, non-isolated region.
    let has_properties = properties.is_some();
    let mut ir = [
        unsigned(&[2, 0]),
        vec![if has_properties { 81 } else { 17 }],
        unsigned(&[2, 0]),
    ]
    .concat();
    if has_properties {
        ir.extend(unsigned(&[0]));
    }
    ir.extend(unsigned(&[2, 1, 0, 0]));
    let properties = properties.map_or_else(
        || unsigned(&[0]),
        |data| [unsigned(&[1, data.len() as u64]), data.to_vec()].concat(),
    );
    let mut result = b"ML\xefR".to_vec();
    result.extend(unsigned(&[6]));
    result.extend(b"Rust-payload-tests\0");
    for (id, bytes) in [
        (0, string_table),
        (
            1,
            unsigned(&[3, 0, 2, 4, 1, 0, 1, if has_properties { 7 } else { 6 }]),
        ),
        (2, data),
        (3, entries),
        (4, ir),
        (8, properties),
    ] {
        result.push(id);
        result.extend(unsigned(&[bytes.len() as u64]));
        result.extend(bytes);
    }
    result
}

fn value(attributes: &[(u64, Vec<u8>)], types: &[(u64, Vec<u8>)]) -> Attribute {
    read_module(&fixture(attributes, types))
        .unwrap()
        .attributes()["test.value"]
        .clone()
}

#[test]
fn native_property_optional_flags_and_payload_boundaries_are_checked() {
    let attributes = [builtin(unsigned(&[2, 5]))];
    let bytes = fixture_with_properties(&attributes, &[], Some(&unsigned(&[7, 0])));
    assert_eq!(read_module(&bytes).unwrap().name(), "hello");
    for properties in [
        unsigned(&[u64::MAX, 0]),
        unsigned(&[7]),
        unsigned(&[7, 0, 0]),
    ] {
        assert!(
            read_module(&fixture_with_properties(
                &attributes,
                &[],
                Some(&properties)
            ))
            .is_err()
        );
    }
    let bytes = fixture_with_properties(&attributes, &[], Some(&unsigned(&[0, 0])));
    assert_eq!(
        read_module(&bytes).unwrap().name(),
        stablehlo::text::parse_module("module {}").unwrap().name()
    );
}

#[test]
fn native_builtin_integer_and_float_payloads_preserve_width_sign_and_bits() {
    for (width, prefix) in [
        (1, 0),
        (2, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
        (64, 0),
        (8, 1),
        (8, 2),
        (64, 2),
    ] {
        let bits = if width == 1 {
            1
        } else {
            (1_u64 << (width - 1)) | 1
        };
        let encoded_bits = if width <= 8 {
            vec![bits as u8]
        } else {
            signed(&[bits as i64])
        };
        let attr = [unsigned(&[8, 0]), encoded_bits].concat();
        let actual = value(
            &[builtin(attr)],
            &[builtin(unsigned(&[0, (width << 2) | prefix]))],
        );
        if width == 1 {
            assert_eq!(actual, Attribute::Bool(true));
        } else if let Attribute::Integer { ty, value } = actual {
            assert_eq!(ty.bit_width(), Some(width as u32));
            let expected = if prefix == 2 {
                i128::from(bits)
            } else {
                i128::from((bits << (64 - width)) as i64 >> (64 - width))
            };
            assert_eq!(value, expected);
        } else {
            panic!("not an integer");
        }
    }
    for (tag, element, bits) in [
        (3, ElementType::Bf16, 0x7fc1),
        (4, ElementType::F16, 0xfc00),
        (5, ElementType::F32, 0x80000000),
        (6, ElementType::F64, 0x7ff8000000000001),
    ] {
        let attr = [unsigned(&[9, 0]), signed(&[bits as i64])].concat();
        assert_eq!(
            value(&[builtin(attr)], &[builtin(unsigned(&[tag]))]),
            Attribute::Float { ty: element, bits }
        );
    }
}

#[test]
fn native_builtin_recursive_types_and_dynamic_dimensions_reconstruct_types() {
    let types = [
        builtin([unsigned(&[13, 2]), signed(&[i64::MIN, 4]), unsigned(&[1])].concat()),
        builtin(unsigned(&[5])),
    ];
    assert_eq!(
        value(&[builtin(unsigned(&[6, 0]))], &types),
        Attribute::Type(Box::new(parse_type("tensor<?x4xf32>").unwrap()))
    );
    let types = [
        builtin(unsigned(&[15, 2, 1, 2])),
        builtin(unsigned(&[13, 0, 3])),
        hlo(unsigned(&[0])),
        builtin(unsigned(&[6])),
    ];
    assert_eq!(
        value(&[builtin(unsigned(&[6, 0]))], &types),
        Attribute::Type(Box::new(
            parse_type("tuple<tensor<f64>, !stablehlo.token>").unwrap()
        ))
    );
}

#[test]
fn native_strided_layout_attributes_and_ranked_memrefs_roundtrip() {
    let layout = [
        unsigned(&[25]),
        signed(&[0]),
        unsigned(&[2]),
        signed(&[8, 1]),
    ]
    .concat();
    assert_eq!(
        value(&[builtin(layout.clone())], &[]),
        parse_attribute("strided<[8, 1], offset: 0>").unwrap()
    );
    assert_eq!(
        parse_attribute("strided<[?, 1], offset: ?>").unwrap(),
        Attribute::StridedLayout {
            offset: None,
            strides: vec![None, Some(1)],
        }
    );
    let memref = [
        unsigned(&[10, 2]),
        signed(&[i64::MIN, 8]),
        unsigned(&[1, 4]),
    ]
    .concat();
    assert_eq!(
        value(
            &[
                builtin(unsigned(&[6, 0])),
                builtin(layout.clone()),
                builtin(unsigned(&[2, 5])),
            ],
            &[builtin(memref), builtin(unsigned(&[5]))],
        ),
        Attribute::Type(Box::new(
            parse_type("memref<?x8xf32, strided<[8, 1], offset: 0>>").unwrap()
        ))
    );
    let memref_with_space = [
        unsigned(&[11, 5, 2]),
        signed(&[i64::MIN, 8]),
        unsigned(&[1, 4]),
    ]
    .concat();
    assert_eq!(
        value(
            &[
                builtin(unsigned(&[6, 0])),
                builtin(layout),
                builtin(unsigned(&[2, 5])),
            ],
            &[builtin(memref_with_space), builtin(unsigned(&[5]))],
        ),
        Attribute::Type(Box::new(
            parse_type("memref<?x8xf32, strided<[8, 1], offset: 0>, \"hello\">").unwrap()
        ))
    );
}

#[test]
fn native_ranked_memrefs_reject_invalid_dimensions_and_layout_rank() {
    let invalid_dimension = [unsigned(&[10, 1]), signed(&[-1]), unsigned(&[1, 4])].concat();
    let invalid_layout = [
        unsigned(&[10, 2]),
        signed(&[i64::MIN, 8]),
        unsigned(&[1, 4]),
    ]
    .concat();
    let short_layout = [unsigned(&[25]), signed(&[0]), unsigned(&[1]), signed(&[8])].concat();
    for (memref, layout) in [
        (invalid_dimension, unsigned(&[25, 0, 0])),
        (invalid_layout, short_layout),
    ] {
        assert!(
            read_module(&fixture(
                &[builtin(unsigned(&[6, 0])), builtin(layout)],
                &[builtin(memref), builtin(unsigned(&[5]))],
            ))
            .is_err()
        );
    }
}

#[test]
fn native_dense_i1_payloads_unpack_splats_packed_bits_and_legacy_bytes() {
    let ty = |count| {
        vec![
            builtin([unsigned(&[13, 1]), signed(&[count]), unsigned(&[1])].concat()),
            builtin(unsigned(&[0, 4])),
        ]
    };
    for (count, bytes, expected) in [
        (5, vec![0xff], vec![true; 5]),
        (5, vec![0], vec![false; 5]),
        (5, vec![0b10101], vec![true, false, true, false, true]),
        (
            9,
            vec![0x81, 1],
            vec![true, false, false, false, false, false, false, true, true],
        ),
        (3, vec![1, 0, 1], vec![true, false, true]),
        (0, vec![], vec![]),
    ] {
        let attr = [unsigned(&[18, 0, bytes.len() as u64]), bytes].concat();
        let Attribute::DenseElements { values, .. } = value(&[builtin(attr)], &ty(count)) else {
            panic!("not dense");
        };
        assert_eq!(
            values,
            expected
                .into_iter()
                .map(ScalarValue::Bool)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn native_dense_array_payloads_require_exact_lengths_not_splat_lengths() {
    let ty = [builtin(unsigned(&[0, 64 << 2]))];
    let bytes = [7_i64.to_le_bytes(), (-2_i64).to_le_bytes()].concat();
    let attr = [unsigned(&[17, 0, 2, bytes.len() as u64]), bytes].concat();
    assert_eq!(
        value(&[builtin(attr)], &ty),
        Attribute::integers(vec![7, -2])
    );
    let malformed = [unsigned(&[17, 0, 3, 8]), vec![0; 8]].concat();
    assert!(read_module(&fixture(&[builtin(malformed)], &ty)).is_err());
}

#[test]
fn cyclic_native_references_and_malformed_custom_payloads_fail_without_panicking() {
    for attributes in [
        vec![builtin(unsigned(&[0, 1, 3]))],
        vec![builtin(unsigned(&[1, 1, 1, 3]))],
        vec![builtin(unsigned(&[6, 0]))],
    ] {
        let error =
            read_module(&fixture(&attributes, &[builtin(unsigned(&[15, 1, 0]))])).unwrap_err();
        assert_eq!(
            error.kind,
            Kind::InvalidPayload("acyclic attribute/type references")
        );
    }
    for attr in [
        unsigned(&[0, u64::MAX]),
        unsigned(&[2, 99]),
        unsigned(&[15, 1]),
        vec![0],
        unsigned(&[255]),
    ] {
        assert!(read_module(&fixture(&[builtin(attr)], &[])).is_err());
    }
}

#[test]
fn stablehlo_custom_enumerations_and_structured_attributes_decode_exactly() {
    for (tag, name, variants) in [
        (
            2,
            "comparison_direction",
            &["EQ", "NE", "GE", "GT", "LE", "LT"][..],
        ),
        (
            3,
            "comparison_type",
            &[
                "NOTYPE",
                "FLOAT",
                "TOTALORDER",
                "SIGNED",
                "UNSIGNED",
                "WEAKORDER",
            ][..],
        ),
        (6, "fft_type", &["FFT", "IFFT", "RFFT", "IRFFT"][..]),
        (8, "precision", &["DEFAULT", "HIGH", "HIGHEST"][..]),
        (9, "rng_algorithm", &["DEFAULT", "THREE_FRY", "PHILOX"][..]),
        (
            12,
            "transpose",
            &["TRANSPOSE_INVALID", "NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"][..],
        ),
        (
            16,
            "result_accuracy_mode",
            &["DEFAULT", "HIGHEST", "TOLERANCE"][..],
        ),
    ] {
        for (index, variant) in variants.iter().enumerate() {
            assert_eq!(
                value(&[hlo(unsigned(&[tag, index as u64]))], &[]),
                Attribute::Enum {
                    dialect: "stablehlo".into(),
                    name: name.into(),
                    value: (*variant).into()
                }
            );
        }
        assert!(
            read_module(&fixture(
                &[hlo(unsigned(&[tag, variants.len() as u64]))],
                &[]
            ))
            .is_err()
        );
    }
    assert_eq!(
        value(&[hlo(payload(1, &[9, 1]))], &[]),
        parse_attribute("#stablehlo.channel_handle<handle = 9, type = 1>").unwrap()
    );
    assert_eq!(
        value(
            &[hlo([unsigned(&[13, 2]), signed(&[i64::MIN, 8])].concat())],
            &[]
        ),
        parse_attribute("#stablehlo.bounds<?, 8>").unwrap()
    );
}

#[test]
fn native_quantized_types_normalize_flags_ranges_and_preserve_scale_bits() {
    for (flags, spelling, minimum, maximum) in [(0, "ui8", 0, 255), (1, "i8", -128, 127)] {
        let quant = [
            unsigned(&[4, flags, 1, 2]),
            signed(&[0.25_f64.to_bits() as i64, 0, minimum, maximum]),
        ]
        .concat();
        let types = [
            (2, quant),
            builtin(unsigned(&[0, 8 << 2])),
            builtin(unsigned(&[5])),
        ];
        assert_eq!(
            value(&[builtin(unsigned(&[6, 0]))], &types),
            Attribute::Type(Box::new(
                parse_type(&format!("!quant.uniform<{spelling}:f32, 0.25>")).unwrap()
            ))
        );
    }
}

#[test]
fn acyclic_native_dependency_depth_and_shared_expansion_are_bounded() {
    let mut attributes = Vec::new();
    for index in 0..140 {
        attributes.push(builtin(unsigned(&[0, 1, index + 4])));
    }
    attributes.push(builtin(unsigned(&[7])));
    assert_eq!(
        read_module(&fixture(&attributes, &[])).unwrap_err().kind,
        Kind::NestingLimit
    );
    let mut attributes = Vec::new();
    for index in 0..30 {
        attributes.push(builtin(unsigned(&[0, 2, index + 4, index + 4])));
    }
    attributes.push(builtin(unsigned(&[2, 5])));
    assert_eq!(
        read_module(&fixture(&attributes, &[])).unwrap_err().kind,
        Kind::ResourceLimit
    );
}

#[test]
fn native_affine_map_header_kinds_and_prefix_expression_codes_decode() {
    for (bytes, expected) in [
        (unsigned(&[23, 8]), "affine_map<(d0, d1) -> (d0, d1)>"),
        (unsigned(&[23, 9, 1, 0]), "affine_map<(d0, d1) -> (d1, d0)>"),
        (unsigned(&[23, 10, 1, 1]), "affine_map<(d0, d1) -> (d1)>"),
        (
            unsigned(&[23, 7, 1, 1, 3, 0, 0, 1, 0]),
            "affine_map<(d0)[s0] -> (d0 + s0)>",
        ),
        (
            [unsigned(&[23, 7, 0, 1, 6, 0, 0, 2]), signed(&[4])].concat(),
            "affine_map<(d0) -> (d0 floordiv 4)>",
        ),
    ] {
        assert_eq!(
            value(&[builtin(bytes)], &[]),
            parse_attribute(expected).unwrap()
        );
    }
    for bytes in [
        unsigned(&[23, u64::MAX]),
        unsigned(&[23, 9, 0, 0]),
        unsigned(&[23, 7, 0, 1, 99]),
    ] {
        assert!(read_module(&fixture(&[builtin(bytes)], &[])).is_err());
    }
}
