use stablehlo::{
    Attribute, ScalarValue,
    text::{TextErrorKind, parse_attribute, print_attribute},
};

fn round_trip(input: &str) -> Attribute {
    let attribute = parse_attribute(input).unwrap_or_else(|error| panic!("{input}: {error}"));
    let output = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&output).unwrap(), attribute, "{output}");
    attribute
}

#[test]
fn raw_buffers_use_pinned_mlir_little_endian_whole_byte_storage() {
    // LLVM 2b35fb9283a44aae92cb72a41d077ee8de2be7ad:
    // BuiltinAttributes.cpp, DenseElementsAttr::isValidRawBuffer.
    for (raw, explicit) in [
        (
            "dense<\"0x01000000FEFFFFFF\"> : tensor<2xi32>",
            "dense<[1, -2]> : tensor<2xi32>",
        ),
        (
            "dense<\"0x0000803F00000080\"> : tensor<2xf32>",
            "dense<[1.0, -0.0]> : tensor<2xf32>",
        ),
        (
            "dense<\"0x000000000000F03F0000000000000080\"> : tensor<2xf64>",
            "dense<[1.0, -0.0]> : tensor<2xf64>",
        ),
        (
            "dense<\"0x0000803F000000C0\"> : tensor<complex<f32>>",
            "dense<(1.0, -2.0)> : tensor<complex<f32>>",
        ),
        (
            "dense<\"0x000000000000F03F00000000000000C0\"> : tensor<complex<f64>>",
            "dense<(1.0, -2.0)> : tensor<complex<f64>>",
        ),
        (
            "dense<\"0x00010001\"> : tensor<4xi1>",
            "dense<[false, true, false, true]> : tensor<4xi1>",
        ),
        (
            "dense<\"0x0F0807\"> : tensor<3xi4>",
            "dense<[-1, -8, 7]> : tensor<3xi4>",
        ),
        (
            "dense<\"0x003C0080\"> : tensor<2xf16>",
            "dense<[0x3C00, 0x8000]> : tensor<2xf16>",
        ),
        (
            "dense<\"0xFEFF\"> : tensor<2xsi8>",
            "dense<[-2, -1]> : tensor<2xsi8>",
        ),
    ] {
        assert_eq!(round_trip(raw), round_trip(explicit), "{raw}");
    }
}

#[test]
fn raw_splats_and_empty_buffers_expand_without_changing_values() {
    assert_eq!(
        round_trip("dense<\"0x01000000\"> : tensor<2x3xi32>"),
        round_trip("dense<1> : tensor<2x3xi32>")
    );
    round_trip("dense<\"0x\"> : tensor<0xi32>");
    round_trip("dense<\"0x00000000\"> : tensor<0xi32>");
    for input in [
        "dense<\"0x0\"> : tensor<i32>",
        "dense<\"0x0102\"> : tensor<2xi32>",
        "dense<\"0xGG\"> : tensor<i8>",
        "dense<\"0x\"> : tensor<i32>",
        "dense<\"not hex\"> : tensor<i32>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
    assert_eq!(
        parse_attribute("dense<\"0x00\"> : tensor<9223372036854775807xi8>")
            .unwrap_err()
            .kind,
        TextErrorKind::ResourceLimit
    );
}

#[test]
fn signless_integer_text_accepts_unsigned_bit_patterns_but_signed_types_do_not() {
    for (input, value) in [
        ("dense<255> : tensor<i8>", -1),
        ("dense<128> : tensor<i8>", -128),
        ("dense<0x8000> : tensor<i16>", -32768),
        ("dense<18446744073709551615> : tensor<i64>", -1),
        ("dense<-0x80> : tensor<i8>", -128),
        ("dense<-1> : tensor<index>", -1),
        (
            "dense<0x7FFFFFFFFFFFFFFF> : tensor<index>",
            i128::from(i64::MAX),
        ),
    ] {
        let Attribute::DenseElements { values, .. } = round_trip(input) else {
            panic!()
        };
        assert_eq!(values, vec![ScalarValue::Integer(value)]);
    }
    for input in [
        "dense<128> : tensor<si8>",
        "dense<0x80> : tensor<si8>",
        "dense<0x8000000000000000> : tensor<index>",
        "dense<-129> : tensor<i8>",
        "dense<-0> : tensor<ui8>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn sparse_coo_assembly_preserves_coordinates_values_and_empty_arrays() {
    for input in [
        "sparse<[[0, 0], [1, 2]], [1, 5]> : tensor<3x4xi32>",
        "sparse<[[0], [4]], [1.0, -2.0]> : tensor<8xf32>",
        "sparse<[[0], [1]], [(1.0, 2.0), (-1.0, -2.0)]> : tensor<2xcomplex<f64>>",
        "sparse<> : tensor<2x3xf64>",
        "sparse<> : tensor<f64>",
        "sparse<[], []> : tensor<3xf64>",
        "sparse<[[0], [2]], 1.0> : tensor<4xf32>",
        "sparse<[[0], [2]], \"0x0000803F00000040\"> : tensor<4xf32>",
        "sparse<[[]], [7]> : tensor<i64>",
    ] {
        round_trip(input);
    }
}

#[test]
fn malformed_sparse_indices_and_values_are_rejected() {
    for input in [
        "sparse<[[3, 0]], [1]> : tensor<3x4xi32>",
        "sparse<[[-1, 0]], [1]> : tensor<3x4xi32>",
        "sparse<[[0]], [1]> : tensor<3x4xi32>",
        "sparse<[[0, 0]], [1, 2]> : tensor<3x4xi32>",
        "sparse<[0, 1], [1, 2]> : tensor<3x4xi32>",
        "sparse<[], []> : tensor<3x4xi32>",
        "sparse<[[0]], [1]> : tensor<?xi32>",
        "sparse<[[0]], [1.0]> : tensor<3xi32>",
        "sparse<[[0.0]], [1]> : tensor<3xi32>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn sparse_coordinate_splats_and_rank_one_shorthand_match_the_reference_parser() {
    // AttributeParser.cpp::parseSparseElementsAttr permits scalar index splats,
    // and SparseElementsAttr::verify also permits a rank-one index list for
    // rank-one tensors. Hexadecimal coordinate buffers are explicitly forbidden.
    for (shorthand, explicit) in [
        (
            "sparse<0, 7> : tensor<2x3xi32>",
            "sparse<[[0, 0]], [7]> : tensor<2x3xi32>",
        ),
        (
            "sparse<1, 7> : tensor<2x3xi32>",
            "sparse<[[1, 1]], [7]> : tensor<2x3xi32>",
        ),
        (
            "sparse<[0, 2], [1, 5]> : tensor<3xi32>",
            "sparse<[[0], [2]], [1, 5]> : tensor<3xi32>",
        ),
        (
            "sparse<0, 7> : tensor<i32>",
            "sparse<[[]], [7]> : tensor<i32>",
        ),
        ("sparse<> : tensor<3xi32>", "sparse<[], []> : tensor<3xi32>"),
    ] {
        assert_eq!(round_trip(shorthand), round_trip(explicit));
    }
    for input in [
        "sparse<\"0x0000000000000000\", [7]> : tensor<3xi32>",
        "sparse<2, 7> : tensor<2x3xi32>",
        "sparse<[[0], [1, 2]], [1, 5]> : tensor<3x4xi32>",
        "sparse<[[0, 0]], [[7]]> : tensor<3x4xi32>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
    assert_eq!(
        print_attribute(&round_trip("sparse<> : tensor<2x3xi32>")).unwrap(),
        "sparse<> : tensor<2x3xi32>"
    );
}

#[test]
fn dense_zero_extent_lists_still_preserve_rank() {
    round_trip("dense<[[], []]> : tensor<2x0xi32>");
    assert!(parse_attribute("dense<[[], []]> : tensor<2x0x4xi32>").is_err());
    assert!(parse_attribute("dense<[]> : tensor<0x4xi32>").is_err());
    assert!(parse_attribute("42 : f64").is_err());
    assert!(parse_attribute("dense<42> : tensor<f32>").is_err());
}
