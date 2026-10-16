use stablehlo::{
    Attribute, ElementType, ScalarValue, TensorType, Type,
    text::{TextErrorKind, parse_attribute, print_attribute},
};

fn round_trip(input: &str) -> Attribute {
    let attribute = parse_attribute(input).unwrap_or_else(|error| panic!("{input}: {error}"));
    let output = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&output).unwrap(), attribute, "{output}");
    attribute
}

#[test]
fn integer_dense_scalars_splats_and_multidimensional_literals() {
    for input in [
        "dense<1> : tensor<i64>",
        "dense<7> : tensor<2x3xi32>",
        "dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>",
        "dense<[0, 2, 1]> : tensor<3xindex>",
        "dense<[]> : tensor<0xi64>",
        "dense<> : tensor<0x2xi64>",
        "dense<0> : tensor<0x2xi64>",
        "dense<[[], []]> : tensor<2x0xi32>",
        "dense<[[[1, 2]], [[3, 4]]]> : tensor<2x1x2xi4>",
        "dense<[-128, 127]> : tensor<2xi8>",
        "dense<[0, 255]> : tensor<2xui8>",
        "dense<[0xFF, 0x80, 0x7F]> : tensor<3xi8>",
        "dense<[0xFF, 0x80, 0x7F]> : tensor<3xui8>",
        "dense<18446744073709551615> : tensor<ui64>",
    ] {
        round_trip(input);
    }
    let scalar = round_trip("dense<7> : tensor<2x3xi32>");
    assert_eq!(
        scalar,
        Attribute::DenseElements {
            ty: Box::new(Type::from(TensorType::static_shape(
                ElementType::I32,
                [2, 3]
            ))),
            values: vec![ScalarValue::Integer(7); 6],
        }
    );
}

#[test]
fn deeply_nested_dense_literals_parse_and_print_without_recursion() {
    let rank = 120;
    let source = format!(
        "dense<{}1{}> : tensor<{}i32>",
        "[".repeat(rank),
        "]".repeat(rank),
        "1x".repeat(rank)
    );
    round_trip(&source);
}

#[test]
fn boolean_literals_support_numeric_and_keyword_forms() {
    for input in [
        "dense<true> : tensor<2xi1>",
        "dense<[false, true, 0, 1, -1, 0x1]> : tensor<6xi1>",
    ] {
        round_trip(input);
    }
    assert!(parse_attribute("dense<2> : tensor<i1>").is_err());
}

#[test]
fn float_and_complex_literals_preserve_bits_and_nan_payloads() {
    for input in [
        "dense<[1.0, -0.0, 2.5]> : tensor<3xf32>",
        "dense<[1.0, -0.0, 2.5]> : tensor<3xf64>",
        "dense<[0x7FC00001, 0x80000000]> : tensor<2xf32>",
        "dense<[0x7FF8000000000123, 0x8000000000000000]> : tensor<2xf64>",
        "dense<[(1.0, -2.0), (0x7FC00001, 0x80000000)]> : tensor<2xcomplex<f32>>",
        "dense<(1.0, -2.0)> : tensor<2xcomplex<f64>>",
        "dense<[0x3C00, 0x8000, 0x7E01]> : tensor<3xf16>",
        "dense<[0x3F80, 0x8000, 0x7FC1]> : tensor<3xbf16>",
        "dense<[0x0, 0xF]> : tensor<2xf4E2M1FN>",
        "dense<[0x0, 0x3F]> : tensor<2xf6E3M2FN>",
        "dense<[0x0, 0xFF]> : tensor<2xf8E4M3FN>",
    ] {
        round_trip(input);
    }
}

#[test]
fn quantized_dense_values_respect_storage_ranges() {
    round_trip("dense<[-127, 15, 127]> : tensor<3x!quant.uniform<i8<-127:127>:f32, 2.0:15>>");
    assert!(
        parse_attribute("dense<-128> : tensor<!quant.uniform<i8<-127:127>:f32, 1.0>>").is_err()
    );
}

#[test]
fn shape_and_element_mismatches_are_not_flattened_away() {
    for input in [
        "dense<[]> : tensor<1xi32>",
        "dense<[1, 2, 3, 4]> : tensor<2x2xi32>",
        "dense<[[1], [2, 3, 4]]> : tensor<2x2xi32>",
        "dense<[[1, 2], [3, 4]]> : tensor<4xi32>",
        "dense<[1]> : tensor<i32>",
        "dense<[[1]]> : tensor<1xi32>",
        "dense<256> : tensor<i8>",
        "dense<-1> : tensor<ui8>",
        "dense<0x100> : tensor<i8>",
        "dense<0x10000> : tensor<f16>",
        "dense<1.0> : tensor<i32>",
        "dense<true> : tensor<i32>",
        "dense<(1, 2)> : tensor<f32>",
        "dense<1.0> : tensor<complex<f32>>",
        "dense<1> : tensor<?xi32>",
        "dense<1> : i32",
        "dense<> : tensor<f32>",
        "dense<[1,]> : tensor<1xi32>",
    ] {
        assert!(parse_attribute(input).is_err(), "{input}");
    }
}

#[test]
fn compressed_literals_have_explicit_resource_and_depth_limits() {
    assert_eq!(
        parse_attribute("dense<0> : tensor<9223372036854775807xi64>")
            .unwrap_err()
            .kind,
        TextErrorKind::ResourceLimit
    );
    let input = format!(
        "dense<{}0{}> : tensor<i64>",
        "[".repeat(129),
        "]".repeat(129)
    );
    assert_eq!(
        parse_attribute(&input).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    // Zero-element shapes do not expand the enormous leading extent.
    round_trip("dense<> : tensor<9223372036854775807x0xi64>");
}

#[test]
fn malformed_dense_printer_input_reports_validation_error() {
    let attribute = Attribute::DenseElements {
        ty: Box::new(Type::from(TensorType::static_shape(ElementType::I8, [2]))),
        values: vec![ScalarValue::Integer(1000), ScalarValue::Integer(2)],
    };
    assert!(matches!(
        print_attribute(&attribute).unwrap_err().kind,
        TextErrorKind::InvalidAttribute(_)
    ));
}
