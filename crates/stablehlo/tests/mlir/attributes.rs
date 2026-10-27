use stablehlo::mlir::assembly::{
    parse_attribute, parse_dictionary, parse_named_attributes, print_attribute, print_dictionary,
};
use stablehlo::{
    attributes::{
        BytesAttr, DenseElementsAttr, NamedAttributesAttr, ScalarBitsAttr, verify_attribute,
    },
    pliron::{
        attribute::AttrObj,
        builtin::attributes::{StringAttr, VecAttr},
        context::Context,
    },
};

#[test]
fn native_attributes_roundtrip_with_fresh_contexts() {
    for source in [
        "unit",
        "true",
        "false",
        "12",
        "-12 : i64",
        "340282366920938463463374607431768211455 : ui128",
        "-170141183460469231731687303715884105728 : i128",
        "0x7F800001 : f32",
        "1.00048828125000000000001 : f16",
        "4 : index",
        r#""plain""#,
        r#""\00\22\\\0A\FF""#,
        r#""typed" : i32"#,
        "[12, true, unit, 0x8000 : f16]",
        "{z = 1, a = [false, unit], flag}",
        r#"{test.note = "kept", "spaced key" = {"1" = true}}"#,
        "@callee::@nested",
        r#"@"quoted-name""#,
        "array<i64: -1, 0, 1>",
        "array<f32: 1.0, 0x7F800001>",
        "array<i64>",
        "dense<1.0> : tensor<2x3xf32>",
        "dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>",
        "dense<[true, false]> : tensor<2xi1>",
        "dense<[]> : tensor<0xi8>",
        r#"dense<"0x00800000"> : tensor<2xi16>"#,
        r#"dense<"0x0180"> : tensor<2xi1>"#,
        r#"dense<"0x1234"> : tensor<complex<i8>>"#,
        r#"dense<"0x01000080"> : tensor<3xf32>"#,
        r#"dense<"0xFF"> : tensor<3xi4>"#,
        r#"dense<"0x"> : tensor<0xf32>"#,
        "sparse<> : tensor<2x3xf32>",
        "sparse<[[0, 1], [1, 2]], [1.0, -2.0]> : tensor<2x3xf32>",
        "sparse<[0, 2], 7> : tensor<3xi32>",
        r#"sparse<[[0], [2]], "0x07000000"> : tensor<3xi32>"#,
        "sparse<[[0], [2]], [(1.0, -1.0), (2.0, 3.0)]> : tensor<3xcomplex<f64>>",
        "sparse<[[]], [42]> : tensor<i32>",
        "dense<[(0x7F800001, 0x80000000), (1.0, -2.0)]> : tensor<2xcomplex<f32>>",
        "dense<(255, -128)> : tensor<complex<i8>>",
        "dense<(340282366920938463463374607431768211455, 0)> : tensor<complex<i128>>",
        "255 : i8",
        "#stablehlo.precision<HIGHEST>",
        "#stablehlo.comparison_direction<LT>",
        "#stablehlo.type_extensions<bounds = [16, ?]>",
        "tensor<2xf32>",
        "(tensor<2xf32>) -> (tensor<2xf32>)",
    ] {
        let ctx = Context::new();
        let attribute =
            parse_attribute(&ctx, source).unwrap_or_else(|error| panic!("{source}: {error}"));
        let printed =
            print_attribute(&ctx, &*attribute).unwrap_or_else(|error| panic!("{source}: {error}"));
        let fresh = Context::new();
        let reparsed =
            parse_attribute(&fresh, &printed).unwrap_or_else(|error| panic!("{printed}: {error}"));
        assert_eq!(
            print_attribute(&fresh, &*reparsed).unwrap(),
            printed,
            "{source}"
        );
    }
}

#[test]
fn string_keyed_dictionaries_preserve_names_and_verify_native_invariants() {
    let ctx = Context::new();
    let source = r#"{test.note = "kept", "spaced key" = {"1" = true}}"#;
    let native = parse_named_attributes(&ctx, source).unwrap();
    assert_eq!(
        native
            .names
            .iter()
            .map(StringAttr::as_str)
            .collect::<Vec<_>>(),
        ["test.note", "spaced key"]
    );
    let printed = print_attribute(&ctx, &native).unwrap();
    assert!(printed.contains("test.note"));
    assert!(printed.contains("\"spaced key\""));
    assert_eq!(
        parse_named_attributes(&ctx, &printed).unwrap().names.len(),
        2
    );
    for names in [vec!["a", "a"], vec![""], vec![]] {
        let malformed: AttrObj = Box::new(NamedAttributesAttr {
            names: names.into_iter().map(StringAttr::from).collect(),
            values: VecAttr::new(vec![Box::new(StringAttr::from("value"))]),
        });
        assert!(verify_attribute(&ctx, &malformed).is_err());
        assert!(print_attribute(&ctx, &*malformed).is_err());
    }
    assert!(parse_attribute(&ctx, r#"{test.a = unit, "test.a" = unit}"#).is_err());
}

#[test]
fn exact_bits_and_bytes_are_not_reinterpreted() {
    let ctx = Context::new();
    let attribute = parse_attribute(&ctx, "0x7F800001 : f32").unwrap();
    assert_eq!(
        attribute.downcast_ref::<ScalarBitsAttr>().unwrap().words,
        [0x7f800001]
    );
    let attribute = parse_attribute(&ctx, r#""\00\FF\22""#).unwrap();
    assert_eq!(
        attribute.downcast_ref::<BytesAttr>().unwrap().value,
        [0, 255, 34]
    );
    let attribute =
        parse_attribute(&ctx, "dense<[0x7F800001, 0x80000000]> : tensor<2xf32>").unwrap();
    let dense = attribute.downcast_ref::<DenseElementsAttr>().unwrap();
    assert!(!dense.is_splat);
    assert_eq!(
        dense.values.0[0]
            .downcast_ref::<ScalarBitsAttr>()
            .unwrap()
            .words,
        [0x7f800001]
    );
}

#[test]
fn dictionary_order_and_source_are_stable() {
    let ctx = Context::new();
    let source = "{z = 1, a = 2, flag}";
    let dictionary = parse_dictionary(&ctx, source).unwrap();
    assert_eq!(
        print_dictionary(&ctx, &dictionary).unwrap(),
        "{a = 2 : i64, flag, z = 1 : i64}"
    );
    assert_eq!(source, "{z = 1, a = 2, flag}");
}

#[test]
fn malformed_attributes_fail_and_later_valid_parse_recovers() {
    let ctx = Context::new();
    for source in [
        "{a = 1, a = 2}",
        "{a = }",
        "[1,]",
        r#""\G0""#,
        r#""\F""#,
        "256 : ui8",
        "-1 : ui8",
        "0x100 : f8E5M2",
        "1.0 : i32",
        "dense<[[1], [2, 3]]> : tensor<2x1xi32>",
        "dense<[1, 2]> : tensor<3xi32>",
        "dense<true> : tensor<i32>",
        "dense<1> : tensor<0xi32>",
        "dense<1> : tensor<?xi32>",
        "-0 : i32",
        "dense<(1.0, 2.0)> : tensor<f32>",
        r#"dense<"0x000"> : tensor<2xi8>"#,
        r#"dense<"0x000102"> : tensor<2xi8>"#,
        r#"dense<"0x0G"> : tensor<i8>"#,
        "sparse<[[2]], [1]> : tensor<2xi32>",
        "sparse<[[0, 1]], [1]> : tensor<2xi32>",
        "sparse<[[0]], [1, 2]> : tensor<2xi32>",
        "sparse<[[0]], [[1]]> : tensor<2xi32>",
        "sparse<[[0]], [1]> : tensor<?xi32>",
        "sparse<[-1], [1]> : tensor<2xi32>",
        "sparse<[[-1, 0]], [1]> : tensor<2x2xi32>",
        "sparse<[[0, -1]], [1]> : tensor<2x2xi32>",
        "sparse<[[0]], [1]> : tensor<-1xi32>",
        "sparse<[[0, 0]], [1]> : tensor<2x-1xi32>",
        "sparse<[-1 x []], [1]> : tensor<2xi32>",
        "sparse<[[0] : -1], [1]> : tensor<2xi32>",
        "#stablehlo.precision<NOT_A_VALUE>",
        "#unknown.attr<1>",
        "@callee::",
    ] {
        assert!(parse_attribute(&ctx, source).is_err(), "{source:?}");
    }
    assert!(parse_attribute(&ctx, "{answer = 42}").is_ok());
}
