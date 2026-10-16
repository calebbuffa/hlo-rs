use stablehlo::{
    Attribute, BufferLayout, BufferType, ElementType, QuantizedTensorType, StableHloAttribute,
    TensorType, Type, TypeExtensions,
    text::{TextErrorKind, parse_type, print_type},
};

fn round_trip(input: &str) -> Type {
    let ty = parse_type(input).unwrap_or_else(|error| panic!("{input}: {error}"));
    let printed = print_type(&ty).unwrap();
    assert_eq!(parse_type(&printed).unwrap(), ty, "{printed}");
    assert_eq!(print_type(&parse_type(&printed).unwrap()).unwrap(), printed);
    ty
}

#[test]
fn future_syntax_from_pinned_reference() {
    // stablehlo/tests/future_type.mlir at the crate's pinned reference.
    for input in [
        "!stablehlo.future<tensor<f32>>",
        "!stablehlo.future<tensor<2x3xf32>>",
        "!stablehlo.future<tensor<?xf32>>",
        "!stablehlo.future<tensor<4xi32>>",
        "!stablehlo.future<tensor<2xcomplex<f32>>>",
        "!stablehlo.future<tensor<f32>, tensor<i32>>",
        "!stablehlo.future<tensor<2x3xf32>, tensor<4xi32>>",
        "!stablehlo.future<>",
    ] {
        round_trip(input);
    }
    for input in [
        "!stablehlo.future<f32>",
        "!stablehlo.future<i32>",
        "!stablehlo.future<tensor<f32>, f32>",
    ] {
        assert!(matches!(
            parse_type(input).unwrap_err().kind,
            TextErrorKind::InvalidType(_)
        ));
    }
}

#[test]
fn every_element_has_distinct_mlir_spelling() {
    let spellings = [
        "index",
        "i1",
        "i2",
        "i4",
        "i8",
        "i16",
        "i32",
        "i64",
        "si2",
        "si4",
        "si8",
        "si16",
        "si32",
        "si64",
        "ui2",
        "ui4",
        "ui8",
        "ui16",
        "ui32",
        "ui64",
        "f4E2M1FN",
        "f6E2M3FN",
        "f6E3M2FN",
        "f16",
        "bf16",
        "f32",
        "f64",
        "tf32",
        "f8E3M4",
        "f8E4M3",
        "f8E4M3FN",
        "f8E4M3FNUZ",
        "f8E4M3B11FNUZ",
        "f8E5M2",
        "f8E5M2FNUZ",
        "f8E8M0FNU",
        "complex<f32>",
        "complex<f64>",
    ];
    let mut elements = std::collections::HashSet::new();
    for spelling in spellings {
        let ty = round_trip(spelling);
        assert!(elements.insert(ty.clone()), "{spelling}");
        assert_eq!(print_type(&ty).unwrap(), spelling);
        round_trip(&format!("tensor<?x0x2x{spelling}>"));
    }
    assert_eq!(elements.len(), 38);
}

#[test]
fn tuple_function_and_nested_types() {
    for input in [
        "tuple<>",
        "tuple<tensor<f32>, !stablehlo.token>",
        "tuple<tuple<tensor<?xf32>>, !stablehlo.future<tensor<i32>>>",
        "() -> ()",
        "(tensor<2xf32>) -> tensor<2xi32>",
        "(tensor<f32>, !stablehlo.token) -> (tensor<i32>, !stablehlo.token)",
        "(tuple<tensor<f32>, tensor<i32>>) -> tuple<tensor<?xf64>>",
    ] {
        round_trip(input);
    }
    assert!(parse_type("tuple<f32>").is_err());
    assert!(parse_type("(tensor<f32>) -> i32").is_err());
}

#[test]
fn bounds_encoding_matches_upstream_assembly_format() {
    // stablehlo/tests/infer_stablehlo.mlir and dialect/AssemblyFormat.cpp.
    let ty = round_trip("tensor<16x?xf32, #stablehlo.bounds<?, 5>>");
    assert_eq!(
        ty,
        Type::from(
            TensorType::new(ElementType::F32, vec![Some(16), None]).with_encoding(
                Attribute::StableHlo(Box::new(StableHloAttribute::TypeExtensions(
                    TypeExtensions {
                        bounds: vec![-1, 5]
                    }
                )))
            )
        )
    );
    round_trip("tensor<?x?x?xi32, #stablehlo.bounds<2, 4, 8>>");
    round_trip("tensor<f32, #stablehlo.bounds<>>");
    let generated = round_trip("tensor<?xf32, #stablehlo.type_extensions<bounds = [5]>>");
    assert_eq!(
        print_type(&generated).unwrap(),
        "tensor<?xf32, #stablehlo.bounds<5>>"
    );
    for input in [
        "tensor<?xf32, #stablehlo.bounds<-2>>",
        "tensor<4xf32, #stablehlo.bounds<5>>",
        "tensor<?x?xf32, #stablehlo.bounds<5>>",
    ] {
        assert!(parse_type(input).is_err(), "{input}");
    }
}

#[test]
fn quantized_syntax_from_reference_and_exact_scale_round_trip() {
    // Per-axis fixture spelling from stablehlo/tests/ops_stablehlo.mlir.
    for input in [
        "tensor<!quant.uniform<i8:f32, 2.000000e+00:15>>",
        "tensor<2x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>",
        "tensor<2x!quant.uniform<ui8:f32, 34.0:16>>",
        "!stablehlo.future<tensor<!quant.uniform<i8:f32, 2:15>>>",
        "tensor<?x!quant.uniform<i8:f32, 0.25>, #stablehlo.bounds<12>>",
        "tensor<!quant.uniform<i8:f32, 0x3FC0000000000000>>",
    ] {
        round_trip(input);
    }
    for bits in [
        1_u64,
        0x0010000000000000,
        0x3fd5555555555555,
        0x7fefffffffffffff,
    ] {
        let ty = Type::QuantizedTensor(QuantizedTensorType::new(
            vec![],
            ElementType::I8,
            None,
            None,
            ElementType::F64,
            None,
            vec![f64::from_bits(bits)],
            vec![0],
        ));
        assert_eq!(parse_type(&print_type(&ty).unwrap()).unwrap(), ty);
    }
    for input in [
        "tensor<!quant.uniform<i8:f32, -1>>",
        "tensor<!quant.uniform<i8:f32, 0>>",
        "tensor<!quant.uniform<i8:f32, inf>>",
        "tensor<!quant.uniform<i8:f32, NaN>>",
        "tensor<!quant.uniform<i8:f32, 1:128>>",
        "tensor<2x!quant.uniform<i8:f32:0, {1}>>",
        "tensor<!quant.uniform<i8<-129:127>:f32, 1>>",
    ] {
        assert!(parse_type(input).is_err(), "{input}");
    }
}

#[test]
fn buffer_layout_and_memory_space_are_preserved() {
    for input in [
        "memref<f32>",
        "memref<2x?xf32>",
        "memref<2x?xf32, 3>",
        "memref<2x?xf32, 3 : i64>",
        "memref<2x?xf32, strided<[?, 1], offset: ?>>",
        "memref<2x?xf32, strided<[?, 1], offset: 2>, 5>",
        "memref<4xf32, strided<[1]>>",
    ] {
        round_trip(input);
    }
    let expected = Type::Buffer(
        BufferType::new(ElementType::F32, vec![Some(2), None])
            .with_layout(BufferLayout::Strided {
                offset: None,
                strides: vec![None, Some(1)],
            })
            .with_memory_space(Attribute::integer(5)),
    );
    assert_eq!(
        parse_type("memref<2x?xf32, strided<[?, 1], offset: ?>, 5>").unwrap(),
        expected
    );
    assert!(parse_type("memref<4xf32, strided<[1, 1]>>").is_err());
}

#[test]
fn whitespace_comments_and_numeric_dimension_limits() {
    round_trip(" // before\n tensor< 2 x ? x complex< f64 > > // after");
    round_trip("tensor<9223372036854775807xf32>");
    round_trip("tensor<0xf32>");
    assert!(parse_type("tensor<9223372036854775808xf32>").is_err());
    assert!(parse_type("tensor<18446744073709551616xf32>").is_err());
    assert!(parse_type("tensor<-1xf32>").is_err());
}

#[test]
fn failures_have_offsets_and_never_ignore_trailing_input() {
    let input = "tensor<2xf32> garbage";
    let error = parse_type(input).unwrap_err();
    assert_eq!(error.kind, TextErrorKind::TrailingInput);
    assert_eq!(error.offset, input.find("garbage").unwrap());
    assert!(error.to_string().contains("byte 14"));
    for input in [
        "",
        "tensor<",
        "tensor<2f32>",
        "tensor<*x2xf32>",
        "tensor<2x>",
        "tensor<f32",
        "tensor<f32>>",
        "tuple<tensor<f32>,>",
        "tensor<f32> tensor<i32>",
        "!stablehlo.token<foo>",
        "complex<i32>",
        "tensor<f32\0>",
        "tensor<é>",
        "tensor<f32>😀",
        "()",
        "(tensor<f32>)",
        "tensor<2xxf32>",
    ] {
        let error = parse_type(input).unwrap_err();
        assert!(input.is_char_boundary(error.offset), "{input}: {error}");
    }
}

#[test]
fn unsupported_metadata_errors_instead_of_dropping_it() {
    let ty = Type::from(
        TensorType::scalar(ElementType::F32).with_encoding(Attribute::Opaque {
            dialect: Some("example".into()),
            syntax: "encoding".into(),
        }),
    );
    assert!(matches!(
        print_type(&ty).unwrap_err().kind,
        TextErrorKind::Unsupported(_)
    ));
    assert!(matches!(
        print_type(&Type::String).unwrap_err().kind,
        TextErrorKind::Unsupported(_)
    ));
    assert!(parse_type("tensor<f32, #unknown.encoding>").is_err());
}

#[test]
fn deeply_nested_type_printing_uses_an_explicit_stack() {
    let mut ty = Type::from(TensorType::scalar(ElementType::F32));
    for _ in 0..100 {
        ty = Type::Tuple(vec![ty]);
    }
    let printed = print_type(&ty).unwrap();
    assert_eq!(parse_type(&printed).unwrap(), ty);
}

#[test]
fn adversarial_nesting_and_truncations_are_bounded() {
    let input = format!("{}tensor<f32>{}", "tuple<".repeat(129), ">".repeat(129));
    assert_eq!(
        parse_type(&input).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    let mut ty = Type::from(TensorType::scalar(ElementType::F32));
    for _ in 0..129 {
        ty = Type::Tuple(vec![ty]);
    }
    assert_eq!(
        print_type(&ty).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    for input in [
        "tensor<2x!quant.uniform<i8<-127:127>:f32:0, {0.25, 0.5}>, #stablehlo.bounds<?>>",
        "(tuple<tensor<?xf32>, !stablehlo.token>) -> (memref<3xf64, strided<[1], offset: ?>>)",
    ] {
        for (index, _) in input.char_indices().skip(1) {
            let _ = parse_type(&input[..index]);
        }
    }
}
