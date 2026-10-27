use stablehlo::mlir::TextErrorKind;
use stablehlo::mlir::assembly::{parse_type, print_type};
use stablehlo::{
    pliron::context::Context,
    types::{ElementType, TensorType, primitive_element},
};

#[test]
fn native_types_roundtrip_in_fresh_contexts() {
    for source in [
        "index",
        "i0",
        "i128",
        "si7",
        "ui65",
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
        "complex<i128>",
        "tensor<f32>",
        "tensor<2x?xcomplex<f64>>",
        "tensor<*xf32>",
        "tuple<>",
        "tuple<tensor<4xi8>, !stablehlo.token>",
        "(tensor<2xf32>, !stablehlo.token) -> (tensor<2xf32>)",
        "() -> ()",
        "vector<4x[8]xf32>",
        "tensor<?x4xf32, #stablehlo.type_extensions<bounds = [16, ?]>>",
        "!quant.uniform<i8:f32, 0.25:3>",
        "!quant.uniform<u8<1:254>:f32, 0x3FD0000000000000:3>",
        "tensor<2x3x!quant.uniform<i8:f32:0, {0.5:2, 0.25:3}>>",
        "!quant.any<i8>",
        "!quant.any<u8:f32>",
        "none",
        "!quant.calibrated<f32<-1.0:2.0>>",
        "memref<?x4xf32>",
        "memref<*xf32, 3>",
        "memref<4x8xf32, strided<[8, 1], offset: ?>>",
        "memref<4x8xf32, affine_map<(d0, d1) -> (d1, d0)>, 3>",
    ] {
        let ctx = Context::new();
        let ty = parse_type(&ctx, source).unwrap_or_else(|error| panic!("{source}: {error}"));
        let printed = print_type(&ctx, ty).unwrap();
        assert_eq!(parse_type(&ctx, &printed).unwrap(), ty, "{source}");
        let fresh = Context::new();
        let reparsed = parse_type(&fresh, &printed).unwrap();
        assert_eq!(print_type(&fresh, reparsed).unwrap(), printed, "{source}");
    }
}

#[test]
fn bounds_are_native_encoding_not_discarded() {
    let ctx = Context::new();
    let ty = parse_type(
        &ctx,
        "tensor<?x4xf32, #stablehlo.type_extensions<bounds = [16, ?]>>",
    )
    .unwrap();
    let ty = ty.deref(&ctx);
    let tensor = ty.downcast_ref::<TensorType>().unwrap();
    assert_eq!(tensor.dimensions, [-1, 4]);
    assert_eq!(tensor.bounds(), Some([16, -1].as_slice()));
    assert_eq!(
        primitive_element(&ctx, tensor.element),
        Some(ElementType::F32)
    );
    assert!(
        parse_type(
            &ctx,
            "tensor<4xf32, #stablehlo.type_extensions<bounds = [16]>>"
        )
        .is_err()
    );
}

#[test]
fn mlir_lexical_rules_and_offsets_are_preserved() {
    let ctx = Context::new();
    assert!(parse_type(&ctx, " \0// comment\rtensor<2x // next\n f32> ").is_ok());
    let error = parse_type(&ctx, "f32 f64").unwrap_err();
    assert_eq!(error.offset, 4);
    assert_eq!(error.kind, TextErrorKind::TrailingInput);
    for source in [
        "\u{b}f32",
        "\u{c}f32",
        "tensor<2xf32junk>",
        "tuple<f32,>",
        "complex<tensor<f32>>",
        "vector<?xf32>",
        "i16777216",
        "tensor<2xf32, #unknown.encoding<1>>",
        "tensor<-2xf32>",
        "tensor<2.0xf32>",
        "tensor<2xf32",
        "tensor<2x>",
    ] {
        assert!(parse_type(&ctx, source).is_err(), "{source:?}");
    }
    assert!(parse_type(&ctx, "tensor<2xf32>").is_ok());
}

#[test]
fn excessive_nesting_is_bounded_and_context_recovers() {
    let ctx = Context::new();
    let source = format!("{}f32{}", "tuple<".repeat(200), ">".repeat(200));
    assert_eq!(
        parse_type(&ctx, &source).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    assert!(parse_type(&ctx, "tuple<tensor<2xf32>>").is_ok());
}
