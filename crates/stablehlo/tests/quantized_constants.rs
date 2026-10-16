use stablehlo::text::{parse_module, print_module};

fn constant(value_type: &str, result_type: &str, values: &str) -> String {
    format!(
        r#"
      func.func @constant() -> {result_type} {{
        %c = "stablehlo.constant"() {{value = dense<{values}> : {value_type}}}
          : () -> {result_type}
        return %c : {result_type}
      }}
    "#
    )
}

#[test]
fn quantized_constants_use_the_exact_storage_tensor_type() {
    for (storage, output, values) in [
        (
            "tensor<2xi8>",
            "tensor<2x!quant.uniform<i8:f32, 2.0:15>>",
            "[1, 2]",
        ),
        (
            "tensor<2xui8>",
            "tensor<2x!quant.uniform<ui8:f32, 34.0:16>>",
            "[1, 255]",
        ),
        (
            "tensor<2xi8>",
            "tensor<2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>",
            "[-128, 127]",
        ),
        (
            "tensor<2xi8>",
            "tensor<2x!quant.uniform<i8<-127:127>:f32, 0.5>>",
            "[-128, 127]",
        ),
    ] {
        let source = constant(storage, output, values);
        let module = parse_module(&source).unwrap_or_else(|error| panic!("{source}\n{error}"));
        assert_eq!(
            parse_module(&print_module(&module).unwrap()).unwrap(),
            module
        );
    }
}

#[test]
fn signedness_shape_precision_and_quantized_value_mismatches_are_rejected() {
    let output = "tensor<2x!quant.uniform<i8:f32, 0.5>>";
    for storage in [
        "tensor<2xui8>",
        "tensor<2xi16>",
        "tensor<2xf32>",
        "tensor<3xi8>",
        "tensor<2x!quant.uniform<i8:f32, 0.5>>",
    ] {
        assert!(
            parse_module(&constant(storage, output, "1")).is_err(),
            "{storage}"
        );
    }
    assert!(parse_module(&constant("tensor<2xi8>", "tensor<2xf32>", "1")).is_err());
}

#[test]
fn quantized_storage_payloads_require_exact_encoding_equality() {
    let plain = "tensor<2xi8>";
    let encoded = "tensor<2xi8, #stablehlo.bounds<?>>";
    let output = "tensor<2x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<?>>";
    assert!(parse_module(&constant(encoded, output, "1")).is_ok());
    assert!(parse_module(&constant(plain, output, "1")).is_err());
    assert!(
        parse_module(&constant(
            encoded,
            "tensor<2x!quant.uniform<i8:f32, 0.5>>",
            "1"
        ))
        .is_err()
    );
}

#[test]
fn sparse_quantized_constants_also_use_storage_typed_payloads() {
    let source = r#"
      func.func @sparse() -> tensor<2x!quant.uniform<i8:f32, 0.5>> {
        %c = "stablehlo.constant"() {
          value = sparse<[[1]], [7]> : tensor<2xi8>
        } : () -> tensor<2x!quant.uniform<i8:f32, 0.5>>
        return %c : tensor<2x!quant.uniform<i8:f32, 0.5>>
      }
    "#;
    let module = parse_module(source).unwrap();
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    assert!(
        parse_module(&source.replace(
            "sparse<[[1]], [7]> : tensor<2xi8>",
            "sparse<[[1]], [7]> : tensor<2xui8>"
        ))
        .is_err()
    );
}
