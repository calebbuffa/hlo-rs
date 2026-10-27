use stablehlo::mlir::{TextErrorKind, export, import};

use stablehlo::pliron::context::Context;

fn roundtrip(source: &str) -> String {
    let mut ctx = Context::new();
    let module = import(&mut ctx, source).unwrap();
    let printed = export(&ctx, &module).unwrap();
    let mut fresh = Context::new();
    let reparsed =
        import(&mut fresh, &printed).unwrap_or_else(|error| panic!("{printed}\n{error}"));
    assert_eq!(export(&fresh, &reparsed).unwrap(), printed);
    printed
}

#[test]
fn generic_native_ssa_and_discardable_metadata_roundtrip() {
    let printed = roundtrip(
        r#"
        module attributes {test.name = "native"} {
          func.func @main(%arg: tensor<2xf32>) -> tensor<2xf32>
              attributes {test.function = unit, test.metadata = {"spaced key" = "kept"}} {
            %c = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
            %sum = "stablehlo.add"(%arg, %c) {test.note = "exact"} :
                (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            return %sum : tensor<2xf32>
          }
        }
    "#,
    );
    assert!(printed.contains("\"stablehlo.add\""));
    assert!(printed.contains("test.note = \"exact\""));
    assert!(printed.contains("test.function"));
    assert!(printed.contains("\"spaced key\" = \"kept\""));
    assert!(printed.contains("test.name = \"native\""));
}

#[test]
fn forward_function_calls_and_result_groups_roundtrip() {
    roundtrip(
        r#"
      module {
        func.func @main(%a: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
          %r:2 = func.call @callee(%a) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
          return %r#0, %r#1 : tensor<i32>, tensor<i32>
        }
        func.func @callee(%a: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
          return %a, %a : tensor<i32>, tensor<i32>
        }
      }
    "#,
    );
}

#[test]
fn empty_zero_result_and_wide_constant_modules_roundtrip() {
    roundtrip("module {}");
    roundtrip("func.func @main() { return }");
    for (ty, value) in [
        ("tensor<2xi1>", "[true, false]"),
        ("tensor<2xcomplex<f64>>", "[(1.0, -1.0), (2.0, 3.0)]"),
        ("tensor<0xf32>", ""),
        ("tensor<2xf64>", "0x7FF0000000000001"),
    ] {
        roundtrip(&format!(
            "module {{ func.func @main() -> {ty} {{ %c = stablehlo.constant dense<{value}> : {ty} return %c : {ty} }} }}"
        ));
    }
}

#[test]
fn invalid_source_preserves_existing_modules_and_context_recovers() {
    let mut ctx = Context::new();
    let existing = import(&mut ctx, "module { func.func @kept() { return } }").unwrap();
    let before = export(&ctx, &existing).unwrap();
    for source in [
        "module { func.func @main() { %a =",
        "module { func.func @main() { %a = stablehlo.constant dense<1> : tensor<i32> %a = stablehlo.constant dense<2> : tensor<i32> return } }",
        "module { func.func @main() { return %missing : tensor<i32> } }",
        "module { func.func @main(%a: tensor<f32>) -> tensor<i32> { return %a : tensor<f32> } }",
        "module { func.func @main(%a: tensor<f32>) -> tensor<f32> { %b = \"stablehlo.add\"(%a) : (tensor<f32>) -> tensor<f32> return %b : tensor<f32> } }",
        "module { func.func @main() { return } func.func @main() { return } }",
        "module { func.func @main() { %a = func.call @missing() : () -> tensor<i32> return } }",
        "module { func.func @main(%a: tensor<f32>) { \"stablehlo.negate\"(%a) {unknown = unit} : (tensor<f32>) -> tensor<f32> return } }",
        "module { func.func @main() -> tensor<i128> { %a = stablehlo.constant dense<1> : tensor<i128> return %a : tensor<i128> } }",
    ] {
        assert!(import(&mut ctx, source).is_err(), "{source}");
        assert_eq!(export(&ctx, &existing).unwrap(), before);
    }
    import(&mut ctx, &before).unwrap();
}

#[test]
fn unknown_ssa_has_a_typed_failure_and_source_offset() {
    let mut ctx = Context::new();
    let source = "module { func.func @main() { return %missing : tensor<i32> } }";
    let error = import(&mut ctx, source).err().unwrap();
    assert!(matches!(error.kind, TextErrorKind::UnknownValue(ref name) if name == "missing"));
    assert_eq!(&source[error.offset..error.offset + 6], "return");
}

#[test]
fn structured_locations_preserve_native_names_files_callsites_and_metadata() {
    for location in [
        r#"unknown"#,
        r#""file.mlir":1:2"#,
        r#""name""#,
        r#""named"("file.mlir":3:4)"#,
        r#"callsite("callee.mlir":3:4 at "caller.mlir":5:6)"#,
        r#"fused<{"metadata key" = "value"}>["file.mlir":1:2, "named", unknown]"#,
        r#""escaped\22\5C\0A"("file.mlir":0:0)"#,
    ] {
        let printed = roundtrip(&format!(
            r#"module {{ func.func @main() -> tensor<i32> {{
                %c = stablehlo.constant dense<1> : tensor<i32> loc({location})
                return %c : tensor<i32> loc({location})
            }} loc({location}) }} loc({location})"#
        ));
        if location != "unknown" {
            assert_eq!(printed.matches(" loc(").count(), 4, "{printed}");
        }
    }
    let mut ctx = Context::new();
    for location in [
        r#""file":-1:0"#,
        r#""file":2147483648:0"#,
        "fused[unknown",
        "callsite(unknown)",
    ] {
        assert!(import(&mut ctx, &format!("module {{}} loc({location})")).is_err());
    }
}

#[test]
fn standard_attribute_names_and_widths_roundtrip_without_native_leakage() {
    for (inputs, body, result, value) in [
        (
            "%a: tensor<2x3xf32>",
            r#"%r = "stablehlo.transpose"(%a) {permutation = array<i64: 1, 0>} : (tensor<2x3xf32>) -> tensor<3x2xf32>"#,
            "tensor<3x2xf32>",
            "%r",
        ),
        (
            "%a: tensor<2xf32>",
            r#"%r = "stablehlo.broadcast_in_dim"(%a) {broadcast_dimensions = array<i64: 1>} : (tensor<2xf32>) -> tensor<3x2xf32>"#,
            "tensor<3x2xf32>",
            "%r",
        ),
        (
            "%a: tensor<2x3xf32>",
            r#"%r = "stablehlo.reverse"(%a) {dimensions = array<i64: 0, 1>} : (tensor<2x3xf32>) -> tensor<2x3xf32>"#,
            "tensor<2x3xf32>",
            "%r",
        ),
        (
            "%a: tensor<4xf32>",
            r#"%r = "stablehlo.slice"(%a) {start_indices = array<i64: 0>, limit_indices = array<i64: 4>, strides = array<i64: 2>} : (tensor<4xf32>) -> tensor<2xf32>"#,
            "tensor<2xf32>",
            "%r",
        ),
        (
            "%a: tensor<2xf32>",
            r#"%r = "stablehlo.concatenate"(%a, %a) {dimension = 0 : i64} : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>"#,
            "tensor<4xf32>",
            "%r",
        ),
        (
            "",
            r#"%r = "stablehlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<4xi32>"#,
            "tensor<4xi32>",
            "%r",
        ),
        (
            "%a: tuple<tensor<i32>, tensor<f32>>",
            r#"%r = "stablehlo.get_tuple_element"(%a) {index = 1 : i32} : (tuple<tensor<i32>, tensor<f32>>) -> tensor<f32>"#,
            "tensor<f32>",
            "%r",
        ),
        (
            "%a: tensor<2x3xf32>",
            r#"%r = "stablehlo.get_dimension_size"(%a) {dimension = 1 : i64} : (tensor<2x3xf32>) -> tensor<i32>"#,
            "tensor<i32>",
            "%r",
        ),
        (
            "%a: tensor<2xf32>",
            r#"%r = "stablehlo.reduce_precision"(%a) {exponent_bits = 5 : i32, mantissa_bits = 10 : i32} : (tensor<2xf32>) -> tensor<2xf32>"#,
            "tensor<2xf32>",
            "%r",
        ),
        (
            "%a: tensor<2xi32>",
            r#"%r = "stablehlo.compare"(%a, %a) {comparison_direction = #stablehlo.comparison_direction<LT>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>"#,
            "tensor<2xi1>",
            "%r",
        ),
        (
            "%a: tensor<2xcomplex<f32>>",
            r#"%r = "stablehlo.fft"(%a) {fft_type = #stablehlo.fft_type<FFT>, fft_length = array<i64: 2>} : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>"#,
            "tensor<2xcomplex<f32>>",
            "%r",
        ),
    ] {
        let source = format!(
            "module {{ func.func @main({inputs}) -> {result} {{ {body} return {value} : {result} }} }}"
        );
        let printed = roundtrip(&source);
        assert!(!printed.contains("stablehlo_"), "{printed}");
    }
}
