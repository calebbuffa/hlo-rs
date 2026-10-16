use stablehlo::text::{TextErrorKind, parse_module, print_module};

fn round_trip(input: &str) -> String {
    let module = parse_module(input).unwrap_or_else(|error| panic!("{input}\n{error}"));
    let canonical = print_module(&module).unwrap();
    let reparsed = parse_module(&canonical).unwrap();
    assert_eq!(print_module(&reparsed).unwrap(), canonical);
    canonical
}

#[test]
fn while_arguments_are_bound_in_both_regions_with_outer_captures() {
    round_trip(
        r#"module {
          func.func @main(%initial: tensor<i64>, %limit: tensor<i64>) -> tensor<i64> {
            %r = stablehlo.while(%iter = %initial) : tensor<i64>
                attributes {consumer.loop = true}
              cond {
                %test = stablehlo.compare LT, %iter, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
                stablehlo.return %test : tensor<i1>
              } do {
                %one = stablehlo.constant dense<1> : tensor<i64>
                %next = stablehlo.add %iter, %one : tensor<i64>
                stablehlo.return %next : tensor<i64>
              }
            return %r : tensor<i64>
          }
        }"#,
    );
}

#[test]
fn while_zero_and_multiple_iteration_arguments_round_trip() {
    round_trip(
        r#"module {
          func.func @main() {
            stablehlo.while () cond {
              %false = stablehlo.constant dense<false> : tensor<i1>
              stablehlo.return %false : tensor<i1>
            } do {
              stablehlo.return
            }
            return
          }
        }"#,
    );
    round_trip(
        r#"module {
          func.func @main(%x: tensor<i32>, %y: tensor<f32>) -> (tensor<i32>, tensor<f32>) {
            %r:2 = stablehlo.while(%a = %x, %b = %y) : tensor<i32>, tensor<f32>
              cond {
                %false = stablehlo.constant dense<false> : tensor<i1>
                stablehlo.return %false : tensor<i1>
              } do {
                stablehlo.return %a, %b : tensor<i32>, tensor<f32>
              }
            return %r#0, %r#1 : tensor<i32>, tensor<f32>
          }
        }"#,
    );
}

#[test]
fn compact_reducers_build_the_same_typed_region_as_explicit_reducers() {
    for (operation, element) in [
        ("add", "f32"),
        ("multiply", "f32"),
        ("maximum", "f32"),
        ("minimum", "f32"),
        ("and", "i1"),
        ("or", "i1"),
        ("xor", "i1"),
    ] {
        let shape = format!("tensor<2x3x{element}>");
        let scalar = format!("tensor<{element}>");
        let output = format!("tensor<2x{element}>");
        let prefix =
            format!("module {{ func.func @main(%x: {shape}, %init: {scalar}) -> {output} {{");
        let compact = format!(
            "{prefix} %r = stablehlo.reduce(%x init: %init) applies stablehlo.{operation}
                across dimensions = [1] : ({shape}, {scalar}) -> {output}
                return %r : {output} }} }}"
        );
        let explicit = format!(
            "{prefix} %r = stablehlo.reduce(%x init: %init)
                across dimensions = [1] : ({shape}, {scalar}) -> {output}
                reducer(%a: {scalar}, %b: {scalar}) {{
                  %s = stablehlo.{operation} %a, %b : {scalar}
                  stablehlo.return %s : {scalar}
                }} return %r : {output} }} }}"
        );
        assert_eq!(round_trip(&compact), round_trip(&explicit));
    }
}

#[test]
fn region_based_reducers_reorder_input_init_pairs_to_the_canonical_operand_order() {
    round_trip(
        r#"module {
          func.func @main(%x: tensor<2xf32>, %y: tensor<2xi32>,
                          %i: tensor<f32>, %j: tensor<i32>) -> (tensor<f32>, tensor<i32>) {
            %r:2 = stablehlo.reduce(%x init: %i), (%y init: %j)
                across dimensions = [0] {consumer.reduce = "kept"}
                : (tensor<2xf32>, tensor<2xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>)
                reducer(%a: tensor<f32>, %a_init: tensor<f32>)
                       (%b: tensor<i32>, %b_init: tensor<i32>) {
                  %sum = stablehlo.add %a, %a_init : tensor<f32>
                  %max = stablehlo.maximum %b, %b_init : tensor<i32>
                  stablehlo.return %sum, %max : tensor<f32>, tensor<i32>
                }
            return %r#0, %r#1 : tensor<f32>, tensor<i32>
          }
        }"#,
    );
}

#[test]
fn invalid_while_bindings_types_regions_and_escaping_values_are_rejected() {
    for operation in [
        "stablehlo.while(%a = %x) : tensor<f32> cond {} do {}",
        "stablehlo.while(%a = %x) : tensor<f32>, tensor<f32> cond {} do {}",
        "stablehlo.while(%a = %x, %a = %x) : tensor<f32>, tensor<f32> cond {} do {}",
        "stablehlo.while(%a = %unknown) : tensor<f32> cond {} do {}",
        "stablehlo.while(%a = %x) : tensor<f32> cond { ^bb(%b: tensor<f32>): } do {}",
        "stablehlo.while(%a = %x) : tensor<f32> cond { stablehlo.return %a : tensor<f32> } do { stablehlo.return %a : tensor<f32> }",
    ] {
        let input = format!(
            "module {{ func.func @main(%x: tensor<f32>) -> tensor<f32> {{
                %r = {operation} return %r : tensor<f32> }} }}"
        );
        assert!(parse_module(&input).is_err(), "{input}");
    }
    let input = r#"module {
      func.func @main(%x: tensor<f32>) -> tensor<f32> {
        %r = stablehlo.while(%a = %x) : tensor<f32> cond {
          %false = stablehlo.constant dense<false> : tensor<i1>
          stablehlo.return %false : tensor<i1>
        } do {
          stablehlo.return %a : tensor<f32>
        }
        return %a : tensor<f32>
      }
    }"#;
    assert!(matches!(
        parse_module(input).unwrap_err().kind,
        TextErrorKind::UnknownValue(_)
    ));
}

#[test]
fn compact_reducers_reject_noncommutative_or_wrong_dialect_operations() {
    for name in [
        "stablehlo.subtract",
        "stablehlo.divide",
        "arith.addf",
        "\"stablehlo.not\"",
    ] {
        let input = format!(
            "module {{ func.func @main(%x: tensor<2xf32>, %i: tensor<f32>) -> tensor<f32> {{
              %r = stablehlo.reduce(%x init: %i) applies {name}
                across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
              return %r : tensor<f32>
            }} }}"
        );
        assert!(parse_module(&input).is_err(), "{input}");
    }
}

#[test]
fn convolution_window_fields_are_canonical_attributes_not_parser_only_metadata() {
    for window in [
        "",
        "stride = [1, 1]",
        "pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 1]",
        "reverse = [-1, 0], rhs_dilate = [1, 1], pad = [[0, 0], [0, 0]], stride = [1, 1],",
    ] {
        round_trip(&format!(
            "module {{ func.func @main(%x: tensor<1x4x4x1xf32>, %k: tensor<2x2x1x1xf32>) -> tensor<1x3x3x1xf32> {{
              %r = stablehlo.convolution(%x, %k)
                dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f],
                window = {{{window}}}
                {{feature_group_count = 1 : i64, batch_group_count = 1 : i64}}
                : (tensor<1x4x4x1xf32>, tensor<2x2x1x1xf32>) -> tensor<1x3x3x1xf32>
              return %r : tensor<1x3x3x1xf32>
            }} }}"
        ));
    }
}

#[test]
fn malformed_or_duplicate_convolution_fields_are_rejected() {
    for window in [
        "pad = [[0]]",
        "pad = [[0, 0, 0]]",
        "stride = [1, 1], stride = [1, 1]",
        "reverse = [true, false]",
        "stride = [0, 1]",
        "unknown = [1, 1]",
    ] {
        let input = format!(
            "module {{ func.func @main(%x: tensor<1x4x4x1xf32>, %k: tensor<2x2x1x1xf32>) -> tensor<1x3x3x1xf32> {{
              %r = stablehlo.convolution(%x, %k)
                dim_numbers = [b, 0, 1, f] x [0, 1, i, o] -> [b, 0, 1, f], window = {{{window}}}
                {{feature_group_count = 1 : i64, batch_group_count = 1 : i64}}
                : (tensor<1x4x4x1xf32>, tensor<2x2x1x1xf32>) -> tensor<1x3x3x1xf32>
              return %r : tensor<1x3x3x1xf32>
            }} }}"
        );
        assert!(parse_module(&input).is_err(), "{input}");
    }
}

#[test]
fn composite_custom_syntax_keeps_the_resolved_decomposition_signature() {
    round_trip(
        r#"module {
          func.func private @decomposition(%x: tensor<f32>) -> tensor<f32> {
            return %x : tensor<f32>
          }
          func.func @main(%x: tensor<f32>) -> tensor<f32> {
            %r = stablehlo.composite "consumer.identity" %x
              {composite_attributes = {}, decomposition = @decomposition}
              : (tensor<f32>) -> tensor<f32>
            return %r : tensor<f32>
          }
        }"#,
    );
}

#[test]
fn truncated_custom_regions_and_deep_custom_nesting_report_errors_without_panics() {
    let input = r#"module { func.func @main(%x: tensor<2xf32>, %init: tensor<f32>) -> tensor<f32> {
      %r = stablehlo.reduce(%x init: %init) across dimensions = [0]
          : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
          reducer(%a: tensor<f32>, %b: tensor<f32>) {
            %r = stablehlo.add %a, %b : tensor<f32>
            stablehlo.return %r : tensor<f32>
          }
      return %r : tensor<f32>
    }}"#;
    for (offset, _) in input.char_indices().skip(1) {
        assert!(parse_module(&input[..offset]).is_err(), "prefix {offset}");
    }
    let mut nested = String::from("module { func.func @main() { ");
    for _ in 0..150 {
        nested.push_str("stablehlo.while () cond { ");
    }
    nested.push_str("stablehlo.return ");
    for _ in 0..150 {
        nested.push_str("} do { stablehlo.return } ");
    }
    nested.push_str("return } }");
    assert!(matches!(
        parse_module(&nested).unwrap_err().kind,
        TextErrorKind::NestingLimit
    ));
    for operation in [
        "%r = stablehlo.reduce(%x init: %i) across dimensions = []
            : (tensor<f32>, tensor<f32>) -> tensor<f32>
            reducer(%a: tensor<f32>, %b: tensor<f32>) { ",
        "%r = stablehlo.composite \"consumer.nested\" ({ ",
    ] {
        let input = format!(
            "module {{ func.func @main(%x: tensor<f32>, %i: tensor<f32>) {{ {}",
            operation.repeat(150),
        );
        assert!(matches!(
            parse_module(&input).unwrap_err().kind,
            TextErrorKind::NestingLimit
        ));
    }
}
