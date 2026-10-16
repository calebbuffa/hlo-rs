use stablehlo::text::{TextErrorKind, parse_module, print_module};

fn canonical(source: &str) -> String {
    let module = parse_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    let printed = print_module(&module).unwrap();
    assert_eq!(
        print_module(&parse_module(&printed).unwrap()).unwrap(),
        printed
    );
    printed
}

#[test]
fn type_and_attribute_aliases_resolve_into_the_same_canonical_module() {
    let aliased = r#"
      !scalar = f32
      #bounds = #stablehlo.bounds<5, ?>
      !tensor = tensor<?x3x!scalar, #bounds>
      #tag = "metadata"
      #attrs = {consumer.label = #tag}
      module attributes #attrs {
        func.func @main(%x: !tensor) -> !tensor {
          %r = stablehlo.negate %x : !tensor
          return %r : !tensor
        }
      }
    "#;
    let explicit = r#"
      module attributes {consumer.label = "metadata"} {
        func.func @main(%x: tensor<?x3xf32, #stablehlo.bounds<5, ?>>)
          -> tensor<?x3xf32, #stablehlo.bounds<5, ?>> {
          %r = stablehlo.negate %x : tensor<?x3xf32, #stablehlo.bounds<5, ?>>
          return %r : tensor<?x3xf32, #stablehlo.bounds<5, ?>>
        }
      }
    "#;
    assert_eq!(canonical(aliased), canonical(explicit));
}

#[test]
fn aliases_work_in_generic_signatures_metadata_literals_and_nested_regions() {
    canonical(
        r#"
      !t = tensor<f32>
      !p = tensor<i1>
      #zero = dense<0.0> : !t
      #name = "consumer"
      #metadata = {nested = [#name, !t, {value = #zero}]}
      module {
        func.func @main(%x: !t) -> !t attributes {consumer.metadata = #metadata} {
          %r = stablehlo.while(%iter = %x) : !t cond {
            %false = "stablehlo.constant"() {value = dense<false> : !p} : () -> !p
            stablehlo.return %false : !p
          } do {
            %zero = "stablehlo.constant"() {value = #zero} : () -> !t
            %sum = stablehlo.add %iter, %zero : !t
            stablehlo.return %sum : !t
          }
          return %r : !t
        }
      }
      #unused = "accepted trailing definition"
      !unused = tensor<i64>
    "#,
    );
}

#[test]
fn prefixed_identifier_grammar_and_separate_namespaces_match_mlir() {
    for name in ["0", "123", "$t", "_t", "-t", "t-2", "t$2"] {
        canonical(&format!(
            "!{name} = tensor<f32>\n#{name} = \"tag\"\n\
             module attributes {{consumer.label = #{name}}} {{\
             func.func @main(%x: !{name}) -> !{name} {{ return %x : !{name} }} }}"
        ));
    }
    for source in [
        "!dialect.type = tensor<f32> module {}",
        "#dialect.attr = true module {}",
        "!0a = tensor<f32> module {}",
        "! t = tensor<f32> module {}",
        "# tag = true module {}",
    ] {
        assert!(parse_module(source).is_err(), "{source}");
    }
}

#[test]
fn redefinitions_unknown_forward_and_recursive_references_are_explicit_errors() {
    for source in [
        "!t = tensor<f32> !t = tensor<i32> module {}",
        "#a = true #a = false module {}",
        "!t = tensor<f32> module {} !t = tensor<i32>",
    ] {
        assert!(matches!(
            parse_module(source).unwrap_err().kind,
            TextErrorKind::DuplicateField(_)
        ));
    }
    for source in [
        "!a = !missing module {}",
        "#a = #missing module {}",
        "!a = !b !b = tensor<f32> module {}",
        "!a = !a module {}",
        "#a = [#a] module {}",
        "module attributes {consumer.tag = #a} {} #a = true",
        "module { func.func @main(%x: !missing) { return } }",
    ] {
        let error = parse_module(source).unwrap_err();
        assert!(
            matches!(error.kind, TextErrorKind::UnknownAlias(_)),
            "{source}: {error}"
        );
        assert!(error.offset <= source.len());
    }
    assert!(
        parse_module("!t = tensor<2xf32> module { func.func @f(%x: tensor<2x!t>) { return } }")
            .is_err()
    );
}

#[test]
fn scalar_aliases_work_in_complex_memref_and_quantized_element_components() {
    canonical(
        r#"
      !f = f32
      !i = i8
      !c = complex<!f>
      !buffer = memref<3x!f>
      !q = tensor<3x!quant.uniform<!i:!f, 0.25:1>>
      module {
        func.func private @external(!buffer, tensor<2x!c>, !q)
      }
    "#,
    );
}

#[test]
fn exponential_alias_expansion_is_bounded_for_types_and_attributes() {
    for (prefix, base, constructor) in [("!", "tensor<f32>", "tuple"), ("#", "true", "")] {
        let mut source = format!("{prefix}a0 = {base}\n");
        for index in 1..30 {
            let previous = index - 1;
            let pair = format!("{prefix}a{previous}, {prefix}a{previous}");
            let value = if constructor.is_empty() {
                format!("[{pair}]")
            } else {
                format!("tuple<{pair}>")
            };
            source.push_str(&format!("{prefix}a{index} = {value}\n"));
        }
        source.push_str("module {}");
        assert_eq!(
            parse_module(&source).unwrap_err().kind,
            TextErrorKind::ResourceLimit
        );
    }
}

#[test]
fn alias_reference_chains_cannot_bypass_the_parser_nesting_limit() {
    for (prefix, base) in [("!", "tensor<f32>"), ("#", "true")] {
        let mut source = format!("{prefix}a0 = {base}\n");
        for index in 1..140 {
            source.push_str(&format!("{prefix}a{index} = {prefix}a{}\n", index - 1));
        }
        source.push_str("module {}");
        assert_eq!(
            parse_module(&source).unwrap_err().kind,
            TextErrorKind::NestingLimit
        );
    }
}

#[test]
fn long_alias_chains_preserve_dialect_terminals_and_combined_nesting_budgets() {
    for (prefix, base) in [
        ("!", "tensor<f32>"),
        ("!", "!quant.uniform<i8:f32, 0.5:0>"),
        ("#", "#complex<number<:f32 1.0, -0.0>>"),
        ("#", "#sparse_tensor<encoding<{map=(i)->(i:compressed)}>>"),
        ("#", "#stablehlo<precision HIGH>"),
        ("#", "{nested=[true]}"),
    ] {
        let mut source = format!("{prefix}a0 = {base}\n");
        for index in 1..=80 {
            source.push_str(&format!("{prefix}a{index} = {prefix}a{}\n", index - 1));
        }
        let definitions = source.clone();
        source.push_str(&format!(
            "module attributes {{consumer.metadata = {prefix}a80}} {{}}"
        ));
        canonical(&source);
        if prefix == "#" {
            let nested = format!("{}#a80{}", "[".repeat(50), "]".repeat(50));
            let source =
                format!("{definitions}module attributes {{consumer.metadata = {nested}}} {{}}");
            assert_eq!(
                parse_module(&source).unwrap_err().kind,
                TextErrorKind::NestingLimit
            );
        }
    }
}
