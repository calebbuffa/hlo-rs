use stablehlo::text::{parse_module, print_module};

#[test]
fn top_level_functions_form_the_same_canonical_module_as_explicit_assembly() {
    let functions = r#"
        func.func private @external(tensor<2xf32>) -> tensor<2xf32>
        func.func @main(%x: tensor<2xf32>) -> tensor<2xf32> {
          %y = func.call @external(%x) : (tensor<2xf32>) -> tensor<2xf32>
          return %y : tensor<2xf32>
        }
    "#;
    let implicit = parse_module(functions).unwrap();
    let explicit = parse_module(&format!("module {{ {functions} }}")).unwrap();
    assert_eq!(implicit, explicit);
    assert_eq!(
        parse_module(&print_module(&implicit).unwrap()).unwrap(),
        implicit
    );
}

#[test]
fn alias_definitions_can_be_interleaved_with_top_level_operations() {
    let source = r#"
        !t = tensor<f32>
        func.func @first(%x: !t loc(#source)) -> !t {
          return %x : !t loc(#source)
        }
        #source = loc("definition")
        #tag = "second"
        func.func @second(%x: !t) -> !t attributes {consumer.tag = #tag} {
          return %x : !t
        } loc(#function)
        #function = loc("second function")
    "#;
    let module = parse_module(source).unwrap();
    assert_eq!(module.computations().len(), 2);
    assert!(module.computations()[0].return_location().is_some());
    assert!(module.computations()[1].location().is_some());
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    assert!(
        parse_module(
            "func.func @a() attributes {consumer.tag = #tag} { return } #tag = \"not yet defined\""
        )
        .is_err()
    );
}

#[test]
fn empty_comment_and_alias_only_files_form_empty_modules() {
    let empty = parse_module("module {}").unwrap();
    for source in [
        "",
        " \r\n",
        "// comment",
        "// comment\n",
        "!t = tensor<f32>",
        "#tag = true",
    ] {
        assert_eq!(parse_module(source).unwrap(), empty, "{source}");
    }
}

#[test]
fn explicit_module_identity_and_locations_are_not_hidden_by_an_extra_wrapper() {
    let module = parse_module(r#"#loc = loc("source") module @named {} loc(#loc)"#).unwrap();
    assert_eq!(module.name(), "named");
    assert!(module.location().is_some());
    assert!(module.computations().is_empty());
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
}

#[test]
fn implicit_module_still_checks_symbols_signatures_and_top_level_kinds() {
    for source in [
        "func.func @f() { return } func.func @f() { return }",
        "func.func @f() { func.call @missing() : () -> () return }",
        "\"stablehlo.create_token\"() : () -> !stablehlo.token",
        "module {} module {}",
        "module {} func.func @f() { return }",
        "func.func @f() { return } trailing",
    ] {
        assert!(parse_module(source).is_err(), "{source}");
    }
}
