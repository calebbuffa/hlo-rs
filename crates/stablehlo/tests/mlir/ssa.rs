use stablehlo::mlir::{export, import};

use stablehlo::pliron::context::Context;

#[test]
fn default_result_zero_and_explicit_group_zero_are_the_same_value() {
    for source in [
        r#"module { func.func @main(%0: tensor<i32>) -> tensor<i32> {
            %v:1 = "stablehlo.negate"(%0#0) : (tensor<i32>) -> tensor<i32>
            return %v : tensor<i32>
        } }"#,
        r#"module { func.func @main() -> tensor<i32> {
            %c = stablehlo.constant dense<1> : tensor<i32>
            return %c#0 : tensor<i32>
        } }"#,
        r#"module {
            func.func @main(%a: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
                %pair:2 = func.call @callee(%a) : (tensor<i32>) -> (tensor<i32>, tensor<i32>)
                return %pair, %pair#1 : tensor<i32>, tensor<i32>
            }
            func.func @callee(%a: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
                return %a, %a#0 : tensor<i32>, tensor<i32>
            }
        }"#,
    ] {
        let mut ctx = Context::new();
        let module = import(&mut ctx, source).unwrap();
        let printed = export(&ctx, &module).unwrap();
        let mut fresh = Context::new();
        let reparsed = import(&mut fresh, &printed).unwrap();
        assert_eq!(export(&fresh, &reparsed).unwrap(), printed);
    }
}

#[test]
fn malformed_names_and_counts_do_not_create_values() {
    let mut ctx = Context::new();
    for result in ["%0abc", "%a#0", "%a:true", "%a:0x1", "%a:0", "%a:"] {
        let source = format!(
            "module {{ func.func @main() {{ {result} = stablehlo.constant dense<1> : tensor<i32> return }} }}"
        );
        assert!(import(&mut ctx, &source).is_err(), "{result}");
    }
    let source = "module { func.func @main() { %a:1 = stablehlo.constant dense<1> : tensor<i32> %a = stablehlo.constant dense<2> : tensor<i32> return } }";
    assert!(import(&mut ctx, source).is_err());
    import(&mut ctx, "module { func.func @recovered() { return } }").unwrap();
}

#[test]
fn quoted_function_names_do_not_become_invalid_bare_identifiers() {
    for name in ["0", "0name", "with-dash", "with space", r"with\22quote"] {
        let mut ctx = Context::new();
        let source = format!(r#"module {{ func.func @"{name}"() {{ return }} }}"#);
        let module = import(&mut ctx, &source).unwrap();
        let printed = export(&ctx, &module).unwrap();
        let mut fresh = Context::new();
        let reparsed = import(&mut fresh, &printed).unwrap();
        assert_eq!(export(&fresh, &reparsed).unwrap(), printed);
    }
}
