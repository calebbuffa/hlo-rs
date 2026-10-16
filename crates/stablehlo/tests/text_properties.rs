use stablehlo::{
    Attribute,
    text::{TextErrorKind, parse_module, print_module},
};

fn map_operation(properties: &str, after_region: &str) -> String {
    format!(
        r#"
        func.func @map(%input: tensor<2xf32>) -> tensor<2xf32> {{
          %r = "stablehlo.map"(%input) {properties} ({{
          ^bb0(%value: tensor<f32>):
            stablehlo.return %value : tensor<f32>
          }}) {after_region} : (tensor<2xf32>) -> tensor<2xf32>
          return %r : tensor<2xf32>
        }}
    "#
    )
}

#[test]
fn properties_precede_regions_and_discardable_attributes_follow_regions() {
    let module = parse_module(&map_operation(
        "<{dimensions = array<i64: 0>}>",
        r#"{consumer.tag = "kept"}"#,
    ))
    .unwrap();
    let instructions = module.computations()[0].instructions();
    let attributes = instructions[1].operation().attributes().unwrap();
    assert_eq!(attributes["consumer.tag"], Attribute::String("kept".into()));
    assert_eq!(attributes["dimensions"], Attribute::integers(vec![0]));
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    let legacy = parse_module(&map_operation(
        "",
        r#"{dimensions = array<i64: 0>, consumer.tag = "kept"}"#,
    ))
    .unwrap();
    assert_eq!(legacy, module);
}

#[test]
fn property_dictionaries_after_regions_are_not_valid_generic_mlir() {
    assert!(parse_module(&map_operation("", "<{dimensions = array<i64: 0>}>")).is_err());
    assert!(
        parse_module(&map_operation(
            "<{dimensions = array<i64: 0>}>",
            "<{consumer.tag = true}>",
        ))
        .is_err()
    );
}

#[test]
fn properties_and_discardable_attributes_cannot_duplicate_keys() {
    let error = parse_module(&map_operation(
        "<{dimensions = array<i64: 0>}>",
        "{dimensions = array<i64: 0>}",
    ))
    .unwrap_err();
    assert!(matches!(error.kind, TextErrorKind::DuplicateField(ref name) if name == "dimensions"));
}

#[test]
fn empty_module_printing_materializes_its_required_single_block() {
    for source in [
        "",
        "module {}",
        "module @named {}",
        r#"module {} loc("source")"#,
        r#"module attributes {consumer.tag = true} {}"#,
    ] {
        let module = parse_module(source).unwrap();
        let printed = print_module(&module).unwrap();
        assert!(printed.contains("^bb0:"), "{printed}");
        assert_eq!(parse_module(&printed).unwrap(), module);
    }
}

#[test]
fn generic_module_body_requires_one_argument_free_block_not_an_empty_region() {
    for source in [
        r#""builtin.module"() ({}) : () -> ()"#,
        r#""builtin.module"() ({ ^bb0: ^bb1: }) : () -> ()"#,
        r#""builtin.module"() ({ ^bb0(%x: tensor<f32>): }) : () -> ()"#,
    ] {
        assert!(parse_module(source).is_err(), "{source}");
    }
    let module = parse_module(r#""builtin.module"() ({ ^bb0: }) : () -> ()"#).unwrap();
    assert_eq!(module, parse_module("module {}").unwrap());
}
