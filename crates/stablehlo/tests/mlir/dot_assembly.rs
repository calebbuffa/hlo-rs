use stablehlo::mlir::{export, import};

use stablehlo::pliron::context::Context;

const DIMENSIONS: &str = "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>";
const ALGORITHM: &str = "algorithm = #stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>";

fn source(name: &str, attributes: &str) -> String {
    format!(
        "module {{ func.func @main(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {{ %v = \"stablehlo.{name}\"(%a, %b) {{{attributes}}} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32> func.return %v : tensor<2x4xf32> }} }}"
    )
}

#[test]
fn dot_precision_presence_and_enum_values_roundtrip_without_manufactured_defaults() {
    for name in ["dot", "dot_general"] {
        for precision in [
            None,
            Some("[]"),
            Some("[#stablehlo<precision HIGH>]"),
            Some("[#stablehlo<precision DEFAULT>, #stablehlo<precision HIGHEST>]"),
        ] {
            let mut attributes = if name == "dot_general" {
                DIMENSIONS.to_owned()
            } else {
                String::new()
            };
            if let Some(value) = precision {
                if !attributes.is_empty() {
                    attributes.push_str(", ");
                }
                attributes.push_str(&format!("precision_config = {value}"));
            }
            let mut ctx = Context::new();
            let module = import(&mut ctx, &source(name, &attributes)).unwrap();
            let printed = export(&ctx, &module).unwrap();
            assert_eq!(
                printed.contains("precision_config"),
                precision.is_some(),
                "{printed}"
            );
            assert!(!printed.contains("stablehlo_dot_precision"));
            assert!(!printed.contains("stablehlo_compute_dot_precision"));
            let mut fresh = Context::new();
            let restored = import(&mut fresh, &printed).unwrap();
            assert_eq!(export(&fresh, &restored).unwrap(), printed);
        }
    }
}

#[test]
fn dot_algorithm_and_optional_default_precision_preserve_the_native_contract() {
    for precision in [
        "",
        ", precision_config = []",
        ", precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]",
    ] {
        let attributes = format!("{DIMENSIONS}, {ALGORITHM}{precision}");
        let mut ctx = Context::new();
        let module = import(&mut ctx, &source("dot_general", &attributes)).unwrap();
        let printed = export(&ctx, &module).unwrap();
        assert!(printed.contains("algorithm = #stablehlo.dot_algorithm"));
        assert_eq!(printed.contains("precision_config"), !precision.is_empty());
        let mut fresh = Context::new();
        let restored = import(&mut fresh, &printed).unwrap();
        assert_eq!(export(&fresh, &restored).unwrap(), printed);
    }
}

#[test]
fn invalid_dot_metadata_does_not_mutate_live_modules_and_context_recovers() {
    let mut ctx = Context::new();
    let original = import(&mut ctx, &source("dot_general", DIMENSIONS)).unwrap();
    let before = export(&ctx, &original).unwrap();
    for attributes in [
        "",
        "dot_dimension_numbers = #stablehlo.gather<>",
        "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>",
        &format!("{DIMENSIONS}, precision_config = array<i64: 0, 0>"),
        &format!("{DIMENSIONS}, precision_config = [0 : i64]"),
        &format!("{DIMENSIONS}, precision_config = [#stablehlo<comparison_type FLOAT>]"),
        &format!(
            "{DIMENSIONS}, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]"
        ),
        &format!("{DIMENSIONS}, {ALGORITHM}, precision_config = [#stablehlo<precision HIGH>]"),
    ] {
        assert!(
            import(&mut ctx, &source("dot_general", attributes)).is_err(),
            "{attributes}"
        );
        assert_eq!(export(&ctx, &original).unwrap(), before);
        let recovered = import(&mut ctx, &source("dot", "")).unwrap();
        export(&ctx, &recovered).unwrap();
    }
}
