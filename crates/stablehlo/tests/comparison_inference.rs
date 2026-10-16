use stablehlo::text::{parse_module, parse_type};
use stablehlo::{ComparisonType, Type, infer_compare_type};

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn module(lhs: &str, rhs: &str, result: &str, comparison: Option<&str>) -> bool {
    let mode = comparison.map_or(String::new(), |mode| {
        format!(", compare_type = #stablehlo<comparison_type {mode}>")
    });
    parse_module(&format!(
        r#"
      func.func @main(%a: {lhs}, %b: {rhs}) -> {result} {{
        %r = "stablehlo.compare"(%a, %b)
          {{comparison_direction = #stablehlo<comparison_direction LT>{mode}}}
          : ({lhs}, {rhs}) -> {result}
        return %r : {result}
      }}
    "#
    ))
    .is_ok()
}

#[test]
fn compare_infers_lhs_shape_bounds_and_predicate_elements() {
    let lhs = ty("tensor<?x2xf32, #stablehlo.bounds<8, ?>>");
    let rhs = ty("tensor<3x2xf32>");
    assert_eq!(
        infer_compare_type(&lhs, &rhs, None).unwrap(),
        ty("tensor<?x2xi1, #stablehlo.bounds<8, ?>>")
    );
    assert_eq!(
        infer_compare_type(&rhs, &lhs, None).unwrap(),
        ty("tensor<3x2xi1>")
    );
    assert!(infer_compare_type(&lhs, &ty("tensor<9x2xf32>"), None).is_err());
    assert!(infer_compare_type(&lhs, &ty("tensor<3x2xf64>"), None).is_err());
}

#[test]
fn compatible_dynamic_comparison_results_are_accepted_like_upstream_fixtures() {
    assert!(module(
        "tensor<3xi32>",
        "tensor<3xi32>",
        "tensor<?xi1>",
        None
    ));
    assert!(module(
        "tensor<3xi32>",
        "tensor<?xi32>",
        "tensor<?xi1>",
        None
    ));
    assert!(module(
        "tensor<?xi32>",
        "tensor<3xi32>",
        "tensor<3xi1>",
        None
    ));
    assert!(!module(
        "tensor<?xi32>",
        "tensor<3xi32>",
        "tensor<4xi1>",
        None
    ));
    assert!(!module(
        "tensor<?xi32>",
        "tensor<4xi32>",
        "tensor<?xi1, #stablehlo.bounds<3>>",
        None
    ));
    assert!(!module(
        "tensor<?xi32, #stablehlo.bounds<3>>",
        "tensor<?xi32>",
        "tensor<4xi1>",
        None
    ));
    assert!(!module(
        "tensor<3xi32>",
        "tensor<3xi32>",
        "tensor<3xi32>",
        None
    ));
}

#[test]
fn weak_order_requires_ordinary_floats_in_inference_and_verification() {
    for source in [
        "tensor<2xf32>",
        "tensor<2xf64>",
        "tensor<2xbf16>",
        "tensor<2xf8E4M3FN>",
    ] {
        assert!(
            infer_compare_type(&ty(source), &ty(source), Some(ComparisonType::WeakOrder)).is_ok()
        );
        assert!(module(source, source, "tensor<2xi1>", Some("WEAKORDER")));
    }
    for source in [
        "tensor<2xi32>",
        "tensor<2xui32>",
        "tensor<2xi1>",
        "tensor<2xcomplex<f32>>",
        "tensor<2x!quant.uniform<i8:f32, 0.5>>",
    ] {
        assert!(
            infer_compare_type(&ty(source), &ty(source), Some(ComparisonType::WeakOrder)).is_err()
        );
        assert!(!module(source, source, "tensor<2xi1>", Some("WEAKORDER")));
    }
    for mode in [
        ComparisonType::NoType,
        ComparisonType::Signed,
        ComparisonType::Unsigned,
    ] {
        assert!(infer_compare_type(&ty("tensor<i32>"), &ty("tensor<i32>"), Some(mode)).is_ok());
    }
}

#[test]
fn explicit_comparison_modes_follow_spec_element_categories() {
    let modes = [
        (ComparisonType::NoType, "NOTYPE"),
        (ComparisonType::Signed, "SIGNED"),
        (ComparisonType::Unsigned, "UNSIGNED"),
        (ComparisonType::Float, "FLOAT"),
        (ComparisonType::TotalOrder, "TOTALORDER"),
        (ComparisonType::WeakOrder, "WEAKORDER"),
    ];
    for (source, allowed_modes) in [
        ("tensor<2xi32>", [true, true, true, false, false, false]),
        ("tensor<2xui32>", [true, true, true, false, false, false]),
        ("tensor<2xi1>", [true, true, true, false, false, false]),
        ("tensor<2xf32>", [false, false, false, true, true, true]),
        (
            "tensor<2xcomplex<f32>>",
            [false, false, false, true, false, false],
        ),
    ] {
        for ((mode, spelling), accepted) in modes.into_iter().zip(allowed_modes) {
            assert_eq!(
                infer_compare_type(&ty(source), &ty(source), Some(mode)).is_ok(),
                accepted,
                "{source}, {spelling}"
            );
            assert_eq!(
                module(source, source, "tensor<2xi1>", Some(spelling)),
                accepted,
                "{source}, {spelling}"
            );
        }
    }
}

#[test]
fn quantized_comparison_preserves_bounds_and_accepts_independent_parameters() {
    let lhs = "tensor<?x!quant.uniform<i8:f32, 0.25:1>, #stablehlo.bounds<5>>";
    let rhs = "tensor<3x!quant.uniform<i8:f32, 0.5:2>>";
    assert_eq!(
        infer_compare_type(&ty(lhs), &ty(rhs), None).unwrap(),
        ty("tensor<?xi1, #stablehlo.bounds<5>>")
    );
    assert!(module(lhs, rhs, "tensor<3xi1>", None));
    assert!(!module(lhs, rhs, "tensor<4xi1>", None));
}
