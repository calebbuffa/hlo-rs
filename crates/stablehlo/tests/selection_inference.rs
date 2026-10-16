use stablehlo::text::{parse_module, parse_type};
use stablehlo::{
    Attribute, ElementType, InferenceError, Type, infer_clamp_type, infer_select_type,
};

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn select_module(pred: &str, a: &str, b: &str, result: &str) -> bool {
    parse_module(&format!(
        "module {{ func.func @main(%p: {pred}, %a: {a}, %b: {b}) -> {result} {{
          %r = \"stablehlo.select\"(%p, %a, %b) : ({pred}, {a}, {b}) -> {result}
          return %r : {result}
        }} }}"
    ))
    .is_ok()
}

fn clamp_module(min: &str, operand: &str, max: &str, result: &str) -> bool {
    parse_module(&format!(
        "module {{ func.func @main(%min: {min}, %x: {operand}, %max: {max}) -> {result} {{
          %r = \"stablehlo.clamp\"(%min, %x, %max) : ({min}, {operand}, {max}) -> {result}
          return %r : {result}
        }} }}"
    ))
    .is_ok()
}

#[test]
fn select_merges_value_shapes_and_bounds_but_not_the_predicate_shape() {
    let a = ty("tensor<?x3xf32, #stablehlo.bounds<8, ?>>");
    let b = ty("tensor<?x3xf32, #stablehlo.bounds<5, ?>>");
    let expected = ty("tensor<?x3xf32, #stablehlo.bounds<5, ?>>");
    assert_eq!(
        infer_select_type(&ty("tensor<i1>"), &a, &b).unwrap(),
        expected
    );
    assert_eq!(
        infer_select_type(&ty("tensor<4x3xi1>"), &a, &b).unwrap(),
        expected
    );
    assert_eq!(
        infer_select_type(&ty("tensor<i1>"), &a, &ty("tensor<4x3xf32>")).unwrap(),
        ty("tensor<4x3xf32>"),
    );
    assert!(infer_select_type(&ty("tensor<6x3xi1>"), &a, &b).is_err());
    assert!(
        infer_select_type(
            &ty("tensor<?x3xi1, #stablehlo.bounds<3, ?>>"),
            &a,
            &ty("tensor<4x3xf32>")
        )
        .is_err()
    );
}

#[test]
fn clamp_retains_operand_encoding_and_does_not_refine_from_limits() {
    let operand = ty("tensor<?x3xf32, #stablehlo.bounds<8, ?>>");
    let scalar = ty("tensor<f32>");
    let limit = ty("tensor<4x3xf32>");
    assert_eq!(
        infer_clamp_type(&scalar, &operand, &limit).unwrap(),
        operand
    );
    assert_eq!(
        infer_clamp_type(&limit, &operand, &scalar).unwrap(),
        operand
    );
    let opaque = stablehlo::TensorType::new(ElementType::F32, vec![None, Some(3)])
        .with_encoding(Attribute::String("retained encoding".into()))
        .into();
    assert_eq!(infer_clamp_type(&scalar, &opaque, &scalar).unwrap(), opaque);
    assert!(infer_clamp_type(&scalar, &operand, &ty("tensor<9x3xf32>")).is_err());
    assert!(infer_clamp_type(&ty("tensor<3xf32>"), &operand, &scalar).is_err());
}

#[test]
fn dynamic_witnesses_do_not_hide_conflicting_predicates_limits_or_results() {
    assert!(!select_module(
        "tensor<3xi1>",
        "tensor<?xf32>",
        "tensor<4xf32>",
        "tensor<?xf32>"
    ));
    assert!(!select_module(
        "tensor<4xi1>",
        "tensor<?xf32>",
        "tensor<?xf32>",
        "tensor<?xf32, #stablehlo.bounds<3>>"
    ));
    assert!(!select_module(
        "tensor<i1>",
        "tensor<?xf32, #stablehlo.bounds<3>>",
        "tensor<?xf32>",
        "tensor<4xf32>"
    ));
    assert!(!clamp_module(
        "tensor<3xf32>",
        "tensor<?xf32>",
        "tensor<4xf32>",
        "tensor<?xf32>"
    ));
    assert!(!clamp_module(
        "tensor<f32>",
        "tensor<?xf32, #stablehlo.bounds<3>>",
        "tensor<f32>",
        "tensor<4xf32>"
    ));
    assert!(!clamp_module(
        "tensor<4xf32>",
        "tensor<?xf32>",
        "tensor<f32>",
        "tensor<?xf32, #stablehlo.bounds<3>>"
    ));
    assert!(select_module(
        "tensor<3xi1>",
        "tensor<?xf32>",
        "tensor<?xf32>",
        "tensor<3xf32>"
    ));
    assert!(clamp_module(
        "tensor<3xf32>",
        "tensor<?xf32>",
        "tensor<f32>",
        "tensor<3xf32>"
    ));
}

#[test]
fn independent_per_tensor_quantization_parameters_are_allowed_by_baseline_rules() {
    let a = "tensor<2x!quant.uniform<i8:f32, 0.25:1>>";
    let b = "tensor<2x!quant.uniform<i8:f32, 0.5:-1>>";
    let output = "tensor<2x!quant.uniform<i8:f32, 1.0:0>>";
    let scalar = "tensor<!quant.uniform<i8:f32, 2.0:2>>";
    assert_eq!(
        infer_select_type(&ty("tensor<i1>"), &ty(a), &ty(b)).unwrap(),
        ty(a)
    );
    assert_eq!(
        infer_clamp_type(&ty(scalar), &ty(a), &ty(b)).unwrap(),
        ty(a)
    );
    assert!(select_module("tensor<i1>", a, b, output));
    assert!(clamp_module(scalar, a, b, output));
    let different_range = "tensor<2x!quant.uniform<i8<-127:127>:f32, 0.5>>";
    assert!(infer_select_type(&ty("tensor<i1>"), &ty(a), &ty(different_range)).is_err());
    assert!(infer_clamp_type(&ty(scalar), &ty(a), &ty(different_range)).is_err());
    assert!(!select_module("tensor<i1>", a, b, different_range));
    assert!(!clamp_module(scalar, a, b, different_range));
    let explicit_default = "tensor<2x!quant.uniform<i8<-128:127>:f32, 0.5>>";
    assert!(select_module("tensor<i1>", a, explicit_default, output));
}

#[test]
fn nonpredicate_per_axis_and_nontensor_inputs_are_rejected() {
    let value = ty("tensor<2xf32>");
    assert!(infer_select_type(&ty("tensor<i32>"), &value, &value).is_err());
    assert!(infer_select_type(&ty("tensor<1xi1>"), &value, &value).is_err());
    assert!(infer_clamp_type(&ty("tensor<i32>"), &value, &value).is_err());
    let quantized = ty("tensor<2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>");
    assert!(infer_select_type(&ty("tensor<i1>"), &quantized, &quantized).is_err());
    assert!(infer_clamp_type(&quantized, &quantized, &quantized).is_err());
    for invalid in [
        Type::Token,
        Type::Tuple(vec![]),
        Type::Element(ElementType::F32),
    ] {
        assert!(infer_select_type(&ty("tensor<i1>"), &invalid, &value).is_err());
        assert!(infer_clamp_type(&invalid, &value, &value).is_err());
    }
    let invalid = stablehlo::TensorType::new(ElementType::F32, vec![Some(u64::MAX)]).into();
    assert!(matches!(
        infer_select_type(&ty("tensor<i1>"), &value, &invalid),
        Err(InferenceError::InvalidType { index: 2, .. }),
    ));
}

#[test]
fn small_scalar_and_vector_broadcast_shapes_match_the_exact_contract() {
    for pred in 0..=4 {
        for a in 0..=3 {
            for b in 0..=3 {
                let predicate = if pred == 4 {
                    ty("tensor<i1>")
                } else {
                    ty(&format!("tensor<{pred}xi1>"))
                };
                let true_value = ty(&format!("tensor<{a}xf32>"));
                let false_value = ty(&format!("tensor<{b}xf32>"));
                let expected = a == b && (pred == 4 || pred == a);
                assert_eq!(
                    infer_select_type(&predicate, &true_value, &false_value).is_ok(),
                    expected,
                    "pred={pred}, true={a}, false={b}"
                );
            }
        }
    }
    for min in 0..=4 {
        for operand in 0..=3 {
            for max in 0..=4 {
                let minimum = if min == 4 {
                    ty("tensor<f32>")
                } else {
                    ty(&format!("tensor<{min}xf32>"))
                };
                let value = ty(&format!("tensor<{operand}xf32>"));
                let maximum = if max == 4 {
                    ty("tensor<f32>")
                } else {
                    ty(&format!("tensor<{max}xf32>"))
                };
                let expected = (min == 4 || min == operand) && (max == 4 || max == operand);
                assert_eq!(
                    infer_clamp_type(&minimum, &value, &maximum).is_ok(),
                    expected,
                    "min={min}, operand={operand}, max={max}"
                );
            }
        }
    }
    assert!(infer_select_type(&ty("tensor<i1>"), &ty("tensor<f32>"), &ty("tensor<f32>")).is_ok());
    assert!(infer_clamp_type(&ty("tensor<f32>"), &ty("tensor<f32>"), &ty("tensor<f32>")).is_ok());
}
