use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{InferenceError, Type, infer_map_type};

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn module(lhs: &str, rhs: &str, output: &str, dimensions: &str) -> bool {
    let source = format!(
        r#"
      func.func @map(%lhs: {lhs}, %rhs: {rhs}) -> {output} {{
        %r = "stablehlo.map"(%lhs, %rhs) ({{
        ^bb0(%a: tensor<f32>, %b: tensor<i32>):
          stablehlo.return %a : tensor<f32>
        }}) {{dimensions = array<i64{dimensions}>}} : ({lhs}, {rhs}) -> {output}
        return %r : {output}
      }}
    "#
    );
    let Ok(module) = parse_module(&source) else {
        return false;
    };
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    true
}

#[test]
fn map_inference_uses_last_input_shape_and_scalar_result_elements_without_encoding() {
    let inputs = [
        ty("tensor<3x2xf32>"),
        ty("tensor<?x2xi32, #stablehlo.bounds<5, ?>>"),
    ];
    assert_eq!(
        infer_map_type(&inputs, &ty("tensor<i1>"), &[0, 1]).unwrap(),
        ty("tensor<?x2xi1>")
    );
    assert!(module(
        "tensor<3x2xf32>",
        "tensor<?x2xi32, #stablehlo.bounds<5, ?>>",
        "tensor<?x2xf32>",
        ": 0, 1"
    ));
}

#[test]
fn map_inference_accepts_scalars_zero_extents_and_independent_quantized_elements() {
    assert_eq!(
        infer_map_type(
            &[ty("tensor<f32>"), ty("tensor<i32>")],
            &ty("tensor<i8>"),
            &[]
        )
        .unwrap(),
        ty("tensor<i8>")
    );
    assert!(module("tensor<f32>", "tensor<i32>", "tensor<f32>", ""));
    let scalar = ty("tensor<!quant.uniform<i8:f32, 0.25:1>>");
    assert_eq!(
        infer_map_type(
            &[ty("tensor<0x!quant.uniform<ui8:f32, 0.5:2>>")],
            &scalar,
            &[0]
        )
        .unwrap(),
        ty("tensor<0x!quant.uniform<i8:f32, 0.25:1>>")
    );
}

#[test]
fn map_dimensions_must_be_every_axis_in_order() {
    let input = ty("tensor<2x3xf32>");
    for axes in [
        vec![],
        vec![0],
        vec![1, 0],
        vec![-1, 0],
        vec![0, 0],
        vec![0, 2],
        vec![0, 1, 2],
    ] {
        assert!(infer_map_type(std::slice::from_ref(&input), &ty("tensor<f32>"), &axes).is_err());
    }
    assert!(!module(
        "tensor<2x3xf32>",
        "tensor<2x3xi32>",
        "tensor<2x3xf32>",
        ": 1, 0"
    ));
}

#[test]
fn map_requires_nonempty_ranked_inputs_and_a_supported_scalar_computation_result() {
    assert_eq!(
        infer_map_type(&[], &ty("tensor<f32>"), &[]),
        Err(InferenceError::EmptyInputs)
    );
    for input in [
        "tensor<*xf32>",
        "!stablehlo.token",
        "tensor<2xindex>",
        "tensor<2xtf32>",
        "tensor<2xsi32>",
        "tensor<2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>",
    ] {
        assert!(
            infer_map_type(&[ty(input)], &ty("tensor<f32>"), &[0]).is_err(),
            "{input}"
        );
    }
    for result in [
        "tensor<1xf32>",
        "tensor<*xf32>",
        "tensor<index>",
        "!stablehlo.token",
        "f32",
    ] {
        assert!(
            infer_map_type(&[ty("tensor<2xf32>")], &ty(result), &[0]).is_err(),
            "{result}"
        );
    }
}

#[test]
fn map_checks_all_common_shape_and_bound_witnesses_without_element_matching() {
    for inputs in [
        vec![ty("tensor<?xf32>"), ty("tensor<3xi32>"), ty("tensor<4xi8>")],
        vec![
            ty("tensor<?xf32, #stablehlo.bounds<3>>"),
            ty("tensor<4xi32>"),
        ],
        vec![
            ty("tensor<4xf32>"),
            ty("tensor<?xi32, #stablehlo.bounds<3>>"),
        ],
        vec![ty("tensor<2x3xf32>"), ty("tensor<6xi32>")],
    ] {
        assert!(
            infer_map_type(&inputs, &ty("tensor<f32>"), &[0]).is_err(),
            "{inputs:?}"
        );
    }
    assert!(
        infer_map_type(
            &[
                ty("tensor<?xf32, #stablehlo.bounds<3>>"),
                ty("tensor<?xi32, #stablehlo.bounds<4>>")
            ],
            &ty("tensor<i1>"),
            &[0],
        )
        .is_ok()
    );
}

#[test]
fn map_declared_output_cannot_hide_input_or_output_bound_conflicts() {
    for (lhs, rhs, result) in [
        (
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<?xi32>",
            "tensor<4xf32>",
        ),
        (
            "tensor<4xf32>",
            "tensor<?xi32>",
            "tensor<?xf32, #stablehlo.bounds<3>>",
        ),
        (
            "tensor<?xf32>",
            "tensor<?xi32, #stablehlo.bounds<3>>",
            "tensor<4xf32>",
        ),
        ("tensor<?xf32>", "tensor<4xi32>", "tensor<3xf32>"),
    ] {
        assert!(!module(lhs, rhs, result, ": 0"));
    }
    assert!(module(
        "tensor<?xf32, #stablehlo.bounds<3>>",
        "tensor<?xi32>",
        "tensor<3xf32>",
        ": 0"
    ));
    assert!(module(
        "tensor<3xf32>",
        "tensor<?xi32>",
        "tensor<?xf32, #stablehlo.bounds<3>>",
        ": 0"
    ));
}

#[test]
fn map_small_bounded_shape_witnesses_match_exact_feasibility() {
    let witnesses = [
        (Some(0), None),
        (Some(1), None),
        (Some(2), None),
        (None, None),
        (None, Some(0)),
        (None, Some(1)),
        (None, Some(2)),
    ];
    let tensor = |(size, bound): (Option<u64>, Option<u64>), element: &str| {
        let size = size.map_or_else(|| "?".into(), |value| value.to_string());
        let encoding =
            bound.map_or_else(String::new, |value| format!(", #stablehlo.bounds<{value}>"));
        format!("tensor<{size}x{element}{encoding}>")
    };
    for lhs in witnesses {
        for rhs in witnesses {
            for output in witnesses {
                let expected = (0..=2).any(|extent| {
                    [lhs, rhs, output].iter().all(|&(size, bound)| {
                        size.is_none_or(|size| size == extent)
                            && bound.is_none_or(|bound| extent <= bound)
                    })
                });
                let inputs = [
                    ty(&tensor(lhs, "f32")),
                    ty(&tensor(rhs, "i32")),
                    ty(&tensor(output, "i8")),
                ];
                assert_eq!(
                    infer_map_type(&inputs, &ty("tensor<f32>"), &[0]).is_ok(),
                    expected,
                    "inference witnesses: {lhs:?}, {rhs:?}, {output:?}",
                );
                assert_eq!(
                    module(
                        &tensor(lhs, "f32"),
                        &tensor(rhs, "i32"),
                        &tensor(output, "f32"),
                        ": 0"
                    ),
                    expected,
                    "verification witnesses: {lhs:?}, {rhs:?}, {output:?}",
                );
            }
        }
    }
}
