use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{InferenceError, Type, infer_sort_types};

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn module(key: &str, payload: &str, output_key: &str, output_payload: &str) -> bool {
    let source = format!(
        r#"
      func.func @sort(%key: {key}, %payload: {payload}) -> ({output_key}, {output_payload}) {{
        %r:2 = "stablehlo.sort"(%key, %payload) ({{
        ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
          %lt = stablehlo.compare LT, %a, %b : (tensor<f32>, tensor<f32>) -> tensor<i1>
          stablehlo.return %lt : tensor<i1>
        }}) {{dimension = -1 : i64}} : ({key}, {payload}) -> ({output_key}, {output_payload})
        return %r#0, %r#1 : {output_key}, {output_payload}
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
fn sort_retains_each_input_shape_bounds_element_and_quantization_metadata() {
    let inputs = [
        ty("tensor<?x2xf32, #stablehlo.bounds<5, ?>>"),
        ty("tensor<3x2xi32>"),
        ty("tensor<?x2x!quant.uniform<i8:f32, 0.25:1>, #stablehlo.bounds<4, ?>>"),
    ];
    for dimension in [-2, -1, 0, 1] {
        assert_eq!(infer_sort_types(&inputs, dimension).unwrap(), inputs);
    }
    assert!(module(
        "tensor<?x2xf32, #stablehlo.bounds<5, ?>>",
        "tensor<3x2xi32>",
        "tensor<?x2xf32, #stablehlo.bounds<5, ?>>",
        "tensor<3x2xi32>"
    ));
}

#[test]
fn sort_rejects_empty_inputs_scalar_inputs_and_out_of_range_signed_dimensions() {
    assert_eq!(infer_sort_types(&[], -1), Err(InferenceError::EmptyInputs));
    for dimension in [i64::MIN, -3, 2, i64::MAX] {
        assert!(matches!(
            infer_sort_types(&[ty("tensor<2x3xf32>")], dimension),
            Err(InferenceError::InvalidSortDimension { .. })
        ));
    }
    for dimension in [-1, 0, 1] {
        assert!(infer_sort_types(&[ty("tensor<f32>")], dimension).is_err());
    }
    assert!(infer_sort_types(&[ty("tensor<0xf32>")], -1).is_ok());
}

#[test]
fn sort_domains_exclude_unranked_metadata_and_per_axis_quantized_tensors() {
    for source in [
        "tensor<*xf32>",
        "tensor<2xindex>",
        "tensor<2xtf32>",
        "tensor<2xsi32>",
        "!stablehlo.token",
        "f32",
        "tuple<tensor<2xf32>>",
        "tensor<2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>",
    ] {
        assert!(infer_sort_types(&[ty(source)], 0).is_err(), "{source}");
    }
}

#[test]
fn sort_common_shape_checks_all_input_bounds_and_static_witnesses() {
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
        assert!(infer_sort_types(&inputs, -1).is_err(), "{inputs:?}");
    }
    let inputs = [
        ty("tensor<?xf32, #stablehlo.bounds<3>>"),
        ty("tensor<?xi32, #stablehlo.bounds<4>>"),
    ];
    assert_eq!(infer_sort_types(&inputs, 0).unwrap(), inputs);
}

#[test]
fn sort_declared_results_cannot_hide_conflicts_with_dynamic_input_bounds() {
    for (key, payload, result_key, result_payload) in [
        (
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<?xi32>",
            "tensor<?xf32>",
            "tensor<4xi32>",
        ),
        (
            "tensor<?xf32>",
            "tensor<?xi32, #stablehlo.bounds<3>>",
            "tensor<4xf32>",
            "tensor<?xi32>",
        ),
        (
            "tensor<4xf32>",
            "tensor<?xi32>",
            "tensor<?xf32>",
            "tensor<?xi32, #stablehlo.bounds<3>>",
        ),
    ] {
        assert!(!module(key, payload, result_key, result_payload));
    }
    assert!(module(
        "tensor<?xf32, #stablehlo.bounds<3>>",
        "tensor<?xi32>",
        "tensor<?xf32>",
        "tensor<3xi32>"
    ));
}

#[test]
fn sort_small_bounded_shape_witnesses_match_exact_feasibility() {
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
    for key in witnesses {
        for payload in witnesses {
            for output_key in witnesses {
                let inputs = [
                    ty(&tensor(key, "f32")),
                    ty(&tensor(payload, "i32")),
                    ty(&tensor(output_key, "i8")),
                ];
                let feasible = |values: &[(Option<u64>, Option<u64>)]| {
                    (0..=2).any(|extent| {
                        values.iter().all(|&(size, bound)| {
                            size.is_none_or(|size| size == extent)
                                && bound.is_none_or(|bound| extent <= bound)
                        })
                    })
                };
                assert_eq!(
                    infer_sort_types(&inputs, 0).is_ok(),
                    feasible(&[key, payload, output_key]),
                    "inference witnesses: {key:?}, {payload:?}, {output_key:?}",
                );
                for output_payload in witnesses {
                    assert_eq!(
                        module(
                            &tensor(key, "f32"),
                            &tensor(payload, "i32"),
                            &tensor(output_key, "f32"),
                            &tensor(output_payload, "i32"),
                        ),
                        feasible(&[key, payload, output_key, output_payload]),
                        "verification witnesses: {key:?}, {payload:?}, {output_key:?}, {output_payload:?}",
                    );
                }
            }
        }
    }
}
