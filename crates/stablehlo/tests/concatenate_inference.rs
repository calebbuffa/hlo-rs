use stablehlo::{
    Attribute, InferenceError, StableHloOp, Type, infer_concatenate_type, text::parse_type,
};

mod common;

fn infer(inputs: &[&str], dimension: usize) -> Result<Type, InferenceError> {
    let types = inputs
        .iter()
        .map(|input| parse_type(input).unwrap())
        .collect::<Vec<_>>();
    infer_concatenate_type(&types, dimension)
}

#[test]
fn concatenation_implements_every_reference_dimension_and_bound_rule() {
    // Base.cpp::inferConcatenatedDimAndBound, rules c0-c5.
    for (lhs, rhs, result) in [
        ("tensor<2xf32>", "tensor<3xf32>", "tensor<5xf32>"),
        ("tensor<2xf32>", "tensor<?xf32>", "tensor<?xf32>"),
        (
            "tensor<2xf32>",
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<?xf32, #stablehlo.bounds<5>>",
        ),
        ("tensor<?xf32>", "tensor<?xf32>", "tensor<?xf32>"),
        (
            "tensor<?xf32>",
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<?xf32>",
        ),
        (
            "tensor<?xf32, #stablehlo.bounds<2>>",
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<?xf32, #stablehlo.bounds<5>>",
        ),
        (
            "tensor<0xf32>",
            "tensor<?xf32, #stablehlo.bounds<0>>",
            "tensor<?xf32, #stablehlo.bounds<0>>",
        ),
    ] {
        let expected = parse_type(result).unwrap();
        assert_eq!(infer(&[lhs, rhs], 0).unwrap(), expected);
        assert_eq!(infer(&[rhs, lhs], 0).unwrap(), expected);
        assert_eq!(infer(&["tensor<0xf32>", lhs, rhs], 0).unwrap(), expected);
    }
}

#[test]
fn other_dimensions_merge_all_sizes_and_bounds() {
    for (lhs, rhs, expected) in [
        ("tensor<2x?xf32>", "tensor<3x4xf32>", "tensor<5x4xf32>"),
        (
            "tensor<2x?xf32, #stablehlo.bounds<?, 8>>",
            "tensor<3x?xf32, #stablehlo.bounds<?, 5>>",
            "tensor<5x?xf32, #stablehlo.bounds<?, 5>>",
        ),
        (
            "tensor<2x?xf32, #stablehlo.bounds<?, 8>>",
            "tensor<3x4xf32>",
            "tensor<5x4xf32>",
        ),
    ] {
        assert_eq!(
            infer(&[lhs, rhs], 0).unwrap(),
            parse_type(expected).unwrap()
        );
        assert_eq!(
            infer(&[rhs, lhs], 0).unwrap(),
            parse_type(expected).unwrap()
        );
    }
    assert!(matches!(
        infer(
            &[
                "tensor<2x?xf32, #stablehlo.bounds<?, 3>>",
                "tensor<3x4xf32>"
            ],
            0
        ),
        Err(InferenceError::BoundExceeded {
            dimension: 1,
            size: 4,
            bound: 3
        })
    ));
    assert!(matches!(
        infer(
            &["tensor<2x?xf32>", "tensor<3x4xf32>", "tensor<1x5xf32>"],
            0
        ),
        Err(InferenceError::DimensionMismatch { dimension: 1, .. })
    ));
}

#[test]
fn verifier_uses_inferred_bounds_without_treating_upper_bounds_as_exact_sizes() {
    let inputs = ["tensor<2xf32>", "tensor<?xf32, #stablehlo.bounds<3>>"]
        .iter()
        .map(|text| parse_type(text).unwrap())
        .collect::<Vec<_>>();
    for (output, accepted) in [
        ("tensor<5xf32>", true),
        ("tensor<4xf32>", true),
        ("tensor<6xf32>", false),
        ("tensor<?xf32>", true),
        ("tensor<?xf32, #stablehlo.bounds<8>>", true),
        ("tensor<?xf32, #stablehlo.bounds<4>>", true),
    ] {
        assert_eq!(
            common::computation(
                StableHloOp::Concatenate,
                inputs.clone(),
                vec![parse_type(output).unwrap()],
                [("dimension".into(), Attribute::integer(0))].into()
            )
            .is_ok(),
            accepted,
            "{output}"
        );
    }
}

#[test]
fn size_and_bound_sums_reject_overflow_and_accept_the_exact_limit() {
    let max = i64::MAX;
    for input in [
        format!("tensor<{max}xf32>"),
        format!("tensor<?xf32, #stablehlo.bounds<{max}>>"),
    ] {
        assert!(matches!(
            infer(&[&input, "tensor<1xf32>"], 0),
            Err(InferenceError::DimensionOverflow { dimension: 0 })
        ));
        assert_eq!(
            infer(&[&input, "tensor<0xf32>"], 0).unwrap(),
            parse_type(&input).unwrap()
        );
    }
}

#[test]
fn inference_errors_are_explicit_for_invalid_inputs_axes_and_types() {
    assert_eq!(infer(&[], 0), Err(InferenceError::EmptyInputs));
    assert!(matches!(
        infer(&["tensor<f32>"], 0),
        Err(InferenceError::DimensionOutOfBounds {
            dimension: 0,
            rank: 0
        })
    ));
    assert!(matches!(
        infer(&["tensor<2xf32>"], 1),
        Err(InferenceError::DimensionOutOfBounds {
            dimension: 1,
            rank: 1
        })
    ));
    assert!(matches!(
        infer(&["tensor<2xf32>", "tensor<1xi32>"], 0),
        Err(InferenceError::IncompatibleTypes { index: 1 })
    ));
    assert!(matches!(
        infer(&["tensor<2xf32>", "tensor<1x2xf32>"], 0),
        Err(InferenceError::RankMismatch {
            expected: 1,
            actual: 2
        })
    ));
    assert!(matches!(
        infer(&["!stablehlo.token"], 0),
        Err(InferenceError::IncompatibleTypes { index: 0 })
    ));
}

#[test]
fn quantization_parameters_are_preserved_and_must_match() {
    let lhs = "tensor<2x!quant.uniform<i8:f32, 0.5:0>>";
    let rhs = "tensor<3x!quant.uniform<i8:f32, 0.5:0>>";
    assert_eq!(
        infer(&[lhs, rhs], 0).unwrap(),
        parse_type("tensor<5x!quant.uniform<i8:f32, 0.5:0>>").unwrap()
    );
    assert!(matches!(
        infer(&[lhs, "tensor<3x!quant.uniform<i8:f32, 0.25:0>>"], 0),
        Err(InferenceError::IncompatibleTypes { index: 1 })
    ));
}
