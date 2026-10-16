use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{InferenceError, RngDistribution, Type, infer_rng_type};

mod upstream_support;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn source(element: &str, distribution: &str, shape: Option<&[i64]>, output: &str) -> String {
    let rank = if let Some(shape) = shape {
        shape.len()
    } else {
        2
    };
    let shape_type = format!("tensor<{rank}xi64>");
    let scalar = format!("tensor<{element}>");
    let (parameter, constant) = if let Some(values) = shape {
        let values = values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        (
            String::new(),
            format!("%s=stablehlo.constant dense<[{values}]>:{shape_type}"),
        )
    } else {
        (format!(",%s:{shape_type}"), String::new())
    };
    format!(
        "func.func @rng(%a:{scalar},%b:{scalar}{parameter})->{output} {{ \
        {constant} %r=\"stablehlo.rng\"(%a,%b,%s){{rng_distribution=#stablehlo<rng_distribution {distribution}>}}:\
        ({scalar},{scalar},{shape_type})->{output} return %r:{output} }}"
    )
}

#[test]
fn constant_and_runtime_shapes_infer_dimensions_without_carrying_encoding() {
    for shape_element in ["i32", "i64", "index"] {
        let a = ty("tensor<f32>");
        let b = ty("tensor<f32,#stablehlo.bounds<>>");
        for values in [vec![], vec![0], vec![2, 0, 3], vec![i32::MAX as i64]] {
            let shape = ty(&format!("tensor<{}x{shape_element}>", values.len()));
            let inferred =
                infer_rng_type(&a, &b, &shape, Some(&values), RngDistribution::Normal).unwrap();
            let Type::Tensor(tensor) = inferred else {
                panic!("expected tensor")
            };
            assert_eq!(
                tensor.dimensions(),
                values
                    .iter()
                    .map(|&value| Some(value as u64))
                    .collect::<Vec<_>>()
            );
            assert!(tensor.encoding().is_none());
            let inferred = infer_rng_type(&a, &b, &shape, None, RngDistribution::Uniform).unwrap();
            let Type::Tensor(tensor) = inferred else {
                panic!("expected tensor")
            };
            assert_eq!(tensor.dimensions(), vec![None; values.len()]);
        }
    }
}

#[test]
fn distributions_share_scalar_element_constraints_and_operation_domains() {
    for element in [
        "i1", "i4", "i8", "i16", "i32", "i64", "ui4", "ui8", "ui16", "ui32", "ui64", "f4E2M1FN",
        "f8E4M3FN", "f8E5M2", "f16", "bf16", "f32", "f64",
    ] {
        let scalar = ty(&format!("tensor<{element}>"));
        let shape = ty("tensor<2xi64>");
        assert!(
            infer_rng_type(
                &scalar,
                &scalar,
                &shape,
                Some(&[2, 3]),
                RngDistribution::Uniform
            )
            .is_ok(),
            "{element}"
        );
        assert_eq!(
            infer_rng_type(&scalar, &scalar, &shape, None, RngDistribution::Normal).is_ok(),
            element.starts_with('f') || element == "bf16",
            "{element}"
        );
    }
    let a = ty("tensor<f32>");
    for b in [
        "tensor<f64>",
        "tensor<1xf32>",
        "tensor<complex<f32>>",
        "tensor<si32>",
        "tensor<tf32>",
        "tensor<!quant.uniform<i8:f32,0.25>>",
    ] {
        assert!(
            infer_rng_type(
                &a,
                &ty(b),
                &ty("tensor<2xi64>"),
                None,
                RngDistribution::Uniform
            )
            .is_err(),
            "{b}"
        );
    }
    for shape in [
        "tensor<i64>",
        "tensor<?xi64>",
        "tensor<2xsi64>",
        "tensor<2xf32>",
        "tensor<1x2xi64>",
    ] {
        assert!(
            infer_rng_type(&a, &a, &ty(shape), None, RngDistribution::Uniform).is_err(),
            "{shape}"
        );
    }
}

#[test]
fn constant_shape_lengths_ranges_and_allocation_failures_return_explicit_errors() {
    let a = ty("tensor<f32>");
    for element in [
        "i2", "ui2", "i4", "ui4", "i8", "ui8", "i16", "ui16", "i32", "ui32", "i64", "ui64", "index",
    ] {
        let shape = ty(&format!("tensor<1x{element}>"));
        let maximum = match element {
            "i2" | "ui2" => 1,
            "i4" | "ui4" => 7,
            "i8" | "ui8" => 127,
            "i16" | "ui16" => 32767,
            "i32" | "ui32" => i32::MAX as i64,
            _ => i64::MAX,
        };
        assert!(
            infer_rng_type(&a, &a, &shape, Some(&[maximum]), RngDistribution::Uniform).is_ok(),
            "{element}"
        );
        if let Some(too_large) = maximum.checked_add(1) {
            assert!(
                infer_rng_type(&a, &a, &shape, Some(&[too_large]), RngDistribution::Uniform)
                    .is_err(),
                "{element}"
            );
        }
    }
    assert!(matches!(
        infer_rng_type(
            &a,
            &a,
            &ty("tensor<2xi64>"),
            Some(&[2]),
            RngDistribution::Uniform
        ),
        Err(InferenceError::AttributeLengthMismatch {
            attribute: "shape",
            expected: 2,
            actual: 1
        })
    ));
    for (shape, values) in [
        ("tensor<1xi64>", vec![-1]),
        ("tensor<1xi32>", vec![i32::MAX as i64 + 1]),
    ] {
        assert!(matches!(
            infer_rng_type(&a, &a, &ty(shape), Some(&values), RngDistribution::Uniform),
            Err(InferenceError::InvalidShapeExtent { dimension: 0, .. })
        ));
    }
    assert!(
        infer_rng_type(
            &a,
            &a,
            &ty("tensor<1xi64>"),
            Some(&[i64::MAX]),
            RngDistribution::Uniform
        )
        .is_ok()
    );
    assert!(matches!(
        infer_rng_type(
            &a,
            &a,
            &ty("tensor<9223372036854775807xi64>"),
            None,
            RngDistribution::Uniform
        ),
        Err(InferenceError::UnrepresentableRank { .. })
    ));
}

#[test]
fn verification_shares_inference_for_runtime_and_constant_shapes() {
    for (values, output, accepted) in [
        (None, "tensor<2x3xf32>", true),
        (None, "tensor<2xf32>", false),
        (Some(vec![2, 3]), "tensor<2x3xf32>", true),
        (Some(vec![2, 3]), "tensor<2x4xf32>", false),
        (
            Some(vec![0, 3]),
            "tensor<?x?xf32,#stablehlo.bounds<0,3>>",
            true,
        ),
        (
            Some(vec![2, 3]),
            "tensor<?x?xf32,#stablehlo.bounds<1,3>>",
            false,
        ),
        (Some(vec![-1, 3]), "tensor<?x3xf32>", false),
        (Some(vec![]), "tensor<f32>", true),
    ] {
        let input = source("f32", "NORMAL", values.as_deref(), output);
        let parsed = parse_module(&input);
        assert_eq!(parsed.is_ok(), accepted, "{input}\n{parsed:?}");
        if let Ok(module) = parsed {
            assert_eq!(
                parse_module(&print_module(&module).unwrap()).unwrap(),
                module
            );
        }
    }
    for (element, distribution, output) in [
        ("i32", "NORMAL", "tensor<2x3xi32>"),
        ("f32", "UNIFORM", "tensor<2x3xf64>"),
    ] {
        assert!(parse_module(&source(element, distribution, None, output)).is_err());
    }
}

#[test]
fn generated_rng_shape_verification_matches_independent_extent_witnesses() {
    let outputs = [
        "tensor<0x0xf32>",
        "tensor<2x3xf32>",
        "tensor<?x?xf32>",
        "tensor<?x?xf32,#stablehlo.bounds<2,3>>",
        "tensor<?x3xf32,#stablehlo.bounds<1,?>>",
    ];
    for first in 0..=3 {
        for second in 0..=4 {
            let expected = [
                first == 0 && second == 0,
                first == 2 && second == 3,
                true,
                first <= 2 && second <= 3,
                first <= 1 && second == 3,
            ];
            for (output, expected) in outputs.into_iter().zip(expected) {
                let input = source("f32", "UNIFORM", Some(&[first, second]), output);
                assert_eq!(parse_module(&input).is_ok(), expected, "{input}");
            }
        }
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn rng_shapes_roundtrip_through_independent_upstream() {
    use upstream_support::UpstreamResult;
    let positives = [
        source("f32", "NORMAL", Some(&[2, 3]), "tensor<2x3xf32>"),
        source("i32", "UNIFORM", Some(&[0, 3]), "tensor<0x3xi32>"),
        source("f32", "UNIFORM", None, "tensor<?x?xf32>"),
        source("f32", "NORMAL", Some(&[]), "tensor<f32>"),
        source(
            "f32",
            "NORMAL",
            Some(&[2, 3]),
            "tensor<?x?xf32,#stablehlo.bounds<2,3>>",
        ),
    ];
    let printed = positives
        .iter()
        .map(|input| print_module(&parse_module(input).unwrap()).unwrap())
        .collect::<Vec<_>>();
    for (input, result) in printed.iter().zip(upstream_support::verify(&printed)) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("{input}\n{error}"),
        };
        parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    }
    let invalid = [
        source("i32", "NORMAL", None, "tensor<2x3xi32>"),
        source("f32", "UNIFORM", Some(&[2, 3]), "tensor<2x4xf32>"),
        source("f32", "UNIFORM", None, "tensor<2xf32>"),
        source("f32", "UNIFORM", None, "tensor<2x3xf64>"),
    ];
    for (input, result) in invalid.iter().zip(upstream_support::verify(&invalid)) {
        assert!(matches!(result, UpstreamResult::Rejected(_)), "{input}");
        assert!(parse_module(input).is_err(), "{input}");
    }
}
