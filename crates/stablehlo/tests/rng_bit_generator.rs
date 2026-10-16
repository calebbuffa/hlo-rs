use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{RngAlgorithm, Type, infer_rng_bit_generator_types};

mod upstream_support;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn source(input: &str, state: &str, output: &str, algorithm: &str) -> String {
    format!(
        "func.func @bits(%s:{input})->({state},{output}) {{ \
        %r:2=stablehlo.rng_bit_generator %s,algorithm={algorithm}:({input})->({state},{output}) \
        return %r#0,%r#1:{state},{output} }}"
    )
}

#[test]
fn inferred_state_preserves_metadata_and_output_is_an_explicit_static_type() {
    let state = ty("tensor<?xui64,#stablehlo.bounds<3>>");
    let output = ty("tensor<0x4xf32>");
    for algorithm in [
        RngAlgorithm::Default,
        RngAlgorithm::ThreeFry,
        RngAlgorithm::Philox,
    ] {
        assert_eq!(
            infer_rng_bit_generator_types(&state, &output, algorithm).unwrap(),
            vec![state.clone(), output.clone()]
        );
    }
    for output in [
        "tensor<2xi1>",
        "tensor<2xcomplex<f32>>",
        "tensor<?xf32>",
        "tensor<2x!quant.uniform<i8:f32,0.5>>",
        "!stablehlo.token",
    ] {
        assert!(
            infer_rng_bit_generator_types(&state, &ty(output), RngAlgorithm::Default).is_err(),
            "{output}"
        );
    }
}

#[test]
fn state_domains_and_algorithm_sizes_follow_the_written_specification() {
    let output = ty("tensor<2xui32>");
    for state in [
        "tensor<2xi64>",
        "tensor<2xf32>",
        "tensor<2xui32>",
        "tensor<ui64>",
        "tensor<1x2xui64>",
        "tensor<2x!quant.uniform<i8:f32,0.5>>",
        "!stablehlo.token",
    ] {
        assert!(
            infer_rng_bit_generator_types(&ty(state), &output, RngAlgorithm::Default).is_err(),
            "{state}"
        );
    }
    for size in 0..=5 {
        let state = ty(&format!("tensor<{size}xui64>"));
        for (algorithm, valid) in [
            (RngAlgorithm::Default, true),
            (RngAlgorithm::ThreeFry, size == 2),
            (RngAlgorithm::Philox, size == 2 || size == 3),
        ] {
            assert_eq!(
                infer_rng_bit_generator_types(&state, &output, algorithm).is_ok(),
                valid,
                "{state:?} {algorithm:?}"
            );
        }
    }
    for bound in 0..=4 {
        let state = ty(&format!("tensor<?xui64,#stablehlo.bounds<{bound}>>"));
        for algorithm in [RngAlgorithm::ThreeFry, RngAlgorithm::Philox] {
            assert_eq!(
                infer_rng_bit_generator_types(&state, &output, algorithm).is_ok(),
                bound >= 2,
                "{bound} {algorithm:?}"
            );
        }
    }
}

#[test]
fn state_result_constraints_are_combined_with_algorithm_and_both_bounds() {
    let witnesses = [
        ("tensor<0xui64>", Some(0), None),
        ("tensor<2xui64>", Some(2), None),
        ("tensor<3xui64>", Some(3), None),
        ("tensor<4xui64>", Some(4), None),
        ("tensor<?xui64>", None, None),
        ("tensor<?xui64,#stablehlo.bounds<1>>", None, Some(1)),
        ("tensor<?xui64,#stablehlo.bounds<2>>", None, Some(2)),
        ("tensor<?xui64,#stablehlo.bounds<3>>", None, Some(3)),
    ];
    for (input, input_size, input_bound) in witnesses {
        for (state, state_size, state_bound) in witnesses {
            for algorithm in ["DEFAULT", "THREE_FRY", "PHILOX"] {
                let expected = (0..=4).any(|size| {
                    input_size.is_none_or(|value| value == size)
                        && state_size.is_none_or(|value| value == size)
                        && input_bound.is_none_or(|value| size <= value)
                        && state_bound.is_none_or(|value| size <= value)
                        && match algorithm {
                            "THREE_FRY" => size == 2,
                            "PHILOX" => size == 2 || size == 3,
                            _ => true,
                        }
                });
                let input = source(input, state, "tensor<2xf32>", algorithm);
                let result = parse_module(&input);
                assert_eq!(result.is_ok(), expected, "{input}\n{result:?}");
                if let Ok(module) = result {
                    assert_eq!(
                        parse_module(&print_module(&module).unwrap()).unwrap(),
                        module
                    );
                }
            }
        }
    }
    assert!(
        parse_module(&source(
            "tensor<2xui64>",
            "tensor<2xf32>",
            "tensor<2xf32>",
            "DEFAULT"
        ))
        .is_err()
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn specified_bit_generator_signatures_roundtrip_through_independent_upstream() {
    use upstream_support::UpstreamResult;
    let positives = [
        source(
            "tensor<2xui64>",
            "tensor<2xui64>",
            "tensor<0x3xui32>",
            "THREE_FRY",
        ),
        source(
            "tensor<3xui64>",
            "tensor<3xui64>",
            "tensor<2xf32>",
            "PHILOX",
        ),
        source(
            "tensor<?xui64>",
            "tensor<?xui64>",
            "tensor<2xi16>",
            "DEFAULT",
        ),
        source(
            "tensor<?xui64,#stablehlo.bounds<2>>",
            "tensor<2xui64>",
            "tensor<2xi32>",
            "THREE_FRY",
        ),
    ];
    let emitted = positives
        .iter()
        .map(|input| print_module(&parse_module(input).unwrap()).unwrap())
        .collect::<Vec<_>>();
    for (input, result) in emitted.iter().zip(upstream_support::verify(&emitted)) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("{input}\n{error}"),
        };
        parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    }
    let invalid = [
        source(
            "tensor<2xui64>",
            "tensor<3xui64>",
            "tensor<2xi32>",
            "DEFAULT",
        ),
        source(
            "tensor<2xui64>",
            "tensor<2xui64>",
            "tensor<?xi32>",
            "DEFAULT",
        ),
    ];
    for (input, result) in invalid.iter().zip(upstream_support::verify(&invalid)) {
        assert!(matches!(result, UpstreamResult::Rejected(_)), "{input}");
        assert!(parse_module(input).is_err(), "{input}");
    }
    // The reference only checks compatible shapes; the written specification is stricter.
    let discrepancies = [
        source("tensor<2xf32>", "tensor<2xf32>", "tensor<2xi32>", "DEFAULT"),
        source(
            "tensor<4xui64>",
            "tensor<4xui64>",
            "tensor<2xi32>",
            "THREE_FRY",
        ),
        source(
            "tensor<2xui64>",
            "tensor<2xf32>",
            "tensor<2xi32>",
            "DEFAULT",
        ),
    ];
    for (input, result) in discrepancies
        .iter()
        .zip(upstream_support::verify(&discrepancies))
    {
        assert!(
            matches!(result, UpstreamResult::Accepted(_)),
            "upstream discrepancy changed: {input}"
        );
        assert!(parse_module(input).is_err(), "{input}");
    }
}
