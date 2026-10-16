use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{Type, infer_collective_broadcast_types, infer_collective_permute_type};

mod upstream_support;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn source(input: &str, output: &str, permute: bool) -> String {
    let (name, attributes) = if permute {
        (
            "collective_permute",
            "source_target_pairs=dense<[[0,1]]>:tensor<1x2xi64>",
        )
    } else {
        (
            "collective_broadcast",
            "replica_groups=dense<[[0,1]]>:tensor<1x2xi64>",
        )
    };
    format!(
        "func.func @identity(%x:{input})->{output} {{ \
         %r=\"stablehlo.{name}\"(%x){{{attributes}}}:({input})->{output} \
         return %r:{output} }}"
    )
}

fn accepted(input: &str, output: &str, permute: bool) -> bool {
    let source = source(input, output, permute);
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
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn identity_collectives_roundtrip_through_independent_upstream() {
    use upstream_support::UpstreamResult;
    let positives = [false, true]
        .into_iter()
        .flat_map(|permute| {
            [
                source("tensor<3xf32>", "tensor<?xf32>", permute),
                source(
                    "tensor<?xf32,#stablehlo.bounds<5>>",
                    "tensor<3xf32>",
                    permute,
                ),
                source(
                    "tensor<?xf32>",
                    "tensor<?xf32,#stablehlo.bounds<5>>",
                    permute,
                ),
            ]
        })
        .collect::<Vec<_>>();
    let emitted = positives
        .iter()
        .map(|source| print_module(&parse_module(source).unwrap()).unwrap())
        .collect::<Vec<_>>();
    for (source, result) in emitted.iter().zip(upstream_support::verify(&emitted)) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("upstream rejected {source}: {error}"),
        };
        parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    }
    let invalid = [false, true]
        .into_iter()
        .flat_map(|permute| {
            [
                source("tensor<3xf32>", "tensor<4xf32>", permute),
                source("tensor<3xf32>", "tensor<3xi32>", permute),
            ]
        })
        .collect::<Vec<_>>();
    for (source, result) in invalid.iter().zip(upstream_support::verify(&invalid)) {
        assert!(matches!(result, UpstreamResult::Rejected(_)), "{source}");
        assert!(parse_module(source).is_err(), "{source}");
    }
    eprintln!(
        "Six collective-broadcast/permute modules roundtrip through independent StableHLO 1.20.0; four malformed static shape/element signatures rejected by both parsers. Not exhaustive bounds or target-1.22 execution validation."
    );
}

#[test]
fn generated_identity_collective_verification_matches_exact_extent_feasibility() {
    let witnesses = [
        (Some(0), None),
        (Some(2), None),
        (Some(5), None),
        (None, None),
        (None, Some(0)),
        (None, Some(2)),
        (None, Some(5)),
    ];
    let tensor = |size: Option<u64>, bound: Option<u64>| {
        let size = size.map_or_else(|| "?".into(), |size| size.to_string());
        let encoding =
            bound.map_or_else(String::new, |bound| format!(",#stablehlo.bounds<{bound}>"));
        format!("tensor<{size}xf32{encoding}>")
    };
    for (left_size, left_bound) in witnesses {
        for (right_size, right_bound) in witnesses {
            let left = tensor(left_size, left_bound);
            let right = tensor(right_size, right_bound);
            let possible = (0..=5).any(|value| {
                left_size.is_none_or(|size| size == value)
                    && right_size.is_none_or(|size| size == value)
                    && left_bound.is_none_or(|bound| value <= bound)
                    && right_bound.is_none_or(|bound| value <= bound)
            });
            for permute in [false, true] {
                assert_eq!(
                    accepted(&left, &right, permute),
                    possible,
                    "{left} -> {right}, permute {permute}"
                );
            }
        }
    }
}

#[test]
fn identity_collectives_preserve_independent_shapes_and_exact_element_metadata() {
    let inputs = vec![
        ty("tensor<2x3xf32>"),
        ty("tensor<?xi32,#stablehlo.bounds<7>>"),
        ty("tensor<3x!quant.uniform<i8:f32,0.25:1>>"),
        ty("tensor<3xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>"),
    ];
    assert_eq!(
        infer_collective_broadcast_types(&inputs, false).unwrap(),
        inputs
    );
    for input in &inputs {
        assert_eq!(infer_collective_permute_type(input).unwrap(), *input);
    }
    let mut dynamic = inputs.clone();
    dynamic.push(ty("tensor<4xi32>"));
    assert_eq!(
        infer_collective_broadcast_types(&dynamic, true).unwrap(),
        inputs
    );
}

#[test]
fn dynamic_roots_require_an_i32_vector_with_one_element_per_data_operand() {
    for root in [
        "tensor<i32>",
        "tensor<2xi32>",
        "tensor<1xi64>",
        "tensor<1xui32>",
        "tensor<?xi32>",
        "tensor<1x1xi32>",
    ] {
        assert!(
            infer_collective_broadcast_types(&[ty("tensor<3xf32>"), ty(root)], true).is_err(),
            "{root}"
        );
    }
    let data = ty("tensor<3xf32>");
    assert_eq!(
        infer_collective_broadcast_types(&[data.clone(), ty("tensor<1xi32>")], true).unwrap(),
        vec![data]
    );
    for (inputs, dynamic) in [
        (vec![], false),
        (vec![], true),
        (vec![ty("tensor<0xi32>")], true),
    ] {
        assert!(infer_collective_broadcast_types(&inputs, dynamic).is_err());
    }
}

#[test]
fn identity_collectives_reject_tokens_buffers_unranked_and_per_axis_values() {
    for input in [
        Type::Token,
        Type::Tuple(vec![]),
        ty("memref<3xf32>"),
        ty("tensor<*xf32>"),
        ty("tensor<2x!quant.uniform<i8:f32:0,{0.25:1,0.5:2}>>"),
    ] {
        assert!(infer_collective_broadcast_types(std::slice::from_ref(&input), false).is_err());
        assert!(infer_collective_permute_type(&input).is_err());
    }
}

#[test]
fn identity_collective_verifiers_reject_contradictory_shape_bound_witnesses() {
    for permute in [false, true] {
        for (input, output, expected) in [
            ("tensor<?xf32,#stablehlo.bounds<3>>", "tensor<4xf32>", false),
            ("tensor<4xf32>", "tensor<?xf32,#stablehlo.bounds<3>>", false),
            ("tensor<?xf32,#stablehlo.bounds<3>>", "tensor<3xf32>", true),
            ("tensor<3xf32>", "tensor<?xf32,#stablehlo.bounds<3>>", true),
            (
                "tensor<?xf32,#stablehlo.bounds<3>>",
                "tensor<?xf32,#stablehlo.bounds<5>>",
                true,
            ),
            ("tensor<3xf32>", "tensor<3xi32>", false),
        ] {
            assert_eq!(
                accepted(input, output, permute),
                expected,
                "{input} -> {output}, permute {permute}"
            );
        }
    }
}
