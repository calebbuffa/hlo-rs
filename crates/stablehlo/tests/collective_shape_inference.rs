use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{ElementType, TensorType, Type, infer_all_gather_types, infer_reduce_scatter_type};

mod upstream_support;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn source(input: &str, output: &str, scatter: bool, group: Option<u64>) -> String {
    let (name, dimension, region) = if scatter {
        (
            "reduce_scatter",
            "scatter_dimension",
            "({^bb0(%a:tensor<f32>,%b:tensor<f32>): \
             \"stablehlo.return\"(%a):(tensor<f32>)->()})",
        )
    } else {
        ("all_gather", "all_gather_dim", "")
    };

    let groups = group.map_or_else(
        || "dense<>:tensor<0x0xi64>".into(),
        |size| {
            let ids = (0..size)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("dense<[[{ids}]]>:tensor<1x{size}xi64>")
        },
    );
    format!(
        "func.func @collective(%x:{input})->{output} {{ \
         %r=\"stablehlo.{name}\"(%x){region}{{{dimension}=0:i64,replica_groups={groups}}} \
         :({input})->{output} return %r:{output} }}"
    )
}

fn accepted(input: &str, output: &str, scatter: bool, group: Option<u64>) -> bool {
    let source = source(input, output, scatter, group);
    let Ok(module) = parse_module(&source) else {
        return false;
    };
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module,
        "{source}"
    );
    true
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn collective_shapes_roundtrip_through_independent_upstream() {
    use upstream_support::UpstreamResult;
    let mut positives = Vec::new();
    for (small, large) in [
        ("tensor<2xf32>", "tensor<4xf32>"),
        ("tensor<?xf32,#stablehlo.bounds<2>>", "tensor<4xf32>"),
        ("tensor<2xf32>", "tensor<?xf32,#stablehlo.bounds<4>>"),
        ("tensor<?xf32>", "tensor<?xf32>"),
    ] {
        positives.push(source(small, large, false, Some(2)));
        positives.push(source(large, small, true, Some(2)));
    }
    let emitted = positives
        .iter()
        .map(|s| print_module(&parse_module(s).unwrap()).unwrap())
        .collect::<Vec<_>>();
    for (source, result) in emitted.iter().zip(upstream_support::verify(&emitted)) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("upstream rejected {source}: {error}"),
        };
        parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    }
    let invalid = [
        source("tensor<3xf32>", "tensor<4xf32>", false, Some(2)),
        source("tensor<5xf32>", "tensor<2xf32>", true, Some(2)),
        source("tensor<2x3xf32>", "tensor<4x4xf32>", false, Some(2)),
        source("tensor<4x4xf32>", "tensor<2x3xf32>", true, Some(2)),
    ];
    for (source, result) in invalid.iter().zip(upstream_support::verify(&invalid)) {
        assert!(matches!(result, UpstreamResult::Rejected(_)), "{source}");
        assert!(parse_module(source).is_err(), "{source}");
    }
    eprintln!(
        "Eight all-gather/reduce-scatter modules roundtrip through independent StableHLO 1.20.0; four malformed static geometries reject in both parsers. Bounds completeness, exact group-factor validation and target-1.22 execution are not established by this audit."
    );
}

#[test]
fn all_gather_infers_each_operand_shape_elements_and_bounds_independently() {
    let operands = vec![
        ty("tensor<2x3xf32>"),
        ty("tensor<?x4xi32,#stablehlo.bounds<7,?>>"),
        ty("tensor<5x!quant.uniform<i8:f32,0.25:1>>"),
    ];
    assert_eq!(
        infer_all_gather_types(&operands, 0, 3).unwrap(),
        vec![
            ty("tensor<6x3xf32>"),
            ty("tensor<?x4xi32,#stablehlo.bounds<21,?>>"),
            ty("tensor<15x!quant.uniform<i8:f32,0.25:1>>"),
        ]
    );
    assert_eq!(infer_all_gather_types(&operands, 0, 1).unwrap(), operands);
    assert_eq!(infer_all_gather_types(&[], 0, 2).unwrap(), vec![]);
}

#[test]
fn reduce_scatter_divides_bounds_and_uses_scalar_reducer_return_elements() {
    let accumulator = ty("tensor<f32>");
    assert_eq!(
        infer_reduce_scatter_type(&ty("tensor<6x3xf16>"), &accumulator, 0, 2).unwrap(),
        ty("tensor<3x3xf32>")
    );
    assert_eq!(
        infer_reduce_scatter_type(
            &ty("tensor<?x3xf16,#stablehlo.bounds<7,?>>"),
            &accumulator,
            0,
            2
        )
        .unwrap(),
        ty("tensor<?x3xf32,#stablehlo.bounds<3,?>>")
    );
    let quantized = ty("tensor<6x!quant.uniform<i8:f32,0.25:1>>");
    let reducer = ty("tensor<!quant.uniform<i8:f32,0.5:2>>");
    assert_eq!(
        infer_reduce_scatter_type(&quantized, &reducer, 0, 2).unwrap(),
        ty("tensor<3x!quant.uniform<i8:f32,0.5:2>>")
    );
}

#[test]
fn collective_inference_rejects_invalid_geometry_domains_and_int64_overflow() {
    assert!(infer_all_gather_types(&[], -1, 2).is_err());
    for count in [i64::MIN, -1, 0] {
        assert!(infer_all_gather_types(&[ty("tensor<2xf32>")], 0, count).is_err());
        assert!(
            infer_reduce_scatter_type(&ty("tensor<2xf32>"), &ty("tensor<f32>"), 0, count).is_err()
        );
    }

    for input in [
        "tensor<f32>",
        "tensor<0xf32>",
        "tensor<?xf32,#stablehlo.bounds<0>>",
        "tensor<*xf32>",
        "memref<2xf32>",
    ] {
        assert!(
            infer_all_gather_types(&[ty(input)], 0, 2).is_err(),
            "{input}"
        );
        assert!(
            infer_reduce_scatter_type(&ty(input), &ty("tensor<f32>"), 0, 2).is_err(),
            "{input}"
        );
    }
    for axis in [-1, 1, i64::MAX] {
        assert!(infer_all_gather_types(&[ty("tensor<2xf32>")], axis, 2).is_err());
    }
    for input in ["tensor<5xf32>", "tensor<?xf32,#stablehlo.bounds<1>>"] {
        assert!(infer_reduce_scatter_type(&ty(input), &ty("tensor<f32>"), 0, 2).is_err());
    }
    assert!(infer_reduce_scatter_type(&ty("tensor<6xf32>"), &ty("tensor<1xf32>"), 0, 2).is_err());
    let large: Type = TensorType::static_shape(ElementType::F32, [i64::MAX as u64]).into();
    assert!(infer_all_gather_types(&[large], 0, 2).is_err());
}

#[test]
fn dynamic_gather_bounds_respect_the_exact_int64_product_domain() {
    for count in [2, 3, 7, i64::MAX] {
        let input = ty(&format!("tensor<?xf32,#stablehlo.bounds<{}>>", i64::MAX));
        let bound = (i64::MAX / count) * count;
        assert_eq!(
            infer_all_gather_types(&[input], 0, count).unwrap(),
            vec![ty(&format!("tensor<?xf32,#stablehlo.bounds<{bound}>>"))]
        );
    }
    assert_eq!(
        infer_all_gather_types(&[ty("tensor<?xf32>")], 0, 2).unwrap(),
        vec![ty("tensor<?xf32>")]
    );
}

#[test]
fn generated_inference_matches_exact_static_products_and_dynamic_bound_extrema() {
    for size in 0..=9 {
        for count in 1..=5 {
            let input = ty(&format!("tensor<{size}xf32>"));
            let gathered = infer_all_gather_types(std::slice::from_ref(&input), 0, count);
            if size == 0 {
                assert!(gathered.is_err());
            } else {
                assert_eq!(
                    gathered.unwrap(),
                    vec![ty(&format!("tensor<{}xf32>", size * count))]
                );
            }
            let scattered = infer_reduce_scatter_type(&input, &ty("tensor<f32>"), 0, count);
            if size == 0 || size % count != 0 {
                assert!(scattered.is_err());
            } else {
                assert_eq!(
                    scattered.unwrap(),
                    ty(&format!("tensor<{}xf32>", size / count))
                );
            }
            let bounded = ty(&format!("tensor<?xf32,#stablehlo.bounds<{size}>>"));
            let gathered = infer_all_gather_types(std::slice::from_ref(&bounded), 0, count);
            let scattered = infer_reduce_scatter_type(&bounded, &ty("tensor<f32>"), 0, count);
            if size == 0 {
                assert!(gathered.is_err());
            } else {
                assert_eq!(
                    gathered.unwrap(),
                    vec![ty(&format!(
                        "tensor<?xf32,#stablehlo.bounds<{}>>",
                        size * count
                    ))]
                );
            }
            if size < count {
                assert!(scattered.is_err());
            } else {
                assert_eq!(
                    scattered.unwrap(),
                    ty(&format!(
                        "tensor<?xf32,#stablehlo.bounds<{}>>",
                        size / count
                    ))
                );
            }
        }
    }
}

#[test]
fn collective_verifier_checks_bounds_in_both_directions_and_nonexchange_axes() {
    for (small, large, expected) in [
        ("tensor<?xf32,#stablehlo.bounds<2>>", "tensor<6xf32>", false),
        ("tensor<3xf32>", "tensor<?xf32,#stablehlo.bounds<5>>", false),
        ("tensor<?xf32>", "tensor<5xf32>", false),
        ("tensor<?xf32,#stablehlo.bounds<2>>", "tensor<4xf32>", true),
        ("tensor<2xf32>", "tensor<?xf32,#stablehlo.bounds<4>>", true),
        (
            "tensor<?x4xf32>",
            "tensor<?x?xf32,#stablehlo.bounds<?,3>>",
            false,
        ),
        (
            "tensor<?x?xf32,#stablehlo.bounds<?,3>>",
            "tensor<?x4xf32>",
            false,
        ),
        ("tensor<?xf32,#stablehlo.bounds<0>>", "tensor<?xf32>", false),
    ] {
        assert_eq!(
            accepted(small, large, false, Some(2)),
            expected,
            "gather {small} -> {large}"
        );
        assert_eq!(
            accepted(large, small, true, Some(2)),
            expected,
            "scatter {large} -> {small}"
        );
    }
    for (small, large, expected) in [
        ("tensor<3xf32>", "tensor<5xf32>", false),
        ("tensor<3xf32>", "tensor<?xf32,#stablehlo.bounds<2>>", false),
        ("tensor<?xf32,#stablehlo.bounds<1>>", "tensor<5xf32>", true),
        ("tensor<3xf32>", "tensor<?xf32,#stablehlo.bounds<5>>", true),
    ] {
        assert_eq!(accepted(small, large, false, None), expected);
        assert_eq!(accepted(large, small, true, None), expected);
    }
}

#[test]
fn generated_collective_bounds_match_enumerated_positive_extent_witnesses() {
    let witnesses = [
        (Some(0), None),
        (Some(1), None),
        (Some(2), None),
        (Some(5), None),
        (None, None),
        (None, Some(0)),
        (None, Some(1)),
        (None, Some(2)),
        (None, Some(5)),
    ];
    let tensor = |size: Option<u64>, bound: Option<u64>| {
        let size = size.map_or_else(|| "?".into(), |size| size.to_string());
        let encoding =
            bound.map_or_else(String::new, |bound| format!(",#stablehlo.bounds<{bound}>"));
        format!("tensor<{size}xf32{encoding}>")
    };
    let fits = |value: u64, size: Option<u64>, bound: Option<u64>| {
        size.is_none_or(|size| size == value) && bound.is_none_or(|bound| value <= bound)
    };
    for (small_size, small_bound) in witnesses {
        for (large_size, large_bound) in witnesses {
            let small = tensor(small_size, small_bound);
            let large = tensor(large_size, large_bound);
            for group in [None, Some(1), Some(2), Some(3)] {
                let possible = (1..=5).any(|s| {
                    fits(s, small_size, small_bound)
                        && (1..=15).any(|l| {
                            fits(l, large_size, large_bound)
                                && group.map_or(l.is_multiple_of(s), |count| l == s * count)
                        })
                });
                assert_eq!(
                    accepted(&small, &large, false, group),
                    possible,
                    "gather {small} -> {large}, group {group:?}"
                );
                assert_eq!(
                    accepted(&large, &small, true, group),
                    possible,
                    "scatter {large} -> {small}, group {group:?}"
                );
            }
        }
    }
}
