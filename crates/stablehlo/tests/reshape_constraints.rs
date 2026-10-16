use stablehlo::text::{parse_module, print_module};

mod upstream_support;

fn source(input: &str, output: &str, dynamic: bool, known: Option<&[i64]>) -> String {
    let opcode = if dynamic {
        "dynamic_reshape"
    } else {
        "reshape"
    };
    let rank = output
        .split('<')
        .nth(1)
        .unwrap()
        .split('x')
        .take_while(|part| *part == "?" || part.bytes().all(|byte| byte.is_ascii_digit()))
        .count();
    let shape_type = format!("tensor<{rank}xi64>");
    let parameter = if dynamic && known.is_none() {
        format!(",%s:{shape_type}")
    } else {
        String::new()
    };
    let constant = known.map_or_else(String::new, |values| {
        format!(
            "%s=stablehlo.constant dense<[{}]>:{shape_type}",
            values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    });
    let (operands, types) = if dynamic {
        ("%x,%s".to_string(), format!("{input},{shape_type}"))
    } else {
        ("%x".to_string(), input.to_string())
    };
    format!(
        "func.func @reshape(%x:{input}{parameter})->{output} {{ {constant} \
        %r=\"stablehlo.{opcode}\"({operands}):({types})->{output} return %r:{output} }}"
    )
}

fn quantized(dimensions: &str, axis: usize, count: usize) -> String {
    let parameters = vec!["0.25:0"; count].join(",");
    format!("tensor<{dimensions}!quant.uniform<i8:f32:{axis},{{{parameters}}}>>")
}

#[test]
fn zero_extents_and_uncancelled_static_factors_cannot_hide_in_dynamic_shapes() {
    for (input, output, expected) in [
        ("tensor<0x?xf32>", "tensor<1xf32>", false),
        ("tensor<0x?xf32>", "tensor<0xf32>", true),
        ("tensor<?x2xf32>", "tensor<3xf32>", false),
        ("tensor<?x2xf32>", "tensor<4xf32>", true),
        ("tensor<?x2xf32>", "tensor<0xf32>", true),
        (
            "tensor<3xf32>",
            "tensor<?x2xf32,#stablehlo.bounds<5,?>>",
            false,
        ),
        (
            "tensor<4xf32>",
            "tensor<?x2xf32,#stablehlo.bounds<5,?>>",
            true,
        ),
        (
            "tensor<1xf32>",
            "tensor<0x?xf32,#stablehlo.bounds<?,5>>",
            false,
        ),
    ] {
        for dynamic in [false, true] {
            let input = source(input, output, dynamic, None);
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

#[test]
fn per_axis_reshape_checks_suffix_even_when_zero_prefix_annuls_total_count() {
    for (input_dims, input_axis, output_dims, output_axis, expected) in [
        ("0x2x3x", 1, "0x2x4x", 1, false),
        ("0x2x3x", 1, "0x1x2x3x", 2, true),
        ("0x2x3x", 1, "0x2x1x3x", 1, true),
        ("0x2x3x", 1, "0x2x3x1x", 1, true),
        ("0x2x3x", 1, "0x2x?x2x", 1, false),
    ] {
        for dynamic in [false, true] {
            let output = if output_dims.contains('?') && !dynamic {
                quantized(output_dims, output_axis, 2)
                    .replace(">>", ">,#stablehlo.bounds<?,?,4,?>>")
            } else {
                quantized(output_dims, output_axis, 2)
            };
            let input = source(
                &quantized(input_dims, input_axis, 2),
                &output,
                dynamic,
                None,
            );
            assert_eq!(parse_module(&input).is_ok(), expected, "{input}");
        }
    }
}

#[test]
fn known_dynamic_reshape_shape_rechecks_per_axis_parameters_and_suffixes() {
    let input = quantized("0x2x3x", 1, 2);
    let output = quantized("0x?x?x", 1, 2);
    for (values, expected) in [([0, 2, 3], true), ([0, 2, 4], false), ([0, 3, 3], false)] {
        let input = source(&input, &output, true, Some(&values));
        assert_eq!(parse_module(&input).is_ok(), expected, "{input}");
    }
}

#[test]
fn generated_zero_prefix_reshapes_match_independent_suffix_products() {
    for lhs_suffix in 0..=5 {
        for rhs_first in 0..=3 {
            for rhs_second in 0..=3 {
                let input = quantized(&format!("0x2x{lhs_suffix}x"), 1, 2);
                let output = quantized(&format!("0x2x{rhs_first}x{rhs_second}x"), 1, 2);
                for dynamic in [false, true] {
                    let input = source(&input, &output, dynamic, None);
                    assert_eq!(
                        parse_module(&input).is_ok(),
                        lhs_suffix == rhs_first * rhs_second,
                        "{input}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn reshape_common_contracts_and_reference_dynamic_gaps_are_classified() {
    use upstream_support::UpstreamResult;
    let positives = [
        source("tensor<2x3xf32>", "tensor<6xf32>", false, None),
        source("tensor<?x2xf32>", "tensor<4xf32>", false, None),
        source("tensor<0x?xf32>", "tensor<0xf32>", true, None),
        source("tensor<2x3xf32>", "tensor<?x?xf32>", true, Some(&[3, 2])),
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
        source("tensor<2x3xf32>", "tensor<7xf32>", false, None),
        source("tensor<2x3xf32>", "tensor<?xf32>", true, Some(&[7])),
    ];
    for (input, result) in invalid.iter().zip(upstream_support::verify(&invalid)) {
        assert!(matches!(result, UpstreamResult::Rejected(_)), "{input}");
        assert!(parse_module(input).is_err(), "{input}");
    }
    let discrepancies = [
        source("tensor<0x?xf32>", "tensor<1xf32>", false, None),
        source("tensor<?x2xf32>", "tensor<3xf32>", true, None),
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
