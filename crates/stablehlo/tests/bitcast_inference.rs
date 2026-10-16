use stablehlo::text::{parse_module, parse_type, print_module, print_type};
use stablehlo::{InferenceError, Type, infer_bitcast_convert_type};

mod upstream_support;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn source(input: &str, output: &str) -> String {
    format!(
        "func.func @bitcast(%x:{input})->{output} {{ \
        %r=stablehlo.bitcast_convert %x:({input})->{output} return %r:{output} }}"
    )
}

#[test]
fn inference_projects_bounds_and_ignores_template_shape_and_encoding() {
    for (input, template, expected) in [
        (
            "tensor<?x3xf32,#stablehlo.bounds<5,?>>",
            "tensor<99xi8>",
            "tensor<?x3x4xi8,#stablehlo.bounds<5,?,?>>",
        ),
        (
            "tensor<?x?xi8,#stablehlo.bounds<5,4>>",
            "tensor<f32>",
            "tensor<?xf32,#stablehlo.bounds<5>>",
        ),
        (
            "tensor<?xf32,#stablehlo.bounds<5>>",
            "tensor<?x?xi32,#stablehlo.bounds<99,99>>",
            "tensor<?xi32,#stablehlo.bounds<5>>",
        ),
        (
            "tensor<complex<f64>>",
            "tensor<complex<f32>>",
            "tensor<2xcomplex<f32>>",
        ),
        (
            "tensor<2xcomplex<f32>>",
            "tensor<complex<f64>>",
            "tensor<complex<f64>>",
        ),
        (
            "tensor<2xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>",
            "tensor<ui32>",
            "tensor<2xui32>",
        ),
    ] {
        let inferred = infer_bitcast_convert_type(&ty(input), &ty(template)).unwrap();
        assert_eq!(inferred, ty(expected), "{input} -> {template}");
        let module = parse_module(&source(input, &print_type(&inferred).unwrap())).unwrap();
        assert_eq!(
            parse_module(&print_module(&module).unwrap()).unwrap(),
            module
        );
    }
}

#[test]
fn storage_widths_and_selected_per_axis_parameters_define_quantized_results() {
    for (input, template, expected) in [
        (
            "tensor<2x!quant.uniform<i8:f32,0.25:1>>",
            "tensor<i4>",
            "tensor<2x2xi4>",
        ),
        (
            "tensor<2xi16>",
            "tensor<1x2x!quant.uniform<i8:f32:1,{0.25:1,0.5:2}>>",
            "tensor<2x2x!quant.uniform<i8:f32:1,{0.25:1,0.5:2}>>",
        ),
        (
            "tensor<2x2x!quant.uniform<i8:f32:1,{0.25:1,0.5:2}>>",
            "tensor<i16>",
            "tensor<2xi16>",
        ),
    ] {
        assert_eq!(
            infer_bitcast_convert_type(&ty(input), &ty(template)).unwrap(),
            ty(expected)
        );
    }
    assert!(
        infer_bitcast_convert_type(
            &ty("tensor<3xi16>"),
            &ty("tensor<3x!quant.uniform<i8:f32:0,{0.25:1,0.5:2,0.75:3}>>")
        )
        .is_ok()
    );
    assert!(
        infer_bitcast_convert_type(
            &ty("tensor<3xi16>"),
            &ty("tensor<2x!quant.uniform<i8:f32:0,{0.25:1,0.5:2}>>")
        )
        .is_err()
    );
}

#[test]
fn invalid_ratios_minor_extents_complex_mixing_and_domains_fail_explicitly() {
    for (input, output) in [
        ("tensor<f6E2M3FN>", "tensor<i4>"),
        ("tensor<f4E2M1FN>", "tensor<f6E3M2FN>"),
    ] {
        assert!(matches!(
            infer_bitcast_convert_type(&ty(input), &ty(output)),
            Err(InferenceError::NonIntegralBitcastRatio { .. })
        ));
    }
    for input in [
        "tensor<i8>",
        "tensor<3xi8>",
        "tensor<?xi8,#stablehlo.bounds<3>>",
    ] {
        assert!(
            matches!(
                infer_bitcast_convert_type(&ty(input), &ty("tensor<f32>")),
                Err(InferenceError::InvalidBitcastMinorExtent)
            ),
            "{input}"
        );
    }
    for (input, output) in [
        ("tensor<complex<f32>>", "tensor<f64>"),
        ("tensor<f64>", "tensor<complex<f32>>"),
        ("tensor<index>", "tensor<i64>"),
        ("tensor<si32>", "tensor<i32>"),
        ("tensor<tf32>", "tensor<i32>"),
        ("tensor<*xf32>", "tensor<i32>"),
        ("!stablehlo.token", "tensor<i32>"),
    ] {
        assert!(
            infer_bitcast_convert_type(&ty(input), &ty(output)).is_err(),
            "{input} -> {output}"
        );
    }
}

#[test]
fn generated_element_pairs_match_independent_rank_and_width_equations() {
    let elements = [
        ("i1", 1, false),
        ("i2", 2, false),
        ("i4", 4, false),
        ("i8", 8, false),
        ("i16", 16, false),
        ("i32", 32, false),
        ("i64", 64, false),
        ("ui4", 4, false),
        ("f4E2M1FN", 4, false),
        ("f6E2M3FN", 6, false),
        ("f8E4M3FN", 8, false),
        ("f16", 16, false),
        ("bf16", 16, false),
        ("f32", 32, false),
        ("f64", 64, false),
        ("complex<f32>", 64, true),
        ("complex<f64>", 128, true),
    ];
    for (input_element, input_bits, input_complex) in elements {
        for (output_element, output_bits, output_complex) in elements {
            let valid = input_complex == output_complex
                && (input_bits.max(output_bits) % input_bits.min(output_bits) == 0);
            let input = if input_bits < output_bits && valid {
                format!("tensor<3x{}x{input_element}>", output_bits / input_bits)
            } else {
                format!("tensor<3x{input_element}>")
            };
            let template = ty(&format!("tensor<{output_element}>"));
            let result = infer_bitcast_convert_type(&ty(&input), &template);
            assert_eq!(result.is_ok(), valid, "{input_element} -> {output_element}");
            if let Ok(result) = result {
                let expected = if input_bits > output_bits {
                    format!("tensor<3x{}x{output_element}>", input_bits / output_bits)
                } else {
                    format!("tensor<3x{output_element}>")
                };
                assert_eq!(result, ty(&expected));
                assert!(parse_module(&source(&input, &expected)).is_ok());
            }
        }
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn inferred_bitcast_shapes_roundtrip_through_independent_upstream() {
    use upstream_support::UpstreamResult;
    let pairs = [
        ("tensor<3xf32>", "tensor<i8>"),
        ("tensor<3x4xi8>", "tensor<f32>"),
        ("tensor<f32>", "tensor<i32>"),
        ("tensor<complex<f64>>", "tensor<complex<f32>>"),
        ("tensor<?xf32,#stablehlo.bounds<5>>", "tensor<i8>"),
        ("tensor<?x?xi8,#stablehlo.bounds<5,4>>", "tensor<f32>"),
    ];
    let modules = pairs
        .iter()
        .map(|(input, template)| {
            let result = infer_bitcast_convert_type(&ty(input), &ty(template)).unwrap();
            print_module(&parse_module(&source(input, &print_type(&result).unwrap())).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    for (input, result) in modules.iter().zip(upstream_support::verify(&modules)) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("{input}\n{error}"),
        };
        parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    }
}
