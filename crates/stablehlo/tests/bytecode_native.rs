use stablehlo::{
    bytecode::{BytecodeContainer, read_module},
    text::parse_module,
};

mod reference_support;
#[path = "upstream_support/structure.rs"]
mod structure;
mod upstream_support;

fn decode(result: upstream_support::UpstreamResult) -> Vec<u8> {
    let hex = match result {
        upstream_support::UpstreamResult::Accepted(hex) => hex,
        upstream_support::UpstreamResult::Rejected(error) => {
            panic!("native bytecode emission failed: {error}")
        }
    };
    assert!(hex.len().is_multiple_of(2));
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn independently_written_native_payloads_decode_to_the_canonical_rust_ir() {
    let sources = [
        "module @math { func.func @main(%x:tensor<2xf32>)->tensor<2xf32> { %r=stablehlo.add %x,%x:tensor<2xf32> return %r:tensor<2xf32> } }",
        "module @meta attributes {test.string=\"hello\", test.bytes=\"\\FF\\00\", test.integer=-42:i64, test.float=1.25:f64, test.dict={a=unit}, test.array=[true, 3:ui8, @symbol]} { func.func @main(%x:tensor<?x2xf32> loc(\"input.rs\":2:3))->tensor<?x2xf32> { return %x:tensor<?x2xf32> } } loc(\"module.rs\":1:2)",
        "module { func.func @main(%t:!stablehlo.token)->!stablehlo.token { return %t:!stablehlo.token } }",
        "module { func.func @main(%x:tensor<complex<f32>>)->tensor<complex<f32>> { return %x:tensor<complex<f32>> } }",
        "module { func.func @main()->tensor<2xi32> { %x=stablehlo.constant dense<[1, -2]>:tensor<2xi32> return %x:tensor<2xi32> } }",
        "module { func.func @main(%x:tensor<?xf32, #stablehlo.bounds<8>>)->tensor<?xf32, #stablehlo.bounds<8>> { return %x:tensor<?xf32, #stablehlo.bounds<8>> } }",
    ];
    let requests = sources
        .iter()
        .flat_map(|source| (0..=6).map(move |version| (version, source.to_string())))
        .collect::<Vec<_>>();
    let expected = upstream_support::verify_native_bytecode(
        &requests
            .iter()
            .map(|(_, source)| source.as_bytes().to_vec())
            .collect::<Vec<_>>(),
    );
    for (((version, source), result), expected) in requests
        .iter()
        .zip(upstream_support::emit_native_bytecode(&requests))
        .zip(expected)
    {
        let bytes = decode(result);
        let container = BytecodeContainer::parse(&bytes).unwrap();
        assert!(
            container
                .attributes()
                .iter()
                .any(|entry| entry.custom_encoding)
        );
        assert!(container.types().iter().any(|entry| entry.custom_encoding));
        let actual =
            read_module(&bytes).unwrap_or_else(|error| panic!("v{version}: {error}\n{source}"));
        let expected = match expected {
            upstream_support::UpstreamResult::Accepted(source) => parse_module(&source).unwrap(),
            upstream_support::UpstreamResult::Rejected(error) => panic!("{error}"),
        };
        structure::assert_equivalent(&expected, &actual);
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE and STABLEHLO_REFERENCE_DIR"]
fn all_reference_modules_import_from_independently_encoded_native_bytecode() {
    let mut labels = Vec::new();
    let mut requests = Vec::new();
    let source = reference_support::pinned_operation_fixtures();
    for (chunk, source) in source
        .split("// -----")
        .enumerate()
        .filter(|(_, source)| !source.contains("expected-error"))
    {
        let Ok(module) = parse_module(source) else {
            continue;
        };
        let source = stablehlo::text::print_module(&module).unwrap();
        for version in 0..=6 {
            labels.push(format!("chunk{} / v{version}", chunk + 1));
            requests.push((version, source.clone()));
        }
    }
    assert_eq!(requests.len(), 195 * 7);
    let mut failures = Vec::new();
    let bytes = upstream_support::emit_native_bytecode(&requests)
        .into_iter()
        .map(decode)
        .collect::<Vec<_>>();
    let expected = upstream_support::verify_native_bytecode(&bytes);
    for ((label, bytes), expected) in labels.iter().zip(bytes).zip(expected) {
        match read_module(&bytes) {
            Ok(actual) => {
                let upstream_support::UpstreamResult::Accepted(source) = expected else {
                    panic!("{label}: reference could not read its output")
                };
                structure::assert_equivalent(&parse_module(&source).unwrap(), &actual);
            }
            Err(error) => failures.push(format!("{label}: {error}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} native import failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn all_stablehlo_custom_attribute_codes_are_independently_decoded() {
    let attributes = [
        "#stablehlo.channel_handle<handle = -7, type = 1>",
        "#stablehlo<comparison_direction LT>",
        "#stablehlo<comparison_type TOTALORDER>",
        "#stablehlo.conv<[b, 0, f]x[o, i, 0]->[b, 0, f]>",
        "#stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>",
        "#stablehlo<fft_type IRFFT>",
        "#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], operand_batching_dims = [2], start_indices_batching_dims = [1], start_index_map = [0], index_vector_dim = 1>",
        "#stablehlo<precision HIGHEST>",
        "#stablehlo<rng_algorithm PHILOX>",
        "#stablehlo<rng_distribution NORMAL>",
        "#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], input_batching_dims = [2], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>",
        "#stablehlo<transpose ADJOINT>",
        "#stablehlo.bounds<?, 8>",
        "#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 1, operand_tuple_indices = [2, 0]>",
        "#stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
        "#stablehlo.result_accuracy_mode<TOLERANCE>",
        "#stablehlo.result_accuracy<atol = 0.01, rtol = 0.02, ulps = 3, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.replica_group_mesh_axes<mesh = #stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = \"x\", size = 4>]>, axes = [#stablehlo.axis_ref<name = \"x\", sub_axis_info = (1)2>]>",
        "#stablehlo<sub_axis_info(1)2>",
        "#stablehlo.axis_ref<name = \"x\", sub_axis_info = (1)2>",
        "#stablehlo.mesh_axis<name = \"x\", size = 4>",
        "#stablehlo.mesh<axes = [#stablehlo.mesh_axis<name = \"x\", size = 4>], device_ids = dense<[2, 0, 3, 1]> : tensor<4xi64>>",
    ];
    let mut module = parse_module("module {}").unwrap();
    module
        .set_attributes(
            [(
                "test.values".to_owned(),
                stablehlo::Attribute::Array(
                    attributes
                        .iter()
                        .map(|source| stablehlo::text::parse_attribute(source).unwrap())
                        .collect(),
                ),
            )]
            .into(),
        )
        .unwrap();
    let source = stablehlo::text::print_module(&module).unwrap();
    let requests = (0..=6)
        .map(|version| (version, source.clone()))
        .collect::<Vec<_>>();
    for (version, result) in upstream_support::emit_native_bytecode(&requests)
        .into_iter()
        .enumerate()
    {
        let bytes = decode(result);
        let container = BytecodeContainer::parse(&bytes).unwrap();
        let codes = container
            .attributes()
            .iter()
            .filter(|entry| {
                entry.custom_encoding && container.dialects()[entry.dialect].name == b"stablehlo"
            })
            .map(|entry| stablehlo::bytecode::decode_varint(entry.data).unwrap().0)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            codes,
            (1..=22).collect(),
            "v{version}: must test all native encodings, not textual fallbacks"
        );
        let actual = read_module(&bytes).unwrap_or_else(|error| panic!("v{version}: {error}"));
        assert_eq!(actual.attributes(), module.attributes());
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn native_builtin_metadata_affine_locations_and_future_types_import() {
    let source = r#"module attributes {
            test.map = affine_map<(d0, d1)[s0] -> (d1 floordiv 4, d0 + s0, d1 mod 2)>,
            test.set = affine_set<(d0)[s0] : (d0 - s0 >= 0, d0 mod 2 == 0)>,
            test.sparse = sparse<[[1, 0]], [3.0]> : tensor<2x2xf32>,
            test.bool_array = array<i1: true, false, true>,
            test.dense = dense<[true, false, true, false, true]> : tensor<5xi1>,
            test.symbol = @root::@child,
            test.future = !stablehlo.future<tensor<f32>, tensor<2xi32>>,
            test.location = loc(fused<"metadata">["file.rs":2:3, "label"]),
            test.range = loc("file.rs":2:3 to 4:5)
        } {}"#;
    let requests = (0..=6)
        .map(|version| (version, source.into()))
        .collect::<Vec<_>>();
    let bytes = upstream_support::emit_native_bytecode(&requests)
        .into_iter()
        .map(decode)
        .collect::<Vec<_>>();
    let expected = upstream_support::verify_native_bytecode(&bytes);
    for (version, (bytes, expected)) in bytes.iter().zip(expected).enumerate() {
        let actual = read_module(bytes).unwrap_or_else(|error| panic!("v{version}: {error}"));
        let upstream_support::UpstreamResult::Accepted(source) = expected else {
            panic!("reference must accept own bytecode");
        };
        structure::assert_equivalent(&parse_module(&source).unwrap(), &actual);
    }
}
