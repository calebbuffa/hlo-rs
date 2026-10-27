use stablehlo::mlir::assembly::{parse_attribute, print_attribute};
use stablehlo::mlir::{export, import};
use stablehlo::{
    attributes::{ConvDimensionNumbersAttr, ResultAccuracyAttr, ResultAccuracyMode},
    pliron::context::Context,
};

#[test]
fn native_dimension_and_metadata_attributes_roundtrip_in_fresh_contexts() {
    for source in [
        "#stablehlo.dot<>",
        "#stablehlo.dot<rhs_contracting_dimensions = [0], lhs_contracting_dimensions = [1]>",
        "#stablehlo<dot<lhs_batching_dimensions = [1, 0], rhs_batching_dimensions = [0, 1]>>",
        "#stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>",
        "#stablehlo.gather<operand_batching_dims = [1], start_indices_batching_dims = [0], index_vector_dim = 2>",
        "#stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>",
        "#stablehlo.scatter<input_batching_dims = [0], scatter_indices_batching_dims = [1]>",
        "#stablehlo.channel_handle<type = 1, handle = 9223372036854775807>",
        "#stablehlo.channel_handle<type = 2, handle = 7>",
        "#stablehlo.channel_handle<type = 3, handle = 7>",
        "#stablehlo.output_operand_alias<output_tuple_indices = [0, 1], operand_index = 2, operand_tuple_indices = []>",
        "#stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>",
        "#stablehlo.conv<[0, b, f]x[i, o, 0]->[f, 0, b]>",
        "#stablehlo.conv<[b, f]x[i, o]->[f, b]>",
        "#stablehlo.conv<raw input_batch_dimension = 7>",
        "#stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
        "#stablehlo.result_accuracy_mode<DEFAULT>",
        "#stablehlo<result_accuracy_mode<TOLERANCE>>",
        "#stablehlo<result_accuracy_mode HIGHEST>",
        "#stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<DEFAULT>>",
        "#stablehlo.result_accuracy<atol = 1.0e-6, rtol = 0.0, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<rtol = 0x7FF0000000000001, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo<type_extensions<bounds = [16, ?]>>",
    ] {
        let ctx = Context::new();
        let attribute =
            parse_attribute(&ctx, source).unwrap_or_else(|error| panic!("{source}: {error}"));
        let printed = print_attribute(&ctx, &*attribute).unwrap();
        let fresh = Context::new();
        let restored =
            parse_attribute(&fresh, &printed).unwrap_or_else(|error| panic!("{printed}: {error}"));
        assert_eq!(
            print_attribute(&fresh, &*restored).unwrap(),
            printed,
            "{source}"
        );
    }
}

#[test]
fn compact_convolution_markers_map_axes_without_sorting_the_spatial_order() {
    let ctx = Context::new();
    let value = parse_attribute(
        &ctx,
        "#stablehlo.conv<[1, b, f, 0]x[o, 1, 0, i]->[f, 0, b, 1]>",
    )
    .unwrap();
    let value = value.downcast_ref::<ConvDimensionNumbersAttr>().unwrap();
    assert_eq!(value.input_batch_dimension, 1);
    assert_eq!(value.input_feature_dimension, 2);
    assert_eq!(value.input_spatial_dimensions, [3, 0]);
    assert_eq!(value.kernel_input_feature_dimension, 3);
    assert_eq!(value.kernel_output_feature_dimension, 0);
    assert_eq!(value.kernel_spatial_dimensions, [2, 1]);
    assert_eq!(value.output_batch_dimension, 2);
    assert_eq!(value.output_feature_dimension, 0);
    assert_eq!(value.output_spatial_dimensions, [1, 3]);
}

#[test]
fn result_accuracy_prints_source_delimiters_and_preserves_double_payloads() {
    let ctx = Context::new();
    assert_eq!(
        print_attribute(&ctx, &ResultAccuracyMode::Tolerance).unwrap(),
        "#stablehlo.result_accuracy_mode<TOLERANCE>"
    );
    let source = "#stablehlo.result_accuracy<atol = 0x7FF0000000000001, rtol = 0x7FF0000000000000, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>";
    let parsed = parse_attribute(&ctx, source).unwrap();
    let value = parsed.downcast_ref::<ResultAccuracyAttr>().unwrap();
    assert_eq!(value.atol_bits, 0x7ff0_0000_0000_0001);
    assert_eq!(value.rtol_bits, 0x7ff0_0000_0000_0000);
    let printed = print_attribute(&ctx, value).unwrap();
    let restored = parse_attribute(&ctx, &printed).unwrap();
    assert_eq!(
        restored.downcast_ref::<ResultAccuracyAttr>().unwrap(),
        value
    );
}

#[test]
fn invalid_dialect_fields_fail_instead_of_defaulting_or_reinterpreting() {
    let ctx = Context::new();
    for source in [
        "#stablehlo.dot<lhs_batching_dimensions = [0], lhs_batching_dimensions = []>",
        "#stablehlo.gather<wrong = [0]>",
        "#stablehlo.scatter<index_vector_dim = -1>",
        "#stablehlo.channel_handle<handle = 1>",
        "#stablehlo.channel_handle<type = 1>",
        "#stablehlo.channel_handle<handle = 9223372036854775808, type = 1>",
        "#stablehlo.channel_handle<handle = 1, type = 0>",
        "#stablehlo.channel_handle<handle = 1, type = 4>",
        "#stablehlo.channel_handle<handle = 1, type = -1>",
        "#stablehlo.output_operand_alias<operand_index = 0, output_tuple_indices = [], operand_tuple_indices = []>",
        "#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = -1, operand_tuple_indices = []>",
        "#stablehlo.conv<[b, f, 1]x[i, o, 0]->[b, f, 0]>",
        "#stablehlo.conv<[b, f, 9223372036854775807]x[i, o, 0]->[b, f, 0]>",
        "#stablehlo.conv<[b, b, 0]x[i, o, 0]->[b, f, 0]>",
        "#stablehlo.conv<[b, f, 0, 0]x[i, o, 0]->[b, f, 0]>",
        "#stablehlo.conv<[b, f, -1]x[i, o, 0]->[b, f, 0]>",
        "#stablehlo.conv<raw unknown_dimension = 1>",
        "#stablehlo.result_accuracy<atol = -0.0, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<ulps = -1, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        "#stablehlo.result_accuracy<atol = 1.0, mode = #stablehlo.result_accuracy_mode<DEFAULT>>",
        "#stablehlo.result_accuracy<mode = #stablehlo<precision DEFAULT>>",
        "#stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<INVALID>>",
        "#stablehlo.dot_algorithm<lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 2, allow_imprecise_accumulation = false>",
    ] {
        assert!(parse_attribute(&ctx, source).is_err(), "{source}");
        parse_attribute(&ctx, "#stablehlo.dot<>").unwrap();
    }
}

#[test]
fn all_fourteen_result_accuracy_operations_preserve_present_and_absent_metadata() {
    for name in [
        "cbrt",
        "cosine",
        "exp2",
        "exponential",
        "exponential_minus_one",
        "log",
        "log2",
        "log_plus_one",
        "logistic",
        "rsqrt",
        "sine",
        "sqrt",
        "tan",
        "tanh",
    ] {
        for attributes in [
            "",
            "result_accuracy = #stablehlo.result_accuracy<atol = 1.0e-6, ulps = 2, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>",
        ] {
            let source = format!(
                "module {{ func.func @main(%arg: tensor<2xf32>) -> tensor<2xf32> {{ %v = \"stablehlo.{name}\"(%arg) {{{attributes}}} : (tensor<2xf32>) -> tensor<2xf32> func.return %v : tensor<2xf32> }} }}"
            );
            let mut ctx = Context::new();
            let module =
                import(&mut ctx, &source).unwrap_or_else(|error| panic!("{name}: {error}"));
            let printed = export(&ctx, &module).unwrap();
            assert_eq!(
                printed.contains("result_accuracy"),
                !attributes.is_empty(),
                "{printed}"
            );
            assert!(!printed.contains("stablehlo_result_accuracy"));
            let mut fresh = Context::new();
            let restored = import(&mut fresh, &printed).unwrap();
            assert_eq!(export(&fresh, &restored).unwrap(), printed);
        }
    }
}
