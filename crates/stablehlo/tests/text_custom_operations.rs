use stablehlo::{
    Attribute, Module,
    text::{TextErrorKind, parse_module, print_module},
};

fn module(arguments: &str, outputs: &str, body: &str) -> String {
    format!("module {{ func.func @main({arguments}) -> {outputs} {{ {body} }} }}")
}

fn parse_and_round_trip(input: &str) -> Module {
    let parsed = parse_module(input).unwrap_or_else(|error| panic!("{input}\n{error}"));
    let printed = print_module(&parsed).unwrap();
    let reparsed = parse_module(&printed).unwrap();
    assert_eq!(print_module(&reparsed).unwrap(), printed);
    assert_eq!(
        reparsed.computations()[0].function_type(),
        parsed.computations()[0].function_type()
    );
    parsed
}

#[test]
fn every_inherited_unary_elementwise_format_accepts_compact_and_function_types() {
    for (name, element) in [
        ("abs", "f32"),
        ("cbrt", "f32"),
        ("ceil", "f32"),
        ("convert", "f32"),
        ("count_leading_zeros", "i32"),
        ("cosine", "f32"),
        ("exponential", "f32"),
        ("exponential_minus_one", "f32"),
        ("floor", "f32"),
        ("imag", "f32"),
        ("log", "f32"),
        ("log_plus_one", "f32"),
        ("logistic", "f32"),
        ("not", "i1"),
        ("negate", "f32"),
        ("popcnt", "i32"),
        ("real", "f32"),
        ("round_nearest_afz", "f32"),
        ("round_nearest_even", "f32"),
        ("rsqrt", "f32"),
        ("sign", "f32"),
        ("sine", "f32"),
        ("sqrt", "f32"),
        ("tan", "f32"),
        ("tanh", "f32"),
    ] {
        let ty = format!("tensor<2x{element}>");
        for signature in [ty.clone(), format!("({ty}) -> {ty}")] {
            parse_and_round_trip(&module(
                &format!("%x: {ty}"),
                &ty,
                &format!("%r = stablehlo.{name} %x : {signature} return %r : {ty}"),
            ));
        }
    }
}

#[test]
fn every_inherited_binary_elementwise_format_accepts_both_signatures() {
    for (name, element) in [
        ("add", "f32"),
        ("atan2", "f32"),
        ("divide", "f32"),
        ("maximum", "f32"),
        ("minimum", "f32"),
        ("multiply", "f32"),
        ("power", "f32"),
        ("remainder", "f32"),
        ("shift_left", "i32"),
        ("shift_right_arithmetic", "i32"),
        ("shift_right_logical", "i32"),
        ("subtract", "f32"),
        ("and", "i1"),
        ("or", "i1"),
        ("xor", "i1"),
    ] {
        let ty = format!("tensor<2x{element}>");
        for signature in [ty.clone(), format!("({ty}, {ty}) -> {ty}")] {
            parse_and_round_trip(&module(
                &format!("%x: {ty}, %y: {ty}"),
                &ty,
                &format!("%r = stablehlo.{name} %x, %y : {signature} return %r : {ty}"),
            ));
        }
    }
}

#[test]
fn complex_and_select_helpers_match_upstream_compact_type_inference() {
    for (arguments, output, operation) in [
        (
            "%x: tensor<2xcomplex<f32>>",
            "tensor<2xf32>",
            "abs %x : (tensor<2xcomplex<f32>>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<2xcomplex<f32>>",
            "tensor<2xf32>",
            "imag %x : (tensor<2xcomplex<f32>>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<2xcomplex<f64>>",
            "tensor<2xf64>",
            "real %x : (tensor<2xcomplex<f64>>) -> tensor<2xf64>",
        ),
        (
            "%x: tensor<2xi32>",
            "tensor<2xf32>",
            "convert %x : (tensor<2xi32>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<2xf32>",
            "tensor<2xi1>",
            "is_finite %x : (tensor<2xf32>) -> tensor<2xi1>",
        ),
        (
            "%x: tensor<2xf32>, %y: tensor<2xf32>",
            "tensor<2xcomplex<f32>>",
            "complex %x, %y : tensor<2xcomplex<f32>>",
        ),
        (
            "%x: tensor<2xf64>, %y: tensor<2xf64>",
            "tensor<2xcomplex<f64>>",
            "complex %x, %y : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xcomplex<f64>>",
        ),
        (
            "%p: tensor<i1>, %x: tensor<2xf32>, %y: tensor<2xf32>",
            "tensor<2xf32>",
            "select %p, %x, %y : tensor<i1>, tensor<2xf32>",
        ),
        (
            "%p: tensor<2xi1>, %x: tensor<2xf32>, %y: tensor<2xf32>",
            "tensor<2xf32>",
            "select %p, %x, %y : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<f32>, %y: tensor<2xf32>, %z: tensor<f32>",
            "tensor<2xf32>",
            "clamp %x, %y, %z : (tensor<f32>, tensor<2xf32>, tensor<f32>) -> tensor<2xf32>",
        ),
    ] {
        parse_and_round_trip(&module(
            arguments,
            output,
            &format!("%r = stablehlo.{operation} return %r : {output}"),
        ));
    }
}

#[test]
fn constants_support_elements_metadata_and_the_unquoted_generic_fallback() {
    for operation in [
        "constant dense<[1, 2]> : tensor<2xi32>",
        "constant {consumer.tag = \"kept\"} dense<[1, 2]> : tensor<2xi32>",
        "constant sparse<[[1]], [2]> : tensor<2xi32>",
        "constant() <{value = dense<[1, 2]> : tensor<2xi32>}> : () -> tensor<2xi32>",
        "constant() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>",
    ] {
        parse_and_round_trip(&module(
            "",
            "tensor<2xi32>",
            &format!("%r = stablehlo.{operation} return %r : tensor<2xi32>"),
        ));
    }
}

#[test]
fn zero_operand_operations_tuple_construction_and_token_merging_round_trip() {
    for (output, operation) in [
        ("!stablehlo.token", "create_token : !stablehlo.token"),
        ("!stablehlo.token", "after_all : !stablehlo.token"),
        ("tensor<ui32>", "replica_id : tensor<ui32>"),
        ("tensor<ui32>", "partition_id : tensor<ui32>"),
        ("tuple<>", "tuple : tuple<>"),
        ("tensor<2xi32>", "iota dim = 0 : tensor<2xi32>"),
    ] {
        parse_and_round_trip(&module(
            "",
            output,
            &format!("%r = stablehlo.{operation} return %r : {output}"),
        ));
    }
    parse_and_round_trip(&module(
        "%x: tensor<f32>, %t: !stablehlo.token",
        "tensor<f32>",
        "%tuple = stablehlo.tuple %x, %t : tuple<tensor<f32>, !stablehlo.token>
         %r = stablehlo.get_tuple_element %tuple[0] : (tuple<tensor<f32>, !stablehlo.token>) -> tensor<f32>
         return %r : tensor<f32>",
    ));
    parse_and_round_trip(&module(
        "%x: !stablehlo.token, %y: !stablehlo.token",
        "!stablehlo.token",
        "%r = stablehlo.after_all %x, %y : !stablehlo.token return %r : !stablehlo.token",
    ));
}

#[test]
fn shape_and_dynamic_index_formats_map_fields_to_canonical_attributes() {
    for (arguments, output, operation) in [
        (
            "%x: tensor<2xf32>",
            "tensor<1x2xf32>",
            "reshape %x : (tensor<2xf32>) -> tensor<1x2xf32>",
        ),
        (
            "%x: tensor<2xf32>, %s: tensor<2xi64>",
            "tensor<1x2xf32>",
            "dynamic_reshape %x, %s : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>",
        ),
        (
            "%x: tensor<2xi32>",
            "tensor<2xf32>",
            "bitcast_convert %x : (tensor<2xi32>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<2xf32>",
            "tensor<3x2xf32>",
            "broadcast %x, sizes = [3] : (tensor<2xf32>) -> tensor<3x2xf32>",
        ),
        (
            "%x: tensor<2xf32>",
            "tensor<3x2xf32>",
            "broadcast_in_dim %x, dims = [1] : (tensor<2xf32>) -> tensor<3x2xf32>",
        ),
        (
            "%x: tensor<2xf32>, %s: tensor<2xi64>",
            "tensor<3x2xf32>",
            "dynamic_broadcast_in_dim %x, %s, dims = [1] : (tensor<2xf32>, tensor<2xi64>) -> tensor<3x2xf32>",
        ),
        (
            "%x: tensor<2x3xf32>",
            "tensor<3x2xf32>",
            "transpose %x, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>",
        ),
        (
            "%x: tensor<2x3xf32>",
            "tensor<2x3xf32>",
            "reverse %x, dims = array<i64: 1> : tensor<2x3xf32>",
        ),
        (
            "%x: tensor<2x3xf32>",
            "tensor<2x3xf32>",
            "reverse %x, dims = [] : tensor<2x3xf32>",
        ),
        (
            "%x: tensor<2xf32>, %y: tensor<2xf32>",
            "tensor<4xf32>",
            "concatenate %x, %y, dim = 0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<4xf32>",
        ),
        (
            "%s: tensor<1xi64>",
            "tensor<2xi32>",
            "dynamic_iota %s, dim = 0 : (tensor<1xi64>) -> tensor<2xi32>",
        ),
        (
            "%x: tensor<2x3xf32>",
            "tensor<i32>",
            "get_dimension_size %x, dim = 1 : (tensor<2x3xf32>) -> tensor<i32>",
        ),
        (
            "%x: tensor<2xf32>, %s: tensor<i32>",
            "tensor<?xf32>",
            "set_dimension_size %x, %s, dim = 0 : (tensor<2xf32>, tensor<i32>) -> tensor<?xf32>",
        ),
        (
            "%x: tensor<4xf32>, %s: tensor<i32>",
            "tensor<2xf32>",
            "dynamic_slice %x, %s, sizes = [2] : (tensor<4xf32>, tensor<i32>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<4xf32>, %u: tensor<2xf32>, %s: tensor<i32>",
            "tensor<4xf32>",
            "dynamic_update_slice %x, %u, %s : (tensor<4xf32>, tensor<2xf32>, tensor<i32>) -> tensor<4xf32>",
        ),
        (
            "%x: tensor<4xf32>, %s: tensor<1xi64>, %e: tensor<1xi64>, %d: tensor<1xi64>",
            "tensor<2xf32>",
            "real_dynamic_slice %x, %s, %e, %d : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xf32>",
        ),
        (
            "%x: tensor<2xf32>, %v: tensor<f32>",
            "tensor<5xf32>",
            "pad %x, %v, low = [1], high = [2], interior = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<5xf32>",
        ),
        (
            "%x: tensor<2xf32>, %v: tensor<f32>, %l: tensor<1xi64>, %h: tensor<1xi64>, %i: tensor<1xi64>",
            "tensor<5xf32>",
            "dynamic_pad %x, %v, %l, %h, %i : (tensor<2xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xf32>",
        ),
    ] {
        parse_and_round_trip(&module(
            arguments,
            output,
            &format!("%r = stablehlo.{operation} return %r : {output}"),
        ));
    }
}

#[test]
fn enum_string_boolean_and_precision_fields_preserve_their_actual_attribute_types() {
    for direction in ["EQ", "#stablehlo<comparison_direction EQ>"] {
        let parsed = parse_and_round_trip(&module(
            "%x: tensor<2xf32>, %y: tensor<2xf32>",
            "tensor<2xi1>",
            &format!(
                "%r = stablehlo.compare {direction}, %x, %y, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1> return %r : tensor<2xi1>"
            ),
        ));
        assert!(
            matches!(&parsed.computations()[0].instructions()[2].operation().attributes().unwrap()["compare_type"], Attribute::Enum { name, value, .. } if name == "comparison_type" && value == "FLOAT")
        );
    }
    for (arguments, output, operation) in [
        (
            "%x: tensor<2x2xf32>",
            "tensor<2x2xf32>",
            "cholesky %x : tensor<2x2xf32>",
        ),
        (
            "%x: tensor<2x2xf32>",
            "tensor<2x2xf32>",
            "cholesky %x, lower = true : tensor<2x2xf32>",
        ),
        (
            "%x: tensor<2x2xf32>",
            "tensor<2x2xf32>",
            "cholesky %x, lower = false : tensor<2x2xf32>",
        ),
        (
            "%x: tensor<2xf32>",
            "tensor<2xf32>",
            "reduce_precision %x, format = e8m23 : tensor<2xf32>",
        ),
        (
            "%x: tensor<2x2xf32>, %y: tensor<2x2xf32>",
            "tensor<2x2xf32>",
            "einsum %x, %y, config = \"ij,jk->ik\" : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>",
        ),
        (
            "%x: tensor<2x3xf32>",
            "tensor<3x2xf32>",
            "unary_einsum %x, config = \"ij->ji\" : (tensor<2x3xf32>) -> tensor<3x2xf32>",
        ),
        (
            "%x: tensor<2xcomplex<f32>>",
            "tensor<2xcomplex<f32>>",
            "fft %x, type = FFT, length = [2] : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>",
        ),
        (
            "%a: tensor<f32>, %b: tensor<f32>, %s: tensor<1xi64>",
            "tensor<2xf32>",
            "rng %a, %b, %s, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<2xf32>",
        ),
    ] {
        parse_and_round_trip(&module(
            arguments,
            output,
            &format!("%r = stablehlo.{operation} return %r : {output}"),
        ));
    }
    parse_and_round_trip(&module(
        "%s: tensor<2xui64>",
        "(tensor<2xui64>, tensor<2xui32>)",
        "%r:2 = stablehlo.rng_bit_generator %s, algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui32>)
         return %r#0, %r#1 : tensor<2xui64>, tensor<2xui32>",
    ));
}

#[test]
fn slice_ranges_barriers_and_custom_calls_round_trip() {
    for (arguments, output, operation) in [
        (
            "%x: tensor<4x6xf32>",
            "tensor<2x3xf32>",
            "slice %x[1:3, 0:6:2] : (tensor<4x6xf32>) -> tensor<2x3xf32>",
        ),
        (
            "%x: tensor<f32>",
            "tensor<f32>",
            "slice %x[] : (tensor<f32>) -> tensor<f32>",
        ),
        (
            "%x: tensor<4xf32>",
            "tensor<0xf32>",
            "slice %x[2:2] : (tensor<4xf32>) -> tensor<0xf32>",
        ),
        (
            "%x: tensor<2xf32>",
            "tensor<2xf32>",
            "custom_call @kernel(%x) : (tensor<2xf32>) -> tensor<2xf32>",
        ),
        (
            "",
            "tensor<f32>",
            "custom_call @\"kernel with spaces\"() : () -> tensor<f32>",
        ),
    ] {
        parse_and_round_trip(&module(
            arguments,
            output,
            &format!("%r = stablehlo.{operation} return %r : {output}"),
        ));
    }
    parse_and_round_trip(&module(
        "%x: tensor<2xf32>, %t: !stablehlo.token",
        "(tensor<2xf32>, !stablehlo.token)",
        "%r:2 = stablehlo.optimization_barrier {consumer.barrier = true} %x, %t : tensor<2xf32>, !stablehlo.token
         return %r#0, %r#1 : tensor<2xf32>, !stablehlo.token",
    ));
    parse_and_round_trip(&module(
        "",
        "()",
        "stablehlo.optimization_barrier () return",
    ));
}

#[test]
fn dot_general_dimensions_precision_and_algorithms_follow_pinned_helpers() {
    for operation in [
        "dot %x, %y",
        "dot %x, %y, precision = [HIGH, HIGHEST]",
        "dot_general %x, %y, contracting_dims = [1] x [0]",
        "dot_general %x, %y, batching_dims = [] x [], contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]",
        "dot_general %x, %y, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
        "dot_general %x, %y, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = f32, rhs_precision_type = f32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false>",
    ] {
        parse_and_round_trip(&module(
            "%x: tensor<2x3xf32>, %y: tensor<3x4xf32>",
            "tensor<2x4xf32>",
            &format!(
                "%r = stablehlo.{operation} : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32> return %r : tensor<2x4xf32>"
            ),
        ));
    }
    parse_and_round_trip(&module(
        "%x: tensor<5x2x3xf32>, %y: tensor<5x3x4xf32>",
        "tensor<5x2x4xf32>",
        "%r = stablehlo.dot_general %x, %y, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x2x3xf32>, tensor<5x3x4xf32>) -> tensor<5x2x4xf32>
         return %r : tensor<5x2x4xf32>",
    ));
}

#[test]
fn quantization_and_bounded_complex_types_use_the_same_custom_signature_helpers() {
    let quantized = "tensor<2x!quant.uniform<i8:f32, 0.5>>";
    parse_and_round_trip(&module(
        "%x: tensor<2xf32>",
        quantized,
        &format!(
            "%r = stablehlo.uniform_quantize %x : (tensor<2xf32>) -> {quantized} return %r : {quantized}"
        ),
    ));
    parse_and_round_trip(&module(
        &format!("%x: {quantized}"),
        "tensor<2xf32>",
        &format!(
            "%r = stablehlo.uniform_dequantize %x : ({quantized}) -> tensor<2xf32> return %r : tensor<2xf32>"
        ),
    ));
    parse_and_round_trip(&module(
        "%x: tensor<?xf32, #stablehlo.bounds<8>>, %y: tensor<?xf32, #stablehlo.bounds<8>>",
        "tensor<?xcomplex<f32>, #stablehlo.bounds<8>>",
        "%r = stablehlo.complex %x, %y : tensor<?xcomplex<f32>, #stablehlo.bounds<8>>
         return %r : tensor<?xcomplex<f32>, #stablehlo.bounds<8>>",
    ));
}

#[test]
fn operations_without_upstream_custom_formats_require_generic_assembly() {
    for name in [
        "infeed",
        "outfeed",
        "send",
        "recv",
        "if",
        "case",
        "all_gather",
        "all_reduce",
        "reduce_scatter",
        "all_to_all",
        "async_start",
        "async_done",
        "batch_norm_grad",
        "batch_norm_inference",
        "batch_norm_training",
        "collective_broadcast",
        "collective_reduce",
        "collective_permute",
        "gather",
        "map",
        "scatter",
        "select_and_scatter",
        "sort",
        "triangular_solve",
        "reduce_window",
        "torch_index_select",
        "dynamic_gather",
        "dynamic_conv",
    ] {
        let input = module("", "()", &format!("stablehlo.{name}() : () -> () return"));
        assert!(
            matches!(
                parse_module(&input).unwrap_err().kind,
                TextErrorKind::Expected("quoted generic operation assembly")
            ),
            "{input}"
        );
    }
}

#[test]
fn custom_formats_reject_inconsistent_types_fields_counts_and_duplicate_attributes() {
    for operation in [
        "add %x : tensor<2xf32>",
        "add %x, %x : (tensor<2xf32>) -> tensor<2xf32>",
        "add %x, %x : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)",
        "add %x, %x : tensor<3xf32>",
        "add %x, %x, : tensor<2xf32>",
        "is_finite %x : tensor<2xf32>",
        "complex %x, %x : tensor<2xf32>",
        "tuple %x : tensor<2xf32>",
        "reverse %x, dims = [0] {dimensions = array<i64: 0>} : tensor<2xf32>",
        "reverse %x, dims = array<i32: 0> : tensor<2xf32>",
        "reverse %x, dims = [0,] : tensor<2xf32>",
        "reduce_precision %x, format = e8m : tensor<2xf32>",
        "reduce_precision %x, format = e8m2147483648 : tensor<2xf32>",
        "compare INVALID, %x, %x : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>",
        "compare #stablehlo<precision HIGH>, %x, %x : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>",
        "constant {value = dense<1.0> : tensor<2xf32>} dense<1.0> : tensor<2xf32>",
        "constant 1.0 : f32",
        "concatenate %x dim = 0 : (tensor<2xf32>) -> tensor<2xf32>",
        "get_tuple_element %x[2147483648] : (tensor<2xf32>) -> tensor<2xf32>",
        "slice %x[0:2:0] : (tensor<2xf32>) -> tensor<2xf32>",
        "slice %x[0:2] {strides = array<i64: 1>} : (tensor<2xf32>) -> tensor<2xf32>",
        "dot %x, %x, precision = [HIGH] : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>",
        "dot_general %x, %x, contracting_dims = [] x [] {dot_dimension_numbers = #stablehlo.dot<>} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>",
        "optimization_barrier %x : tensor<2xf32>, tensor<2xf32>",
    ] {
        let input = module(
            "%x: tensor<2xf32>",
            "tensor<2xf32>",
            &format!("%r = stablehlo.{operation} return %r : tensor<2xf32>"),
        );
        assert!(parse_module(&input).is_err(), "{input}");
    }
    let input = module(
        "",
        "tensor<f32>",
        "%r = stablehlo.unknown : tensor<f32> return %r : tensor<f32>",
    );
    assert!(matches!(
        parse_module(&input).unwrap_err().kind,
        TextErrorKind::Expected("known StableHLO operation")
    ));
}

#[test]
fn custom_assembly_is_accepted_inside_generic_regions() {
    let input = module(
        "%x: tensor<f32>, %y: tensor<f32>, %p: tensor<i1>",
        "tensor<f32>",
        r#"%r = "stablehlo.if"(%p) ({
              %sum = stablehlo.add %x, %y : tensor<f32>
              stablehlo.return %sum : tensor<f32>
            }, {
              %difference = stablehlo.subtract %x, %y : tensor<f32>
              stablehlo.return %difference : tensor<f32>
            }) : (tensor<i1>) -> tensor<f32>
            return %r : tensor<f32>"#,
    );
    parse_and_round_trip(&input);
}
