use stablehlo::text::parse_type;
use stablehlo::{
    Attributes, StableHloOp, Type, infer_uniform_dequantize_type, infer_uniform_quantize_type,
};

mod common;
use common::computation;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn operation(opcode: StableHloOp, operand: &str, result: &str) -> bool {
    computation(
        opcode,
        vec![ty(operand)],
        vec![ty(result)],
        Attributes::new(),
    )
    .is_ok()
}

#[test]
fn dequantization_uses_expressed_elements_and_drops_input_encoding() {
    for (input, output) in [
        ("tensor<2x!quant.uniform<i8:f32, 0.5:1>>", "tensor<2xf32>"),
        (
            "tensor<2x!quant.uniform<i8:f64:0, {0.25, 0.5}>>",
            "tensor<2xf64>",
        ),
        (
            "tensor<?x!quant.uniform<i8:bf16, 0.25>, #stablehlo.bounds<5>>",
            "tensor<?xbf16>",
        ),
        (
            "tensor<2x?x!quant.uniform<i8:f32:0, {0.25, 0.5}>, #stablehlo.bounds<?, 8>>",
            "tensor<2x?xf32>",
        ),
    ] {
        assert_eq!(
            infer_uniform_dequantize_type(&ty(input)).unwrap(),
            ty(output)
        );
        assert!(operation(StableHloOp::UniformDequantize, input, output));
    }
}

#[test]
fn quantization_uses_operand_shape_and_template_elements_not_template_encoding() {
    let input = ty("tensor<?x2xf32, #stablehlo.bounds<8, ?>>");
    let template = ty("tensor<4x2x!quant.uniform<i8:f32:1, {0.25:1, 0.5:2}>>");
    assert_eq!(
        infer_uniform_quantize_type(&input, &template).unwrap(),
        ty("tensor<?x2x!quant.uniform<i8:f32:1, {0.25:1, 0.5:2}>>")
    );
    let template = ty("tensor<?x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<7>>");
    assert_eq!(
        infer_uniform_quantize_type(&ty("tensor<2xf32>"), &template).unwrap(),
        ty("tensor<2x!quant.uniform<i8:f32, 0.5>>")
    );
}

#[test]
fn requantization_can_change_storage_parameters_and_axis() {
    let source = ty("tensor<2x3x!quant.uniform<i8:f32:0, {0.25:1, 0.5:2}>>");
    for output in [
        "tensor<2x3x!quant.uniform<ui8:f32, 0.5:15>>",
        "tensor<2x3x!quant.uniform<i4:f32:1, {0.25, 0.5, 1.0}>>",
        "tensor<2x3x!quant.uniform<i16:f32, 0.5:100>>",
    ] {
        let result = ty(output);
        assert_eq!(
            infer_uniform_quantize_type(&source, &result).unwrap(),
            result
        );
        assert!(
            computation(
                StableHloOp::UniformQuantize,
                vec![source.clone()],
                vec![result],
                Attributes::new()
            )
            .is_ok()
        );
    }
}

#[test]
fn inferred_quantization_checks_per_axis_metadata_against_new_shape() {
    let template = ty("tensor<2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>");
    assert!(infer_uniform_quantize_type(&ty("tensor<3xf32>"), &template).is_err());
    assert!(infer_uniform_quantize_type(&ty("tensor<f32>"), &template).is_err());
    assert!(infer_uniform_quantize_type(&ty("tensor<?xf32>"), &template).is_ok());
}

#[test]
fn quantization_helpers_enforce_domains_and_expressed_types() {
    let quantized = ty("tensor<2x!quant.uniform<i8:f32, 0.5>>");
    for source in [
        "tensor<2xi32>",
        "tensor<2xcomplex<f32>>",
        "tensor<*xf32>",
        "!stablehlo.token",
        "tensor<2xtf32>",
    ] {
        assert!(
            infer_uniform_quantize_type(&ty(source), &quantized).is_err(),
            "{source}"
        );
        assert!(
            infer_uniform_dequantize_type(&ty(source)).is_err(),
            "{source}"
        );
    }
    assert!(infer_uniform_dequantize_type(&ty("tensor<2xf32>")).is_err());
    for output in [
        "tensor<2xf32>",
        "tensor<2x!quant.uniform<i8:f64, 0.5>>",
        "tensor<2x!quant.uniform<i64:f32, 0.5>>",
    ] {
        assert!(
            infer_uniform_quantize_type(&ty("tensor<2xf32>"), &ty(output)).is_err(),
            "{output}"
        );
    }
}

#[test]
fn quantization_result_shapes_respect_input_and_declared_bounds_in_both_directions() {
    for opcode in [StableHloOp::UniformQuantize, StableHloOp::UniformDequantize] {
        let quantized = "!quant.uniform<i8:f32, 0.5>";
        let (input_element, result_element) = if opcode == StableHloOp::UniformQuantize {
            ("f32", quantized)
        } else {
            (quantized, "f32")
        };
        for (input_shape, output_shape, input_bounds, output_bounds, accepted) in [
            ("?", "3", ", #stablehlo.bounds<4>", "", true),
            ("?", "5", ", #stablehlo.bounds<4>", "", false),
            ("3", "?", "", ", #stablehlo.bounds<4>", true),
            ("5", "?", "", ", #stablehlo.bounds<4>", false),
            (
                "?",
                "?",
                ", #stablehlo.bounds<4>",
                ", #stablehlo.bounds<3>",
                true,
            ),
            ("3", "4", "", "", false),
        ] {
            let input = format!("tensor<{input_shape}x{input_element}{input_bounds}>");
            let output = format!("tensor<{output_shape}x{result_element}{output_bounds}>");
            assert_eq!(
                operation(opcode, &input, &output),
                accepted,
                "{opcode:?}: {input} -> {output}"
            );
        }
    }
}
