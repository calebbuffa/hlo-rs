use stablehlo::text::parse_type;
use stablehlo::{Attributes, StableHloOp, Type};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn accepts(input: &str, output: &str) -> bool {
    common::computation(
        StableHloOp::BitcastConvert,
        vec![ty(input)],
        vec![ty(output)],
        Attributes::new(),
    )
    .is_ok()
}

#[test]
fn bitcast_checks_equal_width_shape_bounds_in_both_directions() {
    for (input, output, accepted) in [
        (
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<4xi32>",
            false,
        ),
        ("tensor<?xf32, #stablehlo.bounds<4>>", "tensor<4xi32>", true),
        (
            "tensor<4xf32>",
            "tensor<?xi32, #stablehlo.bounds<3>>",
            false,
        ),
        ("tensor<4xf32>", "tensor<?xi32, #stablehlo.bounds<4>>", true),
        (
            "tensor<?xf32, #stablehlo.bounds<0>>",
            "tensor<?xi32, #stablehlo.bounds<4>>",
            true,
        ),
        ("tensor<f32>", "tensor<i32>", true),
        ("tensor<0xf32>", "tensor<0xi32>", true),
        ("tensor<1xf32>", "tensor<1x1xi32>", false),
    ] {
        assert_eq!(accepts(input, output), accepted, "{input}, {output}");
    }
}

#[test]
fn bitcast_checks_dynamic_minor_dimension_against_exact_width_ratio() {
    for (small, accepted) in [
        ("tensor<2x?xi8, #stablehlo.bounds<?, 3>>", false),
        ("tensor<2x?xi8, #stablehlo.bounds<?, 4>>", true),
        ("tensor<2x?xi8>", true),
        ("tensor<2x0xi8>", false),
        ("tensor<2x3xi8>", false),
        ("tensor<2x4xi8>", true),
        ("tensor<2x5xi8>", false),
    ] {
        assert_eq!(accepts(small, "tensor<2xf32>"), accepted, "{small}");
        assert_eq!(accepts("tensor<2xf32>", small), accepted, "{small}");
    }
}

#[test]
fn bitcast_rejects_nonintegral_width_ratios_even_for_unknown_minor_extents() {
    for (input, output) in [
        ("tensor<2xf6E2M3FN>", "tensor<2x?xi4>"),
        ("tensor<2x?xi4>", "tensor<2xf6E2M3FN>"),
        ("tensor<2xf6E3M2FN>", "tensor<2x?xf4E2M1FN>"),
        ("tensor<2x?xf6E2M3FN>", "tensor<2xi8>"),
        ("tensor<2xf32>", "tensor<2x?xf6E3M2FN>"),
    ] {
        assert!(!accepts(input, output), "{input}, {output}");
    }
}

#[test]
fn bitcast_checks_prefix_bounds_independently_of_minor_width_dimension() {
    for (small, big, accepted) in [
        (
            "tensor<?x4xi8, #stablehlo.bounds<3, ?>>",
            "tensor<4xf32>",
            false,
        ),
        (
            "tensor<?x4xi8, #stablehlo.bounds<4, ?>>",
            "tensor<4xf32>",
            true,
        ),
        (
            "tensor<4x4xi8>",
            "tensor<?xf32, #stablehlo.bounds<3>>",
            false,
        ),
        (
            "tensor<4x4xi8>",
            "tensor<?xf32, #stablehlo.bounds<4>>",
            true,
        ),
        (
            "tensor<?x?xi8, #stablehlo.bounds<3, 3>>",
            "tensor<?xf32>",
            false,
        ),
        ("tensor<0x4xi8>", "tensor<0xf32>", true),
    ] {
        assert_eq!(accepts(small, big), accepted, "{small}, {big}");
        assert_eq!(accepts(big, small), accepted, "{big}, {small}");
    }
}

#[test]
fn bitcast_uses_quantized_storage_widths_without_constraining_scale_or_zero_points() {
    for (input, output) in [
        (
            "tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>",
            "tensor<2xf32>",
        ),
        (
            "tensor<2xf32>",
            "tensor<2x4x!quant.uniform<i8:f32, 0.25:1>>",
        ),
        (
            "tensor<2x!quant.uniform<i8:f32, 0.25:1>>",
            "tensor<2x!quant.uniform<ui8:f64, 0.5:2>>",
        ),
        (
            "tensor<2x2x!quant.uniform<i8:f32:1, {0.25,0.5}>>",
            "tensor<2xi16>",
        ),
    ] {
        assert!(accepts(input, output), "{input}, {output}");
    }
    assert!(!accepts(
        "tensor<2x?x!quant.uniform<i8:f32, 0.25:1>, #stablehlo.bounds<?, 3>>",
        "tensor<2xf32>"
    ));
}

#[test]
fn bitcast_requires_both_elements_to_be_complex_and_checks_complex_total_widths() {
    for (input, output, accepted) in [
        ("tensor<2xcomplex<f64>>", "tensor<2x2xcomplex<f32>>", true),
        ("tensor<2x2xcomplex<f32>>", "tensor<2xcomplex<f64>>", true),
        (
            "tensor<2xcomplex<f64>>",
            "tensor<2x?xcomplex<f32>, #stablehlo.bounds<?, 1>>",
            false,
        ),
        ("tensor<2xcomplex<f32>>", "tensor<2xf64>", false),
        ("tensor<2xf64>", "tensor<2xcomplex<f32>>", false),
        ("tensor<complex<f32>>", "tensor<complex<f64>>", false),
    ] {
        assert_eq!(accepts(input, output), accepted, "{input}, {output}");
    }
}

#[test]
fn bitcast_minor_dimension_witnesses_match_exact_bit_width_equations() {
    for (small, small_width) in [
        ("i1", 1),
        ("i4", 4),
        ("i8", 8),
        ("f6E2M3FN", 6),
        ("f16", 16),
    ] {
        for (big, big_width) in [("i8", 8), ("f16", 16), ("f32", 32), ("f64", 64)] {
            if small_width >= big_width {
                continue;
            }
            for extent in 0..=8 {
                let a = format!("tensor<2x{extent}x{small}>");
                let b = format!("tensor<2x{big}>");
                let feasible = extent * small_width == big_width;
                assert_eq!(accepts(&a, &b), feasible, "{a}, {b}");
                assert_eq!(accepts(&b, &a), feasible, "{b}, {a}");
            }
            for upper in 0..=8 {
                let a = format!("tensor<2x?x{small}, #stablehlo.bounds<?, {upper}>>");
                let b = format!("tensor<2x{big}>");
                let feasible = (0..=upper).any(|extent| extent * small_width == big_width);
                assert_eq!(accepts(&a, &b), feasible, "{a}, {b}");
                assert_eq!(accepts(&b, &a), feasible, "{b}, {a}");
            }
        }
    }
}
