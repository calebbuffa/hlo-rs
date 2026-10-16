use stablehlo::{Attributes, ElementType, QuantizedTensorType, StableHloOp, Type};

mod common;
use common::computation;

fn quantized(axis: Option<usize>, scale: f64, zero: i128) -> Type {
    let count = match axis {
        Some(0) => 2,
        Some(1) => 3,
        _ => 1,
    };
    Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::I8,
        None,
        None,
        ElementType::F32,
        axis,
        vec![scale; count],
        vec![zero; count],
    ))
}

#[test]
fn quantized_elementwise_arithmetic_allows_distinct_scales_and_zero_points() {
    assert!(
        computation(
            StableHloOp::Add,
            vec![quantized(None, 0.5, 0), quantized(None, 0.25, 1)],
            vec![quantized(None, 0.75, -1)],
            Attributes::new(),
        )
        .is_ok()
    );
    for opcode in [
        StableHloOp::Negate,
        StableHloOp::Exponential,
        StableHloOp::Ceil,
    ] {
        assert!(
            computation(
                opcode,
                vec![quantized(None, 0.5, 0)],
                vec![quantized(None, 0.25, 1)],
                Attributes::new()
            )
            .is_ok()
        );
    }
}

#[test]
fn quantized_add_supports_per_tensor_and_per_axis_inputs() {
    for (lhs, rhs, result) in [
        (None, Some(1), Some(1)),
        (Some(1), None, Some(1)),
        (Some(0), Some(0), Some(0)),
    ] {
        assert!(
            computation(
                StableHloOp::Add,
                vec![quantized(lhs, 0.5, 0), quantized(rhs, 0.25, 1)],
                vec![quantized(result, 0.75, -1)],
                Attributes::new()
            )
            .is_ok()
        );
    }
    for (lhs, rhs, result) in [
        (Some(0), Some(1), Some(1)),
        (Some(1), None, None),
        (None, None, Some(1)),
        (None, Some(0), Some(1)),
    ] {
        assert!(
            computation(
                StableHloOp::Add,
                vec![quantized(lhs, 0.5, 0), quantized(rhs, 0.25, 1)],
                vec![quantized(result, 0.75, -1)],
                Attributes::new()
            )
            .is_err()
        );
    }
}

#[test]
fn explicit_default_storage_bounds_match_implicit_bounds() {
    let input = quantized(None, 0.5, 0);
    let result = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::I8,
        Some(-128),
        Some(127),
        ElementType::F32,
        None,
        vec![0.5],
        vec![0],
    ));
    assert!(
        computation(
            StableHloOp::Negate,
            vec![input],
            vec![result],
            Attributes::new()
        )
        .is_ok()
    );
}
