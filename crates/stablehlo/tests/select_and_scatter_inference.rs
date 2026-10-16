use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, Instruction, Operation, Region, StableHloOp, Type,
    ValueId, infer_select_and_scatter_type,
};

mod common;

fn ty(text: &str) -> Type {
    parse_type(text).unwrap()
}

fn verify(
    input: Type,
    source: Type,
    result: Type,
    window: i64,
    stride: i64,
    pad: [i64; 2],
) -> bool {
    let scalar = ty("tensor<f32>");
    let select = Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(100), scalar.clone()),
            BlockArgument::new(ValueId(101), scalar.clone()),
        ],
        vec![
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Compare,
                    [(
                        "comparison_direction".into(),
                        Attribute::Enum {
                            dialect: "stablehlo".into(),
                            name: "comparison_direction".into(),
                            value: "GE".into(),
                        },
                    )]
                    .into(),
                ),
                vec![ValueId(100), ValueId(101)],
                vec![],
                ValueId(102),
                ty("tensor<i1>"),
            ),
            Instruction::with_results(
                Operation::stablehlo(StableHloOp::Return, Attributes::new()),
                vec![ValueId(102)],
                vec![],
                vec![],
            ),
        ],
    )]);
    let scatter = Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(200), scalar.clone()),
            BlockArgument::new(ValueId(201), scalar.clone()),
        ],
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            vec![ValueId(200)],
            vec![],
            vec![],
        )],
    )]);
    let attributes = [
        (
            "window_dimensions".into(),
            Attribute::integers(vec![window]),
        ),
        ("window_strides".into(), Attribute::integers(vec![stride])),
        (
            "padding".into(),
            Attribute::DenseElements {
                ty: Box::new(ty("tensor<1x2xi64>")),
                values: pad
                    .into_iter()
                    .map(|value| stablehlo::ScalarValue::Integer(i128::from(value)))
                    .collect(),
            },
        ),
    ]
    .into();
    common::computation_with_regions(
        StableHloOp::SelectAndScatter,
        vec![input, source, scalar],
        vec![result],
        attributes,
        vec![select, scatter],
    )
    .is_ok()
}

#[test]
fn preserves_input_encoding_and_scatter_return_elements_without_refining_dynamic_shapes() {
    let input = ty("tensor<?x5xf16, #stablehlo.bounds<4, ?>>");
    let source = ty("tensor<2x3xf16>");
    let expected = ty("tensor<?x5xf64, #stablehlo.bounds<4, ?>>");
    assert_eq!(
        infer_select_and_scatter_type(
            &input,
            &source,
            &ty("tensor<f64>"),
            &[2, 3],
            Some(&[2, 1]),
            None
        )
        .unwrap(),
        expected
    );
    let input: Type = stablehlo::TensorType::static_shape(stablehlo::ElementType::F32, [4])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_select_and_scatter_type(
            &input,
            &ty("tensor<2xf32>"),
            &ty("tensor<f32>"),
            &[2],
            Some(&[2]),
            None
        )
        .unwrap(),
        input
    );
}

#[test]
fn preserves_exact_quantized_metadata_and_checks_source_elements() {
    let input = ty("tensor<4x!quant.uniform<i8:f32, 0.25:1>>");
    let source = ty("tensor<2x!quant.uniform<i8:f32, 0.25:1>>");
    let accumulator = ty("tensor<!quant.uniform<i8:f32, 0.5:2>>");
    assert_eq!(
        infer_select_and_scatter_type(&input, &source, &accumulator, &[2], Some(&[2]), None)
            .unwrap(),
        ty("tensor<4x!quant.uniform<i8:f32, 0.5:2>>")
    );
    assert!(
        infer_select_and_scatter_type(
            &input,
            &ty("tensor<2x!quant.uniform<i8:f32, 0.5:2>>"),
            &accumulator,
            &[2],
            Some(&[2]),
            None
        )
        .is_err()
    );
}

#[test]
fn scalar_inputs_use_empty_windows_and_reject_invalid_domains_and_attributes() {
    let scalar = ty("tensor<f32>");
    assert_eq!(
        infer_select_and_scatter_type(&scalar, &scalar, &scalar, &[], None, None).unwrap(),
        scalar
    );
    for (input, source, accumulator, window, strides, padding) in [
        (
            ty("!stablehlo.token"),
            scalar.clone(),
            scalar.clone(),
            vec![],
            None,
            None,
        ),
        (
            scalar.clone(),
            ty("!stablehlo.token"),
            scalar.clone(),
            vec![],
            None,
            None,
        ),
        (
            scalar.clone(),
            scalar.clone(),
            ty("!stablehlo.token"),
            vec![],
            None,
            None,
        ),
        (
            scalar.clone(),
            scalar.clone(),
            ty("tensor<1xf32>"),
            vec![],
            None,
            None,
        ),
        (
            scalar.clone(),
            ty("tensor<1xf32>"),
            scalar.clone(),
            vec![],
            None,
            None,
        ),
        (
            scalar.clone(),
            ty("tensor<i32>"),
            scalar.clone(),
            vec![],
            None,
            None,
        ),
        (
            scalar.clone(),
            scalar.clone(),
            scalar.clone(),
            vec![1],
            None,
            None,
        ),
        (
            ty("tensor<4xf32>"),
            ty("tensor<4xf32>"),
            scalar.clone(),
            vec![0],
            None,
            None,
        ),
        (
            ty("tensor<4xf32>"),
            ty("tensor<4xf32>"),
            scalar.clone(),
            vec![-1],
            None,
            None,
        ),
        (
            ty("tensor<4xf32>"),
            ty("tensor<4xf32>"),
            scalar.clone(),
            vec![1],
            Some(vec![0]),
            None,
        ),
        (
            ty("tensor<4xf32>"),
            ty("tensor<4xf32>"),
            scalar.clone(),
            vec![1],
            Some(vec![]),
            None,
        ),
        (
            ty("tensor<4xf32>"),
            ty("tensor<4xf32>"),
            scalar.clone(),
            vec![1],
            None,
            Some(vec![]),
        ),
    ] {
        assert!(
            infer_select_and_scatter_type(
                &input,
                &source,
                &accumulator,
                &window,
                strides.as_deref(),
                padding.as_deref()
            )
            .is_err()
        );
    }
}

fn witnesses() -> Vec<(Type, Vec<i64>)> {
    vec![
        (ty("tensor<?xf32, #stablehlo.bounds<0>>"), vec![0]),
        (ty("tensor<?xf32, #stablehlo.bounds<1>>"), vec![0, 1]),
        (ty("tensor<?xf32, #stablehlo.bounds<3>>"), vec![0, 1, 2, 3]),
        (ty("tensor<0xf32>"), vec![0]),
        (ty("tensor<1xf32>"), vec![1]),
        (ty("tensor<2xf32>"), vec![2]),
        (ty("tensor<3xf32>"), vec![3]),
    ]
}

fn output(size: i64, window: i64, stride: i64, pad: [i64; 2]) -> i64 {
    let padded = pad[0] + size + pad[1];
    if padded < window {
        0
    } else {
        (padded - window) / stride + 1
    }
}

#[test]
fn generated_source_bounds_and_strided_windows_match_enumerated_feasibility() {
    for (input, inputs) in witnesses() {
        for (source, sources) in witnesses() {
            for window in 1..=3 {
                for stride in 1..=3 {
                    for pad in [[0, 0], [-2, 1], [2, 3], [4, -1]] {
                        let expected = inputs
                            .iter()
                            .any(|&n| sources.contains(&output(n, window, stride, pad)));
                        let actual = infer_select_and_scatter_type(
                            &input,
                            &source,
                            &ty("tensor<f32>"),
                            &[window],
                            Some(&[stride]),
                            Some(&[pad]),
                        )
                        .is_ok();
                        assert_eq!(
                            actual, expected,
                            "{input:?} {source:?} {window} {stride} {pad:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn verifier_combines_input_result_and_source_bounds_into_one_window_witness() {
    for (input, inputs) in witnesses() {
        for (source, sources) in witnesses() {
            for (result, results) in witnesses() {
                for (window, stride, pad) in [
                    (1, 1, [0, 0]),
                    (2, 2, [0, 0]),
                    (3, 1, [2, 3]),
                    (1, 3, [-2, 1]),
                ] {
                    let expected = inputs.iter().any(|&n| {
                        results.contains(&n) && sources.contains(&output(n, window, stride, pad))
                    });
                    assert_eq!(
                        verify(
                            input.clone(),
                            source.clone(),
                            result.clone(),
                            window,
                            stride,
                            pad
                        ),
                        expected,
                        "{input:?} {source:?} {result:?} {window} {stride} {pad:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn unbounded_dimensions_still_obey_minimum_padding_and_int64_input_limits() {
    let input = ty("tensor<?xf32>");
    let accumulator = ty("tensor<f32>");
    for (source, window, stride, pad, expected) in [
        ("tensor<0xf32>", 1, 1, [2, 3], false),
        ("tensor<4xf32>", 1, 1, [2, 3], false),
        ("tensor<5xf32>", 1, 1, [2, 3], true),
        ("tensor<?xf32, #stablehlo.bounds<4>>", 1, 1, [2, 3], false),
        ("tensor<?xf32, #stablehlo.bounds<5>>", 1, 1, [2, 3], true),
        ("tensor<9223372036854775807xf32>", 1, 2, [0, 0], false),
        ("tensor<4611686018427387904xf32>", 1, 2, [0, 0], true),
        ("tensor<0xf32>", 1, 1, [i64::MIN, 0], true),
        ("tensor<9223372036854775807xf32>", 1, 1, [i64::MAX, 0], true),
        (
            "tensor<9223372036854775807xf32>",
            1,
            1,
            [i64::MAX, 1],
            false,
        ),
    ] {
        assert_eq!(
            infer_select_and_scatter_type(
                &input,
                &ty(source),
                &accumulator,
                &[window],
                Some(&[stride]),
                Some(&[pad])
            )
            .is_ok(),
            expected,
            "{source} {pad:?}"
        );
    }
}
