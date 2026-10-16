use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, ElementType, Instruction, Operation, Region,
    ScalarValue, StableHloOp, TensorType, Type, ValueId, infer_reduce_window_types,
};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn scalar() -> Type {
    ty("tensor<f32>")
}

fn region(count: usize) -> Region {
    Region::new(vec![Block::new(
        (0..count * 2)
            .map(|index| BlockArgument::new(ValueId(100 + index as u32), scalar()))
            .collect(),
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            (0..count)
                .map(|index| ValueId(100 + index as u32))
                .collect(),
            vec![],
            vec![],
        )],
    )])
}

#[test]
fn windowed_reduction_keeps_each_input_shape_and_bounds_and_each_reducer_element() {
    let inputs = [
        ty("tensor<?x3xf16, #stablehlo.bounds<7, ?>>"),
        ty("tensor<5x?xi8, #stablehlo.bounds<?, 4>>"),
    ];
    let accumulators = [ty("tensor<f32>"), ty("tensor<i32>")];
    assert_eq!(
        infer_reduce_window_types(
            &inputs,
            &accumulators,
            &[3, 2],
            Some(&[2, 1]),
            None,
            None,
            None,
        )
        .unwrap(),
        vec![
            ty("tensor<?x2xf32, #stablehlo.bounds<3, ?>>"),
            ty("tensor<2x?xi32, #stablehlo.bounds<?, 3>>"),
        ],
    );
}

#[test]
fn windowed_reduction_handles_asymmetric_padding_and_both_dilations() {
    assert_eq!(
        infer_reduce_window_types(
            &[ty("tensor<3x2xf32>")],
            &[scalar()],
            &[2, 1],
            Some(&[4, 1]),
            Some(&[2, 1]),
            Some(&[3, 1]),
            Some(&[[2, 1], [0, 0]]),
        )
        .unwrap(),
        vec![ty("tensor<2x2xf32>")],
    );
    assert_eq!(
        infer_reduce_window_types(
            &[ty("tensor<?xf32, #stablehlo.bounds<3>>")],
            &[scalar()],
            &[2],
            Some(&[4]),
            Some(&[2]),
            Some(&[3]),
            Some(&[[2, 1]]),
        )
        .unwrap(),
        vec![ty("tensor<?xf32, #stablehlo.bounds<2>>")],
    );
}

#[test]
fn windowed_reduction_small_geometry_matches_enumerated_valid_window_positions() {
    for size in 0..=5 {
        for width in 1..=3 {
            for stride in 1..=3 {
                for base in 1..=2 {
                    for dilation in 1..=2 {
                        for low in -2..=2 {
                            for high in -2..=2 {
                                let dilated_size =
                                    if size == 0 { 0 } else { (size - 1) * base + 1 };
                                let dilated_window = (width - 1) * dilation + 1;
                                let padded = dilated_size + low + high;
                                let expected = (0..padded.max(0))
                                    .filter(|&start| {
                                        start % stride == 0 && start + dilated_window <= padded
                                    })
                                    .count();
                                let config = (
                                    &[width][..],
                                    Some(&[stride][..]),
                                    Some(&[base][..]),
                                    Some(&[dilation][..]),
                                    Some(&[[low, high]][..]),
                                );
                                let result = infer_reduce_window_types(
                                    &[ty(&format!("tensor<{size}xf32>"))],
                                    &[scalar()],
                                    config.0,
                                    config.1,
                                    config.2,
                                    config.3,
                                    config.4,
                                )
                                .unwrap();
                                assert_eq!(
                                    result,
                                    vec![ty(&format!("tensor<{expected}xf32>"))],
                                    "size={size}, width={width}, stride={stride}, base={base}, dilation={dilation}, padding={low},{high}",
                                );
                                let result = infer_reduce_window_types(
                                    &[ty(&format!("tensor<?xf32, #stablehlo.bounds<{size}>>"))],
                                    &[scalar()],
                                    config.0,
                                    config.1,
                                    config.2,
                                    config.3,
                                    config.4,
                                )
                                .unwrap();
                                assert_eq!(
                                    result,
                                    vec![ty(&format!(
                                        "tensor<?xf32, #stablehlo.bounds<{expected}>>"
                                    ))],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn windowed_reduction_validates_every_window_vector_even_for_unknown_input_sizes() {
    let input = ty("tensor<?xf32>");
    for invalid in [vec![], vec![0], vec![-1], vec![1, 1]] {
        for field in 0..4 {
            let fields = [vec![1], vec![1], vec![1], vec![1]];
            let mut fields = fields;
            fields[field] = invalid.clone();
            assert!(
                infer_reduce_window_types(
                    std::slice::from_ref(&input),
                    &[scalar()],
                    &fields[0],
                    Some(&fields[1]),
                    Some(&fields[2]),
                    Some(&fields[3]),
                    None,
                )
                .is_err(),
                "field={field}, {invalid:?}"
            );
        }
    }
    for padding in [vec![], vec![[0, 0], [0, 0]]] {
        assert!(
            infer_reduce_window_types(
                std::slice::from_ref(&input),
                &[scalar()],
                &[1],
                None,
                None,
                None,
                Some(&padding),
            )
            .is_err()
        );
    }
}

#[test]
fn windowed_reduction_validates_operand_and_accumulator_counts_shapes_and_domains() {
    assert!(infer_reduce_window_types(&[], &[], &[], None, None, None, None).is_err());
    assert!(
        infer_reduce_window_types(&[ty("tensor<2xf32>")], &[], &[1], None, None, None, None)
            .is_err()
    );
    for invalid in [
        "tensor<*xf32>",
        "tensor<2xindex>",
        "tensor<2xtf32>",
        "tensor<2xsi32>",
        "!stablehlo.token",
        "tensor<2x!quant.uniform<i8:f32:0, {0.25,0.5}>>",
    ] {
        assert!(
            infer_reduce_window_types(&[ty(invalid)], &[scalar()], &[1], None, None, None, None)
                .is_err()
        );
    }
    for accumulator in [
        "tensor<1xf32>",
        "tensor<index>",
        "tensor<*xf32>",
        "!stablehlo.token",
    ] {
        assert!(
            infer_reduce_window_types(
                &[ty("tensor<2xf32>")],
                &[ty(accumulator)],
                &[1],
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
    }
    for inputs in [
        vec![
            ty("tensor<?xf32>"),
            ty("tensor<3xf32>"),
            ty("tensor<4xf32>"),
        ],
        vec![
            ty("tensor<?xf32, #stablehlo.bounds<3>>"),
            ty("tensor<4xf32>"),
        ],
        vec![ty("tensor<3xf32>"), ty("tensor<3x1xf32>")],
    ] {
        assert!(
            infer_reduce_window_types(
                &inputs,
                &vec![scalar(); inputs.len()],
                &[1],
                None,
                None,
                None,
                None,
            )
            .is_err()
        );
    }
}

#[test]
fn windowed_reduction_retains_scalar_and_quantized_reducer_metadata_without_unrelated_encodings() {
    let encoding = Attribute::Opaque {
        dialect: Some("consumer".into()),
        syntax: "layout".into(),
    };
    let input: Type = TensorType::scalar(ElementType::F32)
        .with_encoding(encoding.clone())
        .into();
    let accumulator: Type = TensorType::scalar(ElementType::F32)
        .with_encoding(encoding)
        .into();
    assert_eq!(
        infer_reduce_window_types(&[input], &[accumulator], &[], None, None, None, None).unwrap(),
        vec![scalar()],
    );
    assert_eq!(
        infer_reduce_window_types(
            &[ty(
                "tensor<?x!quant.uniform<i8:f32, 0.25:1>, #stablehlo.bounds<4>>"
            )],
            &[ty("tensor<!quant.uniform<i16:f32, 0.5:2>>")],
            &[2],
            None,
            None,
            None,
            None,
        )
        .unwrap(),
        vec![ty(
            "tensor<?x!quant.uniform<i16:f32, 0.5:2>, #stablehlo.bounds<3>>"
        )],
    );
}

#[test]
fn windowed_reduction_bounds_constrain_every_declared_result_not_only_corresponding_result() {
    for input_bounds in [[3, 5], [5, 3]] {
        for static_result in 0..2 {
            for (extent, accepted) in [(0, true), (2, true), (3, false)] {
                let mut inputs = input_bounds
                    .iter()
                    .map(|bound| ty(&format!("tensor<?xf32, #stablehlo.bounds<{bound}>>")))
                    .collect::<Vec<_>>();
                inputs.extend([scalar(), scalar()]);
                let mut outputs = vec![ty("tensor<?xf32>"); 2];
                outputs[static_result] = ty(&format!("tensor<{extent}xf32>"));
                let result = common::computation_with_regions(
                    StableHloOp::ReduceWindow,
                    inputs,
                    outputs,
                    [("window_dimensions".into(), Attribute::integers(vec![2]))].into(),
                    vec![region(2)],
                );
                assert_eq!(
                    result.is_ok(),
                    accepted,
                    "{input_bounds:?}, {static_result}, {extent}: {result:?}"
                );
            }
        }
    }
}

#[test]
fn windowed_reduction_checks_result_bounds_and_promoted_region_types() {
    let attrs: Attributes = [("window_dimensions".into(), Attribute::integers(vec![2]))].into();
    for (output, accepted) in [
        ("tensor<?xf32, #stablehlo.bounds<2>>", true),
        ("tensor<?xf32, #stablehlo.bounds<1>>", false),
    ] {
        assert_eq!(
            common::computation_with_regions(
                StableHloOp::ReduceWindow,
                vec![ty("tensor<3xf32>"), scalar()],
                vec![ty(output)],
                attrs.clone(),
                vec![region(1)],
            )
            .is_ok(),
            accepted
        );
    }
    let attrs: Attributes = [
        ("window_dimensions".into(), Attribute::integers(vec![2])),
        (
            "padding".into(),
            Attribute::DenseElements {
                ty: Box::new(ty("tensor<1x2xi64>")),
                values: vec![ScalarValue::Integer(-2), ScalarValue::Integer(1)],
            },
        ),
    ]
    .into();
    assert!(
        common::computation_with_regions(
            StableHloOp::ReduceWindow,
            vec![ty("tensor<3xf32>"), scalar()],
            vec![ty("tensor<1xf32>")],
            attrs,
            vec![region(1)],
        )
        .is_ok()
    );
}

#[test]
fn windowed_reduction_rejects_unrepresentable_extents_and_bounds_without_overflow() {
    for source in [
        format!("tensor<{}xf32>", i64::MAX),
        format!("tensor<?xf32, #stablehlo.bounds<{}>>", i64::MAX),
    ] {
        assert!(
            infer_reduce_window_types(
                &[ty(&source)],
                &[scalar()],
                &[1],
                None,
                Some(&[i64::MAX]),
                None,
                None,
            )
            .is_err()
        );
    }
    assert_eq!(
        infer_reduce_window_types(
            &[ty("tensor<0xf32>")],
            &[scalar()],
            &[i64::MAX],
            None,
            Some(&[i64::MAX]),
            Some(&[i64::MAX]),
            Some(&[[i64::MAX, i64::MAX]]),
        )
        .unwrap(),
        vec![ty("tensor<0xf32>")],
    );
}
