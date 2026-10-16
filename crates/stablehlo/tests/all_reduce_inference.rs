use stablehlo::text::parse_type;
use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, Instruction, Operation, Region, ScalarValue,
    StableHloOp, Type, ValueId, infer_all_reduce_types,
};

mod common;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn verify(
    opcode: StableHloOp,
    inputs: Vec<Type>,
    results: Vec<Type>,
    accumulator: Type,
    dynamic_root: bool,
) -> Result<stablehlo::Computation, stablehlo::BuildError> {
    let mut inputs = inputs;
    let mut attributes: Attributes = [(
        "replica_groups".into(),
        Attribute::DenseElements {
            ty: Box::new(ty("tensor<1x2xi64>")),
            values: vec![ScalarValue::Integer(0), ScalarValue::Integer(1)],
        },
    )]
    .into();
    if dynamic_root {
        inputs.push(ty(&format!("tensor<{}xi32>", results.len())));
        attributes.insert("has_dynamic_root".into(), Attribute::Unit);
    }
    let region = Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(100), accumulator.clone()),
            BlockArgument::new(ValueId(101), accumulator),
        ],
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            vec![ValueId(100)],
            vec![],
            vec![],
        )],
    )]);
    common::computation_with_regions(opcode, inputs, results, attributes, vec![region])
}

#[test]
fn all_reduce_preserves_independent_shapes_encodings_and_common_accumulator_elements() {
    let inputs = [
        ty("tensor<?xf16, #stablehlo.bounds<5>>"),
        ty("tensor<3x?xf32, #stablehlo.bounds<?, 7>>"),
        ty("tensor<f32>"),
    ];
    let expected = [
        ty("tensor<?xf64, #stablehlo.bounds<5>>"),
        ty("tensor<3x?xf64, #stablehlo.bounds<?, 7>>"),
        ty("tensor<f64>"),
    ];
    assert_eq!(
        infer_all_reduce_types(&inputs, &ty("tensor<f64>")).unwrap(),
        expected
    );
    for (opcode, root) in [
        (StableHloOp::AllReduce, false),
        (StableHloOp::CollectiveReduce, false),
        (StableHloOp::CollectiveReduce, true),
    ] {
        verify(
            opcode,
            inputs.to_vec(),
            expected.to_vec(),
            ty("tensor<f64>"),
            root,
        )
        .unwrap();
    }
}

#[test]
fn all_reduce_retains_zero_extents_and_unrelated_encodings() {
    let input: Type = stablehlo::TensorType::static_shape(stablehlo::ElementType::I8, [0, 3])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    let expected: Type = stablehlo::TensorType::static_shape(stablehlo::ElementType::I32, [0, 3])
        .with_encoding(Attribute::Opaque {
            dialect: Some("consumer".into()),
            syntax: "layout".into(),
        })
        .into();
    assert_eq!(
        infer_all_reduce_types(&[input], &ty("tensor<i32>")).unwrap(),
        [expected]
    );
    assert!(
        infer_all_reduce_types(&[], &ty("tensor<f32>"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn all_reduce_preserves_quantized_input_encoding_and_scalar_return_parameters() {
    let input = ty("tensor<?x!quant.uniform<i8:f32, 0.25:1>, #stablehlo.bounds<5>>");
    let accumulator = ty("tensor<!quant.uniform<i16:f32, 0.5:2>>");
    let expected = ty("tensor<?x!quant.uniform<i16:f32, 0.5:2>, #stablehlo.bounds<5>>");
    assert_eq!(
        infer_all_reduce_types(std::slice::from_ref(&input), &accumulator).unwrap(),
        std::slice::from_ref(&expected)
    );
    verify(
        StableHloOp::AllReduce,
        vec![input],
        vec![expected],
        accumulator,
        false,
    )
    .unwrap();
}

#[test]
fn all_reduce_rejects_non_scalar_accumulators_and_invalid_domains() {
    for accumulator in [
        "tensor<1xf32>",
        "tensor<*xf32>",
        "tensor<index>",
        "!stablehlo.token",
        "tensor<2x!quant.uniform<i8:f32:0, {0.25,0.5}>>",
    ] {
        assert!(
            infer_all_reduce_types(&[ty("tensor<3xf32>")], &ty(accumulator)).is_err(),
            "{accumulator}"
        );
    }
    for input in [
        "tensor<*xf32>",
        "tensor<3xindex>",
        "tensor<3xtf32>",
        "!stablehlo.token",
        "tensor<2x!quant.uniform<i8:f32:0, {0.25,0.5}>>",
    ] {
        assert!(
            infer_all_reduce_types(&[ty(input)], &ty("tensor<f32>")).is_err(),
            "{input}"
        );
    }
}

#[test]
fn all_reduce_verifier_checks_each_result_bounds_and_accumulator_promotion() {
    for (input, result, accumulator, accepted) in [
        (
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<4xf32>",
            "tensor<f32>",
            false,
        ),
        (
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<3xf32>",
            "tensor<f32>",
            true,
        ),
        (
            "tensor<4xf32>",
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<f32>",
            false,
        ),
        (
            "tensor<4xf32>",
            "tensor<?xf32, #stablehlo.bounds<4>>",
            "tensor<f32>",
            true,
        ),
        ("tensor<4xi32>", "tensor<4xf32>", "tensor<f32>", false),
        ("tensor<4xf32>", "tensor<4xf16>", "tensor<f16>", false),
        ("tensor<4xf16>", "tensor<4xf32>", "tensor<f32>", true),
    ] {
        for (opcode, root) in [
            (StableHloOp::AllReduce, false),
            (StableHloOp::CollectiveReduce, false),
            (StableHloOp::CollectiveReduce, true),
        ] {
            assert_eq!(
                verify(
                    opcode,
                    vec![ty(input)],
                    vec![ty(result)],
                    ty(accumulator),
                    root
                )
                .is_ok(),
                accepted,
                "{opcode:?}, {input}, {result}, {accumulator}"
            );
        }
    }
}

#[test]
fn all_reduce_two_way_bounded_shape_witnesses_match_exact_feasibility() {
    let witnesses = [
        "tensor<0xf32>",
        "tensor<1xf32>",
        "tensor<2xf32>",
        "tensor<3xf32>",
        "tensor<?xf32>",
        "tensor<?xf32, #stablehlo.bounds<1>>",
        "tensor<?xf32, #stablehlo.bounds<2>>",
    ];
    let accepts = |index, v| match index {
        0..=3 => index == v,
        4 => true,
        5 => v <= 1,
        6 => v <= 2,
        _ => unreachable!(),
    };
    for (i, input) in witnesses.into_iter().enumerate() {
        for (j, result) in witnesses.into_iter().enumerate() {
            let feasible = (0..=4).any(|v| accepts(i, v) && accepts(j, v));
            for opcode in [StableHloOp::AllReduce, StableHloOp::CollectiveReduce] {
                assert_eq!(
                    verify(
                        opcode,
                        vec![ty(input)],
                        vec![ty(result)],
                        ty("tensor<f32>"),
                        false
                    )
                    .is_ok(),
                    feasible,
                    "{opcode:?}, {input}, {result}"
                );
            }
        }
    }
}
