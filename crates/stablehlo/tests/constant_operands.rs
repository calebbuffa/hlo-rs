use stablehlo::{
    Attribute, Attributes, Block, BuildError, Computation, ConvDimensionNumbers, ElementType,
    GatherDimensionNumbers, Instruction, Operation, Region, ScalarValue, StableHloAttribute,
    StableHloOp, TensorType, Type, ValueId,
};

fn constant(id: u32, values: &[i128]) -> Instruction {
    let ty = Type::from(TensorType::static_shape(
        ElementType::I64,
        [values.len() as u64],
    ));
    Instruction::new(
        Operation::stablehlo(
            StableHloOp::Constant,
            [(
                "value".to_owned(),
                Attribute::DenseElements {
                    ty: Box::new(ty.clone()),
                    values: values.iter().copied().map(ScalarValue::Integer).collect(),
                },
            )]
            .into(),
        ),
        vec![],
        vec![],
        ValueId(id),
        ty,
    )
}

fn parameter(id: u32, ty: Type) -> Instruction {
    Instruction::new(
        Operation::parameter(id, format!("arg{id}")),
        vec![],
        vec![],
        ValueId(id),
        ty,
    )
}

#[test]
fn dynamic_gather_uses_constant_slice_bounds_and_offset_extents() {
    for (sizes, accepted) in [
        ([1, 4], true),
        ([1, 3], false),
        ([2, 4], false),
        ([1, 5], false),
        ([-1, 4], false),
    ] {
        let input = parameter(0, TensorType::static_shape(ElementType::F32, [5, 4]).into());
        let indices = parameter(1, TensorType::static_shape(ElementType::I64, [2, 1]).into());
        let size = constant(2, &sizes);
        let gather = Instruction::new(
            Operation::stablehlo(
                StableHloOp::DynamicGather,
                [(
                    "dimension_numbers".into(),
                    Attribute::StableHlo(Box::new(StableHloAttribute::GatherDimensionNumbers(
                        GatherDimensionNumbers {
                            offset_dims: vec![1],
                            collapsed_slice_dims: vec![0],
                            start_index_map: vec![0],
                            index_vector_dim: 1,
                            ..Default::default()
                        },
                    ))),
                )]
                .into(),
            ),
            vec![ValueId(0), ValueId(1), ValueId(2)],
            vec![],
            ValueId(3),
            TensorType::static_shape(ElementType::F32, [2, 4]).into(),
        );
        assert_eq!(
            Computation::try_new("gather", vec![input, indices, size, gather], ValueId(3)).is_ok(),
            accepted,
            "sizes: {sizes:?}"
        );
    }
}

#[test]
fn dynamic_convolution_uses_constant_padding_to_verify_spatial_results() {
    for (padding, accepted) in [([0, 0], true), ([1, 1], false), ([-1, 0], false)] {
        let input = parameter(
            0,
            TensorType::static_shape(ElementType::F32, [2, 7, 4]).into(),
        );
        let kernel = parameter(
            1,
            TensorType::static_shape(ElementType::F32, [3, 4, 6]).into(),
        );
        let padding_type: Type = TensorType::static_shape(ElementType::I64, [1, 2]).into();
        let constant = Instruction::new(
            Operation::stablehlo(
                StableHloOp::Constant,
                [(
                    "value".into(),
                    Attribute::DenseElements {
                        ty: Box::new(padding_type.clone()),
                        values: padding.into_iter().map(ScalarValue::Integer).collect(),
                    },
                )]
                .into(),
            ),
            vec![],
            vec![],
            ValueId(2),
            padding_type,
        );
        let attributes = [
            (
                "dimension_numbers".into(),
                Attribute::StableHlo(Box::new(StableHloAttribute::ConvDimensionNumbers(
                    ConvDimensionNumbers {
                        input_batch_dimension: 0,
                        input_feature_dimension: 2,
                        input_spatial_dimensions: vec![1],
                        kernel_input_feature_dimension: 1,
                        kernel_output_feature_dimension: 2,
                        kernel_spatial_dimensions: vec![0],
                        output_batch_dimension: 0,
                        output_feature_dimension: 2,
                        output_spatial_dimensions: vec![1],
                    },
                ))),
            ),
            ("feature_group_count".into(), Attribute::integer(1)),
            ("batch_group_count".into(), Attribute::integer(1)),
        ]
        .into();
        let convolution = Instruction::new(
            Operation::stablehlo(StableHloOp::DynamicConv, attributes),
            vec![ValueId(0), ValueId(1), ValueId(2)],
            vec![],
            ValueId(3),
            TensorType::static_shape(ElementType::F32, [2, 5, 6]).into(),
        );
        assert_eq!(
            Computation::try_new(
                "convolution",
                vec![input, kernel, constant, convolution],
                ValueId(3)
            )
            .is_ok(),
            accepted,
            "padding: {padding:?}"
        );
    }
}

fn dynamic_transform(
    opcode: StableHloOp,
    input: TensorType,
    output: TensorType,
    shape: &[i128],
    attributes: Attributes,
) -> Result<Computation, BuildError> {
    let input = parameter(0, input.into());
    let shape = constant(1, shape);
    let transform = Instruction::new(
        Operation::stablehlo(opcode, attributes),
        vec![ValueId(0), ValueId(1)],
        vec![],
        ValueId(2),
        output.into(),
    );
    Computation::try_new(
        "dynamic_transform",
        vec![input, shape, transform],
        ValueId(2),
    )
}

#[test]
fn dynamic_reshape_checks_constant_shape_even_with_dynamic_result_types() {
    assert!(matches!(
        dynamic_transform(
            StableHloOp::DynamicReshape,
            TensorType::static_shape(ElementType::F32, [2, 3]),
            TensorType::new(ElementType::F32, vec![None, None]),
            &[4, 2],
            Attributes::new()
        ),
        Err(BuildError::InvalidConstantOperand {
            operation: "dynamic_reshape",
            value: ValueId(1),
            ..
        })
    ));
    assert!(
        dynamic_transform(
            StableHloOp::DynamicReshape,
            TensorType::static_shape(ElementType::F32, [2, 3]),
            TensorType::new(ElementType::F32, vec![None, None]),
            &[3, 2],
            Attributes::new(),
        )
        .is_ok()
    );
}

#[test]
fn constant_shapes_must_match_declared_static_extents_and_be_nonnegative() {
    for shape in [[2, 3], [-1, 6]] {
        assert!(matches!(
            dynamic_transform(
                StableHloOp::DynamicReshape,
                TensorType::static_shape(ElementType::F32, [2, 3]),
                TensorType::static_shape(ElementType::F32, [3, 2]),
                &shape,
                Attributes::new()
            ),
            Err(BuildError::InvalidConstantOperand { .. })
        ));
    }
}

#[test]
fn reshape_constant_element_count_comparison_handles_zero_and_overflow() {
    let dimension = i64::MAX as u64;
    assert!(
        dynamic_transform(
            StableHloOp::DynamicReshape,
            TensorType::static_shape(ElementType::F32, [dimension, dimension]),
            TensorType::new(ElementType::F32, vec![None, None]),
            &[i128::from(dimension), i128::from(dimension)],
            Attributes::new(),
        )
        .is_ok()
    );
    assert!(
        dynamic_transform(
            StableHloOp::DynamicReshape,
            TensorType::static_shape(ElementType::F32, [0, dimension]),
            TensorType::new(ElementType::F32, vec![None, None]),
            &[0, 3],
            Attributes::new(),
        )
        .is_ok()
    );
}

#[test]
fn dynamic_broadcast_checks_runtime_shape_against_input_static_dimensions() {
    let attributes = || {
        [(
            "broadcast_dimensions".to_owned(),
            Attribute::integers(vec![0, 1]),
        )]
        .into()
    };
    assert!(matches!(
        dynamic_transform(
            StableHloOp::DynamicBroadcastInDim,
            TensorType::static_shape(ElementType::F32, [1, 3]),
            TensorType::new(ElementType::F32, vec![None, None]),
            &[2, 4],
            attributes()
        ),
        Err(BuildError::InvalidConstantOperand {
            operation: "dynamic_broadcast_in_dim",
            ..
        })
    ));
    assert!(
        dynamic_transform(
            StableHloOp::DynamicBroadcastInDim,
            TensorType::static_shape(ElementType::F32, [1, 3]),
            TensorType::new(ElementType::F32, vec![None, None]),
            &[2, 3],
            attributes(),
        )
        .is_ok()
    );
}

fn dynamic_pad(
    low: &[i128],
    high: &[i128],
    interior: &[i128],
    output: [u64; 2],
) -> Result<Computation, BuildError> {
    let input = parameter(0, TensorType::static_shape(ElementType::F32, [2, 3]).into());
    let padding_value = parameter(1, TensorType::scalar(ElementType::F32).into());
    let pad = Instruction::new(
        Operation::stablehlo(StableHloOp::DynamicPad, Attributes::new()),
        (0..5).map(ValueId).collect(),
        vec![],
        ValueId(5),
        TensorType::static_shape(ElementType::F32, output).into(),
    );
    Computation::try_new(
        "dynamic_pad",
        vec![
            input,
            padding_value,
            constant(2, low),
            constant(3, high),
            constant(4, interior),
            pad,
        ],
        ValueId(5),
    )
}

#[test]
fn dynamic_pad_checks_negative_interior_and_inferred_result_dimensions() {
    assert!(dynamic_pad(&[0, 1], &[2, 1], &[1, 2], [5, 9]).is_ok());
    for (interior, output) in [([1, -1], [5, 9]), ([1, 2], [5, 8])] {
        assert!(matches!(
            dynamic_pad(&[0, 1], &[2, 1], &interior, output),
            Err(BuildError::InvalidConstantOperand {
                operation: "dynamic_pad",
                ..
            })
        ));
    }
    assert!(dynamic_pad(&[-1, 0], &[0, 0], &[0, 0], [1, 3]).is_ok());
    assert!(matches!(
        dynamic_pad(&[-3, 0], &[0, 0], &[0, 0], [0, 3]),
        Err(BuildError::InvalidConstantOperand { .. })
    ));
}

#[test]
fn dynamic_iota_checks_constant_output_shape() {
    let iota = Instruction::new(
        Operation::stablehlo(
            StableHloOp::DynamicIota,
            [("iota_dimension".to_owned(), Attribute::integer(1))].into(),
        ),
        vec![ValueId(0)],
        vec![],
        ValueId(1),
        TensorType::static_shape(ElementType::I32, [2, 3]).into(),
    );
    assert!(matches!(
        Computation::try_new("iota", vec![constant(0, &[2, 4]), iota], ValueId(1)),
        Err(BuildError::InvalidConstantOperand {
            operation: "dynamic_iota",
            ..
        })
    ));
}

fn region_return(operand: ValueId) -> Instruction {
    Instruction::with_results(
        Operation::stablehlo(StableHloOp::Return, Attributes::new()),
        vec![operand],
        vec![],
        vec![],
    )
}

#[test]
fn constant_shape_checks_apply_inside_capturing_regions() {
    let input = parameter(0, TensorType::static_shape(ElementType::F32, [2, 3]).into());
    let predicate = parameter(1, TensorType::scalar(ElementType::I1).into());
    let result = Type::from(TensorType::new(ElementType::F32, vec![None, None]));
    let branch = |id| {
        Region::new(vec![Block::new(
            vec![],
            vec![
                Instruction::new(
                    Operation::stablehlo(StableHloOp::DynamicReshape, Attributes::new()),
                    vec![ValueId(0), ValueId(2)],
                    vec![],
                    ValueId(id),
                    result.clone(),
                ),
                region_return(ValueId(id)),
            ],
        )])
    };
    let if_op = Instruction::new(
        Operation::stablehlo(StableHloOp::If, Attributes::new()),
        vec![ValueId(1)],
        vec![],
        ValueId(3),
        result.clone(),
    )
    .with_regions(vec![branch(10), branch(11)]);
    assert!(matches!(
        Computation::try_new(
            "nested_shape",
            vec![input, predicate, constant(2, &[4, 2]), if_op],
            ValueId(3)
        ),
        Err(BuildError::InvalidConstantOperand {
            operation: "dynamic_reshape",
            ..
        })
    ));
}
