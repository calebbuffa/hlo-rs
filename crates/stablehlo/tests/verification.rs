use stablehlo::{
    Attribute, Block, BlockArgument, BuildError, Computation, ElementType, Instruction, Module,
    Operation, OperationResult, QuantizedTensorType, Region, StableHloOp, TensorType, Type,
    ValueId,
};

fn region_return(operands: Vec<ValueId>) -> Instruction {
    Instruction::with_results(
        Operation::stablehlo(StableHloOp::Return, Default::default()),
        operands,
        Vec::new(),
        Vec::new(),
    )
}

fn comparison() -> Operation {
    Operation::stablehlo(
        StableHloOp::Compare,
        [(
            "comparison_direction".to_owned(),
            stablehlo::Attribute::Enum {
                dialect: "stablehlo".to_owned(),
                name: "comparison_direction".to_owned(),
                value: "EQ".to_owned(),
            },
        )]
        .into(),
    )
}

#[test]
fn computation_rejects_an_invalid_quantized_type() {
    let ty = Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(1)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.0],
        vec![0],
    ));
    let instruction = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        ty,
    );

    assert!(matches!(
        Computation::try_new("invalid", vec![instruction], ValueId(0)),
        Err(BuildError::InvalidType(_))
    ));
}

#[test]
fn verifier_checks_known_fixed_operand_and_result_counts() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let malformed_unary = Instruction::with_results(
        Operation::stablehlo(StableHloOp::Negate, Default::default()),
        Vec::new(),
        Vec::new(),
        vec![OperationResult::new(ValueId(1), tensor.clone())],
    );
    assert_eq!(
        Computation::try_new(
            "bad_operand_count",
            vec![input.clone(), malformed_unary],
            ValueId(0)
        ),
        Err(BuildError::InvalidArity {
            operation: "negate",
            expected: 1,
            actual: 0,
        })
    );

    let malformed_result = Instruction::with_results(
        Operation::stablehlo(StableHloOp::Negate, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(
        Computation::try_new(
            "bad_result_count",
            vec![input, malformed_result],
            ValueId(0)
        ),
        Err(BuildError::InvalidResultArity {
            operation: "negate",
            expected: 1,
            actual: 0,
        })
    );
}

#[test]
fn iota_operations_validate_dimensions_and_dynamic_shape_operands() {
    let static_iota = Instruction::new(
        Operation::stablehlo(
            StableHloOp::Iota,
            [(
                "iota_dimension".to_owned(),
                stablehlo::Attribute::integer(1),
            )]
            .into(),
        ),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
    );
    assert!(Computation::try_new("iota", vec![static_iota], ValueId(0)).is_ok());

    let shape = Instruction::new(
        Operation::parameter(0, "shape"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::Index, [2])),
    );
    let dynamic_iota = Instruction::new(
        Operation::stablehlo(
            StableHloOp::DynamicIota,
            [(
                "iota_dimension".to_owned(),
                stablehlo::Attribute::integer(0),
            )]
            .into(),
        ),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::new(ElementType::F32, vec![None, None])),
    );
    assert!(Computation::try_new("dynamic_iota", vec![shape, dynamic_iota], ValueId(1)).is_ok());

    let invalid_dimension = Instruction::new(
        Operation::stablehlo(
            StableHloOp::Iota,
            [(
                "iota_dimension".to_owned(),
                stablehlo::Attribute::integer(2),
            )]
            .into(),
        ),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
    );
    assert!(matches!(
        Computation::try_new("invalid_iota", vec![invalid_dimension], ValueId(0)),
        Err(BuildError::InvalidResultType {
            operation: "iota",
            ..
        })
    ));
}

#[test]
fn verifier_requires_the_declared_fixed_region_count() {
    let input = Instruction::new(
        Operation::parameter(0, "pred"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::scalar(ElementType::I1)),
    );
    let conditional = Instruction::with_results(
        Operation::stablehlo(StableHloOp::If, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        Vec::new(),
    );
    assert_eq!(
        Computation::try_new("missing_if_regions", vec![input, conditional], ValueId(0)),
        Err(BuildError::InvalidRegionCount {
            operation: "if",
            expected: Some(2),
            actual: 0,
        })
    );
}

#[test]
fn async_start_regions_only_capture_explicit_operands() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let unrelated = Instruction::new(
        Operation::parameter(1, "unrelated"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        tensor.clone(),
    );
    let captured_value = Instruction::new(
        Operation::stablehlo(StableHloOp::Negate, Default::default()),
        vec![ValueId(1)],
        Vec::new(),
        ValueId(10),
        tensor.clone(),
    );
    let async_start = Instruction::new(
        Operation::stablehlo(StableHloOp::AsyncStart, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(2),
        Type::Future(vec![TensorType::scalar(ElementType::F32).into()]),
    )
    .with_regions(vec![Region::new(vec![Block::new(
        Vec::new(),
        vec![captured_value, region_return(vec![ValueId(10)])],
    )])]);
    assert!(matches!(
        Computation::try_new(
            "async_capture",
            vec![input, unrelated, async_start],
            ValueId(2)
        ),
        Err(BuildError::UnknownOrNonDominatingValue {
            instruction: ValueId(10),
            operand: ValueId(1),
        })
    ));
}

#[test]
fn async_start_regions_use_explicit_block_arguments() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let async_start = Instruction::new(
        Operation::stablehlo(StableHloOp::AsyncStart, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::Future(vec![TensorType::scalar(ElementType::F32).into()]),
    )
    .with_regions(vec![Region::new(vec![Block::new(
        vec![BlockArgument::new(ValueId(10), tensor.clone())],
        vec![
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Slice,
                    [
                        ("start_indices".into(), Attribute::integers(vec![])),
                        ("limit_indices".into(), Attribute::integers(vec![])),
                        ("strides".into(), Attribute::integers(vec![])),
                    ]
                    .into(),
                ),
                vec![ValueId(10)],
                vec![],
                ValueId(11),
                tensor,
            ),
            region_return(vec![ValueId(11)]),
        ],
    )])]);
    assert!(
        Computation::try_new(
            "async_operand_capture",
            vec![input, async_start],
            ValueId(1)
        )
        .is_ok()
    );
}

#[test]
fn async_start_return_types_match_its_future_type() {
    let tensor_type = Type::from(TensorType::scalar(ElementType::F32));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor_type.clone(),
    );
    let async_start = Instruction::new(
        Operation::stablehlo(StableHloOp::AsyncStart, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::Future(vec![TensorType::scalar(ElementType::I32).into()]),
    )
    .with_regions(vec![Region::new(vec![Block::new(
        vec![BlockArgument::new(ValueId(10), tensor_type.clone())],
        vec![
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Slice,
                    [
                        ("start_indices".into(), Attribute::integers(vec![])),
                        ("limit_indices".into(), Attribute::integers(vec![])),
                        ("strides".into(), Attribute::integers(vec![])),
                    ]
                    .into(),
                ),
                vec![ValueId(10)],
                vec![],
                ValueId(11),
                tensor_type,
            ),
            region_return(vec![ValueId(11)]),
        ],
    )])]);
    assert!(matches!(
        Computation::try_new("async_type_mismatch", vec![input, async_start], ValueId(1)),
        Err(BuildError::InvalidResultType {
            operation: "async_start",
            result: ValueId(1),
        })
    ));
}

#[test]
fn if_branches_must_return_the_declared_result_types() {
    let pred_type = Type::from(TensorType::scalar(ElementType::I1));
    let tensor_type = Type::from(TensorType::scalar(ElementType::F32));
    let pred = Instruction::new(
        Operation::parameter(0, "pred"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        pred_type,
    );
    let input = Instruction::new(
        Operation::parameter(1, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        tensor_type.clone(),
    );
    let conditional = Instruction::new(
        Operation::stablehlo(StableHloOp::If, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(2),
        tensor_type,
    )
    .with_regions(vec![
        Region::new(vec![Block::new(
            Vec::new(),
            vec![region_return(vec![ValueId(1)])],
        )]),
        Region::new(vec![Block::new(
            Vec::new(),
            vec![region_return(Vec::new())],
        )]),
    ]);
    assert_eq!(
        Computation::try_new(
            "invalid_if_result",
            vec![pred, input, conditional],
            ValueId(2)
        ),
        Err(BuildError::InvalidRegionSignature {
            operation: "if",
            region: 1,
            expected_arguments: 0,
            actual_arguments: 0,
            expected_results: 1,
            actual_results: 0,
        })
    );
}

#[test]
fn while_regions_match_the_carried_state_signature() {
    let state_type = Type::from(TensorType::scalar(ElementType::F32));
    let predicate_type = Type::from(TensorType::scalar(ElementType::I1));
    let input = Instruction::new(
        Operation::parameter(0, "state"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        state_type.clone(),
    );
    let condition = Instruction::with_results(
        comparison(),
        vec![ValueId(10), ValueId(10)],
        Vec::new(),
        vec![OperationResult::new(ValueId(11), predicate_type)],
    );
    let loop_op = Instruction::new(
        Operation::stablehlo(StableHloOp::While, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        state_type.clone(),
    )
    .with_regions(vec![
        Region::new(vec![Block::new(
            vec![BlockArgument::new(ValueId(10), state_type.clone())],
            vec![condition, region_return(vec![ValueId(11)])],
        )]),
        Region::new(vec![Block::new(
            vec![BlockArgument::new(ValueId(12), state_type)],
            vec![region_return(vec![ValueId(12)])],
        )]),
    ]);
    assert!(Computation::try_new("valid_while", vec![input, loop_op], ValueId(1)).is_ok());
}

#[test]
fn verifier_requires_region_terminators_and_rejects_top_level_return() {
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::scalar(ElementType::F32)),
    );
    let loop_op = Instruction::new(
        Operation::stablehlo(StableHloOp::While, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::scalar(ElementType::F32)),
    )
    .with_regions(vec![
        Region::new(vec![Block::default()]),
        Region::new(vec![Block::default()]),
    ]);
    assert_eq!(
        Computation::try_new("unterminated", vec![input, loop_op], ValueId(1)),
        Err(BuildError::MissingRegionTerminator)
    );

    assert_eq!(
        Computation::try_new_with_roots(
            "top_level_return",
            vec![region_return(Vec::new())],
            Vec::new(),
        ),
        Err(BuildError::ReturnOutsideRegion)
    );
}

#[test]
fn verifier_checks_tuple_construction_and_extraction() {
    let f32_ty = Type::from(TensorType::scalar(ElementType::F32));
    let i32_ty = Type::from(TensorType::scalar(ElementType::I32));
    let first = Instruction::new(
        Operation::parameter(0, "first"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        f32_ty.clone(),
    );
    let second = Instruction::new(
        Operation::parameter(1, "second"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        i32_ty.clone(),
    );
    let tuple = Instruction::new(
        Operation::stablehlo(StableHloOp::Tuple, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::Tuple(vec![f32_ty, i32_ty.clone()]),
    );
    let element = Instruction::new(
        Operation::stablehlo(
            StableHloOp::GetTupleElement,
            [("index".to_owned(), stablehlo::Attribute::i32(1))].into(),
        ),
        vec![ValueId(2)],
        Vec::new(),
        ValueId(3),
        i32_ty,
    );
    assert!(
        Computation::try_new(
            "tuple",
            vec![first.clone(), second.clone(), tuple.clone(), element],
            ValueId(3)
        )
        .is_ok()
    );

    let invalid_tuple = Instruction::new(
        Operation::stablehlo(StableHloOp::Tuple, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::Tuple(vec![Type::Token]),
    );
    assert!(matches!(
        Computation::try_new("bad_tuple", vec![first, second, invalid_tuple], ValueId(2)),
        Err(BuildError::InvalidResultType {
            operation: "tuple",
            ..
        })
    ));
}

#[test]
fn compare_produces_a_predicate_tensor_with_the_operand_shape() {
    let input_type = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let predicate_type = Type::from(TensorType::static_shape(ElementType::I1, [2, 3]));
    let lhs = Instruction::new(
        Operation::parameter(0, "lhs"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        input_type.clone(),
    );
    let rhs = Instruction::new(
        Operation::parameter(1, "rhs"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        input_type,
    );
    let compare = Instruction::new(
        comparison(),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        predicate_type,
    );
    assert!(
        Computation::try_new(
            "compare",
            vec![lhs.clone(), rhs.clone(), compare],
            ValueId(2)
        )
        .is_ok()
    );

    let mismatched = Instruction::new(
        comparison(),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::from(TensorType::static_shape(ElementType::I1, [3, 2])),
    );
    assert!(matches!(
        Computation::try_new("bad_compare", vec![lhs, rhs, mismatched], ValueId(2)),
        Err(BuildError::InvalidResultType {
            operation: "compare",
            ..
        })
    ));
}

#[test]
fn compare_rejects_different_operand_element_types() {
    let lhs = Instruction::new(
        Operation::parameter(0, "lhs"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::scalar(ElementType::F32)),
    );
    let rhs = Instruction::new(
        Operation::parameter(1, "rhs"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::scalar(ElementType::I32)),
    );
    let compare = Instruction::new(
        comparison(),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::from(TensorType::scalar(ElementType::I1)),
    );
    assert_eq!(
        Computation::try_new("incompatible_compare", vec![lhs, rhs, compare], ValueId(2)),
        Err(BuildError::IncompatibleOperands {
            operation: "compare",
            lhs: ValueId(0),
            rhs: ValueId(1),
        })
    );
}

#[test]
fn convert_and_complex_enforce_shape_and_element_rules() {
    let input_type = Type::from(TensorType::static_shape(ElementType::F32, [2]));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        input_type,
    );
    let converted = Instruction::new(
        Operation::stablehlo(StableHloOp::Convert, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::static_shape(ElementType::C64, [2])),
    );
    assert!(Computation::try_new("convert", vec![input, converted], ValueId(1)).is_ok());

    let real = Instruction::new(
        Operation::parameter(0, "real"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::F64, [2])),
    );
    let imaginary = Instruction::new(
        Operation::parameter(1, "imaginary"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::static_shape(ElementType::F64, [2])),
    );
    let complex = Instruction::new(
        Operation::stablehlo(StableHloOp::Complex, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::from(TensorType::static_shape(ElementType::C128, [2])),
    );
    assert!(Computation::try_new("complex", vec![real, imaginary, complex], ValueId(2)).is_ok());
}

fn dot_general_computation(
    lhs_shape: Vec<u64>,
    rhs_shape: Vec<u64>,
    result_shape: Vec<u64>,
    dimensions: stablehlo::DotDimensionNumbers,
) -> Result<Computation, BuildError> {
    dot_general_computation_with_precision(
        lhs_shape,
        rhs_shape,
        result_shape,
        dimensions,
        None,
        false,
    )
}

fn dot_general_computation_with_precision(
    lhs_shape: Vec<u64>,
    rhs_shape: Vec<u64>,
    result_shape: Vec<u64>,
    dimensions: stablehlo::DotDimensionNumbers,
    precision_config: Option<Vec<stablehlo::Precision>>,
    has_algorithm: bool,
) -> Result<Computation, BuildError> {
    dot_general_computation_with_types(
        Type::from(TensorType::static_shape(ElementType::F32, lhs_shape)),
        Type::from(TensorType::static_shape(ElementType::F32, rhs_shape)),
        Type::from(TensorType::static_shape(ElementType::F32, result_shape)),
        dimensions,
        precision_config,
        has_algorithm,
    )
}

fn dot_general_computation_with_types(
    lhs_type: Type,
    rhs_type: Type,
    result_type: Type,
    dimensions: stablehlo::DotDimensionNumbers,
    precision_config: Option<Vec<stablehlo::Precision>>,
    has_algorithm: bool,
) -> Result<Computation, BuildError> {
    let lhs = Instruction::new(
        Operation::parameter(0, "lhs"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        lhs_type,
    );
    let rhs = Instruction::new(
        Operation::parameter(1, "rhs"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        rhs_type,
    );
    let mut attributes = std::collections::BTreeMap::from([(
        "dot_dimension_numbers".to_owned(),
        stablehlo::Attribute::StableHlo(Box::new(
            stablehlo::StableHloAttribute::DotDimensionNumbers(dimensions),
        )),
    )]);
    if let Some(precision_config) = precision_config {
        attributes.insert(
            "precision_config".to_owned(),
            stablehlo::Attribute::precision_config(precision_config),
        );
    }
    if has_algorithm {
        let f32_type = Type::Element(ElementType::F32);
        attributes.insert(
            "algorithm".to_owned(),
            stablehlo::Attribute::StableHlo(Box::new(stablehlo::StableHloAttribute::DotAlgorithm(
                Box::new(stablehlo::DotAlgorithm {
                    lhs_precision_type: f32_type.clone(),
                    rhs_precision_type: f32_type.clone(),
                    accumulation_type: f32_type,
                    lhs_component_count: 1,
                    rhs_component_count: 1,
                    num_primitive_operations: 1,
                    allow_imprecise_accumulation: false,
                }),
            ))),
        );
    }
    let dot = Instruction::new(
        Operation::stablehlo(StableHloOp::DotGeneral, attributes),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        result_type,
    );
    Computation::try_new("dot_general", vec![lhs, rhs, dot], ValueId(2))
}

#[test]
fn dot_general_infers_result_shape_from_dimension_numbers() {
    let dimensions = stablehlo::DotDimensionNumbers {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        ..Default::default()
    };
    assert!(
        dot_general_computation(vec![2, 3], vec![3, 4], vec![2, 4], dimensions.clone()).is_ok()
    );
    assert!(matches!(
        dot_general_computation(vec![2, 3], vec![3, 4], vec![2, 5], dimensions),
        Err(BuildError::InvalidResultType {
            operation: "dot_general",
            ..
        })
    ));
}

#[test]
fn dot_general_rejects_invalid_and_incompatible_dimensions() {
    let duplicate = stablehlo::DotDimensionNumbers {
        lhs_contracting_dimensions: vec![0, 0],
        rhs_contracting_dimensions: vec![0, 1],
        ..Default::default()
    };
    assert!(matches!(
        dot_general_computation(vec![2, 3], vec![2, 3], vec![], duplicate),
        Err(BuildError::InvalidDimensionNumbers {
            operation: "dot_general"
        })
    ));

    let mismatch = stablehlo::DotDimensionNumbers {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        ..Default::default()
    };
    assert!(matches!(
        dot_general_computation(vec![2, 3], vec![4, 5], vec![2, 5], mismatch),
        Err(BuildError::IncompatibleOperands {
            operation: "dot_general",
            ..
        })
    ));
}

#[test]
fn dot_general_algorithm_requires_default_precision_config() {
    let dimensions = stablehlo::DotDimensionNumbers {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        ..Default::default()
    };
    let result = dot_general_computation_with_precision(
        vec![2, 3],
        vec![3, 4],
        vec![2, 4],
        dimensions,
        Some(vec![stablehlo::Precision::High]),
        true,
    );
    assert!(
        matches!(
            result,
            Err(BuildError::InvalidDotAlgorithmPrecision {
                operation: "dot_general"
            })
        ),
        "unexpected verifier result: {result:?}"
    );
}

#[test]
fn dot_general_checks_per_tensor_and_per_axis_quantization_constraints() {
    let dimensions = stablehlo::DotDimensionNumbers {
        lhs_contracting_dimensions: vec![1],
        rhs_contracting_dimensions: vec![0],
        ..Default::default()
    };
    let lhs = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.5],
        vec![0],
    ));
    let rhs = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(3), Some(4)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.25],
        vec![0],
    ));
    let non_quantized_lhs = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let non_quantized_result = Type::from(TensorType::static_shape(ElementType::F32, [2, 4]));
    let rhs_with_nonzero_zero_point = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(3), Some(4)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.25],
        vec![1],
    ));
    assert!(matches!(
        dot_general_computation_with_types(
            non_quantized_lhs,
            rhs_with_nonzero_zero_point,
            non_quantized_result,
            dimensions.clone(),
            None,
            false,
        ),
        Err(BuildError::InvalidDotQuantization {
            operation: "dot_general"
        })
    ));

    let result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(2), Some(4)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.125],
        vec![0],
    ));
    assert!(
        dot_general_computation_with_types(
            lhs.clone(),
            rhs,
            result,
            dimensions.clone(),
            None,
            false,
        )
        .is_ok()
    );

    let rhs_per_axis = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(3), Some(4)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.25; 3],
        vec![0; 3],
    ));
    let result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(2), Some(4)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(1),
        vec![0.125; 4],
        vec![0; 4],
    ));
    assert!(matches!(
        dot_general_computation_with_types(lhs, rhs_per_axis, result, dimensions, None, false,),
        Err(BuildError::InvalidDotQuantization {
            operation: "dot_general"
        })
    ));
}

fn shape_transform_computation(
    opcode: StableHloOp,
    attribute_name: Option<&str>,
    attribute_values: Option<Vec<i64>>,
    operand_type: Type,
    result_type: Type,
) -> Result<Computation, BuildError> {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        operand_type,
    );
    let attributes = attribute_name
        .zip(attribute_values)
        .map(|(name, values)| [(name.to_owned(), stablehlo::Attribute::integers(values))].into())
        .unwrap_or_default();
    let transform = Instruction::new(
        Operation::stablehlo(opcode, attributes),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        result_type,
    );
    Computation::try_new("shape_transform", vec![operand, transform], ValueId(1))
}

fn dynamic_broadcast_computation(
    operand_type: Type,
    dimensions_type: Type,
    result_type: Type,
    broadcast_dimensions: Vec<i64>,
    known_expanding_dimensions: Option<Vec<i64>>,
    known_nonexpanding_dimensions: Option<Vec<i64>>,
) -> Result<Computation, BuildError> {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        operand_type,
    );
    let dimensions = Instruction::new(
        Operation::parameter(1, "output_dimensions"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        dimensions_type,
    );
    let mut attributes = std::collections::BTreeMap::from([(
        "broadcast_dimensions".to_owned(),
        stablehlo::Attribute::integers(broadcast_dimensions),
    )]);
    if let Some(values) = known_expanding_dimensions {
        attributes.insert(
            "known_expanding_dimensions".to_owned(),
            stablehlo::Attribute::integers(values),
        );
    }
    if let Some(values) = known_nonexpanding_dimensions {
        attributes.insert(
            "known_nonexpanding_dimensions".to_owned(),
            stablehlo::Attribute::integers(values),
        );
    }
    let broadcast = Instruction::new(
        Operation::stablehlo(StableHloOp::DynamicBroadcastInDim, attributes),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        result_type,
    );
    Computation::try_new(
        "dynamic_broadcast",
        vec![operand, dimensions, broadcast],
        ValueId(2),
    )
}

fn dynamic_pad_computation(
    operand_type: Type,
    padding_type: Type,
    padding_vector_types: [Type; 3],
    result_type: Type,
) -> Result<Computation, BuildError> {
    let mut instructions = vec![
        Instruction::new(
            Operation::parameter(0, "operand"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            operand_type,
        ),
        Instruction::new(
            Operation::parameter(1, "padding_value"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            padding_type,
        ),
    ];
    for (index, ty) in padding_vector_types.into_iter().enumerate() {
        instructions.push(Instruction::new(
            Operation::parameter(index as u32 + 2, "padding_vector"),
            Vec::new(),
            Vec::new(),
            ValueId(index as u32 + 2),
            ty,
        ));
    }
    instructions.push(Instruction::new(
        Operation::stablehlo(StableHloOp::DynamicPad, Default::default()),
        vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3), ValueId(4)],
        Vec::new(),
        ValueId(5),
        result_type,
    ));
    Computation::try_new("dynamic_pad", instructions, ValueId(5))
}

fn dynamic_reshape_computation(
    operand_type: Type,
    output_shape_type: Type,
    result_type: Type,
) -> Result<Computation, BuildError> {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        operand_type,
    );
    let output_shape = Instruction::new(
        Operation::parameter(1, "output_shape"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        output_shape_type,
    );
    let reshape = Instruction::new(
        Operation::stablehlo(StableHloOp::DynamicReshape, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        result_type,
    );
    Computation::try_new(
        "dynamic_reshape",
        vec![operand, output_shape, reshape],
        ValueId(2),
    )
}

#[test]
fn reshape_checks_element_count_for_static_shapes() {
    let operand = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let matching = Type::from(TensorType::static_shape(ElementType::F32, [6]));
    let mismatched = Type::from(TensorType::static_shape(ElementType::F32, [5]));
    assert!(
        shape_transform_computation(StableHloOp::Reshape, None, None, operand.clone(), matching)
            .is_ok()
    );
    assert!(matches!(
        shape_transform_computation(StableHloOp::Reshape, None, None, operand, mismatched),
        Err(BuildError::InvalidResultType {
            operation: "reshape",
            ..
        })
    ));
}

#[test]
fn dynamic_reshape_checks_output_shape_contract_and_element_count() {
    let input = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let output_shape = Type::from(TensorType::static_shape(ElementType::I64, [2]));
    assert!(
        dynamic_reshape_computation(
            input.clone(),
            output_shape.clone(),
            Type::from(TensorType::static_shape(ElementType::F32, [3, 2])),
        )
        .is_ok()
    );
    assert!(matches!(
        dynamic_reshape_computation(
            input.clone(),
            output_shape.clone(),
            Type::from(TensorType::static_shape(ElementType::F32, [4, 2])),
        ),
        Err(BuildError::InvalidResultType {
            operation: "dynamic_reshape",
            ..
        })
    ));
    assert!(
        dynamic_reshape_computation(
            input,
            Type::from(TensorType::static_shape(ElementType::F64, [2])),
            Type::from(TensorType::static_shape(ElementType::F32, [3, 2])),
        )
        .is_err()
    );
}

#[test]
fn transpose_checks_permutation_and_inferred_shape() {
    let operand = Type::from(TensorType::static_shape(ElementType::F32, [2, 3, 4]));
    let expected = Type::from(TensorType::static_shape(ElementType::F32, [4, 2, 3]));
    assert!(
        shape_transform_computation(
            StableHloOp::Transpose,
            Some("permutation"),
            Some(vec![2, 0, 1]),
            operand.clone(),
            expected,
        )
        .is_ok()
    );
    for invalid_permutation in [vec![0, 0, 2], vec![0, 1], vec![0, 1, 3]] {
        assert!(matches!(
            shape_transform_computation(
                StableHloOp::Transpose,
                Some("permutation"),
                Some(invalid_permutation),
                operand.clone(),
                Type::from(TensorType::static_shape(ElementType::F32, [4, 2, 3])),
            ),
            Err(BuildError::InvalidDimensionNumbers {
                operation: "transpose"
            })
        ));
    }
    assert!(matches!(
        shape_transform_computation(
            StableHloOp::Transpose,
            Some("permutation"),
            Some(vec![2, 0, 1]),
            operand,
            Type::from(TensorType::static_shape(ElementType::F32, [3, 2, 4])),
        ),
        Err(BuildError::InvalidResultType {
            operation: "transpose",
            ..
        })
    ));
}

#[test]
fn broadcast_in_dim_checks_mapping_and_dimension_sizes() {
    let operand = Type::from(TensorType::static_shape(ElementType::F32, [1, 3]));
    let result = Type::from(TensorType::static_shape(ElementType::F32, [2, 3, 4]));
    assert!(
        shape_transform_computation(
            StableHloOp::BroadcastInDim,
            Some("broadcast_dimensions"),
            Some(vec![0, 1]),
            operand.clone(),
            result.clone(),
        )
        .is_ok()
    );
    for invalid_dimensions in [vec![0, 0], vec![0], vec![0, 3]] {
        assert!(
            shape_transform_computation(
                StableHloOp::BroadcastInDim,
                Some("broadcast_dimensions"),
                Some(invalid_dimensions),
                operand.clone(),
                result.clone(),
            )
            .is_err()
        );
    }
    assert!(matches!(
        shape_transform_computation(
            StableHloOp::BroadcastInDim,
            Some("broadcast_dimensions"),
            Some(vec![0, 1]),
            Type::from(TensorType::static_shape(ElementType::F32, [2, 4])),
            result,
        ),
        Err(BuildError::InvalidResultType {
            operation: "broadcast_in_dim",
            ..
        })
    ));
}

#[test]
fn dynamic_broadcast_in_dim_checks_dimensions_and_expansion_hints() {
    let operand = Type::from(TensorType::static_shape(ElementType::F32, [1, 3]));
    let dimensions = Type::from(TensorType::static_shape(ElementType::I64, [3]));
    let result = Type::from(TensorType::static_shape(ElementType::F32, [2, 3, 4]));
    assert!(
        dynamic_broadcast_computation(
            operand.clone(),
            dimensions.clone(),
            result.clone(),
            vec![0, 1],
            Some(vec![0]),
            Some(vec![1]),
        )
        .is_ok()
    );
    assert!(matches!(
        dynamic_broadcast_computation(
            operand.clone(),
            Type::from(TensorType::static_shape(ElementType::I64, [2])),
            result.clone(),
            vec![0, 1],
            None,
            None,
        ),
        Err(BuildError::InvalidDimensionNumbers {
            operation: "dynamic_broadcast_in_dim"
        })
    ));
    assert!(matches!(
        dynamic_broadcast_computation(
            operand,
            dimensions,
            result,
            vec![0, 1],
            Some(vec![0]),
            Some(vec![0]),
        ),
        Err(BuildError::InvalidDimensionNumbers {
            operation: "dynamic_broadcast_in_dim"
        })
    ));
}

#[test]
fn dynamic_pad_checks_padding_vector_types_and_ranks() {
    let operand = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let scalar = Type::from(TensorType::scalar(ElementType::F32));
    let vector = Type::from(TensorType::static_shape(ElementType::I64, [2]));
    let result = Type::from(TensorType::static_shape(ElementType::F32, [5, 9]));
    assert!(
        dynamic_pad_computation(
            operand.clone(),
            scalar.clone(),
            [vector.clone(), vector.clone(), vector.clone()],
            result.clone(),
        )
        .is_ok()
    );
    assert!(
        dynamic_pad_computation(
            operand.clone(),
            scalar.clone(),
            [
                vector.clone(),
                Type::from(TensorType::static_shape(ElementType::I32, [2])),
                vector.clone(),
            ],
            result.clone(),
        )
        .is_err()
    );
    assert!(
        dynamic_pad_computation(
            operand,
            scalar,
            [
                vector.clone(),
                vector.clone(),
                Type::from(TensorType::static_shape(ElementType::I64, [1])),
            ],
            result,
        )
        .is_err()
    );
}

#[test]
fn shape_transforms_preserve_per_axis_quantization_constraints() {
    let transpose_input = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.5, 0.25],
        vec![0, 0],
    ));
    let transpose_result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(3), Some(2)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(1),
        vec![0.5, 0.25],
        vec![0, 0],
    ));
    assert!(
        shape_transform_computation(
            StableHloOp::Transpose,
            Some("permutation"),
            Some(vec![1, 0]),
            transpose_input.clone(),
            transpose_result,
        )
        .is_ok()
    );
    let wrong_transpose_result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(3), Some(2)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.5, 0.25, 0.125],
        vec![0, 0, 0],
    ));
    assert!(matches!(
        shape_transform_computation(
            StableHloOp::Transpose,
            Some("permutation"),
            Some(vec![1, 0]),
            transpose_input,
            wrong_transpose_result,
        ),
        Err(BuildError::InvalidQuantizedTransform {
            operation: "transpose"
        })
    ));

    let broadcast_input = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(1), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.5],
        vec![0],
    ));
    let broadcast_result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(4), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.5; 4],
        vec![0; 4],
    ));
    assert!(
        shape_transform_computation(
            StableHloOp::BroadcastInDim,
            Some("broadcast_dimensions"),
            Some(vec![0, 1]),
            broadcast_input.clone(),
            broadcast_result,
        )
        .is_ok()
    );
    let bad_broadcast_result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(4), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.5, 0.25, 0.5, 0.5],
        vec![0; 4],
    ));
    assert!(matches!(
        shape_transform_computation(
            StableHloOp::BroadcastInDim,
            Some("broadcast_dimensions"),
            Some(vec![0, 1]),
            broadcast_input,
            bad_broadcast_result,
        ),
        Err(BuildError::InvalidQuantizedTransform {
            operation: "broadcast_in_dim"
        })
    ));

    let reshape_input = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(1),
        vec![0.5; 3],
        vec![0; 3],
    ));
    let reshape_result = Type::QuantizedTensor(stablehlo::QuantizedTensorType::new(
        vec![Some(3), Some(2)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(0),
        vec![0.5; 3],
        vec![0; 3],
    ));
    assert!(matches!(
        shape_transform_computation(
            StableHloOp::Reshape,
            None,
            None,
            reshape_input,
            reshape_result,
        ),
        Err(BuildError::InvalidQuantizedTransform {
            operation: "reshape"
        })
    ));
}

#[test]
fn reverse_validates_unique_in_range_dimensions_and_result_shape() {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
    );
    let build = |dimensions: Vec<i64>, result_type: Type| {
        let reverse = Instruction::new(
            Operation::stablehlo(
                StableHloOp::Reverse,
                [(
                    "dimensions".to_owned(),
                    stablehlo::Attribute::integers(dimensions),
                )]
                .into(),
            ),
            vec![ValueId(0)],
            Vec::new(),
            ValueId(1),
            result_type,
        );
        Computation::try_new("reverse", vec![operand.clone(), reverse], ValueId(1))
    };

    assert!(
        build(
            vec![1],
            Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
        )
        .is_ok()
    );
    for dimensions in [vec![1, 1], vec![-1], vec![2]] {
        assert!(matches!(
            build(
                dimensions,
                Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
            ),
            Err(BuildError::InvalidDimensionNumbers {
                operation: "reverse"
            })
        ));
    }
    assert!(matches!(
        build(
            vec![1],
            Type::from(TensorType::static_shape(ElementType::I32, [3, 2])),
        ),
        Err(BuildError::InvalidResultType {
            operation: "reverse",
            ..
        })
    ));
}

#[test]
fn slice_validates_bounds_strides_and_inferred_result_shape() {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::I32, [3, 8])),
    );
    let build = |starts: Vec<i64>, limits: Vec<i64>, strides: Vec<i64>, result_type: Type| {
        let attributes = [
            (
                "start_indices".to_owned(),
                stablehlo::Attribute::integers(starts),
            ),
            (
                "limit_indices".to_owned(),
                stablehlo::Attribute::integers(limits),
            ),
            (
                "strides".to_owned(),
                stablehlo::Attribute::integers(strides),
            ),
        ]
        .into();
        let slice = Instruction::new(
            Operation::stablehlo(StableHloOp::Slice, attributes),
            vec![ValueId(0)],
            Vec::new(),
            ValueId(1),
            result_type,
        );
        Computation::try_new("slice", vec![operand.clone(), slice], ValueId(1))
    };

    let valid_result = Type::from(TensorType::static_shape(ElementType::I32, [2, 2]));
    assert!(build(vec![1, 4], vec![3, 8], vec![1, 2], valid_result.clone()).is_ok());
    for (starts, limits, strides) in [
        (vec![-1, 4], vec![3, 8], vec![1, 2]),
        (vec![1, 4], vec![4, 8], vec![1, 2]),
        (vec![1, 4], vec![3, 8], vec![1, 0]),
        (vec![2, 4], vec![1, 8], vec![1, 2]),
    ] {
        assert!(matches!(
            build(starts, limits, strides, valid_result.clone()),
            Err(BuildError::TypeInference {
                operation: "slice",
                ..
            })
        ));
    }
    assert!(matches!(
        build(
            vec![1, 4],
            vec![3, 8],
            vec![1, 2],
            Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
        ),
        Err(BuildError::InvalidResultType {
            operation: "slice",
            ..
        })
    ));
}

#[test]
fn clamp_validates_shapes_element_types_and_result() {
    let scalar = Type::from(TensorType::scalar(ElementType::F32));
    let tensor = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let build = |min_type: Type, max_type: Type, result_type: Type| {
        let min = Instruction::new(
            Operation::parameter(0, "min"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            min_type,
        );
        let operand = Instruction::new(
            Operation::parameter(1, "operand"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            tensor.clone(),
        );
        let max = Instruction::new(
            Operation::parameter(2, "max"),
            Vec::new(),
            Vec::new(),
            ValueId(2),
            max_type,
        );
        let clamp = Instruction::new(
            Operation::stablehlo(StableHloOp::Clamp, Default::default()),
            vec![ValueId(0), ValueId(1), ValueId(2)],
            Vec::new(),
            ValueId(3),
            result_type,
        );
        Computation::try_new("clamp", vec![min, operand, max, clamp], ValueId(3))
    };

    assert!(build(scalar.clone(), tensor.clone(), tensor.clone()).is_ok());
    assert!(matches!(
        build(
            Type::from(TensorType::static_shape(ElementType::F32, [2])),
            scalar.clone(),
            tensor.clone(),
        ),
        Err(BuildError::IncompatibleOperands {
            operation: "clamp",
            ..
        })
    ));
    assert!(matches!(
        build(
            scalar.clone(),
            Type::from(TensorType::scalar(ElementType::I32)),
            tensor.clone(),
        ),
        Err(BuildError::IncompatibleOperands {
            operation: "clamp",
            ..
        })
    ));
    assert!(matches!(
        build(
            scalar,
            tensor.clone(),
            Type::from(TensorType::static_shape(ElementType::F32, [3, 2])),
        ),
        Err(BuildError::InvalidResultType {
            operation: "clamp",
            ..
        })
    ));
}

#[test]
fn select_checks_predicate_broadcast_and_value_shapes() {
    let build = |predicate_type: Type, true_type: Type, false_type: Type, result_type: Type| {
        let predicate = Instruction::new(
            Operation::parameter(0, "predicate"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            predicate_type,
        );
        let on_true = Instruction::new(
            Operation::parameter(1, "on_true"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            true_type,
        );
        let on_false = Instruction::new(
            Operation::parameter(2, "on_false"),
            Vec::new(),
            Vec::new(),
            ValueId(2),
            false_type,
        );
        let select = Instruction::new(
            Operation::stablehlo(StableHloOp::Select, Default::default()),
            vec![ValueId(0), ValueId(1), ValueId(2)],
            Vec::new(),
            ValueId(3),
            result_type,
        );
        Computation::try_new(
            "select",
            vec![predicate, on_true, on_false, select],
            ValueId(3),
        )
    };
    let predicate_scalar = Type::from(TensorType::scalar(ElementType::I1));
    let predicate_tensor = Type::from(TensorType::static_shape(ElementType::I1, [2, 3]));
    let values = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    assert!(
        build(
            predicate_scalar.clone(),
            values.clone(),
            values.clone(),
            values.clone(),
        )
        .is_ok()
    );
    assert!(
        build(
            predicate_tensor.clone(),
            values.clone(),
            values.clone(),
            values.clone(),
        )
        .is_ok()
    );
    assert!(matches!(
        build(
            Type::from(TensorType::static_shape(ElementType::I1, [3])),
            values.clone(),
            values.clone(),
            values.clone(),
        ),
        Err(BuildError::IncompatibleOperands {
            operation: "select",
            ..
        })
    ));
    assert!(matches!(
        build(
            predicate_scalar,
            values.clone(),
            Type::from(TensorType::static_shape(ElementType::F32, [2, 4])),
            values.clone(),
        ),
        Err(BuildError::IncompatibleOperands {
            operation: "select",
            ..
        })
    ));
    assert!(matches!(
        build(
            predicate_tensor,
            values.clone(),
            values,
            Type::from(TensorType::static_shape(ElementType::F32, [3, 2])),
        ),
        Err(BuildError::InvalidResultType {
            operation: "select",
            ..
        })
    ));
}

#[test]
fn concatenate_infers_axis_size_and_checks_other_dimensions() {
    let build = |lhs_shape: Vec<u64>, rhs_shape: Vec<u64>, axis: i64, result_shape: Vec<u64>| {
        let lhs = Instruction::new(
            Operation::parameter(0, "lhs"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            Type::from(TensorType::static_shape(ElementType::I32, lhs_shape)),
        );
        let rhs = Instruction::new(
            Operation::parameter(1, "rhs"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            Type::from(TensorType::static_shape(ElementType::I32, rhs_shape)),
        );
        let concatenate = Instruction::new(
            Operation::stablehlo(
                StableHloOp::Concatenate,
                [("dimension".to_owned(), stablehlo::Attribute::integer(axis))].into(),
            ),
            vec![ValueId(0), ValueId(1)],
            Vec::new(),
            ValueId(2),
            Type::from(TensorType::static_shape(ElementType::I32, result_shape)),
        );
        Computation::try_new("concatenate", vec![lhs, rhs, concatenate], ValueId(2))
    };
    assert!(build(vec![2, 3], vec![4, 3], 0, vec![6, 3]).is_ok());
    assert!(matches!(
        build(vec![2, 3], vec![4, 3], 0, vec![5, 3]),
        Err(BuildError::InvalidResultType {
            operation: "concatenate",
            ..
        })
    ));
    assert!(matches!(
        build(vec![2, 3], vec![4, 4], 0, vec![6, 3]),
        Err(BuildError::TypeInference {
            operation: "concatenate",
            error: stablehlo::InferenceError::DimensionMismatch { dimension: 1, .. }
        })
    ));
    assert!(matches!(
        build(vec![2, 3], vec![4, 3], -1, vec![2, 6]),
        Err(BuildError::InvalidDimensionNumbers {
            operation: "concatenate"
        })
    ));
}

#[test]
fn pad_checks_padding_vectors_and_inferred_shape() {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
    );
    let padding_value = Instruction::new(
        Operation::parameter(1, "padding"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::scalar(ElementType::I32)),
    );
    let build = |low: Vec<i64>, high: Vec<i64>, interior: Vec<i64>, result_type: Type| {
        let attributes = [
            (
                "edge_padding_low".to_owned(),
                stablehlo::Attribute::integers(low),
            ),
            (
                "edge_padding_high".to_owned(),
                stablehlo::Attribute::integers(high),
            ),
            (
                "interior_padding".to_owned(),
                stablehlo::Attribute::integers(interior),
            ),
        ]
        .into();
        let pad = Instruction::new(
            Operation::stablehlo(StableHloOp::Pad, attributes),
            vec![ValueId(0), ValueId(1)],
            Vec::new(),
            ValueId(2),
            result_type,
        );
        Computation::try_new(
            "pad",
            vec![operand.clone(), padding_value.clone(), pad],
            ValueId(2),
        )
    };

    assert!(
        build(
            vec![0, 1],
            vec![2, 1],
            vec![1, 2],
            Type::from(TensorType::static_shape(ElementType::I32, [5, 9])),
        )
        .is_ok()
    );
    for (low, high, interior) in [
        (vec![0], vec![0, 0], vec![0, 0]),
        (vec![0, 0], vec![0], vec![0, 0]),
        (vec![0, 0], vec![0, 0], vec![0]),
        (vec![0, 0], vec![0, 0], vec![-1, 0]),
    ] {
        assert!(matches!(
            build(
                low,
                high,
                interior,
                Type::from(TensorType::static_shape(ElementType::I32, [2, 3])),
            ),
            Err(BuildError::TypeInference {
                operation: "pad",
                ..
            })
        ));
    }
    assert!(matches!(
        build(
            vec![-3, 0],
            vec![0, 0],
            vec![0, 0],
            Type::from(TensorType::static_shape(ElementType::I32, [0, 3])),
        ),
        Err(BuildError::TypeInference {
            operation: "pad",
            error: stablehlo::InferenceError::NegativePaddingResult { dimension: 0 },
        })
    ));
}

#[test]
fn dynamic_slice_checks_scalar_indices_sizes_and_result_shape() {
    let build = |index_types: [Type; 2], sizes: Vec<i64>, result_type: Type| {
        let operand = Instruction::new(
            Operation::parameter(0, "operand"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            Type::from(TensorType::static_shape(ElementType::F32, [4, 4])),
        );
        let first_index = Instruction::new(
            Operation::parameter(1, "index0"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            index_types[0].clone(),
        );
        let second_index = Instruction::new(
            Operation::parameter(2, "index1"),
            Vec::new(),
            Vec::new(),
            ValueId(2),
            index_types[1].clone(),
        );
        let slice = Instruction::new(
            Operation::stablehlo(
                StableHloOp::DynamicSlice,
                [(
                    "slice_sizes".to_owned(),
                    stablehlo::Attribute::integers(sizes),
                )]
                .into(),
            ),
            vec![ValueId(0), ValueId(1), ValueId(2)],
            Vec::new(),
            ValueId(3),
            result_type,
        );
        Computation::try_new(
            "dynamic_slice",
            vec![operand, first_index, second_index, slice],
            ValueId(3),
        )
    };
    let index = Type::from(TensorType::scalar(ElementType::I64));
    let result = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    assert!(build([index.clone(), index.clone()], vec![2, 3], result.clone()).is_ok());
    assert!(matches!(
        build(
            [index, Type::from(TensorType::scalar(ElementType::I32))],
            vec![2, 3],
            result.clone(),
        ),
        Err(BuildError::IncompatibleOperands {
            operation: "dynamic_slice",
            ..
        })
    ));
    assert!(matches!(
        build(
            [
                Type::from(TensorType::scalar(ElementType::I64)),
                Type::from(TensorType::scalar(ElementType::I64)),
            ],
            vec![5, 3],
            result,
        ),
        Err(BuildError::TypeInference {
            operation: "dynamic_slice",
            error: stablehlo::InferenceError::SliceSizeExceeded {
                dimension: 0,
                size: 5,
                size_or_bound: 4,
            },
        })
    ));
}

#[test]
fn dynamic_update_slice_checks_update_bounds_and_result_type() {
    let build = |update_shape: [u64; 2], result_shape: [u64; 2]| {
        let operand = Instruction::new(
            Operation::parameter(0, "operand"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            Type::from(TensorType::static_shape(ElementType::F32, [4, 4])),
        );
        let update = Instruction::new(
            Operation::parameter(1, "update"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            Type::from(TensorType::static_shape(ElementType::F32, update_shape)),
        );
        let index0 = Instruction::new(
            Operation::parameter(2, "index0"),
            Vec::new(),
            Vec::new(),
            ValueId(2),
            Type::from(TensorType::scalar(ElementType::I32)),
        );
        let index1 = Instruction::new(
            Operation::parameter(3, "index1"),
            Vec::new(),
            Vec::new(),
            ValueId(3),
            Type::from(TensorType::scalar(ElementType::I32)),
        );
        let update_slice = Instruction::new(
            Operation::stablehlo(StableHloOp::DynamicUpdateSlice, Default::default()),
            vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
            Vec::new(),
            ValueId(4),
            Type::from(TensorType::static_shape(ElementType::F32, result_shape)),
        );
        Computation::try_new(
            "dynamic_update_slice",
            vec![operand, update, index0, index1, update_slice],
            ValueId(4),
        )
    };
    assert!(build([2, 3], [4, 4]).is_ok());
    assert!(matches!(
        build([5, 3], [4, 4]),
        Err(BuildError::IncompatibleOperands {
            operation: "dynamic_update_slice",
            ..
        })
    ));
    assert!(matches!(
        build([2, 3], [4, 3]),
        Err(BuildError::InvalidResultType {
            operation: "dynamic_update_slice",
            ..
        })
    ));
}

#[test]
fn get_and_set_dimension_size_validate_dimension_and_scalar_contracts() {
    let operand = Instruction::new(
        Operation::parameter(0, "operand"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::F32, [4, 2])),
    );
    let build_get = |dimension: i64, result_type: Type| {
        let get_size = Instruction::new(
            Operation::stablehlo(
                StableHloOp::GetDimensionSize,
                [(
                    "dimension".to_owned(),
                    stablehlo::Attribute::integer(dimension),
                )]
                .into(),
            ),
            vec![ValueId(0)],
            Vec::new(),
            ValueId(1),
            result_type,
        );
        Computation::try_new(
            "get_dimension_size",
            vec![operand.clone(), get_size],
            ValueId(1),
        )
    };
    assert!(build_get(1, Type::from(TensorType::scalar(ElementType::I32)),).is_ok());
    assert!(matches!(
        build_get(2, Type::from(TensorType::scalar(ElementType::I32))),
        Err(BuildError::InvalidDimensionNumbers {
            operation: "get_dimension_size"
        })
    ));

    let build_set = |dimension: i64, size_type: Type| {
        let size = Instruction::new(
            Operation::parameter(1, "size"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            size_type,
        );
        let set_size = Instruction::new(
            Operation::stablehlo(
                StableHloOp::SetDimensionSize,
                [(
                    "dimension".to_owned(),
                    stablehlo::Attribute::integer(dimension),
                )]
                .into(),
            ),
            vec![ValueId(0), ValueId(1)],
            Vec::new(),
            ValueId(2),
            Type::from(TensorType::static_shape(ElementType::F32, [4, 2])),
        );
        Computation::try_new(
            "set_dimension_size",
            vec![operand.clone(), size, set_size],
            ValueId(2),
        )
    };
    assert!(build_set(1, Type::from(TensorType::scalar(ElementType::I32))).is_ok());
    assert!(matches!(
        build_set(2, Type::from(TensorType::scalar(ElementType::I32))),
        Err(BuildError::InvalidDimensionNumbers {
            operation: "set_dimension_size"
        })
    ));
    assert!(matches!(
        build_set(
            1,
            Type::from(TensorType::static_shape(ElementType::I32, [1])),
        ),
        Err(BuildError::InvalidOperandType {
            operation: "set_dimension_size",
            ..
        })
    ));
}

#[test]
fn unary_complex_and_predicate_result_types_follow_the_spec() {
    let input_type = Type::from(TensorType::static_shape(ElementType::C64, [2]));
    let input = Instruction::new(
        Operation::parameter(0, "complex"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        input_type,
    );
    let absolute = Instruction::new(
        Operation::stablehlo(StableHloOp::Abs, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::static_shape(ElementType::F32, [2])),
    );
    assert!(Computation::try_new("complex_abs", vec![input, absolute], ValueId(1)).is_ok());

    let float_input = Instruction::new(
        Operation::parameter(0, "float"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::F32, [2])),
    );
    let finite = Instruction::new(
        Operation::stablehlo(StableHloOp::IsFinite, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::static_shape(ElementType::I1, [2])),
    );
    assert!(Computation::try_new("is_finite", vec![float_input, finite], ValueId(1)).is_ok());

    let real = Instruction::new(
        Operation::stablehlo(StableHloOp::Real, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::static_shape(ElementType::C64, [2])),
    );
    assert!(matches!(
        Computation::try_new(
            "invalid_real_result",
            vec![
                Instruction::new(
                    Operation::parameter(0, "complex"),
                    Vec::new(),
                    Vec::new(),
                    ValueId(0),
                    Type::from(TensorType::static_shape(ElementType::C64, [2])),
                ),
                real,
            ],
            ValueId(1),
        ),
        Err(BuildError::InvalidResultType {
            operation: "real",
            ..
        })
    ));
}

#[test]
fn elementwise_types_accept_compatible_dynamic_dimensions_only() {
    let dynamic = Type::from(TensorType::new(ElementType::F32, vec![None, Some(4)]));
    let static_shape = Type::from(TensorType::static_shape(ElementType::F32, [3, 4]));
    let result_type = dynamic.clone();
    let lhs = Instruction::new(
        Operation::parameter(0, "lhs"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        dynamic,
    );
    let rhs = Instruction::new(
        Operation::parameter(1, "rhs"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        static_shape.clone(),
    );
    let add = Instruction::new(
        Operation::stablehlo(StableHloOp::Add, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        result_type,
    );
    assert!(Computation::try_new("dynamic_add", vec![lhs, rhs, add], ValueId(2)).is_ok());

    let lhs = Instruction::new(
        Operation::parameter(0, "lhs"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::F32, [3, 4])),
    );
    let rhs = Instruction::new(
        Operation::parameter(1, "rhs"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        static_shape,
    );
    let add = Instruction::new(
        Operation::stablehlo(StableHloOp::Add, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::from(TensorType::static_shape(ElementType::F32, [3, 4])),
    );
    assert!(Computation::try_new("static_add", vec![lhs, rhs, add], ValueId(2)).is_ok());

    let incompatible_rhs = Instruction::new(
        Operation::parameter(1, "rhs"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        Type::from(TensorType::static_shape(ElementType::F32, [5, 4])),
    );
    let lhs = Instruction::new(
        Operation::parameter(0, "lhs"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::static_shape(ElementType::F32, [3, 4])),
    );
    let add = Instruction::new(
        Operation::stablehlo(StableHloOp::Add, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        Type::from(TensorType::static_shape(ElementType::F32, [3, 4])),
    );
    assert!(matches!(
        Computation::try_new(
            "incompatible_add",
            vec![lhs, incompatible_rhs, add],
            ValueId(2)
        ),
        Err(BuildError::IncompatibleOperands {
            operation: "add",
            ..
        })
    ));
}

#[test]
fn token_operations_preserve_the_token_type() {
    let token = Instruction::new(
        Operation::stablehlo(StableHloOp::CreateToken, Default::default()),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::Token,
    );
    let after_all = Instruction::new(
        Operation::stablehlo(StableHloOp::AfterAll, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        Type::Token,
    );
    assert!(Computation::try_new("tokens", vec![token, after_all], ValueId(1)).is_ok());
}

#[test]
fn optimization_barrier_preserves_variadic_operand_result_pairs() {
    let f32_ty = Type::from(TensorType::scalar(ElementType::F32));
    let i32_ty = Type::from(TensorType::scalar(ElementType::I32));
    let first = Instruction::new(
        Operation::parameter(0, "first"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        f32_ty.clone(),
    );
    let second = Instruction::new(
        Operation::parameter(1, "second"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        i32_ty.clone(),
    );
    let barrier = Instruction::with_results(
        Operation::stablehlo(StableHloOp::OptimizationBarrier, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        vec![
            OperationResult::new(ValueId(2), f32_ty.clone()),
            OperationResult::new(ValueId(3), i32_ty),
        ],
    );
    assert!(
        Computation::try_new(
            "barrier",
            vec![first.clone(), second.clone(), barrier],
            ValueId(3)
        )
        .is_ok()
    );

    let malformed = Instruction::with_results(
        Operation::stablehlo(StableHloOp::OptimizationBarrier, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        vec![OperationResult::new(ValueId(2), f32_ty)],
    );
    assert_eq!(
        Computation::try_new("bad_barrier", vec![first, second, malformed], ValueId(0)),
        Err(BuildError::MismatchedOperandResultCount {
            operation: "optimization_barrier",
            operands: 2,
            results: 1,
        })
    );
}

#[test]
fn verifier_checks_variadic_operand_and_result_groups() {
    let malformed_all_gather = Instruction::with_results(
        Operation::stablehlo(StableHloOp::AllGather, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        vec![OperationResult::new(
            ValueId(2),
            Type::from(TensorType::scalar(ElementType::F32)),
        )],
    );
    assert_eq!(
        Computation::try_new("bad_all_gather", vec![malformed_all_gather], ValueId(2)),
        Err(BuildError::InvalidVariadicCardinality {
            operation: "all_gather",
            operands: 2,
            results: 1,
        })
    );

    let malformed_reduce = Instruction::with_results(
        Operation::stablehlo(StableHloOp::Reduce, Default::default()),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        Vec::new(),
        vec![OperationResult::new(
            ValueId(3),
            Type::from(TensorType::scalar(ElementType::F32)),
        )],
    )
    .with_regions(vec![Region::new(vec![Block::new(
        Vec::new(),
        vec![region_return(Vec::new())],
    )])]);
    assert_eq!(
        Computation::try_new("bad_reduce_groups", vec![malformed_reduce], ValueId(3)),
        Err(BuildError::InvalidVariadicCardinality {
            operation: "reduce",
            operands: 3,
            results: 1,
        })
    );
}

#[test]
fn reduction_regions_follow_the_variadic_signature() {
    let tensor_type = Type::from(TensorType::scalar(ElementType::F32));
    let inputs = vec![
        Instruction::new(
            Operation::parameter(0, "input"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            tensor_type.clone(),
        ),
        Instruction::new(
            Operation::parameter(1, "init"),
            Vec::new(),
            Vec::new(),
            ValueId(1),
            tensor_type.clone(),
        ),
    ];
    let reducer = Instruction::new(
        Operation::stablehlo(StableHloOp::Add, Default::default()),
        vec![ValueId(10), ValueId(11)],
        Vec::new(),
        ValueId(12),
        tensor_type.clone(),
    );
    let reduction = Instruction::new(
        Operation::stablehlo(
            StableHloOp::Reduce,
            [(
                "dimensions".to_owned(),
                stablehlo::Attribute::integers(vec![]),
            )]
            .into(),
        ),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        tensor_type.clone(),
    )
    .with_regions(vec![Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(10), tensor_type.clone()),
            BlockArgument::new(ValueId(11), tensor_type.clone()),
        ],
        vec![reducer, region_return(vec![ValueId(12)])],
    )])]);
    let mut valid_instructions = inputs.clone();
    valid_instructions.push(reduction);
    assert!(Computation::try_new("valid_reduce", valid_instructions, ValueId(2)).is_ok());

    let malformed = Instruction::new(
        Operation::stablehlo(StableHloOp::Reduce, Default::default()),
        vec![ValueId(0), ValueId(1)],
        Vec::new(),
        ValueId(2),
        tensor_type,
    )
    .with_regions(vec![Region::new(vec![Block::new(
        Vec::new(),
        vec![region_return(vec![ValueId(0)])],
    )])]);
    let mut invalid_instructions = inputs;
    invalid_instructions.push(malformed);
    assert_eq!(
        Computation::try_new("invalid_reduce", invalid_instructions, ValueId(2)),
        Err(BuildError::InvalidRegionSignature {
            operation: "reduce",
            region: 0,
            expected_arguments: 2,
            actual_arguments: 0,
            expected_results: 1,
            actual_results: 1,
        })
    );
}

#[test]
fn computation_accepts_multiple_results_and_zero_result_operations() {
    let tensor = Type::from(TensorType::static_shape(ElementType::F32, [2]));
    let inputs = (0..3)
        .map(|index| {
            Instruction::new(
                Operation::parameter(index, format!("input{index}")),
                Vec::new(),
                Vec::new(),
                ValueId(index),
                tensor.clone(),
            )
        })
        .collect::<Vec<_>>();
    let split = Instruction::with_results(
        Operation::stablehlo(
            StableHloOp::BatchNormTraining,
            [
                ("epsilon".to_owned(), stablehlo::Attribute::f32_bits(0)),
                ("feature_index".to_owned(), stablehlo::Attribute::integer(0)),
            ]
            .into(),
        ),
        vec![ValueId(0), ValueId(1), ValueId(2)],
        Vec::new(),
        vec![
            OperationResult::new(ValueId(3), tensor.clone()),
            OperationResult::new(ValueId(4), tensor.clone()),
            OperationResult::new(ValueId(5), tensor),
        ],
    );
    let mut instructions = inputs;
    instructions.push(split);
    let computation = Computation::try_new("multi_result", instructions, ValueId(5));
    assert!(computation.is_ok());
}

#[test]
fn functions_support_multiple_and_zero_return_values_without_module_entry() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let first = Instruction::new(
        Operation::parameter(0, "first"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let second = Instruction::new(
        Operation::parameter(1, "second"),
        Vec::new(),
        Vec::new(),
        ValueId(1),
        tensor.clone(),
    );
    let function = Computation::try_new_with_roots(
        "two_results",
        vec![first, second],
        vec![ValueId(0), ValueId(1)],
    )
    .unwrap();
    assert_eq!(function.root(), None);
    assert_eq!(
        function.function_type().inputs(),
        &[tensor.clone(), tensor.clone()]
    );
    assert_eq!(
        function.function_type().outputs(),
        &[tensor.clone(), tensor]
    );

    let void_function = Computation::try_new_with_roots(
        "void",
        vec![Instruction::new(
            Operation::stablehlo(StableHloOp::CreateToken, Default::default()),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            Type::Token,
        )],
        Vec::new(),
    )
    .unwrap();
    assert!(void_function.roots().is_empty());

    let mut module = Module::new("program");
    module.add_computation(function);
    module.add_computation(void_function);
    assert_eq!(module.entry(), None);
    assert_eq!(module.verify(), Ok(()));
    assert_eq!(Module::new("empty_program").verify(), Ok(()));
}

#[test]
fn nested_regions_support_block_arguments_and_lexical_captures() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let body_add = Instruction::new(
        Operation::stablehlo(StableHloOp::Add, Default::default()),
        vec![ValueId(0), ValueId(10)],
        Vec::new(),
        ValueId(11),
        tensor.clone(),
    );
    let condition_compare = Instruction::new(
        comparison(),
        vec![ValueId(12), ValueId(12)],
        Vec::new(),
        ValueId(13),
        Type::from(TensorType::scalar(ElementType::I1)),
    );
    let body = Block::new(
        vec![BlockArgument::new(ValueId(10), tensor.clone())],
        vec![body_add, region_return(vec![ValueId(11)])],
    );
    let loop_op = Instruction::with_results(
        Operation::stablehlo(StableHloOp::While, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        vec![OperationResult::new(ValueId(1), tensor.clone())],
    )
    .with_regions(vec![
        Region::new(vec![Block::new(
            vec![BlockArgument::new(ValueId(12), tensor.clone())],
            vec![condition_compare, region_return(vec![ValueId(13)])],
        )]),
        Region::new(vec![body]),
    ]);
    let computation = Computation::try_new("loop", vec![input, loop_op], ValueId(1));
    assert!(computation.is_ok());
}

#[test]
fn nested_region_rejects_values_outside_its_lexical_scope() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let input = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let nested = Instruction::new(
        Operation::stablehlo(StableHloOp::Negate, Default::default()),
        vec![ValueId(99)],
        Vec::new(),
        ValueId(10),
        tensor.clone(),
    );
    let condition_compare = Instruction::new(
        comparison(),
        vec![ValueId(12), ValueId(12)],
        Vec::new(),
        ValueId(13),
        Type::from(TensorType::scalar(ElementType::I1)),
    );
    let loop_op = Instruction::new(
        Operation::stablehlo(StableHloOp::While, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        tensor.clone(),
    )
    .with_regions(vec![
        Region::new(vec![Block::new(
            vec![BlockArgument::new(ValueId(12), tensor.clone())],
            vec![condition_compare, region_return(vec![ValueId(13)])],
        )]),
        Region::new(vec![Block::new(
            vec![BlockArgument::new(ValueId(10), tensor.clone())],
            vec![nested, region_return(vec![ValueId(10)])],
        )]),
    ]);

    assert!(matches!(
        Computation::try_new("invalid_loop", vec![input, loop_op], ValueId(1)),
        Err(BuildError::UnknownOrNonDominatingValue {
            instruction: ValueId(10),
            operand: ValueId(99),
        })
    ));
}

#[test]
fn nested_region_rejects_function_parameter_operations() {
    let nested = Instruction::new(
        Operation::parameter(0, "nested_parameter"),
        Vec::new(),
        Vec::new(),
        ValueId(10),
        Type::from(TensorType::scalar(ElementType::F32)),
    );
    let parent = Instruction::with_results(
        Operation::stablehlo(StableHloOp::While, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        Vec::new(),
    )
    .with_regions(vec![
        Region::new(vec![Block::new(
            Vec::new(),
            vec![nested, region_return(Vec::new())],
        )]),
        Region::new(vec![Block::new(
            Vec::new(),
            vec![region_return(Vec::new())],
        )]),
    ]);
    let root = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::scalar(ElementType::F32)),
    );
    let computation =
        Computation::try_new_with_roots("invalid", vec![root, parent], vec![ValueId(0)]);
    assert_eq!(computation, Err(BuildError::ParameterInsideRegion));
}

#[test]
fn module_checks_called_computations_inside_regions() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let parameter = Instruction::new(
        Operation::parameter(0, "input"),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        tensor.clone(),
    );
    let nested_call = Instruction::new(
        Operation::stablehlo(
            StableHloOp::CustomCall,
            [(
                "call_target_name".to_owned(),
                stablehlo::Attribute::String("test".to_owned()),
            )]
            .into(),
        ),
        vec![ValueId(0)],
        vec![stablehlo::ComputationId(10)],
        ValueId(10),
        tensor.clone(),
    );
    let body = Block::new(
        vec![BlockArgument::new(ValueId(11), tensor.clone())],
        vec![nested_call, region_return(vec![ValueId(10)])],
    );
    let condition_compare = Instruction::new(
        comparison(),
        vec![ValueId(12), ValueId(12)],
        Vec::new(),
        ValueId(13),
        Type::from(TensorType::scalar(ElementType::I1)),
    );
    let parent = Instruction::new(
        Operation::stablehlo(StableHloOp::While, Default::default()),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        tensor.clone(),
    )
    .with_regions(vec![
        Region::new(vec![Block::new(
            vec![BlockArgument::new(ValueId(12), tensor.clone())],
            vec![condition_compare, region_return(vec![ValueId(13)])],
        )]),
        Region::new(vec![body]),
    ]);
    let computation = Computation::try_new("main", vec![parameter, parent], ValueId(1)).unwrap();
    let mut module = Module::new("module");
    let id = module.add_computation(computation);
    module.set_entry(id).unwrap();

    assert_eq!(
        module.verify(),
        Err(BuildError::UnknownComputation(stablehlo::ComputationId(10)))
    );
}

#[test]
fn module_resolves_composite_decomposition_symbols() {
    let tensor = Type::from(TensorType::scalar(ElementType::F32));
    let composite = Instruction::new(
        Operation::stablehlo(
            StableHloOp::Composite,
            [
                (
                    "name".to_owned(),
                    stablehlo::Attribute::String("test.composite".to_owned()),
                ),
                (
                    "decomposition".to_owned(),
                    stablehlo::Attribute::SymbolRef(vec!["decompose".to_owned()]),
                ),
            ]
            .into(),
        ),
        vec![ValueId(0)],
        Vec::new(),
        ValueId(1),
        tensor.clone(),
    );
    let parameter = Instruction::new(
        Operation::parameter(0, "input"),
        vec![],
        vec![],
        ValueId(0),
        tensor.clone(),
    );
    let main = Computation::try_new("main", vec![parameter, composite], ValueId(1)).unwrap();
    let decompose = Computation::try_new(
        "decompose",
        vec![Instruction::new(
            Operation::parameter(0, "input"),
            Vec::new(),
            Vec::new(),
            ValueId(0),
            tensor,
        )],
        ValueId(0),
    )
    .unwrap();
    let mut module = Module::new("module");
    module.add_computation(main);
    module.add_computation(decompose);
    assert_eq!(module.verify(), Ok(()));
}

#[test]
fn module_rejects_unknown_composite_decomposition_symbols() {
    let composite = Instruction::new(
        Operation::stablehlo(
            StableHloOp::Composite,
            [
                (
                    "name".to_owned(),
                    stablehlo::Attribute::String("test.composite".to_owned()),
                ),
                (
                    "decomposition".to_owned(),
                    stablehlo::Attribute::SymbolRef(vec!["missing".to_owned()]),
                ),
            ]
            .into(),
        ),
        Vec::new(),
        Vec::new(),
        ValueId(0),
        Type::from(TensorType::scalar(ElementType::F32)),
    );
    let main = Computation::try_new("main", vec![composite], ValueId(0)).unwrap();
    let mut module = Module::new("module");
    module.add_computation(main);
    assert_eq!(
        module.verify(),
        Err(BuildError::UnknownComputationName {
            operation: "composite",
            name: "missing".to_owned(),
        })
    );
}

#[test]
fn module_computation_names_are_unique() {
    let computation = || {
        Computation::try_new(
            "duplicate",
            vec![Instruction::new(
                Operation::stablehlo(StableHloOp::CreateToken, Default::default()),
                Vec::new(),
                Vec::new(),
                ValueId(0),
                Type::Token,
            )],
            ValueId(0),
        )
        .unwrap()
    };
    let mut module = Module::new("duplicate_symbols");
    module.add_computation(computation());
    module.add_computation(computation());
    assert_eq!(
        module.verify(),
        Err(BuildError::DuplicateComputationName("duplicate".to_owned()))
    );
}
