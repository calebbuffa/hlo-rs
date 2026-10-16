use stablehlo::{
    Attribute, AttributeError, DotAlgorithm, ElementType, Mesh, MeshAxis, MeshReference, Operation,
    QuantizedTensorType, ResultAccuracy, ResultAccuracyMode, ScalarValue, StableHloAttribute,
    StableHloOp, TensorType, Type,
};

#[test]
fn dense_element_attributes_validate_shape_and_scalar_values() {
    let ty = Box::new(Type::from(TensorType::static_shape(ElementType::Ui8, [2])));
    let valid = Attribute::DenseElements {
        ty: ty.clone(),
        values: vec![ScalarValue::Integer(0), ScalarValue::Integer(255)],
    };
    assert_eq!(valid.verify(), Ok(()));

    let invalid_value = Attribute::DenseElements {
        ty: ty.clone(),
        values: vec![ScalarValue::Integer(0), ScalarValue::Integer(256)],
    };
    assert_eq!(
        invalid_value.verify(),
        Err(stablehlo::AttributeValidationError::InvalidDenseElement { index: 1 })
    );

    let wrong_count = Attribute::DenseElements {
        ty,
        values: vec![ScalarValue::Integer(0)],
    };
    assert_eq!(
        wrong_count.verify(),
        Err(
            stablehlo::AttributeValidationError::DenseElementCountMismatch {
                expected: 2,
                actual: 1,
            }
        )
    );
}

#[test]
fn floating_and_complex_literal_bits_fit_the_declared_storage_width() {
    for (element, bits) in [
        (ElementType::F4E2M1Fn, 4),
        (ElementType::F6E2M3Fn, 6),
        (ElementType::F8E4M3Fn, 8),
        (ElementType::F16, 16),
        (ElementType::F32, 32),
        (ElementType::F64, 64),
    ] {
        let attribute = |value| Attribute::DenseElements {
            ty: Box::new(Type::from(TensorType::scalar(element))),
            values: vec![ScalarValue::FloatBits(value)],
        };
        assert!(attribute((1_u128 << bits) - 1).verify().is_ok());
        assert!(attribute(1_u128 << bits).verify().is_err());
    }
    let complex = |element, real, imaginary| Attribute::DenseElements {
        ty: Box::new(Type::from(TensorType::scalar(element))),
        values: vec![ScalarValue::ComplexBits { real, imaginary }],
    };
    assert!(
        complex(ElementType::C64, u64::from(u32::MAX), 0)
            .verify()
            .is_ok()
    );
    assert!(
        complex(ElementType::C64, u64::from(u32::MAX) + 1, 0)
            .verify()
            .is_err()
    );
    assert!(
        complex(ElementType::C64, 0, u64::from(u32::MAX) + 1)
            .verify()
            .is_err()
    );
    assert!(
        complex(ElementType::C128, u64::MAX, u64::MAX)
            .verify()
            .is_ok()
    );
}

#[test]
fn empty_dense_tensor_does_not_overflow_before_reaching_a_zero_dimension() {
    let attribute = Attribute::DenseElements {
        ty: Box::new(Type::from(TensorType::static_shape(
            ElementType::F32,
            [i64::MAX as u64, i64::MAX as u64, 0],
        ))),
        values: vec![],
    };
    assert_eq!(attribute.verify(), Ok(()));
}

#[test]
fn dense_quantized_values_respect_the_declared_storage_range() {
    let ty = Box::new(Type::QuantizedTensor(QuantizedTensorType::new(
        vec![Some(1)],
        ElementType::Ui8,
        Some(0),
        Some(10),
        ElementType::F32,
        None,
        vec![0.5],
        vec![0],
    )));
    let attribute = Attribute::DenseElements {
        ty,
        values: vec![ScalarValue::Integer(11)],
    };
    assert_eq!(
        attribute.verify(),
        Err(stablehlo::AttributeValidationError::InvalidDenseElement { index: 0 })
    );
}

#[test]
fn sparse_element_attributes_validate_coordinate_bounds_and_rank() {
    let ty = Box::new(Type::from(TensorType::static_shape(
        ElementType::F32,
        [2, 3],
    )));
    let valid = Attribute::SparseElements {
        ty: ty.clone(),
        indices: vec![vec![1, 2]],
        values: vec![ScalarValue::FloatBits(0)],
    };
    assert_eq!(valid.verify(), Ok(()));

    let out_of_bounds = Attribute::SparseElements {
        ty: ty.clone(),
        indices: vec![vec![2, 0]],
        values: vec![ScalarValue::FloatBits(0)],
    };
    assert_eq!(
        out_of_bounds.verify(),
        Err(
            stablehlo::AttributeValidationError::SparseIndexOutOfBounds {
                index: 0,
                dimension: 0,
            }
        )
    );

    let wrong_rank = Attribute::SparseElements {
        ty,
        indices: vec![vec![1]],
        values: vec![ScalarValue::FloatBits(0)],
    };
    assert_eq!(
        wrong_rank.verify(),
        Err(
            stablehlo::AttributeValidationError::SparseIndexRankMismatch {
                index: 0,
                expected: 2,
                actual: 1,
            }
        )
    );
}

#[test]
fn result_accuracy_enforces_specified_mode_constraints() {
    let tolerance = StableHloAttribute::ResultAccuracy(ResultAccuracy::new(
        0.01,
        0.02,
        2,
        ResultAccuracyMode::Tolerance,
    ));
    assert_eq!(tolerance.verify(), Ok(()));

    let invalid_default = StableHloAttribute::ResultAccuracy(ResultAccuracy::new(
        0.01,
        0.0,
        0,
        ResultAccuracyMode::Default,
    ));
    assert_eq!(
        invalid_default.verify(),
        Err(AttributeError::InvalidToleranceMode)
    );
}

#[test]
fn dot_algorithm_requires_positive_component_counts() {
    let algorithm = StableHloAttribute::DotAlgorithm(Box::new(DotAlgorithm {
        lhs_precision_type: Type::Element(ElementType::F16),
        rhs_precision_type: Type::Element(ElementType::F16),
        accumulation_type: Type::Element(ElementType::F32),
        lhs_component_count: 0,
        rhs_component_count: 1,
        num_primitive_operations: 1,
        allow_imprecise_accumulation: false,
    }));
    assert_eq!(
        algorithm.verify(),
        Err(AttributeError::NonPositive {
            field: "lhs_component_count",
            value: 0,
        })
    );
}

#[test]
fn dot_algorithm_accepts_only_reference_defined_combinations() {
    let algorithm = |lhs, rhs, accumulation, primitive_ops, allow_imprecise| {
        StableHloAttribute::DotAlgorithm(Box::new(DotAlgorithm {
            lhs_precision_type: Type::Element(lhs),
            rhs_precision_type: Type::Element(rhs),
            accumulation_type: Type::Element(accumulation),
            lhs_component_count: 1,
            rhs_component_count: 1,
            num_primitive_operations: primitive_ops,
            allow_imprecise_accumulation: allow_imprecise,
        }))
    };

    assert!(
        algorithm(
            ElementType::Bf16,
            ElementType::Bf16,
            ElementType::F32,
            3,
            false
        )
        .verify()
        .is_ok()
    );
    assert!(
        algorithm(
            ElementType::F8E4M3FnUz,
            ElementType::F8E5M2,
            ElementType::F32,
            1,
            true
        )
        .verify()
        .is_ok()
    );
    for invalid in [
        algorithm(
            ElementType::F16,
            ElementType::F16,
            ElementType::F64,
            1,
            false,
        ),
        algorithm(
            ElementType::F32,
            ElementType::F32,
            ElementType::F32,
            3,
            false,
        ),
        algorithm(
            ElementType::F8E4M3Fn,
            ElementType::F8E4M3Fn,
            ElementType::F32,
            3,
            true,
        ),
    ] {
        assert_eq!(invalid.verify(), Err(AttributeError::UnknownDotAlgorithm));
    }
}

#[test]
fn mesh_replica_axes_must_reference_defined_axes() {
    let mesh = Mesh {
        axes: vec![MeshAxis {
            name: "x".to_owned(),
            size: 4,
        }],
        device_ids: None,
    };
    let groups = StableHloAttribute::ReplicaGroupMeshAxes(stablehlo::ReplicaGroupMeshAxes {
        mesh: MeshReference::Inline(mesh),
        axes: vec![stablehlo::AxisRef {
            name: "missing".to_owned(),
            sub_axis_info: None,
        }],
    });
    assert_eq!(
        groups.verify(),
        Err(AttributeError::UnknownMeshAxis("missing".to_owned()))
    );
}

#[test]
fn verifier_checks_typed_operation_attributes() {
    let instruction = stablehlo::Instruction::new(
        Operation::stablehlo(
            StableHloOp::DotGeneral,
            [(
                "dot_algorithm".to_owned(),
                Attribute::StableHlo(Box::new(StableHloAttribute::DotAlgorithm(Box::new(
                    DotAlgorithm {
                        lhs_precision_type: Type::Element(ElementType::F16),
                        rhs_precision_type: Type::Element(ElementType::F16),
                        accumulation_type: Type::Element(ElementType::F32),
                        lhs_component_count: 1,
                        rhs_component_count: 1,
                        num_primitive_operations: 0,
                        allow_imprecise_accumulation: false,
                    },
                )))),
            )]
            .into(),
        ),
        vec![],
        vec![],
        stablehlo::ValueId(0),
        Type::from(TensorType::scalar(ElementType::F32)),
    );
    assert!(matches!(
        stablehlo::Computation::try_new("invalid", vec![instruction], stablehlo::ValueId(0)),
        Err(stablehlo::BuildError::InvalidAttribute { .. })
    ));
}

#[test]
fn constant_results_match_their_element_attribute_types() {
    let value_type = Type::from(TensorType::static_shape(ElementType::F32, [2]));
    let value = Attribute::DenseElements {
        ty: Box::new(value_type.clone()),
        values: vec![ScalarValue::FloatBits(0), ScalarValue::FloatBits(0)],
    };
    let valid = stablehlo::Instruction::new(
        Operation::stablehlo(
            StableHloOp::Constant,
            [("value".to_owned(), value.clone())].into(),
        ),
        Vec::new(),
        Vec::new(),
        stablehlo::ValueId(0),
        value_type,
    );
    assert!(
        stablehlo::Computation::try_new("valid_constant", vec![valid], stablehlo::ValueId(0))
            .is_ok()
    );

    let invalid = stablehlo::Instruction::new(
        Operation::stablehlo(StableHloOp::Constant, [("value".to_owned(), value)].into()),
        Vec::new(),
        Vec::new(),
        stablehlo::ValueId(0),
        Type::from(TensorType::static_shape(ElementType::F32, [3])),
    );
    assert!(matches!(
        stablehlo::Computation::try_new("invalid_constant", vec![invalid], stablehlo::ValueId(0)),
        Err(stablehlo::BuildError::InvalidResultType {
            operation: "constant",
            ..
        })
    ));
}
