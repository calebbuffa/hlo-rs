use stablehlo::{
    Attribute, BuildError, Computation, ElementType, Instruction, Operation, StableHloOp,
    TensorType, Type, ValueId,
};

fn build(shapes: &[Vec<Option<u64>>], result: Vec<Option<u64>>) -> Result<Computation, BuildError> {
    let mut instructions = Vec::new();
    for (index, shape) in shapes.iter().enumerate() {
        instructions.push(Instruction::new(
            Operation::parameter(index as u32, format!("arg{index}")),
            vec![],
            vec![],
            ValueId(index as u32),
            Type::from(TensorType::new(ElementType::F32, shape.clone())),
        ));
    }
    let root = ValueId(shapes.len() as u32);
    instructions.push(Instruction::new(
        Operation::stablehlo(
            StableHloOp::Concatenate,
            [("dimension".into(), Attribute::integer(0))].into(),
        ),
        (0..shapes.len())
            .map(|index| ValueId(index as u32))
            .collect(),
        vec![],
        root,
        Type::from(TensorType::new(ElementType::F32, result)),
    ));
    Computation::try_new("concat", instructions, root)
}

#[test]
fn dynamic_first_operand_cannot_mask_conflicting_later_shapes() {
    let shapes = [
        vec![Some(1), None],
        vec![Some(2), Some(3)],
        vec![Some(4), Some(5)],
    ];
    assert!(matches!(
        build(&shapes, vec![Some(7), None]),
        Err(BuildError::TypeInference {
            error: stablehlo::InferenceError::DimensionMismatch { .. },
            ..
        })
    ));
    let shapes = [vec![Some(1), None], vec![Some(2), Some(3)]];
    build(&shapes, vec![Some(3), Some(3)]).unwrap();
    assert!(matches!(
        build(&shapes, vec![Some(3), Some(5)]),
        Err(BuildError::InvalidResultType { .. })
    ));
    build(&shapes, vec![None, None]).unwrap();
}

#[test]
fn static_concatenated_extent_cannot_overflow_mlir_dimension_domain() {
    let shapes = [vec![Some(i64::MAX as u64)], vec![Some(1)]];
    assert!(matches!(
        build(&shapes, vec![None]),
        Err(BuildError::TypeInference {
            error: stablehlo::InferenceError::DimensionOverflow { .. },
            ..
        })
    ));
    let shapes = [vec![Some(i64::MAX as u64 - 1)], vec![Some(1)]];
    build(&shapes, vec![Some(i64::MAX as u64)]).unwrap();
}
