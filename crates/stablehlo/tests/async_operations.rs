use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, BuildError, ElementType, Instruction, Operation,
    QuantizedTensorType, Region, StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::{computation, computation_with_regions};

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

fn slice(input: u32, result: u32, ty: Type) -> Instruction {
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
        vec![ValueId(input)],
        vec![],
        ValueId(result),
        ty,
    )
}

fn returned(values: &[u32]) -> Instruction {
    Instruction::with_results(
        Operation::stablehlo(StableHloOp::Return, Attributes::new()),
        values.iter().copied().map(ValueId).collect(),
        vec![],
        vec![],
    )
}

fn start(
    input: Type,
    future: Type,
    arguments: Vec<BlockArgument>,
    instructions: Vec<Instruction>,
) -> Result<stablehlo::Computation, BuildError> {
    computation_with_regions(
        StableHloOp::AsyncStart,
        vec![input],
        vec![future],
        Attributes::new(),
        vec![Region::new(vec![Block::new(arguments, instructions)])],
    )
}

#[test]
fn futures_and_async_operations_support_quantized_ranked_tensor_elements() {
    for ty in [
        scalar(ElementType::F32),
        Type::QuantizedTensor(QuantizedTensorType::new(
            vec![],
            ElementType::I8,
            None,
            None,
            ElementType::F32,
            None,
            vec![0.5],
            vec![0],
        )),
    ] {
        let future = Type::Future(vec![ty.clone()]);
        assert!(future.verify().is_ok());
        assert!(
            start(
                ty.clone(),
                future.clone(),
                vec![BlockArgument::new(ValueId(100), ty.clone())],
                vec![slice(100, 101, ty.clone()), returned(&[101])],
            )
            .is_ok()
        );
        assert!(
            computation(
                StableHloOp::AsyncDone,
                vec![future],
                vec![ty],
                Attributes::new()
            )
            .is_ok()
        );
    }
}

#[test]
fn async_done_requires_one_future_element_and_a_compatible_result() {
    let ty = scalar(ElementType::F32);
    for (elements, result, accepted) in [
        (vec![ty.clone()], ty.clone(), true),
        (vec![], ty.clone(), false),
        (vec![ty.clone(), ty.clone()], ty.clone(), false),
        (vec![ty], scalar(ElementType::I32), false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::AsyncDone,
                vec![Type::Future(elements)],
                vec![result],
                Attributes::new()
            )
            .is_ok(),
            accepted
        );
    }
    for element in [Type::Token, Type::Tuple(vec![]), Type::Future(vec![])] {
        assert!(matches!(
            Type::Future(vec![element]).verify(),
            Err(stablehlo::TypeError::ExpectedRankedTensor { index: 0, .. })
        ));
    }
}

#[test]
fn async_start_rejects_noncollective_nonslice_operations_and_extra_body_operations() {
    let ty = scalar(ElementType::F32);
    for instructions in [
        vec![returned(&[100])],
        vec![
            Instruction::new(
                Operation::stablehlo(StableHloOp::Negate, Attributes::new()),
                vec![ValueId(100)],
                vec![],
                ValueId(101),
                ty.clone(),
            ),
            returned(&[101]),
        ],
        vec![
            slice(100, 101, ty.clone()),
            slice(101, 102, ty.clone()),
            returned(&[102]),
        ],
    ] {
        assert_eq!(
            start(
                ty.clone(),
                Type::Future(vec![ty.clone()]),
                vec![BlockArgument::new(ValueId(100), ty.clone())],
                instructions
            ),
            Err(BuildError::InvalidAsyncBody)
        );
    }
}

#[test]
fn async_start_binds_operands_to_equal_typed_block_arguments_and_exact_future_results() {
    let ty = scalar(ElementType::F32);
    let other = scalar(ElementType::I32);
    assert!(
        start(
            ty.clone(),
            Type::Future(vec![other.clone()]),
            vec![BlockArgument::new(ValueId(100), ty.clone())],
            vec![slice(100, 101, ty.clone()), returned(&[101])],
        )
        .is_err()
    );
    assert!(
        start(
            ty.clone(),
            Type::Future(vec![other.clone()]),
            vec![BlockArgument::new(ValueId(100), other.clone())],
            vec![slice(100, 101, other), returned(&[101])],
        )
        .is_err()
    );
    assert!(
        start(
            ty.clone(),
            Type::Future(vec![ty.clone(), ty.clone()]),
            vec![BlockArgument::new(ValueId(100), ty.clone())],
            vec![slice(100, 101, ty.clone()), returned(&[101, 101])],
        )
        .is_err()
    );
    assert!(
        start(
            ty.clone(),
            Type::Future(vec![ty.clone()]),
            vec![BlockArgument::new(ValueId(100), ty.clone())],
            vec![slice(0, 101, ty), returned(&[101])],
        )
        .is_err()
    );
}
