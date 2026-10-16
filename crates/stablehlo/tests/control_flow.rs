use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, ElementType, Instruction, Operation, Region,
    ScalarValue, StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::computation_with_regions;

fn tensor(shape: &[Option<u64>]) -> Type {
    TensorType::new(ElementType::F32, shape.to_vec()).into()
}

fn returned(value: u32) -> Instruction {
    Instruction::with_results(
        Operation::stablehlo(StableHloOp::Return, Attributes::new()),
        vec![ValueId(value)],
        vec![],
        vec![],
    )
}

fn condition(argument: Type) -> Region {
    let predicate = Type::from(TensorType::scalar(ElementType::I1));
    Region::new(vec![Block::new(
        vec![BlockArgument::new(ValueId(100), argument)],
        vec![
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Constant,
                    [(
                        "value".into(),
                        Attribute::DenseElements {
                            ty: Box::new(predicate.clone()),
                            values: vec![ScalarValue::Bool(true)],
                        },
                    )]
                    .into(),
                ),
                vec![],
                vec![],
                ValueId(101),
                predicate,
            ),
            returned(101),
        ],
    )])
}

fn body(argument: Type, capture: Option<u32>) -> Region {
    Region::new(vec![Block::new(
        vec![BlockArgument::new(ValueId(200), argument)],
        vec![returned(capture.unwrap_or(200))],
    )])
}

#[test]
fn while_accepts_compatible_dynamic_state_and_rejects_static_conflicts() {
    for (input, cond, carried, result, accepted) in [
        (
            tensor(&[Some(3)]),
            tensor(&[None]),
            tensor(&[None]),
            tensor(&[Some(3)]),
            true,
        ),
        (
            tensor(&[None]),
            tensor(&[Some(3)]),
            tensor(&[Some(3)]),
            tensor(&[None]),
            true,
        ),
        (Type::Token, Type::Token, Type::Token, Type::Token, true),
        (
            tensor(&[Some(3)]),
            tensor(&[Some(3)]),
            tensor(&[Some(3)]),
            tensor(&[Some(4)]),
            false,
        ),
        (
            tensor(&[Some(3)]),
            tensor(&[Some(4)]),
            tensor(&[Some(3)]),
            tensor(&[Some(3)]),
            false,
        ),
        (
            tensor(&[Some(3)]),
            tensor(&[Some(3)]),
            tensor(&[Some(4)]),
            tensor(&[Some(3)]),
            false,
        ),
    ] {
        assert_eq!(
            computation_with_regions(
                StableHloOp::While,
                vec![input],
                vec![result],
                Attributes::new(),
                vec![condition(cond), body(carried, None)],
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn while_checks_body_return_against_input_even_when_result_is_dynamic() {
    let input = tensor(&[Some(3)]);
    let other = tensor(&[Some(4)]);
    let parameters = [input.clone(), other]
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            Instruction::new(
                Operation::parameter(index as u32, format!("arg{index}")),
                vec![],
                vec![],
                ValueId(index as u32),
                ty,
            )
        });
    let loop_op = Instruction::new(
        Operation::stablehlo(StableHloOp::While, Attributes::new()),
        vec![ValueId(0)],
        vec![],
        ValueId(2),
        tensor(&[None]),
    )
    .with_regions(vec![condition(input.clone()), body(input, Some(1))]);
    assert!(
        stablehlo::Computation::try_new(
            "invalid",
            parameters.chain([loop_op]).collect(),
            ValueId(2)
        )
        .is_err()
    );
}

#[test]
fn if_branches_use_compatible_shapes_but_cannot_hide_static_conflicts() {
    for (first, second, accepted) in [
        (tensor(&[Some(3)]), tensor(&[None]), true),
        (tensor(&[Some(3)]), tensor(&[Some(4)]), false),
    ] {
        let predicate = Type::from(TensorType::scalar(ElementType::I1));
        let parameters = [predicate, first, second]
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                Instruction::new(
                    Operation::parameter(index as u32, format!("arg{index}")),
                    vec![],
                    vec![],
                    ValueId(index as u32),
                    ty,
                )
            });
        let branch = Instruction::new(
            Operation::stablehlo(StableHloOp::If, Attributes::new()),
            vec![ValueId(0)],
            vec![],
            ValueId(3),
            tensor(&[None]),
        )
        .with_regions(vec![
            Region::new(vec![Block::new(vec![], vec![returned(1)])]),
            Region::new(vec![Block::new(vec![], vec![returned(2)])]),
        ]);
        assert_eq!(
            stablehlo::Computation::try_new(
                "branch",
                parameters.chain([branch]).collect(),
                ValueId(3)
            )
            .is_ok(),
            accepted
        );
    }
}
