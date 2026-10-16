use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, Computation, ElementType, Instruction, Operation,
    Region, ScalarValue, StableHloAttribute, StableHloOp, TensorType, Type, TypeExtensions,
    ValueId,
};

mod common;
use common::computation;

fn tensor(size: Option<u64>, bound: Option<i64>) -> Type {
    let mut tensor = TensorType::new(ElementType::F32, vec![size]);
    if let Some(bound) = bound {
        tensor = tensor.with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions {
                bounds: vec![bound],
            }),
        )));
    }
    tensor.into()
}

#[test]
fn unary_and_binary_type_compatibility_checks_static_sizes_against_bounds() {
    for (input, output, accepted) in [
        (tensor(Some(3), None), tensor(None, Some(5)), true),
        (tensor(Some(7), None), tensor(None, Some(5)), false),
        (tensor(None, Some(5)), tensor(Some(7), None), false),
        (tensor(None, Some(3)), tensor(None, Some(5)), true),
    ] {
        assert_eq!(
            computation(
                StableHloOp::Negate,
                vec![input.clone()],
                vec![output.clone()],
                Attributes::new()
            )
            .is_ok(),
            accepted
        );
        assert_eq!(
            computation(
                StableHloOp::Add,
                vec![input.clone(), input],
                vec![output],
                Attributes::new()
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn inferred_static_shapes_cannot_exceed_declared_dynamic_bounds() {
    let cases: [(StableHloOp, Attributes); 4] = [
        (
            StableHloOp::Transpose,
            [("permutation".into(), Attribute::integers(vec![0]))].into(),
        ),
        (
            StableHloOp::Broadcast,
            [("broadcast_sizes".into(), Attribute::integers(vec![]))].into(),
        ),
        (
            StableHloOp::Slice,
            [
                ("start_indices".into(), Attribute::integers(vec![0])),
                ("limit_indices".into(), Attribute::integers(vec![3])),
                ("strides".into(), Attribute::integers(vec![1])),
            ]
            .into(),
        ),
        (
            StableHloOp::Reverse,
            [("dimensions".into(), Attribute::integers(vec![0]))].into(),
        ),
    ];
    for (opcode, attributes) in cases {
        for (bound, accepted) in [(2, false), (3, true), (5, true)] {
            assert_eq!(
                computation(
                    opcode,
                    vec![tensor(Some(3), None)],
                    vec![tensor(None, Some(bound))],
                    attributes.clone()
                )
                .is_ok(),
                accepted,
                "{opcode:?}"
            );
        }
    }
}

#[test]
fn constant_shape_operands_respect_dynamic_output_bounds() {
    for (bound, accepted) in [(2, false), (3, true), (5, true)] {
        let input = Instruction::new(
            Operation::parameter(0, "input"),
            vec![],
            vec![],
            ValueId(0),
            tensor(Some(3), None),
        );
        let shape_type = Type::from(TensorType::static_shape(ElementType::I64, [1]));
        let shape = Instruction::new(
            Operation::stablehlo(
                StableHloOp::Constant,
                [(
                    "value".into(),
                    Attribute::DenseElements {
                        ty: Box::new(shape_type.clone()),
                        values: vec![ScalarValue::Integer(3)],
                    },
                )]
                .into(),
            ),
            vec![],
            vec![],
            ValueId(1),
            shape_type,
        );
        let reshape = Instruction::new(
            Operation::stablehlo(StableHloOp::DynamicReshape, Attributes::new()),
            vec![ValueId(0), ValueId(1)],
            vec![],
            ValueId(2),
            tensor(None, Some(bound)),
        );
        assert_eq!(
            Computation::try_new("reshape", vec![input, shape, reshape], ValueId(2)).is_ok(),
            accepted
        );
    }
}

#[test]
fn select_refines_bounds_and_higher_order_shapes_check_every_bounded_witness() {
    let predicate: Type = TensorType::scalar(ElementType::I1).into();
    for (size, accepted) in [(3, true), (5, false)] {
        assert_eq!(
            computation(
                StableHloOp::Select,
                vec![
                    predicate.clone(),
                    tensor(None, Some(3)),
                    tensor(None, Some(4))
                ],
                vec![tensor(Some(size), None)],
                Attributes::new(),
            )
            .is_ok(),
            accepted
        );
    }
    let scalar: Type = TensorType::scalar(ElementType::F32).into();
    let body = Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(100), scalar.clone()),
            BlockArgument::new(ValueId(101), scalar),
        ],
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            vec![ValueId(100)],
            vec![],
            vec![],
        )],
    )]);
    for (size, accepted) in [(3, true), (5, false)] {
        assert_eq!(
            common::computation_with_regions(
                StableHloOp::Map,
                vec![tensor(None, Some(3)), tensor(Some(size), None)],
                vec![tensor(Some(size), None)],
                [("dimensions".into(), Attribute::integers(vec![0]))].into(),
                vec![body.clone()],
            )
            .is_ok(),
            accepted
        );
    }
}
