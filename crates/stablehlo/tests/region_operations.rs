use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, ElementType, Instruction, Operation, Region,
    ScalarValue, StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::computation_with_regions;

fn tensor(element: ElementType, dimensions: &[u64]) -> Type {
    TensorType::static_shape(element, dimensions.iter().copied()).into()
}

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

fn body(arguments: Vec<Type>, returns: &[u32]) -> Region {
    Region::new(vec![Block::new(
        arguments
            .into_iter()
            .enumerate()
            .map(|(index, ty)| BlockArgument::new(ValueId(100 + index as u32), ty))
            .collect(),
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            returns.iter().map(|index| ValueId(100 + index)).collect(),
            vec![],
            vec![],
        )],
    )])
}

fn reduction(
    opcode: StableHloOp,
    input: Type,
    init: Type,
    output: Type,
    accumulator: Type,
    attributes: Attributes,
) -> Result<stablehlo::Computation, stablehlo::BuildError> {
    computation_with_regions(
        opcode,
        vec![input, init],
        vec![output],
        attributes,
        vec![body(vec![accumulator.clone(), accumulator], &[0])],
    )
}

fn dimensions(values: &[i64]) -> Attributes {
    [(
        "dimensions".to_owned(),
        Attribute::integers(values.to_vec()),
    )]
    .into()
}

#[test]
fn map_checks_full_rank_dimensions_scalar_arguments_and_return_types() {
    let input = tensor(ElementType::F32, &[2, 3]);
    let valid = body(vec![scalar(ElementType::F32)], &[0]);
    assert!(
        computation_with_regions(
            StableHloOp::Map,
            vec![input.clone()],
            vec![input.clone()],
            dimensions(&[0, 1]),
            vec![valid.clone()]
        )
        .is_ok()
    );
    for invalid_dimensions in [vec![0], vec![1, 0], vec![0, 0], vec![-1, 1], vec![0, 2]] {
        assert!(
            computation_with_regions(
                StableHloOp::Map,
                vec![input.clone()],
                vec![input.clone()],
                dimensions(&invalid_dimensions),
                vec![valid.clone()]
            )
            .is_err()
        );
    }
    for arguments in [
        vec![tensor(ElementType::F32, &[1])],
        vec![scalar(ElementType::I32)],
        vec![Type::Token],
    ] {
        assert!(
            computation_with_regions(
                StableHloOp::Map,
                vec![input.clone()],
                vec![input.clone()],
                dimensions(&[0, 1]),
                vec![body(arguments, &[0])]
            )
            .is_err()
        );
    }
    assert!(
        computation_with_regions(
            StableHloOp::Map,
            vec![input.clone()],
            vec![tensor(ElementType::I32, &[2, 3])],
            dimensions(&[0, 1]),
            vec![valid.clone()]
        )
        .is_err()
    );
    assert!(
        computation_with_regions(
            StableHloOp::Map,
            vec![input],
            vec![tensor(ElementType::F32, &[3, 2])],
            dimensions(&[0, 1]),
            vec![valid]
        )
        .is_err()
    );
}

#[test]
fn map_and_reductions_reject_empty_operand_groups() {
    for opcode in [
        StableHloOp::Map,
        StableHloOp::Reduce,
        StableHloOp::ReduceWindow,
    ] {
        let results = if opcode == StableHloOp::Map {
            vec![scalar(ElementType::F32)]
        } else {
            vec![]
        };
        assert!(
            computation_with_regions(
                opcode,
                vec![],
                results,
                dimensions(&[]),
                vec![body(vec![], &[])]
            )
            .is_err()
        );
    }
}

#[test]
fn reduce_removes_unique_in_range_dimensions_and_promotes_accumulators() {
    for (input, accumulator) in [
        (ElementType::I8, ElementType::I32),
        (ElementType::Ui8, ElementType::I16),
        (ElementType::F16, ElementType::F32),
        (ElementType::C64, ElementType::C128),
        (ElementType::I1, ElementType::I1),
    ] {
        assert!(
            reduction(
                StableHloOp::Reduce,
                tensor(input, &[2, 3]),
                scalar(input),
                tensor(accumulator, &[2]),
                scalar(accumulator),
                dimensions(&[1])
            )
            .is_ok()
        );
    }
    for (input, accumulator) in [
        (ElementType::I32, ElementType::I8),
        (ElementType::F32, ElementType::F16),
        (ElementType::C128, ElementType::C64),
        (ElementType::I32, ElementType::F32),
    ] {
        assert!(
            reduction(
                StableHloOp::Reduce,
                tensor(input, &[2, 3]),
                scalar(input),
                tensor(accumulator, &[2]),
                scalar(accumulator),
                dimensions(&[1])
            )
            .is_err()
        );
    }
    for reduced in [vec![0, 0], vec![2], vec![-1]] {
        assert!(
            reduction(
                StableHloOp::Reduce,
                tensor(ElementType::F32, &[2, 3]),
                scalar(ElementType::F32),
                tensor(ElementType::F32, &[2]),
                scalar(ElementType::F32),
                dimensions(&reduced)
            )
            .is_err()
        );
    }
    assert!(
        reduction(
            StableHloOp::Reduce,
            tensor(ElementType::F32, &[2, 3]),
            scalar(ElementType::F32),
            tensor(ElementType::F32, &[3]),
            scalar(ElementType::F32),
            dimensions(&[1])
        )
        .is_err()
    );
}

#[test]
fn reduce_checks_init_and_region_types_instead_of_only_counts() {
    for (init, accumulator) in [
        (tensor(ElementType::F32, &[1]), scalar(ElementType::F32)),
        (scalar(ElementType::I32), scalar(ElementType::F32)),
        (scalar(ElementType::F32), tensor(ElementType::F32, &[1])),
    ] {
        assert!(
            reduction(
                StableHloOp::Reduce,
                tensor(ElementType::F32, &[2, 3]),
                init,
                tensor(ElementType::F32, &[2]),
                accumulator,
                dimensions(&[1])
            )
            .is_err()
        );
    }
    assert!(
        computation_with_regions(
            StableHloOp::Reduce,
            vec![tensor(ElementType::F32, &[2]), scalar(ElementType::F32)],
            vec![scalar(ElementType::F32)],
            dimensions(&[0]),
            vec![body(
                vec![scalar(ElementType::F32), scalar(ElementType::I32)],
                &[0]
            )]
        )
        .is_err()
    );
}

#[test]
fn reduce_supports_multiple_typed_accumulators() {
    assert!(
        computation_with_regions(
            StableHloOp::Reduce,
            vec![
                tensor(ElementType::F32, &[2, 3]),
                tensor(ElementType::I8, &[2, 3]),
                scalar(ElementType::F32),
                scalar(ElementType::I8)
            ],
            vec![
                tensor(ElementType::F32, &[3]),
                tensor(ElementType::I32, &[3])
            ],
            dimensions(&[0]),
            vec![body(
                vec![
                    scalar(ElementType::F32),
                    scalar(ElementType::I32),
                    scalar(ElementType::F32),
                    scalar(ElementType::I32)
                ],
                &[0, 1]
            )]
        )
        .is_ok()
    );
}

#[test]
fn common_dynamic_witness_cannot_hide_contradictory_static_shapes() {
    let dynamic: Type = TensorType::new(ElementType::F32, vec![None, Some(3)]).into();
    assert!(
        computation_with_regions(
            StableHloOp::Map,
            vec![
                dynamic.clone(),
                tensor(ElementType::F32, &[2, 3]),
                tensor(ElementType::F32, &[4, 3])
            ],
            vec![dynamic.clone()],
            dimensions(&[0, 1]),
            vec![body(vec![scalar(ElementType::F32); 3], &[0])]
        )
        .is_err()
    );
    assert!(
        computation_with_regions(
            StableHloOp::Reduce,
            vec![
                dynamic.clone(),
                tensor(ElementType::F32, &[2, 3]),
                scalar(ElementType::F32),
                scalar(ElementType::F32)
            ],
            vec![
                tensor(ElementType::F32, &[4]),
                tensor(ElementType::F32, &[4])
            ],
            dimensions(&[1]),
            vec![body(vec![scalar(ElementType::F32); 4], &[0, 1])]
        )
        .is_err()
    );
    assert!(
        computation_with_regions(
            StableHloOp::Reduce,
            vec![
                dynamic.clone(),
                dynamic,
                scalar(ElementType::F32),
                scalar(ElementType::F32)
            ],
            vec![
                tensor(ElementType::F32, &[2]),
                tensor(ElementType::F32, &[4])
            ],
            dimensions(&[1]),
            vec![body(vec![scalar(ElementType::F32); 4], &[0, 1])]
        )
        .is_err()
    );
}

fn padding(rank: u64, values: &[i128]) -> Attribute {
    Attribute::DenseElements {
        ty: Box::new(tensor(ElementType::I64, &[rank, 2])),
        values: values.iter().copied().map(ScalarValue::Integer).collect(),
    }
}

#[test]
fn reduce_window_infers_strides_dilations_and_asymmetric_padding() {
    let attributes: Attributes = [
        (
            "window_dimensions".to_owned(),
            Attribute::integers(vec![2, 1]),
        ),
        ("window_strides".to_owned(), Attribute::integers(vec![4, 1])),
        ("base_dilations".to_owned(), Attribute::integers(vec![2, 1])),
        (
            "window_dilations".to_owned(),
            Attribute::integers(vec![3, 1]),
        ),
        ("padding".to_owned(), padding(2, &[2, 1, 0, 0])),
    ]
    .into();
    for (output, accepted) in [([2, 2], true), ([3, 2], false), ([2, 3], false)] {
        assert_eq!(
            reduction(
                StableHloOp::ReduceWindow,
                tensor(ElementType::F32, &[3, 2]),
                scalar(ElementType::F32),
                tensor(ElementType::F32, &output),
                scalar(ElementType::F32),
                attributes.clone()
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn reduce_window_handles_empty_windows_and_validates_every_vector() {
    let base: Attributes = [("window_dimensions".to_owned(), Attribute::integers(vec![2]))].into();
    for size in [0, 1] {
        assert!(
            reduction(
                StableHloOp::ReduceWindow,
                tensor(ElementType::F32, &[size]),
                scalar(ElementType::F32),
                tensor(ElementType::F32, &[0]),
                scalar(ElementType::F32),
                base.clone()
            )
            .is_ok()
        );
    }
    for name in [
        "window_dimensions",
        "window_strides",
        "base_dilations",
        "window_dilations",
    ] {
        for vector in [vec![], vec![0], vec![-1], vec![1, 1]] {
            let mut attributes = base.clone();
            attributes.insert(name.into(), Attribute::integers(vector));
            assert!(
                reduction(
                    StableHloOp::ReduceWindow,
                    tensor(ElementType::F32, &[4]),
                    scalar(ElementType::F32),
                    tensor(ElementType::F32, &[3]),
                    scalar(ElementType::F32),
                    attributes
                )
                .is_err()
            );
        }
    }
    let mut attributes = base;
    attributes.insert("padding".into(), padding(2, &[0, 0, 0, 0]));
    assert!(
        reduction(
            StableHloOp::ReduceWindow,
            tensor(ElementType::F32, &[4]),
            scalar(ElementType::F32),
            tensor(ElementType::F32, &[3]),
            scalar(ElementType::F32),
            attributes
        )
        .is_err()
    );
}

fn comparator(elements: &[ElementType], predicate: bool) -> Region {
    let arguments = elements
        .iter()
        .flat_map(|&element| [scalar(element), scalar(element)])
        .enumerate()
        .map(|(index, ty)| BlockArgument::new(ValueId(100 + index as u32), ty))
        .collect();
    let compare = Instruction::new(
        Operation::stablehlo(
            StableHloOp::Compare,
            [(
                "comparison_direction".to_owned(),
                Attribute::Enum {
                    dialect: "stablehlo".into(),
                    name: "comparison_direction".into(),
                    value: "LT".into(),
                },
            )]
            .into(),
        ),
        vec![ValueId(100), ValueId(101)],
        vec![],
        ValueId(200),
        scalar(ElementType::I1),
    );
    let terminator = Instruction::with_results(
        Operation::stablehlo(StableHloOp::Return, Attributes::new()),
        vec![ValueId(if predicate { 200 } else { 100 })],
        vec![],
        vec![],
    );
    Region::new(vec![Block::new(arguments, vec![compare, terminator])])
}

#[test]
fn sort_uses_interleaved_comparator_arguments_and_scalar_predicate() {
    let inputs = vec![
        tensor(ElementType::F32, &[2, 3]),
        tensor(ElementType::I32, &[2, 3]),
    ];
    for dimension in [-2, -1, 0, 1] {
        assert!(
            computation_with_regions(
                StableHloOp::Sort,
                inputs.clone(),
                inputs.clone(),
                [("dimension".into(), Attribute::integer(dimension))].into(),
                vec![comparator(&[ElementType::F32, ElementType::I32], true)]
            )
            .is_ok()
        );
    }
    for dimension in [-3, 2] {
        assert!(
            computation_with_regions(
                StableHloOp::Sort,
                inputs.clone(),
                inputs.clone(),
                [("dimension".into(), Attribute::integer(dimension))].into(),
                vec![comparator(&[ElementType::F32, ElementType::I32], true)]
            )
            .is_err()
        );
    }
    assert!(
        computation_with_regions(
            StableHloOp::Sort,
            inputs.clone(),
            inputs,
            Attributes::new(),
            vec![comparator(&[ElementType::F32, ElementType::I32], false)]
        )
        .is_err()
    );
}

#[test]
fn all_reduce_uses_one_scalar_binary_region_even_for_multiple_operands() {
    let inputs = vec![
        tensor(ElementType::F16, &[2]),
        tensor(ElementType::F32, &[3, 4]),
    ];
    let results = vec![
        tensor(ElementType::F32, &[2]),
        tensor(ElementType::F32, &[3, 4]),
    ];
    let attributes: Attributes = [(
        "replica_groups".into(),
        Attribute::DenseElements {
            ty: Box::new(tensor(ElementType::I64, &[1, 2])),
            values: vec![ScalarValue::Integer(0), ScalarValue::Integer(1)],
        },
    )]
    .into();
    assert!(
        computation_with_regions(
            StableHloOp::AllReduce,
            inputs.clone(),
            results.clone(),
            attributes.clone(),
            vec![body(vec![scalar(ElementType::F32); 2], &[0])]
        )
        .is_ok()
    );
    assert!(
        computation_with_regions(
            StableHloOp::AllReduce,
            inputs,
            results,
            attributes,
            vec![body(vec![scalar(ElementType::F32); 4], &[0, 1])]
        )
        .is_err()
    );
}

#[test]
fn reduce_window_rejects_unrepresentable_result_extents_without_overflow() {
    let attributes: Attributes = [
        ("window_dimensions".into(), Attribute::integers(vec![1])),
        ("base_dilations".into(), Attribute::integers(vec![i64::MAX])),
    ]
    .into();
    assert!(
        reduction(
            StableHloOp::ReduceWindow,
            tensor(ElementType::F32, &[i64::MAX as u64]),
            scalar(ElementType::F32),
            TensorType::new(ElementType::F32, vec![None]).into(),
            scalar(ElementType::F32),
            attributes
        )
        .is_err()
    );
}
