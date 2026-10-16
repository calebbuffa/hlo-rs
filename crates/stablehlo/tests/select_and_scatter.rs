use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, ElementType, Instruction, Operation, Region,
    StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::computation_with_regions;

fn tensor(element: ElementType, shape: &[u64]) -> Type {
    TensorType::static_shape(element, shape.iter().copied()).into()
}

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

fn select(argument: Type, result: Type) -> Region {
    Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(200), argument.clone()),
            BlockArgument::new(ValueId(201), argument),
        ],
        vec![
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Compare,
                    [(
                        "comparison_direction".into(),
                        Attribute::Enum {
                            dialect: "stablehlo".into(),
                            name: "comparison_direction".into(),
                            value: "GE".into(),
                        },
                    )]
                    .into(),
                ),
                vec![ValueId(200), ValueId(201)],
                vec![],
                ValueId(202),
                result,
            ),
            Instruction::with_results(
                Operation::stablehlo(StableHloOp::Return, Attributes::new()),
                vec![ValueId(202)],
                vec![],
                vec![],
            ),
        ],
    )])
}

fn scatter(lhs: Type, rhs: Type) -> Region {
    Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(100), lhs),
            BlockArgument::new(ValueId(101), rhs),
        ],
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            vec![ValueId(100)],
            vec![],
            vec![],
        )],
    )])
}

fn attributes() -> Attributes {
    [
        ("window_dimensions".into(), Attribute::integers(vec![2, 3])),
        ("window_strides".into(), Attribute::integers(vec![2, 1])),
    ]
    .into()
}

#[test]
fn validates_window_source_shape_and_promoted_scatter_accumulator() {
    for accumulator in [ElementType::F16, ElementType::F32, ElementType::F64] {
        assert!(
            computation_with_regions(
                StableHloOp::SelectAndScatter,
                vec![
                    tensor(ElementType::F16, &[4, 5]),
                    tensor(ElementType::F16, &[2, 3]),
                    scalar(ElementType::F16)
                ],
                vec![tensor(accumulator, &[4, 5])],
                attributes(),
                vec![
                    select(scalar(ElementType::F16), scalar(ElementType::I1)),
                    scatter(scalar(accumulator), scalar(accumulator)),
                ],
            )
            .is_ok()
        );
    }
}

#[test]
fn rejects_inconsistent_source_init_result_and_scatter_types() {
    let input = tensor(ElementType::F32, &[4, 5]);
    let source = tensor(ElementType::F32, &[2, 3]);
    let init = scalar(ElementType::F32);
    let result = tensor(ElementType::F32, &[4, 5]);
    for (source, init, result, lhs, rhs) in [
        (
            tensor(ElementType::F32, &[2, 4]),
            init.clone(),
            result.clone(),
            ElementType::F32,
            ElementType::F32,
        ),
        (
            tensor(ElementType::I32, &[2, 3]),
            init.clone(),
            result.clone(),
            ElementType::F32,
            ElementType::F32,
        ),
        (
            source.clone(),
            tensor(ElementType::F32, &[1]),
            result.clone(),
            ElementType::F32,
            ElementType::F32,
        ),
        (
            source.clone(),
            scalar(ElementType::I32),
            result.clone(),
            ElementType::F32,
            ElementType::F32,
        ),
        (
            source.clone(),
            init.clone(),
            tensor(ElementType::F32, &[4, 6]),
            ElementType::F32,
            ElementType::F32,
        ),
        (
            source.clone(),
            init.clone(),
            result.clone(),
            ElementType::F32,
            ElementType::F64,
        ),
        (
            source.clone(),
            init.clone(),
            result.clone(),
            ElementType::F16,
            ElementType::F16,
        ),
        (source, init, result, ElementType::I32, ElementType::I32),
    ] {
        assert!(
            computation_with_regions(
                StableHloOp::SelectAndScatter,
                vec![input.clone(), source, init],
                vec![result],
                attributes(),
                vec![
                    select(scalar(ElementType::F32), scalar(ElementType::I1)),
                    scatter(scalar(lhs), scalar(rhs)),
                ],
            )
            .is_err()
        );
    }
}

#[test]
fn rejects_invalid_select_signature_and_window_geometry() {
    for (argument, predicate) in [
        (scalar(ElementType::I32), scalar(ElementType::I1)),
        (
            tensor(ElementType::F32, &[1]),
            tensor(ElementType::I1, &[1]),
        ),
        (scalar(ElementType::F32), scalar(ElementType::F32)),
    ] {
        assert!(
            computation_with_regions(
                StableHloOp::SelectAndScatter,
                vec![
                    tensor(ElementType::F32, &[4, 5]),
                    tensor(ElementType::F32, &[2, 3]),
                    scalar(ElementType::F32)
                ],
                vec![tensor(ElementType::F32, &[4, 5])],
                attributes(),
                vec![
                    select(argument, predicate),
                    scatter(scalar(ElementType::F32), scalar(ElementType::F32))
                ],
            )
            .is_err()
        );
    }
    for (name, value) in [
        ("window_dimensions", vec![]),
        ("window_dimensions", vec![0, 3]),
        ("window_dimensions", vec![2, -1]),
        ("window_strides", vec![2]),
        ("window_strides", vec![0, 1]),
    ] {
        let mut attrs = attributes();
        attrs.insert(name.into(), Attribute::integers(value));
        assert!(
            computation_with_regions(
                StableHloOp::SelectAndScatter,
                vec![
                    tensor(ElementType::F32, &[4, 5]),
                    tensor(ElementType::F32, &[2, 3]),
                    scalar(ElementType::F32)
                ],
                vec![tensor(ElementType::F32, &[4, 5])],
                attrs,
                vec![
                    select(scalar(ElementType::F32), scalar(ElementType::I1)),
                    scatter(scalar(ElementType::F32), scalar(ElementType::F32)),
                ],
            )
            .is_err()
        );
    }
}

#[test]
fn rank_zero_uses_default_windows_and_unrelated_attributes_do_not_dilate() {
    for (shape, source, attrs) in [
        (vec![], vec![], Attributes::new()),
        (vec![4, 5], vec![2, 3], {
            let mut attrs = attributes();
            attrs.insert("base_dilations".into(), Attribute::integers(vec![7, 9]));
            attrs.insert("window_dilations".into(), Attribute::integers(vec![7, 9]));
            attrs
        }),
    ] {
        assert!(
            computation_with_regions(
                StableHloOp::SelectAndScatter,
                vec![
                    tensor(ElementType::F32, &shape),
                    tensor(ElementType::F32, &source),
                    scalar(ElementType::F32)
                ],
                vec![tensor(ElementType::F32, &shape)],
                attrs,
                vec![
                    select(scalar(ElementType::F32), scalar(ElementType::I1)),
                    scatter(scalar(ElementType::F32), scalar(ElementType::F32)),
                ],
            )
            .is_ok()
        );
    }
}
