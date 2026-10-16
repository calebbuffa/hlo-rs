use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, Computation, ElementType, Instruction, Operation,
    Region, ScatterDimensionNumbers, StableHloAttribute, StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::computation_with_regions;

fn tensor(element: ElementType, shape: &[u64]) -> Type {
    TensorType::static_shape(element, shape.iter().copied()).into()
}

fn reducer(elements: &[ElementType]) -> Region {
    let arguments = elements
        .iter()
        .chain(elements)
        .enumerate()
        .map(|(index, &element)| {
            BlockArgument::new(
                ValueId(100 + index as u32),
                TensorType::scalar(element).into(),
            )
        })
        .collect();
    Region::new(vec![Block::new(
        arguments,
        vec![Instruction::with_results(
            Operation::stablehlo(StableHloOp::Return, Attributes::new()),
            (0..elements.len())
                .map(|index| ValueId(100 + index as u32))
                .collect(),
            vec![],
            vec![],
        )],
    )])
}

fn dimensions() -> ScatterDimensionNumbers {
    ScatterDimensionNumbers {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    }
}

fn scatter(
    input: &[u64],
    indices: &[u64],
    update: &[u64],
    output: &[u64],
    dimensions: ScatterDimensionNumbers,
) -> Result<Computation, stablehlo::BuildError> {
    computation_with_regions(
        StableHloOp::Scatter,
        vec![
            tensor(ElementType::F32, input),
            tensor(ElementType::I64, indices),
            tensor(ElementType::F32, update),
        ],
        vec![tensor(ElementType::F32, output)],
        [(
            "scatter_dimension_numbers".into(),
            Attribute::StableHlo(Box::new(StableHloAttribute::ScatterDimensionNumbers(
                dimensions,
            ))),
        )]
        .into(),
        vec![reducer(&[ElementType::F32])],
    )
}

#[test]
fn scatter_accepts_explicit_and_implicit_index_vectors_and_partial_windows() {
    for indices in [vec![2, 1], vec![2]] {
        for window in [0, 2, 4] {
            assert!(scatter(&[5, 4], &indices, &[2, window], &[5, 4], dimensions()).is_ok());
        }
    }
}

#[test]
fn scatter_checks_index_and_update_geometry_and_preserves_input_shape() {
    for (indices, update, output) in [
        (vec![2, 2], vec![2, 4], vec![5, 4]),
        (vec![2, 1], vec![3, 4], vec![5, 4]),
        (vec![2, 1], vec![2, 5], vec![5, 4]),
        (vec![2, 1], vec![2, 4, 1], vec![5, 4]),
        (vec![2, 1], vec![2, 4], vec![4, 5]),
    ] {
        assert!(scatter(&[5, 4], &indices, &update, &output, dimensions()).is_err());
    }
}

#[test]
fn scatter_checks_dimension_uniqueness_ranges_and_disjoint_batch_maps() {
    for dimensions in [
        ScatterDimensionNumbers {
            update_window_dims: vec![1, 1],
            ..dimensions()
        },
        ScatterDimensionNumbers {
            update_window_dims: vec![2],
            ..dimensions()
        },
        ScatterDimensionNumbers {
            inserted_window_dims: vec![2],
            ..dimensions()
        },
        ScatterDimensionNumbers {
            scatter_dims_to_operand_dims: vec![2],
            ..dimensions()
        },
        ScatterDimensionNumbers {
            index_vector_dim: 3,
            ..dimensions()
        },
        ScatterDimensionNumbers {
            input_batching_dims: vec![0],
            scatter_indices_batching_dims: vec![0],
            ..dimensions()
        },
    ] {
        assert!(scatter(&[5, 4], &[2, 1], &[2, 4], &[5, 4], dimensions).is_err());
    }
}

#[test]
fn scatter_supports_the_specification_batched_example() {
    let dimensions = ScatterDimensionNumbers {
        update_window_dims: vec![3, 4],
        inserted_window_dims: vec![1],
        input_batching_dims: vec![0],
        scatter_indices_batching_dims: vec![1],
        scatter_dims_to_operand_dims: vec![2, 1],
        index_vector_dim: 3,
    };
    assert!(
        scatter(
            &[2, 3, 4, 2],
            &[2, 2, 3, 2],
            &[2, 2, 3, 2, 2],
            &[2, 3, 4, 2],
            dimensions.clone()
        )
        .is_ok()
    );
    for index_axis in [0, 2, 3, 4] {
        let mut changed = dimensions.clone();
        changed.scatter_indices_batching_dims = vec![index_axis];
        let accepted = index_axis == 0;
        assert_eq!(
            scatter(
                &[2, 3, 4, 2],
                &[2, 2, 3, 2],
                &[2, 2, 3, 2, 2],
                &[2, 3, 4, 2],
                changed
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn variadic_scatter_promotes_accumulators_and_checks_each_update_element_type() {
    let attributes: Attributes = [(
        "scatter_dimension_numbers".into(),
        Attribute::StableHlo(Box::new(StableHloAttribute::ScatterDimensionNumbers(
            dimensions(),
        ))),
    )]
    .into();
    for (update_element, accepted) in [(ElementType::I8, true), (ElementType::F32, false)] {
        assert_eq!(
            computation_with_regions(
                StableHloOp::Scatter,
                vec![
                    tensor(ElementType::F32, &[5, 4]),
                    tensor(ElementType::I8, &[5, 4]),
                    tensor(ElementType::I64, &[2, 1]),
                    tensor(ElementType::F32, &[2, 4]),
                    tensor(update_element, &[2, 4])
                ],
                vec![
                    tensor(ElementType::F32, &[5, 4]),
                    tensor(ElementType::I32, &[5, 4])
                ],
                attributes.clone(),
                vec![reducer(&[ElementType::F32, ElementType::I32])]
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn scatter_rejects_empty_groups_without_panicking() {
    assert!(
        computation_with_regions(
            StableHloOp::Scatter,
            vec![tensor(ElementType::I64, &[2, 1])],
            vec![],
            [(
                "scatter_dimension_numbers".into(),
                Attribute::StableHlo(Box::new(StableHloAttribute::ScatterDimensionNumbers(
                    dimensions()
                )))
            )]
            .into(),
            vec![reducer(&[])]
        )
        .is_err()
    );
}
