use stablehlo::{
    Attribute, Attributes, ElementType, GatherDimensionNumbers, StableHloAttribute, StableHloOp,
    TensorType, Type,
};

mod common;
use common::computation;

fn tensor(element: ElementType, dimensions: &[u64]) -> Type {
    TensorType::static_shape(element, dimensions.iter().copied()).into()
}

fn attributes(dimensions: GatherDimensionNumbers, sizes: Option<&[i64]>) -> Attributes {
    let mut attributes: Attributes = [(
        "dimension_numbers".into(),
        Attribute::StableHlo(Box::new(StableHloAttribute::GatherDimensionNumbers(
            dimensions,
        ))),
    )]
    .into();
    if let Some(sizes) = sizes {
        attributes.insert("slice_sizes".into(), Attribute::integers(sizes.to_vec()));
    }
    attributes
}

fn row_gather() -> GatherDimensionNumbers {
    GatherDimensionNumbers {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    }
}

#[test]
fn gather_infers_slice_offsets_and_explicit_or_implicit_index_vectors() {
    for indices in [vec![2, 1], vec![2]] {
        assert!(
            computation(
                StableHloOp::Gather,
                vec![
                    tensor(ElementType::F32, &[5, 4]),
                    tensor(ElementType::I64, &indices)
                ],
                vec![tensor(ElementType::F32, &[2, 4])],
                attributes(row_gather(), Some(&[1, 4]))
            )
            .is_ok()
        );
    }
    assert!(
        computation(
            StableHloOp::Gather,
            vec![
                tensor(ElementType::F32, &[5, 4]),
                tensor(ElementType::I64, &[2, 1])
            ],
            vec![tensor(ElementType::F32, &[4, 2])],
            attributes(row_gather(), Some(&[1, 4]))
        )
        .is_err()
    );
}

#[test]
fn gather_checks_slice_bounds_collapsed_extents_and_element_types() {
    for sizes in [vec![2, 4], vec![1, 5], vec![-1, 4], vec![1]] {
        assert!(
            computation(
                StableHloOp::Gather,
                vec![
                    tensor(ElementType::F32, &[5, 4]),
                    tensor(ElementType::I64, &[2, 1])
                ],
                vec![tensor(ElementType::F32, &[2, 4])],
                attributes(row_gather(), Some(&sizes))
            )
            .is_err()
        );
    }
    assert!(
        computation(
            StableHloOp::Gather,
            vec![
                tensor(ElementType::F32, &[5, 4]),
                tensor(ElementType::I64, &[2, 1])
            ],
            vec![tensor(ElementType::I32, &[2, 4])],
            attributes(row_gather(), Some(&[1, 4]))
        )
        .is_err()
    );
}

#[test]
fn gather_rejects_out_of_range_duplicate_and_overlapping_dimension_maps() {
    let base = row_gather();
    let invalid = [
        GatherDimensionNumbers {
            offset_dims: vec![2],
            ..base.clone()
        },
        GatherDimensionNumbers {
            offset_dims: vec![1, 1],
            ..base.clone()
        },
        GatherDimensionNumbers {
            collapsed_slice_dims: vec![2],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_index_map: vec![2],
            ..base.clone()
        },
        GatherDimensionNumbers {
            start_index_map: vec![0, 0],
            ..base.clone()
        },
        GatherDimensionNumbers {
            index_vector_dim: 3,
            ..base.clone()
        },
        GatherDimensionNumbers {
            index_vector_dim: -1,
            ..base.clone()
        },
        GatherDimensionNumbers {
            operand_batching_dims: vec![0],
            start_indices_batching_dims: vec![0],
            ..base
        },
    ];
    for dimensions in invalid {
        assert!(
            computation(
                StableHloOp::Gather,
                vec![
                    tensor(ElementType::F32, &[5, 4]),
                    tensor(ElementType::I64, &[2, 1])
                ],
                vec![tensor(ElementType::F32, &[2, 4])],
                attributes(dimensions, Some(&[1, 4]))
            )
            .is_err()
        );
    }
}

#[test]
fn gather_supports_batching_with_nontrivial_output_positions() {
    let dimensions = GatherDimensionNumbers {
        offset_dims: vec![3, 4],
        collapsed_slice_dims: vec![1],
        operand_batching_dims: vec![0],
        start_indices_batching_dims: vec![1],
        start_index_map: vec![2, 1],
        index_vector_dim: 3,
    };
    assert!(
        computation(
            StableHloOp::Gather,
            vec![
                tensor(ElementType::I32, &[2, 3, 4, 2]),
                tensor(ElementType::I64, &[2, 2, 3, 2])
            ],
            vec![tensor(ElementType::I32, &[2, 2, 3, 2, 2])],
            attributes(dimensions.clone(), Some(&[1, 1, 2, 2]))
        )
        .is_ok()
    );
    for (input_batch, index_batch, accepted) in [(3, 1, false), (2, 3, false), (2, 4, false)] {
        let mut dimensions = dimensions.clone();
        dimensions.start_indices_batching_dims = vec![index_batch];
        assert_eq!(
            computation(
                StableHloOp::Gather,
                vec![
                    tensor(ElementType::I32, &[input_batch, 3, 4, 2]),
                    tensor(ElementType::I64, &[2, 2, 3, 2])
                ],
                vec![tensor(ElementType::I32, &[2, 2, 3, 2, 2])],
                attributes(dimensions, Some(&[1, 1, 2, 2]))
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn dynamic_gather_checks_shape_vector_length_and_known_batch_dimensions() {
    for (slice_shape, output, accepted) in [
        (vec![2], vec![2, 4], true),
        (vec![1], vec![2, 4], false),
        (vec![2], vec![3, 4], false),
        (vec![2], vec![2, 4, 1], false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::DynamicGather,
                vec![
                    tensor(ElementType::F32, &[5, 4]),
                    tensor(ElementType::I64, &[2, 1]),
                    tensor(ElementType::I64, &slice_shape)
                ],
                vec![tensor(ElementType::F32, &output)],
                attributes(row_gather(), None)
            )
            .is_ok(),
            accepted
        );
    }
}
