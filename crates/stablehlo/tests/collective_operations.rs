use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, ChannelHandle, Computation, ElementType,
    Instruction, Operation, Region, ScalarValue, StableHloAttribute, StableHloOp, TensorType, Type,
    ValueId,
};

mod common;
use common::{computation, computation_with_regions};

fn tensor(element: ElementType, shape: &[u64]) -> Type {
    TensorType::static_shape(element, shape.iter().copied()).into()
}

fn dense(shape: &[u64], values: &[i128]) -> Attribute {
    Attribute::DenseElements {
        ty: Box::new(tensor(ElementType::I64, shape)),
        values: values.iter().copied().map(ScalarValue::Integer).collect(),
    }
}

fn groups() -> Attributes {
    [("replica_groups".into(), dense(&[1, 2], &[0, 1]))].into()
}

fn reducer() -> Region {
    let scalar: Type = TensorType::scalar(ElementType::F32).into();
    Region::new(vec![Block::new(
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
    )])
}

fn all_reduce(attributes: Attributes) -> Result<Computation, stablehlo::BuildError> {
    computation_with_regions(
        StableHloOp::AllReduce,
        vec![tensor(ElementType::F32, &[2, 3])],
        vec![tensor(ElementType::F32, &[2, 3])],
        attributes,
        vec![reducer()],
    )
}

#[test]
fn replica_groups_are_rank_two_unique_and_cover_contiguous_device_ids() {
    assert!(all_reduce(groups()).is_ok());
    for value in [
        dense(&[2], &[0, 1]),
        dense(&[1, 2], &[0, 0]),
        dense(&[1, 2], &[0, 2]),
        dense(&[1, 2], &[-2, 0]),
    ] {
        assert!(all_reduce([("replica_groups".into(), value)].into()).is_err());
    }
    let nonuniform = [("replica_groups".into(), dense(&[2, 2], &[0, 1, 2, -1]))].into();
    assert!(all_reduce(nonuniform).is_ok());
}

#[test]
fn global_device_ids_require_positive_channels_and_nonempty_groups() {
    for (channel, accepted) in [
        (None, false),
        (Some(0), false),
        (Some(-1), false),
        (Some(1), true),
    ] {
        let mut attributes = groups();
        attributes.insert("use_global_device_ids".into(), Attribute::Unit);
        if let Some(handle) = channel {
            attributes.insert(
                "channel_handle".into(),
                Attribute::StableHlo(Box::new(StableHloAttribute::ChannelHandle(ChannelHandle {
                    handle,
                    channel_type: 0,
                }))),
            );
        }
        assert_eq!(all_reduce(attributes).is_ok(), accepted);
    }
    let mut attributes = groups();
    attributes.insert("replica_groups".into(), dense(&[0, 0], &[]));
    attributes.insert("use_global_device_ids".into(), Attribute::Unit);
    attributes.insert(
        "channel_handle".into(),
        Attribute::StableHlo(Box::new(StableHloAttribute::ChannelHandle(ChannelHandle {
            handle: 1,
            channel_type: 0,
        }))),
    );
    assert!(all_reduce(attributes).is_err());
}

#[test]
fn all_gather_uses_group_cardinality_and_checks_every_non_gather_axis() {
    let mut attributes = groups();
    attributes.insert("all_gather_dim".into(), Attribute::integer(1));
    for (output, accepted) in [
        (vec![2, 6], true),
        (vec![2, 9], false),
        (vec![4, 6], false),
        (vec![2, 6, 1], false),
        (vec![2, 0], false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::AllGather,
                vec![tensor(ElementType::F32, &[2, 3])],
                vec![tensor(ElementType::F32, &output)],
                attributes.clone()
            )
            .is_ok(),
            accepted
        );
    }
    attributes.insert("replica_groups".into(), dense(&[2, 2], &[0, 1, 2, -1]));
    assert!(
        computation(
            StableHloOp::AllGather,
            vec![tensor(ElementType::F32, &[2, 3])],
            vec![tensor(ElementType::F32, &[2, 6])],
            attributes
        )
        .is_err()
    );
}

#[test]
fn reduce_scatter_checks_group_size_geometry_and_reducer_result_element() {
    let mut attributes = groups();
    attributes.insert("scatter_dimension".into(), Attribute::integer(1));
    for (output, accepted) in [
        (tensor(ElementType::F32, &[2, 3]), true),
        (tensor(ElementType::F32, &[2, 2]), false),
        (tensor(ElementType::F32, &[4, 3]), false),
        (tensor(ElementType::F32, &[2, 0]), false),
        (tensor(ElementType::F16, &[2, 3]), false),
    ] {
        assert_eq!(
            computation_with_regions(
                StableHloOp::ReduceScatter,
                vec![tensor(ElementType::F16, &[2, 6])],
                vec![output],
                attributes.clone(),
                vec![reducer()]
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn all_to_all_splits_concatenates_and_preserves_same_axis_extents() {
    for (split, concat, output, accepted) in [
        (0, 1, vec![2, 6], true),
        (1, 0, vec![8, 1], false),
        (0, 0, vec![4, 3], true),
        (0, 1, vec![4, 6], false),
        (2, 0, vec![4, 3], false),
    ] {
        let mut attributes = groups();
        attributes.insert("split_dimension".into(), Attribute::integer(split));
        attributes.insert("concat_dimension".into(), Attribute::integer(concat));
        attributes.insert("split_count".into(), Attribute::integer(2));
        assert_eq!(
            computation(
                StableHloOp::AllToAll,
                vec![tensor(ElementType::F32, &[4, 3])],
                vec![tensor(ElementType::F32, &output)],
                attributes
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn all_to_all_checks_group_cardinality_and_output_overflow() {
    let mut attributes = groups();
    attributes.insert("split_dimension".into(), Attribute::integer(0));
    attributes.insert("concat_dimension".into(), Attribute::integer(1));
    attributes.insert("split_count".into(), Attribute::integer(3));
    assert!(
        computation(
            StableHloOp::AllToAll,
            vec![tensor(ElementType::F32, &[6, 3])],
            vec![tensor(ElementType::F32, &[2, 9])],
            attributes.clone()
        )
        .is_err()
    );
    attributes.insert("split_count".into(), Attribute::integer(2));
    assert!(
        computation(
            StableHloOp::AllToAll,
            vec![tensor(ElementType::F32, &[2, i64::MAX as u64])],
            vec![TensorType::new(ElementType::F32, vec![Some(1), None]).into()],
            attributes
        )
        .is_err()
    );
}

#[test]
fn collective_permute_checks_pairs_and_preserves_tensor_types() {
    for (pairs, output, accepted) in [
        (
            dense(&[2, 2], &[0, 1, 1, 2]),
            tensor(ElementType::F32, &[2, 3]),
            true,
        ),
        (
            dense(&[2, 2], &[0, 1, 0, 2]),
            tensor(ElementType::F32, &[2, 3]),
            false,
        ),
        (
            dense(&[2, 2], &[0, 1, 2, 1]),
            tensor(ElementType::F32, &[2, 3]),
            false,
        ),
        (
            dense(&[1, 2], &[-1, 0]),
            tensor(ElementType::F32, &[2, 3]),
            false,
        ),
        (
            dense(&[2], &[0, 1]),
            tensor(ElementType::F32, &[2, 3]),
            false,
        ),
        (
            dense(&[1, 2], &[0, 1]),
            tensor(ElementType::F32, &[3, 2]),
            false,
        ),
    ] {
        assert_eq!(
            computation(
                StableHloOp::CollectivePermute,
                vec![tensor(ElementType::F32, &[2, 3])],
                vec![output],
                [("source_target_pairs".into(), pairs)].into()
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn collective_broadcast_allows_partial_groups_and_excludes_dynamic_roots_from_results() {
    let mut attributes: Attributes = [("replica_groups".into(), dense(&[1, 2], &[2, 1]))].into();
    assert!(
        computation(
            StableHloOp::CollectiveBroadcast,
            vec![tensor(ElementType::F32, &[3])],
            vec![tensor(ElementType::F32, &[3])],
            attributes.clone()
        )
        .is_ok()
    );
    attributes.insert("has_dynamic_root".into(), Attribute::Unit);
    for (root, accepted) in [
        (tensor(ElementType::I32, &[1]), true),
        (tensor(ElementType::I32, &[2]), false),
        (tensor(ElementType::I64, &[1]), false),
        (tensor(ElementType::I32, &[]), false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::CollectiveBroadcast,
                vec![tensor(ElementType::F32, &[3]), root],
                vec![tensor(ElementType::F32, &[3])],
                attributes.clone()
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn collective_reduce_uses_only_data_operands_in_the_scalar_reducer() {
    let mut attributes = groups();
    attributes.insert("has_dynamic_root".into(), Attribute::Unit);
    assert!(
        computation_with_regions(
            StableHloOp::CollectiveReduce,
            vec![
                tensor(ElementType::F16, &[3]),
                tensor(ElementType::F32, &[4]),
                tensor(ElementType::I32, &[2])
            ],
            vec![
                tensor(ElementType::F32, &[3]),
                tensor(ElementType::F32, &[4])
            ],
            attributes,
            vec![reducer()]
        )
        .is_ok()
    );
}
