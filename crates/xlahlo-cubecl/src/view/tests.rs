use super::*;
use xlahlo::types::ElementType;

fn storage() -> (BufferId, BufferDescriptor) {
    let mut plan = crate::ExecutionPlanBuilder::new(Vec::new());
    let descriptor = BufferDescriptor::dense_row_major(ElementType::U32, vec![6]);
    (
        plan.add_buffer(crate::BufferRole::Input(0), descriptor.clone()),
        descriptor,
    )
}

#[test]
fn affine_output_validation_checks_signed_reach_type_and_rank() {
    let (buffer, storage) = storage();
    let descriptor = BufferDescriptor::dense_row_major(ElementType::U32, vec![3]);
    for (offset, strides) in [(0, vec![-1]), (5, vec![1]), (0, vec![])] {
        let view = OutputView {
            buffer,
            descriptor: descriptor.clone(),
            map: OutputIndexMap::Affine { offset, strides },
        };
        assert!(view.validate(&storage).is_err());
    }
    let valid = OutputView {
        buffer,
        descriptor,
        map: OutputIndexMap::Affine {
            offset: 5,
            strides: vec![-2],
        },
    };
    valid.validate(&storage).unwrap();
    let mut mismatch = valid;
    mismatch.descriptor.element_type = ElementType::F32;
    assert!(mismatch.validate(&storage).is_err());
}

#[test]
fn composed_output_validation_rejects_unsound_geometry_and_shape_chains() {
    let (buffer, storage) = storage();
    for kind in [
        IndexTransformKind::Transpose(vec![1]),
        IndexTransformKind::Slice {
            starts: vec![5],
            strides: vec![1],
        },
        IndexTransformKind::Slice {
            starts: vec![0],
            strides: vec![0],
        },
        IndexTransformKind::Reshape,
        IndexTransformKind::Broadcast(vec![1]),
    ] {
        let view = OutputView {
            buffer,
            descriptor: BufferDescriptor::dense_row_major(ElementType::U32, vec![3]),
            map: OutputIndexMap::Composed {
                storage_shape: vec![6],
                transforms: vec![IndexTransform {
                    source_shape: vec![6],
                    result_shape: vec![3],
                    kind,
                }],
            },
        };
        assert!(view.validate(&storage).is_err());
    }
    let view = OutputView {
        buffer,
        descriptor: storage.clone(),
        map: OutputIndexMap::Composed {
            storage_shape: vec![6],
            transforms: vec![IndexTransform {
                source_shape: vec![2, 3],
                result_shape: vec![6],
                kind: IndexTransformKind::Reshape,
            }],
        },
    };
    assert!(view.validate(&storage).is_err());
}

#[test]
fn view_geometry_rejects_overflow_and_negative_extents_without_panicking() {
    assert!(matches!(
        dense_strides(&[i64::MAX, 2]),
        Err(PlanError::SizeOverflow)
    ));
    assert!(matches!(dense_strides(&[-1]), Err(PlanError::InvalidShape)));
    let transform = IndexTransform {
        source_shape: vec![6],
        result_shape: vec![3],
        kind: IndexTransformKind::Slice {
            starts: vec![0],
            strides: vec![usize::MAX],
        },
    };
    assert!(matches!(
        transform.validate(ElementType::U32),
        Err(PlanError::SizeOverflow)
    ));
}

#[test]
fn output_views_require_produced_leaf_storage() {
    let mut plan = crate::ExecutionPlanBuilder::new(Vec::new());
    let descriptor = BufferDescriptor::dense_row_major(ElementType::U32, vec![6]);
    let buffer = plan.add_buffer(crate::BufferRole::Temporary, descriptor.clone());
    plan.add_output(crate::PlannedOutput::View(OutputView {
        buffer,
        descriptor,
        map: OutputIndexMap::Affine {
            offset: 0,
            strides: vec![1],
        },
    }));
    assert!(matches!(
        plan.build(),
        Err(PlanError::UnproducedOutput(_) | PlanError::MissingProducer { .. })
    ));
}
