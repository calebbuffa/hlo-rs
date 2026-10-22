use crate::control_support::*;
use xlahlo::{
    Diagnostic,
    inference::control::collective as infer,
    shape::{self, Dimension, Layout},
    types::ElementType,
};

#[test]
fn broadcast_root_requires_one_bounded_index_per_data_operand() {
    let ctx = context();
    let data = shape(&ctx, ElementType::BF16, &[4]);
    let unbounded =
        shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Unbounded]).unwrap();
    let error = infer::infer_collective_broadcast(&ctx, &[data, unbounded], true).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::Inference { .. })
    ));
    for root in [
        shape(&ctx, ElementType::S32, &[]),
        shape(&ctx, ElementType::S32, &[0]),
        shape(&ctx, ElementType::S32, &[2]),
        shape(&ctx, ElementType::S32, &[1, 1]),
        shape(&ctx, ElementType::U32, &[1]),
        shape::token(&ctx),
    ] {
        assert!(infer::infer_collective_broadcast(&ctx, &[data, root], true).is_err());
    }
    let root = shape(&ctx, ElementType::S32, &[1]);
    assert_eq!(
        infer::infer_collective_broadcast(&ctx, &[data, root], true).unwrap(),
        data
    );
}

#[test]
fn broadcast_accepts_bounded_dynamic_roots_without_restricting_layout() {
    let ctx = context();
    let data = shape(&ctx, ElementType::F6E2M3FN, &[3]);
    for root in [
        shape(&ctx, ElementType::S32, &[1]),
        shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Bounded(1)]).unwrap(),
    ] {
        let root = shape::with_layout(
            &ctx,
            root,
            Layout {
                minor_to_major: vec![0],
                memory_space: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            infer::infer_collective_broadcast(&ctx, &[data, root], true).unwrap(),
            data
        );
        assert!(infer::infer_collective_reduce(&ctx, &[data, root], true).is_err());
    }
}

#[test]
fn dynamic_root_variadic_collectives_preserve_every_payload_format() {
    let ctx = context();
    let root = shape::with_default_layout(&ctx, shape(&ctx, ElementType::S32, &[2])).unwrap();
    for &element in ElementType::ALL {
        let data = shape::with_layout(
            &ctx,
            shape(&ctx, element, &[2, 3]),
            Layout {
                minor_to_major: vec![0, 1],
                memory_space: 4,
                ..Default::default()
            },
        )
        .unwrap();
        let expected = shape::tuple(&ctx, vec![data, data]).unwrap();
        assert_eq!(
            infer::infer_collective_broadcast(&ctx, &[data, data, root], true).unwrap(),
            expected,
            "{element:?}"
        );
        assert_eq!(
            infer::infer_collective_reduce(&ctx, &[data, data, root], true).unwrap(),
            expected,
            "{element:?}"
        );
        let unbounded =
            shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Unbounded]).unwrap();
        assert!(infer::infer_collective_broadcast(&ctx, &[data, data, unbounded], true).is_err());
    }
}

#[test]
fn roots_do_not_replace_required_data_or_relax_reduce_geometry() {
    let ctx = context();
    let root = shape::with_default_layout(&ctx, shape(&ctx, ElementType::S32, &[1])).unwrap();
    for operands in [&[][..], &[root][..]] {
        assert!(infer::infer_collective_broadcast(&ctx, operands, true).is_err());
        assert!(infer::infer_collective_reduce(&ctx, operands, true).is_err());
    }
    let data = shape(&ctx, ElementType::C128, &[2]);
    assert_eq!(
        infer::infer_collective_reduce(&ctx, &[data, root], true).unwrap(),
        data
    );
    let dynamic =
        shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Bounded(1)]).unwrap();
    let dynamic = shape::with_default_layout(&ctx, dynamic).unwrap();
    assert!(infer::infer_collective_reduce(&ctx, &[data, dynamic], true).is_err());
    assert!(infer::infer_collective_reduce(&ctx, &[shape::token(&ctx), root], true).is_err());
}
