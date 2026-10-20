#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    Diagnostic,
    attributes::{RandomAlgorithm, RandomDistribution},
    inference::control as infer,
    ops::{
        HloOpInterface,
        control::{RngBitGeneratorOp, RngGetAndUpdateStateOp, RngOp},
    },
    pliron::{builtin::attributes::StringAttr, common_traits::Verify, op::Op},
    shape::{Dimension, Layout, Shape},
    types::ElementType,
    verification::VerifyOptions,
};

#[test]
fn distribution_domains_cover_every_primitive_without_backend_limits() {
    for &element in ElementType::ALL {
        let mut ctx = context();
        let scalar = shape(&ctx, element, &[]);
        let output = shape(&ctx, element, &[2, 3]);
        let parameter = param(&mut ctx, scalar, 0);
        let value = parameter.result(&ctx);
        for (distribution, accepted) in [
            (RandomDistribution::Uniform, !element.is_complex()),
            (RandomDistribution::Normal, element.is_float()),
        ] {
            let inferred = infer::infer_rng(&ctx, &[scalar, scalar], output, distribution, false);
            assert_eq!(inferred.is_ok(), accepted, "{element:?} {distribution:?}");
            if let Ok(result) = inferred {
                assert_eq!(result, output);
            }
            let op = RngOp::new(&mut ctx, output, vec![value, value], distribution);
            assert_eq!(
                op.verify(&ctx).is_ok(),
                accepted,
                "{element:?} {distribution:?}"
            );
        }
    }
}

#[test]
fn mixed_precision_changes_only_real_float_results_and_preserves_layout() {
    let mut ctx = context();
    for &input in ElementType::ALL.iter().filter(|element| element.is_float()) {
        let scalar = shape(&ctx, input, &[]);
        let parameter = param(&mut ctx, scalar, 0);
        let value = parameter.result(&ctx);
        for &output in ElementType::ALL.iter().filter(|element| element.is_float()) {
            let result = Shape::with_layout(
                &ctx,
                Shape::dynamic_array(&ctx, output, vec![Dimension::Bounded(4)]).unwrap(),
                Layout {
                    minor_to_major: vec![0],
                    memory_space: 5,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                infer::infer_rng(
                    &ctx,
                    &[scalar, scalar],
                    result,
                    RandomDistribution::Normal,
                    true,
                )
                .unwrap(),
                result
            );
            let op = RngOp::new(
                &mut ctx,
                result,
                vec![value, value],
                RandomDistribution::Normal,
            );
            assert_eq!(op.verify(&ctx).is_ok(), input == output);
            op.verify_with_options(
                &ctx,
                VerifyOptions {
                    allow_mixed_precision: true,
                    layout_sensitive: true,
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    let f32 = shape(&ctx, ElementType::F32, &[]);
    let bf16 = shape(&ctx, ElementType::BF16, &[]);
    let integer = shape(&ctx, ElementType::S32, &[]);
    for operands in [&[f32, bf16][..], &[integer, integer][..]] {
        assert!(infer::infer_rng(&ctx, operands, bf16, RandomDistribution::Uniform, true).is_err());
    }
}

#[test]
fn malformed_distribution_parameters_return_typed_errors_and_allow_recovery() {
    let mut ctx = context();
    let scalar = shape(&ctx, ElementType::F32, &[]);
    let vector = shape(&ctx, ElementType::F32, &[1]);
    for operands in [vec![], vec![scalar], vec![scalar; 3], vec![scalar, vector]] {
        let error = infer::infer_rng(&ctx, &operands, vector, RandomDistribution::Normal, false)
            .unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::Inference { .. })
        ));
    }
    let tuple = Shape::tuple(&ctx, vec![scalar]).unwrap();
    assert!(
        infer::infer_rng(
            &ctx,
            &[tuple, scalar],
            vector,
            RandomDistribution::Normal,
            false
        )
        .is_err()
    );
    assert!(
        infer::infer_rng(
            &ctx,
            &[scalar, scalar],
            tuple,
            RandomDistribution::Normal,
            false
        )
        .is_err()
    );
    assert_eq!(
        infer::infer_rng(
            &ctx,
            &[scalar, scalar],
            vector,
            RandomDistribution::Normal,
            false
        )
        .unwrap(),
        vector
    );
    let parameter = param(&mut ctx, scalar, 0);
    let value = parameter.result(&ctx);
    let op = RngOp::new(
        &mut ctx,
        vector,
        vec![value, value],
        RandomDistribution::Normal,
    );
    attr(
        &ctx,
        op.get_operation(),
        "hlo_distribution",
        StringAttr::new("normal".into()),
    );
    assert!(op.verify(&ctx).is_err());
    attr(
        &ctx,
        op.get_operation(),
        "hlo_distribution",
        RandomDistribution::Normal,
    );
    op.verify(&ctx).unwrap();
    for operands in [vec![], vec![value], vec![value; 3]] {
        assert!(
            RngOp::new(&mut ctx, vector, operands, RandomDistribution::Normal)
                .verify(&ctx)
                .is_err()
        );
    }
}

#[test]
fn bit_generator_preserves_permissive_state_and_non_tuple_result_contract() {
    let mut ctx = context();
    let state = shape(&ctx, ElementType::U64, &[3]);
    let mut layout = Layout::descending(1);
    layout.memory_space = 4;
    let updated_state = Shape::with_layout(&ctx, state, layout).unwrap();
    let parameter = param(&mut ctx, state, 0);
    let value = parameter.result(&ctx);
    let data = shape(&ctx, ElementType::F4E2M1FN, &[5]);
    let tuple = Shape::tuple(&ctx, vec![updated_state, data]).unwrap();
    for algorithm in [
        RandomAlgorithm::Default,
        RandomAlgorithm::ThreeFry,
        RandomAlgorithm::Philox,
    ] {
        for result in [data, tuple] {
            assert_eq!(
                infer::infer_rng_bit_generator(&ctx, state, result).unwrap(),
                result
            );
            RngBitGeneratorOp::new(&mut ctx, result, vec![value], algorithm)
                .verify_with_options(
                    &ctx,
                    VerifyOptions {
                        layout_sensitive: true,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    for children in [
        vec![],
        vec![state],
        vec![state, data, data],
        vec![data, data],
    ] {
        let result = Shape::tuple(&ctx, children).unwrap();
        assert!(infer::infer_rng_bit_generator(&ctx, state, result).is_err());
        assert!(
            RngBitGeneratorOp::new(&mut ctx, result, vec![value], RandomAlgorithm::Default)
                .verify(&ctx)
                .is_err()
        );
    }
    assert!(
        RngBitGeneratorOp::new(&mut ctx, data, vec![], RandomAlgorithm::Default)
            .verify(&ctx)
            .is_err()
    );
    let op = RngBitGeneratorOp::new(&mut ctx, tuple, vec![value], RandomAlgorithm::Philox);
    attr(
        &ctx,
        op.get_operation(),
        "hlo_algorithm",
        StringAttr::new("philox".into()),
    );
    assert!(op.verify(&ctx).is_err());
    attr(
        &ctx,
        op.get_operation(),
        "hlo_algorithm",
        RandomAlgorithm::Philox,
    );
    op.verify(&ctx).unwrap();
}

#[test]
fn state_counter_has_fixed_shape_and_rejects_operands_or_other_results() {
    let mut ctx = context();
    let state = infer::infer_rng_get_and_update_state(&ctx).unwrap();
    assert_eq!(Shape::element_type(&ctx, state).unwrap(), ElementType::U64);
    assert_eq!(*Shape::dimensions(&ctx, state).unwrap(), [2]);
    RngGetAndUpdateStateOp::new(&mut ctx, state, vec![])
        .verify(&ctx)
        .unwrap();
    for result in [
        shape(&ctx, ElementType::U64, &[]),
        shape(&ctx, ElementType::U64, &[3]),
        shape(&ctx, ElementType::S64, &[2]),
    ] {
        assert!(
            RngGetAndUpdateStateOp::new(&mut ctx, result, vec![])
                .verify(&ctx)
                .is_err()
        );
    }
    let parameter = param(&mut ctx, state, 0);
    let value = parameter.result(&ctx);
    assert!(
        RngGetAndUpdateStateOp::new(&mut ctx, state, vec![value])
            .verify(&ctx)
            .is_err()
    );
}
