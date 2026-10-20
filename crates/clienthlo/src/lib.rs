//! Native pliron implementation of the OpenXLA CHLO dialect.
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]
pub use pliron;
pub mod attributes;
mod error;
pub use error::{Diagnostic, RecoveryCategory};
pub mod inference;
pub mod ops;

#[cfg(test)]
mod semantics_tests;

#[cfg(test)]
mod tests {
    use super::{Diagnostic, RecoveryCategory, attributes::*, inference, ops::*};
    use pliron::{common_traits::Verify, context::Context, op::Op, r#type::Typed};
    use stablehlo::types::{ElementType, tensor};

    fn value(ctx: &mut Context, shape: &[i64], element: ElementType) -> pliron::value::Value {
        let ty = tensor(ctx, shape, element);
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| usize::try_from(*d).ok()?.checked_mul(n))
            .unwrap_or(1);
        let op = stablehlo::ops::ConstantOp::new(ctx, ty, vec![0; count.max(1)], false);
        op.get_operation().deref(ctx).get_result(0)
    }

    #[test]
    fn unary_domains_and_compatibility_are_checked() {
        let mut ctx = Context::new();
        let f = value(&mut ctx, &[3], ElementType::F32);
        let erf = ErfOp::new_inferred(&mut ctx, f).unwrap();
        erf.verify(&ctx).unwrap();
        let int = value(&mut ctx, &[3], ElementType::I32);
        let error = ErfOp::new_inferred(&mut ctx, int).err().unwrap();
        assert_eq!(
            error
                .err
                .downcast_ref::<Diagnostic>()
                .unwrap()
                .recovery_category(),
            RecoveryCategory::InvalidInput
        );
        let complex = value(&mut ctx, &[3], ElementType::C64);
        AcosOp::new_inferred(&mut ctx, complex)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        IsInfOp::new_inferred(&mut ctx, f)
            .unwrap()
            .verify(&ctx)
            .unwrap();
    }
    #[test]
    fn broadcast_inference_handles_dynamic_degenerate_and_bad_shapes() {
        assert_eq!(
            inference::numpy_broadcast_shape(&[&[-1, 1, 4], &[3, 1]]),
            Some(vec![-1, 3, 4])
        );
        assert_eq!(
            inference::numpy_broadcast_shape(&[&[0], &[1]]),
            Some(vec![0])
        );
        assert_eq!(inference::numpy_broadcast_shape(&[&[2, 3], &[4, 3]]), None);
        assert_eq!(
            inference::infer_broadcast_shape(&[2, 3, 4], &[4, 3], Some(&[2, 1])),
            Some(vec![2, 3, 4])
        );
        assert_eq!(
            inference::infer_broadcast_shape(&[2, 3, 4], &[3, 4], Some(&[3, 1])),
            None
        );
    }
    #[test]
    fn broadcast_operation_checks_inferred_result() {
        let mut ctx = Context::new();
        let lhs = value(&mut ctx, &[2, 1], ElementType::F32);
        let rhs = value(&mut ctx, &[3], ElementType::F32);
        BroadcastAddOp::new_inferred(&mut ctx, lhs, rhs, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        let explicit_lhs = value(&mut ctx, &[2, 3, 4], ElementType::F32);
        let explicit_rhs = value(&mut ctx, &[3, 4], ElementType::F32);
        BroadcastAddOp::new_inferred(&mut ctx, explicit_lhs, explicit_rhs, Some(vec![1, 2]))
            .unwrap()
            .verify(&ctx)
            .unwrap();
        // Upstream falls back to an unranked result for illegal broadcasts.
        let unranked = |ctx: &Context, op: &BroadcastAddOp| {
            let ty = op.get_operation().deref(ctx).get_result(0).get_type(ctx);
            inference::shaped(ctx, ty).is_some_and(|s| s.dimensions.is_none())
        };
        let out_of_range =
            BroadcastAddOp::new_inferred(&mut ctx, explicit_lhs, explicit_rhs, Some(vec![1, 3]))
                .unwrap();
        assert!(unranked(&ctx, &out_of_range));
        out_of_range.verify(&ctx).unwrap();
        let bad = value(&mut ctx, &[4, 3], ElementType::F32);
        let mismatched = BroadcastAddOp::new_inferred(&mut ctx, lhs, bad, None).unwrap();
        assert!(unranked(&ctx, &mismatched));
        mismatched.verify(&ctx).unwrap();
        let cmp = BroadcastCompareOp::new_inferred(
            &mut ctx,
            lhs,
            rhs,
            None,
            ComparisonDirection::Gt,
            None,
        )
        .unwrap();
        assert_eq!(
            stablehlo::types::tensor_parts(
                &ctx,
                cmp.get_operation().deref(&ctx).get_result(0).get_type(&ctx)
            ),
            Some((vec![2, 3], ElementType::I1))
        );
        cmp.verify(&ctx).unwrap();
    }
    #[test]
    fn broadcast_select_and_complex_infer_element_types() {
        let mut ctx = Context::new();
        let pred = value(&mut ctx, &[2, 1], ElementType::I1);
        let a = value(&mut ctx, &[1, 3], ElementType::F32);
        let b = value(&mut ctx, &[2, 3], ElementType::F32);
        BroadcastSelectOp::new_inferred(&mut ctx, pred, a, b)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        BroadcastComplexOp::new_inferred(&mut ctx, a, a, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
    }
    #[test]
    fn constants_and_top_k_verify_payload_and_shapes() {
        let mut ctx = Context::new();
        let c =
            ConstantOp::new_inferred(&mut ctx, vec![4], ElementType::F32, vec![0], true).unwrap();
        c.verify(&ctx).unwrap();
        let bad_ty = tensor(&ctx, &[4], ElementType::F32);
        let bad = ConstantOp::new(&mut ctx, bad_ty, vec![0, 1], false);
        assert!(bad.verify(&ctx).is_err());
        let input = value(&mut ctx, &[2, 5], ElementType::F32);
        TopKOp::new_inferred(&mut ctx, input, 3, true)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        assert!(TopKOp::new_inferred(&mut ctx, input, 6, true).is_err());
    }
    #[test]
    fn ragged_dot_infers_contracting_mode_and_rejects_bad_group_shape() {
        let mut ctx = Context::new();
        let lhs = value(&mut ctx, &[2, 3, 4], ElementType::F32);
        let rhs = value(&mut ctx, &[2, 4, 5], ElementType::F32);
        let groups = value(&mut ctx, &[2, 2], ElementType::I32);
        let dims = RaggedDotDimensionNumbersAttr::new(
            &ctx,
            vec![0],
            vec![0],
            vec![2],
            vec![1],
            vec![2],
            vec![],
        );
        let op = RaggedDotOp::new_inferred(&mut ctx, lhs, rhs, groups, dims, None).unwrap();
        assert_eq!(
            stablehlo::types::tensor_parts(
                &ctx,
                op.get_operation().deref(&ctx).get_result(0).get_type(&ctx)
            ),
            Some((vec![2, 2, 3, 5], ElementType::F32))
        );
        op.verify(&ctx).unwrap();
        let mode1_rhs = value(&mut ctx, &[5, 2, 4, 6], ElementType::F32);
        let mode1_groups = value(&mut ctx, &[2, 5], ElementType::I32);
        let mode1_dims = RaggedDotDimensionNumbersAttr::new(
            &ctx,
            vec![0],
            vec![1],
            vec![2],
            vec![2],
            vec![1],
            vec![0],
        );
        let mode1 =
            RaggedDotOp::new_inferred(&mut ctx, lhs, mode1_rhs, mode1_groups, mode1_dims, None)
                .unwrap();
        assert_eq!(
            stablehlo::types::tensor_parts(
                &ctx,
                mode1
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
            ),
            Some((vec![2, 3, 6], ElementType::F32))
        );
        mode1.verify(&ctx).unwrap();
        let mode3_groups = value(&mut ctx, &[7], ElementType::I32);
        let mode3_dims = RaggedDotDimensionNumbersAttr::new(
            &ctx,
            vec![0],
            vec![0],
            vec![2],
            vec![1],
            vec![0],
            vec![],
        );
        let mode3 =
            RaggedDotOp::new_inferred(&mut ctx, lhs, rhs, mode3_groups, mode3_dims, None).unwrap();
        assert_eq!(
            stablehlo::types::tensor_parts(
                &ctx,
                mode3
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
            ),
            Some((vec![2, 3, 5], ElementType::F32))
        );
        mode3.verify(&ctx).unwrap();
        let bad = value(&mut ctx, &[3, 2], ElementType::I32);
        let dims = RaggedDotDimensionNumbersAttr::new(
            &ctx,
            vec![0],
            vec![0],
            vec![2],
            vec![1],
            vec![2],
            vec![],
        );
        assert!(RaggedDotOp::new_inferred(&mut ctx, lhs, rhs, bad, dims, None).is_err());
    }
    #[test]
    fn scan_checks_body_signature_and_return() {
        let mut ctx = Context::new();
        let input = value(&mut ctx, &[2, 3], ElementType::F32);
        let init = value(&mut ctx, &[], ElementType::F32);
        let scan = ScanOp::new_inferred(
            &mut ctx,
            vec![input],
            vec![init],
            0,
            None,
            false,
            None,
            |ctx, block| {
                let args: Vec<_> = block.deref(ctx).arguments().collect();
                let ret = stablehlo::ops::ReturnOp::new_many(ctx, args);
                ret.get_operation().insert_at_back(block, ctx);
                Ok(())
            },
        )
        .unwrap();
        scan.verify(&ctx).unwrap();
    }
    #[test]
    fn numeric_binary_and_logical_domains_are_checked() {
        let mut ctx = Context::new();
        let f = value(&mut ctx, &[2], ElementType::F32);
        let i = value(&mut ctx, &[2], ElementType::I32);
        NextAfterOp::new_inferred(&mut ctx, f, f)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        PolygammaOp::new_inferred(&mut ctx, f, f)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        ZetaOp::new_inferred(&mut ctx, f, f)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        MulhiOp::new_inferred(&mut ctx, i, i)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        assert!(MulhiOp::new_inferred(&mut ctx, f, f).is_err());
        BroadcastAndOp::new_inferred(&mut ctx, i, i, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        assert!(BroadcastAndOp::new_inferred(&mut ctx, f, f, None).is_err());
        BroadcastShiftLeftOp::new_inferred(&mut ctx, i, i, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        let complex = value(&mut ctx, &[2], ElementType::C64);
        BroadcastAddOp::new_inferred(&mut ctx, complex, complex, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        let pred_lhs = value(&mut ctx, &[2], ElementType::I1);
        let pred_rhs = value(&mut ctx, &[2], ElementType::I1);
        BroadcastAndOp::new_inferred(&mut ctx, pred_lhs, pred_rhs, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        BroadcastRemainderOp::new_inferred(&mut ctx, i, i, None)
            .unwrap()
            .verify(&ctx)
            .unwrap();
    }

    #[test]
    fn constant_like_and_chlo_attributes_verify() {
        let mut ctx = Context::new();
        let operand = value(&mut ctx, &[2, 3], ElementType::F32);
        ConstantLikeOp::new_inferred(&mut ctx, vec![0x3f80_0000], ElementType::F32, operand)
            .unwrap()
            .verify(&ctx)
            .unwrap();
        assert!(
            RaggedDotDimensionNumbersAttr::new(
                &ctx,
                vec![-1],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![]
            )
            .verify(&ctx)
            .is_err()
        );
        assert!(Precision::from_u64(3).is_none());
        PrecisionAttr::new(Precision::Highest).verify(&ctx).unwrap();
    }

    #[test]
    fn scan_rejects_bad_body_argument_types_and_terminators() {
        let mut ctx = Context::new();
        let input = value(&mut ctx, &[2, 3], ElementType::F32);
        let init = value(&mut ctx, &[], ElementType::F32);
        let results = vec![input.get_type(&ctx), init.get_type(&ctx)];
        let scan = ScanOp::new(
            &mut ctx,
            vec![input],
            vec![init],
            results,
            0,
            None,
            false,
            None,
        );
        let block = scan.entry_block(&ctx).unwrap();
        let arg = block.deref(&ctx).arguments().next().unwrap();
        arg.set_type(&ctx, tensor(&ctx, &[4], ElementType::F32));
        let args: Vec<_> = block.deref(&ctx).arguments().collect();
        let ret = stablehlo::ops::ReturnOp::new_many(&mut ctx, args);
        ret.get_operation().insert_at_back(block, &ctx);
        assert!(scan.verify(&ctx).is_err());

        let results = vec![input.get_type(&ctx), init.get_type(&ctx)];
        let scan = ScanOp::new(
            &mut ctx,
            vec![input],
            vec![init],
            results,
            0,
            None,
            false,
            None,
        );
        let block = scan.entry_block(&ctx).unwrap();
        let arg = block.deref(&ctx).arguments().next().unwrap();
        let ty = tensor(&ctx, &[3], ElementType::F32);
        let not_terminator = ErfOp::new(&mut ctx, arg, ty);
        not_terminator.get_operation().insert_at_back(block, &ctx);
        assert!(scan.verify(&ctx).is_err());
    }
}
