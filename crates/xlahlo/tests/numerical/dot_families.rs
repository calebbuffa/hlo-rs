use super::super::*;

#[test]
fn dot_inference_uses_default_layout_instead_of_copying_operands() {
    let ctx = context();
    let lhs = Shape::with_layout(
        &ctx,
        a(&ctx, P::F32, &[2, 3]),
        Layout {
            minor_to_major: vec![0, 1],
            ..Default::default()
        },
    )
    .unwrap();
    let rhs = default_layout(&ctx, a(&ctx, P::F32, &[3, 4]));
    let result =
        inference::infer_dot(&ctx, lhs, rhs, &dot_dims(&[1], &[0], &[], &[]), None).unwrap();
    assert_eq!(dims(&ctx, result), [2, 4]);
    assert_eq!(
        Shape::layout(&ctx, result).unwrap().unwrap().minor_to_major,
        [1, 0]
    );
}

#[test]
fn dot_contraction_geometry_includes_scalar_outer_batched_and_dynamic_cases() {
    let ctx = context();
    for m in 0..4 {
        for k in 0..4 {
            for n in 0..4 {
                let result = inference::infer_dot(
                    &ctx,
                    a(&ctx, P::F32, &[m, k]),
                    a(&ctx, P::F32, &[k, n]),
                    &dot_dims(&[1], &[0], &[], &[]),
                    None,
                )
                .unwrap();
                assert!(Shape::equal_ignoring_layout(
                    &ctx,
                    result,
                    a(&ctx, P::F32, &[m, n])
                ));
            }
        }
    }
    let left = dynamic(&ctx, P::F32, &[D::Bounded(2), D::Static(3), D::Static(4)]);
    let right = a(&ctx, P::F32, &[2, 4, 5]);
    let d = dot_dims(&[2], &[1], &[0], &[0]);
    let result = inference::infer_dot(&ctx, left, right, &d, Some(P::F64)).unwrap();
    assert_eq!(dims(&ctx, result), [2, 3, 5]);
    assert_eq!(Shape::dimension(&ctx, result, 0).unwrap(), D::Bounded(2));
    assert_eq!(Shape::element_type(&ctx, result).unwrap(), P::F64);
    let scalar = s(&ctx, P::F32);
    let empty = dot_dims(&[], &[], &[], &[]);
    assert_eq!(
        Shape::rank(
            &ctx,
            inference::infer_dot(&ctx, scalar, scalar, &empty, None).unwrap()
        )
        .unwrap(),
        0
    );
    assert_eq!(
        dims(
            &ctx,
            inference::infer_dot(&ctx, left, scalar, &empty, None).unwrap()
        ),
        dims(&ctx, left)
    );
    for malformed in [
        dot_dims(&[2], &[], &[], &[]),
        dot_dims(&[2], &[0], &[], &[]),
        dot_dims(&[2], &[1], &[2], &[0]),
        dot_dims(&[99], &[1], &[], &[]),
    ] {
        assert!(inference::infer_dot(&ctx, left, right, &malformed, None).is_err());
    }
}

#[test]
fn dot_text_retains_dimension_numbers_precision_and_algorithm() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 3, 4]);
    let rhs = a(&ctx, P::F32, &[2, 4, 5]);
    let result = a(&ctx, P::F32, &[2, 3, 5]);
    let d = dot_dims(&[2], &[1], &[0], &[0]);
    for algorithm in [
        Algorithm::Unset,
        Algorithm::DotAnyF8AnyF8F32,
        Algorithm::DotAnyF8AnyF8F32FastAccum,
        Algorithm::DotF16F16F16,
        Algorithm::DotF16F16F32,
        Algorithm::DotBF16BF16BF16,
        Algorithm::DotBF16BF16F32,
        Algorithm::DotBF16BF16F32X3,
        Algorithm::DotBF16BF16F32X6,
        Algorithm::DotTF32TF32F32,
        Algorithm::DotTF32TF32F32X3,
        Algorithm::DotF32F32F32,
        Algorithm::DotF64F64F64,
        Algorithm::DotBF16BF16F32X9,
        Algorithm::DotBF16BF16FP8X3,
        Algorithm::DotBF16BF16FP8X4,
    ] {
        let config = PrecisionConfig {
            operand_precision: OperandPrecision(vec![Precision::High, Precision::Highest]),
            algorithm,
        };
        roundtrip(&mut ctx, config.clone());
        let args = values(&mut ctx, &[lhs, rhs]);
        let op = DotOp::new(&mut ctx, result, args, d.clone(), config.clone());
        op.verify(&ctx).unwrap();
        assert_eq!(op.precision(&ctx).as_deref(), Some(&config));
    }
    let args = values(&mut ctx, &[lhs]);
    assert!(
        DotOp::new(&mut ctx, result, args, d, precision())
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn dot_requires_the_canonical_typed_precision_configuration() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[4, 3]);
    let result = a(&ctx, P::F32, &[2, 3]);
    let args = values(&mut ctx, &[lhs, rhs]);
    let op = DotOp::new(
        &mut ctx,
        result,
        args,
        dot_dims(&[1], &[0], &[], &[]),
        precision(),
    );
    op.verify(&ctx).unwrap();
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_precision"));
    error_contains(op.verify(&ctx), "missing hlo_precision");
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_precision"), Axes(vec![]));
    let error = op.verify(&ctx).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<xlahlo::Diagnostic>(),
        Some(xlahlo::Diagnostic::InvalidAttribute {
            attribute: "hlo_precision",
            ..
        }),
    ));
}

#[test]
fn unsorted_dot_pairings_retain_declared_batch_and_free_axis_order() {
    let ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 3, 4, 5]);
    let rhs = a(&ctx, P::F32, &[7, 5, 11, 3]);
    for (batch, ragged, rhs_groups, expected) in [
        (true, 3, vec![], vec![5, 3, 2, 4, 7, 11]),
        (false, 3, vec![], vec![1, 2, 4, 7, 11]),
        (false, 0, vec![0], vec![2, 4, 11]),
    ] {
        let dot = if batch {
            dot_dims(&[], &[], &[3, 1], &[1, 3])
        } else {
            dot_dims(&[3, 1], &[1, 3], &[], &[])
        };
        let ordinary = inference::infer_dot(&ctx, lhs, rhs, &dot, None).unwrap();
        assert_eq!(
            dims(&ctx, ordinary),
            if batch {
                vec![5, 3, 2, 4, 7, 11]
            } else {
                vec![2, 4, 7, 11]
            }
        );
        let groups = a(&ctx, P::S32, &[if rhs_groups.is_empty() { 1 } else { 7 }]);
        let config = RaggedDotDimensionNumbers {
            dot_dimension_numbers: dot,
            lhs_ragged_dimensions: Axes(vec![ragged]),
            rhs_group_dimensions: Axes(rhs_groups),
        };
        let result = inference::infer_ragged_dot(&ctx, lhs, rhs, groups, &config, None).unwrap();
        assert_eq!(dims(&ctx, result), expected);
    }
}

#[test]
fn high_rank_dot_and_ragged_modes_preserve_dimension_order() {
    let ctx = context();
    const RANK: usize = 32_768;
    let input = a(&ctx, P::F32, &vec![1; RANK]);
    let groups = a(&ctx, P::S32, &[1]);
    let axes = (0..RANK as i64).step_by(2).collect::<Vec<_>>();
    let reversed = axes.iter().rev().copied().collect::<Vec<_>>();
    let batch = dot_dims(&[], &[], &axes, &reversed);
    let contracting = dot_dims(&axes, &reversed, &[], &[]);
    for (dot, ragged, rhs_groups, output_rank) in [
        (batch, 0, vec![], RANK * 3 / 2),
        (contracting.clone(), 0, vec![], RANK + 1),
        (contracting, 1, vec![1], RANK - 1),
    ] {
        let ordinary = inference::infer_dot(&ctx, input, input, &dot, None).unwrap();
        assert_eq!(
            dims(&ctx, ordinary),
            vec![
                1;
                if dot.lhs_batch_dimensions.0.is_empty() {
                    RANK
                } else {
                    RANK * 3 / 2
                }
            ]
        );
        let config = RaggedDotDimensionNumbers {
            dot_dimension_numbers: dot,
            lhs_ragged_dimensions: Axes(vec![ragged]),
            rhs_group_dimensions: Axes(rhs_groups),
        };
        let result =
            inference::infer_ragged_dot(&ctx, input, input, groups, &config, None).unwrap();
        assert_eq!(dims(&ctx, result), vec![1; output_rank]);
        assert_eq!(
            Shape::layout(&ctx, result).unwrap().unwrap().minor_to_major,
            (0..output_rank as i64).rev().collect::<Vec<_>>()
        );
    }
}

#[test]
fn dot_checks_all_dimension_bounds_before_uniqueness() {
    let ctx = context();
    error_contains(
        inference::infer_dot(
            &ctx,
            a(&ctx, P::F32, &[2]),
            a(&ctx, P::F32, &[2, 2]),
            &dot_dims(&[0, 0], &[0, 0], &[], &[3]),
            None,
        ),
        "out of range",
    );
}

#[test]
fn dot_checks_contracting_sizes_before_batch_cardinality() {
    let ctx = context();
    error_contains(
        inference::infer_dot(
            &ctx,
            a(&ctx, P::F32, &[2, 3]),
            a(&ctx, P::F32, &[4, 5]),
            &dot_dims(&[1], &[0], &[0], &[]),
            None,
        ),
        "contracting dimension sizes differ",
    );
}

#[test]
fn small_ragged_geometries_match_all_three_mode_formulas() {
    let ctx = context();
    for b in 0..4 {
        for m in 0..4 {
            for k in 0..4 {
                for g in 0..4 {
                    let lhs = a(&ctx, P::F32, &[b, m, k]);
                    for (ragged, rhs_dims, rhs_batch, rhs_contract, groups, rhs_groups, expected) in [
                        (
                            1,
                            vec![g, b, k, 2],
                            1,
                            2,
                            vec![b, g],
                            vec![0],
                            vec![b, m, 2],
                        ),
                        (2, vec![b, k, 2], 0, 1, vec![b, g], vec![], vec![g, b, m, 2]),
                        (0, vec![b, k, 2], 0, 1, vec![g], vec![], vec![b, m, 2]),
                    ] {
                        let config = RaggedDotDimensionNumbers {
                            dot_dimension_numbers: dot_dims(
                                &[2],
                                &[rhs_contract],
                                &[0],
                                &[rhs_batch],
                            ),
                            lhs_ragged_dimensions: Axes(vec![ragged]),
                            rhs_group_dimensions: Axes(rhs_groups),
                        };
                        let result = inference::infer_ragged_dot(
                            &ctx,
                            lhs,
                            a(&ctx, P::F32, &rhs_dims),
                            a(&ctx, P::S32, &groups),
                            &config,
                            None,
                        )
                        .unwrap();
                        assert_eq!(dims(&ctx, result), expected);
                        assert_eq!(
                            Shape::layout(&ctx, result).unwrap().unwrap().minor_to_major,
                            (0..expected.len() as i64).rev().collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn all_three_ragged_modes_round_trip() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 3, 4]);
    for (ragged, rhs_dims, group_dims, rhs_batch, rhs_contract, groups, expected) in [
        (
            1,
            vec![5, 2, 4, 6],
            vec![2, 5],
            1,
            2,
            vec![0],
            vec![2, 3, 6],
        ),
        (2, vec![2, 4, 6], vec![2, 5], 0, 1, vec![], vec![5, 2, 3, 6]),
        (0, vec![2, 4, 6], vec![5], 0, 1, vec![], vec![2, 3, 6]),
    ] {
        let config = RaggedDotDimensionNumbers {
            dot_dimension_numbers: dot_dims(&[2], &[rhs_contract], &[0], &[rhs_batch]),
            lhs_ragged_dimensions: Axes(vec![ragged]),
            rhs_group_dimensions: Axes(groups),
        };
        roundtrip(&mut ctx, config.clone());
        let rhs = a(&ctx, P::F32, &rhs_dims);
        let group = a(&ctx, P::F64, &group_dims);
        let result = inference::infer_ragged_dot(&ctx, lhs, rhs, group, &config, None).unwrap();
        assert_eq!(dims(&ctx, result), expected);
        let args = values(&mut ctx, &[lhs, rhs, group]);
        RaggedDotOp::new(&mut ctx, result, args, config, precision())
            .verify(&ctx)
            .unwrap();
    }
}

#[test]
fn ragged_group_shape_and_dimension_constraints_reject_single_changes() {
    let ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 3, 4]);
    let rhs = a(&ctx, P::F32, &[5, 2, 4, 6]);
    let group = a(&ctx, P::S32, &[2, 5]);
    let config = RaggedDotDimensionNumbers {
        dot_dimension_numbers: dot_dims(&[2], &[2], &[0], &[1]),
        lhs_ragged_dimensions: Axes(vec![1]),
        rhs_group_dimensions: Axes(vec![0]),
    };
    for ragged in [vec![], vec![1, 2], vec![3]] {
        let config = RaggedDotDimensionNumbers {
            lhs_ragged_dimensions: Axes(ragged),
            ..config.clone()
        };
        assert!(inference::infer_ragged_dot(&ctx, lhs, rhs, group, &config, None).is_err());
    }
    for groups in [vec![1], vec![0, 0], vec![]] {
        let config = RaggedDotDimensionNumbers {
            rhs_group_dimensions: Axes(groups),
            ..config.clone()
        };
        assert!(inference::infer_ragged_dot(&ctx, lhs, rhs, group, &config, None).is_err());
    }
    for group in [vec![], vec![3, 5], vec![2, 6], vec![1, 2, 5]] {
        assert!(
            inference::infer_ragged_dot(&ctx, lhs, rhs, a(&ctx, P::S32, &group), &config, None)
                .is_err()
        );
    }
}

#[test]
fn ragged_dynamic_group_count_and_nested_noncontracting_prefix_are_preserved() {
    let ctx = context();
    let config = RaggedDotDimensionNumbers {
        dot_dimension_numbers: dot_dims(&[2], &[1], &[0], &[0]),
        lhs_ragged_dimensions: Axes(vec![2]),
        rhs_group_dimensions: Axes(vec![]),
    };
    let result = inference::infer_ragged_dot(
        &ctx,
        a(&ctx, P::F32, &[2, 3, 4]),
        a(&ctx, P::F32, &[2, 4, 6]),
        dynamic(&ctx, P::S32, &[D::Static(2), D::Bounded(5)]),
        &config,
        None,
    )
    .unwrap();
    assert_eq!(Shape::dimension(&ctx, result, 0).unwrap(), D::Bounded(5));
    assert_eq!(dims(&ctx, result), [5, 2, 3, 6]);
    let config = RaggedDotDimensionNumbers {
        dot_dimension_numbers: dot_dims(&[3], &[2], &[0], &[1]),
        lhs_ragged_dimensions: Axes(vec![2]),
        rhs_group_dimensions: Axes(vec![0]),
    };
    let result = inference::infer_ragged_dot(
        &ctx,
        a(&ctx, P::F32, &[2, 3, 7, 4]),
        a(&ctx, P::F32, &[5, 2, 4, 6]),
        a(&ctx, P::S32, &[2, 3, 5]),
        &config,
        None,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [2, 3, 7, 6]);
}

#[test]
fn excluded_ragged_group_axis_does_not_cause_spurious_output_overflow() {
    let ctx = context();
    let config = RaggedDotDimensionNumbers {
        dot_dimension_numbers: dot_dims(&[1], &[1], &[], &[]),
        lhs_ragged_dimensions: Axes(vec![0]),
        rhs_group_dimensions: Axes(vec![0]),
    };
    let result = inference::infer_ragged_dot(
        &ctx,
        a(&ctx, P::Pred, &[1, 0]),
        a(&ctx, P::Pred, &[i64::MAX, 0, i64::MAX]),
        a(&ctx, P::Pred, &[i64::MAX]),
        &config,
        None,
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [1, i64::MAX]);
}

fn sparse(lhs: bool, nonzero: i64, block: i64, axis: i64, idx: i32) -> SparsityConfig {
    let side = TensorSparsityConfig {
        num_non_zero: nonzero,
        block_size: block,
        dimension: axis,
        stride: 1,
        idx,
    };
    if lhs {
        SparsityConfig {
            lhs: side.into(),
            rhs: OptionalTensorSparsityConfig::Absent,
        }
    } else {
        SparsityConfig {
            lhs: OptionalTensorSparsityConfig::Absent,
            rhs: side.into(),
        }
    }
}

#[test]
fn sparse_dot_expands_only_configured_contracting_dimensions() {
    let ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[8, 3]);
    let d = dot_dims(&[1], &[0], &[], &[]);
    let config = sparse(true, 2, 4, 1, 2);
    for lhs in [lhs, Shape::buffer(&ctx, lhs).unwrap()] {
        let result =
            inference::infer_dot_with_sparsity(&ctx, lhs, rhs, &d, None, Some(&config)).unwrap();
        assert_eq!(dims(&ctx, result), [2, 3]);
    }
    let result = inference::infer_dot_with_sparsity(
        &ctx,
        a(&ctx, P::F32, &[2, 8]),
        a(&ctx, P::F32, &[4, 3]),
        &d,
        None,
        Some(&sparse(true, 4, 2, 1, 2)),
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [2, 3]);
    assert!(
        inference::infer_dot_with_sparsity(
            &ctx,
            lhs,
            a(&ctx, P::F32, &[4, 3]),
            &d,
            None,
            Some(&sparse(true, 0, 0, -1, 2))
        )
        .is_ok()
    );
}

#[test]
fn rhs_sparse_dot_expands_its_configured_contracting_dimension() {
    let ctx = context();
    let result = inference::infer_dot_with_sparsity(
        &ctx,
        a(&ctx, P::F32, &[2, 8]),
        a(&ctx, P::F32, &[4, 3]),
        &dot_dims(&[1], &[0], &[], &[]),
        None,
        Some(&sparse(false, 2, 4, 0, 2)),
    )
    .unwrap();
    assert_eq!(dims(&ctx, result), [2, 3]);
}

#[test]
fn sparse_geometry_and_auxiliary_operand_references_are_checked() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[8, 3]);
    let metadata = a(&ctx, P::U16, &[2, 1]);
    let result = a(&ctx, P::F32, &[2, 3]);
    let d = dot_dims(&[1], &[0], &[], &[]);
    for (nonzero, block, idx, extra, valid) in [
        (2, 4, 2, 1, true),
        (0, 4, 2, 1, false),
        (2, 0, 2, 1, false),
        (3, 4, 2, 1, false),
        (2, 4, 1, 1, false),
        (2, 4, 3, 1, false),
        (2, 4, -1, 1, false),
        (2, 4, 2, 2, false),
    ] {
        let mut shapes = vec![lhs, rhs];
        shapes.extend(vec![metadata; extra]);
        let args = values(&mut ctx, &shapes);
        let op = DotOp::new(&mut ctx, result, args, d.clone(), precision());
        op.get_operation().deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_sparsity"),
            sparse(true, nonzero, block, 1, idx),
        );
        assert_eq!(op.verify(&ctx).is_ok(), valid);
    }
    let token = Shape::token(&ctx);
    let args = values(&mut ctx, &[lhs, rhs, token]);
    let op = DotOp::new(&mut ctx, result, args, d, precision());
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_sparsity"), sparse(true, 2, 4, 1, 2));
    assert!(op.verify(&ctx).is_err());
}

#[test]
fn block_scaling_tracks_all_scales_and_optional_zero_points_without_inventing_geometry() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 8]);
    let rhs = a(&ctx, P::F32, &[8, 3]);
    let metadata = a(&ctx, P::U16, &[2, 1]);
    let result = a(&ctx, P::F32, &[2, 3]);
    for (scale_idx, zero_idx, extras, valid) in [
        (2, OptionalInteger::Absent, 1, true),
        (2, OptionalInteger::Value(3), 2, true),
        (2, OptionalInteger::Value(2), 2, false),
        (2, OptionalInteger::Value(4), 2, false),
        (0, OptionalInteger::Absent, 1, false),
        (2, OptionalInteger::Absent, 2, false),
    ] {
        let mut shapes = vec![lhs, rhs];
        shapes.extend(vec![metadata; extras]);
        let args = values(&mut ctx, &shapes);
        let op = DotOp::new(
            &mut ctx,
            result,
            args,
            dot_dims(&[1], &[0], &[], &[]),
            precision(),
        );
        let config = BlockScalingConfig {
            lhs: TensorBlockScalingConfig {
                scale_idx,
                zero_idx,
                strides: Axes(vec![3, 7]),
                steps: Axes(vec![-1, 0]),
            }
            .into(),
            rhs: OptionalTensorBlockScalingConfig::Absent,
        };
        roundtrip(&mut ctx, config.clone());
        op.get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new("hlo_block_scaling"), config);
        assert_eq!(op.verify(&ctx).is_ok(), valid);
    }
}

#[test]
fn scaled_dot_geometry_and_precision_round_trip() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[4, 8]);
    let rhs = a(&ctx, P::F32, &[8, 6]);
    let ls = a(&ctx, P::F16, &[2, 2]);
    let rs = a(&ctx, P::S32, &[4, 3]);
    let d = dot_dims(&[1], &[0], &[], &[]);
    let result = inference::infer_scaled_dot(&ctx, lhs, rhs, ls, rs, &d, None).unwrap();
    assert_eq!(dims(&ctx, result), [4, 6]);
    let args = values(&mut ctx, &[lhs, rhs, ls, rs]);
    let config = PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::High, Precision::Highest]),
        algorithm: Algorithm::DotF32F32F32,
    };
    ScaledDotOp::new(&mut ctx, result, args, d.clone(), config.clone())
        .verify(&ctx)
        .unwrap();
    roundtrip(&mut ctx, config);
    for bad in [
        a(&ctx, P::F16, &[3, 2]),
        a(&ctx, P::F16, &[0, 2]),
        a(&ctx, P::F16, &[2]),
        a(&ctx, P::F16, &[1, 2, 2]),
    ] {
        assert!(inference::infer_scaled_dot(&ctx, lhs, rhs, bad, rs, &d, None).is_err());
    }
    assert!(
        inference::infer_scaled_dot(&ctx, lhs, a(&ctx, P::F32, &[9, 6]), ls, rs, &d, None).is_err()
    );
    assert!(
        inference::infer_scaled_dot(&ctx, lhs, rhs, ls, rs, &dot_dims(&[], &[0], &[], &[]), None)
            .is_err()
    );
}

#[test]
fn bf16_dummy_scales_allow_all_one_scale_shapes() {
    let ctx = context();
    let lhs = a(&ctx, P::BF16, &[4, 8]);
    let rhs = a(&ctx, P::BF16, &[8, 6]);
    let ls = s(&ctx, P::BF16);
    let rs = a(&ctx, P::BF16, &[1, 1, 1]);
    let d = dot_dims(&[1], &[0], &[], &[]);
    assert!(inference::infer_scaled_dot(&ctx, lhs, rhs, ls, rs, &d, None).is_ok());
    assert!(inference::infer_scaled_dot(&ctx, lhs, rhs, ls, ls, &d, None).is_err());
    assert!(inference::infer_scaled_dot(&ctx, lhs, rhs, s(&ctx, P::F16), rs, &d, None).is_err());
}

#[test]
fn sparse_and_block_scaled_convolution_share_auxiliary_operand_contracts() {
    let mut ctx = context();
    let lhs = a(&ctx, P::F32, &[2, 4]);
    let rhs = a(&ctx, P::F32, &[8, 3]);
    let metadata = a(&ctx, P::U16, &[2, 1]);
    let result = a(&ctx, P::F32, &[2, 3]);
    let d = ConvolutionDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: Axes(vec![]),
        kernel_input_feature_dimension: 0,
        kernel_output_feature_dimension: 1,
        kernel_spatial_dimensions: Axes(vec![]),
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: Axes(vec![]),
    };
    for (idx, nonzero, extra, valid) in [
        (2, 2, 1, true),
        (1, 2, 1, false),
        (2, 0, 1, false),
        (2, 2, 2, false),
    ] {
        let mut shapes = vec![lhs, rhs];
        shapes.extend(vec![metadata; extra]);
        let args = values(&mut ctx, &shapes);
        let op = ConvolutionOp::new(
            &mut ctx,
            result,
            args,
            d.clone(),
            Window(vec![]),
            NumericalInteger(1),
            NumericalInteger(1),
            ConvolutionKind::Unset,
            precision(),
        );
        op.get_operation().deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_sparsity"),
            sparse(true, nonzero, 4, 1, idx),
        );
        assert_eq!(op.verify(&ctx).is_ok(), valid);
    }
}
