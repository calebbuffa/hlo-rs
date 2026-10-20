use super::super::*;
use super::windows::window;
use xlahlo::literal::Literal;

fn convolution_dimensions() -> ConvolutionDimensionNumbers {
    ConvolutionDimensionNumbers {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: Axes(vec![]),
        kernel_input_feature_dimension: 0,
        kernel_output_feature_dimension: 1,
        kernel_spatial_dimensions: Axes(vec![]),
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: Axes(vec![]),
    }
}

#[test]
fn contraction_and_selection_geometry_accepts_every_array_element_format() {
    let mut ctx = context();
    for &et in P::ALL {
        let lhs = a(&ctx, et, &[2, 4]);
        let rhs = a(&ctx, et, &[4, 3]);
        let expected = a(&ctx, et, &[2, 3]);
        let d = dot_dims(&[1], &[0], &[], &[]);
        let inferred = inference::infer_dot(&ctx, lhs, rhs, &d, None).unwrap();
        assert!(Shape::compatible(&ctx, inferred, expected), "{et:?}");
        let args = values(&mut ctx, &[lhs, rhs]);
        DotOp::new(&mut ctx, expected, args, d.clone(), precision())
            .verify(&ctx)
            .unwrap();

        let dimensions = convolution_dimensions();
        let inferred =
            inference::infer_convolution(&ctx, lhs, rhs, 1, 1, &Window(vec![]), &dimensions, None)
                .unwrap();
        assert!(Shape::compatible(&ctx, inferred, expected), "{et:?}");
        let args = values(&mut ctx, &[lhs, rhs]);
        ConvolutionOp::new(
            &mut ctx,
            expected,
            args,
            dimensions,
            Window(vec![]),
            NumericalInteger(1),
            NumericalInteger(1),
            ConvolutionKind::Fprop,
            precision(),
        )
        .verify(&ctx)
        .unwrap();

        let group_sizes = a(&ctx, P::S32, &[5]);
        let ragged_rhs = a(&ctx, et, &[5, 4, 3]);
        let ragged_dimensions = RaggedDotDimensionNumbers {
            dot_dimension_numbers: dot_dims(&[1], &[1], &[], &[]),
            lhs_ragged_dimensions: Axes(vec![0]),
            rhs_group_dimensions: Axes(vec![0]),
        };
        let inferred = inference::infer_ragged_dot(
            &ctx,
            lhs,
            ragged_rhs,
            group_sizes,
            &ragged_dimensions,
            None,
        )
        .unwrap();
        assert!(Shape::compatible(&ctx, inferred, expected), "{et:?}");
        let args = values(&mut ctx, &[lhs, ragged_rhs, group_sizes]);
        RaggedDotOp::new(&mut ctx, expected, args, ragged_dimensions, precision())
            .verify(&ctx)
            .unwrap();

        let lhs_scale = a(&ctx, P::F8E8M0FNU, &[1, 2]);
        let rhs_scale = a(&ctx, P::F8E8M0FNU, &[2, 1]);
        let inferred =
            inference::infer_scaled_dot(&ctx, lhs, rhs, lhs_scale, rhs_scale, &d, None).unwrap();
        assert!(Shape::compatible(&ctx, inferred, expected), "{et:?}");
        let args = values(&mut ctx, &[lhs, rhs, lhs_scale, rhs_scale]);
        ScaledDotOp::new(&mut ctx, expected, args, d, precision())
            .verify(&ctx)
            .unwrap();

        let top_k = inference::infer_top_k(&ctx, lhs, 2).unwrap();
        let outputs = Shape::children(&ctx, top_k);
        assert_eq!(Shape::element_type(&ctx, outputs[0]).unwrap(), et);
        assert_eq!(Shape::element_type(&ctx, outputs[1]).unwrap(), P::S32);
        drop(outputs);
        let args = values(&mut ctx, &[lhs]);
        TopKOp::new(
            &mut ctx,
            top_k,
            args,
            NumericalInteger(2),
            SortOptions { descending: true },
        )
        .verify(&ctx)
        .unwrap();

        let scalar = s(&ctx, et);
        let mask = match et.bit_width() {
            128 => u128::MAX,
            width => (1u128 << width) - 1,
        };
        let literal = Literal::from_bits(&ctx, scalar, &[mask]).unwrap();
        assert_eq!(literal.bits(&ctx).unwrap(), [mask], "{et:?}");
        let constant = ConstantOp::new(&mut ctx, scalar, vec![], literal);
        constant.verify(&ctx).unwrap();
    }
}

#[test]
fn callable_contracts_preserve_every_array_element_format() {
    let mut ctx = context();
    for &et in P::ALL {
        let input = a(&ctx, et, &[4]);
        let scalar = s(&ctx, et);
        let reducer = (vec![scalar; 2], scalar);
        let args = values(&mut ctx, &[input, scalar]);
        let reduced = inference::infer_reduce(&ctx, &[input, scalar], &[0], &reducer).unwrap();
        assert_eq!(Shape::element_type(&ctx, reduced).unwrap(), et);
        let op = ReduceOp::new(&mut ctx, reduced, args, Axes(vec![0]), callees(1));
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Reduce,
            VerifyOptions::default(),
            &[&reducer],
        )
        .unwrap();

        let config = window(2, 1, 0, 0, 1, 1);
        let reduced =
            inference::infer_reduce_window(&ctx, &[input, scalar], &config, &reducer).unwrap();
        assert_eq!(Shape::element_type(&ctx, reduced).unwrap(), et);
        let args = values(&mut ctx, &[input, scalar]);
        let op = ReduceWindowOp::new(&mut ctx, reduced, args, config.clone(), callees(1));
        verify_callees(
            &ctx,
            op.get_operation(),
            O::ReduceWindow,
            VerifyOptions::default(),
            &[&reducer],
        )
        .unwrap();

        let body = (vec![scalar], scalar);
        let mapped = inference::infer_map(&ctx, &[input], &body, &[0]).unwrap();
        assert_eq!(Shape::element_type(&ctx, mapped).unwrap(), et);
        let args = values(&mut ctx, &[input]);
        let op = MapOp::new(&mut ctx, mapped, args, callees(1));
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Map,
            VerifyOptions::default(),
            &[&body],
        )
        .unwrap();

        let scanned = inference::infer_scan(&ctx, &[input], &body, 0, 0, None).unwrap();
        assert_eq!(Shape::element_type(&ctx, scanned).unwrap(), et);
        let args = values(&mut ctx, &[input]);
        let op = ScanOp::new(
            &mut ctx,
            scanned,
            args,
            Axes(vec![0]),
            NumericalInteger(0),
            TriState::True,
            callees(1),
        );
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Scan,
            VerifyOptions::default(),
            &[&body],
        )
        .unwrap();

        let comparator = (vec![scalar; 2], s(&ctx, P::Pred));
        let sorted = inference::infer_sort(&ctx, &[input], 0, &comparator).unwrap();
        assert_eq!(sorted, input);
        let args = values(&mut ctx, &[input]);
        let op = SortOp::new(&mut ctx, sorted, args, Axes(vec![0]), callees(1));
        verify_callees(
            &ctx,
            op.get_operation(),
            O::Sort,
            VerifyOptions::default(),
            &[&comparator],
        )
        .unwrap();

        let source = a(&ctx, et, &[3]);
        assert_eq!(
            inference::infer_select_and_scatter(
                &ctx,
                input,
                source,
                scalar,
                &config,
                &comparator,
                &reducer
            )
            .unwrap(),
            input,
        );
        let args = values(&mut ctx, &[input, source, scalar]);
        let op = SelectAndScatterOp::new(&mut ctx, input, args, config, callees(2));
        verify_callees(
            &ctx,
            op.get_operation(),
            O::SelectAndScatter,
            VerifyOptions::default(),
            &[&comparator, &reducer],
        )
        .unwrap();
    }
}

#[test]
fn decomposition_and_batch_normalization_domains_include_all_floating_formats() {
    let mut ctx = context();
    for &et in P::ALL {
        let matrix = a(&ctx, et, &[2, 2]);
        let vector = a(&ctx, et, &[2]);
        let options = TriangularSolveOptions {
            left_side: true,
            lower: true,
            unit_diagonal: false,
            transpose_a: Transpose::NoTranspose,
        };
        let cholesky = inference::infer_cholesky(&ctx, matrix);
        let solve = inference::infer_triangular_solve(&ctx, matrix, matrix, &options);
        if et.is_float() || et.is_complex() {
            assert_eq!(cholesky.unwrap(), matrix, "{et:?}");
            assert_eq!(solve.unwrap(), matrix, "{et:?}");
            let args = values(&mut ctx, &[matrix]);
            CholeskyOp::new(&mut ctx, matrix, args, CholeskyOptions { lower: true })
                .verify(&ctx)
                .unwrap();
            let args = values(&mut ctx, &[matrix, matrix]);
            TriangularSolveOp::new(&mut ctx, matrix, args, options)
                .verify(&ctx)
                .unwrap();
        } else {
            assert!(cholesky.is_err(), "{et:?}");
            assert!(solve.is_err(), "{et:?}");
        }
        for opcode in [
            O::BatchNormTraining,
            O::BatchNormInference,
            O::BatchNormGrad,
        ] {
            let inputs = match opcode {
                O::BatchNormTraining => vec![matrix, vector, vector],
                O::BatchNormInference => vec![matrix, vector, vector, vector, vector],
                _ => vec![matrix, vector, vector, vector, matrix],
            };
            let inferred = inference::infer_batch_norm(&ctx, opcode, &inputs, 1);
            if !et.is_float() {
                assert!(inferred.is_err(), "{opcode:?} {et:?}");
                continue;
            }
            let result = inferred.unwrap();
            let args = values(&mut ctx, &inputs);
            let options = BatchNormOptions {
                epsilon_bits: 0.01f32.to_bits(),
                feature_index: 1,
            };
            let op = match opcode {
                O::BatchNormTraining => {
                    BatchNormTrainingOp::new(&mut ctx, result, args, options).get_operation()
                }
                O::BatchNormInference => {
                    BatchNormInferenceOp::new(&mut ctx, result, args, options).get_operation()
                }
                _ => BatchNormGradOp::new(&mut ctx, result, args, options).get_operation(),
            };
            instruction(&ctx, op, opcode, VerifyOptions::default()).unwrap();
        }
    }
}

#[test]
fn fft_datatype_limits_are_semantic_not_backend_capability_filters() {
    let mut ctx = context();
    for &et in P::ALL {
        let input = a(&ctx, et, &[4]);
        for kind in [FftType::FFT, FftType::IFFT, FftType::RFFT, FftType::IRFFT] {
            let length = if kind == FftType::IRFFT { 6 } else { 4 };
            let inferred = inference::infer_fft(&ctx, input, kind, &[length]);
            let valid = if kind == FftType::RFFT {
                matches!(et, P::F32 | P::F64)
            } else {
                et.is_complex()
            };
            if valid {
                let result = inferred.unwrap();
                let args = values(&mut ctx, &[input]);
                FftOp::new(&mut ctx, result, args, kind, Axes(vec![length]))
                    .verify(&ctx)
                    .unwrap();
            } else {
                assert!(inferred.is_err(), "{kind:?} {et:?}");
            }
        }
    }
}
