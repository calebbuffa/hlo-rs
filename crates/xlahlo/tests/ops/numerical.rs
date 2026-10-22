use xlahlo::{
    attributes::*,
    inference::{Signature, numerical as inference},
    opcode::Opcode as O,
    ops::{
        ComputationOp, HloOpInterface, ReturnOp,
        control::{ConstantOp, ParameterOp, TupleOp},
        numerical::*,
    },
    pliron::{
        attribute::{AttrObj, Attribute},
        common_traits::Verify,
        context::Context,
        context::Ptr,
        identifier::Identifier,
        op::Op,
        op::op_cast,
        operation::Operation,
        parsable::{Parsable, parse_from_str},
        printable::Printable,
        result::Result,
        r#type::TypeHandle,
        value::Value,
    },
    shape::{self, Dimension as D, Layout},
    types::ElementType as P,
    verification::VerifyOptions,
};

mod numerical {
    mod convolution;
    mod datatypes;
    mod dot_families;
    mod errors;
    pub(super) mod native_graph;
    mod precision_ordering;
    mod reducers_and_maps;
    mod scans_and_shuffles;
    mod windows;
}
use numerical::native_graph::verify_callees;

fn instruction(ctx: &Context, op: Ptr<Operation>, opcode: O, options: VerifyOptions) -> Result<()> {
    let dynamic = Operation::get_op_dyn(op, ctx);
    let hlo = op_cast::<dyn HloOpInterface>(dynamic.as_ref()).unwrap();
    assert_eq!(hlo.opcode(), opcode);
    hlo.verify_with_options(ctx, options)
}

fn context() -> Context {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    ctx
}
fn a(ctx: &Context, et: P, dims: &[i64]) -> TypeHandle {
    shape::array(ctx, et, dims.to_vec()).unwrap()
}
fn dynamic(ctx: &Context, et: P, dims: &[D]) -> TypeHandle {
    shape::dynamic_array(ctx, et, dims.to_vec()).unwrap()
}
fn s(ctx: &Context, et: P) -> TypeHandle {
    shape::scalar(ctx, et).unwrap()
}
fn dims(ctx: &Context, shape: TypeHandle) -> Vec<i64> {
    shape::dimensions(ctx, shape).unwrap().to_vec()
}
fn default_layout(ctx: &Context, shape: TypeHandle) -> TypeHandle {
    shape::with_default_layout(ctx, shape).unwrap()
}
fn values(ctx: &mut Context, shapes: &[TypeHandle]) -> Vec<Value> {
    shapes
        .iter()
        .enumerate()
        .map(|(index, &shape)| {
            ParameterOp::new(ctx, shape, vec![], Index(index as u32)).result(ctx)
        })
        .collect()
}
fn callees(count: usize) -> Callees {
    Callees(
        (0..count)
            .map(|i| format!("callee{i}").try_into().unwrap())
            .collect(),
    )
}
fn precision() -> PrecisionConfig {
    PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::Default; 2]),
        algorithm: Algorithm::Unset,
    }
}
fn dot_dims(lc: &[i64], rc: &[i64], lb: &[i64], rb: &[i64]) -> DotDimensionNumbers {
    DotDimensionNumbers {
        lhs_contracting_dimensions: Axes(lc.to_vec()),
        rhs_contracting_dimensions: Axes(rc.to_vec()),
        lhs_batch_dimensions: Axes(lb.to_vec()),
        rhs_batch_dimensions: Axes(rb.to_vec()),
    }
}
fn signature(ctx: &Context, types: &[P], result: P) -> Signature {
    (types.iter().map(|&et| s(ctx, et)).collect(), s(ctx, result))
}
fn error_contains<T: std::fmt::Debug>(result: xlahlo::pliron::result::Result<T>, message: &str) {
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains(message),
        "{error:?} does not contain {message:?}"
    );
}
fn roundtrip<T: Attribute + PartialEq + std::fmt::Debug + Clone>(ctx: &mut Context, value: T) {
    let attr: AttrObj = Box::new(value.clone());
    let text = attr.disp(ctx).to_string();
    let parsed = parse_from_str(AttrObj::parser(()), ctx, &text).unwrap();
    assert_eq!(parsed.downcast_ref::<T>().unwrap(), &value);
}

#[test]
fn sort_comparator_pairs_are_interleaved_by_operand_and_result_can_be_tuple() {
    let mut ctx = context();
    let keys = a(&ctx, P::F32, &[2, 3]);
    let vals = a(&ctx, P::S32, &[2, 3]);
    let comparator = signature(&ctx, &[P::F32, P::F32, P::S32, P::S32], P::Pred);
    let result = inference::infer_sort(&ctx, &[keys, vals], 1, &comparator).unwrap();
    assert_eq!(&*shape::children(&ctx, result), &[keys, vals]);
    let operands = values(&mut ctx, &[keys, vals]);
    let op = SortOp::new(&mut ctx, result, operands, Axes(vec![1]), callees(1));
    op.verify(&ctx).unwrap();
    verify_callees(
        &ctx,
        op.get_operation(),
        O::Sort,
        VerifyOptions::default(),
        &[&comparator],
    )
    .unwrap();
    let wrong = signature(&ctx, &[P::F32, P::S32, P::F32, P::S32], P::Pred);
    assert!(inference::infer_sort(&ctx, &[keys, vals], 1, &wrong).is_err());
    assert!(
        inference::infer_sort(&ctx, &[keys, a(&ctx, P::S32, &[2, 4])], 1, &comparator).is_err()
    );
    assert!(inference::infer_sort(&ctx, &[keys, vals], 2, &comparator).is_err());
    assert!(inference::infer_sort(&ctx, &[], 1, &comparator).is_err());
    let single = signature(&ctx, &[P::S32, P::S32], P::Pred);
    assert_eq!(
        inference::infer_sort(&ctx, &[vals], 1, &single).unwrap(),
        vals
    );
    let mixed = signature(&ctx, &[P::F64, P::F16, P::S32, P::S32], P::Pred);
    assert!(inference::infer_sort(&ctx, &[keys, vals], 1, &mixed).is_ok());
}

#[test]
fn sort_inference_checks_comparator_before_dimensions() {
    let ctx = context();
    let operand = a(&ctx, P::F32, &[3]);
    error_contains(
        inference::infer_sort(
            &ctx,
            &[operand],
            3,
            &signature(&ctx, &[P::F32, P::F32], P::F32),
        ),
        "predicate scalar",
    );
    error_contains(
        inference::infer_sort(&ctx, &[operand], 3, &signature(&ctx, &[], P::Pred)),
        "two scalar parameters",
    );
    error_contains(
        inference::infer_sort(
            &ctx,
            &[operand],
            3,
            &signature(&ctx, &[P::F32, P::S32], P::Pred),
        ),
        "incorrect scalar type",
    );
    let comparator = signature(&ctx, &[P::F32, P::F32], P::Pred);
    assert_eq!(
        inference::infer_sort(&ctx, &[operand], -1, &comparator).unwrap(),
        operand
    );
    let buffered = shape::buffer(&ctx, operand).unwrap();
    assert_eq!(
        inference::infer_sort(&ctx, &[buffered], 0, &comparator).unwrap(),
        buffered
    );
    let bad = (
        vec![shape::buffer(&ctx, s(&ctx, P::F32)).unwrap(); 2],
        s(&ctx, P::Pred),
    );
    assert!(inference::infer_sort(&ctx, &[buffered], 0, &bad).is_err());
}

#[test]
fn fft_shapes_preserve_layout_and_distinguish_real_transform_geometry() {
    let ctx = context();
    let input = default_layout(&ctx, a(&ctx, P::F32, &[2, 4, 6]));
    let complex = inference::infer_fft(&ctx, input, FftKind::Rfft, &[4, 6]).unwrap();
    assert_eq!(dims(&ctx, complex), [2, 4, 4]);
    assert_eq!(shape::element_type(&ctx, complex).unwrap(), P::C64);
    assert_eq!(
        shape::layout(&ctx, complex).unwrap().as_deref(),
        shape::layout(&ctx, input).unwrap().as_deref()
    );
    assert_eq!(
        inference::infer_fft(&ctx, complex, FftKind::Irfft, &[4, 6]).unwrap(),
        input
    );
    for kind in [FftKind::Fft, FftKind::Ifft] {
        assert_eq!(
            inference::infer_fft(&ctx, complex, kind, &[1]).unwrap(),
            complex
        );
        assert!(inference::infer_fft(&ctx, input, kind, &[6]).is_err());
    }
    for lengths in [vec![], vec![1, 2, 3, 4], vec![5, 6], vec![4, 7]] {
        assert!(inference::infer_fft(&ctx, input, FftKind::Rfft, &lengths).is_err());
    }
    assert!(inference::infer_fft(&ctx, s(&ctx, P::F32), FftKind::Fft, &[1]).is_err());
    assert!(inference::infer_fft(&ctx, s(&ctx, P::C64), FftKind::Fft, &[1]).is_err());
    let buffered = shape::buffer(&ctx, a(&ctx, P::C64, &[4])).unwrap();
    for kind in [FftKind::Fft, FftKind::Ifft] {
        assert_eq!(
            inference::infer_fft(&ctx, buffered, kind, &[4]).unwrap(),
            buffered
        );
    }
    assert!(inference::infer_fft(&ctx, buffered, FftKind::Irfft, &[4]).is_err());
    assert!(
        inference::infer_fft(
            &ctx,
            shape::buffer(&ctx, a(&ctx, P::F32, &[4])).unwrap(),
            FftKind::Rfft,
            &[4]
        )
        .is_err()
    );
    let zero = a(&ctx, P::F64, &[0]);
    let zero_complex = inference::infer_fft(&ctx, zero, FftKind::Rfft, &[0]).unwrap();
    assert_eq!(dims(&ctx, zero_complex), [0]);
    assert_eq!(
        inference::infer_fft(&ctx, zero_complex, FftKind::Irfft, &[0]).unwrap(),
        zero
    );
    for (input, expected) in [(D::Bounded(6), D::Bounded(4)), (D::Unbounded, D::Static(4))] {
        let result =
            inference::infer_fft(&ctx, dynamic(&ctx, P::F32, &[input]), FftKind::Rfft, &[6])
                .unwrap();
        assert_eq!(shape::dimension(&ctx, result, 0).unwrap(), expected);
    }
}

#[test]
fn fft_assembly_round_trips_all_four_kinds_and_rejects_unknown_enums() {
    let mut ctx = context();
    for (input, output, kind, length) in [
        (P::C64, P::C64, FftKind::Fft, 4),
        (P::C128, P::C128, FftKind::Ifft, 4),
        (P::F32, P::C64, FftKind::Rfft, 6),
        (P::C128, P::F64, FftKind::Irfft, 6),
    ] {
        roundtrip(&mut ctx, kind);
        let input_size = if kind == FftKind::Rfft { 6 } else { 4 };
        let output_size = if kind == FftKind::Irfft { 6 } else { 4 };
        let input_shape = a(&ctx, input, &[input_size]);
        let args = values(&mut ctx, &[input_shape]);
        let result = a(&ctx, output, &[output_size]);
        let op = FftOp::new(&mut ctx, result, args, kind, Axes(vec![length]));
        op.verify(&ctx).unwrap();
        op.get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new("hlo_kind"), Axes(vec![999]));
        assert!(op.verify(&ctx).is_err());
    }
}

#[test]
fn decompositions_preserve_shapes_and_require_square_compatible_matrices() {
    let mut ctx = context();
    let matrix = a(&ctx, P::C128, &[2, 3, 3]);
    let rhs = a(&ctx, P::C128, &[2, 3, 5]);
    let mut options = TriangularSolveOptions {
        left_side: true,
        lower: true,
        unit_diagonal: true,
        transpose_a: Transpose::Adjoint,
    };
    assert_eq!(inference::infer_cholesky(&ctx, matrix).unwrap(), matrix);
    assert_eq!(
        inference::infer_triangular_solve(&ctx, matrix, rhs, &options).unwrap(),
        rhs
    );
    options.left_side = false;
    assert!(inference::infer_triangular_solve(&ctx, matrix, rhs, &options).is_err());
    let right = a(&ctx, P::C128, &[2, 5, 3]);
    assert_eq!(
        inference::infer_triangular_solve(&ctx, matrix, right, &options).unwrap(),
        right
    );
    assert!(
        inference::infer_triangular_solve(&ctx, s(&ctx, P::S32), s(&ctx, P::S32), &options)
            .is_err()
    );
    assert!(
        inference::infer_triangular_solve(
            &ctx,
            shape::tuple(&ctx, vec![s(&ctx, P::F32)]).unwrap(),
            s(&ctx, P::F32),
            &options
        )
        .is_err()
    );
    for invalid in [
        a(&ctx, P::S32, &[3, 3]),
        a(&ctx, P::F32, &[3]),
        a(&ctx, P::F32, &[3, 4]),
        shape::tuple(&ctx, vec![matrix]).unwrap(),
    ] {
        assert!(inference::infer_cholesky(&ctx, invalid).is_err());
    }
    for transpose in [
        Transpose::NoTranspose,
        Transpose::Transpose,
        Transpose::Adjoint,
    ] {
        options.transpose_a = transpose;
        roundtrip(&mut ctx, options.clone());
        let args = values(&mut ctx, &[matrix, right]);
        TriangularSolveOp::new(&mut ctx, right, args, options.clone())
            .verify(&ctx)
            .unwrap();
    }
    let args = values(&mut ctx, &[matrix]);
    CholeskyOp::new(&mut ctx, matrix, args, CholeskyOptions { lower: true })
        .verify(&ctx)
        .unwrap();
}

#[test]
fn top_k_exhaustive_small_shapes_and_text_defaults() {
    let mut ctx = context();
    for batch in 0..4 {
        for n in 0..6 {
            for k in 0..=n {
                let operand = dynamic(&ctx, P::F32, &[D::Bounded(batch), D::Bounded(n)]);
                let output = inference::infer_top_k(&ctx, operand, k).unwrap();
                let values_shape = default_layout(
                    &ctx,
                    dynamic(&ctx, P::F32, &[D::Bounded(batch), D::Static(k)]),
                );
                assert_eq!(
                    output,
                    shape::tuple(
                        &ctx,
                        vec![
                            values_shape,
                            shape::with_element_type(&ctx, values_shape, P::S32).unwrap()
                        ]
                    )
                    .unwrap()
                );
                for descending in [false, true] {
                    let args = values(&mut ctx, &[operand]);
                    TopKOp::new(
                        &mut ctx,
                        output,
                        args,
                        NumericalInteger(k),
                        SortOptions { descending },
                    )
                    .verify(&ctx)
                    .unwrap();
                }
            }
            assert!(inference::infer_top_k(&ctx, a(&ctx, P::F32, &[n]), n + 1).is_err());
        }
    }
    assert!(inference::infer_top_k(&ctx, s(&ctx, P::F32), 0).is_err());
    assert!(inference::infer_top_k(&ctx, a(&ctx, P::F32, &[3]), -1).is_err());
    assert!(inference::infer_top_k(&ctx, dynamic(&ctx, P::F32, &[D::Unbounded]), 0).is_err());
}

#[test]
fn batch_norm_training_inference_and_grad_share_feature_geometry() {
    let mut ctx = context();
    let input = dynamic(&ctx, P::F32, &[D::Static(2), D::Bounded(3), D::Static(4)]);
    let vector = a(&ctx, P::F32, &[3]);
    let feature = default_layout(&ctx, dynamic(&ctx, P::F32, &[D::Bounded(3)]));
    let expected = shape::tuple(&ctx, vec![input, feature, feature]).unwrap();
    let training = vec![input, vector, vector];
    assert_eq!(
        inference::infer_batch_norm(&ctx, O::BatchNormTraining, &training, 1).unwrap(),
        expected
    );
    let params = vec![input, vector, vector, vector, vector];
    assert_eq!(
        inference::infer_batch_norm(&ctx, O::BatchNormInference, &params, 1).unwrap(),
        input
    );
    let mut inference_params = params.clone();
    inference_params[3] = a(&ctx, P::F32, &[3, 1]);
    inference_params[4] = a(&ctx, P::F32, &[3, 1]);
    assert_eq!(
        inference::infer_batch_norm(&ctx, O::BatchNormInference, &inference_params, 1).unwrap(),
        input
    );
    inference_params[3] = shape::buffer(&ctx, vector).unwrap();
    inference_params[4] = shape::buffer(&ctx, a(&ctx, P::F32, &[3, 1])).unwrap();
    assert_eq!(
        inference::infer_batch_norm(&ctx, O::BatchNormInference, &inference_params, 1).unwrap(),
        input
    );
    for index in [1, 2] {
        let mut invalid = params.clone();
        invalid[index] = shape::buffer(&ctx, vector).unwrap();
        assert!(inference::infer_batch_norm(&ctx, O::BatchNormInference, &invalid, 1).is_err());
    }
    let mut grad = params.clone();
    grad[4] = input;
    assert_eq!(
        inference::infer_batch_norm(&ctx, O::BatchNormGrad, &grad, 1).unwrap(),
        expected
    );
    for axis in [-1, 3] {
        assert!(inference::infer_batch_norm(&ctx, O::BatchNormTraining, &training, axis).is_err());
    }
    for bad in [
        a(&ctx, P::F32, &[4]),
        s(&ctx, P::F32),
        a(&ctx, P::S32, &[3]),
    ] {
        let mut parameters = training.clone();
        parameters[1] = bad;
        assert!(inference::infer_batch_norm(&ctx, O::BatchNormTraining, &parameters, 1).is_err());
    }
    grad[4] = a(&ctx, P::F32, &[2, 3, 5]);
    assert!(inference::infer_batch_norm(&ctx, O::BatchNormGrad, &grad, 1).is_err());
    grad[4] = dynamic(&ctx, P::F64, &[D::Static(2), D::Bounded(3), D::Static(4)]);
    assert!(inference::infer_batch_norm(&ctx, O::BatchNormGrad, &grad, 1).is_ok());
    let config = BatchNormOptions {
        epsilon_bits: 0.001f32.to_bits(),
        feature_index: 1,
    };
    let args = values(&mut ctx, &training);
    BatchNormTrainingOp::new(&mut ctx, expected, args, config.clone())
        .verify(&ctx)
        .unwrap();
    let args = values(&mut ctx, &params);
    BatchNormInferenceOp::new(&mut ctx, input, args, config.clone())
        .verify(&ctx)
        .unwrap();
    let mut grad = params;
    grad[4] = input;
    let args = values(&mut ctx, &grad);
    BatchNormGradOp::new(&mut ctx, expected, args, config.clone())
        .verify(&ctx)
        .unwrap();
    grad[1] = a(&ctx, P::F64, &[3]);
    let args = values(&mut ctx, &grad);
    let op = BatchNormGradOp::new(&mut ctx, expected, args, config);
    assert!(op.verify(&ctx).is_err());
    instruction(
        &ctx,
        op.get_operation(),
        O::BatchNormGrad,
        VerifyOptions {
            allow_mixed_precision: true,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn numerical_instructions_reject_malformed_attributes_arity_and_results() {
    let mut ctx = context();
    let matrix = a(&ctx, P::F32, &[3, 3]);
    let scalar = s(&ctx, P::F32);
    let vector = a(&ctx, P::F32, &[3]);
    let result = shape::tuple(&ctx, vec![vector, a(&ctx, P::S32, &[3])]).unwrap();
    let cases = [
        (
            O::Dot,
            DotOp::new(
                &mut ctx,
                matrix,
                vec![],
                dot_dims(&[1], &[0], &[], &[]),
                precision(),
            )
            .get_operation(),
        ),
        (
            O::Reduce,
            ReduceOp::new(&mut ctx, scalar, vec![], Axes(vec![0]), callees(1)).get_operation(),
        ),
        (
            O::ReduceWindow,
            ReduceWindowOp::new(&mut ctx, vector, vec![], Window(vec![]), callees(1))
                .get_operation(),
        ),
        (
            O::SelectAndScatter,
            SelectAndScatterOp::new(&mut ctx, matrix, vec![], Window(vec![]), callees(2))
                .get_operation(),
        ),
        (
            O::Map,
            MapOp::new(&mut ctx, matrix, vec![], callees(1)).get_operation(),
        ),
        (
            O::RaggedDot,
            RaggedDotOp::new(
                &mut ctx,
                matrix,
                vec![],
                RaggedDotDimensionNumbers {
                    dot_dimension_numbers: dot_dims(&[1], &[0], &[], &[]),
                    lhs_ragged_dimensions: Axes(vec![0]),
                    rhs_group_dimensions: Axes(vec![0]),
                },
                precision(),
            )
            .get_operation(),
        ),
        (
            O::ScaledDot,
            ScaledDotOp::new(
                &mut ctx,
                matrix,
                vec![],
                dot_dims(&[1], &[0], &[], &[]),
                precision(),
            )
            .get_operation(),
        ),
        (
            O::Sort,
            SortOp::new(&mut ctx, matrix, vec![], Axes(vec![0]), callees(1)).get_operation(),
        ),
        (
            O::TopK,
            TopKOp::new(
                &mut ctx,
                result,
                vec![],
                NumericalInteger(3),
                SortOptions { descending: true },
            )
            .get_operation(),
        ),
        (
            O::Fft,
            FftOp::new(&mut ctx, matrix, vec![], FftKind::Rfft, Axes(vec![3])).get_operation(),
        ),
        (
            O::Cholesky,
            CholeskyOp::new(&mut ctx, matrix, vec![], CholeskyOptions { lower: false })
                .get_operation(),
        ),
        (
            O::TriangularSolve,
            TriangularSolveOp::new(
                &mut ctx,
                matrix,
                vec![],
                TriangularSolveOptions {
                    left_side: true,
                    lower: true,
                    unit_diagonal: true,
                    transpose_a: Transpose::NoTranspose,
                },
            )
            .get_operation(),
        ),
        (
            O::BatchNormTraining,
            BatchNormTrainingOp::new(
                &mut ctx,
                matrix,
                vec![],
                BatchNormOptions {
                    epsilon_bits: 0,
                    feature_index: 1,
                },
            )
            .get_operation(),
        ),
        (
            O::BatchNormInference,
            BatchNormInferenceOp::new(
                &mut ctx,
                matrix,
                vec![],
                BatchNormOptions {
                    epsilon_bits: 0,
                    feature_index: 1,
                },
            )
            .get_operation(),
        ),
        (
            O::BatchNormGrad,
            BatchNormGradOp::new(
                &mut ctx,
                matrix,
                vec![],
                BatchNormOptions {
                    epsilon_bits: 0,
                    feature_index: 1,
                },
            )
            .get_operation(),
        ),
    ];
    for (opcode, op) in cases {
        assert!(
            instruction(&ctx, op, opcode, VerifyOptions::default()).is_err(),
            "{opcode}"
        );
    }

    let args = values(&mut ctx, &[matrix]);
    let op = CholeskyOp::new(&mut ctx, matrix, args, CholeskyOptions { lower: false });
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_options"), Axes(vec![]));
    assert!(op.verify(&ctx).is_err());
    let args = values(&mut ctx, &[matrix]);
    let op = SortOp::new(&mut ctx, matrix, args, Axes(vec![0]), callees(0));
    assert!(op.verify(&ctx).is_err());
    let args = values(&mut ctx, &[matrix]);
    let op = CholeskyOp::new(&mut ctx, vector, args, CholeskyOptions { lower: false });
    assert!(op.verify(&ctx).is_err());
    let args = values(&mut ctx, &[matrix]);
    let op = ScanOp::new(
        &mut ctx,
        matrix,
        args,
        Axes(vec![]),
        NumericalInteger(0),
        TriState::True,
        callees(1),
    );
    assert!(op.verify(&ctx).is_err());
}

#[test]
fn batch_norm_checks_parameter_ranks_before_element_domains() {
    let ctx = context();
    let input = dynamic(&ctx, P::F32, &[D::Static(2), D::Bounded(3), D::Static(4)]);
    let vector = a(&ctx, P::F32, &[3]);
    let scalar = s(&ctx, P::F32);
    for (opcode, parameters) in [
        (O::BatchNormTraining, vec![input, vector, scalar]),
        (
            O::BatchNormInference,
            vec![input, vector, scalar, vector, vector],
        ),
        (O::BatchNormGrad, vec![input, scalar, vector, vector, input]),
    ] {
        error_contains(
            inference::infer_batch_norm(&ctx, opcode, &parameters, 3),
            "axis",
        );
    }
    for (opcode, mut parameters) in [
        (O::BatchNormTraining, vec![input, vector, vector]),
        (
            O::BatchNormInference,
            vec![input, vector, vector, vector, vector],
        ),
        (O::BatchNormGrad, vec![input, vector, vector, vector, input]),
    ] {
        parameters[1] = a(&ctx, P::S32, &[3]);
        assert!(inference::infer_batch_norm(&ctx, opcode, &parameters, 1).is_err());
        assert!(inference::infer_batch_norm(&ctx, opcode, &[], 1).is_err());
    }
    for (index, bad) in [(3, a(&ctx, P::F32, &[2, 3])), (4, scalar)] {
        let mut parameters = vec![input, vector, vector, vector, vector];
        parameters[index] = bad;
        assert!(inference::infer_batch_norm(&ctx, O::BatchNormInference, &parameters, 1).is_err());
    }
    let invalid_type_with_bad_scale = [a(&ctx, P::S32, &[2, 3]), scalar, vector];
    error_contains(
        inference::infer_batch_norm(&ctx, O::BatchNormTraining, &invalid_type_with_bad_scale, 1),
        "must be a vector",
    );
    let invalid_grad = [
        a(&ctx, P::S32, &[2, 3]),
        vector,
        scalar,
        vector,
        a(&ctx, P::F32, &[2, 3]),
    ];
    error_contains(
        inference::infer_batch_norm(&ctx, O::BatchNormGrad, &invalid_grad, 1),
        "must be a vector",
    );
    let invalid_grad_types = [
        input,
        a(&ctx, P::S32, &[3]),
        vector,
        vector,
        a(&ctx, P::S32, &[2, 3, 4]),
    ];
    error_contains(
        inference::infer_batch_norm(&ctx, O::BatchNormGrad, &invalid_grad_types, 1),
        "floating point",
    );
}

#[test]
fn matrix_decompositions_preserve_layout_buffers_and_unbounded_square_compatibility() {
    let ctx = context();
    let matrix = shape::with_layout(
        &ctx,
        dynamic(&ctx, P::F64, &[D::Static(2), D::Unbounded, D::Static(3)]),
        Layout {
            minor_to_major: vec![0, 1, 2],
            ..Default::default()
        },
    )
    .unwrap();
    let rhs = shape::with_layout(
        &ctx,
        dynamic(&ctx, P::F64, &[D::Static(2), D::Bounded(3), D::Static(5)]),
        Layout {
            minor_to_major: vec![1, 0, 2],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(inference::infer_cholesky(&ctx, matrix).unwrap(), matrix);
    let options = TriangularSolveOptions {
        left_side: true,
        lower: false,
        unit_diagonal: false,
        transpose_a: Transpose::NoTranspose,
    };
    assert_eq!(
        inference::infer_triangular_solve(&ctx, matrix, rhs, &options).unwrap(),
        rhs
    );
    let matrix = shape::buffer(&ctx, matrix).unwrap();
    let rhs = shape::buffer(&ctx, rhs).unwrap();
    assert_eq!(inference::infer_cholesky(&ctx, matrix).unwrap(), matrix);
    assert_eq!(
        inference::infer_triangular_solve(&ctx, matrix, rhs, &options).unwrap(),
        rhs
    );
    error_contains(
        inference::infer_triangular_solve(
            &ctx,
            a(&ctx, P::F32, &[3, 4]),
            s(&ctx, P::F32),
            &options,
        ),
        "identical rank",
    );
    error_contains(
        inference::infer_cholesky(&ctx, shape::tuple(&ctx, vec![matrix]).unwrap()),
        "floating or complex",
    );
}
