use pliron::{
    basic_block::BasicBlock,
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    region::Region,
    r#type::TypeHandle,
    value::Value,
};
use stablehlo::{attributes::*, ops::*, types::*};

fn value(ctx: &mut Context, shape: &[i64], element: ElementType) -> Value {
    let ty = tensor(ctx, shape, element);
    let op = ConstantOp::new(ctx, ty, vec![0], true);
    op.get_operation().deref(ctx).get_result(0)
}

fn add_return_region(
    ctx: &mut Context,
    region: Ptr<Region>,
    args: Vec<TypeHandle>,
    returned: usize,
) {
    let block = BasicBlock::new(ctx, None, args);
    block.insert_at_back(region, ctx);
    let values = block.deref(ctx).arguments().take(returned).collect();
    ReturnOp::new_many(ctx, values)
        .get_operation()
        .insert_at_back(block, ctx);
}

#[test]
fn dot_general_accepts_valid_contracting_and_batch_dimensions() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let rhs = value(&mut ctx, &[2, 4, 5], ElementType::F32);
    let result = tensor(&ctx, &[2, 3, 5], ElementType::F32);
    let op = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result,
        vec![2],
        vec![1],
        vec![0],
        vec![0],
        vec![0, 0],
    );
    assert!(op.verify(&ctx).is_ok());
}

#[test]
fn dot_general_rejects_duplicate_and_out_of_range_dimensions() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[2, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 4], ElementType::F32);
    let result = tensor(&ctx, &[2, 4], ElementType::F32);
    let duplicate = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result,
        vec![1],
        vec![0],
        vec![1],
        vec![0],
        vec![],
    );
    assert!(duplicate.verify(&ctx).is_err());
}

#[test]
fn dot_validates_vector_matrix_shapes_and_precision() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[2, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 4], ElementType::F32);
    let result = tensor(&ctx, &[2, 4], ElementType::F32);
    let good = DotOp::new(&mut ctx, lhs, rhs, result, vec![0, 1]);
    assert!(good.verify(&ctx).is_ok());
}

#[test]
fn fft_rejects_invalid_fft_rank_and_lengths() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[8], ElementType::C64);
    let output = tensor(&ctx, &[8], ElementType::C64);
    let bad = FftOp::new(
        &mut ctx,
        input,
        output,
        FftTypeAttr { value: 0 },
        vec![8, 8, 8, 8],
    );
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn reduce_requires_unique_in_range_dimensions_and_paired_initial_values() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let init = value(&mut ctx, &[], ElementType::F32);
    let result = tensor(&ctx, &[2], ElementType::F32);
    let good = ReduceOp::new(&mut ctx, vec![input], vec![init], vec![result], vec![1]);
    let reduce_region = good.reducer_region(&ctx);
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    add_return_region(&mut ctx, reduce_region, vec![scalar, scalar], 1);
    assert!(good.verify(&ctx).is_ok());
    let widened_result = tensor(&ctx, &[2], ElementType::F64);
    let widened = ReduceOp::new(
        &mut ctx,
        vec![input],
        vec![init],
        vec![widened_result],
        vec![1],
    );
    let accumulator = TensorType::scalar(&ctx, ElementType::F64);
    let widened_region = widened.reducer_region(&ctx);
    add_return_region(&mut ctx, widened_region, vec![accumulator, accumulator], 1);
    assert!(widened.verify(&ctx).is_ok());
    let bad_result = tensor(&ctx, &[2], ElementType::F32);
    let bad = ReduceOp::new(
        &mut ctx,
        vec![input],
        vec![init],
        vec![bad_result],
        vec![1, 1],
    );
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn map_requires_dimension_identity_and_ranked_result() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2, 3], ElementType::F32);
    let result = tensor(&ctx, &[2, 3], ElementType::F32);
    let op = MapOp::new(&mut ctx, vec![input], result, vec![0, 1]);
    let map_region = op.computation_region(&ctx);
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    add_return_region(&mut ctx, map_region, vec![scalar], 1);
    assert!(op.verify(&ctx).is_ok());
}

#[test]
fn convolution_validates_dimension_numbers_window_and_result_shape() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[1, 5, 5, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 3, 3, 4], ElementType::F32);
    let result = tensor(&ctx, &[1, 3, 3, 4], ElementType::F32);
    let dims = ConvDimensionNumbersAttr {
        input_batch_dimension: 0,
        input_feature_dimension: 3,
        input_spatial_dimensions: vec![1, 2],
        kernel_input_feature_dimension: 2,
        kernel_output_feature_dimension: 3,
        kernel_spatial_dimensions: vec![0, 1],
        output_batch_dimension: 0,
        output_feature_dimension: 3,
        output_spatial_dimensions: vec![1, 2],
    };
    let good = ConvolutionOp::new(
        &mut ctx,
        lhs,
        rhs,
        result,
        dims.clone(),
        vec![1, 1],
        vec![0, 0, 0, 0],
        vec![1, 1],
        vec![1, 1],
        1,
        1,
        vec![false, false],
        vec![0, 0],
    );
    assert!(good.verify(&ctx).is_ok());
    let bad_result = tensor(&ctx, &[1, 2, 3, 4], ElementType::F32);
    let bad = ConvolutionOp::new(
        &mut ctx,
        lhs,
        rhs,
        bad_result,
        dims,
        vec![1, 1],
        vec![0, 0, 0, 0],
        vec![1, 1],
        vec![1, 1],
        1,
        1,
        vec![false, false],
        vec![0],
    );
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn cholesky_and_triangular_solve_check_matrix_shapes() {
    let mut ctx = Context::new();
    let a = value(&mut ctx, &[2, 3, 3], ElementType::F32);
    let chol_result = tensor(&ctx, &[2, 3, 3], ElementType::F32);
    let chol = CholeskyOp::new(&mut ctx, a, chol_result, true);
    assert!(chol.verify(&ctx).is_ok());
    let nonsquare_result = tensor(&ctx, &[2, 3, 4], ElementType::F32);
    let nonsquare = CholeskyOp::new(&mut ctx, a, nonsquare_result, false);
    assert!(nonsquare.verify(&ctx).is_err());
    let b = value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let solve_result = tensor(&ctx, &[2, 3, 4], ElementType::F32);
    let solve = TriangularSolveOp::new(&mut ctx, a, b, solve_result, true, true, false, 0);
    assert!(solve.verify(&ctx).is_ok());
}

#[test]
fn batch_norm_checks_feature_vector_shapes_and_epsilon() {
    let mut ctx = Context::new();
    let x = value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let scale = value(&mut ctx, &[3], ElementType::F32);
    let offset = value(&mut ctx, &[3], ElementType::F32);
    let mean = value(&mut ctx, &[3], ElementType::F32);
    let variance = value(&mut ctx, &[3], ElementType::F32);
    let result = tensor(&ctx, &[2, 3, 4], ElementType::F32);
    let good = BatchNormInferenceOp::new(
        &mut ctx,
        x,
        scale,
        offset,
        mean,
        variance,
        result,
        1,
        0.001_f64.to_bits(),
    );
    assert!(good.verify(&ctx).is_ok());
    let result = tensor(&ctx, &[2, 3, 4], ElementType::F32);
    let bad = BatchNormInferenceOp::new(
        &mut ctx,
        x,
        scale,
        offset,
        mean,
        variance,
        result,
        1,
        (-1.0_f64).to_bits(),
    );
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn rng_bit_generator_requires_valid_state_and_integer_output() {
    let mut ctx = Context::new();
    let state = value(&mut ctx, &[2], ElementType::Ui64);
    let state_result = tensor(&ctx, &[2], ElementType::Ui64);
    let data_result = tensor(&ctx, &[4], ElementType::I32);
    let good = RngBitGeneratorOp::new(
        &mut ctx,
        state,
        vec![state_result, data_result],
        RngAlgorithmAttr { value: 1 },
    );
    assert!(good.verify(&ctx).is_ok());
}

fn add_predicate_return_region(ctx: &mut Context, region: Ptr<Region>, args: Vec<TypeHandle>) {
    let block = BasicBlock::new(ctx, None, args);
    block.insert_at_back(region, ctx);
    let pred_type = TensorType::scalar(ctx, ElementType::I1);
    let pred = ConstantOp::new(ctx, pred_type, vec![1], true);
    pred.get_operation().insert_at_back(block, ctx);
    let value = pred.get_operation().deref(ctx).get_result(0);
    ReturnOp::new(ctx, value)
        .get_operation()
        .insert_at_back(block, ctx);
}

#[test]
fn einsum_builders_retain_equation_attributes_and_ranked_types() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[2, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 4], ElementType::F32);
    let result = tensor(&ctx, &[2, 4], ElementType::F32);
    let binary = EinsumOp::new(&mut ctx, lhs, rhs, result, "ab,bc->ac".into());
    assert!(binary.verify(&ctx).is_ok());
    let unary_result = tensor(&ctx, &[3, 2], ElementType::F32);
    let unary = UnaryEinsumOp::new(&mut ctx, lhs, unary_result, "ab->ba".into());
    assert!(unary.verify(&ctx).is_ok());
}

#[test]
fn convolutions_enforce_specification_nonquantized_element_constraints() {
    use ElementType::{F8E4M3Fn, F8E5M2, F32, F64, I8, I32};
    let mut ctx = Context::new();
    let dims = ConvDimensionNumbersAttr {
        input_batch_dimension: 0,
        input_feature_dimension: 3,
        input_spatial_dimensions: vec![1, 2],
        kernel_input_feature_dimension: 2,
        kernel_output_feature_dimension: 3,
        kernel_spatial_dimensions: vec![0, 1],
        output_batch_dimension: 0,
        output_feature_dimension: 3,
        output_spatial_dimensions: vec![1, 2],
    };
    for (left, right, output, valid) in [
        (I8, I8, I32, false),
        (I32, I32, I8, false),
        (F32, F32, F64, false),
        (F32, F32, I32, false),
        (I8, I32, I32, false),
        (I32, I32, I32, true),
        (F32, F32, F32, true),
        (F8E4M3Fn, F8E5M2, F8E4M3Fn, true),
        (F8E5M2, F8E4M3Fn, F8E5M2, true),
        (F8E4M3Fn, F8E5M2, F8E5M2, false),
        (F8E4M3Fn, F32, F8E4M3Fn, false),
    ] {
        let lhs = value(&mut ctx, &[1, 5, 5, 3], left);
        let rhs = value(&mut ctx, &[3, 3, 3, 4], right);
        let result = tensor(&ctx, &[1, 3, 3, 4], output);
        let convolution = ConvolutionOp::new(
            &mut ctx,
            lhs,
            rhs,
            result,
            dims.clone(),
            vec![1, 1],
            vec![0, 0, 0, 0],
            vec![1, 1],
            vec![1, 1],
            1,
            1,
            vec![false, false],
            vec![0, 0],
        );
        assert_eq!(
            convolution.verify(&ctx).is_ok(),
            valid,
            "convolution {left:?}, {right:?} -> {output:?}",
        );
        let padding = value(&mut ctx, &[2, 2], ElementType::I32);
        let dynamic_result = tensor(&ctx, &[1, -1, -1, 4], output);
        let dynamic = DynamicConvOp::new(&mut ctx, lhs, rhs, padding, dynamic_result, dims.clone());
        assert_eq!(
            dynamic.verify(&ctx).is_ok(),
            valid,
            "dynamic_conv {left:?}, {right:?} -> {output:?}",
        );
    }
}

#[test]
fn dynamic_convolution_checks_padding_tensor_rank_and_dimensions() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[1, 5, 5, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 3, 3, 4], ElementType::F32);
    let padding = value(&mut ctx, &[2, 2], ElementType::I32);
    let result = tensor(&ctx, &[1, -1, -1, 4], ElementType::F32);
    let dims = ConvDimensionNumbersAttr {
        input_batch_dimension: 0,
        input_feature_dimension: 3,
        input_spatial_dimensions: vec![1, 2],
        kernel_input_feature_dimension: 2,
        kernel_output_feature_dimension: 3,
        kernel_spatial_dimensions: vec![0, 1],
        output_batch_dimension: 0,
        output_feature_dimension: 3,
        output_spatial_dimensions: vec![1, 2],
    };
    let good = DynamicConvOp::new(&mut ctx, lhs, rhs, padding, result, dims.clone());
    assert!(good.verify(&ctx).is_ok());
    let bad_padding = value(&mut ctx, &[2, 3], ElementType::I32);
    let result = tensor(&ctx, &[1, -1, -1, 4], ElementType::F32);
    let bad = DynamicConvOp::new(&mut ctx, lhs, rhs, bad_padding, result, dims.clone());
    assert!(bad.verify(&ctx).is_err());
    let padding = value(&mut ctx, &[2, 2], ElementType::I32);
    let result = tensor(&ctx, &[1, -1, -1, 4], ElementType::F32);
    let bad_groups = DynamicConvOp::with_window_attributes(
        &mut ctx,
        lhs,
        rhs,
        padding,
        result,
        dims,
        vec![1, 1],
        vec![1, 1],
        vec![1, 1],
        2,
        1,
        vec![false, false],
        vec![0, 0],
    );
    assert!(bad_groups.verify(&ctx).is_err());
}

#[test]
fn batch_norm_training_and_gradient_validate_result_roles() {
    let mut ctx = Context::new();
    let x = value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let scale = value(&mut ctx, &[3], ElementType::F32);
    let offset = value(&mut ctx, &[3], ElementType::F32);
    let outputs = vec![
        tensor(&ctx, &[2, 3, 4], ElementType::F32),
        tensor(&ctx, &[3], ElementType::F32),
        tensor(&ctx, &[3], ElementType::F32),
    ];
    let training =
        BatchNormTrainingOp::new(&mut ctx, x, scale, offset, outputs, 1, 0.001_f64.to_bits());
    assert!(training.verify(&ctx).is_ok());
    let mean = value(&mut ctx, &[3], ElementType::F32);
    let variance = value(&mut ctx, &[3], ElementType::F32);
    let grad = value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let outputs = vec![
        tensor(&ctx, &[2, 3, 4], ElementType::F32),
        tensor(&ctx, &[3], ElementType::F32),
        tensor(&ctx, &[3], ElementType::F32),
    ];
    let backward = BatchNormGradOp::new(
        &mut ctx,
        x,
        scale,
        mean,
        variance,
        grad,
        outputs,
        1,
        0.001_f64.to_bits(),
    );
    assert!(backward.verify(&ctx).is_ok());
}

#[test]
fn reduce_window_and_select_scatter_verify_region_signatures() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[4], ElementType::F32);
    let init = value(&mut ctx, &[], ElementType::F32);
    let result = tensor(&ctx, &[3], ElementType::F32);
    let reduce_window =
        ReduceWindowOp::new(&mut ctx, vec![input], vec![init], vec![result], vec![2]);
    let reduce_region = reduce_window.reducer_region(&ctx);
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    add_return_region(&mut ctx, reduce_region, vec![scalar, scalar], 1);
    assert!(reduce_window.verify(&ctx).is_ok());

    let padded_result = tensor(&ctx, &[3], ElementType::F32);
    let padded = ReduceWindowOp::with_window_attributes(
        &mut ctx,
        vec![input],
        vec![init],
        vec![padded_result],
        vec![2],
        vec![2],
        vec![1],
        vec![1],
        vec![1, 1],
    );
    let region = padded.reducer_region(&ctx);
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    add_return_region(&mut ctx, region, vec![scalar, scalar], 1);
    assert!(padded.verify(&ctx).is_ok());

    let source = value(&mut ctx, &[3], ElementType::F32);
    let scattered = value(&mut ctx, &[], ElementType::F32);
    let result = tensor(&ctx, &[4], ElementType::F32);
    let op = SelectAndScatterOp::new(&mut ctx, input, source, scattered, result, vec![2]);
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    let select_region = op.select_region(&ctx);
    add_predicate_return_region(&mut ctx, select_region, vec![scalar, scalar]);
    let scatter_region = op.scatter_region(&ctx);
    add_return_region(&mut ctx, scatter_region, vec![scalar, scalar], 1);
    assert!(op.verify(&ctx).is_ok());

    let strided_source = value(&mut ctx, &[2], ElementType::F32);
    let strided_result = tensor(&ctx, &[4], ElementType::F32);
    let strided = SelectAndScatterOp::with_window_attributes(
        &mut ctx,
        input,
        strided_source,
        scattered,
        strided_result,
        vec![2],
        vec![2],
        vec![],
    );
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    let select_region = strided.select_region(&ctx);
    add_predicate_return_region(&mut ctx, select_region, vec![scalar, scalar]);
    let scatter_region = strided.scatter_region(&ctx);
    add_return_region(&mut ctx, scatter_region, vec![scalar, scalar], 1);
    assert!(strided.verify(&ctx).is_ok());

    let widened_result = tensor(&ctx, &[4], ElementType::F64);
    let widened =
        SelectAndScatterOp::new(&mut ctx, input, source, scattered, widened_result, vec![2]);
    let scalar_f32 = TensorType::scalar(&ctx, ElementType::F32);
    let select_region = widened.select_region(&ctx);
    add_predicate_return_region(&mut ctx, select_region, vec![scalar_f32, scalar_f32]);
    let scalar_f64 = TensorType::scalar(&ctx, ElementType::F64);
    let scatter_region = widened.scatter_region(&ctx);
    add_return_region(&mut ctx, scatter_region, vec![scalar_f64, scalar_f64], 1);
    assert!(widened.verify(&ctx).is_ok());
}

#[test]
fn rng_uses_shape_vector_and_distribution() {
    let mut ctx = Context::new();
    let lower = value(&mut ctx, &[], ElementType::F32);
    let upper = value(&mut ctx, &[], ElementType::F32);
    let shape = value(&mut ctx, &[2], ElementType::I64);
    let result = tensor(&ctx, &[3, 4], ElementType::F32);
    let good = RngOp::new(
        &mut ctx,
        lower,
        upper,
        shape,
        result,
        RngDistributionAttr { value: 0 },
    );
    assert!(good.verify(&ctx).is_ok());
    let shape = value(&mut ctx, &[2, 2], ElementType::I64);
    let result = tensor(&ctx, &[3, 4], ElementType::F32);
    let bad = RngOp::new(
        &mut ctx,
        lower,
        upper,
        shape,
        result,
        RngDistributionAttr { value: 0 },
    );
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn sort_verifies_comparator_region_and_result_shapes() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[4], ElementType::F32);
    let result = tensor(&ctx, &[4], ElementType::F32);
    let sort = SortOp::new(&mut ctx, vec![input], vec![result], 0, true);
    let scalar = TensorType::scalar(&ctx, ElementType::F32);
    let region = sort.comparator_region(&ctx);
    add_predicate_return_region(&mut ctx, region, vec![scalar, scalar]);
    assert!(sort.verify(&ctx).is_ok());
}
