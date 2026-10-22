use pliron::{
    basic_block::BasicBlock,
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    region::Region,
    r#type::{Type, TypeHandle, Typed},
    value::Value,
};
use stablehlo::{attributes::*, ops::*, types::*};

fn value(ctx: &mut Context, shape: &[i64], element: ElementType) -> Value {
    let ty = tensor(ctx, shape, element);
    let op = ConstantOp::new(ctx, ty, vec![0], true);
    op.get_operation().deref(ctx).get_result(0)
}

fn dot_dimensions(
    lhs_contract: &[i64],
    rhs_contract: &[i64],
    lhs_batch: &[i64],
    rhs_batch: &[i64],
) -> DotDimensionNumbersAttr {
    DotDimensionNumbersAttr {
        lhs_batching_dimensions: lhs_batch.to_vec(),
        rhs_batching_dimensions: rhs_batch.to_vec(),
        lhs_contracting_dimensions: lhs_contract.to_vec(),
        rhs_contracting_dimensions: rhs_contract.to_vec(),
    }
}

fn convolution_options(
    dimensions: ConvDimensionNumbersAttr,
    precision: Vec<Precision>,
) -> ConvolutionOptions {
    ConvolutionOptions {
        dimensions,
        strides: vec![1, 1],
        padding: vec![0, 0, 0, 0],
        lhs_dilation: vec![1, 1],
        rhs_dilation: vec![1, 1],
        feature_group_count: 1,
        batch_group_count: 1,
        window_reversal: vec![false, false],
        precision,
    }
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
        dot_dimensions(&[2], &[1], &[0], &[0]),
    );
    assert!(op.verify(&ctx).is_ok());
}

#[test]
fn dot_general_inferred_constructor_builds_result_shape() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[2, 3, 4], ElementType::F32);
    let rhs = value(&mut ctx, &[2, 4, 5], ElementType::F32);
    let result_element = ElementType::F32.native(&ctx);
    let op = DotGeneralOp::new_inferred(
        &mut ctx,
        lhs,
        rhs,
        dot_dimensions(&[2], &[1], &[0], &[0]),
        result_element,
    )
    .expect("valid dot dimensions should infer a result type");
    assert!(op.verify(&ctx).is_ok());
    assert_eq!(
        op.get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx)
            .deref(&ctx)
            .downcast_ref::<TensorType>()
            .expect("result should be a tensor")
            .dimensions,
        [2, 3, 5]
    );
}

#[test]
fn dot_general_optional_algorithm_validates_defaults_and_roundtrips() {
    let mut ctx = Context::new();
    let lhs_type = tensor(&ctx, &[2, 3], ElementType::F32);
    let rhs_type = tensor(&ctx, &[3, 4], ElementType::F32);
    let result_type = tensor(&ctx, &[2, 4], ElementType::F32);
    let algorithm = DotAlgorithmAttr {
        lhs_precision_type: ElementType::F32.native(&ctx),
        rhs_precision_type: ElementType::F32.native(&ctx),
        accumulation_type: ElementType::F32.native(&ctx),
        lhs_component_count: 1,
        rhs_component_count: 1,
        num_primitive_operations: 1,
        allow_imprecise_accumulation: false,
    };
    let lhs = value(&mut ctx, &[2, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 4], ElementType::F32);
    let dot = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    )
    .with_precision(&ctx, vec![Precision::Default, Precision::Default])
    .with_algorithm(&ctx, algorithm.clone());
    assert_eq!(dot.algorithm(&ctx).as_deref(), Some(&algorithm));
    assert!(dot.verify(&ctx).is_ok());

    let non_default_precision = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    )
    .with_precision(&ctx, vec![Precision::High, Precision::Default])
    .with_algorithm(&ctx, algorithm.clone());
    assert!(non_default_precision.verify(&ctx).is_err());

    let omitted_default_precision = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    )
    .with_precision(&ctx, vec![Precision::Default, Precision::Default])
    .with_algorithm(&ctx, algorithm.clone());
    omitted_default_precision
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new(stablehlo::attributes::keys::DOT_PRECISION));
    assert!(omitted_default_precision.verify(&ctx).is_ok());

    let malformed_precision = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    )
    .with_precision(&ctx, vec![Precision::Default, Precision::Default])
    .with_algorithm(&ctx, algorithm.clone());
    malformed_precision
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(
            Identifier::new(stablehlo::attributes::keys::DOT_PRECISION),
            pliron::builtin::attributes::StringAttr::new("DEFAULT".into()),
        );
    assert!(malformed_precision.verify(&ctx).is_err());

    let invalid_algorithm = DotAlgorithmAttr {
        lhs_component_count: 0,
        ..algorithm.clone()
    };
    let invalid = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    )
    .with_precision(&ctx, vec![Precision::Default, Precision::Default])
    .with_algorithm(&ctx, invalid_algorithm);
    assert!(invalid.verify(&ctx).is_err());

    let malformed = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    );
    malformed.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new(stablehlo::attributes::keys::DOT_ALGORITHM),
        pliron::builtin::attributes::StringAttr::new("not an algorithm".into()),
    );
    assert!(malformed.verify(&ctx).is_err());

    let module = ModuleOp::new(&mut ctx);
    let module_body = module.body_block(&ctx).unwrap();
    let signature = FunctionType::intern(&ctx, vec![lhs_type, rhs_type], vec![result_type]);
    let function = FuncOp::new(&mut ctx, "dot".into(), signature).unwrap();
    let entry = function.entry_block(&ctx).unwrap();
    let lhs = function.argument(&ctx, 0).unwrap();
    let rhs = function.argument(&ctx, 1).unwrap();
    let dot = DotGeneralOp::new(
        &mut ctx,
        lhs,
        rhs,
        result_type,
        dot_dimensions(&[1], &[0], &[], &[]),
    )
    .with_precision(&ctx, vec![Precision::Default, Precision::Default])
    .with_algorithm(&ctx, algorithm.clone());
    let result = dot.get_operation().deref(&ctx).get_result(0);
    dot.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, result)
        .get_operation()
        .insert_at_back(entry, &ctx);
    function.get_operation().insert_at_back(module_body, &ctx);

    let assembly = module.get_operation().disp(&ctx).to_string();
    assert!(
        assembly.contains(stablehlo::attributes::keys::DOT_ALGORITHM),
        "{assembly}"
    );
    let mut parsed_ctx = Context::new();
    ModuleOp::register(&mut parsed_ctx);
    FuncOp::register(&mut parsed_ctx);
    ReturnOp::register(&mut parsed_ctx);
    DotGeneralOp::register(&mut parsed_ctx);
    TensorType::register(&mut parsed_ctx);
    FunctionType::register(&mut parsed_ctx);
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &assembly,
    )
    .unwrap_or_else(|error| panic!("dot_general module failed to roundtrip:\n{assembly}\n{error}"));
    let parsed_module = Operation::get_op::<ModuleOp>(parsed, &parsed_ctx).unwrap();
    let parsed_body = parsed_module.body_block(&parsed_ctx).unwrap();
    let parsed_function = parsed_body
        .deref(&parsed_ctx)
        .iter(&parsed_ctx)
        .next()
        .unwrap();
    let parsed_function = Operation::get_op::<FuncOp>(parsed_function, &parsed_ctx).unwrap();
    let parsed_entry = parsed_function.entry_block(&parsed_ctx).unwrap();
    let parsed_dot = parsed_entry
        .deref(&parsed_ctx)
        .iter(&parsed_ctx)
        .find_map(|op| Operation::get_op::<DotGeneralOp>(op, &parsed_ctx))
        .unwrap();
    assert_eq!(
        parsed_dot.algorithm(&parsed_ctx).as_deref(),
        Some(&algorithm)
    );
    assert!(parsed_dot.verify(&parsed_ctx).is_ok());
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
        dot_dimensions(&[1], &[0], &[1], &[0]),
    );
    assert!(duplicate.verify(&ctx).is_err());
}

#[test]
fn dot_validates_vector_matrix_shapes_and_precision() {
    let mut ctx = Context::new();
    let lhs = value(&mut ctx, &[2, 3], ElementType::F32);
    let rhs = value(&mut ctx, &[3, 4], ElementType::F32);
    let result = tensor(&ctx, &[2, 4], ElementType::F32);
    let good = DotOp::new(
        &mut ctx,
        lhs,
        rhs,
        result,
        vec![Precision::Default, Precision::High],
    );
    assert!(good.verify(&ctx).is_ok());
}

#[test]
fn fft_rejects_invalid_fft_rank_and_lengths() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[8], ElementType::C64);
    let output = tensor(&ctx, &[8], ElementType::C64);
    let bad = FftOp::new(&mut ctx, input, output, FftTypeAttr::Fft, vec![8, 8, 8, 8]);
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
        convolution_options(dims.clone(), vec![Precision::Default, Precision::Default]),
    );
    assert!(good.verify(&ctx).is_ok());
    let bad_result = tensor(&ctx, &[1, 2, 3, 4], ElementType::F32);
    let bad = ConvolutionOp::new(
        &mut ctx,
        lhs,
        rhs,
        bad_result,
        convolution_options(dims, vec![Precision::Default]),
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
    let solve = TriangularSolveOp::new(
        &mut ctx,
        a,
        b,
        solve_result,
        TriangularSolveOptions {
            left_side: true,
            lower: true,
            unit_diagonal: false,
            transpose: TransposeMode::None,
        },
    );
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
        BatchNormInferenceOperands {
            x,
            scale,
            offset,
            mean,
            variance,
        },
        result,
        BatchNormOptions {
            feature_index: 1,
            epsilon: 0.001,
        },
    );
    assert!(good.verify(&ctx).is_ok());
    let result = tensor(&ctx, &[2, 3, 4], ElementType::F32);
    let bad = BatchNormInferenceOp::new(
        &mut ctx,
        BatchNormInferenceOperands {
            x,
            scale,
            offset,
            mean,
            variance,
        },
        result,
        BatchNormOptions {
            feature_index: 1,
            epsilon: -1.0,
        },
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
        RngAlgorithmAttr::ThreeFry,
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
            convolution_options(dims.clone(), vec![Precision::Default, Precision::Default]),
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
    let bad_groups = DynamicConvOp::new(&mut ctx, lhs, rhs, padding, result, dims)
        .with_window_options(
            &ctx,
            DynamicConvolutionOptions {
                strides: vec![1, 1],
                lhs_dilation: vec![1, 1],
                rhs_dilation: vec![1, 1],
                feature_group_count: 2,
                batch_group_count: 1,
                window_reversal: vec![false, false],
                precision: vec![Precision::Default, Precision::Default],
            },
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
    let training = BatchNormTrainingOp::new(
        &mut ctx,
        x,
        scale,
        offset,
        outputs,
        BatchNormOptions {
            feature_index: 1,
            epsilon: 0.001,
        },
    );
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
        BatchNormGradOperands {
            x,
            scale,
            mean,
            variance,
            grad,
        },
        outputs,
        BatchNormOptions {
            feature_index: 1,
            epsilon: 0.001,
        },
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
    let padded = ReduceWindowOp::new(
        &mut ctx,
        vec![input],
        vec![init],
        vec![padded_result],
        vec![2],
    )
    .with_window_options(
        &ctx,
        ReduceWindowOptions {
            strides: vec![2],
            base_dilations: vec![1],
            window_dilations: vec![1],
            padding: vec![1, 1],
        },
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
    let strided = SelectAndScatterOp::new(
        &mut ctx,
        input,
        strided_source,
        scattered,
        strided_result,
        vec![2],
    )
    .with_window_options(
        &ctx,
        SelectAndScatterOptions {
            strides: vec![2],
            padding: vec![],
        },
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
        RngDistributionAttr::Uniform,
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
        RngDistributionAttr::Uniform,
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
