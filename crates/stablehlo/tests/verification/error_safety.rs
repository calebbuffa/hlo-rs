use pliron::{context::Context, op::Op};
use stablehlo::{
    Diagnostic,
    attributes::{
        AffineBinaryOp, AffineExpr, AffineMap, ConvDimensionNumbersAttr, DotDimensionNumbersAttr,
        MeshAttr, MeshAxisAttr,
    },
    inference::{infer_convolution_type, infer_dot_general_type, infer_elementwise, infer_reshape},
    ops::{CallOp, FuncOp, ModuleOp, ReturnOp},
    pliron::{
        builtin::attributes::StringAttr,
        location::{Located, Location},
    },
    sparse::{SparseLevelFormat, SparseLevelType, SparseTensorEncoding},
    types::{ElementType, FunctionType, TensorType},
    verification::verify_module_fast,
};

fn diagnostic(error: &pliron::result::Error) -> &Diagnostic {
    error
        .err
        .downcast_ref::<Diagnostic>()
        .expect("pliron error should preserve the StableHLO diagnostic payload")
}

#[test]
fn public_errors_downcast_preserve_context_and_recover_after_repair() {
    let mut ctx = Context::new();
    let element = ElementType::F32.native(&ctx);

    let lhs = TensorType::get(&ctx, vec![2, 3], element).into();
    let rhs = TensorType::get(&ctx, vec![3, 4], element).into();
    let mut dimensions = DotDimensionNumbersAttr {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![9],
        rhs_contracting_dimensions: vec![0],
    };
    let inference_error = infer_dot_general_type(&ctx, lhs, rhs, &dimensions, element)
        .expect_err("invalid inference input must return a pliron error");
    let inference_diagnostic = diagnostic(&inference_error);
    assert_eq!(
        inference_diagnostic.recovery_category(),
        stablehlo::RecoveryCategory::InvalidInput
    );
    assert!(matches!(
        inference_diagnostic,
        Diagnostic::InvalidDimensionIndex {
            operation,
            dimension_index: 9,
            value: 9,
            rank: 2,
            ..
        } if operation == "dot_general"
    ));
    dimensions.lhs_contracting_dimensions[0] = 1;
    assert!(infer_dot_general_type(&ctx, lhs, rhs, &dimensions, element).is_ok());

    let module = ModuleOp::new(&mut ctx);
    let module_block = module.body_block(&ctx).expect("module has a body block");
    let signature = FunctionType::intern(&ctx, vec![], vec![]);
    let caller = FuncOp::new(&mut ctx, "caller".into(), signature).expect("valid function");
    let block = caller
        .entry_block(&ctx)
        .expect("function has an entry block");
    let call = CallOp::new(&mut ctx, "missing".into(), vec![], vec![]);
    let call_location = Location::Named {
        name: "recoverable-call".to_owned(),
        child_loc: Box::new(Location::Unknown),
    };
    call.get_operation()
        .deref_mut(&ctx)
        .set_loc(call_location.clone());
    call.get_operation().insert_at_back(block, &ctx);
    ReturnOp::new_many(&mut ctx, vec![])
        .get_operation()
        .insert_at_back(block, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let graph_error = verify_module_fast(&ctx, module.get_operation())
        .expect_err("unresolved call must fail graph verification");
    assert_eq!(graph_error.loc, call_location);
    assert_eq!(
        diagnostic(&graph_error).recovery_category(),
        stablehlo::RecoveryCategory::InvalidInput
    );
    assert!(matches!(
        diagnostic(&graph_error),
        Diagnostic::InvalidOperation {
            operation,
            location,
            reason: "call references an unresolved function symbol"
        } if operation == "stablehlo.call" && location.contains("recoverable-call")
    ));

    let callee = FuncOp::new(&mut ctx, "callee".into(), signature).expect("valid function");
    let callee_block = callee
        .entry_block(&ctx)
        .expect("function has an entry block");
    ReturnOp::new_many(&mut ctx, vec![])
        .get_operation()
        .insert_at_back(callee_block, &ctx);
    callee.get_operation().insert_at_back(module_block, &ctx);
    call.set_attr_stablehlo_call_callee(&ctx, StringAttr::new("callee".into()));
    assert!(verify_module_fast(&ctx, module.get_operation()).is_ok());

    let sparse = SparseTensorEncoding::new(
        &ctx,
        vec![SparseLevelType::new(SparseLevelFormat::Dense)],
        AffineMap {
            dimensions: 1,
            symbols: 0,
            results: vec![AffineExpr::dimension(0)],
        },
    )
    .expect("valid sparse encoding");
    let malformed_type = TensorType::get(&ctx, vec![2, 3], element)
        .deref(&ctx)
        .with_encoding(&ctx, Box::new(sparse.clone()));
    let source_error = malformed_type
        .deref(&ctx)
        .verify(&ctx)
        .expect_err("rank-mismatched sparse encoding must fail type verification");
    let source_diagnostic = diagnostic(&source_error);
    assert_eq!(
        source_diagnostic.recovery_category(),
        stablehlo::RecoveryCategory::InvalidInput
    );
    assert!(matches!(
        source_diagnostic,
        Diagnostic::Source {
            context: "invalid sparse tensor encoding",
            ..
        }
    ));
    let source = std::error::Error::source(source_diagnostic)
        .expect("sparse type diagnostic must preserve its source chain");
    let source_pliron_error = source
        .downcast_ref::<pliron::result::Error>()
        .expect("source should retain the nested pliron error");
    assert!(matches!(
        source_pliron_error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::TensorRankMismatch)
    ));
    let repaired_type = TensorType::get(&ctx, vec![2], element)
        .deref(&ctx)
        .with_encoding(&ctx, Box::new(sparse));
    assert!(repaired_type.deref(&ctx).verify(&ctx).is_ok());
}

#[test]
fn malformed_dimension_returns_structured_index_context() {
    let ctx = Context::new();
    let element = ElementType::F32.native(&ctx);
    let lhs = TensorType::get(&ctx, vec![2, 3], element).into();
    let rhs = TensorType::get(&ctx, vec![3, 4], element).into();
    let dimensions = DotDimensionNumbersAttr {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![7],
        rhs_contracting_dimensions: vec![0],
    };
    let error = infer_dot_general_type(&ctx, lhs, rhs, &dimensions, element)
        .expect_err("out-of-range dimension must be rejected");
    assert!(matches!(
        diagnostic(&error),
        Diagnostic::InvalidDimensionIndex {
            operation,
            dimension_index: 7,
            value: 7,
            rank: 2,
            ..
        } if operation == "dot_general"
    ));
}

#[test]
fn empty_elementwise_operands_return_a_recoverable_diagnostic() {
    let ctx = Context::new();
    let error =
        infer_elementwise(&ctx, "add", &[]).expect_err("empty operand list must be rejected");
    assert!(matches!(
        diagnostic(&error),
        Diagnostic::EmptyOperands {
            operation: "elementwise",
            role: "input"
        }
    ));
}

#[test]
fn reshape_reports_shape_product_overflow() {
    let ctx = Context::new();
    let element = ElementType::F32.native(&ctx);
    let huge = TensorType::get(&ctx, vec![i64::MAX, 2], element).into();
    let error = infer_reshape(&ctx, huge, &[i64::MAX, 2])
        .expect_err("overflowing element count must not be accepted");
    assert!(matches!(diagnostic(&error), Diagnostic::Overflow));
}

#[test]
fn affine_division_by_zero_returns_a_typed_error() {
    let expression = AffineExpr::binary(
        AffineBinaryOp::Mod,
        AffineExpr::dimension(0),
        AffineExpr::constant(0),
    );
    let error = expression
        .evaluate_at(&[9], &[])
        .expect_err("zero divisor must be rejected");
    assert!(matches!(diagnostic(&error), Diagnostic::DivisionByZero));
}

#[test]
fn oversized_mesh_product_returns_a_typed_error() {
    let ctx = Context::new();
    let mesh = MeshAttr {
        axes: vec![
            MeshAxisAttr {
                name: StringAttr::new("x".into()),
                size: i64::MAX,
            },
            MeshAxisAttr {
                name: StringAttr::new("y".into()),
                size: 2,
            },
        ],
        device_ids: vec![],
    };
    let error = mesh
        .replica_group_size(&ctx, &[])
        .expect_err("mesh size overflow must be rejected");
    assert!(matches!(diagnostic(&error), Diagnostic::MeshTooLarge));
}

#[test]
fn recursive_function_calls_are_reported_as_a_cycle() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let module_block = module.body_block(&ctx).expect("module has a body block");

    let signature = FunctionType::intern(&ctx, vec![], vec![]);
    let function = FuncOp::new(&mut ctx, "recursive".into(), signature).expect("valid function");
    let block = function
        .entry_block(&ctx)
        .expect("function has an entry block");
    CallOp::new(&mut ctx, "recursive".into(), vec![], vec![])
        .get_operation()
        .insert_at_back(block, &ctx);
    ReturnOp::new_many(&mut ctx, vec![])
        .get_operation()
        .insert_at_back(block, &ctx);
    function.get_operation().insert_at_back(module_block, &ctx);

    let error = verify_module_fast(&ctx, module.get_operation())
        .expect_err("recursive calls are unsupported");
    assert!(matches!(
        diagnostic(&error),
        Diagnostic::CycleDetected {
            graph: "function call graph"
        }
    ));
}

#[test]
fn invalid_convolution_stride_is_rejected_without_division() {
    let ctx = Context::new();
    let element = ElementType::F32.native(&ctx);
    let lhs = TensorType::get(&ctx, vec![1, 5, 3], element).into();
    let rhs = TensorType::get(&ctx, vec![3, 3, 4], element).into();
    let dimensions = ConvDimensionNumbersAttr {
        input_batch_dimension: 0,
        input_feature_dimension: 2,
        input_spatial_dimensions: vec![1],
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 2,
        kernel_spatial_dimensions: vec![0],
        output_batch_dimension: 0,
        output_feature_dimension: 2,
        output_spatial_dimensions: vec![1],
    };
    let error = infer_convolution_type(
        &ctx,
        lhs,
        rhs,
        element,
        stablehlo::inference::ConvolutionOptions {
            dimensions: &dimensions,
            strides: Some(&[0]),
            padding: None,
            lhs_dilation: None,
            rhs_dilation: None,
            feature_group_count: 1,
            batch_group_count: 1,
        },
    )
    .expect_err("zero convolution stride must be rejected");
    assert!(matches!(diagnostic(&error), Diagnostic::InvalidDimensions));
}
