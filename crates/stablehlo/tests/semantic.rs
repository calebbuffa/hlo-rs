use pliron::{
    basic_block::BasicBlock,
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
    r#type::TypeHandle,
};
use stablehlo::{ops::*, types::*, verification::verify_module_fast};

fn function(
    ctx: &mut Context,
    name: &str,
    inputs: Vec<TypeHandle>,
    outputs: Vec<TypeHandle>,
) -> FuncOp {
    let signature = FunctionType::intern(ctx, inputs, outputs);
    FuncOp::new(ctx, name.to_owned(), signature).expect("valid function signature")
}

#[test]
fn typed_dimension_and_comparison_attributes_verify() {
    let mut ctx = Context::new();
    let input_ty = tensor(&ctx, &[2, 3], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, input_ty, vec![0; 6], false);
    let input_value = input.get_operation().deref(&ctx).get_result(0);
    let transpose_ty = tensor(&ctx, &[3, 2], ElementType::F32);
    let transposed = TransposeOp::new(&mut ctx, input_value, transpose_ty, vec![1, 0]);
    assert!(transposed.verify(&ctx).is_ok());
    let invalid = TransposeOp::new(&mut ctx, input_value, transpose_ty, vec![0, 0]);
    assert!(invalid.verify(&ctx).is_err());

    let pred_ty = tensor(&ctx, &[2, 3], ElementType::I1);
    let compare = CompareOp::new(
        &mut ctx,
        input_value,
        input_value,
        pred_ty,
        CompareDirection::Ge,
        CompareType::Float,
    );
    assert!(compare.verify(&ctx).is_ok());
}

#[test]
fn full_and_fast_verifiers_accept_valid_function() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let function = function(&mut ctx, "valid", vec![], vec![ty]);
    let block = function.entry_block(&ctx).expect("function entry block");
    let c = ConstantOp::new(&mut ctx, ty, vec![0], true);
    let cv = c.get_operation().deref(&ctx).get_result(0);
    c.get_operation().insert_at_back(block, &ctx);
    let add = AddOp::new_inferred(&mut ctx, cv, cv).unwrap();
    let av = add.get_operation().deref(&ctx).get_result(0);
    add.get_operation().insert_at_back(block, &ctx);
    let ret = ReturnOp::new(&mut ctx, av);
    ret.get_operation().insert_at_back(block, &ctx);

    verify_module_fast(&ctx, function.get_operation()).unwrap();
    let dynamic = Operation::get_op_dyn(function.get_operation(), &ctx);
    verify_op(&*dynamic, &ctx).unwrap();
}

#[test]
fn broadcast_attributes_are_typed_integer_vectors() {
    let mut ctx = Context::new();
    let scalar_ty = tensor(&ctx, &[], ElementType::F32);
    let scalar = ConstantOp::new(&mut ctx, scalar_ty, vec![0], true);
    let value = scalar.get_operation().deref(&ctx).get_result(0);
    let broadcast = BroadcastInDimOp::new_inferred(&mut ctx, value, &[4, 2], vec![]).unwrap();
    assert!(broadcast.verify(&ctx).is_ok());
}

#[test]
fn transpose_requires_a_static_permutation() {
    let mut ctx = Context::new();
    let input_ty = tensor(&ctx, &[0, 2, 3], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, input_ty, vec![0], true);
    let value = input.get_operation().deref(&ctx).get_result(0);
    let cases = [
        (vec![0, 1, 2], vec![0, 2, 3], true),
        (vec![0, 2, 1], vec![0, 3, 2], true),
        (vec![1, 0, 2], vec![2, 0, 3], true),
        (vec![1, 2, 0], vec![2, 3, 0], true),
        (vec![2, 0, 1], vec![3, 0, 2], true),
        (vec![2, 1, 0], vec![3, 2, 0], true),
        (vec![], vec![0, 2, 3], false),
        (vec![0], vec![0, 2, 3], false),
        (vec![0, 1, 2, 3], vec![0, 2, 3], false),
        (vec![0, 0, 2], vec![0, 2, 3], false),
        (vec![0, 1, 3], vec![0, 2, 3], false),
        (vec![0, 1, 2], vec![0, 3, 2], false),
    ];
    for (permutation, shape, accepted) in cases {
        let result_ty = tensor(&ctx, &shape, ElementType::F32);
        let transpose = TransposeOp::new(&mut ctx, value, result_ty, permutation);
        assert_eq!(transpose.verify(&ctx).is_ok(), accepted);
    }
}

#[test]
fn fast_verifier_cache_keeps_operand_and_result_types_in_its_key() {
    for multiply in [false, true] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[2], ElementType::F32);
        let wrong = tensor(&ctx, &[3], ElementType::F32);
        let function = function(&mut ctx, "cache", vec![], vec![ty]);
        let block = function.entry_block(&ctx).expect("function entry block");
        let c = ConstantOp::new(&mut ctx, ty, vec![0], true);
        let value = c.get_operation().deref(&ctx).get_result(0);
        c.get_operation().insert_at_back(block, &ctx);
        let good = AddOp::new(&mut ctx, value, value, ty);
        good.get_operation().insert_at_back(block, &ctx);
        let bad = if multiply {
            stablehlo::ops::MultiplyOp::new(&mut ctx, value, value, wrong).get_operation()
        } else {
            AddOp::new(&mut ctx, value, value, wrong).get_operation()
        };
        let result = bad.deref(&ctx).get_result(0);
        bad.insert_at_back(block, &ctx);
        ReturnOp::new(&mut ctx, result)
            .get_operation()
            .insert_at_back(block, &ctx);
        assert!(verify_module_fast(&ctx, function.get_operation()).is_err());
    }
}

#[test]
fn fast_verifier_rejects_later_and_sibling_block_definitions() {
    for sibling in [false, true] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[2], ElementType::F32);
        let function = function(&mut ctx, "dominance", vec![], vec![ty]);
        let region = function.region(&ctx);
        let block = function.entry_block(&ctx).expect("function entry block");
        let c = ConstantOp::new(&mut ctx, ty, vec![0], true);
        let value = c.get_operation().deref(&ctx).get_result(0);
        let add = AddOp::new(&mut ctx, value, value, ty);
        add.get_operation().insert_at_back(block, &ctx);
        let result = add.get_operation().deref(&ctx).get_result(0);
        if sibling {
            let other = BasicBlock::new(&mut ctx, None, vec![]);
            other.insert_at_back(region, &ctx);
            c.get_operation().insert_at_back(other, &ctx);
            ReturnOp::new(&mut ctx, value)
                .get_operation()
                .insert_at_back(other, &ctx);
        } else {
            c.get_operation().insert_at_back(block, &ctx);
        }
        ReturnOp::new(&mut ctx, result)
            .get_operation()
            .insert_at_back(block, &ctx);
        assert!(verify_module_fast(&ctx, function.get_operation()).is_err());
    }
}

#[test]
fn fast_verifier_indexes_nested_capture_at_its_parent_position() {
    for late in [false, true] {
        let mut ctx = Context::new();
        let ty = tensor(&ctx, &[2], ElementType::F32);
        let function = function(&mut ctx, "nested", vec![], vec![ty]);
        let block = function.entry_block(&ctx).expect("function entry block");
        let pred_ty = tensor(&ctx, &[], ElementType::I1);
        let pred = ConstantOp::new(&mut ctx, pred_ty, vec![1], true);
        let pv = pred.get_operation().deref(&ctx).get_result(0);
        pred.get_operation().insert_at_back(block, &ctx);
        let ty = tensor(&ctx, &[2], ElementType::F32);
        let c = ConstantOp::new(&mut ctx, ty, vec![0], true);
        let value = c.get_operation().deref(&ctx).get_result(0);
        let branch = stablehlo::ops::IfOp::new(&mut ctx, pv, vec![ty]);
        for region in [branch.true_region(&ctx), branch.false_region(&ctx)] {
            let nested = BasicBlock::new(&mut ctx, None, vec![]);
            nested.insert_at_back(region, &ctx);
            ReturnOp::new(&mut ctx, value)
                .get_operation()
                .insert_at_back(nested, &ctx);
        }
        if !late {
            c.get_operation().insert_at_back(block, &ctx);
        }
        branch.get_operation().insert_at_back(block, &ctx);
        if late {
            c.get_operation().insert_at_back(block, &ctx);
        }
        let result = branch.get_operation().deref(&ctx).get_result(0);
        ReturnOp::new(&mut ctx, result)
            .get_operation()
            .insert_at_back(block, &ctx);
        assert_eq!(
            verify_module_fast(&ctx, function.get_operation()).is_ok(),
            !late
        );
    }
}

#[test]
fn stablehlo_function_pipeline_verifies_and_repairs_return_type() {
    use stablehlo::Diagnostic;

    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let tensor_type = tensor(&ctx, &[4], ElementType::F32);
    let signature = function_type(&ctx, vec![tensor_type, tensor_type], vec![tensor_type]);
    let function = FuncOp::new(&mut ctx, "pipeline".into(), signature).expect("valid function");
    let block = function.entry_block(&ctx).expect("function entry block");
    let arguments = function.arguments(&ctx);
    let [x, y] = arguments.as_slice() else {
        panic!("function signature creates two arguments");
    };

    let add = AddOp::new_inferred(&mut ctx, *x, *y).expect("same-shaped tensors add");
    let sum = add.get_operation().deref(&ctx).get_result(0);
    add.verify(&ctx).expect("valid add");
    add.get_operation().insert_at_back(block, &ctx);

    let constant = ConstantOp::new(&mut ctx, tensor_type, vec![2.0f32.to_bits() as u64], true);
    let two = constant.get_operation().deref(&ctx).get_result(0);
    constant.verify(&ctx).expect("valid splat constant");
    constant.get_operation().insert_at_back(block, &ctx);

    let multiply =
        MultiplyOp::new_inferred(&mut ctx, sum, two).expect("same-shaped tensors multiply");
    let product = multiply.get_operation().deref(&ctx).get_result(0);
    multiply.verify(&ctx).expect("valid multiply");
    multiply.get_operation().insert_at_back(block, &ctx);

    let exponential =
        ExponentialOp::new_inferred(&mut ctx, product).expect("valid exponential input");
    let result = exponential.get_operation().deref(&ctx).get_result(0);
    exponential.verify(&ctx).expect("valid exponential");
    exponential.get_operation().insert_at_back(block, &ctx);

    let wrong_type = tensor(&ctx, &[3], ElementType::F32);
    let wrong_constant = ConstantOp::new(&mut ctx, wrong_type, vec![1.0f32.to_bits() as u64], true);
    let wrong_result = wrong_constant.get_operation().deref(&ctx).get_result(0);
    wrong_constant.get_operation().insert_at_back(block, &ctx);
    let ret = ReturnOp::new(&mut ctx, wrong_result);
    ret.verify(&ctx).expect("return has no operation results");
    ret.get_operation().insert_at_back(block, &ctx);
    function
        .get_operation()
        .insert_at_back(module.body_block(&ctx).expect("module body"), &ctx);

    function
        .verify(&ctx)
        .expect_err("wrong return type must fail function verification");
    let error = verify_module_fast(&ctx, module.get_operation()).expect_err("wrong return type");
    let diagnostic = error
        .err
        .downcast_ref::<Diagnostic>()
        .expect("typed diagnostic payload");
    assert!(
        matches!(diagnostic, Diagnostic::InvalidOperation { operation, reason: "function return types do not match signature", .. } if operation == "stablehlo.return"),
        "unexpected diagnostic: {diagnostic:?}"
    );
    wrong_result.replace_all_uses_with(&ctx, &result);
    function.verify(&ctx).expect("repaired function verifies");
    verify_module_fast(&ctx, module.get_operation()).expect("repaired return type verifies");
}

#[test]
fn inferred_builder_preserves_typed_error_and_operation_name() {
    use stablehlo::Diagnostic;

    let mut ctx = Context::new();
    let float_type = tensor(&ctx, &[4], ElementType::F32);
    let integer_type = tensor(&ctx, &[4], ElementType::I32);
    let float = ConstantOp::new(&mut ctx, float_type, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let integer = ConstantOp::new(&mut ctx, integer_type, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let error = match AddOp::new_inferred(&mut ctx, float, integer) {
        Err(error) => error,
        Ok(_) => panic!("mixed element types must be rejected"),
    };
    let diagnostic = error
        .err
        .downcast_ref::<Diagnostic>()
        .expect("typed diagnostic payload");
    assert!(matches!(
        diagnostic,
        Diagnostic::BuilderFailure {
            operation: "stablehlo.add",
            category: stablehlo::RecoveryCategory::InvalidInput,
            ..
        }
    ));
    assert!(std::error::Error::source(diagnostic).is_some());
}

#[test]
fn function_and_module_constructors_enforce_signature_and_symbol_contracts() {
    use stablehlo::Diagnostic;

    let mut ctx = Context::new();
    let invalid_signature = tensor(&ctx, &[4], ElementType::F32);
    let error = match FuncOp::new(&mut ctx, "invalid".into(), invalid_signature) {
        Err(error) => error,
        Ok(_) => panic!("non-function signature must be rejected"),
    };
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidOperation { operation, reason: "signature must be a StableHLO FunctionType", .. }) if operation == "stablehlo.func"
    ));

    let module = ModuleOp::new(&mut ctx);
    let signature = function_type(&ctx, vec![], vec![]);
    let first = FuncOp::new(&mut ctx, "duplicate".into(), signature).expect("valid function");
    let second = FuncOp::new(&mut ctx, "duplicate".into(), signature).expect("valid function");
    let body = module.body_block(&ctx).expect("module body block");
    first.get_operation().insert_at_back(body, &ctx);
    second.get_operation().insert_at_back(body, &ctx);
    let error = module
        .verify(&ctx)
        .expect_err("duplicate symbols must fail module verification");
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidOperation { operation, reason: "function symbol is empty or duplicated", .. }) if operation == "stablehlo.func"
    ));
}
