use pliron::{
    basic_block::BasicBlock,
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
};
use stablehlo::{ops::*, types::*, verification::verify_module_fast};

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
    let function = FuncOp::new(&mut ctx);
    let region = function.region(&ctx);
    let block = BasicBlock::new(&mut ctx, None, vec![]);
    block.insert_at_back(region, &ctx);
    let ty = tensor(&ctx, &[2], ElementType::F32);
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
        let function = FuncOp::new(&mut ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(function.region(&ctx), &ctx);
        let ty = tensor(&ctx, &[2], ElementType::F32);
        let wrong = tensor(&ctx, &[3], ElementType::F32);
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
        let function = FuncOp::new(&mut ctx);
        let region = function.region(&ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);
        let ty = tensor(&ctx, &[2], ElementType::F32);
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
        let function = FuncOp::new(&mut ctx);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(function.region(&ctx), &ctx);
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
