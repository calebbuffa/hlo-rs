#![cfg(feature = "mlir-interop")]

use pliron::{builtin::types::FP32Type, common_traits::Verify, context::Context, op::Op};
use pliron_arith::ops::{MulFOp, NegFOp};
use pliron_iree_linalg_ext::ops::SortOp;
use pliron_linalg::{
    attributes::{IteratorType, identity_map},
    ops::GenericOp,
    pliron_math::ops::ExpOp,
};
use pliron_tensor::ops::EmptyOp;
use stablehlo::{
    ops::{AddOp, FuncOp, ModuleOp, ReturnOp},
    types::{FunctionType, TensorType},
};

#[test]
fn mixed_frontend_module() {
    for dynamic in [false, true] {
        let ctx = &mut Context::new();
        let ty = TensorType::from_element(
            ctx,
            FP32Type::get(ctx).into(),
            vec![if dynamic { -1 } else { 4 }],
        );
        let module = ModuleOp::new(ctx);
        let signature = FunctionType::intern(ctx, vec![ty, ty], vec![ty]);
        let name = if dynamic { "mixed_dynamic" } else { "mixed" };
        let function = FuncOp::new(ctx, name.into(), signature).unwrap();
        function
            .get_operation()
            .insert_at_back(module.body_block(ctx).unwrap(), ctx);
        let block = function.entry_block(ctx).unwrap();
        let a = function.arguments(ctx);
        let add = AddOp::new(ctx, a[0], a[1], ty);
        add.get_operation().insert_at_back(block, ctx);
        let sum = add.get_operation().deref(ctx).get_result(0);
        let sizes = if dynamic {
            let index = pliron_arith::shaped::index(ctx);
            let zero = pliron_arith::ops::ConstantOp::new(
                ctx,
                Box::new(stablehlo::attributes::IndexAttr { value: 0 }),
                index,
            );
            zero.get_operation().insert_at_back(block, ctx);
            let dim = pliron_tensor::ops::DimOp::new(ctx, a[0], zero.result(ctx));
            dim.get_operation().insert_at_back(block, ctx);
            vec![dim.result(ctx).unwrap()]
        } else {
            vec![]
        };
        let empty = EmptyOp::new(ctx, ty, sizes);
        empty.get_operation().insert_at_back(block, ctx);
        let init = empty.get_operation().deref(ctx).get_result(0);
        let generic = GenericOp::build(
            ctx,
            vec![sum],
            vec![init],
            vec![identity_map(1); 2],
            vec![IteratorType::Parallel],
            Box::new(|ctx, block, args| {
                let square = MulFOp::new(ctx, args[0], args[0]);
                square.get_operation().insert_at_back(block, ctx);
                let neg = NegFOp::new(ctx, square.result(ctx));
                neg.get_operation().insert_at_back(block, ctx);
                let exp = ExpOp::new(ctx, neg.result(ctx));
                exp.get_operation().insert_at_back(block, ctx);
                Ok(vec![exp.result(ctx)])
            }),
        )
        .unwrap();
        generic.get_operation().insert_at_back(block, ctx);
        let value = generic.get_operation().deref(ctx).get_result(0);
        let sort = SortOp::build(
            ctx,
            vec![value],
            vec![init],
            0,
            Box::new(|ctx, block, args| {
                let cmp = pliron_arith::ops::CmpFOp::new(
                    ctx,
                    pliron_arith::attributes::CmpFPredicate::Olt,
                    args[0],
                    args[1],
                );
                cmp.get_operation().insert_at_back(block, ctx);
                Ok(vec![cmp.result(ctx)])
            }),
        )
        .unwrap();
        sort.get_operation().insert_at_back(block, ctx);
        let result = sort.get_operation().deref(ctx).get_result(0);
        ReturnOp::new(ctx, result)
            .get_operation()
            .insert_at_back(block, ctx);
        module.verify(ctx).unwrap();
        let text = stablehlo::mlir::export(ctx, &module).unwrap();
        for dialect in [
            "stablehlo.",
            "arith.",
            "math.",
            "tensor.",
            "linalg.",
            "iree_linalg_ext.",
        ] {
            assert!(text.contains(dialect), "missing {dialect}: {text}");
        }
        if let Some(directory) = std::env::var_os("PLIRON_IREE_FIXTURES") {
            let directory = std::path::PathBuf::from(directory).join("compile");
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join(format!("{name}.mlir")), text).unwrap();
            let expected = [-16.0_f32, -9.0, -4.0, -1.0].map(f32::exp);
            let mut cases = vec![format!(
                r#"{{"inputs":[{{"dtype":"float32","values":[-3,1,0,2]}},{{"dtype":"float32","values":[1,0,3,-6]}}],"expected":[{{"dtype":"float32","values":{expected:?}}}]}}"#
            )];
            if dynamic {
                for (a, b) in [
                    (vec![], vec![]),
                    (vec![-0.5_f32], vec![1.5]),
                    (vec![-1., 0., 1., 2., -2., 3., -3.], vec![0.; 7]),
                ] {
                    let mut expected = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (-(a + b).powi(2)).exp())
                        .collect::<Vec<_>>();
                    expected.sort_by(f32::total_cmp);
                    cases.push(format!(
                        r#"{{"inputs":[{{"dtype":"float32","values":{a:?}}},{{"dtype":"float32","values":{b:?}}}],"expected":[{{"dtype":"float32","values":{expected:?}}}]}}"#
                    ));
                }
            }
            let cases = cases.join(",");
            std::fs::write(directory.join(format!("{name}.json")), format!(
            r#"{{"function":"{name}","cases":[{cases}],"rtol":0.00001,"atol":0.0000000001}}"#
        )).unwrap();
        }
        // Export must also reject a malformed foreign operation nested in a region.
        let square = {
            use pliron::linked_list::ContainsLinkedList;
            generic
                .body(ctx)
                .unwrap()
                .deref(ctx)
                .iter(ctx)
                .next()
                .unwrap()
        };
        let cmp = pliron_arith::ops::CmpIOp::new(
            ctx,
            pliron_arith::attributes::CmpIPredicate::Eq,
            a[0],
            a[1],
        );
        let wrong_owner = cmp.get_operation().deref(ctx).attributes.0.clone();
        square.deref_mut(ctx).attributes.0.extend(wrong_owner);
        assert!(stablehlo::mlir::export(ctx, &module).is_err());
    }
}

#[test]
fn module_export_rejects_foreign_ssa_captures() {
    let ctx = &mut Context::new();
    let ty = TensorType::from_element(ctx, FP32Type::get(ctx).into(), vec![4]);
    let external = pliron::basic_block::BasicBlock::new(ctx, None, vec![ty]);
    let captured = external.deref(ctx).arguments().next().unwrap();
    let module = ModuleOp::new(ctx);
    let signature = FunctionType::intern(ctx, vec![], vec![ty]);
    let function = FuncOp::new(ctx, "invalid".into(), signature).unwrap();
    function
        .get_operation()
        .insert_at_back(module.body_block(ctx).unwrap(), ctx);
    let block = function.entry_block(ctx).unwrap();
    let add = pliron_arith::ops::AddFOp::new(ctx, captured, captured);
    add.get_operation().insert_at_back(block, ctx);
    ReturnOp::new(ctx, add.result(ctx))
        .get_operation()
        .insert_at_back(block, ctx);
    assert!(stablehlo::mlir::export(ctx, &module).is_err());
}
