use stablehlo::{
    ops::{BroadcastInDimOp, DynamicBroadcastInDimOp, FuncOp},
    pliron::{
        builtin::attributes::StringAttr, common_traits::Verify, context::Context,
        identifier::Identifier, op::Op,
    },
    types::{ElementType, FunctionType, TensorType},
};

#[test]
fn optional_hints_use_operand_axes_and_need_not_be_sorted() {
    // Pinned TypeInference.cpp verifyDynamicBroadcastInDimOp c4/c8/c9/c10.
    for (source, result, mapping, expanding, nonexpanding, valid) in [
        (vec![2], vec![3, 2], vec![1], None, None, true),
        (vec![2], vec![3, 2], vec![1], None, Some(vec![0]), true),
        (vec![2], vec![3, 2], vec![1], Some(vec![0]), None, true),
        (vec![2], vec![3, 2], vec![1], None, Some(vec![1]), false),
        (
            vec![2],
            vec![3, 2],
            vec![1],
            Some(vec![0]),
            Some(vec![0]),
            false,
        ),
        (vec![2], vec![3, 2], vec![1], None, Some(vec![0, 0]), false),
        (
            vec![2, 3],
            vec![3, 2],
            vec![1, 0],
            None,
            Some(vec![1, 0]),
            true,
        ),
        (vec![2, 3], vec![3, 2], vec![0, 0], None, None, false),
        (vec![1], vec![0], vec![0], None, None, true),
        (vec![0], vec![1], vec![0], None, None, false),
        (vec![0], vec![0], vec![0], None, None, true),
    ] {
        let mut ctx = Context::new();
        let element = ElementType::F32.native(&ctx);
        let input = TensorType::get(&ctx, source, element).into();
        let output = TensorType::get(&ctx, result.clone(), element).into();
        let dimensions = TensorType::get(
            &ctx,
            vec![result.len() as i64],
            ElementType::I64.native(&ctx),
        )
        .into();
        let signature = FunctionType::intern(&ctx, vec![input, dimensions], vec![output]);
        let function = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
        let arguments = function.arguments(&ctx);
        let op = DynamicBroadcastInDimOp::new(
            &mut ctx,
            arguments[0],
            arguments[1],
            output,
            mapping,
            expanding.clone().unwrap_or_default(),
            nonexpanding.clone().unwrap_or_default(),
        );
        for (name, absent) in [
            (
                "stablehlo_dynamic_broadcast_known_expanding",
                expanding.is_none(),
            ),
            (
                "stablehlo_dynamic_broadcast_known_nonexpanding",
                nonexpanding.is_none(),
            ),
        ] {
            if absent {
                op.get_operation()
                    .deref_mut(&ctx)
                    .attributes
                    .0
                    .remove(&Identifier::new(name));
            }
        }
        assert_eq!(
            op.verify(&ctx).is_ok(),
            valid,
            "{result:?}, {expanding:?}, {nonexpanding:?}"
        );
    }
}

#[test]
fn static_broadcast_uses_unique_mappings_and_size_one_expansion() {
    // Pinned TypeInference.cpp verifyBroadcastInDimOp c4/c5.
    for (source, result, mapping, valid) in [
        (vec![2, 3], vec![3, 2], vec![1, 0], true),
        (vec![2, 3], vec![3, 2], vec![0, 0], false),
        (vec![1], vec![0], vec![0], true),
        (vec![0], vec![1], vec![0], false),
        (vec![0], vec![0], vec![0], true),
        (vec![2], vec![3], vec![0], false),
    ] {
        let mut ctx = Context::new();
        let element = ElementType::F32.native(&ctx);
        let input = TensorType::get(&ctx, source, element).into();
        let output = TensorType::get(&ctx, result.clone(), element).into();
        let signature = FunctionType::intern(&ctx, vec![input], vec![output]);
        let function = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
        let argument = function.arguments(&ctx)[0];
        let op = BroadcastInDimOp::new(&mut ctx, argument, output, mapping);
        assert_eq!(op.verify(&ctx).is_ok(), valid, "{result:?}");
    }
}

#[test]
fn malformed_optional_attribute_is_not_treated_as_absent() {
    let mut ctx = Context::new();
    let input = TensorType::get(&ctx, vec![2], ElementType::F32.native(&ctx)).into();
    let dimensions = TensorType::get(&ctx, vec![1], ElementType::I64.native(&ctx)).into();
    let signature = FunctionType::intern(&ctx, vec![input, dimensions], vec![input]);
    let function = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
    let arguments = function.arguments(&ctx);
    let op = DynamicBroadcastInDimOp::new(
        &mut ctx,
        arguments[0],
        arguments[1],
        input,
        vec![0],
        vec![],
        vec![],
    );
    op.get_operation().deref_mut(&ctx).attributes.0.insert(
        Identifier::new("stablehlo_dynamic_broadcast_known_expanding"),
        Box::new(StringAttr::from("not an integer list")),
    );
    assert!(op.verify(&ctx).is_err());
}
