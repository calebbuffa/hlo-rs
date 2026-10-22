use pliron::{
    context::{Context, Ptr},
    identifier::Identifier,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    r#type::{Type, TypeHandle},
};
use stablehlo::{
    Diagnostic,
    ops::{CallOp, FuncOp, ModuleOp, ReturnOp},
    types::{ElementType, FunctionType, TensorType, TokenType, TupleType, tensor},
    verification::verify_module_fast,
};

fn build(ctx: &mut Context, ty: TypeHandle) -> (ModuleOp, CallOp) {
    let module = ModuleOp::new(ctx);
    let body = module.body_block(ctx).unwrap();
    let signature = FunctionType::intern(ctx, vec![ty], vec![ty]);
    let caller = FuncOp::new(ctx, "main".into(), signature).unwrap();
    let block = caller.entry_block(ctx).unwrap();
    let argument = caller.argument(ctx, 0).unwrap();
    let call = CallOp::new(ctx, "helper".into(), vec![argument], vec![ty]);
    let result = call.get_operation().deref(ctx).get_result(0);
    call.get_operation().insert_at_back(block, ctx);
    ReturnOp::new(ctx, result)
        .get_operation()
        .insert_at_back(block, ctx);
    caller.get_operation().insert_at_back(body, ctx);
    let helper = FuncOp::new(ctx, "helper".into(), signature).unwrap();
    let block = helper.entry_block(ctx).unwrap();
    let argument = helper.argument(ctx, 0).unwrap();
    ReturnOp::new(ctx, argument)
        .get_operation()
        .insert_at_back(block, ctx);
    helper.get_operation().insert_at_back(body, ctx);
    (module, call)
}

fn value_types(ctx: &Context) -> Vec<TypeHandle> {
    let tensor = tensor(ctx, &[2, 3], ElementType::F32);
    let token = TokenType::get(ctx).into();
    let empty = TupleType::get(ctx, vec![]).into();
    let tuple = TupleType::get(ctx, vec![tensor, token, empty]).into();
    let nested = TupleType::get(ctx, vec![tuple, tuple, empty]).into();
    vec![tensor, token, empty, tuple, nested]
}

#[test]
fn function_calls_accept_token_empty_and_nested_tuple_values_without_repacking() {
    let mut ctx = Context::new();
    for ty in value_types(&ctx) {
        let (module, call) = build(&mut ctx, ty);
        verify_module_fast(&ctx, module.get_operation()).unwrap();
        assert_eq!(call.get_operation().deref(&ctx).get_type(0), ty);
    }
}

#[test]
fn function_value_calls_roundtrip_and_preserve_forward_symbol_references() {
    let mut ctx = Context::new();
    for ty in value_types(&ctx) {
        let (module, _) = build(&mut ctx, ty);
        let text = module.get_operation().disp(&ctx).to_string();
        let mut fresh = Context::new();
        TensorType::register(&mut fresh);
        TokenType::register(&mut fresh);
        TupleType::register(&mut fresh);
        FunctionType::register(&mut fresh);
        ModuleOp::register(&mut fresh);
        FuncOp::register(&mut fresh);
        CallOp::register(&mut fresh);
        ReturnOp::register(&mut fresh);
        let parsed =
            parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
        verify_module_fast(&fresh, parsed).unwrap();
        let original_shape = ty.disp(&ctx).to_string();
        let call = first_call(&fresh, parsed);
        assert_eq!(
            call.deref(&fresh).get_type(0).disp(&fresh).to_string(),
            original_shape,
        );
    }
}

fn first_call(ctx: &Context, module: Ptr<Operation>) -> Ptr<Operation> {
    let module = Operation::get_op::<ModuleOp>(module, ctx).unwrap();
    let block = module.body_block(ctx).unwrap();
    let function = block.deref(ctx).iter(ctx).next().unwrap();
    let function = Operation::get_op::<FuncOp>(function, ctx).unwrap();
    let block = function.entry_block(ctx).unwrap();
    block.deref(ctx).iter(ctx).next().unwrap()
}

#[test]
fn non_tensor_calls_keep_signature_symbol_and_recursion_diagnostics_with_recovery() {
    use pliron::builtin::attributes::StringAttr;
    let mut ctx = Context::new();
    let token = TokenType::get(&ctx).into();
    let (module, call) = build(&mut ctx, token);
    call.set_attr_stablehlo_call_callee(&ctx, StringAttr::new("main".into()));
    let error = verify_module_fast(&ctx, module.get_operation()).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::CycleDetected {
            graph: "function call graph"
        }),
    ));
    call.set_attr_stablehlo_call_callee(&ctx, StringAttr::new("missing".into()));
    assert!(verify_module_fast(&ctx, module.get_operation()).is_err());
    call.set_attr_stablehlo_call_callee(&ctx, StringAttr::new("helper".into()));
    verify_module_fast(&ctx, module.get_operation()).unwrap();
    let body = module.body_block(&ctx).unwrap();
    let helper = body.deref(&ctx).iter(&ctx).nth(1).unwrap();
    let function_type = FunctionType::intern(&ctx, vec![token], vec![]);
    let helper = Operation::get_op::<FuncOp>(helper, &ctx).unwrap();
    let saved = helper
        .get_attr_stablehlo_function_signature(&ctx)
        .unwrap()
        .clone();
    helper.set_attr_stablehlo_function_signature(
        &ctx,
        pliron::builtin::attributes::TypeAttr::new(function_type),
    );
    let before = module.get_operation().disp(&ctx).to_string();
    for _ in 0..2 {
        assert!(verify_module_fast(&ctx, module.get_operation()).is_err());
        assert_eq!(module.get_operation().disp(&ctx).to_string(), before);
    }
    helper.set_attr_stablehlo_function_signature(&ctx, saved);
    verify_module_fast(&ctx, module.get_operation()).unwrap();
    call.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new(stablehlo::attributes::keys::CALL_CALLEE));
    assert!(verify_module_fast(&ctx, module.get_operation()).is_err());
}

#[test]
fn calls_preserve_mixed_value_order_and_zero_result_signatures() {
    use pliron::{common_traits::Verify, r#type::Typed};
    let mut ctx = Context::new();
    let array = tensor(&ctx, &[2, 3], ElementType::F32);
    let token = TokenType::get(&ctx).into();
    let tuple = TupleType::get(&ctx, vec![array, token]).into();
    let values = vec![token, tuple, array, token];
    for results in [vec![], values.clone()] {
        let module = ModuleOp::new(&mut ctx);
        let body = module.body_block(&ctx).unwrap();
        let signature = FunctionType::intern(&ctx, values.clone(), results.clone());
        let caller = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
        let block = caller.entry_block(&ctx).unwrap();
        let arguments = caller.arguments(&ctx);
        let call = CallOp::new(&mut ctx, "helper".into(), arguments, results.clone());
        call.verify(&ctx).unwrap();
        let outputs = call.get_operation().deref(&ctx).results().collect();
        call.get_operation().insert_at_back(block, &ctx);
        ReturnOp::new_many(&mut ctx, outputs)
            .get_operation()
            .insert_at_back(block, &ctx);
        caller.get_operation().insert_at_back(body, &ctx);
        let helper = FuncOp::new(&mut ctx, "helper".into(), signature).unwrap();
        let block = helper.entry_block(&ctx).unwrap();
        let outputs = if results.is_empty() {
            vec![]
        } else {
            helper.arguments(&ctx)
        };
        ReturnOp::new_many(&mut ctx, outputs)
            .get_operation()
            .insert_at_back(block, &ctx);
        helper.get_operation().insert_at_back(body, &ctx);
        verify_module_fast(&ctx, module.get_operation()).unwrap();
        assert_eq!(
            call.get_operation()
                .deref(&ctx)
                .operands()
                .map(|value| value.get_type(&ctx))
                .collect::<Vec<_>>(),
            values,
        );
        assert_eq!(
            call.get_operation()
                .deref(&ctx)
                .result_types()
                .collect::<Vec<_>>(),
            results,
        );
    }
}
