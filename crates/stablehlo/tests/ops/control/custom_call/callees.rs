use pliron::{
    attribute::AttrObj,
    builtin::attributes::{StringAttr, VecAttr},
    common_traits::Verify,
    context::Context,
    identifier::Identifier,
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    r#type::Type,
};
use stablehlo::{
    Diagnostic,
    attributes::SymbolRefAttr,
    ops::{CustomCallOp, FuncOp, ModuleOp, ReturnOp},
    types::{ElementType, FunctionType, TensorType, tensor},
    verification::verify_module_fast,
};

fn module_with_callees(ctx: &mut Context) -> (ModuleOp, CustomCallOp) {
    let module = ModuleOp::new(ctx);
    let body = module.body_block(ctx).unwrap();
    let ty = tensor(ctx, &[], ElementType::F32);
    let signature = FunctionType::intern(ctx, vec![], vec![ty]);
    let function = FuncOp::new(ctx, "main".into(), signature).unwrap();
    let block = function.entry_block(ctx).unwrap();
    let call = CustomCallOp::new(ctx, "external".into(), vec![], vec![ty], 1);
    call.set_called_computations(ctx, vec!["second".into(), "first".into(), "second".into()]);
    let result = call.get_operation().deref(ctx).get_result(0);
    call.get_operation().insert_at_back(block, ctx);
    ReturnOp::new(ctx, result)
        .get_operation()
        .insert_at_back(block, ctx);
    function.get_operation().insert_at_back(body, ctx);

    let other = tensor(ctx, &[3], ElementType::F64);
    let signature = FunctionType::intern(ctx, vec![other], vec![other]);
    for name in ["first", "second"] {
        let function = FuncOp::new(ctx, name.into(), signature).unwrap();
        let block = function.entry_block(ctx).unwrap();
        let argument = function.argument(ctx, 0).unwrap();
        ReturnOp::new(ctx, argument)
            .get_operation()
            .insert_at_back(block, ctx);
        function.get_operation().insert_at_back(body, ctx);
    }
    (module, call)
}

#[test]
fn called_computations_default_empty_and_preserve_ordered_flat_symbols() {
    let mut ctx = Context::new();
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![], 1);
    assert!(call.called_computations(&ctx).unwrap().is_empty());
    call.verify(&ctx).unwrap();
    let names = vec!["z".into(), "quoted \"symbol\"".into(), "z".into()];
    call.set_called_computations(&mut ctx, names.clone());
    assert_eq!(call.called_computations(&ctx).unwrap(), names);
    let references = call
        .get_attr_stablehlo_custom_call_called_computations(&ctx)
        .unwrap();
    for (reference, name) in references.0.iter().zip(&names) {
        let reference = reference.downcast_ref::<SymbolRefAttr>().unwrap();
        assert_eq!(reference.path.len(), 1);
        assert_eq!(reference.path[0].as_str(), name);
    }
    drop(references);
    call.verify(&ctx).unwrap();
    call.set_called_computations(&mut ctx, vec![]);
    assert!(call.called_computations(&ctx).unwrap().is_empty());
    call.verify(&ctx).unwrap();
}

#[test]
fn malformed_called_computations_are_diagnosed_without_mutation_and_can_be_repaired() {
    let mut ctx = Context::new();
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![], 1);
    let malformed: Vec<AttrObj> = vec![
        Box::new(StringAttr::new("callee".into())),
        Box::new(VecAttr::new(vec![Box::new(StringAttr::new(
            "callee".into(),
        ))])),
        Box::new(VecAttr::new(vec![Box::new(SymbolRefAttr { path: vec![] })])),
        Box::new(VecAttr::new(vec![Box::new(SymbolRefAttr {
            path: vec![StringAttr::new(String::new())],
        })])),
        Box::new(VecAttr::new(vec![Box::new(SymbolRefAttr {
            path: vec![
                StringAttr::new("outer".into()),
                StringAttr::new("inner".into()),
            ],
        })])),
    ];
    for attribute in malformed {
        call.get_operation().deref_mut(&ctx).attributes.0.insert(
            Identifier::new("stablehlo_custom_call_called_computations"),
            attribute,
        );
        let before = call.get_operation().disp(&ctx).to_string();
        for _ in 0..2 {
            let error = call.called_computations(&ctx).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<Diagnostic>(),
                Some(Diagnostic::InvalidAttribute {
                    attribute: "called_computations",
                    ..
                }),
            ));
            assert!(call.verify(&ctx).is_err());
            assert_eq!(call.get_operation().disp(&ctx).to_string(), before);
        }
        call.set_called_computations(&mut ctx, vec!["repaired".into()]);
        call.verify(&ctx).unwrap();
        assert_eq!(call.called_computations(&ctx).unwrap(), ["repaired"]);
    }
}

#[test]
fn module_resolves_custom_callees_without_ordinary_call_signature_or_recursion_rules() {
    let mut ctx = Context::new();
    let (module, call) = module_with_callees(&mut ctx);
    verify_module_fast(&ctx, module.get_operation()).unwrap();
    call.set_called_computations(&mut ctx, vec!["main".into()]);
    verify_module_fast(&ctx, module.get_operation()).unwrap();
    call.set_called_computations(&mut ctx, vec!["missing".into()]);
    call.verify(&ctx).unwrap();
    let before = module.get_operation().disp(&ctx).to_string();
    for _ in 0..2 {
        let error = verify_module_fast(&ctx, module.get_operation()).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidOperation {
                reason: "custom_call references an unresolved function symbol",
                ..
            }),
        ));
        assert_eq!(module.get_operation().disp(&ctx).to_string(), before);
    }
    call.set_called_computations(&mut ctx, vec!["second".into(), "first".into()]);
    verify_module_fast(&ctx, module.get_operation()).unwrap();
}

#[test]
fn custom_callees_roundtrip_with_forward_references_in_a_fresh_context() {
    let mut ctx = Context::new();
    let (module, _) = module_with_callees(&mut ctx);
    verify_module_fast(&ctx, module.get_operation()).unwrap();
    let text = module.get_operation().disp(&ctx).to_string();
    assert!(text.contains("stablehlo.symbol_ref"));
    let mut fresh = Context::new();
    TensorType::register(&mut fresh);
    FunctionType::register(&mut fresh);
    ModuleOp::register(&mut fresh);
    FuncOp::register(&mut fresh);
    ReturnOp::register(&mut fresh);
    CustomCallOp::register(&mut fresh);
    let parsed = parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
    verify_module_fast(&fresh, parsed).unwrap();
    let module = Operation::get_op::<ModuleOp>(parsed, &fresh).unwrap();
    let body = module.body_block(&fresh).unwrap();
    let function = body.deref(&fresh).iter(&fresh).next().unwrap();
    let function = Operation::get_op::<FuncOp>(function, &fresh).unwrap();
    let block = function.entry_block(&fresh).unwrap();
    let call = block.deref(&fresh).iter(&fresh).next().unwrap();
    let call = Operation::get_op::<CustomCallOp>(call, &fresh).unwrap();
    assert_eq!(
        call.called_computations(&fresh).unwrap(),
        ["second", "first", "second"],
    );
}
