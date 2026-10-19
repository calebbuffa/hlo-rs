use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, VecAttr},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    region::Region,
    r#type::{TypeHandle, Typed},
    utils::apint::APInt,
};
use stablehlo::{attributes::*, ops::*, types::*, verification::verify_module_fast};

fn replica_groups(ctx: &Context, groups: &[&[i64]]) -> VecAttr {
    let ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let values = groups
        .iter()
        .map(|group| {
            let members = group
                .iter()
                .map(|&value| {
                    Box::new(IntegerAttr::new(
                        ty,
                        APInt::from_str(&value.to_string(), 64, 10).unwrap(),
                    )) as AttrObj
                })
                .collect();
            Box::new(VecAttr::new(members)) as AttrObj
        })
        .collect();
    VecAttr::new(values)
}

fn add_block(
    ctx: &mut Context,
    region: Ptr<Region>,
    args: Vec<TypeHandle>,
) -> pliron::context::Ptr<BasicBlock> {
    let b = BasicBlock::new(ctx, None, args);
    b.insert_at_back(region, ctx);
    b
}

#[test]
fn tuple_build_and_projection_use_native_types() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let c = ConstantOp::new(&mut ctx, ty, vec![0], true);
    let value = c.get_operation().deref(&ctx).get_result(0);
    let tuple = TupleOp::new(&mut ctx, vec![value]);
    assert!(tuple.verify(&ctx).is_ok());
    let tv = tuple.get_operation().deref(&ctx).get_result(0);
    let get = GetTupleElementOp::new(&mut ctx, tv, 0, ty);
    assert!(get.verify(&ctx).is_ok());
    let bad = GetTupleElementOp::new(&mut ctx, tv, 1, ty);
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn token_operations_require_native_token_types() {
    let mut ctx = Context::new();
    let token = CreateTokenOp::new(&mut ctx);
    assert!(token.verify(&ctx).is_ok());
    let v = token.get_operation().deref(&ctx).get_result(0);
    assert!(AfterAllOp::new(&mut ctx, vec![v]).verify(&ctx).is_ok());
    let ty = tensor(&ctx, &[], ElementType::I32);
    let c = ConstantOp::new(&mut ctx, ty, vec![0], true);
    let cv = c.get_operation().deref(&ctx).get_result(0);
    let bad = AfterAllOp::new(&mut ctx, vec![cv]);
    assert!(bad.verify(&ctx).is_err());
}

#[test]
fn if_and_case_enforce_region_return_contracts() {
    let mut ctx = Context::new();
    let bool_ty = tensor(&ctx, &[], ElementType::I1);
    let pred = ConstantOp::new(&mut ctx, bool_ty, vec![1], true);
    let pv = pred.get_operation().deref(&ctx).get_result(0);
    let ty = tensor(&ctx, &[3], ElementType::F32);
    let value = ConstantOp::new(&mut ctx, ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let ifop = IfOp::new(&mut ctx, pv, vec![ty]);
    for region in [ifop.true_region(&ctx), ifop.false_region(&ctx)] {
        let block = add_block(&mut ctx, region, vec![]);
        ReturnOp::new(&mut ctx, value)
            .get_operation()
            .insert_at_back(block, &ctx);
    }
    assert!(ifop.verify(&ctx).is_ok());
    let case = CaseOp::new(&mut ctx, pv, vec![ty], 1);
    let case_region = case.get_operation().deref(&ctx).get_region(0);
    let block = add_block(&mut ctx, case_region, vec![]);
    ReturnOp::new(&mut ctx, value)
        .get_operation()
        .insert_at_back(block, &ctx);
    assert!(case.verify(&ctx).is_err());
}

#[test]
fn while_regions_match_carried_arguments_and_results() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let init = ConstantOp::new(&mut ctx, ty, vec![0], true);
    let iv = init.get_operation().deref(&ctx).get_result(0);
    let loopop = WhileOp::new(&mut ctx, vec![iv], vec![ty]);
    let cond_region = loopop.condition_region(&ctx);
    let cond = add_block(&mut ctx, cond_region, vec![ty]);
    let pty = tensor(&ctx, &[], ElementType::I1);
    let p = ConstantOp::new(&mut ctx, pty, vec![1], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    ReturnOp::new(&mut ctx, p)
        .get_operation()
        .insert_at_back(cond, &ctx);
    let loop_body = loopop.body_region(&ctx);
    let body = add_block(&mut ctx, loop_body, vec![ty]);
    let arg = body.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, arg)
        .get_operation()
        .insert_at_back(body, &ctx);
    assert!(loopop.verify(&ctx).is_ok());
}

#[test]
fn custom_call_checks_api_version_and_alias_paths() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let tuple = TupleOp::new(&mut ctx, vec![input]);
    let tuple_value = tuple.get_operation().deref(&ctx).get_result(0);
    let tuple_ty = tuple_value.get_type(&ctx);
    let good = CustomCallOp::new(
        &mut ctx,
        "target".into(),
        vec![tuple_value],
        vec![tuple_ty],
        4,
    );
    good.set_attr_stablehlo_custom_call_output_operand_aliases(
        &ctx,
        VecAttr::new(vec![Box::new(OutputOperandAliasAttr {
            output_tuple_indices: vec![0],
            operand_index: 0,
            operand_tuple_indices: vec![0],
        }) as AttrObj]),
    );
    assert!(good.verify(&ctx).is_ok());
    let bad = CustomCallOp::new(
        &mut ctx,
        "target".into(),
        vec![tuple_value],
        vec![tuple_ty],
        4,
    );
    bad.set_attr_stablehlo_custom_call_output_operand_aliases(
        &ctx,
        VecAttr::new(vec![Box::new(OutputOperandAliasAttr {
            output_tuple_indices: vec![1],
            operand_index: 0,
            operand_tuple_indices: vec![0],
        }) as AttrObj]),
    );
    assert!(bad.verify(&ctx).is_err());
    assert!(
        CustomCallOp::new(&mut ctx, "target".into(), vec![input], vec![ty], 5)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn module_fast_resolves_function_symbols_and_call_signatures() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let body = module.body_block(&ctx).expect("module has a body block");
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let sig = FunctionType::intern(&ctx, vec![ty], vec![ty]);
    let callee = FuncOp::new(&mut ctx, "callee".into(), sig).expect("valid callee");
    let callee_block = callee.entry_block(&ctx).expect("callee entry block");
    let arg = callee.argument(&ctx, 0).expect("callee argument");
    ReturnOp::new(&mut ctx, arg)
        .get_operation()
        .insert_at_back(callee_block, &ctx);
    callee.get_operation().insert_at_back(body, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".into(), sig).expect("valid caller");
    let caller_block = caller.entry_block(&ctx).expect("caller entry block");
    let constant = ConstantOp::new(&mut ctx, ty, vec![0], true);
    let value = constant.get_operation().deref(&ctx).get_result(0);
    constant.get_operation().insert_at_back(caller_block, &ctx);
    let call = CallOp::new(&mut ctx, "callee".into(), vec![value], vec![ty]);
    let result = call.get_operation().deref(&ctx).get_result(0);
    call.get_operation().insert_at_back(caller_block, &ctx);
    ReturnOp::new(&mut ctx, result)
        .get_operation()
        .insert_at_back(caller_block, &ctx);
    caller.get_operation().insert_at_back(body, &ctx);
    assert!(verify_module_fast(&ctx, module.get_operation()).is_ok());
    let unresolved = FuncOp::new(&mut ctx, "unresolved".into(), sig).expect("valid function");
    let b = unresolved.entry_block(&ctx).expect("function entry block");
    let call = CallOp::new(&mut ctx, "missing".into(), vec![], vec![]);
    call.get_operation().insert_at_back(b, &ctx);
    unresolved.get_operation().insert_at_back(body, &ctx);
    assert!(verify_module_fast(&ctx, module.get_operation()).is_err());
}

#[test]
fn async_start_and_done_preserve_future_signature() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let start = AsyncStartOp::new(&mut ctx, vec![input], vec![ty]);
    let async_body = start.body_region(&ctx);
    let block = add_block(&mut ctx, async_body, vec![ty]);
    let arg = block.deref(&ctx).get_argument(0);
    let sliced = SliceOp::new(&mut ctx, arg, ty, vec![0], vec![2], vec![1]);
    let result = sliced.get_operation().deref(&ctx).get_result(0);
    sliced.get_operation().insert_at_back(block, &ctx);
    ReturnOp::new(&mut ctx, result)
        .get_operation()
        .insert_at_back(block, &ctx);
    assert!(start.verify(&ctx).is_ok());
    let future = start.get_operation().deref(&ctx).get_result(0);
    assert!(
        AsyncDoneOp::new(&mut ctx, future, vec![ty])
            .verify(&ctx)
            .is_ok()
    );
    let bad_ty = tensor(&ctx, &[2], ElementType::I32);
    assert!(
        AsyncDoneOp::new(&mut ctx, future, vec![bad_ty])
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn communication_channel_kinds_follow_host_transfer_mode() {
    let mut ctx = Context::new();
    let token = CreateTokenOp::new(&mut ctx);
    let tv = token.get_operation().deref(&ctx).get_result(0);
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let channel = ChannelHandleAttr {
        handle: 7,
        channel_type: 1,
    };
    assert!(
        stablehlo::ops::SendOp::new(&mut ctx, input, tv, channel, false)
            .verify(&ctx)
            .is_ok()
    );
    let bad = ChannelHandleAttr {
        handle: 7,
        channel_type: 2,
    };
    assert!(
        stablehlo::ops::SendOp::new(&mut ctx, input, tv, bad, false)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn collective_inference_preserves_shape_contracts() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2, 3], ElementType::F32);
    let gathered = stablehlo::inference::infer_all_gather_types(&ctx, &[ty], 1, 2).unwrap();
    assert_eq!(
        stablehlo::types::tensor_parts(&ctx, gathered[0]).unwrap().0,
        vec![2, 6]
    );
    let exchanged = stablehlo::inference::infer_all_to_all_types(&ctx, &[ty], 0, 1, 2).unwrap();
    assert_eq!(
        stablehlo::types::tensor_parts(&ctx, exchanged[0])
            .unwrap()
            .0,
        vec![1, 6]
    );
    let groups = replica_groups(&ctx, &[&[0, 1]]);
    let input = ConstantOp::new(&mut ctx, ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let gather = AllGatherOp::new(&mut ctx, vec![input], gathered, 1, groups.clone());
    assert!(gather.verify(&ctx).is_ok());
    let group_size_ty = tensor(&ctx, &[1], ElementType::I32);
    let root = ConstantOp::new(&mut ctx, group_size_ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let broadcast = CollectiveBroadcastOp::new(&mut ctx, vec![input, root], vec![ty], true, groups);
    assert!(broadcast.verify(&ctx).is_ok());
}

#[test]
fn collective_reducer_regions_have_typed_scalar_signatures() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let scalar = tensor(&ctx, &[], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let groups = VecAttr::new(vec![]);
    let op = AllReduceOp::new(&mut ctx, vec![input], vec![ty], groups);
    let region = op.reducer_region(&ctx);
    let block = add_block(&mut ctx, region, vec![scalar, scalar]);
    let arg = block.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, arg)
        .get_operation()
        .insert_at_back(block, &ctx);
    assert!(op.verify(&ctx).is_ok());
}

#[test]
fn all_to_all_and_reduce_scatter_match_replica_geometry() {
    let mut ctx = Context::new();
    let groups = replica_groups(&ctx, &[&[0, 1]]);
    let input_ty = tensor(&ctx, &[2, 4], ElementType::F32);
    let input = ConstantOp::new(&mut ctx, input_ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let all_to_all_ty = tensor(&ctx, &[1, 8], ElementType::F32);
    let all_to_all = AllToAllOp::new(
        &mut ctx,
        vec![input],
        vec![all_to_all_ty],
        0,
        1,
        2,
        groups.clone(),
    );
    assert!(all_to_all.verify(&ctx).is_ok());
    let scalar = tensor(&ctx, &[], ElementType::F32);
    let scatter_ty = tensor(&ctx, &[2, 2], ElementType::F32);
    let scatter = ReduceScatterOp::new(&mut ctx, vec![input], vec![scatter_ty], 1, groups);
    let region = scatter.reducer_region(&ctx);
    let block = add_block(&mut ctx, region, vec![scalar, scalar]);
    let arg = block.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, arg)
        .get_operation()
        .insert_at_back(block, &ctx);
    assert!(scatter.verify(&ctx).is_ok());
}

#[test]
fn custom_call_backend_config_matches_api_version() {
    use pliron::builtin::attributes::{BoolAttr, DictAttr, StringAttr};
    let mut ctx = Context::new();
    for version in 0..=4 {
        let dictionary = version == 4;
        for (config, accepted) in [
            (
                Box::new(StringAttr::new("config".into())) as AttrObj,
                !dictionary,
            ),
            (
                Box::new(stablehlo::attributes::BytesAttr {
                    value: vec![0, 255],
                }) as AttrObj,
                !dictionary,
            ),
            (Box::new(DictAttr::new(vec![])) as AttrObj, dictionary),
            (Box::new(BoolAttr::new(true)) as AttrObj, false),
        ] {
            let call = CustomCallOp::new(&mut ctx, "target".into(), vec![], vec![], version);
            assert!(
                call.verify(&ctx).is_ok(),
                "absent config is accepted for API {version}"
            );
            call.set_backend_config(&mut ctx, config);
            assert_eq!(call.verify(&ctx).is_ok(), accepted, "API {version}");
            assert!(call.backend_config(&ctx).is_some());
        }
    }
}
