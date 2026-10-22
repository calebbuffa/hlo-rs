use pliron::{
    attribute::AttrObj,
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, VecAttr},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    region::Region,
    r#type::{Type as PlironType, TypeHandle, Typed},
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
    assert_eq!(get.index(&ctx), Some(0));
    let bad = GetTupleElementOp::new(&mut ctx, tv, 1, ty);
    assert_eq!(bad.index(&ctx), Some(1));
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
fn replica_and_partition_ids_return_scalar_ui32() {
    let mut ctx = Context::new();
    for op in [
        ReplicaIdOp::new(&mut ctx).get_operation(),
        PartitionIdOp::new(&mut ctx).get_operation(),
    ] {
        let result = op.deref(&ctx).get_result(0);
        assert_eq!(
            stablehlo::types::tensor_parts(&ctx, result.get_type(&ctx)).unwrap(),
            (vec![], ElementType::Ui32)
        );
        assert!(op.verify(&ctx).is_ok());
        result.set_type(&ctx, tensor(&ctx, &[], ElementType::I32));
        assert!(op.verify(&ctx).is_err());
    }
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
fn module_and_function_native_assembly_roundtrip_preserves_signature_and_body() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let body = module.body_block(&ctx).expect("module body block");
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let result_ty = tensor(&ctx, &[2], ElementType::F64);
    let signature = FunctionType::intern(&ctx, vec![ty], vec![result_ty]);
    let function = FuncOp::new(&mut ctx, "entry".into(), signature).expect("valid function");
    let function_block = function.entry_block(&ctx).expect("function entry block");
    let argument = function.argument(&ctx, 0).expect("function argument");
    let constant = ConstantOp::new(&mut ctx, ty, vec![0x8000_0000], true);
    let constant_value = constant.get_operation().deref(&ctx).get_result(0);
    constant
        .get_operation()
        .insert_at_back(function_block, &ctx);
    let add = AddOp::new_inferred(&mut ctx, argument, constant_value).unwrap();
    let result = add.get_operation().deref(&ctx).get_result(0);
    add.get_operation().insert_at_back(function_block, &ctx);
    let convert = ConvertOp::new(&mut ctx, result, result_ty);
    let result = convert.get_operation().deref(&ctx).get_result(0);
    convert.get_operation().insert_at_back(function_block, &ctx);
    ReturnOp::new(&mut ctx, result)
        .get_operation()
        .insert_at_back(function_block, &ctx);
    function.get_operation().insert_at_back(body, &ctx);
    assert!(verify_module_fast(&ctx, module.get_operation()).is_ok());

    let assembly = module.get_operation().disp(&ctx).to_string();
    assert!(assembly.contains("stablehlo.module"), "{assembly}");
    assert!(assembly.contains("stablehlo.func"), "{assembly}");
    assert!(assembly.contains("entry"), "{assembly}");
    assert!(assembly.contains("stablehlo.return"), "{assembly}");

    let mut parsed_ctx = Context::new();
    TensorType::register(&mut parsed_ctx);
    FunctionType::register(&mut parsed_ctx);
    ModuleOp::register(&mut parsed_ctx);
    FuncOp::register(&mut parsed_ctx);
    ReturnOp::register(&mut parsed_ctx);
    ConstantOp::register(&mut parsed_ctx);
    AddOp::register(&mut parsed_ctx);
    ConvertOp::register(&mut parsed_ctx);
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &assembly,
    )
    .unwrap_or_else(|error| panic!("StableHLO module failed to parse:\n{assembly}\n{error}"));
    let parsed_module = Operation::get_op_dyn(parsed, &parsed_ctx);
    verify_module_fast(&parsed_ctx, parsed)
        .unwrap_or_else(|error| panic!("reparsed StableHLO module failed verification:\n{error}"));
    let parsed_module = parsed_module
        .as_ref()
        .downcast_ref::<ModuleOp>()
        .expect("parsed StableHLO module");
    let parsed_body = parsed_module
        .body_block(&parsed_ctx)
        .expect("parsed body block");
    let parsed_function = Operation::get_op_dyn(
        parsed_body
            .deref(&parsed_ctx)
            .iter(&parsed_ctx)
            .next()
            .unwrap(),
        &parsed_ctx,
    );
    let parsed_function = parsed_function
        .as_ref()
        .downcast_ref::<FuncOp>()
        .expect("parsed StableHLO function");
    assert_eq!(parsed_function.name(&parsed_ctx).as_deref(), Some("entry"));
    let parsed_signature = parsed_function.signature(&parsed_ctx).unwrap();
    let parsed_signature = parsed_signature.deref(&parsed_ctx);
    let parsed_signature = parsed_signature.downcast_ref::<FunctionType>().unwrap();
    assert_eq!(parsed_signature.inputs().len(), 1);
    assert_eq!(parsed_signature.outputs().len(), 1);
    assert_eq!(
        tensor_parts(&parsed_ctx, parsed_signature.inputs()[0]),
        Some((vec![2], ElementType::F32))
    );
    assert_eq!(
        tensor_parts(&parsed_ctx, parsed_signature.outputs()[0]),
        Some((vec![2], ElementType::F64))
    );
    let parsed_entry = parsed_function.entry_block(&parsed_ctx).unwrap();
    let parsed_ops: Vec<_> = parsed_entry.deref(&parsed_ctx).iter(&parsed_ctx).collect();
    assert_eq!(parsed_ops.len(), 4);
    let parsed_constant = Operation::get_op::<ConstantOp>(parsed_ops[0], &parsed_ctx).unwrap();
    assert_eq!(parsed_constant.bits(&parsed_ctx), Some(vec![0x8000_0000]));
    assert_eq!(parsed_constant.is_splat(&parsed_ctx), Some(true));
    assert!(Operation::is_op::<AddOp>(parsed_ops[1], &parsed_ctx));
    assert!(Operation::is_op::<ConvertOp>(parsed_ops[2], &parsed_ctx));
    let reparsed_assembly = parsed.disp(&parsed_ctx).to_string();
    let mut second_ctx = Context::new();
    TensorType::register(&mut second_ctx);
    FunctionType::register(&mut second_ctx);
    ModuleOp::register(&mut second_ctx);
    FuncOp::register(&mut second_ctx);
    ReturnOp::register(&mut second_ctx);
    ConstantOp::register(&mut second_ctx);
    AddOp::register(&mut second_ctx);
    ConvertOp::register(&mut second_ctx);
    let second = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut second_ctx,
        &reparsed_assembly,
    )
    .unwrap_or_else(|error| {
        panic!("reprinted StableHLO module failed to parse:\n{reparsed_assembly}\n{error}")
    });
    verify_module_fast(&second_ctx, second)
        .unwrap_or_else(|error| panic!("reprinted StableHLO module failed verification:\n{error}"));
}

#[test]
fn native_assembly_preserves_calls_nested_regions_and_backend_attributes() {
    use pliron::builtin::attributes::{BoolAttr, DictAttr};

    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let module_body = module.body_block(&ctx).unwrap();
    let ty = tensor(&ctx, &[], ElementType::F32);
    let signature = FunctionType::intern(&ctx, vec![ty], vec![ty]);
    let helper = FuncOp::new(&mut ctx, "helper".into(), signature).unwrap();
    let argument = helper.argument(&ctx, 0).unwrap();
    ReturnOp::new(&mut ctx, argument)
        .get_operation()
        .insert_at_back(helper.entry_block(&ctx).unwrap(), &ctx);
    helper.get_operation().insert_at_back(module_body, &ctx);

    let main = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
    let main_body = main.entry_block(&ctx).unwrap();
    let argument = main.argument(&ctx, 0).unwrap();
    let call = CallOp::new(&mut ctx, "helper".into(), vec![argument], vec![ty]);
    let called_value = call.get_operation().deref(&ctx).get_result(0);
    call.get_operation().insert_at_back(main_body, &ctx);
    let while_op = WhileOp::new(&mut ctx, vec![called_value], vec![ty]);
    let condition_region = while_op.condition_region(&ctx);
    let condition = add_block(&mut ctx, condition_region, vec![ty]);
    let pred_ty = tensor(&ctx, &[], ElementType::I1);
    let pred = ConstantOp::new(&mut ctx, pred_ty, vec![0], true);
    let pred_value = pred.get_operation().deref(&ctx).get_result(0);
    pred.get_operation().insert_at_back(condition, &ctx);
    ReturnOp::new(&mut ctx, pred_value)
        .get_operation()
        .insert_at_back(condition, &ctx);
    let body_region = while_op.body_region(&ctx);
    let body = add_block(&mut ctx, body_region, vec![ty]);
    let carried = body.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, carried)
        .get_operation()
        .insert_at_back(body, &ctx);
    let loop_value = while_op.get_operation().deref(&ctx).get_result(0);
    while_op.get_operation().insert_at_back(main_body, &ctx);
    let custom = CustomCallOp::new(
        &mut ctx,
        "native_target".into(),
        vec![loop_value],
        vec![ty],
        4,
    );
    custom.set_backend_config(
        &mut ctx,
        Box::new(DictAttr::new(vec![(
            "enabled".try_into().unwrap(),
            Box::new(BoolAttr::new(true)) as AttrObj,
        )])),
    );
    let output = custom.get_operation().deref(&ctx).get_result(0);
    custom.get_operation().insert_at_back(main_body, &ctx);
    ReturnOp::new(&mut ctx, output)
        .get_operation()
        .insert_at_back(main_body, &ctx);
    main.get_operation().insert_at_back(module_body, &ctx);
    verify_module_fast(&ctx, module.get_operation()).unwrap();

    let assembly = module.get_operation().disp(&ctx).to_string();
    let mut parsed_ctx = Context::new();
    TensorType::register(&mut parsed_ctx);
    FunctionType::register(&mut parsed_ctx);
    for register in [
        ModuleOp::register,
        FuncOp::register,
        ReturnOp::register,
        ConstantOp::register,
        CallOp::register,
        WhileOp::register,
        CustomCallOp::register,
    ] {
        register(&mut parsed_ctx);
    }
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &assembly,
    )
    .unwrap_or_else(|error| panic!("nested module failed to parse:\n{assembly}\n{error}"));
    verify_module_fast(&parsed_ctx, parsed).unwrap();
    let parsed_module = Operation::get_op::<ModuleOp>(parsed, &parsed_ctx).unwrap();
    let parsed_body = parsed_module.body_block(&parsed_ctx).unwrap();
    let functions: Vec<_> = parsed_body.deref(&parsed_ctx).iter(&parsed_ctx).collect();
    assert_eq!(functions.len(), 2);
    let parsed_main = Operation::get_op::<FuncOp>(functions[1], &parsed_ctx).unwrap();
    let parsed_entry = parsed_main.entry_block(&parsed_ctx).unwrap();
    let operations: Vec<_> = parsed_entry.deref(&parsed_ctx).iter(&parsed_ctx).collect();
    assert_eq!(operations.len(), 4);
    let parsed_call = Operation::get_op::<CallOp>(operations[0], &parsed_ctx).unwrap();
    assert_eq!(
        parsed_call
            .get_attr_stablehlo_call_callee(&parsed_ctx)
            .unwrap()
            .as_str(),
        "helper"
    );
    let parsed_while = Operation::get_op::<WhileOp>(operations[1], &parsed_ctx).unwrap();
    assert_eq!(
        parsed_while
            .condition_region(&parsed_ctx)
            .deref(&parsed_ctx)
            .iter(&parsed_ctx)
            .count(),
        1
    );
    let parsed_custom = Operation::get_op::<CustomCallOp>(operations[2], &parsed_ctx).unwrap();
    let config = parsed_custom.backend_config(&parsed_ctx).unwrap();
    let config = config.downcast_ref::<DictAttr>().unwrap();
    assert_eq!(
        config,
        &DictAttr::new(vec![(
            "enabled".try_into().unwrap(),
            Box::new(BoolAttr::new(true)) as AttrObj,
        )])
    );
}

#[test]
fn native_assembly_preserves_movement_gather_reduction_and_compute_metadata() {
    fn append(
        ctx: &Context,
        block: Ptr<BasicBlock>,
        operation: Ptr<Operation>,
    ) -> pliron::value::Value {
        operation.insert_at_back(block, ctx);
        operation.deref(ctx).get_result(0)
    }

    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx);
    let input_ty = tensor(&ctx, &[2, 3], ElementType::F32);
    let scalar_ty = tensor(&ctx, &[], ElementType::F32);
    let signature = FunctionType::intern(&ctx, vec![input_ty], vec![scalar_ty]);
    let function = FuncOp::new(&mut ctx, "main".into(), signature).unwrap();
    let entry = function.entry_block(&ctx).unwrap();
    let input = function.argument(&ctx, 0).unwrap();
    let slice = SliceOp::new_inferred(&mut ctx, input, vec![0, 1], vec![2, 3], vec![1, 1]).unwrap();
    let sliced = append(&ctx, entry, slice.get_operation());
    let reshape = ReshapeOp::new_inferred(&mut ctx, sliced, &[4]).unwrap();
    let reshaped = append(&ctx, entry, reshape.get_operation());
    let broadcast = BroadcastInDimOp::new_inferred(&mut ctx, reshaped, &[2, 4], vec![1]).unwrap();
    let broadcasted = append(&ctx, entry, broadcast.get_operation());
    let initial = ConstantOp::new(&mut ctx, scalar_ty, vec![0], true);
    let initial = append(&ctx, entry, initial.get_operation());
    let vector_ty = tensor(&ctx, &[2], ElementType::F32);
    let reduce = ReduceOp::new(
        &mut ctx,
        vec![broadcasted],
        vec![initial],
        vec![vector_ty],
        vec![1],
    );
    let reducer_region = reduce.reducer_region(&ctx);
    let reducer = add_block(&mut ctx, reducer_region, vec![scalar_ty, scalar_ty]);
    let lhs = reducer.deref(&ctx).get_argument(0);
    let rhs = reducer.deref(&ctx).get_argument(1);
    let sum = AddOp::new_inferred(&mut ctx, lhs, rhs).unwrap();
    let sum = append(&ctx, reducer, sum.get_operation());
    ReturnOp::new(&mut ctx, sum)
        .get_operation()
        .insert_at_back(reducer, &ctx);
    let reduced = append(&ctx, entry, reduce.get_operation());
    let indices_ty = tensor(&ctx, &[1], ElementType::I32);
    let indices = ConstantOp::new(&mut ctx, indices_ty, vec![1], true);
    let indices = append(&ctx, entry, indices.get_operation());
    let dimensions = GatherDimensionNumbersAttr {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        operand_batching_dims: vec![],
        start_indices_batching_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
    };
    let gather =
        GatherOp::new_inferred(&mut ctx, reduced, indices, dimensions.clone(), vec![1]).unwrap();
    let gathered = append(&ctx, entry, gather.get_operation());
    let dot = DotOp::new(
        &mut ctx,
        gathered,
        gathered,
        scalar_ty,
        vec![Precision::Default, Precision::High],
    );
    let dot_result = append(&ctx, entry, dot.get_operation());
    ReturnOp::new(&mut ctx, dot_result)
        .get_operation()
        .insert_at_back(entry, &ctx);
    function
        .get_operation()
        .insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
    verify_module_fast(&ctx, module.get_operation()).unwrap();

    let assembly = module.get_operation().disp(&ctx).to_string();
    let mut parsed_ctx = Context::new();
    TensorType::register(&mut parsed_ctx);
    FunctionType::register(&mut parsed_ctx);
    for register in [
        ModuleOp::register,
        FuncOp::register,
        ReturnOp::register,
        ConstantOp::register,
        SliceOp::register,
        ReshapeOp::register,
        BroadcastInDimOp::register,
        ReduceOp::register,
        AddOp::register,
        GatherOp::register,
        DotOp::register,
    ] {
        register(&mut parsed_ctx);
    }
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed_ctx,
        &assembly,
    )
    .unwrap_or_else(|error| panic!("structured module failed to parse:\n{assembly}\n{error}"));
    verify_module_fast(&parsed_ctx, parsed).unwrap();
    let module = Operation::get_op::<ModuleOp>(parsed, &parsed_ctx).unwrap();
    let entry = module.body_block(&parsed_ctx).unwrap();
    let function = entry.deref(&parsed_ctx).iter(&parsed_ctx).next().unwrap();
    let function = Operation::get_op::<FuncOp>(function, &parsed_ctx).unwrap();
    let entry = function.entry_block(&parsed_ctx).unwrap();
    let operations: Vec<_> = entry.deref(&parsed_ctx).iter(&parsed_ctx).collect();
    let gather = operations
        .iter()
        .find_map(|&ptr| Operation::get_op::<GatherOp>(ptr, &parsed_ctx))
        .unwrap();
    assert_eq!(gather.dimensions(&parsed_ctx).as_deref(), Some(&dimensions));
    assert_eq!(gather.slice_sizes(&parsed_ctx), Some(vec![1]));
    let broadcast = operations
        .iter()
        .find_map(|&ptr| Operation::get_op::<BroadcastInDimOp>(ptr, &parsed_ctx))
        .unwrap();
    assert_eq!(broadcast.dimensions(&parsed_ctx), Some(vec![1]));
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
        channel_type: ChannelType::DeviceToDevice,
    };
    assert!(
        stablehlo::ops::SendOp::new(&mut ctx, input, tv, channel, false)
            .verify(&ctx)
            .is_ok()
    );
    let bad = ChannelHandleAttr {
        handle: 7,
        channel_type: ChannelType::DeviceToHost,
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
