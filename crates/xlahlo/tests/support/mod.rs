use xlahlo::{
    attributes::Index,
    ops::{ComputationOp, ReturnOp, control::ParameterOp},
    pliron::{
        basic_block::BasicBlock,
        builtin::{ops::ModuleOp, types::FunctionType},
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        r#type::TypeHandle,
        value::Value,
    },
};

pub(super) fn context() -> Context {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    ctx
}

pub(super) fn module(ctx: &mut Context) -> (ModuleOp, Ptr<BasicBlock>) {
    let module = ModuleOp::new(ctx, Identifier::new("test"));
    xlahlo::ir::set_entry(ctx, module.get_operation(), Identifier::new("main")).unwrap();
    let block = module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .unwrap();
    (module, block)
}

pub(super) fn computation(
    ctx: &mut Context,
    container: Ptr<BasicBlock>,
    name: &'static str,
    inputs: Vec<TypeHandle>,
    output: TypeHandle,
) -> (ComputationOp, Ptr<BasicBlock>, Vec<Value>) {
    let signature = FunctionType::get(ctx, inputs.clone(), vec![output]).into();
    let op = ComputationOp::new(ctx, Identifier::new(name), signature);
    op.get_operation().insert_at_back(container, ctx);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(op.body(ctx), ctx);
    let mut parameters = Vec::new();
    for (index, ty) in inputs.into_iter().enumerate() {
        let parameter = ParameterOp::new(ctx, ty, vec![], Index(index.try_into().unwrap()));
        parameter.get_operation().insert_at_back(block, ctx);
        parameters.push(parameter.result(ctx));
    }
    (op, block, parameters)
}

pub(super) fn ret(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) {
    ReturnOp::new(ctx, value)
        .get_operation()
        .insert_at_back(block, ctx);
}
