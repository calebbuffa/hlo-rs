#![allow(dead_code)]
use xlahlo::{
    attributes::{Callees, Index, ReplicaGroup, ReplicaGroups},
    ops::{ComputationOp, ReturnOp, control::ParameterOp},
    pliron::{
        attribute::Attribute,
        basic_block::BasicBlock,
        builtin::{ops::ModuleOp, types::FunctionType},
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::TypeHandle,
        value::Value,
    },
    types::{ElementType, TensorType},
};

pub(super) fn context() -> Context {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    ctx
}
pub(super) fn shape(ctx: &Context, element: ElementType, dims: &[i64]) -> TypeHandle {
    TensorType::get(ctx, element, dims.to_vec()).unwrap()
}
pub(super) fn param(ctx: &mut Context, ty: TypeHandle, index: u32) -> ParameterOp {
    ParameterOp::new(ctx, ty, vec![], Index(index))
}
pub(super) fn attr<T: Attribute>(ctx: &Context, op: Ptr<Operation>, key: &'static str, value: T) {
    op.deref_mut(ctx)
        .attributes
        .set(Identifier::new(key), value);
}
pub(super) fn computation(
    ctx: &mut Context,
    name: &'static str,
    args: Vec<TypeHandle>,
    result: TypeHandle,
) -> (ComputationOp, Ptr<BasicBlock>, Vec<Value>) {
    let parameters: Vec<_> = args
        .iter()
        .enumerate()
        .map(|(i, &ty)| param(ctx, ty, u32::try_from(i).unwrap()))
        .collect();
    let signature = FunctionType::get(ctx, args, vec![result]).into();
    let comp = ComputationOp::new(ctx, Identifier::new(name), signature);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(comp.body(ctx), ctx);
    let parameters = parameters
        .into_iter()
        .map(|p| {
            p.get_operation().insert_at_back(block, ctx);
            p.result(ctx)
        })
        .collect();
    (comp, block, parameters)
}
pub(super) fn ret(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) {
    ReturnOp::new(ctx, value)
        .get_operation()
        .insert_at_back(block, ctx);
}
pub(super) fn module(
    ctx: &mut Context,
    computations: &[ComputationOp],
    entry: &'static str,
) -> ModuleOp {
    let module = ModuleOp::new(ctx, Identifier::new("control_test"));
    let block = module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .unwrap();
    for comp in computations {
        comp.get_operation().insert_at_back(block, ctx);
    }
    xlahlo::ir::set_entry(ctx, module.get_operation(), Identifier::new(entry)).unwrap();
    module
}
pub(super) fn callees(names: &[&'static str]) -> Callees {
    Callees(names.iter().map(|&s| Identifier::new(s)).collect())
}
pub(super) fn groups(n: i64) -> ReplicaGroups {
    ReplicaGroups(vec![ReplicaGroup((0..n).collect())])
}
