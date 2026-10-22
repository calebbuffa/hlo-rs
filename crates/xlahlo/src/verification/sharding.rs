use super::{VerifyOptions, error, optional_attr, require};
use crate::{
    attributes::Callees,
    opcode::Opcode,
    ops::{HloOpInterface, control::ParameterOp},
    sharding::{NamedSharding, Sharding},
};
use pliron::{
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    result::Result,
    value::DefiningEntity,
};
use std::collections::HashMap;

pub(super) fn verify(
    ctx: &Context,
    instructions: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
    options: VerifyOptions,
) -> Result<()> {
    let partitions = options.collective_context.partition_count;
    require(partitions > 0, "partition count must be positive")?;
    let policy = options.sharding_context;
    let devices = (policy.use_spmd_partitioning && policy.verify_device_numbers && partitions > 1)
        .then_some(partitions);
    let mut shardings = HashMap::new();
    let mut control_flow = Vec::new();
    for &(op, _) in instructions {
        let data = op.deref(ctx);
        if data.get_num_results() != 1 {
            continue;
        }
        if let Some(sharding) =
            optional_attr::<Sharding>(ctx, op, crate::attributes::keys::SHARDING)?
        {
            let shape = data.get_type(0);
            let sharding = sharding.normalize_tuple(ctx, shape)?;
            sharding.validate(ctx, shape, devices)?;
            shardings.insert(op, sharding);
        }
        if let Some(named) =
            optional_attr::<NamedSharding>(ctx, op, crate::attributes::keys::NAMED_SHARDING)?
        {
            named.validate_shape(ctx, data.get_type(0), devices)?;
        }
        let dynamic = Operation::get_op_dyn(op, ctx);
        if let Some(hlo) = op_cast::<dyn HloOpInterface>(dynamic.as_ref())
            && matches!(hlo.opcode(), Opcode::While | Opcode::Conditional)
        {
            control_flow.push((op, hlo.opcode()));
        }
    }
    if control_flow.is_empty() {
        return Ok(());
    }
    let mut roots = HashMap::new();
    let mut parameters = HashMap::new();
    for &computation in computations.values() {
        let block = super::single_block(ctx, computation.deref(ctx).get_region(0))?;
        let term = block
            .deref(ctx)
            .get_terminator(ctx)
            .ok_or_else(|| error("missing computation return"))?;
        if let DefiningEntity::Op(root) = term.deref(ctx).get_operand(0).defining_entity() {
            roots.insert(computation, root);
        }
        for op in block.deref(ctx).iter(ctx) {
            if let Some(parameter) = Operation::get_op::<ParameterOp>(op, ctx)
                && parameter.index(ctx).is_some_and(|index| index.0 == 0)
            {
                parameters.insert(computation, op);
            }
        }
    }
    for (op, opcode) in control_flow {
        let callees = optional_attr::<Callees>(ctx, op, crate::attributes::keys::CALLEES)?
            .ok_or_else(|| error("missing control-flow callees"))?;
        let mut related = vec![op];
        for (index, callee) in callees.0.iter().enumerate() {
            let computation = computations
                .get(callee)
                .ok_or_else(|| error("control-flow callee not found"))?;
            if opcode == Opcode::Conditional || index == 0 {
                related.push(
                    *roots
                        .get(computation)
                        .ok_or_else(|| error("control-flow callee root not found"))?,
                );
            }
            if opcode == Opcode::While {
                related.push(
                    *parameters
                        .get(computation)
                        .ok_or_else(|| error("missing while parameter"))?,
                );
            }
        }
        let mut previous: Option<&Sharding> = None;
        for related in related {
            if let Some(sharding) = shardings.get(&related) {
                if let Some(previous) = previous {
                    require(
                        previous.equal_semantics(sharding)?,
                        "inconsistent control-flow sharding",
                    )?;
                }
                previous = Some(sharding);
            }
        }
    }
    Ok(())
}
