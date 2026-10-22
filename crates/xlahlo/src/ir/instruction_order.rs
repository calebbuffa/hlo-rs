//! Logical HLO ordering, independent of physical SSA dominance order.
//!
//! Graph membership always comes from native operands and control predecessors.
//! Optional permutations retain ordering that cannot be derived from that graph
//! (notably XLA's swap-removal user order). Stale permutations are errors.
//! Graph editors must update exact permutations or explicitly invalidate them
//! at mutation commit. Arbitrary generic rewriters are not intercepted here.
//! Module verification validates these hints only after physical SSA checks;
//! logical order never grants forward operand references.
//! Based on OpenXLA's instruction and computation ordering in
//! `hlo_computation.cc` and `hlo_instruction.cc`.
use super::{ControlPredecessors, InstructionName, fusion_parameters};
use crate::{
    ops::{ComputationOp, ReturnOp},
    verification::{error, optional_attr, require},
};
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    context::{Context, Ptr},
    derive::pliron_attr,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    operation::Operation,
    result::Result,
    value::{DefiningEntity, Value},
};
use std::collections::{HashMap, HashSet};

#[pliron_attr(
    name = "xlahlo.order_permutation",
    format = "`<` $subject `,` `[` vec($members, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// An exact ordering override for the named subject's graph-derived members.
pub struct OrderPermutation {
    /// Named instruction or fusion parameter whose order is overridden.
    pub subject: String,
    /// Exact permutation of the graph-derived members.
    pub members: Vec<String>,
}

#[pliron_attr(
    name = "xlahlo.instruction_order",
    format = "`<` `[` vec($instructions, CharSpace(`,`)) `]` `,` `[` vec($blocks, CharSpace(`,`)) `]` `,` `[` vec($users, CharSpace(`,`)) `]` `,` `[` vec($control_successors, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Optional logical instruction, nested-block, user, and control ordering hints.
///
/// These hints never override graph membership or physical SSA dominance.
/// Nonempty permutations must exactly match the members derived from the current
/// IR; stale names or incomplete/duplicate member lists are invalid.
pub struct InstructionOrder {
    /// Empty means derive from physical order. Nonempty is an exact permutation
    /// of the containing computation's direct instructions, excluding return.
    pub instructions: Vec<String>,
    /// Exact nested fusion parameter/instruction order, keyed by its owner.
    pub blocks: Vec<OrderPermutation>,
    /// Deduplicated data users, keyed by instruction or fusion parameter name.
    pub users: Vec<OrderPermutation>,
    /// Outgoing control order; incoming control metadata remains authoritative.
    pub control_successors: Vec<OrderPermutation>,
}

const KEY: &str = "hlo_instruction_order";

fn computation(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Ptr<Operation>> {
    let mut owner = block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| error("ordering block is detached"))?;
    loop {
        if Operation::is_op::<ComputationOp>(owner, ctx) {
            return Ok(owner);
        }
        owner = owner
            .deref(ctx)
            .get_parent_block()
            .and_then(|block| block.deref(ctx).get_parent_op(ctx))
            .ok_or_else(|| error("ordering value has no computation owner"))?;
    }
}

fn value_block(ctx: &Context, value: Value) -> Result<Ptr<BasicBlock>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => {
            require(
                operation.deref(ctx).get_num_results() == 1
                    && operation.deref(ctx).get_result(0) == value,
                "ordering requires a single-result instruction",
            )?;
            operation
                .deref(ctx)
                .get_parent_block()
                .ok_or_else(|| error("ordering instruction is detached"))
        }
        DefiningEntity::Block(block) => {
            require(
                block
                    .deref(ctx)
                    .arguments()
                    .any(|argument| argument == value),
                "invalid ordering block argument",
            )?;
            Ok(block)
        }
    }
}

fn name(ctx: &Context, value: Value) -> Result<Option<String>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => Ok(optional_attr::<InstructionName>(
            ctx, operation, "hlo_name",
        )?
        .map(|name| name.0.clone())),
        DefiningEntity::Block(block) => {
            let index = block
                .deref(ctx)
                .arguments()
                .position(|argument| argument == value)
                .ok_or_else(|| error("invalid ordering block argument"))?;
            Ok(fusion_parameters(ctx, block)?.0[index].name.clone())
        }
    }
}

fn instructions(ctx: &Context, block: Ptr<BasicBlock>) -> Vec<Ptr<Operation>> {
    block
        .deref(ctx)
        .iter(ctx)
        .filter(|&operation| !Operation::is_op::<ReturnOp>(operation, ctx))
        .collect()
}

fn named_values(ctx: &Context, block: Ptr<BasicBlock>) -> Result<HashMap<String, Value>> {
    let mut names = HashMap::new();
    let mut values: Vec<_> = block.deref(ctx).arguments().collect();
    for operation in instructions(ctx, block) {
        require(
            operation.deref(ctx).get_num_results() == 1,
            "ordering block contains a noninstruction",
        )?;
        values.push(operation.deref(ctx).get_result(0));
    }
    for value in values {
        if let Some(name) = name(ctx, value)? {
            require(
                !name.is_empty() && names.insert(name, value).is_none(),
                "empty or ambiguous ordering name",
            )?;
        }
    }
    Ok(names)
}

fn incoming_names(ctx: &Context, value: Value) -> Result<Vec<String>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => {
            Ok(
                optional_attr::<ControlPredecessors>(ctx, operation, "hlo_control_predecessors")?
                    .map(|predecessors| predecessors.0.clone())
                    .unwrap_or_default(),
            )
        }
        DefiningEntity::Block(block) => {
            let index = block
                .deref(ctx)
                .arguments()
                .position(|argument| argument == value)
                .ok_or_else(|| error("invalid control parameter"))?;
            Ok(fusion_parameters(ctx, block)?.0[index]
                .control_predecessors
                .clone())
        }
    }
}

fn permutation<T: Copy + Eq + std::hash::Hash>(actual: &[T], requested: &[T]) -> Result<Vec<T>> {
    let actual: HashSet<_> = actual.iter().copied().collect();
    let requested_set: HashSet<_> = requested.iter().copied().collect();
    require(
        requested.len() == actual.len() && requested_set == actual,
        "stale or duplicate logical ordering permutation",
    )?;
    Ok(requested.to_vec())
}

fn state(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Option<InstructionOrder>> {
    Ok(
        optional_attr::<InstructionOrder>(ctx, computation(ctx, block)?, KEY)?
            .map(|order| order.clone()),
    )
}

/// Direct logical instruction storage order; never used as SSA dominance order.
pub fn instruction_order(ctx: &Context, owner: Ptr<Operation>) -> Result<Vec<Ptr<Operation>>> {
    require(
        Operation::is_op::<ComputationOp>(owner, ctx),
        "instruction order needs a computation",
    )?;
    let regions: Vec<_> = owner.deref(ctx).regions().collect();
    require(regions.len() == 1, "ordering computation needs one region")?;
    let blocks: Vec<_> = regions[0].deref(ctx).iter(ctx).collect();
    require(blocks.len() == 1, "ordering computation needs one block")?;
    let block = blocks[0];
    let actual = instructions(ctx, block);
    let Some(order) = state(ctx, block)? else {
        return Ok(actual);
    };
    if order.instructions.is_empty() {
        return Ok(actual);
    }
    let names = named_values(ctx, block)?;
    let requested = order
        .instructions
        .iter()
        .map(
            |name| match names.get(name).map(|value| value.defining_entity()) {
                Some(DefiningEntity::Op(operation)) => Ok(operation),
                _ => Err(error(
                    "logical instruction name is missing or a block argument",
                )),
            },
        )
        .collect::<Result<Vec<_>>>()?;
    permutation(&actual, &requested)
}

/// Operation-only projection of logical storage, excluding fusion arguments.
pub fn block_instruction_order(
    ctx: &Context,
    block: Ptr<BasicBlock>,
) -> Result<Vec<Ptr<Operation>>> {
    Ok(block_value_order(ctx, block)?
        .into_iter()
        .filter_map(|value| match value.defining_entity() {
            DefiningEntity::Op(operation) => Some(operation),
            DefiningEntity::Block(_) => None,
        })
        .collect())
}

/// Logical parameter and instruction identities, excluding return. Fusion
/// arguments retain their physical indices regardless of this permutation.
pub fn block_value_order(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Vec<Value>> {
    let owner = block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| error("ordering block is detached"))?;
    if Operation::is_op::<ComputationOp>(owner, ctx) {
        Ok(instruction_order(ctx, owner)?
            .into_iter()
            .map(|operation| operation.deref(ctx).get_result(0))
            .collect())
    } else {
        require(
            Operation::is_op::<crate::ops::control::FusionOp>(owner, ctx),
            "nested logical storage requires a fusion body",
        )?;
        let mut actual: Vec<_> = block.deref(ctx).arguments().collect();
        for operation in instructions(ctx, block) {
            require(
                operation.deref(ctx).get_num_results() == 1,
                "ordering block contains a noninstruction",
            )?;
            actual.push(operation.deref(ctx).get_result(0));
        }
        let Some(order) = state(ctx, block)? else {
            return Ok(actual);
        };
        let Some(owner_name) =
            optional_attr::<InstructionName>(ctx, owner, "hlo_name")?.map(|name| name.0.clone())
        else {
            return Ok(actual);
        };
        let mut records = order
            .blocks
            .iter()
            .filter(|record| record.subject == owner_name);
        let Some(record) = records.next() else {
            return Ok(actual);
        };
        require(
            records.next().is_none(),
            "duplicate nested block ordering subject",
        )?;
        let names = named_values(ctx, block)?;
        let requested = record
            .members
            .iter()
            .map(|name| {
                names
                    .get(name)
                    .copied()
                    .ok_or_else(|| error("nested storage order refers to a missing value"))
            })
            .collect::<Result<Vec<_>>>()?;
        permutation(&actual, &requested)
    }
}

fn apply_value_override(
    ctx: &Context,
    subject: Value,
    actual: &[Value],
    records: &[OrderPermutation],
) -> Result<Vec<Value>> {
    let subject_name = name(ctx, subject)?;
    let mut matches = records
        .iter()
        .filter(|record| subject_name.as_ref() == Some(&record.subject));
    let Some(record) = matches.next() else {
        return Ok(actual.to_vec());
    };
    require(
        matches.next().is_none(),
        "duplicate subject ordering override",
    )?;
    let names = named_values(ctx, value_block(ctx, subject)?)?;
    let requested = record
        .members
        .iter()
        .map(|name| {
            names
                .get(name)
                .copied()
                .ok_or_else(|| error("unresolved ordering member"))
        })
        .collect::<Result<Vec<_>>>()?;
    permutation(actual, &requested)
}

/// Deduplicated HLO users, not SSA operand uses and never the return terminator.
pub fn users(ctx: &Context, value: Value) -> Result<Vec<Value>> {
    let block = value_block(ctx, value)?;
    let actual: Vec<_> = block_instruction_order(ctx, block)?
        .into_iter()
        .filter(|operation| {
            operation
                .deref(ctx)
                .operands()
                .any(|operand| operand == value)
        })
        .map(|operation| operation.deref(ctx).get_result(0))
        .collect();
    let Some(order) = state(ctx, block)? else {
        return Ok(actual);
    };
    apply_value_override(ctx, value, &actual, &order.users)
}

/// Incoming control edges are authoritative native metadata, in stored order.
pub fn control_predecessors(ctx: &Context, value: Value) -> Result<Vec<Value>> {
    let names = named_values(ctx, value_block(ctx, value)?)?;
    let incoming = incoming_names(ctx, value)?;
    let mut seen = HashSet::new();
    incoming
        .into_iter()
        .map(|name| {
            let predecessor = *names
                .get(&name)
                .ok_or_else(|| error("unresolved control predecessor"))?;
            require(
                predecessor != value && seen.insert(predecessor),
                "self or duplicate control predecessor",
            )?;
            Ok(predecessor)
        })
        .collect()
}

/// Ordered projection of the same incoming-edge graph, never a second edge set.
pub fn control_successors(ctx: &Context, value: Value) -> Result<Vec<Value>> {
    let block = value_block(ctx, value)?;
    let candidates = block_value_order(ctx, block)?;
    let mut actual = Vec::new();
    for candidate in candidates {
        if control_predecessors(ctx, candidate)?.contains(&value) {
            actual.push(candidate);
        }
    }
    let Some(order) = state(ctx, block)? else {
        return Ok(actual);
    };
    apply_value_override(ctx, value, &actual, &order.control_successors)
}

/// Reject dangling/duplicate records even when their subjects are not queried.
pub(crate) fn verify_order(ctx: &Context, owner: Ptr<Operation>) -> Result<()> {
    instruction_order(ctx, owner)?;
    let Some(order) =
        optional_attr::<InstructionOrder>(ctx, owner, KEY)?.map(|order| order.clone())
    else {
        return Ok(());
    };
    let mut subjects = HashMap::new();
    let mut block_owners = HashMap::new();
    let mut pending = vec![owner];
    while let Some(operation) = pending.pop() {
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                    require(
                        block_owners.insert(name.0.clone(), block).is_none(),
                        "ambiguous nested block owner",
                    )?;
                }
                for (name, value) in named_values(ctx, block)? {
                    require(
                        subjects.insert(name, value).is_none(),
                        "ambiguous ordering subject",
                    )?;
                }
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    let mut seen = HashSet::new();
    for record in &order.blocks {
        require(
            seen.insert(&record.subject),
            "duplicate nested block ordering subject",
        )?;
        let block = *block_owners
            .get(&record.subject)
            .ok_or_else(|| error("stale nested block ordering subject"))?;
        block_instruction_order(ctx, block)?;
    }
    for records in [&order.users, &order.control_successors] {
        let mut seen = HashSet::new();
        for record in records {
            require(seen.insert(&record.subject), "duplicate ordering subject")?;
            let value = *subjects
                .get(&record.subject)
                .ok_or_else(|| error("stale ordering subject"))?;
            users(ctx, value)?;
            control_successors(ctx, value)?;
        }
    }
    Ok(())
}

/// Set only a complete validated permutation record; restore the prior record on
/// error. This changes ordering annotations, not operand or control membership.
pub fn set_order(ctx: &Context, owner: Ptr<Operation>, order: InstructionOrder) -> Result<()> {
    require(
        Operation::is_op::<ComputationOp>(owner, ctx),
        "ordering needs a computation",
    )?;
    let old = owner.deref(ctx).attributes.clone();
    owner
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new(KEY), order);
    if let Err(error) = verify_order(ctx, owner) {
        owner.deref_mut(ctx).attributes = old;
        return Err(error);
    }
    Ok(())
}

/// Call before graph mutation when exact order cannot be updated. Clearing the
/// ordering hint never clears graph edges. Structural mutations with stale hints
/// otherwise fail access/verification rather than silently reverting order.
/// Detached fusion construction has no containing computation state to clear.
pub fn invalidate_order(ctx: &Context, block: Ptr<BasicBlock>) -> Result<()> {
    let mut block = block;
    loop {
        let owner = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("ordering block is detached"))?;
        if Operation::is_op::<ComputationOp>(owner, ctx) {
            owner
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&Identifier::new(KEY));
            return Ok(());
        }
        require(
            Operation::is_op::<crate::ops::control::FusionOp>(owner, ctx),
            "ordering invalidation requires a computation or fusion body",
        )?;
        let Some(parent) = owner.deref(ctx).get_parent_block() else {
            return Ok(());
        };
        block = parent;
    }
}

pub(crate) fn register(ctx: &mut Context) {
    OrderPermutation::register::<OrderPermutation>(ctx);
    InstructionOrder::register::<InstructionOrder>(ctx);
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        attributes::Index,
        ops::{ReturnOp, control::ParameterOp, elementwise::AddOp},
        types::{ElementType, TensorType},
    };
    use pliron::{builtin::types::FunctionType, op::Op, printable::Printable};

    #[test]
    fn module_verification_checks_serialized_nested_and_relation_orders_after_dominance() {
        use crate::{
            attributes::FusionKind,
            ir::{FusionParameter, FusionParameters},
            ops::control::FusionOp,
        };
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let ty = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
        let signature = FunctionType::get(&ctx, vec![ty], vec![ty]).into();
        let owner = ComputationOp::new(&mut ctx, Identifier::new("main"), signature);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(owner.body(&ctx), &ctx);
        let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
        parameter.get_operation().insert_at_back(block, &ctx);
        let input = parameter.result(&ctx);
        let first = AddOp::new(&mut ctx, ty, vec![input; 2]);
        first.get_operation().insert_at_back(block, &ctx);
        let first_value = first.result(&ctx);
        let fusion = FusionOp::new(&mut ctx, ty, vec![first_value], FusionKind::Loop);
        fusion.get_operation().insert_at_back(block, &ctx);
        let body = BasicBlock::new(&mut ctx, None, vec![ty]);
        body.insert_at_back(fusion.body(&ctx), &ctx);
        let argument = body.deref(&ctx).get_argument(0);
        let inner = AddOp::new(&mut ctx, ty, vec![argument; 2]);
        inner.get_operation().insert_at_back(body, &ctx);
        let inner_value = inner.result(&ctx);
        ReturnOp::new(&mut ctx, inner_value)
            .get_operation()
            .insert_at_back(body, &ctx);
        let result = fusion.result(&ctx);
        ReturnOp::new(&mut ctx, result)
            .get_operation()
            .insert_at_back(block, &ctx);
        for (operation, name) in [
            (parameter.get_operation(), "p"),
            (first.get_operation(), "a"),
            (fusion.get_operation(), "f"),
            (inner.get_operation(), "n"),
        ] {
            operation
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_name"), InstructionName(name.into()));
        }
        crate::ir::set_fusion_parameters(
            &ctx,
            body,
            FusionParameters(vec![FusionParameter {
                name: Some("q".into()),
                ..Default::default()
            }]),
        )
        .unwrap();
        fusion.get_operation().unlink(&ctx);
        invalidate_order(&ctx, body).unwrap();
        crate::ir::set_fusion_parameters(
            &ctx,
            body,
            FusionParameters(vec![FusionParameter {
                name: Some("renamed_q".into()),
                ..Default::default()
            }]),
        )
        .unwrap();
        crate::ir::set_fusion_parameters(
            &ctx,
            body,
            FusionParameters(vec![FusionParameter {
                name: Some("q".into()),
                ..Default::default()
            }]),
        )
        .unwrap();
        fusion
            .get_operation()
            .insert_before(&ctx, block.deref(&ctx).get_terminator(&ctx).unwrap());
        fusion.get_operation().deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(vec!["a".into()]),
        );
        let module = pliron::builtin::ops::ModuleOp::new(&mut ctx, Identifier::new("test"));
        let module_block = module
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        owner.get_operation().insert_at_back(module_block, &ctx);
        crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("main")).unwrap();
        let physical = block.deref(&ctx).iter(&ctx).collect::<Vec<_>>();
        assert_eq!(
            instruction_order(&ctx, owner.get_operation()).unwrap(),
            vec![
                parameter.get_operation(),
                first.get_operation(),
                fusion.get_operation()
            ]
        );
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        let valid = InstructionOrder {
            instructions: vec!["f".into(), "a".into(), "p".into()],
            blocks: vec![OrderPermutation {
                subject: "f".into(),
                members: vec!["n".into(), "q".into()],
            }],
            users: vec![OrderPermutation {
                subject: "q".into(),
                members: vec!["n".into()],
            }],
            control_successors: vec![OrderPermutation {
                subject: "a".into(),
                members: vec!["f".into()],
            }],
        };
        set_order(&ctx, owner.get_operation(), valid.clone()).unwrap();
        assert_eq!(
            block_value_order(&ctx, body).unwrap(),
            vec![inner_value, argument]
        );
        assert_eq!(
            block_instruction_order(&ctx, body).unwrap(),
            vec![inner.get_operation()]
        );
        assert_eq!(body.deref(&ctx).get_argument(0), argument);
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        assert_eq!(physical, block.deref(&ctx).iter(&ctx).collect::<Vec<_>>());
        let mut invalid = Vec::new();
        for subject in ["missing", "p"] {
            let mut order = valid.clone();
            order.blocks[0].subject = subject.into();
            invalid.push(order);
        }
        let mut order = valid.clone();
        order.blocks.push(order.blocks[0].clone());
        invalid.push(order);
        for members in [
            vec!["missing".into(), "q".into()],
            vec!["n".into(), "n".into()],
            vec!["n".into()],
            vec!["q".into(), "q".into()],
        ] {
            let mut order = valid.clone();
            order.blocks[0].members = members;
            invalid.push(order);
        }
        for controls in [false, true] {
            for subject in ["missing", "p"] {
                let mut order = valid.clone();
                let records = if controls {
                    &mut order.control_successors
                } else {
                    &mut order.users
                };
                records[0].subject = subject.into();
                invalid.push(order);
            }
            let mut order = valid.clone();
            let records = if controls {
                &mut order.control_successors
            } else {
                &mut order.users
            };
            records.push(records[0].clone());
            invalid.push(order);
        }
        let text = module.get_operation().disp(&ctx).to_string();
        let mut parsed_context = Context::new();
        crate::register(&mut parsed_context);
        let parsed = pliron::parsable::parse_from_str(
            pliron::irfmt::parsers::spaced(Operation::top_level_parser()),
            &mut parsed_context,
            &text,
        )
        .unwrap();
        crate::verification::verify(&parsed_context, parsed).unwrap();
        assert_eq!(
            snapshot(&ctx, module.get_operation()),
            snapshot(&parsed_context, parsed)
        );
        Operation::erase(parsed, &mut parsed_context);
        for order in invalid {
            owner
                .get_operation()
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new(KEY), order);
            assert!(crate::verification::verify(&ctx, module.get_operation()).is_err());
            let text = module.get_operation().disp(&ctx).to_string();
            let mut parsed_context = Context::new();
            crate::register(&mut parsed_context);
            let parsed = pliron::parsable::parse_from_str(
                pliron::irfmt::parsers::spaced(Operation::top_level_parser()),
                &mut parsed_context,
                &text,
            )
            .unwrap();
            assert!(crate::verification::verify(&parsed_context, parsed).is_err());
            Operation::erase(parsed, &mut parsed_context);
        }
        set_order(&ctx, owner.get_operation(), valid).unwrap();
        Operation::replace_operand(first.get_operation(), &ctx, 0, result);
        assert!(crate::verification::verify(&ctx, module.get_operation()).is_err());
        let input = parameter.result(&ctx);
        Operation::replace_operand(first.get_operation(), &ctx, 0, input);
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        invalidate_order(&ctx, body).unwrap();
        assert_eq!(
            instruction_order(&ctx, owner.get_operation()).unwrap(),
            vec![
                parameter.get_operation(),
                first.get_operation(),
                fusion.get_operation()
            ]
        );
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
    }

    fn snapshot(ctx: &Context, owner: Ptr<Operation>) -> Vec<String> {
        let mut rows = Vec::new();
        let mut pending = vec![owner];
        while let Some(operation) = pending.pop() {
            let data = operation.deref(ctx);
            let mut attributes: Vec<_> = data
                .attributes
                .0
                .iter()
                .filter(|(name, _)| **name != Identifier::new("builtin_given_names"))
                .map(|(name, value)| format!("{name}: {}", value.disp(ctx)))
                .collect();
            attributes.sort();
            let types: Vec<_> = data
                .result_types()
                .map(|ty| ty.disp(ctx).to_string())
                .collect();
            rows.push(format!(
                "{} {types:?} {attributes:?}",
                Operation::get_op_dyn(operation, ctx).get_opid().disp(ctx),
            ));
            for region in data.regions() {
                for block in region.deref(ctx).iter(ctx) {
                    pending.extend(block.deref(ctx).iter(ctx));
                }
            }
        }
        rows
    }

    #[test]
    fn logical_order_and_deduplicated_users_do_not_reorder_physical_ssa() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        register(&mut ctx);
        let ty = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
        let signature = FunctionType::get(&ctx, vec![ty], vec![ty]).into();
        let owner = ComputationOp::new(&mut ctx, Identifier::new("main"), signature);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(owner.body(&ctx), &ctx);
        let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
        parameter.get_operation().insert_at_back(block, &ctx);
        let value = parameter.result(&ctx);
        let first = AddOp::new(&mut ctx, ty, vec![value, value]);
        first.get_operation().insert_at_back(block, &ctx);
        let second = AddOp::new(&mut ctx, ty, vec![value, value]);
        second.get_operation().insert_at_back(block, &ctx);
        let root = second.result(&ctx);
        ReturnOp::new(&mut ctx, root)
            .get_operation()
            .insert_at_back(block, &ctx);
        for (operation, name) in [
            (parameter.get_operation(), "p"),
            (first.get_operation(), "a"),
            (second.get_operation(), "b"),
        ] {
            operation
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_name"), InstructionName(name.into()));
        }
        let physical: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        set_order(
            &ctx,
            owner.get_operation(),
            InstructionOrder {
                instructions: vec!["b".into(), "p".into(), "a".into()],
                users: vec![OrderPermutation {
                    subject: "p".into(),
                    members: vec!["a".into(), "b".into()],
                }],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            users(&ctx, value).unwrap(),
            vec![first.result(&ctx), second.result(&ctx)]
        );
        assert_eq!(block.deref(&ctx).iter(&ctx).collect::<Vec<_>>(), physical);
        assert_eq!(
            instruction_order(&ctx, owner.get_operation()).unwrap()[0],
            second.get_operation()
        );
        crate::verification::verify(&ctx, owner.get_operation()).unwrap();
        let text = owner.get_operation().disp(&ctx).to_string();
        let mut parsed_context = Context::new();
        crate::register(&mut parsed_context);
        register(&mut parsed_context);
        let parsed = pliron::parsable::parse_from_str(
            pliron::irfmt::parsers::spaced(Operation::top_level_parser()),
            &mut parsed_context,
            &text,
        )
        .unwrap();
        verify_order(&parsed_context, parsed).unwrap();
        assert_eq!(
            snapshot(&ctx, owner.get_operation()),
            snapshot(&parsed_context, parsed)
        );
        let cloned =
            pliron::irbuild::cloning::clone_operation(
                owner.get_operation(),
                &mut ctx,
                &mut pliron::irbuild::rewriter::IRRewriter::<
                    pliron::irbuild::listener::DummyListener,
                >::default(),
                &mut pliron::irbuild::cloning::IrMapping::new(),
            );
        verify_order(&ctx, cloned).unwrap();
        assert_eq!(
            snapshot(&ctx, owner.get_operation()),
            snapshot(&ctx, cloned)
        );
        Operation::erase(cloned, &mut ctx);
        let before = owner.get_operation().disp(&ctx).to_string();
        assert!(
            set_order(
                &ctx,
                owner.get_operation(),
                InstructionOrder {
                    instructions: vec!["p".into(), "p".into(), "a".into()],
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert_eq!(before, owner.get_operation().disp(&ctx).to_string());
        second.get_operation().deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(vec!["a".into()]),
        );
        assert_eq!(
            control_predecessors(&ctx, second.result(&ctx)).unwrap(),
            vec![first.result(&ctx)]
        );
        assert_eq!(
            control_successors(&ctx, first.result(&ctx)).unwrap(),
            vec![second.result(&ctx)]
        );
        let before = owner.get_operation().disp(&ctx).to_string();
        assert!(
            set_order(
                &ctx,
                owner.get_operation(),
                InstructionOrder {
                    control_successors: vec![OrderPermutation {
                        subject: "a".into(),
                        members: vec!["p".into()],
                    }],
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert_eq!(before, owner.get_operation().disp(&ctx).to_string());
        let third = AddOp::new(&mut ctx, ty, vec![value, value]);
        third
            .get_operation()
            .insert_before(&ctx, second.get_operation());
        assert!(users(&ctx, value).is_err());
        invalidate_order(&ctx, block).unwrap();
        assert_eq!(users(&ctx, value).unwrap().len(), 3);
    }
}
