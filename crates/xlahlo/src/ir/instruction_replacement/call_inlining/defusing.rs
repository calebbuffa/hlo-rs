//! Transactional native fusion defusing.
//!
//! Mirrors the pinned `HloInstruction::Defuse` graph rewrite. Pass traversal,
//! single-caller eligibility, and execution-thread filtering belong to the
//! optimizer consumer.

use super::*;
use crate::{
    ir::{
        fusion_parameters, instruction_order::invalidate_order, verify_fusion_parameter_structure,
    },
    ops::control::FusionOp,
    verification::VerifyOptions,
};
use pliron::{
    irbuild::{
        cloning::IrMapping,
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    value::DefiningEntity,
};
use std::collections::{HashMap, HashSet, VecDeque};

use super::super::control_graph::{ControlGraph, ControlNode};

/// The result of defusing one fusion.
///
/// `instructions` maps each source instruction in the fused body to its
/// caller-body clone. Parameters are mapped to caller operands and are omitted.
pub struct DefusedFusion {
    /// The mapped fusion root, either a caller operand or a cloned instruction.
    pub root: Value,
    /// Source fused-body instruction to its caller-body clone.
    pub instructions: HashMap<Ptr<Operation>, Ptr<Operation>>,
}

struct StagedDefuse {
    root: Value,
    instructions: HashMap<Ptr<Operation>, Value>,
}

#[derive(Clone, Copy)]
enum BodyNode {
    Parameter,
    Instruction(Ptr<Operation>),
}

fn module_for(ctx: &Context, mut operation: Ptr<Operation>) -> Result<Ptr<Operation>> {
    loop {
        if Operation::is_op::<ModuleOp>(operation, ctx) {
            return Ok(operation);
        }
        operation = operation
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("defuse requires an attached native module"))?;
    }
}

fn fusion_body(ctx: &Context, fusion: Ptr<Operation>) -> Result<Ptr<BasicBlock>> {
    require(
        Operation::is_op::<FusionOp>(fusion, ctx),
        "defuse requires a native Fusion",
    )?;
    let regions: Vec<_> = fusion.deref(ctx).regions().collect();
    require(regions.len() == 1, "fusion requires one body region")?;
    let blocks: Vec<_> = regions[0].deref(ctx).iter(ctx).collect();
    require(blocks.len() == 1, "fusion requires one body block")?;
    require(
        blocks[0].deref(ctx).get_num_arguments() == fusion.deref(ctx).get_num_operands(),
        "fusion parameter count does not match its operands",
    )?;
    Ok(blocks[0])
}

fn dependency_order(ctx: &Context, block: Ptr<BasicBlock>) -> Result<(Vec<Ptr<Operation>>, Value)> {
    verify_fusion_parameter_structure(ctx, block)?;
    let parameters = fusion_parameters(ctx, block)?;
    let all: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let terminators: Vec<_> = all
        .iter()
        .copied()
        .filter(|&op| Operation::is_op::<ReturnOp>(op, ctx))
        .collect();
    require(terminators.len() == 1, "fusion requires one return")?;
    let terminator = terminators[0];
    require(
        all.last() == Some(&terminator) && terminator.deref(ctx).get_num_operands() == 1,
        "fusion return must be the final single-root terminator",
    )?;
    let root = terminator.deref(ctx).get_operand(0);
    let instructions: Vec<_> = all
        .into_iter()
        .filter(|&op| !Operation::is_op::<ReturnOp>(op, ctx))
        .collect();
    require(
        instructions
            .iter()
            .all(|op| op.deref(ctx).get_num_results() == 1),
        "fusion body contains a non-instruction",
    )?;

    let mut members = HashMap::new();
    let mut names = HashMap::<String, BodyNode>::new();
    for parameter in &parameters.0 {
        if let Some(name) = &parameter.name {
            require(
                !name.is_empty() && names.insert(name.clone(), BodyNode::Parameter).is_none(),
                "fusion has an invalid or duplicate parameter name",
            )?;
        }
    }
    for &instruction in &instructions {
        members.insert(instruction, members.len());
        if let Some(name) = optional_attr::<InstructionName>(ctx, instruction, "hlo_name")? {
            require(
                !name.0.is_empty()
                    && names
                        .insert(name.0.clone(), BodyNode::Instruction(instruction))
                        .is_none(),
                "fusion has an invalid or duplicate instruction name",
            )?;
        }
    }

    let mut indegree = vec![0usize; instructions.len()];
    let mut successors = vec![Vec::new(); instructions.len()];
    let mut edges = HashSet::new();
    for (target_index, &target) in instructions.iter().enumerate() {
        let mut dependencies = Vec::new();
        for operand in target.deref(ctx).operands() {
            match operand.defining_entity() {
                DefiningEntity::Op(definition) => {
                    require(
                        members.contains_key(&definition),
                        "fusion instruction captures an operation outside its body",
                    )?;
                    dependencies.push(definition);
                }
                DefiningEntity::Block(owner) => {
                    require(
                        owner == block
                            && (0..block.deref(ctx).get_num_arguments())
                                .any(|index| block.deref(ctx).get_argument(index) == operand),
                        "fusion instruction captures a value outside its body",
                    )?;
                }
            }
        }
        if let Some(controls) =
            optional_attr::<ControlPredecessors>(ctx, target, "hlo_control_predecessors")?
        {
            for name in &controls.0 {
                match names
                    .get(name)
                    .copied()
                    .ok_or_else(|| error(format!("unknown fusion control predecessor: {name}")))?
                {
                    BodyNode::Instruction(operation) => dependencies.push(operation),
                    BodyNode::Parameter => {}
                }
            }
        }
        for dependency in dependencies {
            let source_index = *members
                .get(&dependency)
                .ok_or_else(|| error("fusion dependency is outside its body"))?;
            if edges.insert((source_index, target_index)) {
                indegree[target_index] += 1;
                successors[source_index].push(target_index);
            }
        }
    }

    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut order = Vec::with_capacity(instructions.len());
    while let Some(index) = ready.pop_front() {
        order.push(instructions[index]);
        for &successor in &successors[index] {
            indegree[successor] = indegree[successor]
                .checked_sub(1)
                .ok_or_else(|| error("inconsistent fusion dependency graph"))?;
            if indegree[successor] == 0 {
                ready.push_back(successor);
            }
        }
    }
    require(
        order.len() == instructions.len(),
        "cycle in fusion data/control dependencies",
    )?;
    require(
        match root.defining_entity() {
            DefiningEntity::Op(operation) => members.contains_key(&operation),
            DefiningEntity::Block(owner) => owner == block,
        },
        "fusion root is outside its body",
    )?;
    Ok((order, root))
}

fn append_edge(
    nodes: &mut [super::super::control_graph::ControlNode],
    target: usize,
    source: usize,
) {
    if !nodes[target].incoming.contains(&source) {
        nodes[target].incoming.push(source);
    }
}

fn defuse_impl(
    ctx: &mut Context,
    module: Ptr<Operation>,
    fusion: Ptr<Operation>,
) -> Result<StagedDefuse> {
    let fusion_block = fusion_body(ctx, fusion)?;
    let (order, source_root) = dependency_order(ctx, fusion_block)?;
    let caller_block = fusion
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| error("detached fusion"))?;
    let mut controls = ControlGraph::read(ctx, caller_block)?;
    let fusion_control_index = controls.index(fusion.deref(ctx).get_result(0))?;
    controls.ensure_names(ctx, module)?;
    let external_predecessors = controls.nodes[fusion_control_index].incoming.clone();
    let fusion_name = controls.nodes[fusion_control_index].name.clone();
    let successors = controls
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            node.incoming
                .contains(&fusion_control_index)
                .then_some(index)
        })
        .collect::<Vec<_>>();

    let operands = fusion.deref(ctx).operands().collect::<Vec<_>>();
    let parameters = fusion_parameters(ctx, fusion_block)?;
    let mut mapping = IrMapping::new();
    for (index, operand) in operands.iter().copied().enumerate() {
        mapping.map_value(fusion_block.deref(ctx).get_argument(index), operand);
    }
    let mut cloner = InstructionCloner::new(ctx, module)?;
    let mut cloned = HashMap::new();
    let mut clone_order = Vec::with_capacity(order.len());
    for source in order {
        let operation = cloner.clone_instruction(ctx, module, source, fusion, &mut mapping)?;
        cloned.insert(source, operation);
        clone_order.push(operation);
    }
    let root = mapping
        .lookup_value(source_root)
        .ok_or_else(|| error("fusion root was not mapped"))?;
    require(
        shape::compatible(
            ctx,
            fusion.deref(ctx).get_result(0).get_type(ctx),
            root.get_type(ctx),
        ),
        "mapped fusion root has an incompatible shape",
    )?;

    let mut source_control_nodes = HashMap::<String, usize>::new();
    for (index, parameter) in parameters.0.iter().enumerate() {
        if let Some(name) = &parameter.name {
            source_control_nodes.insert(name.clone(), controls.index(operands[index])?);
        }
    }
    for (&source, &clone) in &cloned {
        let clone_name = optional_attr::<InstructionName>(ctx, clone, "hlo_name")?
            .ok_or_else(|| error("cloned instruction has no HLO name"))?;
        let index = controls.nodes.len();
        controls.nodes.push(ControlNode {
            value: clone.deref(ctx).get_result(0),
            name: Some(clone_name.0.clone()),
            incoming: Vec::new(),
        });
        if let Some(source_name) = optional_attr::<InstructionName>(ctx, source, "hlo_name")? {
            source_control_nodes.insert(source_name.0.clone(), index);
        }
    }

    for (&source, &clone) in &cloned {
        let index = controls.index(clone.deref(ctx).get_result(0))?;
        for predecessor in &external_predecessors {
            append_edge(&mut controls.nodes, index, *predecessor);
        }
        if let Some(predecessors) =
            optional_attr::<ControlPredecessors>(ctx, source, "hlo_control_predecessors")?
        {
            for name in &predecessors.0 {
                let predecessor = *source_control_nodes
                    .get(name)
                    .ok_or_else(|| error(format!("unmapped fusion control predecessor: {name}")))?;
                append_edge(&mut controls.nodes, index, predecessor);
            }
        }
    }
    for (parameter_index, parameter) in parameters.0.iter().enumerate() {
        let target = controls.index(operands[parameter_index])?;
        for name in &parameter.control_predecessors {
            let predecessor = *source_control_nodes
                .get(name)
                .ok_or_else(|| error(format!("unmapped fusion parameter control: {name}")))?;
            append_edge(&mut controls.nodes, target, predecessor);
        }
    }

    let root_control_index = match root.defining_entity() {
        DefiningEntity::Op(operation) => controls.index(operation.deref(ctx).get_result(0))?,
        DefiningEntity::Block(owner) => {
            let index = owner
                .deref(ctx)
                .arguments()
                .position(|argument| argument == root)
                .ok_or_else(|| error("mapped fusion root is not a caller parameter"))?;
            controls.index(owner.deref(ctx).get_argument(index))?
        }
    };
    if cloned.is_empty() {
        for &successor in &successors {
            controls.nodes[successor]
                .incoming
                .retain(|&predecessor| predecessor != fusion_control_index);
            for &predecessor in &external_predecessors {
                append_edge(&mut controls.nodes, successor, predecessor);
            }
        }
    } else {
        for &successor in &successors {
            controls.nodes[successor]
                .incoming
                .retain(|&predecessor| predecessor != fusion_control_index);
            append_edge(&mut controls.nodes, successor, root_control_index);
        }
    }
    controls.nodes[fusion_control_index].incoming.clear();
    if fusion_name.is_some() && root_control_index == fusion_control_index {
        return Err(error("fusion root cannot be the fusion itself"));
    }
    controls.ensure_names(ctx, module)?;
    controls.write(ctx)?;
    invalidate_order(ctx, caller_block)?;

    let original_fusion_value = fusion.deref(ctx).get_result(0);
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.replace_value_uses_with(ctx, original_fusion_value, root);
    Operation::erase(fusion, ctx);

    let existing_schedule = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .as_deref()
        .cloned();
    if let Some(mut schedule) = existing_schedule {
        if let Some(computation) = caller_block
            .deref(ctx)
            .get_parent_op(ctx)
            .and_then(|owner| Operation::get_op::<ComputationOp>(owner, ctx))
            && let Some(symbol) = computation.get_attr_hlo_computation_name(ctx)
            && let Some(sequence) = schedule
                .0
                .iter_mut()
                .find(|sequence| sequence.computation == *symbol)
        {
            if let Some(old_name) = fusion_name {
                let slots = sequence
                    .instructions
                    .iter()
                    .enumerate()
                    .filter_map(|(index, name)| (name == &old_name).then_some(index))
                    .collect::<Vec<_>>();
                require(
                    slots.len() == 1,
                    "scheduled fusion must occur exactly once in its caller sequence",
                )?;
                let names = clone_order
                    .iter()
                    .map(|&operation| {
                        optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
                            .map(|name| name.0.clone())
                            .ok_or_else(|| error("cloned instruction has no schedule name"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                sequence.instructions.splice(slots[0]..=slots[0], names);
            } else {
                return Err(error("scheduled fusion has no instruction name"));
            }
        }
        module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_schedule"), schedule);
        crate::ir::schedule_update::update_schedule(ctx, module)?;
    }

    Ok(StagedDefuse {
        root,
        instructions: cloned
            .into_iter()
            .map(|(source, operation)| (source, operation.deref(ctx).get_result(0)))
            .collect(),
    })
}

/// Expand one fusion into its caller body transactionally.
///
/// Every non-parameter instruction is cloned, including unused or effectful
/// instructions. Inner and boundary control dependencies are remapped, and an
/// existing caller schedule expands at the fusion's slot in dependency order.
/// Fusion-local output/operand alias hints expire with the fusion; aliases on
/// cloned instructions and module input/output alias metadata are preserved.
/// No schedule is invented. `Some` rehearses and verifies the complete edit
/// before publishing it.
pub fn defuse(
    ctx: &mut Context,
    fusion: Ptr<Operation>,
    verification: Option<VerifyOptions>,
) -> Result<DefusedFusion> {
    let module = module_for(ctx, fusion)?;
    let source_instructions = fusion_body(ctx, fusion)?
        .deref(ctx)
        .iter(ctx)
        .filter(|&operation| !Operation::is_op::<ReturnOp>(operation, ctx))
        .collect::<Vec<_>>();
    if let Some(options) = verification {
        super::super::require_rehearsable_policy(options)?;
    }
    edit_staging::staged_edit(
        ctx,
        module,
        verification,
        |ctx, scratch, initial| {
            let staged_fusion = initial
                .lookup_op(fusion)
                .ok_or_else(|| error("fusion has no rehearsal identity"))?;
            let result = defuse_impl(ctx, scratch, staged_fusion)?;
            let instructions = source_instructions
                .iter()
                .map(|&source| {
                    let staged_source = initial
                        .lookup_op(source)
                        .ok_or_else(|| error("fusion instruction has no rehearsal identity"))?;
                    let value = result
                        .instructions
                        .get(&staged_source)
                        .copied()
                        .ok_or_else(|| error("fusion instruction was not defused"))?;
                    Ok((source, value))
                })
                .collect::<Result<_>>()?;
            Ok(StagedDefuse {
                root: result.root,
                instructions,
            })
        },
        |result, live| {
            Ok(DefusedFusion {
                root: live.value(result.root)?,
                instructions: result
                    .instructions
                    .into_iter()
                    .map(|(source, value)| {
                        let value = live.value(value)?;
                        let operation = value
                            .defining_op()
                            .ok_or_else(|| error("defused instruction lost its operation"))?;
                        Ok((source, operation))
                    })
                    .collect::<Result<_>>()?,
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Axes, Bools, FusionKind, OutputOperandAliases, OutputOperandAliasing},
        ir::{
            AliasEntry, AliasKind, FusionParameter, FusionParameters, InputOutputAliases,
            InstructionSequence, Schedule,
        },
        ops::{
            control::{CustomCallOp, FusionOp},
            elementwise::AddOp,
            movement::CopyOp,
        },
        verification::verify,
    };
    use pliron::{
        builtin::attributes::StringAttr, identifier::Identifier, linked_list::ContainsLinkedList,
        op::Op, operation::Operation, printable::Printable,
    };

    use super::super::tests::{attr, fixture, name};

    fn replace_call_with_fusion(
        ctx: &mut Context,
        fixture: &super::super::tests::Fixture,
        with_root_instruction: bool,
    ) -> (Ptr<Operation>, Ptr<Operation>, Vec<Ptr<Operation>>) {
        Operation::replace_operand(fixture.ret, ctx, 0, fixture.parameter);
        Operation::erase(fixture.call, ctx);
        let producer = CopyOp::new(ctx, fixture.ty, vec![fixture.parameter]).get_operation();
        producer.insert_before(ctx, fixture.ret);
        name(ctx, producer, "producer");

        let producer_value = producer.deref(ctx).get_result(0);
        let fusion =
            FusionOp::new(ctx, fixture.ty, vec![producer_value], FusionKind::Loop).get_operation();
        fusion.insert_before(ctx, fixture.ret);
        name(ctx, fusion, "fusion");
        attr(
            ctx,
            fusion,
            "hlo_aliases",
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![]),
                operand_index: 0,
                operand_shape_index: Axes(vec![]),
            }]),
        );
        attr(
            ctx,
            fusion,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["producer".into()]),
        );
        let body = BasicBlock::new(ctx, None, vec![fixture.ty]);
        body.insert_at_back(fusion.deref(ctx).get_region(0), ctx);
        crate::ir::set_fusion_parameters(
            ctx,
            body,
            FusionParameters(vec![FusionParameter {
                name: Some("inner.arg".into()),
                ..Default::default()
            }]),
        )
        .unwrap();
        let argument = body.deref(ctx).get_argument(0);
        let mut sources = Vec::new();
        let root = if with_root_instruction {
            let dead = CustomCallOp::new(
                ctx,
                fixture.ty,
                vec![argument],
                StringAttr::new("side_effect".into()),
            )
            .get_operation();
            dead.insert_at_back(body, ctx);
            dead.deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            name(ctx, dead, "inner.dead");
            sources.push(dead);
            let add = AddOp::new(ctx, fixture.ty, vec![argument, argument]).get_operation();
            add.insert_at_back(body, ctx);
            name(ctx, add, "inner.root");
            sources.push(add);
            add.deref(ctx).get_result(0)
        } else {
            let side = CustomCallOp::new(
                ctx,
                fixture.ty,
                vec![],
                StringAttr::new("side_effect".into()),
            )
            .get_operation();
            side.insert_at_back(body, ctx);
            side.deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            name(ctx, side, "inner.side");
            sources.push(side);
            crate::ir::set_fusion_parameters(
                ctx,
                body,
                FusionParameters(vec![FusionParameter {
                    name: Some("inner.arg".into()),
                    control_predecessors: vec!["inner.side".into()],
                    ..Default::default()
                }]),
            )
            .unwrap();
            let side_value = side.deref(ctx).get_result(0);
            let root = AddOp::new(ctx, fixture.ty, vec![argument, side_value]).get_operation();
            root.insert_at_back(body, ctx);
            name(ctx, root, "inner.root");
            sources.push(root);
            root.deref(ctx).get_result(0)
        };
        ReturnOp::new(ctx, root)
            .get_operation()
            .insert_at_back(body, ctx);

        if with_root_instruction {
            let fusion_value = fusion.deref(ctx).get_result(0);
            let after = CopyOp::new(ctx, fixture.ty, vec![fusion_value]).get_operation();
            after.insert_before(ctx, fixture.ret);
            name(ctx, after, "after");
            attr(
                ctx,
                after,
                "hlo_control_predecessors",
                ControlPredecessors(vec!["fusion".into()]),
            );
            let after_value = after.deref(ctx).get_result(0);
            Operation::replace_operand(fixture.ret, ctx, 0, after_value);
            (fusion, producer, sources)
        } else {
            let fusion_value = fusion.deref(ctx).get_result(0);
            Operation::replace_operand(fixture.ret, ctx, 0, fusion_value);
            (fusion, producer, sources)
        }
    }

    #[test]
    fn defuse_clones_dead_instructions_relays_controls_and_expands_the_schedule() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx, true);
        let (fusion, _, sources) = replace_call_with_fusion(ctx, &fixture, true);
        fixture.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![
                InstructionSequence {
                    computation: "callee".into(),
                    instructions: vec!["callee.p".into()],
                },
                InstructionSequence {
                    computation: "main".into(),
                    instructions: vec![
                        "main.p".into(),
                        "producer".into(),
                        "fusion".into(),
                        "after".into(),
                    ],
                },
            ]),
        );
        let module_aliases = InputOutputAliases(vec![AliasEntry {
            output_shape_index: Axes(vec![]),
            parameter_number: 0,
            parameter_shape_index: Axes(vec![]),
            kind: AliasKind::MayAlias,
        }]);
        attr(
            ctx,
            fixture.module,
            "hlo_input_output_aliases",
            module_aliases.clone(),
        );
        verify(ctx, fixture.module).unwrap();

        let result = defuse(ctx, fusion, Some(VerifyOptions::default())).unwrap();
        assert_eq!(result.instructions.len(), 2);
        assert!(
            sources
                .iter()
                .all(|source| result.instructions.contains_key(source))
        );
        verify(ctx, fixture.module).unwrap();

        let dead = result.instructions[&sources[0]];
        let root = result.instructions[&sources[1]];
        assert!(
            optional_attr::<OutputOperandAliases>(ctx, root, "hlo_aliases")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            optional_attr::<InputOutputAliases>(ctx, fixture.module, "hlo_input_output_aliases")
                .unwrap()
                .as_deref(),
            Some(&module_aliases)
        );
        let dead_name = optional_attr::<InstructionName>(ctx, dead, "hlo_name")
            .unwrap()
            .unwrap()
            .0
            .clone();
        let root_name = optional_attr::<InstructionName>(ctx, root, "hlo_name")
            .unwrap()
            .unwrap()
            .0
            .clone();
        assert!(
            optional_attr::<ControlPredecessors>(ctx, dead, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0
                .contains(&"producer".to_owned())
        );
        let after = fixture
            .caller_block
            .deref(ctx)
            .iter(ctx)
            .find(|&operation| {
                optional_attr::<InstructionName>(ctx, operation, "hlo_name")
                    .unwrap()
                    .is_some_and(|name| name.0 == "after")
            })
            .unwrap();
        assert_eq!(
            optional_attr::<ControlPredecessors>(ctx, after, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec![root_name.clone()]
        );
        let schedule = optional_attr::<Schedule>(ctx, fixture.module, "hlo_schedule")
            .unwrap()
            .unwrap();
        let sequence = &schedule
            .0
            .iter()
            .find(|sequence| sequence.computation == "main")
            .unwrap()
            .instructions;
        let producer_index = sequence.iter().position(|name| name == "producer").unwrap();
        let dead_index = sequence.iter().position(|name| name == &dead_name).unwrap();
        let root_index = sequence.iter().position(|name| name == &root_name).unwrap();
        let after_index = sequence.iter().position(|name| name == "after").unwrap();
        assert!(producer_index < dead_index && dead_index < root_index && root_index < after_index);
        assert!(
            !fixture
                .caller_block
                .deref(ctx)
                .iter(ctx)
                .any(|operation| Operation::is_op::<FusionOp>(operation, ctx))
        );
    }

    #[test]
    fn defuse_parameter_passthrough_splices_external_control_edges() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx, true);
        Operation::replace_operand(fixture.ret, ctx, 0, fixture.parameter);
        Operation::erase(fixture.call, ctx);
        let producer = CopyOp::new(ctx, fixture.ty, vec![fixture.parameter]).get_operation();
        producer.insert_before(ctx, fixture.ret);
        name(ctx, producer, "producer");

        let producer_value = producer.deref(ctx).get_result(0);
        let fusion =
            FusionOp::new(ctx, fixture.ty, vec![producer_value], FusionKind::Loop).get_operation();
        fusion.insert_before(ctx, fixture.ret);
        name(ctx, fusion, "fusion");
        attr(
            ctx,
            fusion,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["producer".into()]),
        );
        attr(
            ctx,
            fusion,
            "hlo_aliases",
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![]),
                operand_index: 0,
                operand_shape_index: Axes(vec![]),
            }]),
        );
        let body = BasicBlock::new(ctx, None, vec![fixture.ty]);
        body.insert_at_back(fusion.deref(ctx).get_region(0), ctx);
        crate::ir::set_fusion_parameters(
            ctx,
            body,
            FusionParameters(vec![FusionParameter {
                name: Some("inner.arg".into()),
                ..Default::default()
            }]),
        )
        .unwrap();
        let argument = body.deref(ctx).get_argument(0);
        ReturnOp::new(ctx, argument)
            .get_operation()
            .insert_at_back(body, ctx);
        let fusion_value = fusion.deref(ctx).get_result(0);
        let after = CopyOp::new(ctx, fixture.ty, vec![fusion_value]).get_operation();
        after.insert_before(ctx, fixture.ret);
        name(ctx, after, "after");
        attr(
            ctx,
            after,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["fusion".into()]),
        );
        let after_value = after.deref(ctx).get_result(0);
        Operation::replace_operand(fixture.ret, ctx, 0, after_value);
        verify(ctx, fixture.module).unwrap();

        let result = defuse(ctx, fusion, Some(VerifyOptions::default())).unwrap();
        assert!(result.instructions.is_empty());
        assert_eq!(result.root, producer.deref(ctx).get_result(0));
        assert!(
            optional_attr::<OutputOperandAliases>(ctx, producer, "hlo_aliases")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            optional_attr::<ControlPredecessors>(ctx, after, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec!["producer".to_owned()]
        );
        verify(ctx, fixture.module).unwrap();
    }

    #[test]
    fn defuse_rejects_relay_cycle_without_mutating_the_module() {
        let ctx = &mut Context::new();
        let fixture = fixture(ctx, true);
        let (fusion, producer, _) = replace_call_with_fusion(ctx, &fixture, false);
        verify(ctx, fixture.module).unwrap();
        let producer_value = producer.deref(ctx).get_result(0);
        let before = fixture.module.disp(ctx).to_string();

        let error = match defuse(ctx, fusion, Some(VerifyOptions::default())) {
            Ok(_) => panic!("cycle-producing defuse unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("cycle"));
        assert_eq!(fixture.module.disp(ctx).to_string(), before);
        assert_eq!(producer.deref(ctx).get_result(0), producer_value);
        verify(ctx, fixture.module).unwrap();
    }
}
