//! Typed metadata and mutation helpers for fusion block arguments.
use super::{ControlPredecessors, InstructionName};
use crate::{
    Diagnostic,
    attributes::{FrontendAttributes, OpMetadata},
    ops::control::FusionOp,
    original_value::OriginalValue,
    sharding::Sharding,
    verification::require,
};
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    combine::{Parser, attempt, optional, parser::char::char},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_attr,
    identifier::Identifier,
    irfmt::parsers::{delimited_list_parser, spaced},
    linked_list::ContainsLinkedList,
    operation::Operation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, ListSeparator, Printable},
    result::Result,
    r#type::Typed,
    value::DefiningEntity,
};
use std::collections::{HashMap, HashSet};

#[pliron_attr(name = "xlahlo.fusion_parameter", verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Metadata for one fusion block argument, indexed by its position and parameter number.
pub struct FusionParameter {
    /// HLO parameter instruction name, when one was present in the source.
    pub name: Option<String>,
    /// Frontend annotations carried by the source parameter instruction.
    pub frontend_attributes: FrontendAttributes,
    /// Names of instructions or parameters that must precede this parameter.
    pub control_predecessors: Vec<String>,
    /// Source parameter metadata; absence is distinct from an explicit default.
    pub metadata: Option<OpMetadata>,
    /// Source parameter sharding, checked against this exact argument shape.
    pub sharding: Option<Sharding>,
    /// Original-value recovery tree for this parameter's shape.
    pub original_value: Option<OriginalValue>,
}

impl Printable for FusionParameter {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<")?;
        if let Some(name) = &self.name {
            name.fmt(ctx, state, f)?;
        }
        write!(f, ", ")?;
        self.frontend_attributes.fmt(ctx, state, f)?;
        write!(f, ", [")?;
        pliron::irfmt::printers::list_with_sep(
            &self.control_predecessors,
            ListSeparator::CharSpace(','),
        )
        .fmt(ctx, state, f)?;
        write!(f, "], ")?;
        if let Some(metadata) = &self.metadata {
            metadata.fmt(ctx, state, f)?;
        }
        write!(f, ", ")?;
        if let Some(sharding) = &self.sharding {
            sharding.fmt(ctx, state, f)?;
        }
        write!(f, ", ")?;
        if let Some(original) = &self.original_value {
            original.fmt(ctx, state, f)?;
        }
        write!(f, ">")
    }
}

impl Parsable for FusionParameter {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(stream: &mut StateStream<'a>, _: ()) -> ParseResult<'a, Self> {
        let annotations = optional(attempt(spaced(char(','))).with((
            optional(OpMetadata::parser(())).skip(spaced(char(','))),
            optional(Sharding::parser(())).skip(spaced(char(','))),
            optional(OriginalValue::parser(())),
        )));
        let ((name, frontend_attributes, control_predecessors, annotations), _) = spaced(char('<'))
            .with((
                optional(String::parser(())).skip(spaced(char(','))),
                FrontendAttributes::parser(()).skip(spaced(char(','))),
                delimited_list_parser('[', ']', ',', String::parser(())),
                annotations,
            ))
            .skip(spaced(char('>')))
            .parse_stream(stream)
            .into_result()?;
        let (metadata, sharding, original_value) = annotations.unwrap_or_default();
        Ok(Self {
            name,
            frontend_attributes,
            control_predecessors,
            metadata,
            sharding,
            original_value,
        })
        .into_parse_result()
    }
}

#[pliron_attr(
    name = "xlahlo.fusion_parameters",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Per-argument metadata on a fusion body block, in block-argument order.
pub struct FusionParameters(pub Vec<FusionParameter>);

fn fusion_block(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Ptr<Operation>> {
    let parent = block.deref(ctx).get_parent_op(ctx).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "fusion parameter block is detached".into(),
        })
    })?;
    let Some(fusion) = Operation::get_op::<FusionOp>(parent, ctx) else {
        return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "fusion parameter metadata requires a fusion body block".into(),
        }));
    };
    require(
        block.deref(ctx).get_parent_region() == Some(fusion.body(ctx)),
        "fusion parameter metadata requires a fusion body block",
    )?;
    Ok(parent)
}

/// Return metadata aligned with the fusion's block arguments.
///
/// Missing metadata is interpreted as the default record for each existing
/// block argument, preserving compatibility with older native assemblies.
pub fn fusion_parameters(ctx: &Context, block: Ptr<BasicBlock>) -> Result<FusionParameters> {
    fusion_block(ctx, block)?;
    let block_data = block.deref(ctx);
    let key = Identifier::new("hlo_fusion_parameters");
    if block_data.attributes.0.contains_key(&key)
        && block_data
            .attributes
            .get::<FusionParameters>(&key)
            .is_none()
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "hlo_fusion_parameters",
            message: "expected an ir::FusionParameters attribute".into(),
        }));
    }
    let parameters = match block_data.attributes.get::<FusionParameters>(&key) {
        Some(parameters) => parameters.clone(),
        None => FusionParameters(vec![
            FusionParameter::default();
            block_data.get_num_arguments()
        ]),
    };
    require(
        parameters.0.len() == block_data.get_num_arguments(),
        "fusion parameter metadata count differs from block arguments",
    )?;
    Ok(parameters)
}

/// Replace the complete parameter metadata vector after checking its alignment and names.
pub fn set_fusion_parameters(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    parameters: FusionParameters,
) -> Result<()> {
    fusion_block(ctx, block)?;
    validate_parameter_records(
        ctx,
        block,
        &parameters,
        block.deref(ctx).get_num_arguments(),
        None,
        true,
    )?;
    let order_changed = {
        let data = block.deref(ctx);
        let key = Identifier::new("hlo_fusion_parameters");
        match data.attributes.get::<FusionParameters>(&key) {
            Some(previous) => {
                previous.0.len() != parameters.0.len()
                    || previous.0.iter().zip(&parameters.0).any(|(old, new)| {
                        old.name != new.name || old.control_predecessors != new.control_predecessors
                    })
            }
            None => {
                data.attributes.0.contains_key(&key)
                    || parameters.0.iter().any(|parameter| {
                        parameter.name.is_some() || !parameter.control_predecessors.is_empty()
                    })
            }
        }
    };
    if order_changed {
        super::instruction_order::invalidate_order(ctx, block)?;
    }
    block
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_fusion_parameters"), parameters);
    Ok(())
}

/// Return whether a fusion parameter participates in any control dependency.
pub fn parameter_has_control_dependency(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    parameter_index: usize,
) -> Result<bool> {
    let parameters = fusion_parameters(ctx, block)?;
    let parameter = parameters.0.get(parameter_index).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "fusion parameter index is out of range".into(),
        })
    })?;
    if !parameter.control_predecessors.is_empty() {
        return Ok(true);
    }
    let Some(name) = parameter.name.as_deref() else {
        return Ok(false);
    };
    if parameters.0.iter().any(|target| {
        target
            .control_predecessors
            .iter()
            .any(|predecessor| predecessor == name)
    }) {
        return Ok(true);
    }
    Ok(block.deref(ctx).iter(ctx).any(|instruction| {
        instruction
            .deref(ctx)
            .attributes
            .get::<ControlPredecessors>(&Identifier::new("hlo_control_predecessors"))
            .is_some_and(|predecessors| predecessors.0.iter().any(|pred| pred == name))
    }))
}

/// Add a control edge to a parameter or instruction target in a fusion body.
///
/// The target stores the predecessor name, matching the native instruction
/// control-dependency representation.
pub fn add_fusion_control_dependency(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    predecessor: &str,
    target: &str,
) -> Result<()> {
    let mut parameters = fusion_parameters(ctx, block)?;
    require(
        predecessor != target,
        "control dependency cannot be a self-edge",
    )?;
    let predecessor_exists = parameters
        .0
        .iter()
        .any(|parameter| parameter.name.as_deref() == Some(predecessor))
        || block.deref(ctx).iter(ctx).any(|instruction| {
            instruction
                .deref(ctx)
                .attributes
                .get::<InstructionName>(&Identifier::new("hlo_name"))
                .is_some_and(|name| name.0 == predecessor)
        });
    require(predecessor_exists, "unknown fusion control predecessor")?;

    if let Some(index) = parameters
        .0
        .iter()
        .position(|parameter| parameter.name.as_deref() == Some(target))
    {
        let target_predecessors = &mut parameters.0[index].control_predecessors;
        require(
            !target_predecessors.iter().any(|value| value == predecessor),
            "duplicate fusion control dependency",
        )?;
        target_predecessors.push(predecessor.to_owned());
        set_fusion_parameters(ctx, block, parameters)
    } else {
        let target_operation = block.deref(ctx).iter(ctx).find(|instruction| {
            instruction
                .deref(ctx)
                .attributes
                .get::<InstructionName>(&Identifier::new("hlo_name"))
                .is_some_and(|name| name.0 == target)
        });
        let Some(target_operation) = target_operation else {
            return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "unknown fusion control dependency target".into(),
            }));
        };
        let key = Identifier::new("hlo_control_predecessors");
        let mut predecessors = target_operation
            .deref(ctx)
            .attributes
            .get::<ControlPredecessors>(&key)
            .cloned()
            .unwrap_or_default();
        require(
            !predecessors.0.iter().any(|value| value == predecessor),
            "duplicate fusion control dependency",
        )?;
        predecessors.0.push(predecessor.to_owned());
        super::instruction_order::invalidate_order(ctx, block)?;
        target_operation
            .deref_mut(ctx)
            .attributes
            .set(key, predecessors);
        Ok(())
    }
}

/// Remove one control edge while retaining all other parameter metadata.
pub fn remove_fusion_control_dependency(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    predecessor: &str,
    target: &str,
) -> Result<()> {
    let mut parameters = fusion_parameters(ctx, block)?;
    if let Some(index) = parameters
        .0
        .iter()
        .position(|parameter| parameter.name.as_deref() == Some(target))
    {
        let predecessor_list = &mut parameters.0[index].control_predecessors;
        let Some(edge_index) = predecessor_list
            .iter()
            .position(|value| value == predecessor)
        else {
            return Err(crate::verification::error(
                "fusion control dependency does not exist",
            ));
        };
        predecessor_list.remove(edge_index);
        return set_fusion_parameters(ctx, block, parameters);
    }

    let target_operation = block.deref(ctx).iter(ctx).find(|instruction| {
        instruction
            .deref(ctx)
            .attributes
            .get::<InstructionName>(&Identifier::new("hlo_name"))
            .is_some_and(|name| name.0 == target)
    });
    let Some(target_operation) = target_operation else {
        return Err(crate::verification::error(
            "unknown fusion control dependency target",
        ));
    };
    let key = Identifier::new("hlo_control_predecessors");
    let mut predecessors = target_operation
        .deref(ctx)
        .attributes
        .get::<ControlPredecessors>(&key)
        .cloned()
        .ok_or_else(|| crate::verification::error("fusion control dependency does not exist"))?;
    let Some(edge_index) = predecessors.0.iter().position(|value| value == predecessor) else {
        return Err(crate::verification::error(
            "fusion control dependency does not exist",
        ));
    };
    predecessors.0.remove(edge_index);
    super::instruction_order::invalidate_order(ctx, block)?;
    target_operation
        .deref_mut(ctx)
        .attributes
        .set(key, predecessors);
    Ok(())
}

/// Remove one unused fusion parameter and its corresponding fusion operand.
///
/// The operation validates all preconditions before mutation; parameters with
/// control edges cannot be removed. Block-argument indices and operand indices
/// are identical in valid HLO fusion bodies.
pub fn remove_fusion_parameter(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    parameter_index: usize,
) -> Result<FusionParameter> {
    let fusion = fusion_block(ctx, block)?;
    let mut parameters = fusion_parameters(ctx, block)?;
    let block_data = block.deref(ctx);
    let fusion_data = fusion.deref(ctx);
    require(
        block_data.get_num_arguments() == fusion_data.get_num_operands(),
        "fusion parameters and operands are not aligned",
    )?;
    require(
        parameter_index < block_data.get_num_arguments(),
        "fusion parameter index is out of range",
    )?;
    let argument = block_data.get_argument(parameter_index);
    require(!argument.is_used(ctx), "fusion parameter is still used")?;
    require(
        !parameter_has_control_dependency(ctx, block, parameter_index)?,
        "fusion parameter participates in a control dependency",
    )?;
    let removed = parameters.0.remove(parameter_index);
    validate_parameter_records(
        ctx,
        block,
        &parameters,
        block_data.get_num_arguments() - 1,
        Some(parameter_index),
        true,
    )?;
    drop(block_data);
    drop(fusion_data);

    super::instruction_order::invalidate_order(ctx, block)?;
    BasicBlock::remove_argument(block, ctx, parameter_index);
    Operation::remove_operand(fusion, ctx, parameter_index);
    block
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_fusion_parameters"), parameters);
    Ok(removed)
}

fn validate_parameter_records(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    parameters: &FusionParameters,
    expected_count: usize,
    removed_argument: Option<usize>,
    validate_annotations: bool,
) -> Result<()> {
    require(
        parameters.0.len() == expected_count,
        "fusion parameter metadata count differs from block arguments",
    )?;
    let mut names = HashSet::new();
    for (index, parameter) in parameters.0.iter().enumerate() {
        if validate_annotations {
            parameter.frontend_attributes.verify(ctx)?;
            if let Some(metadata) = &parameter.metadata {
                metadata.verify(ctx)?;
            }
            let argument_index =
                index + usize::from(removed_argument.is_some_and(|removed| index >= removed));
            let shape = block.deref(ctx).get_argument(argument_index).get_type(ctx);
            if let Some(sharding) = &parameter.sharding {
                sharding.validate(ctx, shape, None)?;
            }
            if let Some(original) = &parameter.original_value {
                original.verify_shape(ctx, shape)?;
            }
        }
        if let Some(name) = &parameter.name {
            require(
                !name.is_empty() && names.insert(name.as_str()),
                "fusion parameter names must be non-empty and unique",
            )?;
        }
    }
    for instruction in block.deref(ctx).iter(ctx) {
        if let Some(name) = instruction
            .deref(ctx)
            .attributes
            .get::<InstructionName>(&Identifier::new("hlo_name"))
        {
            require(
                !names.contains(name.0.as_str()),
                "fusion parameter name duplicates an instruction name",
            )?;
        }
    }
    for parameter in &parameters.0 {
        for predecessor in &parameter.control_predecessors {
            require(
                names.contains(predecessor.as_str())
                    || block.deref(ctx).iter(ctx).any(|instruction| {
                        instruction
                            .deref(ctx)
                            .attributes
                            .get::<InstructionName>(&Identifier::new("hlo_name"))
                            .is_some_and(|name| name.0 == *predecessor)
                    }),
                "unknown fusion parameter control predecessor",
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum FusionNode {
    Parameter(usize),
    Instruction(Ptr<Operation>),
}

fn add_fusion_edge(
    from: FusionNode,
    to: FusionNode,
    indegree: &mut HashMap<FusionNode, usize>,
    successors: &mut HashMap<FusionNode, Vec<FusionNode>>,
    edges: &mut HashSet<(FusionNode, FusionNode)>,
) -> Result<()> {
    require(
        indegree.contains_key(&from) && indegree.contains_key(&to),
        "fusion dependency is outside its body",
    )?;
    if edges.insert((from, to)) {
        *indegree
            .get_mut(&to)
            .ok_or_else(|| crate::verification::error("fusion dependency target missing"))? += 1;
        successors.entry(from).or_default().push(to);
    }
    Ok(())
}

/// Validate fusion parameter names and the combined data/control dependency graph.
///
/// Fusion parameters are SSA block arguments rather than operations, but XLA
/// represents them as parameter instructions with names and control edges.
pub fn verify_fusion_parameters(ctx: &Context, block: Ptr<BasicBlock>) -> Result<()> {
    verify_fusion_parameters_impl(ctx, block, true)
}

/// Check graph closure for raw cloning without selecting annotation verification policy.
#[cfg(test)]
pub(crate) fn verify_fusion_parameter_structure(
    ctx: &Context,
    block: Ptr<BasicBlock>,
) -> Result<()> {
    verify_fusion_parameters_impl(ctx, block, false)
}

fn verify_fusion_parameters_impl(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    validate_annotations: bool,
) -> Result<()> {
    let parameters = fusion_parameters(ctx, block)?;
    validate_parameter_records(
        ctx,
        block,
        &parameters,
        block.deref(ctx).get_num_arguments(),
        None,
        validate_annotations,
    )?;

    let instructions: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let mut names = HashMap::<String, FusionNode>::new();
    let mut nodes = Vec::with_capacity(parameters.0.len() + instructions.len());
    for index in 0..parameters.0.len() {
        let node = FusionNode::Parameter(index);
        nodes.push(node);
        if let Some(name) = &parameters.0[index].name {
            require(
                names.insert(name.clone(), node).is_none(),
                "duplicate fusion parameter name",
            )?;
        }
    }
    for &instruction in &instructions {
        let node = FusionNode::Instruction(instruction);
        nodes.push(node);
        if let Some(name) = instruction
            .deref(ctx)
            .attributes
            .get::<InstructionName>(&Identifier::new("hlo_name"))
        {
            require(!name.0.is_empty(), "instruction name cannot be empty")?;
            require(
                names.insert(name.0.clone(), node).is_none(),
                "duplicate fusion parameter or instruction name",
            )?;
        }
    }

    let mut indegree: HashMap<_, usize> = nodes.iter().copied().map(|node| (node, 0)).collect();
    let mut successors = HashMap::<FusionNode, Vec<FusionNode>>::new();
    let mut edges = HashSet::new();

    for &instruction in &instructions {
        let target = FusionNode::Instruction(instruction);
        for operand in instruction.deref(ctx).operands() {
            let source = match operand.defining_entity() {
                DefiningEntity::Op(definition) => FusionNode::Instruction(definition),
                DefiningEntity::Block(owner) if owner == block => {
                    let index = block
                        .deref(ctx)
                        .arguments()
                        .position(|argument| argument == operand)
                        .ok_or_else(|| {
                            crate::verification::error("fusion dependency uses an invalid argument")
                        })?;
                    FusionNode::Parameter(index)
                }
                DefiningEntity::Block(_) => {
                    return Err(crate::verification::error(
                        "fusion dependency captures a value from outside its body",
                    ));
                }
            };
            add_fusion_edge(source, target, &mut indegree, &mut successors, &mut edges)?;
        }
        if let Some(predecessors) = instruction
            .deref(ctx)
            .attributes
            .get::<ControlPredecessors>(&Identifier::new("hlo_control_predecessors"))
        {
            for predecessor in &predecessors.0 {
                let source = *names.get(predecessor).ok_or_else(|| {
                    crate::verification::error(format!(
                        "unknown fusion control predecessor: {predecessor}"
                    ))
                })?;
                require(
                    !matches!(source, FusionNode::Instruction(operation) if Operation::is_op::<crate::ops::ReturnOp>(operation, ctx)),
                    "return cannot be a control predecessor",
                )?;
                add_fusion_edge(source, target, &mut indegree, &mut successors, &mut edges)?;
            }
        }
    }
    for (index, parameter) in parameters.0.iter().enumerate() {
        for predecessor in &parameter.control_predecessors {
            let source = *names.get(predecessor).ok_or_else(|| {
                crate::verification::error(format!(
                    "unknown fusion parameter control predecessor: {predecessor}"
                ))
            })?;
            require(
                !matches!(source, FusionNode::Instruction(operation) if Operation::is_op::<crate::ops::ReturnOp>(operation, ctx)),
                "return cannot be a control predecessor",
            )?;
            add_fusion_edge(
                source,
                FusionNode::Parameter(index),
                &mut indegree,
                &mut successors,
                &mut edges,
            )?;
        }
    }

    let mut ready: Vec<_> = indegree
        .iter()
        .filter_map(|(&node, &degree)| (degree == 0).then_some(node))
        .collect();
    let mut visited = 0;
    while let Some(node) = ready.pop() {
        visited += 1;
        if let Some(users) = successors.get(&node) {
            for user in users {
                let degree = indegree.get_mut(user).ok_or_else(|| {
                    crate::verification::error("inconsistent fusion dependency indegree")
                })?;
                *degree = degree.checked_sub(1).ok_or_else(|| {
                    crate::verification::error("inconsistent fusion dependency indegree")
                })?;
                if *degree == 0 {
                    ready.push(*user);
                }
            }
        }
    }
    require(
        visited == nodes.len(),
        "cycle in fusion data/control dependencies",
    )
}

/// Register fusion-parameter metadata for native assembly.
pub(crate) fn register(ctx: &mut Context) {
    FusionParameter::register::<FusionParameter>(ctx);
    FusionParameters::register::<FusionParameters>(ctx);
}
