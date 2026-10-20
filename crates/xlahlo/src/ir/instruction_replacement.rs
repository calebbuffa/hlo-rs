//! Native instruction forwarding and recursive operand cleanup.
//!
//! Follows `HloComputation::ReplaceInstruction` and its default
//! `RemoveInstructionAndUnusedOperands` path. Ordinary parameters are protected;
//! fusion parameters repair their owning fusion's operands when removed.
use super::{
    ControlPredecessors, InstructionName, Schedule, fusion_parameters, remove_fusion_parameter,
    set_fusion_parameters,
};
use crate::{
    attributes::{FrontendAttributes, FusionKind, OpMetadata, OutputOperandAliases, Payload},
    effects::SideEffects,
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp, control::FusionOp},
    original_value::OriginalValue,
    shape::Shape,
    sharding::Sharding,
    verification::{VerifyOptions, error, optional_attr, require, verify_with_options},
};
use pliron::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::{IrMapping, clone_operation},
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    result::Result,
    r#type::Typed,
    value::{DefiningEntity, Value},
};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Policies corresponding to the pinned `ReplaceInstruction` arguments.
///
/// These are graph-edit policies, not verifier options.
#[derive(Clone, Copy, Debug)]
pub struct ReplaceInstructionOptions {
    pub preserve_sharding: bool,
    pub relay_control_dependency: bool,
    pub remove_unused_operands: bool,
    pub preserve_frontend_attributes: bool,
}

impl Default for ReplaceInstructionOptions {
    fn default() -> Self {
        Self {
            preserve_sharding: false,
            relay_control_dependency: false,
            remove_unused_operands: true,
            preserve_frontend_attributes: true,
        }
    }
}

/// The changed bit and erased operations, for consumers holding pointer worklists.
#[derive(Default)]
pub struct InstructionReplacement {
    pub changed: bool,
    pub removed: HashSet<Ptr<Operation>>,
}

/// Replace a one-result instruction by a compatible value in the same body.
///
/// A fusion block argument is an instruction endpoint for annotation/control
/// purposes, even though pliron represents it as a `Value`. Conflicting sharding
/// or unrelayed controls produce `changed=false`, matching XLA. Other malformed
/// input is an error, never an optimization nonmatch. Existing replacement
/// annotations are retained except for the pinned metadata/provenance rules.
/// `None` is the raw pinned edit, including its alias-maintenance gap during
/// recursive parameter removal. `Some` stages the complete edit, preserves
/// alias owners, repairs an existing schedule, and verifies before committing.
/// No verification policy is chosen implicitly and no schedule is invented.
/// Actual graph changes invalidate optional logical ordering permutations in
/// their containing computation; this does not preserve XLA's historical order.
pub fn replace_instruction(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    replacement: Value,
    options: ReplaceInstructionOptions,
    verification: Option<VerifyOptions>,
) -> Result<InstructionReplacement> {
    if let Some(verification) = verification {
        return verified_edit(
            ctx,
            instruction,
            Edit::Replace(replacement, options),
            verification,
        );
    }
    replace_instruction_impl(ctx, instruction, replacement, options)
}

fn replace_instruction_impl(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    replacement: Value,
    options: ReplaceInstructionOptions,
) -> Result<InstructionReplacement> {
    let original = result(ctx, instruction)?;
    if original == replacement {
        return Ok(InstructionReplacement::default());
    }
    let block = endpoint_block(ctx, original)?;
    require(
        endpoint_block(ctx, replacement)? == block,
        "replacement is outside the instruction body",
    )?;
    require(
        Shape::compatible(ctx, original.get_type(ctx), replacement.get_type(ctx)),
        "replacement instruction shapes are incompatible",
    )?;
    let mut controls = ControlGraph::read(ctx, block)?;
    let old_node = controls.index(original)?;
    let replacement_node = controls.index(replacement)?;
    let old_annotations = Annotations::read(ctx, original)?;
    let mut annotations = Annotations::read(ctx, replacement)?;
    if options.preserve_sharding
        && let (Some(old), Some(new)) = (&old_annotations.sharding, &annotations.sharding)
        && !old.equal_semantics(new)?
    {
        return Ok(InstructionReplacement::default());
    }
    if !options.relay_control_dependency && controls.has_controls(old_node) {
        return Ok(InstructionReplacement::default());
    }
    let module = containing_module(ctx, block)?;
    let mut analyses = AnalysisManager::default();
    let effects = SideEffects::compute(module, ctx, &mut analyses)?;
    let operands: Vec<_> = instruction.deref(ctx).operands().collect();
    if let Some(operation) = replacement.defining_op() {
        require(
            !depends_on(ctx, operation, instruction)?,
            "replacement depends on the removed instruction",
        )?;
    }
    require(
        opcode(ctx, instruction)? != Opcode::Parameter,
        "parameter replacement requires caller analysis",
    )?;
    let old_metadata = old_annotations.metadata.clone().unwrap_or_default();
    let new_metadata = annotations.metadata.clone().unwrap_or_default();
    if new_metadata.op_name.is_empty() && !old_metadata.op_name.is_empty() {
        annotations.metadata = Some(old_metadata);
    } else if matches!(new_metadata.metadata_payload, Payload::Absent)
        && !matches!(old_metadata.metadata_payload, Payload::Absent)
    {
        let mut metadata = new_metadata;
        metadata.metadata_payload = old_metadata.metadata_payload;
        annotations.metadata = Some(metadata);
    }
    if options.preserve_frontend_attributes
        && annotations.frontend.0.is_empty()
        && !operands.contains(&replacement)
    {
        annotations.frontend = old_annotations.frontend;
    }
    if let Some(provenance) = old_annotations.original_value {
        annotations.original_value = Some(provenance);
    }
    if annotations.sharding.is_none() {
        annotations.sharding = old_annotations.sharding;
    }
    annotations.validate(ctx, replacement)?;
    if options.relay_control_dependency {
        controls.relay(old_node, replacement_node);
        controls.ensure_names(ctx, module)?;
    }
    // Rename before deleting, updating references in schedules and controls.
    // TupleSimplifier normally replaces different opcodes; shared users of this
    // primitive also need the same-opcode identity rule.
    let replacement_opcode = replacement
        .defining_op()
        .map(|op| opcode(ctx, op))
        .transpose()?
        .unwrap_or(Opcode::Parameter);
    let preserve_name = opcode(ctx, instruction)? == replacement_opcode
        && (replacement_opcode != Opcode::CustomCall
            || same_custom_target(ctx, instruction, replacement.defining_op())?);
    if preserve_name {
        drop(optional_attr::<Schedule>(ctx, module, "hlo_schedule")?);
    }
    let order_owner = order_owner(ctx, block)?;
    invalidate_order(ctx, order_owner);
    if preserve_name && let Some(name) = controls.nodes[old_node].name.take() {
        let previous = controls.nodes[replacement_node].name.replace(name.clone());
        if let Some(previous) = previous {
            rename_schedule(ctx, module, &previous, &name)?;
        }
    }
    controls.write(ctx)?;
    annotations.write(ctx, replacement)?;
    let users: HashSet<_> = original
        .uses(ctx)
        .iter()
        .map(|use_| use_.user_op())
        .collect();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    rewriter.get_config_mut().set_name_on_value_replacement = false;
    rewriter.replace_value_uses_with(ctx, original, replacement);
    let mut cleanup = Cleanup::new(module, &effects);
    if options.remove_unused_operands {
        cleanup.queue.push_back(original);
        cleanup.initial = Some(instruction);
        cleanup.ignore_controls = options.relay_control_dependency;
        cleanup.drain(ctx, &mut rewriter)?;
    } else {
        require(!original.is_used(ctx), "replaced instruction is still used")?;
        cleanup.record_erased_tree(ctx, instruction);
        rewriter.erase_operation(ctx, instruction);
    }
    for user in users {
        if !cleanup.removed.contains(&user) && Operation::is_op::<FusionOp>(user, ctx) {
            deduplicate_fusion_operands(ctx, user, &mut rewriter, &mut cleanup)?;
        }
    }
    cleanup.drain(ctx, &mut rewriter)?;
    Ok(InstructionReplacement {
        changed: true,
        removed: cleanup.removed,
    })
}

/// Remove a dead instruction and recursively dead pure operands.
///
/// Uses the pinned default caller-analysis policy: ordinary computation
/// parameters are retained; unused fusion parameters remove matching caller
/// operands. The initial instruction may be effectful if the caller explicitly
/// requests its removal, but recursively reached effectful producers are kept.
/// Ignored control dependencies are spliced predecessor-to-successor before
/// deletion, rather than silently discarded. `None` is raw pinned removal;
/// `Some` validates an isolated final edit plan before mutating the source.
pub fn remove_instruction_and_unused_operands(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    ignore_control_dependencies: bool,
    verification: Option<VerifyOptions>,
) -> Result<HashSet<Ptr<Operation>>> {
    if let Some(verification) = verification {
        return verified_edit(
            ctx,
            instruction,
            Edit::Remove(ignore_control_dependencies),
            verification,
        )
        .map(|edit| edit.removed);
    }
    remove_instruction_impl(ctx, instruction, ignore_control_dependencies)
}

fn remove_instruction_impl(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    ignore_control_dependencies: bool,
) -> Result<HashSet<Ptr<Operation>>> {
    let value = result(ctx, instruction)?;
    let block = endpoint_block(ctx, value)?;
    require(!value.is_used(ctx), "cannot remove a live instruction")?;
    let graph = ControlGraph::read(ctx, block)?;
    require(
        ignore_control_dependencies || !graph.has_controls(graph.index(value)?),
        "cannot remove instruction with control dependencies",
    )?;
    require(
        opcode(ctx, instruction)? != Opcode::Parameter,
        "ordinary parameters require caller analysis",
    )?;
    let module = containing_module(ctx, block)?;
    let effects = SideEffects::compute(module, ctx, &mut AnalysisManager::default())?;
    let mut cleanup = Cleanup::new(module, &effects);
    cleanup.initial = Some(instruction);
    cleanup.ignore_controls = ignore_control_dependencies;
    cleanup.queue.push_back(value);
    cleanup.drain(ctx, &mut IRRewriter::<DummyListener>::default())?;
    Ok(cleanup.removed)
}

#[derive(Clone, Copy)]
enum Edit {
    Replace(Value, ReplaceInstructionOptions),
    Remove(bool),
}

struct OperationEdit {
    operation: Ptr<Operation>,
    operands: Vec<Value>,
    attributes: AttributeDict,
}

struct EditPlan {
    operations: Vec<OperationEdit>,
    arguments: Vec<(Ptr<BasicBlock>, Vec<usize>, AttributeDict)>,
    erase: Vec<Ptr<Operation>>,
    outcome: InstructionReplacement,
}

impl EditPlan {
    // All indices, uses, ownership and the final verifier policy were checked
    // against the isolated graph. Commit creates no IR and invokes no fallible
    // analysis, validation, naming or alias-repair helper.
    fn commit(self, ctx: &mut Context) -> InstructionReplacement {
        for edit in self.operations {
            for (index, value) in edit.operands.iter().copied().enumerate() {
                if edit.operation.deref(ctx).get_operand(index) != value {
                    Operation::replace_operand(edit.operation, ctx, index, value);
                }
            }
            let old_count = edit.operation.deref(ctx).get_num_operands();
            for index in (edit.operands.len()..old_count).rev() {
                Operation::remove_operand(edit.operation, ctx, index);
            }
            edit.operation.deref_mut(ctx).attributes = edit.attributes;
        }
        for operation in self.erase {
            Operation::erase(operation, ctx);
        }
        for (block, indices, attributes) in self.arguments {
            for index in indices.into_iter().rev() {
                BasicBlock::remove_argument(block, ctx, index);
            }
            block.deref_mut(ctx).attributes = attributes;
        }
        self.outcome
    }
}

fn operation_tree(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut operations = vec![root];
    let mut index = 0;
    while index < operations.len() {
        let operation = operations[index];
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                operations.extend(block.deref(ctx).iter(ctx));
            }
        }
        index += 1;
    }
    operations
}

struct AliasOwners {
    fusion: Ptr<Operation>,
    aliases: OutputOperandAliases,
    owners: Vec<Value>,
}

fn alias_owners(ctx: &Context, operations: &[Ptr<Operation>]) -> Result<Vec<AliasOwners>> {
    let mut snapshots = Vec::new();
    for &fusion in operations {
        if !Operation::is_op::<FusionOp>(fusion, ctx) {
            continue;
        }
        let Some(aliases) = optional_attr::<OutputOperandAliases>(ctx, fusion, "hlo_aliases")?
            .as_deref()
            .cloned()
        else {
            continue;
        };
        let owners = aliases
            .0
            .iter()
            .map(|alias| {
                let index = usize::try_from(alias.operand_index)
                    .map_err(|_| error("negative fusion alias operand"))?;
                require(
                    index < fusion.deref(ctx).get_num_operands(),
                    "fusion alias operand is outside caller",
                )?;
                Ok(fusion.deref(ctx).get_operand(index))
            })
            .collect::<Result<Vec<_>>>()?;
        snapshots.push(AliasOwners {
            fusion,
            aliases,
            owners,
        });
    }
    Ok(snapshots)
}

fn restore_alias_owners(
    ctx: &Context,
    snapshots: Vec<AliasOwners>,
    removed: &HashSet<Ptr<Operation>>,
) -> Result<()> {
    for mut snapshot in snapshots {
        if removed.contains(&snapshot.fusion) {
            continue;
        }
        for (alias, owner) in snapshot.aliases.0.iter_mut().zip(snapshot.owners) {
            let index = snapshot
                .fusion
                .deref(ctx)
                .operands()
                .position(|operand| operand == owner)
                .ok_or_else(|| error("verified edit would remove a fusion alias owner"))?;
            alias.operand_index =
                i64::try_from(index).map_err(|_| error("fusion alias operand index overflow"))?;
        }
        snapshot
            .fusion
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_aliases"), snapshot.aliases);
    }
    Ok(())
}

fn verified_edit(
    ctx: &mut Context,
    instruction: Ptr<Operation>,
    edit: Edit,
    verification: VerifyOptions,
) -> Result<InstructionReplacement> {
    let block = endpoint_block(ctx, result(ctx, instruction)?)?;
    let module = containing_module(ctx, block)?;
    verify_with_options(ctx, module, verification)?;
    let original = operation_tree(ctx, module);
    let mut mapping = IrMapping::new();
    let scratch = clone_operation(
        module,
        ctx,
        &mut IRRewriter::<DummyListener>::default(),
        &mut mapping,
    );
    let plan = (|| {
        let mut reverse_values = HashMap::new();
        let mut reverse_operations = HashMap::new();
        let mut blocks = Vec::new();
        for &operation in &original {
            let cloned = mapping
                .lookup_op(operation)
                .ok_or_else(|| error("edit staging omitted an operation"))?;
            reverse_operations.insert(cloned, operation);
            for index in 0..operation.deref(ctx).get_num_results() {
                let value = operation.deref(ctx).get_result(index);
                let cloned = mapping
                    .lookup_value(value)
                    .ok_or_else(|| error("edit staging omitted a result"))?;
                reverse_values.insert(cloned, value);
            }
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    let cloned = mapping
                        .lookup_block(block)
                        .ok_or_else(|| error("edit staging omitted a block"))?;
                    let mut arguments = Vec::new();
                    for value in block.deref(ctx).arguments() {
                        let copy = mapping
                            .lookup_value(value)
                            .ok_or_else(|| error("edit staging omitted an argument"))?;
                        reverse_values.insert(copy, value);
                        arguments.push(copy);
                    }
                    blocks.push((operation, block, cloned, arguments));
                }
            }
        }
        let aliases = alias_owners(ctx, &operation_tree(ctx, scratch))?;
        let staged_instruction = mapping
            .lookup_op(instruction)
            .ok_or_else(|| error("edit instruction is outside its module"))?;
        let outcome = match edit {
            Edit::Replace(value, options) => {
                let value = mapping
                    .lookup_value(value)
                    .ok_or_else(|| error("replacement is outside its module"))?;
                replace_instruction_impl(ctx, staged_instruction, value, options)?
            }
            Edit::Remove(ignore_controls) => InstructionReplacement {
                changed: true,
                removed: remove_instruction_impl(ctx, staged_instruction, ignore_controls)?,
            },
        };
        if !outcome.changed {
            return Ok(EditPlan {
                operations: Vec::new(),
                arguments: Vec::new(),
                erase: Vec::new(),
                outcome,
            });
        }
        restore_alias_owners(ctx, aliases, &outcome.removed)?;
        super::schedule_update::update_schedule(ctx, scratch)?;
        verify_with_options(ctx, scratch, verification)?;
        require(
            operation_tree(ctx, scratch)
                .iter()
                .all(|operation| reverse_operations.contains_key(operation)),
            "edit created an unplanned operation",
        )?;
        let removed = outcome
            .removed
            .iter()
            .map(|operation| {
                reverse_operations
                    .get(operation)
                    .copied()
                    .ok_or_else(|| error("edit created an unplanned operation"))
            })
            .collect::<Result<HashSet<_>>>()?;
        let mut operations = Vec::new();
        for &operation in &original {
            if removed.contains(&operation) {
                continue;
            }
            let cloned = mapping
                .lookup_op(operation)
                .ok_or_else(|| error("edit lost a surviving operation"))?;
            let operands = cloned
                .deref(ctx)
                .operands()
                .map(|value| {
                    reverse_values
                        .get(&value)
                        .copied()
                        .ok_or_else(|| error("edit created an unplanned value"))
                })
                .collect::<Result<Vec<_>>>()?;
            require(
                operands.len() <= operation.deref(ctx).get_num_operands(),
                "edit unexpectedly added operands",
            )?;
            operations.push(OperationEdit {
                operation,
                operands,
                attributes: cloned.deref(ctx).attributes.clone(),
            });
        }
        let mut arguments = Vec::new();
        for (owner, block, cloned, before) in blocks {
            if removed.contains(&owner) {
                continue;
            }
            let after: HashSet<_> = cloned.deref(ctx).arguments().collect();
            let indices = before
                .iter()
                .enumerate()
                .filter_map(|(index, value)| (!after.contains(value)).then_some(index))
                .collect();
            arguments.push((block, indices, cloned.deref(ctx).attributes.clone()));
        }
        let mut erase = Vec::new();
        for &operation in original.iter().rev() {
            if !removed.contains(&operation) {
                continue;
            }
            let mut parent = operation
                .deref(ctx)
                .get_parent_block()
                .and_then(|block| block.deref(ctx).get_parent_op(ctx));
            let mut nested = false;
            while let Some(owner) = parent {
                if removed.contains(&owner) {
                    nested = true;
                    break;
                }
                parent = owner
                    .deref(ctx)
                    .get_parent_block()
                    .and_then(|block| block.deref(ctx).get_parent_op(ctx));
            }
            if !nested {
                erase.push(operation);
            }
        }
        Ok(EditPlan {
            operations,
            arguments,
            erase,
            outcome: InstructionReplacement {
                changed: outcome.changed,
                removed,
            },
        })
    })();
    Operation::erase(scratch, ctx);
    plan.map(|plan| plan.commit(ctx))
}

fn result(ctx: &Context, operation: Ptr<Operation>) -> Result<Value> {
    opcode(ctx, operation)?;
    let ir = operation.deref(ctx);
    require(
        ir.get_num_results() == 1,
        "instruction replacement requires exactly one result",
    )?;
    Ok(ir.get_result(0))
}

fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref())
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| error("instruction edit requires native HLO"))
}

fn endpoint_block(ctx: &Context, value: Value) -> Result<Ptr<BasicBlock>> {
    let block = match value.defining_entity() {
        DefiningEntity::Op(operation) => operation
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("instruction endpoint is detached"))?,
        DefiningEntity::Block(block) => block,
    };
    let parent = block
        .deref(ctx)
        .get_parent_op(ctx)
        .ok_or_else(|| error("instruction body is detached"))?;
    require(
        Operation::is_op::<ComputationOp>(parent, ctx) || Operation::is_op::<FusionOp>(parent, ctx),
        "instruction edit requires computation or fusion body",
    )?;
    Ok(block)
}

fn containing_module(ctx: &Context, mut block: Ptr<BasicBlock>) -> Result<Ptr<Operation>> {
    loop {
        let parent = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("instruction body is detached"))?;
        if Operation::is_op::<ModuleOp>(parent, ctx) {
            return Ok(parent);
        }
        block = parent
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("instruction has no containing module"))?;
    }
}

fn order_owner(ctx: &Context, mut block: Ptr<BasicBlock>) -> Result<Ptr<Operation>> {
    loop {
        let owner = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("ordering body is detached"))?;
        if Operation::is_op::<ComputationOp>(owner, ctx) {
            return Ok(owner);
        }
        require(
            Operation::is_op::<FusionOp>(owner, ctx),
            "invalid ordering body owner",
        )?;
        block = owner
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("ordering body has no computation"))?;
    }
}

fn invalidate_order(ctx: &Context, owner: Ptr<Operation>) {
    owner
        .deref_mut(ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_instruction_order"));
}

fn depends_on(ctx: &Context, operation: Ptr<Operation>, needle: Ptr<Operation>) -> Result<bool> {
    let mut pending = vec![operation];
    let mut seen = HashSet::new();
    while let Some(operation) = pending.pop() {
        if operation == needle {
            return Ok(true);
        }
        if !seen.insert(operation) {
            continue;
        }
        opcode(ctx, operation)?;
        pending.extend(
            operation
                .deref(ctx)
                .operands()
                .filter_map(|operand| operand.defining_op()),
        );
    }
    Ok(false)
}

#[derive(Clone)]
struct Annotations {
    metadata: Option<OpMetadata>,
    frontend: FrontendAttributes,
    sharding: Option<Sharding>,
    original_value: Option<OriginalValue>,
}

impl Annotations {
    fn read(ctx: &Context, value: Value) -> Result<Self> {
        let annotations = match value.defining_entity() {
            DefiningEntity::Op(operation) => Self {
                metadata: optional_attr::<OpMetadata>(ctx, operation, "hlo_metadata")?
                    .as_deref()
                    .cloned(),
                frontend: optional_attr::<FrontendAttributes>(
                    ctx,
                    operation,
                    "hlo_frontend_attributes",
                )?
                .as_deref()
                .cloned()
                .unwrap_or_default(),
                sharding: optional_attr::<Sharding>(ctx, operation, "hlo_sharding")?
                    .as_deref()
                    .cloned(),
                original_value: optional_attr::<OriginalValue>(
                    ctx,
                    operation,
                    "hlo_original_value",
                )?
                .as_deref()
                .cloned(),
            },
            DefiningEntity::Block(block) => {
                let parameters = fusion_parameters(ctx, block)?;
                let index = argument_index(ctx, block, value)?;
                let parameter = &parameters.0[index];
                Self {
                    metadata: parameter.metadata.clone(),
                    frontend: parameter.frontend_attributes.clone(),
                    sharding: parameter.sharding.clone(),
                    original_value: parameter.original_value.clone(),
                }
            }
        };
        annotations.validate(ctx, value)?;
        Ok(annotations)
    }

    fn validate(&self, ctx: &Context, value: Value) -> Result<()> {
        self.frontend.verify(ctx)?;
        if let Some(metadata) = &self.metadata {
            metadata.verify(ctx)?;
        }
        if let Some(sharding) = &self.sharding {
            sharding.validate(ctx, value.get_type(ctx), None)?;
        }
        if let Some(provenance) = &self.original_value {
            provenance.verify_shape(ctx, value.get_type(ctx))?;
        }
        Ok(())
    }

    fn write(self, ctx: &Context, value: Value) -> Result<()> {
        match value.defining_entity() {
            DefiningEntity::Op(operation) => {
                let mut ir = operation.deref_mut(ctx);
                if let Some(metadata) = self.metadata {
                    ir.attributes.set(Identifier::new("hlo_metadata"), metadata);
                }
                if !self.frontend.0.is_empty() {
                    ir.attributes
                        .set(Identifier::new("hlo_frontend_attributes"), self.frontend);
                }
                if let Some(sharding) = self.sharding {
                    ir.attributes.set(Identifier::new("hlo_sharding"), sharding);
                }
                if let Some(original) = self.original_value {
                    ir.attributes
                        .set(Identifier::new("hlo_original_value"), original);
                }
            }
            DefiningEntity::Block(block) => {
                let mut parameters = fusion_parameters(ctx, block)?;
                let index = argument_index(ctx, block, value)?;
                let parameter = &mut parameters.0[index];
                parameter.metadata = self.metadata;
                parameter.frontend_attributes = self.frontend;
                parameter.sharding = self.sharding;
                parameter.original_value = self.original_value;
                set_fusion_parameters(ctx, block, parameters)?;
            }
        }
        Ok(())
    }
}

fn argument_index(ctx: &Context, block: Ptr<BasicBlock>, value: Value) -> Result<usize> {
    (0..block.deref(ctx).get_num_arguments())
        .find(|&index| block.deref(ctx).get_argument(index) == value)
        .ok_or_else(|| error("fusion argument does not belong to its body"))
}

struct ControlNode {
    value: Value,
    name: Option<String>,
    incoming: Vec<usize>,
}

struct ControlGraph {
    block: Ptr<BasicBlock>,
    nodes: Vec<ControlNode>,
}

impl ControlGraph {
    fn read(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Self> {
        let parent = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("control body is detached"))?;
        let mut raw = Vec::new();
        if Operation::is_op::<FusionOp>(parent, ctx) {
            for (index, parameter) in fusion_parameters(ctx, block)?.0.into_iter().enumerate() {
                raw.push((
                    block.deref(ctx).get_argument(index),
                    parameter.name,
                    parameter.control_predecessors,
                ));
            }
        } else {
            require(
                Operation::is_op::<ComputationOp>(parent, ctx),
                "control body must be HLO",
            )?;
        }
        for instruction in block.deref(ctx).iter(ctx) {
            if Operation::is_op::<ReturnOp>(instruction, ctx) {
                continue;
            }
            raw.push((
                result(ctx, instruction)?,
                optional_attr::<InstructionName>(ctx, instruction, "hlo_name")?
                    .map(|name| name.0.clone()),
                optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
                    .map(|controls| controls.0.clone())
                    .unwrap_or_default(),
            ));
        }
        let mut names = HashMap::new();
        for (index, (_, name, _)) in raw.iter().enumerate() {
            if let Some(name) = name {
                require(
                    !name.is_empty() && names.insert(name.clone(), index).is_none(),
                    "invalid or duplicate control endpoint name",
                )?;
            }
        }
        let nodes = raw
            .into_iter()
            .map(|(value, name, controls)| {
                let incoming = controls
                    .into_iter()
                    .map(|name| {
                        names
                            .get(&name)
                            .copied()
                            .ok_or_else(|| error("unknown control predecessor"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ControlNode {
                    value,
                    name,
                    incoming,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { block, nodes })
    }

    fn index(&self, value: Value) -> Result<usize> {
        self.nodes
            .iter()
            .position(|node| node.value == value)
            .ok_or_else(|| error("control endpoint is outside its body"))
    }

    fn has_controls(&self, index: usize) -> bool {
        !self.nodes[index].incoming.is_empty()
            || self.nodes.iter().any(|node| node.incoming.contains(&index))
    }

    fn relay(&mut self, from: usize, to: usize) {
        let incoming = self.nodes[from].incoming.clone();
        for predecessor in incoming {
            if !self.nodes[to].incoming.contains(&predecessor) {
                self.nodes[to].incoming.push(predecessor);
            }
        }
        for node in &mut self.nodes {
            if node.incoming.contains(&from) && !node.incoming.contains(&to) {
                node.incoming.push(to);
            }
        }
        self.drop_edges(from);
    }

    fn splice(&mut self, index: usize) {
        let incoming = self.nodes[index].incoming.clone();
        for node in &mut self.nodes {
            if node.incoming.contains(&index) {
                for &predecessor in &incoming {
                    if !node.incoming.contains(&predecessor) {
                        node.incoming.push(predecessor);
                    }
                }
            }
        }
        self.drop_edges(index);
    }

    fn drop_edges(&mut self, index: usize) {
        for node in &mut self.nodes {
            node.incoming.retain(|&predecessor| predecessor != index);
        }
        self.nodes[index].incoming.clear();
    }

    fn ensure_names(&mut self, ctx: &Context, module: Ptr<Operation>) -> Result<()> {
        let mut occupied = HashSet::new();
        let mut pending = vec![module];
        while let Some(operation) = pending.pop() {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                occupied.insert(name.0.clone());
            }
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    if Operation::is_op::<FusionOp>(operation, ctx) {
                        occupied.extend(
                            fusion_parameters(ctx, block)?
                                .0
                                .into_iter()
                                .filter_map(|parameter| parameter.name),
                        );
                    }
                    pending.extend(block.deref(ctx).iter(ctx));
                }
            }
        }
        for index in 0..self.nodes.len() {
            if self.nodes[index].name.is_some()
                || !self.nodes.iter().any(|node| node.incoming.contains(&index))
            {
                continue;
            }
            let mut suffix = 0;
            let name = loop {
                let name = format!("control_relay.{suffix}");
                if occupied.insert(name.clone()) {
                    break name;
                }
                suffix += 1;
            };
            self.nodes[index].name = Some(name);
        }
        Ok(())
    }

    fn write(&self, ctx: &Context) -> Result<()> {
        let parent = self
            .block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("control body detached"))?;
        let mut parameters = if Operation::is_op::<FusionOp>(parent, ctx) {
            Some(fusion_parameters(ctx, self.block)?)
        } else {
            None
        };
        for node in &self.nodes {
            let controls = node
                .incoming
                .iter()
                .map(|&index| {
                    self.nodes[index]
                        .name
                        .clone()
                        .ok_or_else(|| error("control predecessor has no name"))
                })
                .collect::<Result<Vec<_>>>()?;
            match node.value.defining_entity() {
                DefiningEntity::Op(operation) => {
                    let mut ir = operation.deref_mut(ctx);
                    if let Some(name) = &node.name {
                        ir.attributes
                            .set(Identifier::new("hlo_name"), InstructionName(name.clone()));
                    } else {
                        ir.attributes.0.remove(&Identifier::new("hlo_name"));
                    }
                    let key = Identifier::new("hlo_control_predecessors");
                    if !controls.is_empty() || ir.attributes.0.contains_key(&key) {
                        ir.attributes.set(key, ControlPredecessors(controls));
                    }
                }
                DefiningEntity::Block(block) => {
                    let index = argument_index(ctx, block, node.value)?;
                    let parameter = parameters
                        .as_mut()
                        .ok_or_else(|| error("ordinary block argument is not an instruction"))?
                        .0
                        .get_mut(index)
                        .ok_or_else(|| error("missing fusion parameter"))?;
                    parameter.name = node.name.clone();
                    parameter.control_predecessors = controls;
                }
            }
        }
        if let Some(parameters) = parameters {
            set_fusion_parameters(ctx, self.block, parameters)?;
        }
        Ok(())
    }
}

fn same_custom_target(
    ctx: &Context,
    old: Ptr<Operation>,
    new: Option<Ptr<Operation>>,
) -> Result<bool> {
    let Some(new) = new else {
        return Ok(false);
    };
    Ok(
        optional_attr::<pliron::builtin::attributes::StringAttr>(ctx, old, "hlo_target")?
            .as_deref()
            == optional_attr::<pliron::builtin::attributes::StringAttr>(ctx, new, "hlo_target")?
                .as_deref(),
    )
}

fn rename_schedule(ctx: &Context, module: Ptr<Operation>, old: &str, new: &str) -> Result<()> {
    let schedule = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .as_deref()
        .cloned();
    if let Some(mut schedule) = schedule {
        for sequence in &mut schedule.0 {
            sequence.instructions.retain(|name| name != new);
            for name in &mut sequence.instructions {
                if name == old {
                    *name = new.to_owned();
                }
            }
        }
        module
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_schedule"), schedule);
    }
    Ok(())
}

struct Cleanup<'a> {
    module: Ptr<Operation>,
    effects: &'a SideEffects,
    removed: HashSet<Ptr<Operation>>,
    queue: VecDeque<Value>,
    initial: Option<Ptr<Operation>>,
    ignore_controls: bool,
}

impl<'a> Cleanup<'a> {
    fn new(module: Ptr<Operation>, effects: &'a SideEffects) -> Self {
        Self {
            module,
            effects,
            removed: HashSet::new(),
            queue: VecDeque::new(),
            initial: None,
            ignore_controls: false,
        }
    }

    fn record_erased_tree(&mut self, ctx: &Context, operation: Ptr<Operation>) {
        let mut pending = vec![operation];
        while let Some(operation) = pending.pop() {
            if !self.removed.insert(operation) {
                continue;
            }
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    pending.extend(block.deref(ctx).iter(ctx));
                }
            }
        }
    }

    fn drain(&mut self, ctx: &mut Context, rewriter: &mut IRRewriter<DummyListener>) -> Result<()> {
        loop {
            let mut arguments: HashMap<Ptr<BasicBlock>, BTreeSet<usize>> = HashMap::new();
            while let Some(value) = self.queue.pop_front() {
                if let Some(operation) = value.defining_op() {
                    if self.removed.contains(&operation) {
                        continue;
                    }
                    if value.is_used(ctx) || opcode(ctx, operation)? == Opcode::Parameter {
                        continue;
                    }
                    if Some(operation) != self.initial && self.effects.has_side_effect(operation)? {
                        continue;
                    }
                    if Some(operation) == self.initial
                        && opcode(ctx, operation)? == Opcode::AfterAll
                        && after_all_depends_on_effect(ctx, operation, self.effects)?
                    {
                        continue;
                    }
                    let block = endpoint_block(ctx, value)?;
                    let mut graph = ControlGraph::read(ctx, block)?;
                    let index = graph.index(value)?;
                    if !self.ignore_controls && graph.has_controls(index) {
                        continue;
                    }
                    let order_owner = order_owner(ctx, block)?;
                    if self.ignore_controls {
                        graph.splice(index);
                        graph.ensure_names(ctx, self.module)?;
                        invalidate_order(ctx, order_owner);
                        graph.write(ctx)?;
                    }
                    invalidate_order(ctx, order_owner);
                    self.queue.extend(operation.deref(ctx).operands());
                    self.record_erased_tree(ctx, operation);
                    rewriter.erase_operation(ctx, operation);
                } else if !value.is_used(ctx) {
                    let block = endpoint_block(ctx, value)?;
                    let index = argument_index(ctx, block, value)?;
                    let mut graph = ControlGraph::read(ctx, block)?;
                    let node = graph.index(value)?;
                    if !self.ignore_controls && graph.has_controls(node) {
                        continue;
                    }
                    let order_owner = order_owner(ctx, block)?;
                    if self.ignore_controls {
                        graph.splice(node);
                        graph.ensure_names(ctx, self.module)?;
                        invalidate_order(ctx, order_owner);
                        graph.write(ctx)?;
                    }
                    arguments.entry(block).or_default().insert(index);
                }
            }
            if arguments.is_empty() {
                return Ok(());
            }
            for (block, indices) in arguments {
                let fusion = block
                    .deref(ctx)
                    .get_parent_op(ctx)
                    .ok_or_else(|| error("fusion parameter has no caller"))?;
                for index in indices.into_iter().rev() {
                    let operand = fusion.deref(ctx).get_operand(index);
                    let owner = order_owner(ctx, block)?;
                    invalidate_order(ctx, owner);
                    remove_fusion_parameter(ctx, block, index)?;
                    if !fusion
                        .deref(ctx)
                        .operands()
                        .any(|remaining| remaining == operand)
                    {
                        self.queue.push_back(operand);
                    }
                }
            }
        }
    }
}

fn after_all_depends_on_effect(
    ctx: &Context,
    instruction: Ptr<Operation>,
    effects: &SideEffects,
) -> Result<bool> {
    for operand in instruction.deref(ctx).operands() {
        if let Some(mut operation) = operand.defining_op() {
            if opcode(ctx, operation)? == Opcode::GetTupleElement
                && let Some(producer) = operation.deref(ctx).get_operand(0).defining_op()
            {
                operation = producer;
            }
            if effects.has_side_effect(operation)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn deduplicate_fusion_operands(
    ctx: &mut Context,
    fusion: Ptr<Operation>,
    rewriter: &mut IRRewriter<DummyListener>,
    cleanup: &mut Cleanup<'_>,
) -> Result<()> {
    let fusion_op =
        Operation::get_op::<FusionOp>(fusion, ctx).ok_or_else(|| error("expected fusion"))?;
    if fusion_op
        .get_attr_hlo_fusion_kind(ctx)
        .is_some_and(|kind| *kind == FusionKind::Custom)
    {
        return Ok(());
    }
    let block = fusion_op
        .body(ctx)
        .deref(ctx)
        .get_entry_block()
        .ok_or_else(|| error("fusion has no body"))?;
    let operands: Vec<_> = fusion.deref(ctx).operands().collect();
    let mut first = HashMap::new();
    let mut duplicates = Vec::new();
    for (index, operand) in operands.into_iter().enumerate() {
        if let Some(&kept) = first.get(&operand) {
            let argument = block.deref(ctx).get_argument(index);
            let replacement = block.deref(ctx).get_argument(kept);
            let owner = order_owner(ctx, block)?;
            invalidate_order(ctx, owner);
            rewriter.replace_value_uses_with(ctx, argument, replacement);
            // ReplaceAllUses copies provenance, not metadata or controls.
            let original = Annotations::read(ctx, argument)?.original_value;
            if let Some(original) = original {
                let mut annotations = Annotations::read(ctx, replacement)?;
                annotations.original_value = Some(original);
                annotations.write(ctx, replacement)?;
            }
            duplicates.push((index, kept));
        } else {
            first.insert(operand, index);
        }
    }
    for (index, kept) in duplicates.into_iter().rev() {
        remap_removed_operand_aliases(ctx, fusion, index, kept)?;
        remove_fusion_parameter(ctx, block, index)?;
    }
    // Keep the original cleanup snapshot; deduplication creates no instructions.
    cleanup.drain(ctx, rewriter)
}

fn remap_removed_operand_aliases(
    ctx: &Context,
    fusion: Ptr<Operation>,
    removed: usize,
    replacement: usize,
) -> Result<()> {
    let Some(mut aliases) = optional_attr::<OutputOperandAliases>(ctx, fusion, "hlo_aliases")?
        .as_deref()
        .cloned()
    else {
        return Ok(());
    };
    for alias in &mut aliases.0 {
        let index = usize::try_from(alias.operand_index)
            .map_err(|_| error("negative fusion alias operand"))?;
        if index == removed {
            alias.operand_index = i64::try_from(replacement)
                .map_err(|_| error("fusion alias operand index overflow"))?;
        } else if index > removed {
            alias.operand_index -= 1;
        }
    }
    fusion
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_aliases"), aliases);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Bytes, Index},
        ir::{self, FusionParameter, FusionParameters, InstructionSequence},
        ops::control::{AfterAllOp, CustomCallOp, GetTupleElementOp, ParameterOp, TupleOp},
        ops::elementwise::AddOp,
        ops::movement::CopyOp,
        original_value::{OriginalArray, OriginalValueElement},
        shape::{Dimension, Layout},
        types::ElementType,
    };
    use pliron::{
        builtin::{attributes::StringAttr, types::FunctionType},
        op::Op,
        printable::Printable,
        r#type::TypeHandle,
    };

    struct Fixture {
        module: Ptr<Operation>,
        block: Ptr<BasicBlock>,
        ty: TypeHandle,
        parameters: Vec<Value>,
    }

    fn fixture(ctx: &mut Context, inputs: usize) -> Fixture {
        crate::register(ctx);
        let ty = Shape::array(ctx, ElementType::F32, vec![2, 3]).unwrap();
        fixture_with_type(ctx, inputs, ty)
    }

    fn fixture_with_type(ctx: &mut Context, inputs: usize, ty: TypeHandle) -> Fixture {
        let module = ModuleOp::new(ctx, Identifier::new("replacement")).get_operation();
        ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let signature = FunctionType::get(ctx, vec![ty; inputs], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new("main"), signature);
        computation.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let mut parameters = Vec::new();
        for index in 0..inputs {
            let parameter = ParameterOp::new(ctx, ty, vec![], Index(index as u32));
            parameter.get_operation().insert_at_back(block, ctx);
            name(ctx, parameter.get_operation(), &format!("p{index}"));
            parameters.push(parameter.result(ctx));
        }
        Fixture {
            module,
            block,
            ty,
            parameters,
        }
    }

    fn name(ctx: &Context, operation: Ptr<Operation>, name: &str) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name.into()));
    }

    fn controls(ctx: &Context, operation: Ptr<Operation>, predecessors: &[&str]) {
        operation.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_control_predecessors"),
            ControlPredecessors(predecessors.iter().map(|name| (*name).into()).collect()),
        );
    }

    fn finish(ctx: &mut Context, fixture: &Fixture, value: Value) -> Ptr<Operation> {
        let ret = ReturnOp::new(ctx, value).get_operation();
        ret.insert_at_back(fixture.block, ctx);
        crate::verification::verify(ctx, fixture.module).unwrap();
        ret
    }

    fn copy(
        ctx: &mut Context,
        fixture: &Fixture,
        value: Value,
        instruction_name: &str,
    ) -> Ptr<Operation> {
        let operation = CopyOp::new(ctx, value.get_type(ctx), vec![value]).get_operation();
        operation.insert_at_back(fixture.block, ctx);
        name(ctx, operation, instruction_name);
        operation
    }

    fn provenance(instruction_name: &str) -> OriginalValue {
        OriginalValue {
            elements: vec![OriginalValueElement {
                shape_index: vec![],
                original_array: vec![OriginalArray {
                    instruction_name: instruction_name.into(),
                    shape_index: vec![],
                }],
            }],
            ..Default::default()
        }
    }

    #[test]
    fn defaults_are_the_pinned_replacement_arguments() {
        let options = ReplaceInstructionOptions::default();
        assert!(!options.preserve_sharding);
        assert!(!options.relay_control_dependency);
        assert!(options.remove_unused_operands);
        assert!(options.preserve_frontend_attributes);
    }

    #[test]
    fn compatible_layout_replacement_retains_ordinary_parameters_and_transfers_annotations() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let laid_out = Shape::with_layout(
            ctx,
            f.ty,
            Layout {
                minor_to_major: vec![0, 1],
                ..Default::default()
            },
        )
        .unwrap();
        let old = CopyOp::new(ctx, laid_out, vec![f.parameters[0]]).get_operation();
        old.insert_at_back(f.block, ctx);
        let metadata = OpMetadata {
            op_name: "source".into(),
            ..Default::default()
        };
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_metadata"), metadata.clone());
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::replicated());
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_original_value"), provenance("old"));
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_frontend_attributes"),
            FrontendAttributes::new([("do_not_flow_back".into(), "true".into())]),
        );
        let ret = finish(ctx, &f, result(ctx, old).unwrap());
        let changed = replace_instruction(
            ctx,
            old,
            f.parameters[0],
            ReplaceInstructionOptions {
                preserve_sharding: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert!(changed.changed);
        assert!(changed.removed.contains(&old));
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
        let annotations = Annotations::read(ctx, f.parameters[0]).unwrap();
        assert_eq!(annotations.metadata, Some(metadata));
        assert!(annotations.sharding.is_some());
        assert_eq!(annotations.original_value, Some(provenance("old")));
        assert!(annotations.frontend.0.is_empty());
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn existing_metadata_keeps_name_but_receives_missing_payload_and_provenance() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "discarded_name".into(),
                metadata_payload: Payload::Value(Bytes(vec![1, 2, 3])),
                ..Default::default()
            },
        );
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("from_old"),
        );
        old.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_frontend_attributes"),
            FrontendAttributes::new([("kept".into(), "yes".into())]),
        );
        let destination = f.parameters[1].defining_op().unwrap();
        destination.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "existing".into(),
                source_line: 42,
                ..Default::default()
            },
        );
        finish(ctx, &f, result(ctx, old).unwrap());
        replace_instruction(ctx, old, f.parameters[1], Default::default(), None).unwrap();
        let annotations = Annotations::read(ctx, f.parameters[1]).unwrap();
        let metadata = annotations.metadata.unwrap();
        assert_eq!(metadata.op_name, "existing");
        assert_eq!(metadata.source_line, 42);
        assert_eq!(
            metadata.metadata_payload,
            Payload::Value(Bytes(vec![1, 2, 3]))
        );
        assert_eq!(annotations.frontend.get("kept"), Some("yes"));
        assert_eq!(annotations.original_value, Some(provenance("from_old")));
    }

    #[test]
    fn incompatible_sharding_is_an_unchanged_nonmatch_without_metadata_leaks() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let old = copy(ctx, &f, f.parameters[0], "old");
        old.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::replicated());
        f.parameters[0]
            .defining_op()
            .unwrap()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::maximal(0));
        finish(ctx, &f, result(ctx, old).unwrap());
        let before = f.module.disp(ctx).to_string();
        let outcome = replace_instruction(
            ctx,
            old,
            f.parameters[0],
            ReplaceInstructionOptions {
                preserve_sharding: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert!(!outcome.changed);
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    #[test]
    fn whole_tuple_style_replacement_skips_incoming_and_outgoing_controls() {
        for outgoing in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 2);
            let old = copy(ctx, &f, f.parameters[0], "old");
            let consumer = copy(ctx, &f, f.parameters[1], "consumer");
            if outgoing {
                controls(ctx, consumer, &["old"]);
            } else {
                controls(ctx, old, &["p1"]);
            }
            let add = AddOp::new(
                ctx,
                f.ty,
                vec![result(ctx, old).unwrap(), result(ctx, consumer).unwrap()],
            );
            add.get_operation().insert_at_back(f.block, ctx);
            finish(ctx, &f, add.result(ctx));
            let before = f.module.disp(ctx).to_string();
            assert!(
                !replace_instruction(ctx, old, f.parameters[0], Default::default(), None)
                    .unwrap()
                    .changed
            );
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn relay_moves_controls_and_stable_schedule_defers_old_nodes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        controls(ctx, old, &["p1"]);
        let consumer = copy(ctx, &f, f.parameters[1], "consumer");
        controls(ctx, consumer, &["old"]);
        let add = AddOp::new(
            ctx,
            f.ty,
            vec![result(ctx, old).unwrap(), result(ctx, consumer).unwrap()],
        );
        add.get_operation().insert_at_back(f.block, ctx);
        name(ctx, add.get_operation(), "sum");
        finish(ctx, &f, add.result(ctx));
        f.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("main"),
                instructions: ["p0", "p1", "old", "consumer", "sum"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            }]),
        );
        replace_instruction(
            ctx,
            old,
            f.parameters[0],
            ReplaceInstructionOptions {
                relay_control_dependency: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        super::super::schedule_update::update_schedule(ctx, f.module).unwrap();
        assert_eq!(
            optional_attr::<ControlPredecessors>(ctx, consumer, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec!["p0"]
        );
        let schedule = optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
            .unwrap()
            .unwrap();
        assert_eq!(schedule.0[0].instructions, ["p1", "p0", "consumer", "sum"]);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn recursive_cleanup_removes_pure_operands_but_keeps_effects_and_parameters() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let pure = copy(ctx, &f, f.parameters[0], "pure");
        let effect = CustomCallOp::new(ctx, f.ty, vec![], StringAttr::new("effect".into()));
        effect.get_operation().insert_at_back(f.block, ctx);
        effect.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_has_side_effect"),
            crate::attributes::Bools(vec![true]),
        );
        let tuple_type = Shape::tuple(ctx, vec![f.ty, f.ty]).unwrap();
        let tuple = TupleOp::new(
            ctx,
            tuple_type,
            vec![result(ctx, pure).unwrap(), effect.result(ctx)],
        );
        tuple.get_operation().insert_at_back(f.block, ctx);
        let projection = GetTupleElementOp::new(ctx, f.ty, vec![tuple.result(ctx)], Index(0));
        projection.get_operation().insert_at_back(f.block, ctx);
        let ret = finish(ctx, &f, projection.result(ctx));
        let outcome = replace_instruction(
            ctx,
            projection.get_operation(),
            f.parameters[0],
            Default::default(),
            None,
        )
        .unwrap();
        assert!(outcome.removed.contains(&pure));
        assert!(outcome.removed.contains(&tuple.get_operation()));
        assert!(outcome.removed.contains(&projection.get_operation()));
        assert!(!outcome.removed.contains(&effect.get_operation()));
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn same_opcode_replacement_keeps_destination_schedule_position_under_source_name() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let destination = copy(ctx, &f, f.parameters[0], "destination");
        let between = copy(ctx, &f, f.parameters[1], "between");
        let old = copy(ctx, &f, f.parameters[0], "old");
        let sum = AddOp::new(
            ctx,
            f.ty,
            vec![result(ctx, old).unwrap(), result(ctx, between).unwrap()],
        );
        sum.get_operation().insert_at_back(f.block, ctx);
        name(ctx, sum.get_operation(), "sum");
        finish(ctx, &f, sum.result(ctx));
        f.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("main"),
                instructions: ["p0", "p1", "destination", "between", "old", "sum"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            }]),
        );
        replace_instruction(
            ctx,
            old,
            result(ctx, destination).unwrap(),
            Default::default(),
            None,
        )
        .unwrap();
        super::super::schedule_update::update_schedule(ctx, f.module).unwrap();
        assert_eq!(
            optional_attr::<Schedule>(ctx, f.module, "hlo_schedule")
                .unwrap()
                .unwrap()
                .0[0]
                .instructions,
            ["p0", "p1", "old", "between", "sum"],
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn after_all_effect_guard_applies_to_the_initial_removal_only() {
        for recursive in [false, true] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1);
            let token = Shape::token(ctx);
            let effect = CustomCallOp::new(ctx, token, vec![], StringAttr::new("effect".into()));
            effect.get_operation().insert_at_back(f.block, ctx);
            effect.get_operation().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_has_side_effect"),
                crate::attributes::Bools(vec![true]),
            );
            let after = AfterAllOp::new(ctx, token, vec![effect.result(ctx)]);
            after.get_operation().insert_at_back(f.block, ctx);
            let requested = if recursive {
                let tuple_ty = Shape::tuple(ctx, vec![token]).unwrap();
                let tuple = TupleOp::new(ctx, tuple_ty, vec![after.result(ctx)]);
                tuple.get_operation().insert_at_back(f.block, ctx);
                tuple.get_operation()
            } else {
                after.get_operation()
            };
            finish(ctx, &f, f.parameters[0]);
            let removed =
                remove_instruction_and_unused_operands(ctx, requested, false, None).unwrap();
            assert_eq!(removed.contains(&after.get_operation()), recursive);
            assert!(!removed.contains(&effect.get_operation()));
            crate::verification::verify(ctx, f.module).unwrap();
        }
    }

    #[test]
    fn recursive_cleanup_splices_dead_operand_controls_instead_of_dropping_order() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        controls(ctx, dead, &["p0"]);
        let user = copy(ctx, &f, f.parameters[1], "user");
        controls(ctx, user, &["dead"]);
        finish(ctx, &f, result(ctx, user).unwrap());
        let removed = remove_instruction_and_unused_operands(ctx, dead, true, None).unwrap();
        assert!(removed.contains(&dead));
        assert_eq!(
            optional_attr::<ControlPredecessors>(ctx, user, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec!["p0"]
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn fusion_argument_replacement_preserves_annotations_and_parameter_controls() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let fusion = FusionOp::new(ctx, f.ty, f.parameters.clone(), FusionKind::Loop);
        fusion.get_operation().insert_at_back(f.block, ctx);
        let body = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
        body.insert_at_back(fusion.body(ctx), ctx);
        let a = body.deref(ctx).get_argument(0);
        let b = body.deref(ctx).get_argument(1);
        let tuple_type = Shape::tuple(ctx, vec![f.ty, f.ty]).unwrap();
        let tuple = TupleOp::new(ctx, tuple_type, vec![a, b]);
        tuple.get_operation().insert_at_back(body, ctx);
        let projection = GetTupleElementOp::new(ctx, f.ty, vec![tuple.result(ctx)], Index(0));
        projection.get_operation().insert_at_back(body, ctx);
        name(ctx, projection.get_operation(), "project");
        controls(ctx, projection.get_operation(), &["arg1"]);
        projection.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                op_name: "fusion_source".into(),
                ..Default::default()
            },
        );
        projection
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_sharding"), Sharding::replicated());
        projection.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_original_value"),
            provenance("fusion_source"),
        );
        let sum = AddOp::new(ctx, f.ty, vec![projection.result(ctx), b]);
        sum.get_operation().insert_at_back(body, ctx);
        ReturnOp::new(ctx, sum.result(ctx))
            .get_operation()
            .insert_at_back(body, ctx);
        set_fusion_parameters(
            ctx,
            body,
            FusionParameters(vec![
                FusionParameter {
                    name: Some("arg0".into()),
                    ..Default::default()
                },
                FusionParameter {
                    name: Some("arg1".into()),
                    ..Default::default()
                },
            ]),
        )
        .unwrap();
        finish(ctx, &f, fusion.result(ctx));
        replace_instruction(
            ctx,
            projection.get_operation(),
            a,
            ReplaceInstructionOptions {
                preserve_sharding: true,
                relay_control_dependency: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let parameters = fusion_parameters(ctx, body).unwrap();
        assert_eq!(parameters.0.len(), 2); // arg1 remains a control predecessor.
        assert_eq!(parameters.0[0].control_predecessors, vec!["arg1"]);
        assert_eq!(
            parameters.0[0].metadata.as_ref().unwrap().op_name,
            "fusion_source"
        );
        assert!(parameters.0[0].sharding.is_some());
        assert_eq!(
            parameters.0[0].original_value,
            Some(provenance("fusion_source"))
        );
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn cleanup_removes_unused_fusion_arguments_and_recursively_dead_caller_operands() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let outer_dead = copy(ctx, &f, f.parameters[1], "outer_dead");
        let fusion = FusionOp::new(
            ctx,
            f.ty,
            vec![f.parameters[0], result(ctx, outer_dead).unwrap()],
            FusionKind::Loop,
        );
        fusion.get_operation().insert_at_back(f.block, ctx);
        let body = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
        body.insert_at_back(fusion.body(ctx), ctx);
        let a = body.deref(ctx).get_argument(0);
        let b = body.deref(ctx).get_argument(1);
        let tuple_type = Shape::tuple(ctx, vec![f.ty, f.ty]).unwrap();
        let tuple = TupleOp::new(ctx, tuple_type, vec![a, b]);
        tuple.get_operation().insert_at_back(body, ctx);
        let projection = GetTupleElementOp::new(ctx, f.ty, vec![tuple.result(ctx)], Index(0));
        projection.get_operation().insert_at_back(body, ctx);
        ReturnOp::new(ctx, projection.result(ctx))
            .get_operation()
            .insert_at_back(body, ctx);
        finish(ctx, &f, fusion.result(ctx));
        let outcome =
            replace_instruction(ctx, projection.get_operation(), a, Default::default(), None)
                .unwrap();
        assert!(outcome.removed.contains(&outer_dead));
        assert_eq!(body.deref(ctx).get_num_arguments(), 1);
        assert_eq!(fusion.get_operation().deref(ctx).get_num_operands(), 1);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn replacement_deduplicates_noncustom_fusion_operands() {
        for kind in [FusionKind::Loop, FusionKind::Custom] {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1);
            let old = copy(ctx, &f, f.parameters[0], "old");
            let fusion = FusionOp::new(
                ctx,
                f.ty,
                vec![result(ctx, old).unwrap(), f.parameters[0]],
                kind,
            );
            fusion.get_operation().insert_at_back(f.block, ctx);
            let body = BasicBlock::new(ctx, None, vec![f.ty, f.ty]);
            body.insert_at_back(fusion.body(ctx), ctx);
            let arguments = vec![
                body.deref(ctx).get_argument(0),
                body.deref(ctx).get_argument(1),
            ];
            let add = AddOp::new(ctx, f.ty, arguments);
            add.get_operation().insert_at_back(body, ctx);
            ReturnOp::new(ctx, add.result(ctx))
                .get_operation()
                .insert_at_back(body, ctx);
            fusion.get_operation().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_aliases"),
                OutputOperandAliases(vec![crate::attributes::OutputOperandAliasing {
                    output_shape_index: crate::attributes::Axes(vec![]),
                    operand_index: 1,
                    operand_shape_index: crate::attributes::Axes(vec![]),
                }]),
            );
            finish(ctx, &f, fusion.result(ctx));
            replace_instruction(ctx, old, f.parameters[0], Default::default(), None).unwrap();
            assert_eq!(
                body.deref(ctx).get_num_arguments(),
                if kind == FusionKind::Custom { 2 } else { 1 }
            );
            assert_eq!(
                optional_attr::<OutputOperandAliases>(ctx, fusion.get_operation(), "hlo_aliases",)
                    .unwrap()
                    .unwrap()
                    .0[0]
                    .operand_index,
                if kind == FusionKind::Custom { 1 } else { 0 },
            );
            crate::verification::verify(ctx, f.module).unwrap();
        }
    }

    #[test]
    fn malformed_metadata_cross_body_shape_and_data_cycle_errors_precede_rewrite() {
        for case in 0..3 {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 1);
            let old = copy(ctx, &f, f.parameters[0], "old");
            let later = copy(ctx, &f, result(ctx, old).unwrap(), "later");
            finish(ctx, &f, result(ctx, later).unwrap());
            let replacement = match case {
                0 => {
                    old.deref_mut(ctx).attributes.set(
                        Identifier::new("hlo_metadata"),
                        StringAttr::new("wrong".into()),
                    );
                    f.parameters[0]
                }
                1 => {
                    let different = Shape::array(ctx, ElementType::F32, vec![6]).unwrap();
                    f.parameters[0].set_type(ctx, different);
                    f.parameters[0]
                }
                _ => result(ctx, later).unwrap(),
            };
            let before = f.module.disp(ctx).to_string();
            assert!(replace_instruction(ctx, old, replacement, Default::default(), None).is_err());
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn no_implicit_verification_policy_rejects_or_authorizes_unbounded_shapes() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dynamic = Shape::dynamic_array(
            ctx,
            ElementType::F32,
            vec![Dimension::Unbounded, Dimension::Static(3)],
        )
        .unwrap();
        let old = CopyOp::new(ctx, dynamic, vec![f.parameters[0]]).get_operation();
        old.insert_at_back(f.block, ctx);
        ReturnOp::new(ctx, result(ctx, old).unwrap())
            .get_operation()
            .insert_at_back(f.block, ctx);
        assert!(crate::verification::verify(ctx, f.module).is_err());
        replace_instruction(ctx, old, f.parameters[0], Default::default(), None).unwrap();
        crate::verification::verify(ctx, f.module).unwrap();
    }

    struct AliasFixture {
        outer: Fixture,
        fusion: Ptr<Operation>,
        body: Ptr<BasicBlock>,
        projection: Ptr<Operation>,
        selected: Value,
    }

    fn alias_fixture(ctx: &mut Context, selected_index: usize, inputs: usize) -> AliasFixture {
        let outer = fixture(ctx, inputs);
        let fusion = FusionOp::new(ctx, outer.ty, outer.parameters.clone(), FusionKind::Loop);
        let fusion_ptr = fusion.get_operation();
        fusion_ptr.insert_at_back(outer.block, ctx);
        name(ctx, fusion_ptr, "fusion");
        let body = BasicBlock::new(ctx, None, vec![outer.ty; inputs]);
        body.insert_at_back(fusion.body(ctx), ctx);
        let a = body.deref(ctx).get_argument(0);
        let b = body.deref(ctx).get_argument(1);
        set_fusion_parameters(
            ctx,
            body,
            FusionParameters(
                (0..inputs)
                    .map(|index| FusionParameter {
                        name: Some(format!("arg{index}")),
                        metadata: Some(OpMetadata {
                            op_name: format!("input{index}"),
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .collect(),
            ),
        )
        .unwrap();
        let ty = Shape::tuple(ctx, vec![outer.ty, outer.ty]).unwrap();
        let tuple = TupleOp::new(ctx, ty, vec![a, b]);
        tuple.get_operation().insert_at_back(body, ctx);
        name(ctx, tuple.get_operation(), "tuple");
        let projection = GetTupleElementOp::new(
            ctx,
            outer.ty,
            vec![tuple.result(ctx)],
            Index(selected_index as u32),
        );
        let projection_ptr = projection.get_operation();
        projection_ptr.insert_at_back(body, ctx);
        name(ctx, projection_ptr, "project");
        controls(
            ctx,
            projection_ptr,
            &[if selected_index == 0 { "arg1" } else { "arg0" }],
        );
        let root = if inputs == 3 {
            let arguments = vec![projection.result(ctx), body.deref(ctx).get_argument(2)];
            let add = AddOp::new(ctx, outer.ty, arguments);
            add.get_operation().insert_at_back(body, ctx);
            name(ctx, add.get_operation(), "sum");
            add.result(ctx)
        } else {
            projection.result(ctx)
        };
        ReturnOp::new(ctx, root)
            .get_operation()
            .insert_at_back(body, ctx);
        fusion_ptr.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_aliases"),
            OutputOperandAliases(vec![crate::attributes::OutputOperandAliasing {
                output_shape_index: crate::attributes::Axes(vec![]),
                operand_index: 1,
                operand_shape_index: crate::attributes::Axes(vec![]),
            }]),
        );
        finish(ctx, &outer, fusion.result(ctx));
        outer.module.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("main"),
                instructions: (0..inputs)
                    .map(|index| format!("p{index}"))
                    .chain(std::iter::once("fusion".into()))
                    .collect(),
            }]),
        );
        verify_with_options(ctx, outer.module, VerifyOptions::default()).unwrap();
        AliasFixture {
            outer,
            fusion: fusion_ptr,
            body,
            projection: projection_ptr,
            selected: body.deref(ctx).get_argument(selected_index),
        }
    }

    fn alias_owner(ctx: &Context, fusion: Ptr<Operation>) -> Value {
        let aliases = optional_attr::<OutputOperandAliases>(ctx, fusion, "hlo_aliases")
            .unwrap()
            .unwrap();
        fusion
            .deref(ctx)
            .get_operand(aliases.0[0].operand_index as usize)
    }

    fn forwarding_options() -> ReplaceInstructionOptions {
        ReplaceInstructionOptions {
            preserve_sharding: true,
            relay_control_dependency: true,
            ..Default::default()
        }
    }

    fn rich_order(ctx: &Context, block: Ptr<BasicBlock>) -> Ptr<Operation> {
        use super::super::instruction_order::{self, InstructionOrder, OrderPermutation};
        let owner = order_owner(ctx, block).unwrap();
        let root = owner
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        let value_name = |value: Value| match value.defining_entity() {
            DefiningEntity::Op(operation) => {
                optional_attr::<InstructionName>(ctx, operation, "hlo_name")
                    .unwrap()
                    .unwrap()
                    .0
                    .clone()
            }
            DefiningEntity::Block(block) => fusion_parameters(ctx, block).unwrap().0
                [argument_index(ctx, block, value).unwrap()]
            .name
            .clone()
            .unwrap(),
        };
        let mut order = InstructionOrder {
            instructions: instruction_order::block_instruction_order(ctx, root)
                .unwrap()
                .into_iter()
                .rev()
                .map(|operation| value_name(result(ctx, operation).unwrap()))
                .collect(),
            ..Default::default()
        };
        let mut pending = vec![root];
        while let Some(body) = pending.pop() {
            let values = instruction_order::block_value_order(ctx, body).unwrap();
            if body != root {
                let fusion = body.deref(ctx).get_parent_op(ctx).unwrap();
                order.blocks.push(OrderPermutation {
                    subject: value_name(result(ctx, fusion).unwrap()),
                    members: values
                        .iter()
                        .rev()
                        .map(|&value| value_name(value))
                        .collect(),
                });
            }
            for value in values {
                order.users.push(OrderPermutation {
                    subject: value_name(value),
                    members: instruction_order::users(ctx, value)
                        .unwrap()
                        .into_iter()
                        .rev()
                        .map(&value_name)
                        .collect(),
                });
                order.control_successors.push(OrderPermutation {
                    subject: value_name(value),
                    members: instruction_order::control_successors(ctx, value)
                        .unwrap()
                        .into_iter()
                        .rev()
                        .map(&value_name)
                        .collect(),
                });
                if let Some(operation) = value.defining_op() {
                    for region in operation.deref(ctx).regions() {
                        pending.extend(region.deref(ctx).iter(ctx));
                    }
                }
            }
        }
        instruction_order::set_order(ctx, owner, order).unwrap();
        owner
    }

    fn has_order(ctx: &Context, owner: Ptr<Operation>) -> bool {
        owner
            .deref(ctx)
            .attributes
            .0
            .contains_key(&Identifier::new("hlo_instruction_order"))
    }

    #[test]
    fn actual_raw_and_verified_replacements_invalidate_rich_order_only_on_commit() {
        for verification in [None, Some(VerifyOptions::default())] {
            let ctx = &mut Context::new();
            let f = alias_fixture(ctx, 1, 3);
            let owner = rich_order(ctx, f.body);
            assert!(has_order(ctx, owner));
            replace_instruction(
                ctx,
                f.projection,
                f.selected,
                forwarding_options(),
                verification,
            )
            .unwrap();
            assert!(!has_order(ctx, owner));
            verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
        }
    }

    #[test]
    fn ordering_survives_self_control_sharding_and_malformed_preflight_nonchanges() {
        for (case, verification) in (0..4)
            .flat_map(|case| [None, Some(VerifyOptions::default())].map(|policy| (case, policy)))
        {
            let ctx = &mut Context::new();
            let f = fixture(ctx, 2);
            let old = copy(ctx, &f, f.parameters[0], "old");
            if case == 1 {
                controls(ctx, old, &["p1"]);
            }
            if case == 2 {
                old.deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_sharding"), Sharding::replicated());
                f.parameters[0]
                    .defining_op()
                    .unwrap()
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_sharding"), Sharding::maximal(0));
            }
            finish(ctx, &f, result(ctx, old).unwrap());
            let owner = rich_order(ctx, f.block);
            if case == 3 {
                old.deref_mut(ctx).attributes.set(
                    Identifier::new("hlo_metadata"),
                    StringAttr::new("wrong".into()),
                );
            }
            let before = f.module.disp(ctx).to_string();
            let replacement = if case == 0 {
                result(ctx, old).unwrap()
            } else {
                f.parameters[0]
            };
            let outcome = replace_instruction(
                ctx,
                old,
                replacement,
                ReplaceInstructionOptions {
                    preserve_sharding: true,
                    ..Default::default()
                },
                verification,
            );
            if case == 3 {
                assert!(outcome.is_err());
            } else {
                assert!(!outcome.unwrap().changed);
            }
            assert!(has_order(ctx, owner));
            assert_eq!(f.module.disp(ctx).to_string(), before);
        }
    }

    #[test]
    fn raw_cleanup_invalidation_does_not_clear_unrelated_computation_order() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        finish(ctx, &f, f.parameters[0]);
        let owner = rich_order(ctx, f.block);
        let signature = FunctionType::get(ctx, vec![f.ty], vec![f.ty]).into();
        let other = ComputationOp::new(ctx, Identifier::new("other"), signature);
        let container = owner.deref(ctx).get_parent_block().unwrap();
        other.get_operation().insert_at_back(container, ctx);
        let other_block = BasicBlock::new(ctx, None, vec![]);
        other_block.insert_at_back(other.body(ctx), ctx);
        let input = ParameterOp::new(ctx, f.ty, vec![], Index(0));
        input.get_operation().insert_at_back(other_block, ctx);
        name(ctx, input.get_operation(), "other_input");
        ReturnOp::new(ctx, input.result(ctx))
            .get_operation()
            .insert_at_back(other_block, ctx);
        let other_owner = rich_order(ctx, other_block);
        let before = other.get_operation().disp(ctx).to_string();
        remove_instruction_and_unused_operands(ctx, dead, false, None).unwrap();
        assert!(!has_order(ctx, owner));
        assert!(has_order(ctx, other_owner));
        assert_eq!(other.get_operation().disp(ctx).to_string(), before);
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn raw_deduplication_invalidates_nested_storage_and_relation_permutations() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 0, 2);
        controls(ctx, f.projection, &[]);
        let old = CopyOp::new(ctx, f.outer.ty, vec![f.outer.parameters[0]]).get_operation();
        old.insert_before(ctx, f.fusion);
        name(ctx, old, "duplicate");
        Operation::replace_operand(f.fusion, ctx, 1, result(ctx, old).unwrap());
        let owner = rich_order(ctx, f.body);
        replace_instruction(ctx, old, f.outer.parameters[0], Default::default(), None).unwrap();
        assert!(!has_order(ctx, owner));
        assert_eq!(f.body.deref(ctx).get_num_arguments(), 1);
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 1);
        verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_final_control_cycle_rejection_preserves_rich_order_and_source() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        controls(ctx, f.parameters[1].defining_op().unwrap(), &["p0"]);
        controls(ctx, old, &["p1"]);
        finish(ctx, &f, result(ctx, old).unwrap());
        let owner = rich_order(ctx, f.block);
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx);
        assert!(
            replace_instruction(
                ctx,
                old,
                f.parameters[0],
                forwarding_options(),
                Some(VerifyOptions::default()),
            )
            .is_err()
        );
        assert!(has_order(ctx, owner));
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert!(f.parameters[0].uses(ctx) == uses);
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn protected_after_all_cleanup_noop_preserves_ordering() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let token = Shape::token(ctx);
        let effect = CustomCallOp::new(ctx, token, vec![], StringAttr::new("effect".into()));
        effect.get_operation().insert_at_back(f.block, ctx);
        name(ctx, effect.get_operation(), "effect");
        effect.get_operation().deref_mut(ctx).attributes.set(
            Identifier::new("hlo_has_side_effect"),
            crate::attributes::Bools(vec![true]),
        );
        let after = AfterAllOp::new(ctx, token, vec![effect.result(ctx)]);
        after.get_operation().insert_at_back(f.block, ctx);
        name(ctx, after.get_operation(), "after");
        finish(ctx, &f, f.parameters[0]);
        let owner = rich_order(ctx, f.block);
        let before = f.module.disp(ctx).to_string();
        assert!(
            remove_instruction_and_unused_operands(ctx, after.get_operation(), false, None)
                .unwrap()
                .is_empty()
        );
        assert!(has_order(ctx, owner));
        assert_eq!(f.module.disp(ctx).to_string(), before);
    }

    #[test]
    fn raw_direct_alias_owner_removal_succeeds_but_final_verification_fails() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 0, 2);
        let owner = rich_order(ctx, f.body);
        let edit =
            replace_instruction(ctx, f.projection, f.selected, forwarding_options(), None).unwrap();
        assert!(edit.changed);
        assert!(!has_order(ctx, owner));
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 1);
        assert_eq!(
            optional_attr::<OutputOperandAliases>(ctx, f.fusion, "hlo_aliases")
                .unwrap()
                .unwrap()
                .0[0]
                .operand_index,
            1
        );
        assert!(verify_with_options(ctx, f.outer.module, VerifyOptions::default()).is_err());
    }

    #[test]
    fn raw_shifted_alias_can_silently_rebind_to_a_different_same_shaped_owner() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 1, 3);
        let owner = alias_owner(ctx, f.fusion);
        replace_instruction(ctx, f.projection, f.selected, forwarding_options(), None).unwrap();
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 2);
        assert_ne!(alias_owner(ctx, f.fusion), owner);
        assert_eq!(alias_owner(ctx, f.fusion), f.outer.parameters[2]);
        verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_shifted_alias_preserves_owner_and_surviving_native_handles() {
        for inputs in [2, 3] {
            let ctx = &mut Context::new();
            let f = alias_fixture(ctx, 1, inputs);
            let owner = alias_owner(ctx, f.fusion);
            let edit = replace_instruction(
                ctx,
                f.projection,
                f.selected,
                forwarding_options(),
                Some(VerifyOptions::default()),
            )
            .unwrap();
            assert!(edit.changed);
            assert!(edit.removed.contains(&f.projection));
            assert_eq!(alias_owner(ctx, f.fusion), owner);
            assert_eq!(f.fusion.deref(ctx).get_num_operands(), inputs - 1);
            assert_eq!(f.body.deref(ctx).get_argument(0), f.selected);
            assert_eq!(
                fusion_parameters(ctx, f.body).unwrap().0[0].name.as_deref(),
                Some("arg1")
            );
            assert_eq!(f.fusion.deref(ctx).get_parent_block(), Some(f.outer.block));
            verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
        }
    }

    #[test]
    fn verified_alias_owner_loss_repeatedly_rejects_without_mutation_and_recovers() {
        for _ in 0..2 {
            let ctx = &mut Context::new();
            let f = alias_fixture(ctx, 0, 2);
            let owner = rich_order(ctx, f.body);
            let before = f.outer.module.disp(ctx).to_string();
            let values: Vec<_> = f
                .outer
                .parameters
                .iter()
                .copied()
                .chain(f.body.deref(ctx).arguments())
                .chain(std::iter::once(result(ctx, f.projection).unwrap()))
                .collect();
            let uses: Vec<_> = values.iter().map(|value| value.uses(ctx)).collect();
            for _ in 0..3 {
                let failure = replace_instruction(
                    ctx,
                    f.projection,
                    f.selected,
                    forwarding_options(),
                    Some(VerifyOptions::default()),
                )
                .err()
                .expect("lost alias owner must be rejected");
                assert!(failure.to_string().contains("fusion alias owner"));
                assert_eq!(f.outer.module.disp(ctx).to_string(), before);
                assert!(has_order(ctx, owner));
                assert!(
                    values
                        .iter()
                        .map(|value| value.uses(ctx))
                        .collect::<Vec<_>>()
                        == uses
                );
                assert_eq!(f.projection.deref(ctx).get_parent_block(), Some(f.body));
                verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
            }
            // The caller changes its contract explicitly; the edit never drops it.
            f.fusion
                .deref_mut(ctx)
                .attributes
                .0
                .remove(&Identifier::new("hlo_aliases"));
            replace_instruction(
                ctx,
                f.projection,
                f.selected,
                forwarding_options(),
                Some(VerifyOptions::default()),
            )
            .unwrap();
            verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
        }
    }

    #[test]
    fn verified_removal_commits_without_replacing_surviving_operations() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let dead = copy(ctx, &f, f.parameters[0], "dead");
        let ret = finish(ctx, &f, f.parameters[0]);
        let removed = remove_instruction_and_unused_operands(
            ctx,
            dead,
            false,
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert_eq!(removed, HashSet::from([dead]));
        assert_eq!(ret.deref(ctx).get_parent_block(), Some(f.block));
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
        verify_with_options(ctx, f.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_alias_owner_survives_a_duplicate_caller_occurrence_removal() {
        let ctx = &mut Context::new();
        let f = alias_fixture(ctx, 1, 2);
        Operation::replace_operand(f.fusion, ctx, 1, f.outer.parameters[0]);
        let owner = alias_owner(ctx, f.fusion);
        replace_instruction(
            ctx,
            f.projection,
            f.selected,
            forwarding_options(),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert_eq!(f.fusion.deref(ctx).get_num_operands(), 1);
        assert_eq!(alias_owner(ctx, f.fusion), owner);
        verify_with_options(ctx, f.outer.module, VerifyOptions::default()).unwrap();
    }

    #[test]
    fn verified_no_match_leaves_source_and_uses_unchanged() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let old = copy(ctx, &f, f.parameters[0], "old");
        controls(ctx, old, &["p1"]);
        finish(ctx, &f, result(ctx, old).unwrap());
        let before = f.module.disp(ctx).to_string();
        let uses = f.parameters[0].uses(ctx).len();
        let outcome = replace_instruction(
            ctx,
            old,
            f.parameters[0],
            Default::default(),
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert!(!outcome.changed);
        assert!(outcome.removed.is_empty());
        assert_eq!(f.module.disp(ctx).to_string(), before);
        assert_eq!(f.parameters[0].uses(ctx).len(), uses);
    }

    #[test]
    fn verified_edit_uses_callers_unbounded_policy_without_an_implicit_default() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = Shape::dynamic_array(ctx, ElementType::F32, vec![Dimension::Unbounded]).unwrap();
        let f = fixture_with_type(ctx, 1, ty);
        let pair = Shape::tuple(ctx, vec![ty, ty]).unwrap();
        let tuple = TupleOp::new(ctx, pair, vec![f.parameters[0], f.parameters[0]]);
        tuple.get_operation().insert_at_back(f.block, ctx);
        let projection = GetTupleElementOp::new(ctx, ty, vec![tuple.result(ctx)], Index(1));
        projection.get_operation().insert_at_back(f.block, ctx);
        ReturnOp::new(ctx, projection.result(ctx))
            .get_operation()
            .insert_at_back(f.block, ctx);
        let before = f.module.disp(ctx).to_string();
        assert!(
            replace_instruction(
                ctx,
                projection.get_operation(),
                f.parameters[0],
                Default::default(),
                Some(VerifyOptions::default()),
            )
            .is_err()
        );
        assert_eq!(f.module.disp(ctx).to_string(), before);
        let options = VerifyOptions {
            supported_unbounded_dynamic_op: Some(|ctx, operation| {
                matches!(
                    opcode(ctx, operation),
                    Ok(Opcode::Parameter | Opcode::Tuple | Opcode::GetTupleElement)
                )
            }),
            ..Default::default()
        };
        replace_instruction(
            ctx,
            projection.get_operation(),
            f.parameters[0],
            Default::default(),
            Some(options),
        )
        .unwrap();
        verify_with_options(ctx, f.module, options).unwrap();
        assert_eq!(
            f.block
                .deref(ctx)
                .get_terminator(ctx)
                .unwrap()
                .deref(ctx)
                .get_operand(0),
            f.parameters[0]
        );
    }

    #[test]
    fn disabling_operand_cleanup_retains_dead_producers_and_given_names() {
        use pliron::common_traits::Named;
        let ctx = &mut Context::new();
        let f = fixture(ctx, 2);
        let producer = CopyOp::new(ctx, f.ty, vec![f.parameters[0]]);
        producer.get_operation().insert_at_back(f.block, ctx);
        let old = CopyOp::new(ctx, f.ty, vec![producer.result(ctx)]);
        old.get_operation().insert_at_back(f.block, ctx);
        old.result(ctx)
            .set_name(ctx, Some(Identifier::new("old_ssa")));
        let before_name = f.parameters[1].given_name(ctx);
        let ret = finish(ctx, &f, old.result(ctx));
        let edit = replace_instruction(
            ctx,
            old.get_operation(),
            f.parameters[1],
            ReplaceInstructionOptions {
                remove_unused_operands: false,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(edit.removed, HashSet::from([old.get_operation()]));
        assert_eq!(
            producer.get_operation().deref(ctx).get_parent_block(),
            Some(f.block)
        );
        assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[1]);
        assert_eq!(f.parameters[1].given_name(ctx), before_name);
        crate::verification::verify(ctx, f.module).unwrap();
    }

    #[test]
    fn cross_body_replacement_is_rejected_without_mutating_either_body() {
        let ctx = &mut Context::new();
        let f = fixture(ctx, 1);
        let other = fixture_with_type(ctx, 1, f.ty);
        let old = CopyOp::new(ctx, f.ty, vec![f.parameters[0]]);
        old.get_operation().insert_at_back(f.block, ctx);
        finish(ctx, &f, old.result(ctx));
        finish(ctx, &other, other.parameters[0]);
        let before = [
            f.module.disp(ctx).to_string(),
            other.module.disp(ctx).to_string(),
        ];
        assert!(
            replace_instruction(
                ctx,
                old.get_operation(),
                other.parameters[0],
                Default::default(),
                None,
            )
            .is_err()
        );
        assert_eq!(
            [
                f.module.disp(ctx).to_string(),
                other.module.disp(ctx).to_string()
            ],
            before
        );
    }

    #[test]
    fn tuple_projection_forwarding_preserves_all_element_formats_and_dynamic_geometry() {
        for &element in ElementType::ALL {
            for dimension in [
                Dimension::Static(3),
                Dimension::Bounded(3),
                Dimension::Unbounded,
            ] {
                let ctx = &mut Context::new();
                crate::register(ctx);
                let ty = Shape::dynamic_array(ctx, element, vec![dimension]).unwrap();
                let f = fixture_with_type(ctx, 1, ty);
                let pair = Shape::tuple(ctx, vec![ty, ty]).unwrap();
                let tuple = TupleOp::new(ctx, pair, vec![f.parameters[0], f.parameters[0]]);
                tuple.get_operation().insert_at_back(f.block, ctx);
                let projection = GetTupleElementOp::new(ctx, ty, vec![tuple.result(ctx)], Index(1));
                projection.get_operation().insert_at_back(f.block, ctx);
                let ret = ReturnOp::new(ctx, projection.result(ctx)).get_operation();
                ret.insert_at_back(f.block, ctx);
                let options = crate::verification::VerifyOptions {
                    supported_unbounded_dynamic_op: Some(|ctx, operation| {
                        matches!(
                            opcode(ctx, operation),
                            Ok(Opcode::Parameter | Opcode::Tuple | Opcode::GetTupleElement)
                        )
                    }),
                    ..Default::default()
                };
                crate::verification::verify_with_options(ctx, f.module, options).unwrap();
                replace_instruction(
                    ctx,
                    projection.get_operation(),
                    f.parameters[0],
                    Default::default(),
                    None,
                )
                .unwrap();
                assert_eq!(ret.deref(ctx).get_operand(0).get_type(ctx), ty);
                assert_eq!(ret.deref(ctx).get_operand(0), f.parameters[0]);
                crate::verification::verify_with_options(ctx, f.module, options).unwrap();
            }
        }
    }
}
