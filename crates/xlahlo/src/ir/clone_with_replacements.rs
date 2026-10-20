//! Private, unpublished computation-cloning prerequisite.
//!
//! Source: OpenXLA 941335fe721cdc5fad5b6c37a0c2e3a7a36fea24,
//! hlo_computation.cc:2248-2454 and hlo_instruction.cc:3060-3080.
//! Extra parameters/external dependencies: hlo_computation.h:744-800.
//! Async ownership: hlo_instructions.cc:795-825. Accuracy normalization:
//! hlo_instruction.cc:2573-2608 and IsUnaryOpWithResultAccuracy:5701-5728.
//! Replacement lookup occurs once at each operand edge, not recursively.
//! An omitted instruction must have no surviving data use. External operand
//! dependencies are copied, never captured through same-context clone fallback.
//!
//! A replacement's attributes, including OriginalValue, belong to that
//! replacement, not the replaced instruction. Shape-index rebasing is the
//! caller's responsibility (as in RemoveDeadTupleIndices). The original module,
//! templates, module schedule and configuration are never edited. The caller
//! owns the returned detached computation and must verify its eventual complete
//! module with its original VerifyOptions after publishing/remapping callsites.
//! Passing no VerifyOptions is a raw structural clone, not implicit verification;
//! upstream can clone intermediate computations that are not yet semantically valid.
//! OriginalValue is copied without validating its tree against the shape, matching
//! hlo_original_value_util.cc:30-58; explicit verification checks that contract.
//!
//! Qualification pending. Logical storage/user/control permutations and the
//! pending full FusionParameter annotation representation require qualification;
//! this is not a public parity claim or a while-state transformation.
use super::{
    ControlPredecessors, FusionParameters, InstructionName, InstructionSequence, Schedule,
    instruction_order::{self, InstructionOrder, OrderPermutation},
};
use crate::{
    attributes::{Callees, Index, ResultAccuracy},
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp, control::TupleOp},
    original_value::OriginalValue,
    types::TupleType,
    verification::{VerifyOptions, error, optional_attr, require},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{attributes::TypeAttr, ops::ModuleOp, types::FunctionType},
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::{IrMapping, clone_operation},
        listener::DummyListener,
        rewriter::IRRewriter,
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
    value::{DefiningEntity, Value},
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Default)]
pub(crate) enum CloneRoot {
    #[default]
    Keep,
    Instruction(Ptr<Operation>),
    Tuple(Vec<Ptr<Operation>>),
}

/// Detached templates have no native name scope. Explicit edges supply the
/// pointer identity that upstream stores on HloInstruction, including outgoing
/// template edges; existing attached control edges use their actual block scope.
#[derive(Default)]
pub(crate) struct Replacements {
    pub instructions: HashMap<Ptr<Operation>, Option<Ptr<Operation>>>,
    pub extra_parameters: Vec<Ptr<Operation>>,
    pub template_control_edges: Vec<(Ptr<Operation>, Ptr<Operation>)>,
    pub root: CloneRoot,
}

pub(crate) struct ComputationClone {
    pub computation: ComputationOp,
    /// AsyncStart owns a new wrapped computation even in the same module.
    /// Publish these alongside `computation`; all remain detached on success.
    pub embedded_computations: Vec<ComputationOp>,
    /// Pinned AsyncStart cloning gives a scheduled wrapped clone a trivial
    /// postorder schedule. Apply these only when publishing the embedded clones.
    pub embedded_schedules: Vec<InstructionSequence>,
    /// Pinned HloCloneContext maps selected templates, not replaced originals.
    pub mapping: IrMapping,
}

/// Cached native handles must outlive this context. Mappings and reserved names
/// are committed only on success; the caller still owns every returned clone.
#[derive(Clone)]
pub(crate) struct ComputationCloneContext {
    destination: Option<Ptr<Operation>>,
    suffix: String,
    computations: HashMap<Ptr<Operation>, Ptr<Operation>>,
    instructions: HashMap<Ptr<Operation>, Ptr<Operation>>,
    computations_by_name: HashMap<String, Ptr<Operation>>,
    instructions_by_name: HashMap<String, Ptr<Operation>>,
    instruction_names: HashSet<String>,
    computation_names: HashSet<String>,
    schedules: Vec<InstructionSequence>,
    in_progress: HashSet<Ptr<Operation>>,
}

impl ComputationCloneContext {
    pub(crate) fn new(destination: Option<Ptr<Operation>>, suffix: String) -> Self {
        Self {
            destination,
            suffix,
            computations: HashMap::new(),
            instructions: HashMap::new(),
            computations_by_name: HashMap::new(),
            instructions_by_name: HashMap::new(),
            instruction_names: HashSet::new(),
            computation_names: HashSet::new(),
            schedules: Vec::new(),
            in_progress: HashSet::new(),
        }
    }

    pub(crate) fn computation(&self, source: Ptr<Operation>) -> Option<Ptr<Operation>> {
        self.computations.get(&source).copied()
    }

    pub(crate) fn instruction(&self, source: Ptr<Operation>) -> Option<Ptr<Operation>> {
        self.instructions.get(&source).copied()
    }

    pub(crate) fn computation_by_name(&self, name: &str) -> Option<Ptr<Operation>> {
        self.computations_by_name.get(name).copied()
    }

    pub(crate) fn instruction_by_name(&self, name: &str) -> Option<Ptr<Operation>> {
        self.instructions_by_name.get(name).copied()
    }
}

fn owning_module(ctx: &Context, mut operation: Ptr<Operation>) -> Option<Ptr<Operation>> {
    loop {
        if Operation::is_op::<ModuleOp>(operation, ctx) {
            return Some(operation);
        }
        operation = operation
            .deref(ctx)
            .get_parent_block()?
            .deref(ctx)
            .get_parent_op(ctx)?;
    }
}

struct Plan {
    original: Vec<Ptr<Operation>>,
    postorder: Vec<Ptr<Operation>>,
    roots: Vec<Ptr<Operation>>,
    tuple_root: bool,
    tuple_name: Option<String>,
    parameters: Vec<TypeHandle>,
    result: TypeHandle,
    names: HashMap<Ptr<Operation>, String>,
    parameter_names: HashMap<(Ptr<BasicBlock>, usize), String>,
    controls: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>>,
}

fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref())
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| error("clone dependency is not a native HLO instruction"))
}

fn selected(operation: Ptr<Operation>, replacements: &Replacements) -> Option<Ptr<Operation>> {
    replacements
        .instructions
        .get(&operation)
        .copied()
        .unwrap_or(Some(operation))
}

fn operand_definition(ctx: &Context, value: Value) -> Result<Ptr<Operation>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation)
            if operation.deref(ctx).get_num_results() == 1
                && operation.deref(ctx).get_result(0) == value =>
        {
            Ok(operation)
        }
        _ => Err(error(
            "computation clone operand is not a single HLO result",
        )),
    }
}

fn body(ctx: &Context, computation: &ComputationOp) -> Result<Ptr<BasicBlock>> {
    let operation = computation.get_operation().deref(ctx);
    require(
        operation.num_regions() == 1
            && operation.get_num_operands() == 0
            && operation.get_num_results() == 0
            && operation.get_num_successors() == 0,
        "invalid computation clone source",
    )?;
    let blocks: Vec<_> = computation.body(ctx).deref(ctx).iter(ctx).collect();
    require(blocks.len() == 1, "computation cloning requires one block")?;
    require(
        blocks[0].deref(ctx).get_num_arguments() == 0,
        "computation cloning requires parameter instructions",
    )?;
    Ok(blocks[0])
}

fn validate_region_closure(ctx: &Context, instruction: Ptr<Operation>) -> Result<()> {
    let mut pending = vec![instruction];
    while let Some(operation) = pending.pop() {
        let regions: Vec<_> = operation.deref(ctx).regions().collect();
        for region in regions {
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            require(blocks.len() == 1, "HLO clone region must have one block")?;
            let block = blocks[0];
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            let definitions: HashSet<_> = children.iter().copied().collect();
            for child in children {
                let data = child.deref(ctx);
                require(
                    data.get_num_successors() == 0,
                    "HLO clone has block successors",
                )?;
                for operand in data.operands() {
                    let inside = match operand.defining_entity() {
                        DefiningEntity::Op(owner) => definitions.contains(&owner),
                        DefiningEntity::Block(owner) => owner == block,
                    };
                    require(inside, "inline clone region captures an external value")?;
                }
                pending.push(child);
            }
        }
    }
    Ok(())
}

fn plan(ctx: &Context, computation: &ComputationOp, replacements: &Replacements) -> Result<Plan> {
    let block = body(ctx, computation)?;
    let mut physical: Vec<_> = block.deref(ctx).iter(ctx).collect();
    let terminator = physical.pop().ok_or_else(|| error("empty clone source"))?;
    require(
        Operation::is_op::<ReturnOp>(terminator, ctx)
            && terminator.deref(ctx).get_num_operands() == 1,
        "computation clone source needs a return",
    )?;
    let original = instruction_order::instruction_order(ctx, computation.get_operation())?;
    let old_root = operand_definition(ctx, terminator.deref(ctx).get_operand(0))?;
    let originals: HashSet<_> = original.iter().copied().collect();
    require(
        originals.contains(&old_root),
        "root is outside source computation",
    )?;
    for (&key, &replacement) in &replacements.instructions {
        opcode(ctx, key)?;
        if let Some(template) = replacement {
            require(
                template.deref(ctx).get_parent_block().is_none(),
                "replacement value must be an owned detached instruction",
            )?;
        }
    }
    for &parameter in &replacements.extra_parameters {
        require(
            opcode(ctx, parameter)? == Opcode::Parameter
                && parameter.deref(ctx).get_num_operands() == 0
                && parameter.deref(ctx).get_num_results() == 1,
            "extra parameter must be a native parameter without operands",
        )?;
    }
    let (root_inputs, tuple_root) = match &replacements.root {
        CloneRoot::Keep => (vec![old_root], false),
        CloneRoot::Instruction(instruction) => {
            require(
                originals.contains(instruction),
                "new root is outside source computation",
            )?;
            (vec![*instruction], false)
        }
        CloneRoot::Tuple(instructions) => {
            require(
                instructions.iter().all(|op| originals.contains(op)),
                "tuple root contains an instruction outside source computation",
            )?;
            (instructions.clone(), true)
        }
    };
    let roots: Vec<_> = root_inputs
        .into_iter()
        .map(|root| selected(root, replacements).ok_or_else(|| error("clone root is omitted")))
        .collect::<Result<_>>()?;
    let starts = original.iter().filter_map(|&op| selected(op, replacements));
    let mut state = HashMap::<Ptr<Operation>, bool>::new();
    let mut postorder = Vec::new();
    for start in starts {
        let mut pending = vec![(start, false)];
        while let Some((operation, finish)) = pending.pop() {
            if finish {
                state.insert(operation, true);
                postorder.push(operation);
                continue;
            }
            if let Some(&finished) = state.get(&operation) {
                require(finished, "cycle in replaced operand graph")?;
                continue;
            }
            let data = operation.deref(ctx);
            require(
                data.get_num_results() == 1 && data.get_num_successors() == 0,
                "clone instruction requires one result and no block successors",
            )?;
            opcode(ctx, operation)?;
            validate_region_closure(ctx, operation)?;
            state.insert(operation, false);
            pending.push((operation, true));
            for operand in data.operands() {
                let definition = operand_definition(ctx, operand)?;
                let dependency = selected(definition, replacements)
                    .ok_or_else(|| error("replacements omit a surviving data operand"))?;
                pending.push((dependency, false));
            }
        }
    }
    require(
        roots.iter().all(|root| state.get(root) == Some(&true)),
        "new root is not in the cloned dependency graph",
    )?;
    let edges = control_edges(ctx, &original, &postorder, replacements)?;
    let selected_set: HashSet<_> = postorder.iter().copied().collect();
    // Extras are independent Clone() occurrences, not DFS roots or shared
    // context mappings. An occurrence also reached by DFS must not deduplicate
    // away its duplicate parameter index.
    postorder.splice(0..0, replacements.extra_parameters.iter().copied());
    let mut parameters = BTreeMap::new();
    let mut names = HashMap::new();
    let mut used_names = HashSet::new();
    for &operation in &postorder {
        if opcode(ctx, operation)? == Opcode::Parameter {
            let index = optional_attr::<Index>(ctx, operation, "hlo_index")?
                .ok_or_else(|| error("clone parameter requires an index"))?
                .0;
            require(
                operation.deref(ctx).get_num_operands() == 0
                    && parameters
                        .insert(index, operation.deref(ctx).get_type(0))
                        .is_none(),
                "duplicate clone parameter index or parameter operands",
            )?;
        }
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            require(!name.0.is_empty(), "empty output instruction name")?;
            used_names.insert(name.0.clone());
            names.insert(operation, name.0.clone());
        }
        optional_attr::<OriginalValue>(ctx, operation, "hlo_original_value")?;
    }
    let parameter_count =
        u32::try_from(parameters.len()).map_err(|_| error("too many clone parameters"))?;
    require(
        parameters.keys().copied().eq(0..parameter_count),
        "clone parameter indices must be contiguous",
    )?;
    let mut controls = HashMap::<_, Vec<_>>::new();
    for (predecessor, successor) in &edges {
        if !selected_set.contains(predecessor) {
            continue;
        }
        let Some(successor) = selected(*successor, replacements) else {
            continue;
        };
        require(
            selected_set.contains(&successor),
            "control successor has no clone",
        )?;
        if !names.contains_key(predecessor) {
            let mut index = 0;
            let name = loop {
                let name = format!("clone_control.{index}");
                if used_names.insert(name.clone()) {
                    break name;
                }
                index += 1;
            };
            names.insert(*predecessor, name);
        }
        let incoming = controls.entry(successor).or_default();
        if !incoming.contains(predecessor) {
            incoming.push(*predecessor);
        }
    }
    let result = if tuple_root {
        TupleType::get(
            ctx,
            roots.iter().map(|op| op.deref(ctx).get_type(0)).collect(),
        )?
    } else {
        roots[0].deref(ctx).get_type(0)
    };
    Ok(Plan {
        original,
        postorder,
        roots,
        tuple_root,
        tuple_name: None,
        parameters: parameters.into_values().collect(),
        result,
        names,
        parameter_names: HashMap::new(),
        controls,
    })
}

fn control_edges(
    ctx: &Context,
    original: &[Ptr<Operation>],
    postorder: &[Ptr<Operation>],
    replacements: &Replacements,
) -> Result<Vec<(Ptr<Operation>, Ptr<Operation>)>> {
    let mut blocks = Vec::new();
    let mut seen_blocks = HashSet::new();
    let mut edges = Vec::new();
    for &instruction in original.iter().chain(postorder) {
        if let Some(block) = instruction.deref(ctx).get_parent_block() {
            if seen_blocks.insert(block) {
                blocks.push(block);
            }
        } else {
            require(
                optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
                    .is_none_or(|predecessors| predecessors.0.is_empty()),
                "detached symbolic controls need explicit template control edges",
            )?;
        }
    }
    for block in blocks {
        let instructions: Vec<_> = block.deref(ctx).iter(ctx).collect();
        let mut names = HashMap::new();
        for &instruction in &instructions {
            if let Some(name) = optional_attr::<InstructionName>(ctx, instruction, "hlo_name")? {
                require(
                    names.insert(name.0.clone(), instruction).is_none(),
                    "duplicate source instruction name",
                )?;
            }
        }
        for instruction in instructions {
            if let Some(predecessors) =
                optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
            {
                for name in &predecessors.0 {
                    let predecessor = *names
                        .get(name)
                        .ok_or_else(|| error("unresolved source control predecessor"))?;
                    edges.push((predecessor, instruction));
                }
            }
        }
    }
    let known: HashSet<_> = original.iter().chain(postorder).copied().collect();
    for &(predecessor, successor) in &replacements.template_control_edges {
        require(
            known.contains(&predecessor) && known.contains(&successor),
            "template control endpoint is outside the clone graph",
        )?;
        edges.push((predecessor, successor));
    }
    Ok(edges)
}

/// Clone without publishing or editing the source. Errors after allocation erase
/// the entire staging computation, including its operand uses. This routine does
/// not supply a default verifier policy or pretend detached named callees resolve.
pub(crate) fn clone_with_replacements(
    ctx: &mut Context,
    source: &ComputationOp,
    name: Identifier,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
) -> Result<ComputationClone> {
    let module = owning_module(ctx, source.get_operation());
    let mut context = ComputationCloneContext::new(module, "clone".into());
    clone_in_context(ctx, source, name, replacements, options, &mut context)
}

pub(crate) fn clone_in_context(
    ctx: &mut Context,
    source: &ComputationOp,
    name: Identifier,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<ComputationClone> {
    let mut transaction = context.clone();
    let result = clone_impl(ctx, source, name, replacements, options, &mut transaction);
    if result.is_ok() {
        *context = transaction;
    }
    result
}

fn clone_impl(
    ctx: &mut Context,
    source: &ComputationOp,
    name: Identifier,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<ComputationClone> {
    require(
        context.in_progress.insert(source.get_operation()),
        "cycle in cloned computation dependencies",
    )?;
    let source_module = source
        .get_operation()
        .deref(ctx)
        .get_parent_block()
        .and_then(|block| block.deref(ctx).get_parent_op(ctx));
    if let Some(module) = source_module {
        require(
            Operation::is_op::<ModuleOp>(module, ctx),
            "source owner is not a module",
        )?;
    }
    if let Some(options) = options {
        crate::verification::verify_with_options(
            ctx,
            source_module.unwrap_or(source.get_operation()),
            options,
        )?;
    }
    let module = context.destination;
    if let Some(module) = module {
        require(
            Operation::is_op::<ModuleOp>(module, ctx),
            "clone destination is not a module",
        )?;
        if Some(module) != source_module
            && let Some(options) = options
        {
            crate::verification::verify_with_options(ctx, module, options)?;
        }
    }
    let mut plan = plan(ctx, source, replacements)?;
    let source_name = source
        .get_attr_hlo_computation_name(ctx)
        .ok_or_else(|| error("clone source has no computation name"))?
        .as_ref()
        .to_string();
    let mut source_instruction_names = HashMap::new();
    for operation in nested_operations(ctx, &plan.postorder) {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            source_instruction_names.insert(operation, name.0.clone());
        }
    }
    // HloModule uniquifies instruction names when adding clones. Native named
    // controls need those final names before the detached clone can be verified.
    uniquify_names(ctx, module, &mut plan, &mut context.instruction_names)?;
    let block_attributes = body(ctx, source)?.deref(ctx).attributes.clone();
    let (embedded_computations, embedded_schedules, async_callees) =
        clone_called_computations(ctx, source_module, &plan, options, context)?;
    let mut computation_names = context.computation_names.clone();
    if let Some(module) = module {
        for operation in nested_operations(ctx, &[module]) {
            if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx)
                && let Some(name) = computation.get_attr_hlo_computation_name(ctx)
            {
                computation_names.insert(name.as_ref().to_string());
            }
        }
    }
    for computation in &embedded_computations {
        if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
            computation_names.insert(name.as_ref().to_string());
        }
    }
    let name = Identifier::try_new(unique_symbol(&mut computation_names, name.as_ref()));
    let name = match name {
        Ok(name) => name,
        Err(error) => {
            for embedded in embedded_computations {
                Operation::erase(embedded.get_operation(), ctx);
            }
            return Err(error);
        }
    };
    context.computation_names = computation_names;
    let signature: TypeHandle =
        FunctionType::get(ctx, plan.parameters.clone(), vec![plan.result]).into();
    let computation = ComputationOp::new(ctx, name.clone(), signature);
    let attributes = source.get_operation().deref(ctx).attributes.clone();
    let location = source.get_operation().deref(ctx).loc();
    {
        let mut data = computation.get_operation().deref_mut(ctx);
        data.attributes = attributes;
        data.attributes
            .0
            .remove(&Identifier::new("hlo_instruction_order"));
        data.set_loc(location);
    }
    computation
        .set_attr_hlo_computation_name(ctx, pliron::builtin::attributes::IdentifierAttr::new(name));
    computation.set_attr_hlo_computation_signature(ctx, TypeAttr::new(signature));
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(computation.body(ctx), ctx);
    block.deref_mut(ctx).attributes = block_attributes;
    let outcome = build(
        ctx,
        &computation,
        block,
        replacements,
        &plan,
        &async_callees,
        options,
    )
    .and_then(|mapping| {
        if let Some(options) = options {
            verify_in_source_context(
                ctx,
                module,
                &computation,
                &embedded_computations,
                &embedded_schedules,
                context,
                options,
            )?;
        }
        Ok(mapping)
    });
    match outcome {
        Ok(mapping) => {
            context.in_progress.remove(&source.get_operation());
            context
                .computations
                .insert(source.get_operation(), computation.get_operation());
            context
                .computations_by_name
                .insert(source_name, computation.get_operation());
            for operation in nested_operations(ctx, &plan.postorder) {
                // Pinned extra_parameters use Clone() without the clone context.
                if replacements.extra_parameters.contains(&operation) {
                    continue;
                }
                if let Some(cloned) = mapping.lookup_op(operation) {
                    context.instructions.insert(operation, cloned);
                    if let Some(name) = source_instruction_names.get(&operation) {
                        context.instructions_by_name.insert(name.clone(), cloned);
                    }
                }
            }
            context.schedules.extend(embedded_schedules.iter().cloned());
            Ok(ComputationClone {
                computation,
                embedded_computations,
                embedded_schedules,
                mapping,
            })
        }
        Err(error) => {
            Operation::erase(computation.get_operation(), ctx);
            for embedded in embedded_computations {
                Operation::erase(embedded.get_operation(), ctx);
            }
            Err(error)
        }
    }
}

fn nested_operations(ctx: &Context, roots: &[Ptr<Operation>]) -> Vec<Ptr<Operation>> {
    let mut result = Vec::new();
    let mut pending: Vec<_> = roots.iter().rev().copied().collect();
    while let Some(operation) = pending.pop() {
        result.push(operation);
        let mut children = Vec::new();
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                children.extend(block.deref(ctx).iter(ctx));
            }
        }
        pending.extend(children.into_iter().rev());
    }
    result
}

type AsyncCallees = HashMap<Ptr<Operation>, Callees>;

fn clone_called_computations(
    ctx: &mut Context,
    module: Option<Ptr<Operation>>,
    plan: &Plan,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<(Vec<ComputationOp>, Vec<InstructionSequence>, AsyncCallees)> {
    let mut embedded = Vec::<ComputationOp>::new();
    let mut schedules = Vec::new();
    let result = (|| {
        let mut remapped = HashMap::new();
        for operation in nested_operations(ctx, &plan.postorder) {
            if Operation::is_op::<ReturnOp>(operation, ctx) {
                continue;
            }
            let kind = opcode(ctx, operation)?;
            let callees = optional_attr::<Callees>(ctx, operation, "hlo_callees")?
                .map(|callees| callees.0.clone())
                .unwrap_or_default();
            if kind == Opcode::AsyncStart {
                require(
                    callees.len() == 1,
                    "async start requires one wrapped computation",
                )?;
            }
            if callees.is_empty() {
                continue;
            }
            let owner = owning_module(ctx, operation)
                .or(module)
                .ok_or_else(|| error("named clone dependency needs a source module"))?;
            let mut available = HashMap::new();
            for candidate in nested_operations(ctx, &[owner]) {
                if let Some(computation) = Operation::get_op::<ComputationOp>(candidate, ctx) {
                    let name = computation
                        .get_attr_hlo_computation_name(ctx)
                        .ok_or_else(|| error("missing source callee name"))?;
                    available.insert(name.as_ref().clone(), candidate);
                }
            }
            let mut names = Vec::new();
            for callee_name in callees {
                let original = *available
                    .get(&callee_name)
                    .ok_or_else(|| error("unresolved clone dependency computation"))?;
                let is_async = matches!(
                    kind,
                    Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone,
                );
                if Some(owner) == context.destination && !is_async {
                    names.push(callee_name);
                    continue;
                }
                let cloned = if let Some(cloned) = context.computation(original) {
                    cloned
                } else {
                    let callee = Operation::get_op::<ComputationOp>(original, ctx)
                        .ok_or_else(|| error("callee is not a native computation"))?;
                    let mut names = HashSet::new();
                    let name = Identifier::try_new(unique_symbol(
                        &mut names,
                        &format!("{callee_name}.{}", context.suffix),
                    ))?;
                    let output = clone_impl(
                        ctx,
                        &callee,
                        name,
                        &Replacements::default(),
                        options,
                        context,
                    )?;
                    let cloned = output.computation.get_operation();
                    schedules.extend(output.embedded_schedules);
                    embedded.extend(output.embedded_computations);
                    embedded.push(output.computation);
                    if kind == Opcode::AsyncStart
                        && optional_attr::<Schedule>(ctx, owner, "hlo_schedule")?.is_some_and(
                            |schedule| {
                                schedule
                                    .0
                                    .iter()
                                    .any(|sequence| sequence.computation == callee_name)
                            },
                        )
                    {
                        let computation = embedded
                            .last()
                            .ok_or_else(|| error("missing async wrapped clone"))?;
                        let order =
                            topological_order(ctx, &ordered_instructions(ctx, computation)?)?;
                        schedules.push(sequence(ctx, computation, &order)?);
                    }
                    cloned
                };
                let computation = Operation::get_op::<ComputationOp>(cloned, ctx)
                    .ok_or_else(|| error("context mapping is not a computation"))?;
                names.push(
                    computation
                        .get_attr_hlo_computation_name(ctx)
                        .ok_or_else(|| error("cloned callee has no name"))?
                        .as_ref()
                        .clone(),
                );
            }
            remapped.insert(operation, Callees(names));
        }
        Ok(remapped)
    })();
    match result {
        Ok(callees) => Ok((embedded, schedules, callees)),
        Err(error) => {
            for computation in embedded {
                Operation::erase(computation.get_operation(), ctx);
            }
            Err(error)
        }
    }
}

fn uniquify_names(
    ctx: &Context,
    module: Option<Ptr<Operation>>,
    plan: &mut Plan,
    reserved: &mut HashSet<String>,
) -> Result<()> {
    let mut used = reserved.clone();
    let mut pending: Vec<_> = module.into_iter().collect();
    while let Some(operation) = pending.pop() {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            used.insert(name.0.clone());
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                if let Some(parameters) = block
                    .deref(ctx)
                    .attributes
                    .get::<FusionParameters>(&Identifier::new("hlo_fusion_parameters"))
                {
                    used.extend(
                        parameters
                            .0
                            .iter()
                            .filter_map(|parameter| parameter.name.clone()),
                    );
                }
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    // Iteration order must not decide names when a template already has a
    // .clone suffix. Match the deterministic instruction traversal.
    for operation in nested_operations(ctx, &plan.postorder) {
        if let std::collections::hash_map::Entry::Vacant(entry) = plan.names.entry(operation)
            && let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
        {
            require(!name.0.is_empty(), "empty nested instruction name")?;
            entry.insert(name.0.clone());
        }
        if let Some(name) = plan.names.get_mut(&operation) {
            *name = unique_name(&mut used, name);
        } else if !Operation::is_op::<ReturnOp>(operation, ctx) {
            let name = unique_name(&mut used, &format!("{}.clone", opcode(ctx, operation)?));
            plan.names.insert(operation, name);
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                if Operation::is_op::<crate::ops::control::FusionOp>(operation, ctx) {
                    let parameters = super::fusion_parameters(ctx, block)?;
                    for (index, parameter) in parameters.0.iter().enumerate() {
                        let default = format!("parameter_{index}.clone");
                        let name = parameter.name.as_deref().unwrap_or(&default);
                        plan.parameter_names
                            .insert((block, index), unique_name(&mut used, name));
                    }
                }
            }
        }
    }
    if plan.tuple_root {
        plan.tuple_name = Some(unique_name(&mut used, "tuple.clone"));
    }
    *reserved = used;
    Ok(())
}

fn unique_name(used: &mut HashSet<String>, old: &str) -> String {
    let mut candidate = old.to_owned();
    let mut suffix = 0;
    while !used.insert(candidate.clone()) {
        candidate = if suffix == 0 {
            format!("{old}.clone")
        } else {
            format!("{old}.clone.{suffix}")
        };
        suffix += 1;
    }
    candidate
}

fn unique_symbol(used: &mut HashSet<String>, requested: &str) -> String {
    let mut base: String = requested
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    if !base.starts_with(|character: char| character.is_ascii_alphabetic() || character == '_') {
        base.insert(0, '_');
    }
    let mut candidate = base.clone();
    let mut index = 0;
    while !used.insert(candidate.clone()) {
        candidate = format!("{base}_clone_{index}");
        index += 1;
    }
    candidate
}

fn normalize_nested_metadata(
    ctx: &Context,
    template: Ptr<Operation>,
    mapping: &IrMapping,
    plan: &Plan,
) -> Result<()> {
    for original in nested_operations(ctx, &[template]) {
        let cloned = mapping
            .lookup_op(original)
            .ok_or_else(|| error("nested operation has no clone mapping"))?;
        optional_attr::<OriginalValue>(ctx, original, "hlo_original_value")?;
        {
            let mut data = cloned.deref_mut(ctx);
            if let Some(name) = plan.names.get(&original) {
                data.attributes
                    .set(Identifier::new("hlo_name"), InstructionName(name.clone()));
            }
            if !Operation::is_op::<ReturnOp>(original, ctx)
                && !opcode(ctx, original)?.supports_result_accuracy()
                && data
                    .attributes
                    .get::<ResultAccuracy>(&Identifier::new("hlo_result_accuracy"))
                    .is_some()
            {
                data.attributes.set(
                    Identifier::new("hlo_result_accuracy"),
                    ResultAccuracy::Default,
                );
            }
        }
        for region in original.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                let cloned_block = mapping
                    .lookup_block(block)
                    .ok_or_else(|| error("nested block has no clone mapping"))?;
                let mut renamed = HashMap::new();
                let mut ambiguous = HashSet::new();
                for child in block.deref(ctx).iter(ctx) {
                    if let Some(name) = optional_attr::<InstructionName>(ctx, child, "hlo_name")?
                        && renamed
                            .insert(name.0.clone(), plan.names[&child].clone())
                            .is_some()
                    {
                        ambiguous.insert(name.0.clone());
                    }
                }
                let parameters = if Operation::is_op::<crate::ops::control::FusionOp>(original, ctx)
                {
                    Some(super::fusion_parameters(ctx, block)?)
                } else {
                    None
                };
                if let Some(parameters) = &parameters {
                    for (index, parameter) in parameters.0.iter().enumerate() {
                        if let Some(name) = &parameter.name {
                            let mapped =
                                plan.parameter_names.get(&(block, index)).ok_or_else(|| {
                                    error("fusion parameter lacks a planned clone name")
                                })?;
                            if renamed.insert(name.clone(), mapped.clone()).is_some() {
                                ambiguous.insert(name.clone());
                            }
                        }
                    }
                }
                let remap_names = |names: &[String]| -> Result<Vec<String>> {
                    names
                        .iter()
                        .map(|name| {
                            require(!ambiguous.contains(name), "ambiguous nested control name")?;
                            renamed
                                .get(name)
                                .cloned()
                                .ok_or_else(|| error("unresolved nested control predecessor"))
                        })
                        .collect()
                };
                if let Some(mut parameters) = parameters {
                    for (index, parameter) in parameters.0.iter_mut().enumerate() {
                        parameter.name = plan.parameter_names.get(&(block, index)).cloned();
                        parameter.control_predecessors =
                            remap_names(&parameter.control_predecessors)?;
                    }
                    cloned_block
                        .deref_mut(ctx)
                        .attributes
                        .set(Identifier::new("hlo_fusion_parameters"), parameters);
                }
                for child in block.deref(ctx).iter(ctx) {
                    if let Some(controls) = optional_attr::<ControlPredecessors>(
                        ctx,
                        child,
                        "hlo_control_predecessors",
                    )? {
                        let cloned_child = mapping
                            .lookup_op(child)
                            .ok_or_else(|| error("control target lacks a clone mapping"))?;
                        cloned_child.deref_mut(ctx).attributes.set(
                            Identifier::new("hlo_control_predecessors"),
                            ControlPredecessors(remap_names(&controls.0)?),
                        );
                    }
                }
            }
        }
    }
    for original in nested_operations(ctx, &[template]) {
        if Operation::is_op::<crate::ops::control::FusionOp>(original, ctx) {
            let cloned = mapping
                .lookup_op(original)
                .ok_or_else(|| error("fusion has no clone mapping"))?;
            for region in cloned.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    super::verify_fusion_parameter_structure(ctx, block)?;
                }
            }
        }
    }
    Ok(())
}

/// Reuse the authoritative module verifier in an isolated copy of the source
/// context. The real module is never used as a temporary publication surface.
fn verify_in_source_context(
    ctx: &mut Context,
    module: Option<Ptr<Operation>>,
    computation: &ComputationOp,
    embedded: &[ComputationOp],
    embedded_schedules: &[InstructionSequence],
    context: &ComputationCloneContext,
    options: VerifyOptions,
) -> Result<()> {
    let Some(module) = module else {
        require(
            embedded.is_empty(),
            "embedded computations require a module context",
        )?;
        return crate::verification::verify_with_options(ctx, computation.get_operation(), options);
    };
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let staging = clone_operation(module, ctx, &mut rewriter, &mut IrMapping::new());
    let block = staging
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block();
    let Some(block) = block else {
        Operation::erase(staging, ctx);
        return Err(error("staging module has no block"));
    };
    let mut present = HashSet::new();
    for operation in nested_operations(ctx, &[staging]) {
        if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx)
            && let Some(name) = computation.get_attr_hlo_computation_name(ctx)
        {
            present.insert(name.as_ref().clone());
        }
    }
    computation.get_operation().insert_at_back(block, ctx);
    if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
        present.insert(name.as_ref().clone());
    }
    for computation in embedded {
        computation.get_operation().insert_at_back(block, ctx);
        if let Some(name) = computation.get_attr_hlo_computation_name(ctx) {
            present.insert(name.as_ref().clone());
        }
    }
    // Earlier context outputs may remain detached. Copy rather than temporarily
    // relink them: only this transaction's new results are moved into staging.
    let result = (|| {
        for &operation in context.computations.values() {
            let computation = Operation::get_op::<ComputationOp>(operation, ctx)
                .ok_or_else(|| error("cached clone is not a native computation"))?;
            let name = computation
                .get_attr_hlo_computation_name(ctx)
                .ok_or_else(|| error("cached clone computation lacks a name"))?
                .as_ref()
                .clone();
            if present.insert(name) {
                let cloned = clone_operation(operation, ctx, &mut rewriter, &mut IrMapping::new());
                cloned.insert_at_back(block, ctx);
            }
        }
        if !embedded_schedules.is_empty() || !context.schedules.is_empty() {
            let mut schedule = optional_attr::<Schedule>(ctx, staging, "hlo_schedule")?
                .map(|schedule| schedule.clone())
                .unwrap_or_default();
            let mut names: HashSet<_> = schedule
                .0
                .iter()
                .map(|sequence| sequence.computation.clone())
                .collect();
            for sequence in context.schedules.iter().chain(embedded_schedules) {
                if names.insert(sequence.computation.clone()) {
                    schedule.0.push(sequence.clone());
                } else {
                    require(
                        schedule.0.iter().any(|existing| existing == sequence),
                        "cached clone schedule conflicts with destination schedule",
                    )?;
                }
            }
            staging
                .deref_mut(ctx)
                .attributes
                .set(Identifier::new("hlo_schedule"), schedule);
        }
        crate::verification::verify_with_options(ctx, staging, options)
    })();
    computation.get_operation().unlink(ctx);
    for computation in embedded {
        computation.get_operation().unlink(ctx);
    }
    Operation::erase(staging, ctx);
    result
}

fn ordered_instructions(ctx: &Context, computation: &ComputationOp) -> Result<Vec<Ptr<Operation>>> {
    Ok(body(ctx, computation)?
        .deref(ctx)
        .iter(ctx)
        .filter(|&operation| !Operation::is_op::<ReturnOp>(operation, ctx))
        .collect())
}

fn topological_order(ctx: &Context, preferred: &[Ptr<Operation>]) -> Result<Vec<Ptr<Operation>>> {
    let mut names = HashMap::new();
    for &operation in preferred {
        if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
            require(
                names.insert(name.0.clone(), operation).is_none(),
                "duplicate schedule name",
            )?;
        }
    }
    let mut dependencies = HashMap::new();
    for &operation in preferred {
        let mut incoming = HashSet::new();
        for operand in operation.deref(ctx).operands() {
            incoming.insert(operand_definition(ctx, operand)?);
        }
        if let Some(controls) =
            optional_attr::<ControlPredecessors>(ctx, operation, "hlo_control_predecessors")?
        {
            for name in &controls.0 {
                incoming.insert(
                    *names
                        .get(name)
                        .ok_or_else(|| error("unresolved scheduled control predecessor"))?,
                );
            }
        }
        dependencies.insert(operation, incoming);
    }
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    while ordered.len() < preferred.len() {
        let next = preferred
            .iter()
            .copied()
            .find(|operation| {
                !seen.contains(operation)
                    && dependencies[operation]
                        .iter()
                        .all(|dependency| seen.contains(dependency))
            })
            .ok_or_else(|| error("cycle or external dependency in cloned schedule"))?;
        seen.insert(next);
        ordered.push(next);
    }
    Ok(ordered)
}

fn sequence(
    ctx: &Context,
    computation: &ComputationOp,
    order: &[Ptr<Operation>],
) -> Result<InstructionSequence> {
    let name = computation
        .get_attr_hlo_computation_name(ctx)
        .ok_or_else(|| error("scheduled clone has no computation name"))?
        .as_ref()
        .clone();
    let instructions = order
        .iter()
        .map(|&instruction| {
            optional_attr::<InstructionName>(ctx, instruction, "hlo_name")?
                .map(|name| name.0.clone())
                .ok_or_else(|| error("scheduled clone instruction has no name"))
        })
        .collect::<Result<_>>()?;
    Ok(InstructionSequence {
        computation: name,
        instructions,
    })
}

pub(crate) struct ScheduledComputationClone {
    pub cloned: ComputationClone,
    pub sequence: InstructionSequence,
}

/// The separate pinned schedule companion, not a requirement on ordinary clones.
/// Schedule construction failures dispose the already-detached clone transaction.
pub(crate) fn clone_with_schedule_and_replacements(
    ctx: &mut Context,
    source: &ComputationOp,
    name: Identifier,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
) -> Result<ScheduledComputationClone> {
    let module = owning_module(ctx, source.get_operation());
    let mut context = ComputationCloneContext::new(module, "clone".into());
    clone_with_schedule_in_context(ctx, source, name, replacements, options, &mut context)
}

pub(crate) fn clone_with_schedule_in_context(
    ctx: &mut Context,
    source: &ComputationOp,
    name: Identifier,
    replacements: &Replacements,
    options: Option<VerifyOptions>,
    context: &mut ComputationCloneContext,
) -> Result<ScheduledComputationClone> {
    let module = source
        .get_operation()
        .deref(ctx)
        .get_parent_block()
        .and_then(|block| block.deref(ctx).get_parent_op(ctx))
        .ok_or_else(|| error("scheduled clone requires a module"))?;
    let source_name = source
        .get_attr_hlo_computation_name(ctx)
        .ok_or_else(|| error("scheduled source requires a name"))?
        .as_ref()
        .clone();
    let source_order = optional_attr::<Schedule>(ctx, module, "hlo_schedule")?
        .and_then(|schedule| {
            schedule
                .0
                .iter()
                .find(|sequence| sequence.computation == source_name)
                .cloned()
        })
        .ok_or_else(|| error("source computation is not scheduled"))?;
    let source_instructions = ordered_instructions(ctx, source)?;
    let mut by_name = HashMap::new();
    for instruction in source_instructions {
        if let Some(name) = optional_attr::<InstructionName>(ctx, instruction, "hlo_name")? {
            by_name.insert(name.0.clone(), instruction);
        }
    }
    let mut transaction = context.clone();
    let output = clone_impl(ctx, source, name, replacements, options, &mut transaction)?;
    let result = (|| {
        let mut ordered = Vec::new();
        let mut mapped = HashSet::new();
        let mut after_last_parameter = 0;
        for name in &source_order.instructions {
            let original = *by_name
                .get(name)
                .ok_or_else(|| error("scheduled source instruction is unresolved"))?;
            let Some(selected) = selected(original, replacements) else {
                continue;
            };
            let cloned = output
                .mapping
                .lookup_op(selected)
                .ok_or_else(|| error("scheduled replacement has no clone mapping"))?;
            if mapped.insert(cloned) {
                ordered.push(cloned);
            }
            if opcode(ctx, original)? == Opcode::Parameter {
                after_last_parameter = ordered.len();
            }
        }
        let instructions = ordered_instructions(ctx, &output.computation)?;
        let mut extras = Vec::new();
        for &operation in &instructions {
            if !mapped.contains(&operation) && opcode(ctx, operation)? == Opcode::Parameter {
                extras.push(operation);
            }
        }
        ordered.splice(after_last_parameter..after_last_parameter, extras);
        ordered.extend(
            instructions.iter().copied().filter(|operation| {
                !ordered_contains_mapped_or_parameter(ctx, *operation, &mapped)
            }),
        );
        // Upstream instruction construction always assigns a name. Native
        // constructors allow unnamed extras, so supply deterministic names here.
        let mut used = transaction.instruction_names.clone();
        for operation in nested_operations(ctx, &[module]) {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                used.insert(name.0.clone());
            }
        }
        for &operation in &instructions {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                used.insert(name.0.clone());
            }
        }
        for &operation in &instructions {
            if optional_attr::<InstructionName>(ctx, operation, "hlo_name")?.is_none() {
                let name = unique_name(&mut used, &format!("{}.clone", opcode(ctx, operation)?));
                operation
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_name"), InstructionName(name));
            }
        }
        transaction.instruction_names = used;
        let sequence = sequence(ctx, &output.computation, &ordered)?;
        if let Some(options) = options {
            // Module verification skips dependency checks for unreachable
            // computations; this detached result still needs its supplied order
            // checked at the explicitly verified clone boundary.
            require(
                topological_order(ctx, &ordered)? == ordered,
                "mapped clone schedule violates replacement dependencies",
            )?;
            let mut schedules = output.embedded_schedules.clone();
            schedules.push(sequence.clone());
            verify_in_source_context(
                ctx,
                transaction.destination,
                &output.computation,
                &output.embedded_computations,
                &schedules,
                &transaction,
                options,
            )?;
        }
        Ok(sequence)
    })();
    match result {
        Ok(sequence) => {
            transaction.schedules.push(sequence.clone());
            *context = transaction;
            Ok(ScheduledComputationClone {
                cloned: output,
                sequence,
            })
        }
        Err(error) => {
            Operation::erase(output.computation.get_operation(), ctx);
            for computation in output.embedded_computations {
                Operation::erase(computation.get_operation(), ctx);
            }
            Err(error)
        }
    }
}

fn ordered_contains_mapped_or_parameter(
    ctx: &Context,
    instruction: Ptr<Operation>,
    mapped: &HashSet<Ptr<Operation>>,
) -> bool {
    mapped.contains(&instruction)
        || Operation::is_op::<crate::ops::control::ParameterOp>(instruction, ctx)
}

fn build(
    ctx: &mut Context,
    computation: &ComputationOp,
    block: Ptr<BasicBlock>,
    replacements: &Replacements,
    plan: &Plan,
    async_callees: &AsyncCallees,
    options: Option<VerifyOptions>,
) -> Result<IrMapping> {
    let mut mapping = IrMapping::new();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    for &template in &plan.postorder {
        // Pre-map EVERY direct operand. clone_operation's same-context fallback
        // is intentionally unreachable, including for external dependencies.
        let operands: Vec<_> = template.deref(ctx).operands().collect();
        for operand in operands {
            let original = operand_definition(ctx, operand)?;
            let selected = selected(original, replacements)
                .ok_or_else(|| error("omitted operand after clone planning"))?;
            let cloned = mapping
                .lookup_op(selected)
                .ok_or_else(|| error("unmapped operand after clone planning"))?;
            mapping.map_value(operand, cloned.deref(ctx).get_result(0));
        }
        let cloned = clone_operation(template, ctx, &mut rewriter, &mut mapping);
        cloned.insert_at_back(block, ctx);
        normalize_nested_metadata(ctx, template, &mapping, plan)?;
        for original in nested_operations(ctx, &[template]) {
            if let Some(callees) = async_callees.get(&original) {
                let cloned = mapping
                    .lookup_op(original)
                    .ok_or_else(|| error("async start missing from clone mapping"))?;
                cloned
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_callees"), callees.clone());
            }
        }
        {
            let mut data = cloned.deref_mut(ctx);
            if let Some(controls) = plan.controls.get(&template) {
                let controls = controls
                    .iter()
                    .map(|predecessor| {
                        plan.names
                            .get(predecessor)
                            .cloned()
                            .ok_or_else(|| error("control predecessor lacks a cloned name"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                data.attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(controls),
                );
            } else if data
                .attributes
                .0
                .contains_key(&Identifier::new("hlo_control_predecessors"))
            {
                data.attributes.set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors::default(),
                );
            }
            if let Some(name) = plan.names.get(&template) {
                data.attributes
                    .set(Identifier::new("hlo_name"), InstructionName(name.clone()));
            }
        }
        if let Some(options) = options {
            let dynamic = Operation::get_op_dyn(cloned, ctx);
            op_cast::<dyn HloOpInterface>(dynamic.as_ref())
                .ok_or_else(|| error("cloned instruction lost HLO interface"))?
                .verify_with_options(ctx, options)?;
        }
    }
    let roots: Vec<_> = plan
        .roots
        .iter()
        .map(|&root| {
            mapping
                .lookup_op(root)
                .map(|cloned| cloned.deref(ctx).get_result(0))
                .ok_or_else(|| error("selected root has no clone"))
        })
        .collect::<Result<_>>()?;
    let root = if plan.tuple_root {
        let tuple = TupleOp::new(ctx, plan.result, roots);
        tuple.get_operation().insert_at_back(block, ctx);
        let name = plan
            .tuple_name
            .clone()
            .ok_or_else(|| error("tuple root has no planned unique name"))?;
        tuple
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name));
        tuple.result(ctx)
    } else {
        roots[0]
    };
    ReturnOp::new(ctx, root)
        .get_operation()
        .insert_at_back(block, ctx);
    computation.verify(ctx)?;
    topological_order(ctx, &ordered_instructions(ctx, computation)?)?;
    install_logical_order(ctx, computation, replacements, plan, &mapping)?;
    Ok(mapping)
}

fn instruction_name(ctx: &Context, operation: Ptr<Operation>) -> Result<String> {
    optional_attr::<InstructionName>(ctx, operation, "hlo_name")?
        .map(|name| name.0.clone())
        .ok_or_else(|| error("logical clone ordering requires a planned instruction name"))
}

fn mapped_member(
    ctx: &Context,
    value: Value,
    replacements: &Replacements,
    mapping: &IrMapping,
) -> Result<Option<Value>> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => {
            let Some(template) = selected(operation, replacements) else {
                return Ok(None);
            };
            Ok(mapping
                .lookup_op(template)
                .map(|op| op.deref(ctx).get_result(0)))
        }
        DefiningEntity::Block(_) => Ok(mapping.lookup_value(value)),
    }
}

/// Stable mapped-element ordering with unmatched members appended, as in
/// MappedPtrContainerSorter. Membership comes exclusively from the actual graph.
fn mapped_order(actual: &[Value], preferred: impl IntoIterator<Item = Value>) -> Vec<Value> {
    let members: HashSet<_> = actual.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut result: Vec<_> = preferred
        .into_iter()
        .filter(|value| members.contains(value) && seen.insert(*value))
        .collect();
    result.extend(actual.iter().copied().filter(|value| seen.insert(*value)));
    result
}

fn value_name(ctx: &Context, value: Value) -> Result<String> {
    match value.defining_entity() {
        DefiningEntity::Op(operation) => instruction_name(ctx, operation),
        DefiningEntity::Block(block) => {
            let index = block
                .deref(ctx)
                .arguments()
                .position(|argument| argument == value)
                .ok_or_else(|| error("logical clone parameter is missing"))?;
            super::fusion_parameters(ctx, block)?.0[index]
                .name
                .clone()
                .ok_or_else(|| error("logical clone parameter has no name"))
        }
    }
}

fn install_logical_order(
    ctx: &Context,
    computation: &ComputationOp,
    replacements: &Replacements,
    plan: &Plan,
    mapping: &IrMapping,
) -> Result<()> {
    let actual = ordered_instructions(ctx, computation)?;
    let mut seen = HashSet::new();
    let mut ordered = Vec::new();
    let mut after_last_parameter = 0;
    for &original in &plan.original {
        let Some(template) = selected(original, replacements) else {
            continue;
        };
        let cloned = mapping
            .lookup_op(template)
            .ok_or_else(|| error("logical instruction has no clone"))?;
        if seen.insert(cloned) {
            ordered.push(cloned);
        }
        if opcode(ctx, original)? == Opcode::Parameter {
            after_last_parameter = ordered.len();
        }
    }
    let mut parameters = Vec::new();
    let mut external = Vec::new();
    for &operation in &actual {
        if seen.insert(operation) {
            if opcode(ctx, operation)? == Opcode::Parameter {
                parameters.push(operation);
            } else {
                external.push(operation);
            }
        }
    }
    ordered.splice(after_last_parameter..after_last_parameter, parameters);
    ordered.extend(external);
    let mut order = InstructionOrder {
        instructions: ordered
            .iter()
            .map(|&operation| instruction_name(ctx, operation))
            .collect::<Result<_>>()?,
        ..Default::default()
    };
    // Before installing logical storage order, capture insertion-derived users
    // and control successors. Source-original permutations then take precedence.
    let mut user_orders = HashMap::new();
    let mut successor_orders = HashMap::new();
    for &operation in &actual {
        let value = operation.deref(ctx).get_result(0);
        user_orders.insert(value, instruction_order::users(ctx, value)?);
        successor_orders.insert(value, instruction_order::control_successors(ctx, value)?);
    }
    for &original in &plan.original {
        let Some(template) = selected(original, replacements) else {
            continue;
        };
        let cloned = mapping
            .lookup_op(template)
            .ok_or_else(|| error("logical source member has no clone"))?;
        let value = cloned.deref(ctx).get_result(0);
        let old_value = original.deref(ctx).get_result(0);
        let mapped = |values: Vec<Value>| -> Result<Vec<Value>> {
            values
                .into_iter()
                .map(|value| mapped_member(ctx, value, replacements, mapping))
                .collect::<Result<Vec<_>>>()
                .map(|values| values.into_iter().flatten().collect())
        };
        let incoming = instruction_order::control_predecessors(ctx, value)?;
        let preferred = mapped(instruction_order::control_predecessors(ctx, old_value)?)?;
        let incoming = mapped_order(&incoming, preferred);
        if !incoming.is_empty() {
            cloned.deref_mut(ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(
                    incoming
                        .into_iter()
                        .map(|value| value_name(ctx, value))
                        .collect::<Result<_>>()?,
                ),
            );
        }
        let users = user_orders
            .get(&value)
            .ok_or_else(|| error("cloned user ordering snapshot is missing"))?;
        user_orders.insert(
            value,
            mapped_order(users, mapped(instruction_order::users(ctx, old_value)?)?),
        );
        let successors = successor_orders
            .get(&value)
            .ok_or_else(|| error("cloned control ordering snapshot is missing"))?;
        successor_orders.insert(
            value,
            mapped_order(
                successors,
                mapped(instruction_order::control_successors(ctx, old_value)?)?,
            ),
        );
    }
    for &operation in &actual {
        let value = operation.deref(ctx).get_result(0);
        let subject = value_name(ctx, value)?;
        let members = user_orders
            .remove(&value)
            .ok_or_else(|| error("missing logical user order"))?;
        if !members.is_empty() {
            order.users.push(OrderPermutation {
                subject: subject.clone(),
                members: members
                    .into_iter()
                    .map(|member| value_name(ctx, member))
                    .collect::<Result<_>>()?,
            });
        }
        let members = successor_orders
            .remove(&value)
            .ok_or_else(|| error("missing logical successor order"))?;
        if !members.is_empty() {
            order.control_successors.push(OrderPermutation {
                subject,
                members: members
                    .into_iter()
                    .map(|member| value_name(ctx, member))
                    .collect::<Result<_>>()?,
            });
        }
    }
    for original in nested_operations(ctx, &plan.postorder) {
        if !Operation::is_op::<crate::ops::control::FusionOp>(original, ctx) {
            continue;
        }
        for region in original.deref(ctx).regions() {
            for source_block in region.deref(ctx).iter(ctx) {
                let cloned_owner = mapping
                    .lookup_op(original)
                    .ok_or_else(|| error("fusion ordering has no cloned owner"))?;
                let source_order = instruction_order::block_value_order(ctx, source_block)?;
                let cloned_order = source_order
                    .iter()
                    .map(|&value| {
                        mapping
                            .lookup_value(value)
                            .ok_or_else(|| error("fusion ordering member has no clone"))
                            .and_then(|value| value_name(ctx, value))
                    })
                    .collect::<Result<Vec<_>>>()?;
                order.blocks.push(OrderPermutation {
                    subject: instruction_name(ctx, cloned_owner)?,
                    members: cloned_order,
                });
                for source_value in source_order {
                    let cloned = mapping
                        .lookup_value(source_value)
                        .ok_or_else(|| error("fusion ordering value has no clone mapping"))?;
                    let subject = value_name(ctx, cloned)?;
                    for (members, records) in [
                        (
                            instruction_order::users(ctx, source_value)?,
                            &mut order.users,
                        ),
                        (
                            instruction_order::control_successors(ctx, source_value)?,
                            &mut order.control_successors,
                        ),
                    ] {
                        if members.is_empty() {
                            continue;
                        }
                        let members = members
                            .into_iter()
                            .map(|member| {
                                mapping
                                    .lookup_value(member)
                                    .ok_or_else(|| error("fusion ordering relation has no clone"))
                                    .and_then(|member| value_name(ctx, member))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        records.push(OrderPermutation {
                            subject: subject.clone(),
                            members,
                        });
                    }
                }
            }
        }
    }
    instruction_order::set_order(ctx, computation.get_operation(), order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Bytes, Callees, FrontendAttributes, Payload},
        ir::{BackendConfigPayload, ExecutionThread},
        ops::{
            control::{CallOp, ParameterOp},
            elementwise::{AddOp, ExpOp, NegateOp},
        },
        types::{ElementType, TensorType},
    };
    use pliron::{attribute::Attribute, printable::Printable};

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        instruction_order::register(&mut ctx);
        ctx
    }

    fn scalar(ctx: &Context) -> TypeHandle {
        TensorType::get(ctx, ElementType::F32, vec![]).unwrap()
    }

    fn attribute<T: Attribute>(
        ctx: &Context,
        operation: Ptr<Operation>,
        name: &'static str,
        value: T,
    ) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new(name), value);
    }

    fn computation(
        ctx: &mut Context,
        name: &'static str,
        parameter_count: u32,
    ) -> (ComputationOp, Ptr<BasicBlock>, Vec<Ptr<Operation>>) {
        let ty = scalar(ctx);
        let signature: TypeHandle =
            FunctionType::get(ctx, vec![ty; parameter_count as usize], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new(name), signature);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let parameters = (0..parameter_count)
            .map(|index| {
                let parameter = ParameterOp::new(ctx, ty, vec![], Index(index)).get_operation();
                parameter.insert_at_back(block, ctx);
                parameter
            })
            .collect();
        (computation, block, parameters)
    }

    fn returned(ctx: &mut Context, block: Ptr<BasicBlock>, root: Ptr<Operation>) {
        let value = root.deref(ctx).get_result(0);
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(block, ctx);
    }

    fn clone(
        ctx: &mut Context,
        source: &ComputationOp,
        replacements: &Replacements,
    ) -> Result<ComputationClone> {
        clone_with_replacements(
            ctx,
            source,
            Identifier::new("cloned"),
            replacements,
            Some(VerifyOptions::default()),
        )
    }

    fn assert_closed(ctx: &Context, computation: &ComputationOp) {
        let block = body(ctx, computation).unwrap();
        let instructions: HashSet<_> = block.deref(ctx).iter(ctx).collect();
        for instruction in &instructions {
            for operand in instruction.deref(ctx).operands() {
                assert!(instructions.contains(&operand_definition(ctx, operand).unwrap()));
            }
        }
    }

    #[test]
    fn unchanged_clone_preserves_attributes_and_does_not_capture_source_values() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let parameter = parameters[0].deref(&ctx).get_result(0);
        let root = NegateOp::new(&mut ctx, ty, vec![parameter]).get_operation();
        root.insert_at_back(block, &ctx);
        attribute(
            &ctx,
            source.get_operation(),
            "hlo_execution_thread",
            ExecutionThread("worker".into()),
        );
        attribute(
            &ctx,
            root,
            "hlo_frontend_attributes",
            FrontendAttributes::new([("key".into(), "value".into())]),
        );
        attribute(
            &ctx,
            root,
            "hlo_backend_config_payload",
            BackendConfigPayload(Payload::Value(Bytes(vec![1, 2, 3]))),
        );
        returned(&mut ctx, block, root);
        let before = source.get_operation().disp(&ctx).to_string();
        let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        assert_closed(&ctx, &output.computation);
        let cloned_root = output.mapping.lookup_op(root).unwrap();
        let mut cloned_attributes = cloned_root.deref(&ctx).attributes.clone();
        cloned_attributes.0.remove(&Identifier::new("hlo_name"));
        assert_eq!(root.deref(&ctx).attributes, cloned_attributes);
        assert_eq!(
            optional_attr::<ExecutionThread>(
                &ctx,
                output.computation.get_operation(),
                "hlo_execution_thread"
            )
            .unwrap()
            .unwrap()
            .0,
            "worker",
        );
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
    }

    #[test]
    fn omissions_replacements_extra_parameters_and_root_substitution() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 2);
        returned(&mut ctx, block, parameters[0]);
        let ty = scalar(&ctx);
        let replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
        let extra = ParameterOp::new(&mut ctx, ty, vec![], Index(1)).get_operation();
        let replacements = Replacements {
            instructions: HashMap::from([
                (parameters[0], Some(replacement)),
                (parameters[1], None),
            ]),
            extra_parameters: vec![extra],
            root: CloneRoot::Instruction(parameters[0]),
            ..Default::default()
        };
        let mut context = ComputationCloneContext::new(None, "clone".into());
        let output = clone_in_context(
            &mut ctx,
            &source,
            Identifier::new("cloned"),
            &replacements,
            Some(VerifyOptions::default()),
            &mut context,
        )
        .unwrap();
        assert!(output.mapping.lookup_op(parameters[1]).is_none());
        assert!(output.mapping.lookup_op(parameters[0]).is_none());
        assert!(output.mapping.lookup_op(replacement).is_some());
        assert!(output.mapping.lookup_op(extra).is_some());
        assert!(context.instruction(extra).is_none());
        assert_eq!(
            context.instruction(replacement),
            output.mapping.lookup_op(replacement)
        );
        assert_closed(&ctx, &output.computation);
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        Operation::erase(output.computation.get_operation(), &mut ctx);
        Operation::erase(replacement, &mut ctx);
        Operation::erase(extra, &mut ctx);
    }

    #[test]
    fn attached_extra_is_independent_of_operand_closure_and_shared_context() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        returned(&mut ctx, block, parameters[0]);
        let (external, external_block, external_parameters) = computation(&mut ctx, "external", 2);
        let extra = external_parameters[1];
        returned(&mut ctx, external_block, extra);
        attribute(
            &ctx,
            external_parameters[0],
            "hlo_name",
            InstructionName("external_input".into()),
        );
        attribute(
            &ctx,
            extra,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["external_input".into()]),
        );
        let source_before = source.get_operation().disp(&ctx).to_string();
        let external_before = external.get_operation().disp(&ctx).to_string();
        let uses_before = extra.deref(&ctx).get_result(0).num_uses(&ctx);
        let replacements = Replacements {
            extra_parameters: vec![extra],
            ..Default::default()
        };
        let mut shared = ComputationCloneContext::new(None, "clone".into());
        let output = clone_in_context(
            &mut ctx,
            &source,
            Identifier::new("cloned"),
            &replacements,
            Some(VerifyOptions::default()),
            &mut shared,
        )
        .unwrap();
        assert!(output.mapping.lookup_op(extra).is_some());
        assert!(shared.instruction(extra).is_none());
        assert_closed(&ctx, &output.computation);
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        Operation::erase(output.computation.get_operation(), &mut ctx);
        let ty = scalar(&ctx);
        let value = extra.deref(&ctx).get_result(0);
        let template = NegateOp::new(&mut ctx, ty, vec![value]).get_operation();
        let replacements = Replacements {
            extra_parameters: vec![extra],
            instructions: HashMap::from([(parameters[0], Some(template))]),
            ..Default::default()
        };
        let mut fresh = ComputationCloneContext::new(None, "clone".into());
        // The extra and DFS dependency are separate parameter occurrences.
        // Both carry index 1, so upstream's computation builder rejects them;
        // return an error before allocation rather than deduplicating them.
        let error = clone_in_context(
            &mut ctx,
            &source,
            Identifier::new("invalid"),
            &replacements,
            None,
            &mut fresh,
        )
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains("duplicate clone parameter index")
        );
        assert!(fresh.instructions.is_empty());
        assert!(fresh.computations.is_empty());
        assert!(fresh.instruction_names.is_empty());
        assert!(fresh.computation_names.is_empty());
        assert!(fresh.schedules.is_empty());
        Operation::erase(template, &mut ctx);
        assert_eq!(uses_before, extra.deref(&ctx).get_result(0).num_uses(&ctx));
        assert_eq!(source_before, source.get_operation().disp(&ctx).to_string());
        assert_eq!(
            external_before,
            external.get_operation().disp(&ctx).to_string()
        );
    }

    #[test]
    fn external_attached_operand_closure_is_copied_without_editing_its_owner() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        returned(&mut ctx, block, parameters[0]);
        let (external, external_block, external_parameters) = computation(&mut ctx, "external", 1);
        let ty = scalar(&ctx);
        let value = external_parameters[0].deref(&ctx).get_result(0);
        let dependency = NegateOp::new(&mut ctx, ty, vec![value]).get_operation();
        dependency.insert_at_back(external_block, &ctx);
        returned(&mut ctx, external_block, dependency);
        let dependency_value = dependency.deref(&ctx).get_result(0);
        let replacement = NegateOp::new(&mut ctx, ty, vec![dependency_value]).get_operation();
        let before = external.get_operation().disp(&ctx).to_string();
        let replacements = Replacements {
            instructions: HashMap::from([(parameters[0], Some(replacement))]),
            ..Default::default()
        };
        let output = clone(&mut ctx, &source, &replacements).unwrap();
        assert_eq!(before, external.get_operation().disp(&ctx).to_string());
        assert!(output.mapping.lookup_op(dependency).is_some());
        assert!(output.mapping.lookup_op(external_parameters[0]).is_some());
        assert_closed(&ctx, &output.computation);
        let logical =
            instruction_order::instruction_order(&ctx, output.computation.get_operation()).unwrap();
        // Pinned storage puts mapped replacement first and new nonparameters
        // last, despite the reverse physical SSA dependency order.
        assert_eq!(logical[0], output.mapping.lookup_op(replacement).unwrap());
        assert_eq!(
            logical.last(),
            output.mapping.lookup_op(dependency).as_ref()
        );
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        Operation::erase(output.computation.get_operation(), &mut ctx);
        let external_replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
        let mut replacements = replacements;
        replacements
            .instructions
            .insert(external_parameters[0], Some(external_replacement));
        let output = clone(&mut ctx, &source, &replacements).unwrap();
        assert!(output.mapping.lookup_op(external_parameters[0]).is_none());
        assert!(output.mapping.lookup_op(external_replacement).is_some());
        assert_eq!(before, external.get_operation().disp(&ctx).to_string());
        assert_closed(&ctx, &output.computation);
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        Operation::erase(output.computation.get_operation(), &mut ctx);
        Operation::erase(external_replacement, &mut ctx);
        Operation::erase(replacement, &mut ctx);
    }

    #[test]
    fn replacement_lookup_is_once_per_operand_edge() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let parameter = parameters[0].deref(&ctx).get_result(0);
        let old = NegateOp::new(&mut ctx, ty, vec![parameter]).get_operation();
        old.insert_at_back(block, &ctx);
        returned(&mut ctx, block, old);
        let replacement = NegateOp::new(&mut ctx, ty, vec![parameter]).get_operation();
        let new_parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
        let replacements = Replacements {
            instructions: HashMap::from([
                (old, Some(replacement)),
                (parameters[0], Some(new_parameter)),
            ]),
            ..Default::default()
        };
        let output = clone(&mut ctx, &source, &replacements).unwrap();
        assert_eq!(
            output
                .mapping
                .lookup_op(replacement)
                .unwrap()
                .deref(&ctx)
                .get_operand(0),
            output
                .mapping
                .lookup_op(new_parameter)
                .unwrap()
                .deref(&ctx)
                .get_result(0),
        );
        assert_closed(&ctx, &output.computation);
        Operation::erase(output.computation.get_operation(), &mut ctx);
        Operation::erase(replacement, &mut ctx);
        Operation::erase(new_parameter, &mut ctx);
    }

    #[test]
    fn tuple_root_supports_empty_and_repeated_elements() {
        for elements in [vec![], vec![0], vec![1, 0, 1]] {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 2);
            returned(&mut ctx, block, parameters[0]);
            let replacements = Replacements {
                root: CloneRoot::Tuple(elements.iter().map(|&i| parameters[i]).collect()),
                ..Default::default()
            };
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            let block = body(&ctx, &output.computation).unwrap();
            let instructions: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
            let tuple = instructions[instructions.len() - 2];
            assert_eq!(opcode(&ctx, tuple).unwrap(), Opcode::Tuple);
            assert_eq!(tuple.deref(&ctx).get_num_operands(), elements.len());
            assert_closed(&ctx, &output.computation);
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        }
    }

    #[test]
    fn omitted_used_operand_and_omitted_root_fail_before_source_changes() {
        for omit_root in [false, true] {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let value = parameters[0].deref(&ctx).get_result(0);
            let root = NegateOp::new(&mut ctx, ty, vec![value]).get_operation();
            root.insert_at_back(block, &ctx);
            returned(&mut ctx, block, root);
            let before = source.get_operation().disp(&ctx).to_string();
            let replacements = Replacements {
                instructions: HashMap::from([(if omit_root { root } else { parameters[0] }, None)]),
                ..Default::default()
            };
            assert!(clone(&mut ctx, &source, &replacements).is_err());
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        }
    }

    #[test]
    fn controls_follow_selected_predecessor_and_replaced_successor_identity() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 2);
        attribute(
            &ctx,
            parameters[0],
            "hlo_name",
            InstructionName("left".into()),
        );
        attribute(
            &ctx,
            parameters[1],
            "hlo_name",
            InstructionName("right".into()),
        );
        attribute(
            &ctx,
            parameters[1],
            "hlo_control_predecessors",
            ControlPredecessors(vec!["left".into()]),
        );
        returned(&mut ctx, block, parameters[1]);
        let ty = scalar(&ctx);
        let replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(1)).get_operation();
        attribute(
            &ctx,
            replacement,
            "hlo_name",
            InstructionName("right".into()),
        );
        let replacements = Replacements {
            instructions: HashMap::from([(parameters[1], Some(replacement))]),
            ..Default::default()
        };
        let output = clone(&mut ctx, &source, &replacements).unwrap();
        assert_eq!(
            optional_attr::<ControlPredecessors>(
                &ctx,
                output.mapping.lookup_op(replacement).unwrap(),
                "hlo_control_predecessors"
            )
            .unwrap()
            .unwrap()
            .0,
            vec!["left"],
        );
        assert!(replacement.deref(&ctx).get_parent_block().is_none());
        Operation::erase(output.computation.get_operation(), &mut ctx);
        Operation::erase(replacement, &mut ctx);
    }

    #[test]
    fn replacing_control_source_does_not_inherit_its_outgoing_edges() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 2);
        attribute(
            &ctx,
            parameters[0],
            "hlo_name",
            InstructionName("left".into()),
        );
        attribute(
            &ctx,
            parameters[1],
            "hlo_control_predecessors",
            ControlPredecessors(vec!["left".into()]),
        );
        returned(&mut ctx, block, parameters[1]);
        let ty = scalar(&ctx);
        let replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
        attribute(
            &ctx,
            replacement,
            "hlo_name",
            InstructionName("left".into()),
        );
        let mut replacements = Replacements {
            instructions: HashMap::from([(parameters[0], Some(replacement))]),
            ..Default::default()
        };
        for explicit_edge in [false, true] {
            if explicit_edge {
                replacements
                    .template_control_edges
                    .push((replacement, parameters[1]));
            }
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            let incoming = optional_attr::<ControlPredecessors>(
                &ctx,
                output.mapping.lookup_op(parameters[1]).unwrap(),
                "hlo_control_predecessors",
            )
            .unwrap()
            .unwrap()
            .0
            .clone();
            assert_eq!(incoming.len(), usize::from(explicit_edge));
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
        }
        Operation::erase(replacement, &mut ctx);
    }

    #[test]
    fn output_names_are_uniquified_but_duplicate_parameter_indices_fail() {
        for duplicate_name in [false, true] {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 2);
            returned(&mut ctx, block, parameters[0]);
            attribute(
                &ctx,
                parameters[0],
                "hlo_name",
                InstructionName("left".into()),
            );
            let ty = scalar(&ctx);
            let replacement = ParameterOp::new(
                &mut ctx,
                ty,
                vec![],
                Index(if duplicate_name { 1 } else { 0 }),
            )
            .get_operation();
            if duplicate_name {
                attribute(
                    &ctx,
                    replacement,
                    "hlo_name",
                    InstructionName("left".into()),
                );
            }
            let before = source.get_operation().disp(&ctx).to_string();
            let replacements = Replacements {
                instructions: HashMap::from([(parameters[1], Some(replacement))]),
                ..Default::default()
            };
            let result = clone(&mut ctx, &source, &replacements);
            if duplicate_name {
                let output = result.unwrap();
                let original = output.mapping.lookup_op(parameters[0]).unwrap();
                let replaced = output.mapping.lookup_op(replacement).unwrap();
                assert_ne!(
                    optional_attr::<InstructionName>(&ctx, original, "hlo_name")
                        .unwrap()
                        .unwrap()
                        .0,
                    optional_attr::<InstructionName>(&ctx, replaced, "hlo_name")
                        .unwrap()
                        .unwrap()
                        .0,
                );
                Operation::erase(output.computation.get_operation(), &mut ctx);
            } else {
                assert!(result.is_err());
            }
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            Operation::erase(replacement, &mut ctx);
        }
    }

    #[test]
    fn local_verification_error_drops_all_staging_operand_uses() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        returned(&mut ctx, block, parameters[0]);
        let ty = scalar(&ctx);
        let replacement = AddOp::new(&mut ctx, ty, vec![]).get_operation();
        let replacements = Replacements {
            instructions: HashMap::from([(parameters[0], Some(replacement))]),
            ..Default::default()
        };
        let before = source.get_operation().disp(&ctx).to_string();
        assert!(clone(&mut ctx, &source, &replacements).is_err());
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        // Erasing the template would assert if a failed clone retained its result.
        Operation::erase(replacement, &mut ctx);
        crate::verification::verify(&ctx, source.get_operation()).unwrap();
    }

    #[test]
    fn unary_accuracy_is_preserved_but_nonunary_accuracy_is_reset() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let value = parameters[0].deref(&ctx).get_result(0);
        let unary = ExpOp::new(&mut ctx, ty, vec![value]).get_operation();
        unary.insert_at_back(block, &ctx);
        let unary_result = unary.deref(&ctx).get_result(0);
        let binary = AddOp::new(&mut ctx, ty, vec![value, unary_result]).get_operation();
        binary.insert_at_back(block, &ctx);
        for operation in [unary, binary] {
            attribute(
                &ctx,
                operation,
                "hlo_result_accuracy",
                ResultAccuracy::Highest,
            );
        }
        returned(&mut ctx, block, binary);
        let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        for (operation, expected) in [
            (unary, ResultAccuracy::Highest),
            (binary, ResultAccuracy::Default),
        ] {
            assert_eq!(
                *optional_attr::<ResultAccuracy>(
                    &ctx,
                    output.mapping.lookup_op(operation).unwrap(),
                    "hlo_result_accuracy"
                )
                .unwrap()
                .unwrap(),
                expected,
            );
        }
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
    }

    #[test]
    fn named_callees_are_verified_in_an_isolated_complete_module_context() {
        let mut ctx = context();
        let (callee, block, parameters) = computation(&mut ctx, "callee", 1);
        returned(&mut ctx, block, parameters[0]);
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let value = parameters[0].deref(&ctx).get_result(0);
        let call = CallOp::new(
            &mut ctx,
            ty,
            vec![value],
            Callees(vec![Identifier::new("callee")]),
        )
        .get_operation();
        attribute(&ctx, call, "hlo_name", InstructionName("call".into()));
        call.insert_at_back(block, &ctx);
        returned(&mut ctx, block, call);
        let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
        let block = module
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        callee.get_operation().insert_at_back(block, &ctx);
        source.get_operation().insert_at_back(block, &ctx);
        crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        let before = module.get_operation().disp(&ctx).to_string();
        let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        assert_eq!(before, module.get_operation().disp(&ctx).to_string());
        assert!(
            output
                .computation
                .get_operation()
                .deref(&ctx)
                .get_parent_block()
                .is_none()
        );
        output
            .computation
            .get_operation()
            .insert_at_back(block, &ctx);
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
    }

    #[test]
    fn provenance_belongs_to_selected_template_and_shape_rebasing_is_explicit() {
        use crate::original_value::{OriginalArray, OriginalValueElement};
        let mut ctx = context();
        let scalar = scalar(&ctx);
        let tuple = TupleType::get(&ctx, vec![scalar, scalar]).unwrap();
        let signature = FunctionType::get(&ctx, vec![tuple], vec![tuple]).into();
        let source = ComputationOp::new(&mut ctx, Identifier::new("source"), signature);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(source.body(&ctx), &ctx);
        let parameter = ParameterOp::new(&mut ctx, tuple, vec![], Index(0)).get_operation();
        parameter.insert_at_back(block, &ctx);
        let provenance = |paths: Vec<Vec<i64>>| OriginalValue {
            elements: paths
                .into_iter()
                .map(|path| OriginalValueElement {
                    shape_index: path.clone(),
                    original_array: vec![OriginalArray {
                        instruction_name: "original_source".into(),
                        shape_index: path,
                    }],
                })
                .collect(),
            ..Default::default()
        };
        let original_value = provenance(vec![vec![0], vec![1]]);
        attribute(
            &ctx,
            parameter,
            "hlo_original_value",
            original_value.clone(),
        );
        returned(&mut ctx, block, parameter);
        let template = ParameterOp::new(&mut ctx, scalar, vec![], Index(0)).get_operation();
        attribute(&ctx, template, "hlo_original_value", original_value.clone());
        let replacements = Replacements {
            instructions: HashMap::from([(parameter, Some(template))]),
            ..Default::default()
        };
        assert!(clone(&mut ctx, &source, &replacements).is_err());
        let raw = clone_with_replacements(
            &mut ctx,
            &source,
            Identifier::new("raw"),
            &replacements,
            None,
        )
        .unwrap();
        assert_eq!(
            optional_attr::<OriginalValue>(
                &ctx,
                raw.mapping.lookup_op(template).unwrap(),
                "hlo_original_value",
            )
            .unwrap()
            .as_deref(),
            Some(&original_value),
        );
        assert!(crate::verification::verify(&ctx, raw.computation.get_operation()).is_err());
        Operation::erase(raw.computation.get_operation(), &mut ctx);
        let rebased = provenance(vec![vec![]]);
        attribute(&ctx, template, "hlo_original_value", rebased.clone());
        let output = clone(&mut ctx, &source, &replacements).unwrap();
        assert_eq!(
            *optional_attr::<OriginalValue>(
                &ctx,
                output.mapping.lookup_op(template).unwrap(),
                "hlo_original_value"
            )
            .unwrap()
            .unwrap(),
            rebased,
        );
        assert_eq!(
            optional_attr::<OriginalValue>(&ctx, parameter, "hlo_original_value")
                .unwrap()
                .unwrap()
                .elements
                .len(),
            2,
        );
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        Operation::erase(output.computation.get_operation(), &mut ctx);
        Operation::erase(template, &mut ctx);
    }

    #[test]
    fn async_start_clones_wrapped_ownership_instead_of_sharing_original() {
        use crate::ops::control::{AsyncDoneOp, AsyncStartOp, AsyncUpdateOp};
        let mut ctx = context();
        let (work, block, parameters) = computation(&mut ctx, "work", 1);
        returned(&mut ctx, block, parameters[0]);
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let state = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
        let inputs = TupleType::get(&ctx, vec![ty]).unwrap();
        let async_type = TupleType::get(&ctx, vec![inputs, ty, state]).unwrap();
        let input = parameters[0].deref(&ctx).get_result(0);
        let start = AsyncStartOp::new(
            &mut ctx,
            async_type,
            vec![input],
            Callees(vec![Identifier::new("work")]),
        );
        attribute(
            &ctx,
            start.get_operation(),
            "hlo_name",
            InstructionName("async_start".into()),
        );
        start.get_operation().insert_at_back(block, &ctx);
        let value = start.result(&ctx);
        let update = AsyncUpdateOp::new(&mut ctx, async_type, vec![value]);
        update.get_operation().insert_at_back(block, &ctx);
        let value = update.result(&ctx);
        let done = AsyncDoneOp::new(&mut ctx, ty, vec![value]);
        attribute(
            &ctx,
            done.get_operation(),
            "hlo_name",
            InstructionName("async_done".into()),
        );
        done.get_operation().insert_at_back(block, &ctx);
        returned(&mut ctx, block, done.get_operation());
        let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
        let block = module
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        for computation in [&work, &source] {
            computation.get_operation().insert_at_back(block, &ctx);
        }
        crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        let before = module.get_operation().disp(&ctx).to_string();
        let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        assert_eq!(output.embedded_computations.len(), 1);
        assert_eq!(before, module.get_operation().disp(&ctx).to_string());
        let cloned_start = output.mapping.lookup_op(start.get_operation()).unwrap();
        assert_ne!(
            optional_attr::<Callees>(&ctx, cloned_start, "hlo_callees")
                .unwrap()
                .unwrap()
                .0[0],
            Identifier::new("work"),
        );
        for embedded in &output.embedded_computations {
            embedded.get_operation().insert_at_back(block, &ctx);
        }
        output
            .computation
            .get_operation()
            .insert_at_back(block, &ctx);
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        let second = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        for original in [start.get_operation(), done.get_operation()] {
            let first_name = optional_attr::<InstructionName>(
                &ctx,
                output.mapping.lookup_op(original).unwrap(),
                "hlo_name",
            )
            .unwrap()
            .unwrap()
            .0
            .clone();
            let second_name = optional_attr::<InstructionName>(
                &ctx,
                second.mapping.lookup_op(original).unwrap(),
                "hlo_name",
            )
            .unwrap()
            .unwrap()
            .0
            .clone();
            assert_ne!(first_name, second_name);
        }
        assert_ne!(
            output.embedded_computations[0].get_operation(),
            second.embedded_computations[0].get_operation(),
        );
        for computation in &second.embedded_computations {
            computation.get_operation().insert_at_back(block, &ctx);
        }
        second
            .computation
            .get_operation()
            .insert_at_back(block, &ctx);
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
    }

    #[test]
    fn scheduled_companion_maps_order_and_inserts_extra_parameters_after_existing_ones() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let input = parameters[0].deref(&ctx).get_result(0);
        let root = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
        root.insert_at_back(block, &ctx);
        attribute(
            &ctx,
            parameters[0],
            "hlo_name",
            InstructionName("input".into()),
        );
        attribute(&ctx, root, "hlo_name", InstructionName("root".into()));
        returned(&mut ctx, block, root);
        let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
        let block = module
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        source.get_operation().insert_at_back(block, &ctx);
        crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
        attribute(
            &ctx,
            module.get_operation(),
            "hlo_schedule",
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("source"),
                instructions: vec!["input".into(), "root".into()],
            }]),
        );
        let extra = ParameterOp::new(&mut ctx, ty, vec![], Index(1)).get_operation();
        let replacements = Replacements {
            extra_parameters: vec![extra],
            ..Default::default()
        };
        let before = module.get_operation().disp(&ctx).to_string();
        let output = clone_with_schedule_and_replacements(
            &mut ctx,
            &source,
            Identifier::new("cloned"),
            &replacements,
            Some(VerifyOptions::default()),
        )
        .unwrap();
        assert_eq!(before, module.get_operation().disp(&ctx).to_string());
        let mapped_name = |original| {
            optional_attr::<InstructionName>(
                &ctx,
                output.cloned.mapping.lookup_op(original).unwrap(),
                "hlo_name",
            )
            .unwrap()
            .unwrap()
            .0
            .clone()
        };
        assert_eq!(
            output.sequence.instructions,
            vec![
                mapped_name(parameters[0]),
                mapped_name(extra),
                mapped_name(root)
            ],
        );
        output
            .cloned
            .computation
            .get_operation()
            .insert_at_back(block, &ctx);
        let schedule = Schedule(vec![
            InstructionSequence {
                computation: Identifier::new("source"),
                instructions: vec!["input".into(), "root".into()],
            },
            output.sequence,
        ]);
        attribute(&ctx, module.get_operation(), "hlo_schedule", schedule);
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        Operation::erase(extra, &mut ctx);
    }

    #[test]
    fn raw_schedule_preserves_mapped_order_and_verified_failure_rolls_back_for_recovery() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let input = parameters[0].deref(&ctx).get_result(0);
        let root = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
        root.insert_at_back(block, &ctx);
        returned(&mut ctx, block, root);
        attribute(
            &ctx,
            parameters[0],
            "hlo_name",
            InstructionName("input".into()),
        );
        attribute(&ctx, root, "hlo_name", InstructionName("root".into()));
        let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
        let module_block = module
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        source.get_operation().insert_at_back(module_block, &ctx);
        crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
        attribute(
            &ctx,
            module.get_operation(),
            "hlo_schedule",
            Schedule(vec![InstructionSequence {
                computation: Identifier::new("source"),
                instructions: vec!["input".into(), "root".into()],
            }]),
        );
        crate::verification::verify(&ctx, module.get_operation()).unwrap();
        let dependency = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
        let dependency_value = dependency.deref(&ctx).get_result(0);
        let replacement = NegateOp::new(&mut ctx, ty, vec![dependency_value]).get_operation();
        let replacements = Replacements {
            instructions: HashMap::from([(root, Some(replacement))]),
            root: CloneRoot::Tuple(vec![root]),
            ..Default::default()
        };
        let before = module.get_operation().disp(&ctx).to_string();
        let mut raw_sequences = Vec::new();
        for _ in 0..2 {
            let raw = clone_with_schedule_and_replacements(
                &mut ctx,
                &source,
                Identifier::new("cloned"),
                &replacements,
                None,
            )
            .unwrap();
            let mapped: Vec<_> = [parameters[0], replacement, dependency]
                .into_iter()
                .map(|original| {
                    instruction_name(&ctx, raw.cloned.mapping.lookup_op(original).unwrap()).unwrap()
                })
                .collect();
            assert_eq!(&raw.sequence.instructions[..3], mapped);
            assert_eq!(raw.sequence.instructions.len(), 4);
            assert_ne!(
                raw.sequence.instructions,
                sequence(
                    &ctx,
                    &raw.cloned.computation,
                    &topological_order(
                        &ctx,
                        &ordered_instructions(&ctx, &raw.cloned.computation).unwrap()
                    )
                    .unwrap(),
                )
                .unwrap()
                .instructions
            );
            assert_closed(&ctx, &raw.cloned.computation);
            raw_sequences.push(raw.sequence);
            Operation::erase(raw.cloned.computation.get_operation(), &mut ctx);
        }
        assert_eq!(raw_sequences[0], raw_sequences[1]);
        let mut shared = ComputationCloneContext::new(Some(module.get_operation()), "clone".into());
        for _ in 0..2 {
            assert!(
                clone_with_schedule_in_context(
                    &mut ctx,
                    &source,
                    Identifier::new("cloned"),
                    &replacements,
                    Some(VerifyOptions::default()),
                    &mut shared,
                )
                .is_err()
            );
            assert!(shared.computations.is_empty());
            assert!(shared.instructions.is_empty());
            assert!(shared.computations_by_name.is_empty());
            assert!(shared.instructions_by_name.is_empty());
            assert!(shared.instruction_names.is_empty());
            assert!(shared.computation_names.is_empty());
            assert!(shared.schedules.is_empty());
            assert!(shared.in_progress.is_empty());
            assert_eq!(before, module.get_operation().disp(&ctx).to_string());
        }
        let mut fresh = ComputationCloneContext::new(Some(module.get_operation()), "clone".into());
        let mut recovered_sequences = Vec::new();
        for context in [&mut shared, &mut fresh] {
            let recovered = clone_with_schedule_in_context(
                &mut ctx,
                &source,
                Identifier::new("cloned"),
                &Replacements::default(),
                Some(VerifyOptions::default()),
                context,
            )
            .unwrap();
            assert_eq!(
                context.instruction(root),
                recovered.cloned.mapping.lookup_op(root)
            );
            assert_eq!(context.schedules, vec![recovered.sequence.clone()]);
            recovered_sequences.push(recovered.sequence);
            Operation::erase(recovered.cloned.computation.get_operation(), &mut ctx);
        }
        assert_eq!(recovered_sequences[0], recovered_sequences[1]);
        assert_eq!(before, module.get_operation().disp(&ctx).to_string());
        Operation::erase(replacement, &mut ctx);
        Operation::erase(dependency, &mut ctx);
    }

    #[test]
    fn raw_clone_accepts_semantically_intermediate_ir_without_a_hidden_verifier() {
        let mut ctx = context();
        let (source, block, _) = computation(&mut ctx, "source", 0);
        let ty = scalar(&ctx);
        let intermediate = AddOp::new(&mut ctx, ty, vec![]).get_operation();
        intermediate.insert_at_back(block, &ctx);
        returned(&mut ctx, block, intermediate);
        assert!(crate::verification::verify(&ctx, source.get_operation()).is_err());
        let before = source.get_operation().disp(&ctx).to_string();
        let output = clone_with_replacements(
            &mut ctx,
            &source,
            Identifier::new("cloned"),
            &Replacements::default(),
            None,
        )
        .unwrap();
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        assert!(crate::verification::verify(&ctx, output.computation.get_operation()).is_err());
        assert_closed(&ctx, &output.computation);
        assert!(clone(&mut ctx, &source, &Replacements::default()).is_err());
    }

    #[test]
    fn mapped_order_matrix_is_stable_with_aliases_omissions_and_unmapped_members() {
        let mut ctx = context();
        let (_, _, parameters) = computation(&mut ctx, "ordering", 5);
        let values: Vec<_> = parameters
            .iter()
            .map(|parameter| parameter.deref(&ctx).get_result(0))
            .collect();
        for (preferred, expected) in [
            (vec![], vec![0, 1, 2, 3]),
            (vec![3, 1, 0, 2], vec![3, 1, 0, 2]),
            (vec![2, 2, 0, 0], vec![2, 0, 1, 3]),
            (vec![4, 2, 4, 0], vec![2, 0, 1, 3]),
            (vec![3, 4, 1, 3, 2], vec![3, 1, 2, 0]),
        ] {
            assert_eq!(
                mapped_order(
                    &values[..4],
                    preferred.into_iter().map(|index| values[index])
                ),
                expected
                    .into_iter()
                    .map(|index| values[index])
                    .collect::<Vec<_>>(),
            );
        }
    }

    #[test]
    fn clone_preserves_explicit_source_user_and_successor_permutations() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let input = parameters[0].deref(&ctx).get_result(0);
        let first = AddOp::new(&mut ctx, ty, vec![input, input]).get_operation();
        first.insert_at_back(block, &ctx);
        let second = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
        second.insert_at_back(block, &ctx);
        for (operation, name) in [(parameters[0], "p"), (first, "first"), (second, "second")] {
            attribute(&ctx, operation, "hlo_name", InstructionName(name.into()));
        }
        for operation in [first, second] {
            attribute(
                &ctx,
                operation,
                "hlo_control_predecessors",
                ControlPredecessors(vec!["p".into()]),
            );
        }
        returned(&mut ctx, block, second);
        instruction_order::set_order(
            &ctx,
            source.get_operation(),
            InstructionOrder {
                users: vec![OrderPermutation {
                    subject: "p".into(),
                    members: vec!["second".into(), "first".into()],
                }],
                control_successors: vec![OrderPermutation {
                    subject: "p".into(),
                    members: vec!["second".into(), "first".into()],
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let before = source.get_operation().disp(&ctx).to_string();
        let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        let parameter = output
            .mapping
            .lookup_op(parameters[0])
            .unwrap()
            .deref(&ctx)
            .get_result(0);
        let expected: Vec<_> = [second, first]
            .into_iter()
            .map(|operation| {
                output
                    .mapping
                    .lookup_op(operation)
                    .unwrap()
                    .deref(&ctx)
                    .get_result(0)
            })
            .collect();
        assert_eq!(instruction_order::users(&ctx, parameter).unwrap(), expected);
        assert_eq!(
            instruction_order::control_successors(&ctx, parameter).unwrap(),
            expected
        );
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        instruction_order::verify_order(&ctx, output.computation.get_operation()).unwrap();
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
    }

    #[test]
    fn omitted_control_successors_are_filtered_and_unresolved_edges_error() {
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 2);
        attribute(&ctx, parameters[0], "hlo_name", InstructionName("p".into()));
        attribute(
            &ctx,
            parameters[1],
            "hlo_name",
            InstructionName("unused".into()),
        );
        attribute(
            &ctx,
            parameters[1],
            "hlo_control_predecessors",
            ControlPredecessors(vec!["p".into()]),
        );
        returned(&mut ctx, block, parameters[0]);
        let before = source.get_operation().disp(&ctx).to_string();
        let replacements = Replacements {
            instructions: HashMap::from([(parameters[1], None)]),
            ..Default::default()
        };
        let output = clone(&mut ctx, &source, &replacements).unwrap();
        let parameter = output
            .mapping
            .lookup_op(parameters[0])
            .unwrap()
            .deref(&ctx)
            .get_result(0);
        assert!(
            instruction_order::control_successors(&ctx, parameter)
                .unwrap()
                .is_empty()
        );
        assert!(output.mapping.lookup_op(parameters[1]).is_none());
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        Operation::erase(output.computation.get_operation(), &mut ctx);
        attribute(
            &ctx,
            parameters[1],
            "hlo_control_predecessors",
            ControlPredecessors(vec!["missing".into()]),
        );
        let malformed = source.get_operation().disp(&ctx).to_string();
        // Raw cloning must still reject ambiguous/unresolved symbolic graph
        // references, independent of optional semantic verification.
        assert!(
            clone_with_replacements(
                &mut ctx,
                &source,
                Identifier::new("raw"),
                &Replacements::default(),
                None,
            )
            .is_err()
        );
        assert_eq!(malformed, source.get_operation().disp(&ctx).to_string());
    }

    #[test]
    fn cross_module_context_clones_foreign_callees_once_and_commits_mappings_on_success() {
        let mut ctx = context();
        let (callee, block, parameters) = computation(&mut ctx, "callee", 1);
        returned(&mut ctx, block, parameters[0]);
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let input = parameters[0].deref(&ctx).get_result(0);
        let call = CallOp::new(
            &mut ctx,
            ty,
            vec![input],
            Callees(vec![Identifier::new("callee")]),
        )
        .get_operation();
        attribute(
            &ctx,
            call,
            "hlo_name",
            InstructionName("foreign_call".into()),
        );
        call.insert_at_back(block, &ctx);
        returned(&mut ctx, block, call);
        let source_module = ModuleOp::new(&mut ctx, Identifier::new("foreign"));
        let source_block = source_module
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        callee.get_operation().insert_at_back(source_block, &ctx);
        source.get_operation().insert_at_back(source_block, &ctx);
        crate::ir::set_entry(
            &ctx,
            source_module.get_operation(),
            Identifier::new("source"),
        )
        .unwrap();
        let (entry, block, parameters) = computation(&mut ctx, "entry", 1);
        returned(&mut ctx, block, parameters[0]);
        let destination = ModuleOp::new(&mut ctx, Identifier::new("destination"));
        let destination_block = destination
            .get_operation()
            .deref(&ctx)
            .get_region(0)
            .deref(&ctx)
            .get_entry_block()
            .unwrap();
        entry
            .get_operation()
            .insert_at_back(destination_block, &ctx);
        crate::ir::set_entry(&ctx, destination.get_operation(), Identifier::new("entry")).unwrap();
        let before_source = source_module.get_operation().disp(&ctx).to_string();
        let before_destination = destination.get_operation().disp(&ctx).to_string();
        let mut context =
            ComputationCloneContext::new(Some(destination.get_operation()), "clone".into());
        let first = clone_in_context(
            &mut ctx,
            &source,
            Identifier::new("copy"),
            &Replacements::default(),
            Some(VerifyOptions::default()),
            &mut context,
        )
        .unwrap();
        assert_eq!(first.embedded_computations.len(), 1);
        assert_eq!(context.instruction(call), first.mapping.lookup_op(call));
        assert_eq!(
            context.instruction_by_name("foreign_call"),
            first.mapping.lookup_op(call)
        );
        assert_eq!(
            context.computation_by_name("source"),
            Some(first.computation.get_operation())
        );
        let shared_callee = context.computation(callee.get_operation()).unwrap();
        let second = clone_in_context(
            &mut ctx,
            &source,
            Identifier::new("copy"),
            &Replacements::default(),
            Some(VerifyOptions::default()),
            &mut context,
        )
        .unwrap();
        assert!(second.embedded_computations.is_empty());
        assert_eq!(
            context.computation(callee.get_operation()),
            Some(shared_callee)
        );
        assert_eq!(context.instruction(call), second.mapping.lookup_op(call));
        assert_eq!(
            context.instruction_by_name("foreign_call"),
            second.mapping.lookup_op(call)
        );
        assert_eq!(
            context.computation_by_name("source"),
            Some(second.computation.get_operation())
        );
        let names = context.computation_names.clone();
        let mappings = context.computations.clone();
        let invalid = Replacements {
            instructions: HashMap::from([(call, None)]),
            ..Default::default()
        };
        assert!(
            clone_in_context(
                &mut ctx,
                &source,
                Identifier::new("failed"),
                &invalid,
                None,
                &mut context,
            )
            .is_err()
        );
        assert_eq!(context.computation_names, names);
        assert_eq!(context.computations, mappings);
        assert_eq!(
            before_source,
            source_module.get_operation().disp(&ctx).to_string()
        );
        assert_eq!(
            before_destination,
            destination.get_operation().disp(&ctx).to_string()
        );
        for computation in &first.embedded_computations {
            computation
                .get_operation()
                .insert_at_back(destination_block, &ctx);
        }
        for output in [&first, &second] {
            output
                .computation
                .get_operation()
                .insert_at_back(destination_block, &ctx);
        }
        crate::verification::verify(&ctx, destination.get_operation()).unwrap();
        crate::verification::verify(&ctx, source_module.get_operation()).unwrap();
    }

    #[test]
    fn inline_fusion_clones_block_arguments_controls_and_nested_accuracy() {
        use crate::{
            attributes::{FusionKind, OpMetadata},
            ir::FusionParameter,
            ops::control::FusionOp,
            original_value::{OriginalArray, OriginalValueElement},
            sharding::Sharding,
        };
        let mut ctx = context();
        let (source, block, parameters) = computation(&mut ctx, "source", 1);
        let ty = scalar(&ctx);
        let input = parameters[0].deref(&ctx).get_result(0);
        let fusion = FusionOp::new(&mut ctx, ty, vec![input], FusionKind::Loop);
        fusion.get_operation().insert_at_back(block, &ctx);
        let fusion_block = BasicBlock::new(&mut ctx, None, vec![ty]);
        fusion_block.insert_at_back(fusion.body(&ctx), &ctx);
        let argument = fusion_block.deref(&ctx).get_argument(0);
        let root = ExpOp::new(&mut ctx, ty, vec![argument]).get_operation();
        root.insert_at_back(fusion_block, &ctx);
        attribute(&ctx, root, "hlo_name", InstructionName("inside".into()));
        attribute(&ctx, root, "hlo_result_accuracy", ResultAccuracy::Highest);
        attribute(
            &ctx,
            root,
            "hlo_control_predecessors",
            ControlPredecessors(vec!["argument".into()]),
        );
        crate::ir::set_fusion_parameters(
            &ctx,
            fusion_block,
            FusionParameters(vec![FusionParameter {
                name: Some("argument".into()),
                frontend_attributes: FrontendAttributes::new([(
                    "annotation".into(),
                    "preserved".into(),
                )]),
                metadata: Some(OpMetadata {
                    op_name: "original_argument".into(),
                    source_file: "clone_fixture".into(),
                    source_line: 7,
                    ..Default::default()
                }),
                sharding: Some(Sharding::replicated()),
                original_value: Some(OriginalValue {
                    elements: vec![OriginalValueElement {
                        shape_index: vec![],
                        original_array: vec![OriginalArray {
                            instruction_name: "original_argument".into(),
                            shape_index: vec![],
                        }],
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            }]),
        )
        .unwrap();
        returned(&mut ctx, fusion_block, root);
        returned(&mut ctx, block, fusion.get_operation());
        attribute(
            &ctx,
            fusion.get_operation(),
            "hlo_name",
            InstructionName("fusion".into()),
        );
        instruction_order::set_order(
            &ctx,
            source.get_operation(),
            InstructionOrder {
                blocks: vec![OrderPermutation {
                    subject: "fusion".into(),
                    members: vec!["inside".into(), "argument".into()],
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let before = source.get_operation().disp(&ctx).to_string();
        let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
        let cloned_block = output.mapping.lookup_block(fusion_block).unwrap();
        let metadata = crate::ir::fusion_parameters(&ctx, cloned_block).unwrap();
        let mut expected = crate::ir::fusion_parameters(&ctx, fusion_block).unwrap();
        expected.0[0].name.clone_from(&metadata.0[0].name);
        assert_eq!(metadata, expected);
        let cloned_root = output.mapping.lookup_op(root).unwrap();
        assert_eq!(
            instruction_order::block_value_order(&ctx, cloned_block).unwrap(),
            vec![
                cloned_root.deref(&ctx).get_result(0),
                cloned_block.deref(&ctx).get_argument(0)
            ]
        );
        assert_eq!(
            optional_attr::<ControlPredecessors>(&ctx, cloned_root, "hlo_control_predecessors")
                .unwrap()
                .unwrap()
                .0,
            vec![metadata.0[0].name.clone().unwrap()],
        );
        assert_eq!(
            *optional_attr::<ResultAccuracy>(&ctx, cloned_root, "hlo_result_accuracy")
                .unwrap()
                .unwrap(),
            ResultAccuracy::Highest,
        );
        assert_eq!(
            cloned_root.deref(&ctx).get_operand(0),
            cloned_block.deref(&ctx).get_argument(0)
        );
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        Operation::erase(output.computation.get_operation(), &mut ctx);
        let mut invalid = crate::ir::fusion_parameters(&ctx, fusion_block).unwrap();
        invalid.0[0].original_value.as_mut().unwrap().elements[0].shape_index = vec![0];
        fusion_block
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new("hlo_fusion_parameters"), invalid.clone());
        let before = source.get_operation().disp(&ctx).to_string();
        assert!(clone(&mut ctx, &source, &Replacements::default()).is_err());
        let raw = clone_with_replacements(
            &mut ctx,
            &source,
            Identifier::new("raw_fusion"),
            &Replacements::default(),
            None,
        )
        .unwrap();
        let raw_block = raw.mapping.lookup_block(fusion_block).unwrap();
        assert_eq!(
            crate::ir::fusion_parameters(&ctx, raw_block).unwrap().0[0].original_value,
            invalid.0[0].original_value,
        );
        assert!(crate::verification::verify(&ctx, raw.computation.get_operation()).is_err());
        assert_eq!(before, source.get_operation().disp(&ctx).to_string());
        Operation::erase(raw.computation.get_operation(), &mut ctx);
    }
}
