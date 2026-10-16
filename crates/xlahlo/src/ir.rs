use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Error, Opcode, Result, Shape,
    proto::{
        hlo::{HloComputationProto, HloInstructionProto, HloModuleProto},
        xla_data::ProgramShapeProto,
    },
};

pub(crate) mod channels;
mod effects;
pub(crate) mod names;
mod parameters;
pub use effects::EffectContext;

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn execution_thread(name: &str) -> &str {
    if name.is_empty() { "main" } else { name }
}

fn owner() -> u64 {
    NEXT_OWNER
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("module identity space exhausted")
}

/// A handle belongs to one module instance, including across coordinated edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputationId {
    owner: u64,
    wire: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstructionId {
    owner: u64,
    computation: i64,
    wire: i64,
}

#[derive(Debug)]
pub struct ClonedComputation {
    pub computation: ComputationId,
    pub instructions: BTreeMap<InstructionId, InstructionId>,
}

/// A checked instruction prototype. Its references are supplied on insertion.
/// All pinned typed operation/metadata fields are available through the schema.
#[derive(Clone, Debug, PartialEq)]
pub struct Instruction {
    wire: HloInstructionProto,
    explicit_channel: bool,
}

impl Instruction {
    pub fn new(name: impl Into<String>, opcode: Opcode, shape: Shape) -> Self {
        Self {
            wire: HloInstructionProto {
                name: name.into(),
                opcode: opcode.name().into(),
                shape: Some(shape.into_proto()).into(),
                ..Default::default()
            },
            explicit_channel: false,
        }
    }

    pub fn from_proto(wire: HloInstructionProto) -> Result<Self> {
        validate_instruction_data(&wire)?;
        Ok(Self {
            wire,
            explicit_channel: false,
        })
    }

    pub fn as_proto(&self) -> &HloInstructionProto {
        &self.wire
    }
    /// Export raw wire fields. Like native HLO, binary reimport loses explicit
    /// zero/negative channel presence because the channel field is proto3 scalar.
    pub fn into_proto(self) -> HloInstructionProto {
        self.wire
    }

    /// Edit schema-backed payloads without access to graph storage. Insertion
    /// validates names, shapes, opcodes and references again.
    /// Changing channel scalar fields resets presence to wire-import rules;
    /// use `with_channel_id` to preserve explicit nonpositive presence.
    pub fn with_fields(mut self, edit: impl FnOnce(&mut HloInstructionProto)) -> Result<Self> {
        let channel = (self.wire.channel_id, self.wire.all_reduce_id);
        edit(&mut self.wire);
        let explicit =
            self.explicit_channel && channel == (self.wire.channel_id, self.wire.all_reduce_id);
        if explicit && !channels::supported(&self.wire.opcode) {
            return Err(Error::new("channel", "opcode does not carry a channel"));
        }
        let mut updated = Self::from_proto(self.wire)?;
        updated.explicit_channel = explicit;
        Ok(updated)
    }
}

/// Supported module configuration stored outside `HloModuleProto`, as in XLA.
/// Counts remain unchecked until semantic verification, like unverified text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleConfig {
    pub replica_count: i64,
    pub num_partitions: i64,
    pub use_spmd_partitioning: bool,
    pub alias_passthrough_params: bool,
    pub allow_spmd_sharding_propagation_to_parameters: Vec<bool>,
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            replica_count: 1,
            num_partitions: 1,
            use_spmd_partitioning: false,
            alias_passthrough_params: false,
            allow_spmd_sharding_propagation_to_parameters: vec![false],
            allow_spmd_sharding_propagation_to_output: vec![false],
        }
    }
}

/// An owned HLO graph. Wire-only fields, payload bytes and unknown fields survive
/// import/export. Import validates structure, not every operation's semantics.
#[derive(Debug)]
pub struct Module {
    pub(crate) wire: HloModuleProto,
    pub(crate) channel_presence: BTreeSet<i64>,
    config: ModuleConfig,
    owner: u64,
    next_instruction: i64,
    next_computation: i64,
}

pub(crate) fn computation_program_shape(
    computation: &HloComputationProto,
) -> Result<ProgramShapeProto> {
    let mut parameters: Vec<_> = computation
        .instructions
        .iter()
        .filter(|i| i.opcode == "parameter")
        .collect();
    parameters.sort_by_key(|i| i.parameter_number);
    let mut program = ProgramShapeProto::new();
    for (index, parameter) in parameters.iter().enumerate() {
        if parameter.parameter_number != index as i64 {
            return Err(Error::new(
                "parameter",
                "parameter numbers must be consecutive from zero",
            ));
        }
        program.parameters.push(
            parameter
                .shape
                .as_ref()
                .ok_or_else(|| Error::new("shape", "missing parameter shape"))?
                .clone(),
        );
        program.parameter_names.push(parameter.name.clone());
    }
    program.result = computation
        .instructions
        .iter()
        .find(|i| i.id == computation.root_id)
        .ok_or_else(|| Error::new("reference", "missing computation root"))?
        .shape
        .clone();
    Ok(program)
}

impl Module {
    pub(crate) fn clone_for_printing(&self) -> Self {
        Self {
            wire: self.wire.clone(),
            channel_presence: self.channel_presence.clone(),
            config: self.config.clone(),
            owner: self.owner,
            next_instruction: self.next_instruction,
            next_computation: self.next_computation,
        }
    }

    pub fn new(name: impl Into<String>) -> Self {
        Self {
            wire: HloModuleProto {
                name: name.into(),
                ..Default::default()
            },
            owner: owner(),
            channel_presence: BTreeSet::new(),
            config: ModuleConfig::default(),
            next_instruction: 0,
            next_computation: 0,
        }
    }

    /// Import graph data with default configuration. `HloModuleProto` does not
    /// contain owned configuration; use `from_proto_with_config` to retain it.
    pub fn from_proto(wire: HloModuleProto) -> Result<Self> {
        Self::from_proto_with_config(wire, ModuleConfig::default())
    }

    pub fn from_proto_with_config(wire: HloModuleProto, config: ModuleConfig) -> Result<Self> {
        let mut module = Self {
            wire,
            channel_presence: BTreeSet::new(),
            config,
            owner: owner(),
            next_instruction: 0,
            next_computation: 0,
        };
        module.refresh_computation_kinds();
        module.validate_structure()?;
        module.next_computation = next_id(module.wire.computations.iter().map(|c| c.id))?;
        module.next_instruction = next_id(
            module
                .wire
                .computations
                .iter()
                .flat_map(|c| c.instructions.iter().map(|i| i.id)),
        )?;
        Ok(module)
    }

    pub fn name(&self) -> &str {
        &self.wire.name
    }
    pub fn config(&self) -> ModuleConfig {
        self.config.clone()
    }
    pub fn set_config(&mut self, config: ModuleConfig) {
        self.config = config;
    }
    /// Graph data only. Module configuration must be transferred separately.
    pub fn as_proto(&self) -> &HloModuleProto {
        &self.wire
    }

    /// Export in dependency order, retaining all wire metadata. Configuration
    /// is outside this native wire message; transfer `config()` separately.
    pub fn to_proto(&self) -> Result<HloModuleProto> {
        self.validate_structure()?;
        let mut wire = self.wire.clone();
        let computation_order = topo(wire.computations.iter().map(|c| {
            (
                c.id,
                c.instructions
                    .iter()
                    .flat_map(|i| i.called_computation_ids.iter().copied())
                    .collect(),
            )
        }))?;
        let mut computations: BTreeMap<_, _> =
            wire.computations.drain(..).map(|c| (c.id, c)).collect();
        for id in computation_order {
            let mut computation = computations
                .remove(&id)
                .ok_or_else(|| Error::new("graph", "missing computation during export"))?;
            let order = instruction_order(&computation)?;
            let mut instructions: BTreeMap<_, _> = computation
                .instructions
                .drain(..)
                .map(|i| (i.id, i))
                .collect();
            for id in order {
                computation.instructions.push(
                    instructions
                        .remove(&id)
                        .ok_or_else(|| Error::new("graph", "missing instruction during export"))?,
                );
            }
            wire.computations.push(computation);
        }
        Ok(wire)
    }

    /// Export with missing host/computation signatures and parameter names
    /// materialized for native HLO import. Existing shapes, names and unknown
    /// fields survive. Unlike raw export, this requires consecutive parameter
    /// numbers and complete signature name counts.
    pub fn to_proto_with_program_shapes(&self) -> Result<HloModuleProto> {
        let mut wire = self.to_proto()?;
        if wire.host_program_shape.is_none() {
            let mut program = self.program_shape(self.entry()?)?;
            program.parameter_names.clear();
            wire.host_program_shape = Some(program).into();
        }
        if let Some(program) = wire.host_program_shape.as_mut() {
            if program.parameter_names.is_empty() {
                program.parameter_names = (0..program.parameters.len())
                    .map(|i| format!("p{i}"))
                    .collect();
            }
            validate_program_names(program)?;
        }
        for computation in &mut wire.computations {
            if computation.program_shape.is_none() {
                computation.program_shape =
                    Some(self.program_shape(self.computation_handle(computation.id))?).into();
            }
            if let Some(program) = computation.program_shape.as_mut() {
                if program.parameter_names.is_empty() && !program.parameters.is_empty() {
                    program.parameter_names = self
                        .program_shape(self.computation_handle(computation.id))?
                        .parameter_names;
                }
                validate_program_names(program)?;
            }
        }
        Ok(wire)
    }

    pub fn computations(&self) -> impl Iterator<Item = ComputationId> + '_ {
        self.wire
            .computations
            .iter()
            .map(|c| self.computation_handle(c.id))
    }

    pub fn computation(&self, id: ComputationId) -> Result<&HloComputationProto> {
        if id.owner != self.owner {
            return Err(Error::new("handle", "foreign computation handle"));
        }
        self.wire
            .computations
            .iter()
            .find(|c| c.id == id.wire)
            .ok_or_else(|| Error::new("handle", "deleted or unknown computation"))
    }

    pub fn instructions(&self, computation: ComputationId) -> Result<Vec<InstructionId>> {
        Ok(self
            .computation(computation)?
            .instructions
            .iter()
            .map(|i| self.instruction_handle(computation.wire, i.id))
            .collect())
    }

    pub fn instruction(&self, id: InstructionId) -> Result<&HloInstructionProto> {
        self.computation(ComputationId {
            owner: id.owner,
            wire: id.computation,
        })?
        .instructions
        .iter()
        .find(|i| i.id == id.wire)
        .ok_or_else(|| Error::new("handle", "deleted or unknown instruction"))
    }

    pub fn shape(&self, id: InstructionId) -> Result<Shape> {
        Shape::from_proto(
            self.instruction(id)?
                .shape
                .as_ref()
                .ok_or_else(|| Error::new("shape", "missing instruction shape"))?
                .clone(),
        )
    }

    pub fn entry(&self) -> Result<ComputationId> {
        let id = self.computation_handle(self.wire.entry_computation_id);
        self.computation(id)?;
        Ok(id)
    }

    pub fn root(&self, computation: ComputationId) -> Result<InstructionId> {
        let id = self.instruction_handle(computation.wire, self.computation(computation)?.root_id);
        self.instruction(id)?;
        Ok(id)
    }

    pub fn operands(&self, instruction: InstructionId) -> Result<Vec<InstructionId>> {
        Ok(self
            .instruction(instruction)?
            .operand_ids
            .iter()
            .map(|&id| self.instruction_handle(instruction.computation, id))
            .collect())
    }

    /// Unique users, even when one instruction consumes this value repeatedly.
    pub fn users(&self, instruction: InstructionId) -> Result<Vec<InstructionId>> {
        self.instruction(instruction)?;
        Ok(self
            .computation(self.computation_handle(instruction.computation))?
            .instructions
            .iter()
            .filter(|i| i.operand_ids.contains(&instruction.wire))
            .map(|i| self.instruction_handle(instruction.computation, i.id))
            .collect())
    }

    pub fn postorder(&self, computation: ComputationId) -> Result<Vec<InstructionId>> {
        Ok(instruction_order(self.computation(computation)?)?
            .into_iter()
            .map(|id| self.instruction_handle(computation.wire, id))
            .collect())
    }

    pub fn add_computation(&mut self, name: impl Into<String>) -> Result<ComputationId> {
        let name = name.into();
        if name.is_empty() || self.wire.computations.iter().any(|c| c.name == name) {
            return Err(Error::new("name", "empty or duplicate computation name"));
        }
        let id = self.next_computation;
        self.next_computation = id
            .checked_add(1)
            .ok_or_else(|| Error::new("overflow", "computation ID exhausted"))?;
        self.wire.computations.push(HloComputationProto {
            name,
            id,
            root_id: -1,
            ..Default::default()
        });
        Ok(self.computation_handle(id))
    }

    /// Insert with a sanitized, module-unique name, as in native HLO construction.
    pub fn add_instruction(
        &mut self,
        computation: ComputationId,
        instruction: Instruction,
        operands: &[InstructionId],
        callees: &[ComputationId],
    ) -> Result<InstructionId> {
        self.computation(computation)?;
        let channel_present = instruction.explicit_channel;
        let mut wire = instruction.into_proto();
        validate_instruction_data(&wire)?;
        validate_metadata_reference(&wire, self.wire.payloads.len())?;
        if !wire.operand_ids.is_empty()
            || !wire.called_computation_ids.is_empty()
            || !wire.control_predecessor_ids.is_empty()
        {
            return Err(Error::new(
                "reference",
                "prototype references must be provided as typed handles",
            ));
        }
        let opcode: Opcode = wire.opcode.parse()?;
        if opcode.arity().is_some_and(|n| n != operands.len()) {
            return Err(Error::new(
                "arity",
                format!("{opcode} operand count mismatch"),
            ));
        }
        for &operand in operands {
            self.instruction(operand)?;
            if operand.computation != computation.wire {
                return Err(Error::new(
                    "reference",
                    "operand belongs to another computation",
                ));
            }
        }
        for &callee in callees {
            self.computation(callee)?;
        }
        wire.name = names::NameUniquer::for_module(&self.wire).unique(&wire.name)?;
        if channel_present && !channels::supported(&wire.opcode) {
            return Err(Error::new("channel", "opcode does not carry a channel"));
        }
        let id = self.next_instruction;
        self.next_instruction = id
            .checked_add(1)
            .ok_or_else(|| Error::new("overflow", "instruction ID exhausted"))?;
        wire.id = id;
        wire.operand_ids = operands.iter().map(|i| i.wire).collect();
        wire.called_computation_ids = callees.iter().map(|c| c.wire).collect();
        self.computation_mut(computation)?.instructions.push(wire);
        if channel_present {
            self.channel_presence.insert(id);
        }
        self.invalidate_schedule();
        Ok(self.instruction_handle(computation.wire, id))
    }

    pub fn set_root(&mut self, computation: ComputationId, root: InstructionId) -> Result<()> {
        self.computation(computation)?;
        self.instruction(root)?;
        if root.computation != computation.wire {
            return Err(Error::new("reference", "foreign root"));
        }
        self.computation_mut(computation)?.root_id = root.wire;
        self.clear_signatures();
        Ok(())
    }

    pub fn set_entry(&mut self, computation: ComputationId) -> Result<()> {
        let name = self.computation(computation)?.name.clone();
        self.wire.entry_computation_id = computation.wire;
        self.wire.entry_computation_name = name;
        self.wire.host_program_shape.clear();
        self.invalidate_schedule();
        Ok(())
    }

    /// A coordinated rewrite commits only after structural validation. Semantic
    /// checking is a separate explicit call, so signature-changing passes work.
    /// Failed transactions retain consumed ID high-water marks to prevent stale
    /// handles escaping a failed edit from ever identifying a later instruction.
    pub fn edit(&mut self, edit: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let mut staged = Self {
            wire: self.wire.clone(),
            channel_presence: self.channel_presence.clone(),
            config: self.config.clone(),
            owner: self.owner,
            next_instruction: self.next_instruction,
            next_computation: self.next_computation,
        };
        let outcome = edit(&mut staged).and_then(|()| staged.validate_structure());
        self.next_instruction = self.next_instruction.max(staged.next_instruction);
        self.next_computation = self.next_computation.max(staged.next_computation);
        outcome?;
        self.wire = staged.wire;
        self.channel_presence = staged.channel_presence;
        self.config = staged.config;
        Ok(())
    }

    /// Atomically commit a coordinated rewrite only if semantic verification
    /// succeeds. Unsupported contracts are errors and roll back the rewrite.
    pub fn edit_verified(
        &mut self,
        options: crate::VerifyOptions,
        edit: impl FnOnce(&mut Self) -> Result<()>,
    ) -> Result<()> {
        self.edit(|staged| {
            edit(staged)?;
            staged.verify(options)
        })
    }

    /// Assign an explicit schedule after graph editing has finished.
    /// Semantic thread coverage is checked by `verify_schedule` or `verify`.
    pub fn set_schedule(
        &mut self,
        schedule: Option<crate::proto::hlo::HloScheduleProto>,
    ) -> Result<()> {
        self.edit(|staged| {
            staged.wire.schedule = schedule.into();
            Ok(())
        })
    }

    pub fn set_input_output_aliases(
        &mut self,
        config: Option<crate::proto::hlo::HloInputOutputAliasProto>,
    ) -> Result<()> {
        self.edit(|staged| {
            staged.wire.input_output_alias = config.into();
            Ok(())
        })
    }

    pub fn set_buffer_donors(
        &mut self,
        config: Option<crate::proto::hlo::HloBufferDonorProto>,
    ) -> Result<()> {
        self.edit(|staged| {
            staged.wire.buffer_donor = config.into();
            Ok(())
        })
    }

    pub fn replace_operand(
        &mut self,
        user: InstructionId,
        index: usize,
        replacement: InstructionId,
    ) -> Result<()> {
        self.instruction(user)?;
        self.instruction(replacement)?;
        if user.computation != replacement.computation {
            return Err(Error::new("reference", "cross-computation replacement"));
        }
        if index >= self.instruction(user)?.operand_ids.len() {
            return Err(Error::new("operand", "operand index out of range"));
        }
        let mut staged = self.wire.clone();
        let instruction = staged
            .computations
            .iter_mut()
            .find(|c| c.id == user.computation)
            .and_then(|c| c.instructions.iter_mut().find(|i| i.id == user.wire))
            .ok_or_else(|| Error::new("handle", "missing user"))?;
        instruction.operand_ids[index] = replacement.wire;
        let computation = staged
            .computations
            .iter()
            .find(|c| c.id == user.computation)
            .ok_or_else(|| Error::new("handle", "missing computation"))?;
        instruction_order(computation)?;
        self.wire = staged;
        self.invalidate_schedule();
        self.clear_signatures();
        Ok(())
    }

    pub fn replace_all_uses(
        &mut self,
        old: InstructionId,
        replacement: InstructionId,
    ) -> Result<()> {
        self.instruction(old)?;
        self.instruction(replacement)?;
        if old.computation != replacement.computation {
            return Err(Error::new("reference", "cross-computation replacement"));
        }
        if old == replacement {
            return Ok(());
        }
        self.edit(|module| {
            let computation = module.computation_mut(module.computation_handle(old.computation))?;
            for instruction in &mut computation.instructions {
                for operand in &mut instruction.operand_ids {
                    if *operand == old.wire {
                        *operand = replacement.wire;
                    }
                }
            }
            if computation.root_id == old.wire {
                computation.root_id = replacement.wire;
            }
            module.invalidate_schedule();
            module.clear_signatures();
            Ok(())
        })
    }

    pub fn add_control_dependency(
        &mut self,
        predecessor: InstructionId,
        successor: InstructionId,
    ) -> Result<()> {
        self.instruction(predecessor)?;
        self.instruction(successor)?;
        if predecessor.computation != successor.computation {
            return Err(Error::new("reference", "cross-computation control edge"));
        }
        self.edit(|module| {
            let instruction = module.instruction_mut(successor)?;
            if instruction
                .control_predecessor_ids
                .contains(&predecessor.wire)
            {
                return Err(Error::new("control", "duplicate control predecessor"));
            }
            instruction.control_predecessor_ids.push(predecessor.wire);
            module.invalidate_schedule();
            Ok(())
        })
    }

    pub fn set_shape(&mut self, instruction: InstructionId, shape: Shape) -> Result<()> {
        self.instruction_mut(instruction)?.shape = Some(shape.into_proto()).into();
        self.clear_signatures();
        Ok(())
    }

    /// Removes a dead, non-parameter instruction without effects or control edges.
    /// Defaults to non-SPMD channel sensitivity.
    pub fn remove_instruction(&mut self, instruction: InstructionId) -> Result<()> {
        self.remove_instruction_with_effect_context(instruction, EffectContext::default())
    }

    pub fn remove_instruction_with_effect_context(
        &mut self,
        instruction: InstructionId,
        context: EffectContext,
    ) -> Result<()> {
        let data = self.instruction(instruction)?;
        let computation = self.computation(self.computation_handle(instruction.computation))?;
        if computation.root_id == instruction.wire
            || !data.control_predecessor_ids.is_empty()
            || computation.instructions.iter().any(|i| {
                i.operand_ids.contains(&instruction.wire)
                    || i.control_predecessor_ids.contains(&instruction.wire)
            })
        {
            return Err(Error::new(
                "live-instruction",
                "root, data use or control dependency prevents removal",
            ));
        }
        let opcode: Opcode = self.instruction(instruction)?.opcode.parse()?;
        if opcode == Opcode::Parameter || self.instruction_has_side_effect(instruction, context)? {
            return Err(Error::new(
                "unsupported",
                format!("parameter or side effect prevents safe deletion of {opcode}"),
            ));
        }
        self.computation_mut(self.computation_handle(instruction.computation))?
            .instructions
            .retain(|i| i.id != instruction.wire);
        self.channel_presence.remove(&instruction.wire);
        self.invalidate_schedule();
        self.clear_signatures();
        Ok(())
    }

    /// Remove a dead instruction and recursively remove newly unused operands.
    /// Parameters, computation roots, effects and control dependencies are
    /// retained. Parameter/signature cleanup requires `remove_unused_parameter`.
    /// Like native HLO, an initial after-all depending directly (or through one
    /// tuple projection) on effects is retained, returning an empty deletion list.
    /// Unlike native low-level removal, the explicit target must also be pure.
    /// Failures leave the graph unchanged; returned handles are now invalid.
    pub fn remove_instruction_and_unused_operands(
        &mut self,
        instruction: InstructionId,
    ) -> Result<Vec<InstructionId>> {
        self.remove_instruction_and_unused_operands_with_effect_context(
            instruction,
            EffectContext::default(),
        )
    }

    pub fn remove_instruction_and_unused_operands_with_effect_context(
        &mut self,
        instruction: InstructionId,
        context: EffectContext,
    ) -> Result<Vec<InstructionId>> {
        self.instruction(instruction)?;
        let mut removed = Vec::new();
        self.edit(|module| {
            let owner = module.computation_handle(instruction.computation);
            let operands = module.operands(instruction)?;
            if module.root(owner)? == instruction || !module.users(instruction)?.is_empty() {
                return Err(Error::new(
                    "live-instruction",
                    "root or data use prevents removal",
                ));
            }
            if module.instruction(instruction)?.opcode == "after-all" {
                for &operand in &operands {
                    let source = if module.instruction(operand)?.opcode == "get-tuple-element" {
                        *module
                            .operands(operand)?
                            .first()
                            .ok_or_else(|| Error::new("arity", "tuple projection has no operand"))?
                    } else {
                        operand
                    };
                    if module.instruction_has_side_effect(source, context)? {
                        return Ok(());
                    }
                }
            }
            module.remove_instruction_with_effect_context(instruction, context)?;
            removed.push(instruction);
            let mut deleted = BTreeSet::from([instruction]);
            let mut pending = std::collections::VecDeque::from(operands);
            while let Some(candidate) = pending.pop_front() {
                if deleted.contains(&candidate) {
                    continue;
                }
                let data = module.instruction(candidate)?;
                if data.opcode == "parameter"
                    || module.root(owner)? == candidate
                    || !module.users(candidate)?.is_empty()
                    || !data.control_predecessor_ids.is_empty()
                    || module
                        .computation(owner)?
                        .instructions
                        .iter()
                        .any(|i| i.control_predecessor_ids.contains(&candidate.wire))
                    || module.instruction_has_side_effect(candidate, context)?
                {
                    continue;
                }
                pending.extend(module.operands(candidate)?);
                module.remove_instruction_with_effect_context(candidate, context)?;
                deleted.insert(candidate);
                removed.push(candidate);
            }
            Ok(())
        })?;
        Ok(removed)
    }

    /// Clone with a fresh handle namespace. Wire IDs and payloads are preserved.
    pub fn fork(&self) -> Result<Self> {
        let mut fork = Self::from_proto_with_config(self.to_proto()?, self.config.clone())?;
        fork.channel_presence = self.channel_presence.clone();
        Ok(fork)
    }

    /// Mutate typed payload fields without exposing graph references or IDs.
    /// Structural validation commits atomically; opcode semantics are verified
    /// separately at the enclosing pass boundary.
    /// Changing channel scalar fields resets presence to wire-import rules;
    /// use `set_channel_id` for explicit nonpositive presence.
    pub fn edit_instruction_fields(
        &mut self,
        id: InstructionId,
        edit: impl FnOnce(&mut HloInstructionProto) -> Result<()>,
    ) -> Result<()> {
        self.instruction(id)?;
        self.edit(|module| {
            let before = module.instruction(id)?.clone();
            let mut updated = before.clone();
            edit(&mut updated)?;
            if updated.id != before.id
                || updated.operand_ids != before.operand_ids
                || updated.control_predecessor_ids != before.control_predecessor_ids
                || updated.called_computation_ids != before.called_computation_ids
            {
                return Err(Error::new(
                    "reference",
                    "use typed graph APIs to change identities or references",
                ));
            }
            validate_instruction_data(&updated)?;
            if (before.channel_id, before.all_reduce_id)
                != (updated.channel_id, updated.all_reduce_id)
            {
                module.channel_presence.remove(&id.wire);
            }
            let opcode: Opcode = updated.opcode.parse()?;
            if opcode
                .arity()
                .is_some_and(|arity| arity != updated.operand_ids.len())
            {
                return Err(Error::new(
                    "arity",
                    "updated opcode has incompatible operand count",
                ));
            }
            *module.instruction_mut(id)? = updated;
            module.invalidate_schedule();
            module.clear_signatures();
            Ok(())
        })
    }

    pub fn set_called_computations(
        &mut self,
        id: InstructionId,
        callees: &[ComputationId],
    ) -> Result<()> {
        self.instruction(id)?;
        for &callee in callees {
            self.computation(callee)?;
        }
        self.edit(|module| {
            module.instruction_mut(id)?.called_computation_ids =
                callees.iter().map(|c| c.wire).collect();
            module.invalidate_schedule();
            Ok(())
        })
    }

    pub fn remove_control_dependency(
        &mut self,
        predecessor: InstructionId,
        successor: InstructionId,
    ) -> Result<()> {
        self.instruction(predecessor)?;
        self.instruction(successor)?;
        if predecessor.computation != successor.computation
            || !self
                .instruction(successor)?
                .control_predecessor_ids
                .contains(&predecessor.wire)
        {
            return Err(Error::new("control", "control dependency does not exist"));
        }
        self.instruction_mut(successor)?
            .control_predecessor_ids
            .retain(|id| *id != predecessor.wire);
        self.invalidate_schedule();
        Ok(())
    }

    /// Clone one computation into this module with fresh instruction/wire IDs.
    /// External callee references are retained; local data/control/root
    /// references are all remapped. No optimization is performed.
    pub fn clone_computation(
        &mut self,
        source: ComputationId,
        name: impl Into<String>,
    ) -> Result<ClonedComputation> {
        let source_wire = self.computation(source)?.clone();
        let name = name.into();
        if name.is_empty() || self.wire.computations.iter().any(|c| c.name == name) {
            return Err(Error::new("name", "empty or duplicate clone name"));
        }
        let computation_id = self.next_computation;
        let next_computation = computation_id
            .checked_add(1)
            .ok_or_else(|| Error::new("overflow", "computation ID exhausted"))?;
        let mut next_instruction = self.next_instruction;
        let mut mapping = BTreeMap::new();
        for instruction in &source_wire.instructions {
            mapping.insert(instruction.id, next_instruction);
            next_instruction = next_instruction
                .checked_add(1)
                .ok_or_else(|| Error::new("overflow", "instruction ID exhausted"))?;
        }
        let mut clone = source_wire;
        let mut names = names::NameUniquer::for_module(&self.wire);
        clone.id = computation_id;
        clone.name = name;
        clone.root_id = *mapping
            .get(&clone.root_id)
            .ok_or_else(|| Error::new("root", "cannot clone missing root"))?;
        for instruction in &mut clone.instructions {
            instruction.name = names.unique(&format!("{}.clone", instruction.name))?;
            instruction.id = *mapping
                .get(&instruction.id)
                .ok_or_else(|| Error::new("reference", "clone identity missing"))?;
            for id in instruction
                .operand_ids
                .iter_mut()
                .chain(&mut instruction.control_predecessor_ids)
            {
                *id = *mapping
                    .get(id)
                    .ok_or_else(|| Error::new("reference", "clone dependency missing"))?;
            }
        }
        self.next_computation = next_computation;
        self.next_instruction = next_instruction;
        self.wire.computations.push(clone);
        for (&old, &new) in &mapping {
            if self.channel_presence.contains(&old) {
                self.channel_presence.insert(new);
            }
        }
        self.invalidate_schedule();
        Ok(ClonedComputation {
            computation: self.computation_handle(computation_id),
            instructions: mapping
                .into_iter()
                .map(|(old, new)| {
                    (
                        self.instruction_handle(source.wire, old),
                        self.instruction_handle(computation_id, new),
                    )
                })
                .collect(),
        })
    }

    pub fn validate_structure(&self) -> Result<()> {
        if self.wire.name.is_empty() {
            return Err(Error::new("name", "empty module name"));
        }
        let mut computations = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut instructions = BTreeSet::new();
        let mut instruction_names = BTreeSet::new();
        for computation in &self.wire.computations {
            if computation.id < 0
                || !computations.insert(computation.id)
                || computation.name.is_empty()
                || !names.insert(&computation.name)
            {
                return Err(Error::new(
                    "computation",
                    "invalid/duplicate computation identity",
                ));
            }
            for instruction in &computation.instructions {
                validate_instruction_data(instruction).map_err(|e| e.context(&instruction.name))?;
                if self.channel_presence.contains(&instruction.id)
                    && !channels::supported(&instruction.opcode)
                {
                    return Err(Error::new("channel", "opcode does not carry a channel"));
                }
                if instruction.id < 0
                    || !instructions.insert(instruction.id)
                    || !instruction_names.insert(&instruction.name)
                {
                    return Err(Error::new(
                        "instruction",
                        "invalid/duplicate instruction identity",
                    ));
                }
                validate_metadata_reference(instruction, self.wire.payloads.len())?;
            }
            if !computation
                .instructions
                .iter()
                .any(|i| i.id == computation.root_id)
            {
                return Err(Error::new(
                    "root",
                    format!("missing root in {}", computation.name),
                ));
            }
            instruction_order(computation).map_err(|e| e.context(&computation.name))?;
        }
        if !self.channel_presence.is_subset(&instructions) {
            return Err(Error::new(
                "channel",
                "channel presence refers to a missing instruction",
            ));
        }
        if !computations.contains(&self.wire.entry_computation_id) {
            return Err(Error::new("entry", "missing entry computation"));
        }
        let entry = self.computation(self.computation_handle(self.wire.entry_computation_id))?;
        if !self.wire.entry_computation_name.is_empty()
            && self.wire.entry_computation_name != entry.name
        {
            return Err(Error::new("entry", "entry name/ID mismatch"));
        }
        topo(self.wire.computations.iter().map(|c| {
            (
                c.id,
                c.instructions
                    .iter()
                    .flat_map(|i| i.called_computation_ids.iter().copied())
                    .collect(),
            )
        }))?;
        if let Some(schedule) = self.wire.schedule.as_ref() {
            let scheduled_threads = schedule
                .sequences
                .keys()
                .filter_map(|id| self.wire.computations.iter().find(|c| c.id == *id))
                .map(|computation| execution_thread(&computation.execution_thread))
                .collect::<BTreeSet<_>>();
            let called = self
                .wire
                .computations
                .iter()
                .flat_map(|c| &c.instructions)
                .flat_map(|i| i.called_computation_ids.iter().copied())
                .collect::<BTreeSet<_>>();
            for (&id, sequence) in &schedule.sequences {
                let computation = self.computation(self.computation_handle(id))?;
                let instructions: BTreeSet<_> =
                    computation.instructions.iter().map(|i| i.id).collect();
                if sequence
                    .instruction_ids
                    .iter()
                    .any(|instruction| !instructions.contains(instruction))
                {
                    return Err(Error::new("schedule", "unknown instruction"));
                }
                let live = computation.id == self.wire.entry_computation_id
                    || called.contains(&computation.id);
                let verify_order = live
                    && !computation.is_fusion_computation
                    && scheduled_threads.contains(execution_thread(&computation.execution_thread));
                if !verify_order {
                    continue;
                }
                let all: BTreeSet<_> = computation.instructions.iter().map(|i| i.id).collect();
                let sequence_ids: BTreeSet<_> = sequence.instruction_ids.iter().copied().collect();
                if sequence_ids != all || sequence_ids.len() != sequence.instruction_ids.len() {
                    return Err(Error::new(
                        "schedule",
                        "schedule must contain every instruction exactly once",
                    ));
                }
                let mut seen = BTreeSet::new();
                for id in &sequence.instruction_ids {
                    let instruction = computation
                        .instructions
                        .iter()
                        .find(|i| i.id == *id)
                        .ok_or_else(|| Error::new("schedule", "unknown instruction"))?;
                    if instruction
                        .operand_ids
                        .iter()
                        .chain(&instruction.control_predecessor_ids)
                        .any(|i| !seen.contains(i))
                    {
                        return Err(Error::new("schedule", "schedule violates dependency order"));
                    }
                    seen.insert(*id);
                }
            }
        }
        Ok(())
    }

    /// Compute current signatures rather than trusting imported cached ones.
    pub fn program_shape(&self, computation: ComputationId) -> Result<ProgramShapeProto> {
        computation_program_shape(self.computation(computation)?)
    }

    pub(crate) fn computation_handle(&self, wire: i64) -> ComputationId {
        ComputationId {
            owner: self.owner,
            wire,
        }
    }
    pub(crate) fn instruction_handle(&self, computation: i64, wire: i64) -> InstructionId {
        InstructionId {
            owner: self.owner,
            computation,
            wire,
        }
    }
    fn computation_mut(&mut self, id: ComputationId) -> Result<&mut HloComputationProto> {
        self.computation(id)?;
        self.wire
            .computations
            .iter_mut()
            .find(|c| c.id == id.wire)
            .ok_or_else(|| Error::new("handle", "missing computation"))
    }
    fn instruction_mut(&mut self, id: InstructionId) -> Result<&mut HloInstructionProto> {
        self.instruction(id)?;
        self.computation_mut(self.computation_handle(id.computation))?
            .instructions
            .iter_mut()
            .find(|i| i.id == id.wire)
            .ok_or_else(|| Error::new("handle", "missing instruction"))
    }
    fn invalidate_schedule(&mut self) {
        self.wire.schedule.clear();
        self.refresh_computation_kinds();
    }
    fn refresh_computation_kinds(&mut self) {
        let fused: BTreeSet<_> = self
            .wire
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .filter(|i| i.opcode == "fusion")
            .flat_map(|i| i.called_computation_ids.iter().copied())
            .collect();
        for computation in &mut self.wire.computations {
            computation.is_fusion_computation = fused.contains(&computation.id);
        }
    }
    fn clear_signatures(&mut self) {
        self.wire.host_program_shape.clear();
        for computation in &mut self.wire.computations {
            computation.program_shape.clear();
        }
    }
}

fn next_id(ids: impl Iterator<Item = i64>) -> Result<i64> {
    ids.max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(|| Error::new("overflow", "ID space exhausted"))
}

fn validate_program_names(program: &ProgramShapeProto) -> Result<()> {
    if program.parameters.len() != program.parameter_names.len() {
        return Err(Error::new(
            "signature",
            "parameter shape/name count mismatch",
        ));
    }
    Ok(())
}

fn validate_metadata_reference(
    instruction: &HloInstructionProto,
    payload_count: usize,
) -> Result<()> {
    if let Some(payload) = instruction
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.metadata_payload.as_ref())
        && payload.has_id()
        && usize::try_from(payload.id())
            .ok()
            .is_none_or(|id| id >= payload_count)
    {
        return Err(Error::new(
            "metadata",
            format!("invalid metadata payload ID on {}", instruction.name),
        ));
    }
    Ok(())
}

fn validate_instruction_data(wire: &HloInstructionProto) -> Result<()> {
    if wire.name.is_empty() {
        return Err(Error::new("name", "empty instruction name"));
    }
    let opcode = wire.opcode.parse::<Opcode>()?;
    if opcode.supports_result_accuracy()
        && let Some(accuracy) = wire.result_accuracy.as_ref()
    {
        crate::accuracy::validate(accuracy)?;
    }
    if let Some(sharding) = wire.sharding.as_ref() {
        crate::Sharding::from_proto(sharding.clone())?;
    }
    if opcode == Opcode::Domain {
        for sharding in [
            wire.domain_entry_sharding.as_ref(),
            wire.domain_exit_sharding.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            crate::Sharding::from_proto(sharding.clone())?;
        }
    }
    Shape::from_proto(
        wire.shape
            .as_ref()
            .ok_or_else(|| Error::new("shape", "instruction shape missing"))?
            .clone(),
    )?;
    Ok(())
}

pub(crate) fn instruction_order(computation: &HloComputationProto) -> Result<Vec<i64>> {
    for instruction in &computation.instructions {
        let unique: BTreeSet<_> = instruction.control_predecessor_ids.iter().collect();
        if unique.len() != instruction.control_predecessor_ids.len() {
            return Err(Error::new("control", "duplicate control predecessor"));
        }
    }
    topo(computation.instructions.iter().map(|i| {
        (
            i.id,
            i.operand_ids
                .iter()
                .chain(&i.control_predecessor_ids)
                .copied()
                .collect(),
        )
    }))
}

/// Iterative Kahn ordering includes disconnected instructions and control edges.
fn topo(nodes: impl Iterator<Item = (i64, Vec<i64>)>) -> Result<Vec<i64>> {
    let graph: BTreeMap<_, _> = nodes.collect();
    let mut counts = BTreeMap::new();
    let mut users: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for (&id, dependencies) in &graph {
        let dependencies: BTreeSet<_> = dependencies.iter().copied().collect();
        counts.insert(id, dependencies.len());
        for dependency in dependencies {
            if !graph.contains_key(&dependency) {
                return Err(Error::new(
                    "reference",
                    format!("missing dependency {dependency}"),
                ));
            }
            users.entry(dependency).or_default().insert(id);
        }
    }
    let mut ready: BTreeSet<_> = counts
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(&id, _)| id)
        .collect();
    let mut order = Vec::with_capacity(graph.len());
    while let Some(id) = ready.pop_first() {
        order.push(id);
        if let Some(successors) = users.get(&id) {
            for successor in successors {
                let count = counts
                    .get_mut(successor)
                    .ok_or_else(|| Error::new("graph", "missing successor count"))?;
                *count -= 1;
                if *count == 0 {
                    ready.insert(*successor);
                }
            }
        }
    }
    if order.len() != graph.len() {
        return Err(Error::new("cycle", "dependency graph contains a cycle"));
    }
    Ok(order)
}
