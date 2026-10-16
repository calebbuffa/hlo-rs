use std::collections::BTreeMap;

use super::{Instruction, InstructionId, Module};
use crate::{Error, Result};

impl Module {
    /// Remove an unused parameter and its argument from every call, fusion or
    /// async-start caller. Entry signature changes require explicit permission.
    /// Parameters after the removed slot receive fresh identities; the returned
    /// map relates their old handles to their replacements.
    ///
    /// Caller operand instructions are retained, including effects. Async-start
    /// operand context shapes are updated; additional async state/projection
    /// shape rewrites may be needed in the surrounding coordinated transaction.
    /// Aliases/donors/prefetches referencing the removed slot must be explicitly
    /// rewritten first. Other parameter/operand indices are renumbered.
    pub fn remove_unused_parameter(
        &mut self,
        parameter: InstructionId,
        allow_entry: bool,
    ) -> Result<BTreeMap<InstructionId, InstructionId>> {
        self.instruction(parameter)?;
        let mut replacements = BTreeMap::new();
        self.edit(|module| {
            replacements = module.remove_parameter_inner(parameter, allow_entry)?;
            Ok(())
        })?;
        Ok(replacements)
    }

    fn remove_parameter_inner(
        &mut self,
        parameter: InstructionId,
        allow_entry: bool,
    ) -> Result<BTreeMap<InstructionId, InstructionId>> {
        let data = self.instruction(parameter)?;
        if data.opcode != "parameter" {
            return Err(Error::new("parameter", "instruction is not a parameter"));
        }
        if !data.operand_ids.is_empty() || !data.called_computation_ids.is_empty() {
            return Err(Error::new(
                "parameter",
                "parameter has unexpected references",
            ));
        }
        let number = data.parameter_number;
        let index = usize::try_from(number)
            .map_err(|_| Error::new("parameter", "negative parameter number"))?;
        let owner = self.computation_handle(parameter.computation);
        self.program_shape(owner)?;
        let computation = self.computation(owner)?;
        if computation.root_id == parameter.wire
            || !self.users(parameter)?.is_empty()
            || !data.control_predecessor_ids.is_empty()
            || computation
                .instructions
                .iter()
                .any(|i| i.control_predecessor_ids.contains(&parameter.wire))
        {
            return Err(Error::new(
                "live-instruction",
                "parameter is live or has control dependencies",
            ));
        }
        let entry = self.wire.entry_computation_id == parameter.computation;
        if entry && !allow_entry {
            return Err(Error::new(
                "parameter",
                "entry parameter removal requires explicit permission",
            ));
        }
        let mut remaining = Vec::new();
        for id in self.instructions(owner)? {
            let data = self.instruction(id)?;
            if data.opcode == "parameter" && data.parameter_number > number {
                remaining.push((data.parameter_number, id));
            }
        }
        remaining.sort_by_key(|&(number, _)| number);
        let mut callers = Vec::new();
        for computation in self.computations() {
            for id in self.instructions(computation)? {
                if self
                    .instruction(id)?
                    .called_computation_ids
                    .contains(&parameter.computation)
                {
                    callers.push(id);
                }
            }
        }
        if callers.is_empty() && !entry {
            return Err(Error::new(
                "parameter",
                "cannot remove a parameter without a caller",
            ));
        }
        for caller in callers {
            let instruction = self.instruction_mut(caller)?;
            if !matches!(
                instruction.opcode.as_str(),
                "call" | "fusion" | "async-start"
            ) || instruction.called_computation_ids != [parameter.computation]
            {
                return Err(Error::new(
                    "unsupported",
                    "parameter caller requires a coordinated control-flow signature rewrite",
                ));
            }
            if index >= instruction.operand_ids.len() {
                return Err(Error::new(
                    "parameter",
                    "caller does not bind the parameter slot",
                ));
            }
            for alias in &mut instruction.output_operand_aliasing {
                renumber(&mut alias.operand_index, number)?;
                if instruction.opcode == "async-start"
                    && alias.output_shape_index.first() == Some(&0)
                    && let Some(slot) = alias.output_shape_index.get_mut(1)
                {
                    renumber(slot, number)?;
                }
            }
            if instruction.opcode == "async-start" {
                let context = instruction
                    .shape
                    .as_mut()
                    .and_then(|shape| shape.tuple_shapes.get_mut(0))
                    .ok_or_else(|| Error::new("async", "missing async operand context"))?;
                if context.tuple_shapes.len() != instruction.operand_ids.len() {
                    return Err(Error::new("async", "async context/operand count mismatch"));
                }
                context.tuple_shapes.remove(index);
            }
            instruction.operand_ids.remove(index);
        }
        if entry {
            for reference in self
                .wire
                .input_output_alias
                .as_mut()
                .into_iter()
                .flat_map(|a| &mut a.entries)
                .map(|a| &mut a.parameter_number)
                .chain(
                    self.wire
                        .buffer_donor
                        .as_mut()
                        .into_iter()
                        .flat_map(|d| &mut d.entries)
                        .map(|d| &mut d.parameter_number),
                )
                .chain(
                    self.wire
                        .cross_program_prefetches
                        .iter_mut()
                        .map(|p| &mut p.parameter),
                )
            {
                renumber(reference, number)?;
            }
        }
        self.computation_mut(owner)?
            .instructions
            .retain(|i| i.id != parameter.wire);
        let mut replacements = BTreeMap::new();
        for (number, old) in remaining {
            let mut prototype = self.instruction(old)?.clone();
            let controls = std::mem::take(&mut prototype.control_predecessor_ids);
            prototype.parameter_number = number - 1;
            prototype.name = format!("param_{}", number - 1);
            let new = self.add_instruction(owner, Instruction::from_proto(prototype)?, &[], &[])?;
            let computation = self.computation_mut(owner)?;
            for instruction in &mut computation.instructions {
                for reference in instruction
                    .operand_ids
                    .iter_mut()
                    .chain(&mut instruction.control_predecessor_ids)
                {
                    if *reference == old.wire {
                        *reference = new.wire;
                    }
                }
            }
            if computation.root_id == old.wire {
                computation.root_id = new.wire;
            }
            self.instruction_mut(new)?.control_predecessor_ids = controls;
            self.computation_mut(owner)?
                .instructions
                .retain(|i| i.id != old.wire);
            replacements.insert(old, new);
        }
        self.invalidate_schedule();
        self.clear_signatures();
        Ok(replacements)
    }
}

fn renumber(reference: &mut i64, removed: i64) -> Result<()> {
    if *reference == removed {
        return Err(Error::new(
            "alias",
            "removed parameter/operand still has an alias, donor or prefetch reference",
        ));
    }
    if *reference > removed {
        *reference -= 1;
    }
    Ok(())
}
