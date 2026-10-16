use std::collections::BTreeSet;

use super::{ComputationId, Instruction, InstructionId, Module};
use crate::{Opcode, Result, proto::hlo::HloInstructionProto};

/// External module policy corresponding to `HloModuleConfig::ChannelIdSensitive`.
#[derive(Clone, Copy, Debug)]
pub struct EffectContext {
    pub channel_id_sensitive: bool,
}

impl Default for EffectContext {
    fn default() -> Self {
        Self {
            channel_id_sensitive: true,
        }
    }
}

fn local(
    instruction: &HloInstructionProto,
    channel: Option<i64>,
    context: EffectContext,
) -> Result<bool> {
    use Opcode::*;
    Ok(match instruction.opcode.parse()? {
        Send
        | SendDone
        | Recv
        | RecvDone
        | Rng
        | RngGetAndUpdateState
        | Infeed
        | Outfeed
        | AllReduceStart
        | AllReduceDone
        | AllGatherStart
        | AllGatherDone
        | CollectiveBroadcast
        | CollectivePermuteStart
        | CollectivePermuteDone => true,
        AllToAll | AllGather | AllReduce | ReduceScatter | CollectiveReduce => {
            instruction.constrain_layout || (channel.is_some() && context.channel_id_sensitive)
        }
        CollectivePermute => channel.is_some() && context.channel_id_sensitive,
        CustomCall => instruction.custom_call_has_side_effect,
        _ => false,
    })
}

impl Instruction {
    /// Local effects only, without following called computations.
    /// Explicit channel setters retain zero/negative presence; wire import uses
    /// the native positive-ID convention.
    pub fn has_local_side_effect(&self, context: EffectContext) -> Result<bool> {
        local(self.as_proto(), self.channel_id(), context)
    }
}

impl Module {
    pub fn instruction_has_local_side_effect(
        &self,
        instruction: InstructionId,
        context: EffectContext,
    ) -> Result<bool> {
        local(
            self.instruction(instruction)?,
            self.channel_id(instruction)?,
            context,
        )
    }

    /// Includes every instruction in called computations, even dead instructions.
    pub fn instruction_has_side_effect(
        &self,
        instruction: InstructionId,
        context: EffectContext,
    ) -> Result<bool> {
        self.effects_from(vec![instruction], context)
    }

    pub fn computation_has_side_effect(
        &self,
        computation: ComputationId,
        context: EffectContext,
    ) -> Result<bool> {
        self.effects_from(self.instructions(computation)?, context)
    }

    fn effects_from(&self, mut work: Vec<InstructionId>, context: EffectContext) -> Result<bool> {
        let mut visited = BTreeSet::new();
        while let Some(id) = work.pop() {
            let instruction = self.instruction(id)?;
            if local(instruction, self.channel_id(id)?, context)? {
                return Ok(true);
            }
            for &callee in &instruction.called_computation_ids {
                let computation = self.computation_handle(callee);
                if visited.insert(computation) {
                    work.extend(self.instructions(computation)?);
                }
            }
        }
        Ok(false)
    }
}
