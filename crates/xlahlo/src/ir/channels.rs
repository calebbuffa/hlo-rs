use super::{Instruction, InstructionId, Module};
use crate::{Error, Result, proto::hlo::HloInstructionProto};

pub(crate) fn supported(opcode: &str) -> bool {
    matches!(
        opcode,
        "send"
            | "send-done"
            | "recv"
            | "recv-done"
            | "all-gather"
            | "all-gather-start"
            | "all-reduce"
            | "all-reduce-start"
            | "reduce-scatter"
            | "all-to-all"
            | "ragged-all-to-all"
            | "collective-broadcast"
            | "collective-reduce"
            | "collective-permute"
            | "collective-permute-start"
    )
}

pub(crate) fn value(wire: &HloInstructionProto, explicit: bool) -> Option<i64> {
    if !supported(&wire.opcode) {
        None
    } else if explicit || wire.channel_id > 0 {
        Some(wire.channel_id)
    } else if matches!(wire.opcode.as_str(), "all-reduce" | "all-reduce-start")
        && wire.all_reduce_id > 0
    {
        Some(wire.all_reduce_id)
    } else {
        None
    }
}

impl Instruction {
    /// Native semantic presence, distinct from the lossy proto3 scalar field.
    pub fn channel_id(&self) -> Option<i64> {
        value(&self.wire, self.explicit_channel)
    }

    /// Set a semantic channel, retaining explicit zero/negative presence in
    /// memory. Native protobuf import loses nonpositive channel presence.
    pub fn with_channel_id(mut self, channel: Option<i64>) -> Result<Self> {
        if !supported(&self.wire.opcode) {
            return Err(Error::new("channel", "opcode does not carry a channel"));
        }
        self.wire.channel_id = channel.unwrap_or(0);
        if matches!(self.wire.opcode.as_str(), "all-reduce" | "all-reduce-start") {
            self.wire.all_reduce_id = 0;
        }
        self.explicit_channel = channel.is_some();
        Ok(self)
    }
}

impl Module {
    /// Native semantic channel presence. Wire imports use positive IDs and the
    /// legacy all-reduce ID; text/explicit setters also retain nonpositive IDs.
    pub fn channel_id(&self, instruction: InstructionId) -> Result<Option<i64>> {
        Ok(value(
            self.instruction(instruction)?,
            self.channel_presence.contains(&instruction.wire),
        ))
    }

    pub fn set_channel_id(
        &mut self,
        instruction: InstructionId,
        channel: Option<i64>,
    ) -> Result<()> {
        self.instruction(instruction)?;
        self.edit(|module| {
            let wire = module.instruction_mut(instruction)?;
            if !supported(&wire.opcode) {
                return Err(Error::new("channel", "opcode does not carry a channel"));
            }
            wire.channel_id = channel.unwrap_or(0);
            if matches!(wire.opcode.as_str(), "all-reduce" | "all-reduce-start") {
                wire.all_reduce_id = 0;
            }
            if channel.is_some() {
                module.channel_presence.insert(instruction.wire);
            } else {
                module.channel_presence.remove(&instruction.wire);
            }
            module.invalidate_schedule();
            Ok(())
        })
    }
}
