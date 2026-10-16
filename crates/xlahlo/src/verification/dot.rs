use crate::{Error, PrimitiveType, Result, Shape, proto::hlo::HloInstructionProto};
use std::collections::BTreeSet;

pub(super) fn verify_auxiliary_operands(
    instruction: &HloInstructionProto,
    operands: &[Shape],
) -> Result<()> {
    if operands.len() < 2 {
        return Err(Error::new(
            "arity",
            "dot/convolution requires at least two operands",
        ));
    }
    let mut seen = BTreeSet::new();
    let mut check = |index: i32| -> Result<()> {
        let index = usize::try_from(index)
            .map_err(|_| Error::new("auxiliary-operand", "negative operand index"))?;
        if index < 2 || index >= operands.len() || !seen.insert(index) {
            return Err(Error::new(
                "auxiliary-operand",
                "auxiliary operand index is repeated or out of range",
            ));
        }
        if !operands[index].is_array() && operands[index].element_type() != PrimitiveType::BUFFER {
            return Err(Error::new(
                "auxiliary-operand",
                "auxiliary operand must be an array",
            ));
        }
        Ok(())
    };
    if let Some(config) = instruction.sparsity_config.as_ref() {
        for side in [config.lhs.as_ref(), config.rhs.as_ref()]
            .into_iter()
            .flatten()
        {
            check(side.idx)?;
        }
    }
    if let Some(config) = instruction.block_scaling_config.as_ref() {
        for side in [config.lhs.as_ref(), config.rhs.as_ref()]
            .into_iter()
            .flatten()
        {
            check(side.scale_idx)?;
            if let Some(index) = side.zero_idx {
                check(index)?;
            }
        }
    }
    if seen.len() != operands.len() - 2 {
        return Err(Error::new(
            "auxiliary-operand",
            "every extra operand must be referenced exactly once",
        ));
    }
    Ok(())
}
