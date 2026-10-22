use super::{VerifyOptions, require};
use crate::{
    opcode::Opcode,
    shape,
    types::{ElementType, tensor},
};
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
};

pub(super) fn verify_instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    if !options.layout_sensitive && options.instruction_can_change_layout.is_none() {
        return Ok(());
    }
    let data = op.deref(ctx);
    let result = data.get_type(0);
    if !shape::is_array_or_buffer(ctx, result) {
        return Ok(());
    }
    let result = shape::array_view(ctx, result)?;
    let result_array = tensor(ctx, result)?;
    let Some(result_layout) = shape::layout(ctx, result)? else {
        return Ok(());
    };
    if options.layout_sensitive {
        require(
            result_layout.minor_to_major.len() == result_array.dimensions.len(),
            "layout-sensitive verification requires minor-to-major rank to match shape rank",
        )?;
    }
    if options
        .instruction_can_change_layout
        .is_none_or(|allowed| allowed(ctx, op))
    {
        return Ok(());
    }
    for operand in data.operand_types(ctx) {
        if !shape::is_array_or_buffer(ctx, operand) {
            continue;
        }
        let operand = shape::array_view(ctx, operand)?;
        let operand_array = tensor(ctx, operand)?;
        if operand_array.dimensions.len() != result_array.dimensions.len() {
            continue;
        }
        let Some(operand_layout) = shape::layout(ctx, operand)? else {
            continue;
        };
        let (mut output, mut input) = (result_layout.clone(), operand_layout.clone());
        output.tiles.clear();
        input.tiles.clear();
        output.memory_space = 0;
        input.memory_space = 0;
        if matches!(
            opcode,
            Opcode::Convert | Opcode::Compare | Opcode::IsFinite | Opcode::Scatter
        ) || (opcode == Opcode::Select && operand_array.element == ElementType::Pred)
        {
            output.element_size_in_bits = 0;
            input.element_size_in_bits = 0;
        } else if matches!(
            opcode,
            Opcode::DynamicSlice | Opcode::DynamicUpdateSlice | Opcode::Copy
        ) {
            let (from, to) = (operand_layout.memory_space, result_layout.memory_space);
            require(
                !((from == 1 && to != 1)
                    || (opcode == Opcode::DynamicSlice && from == 0 && to == 5)
                    || (opcode == Opcode::DynamicUpdateSlice && from == 5 && to == 0)),
                "instruction violates host-offload layout transition policy",
            )?;
        }
        require(
            crate::shape::layouts_equal(ctx, &output, &input),
            "instruction is not permitted to change operand layouts",
        )?;
    }
    Ok(())
}
