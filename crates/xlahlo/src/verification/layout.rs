use crate::{Error, InstructionId, Module, PrimitiveType, Result, Shape, VerifyOptions};

pub(super) fn verify_instruction(
    module: &Module,
    id: InstructionId,
    options: VerifyOptions,
) -> Result<()> {
    if options
        .instruction_can_change_layout
        .is_none_or(|allowed| allowed(module, id))
    {
        return Ok(());
    }
    fn array(shape: &Shape) -> Result<Option<Shape>> {
        if shape.element_type() == PrimitiveType::BUFFER {
            Ok(Some(shape.array_view()?))
        } else if shape.is_array() {
            Ok(Some(shape.clone()))
        } else {
            Ok(None)
        }
    }
    let result = module.shape(id)?;
    let Some(result) = array(&result)? else {
        return Ok(());
    };
    let Some(result_layout) = result.layout() else {
        return Ok(());
    };
    let instruction = module.instruction(id)?;
    for operand in module.operands(id)? {
        let shape = module.shape(operand)?;
        let Some(operand) = array(&shape)? else {
            continue;
        };
        if operand.dimensions().len() != result.dimensions().len() {
            continue;
        }
        let Some(operand_layout) = operand.layout() else {
            continue;
        };
        let (mut output, mut input) = (result_layout.clone(), operand_layout.clone());
        output.tiles.clear();
        input.tiles.clear();
        output.memory_space = 0;
        input.memory_space = 0;
        let opcode = instruction.opcode.as_str();
        if matches!(opcode, "convert" | "compare" | "is-finite" | "scatter")
            || (opcode == "select" && operand.element_type() == PrimitiveType::PRED)
        {
            output.element_size_in_bits = 0;
            input.element_size_in_bits = 0;
        } else if matches!(opcode, "dynamic-slice" | "dynamic-update-slice" | "copy") {
            let (from, to) = (operand_layout.memory_space, result_layout.memory_space);
            if (from == 1 && to != 1)
                || (opcode == "dynamic-slice" && from == 0 && to == 5)
                || (opcode == "dynamic-update-slice" && from == 5 && to == 0)
            {
                return Err(Error::new(
                    "memory-space",
                    "instruction violates host-offload layout transition policy",
                ));
            }
        }
        if !crate::shape::layouts_equal(&output, &input) {
            return Err(Error::new(
                "layout",
                "instruction is not permitted to change operand layouts",
            ));
        }
    }
    Ok(())
}

pub(super) fn allowed_async_intermediary(
    instruction: &crate::proto::hlo::HloInstructionProto,
) -> bool {
    matches!(
        instruction.opcode.as_str(),
        "tuple" | "get-tuple-element" | "opt-barrier" | "copy" | "parameter" | "while"
    ) || instruction.opcode == "custom-call"
        && matches!(
            instruction.custom_call_target.as_str(),
            "Sharding"
                | "LocalToGlobalShape"
                | "GlobalToLocalShape"
                | "xla.sdy.LocalToGlobalShape"
                | "xla.sdy.GlobalToLocalShape"
                | "xla.sdy.FuncResultSharding"
        )
}

pub(super) fn verify_bitcast(
    module: &Module,
    id: InstructionId,
    options: VerifyOptions,
) -> Result<()> {
    if !options.layout_sensitive {
        return Ok(());
    }
    let operand_id = module.operands(id)?[0];
    let operand = module.shape(operand_id)?;
    let output = module.shape(id)?;
    let size = |shape: &Shape| options.size_of(shape);
    if size(&operand)? != size(&output)? {
        let same_data = options.allow_bitcast_to_have_different_size
            && !operand.is_dynamic()
            && !output.is_dynamic()
            && operand.array_data_size()? == output.array_data_size()?;
        if !same_data {
            return Err(Error::new(
                "bitcast",
                "bitcast operand and result have different shape sizes",
            ));
        }
    }
    let operand = if operand.element_type() == PrimitiveType::BUFFER {
        operand.array_view()?
    } else {
        operand
    };
    let output = if output.element_type() == PrimitiveType::BUFFER {
        output.array_view()?
    } else {
        output
    };
    if let (Some(input), Some(result)) = (operand.layout(), output.layout())
        && (input.memory_space == 5) != (result.memory_space == 5)
    {
        let mut source = operand_id;
        while module.instruction(source)?.opcode == "copy" {
            source = module.operands(source)?[0];
        }
        if module.instruction(source)?.opcode != "constant" {
            return Err(Error::new(
                "bitcast",
                "bitcast cannot cross host/device memory spaces except through constant copies",
            ));
        }
    }
    Ok(())
}
