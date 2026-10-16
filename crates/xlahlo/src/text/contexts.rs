use super::*;

fn comparison_order(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
) -> Result<String> {
    if !matches!(
        instruction.comparison_direction.as_str(),
        "EQ" | "NE" | "GE" | "GT" | "LE" | "LT"
    ) {
        return Err(Error::new(
            "comparison",
            "missing or invalid comparison direction",
        ));
    }
    if !instruction.comparison_order.is_empty() {
        if !instruction.comparison_type.is_empty()
            || !matches!(
                instruction.comparison_order.as_str(),
                "TOTAL" | "WEAK" | "PARTIAL"
            )
        {
            return Err(Error::new(
                "comparison",
                "invalid or conflicting comparison order/type",
            ));
        }
        return Ok(instruction.comparison_order.clone());
    }
    if !instruction.comparison_type.is_empty() {
        return match instruction.comparison_type.as_str() {
            "FLOAT" => Ok("PARTIAL".into()),
            "TOTALORDER" | "SIGNED" | "UNSIGNED" => Ok("TOTAL".into()),
            _ => Err(Error::new("comparison", "invalid legacy comparison type")),
        };
    }
    let operand = instruction
        .operand_ids
        .first()
        .and_then(|id| instructions.iter().find(|i| i.id == *id))
        .and_then(|i| i.shape.as_ref())
        .ok_or_else(|| Error::new("comparison", "missing comparison operand"))?;
    let ty = Shape::from_proto(operand.clone())?
        .array_view()?
        .element_type();
    Ok(if ty.is_floating() || ty.is_complex() {
        "PARTIAL"
    } else {
        "TOTAL"
    }
    .into())
}

fn map_dimensions(instruction: &HloInstructionProto) -> Result<Vec<i64>> {
    let shape = Shape::from_proto(
        instruction
            .shape
            .as_ref()
            .ok_or_else(|| Error::new("shape", "map result shape missing"))?
            .clone(),
    )?;
    let shape = if shape.element_type() == PrimitiveType::BUFFER {
        shape.array_view()?
    } else {
        shape
    };
    (0..shape.dimensions().len())
        .map(|axis| i64::try_from(axis).map_err(|_| Error::new("overflow", "map rank exceeds i64")))
        .collect()
}

fn validate_reshape(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
) -> Result<()> {
    if !matches!(instruction.opcode.as_str(), "reshape" | "dynamic-reshape") {
        return Ok(());
    }
    let operand = instruction
        .operand_ids
        .first()
        .and_then(|id| instructions.iter().find(|i| i.id == *id))
        .and_then(|i| i.shape.as_ref())
        .ok_or_else(|| Error::new("shape", "reshape requires an array operand"))?;
    let operand = Shape::from_proto(operand.clone())?.array_view()?;
    let result = Shape::from_proto(
        instruction
            .shape
            .as_ref()
            .ok_or_else(|| Error::new("shape", "reshape result shape missing"))?
            .clone(),
    )?
    .array_view()?;
    let product = |shape: &Shape| {
        shape
            .dimensions()
            .iter()
            .filter(|&&size| size != crate::UNBOUNDED_SIZE)
            .try_fold(1_i64, |product, &size| {
                product
                    .checked_mul(size)
                    .ok_or_else(|| Error::new("overflow", "reshape static extent product overflow"))
            })
    };
    let dynamic = instruction.opcode == "dynamic-reshape";
    if (dynamic || !operand.dimensions().contains(&crate::UNBOUNDED_SIZE))
        && product(&operand)? != product(&result)?
    {
        return Err(Error::new(
            "shape",
            "reshape text input and output static extent products differ",
        ));
    }
    if dynamic && instruction.operand_ids.len() - 1 != result.rank()? {
        return Err(Error::new(
            "arity",
            "dynamic reshape text requires one size operand per result dimension",
        ));
    }
    Ok(())
}

fn canonical_shape(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
) -> Result<Option<Shape>> {
    let operand = |index: usize| -> Result<Shape> {
        let id = instruction
            .operand_ids
            .get(index)
            .ok_or_else(|| Error::new("arity", "context operation missing operand"))?;
        let shape = instructions
            .iter()
            .find(|i| i.id == *id)
            .and_then(|i| i.shape.as_ref())
            .ok_or_else(|| Error::new("reference", "context operand shape missing"))?;
        Shape::from_proto(shape.clone())
    };
    let data = || -> Result<Shape> {
        let shape = Shape::from_proto(
            instruction
                .shape
                .as_ref()
                .ok_or_else(|| Error::new("shape", "missing context result"))?
                .clone(),
        )?;
        shape.subshape(&[0])
    };
    Ok(match instruction.opcode.as_str() {
        "send" => Some(Shape::tuple(vec![
            operand(0)?,
            Shape::scalar(PrimitiveType::U32)?.with_default_layout()?,
            Shape::token(),
        ])?),
        "recv" => Some(Shape::tuple(vec![
            data()?,
            Shape::scalar(PrimitiveType::U32)?.with_default_layout()?,
            Shape::token(),
        ])?),
        "recv-done" => Some(Shape::tuple(vec![
            operand(0)?.subshape(&[0])?,
            Shape::token(),
        ])?),
        "send-done" | "outfeed" => Some(Shape::token()),
        "infeed" => Some(Shape::tuple(vec![data()?, Shape::token()])?),
        _ => None,
    })
}

pub(super) fn normalize(
    computation: &mut HloComputationProto,
    channels: &BTreeSet<i64>,
) -> Result<()> {
    normalize_with_inference(computation, channels, |_, _| Ok(None))
}

pub(super) fn normalize_with_inference(
    computation: &mut HloComputationProto,
    channels: &BTreeSet<i64>,
    mut infer: impl FnMut(&HloInstructionProto, &[HloInstructionProto]) -> Result<Option<Shape>>,
) -> Result<()> {
    for id in crate::ir::instruction_order(computation)? {
        let index = computation
            .instructions
            .iter()
            .position(|i| i.id == id)
            .ok_or_else(|| Error::new("reference", "missing context instruction"))?;
        if let Some(shape) = infer(&computation.instructions[index], &computation.instructions)? {
            computation.instructions[index].shape = Some(shape.into_proto()).into();
        }
        if computation.instructions[index].opcode == "outfeed"
            && computation.instructions[index].outfeed_shape.is_none()
        {
            let shape = computation.instructions[index]
                .operand_ids
                .first()
                .and_then(|id| computation.instructions.iter().find(|i| i.id == *id))
                .and_then(|instruction| instruction.shape.as_ref())
                .ok_or_else(|| Error::new("shape", "outfeed requires a shaped data operand"))?
                .clone();
            computation.instructions[index].outfeed_shape = Some(shape).into();
        }
        let instruction = &computation.instructions[index];
        validate_reshape(instruction, &computation.instructions)?;
        let comparison = if instruction.opcode == "compare" {
            Some(comparison_order(instruction, &computation.instructions)?)
        } else {
            None
        };
        let dimensions = if instruction.opcode == "map" {
            Some(map_dimensions(instruction)?)
        } else {
            None
        };
        check_channels(instruction, &computation.instructions, channels)?;
        if let Some(shape) = canonical_shape(instruction, &computation.instructions)? {
            computation.instructions[index].shape = Some(shape.into_proto()).into();
        }
        if let Some(order) = comparison {
            computation.instructions[index].comparison_order = order;
            computation.instructions[index].comparison_type.clear();
        }
        if let Some(dimensions) = dimensions {
            computation.instructions[index].dimensions = dimensions;
        }
    }
    Ok(())
}

pub(super) fn check_print(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
    channels: &BTreeSet<i64>,
) -> Result<()> {
    check_channels(instruction, instructions, channels)
        .map_err(|error| Error::new("unsupported", error.message))?;
    validate_reshape(instruction, instructions)?;
    if instruction.opcode == "compare"
        && (!instruction.comparison_type.is_empty()
            || comparison_order(instruction, instructions)? != instruction.comparison_order)
    {
        return Err(Error::new(
            "unsupported",
            "comparison attributes would be normalized by the pinned text constructor",
        ));
    }
    if instruction.opcode == "map" && map_dimensions(instruction)? != instruction.dimensions {
        return Err(Error::new(
            "unsupported",
            "map dimensions would be normalized by the pinned text constructor",
        ));
    }
    if let Some(expected) = canonical_shape(instruction, instructions)? {
        let actual = Shape::from_proto(
            instruction
                .shape
                .as_ref()
                .ok_or_else(|| Error::new("shape", "missing context result"))?
                .clone(),
        )?;
        if actual != expected {
            return Err(Error::new(
                "unsupported",
                "context result would be changed by pinned text constructor normalization",
            ));
        }
    }
    Ok(())
}

fn check_channels(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
    channels: &BTreeSet<i64>,
) -> Result<()> {
    if matches!(instruction.opcode.as_str(), "recv-done" | "send-done") {
        let start = instruction
            .operand_ids
            .first()
            .and_then(|id| instructions.iter().find(|i| i.id == *id))
            .ok_or_else(|| Error::new("arity", "send/recv done missing operand"))?;
        if matches!(
            start.opcode.as_str(),
            "send" | "recv" | "send-done" | "recv-done"
        ) && crate::ir::channels::value(start, channels.contains(&start.id))
            != crate::ir::channels::value(instruction, channels.contains(&instruction.id))
        {
            return Err(Error::new("channel", "start/done channel differs in text"));
        }
    }
    Ok(())
}
