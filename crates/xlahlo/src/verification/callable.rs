use super::{LayoutComparison, same_shapes};
use crate::{
    ComputationId, Error, Module, Opcode, PrimitiveType, Result, Shape, VerifyOptions,
    proto::{
        hlo::HloInstructionProto,
        xla_data::{ProgramShapeProto, ShapeProto},
    },
};

pub(super) fn verify_parameters(
    operands: &[Shape],
    program: &ProgramShapeProto,
    options: VerifyOptions,
    allow_extra: bool,
) -> Result<()> {
    if program.parameters.len() > operands.len()
        || (!allow_extra && program.parameters.len() != operands.len())
    {
        return Err(Error::new(
            "callee",
            "operand/parameter cardinality mismatch",
        ));
    }
    for (operand, parameter) in operands.iter().zip(&program.parameters) {
        let parameter = Shape::from_proto(parameter.clone())?;
        if if options.layout_sensitive {
            !same_shapes(operand, &parameter, LayoutComparison::Exact)
        } else {
            !operand.compatible(&parameter)
        } {
            return Err(Error::new("callee", "operand and parameter shapes differ"));
        }
    }
    Ok(())
}

pub(super) fn verify_ownership(module: &Module) -> Result<()> {
    for computation in module.computations() {
        let computation = module.computation(computation)?;
        let callers: Vec<_> = module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .filter(|i| i.called_computation_ids.contains(&computation.id))
            .collect();
        if !callers.iter().any(|i| i.opcode == "fusion") {
            continue;
        }
        if callers.len() != 1
            || callers[0].opcode != "fusion"
            || computation.id == module.as_proto().entry_computation_id
        {
            return Err(Error::new(
                "fusion",
                "a fusion computation must have one owning fusion and cannot be entry",
            ));
        }
        let used: std::collections::BTreeSet<_> = computation
            .instructions
            .iter()
            .flat_map(|i| i.operand_ids.iter().copied())
            .collect();
        for instruction in &computation.instructions {
            if instruction.id == computation.root_id {
                if used.contains(&instruction.id) {
                    return Err(Error::new("fusion", "fusion root cannot have users"));
                }
            } else if !used.contains(&instruction.id)
                && !module.instruction_has_local_side_effect(
                    module.instruction_handle(computation.id, instruction.id),
                    Default::default(),
                )?
            {
                return Err(Error::new(
                    "fusion",
                    format!("unused non-root instruction {}", instruction.name),
                ));
            }
        }
    }
    Ok(())
}

fn has_layout(shape: &ShapeProto) -> bool {
    if shape.element_type.enum_value_or_default() == PrimitiveType::TUPLE {
        shape.tuple_shapes.iter().all(has_layout)
    } else {
        !shape
            .element_type
            .enum_value_or_default()
            .is_array_element()
            || shape.layout.is_some()
    }
}

pub(super) fn verify_custom_call(
    instruction: &HloInstructionProto,
    output: &Shape,
    operands: &[Shape],
) -> Result<()> {
    if instruction.constrain_layout
        && !matches!(
            instruction.custom_call_target.as_str(),
            "LayoutConstraint" | "control_dep"
        )
    {
        if !has_layout(output.as_proto())
            || operands.len() != instruction.operand_shapes_with_layout.len()
        {
            return Err(Error::new(
                "custom-call",
                "layout-constrained custom call requires result and operand layouts",
            ));
        }
        for (operand, constraint) in operands.iter().zip(&instruction.operand_shapes_with_layout) {
            let constraint = Shape::from_proto(constraint.clone())?;
            if !has_layout(constraint.as_proto()) || !constraint.compatible(operand) {
                return Err(Error::new(
                    "custom-call",
                    "operand layout constraint is missing layout or has incompatible geometry",
                ));
            }
        }
    }
    if instruction.custom_call_target == "GetRngSeed"
        && (!operands.is_empty()
            || output.element_type() != PrimitiveType::U64
            || output.rank()? != 0)
    {
        return Err(Error::new(
            "custom-call",
            "GetRngSeed requires no operands and a U64 scalar result",
        ));
    }
    Ok(())
}

pub(super) fn verify_aliases(
    module: &Module,
    computation: ComputationId,
    instruction: &HloInstructionProto,
    output: &Shape,
    operands: &[Shape],
    options: VerifyOptions,
) -> Result<()> {
    let opcode: Opcode = instruction.opcode.parse()?;
    if !matches!(opcode, Opcode::Call | Opcode::Fusion | Opcode::CustomCall) {
        return Ok(());
    }
    let computation = module.computation(computation)?;
    let dead = computation.id != module.as_proto().entry_computation_id
        && !module
            .as_proto()
            .computations
            .iter()
            .flat_map(|c| &c.instructions)
            .any(|i| i.called_computation_ids.contains(&computation.id));
    let skip = instruction
        .frontend_attributes
        .as_ref()
        .is_some_and(|a| a.map.contains_key("xla_skip_custom_call_alias_shape_check"));
    for alias in &instruction.output_operand_aliasing {
        let operand = usize::try_from(alias.operand_index)
            .ok()
            .and_then(|i| operands.get(i))
            .ok_or_else(|| Error::new("alias", "alias operand index out of range"))?;
        let index = |index: &[i64]| {
            index
                .iter()
                .map(|&i| {
                    usize::try_from(i).map_err(|_| Error::new("alias", "negative shape index"))
                })
                .collect::<Result<Vec<_>>>()
        };
        let mut operand = operand.subshape(&index(&alias.operand_shape_index)?)?;
        let mut output = output.subshape(&index(&alias.output_shape_index)?)?;
        let scalar = if operand.element_type() == PrimitiveType::BUFFER {
            operand.array_view()?.element_count()? == 1
        } else {
            operand.is_array() && operand.element_count()? == 1
        };
        if opcode == Opcode::CustomCall && skip {
            return Ok(());
        }
        if opcode == Opcode::CustomCall
            && matches!(instruction.custom_call_target.as_str(), "Pin" | "Unpin")
        {
            for shape in [&mut operand, &mut output] {
                if shape.element_type() == PrimitiveType::BUFFER {
                    *shape = shape.array_view()?;
                }
            }
        }
        if opcode == Opcode::Call || (opcode == Opcode::CustomCall && !options.layout_sensitive) {
            fn clear_dynamic(shape: &mut ShapeProto) {
                let mut pending = vec![shape];
                while let Some(shape) = pending.pop() {
                    if shape
                        .element_type
                        .enum_value()
                        .is_ok_and(PrimitiveType::is_array_element)
                    {
                        shape.is_dynamic_dimension = vec![false; shape.dimensions.len()];
                    }
                    if matches!(
                        shape.element_type.enum_value(),
                        Ok(PrimitiveType::TUPLE | PrimitiveType::BUFFER)
                    ) {
                        pending.extend(&mut shape.tuple_shapes);
                    }
                }
            }
            let (mut a, mut b) = (operand.into_proto(), output.into_proto());
            clear_dynamic(&mut a);
            clear_dynamic(&mut b);
            operand = Shape::from_proto(a)?;
            output = Shape::from_proto(b)?;
        }
        let mode = if opcode == Opcode::Fusion && (dead || computation.is_fusion_computation) {
            LayoutComparison::IgnoreMemoryAndTiles
        } else if opcode == Opcode::CustomCall && scalar {
            LayoutComparison::IgnoreMemory
        } else {
            LayoutComparison::Exact
        };
        let equal = if options.layout_sensitive {
            same_shapes(&operand, &output, mode)
        } else {
            operand.compatible(&output)
        };
        if !equal {
            return Err(Error::new(
                "alias",
                "aliased output and operand subshapes differ",
            ));
        }
    }
    Ok(())
}
