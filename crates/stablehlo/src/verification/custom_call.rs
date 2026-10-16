//! Custom-call ABI, layout and alias contracts, including buffer intrinsics.

use std::collections::{HashMap, HashSet};

use super::{BuildError, compatible_dimensions, shaped_dimensions, validate_dimensions};
use crate::{
    Attribute, Instruction, Layout, OutputOperandAlias, StableHloAttribute, StableHloOp, Type,
    ValueId,
};

fn invalid(instruction: &Instruction, attribute: &'static str) -> BuildError {
    BuildError::MissingOrInvalidAttribute {
        operation: instruction.operation().name(),
        attribute,
    }
}

fn layouts(instruction: &Instruction, name: &str) -> Option<Vec<Layout>> {
    match instruction.operation().attributes()?.get(name)? {
        Attribute::Array(values) => Some(
            values
                .iter()
                .map(|attribute| {
                    Layout(
                        crate::attribute::builtin::index_tensor_values(attribute)
                            .expect("verified index tensor layout"),
                    )
                })
                .collect(),
        ),
        _ => unreachable!("verified layout attribute"),
    }
}

fn verify_layouts<'a>(
    instruction: &Instruction,
    name: &'static str,
    types: impl IntoIterator<Item = &'a Type>,
    layouts: &[Layout],
) -> Result<(), BuildError> {
    let types = types.into_iter().collect::<Vec<_>>();
    if types.len() != layouts.len() {
        return Err(invalid(instruction, name));
    }
    for (ty, layout) in types.into_iter().zip(layouts) {
        if matches!(ty, Type::Tuple(_)) {
            return Err(invalid(instruction, name));
        }
        if matches!(ty, Type::UnrankedTensor(_)) {
            continue;
        }
        let shape = match ty {
            Type::Buffer(buffer) => Some(buffer.dimensions()),
            _ => shaped_dimensions(ty),
        };
        let rank = shape.map_or(0, <[Option<u64>]>::len);
        if layout.0.len() != rank || validate_dimensions(&layout.0, rank).is_none() {
            return Err(invalid(instruction, name));
        }
    }
    Ok(())
}

fn part<'a>(mut ty: &'a Type, path: &[i64]) -> Option<&'a Type> {
    for &index in path {
        let Type::Tuple(elements) = ty else {
            return None;
        };
        ty = elements.get(usize::try_from(index).ok()?)?;
    }
    Some(ty)
}

fn aliases(instruction: &Instruction) -> Vec<&OutputOperandAlias> {
    match instruction
        .operation()
        .attributes()
        .and_then(|attributes| attributes.get("output_operand_aliases"))
    {
        None => vec![],
        Some(Attribute::Array(values)) => values
            .iter()
            .map(|value| match value {
                Attribute::StableHlo(value) => match value.as_ref() {
                    StableHloAttribute::OutputOperandAlias(value) => value,
                    _ => unreachable!("verified alias attribute"),
                },
                _ => unreachable!("verified alias attribute"),
            })
            .collect(),
        _ => unreachable!("verified alias attribute"),
    }
}

fn buffer_paths(ty: &Type, path: &mut Vec<i64>, paths: &mut Vec<Vec<i64>>) {
    match ty {
        Type::Buffer(_) => paths.push(path.clone()),
        Type::Tuple(elements) => {
            for (index, element) in elements.iter().enumerate() {
                path.push(index as i64);
                buffer_paths(element, path, paths);
                path.pop();
            }
        }
        _ => {}
    }
}

fn verify_intrinsic(
    instruction: &Instruction,
    target: &str,
    available: &HashMap<ValueId, Type>,
) -> Result<bool, BuildError> {
    let inputs = instruction.operands();
    let results = instruction.results();
    match target {
        "CreateBuffer" => {
            if !inputs.is_empty()
                || results.len() != 1
                || !matches!(results[0].ty(), Type::Buffer(_))
            {
                return Err(invalid(instruction, "call_target_name"));
            }
            Ok(true)
        }
        "Pin" | "Unpin" => {
            if inputs.len() != 1 || results.len() != 1 {
                return Err(invalid(instruction, "call_target_name"));
            }
            let (tensor, buffer) = if target == "Pin" {
                (&available[&inputs[0]], results[0].ty())
            } else {
                (results[0].ty(), &available[&inputs[0]])
            };
            if !matches!((tensor, buffer), (Type::Tensor(tensor), Type::Buffer(buffer))
                if tensor.element() == buffer.element()
                    && compatible_dimensions(tensor.dimensions(), buffer.dimensions()))
            {
                return Err(invalid(instruction, "call_target_name"));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if opcode != StableHloOp::CustomCall {
        return Ok(());
    }
    let attributes = instruction
        .operation()
        .attributes()
        .expect("StableHLO attributes");
    let typed_ffi = matches!(
        attributes.get("api_version"),
        Some(Attribute::Integer { value: 4, .. })
    );
    if let Some(config) = attributes.get("backend_config")
        && !matches!(
            (typed_ffi, config),
            (true, Attribute::Dictionary(_)) | (false, Attribute::String(_) | Attribute::Bytes(_))
        )
    {
        return Err(invalid(instruction, "backend_config"));
    }
    match (
        layouts(instruction, "operand_layouts"),
        layouts(instruction, "result_layouts"),
    ) {
        (Some(inputs), Some(outputs)) => {
            verify_layouts(
                instruction,
                "operand_layouts",
                instruction.operands().iter().map(|id| &available[id]),
                &inputs,
            )?;
            let result_types = match instruction.results() {
                [result] => match result.ty() {
                    Type::Tuple(elements) => elements.iter().collect::<Vec<_>>(),
                    ty => vec![ty],
                },
                results => results.iter().map(|result| result.ty()).collect(),
            };
            verify_layouts(instruction, "result_layouts", result_types, &outputs)?;
        }
        (None, None) => {}
        _ => return Err(invalid(instruction, "operand_layouts")),
    }
    let output = match instruction.results() {
        [result] => result.ty().clone(),
        results => Type::Tuple(results.iter().map(|result| result.ty().clone()).collect()),
    };
    let aliases = aliases(instruction);
    let mut buffer_inputs = HashSet::new();
    let mut buffer_outputs = HashSet::new();
    for alias in aliases {
        let input = usize::try_from(alias.operand_index)
            .ok()
            .and_then(|index| instruction.operands().get(index))
            .and_then(|id| part(&available[id], &alias.operand_tuple_indices));
        let result = part(&output, &alias.output_tuple_indices);
        let (Some(input), Some(result)) = (input, result) else {
            return Err(invalid(instruction, "output_operand_aliases"));
        };
        if input != result {
            return Err(invalid(instruction, "output_operand_aliases"));
        }
        if matches!(input, Type::Buffer(_))
            && (!buffer_inputs.insert((alias.operand_index, alias.operand_tuple_indices.clone()))
                || !buffer_outputs.insert(alias.output_tuple_indices.clone()))
        {
            return Err(invalid(instruction, "output_operand_aliases"));
        }
    }
    let Some(Attribute::String(target)) = attributes.get("call_target_name") else {
        unreachable!("verified custom-call target");
    };
    if verify_intrinsic(instruction, target, available)? {
        return Ok(());
    }
    let mut paths = vec![];
    buffer_paths(&output, &mut vec![], &mut paths);
    if paths.iter().any(|path| !buffer_outputs.contains(path)) {
        return Err(invalid(instruction, "output_operand_aliases"));
    }
    Ok(())
}
