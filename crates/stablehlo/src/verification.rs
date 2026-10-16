use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use crate::{
    Attribute, AttributeValidationError, Block, Computation, ComputationId, ElementType,
    Instruction, Operation, StableHloOp, TensorType, Type, TypeError, ValueId,
};

mod attributes;
mod collectives;
mod constants;
mod convolution;
mod custom_call;
mod functions;
mod gather;
mod regions;
mod scatter;
mod schema;
mod tensor_ops;
mod window;

pub(crate) use functions::{
    verify_function_attributes, verify_module_attributes, verify_resolved_mesh,
};

pub(crate) fn verify_computation(computation: &Computation) -> Result<(), BuildError> {
    verify_location(computation.location())?;
    verify_location(computation.return_location())?;
    if computation.name.is_empty() {
        return Err(BuildError::InvalidFunctionName);
    }
    verify_function_attributes(computation, &computation.attributes)?;
    if let Some(signature) = &computation.declaration {
        Type::Function(signature.clone())
            .verify()
            .map_err(BuildError::InvalidFunctionType)?;
        return Ok(());
    }
    let mut parameter_numbers = Vec::new();
    let mut available = HashMap::new();
    let mut all_values = HashSet::new();
    let mut next_top_level_id = 0_u32;
    for instruction in &computation.instructions {
        verify_instruction(
            instruction,
            &mut available,
            &mut all_values,
            &mut next_top_level_id,
            &mut parameter_numbers,
        )?;
    }
    parameter_numbers.sort_unstable();
    if parameter_numbers
        .iter()
        .copied()
        .ne(0..parameter_numbers.len() as u32)
    {
        return Err(BuildError::InvalidParameterNumbers);
    }
    for root in &computation.roots {
        if !available.contains_key(root) {
            return Err(BuildError::UnknownValue(*root));
        }
    }
    constants::verify(computation)
}

pub(crate) fn tensor_element_types_match(lhs: &Type, rhs: &Type) -> bool {
    match (lhs, rhs) {
        (Type::Tensor(lhs), Type::Tensor(rhs)) => lhs.element() == rhs.element(),
        (Type::QuantizedTensor(lhs), Type::QuantizedTensor(rhs)) => {
            same_quantization_storage(lhs, rhs)
                && lhs.quantization_dimension() == rhs.quantization_dimension()
                && same_quantization_parameters(lhs, rhs)
        }
        _ => false,
    }
}

fn verify_instruction(
    instruction: &Instruction,
    available: &mut HashMap<ValueId, Type>,
    all_values: &mut HashSet<ValueId>,
    next_top_level_id: &mut u32,
    parameter_numbers: &mut Vec<u32>,
) -> Result<(), BuildError> {
    if matches!(
        &instruction.operation,
        Operation::StableHlo {
            opcode: StableHloOp::Return,
            ..
        }
    ) {
        return Err(BuildError::ReturnOutsideRegion);
    }
    verify_attributes(instruction)?;
    verify_region_arity(instruction)?;
    if let Some(expected) = instruction.operation.expected_arity()
        && instruction.operands.len() != expected
    {
        return Err(BuildError::InvalidArity {
            operation: instruction.operation.name(),
            expected,
            actual: instruction.operands.len(),
        });
    }
    if let Some(expected) = instruction.operation.expected_result_arity()
        && instruction.results.len() != expected
    {
        return Err(BuildError::InvalidResultArity {
            operation: instruction.operation.name(),
            expected,
            actual: instruction.results.len(),
        });
    }
    verify_variadic_cardinality(instruction)?;
    verify_result_types(instruction)?;
    for &operand in &instruction.operands {
        if !available.contains_key(&operand) {
            return Err(BuildError::UnknownOrNonDominatingValue {
                instruction: instruction
                    .results
                    .first()
                    .map_or(operand, |result| result.value()),
                operand,
            });
        }
    }

    for region in &instruction.regions {
        if region.blocks().is_empty() {
            return Err(BuildError::EmptyRegion);
        }
        if let Some(expected) = instruction.operation.expected_region_block_count()
            && region.blocks().len() != expected
        {
            return Err(BuildError::InvalidRegionBlockCount {
                operation: instruction.operation.name(),
                expected,
                actual: region.blocks().len(),
            });
        }
        let region_values = visible_region_values(instruction, available);
        for block in region.blocks() {
            verify_block(
                block,
                &region_values,
                all_values,
                parameter_numbers,
                instruction.operation.requires_region_return(),
            )?;
        }
    }
    verify_control_flow_region_contract(instruction, available)?;
    verify_region_signature_counts(instruction)?;
    verify_async_start_region(instruction, available)?;

    match &instruction.operation {
        Operation::Parameter { number, .. } => {
            if instruction.results().len() != 1 {
                return Err(BuildError::InvalidParameterResultCount);
            }
            parameter_numbers.push(*number);
        }
        Operation::StableHlo { opcode, .. } => {
            verify_builtin_type_rule(instruction, *opcode, available)?;
        }
        Operation::Call { .. } => {}
    }

    for result in instruction.results() {
        if !all_values.insert(result.value()) {
            return Err(BuildError::DuplicateValue(result.value()));
        }
        if result.value() != ValueId(*next_top_level_id) {
            return Err(BuildError::InvalidResultId {
                expected: ValueId(*next_top_level_id),
                actual: result.value(),
            });
        }
        *next_top_level_id = next_top_level_id
            .checked_add(1)
            .ok_or(BuildError::TooManyValues)?;
        available.insert(result.value(), result.ty().clone());
    }
    Ok(())
}

fn verify_block(
    block: &Block,
    outer_values: &HashMap<ValueId, Type>,
    all_values: &mut HashSet<ValueId>,
    parameter_numbers: &mut Vec<u32>,
    requires_return: bool,
) -> Result<(), BuildError> {
    if requires_return {
        if !matches!(
            block.instructions().last().map(Instruction::operation),
            Some(Operation::StableHlo {
                opcode: StableHloOp::Return,
                ..
            })
        ) {
            return Err(BuildError::MissingRegionTerminator);
        }
        if block.instructions()[..block.instructions().len() - 1]
            .iter()
            .any(|instruction| {
                matches!(
                    instruction.operation(),
                    Operation::StableHlo {
                        opcode: StableHloOp::Return,
                        ..
                    }
                )
            })
        {
            return Err(BuildError::UnexpectedRegionTerminator);
        }
    }
    let mut available = outer_values.clone();
    for argument in block.arguments() {
        verify_location(argument.location())?;
        if !argument.ty().is_value_type() {
            return Err(BuildError::InvalidType(TypeError::ExpectedValueType {
                context: "block argument",
                index: 0,
            }));
        }
        argument.ty().verify().map_err(BuildError::InvalidType)?;
        if !all_values.insert(argument.value()) {
            return Err(BuildError::DuplicateValue(argument.value()));
        }
        available.insert(argument.value(), argument.ty().clone());
    }
    for instruction in block.instructions() {
        verify_nested_instruction(instruction, &mut available, all_values, parameter_numbers)?;
    }
    Ok(())
}

fn verify_nested_instruction(
    instruction: &Instruction,
    available: &mut HashMap<ValueId, Type>,
    all_values: &mut HashSet<ValueId>,
    parameter_numbers: &mut Vec<u32>,
) -> Result<(), BuildError> {
    if matches!(&instruction.operation, Operation::Parameter { .. }) {
        return Err(BuildError::ParameterInsideRegion);
    }
    verify_attributes(instruction)?;
    verify_region_arity(instruction)?;
    if let Some(expected) = instruction.operation.expected_arity()
        && instruction.operands.len() != expected
    {
        return Err(BuildError::InvalidArity {
            operation: instruction.operation.name(),
            expected,
            actual: instruction.operands.len(),
        });
    }
    if let Some(expected) = instruction.operation.expected_result_arity()
        && instruction.results.len() != expected
    {
        return Err(BuildError::InvalidResultArity {
            operation: instruction.operation.name(),
            expected,
            actual: instruction.results.len(),
        });
    }
    verify_variadic_cardinality(instruction)?;
    verify_result_types(instruction)?;
    for &operand in &instruction.operands {
        if !available.contains_key(&operand) {
            return Err(BuildError::UnknownOrNonDominatingValue {
                instruction: instruction
                    .results
                    .first()
                    .map_or(operand, |result| result.value()),
                operand,
            });
        }
    }
    for region in &instruction.regions {
        if region.blocks().is_empty() {
            return Err(BuildError::EmptyRegion);
        }
        if let Some(expected) = instruction.operation.expected_region_block_count()
            && region.blocks().len() != expected
        {
            return Err(BuildError::InvalidRegionBlockCount {
                operation: instruction.operation.name(),
                expected,
                actual: region.blocks().len(),
            });
        }
        let region_values = visible_region_values(instruction, available);
        for block in region.blocks() {
            verify_block(
                block,
                &region_values,
                all_values,
                parameter_numbers,
                instruction.operation.requires_region_return(),
            )?;
        }
    }
    verify_control_flow_region_contract(instruction, available)?;
    verify_region_signature_counts(instruction)?;
    verify_async_start_region(instruction, available)?;
    if let Operation::StableHlo { opcode, .. } = &instruction.operation {
        verify_builtin_type_rule(instruction, *opcode, available)?;
    }
    for result in instruction.results() {
        if !all_values.insert(result.value()) {
            return Err(BuildError::DuplicateValue(result.value()));
        }
        available.insert(result.value(), result.ty().clone());
    }
    Ok(())
}

fn verify_result_types(instruction: &Instruction) -> Result<(), BuildError> {
    for (index, result) in instruction.results().iter().enumerate() {
        if !result.ty().is_value_type() {
            return Err(BuildError::InvalidType(TypeError::ExpectedValueType {
                context: "operation result",
                index,
            }));
        }
        result.ty().verify().map_err(BuildError::InvalidType)?;
    }
    Ok(())
}

fn visible_region_values(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> HashMap<ValueId, Type> {
    if instruction.operation.is_isolated_from_above() {
        HashMap::new()
    } else {
        available.clone()
    }
}

fn verify_control_flow_region_contract(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let opcode = match &instruction.operation {
        Operation::StableHlo { opcode, .. } => *opcode,
        Operation::Parameter { .. } | Operation::Call { .. } => return Ok(()),
    };
    if !matches!(
        opcode,
        StableHloOp::If | StableHloOp::Case | StableHloOp::While
    ) {
        return Ok(());
    }

    if matches!(opcode, StableHloOp::If | StableHloOp::Case) {
        let selector = instruction.operands[0];
        let selector_type = available.get(&selector);
        let valid = match opcode {
            StableHloOp::If => {
                selector_type == Some(&Type::from(TensorType::scalar(ElementType::I1)))
            }
            StableHloOp::Case => matches!(
                selector_type,
                Some(Type::Tensor(tensor))
                    if tensor.rank() == 0
                        && tensor.element().is_integer()
                        && tensor.element() != ElementType::I1
            ),
            _ => false,
        };
        if !valid {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value: selector,
            });
        }
    }

    if opcode == StableHloOp::While && instruction.operands.len() != instruction.results.len() {
        return Err(BuildError::MismatchedOperandResultCount {
            operation: instruction.operation.name(),
            operands: instruction.operands.len(),
            results: instruction.results.len(),
        });
    }
    if opcode == StableHloOp::While {
        let inputs = instruction
            .operands
            .iter()
            .map(|id| available[id].clone())
            .collect::<Vec<_>>();
        let inferred =
            crate::infer_while_types(&inputs).map_err(|error| BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            })?;
        for (expected, result) in inferred.iter().zip(&instruction.results) {
            if !compatible_value_types(expected, result.ty()) {
                return Err(BuildError::InvalidResultType {
                    operation: instruction.operation.name(),
                    result: result.value(),
                });
            }
        }
    }

    for (region_index, region) in instruction.regions.iter().enumerate() {
        let block = &region.blocks()[0];
        let expected_arguments = if opcode == StableHloOp::While {
            instruction.operands.len()
        } else {
            0
        };
        if block.arguments().len() != expected_arguments {
            return Err(BuildError::InvalidRegionSignature {
                operation: instruction.operation.name(),
                region: region_index,
                expected_arguments,
                actual_arguments: block.arguments().len(),
                expected_results: instruction.results.len(),
                actual_results: block
                    .instructions()
                    .last()
                    .map_or(0, |terminator| terminator.operands().len()),
            });
        }

        if opcode == StableHloOp::While {
            for (argument, operand) in block.arguments().iter().zip(&instruction.operands) {
                if !compatible_value_types(&available[operand], argument.ty()) {
                    return Err(BuildError::InvalidResultType {
                        operation: instruction.operation.name(),
                        result: argument.value(),
                    });
                }
            }
        }

        let Some(terminator) = block.instructions().last() else {
            return Err(BuildError::MissingRegionTerminator);
        };
        let actual_results = terminator.operands().len();
        let expected_results = if opcode == StableHloOp::While && region_index == 0 {
            1
        } else {
            instruction.results.len()
        };
        if actual_results != expected_results {
            return Err(BuildError::InvalidRegionSignature {
                operation: instruction.operation.name(),
                region: region_index,
                expected_arguments,
                actual_arguments: block.arguments().len(),
                expected_results,
                actual_results,
            });
        }

        if opcode == StableHloOp::While && region_index == 0 {
            let returned = terminator.operands()[0];
            if resolve_region_value_type(block, available, returned)
                != Some(&Type::from(TensorType::scalar(ElementType::I1)))
            {
                return Err(BuildError::InvalidOperandType {
                    operation: instruction.operation.name(),
                    value: returned,
                });
            }
        } else {
            for (index, (&returned, result)) in terminator
                .operands()
                .iter()
                .zip(&instruction.results)
                .enumerate()
            {
                let expected = if opcode == StableHloOp::While {
                    &available[&instruction.operands[index]]
                } else {
                    result.ty()
                };
                let returned_type = resolve_region_value_type(block, available, returned)
                    .expect("verified region return");
                if !compatible_value_types(returned_type, expected) {
                    return Err(BuildError::InvalidResultType {
                        operation: instruction.operation.name(),
                        result: result.value(),
                    });
                }
                if opcode != StableHloOp::While && region_index > 0 {
                    let first = &instruction.regions[0].blocks()[0];
                    let first_return = first
                        .instructions()
                        .last()
                        .expect("verified terminator")
                        .operands()[index];
                    let first_type = resolve_region_value_type(first, available, first_return)
                        .expect("verified region return");
                    if !compatible_value_types(first_type, returned_type) {
                        return Err(BuildError::InvalidResultType {
                            operation: instruction.operation.name(),
                            result: result.value(),
                        });
                    }
                }
            }
        }
    }
    if opcode != StableHloOp::While {
        let branches = instruction
            .regions
            .iter()
            .map(|region| {
                let block = &region.blocks()[0];
                block
                    .instructions()
                    .last()
                    .expect("verified terminator")
                    .operands()
                    .iter()
                    .map(|&returned| {
                        resolve_region_value_type(block, available, returned)
                            .expect("verified region return")
                            .clone()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inferred = crate::inference::infer_conditional_types(
            opcode,
            &available[&instruction.operands[0]],
            &branches.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
        .map_err(|error| BuildError::TypeInference {
            operation: instruction.operation.name(),
            error,
        })?;
        for (expected, result) in inferred.iter().zip(&instruction.results) {
            if !compatible_value_types(expected, result.ty()) {
                return Err(BuildError::InvalidResultType {
                    operation: instruction.operation.name(),
                    result: result.value(),
                });
            }
        }
    }
    Ok(())
}

fn resolve_region_value_type<'a>(
    block: &'a Block,
    outer_values: &'a HashMap<ValueId, Type>,
    value: ValueId,
) -> Option<&'a Type> {
    if let Some(argument) = block
        .arguments()
        .iter()
        .find(|argument| argument.value() == value)
    {
        return Some(argument.ty());
    }
    for instruction in block.instructions() {
        if let Some(result) = instruction
            .results()
            .iter()
            .find(|result| result.value() == value)
        {
            return Some(result.ty());
        }
        for region in instruction.regions() {
            for nested_block in region.blocks() {
                if let Some(ty) = resolve_region_value_type(nested_block, outer_values, value) {
                    return Some(ty);
                }
            }
        }
    }
    outer_values.get(&value)
}

fn verify_region_signature_counts(instruction: &Instruction) -> Result<(), BuildError> {
    let opcode = match &instruction.operation {
        Operation::StableHlo { opcode, .. } => *opcode,
        Operation::Parameter { .. } | Operation::Call { .. } => return Ok(()),
    };
    let expected = match opcode {
        StableHloOp::AllReduce | StableHloOp::CollectiveReduce => Some((2, 1)),
        StableHloOp::Reduce | StableHloOp::ReduceWindow => {
            Some((instruction.operands.len(), instruction.results.len()))
        }
        StableHloOp::ReduceScatter => Some((2, instruction.results.len())),
        StableHloOp::Map => Some((instruction.operands.len(), instruction.results.len())),
        StableHloOp::Sort => Some((instruction.operands.len().saturating_mul(2), 1)),
        StableHloOp::Scatter => Some((
            instruction.results.len().saturating_mul(2),
            instruction.results.len(),
        )),
        StableHloOp::SelectAndScatter => Some((2, 1)),
        _ => None,
    };
    let Some((expected_arguments, expected_results)) = expected else {
        return Ok(());
    };
    for (region_index, region) in instruction.regions.iter().enumerate() {
        let block = &region.blocks()[0];
        let actual_results = block
            .instructions()
            .last()
            .map_or(0, |terminator| terminator.operands().len());
        if block.arguments().len() != expected_arguments || actual_results != expected_results {
            return Err(BuildError::InvalidRegionSignature {
                operation: instruction.operation.name(),
                region: region_index,
                expected_arguments,
                actual_arguments: block.arguments().len(),
                expected_results,
                actual_results,
            });
        }
    }
    Ok(())
}

fn verify_async_start_region(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if !matches!(
        &instruction.operation,
        Operation::StableHlo {
            opcode: StableHloOp::AsyncStart,
            ..
        }
    ) {
        return Ok(());
    }
    let Some(result) = instruction.results.first() else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: ValueId(0),
        });
    };
    let Type::Future(future_types) = result.ty() else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    let block = &instruction.regions[0].blocks()[0];
    let Some(terminator) = block.instructions().last() else {
        return Err(BuildError::MissingRegionTerminator);
    };
    if block.arguments().len() != instruction.operands.len()
        || terminator.operands().len() != 1
        || future_types.len() != 1
    {
        return Err(BuildError::InvalidRegionSignature {
            operation: instruction.operation.name(),
            region: 0,
            expected_arguments: instruction.operands.len(),
            actual_arguments: block.arguments().len(),
            expected_results: 1,
            actual_results: terminator.operands().len(),
        });
    }
    for (argument, operand) in block.arguments().iter().zip(&instruction.operands) {
        if argument.ty() != &available[operand] {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value: argument.value(),
            });
        }
    }
    if block.instructions().len() != 2
        || !matches!(
            block.instructions()[0].operation(),
            Operation::StableHlo {
                opcode: StableHloOp::AllGather
                    | StableHloOp::AllReduce
                    | StableHloOp::AllToAll
                    | StableHloOp::CollectiveBroadcast
                    | StableHloOp::CollectivePermute
                    | StableHloOp::CollectiveReduce
                    | StableHloOp::DynamicSlice
                    | StableHloOp::DynamicUpdateSlice
                    | StableHloOp::ReduceScatter
                    | StableHloOp::Slice,
                ..
            }
        )
    {
        return Err(BuildError::InvalidAsyncBody);
    }
    for (&returned, expected) in terminator.operands().iter().zip(future_types) {
        if resolve_region_value_type(block, available, returned) != Some(expected) {
            return Err(BuildError::InvalidResultType {
                operation: instruction.operation.name(),
                result: instruction.results[0].value(),
            });
        }
    }
    Ok(())
}

fn verify_region_arity(instruction: &Instruction) -> Result<(), BuildError> {
    let actual = instruction.regions.len();
    let expected = instruction.operation.expected_region_arity();
    if let Some(expected) = expected
        && actual != expected
    {
        return Err(BuildError::InvalidRegionCount {
            operation: instruction.operation.name(),
            expected: Some(expected),
            actual,
        });
    }
    if actual < instruction.operation.minimum_region_arity() {
        return Err(BuildError::InvalidRegionCount {
            operation: instruction.operation.name(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn verify_variadic_cardinality(instruction: &Instruction) -> Result<(), BuildError> {
    let operation = instruction.operation.name();
    let operands = instruction.operands.len();
    let results = instruction.results.len();
    let valid = match instruction.operation {
        Operation::StableHlo {
            opcode:
                StableHloOp::AllGather
                | StableHloOp::AllReduce
                | StableHloOp::AllToAll
                | StableHloOp::OptimizationBarrier
                | StableHloOp::Sort,
            ..
        } => operands == results,
        Operation::StableHlo {
            opcode: StableHloOp::CollectiveBroadcast | StableHloOp::CollectiveReduce,
            ..
        } => {
            let dynamic_root = instruction
                .operation
                .attributes()
                .is_some_and(|attributes| attributes.contains_key("has_dynamic_root"));
            results > 0 && operands == results.saturating_add(usize::from(dynamic_root))
        }
        Operation::StableHlo {
            opcode: StableHloOp::Reduce | StableHloOp::ReduceWindow,
            ..
        } => operands > 0 && operands.is_multiple_of(2) && results == operands / 2,
        Operation::StableHlo {
            opcode: StableHloOp::Scatter,
            ..
        } => results > 0 && operands == results.saturating_mul(2).saturating_add(1),
        Operation::StableHlo {
            opcode: StableHloOp::Concatenate | StableHloOp::Map,
            ..
        } => operands > 0 && results == 1,
        Operation::StableHlo {
            opcode: StableHloOp::DynamicSlice,
            ..
        } => operands > 0 && results == 1,
        Operation::StableHlo {
            opcode: StableHloOp::DynamicUpdateSlice,
            ..
        } => operands >= 2 && results == 1,
        Operation::StableHlo {
            opcode: StableHloOp::Outfeed | StableHloOp::Send,
            ..
        } => operands >= 1 && results == 1,
        _ => true,
    };
    if valid {
        return Ok(());
    }
    if matches!(
        instruction.operation,
        Operation::StableHlo {
            opcode: StableHloOp::OptimizationBarrier,
            ..
        }
    ) {
        return Err(BuildError::MismatchedOperandResultCount {
            operation,
            operands,
            results,
        });
    }
    Err(BuildError::InvalidVariadicCardinality {
        operation,
        operands,
        results,
    })
}

fn verify_attributes(instruction: &Instruction) -> Result<(), BuildError> {
    verify_location(instruction.location())?;
    if let Some(attributes) = instruction.operation.attributes() {
        for (name, attribute) in attributes {
            attribute
                .verify()
                .map_err(|error| BuildError::InvalidAttribute {
                    name: name.clone(),
                    error,
                })?;
        }
        if let Operation::Call { attributes } = &instruction.operation {
            if !matches!(attributes.get("callee"), Some(Attribute::SymbolRef(path))
                if matches!(path.as_slice(), [name] if !name.is_empty()))
            {
                return Err(BuildError::MissingOrInvalidAttribute {
                    operation: "func.call",
                    attribute: "callee",
                });
            }
            functions::verify_signature_attributes(
                attributes,
                instruction.operands.len(),
                instruction.results.len(),
            )?;
            if !instruction.called_computations.is_empty() {
                return Err(BuildError::InvalidSymbolReference {
                    operation: "func.call",
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn verify_location(location: Option<&crate::Location>) -> Result<(), BuildError> {
    if let Some(location) = location {
        location
            .verify()
            .map_err(|error| BuildError::InvalidAttribute {
                name: "location".into(),
                error,
            })?;
    }
    Ok(())
}

fn verify_builtin_type_rule(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    match opcode {
        StableHloOp::Constant => {
            let value = instruction
                .operation
                .attributes()
                .and_then(|attributes| attributes.get("value"));
            let ty = match value {
                Some(Attribute::DenseElements { ty, .. })
                | Some(Attribute::SparseElements { ty, .. }) => ty.as_ref(),
                _ => {
                    return Err(BuildError::MissingOrInvalidAttribute {
                        operation: instruction.operation.name(),
                        attribute: "value",
                    });
                }
            };
            let result = &instruction.results[0];
            let expected_storage = if let Type::QuantizedTensor(tensor) = result.ty() {
                let mut storage =
                    TensorType::new(tensor.storage_type(), tensor.dimensions().to_vec());
                if let Some(encoding) = tensor.encoding() {
                    storage = storage.with_encoding(encoding.clone());
                }
                Type::from(storage)
            } else {
                result.ty().clone()
            };
            if *ty != expected_storage {
                return Err(BuildError::InvalidResultType {
                    operation: instruction.operation.name(),
                    result: result.value(),
                });
            }
        }
        StableHloOp::Iota | StableHloOp::DynamicIota => {
            verify_iota(instruction, opcode, available)?;
        }
        StableHloOp::Convert => verify_convert(instruction, available)?,
        StableHloOp::Complex => verify_complex(instruction, available)?,
        StableHloOp::DotGeneral => verify_dot_general(instruction, available)?,
        StableHloOp::Reshape => verify_reshape(instruction, available)?,
        StableHloOp::DynamicReshape => verify_dynamic_reshape(instruction, available)?,
        StableHloOp::Transpose => verify_transpose(instruction, available)?,
        StableHloOp::BroadcastInDim => verify_broadcast_in_dim(instruction, available)?,
        StableHloOp::DynamicBroadcastInDim => {
            verify_dynamic_broadcast_in_dim(instruction, available)?
        }
        StableHloOp::Reverse => verify_reverse(instruction, available)?,
        StableHloOp::Slice => verify_slice(instruction, available)?,
        StableHloOp::Clamp => verify_clamp(instruction, available)?,
        StableHloOp::Select => verify_select(instruction, available)?,
        StableHloOp::Concatenate => verify_concatenate(instruction, available)?,
        StableHloOp::Pad => verify_pad(instruction, available)?,
        StableHloOp::DynamicPad => verify_dynamic_pad(instruction, available)?,
        StableHloOp::DynamicSlice => verify_dynamic_slice(instruction, available)?,
        StableHloOp::DynamicUpdateSlice => verify_dynamic_update_slice(instruction, available)?,
        StableHloOp::GetDimensionSize => verify_get_dimension_size(instruction, available)?,
        StableHloOp::SetDimensionSize => verify_set_dimension_size(instruction, available)?,
        StableHloOp::CreateToken => {
            verify_single_result_type(instruction, &Type::Token)?;
        }
        StableHloOp::AfterAll => {
            let invalid_operand = instruction
                .operands
                .iter()
                .find(|operand| available.get(operand) != Some(&Type::Token));
            if let Some(&value) = invalid_operand {
                return Err(BuildError::InvalidOperandType {
                    operation: instruction.operation.name(),
                    value,
                });
            }
            verify_single_result_type(instruction, &Type::Token)?;
        }
        StableHloOp::Tuple => {
            let inputs = instruction
                .operands
                .iter()
                .map(|operand| available[operand].clone())
                .collect::<Vec<_>>();
            let expected =
                crate::infer_tuple_type(&inputs).map_err(|error| BuildError::TypeInference {
                    operation: instruction.operation.name(),
                    error,
                })?;
            verify_single_result_type(instruction, &expected)?;
        }
        StableHloOp::GetTupleElement => {
            let operand = instruction.operands.first().copied().unwrap_or(ValueId(0));
            let Some(Type::Tuple(elements)) = available.get(&operand) else {
                return Err(BuildError::InvalidOperandType {
                    operation: instruction.operation.name(),
                    value: operand,
                });
            };
            let Some(Attribute::Integer { value: index, .. }) = instruction
                .operation
                .attributes()
                .and_then(|attributes| attributes.get("index"))
            else {
                return Err(BuildError::MissingOrInvalidAttribute {
                    operation: instruction.operation.name(),
                    attribute: "index",
                });
            };
            let expected = crate::infer_get_tuple_element_type(
                &available[&operand],
                i32::try_from(*index).expect("verified i32 tuple index"),
            )
            .map_err(|_| BuildError::TupleIndexOutOfRange {
                operation: instruction.operation.name(),
                index: i64::try_from(*index).expect("verified i32 tuple index"),
                tuple_size: elements.len(),
            })?;
            verify_single_result_type(instruction, &expected)?;
        }
        StableHloOp::Compare => {
            let lhs = instruction.operands[0];
            let rhs = instruction.operands[1];
            let Some(lhs_type) = available.get(&lhs) else {
                return Err(BuildError::UnknownValue(lhs));
            };
            let Some(rhs_type) = available.get(&rhs) else {
                return Err(BuildError::UnknownValue(rhs));
            };
            let invalid_comparison = || BuildError::MissingOrInvalidAttribute {
                operation: instruction.operation.name(),
                attribute: "compare_type",
            };
            let comparison = instruction
                .operation
                .attributes()
                .and_then(|attributes| attributes.get("compare_type"))
                .map(|attribute| match attribute {
                    Attribute::Enum {
                        dialect,
                        name,
                        value,
                    } if dialect == "stablehlo" && name == "comparison_type" => {
                        crate::ComparisonType::from_name(value).ok_or_else(invalid_comparison)
                    }
                    _ => Err(invalid_comparison()),
                })
                .transpose()?;
            let expected =
                crate::infer_compare_type(lhs_type, rhs_type, comparison).map_err(|_| {
                    BuildError::IncompatibleOperands {
                        operation: instruction.operation.name(),
                        lhs,
                        rhs,
                    }
                })?;
            let result = &instruction.results[0];
            if !compatible_tensor_types(Some(&expected), Some(result.ty()))
                || !compatible_shapes(rhs_type, result.ty())
            {
                return Err(BuildError::InvalidResultType {
                    operation: instruction.operation.name(),
                    result: result.value(),
                });
            }
        }
        StableHloOp::Abs | StableHloOp::Imag | StableHloOp::IsFinite | StableHloOp::Real => {
            verify_unary_special_type(instruction, opcode, available)?;
        }
        _ => {}
    }
    if opcode == StableHloOp::OptimizationBarrier {
        if instruction.operands.len() != instruction.results.len() {
            return Err(BuildError::MismatchedOperandResultCount {
                operation: instruction.operation.name(),
                operands: instruction.operands.len(),
                results: instruction.results.len(),
            });
        }
        let inputs = instruction
            .operands
            .iter()
            .map(|id| available[id].clone())
            .collect::<Vec<_>>();
        let inferred = crate::infer_optimization_barrier_types(&inputs).map_err(|error| {
            BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            }
        })?;
        for (expected, result) in inferred.iter().zip(&instruction.results) {
            if expected != result.ty() {
                return Err(BuildError::InvalidResultType {
                    operation: instruction.operation.name(),
                    result: result.value(),
                });
            }
        }
    }
    let is_unary_same_type = matches!(
        opcode,
        StableHloOp::CountLeadingZeros
            | StableHloOp::Negate
            | StableHloOp::Not
            | StableHloOp::PopulationCount
            | StableHloOp::RoundNearestAfz
            | StableHloOp::RoundNearestEven
            | StableHloOp::Sign
            | StableHloOp::Cbrt
            | StableHloOp::Ceil
            | StableHloOp::Cosine
            | StableHloOp::Exponential
            | StableHloOp::ExponentialMinusOne
            | StableHloOp::Floor
            | StableHloOp::Log
            | StableHloOp::LogPlusOne
            | StableHloOp::Logistic
            | StableHloOp::Rsqrt
            | StableHloOp::Sine
            | StableHloOp::Sqrt
            | StableHloOp::Tan
            | StableHloOp::Tanh
    );
    if is_unary_same_type
        && let Some(operand) = instruction.operands.first()
        && let Some(result) = instruction.results.first()
        && !compatible_tensor_types(available.get(operand), Some(result.ty()))
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    let is_binary_same_type = matches!(
        opcode,
        StableHloOp::Atan2
            | StableHloOp::Divide
            | StableHloOp::Maximum
            | StableHloOp::Minimum
            | StableHloOp::Multiply
            | StableHloOp::Power
            | StableHloOp::Remainder
            | StableHloOp::ShiftLeft
            | StableHloOp::ShiftRightArithmetic
            | StableHloOp::ShiftRightLogical
            | StableHloOp::Subtract
    );
    if is_binary_same_type
        && instruction.operands.len() == 2
        && !compatible_tensor_types(
            available.get(&instruction.operands[0]),
            available.get(&instruction.operands[1]),
        )
    {
        return Err(BuildError::IncompatibleOperands {
            operation: instruction.operation.name(),
            lhs: instruction.operands[0],
            rhs: instruction.operands[1],
        });
    }
    if is_binary_same_type
        && let Some(result) = instruction.results.first()
        && instruction
            .operands
            .iter()
            .any(|operand| !compatible_tensor_types(available.get(operand), Some(result.ty())))
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    schema::verify(instruction, opcode, available)?;
    if let Some(attributes) = instruction.operation.attributes() {
        attributes::verify(opcode, attributes)?;
    }
    tensor_ops::verify(instruction, opcode, available)?;
    gather::verify(instruction, opcode, available)?;
    convolution::verify(instruction, opcode, available)?;
    custom_call::verify(instruction, opcode, available)?;
    collectives::verify(instruction, opcode, available)?;
    scatter::verify(instruction, opcode, available)?;
    regions::verify(instruction, opcode, available)
}

fn verify_iota(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let Some(Attribute::Integer {
        value: dimension, ..
    }) = instruction
        .operation
        .attributes()
        .and_then(|attributes| attributes.get("iota_dimension"))
    else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "iota_dimension",
        });
    };
    let Some(dimension) = usize::try_from(*dimension).ok() else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "iota_dimension",
        });
    };
    let Some(result) = instruction.results.first() else {
        return Err(BuildError::InvalidResultArity {
            operation: instruction.operation.name(),
            expected: 1,
            actual: 0,
        });
    };
    let Some(shape) = shaped_dimensions(result.ty()) else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    let allowed_element = match result.ty() {
        Type::Tensor(tensor) => {
            tensor.element() != ElementType::Index
                && tensor.element() != ElementType::Tf32
                && (opcode == StableHloOp::DynamicIota || tensor.element().is_numeric())
        }
        Type::QuantizedTensor(tensor) => tensor.quantization_dimension().is_none(),
        _ => false,
    };
    if !allowed_element {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    if dimension >= shape.len() {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    if opcode == StableHloOp::Iota {
        if shape.iter().any(Option::is_none) {
            return Err(BuildError::InvalidResultType {
                operation: instruction.operation.name(),
                result: result.value(),
            });
        }
    } else {
        let operand = instruction.operands[0];
        let Some(Type::Tensor(shape_tensor)) = available.get(&operand) else {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value: operand,
            });
        };
        if shape_tensor.rank() != 1
            || !shape_tensor.element().is_dimension_value()
            || shape_tensor.dimensions()[0] != Some(shape.len() as u64)
        {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value: operand,
            });
        }
    }
    Ok(())
}

fn verify_convert(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let Some(Type::Tensor(input)) = available.get(&operand) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let result = &instruction.results[0];
    let Type::Tensor(output) = result.ty() else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    if crate::inference::validate_convert_types(&Type::from(input.clone()), output.element())
        .is_err()
        || !compatible_shapes(&Type::from(input.clone()), &Type::from(output.clone()))
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_complex(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let lhs = instruction.operands[0];
    let rhs = instruction.operands[1];
    let expected = crate::infer_complex_type(&available[&lhs], &available[&rhs]).map_err(|_| {
        BuildError::IncompatibleOperands {
            operation: instruction.operation.name(),
            lhs,
            rhs,
        }
    })?;
    let result = &instruction.results[0];
    if !compatible_tensor_types(Some(&expected), Some(result.ty()))
        || !compatible_shapes(&available[&rhs], result.ty())
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_dot_general(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let lhs = instruction.operands[0];
    let rhs = instruction.operands[1];
    let (Some(lhs_type), Some(rhs_type)) = (available.get(&lhs), available.get(&rhs)) else {
        return Err(BuildError::UnknownValue(lhs));
    };
    if !matches!(lhs_type, Type::Tensor(_))
        && !matches!(
            lhs_type,
            Type::QuantizedTensor(quantized) if quantized.quantization_dimension().is_none()
        )
    {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: lhs,
        });
    }
    if !matches!(rhs_type, Type::Tensor(_) | Type::QuantizedTensor(_)) {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: rhs,
        });
    }
    let (Some(lhs_shape), Some(rhs_shape)) =
        (shaped_dimensions(lhs_type), shaped_dimensions(rhs_type))
    else {
        return Err(BuildError::IncompatibleOperands {
            operation: instruction.operation.name(),
            lhs,
            rhs,
        });
    };
    let Some(attributes) = instruction.operation.attributes() else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "dot_dimension_numbers",
        });
    };
    let Some(Attribute::StableHlo(attribute)) = attributes.get("dot_dimension_numbers") else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "dot_dimension_numbers",
        });
    };
    let crate::StableHloAttribute::DotDimensionNumbers(dimensions) = attribute.as_ref() else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "dot_dimension_numbers",
        });
    };
    if dimensions.lhs_batching_dimensions.len() != dimensions.rhs_batching_dimensions.len()
        || dimensions.lhs_contracting_dimensions.len()
            != dimensions.rhs_contracting_dimensions.len()
    {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let lhs_batch = validate_dimensions(&dimensions.lhs_batching_dimensions, lhs_shape.len());
    let rhs_batch = validate_dimensions(&dimensions.rhs_batching_dimensions, rhs_shape.len());
    let lhs_contract = validate_dimensions(&dimensions.lhs_contracting_dimensions, lhs_shape.len());
    let rhs_contract = validate_dimensions(&dimensions.rhs_contracting_dimensions, rhs_shape.len());
    let (Some(lhs_batch), Some(rhs_batch), Some(lhs_contract), Some(rhs_contract)) =
        (lhs_batch, rhs_batch, lhs_contract, rhs_contract)
    else {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    };
    if !disjoint_dimensions(&lhs_batch, &lhs_contract)
        || !disjoint_dimensions(&rhs_batch, &rhs_contract)
    {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let precision_config = match attributes.get("precision_config") {
        None => true,
        Some(Attribute::Array(precision)) => precision.iter().all(
            |precision| matches!(precision, Attribute::Enum { value, .. } if value == "DEFAULT"),
        ),
        Some(_) => {
            return Err(BuildError::MissingOrInvalidAttribute {
                operation: instruction.operation.name(),
                attribute: "precision_config",
            });
        }
    };
    let has_algorithm = match attributes.get("algorithm") {
        None => false,
        Some(Attribute::StableHlo(attribute)) => {
            if !matches!(
                attribute.as_ref(),
                crate::StableHloAttribute::DotAlgorithm(_)
            ) {
                return Err(BuildError::MissingOrInvalidAttribute {
                    operation: instruction.operation.name(),
                    attribute: "algorithm",
                });
            }
            true
        }
        Some(_) => {
            return Err(BuildError::MissingOrInvalidAttribute {
                operation: instruction.operation.name(),
                attribute: "algorithm",
            });
        }
    };
    if has_algorithm && !precision_config {
        return Err(BuildError::InvalidDotAlgorithmPrecision {
            operation: instruction.operation.name(),
        });
    }
    let Some(result) = instruction.results.first() else {
        return Err(BuildError::InvalidResultArity {
            operation: instruction.operation.name(),
            expected: 1,
            actual: 0,
        });
    };
    if !matches!(result.ty(), Type::Tensor(_) | Type::QuantizedTensor(_)) {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    let expected = crate::infer_dot_general_type(lhs_type, rhs_type, dimensions, result.ty())
        .map_err(|error| match error {
            crate::InferenceError::DimensionMismatch { .. } => BuildError::IncompatibleOperands {
                operation: instruction.operation.name(),
                lhs,
                rhs,
            },
            error => BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            },
        })?;
    if crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err() {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    verify_dot_general_quantization(instruction, lhs_type, rhs_type, result.ty(), &rhs_contract)?;
    Ok(())
}

fn verify_dot_general_quantization(
    instruction: &Instruction,
    lhs: &Type,
    rhs: &Type,
    result: &Type,
    rhs_contracting_dimensions: &[usize],
) -> Result<(), BuildError> {
    let operation = instruction.operation.name();
    if ![lhs, rhs, result]
        .iter()
        .any(|ty| matches!(ty, Type::QuantizedTensor(_)))
    {
        return Ok(());
    }
    if !convolution_dot_quantization_compatible(lhs, rhs, result) {
        return Err(BuildError::InvalidDotQuantization { operation });
    }
    let Type::QuantizedTensor(rhs) = rhs else {
        return Err(BuildError::InvalidDotQuantization { operation });
    };
    if rhs.zero_points().iter().any(|zero_point| *zero_point != 0) {
        return Err(BuildError::InvalidDotQuantization { operation });
    }
    if rhs
        .quantization_dimension()
        .is_some_and(|dimension| rhs_contracting_dimensions.contains(&dimension))
    {
        return Err(BuildError::InvalidDotQuantization { operation });
    }
    Ok(())
}

fn convolution_dot_quantization_compatible(lhs: &Type, rhs: &Type, result: &Type) -> bool {
    match (lhs, rhs, result) {
        (Type::QuantizedTensor(lhs), Type::QuantizedTensor(rhs), Type::QuantizedTensor(result)) => {
            lhs.storage_type() == rhs.storage_type()
                && lhs.expressed_type() == rhs.expressed_type()
                && lhs.expressed_type() == result.expressed_type()
                && (rhs.quantization_dimension().is_some()
                    || result.quantization_dimension().is_none())
        }
        (Type::Tensor(lhs), Type::QuantizedTensor(rhs), Type::Tensor(result)) => {
            lhs.element() == rhs.expressed_type() && lhs.element() == result.element()
        }
        _ => false,
    }
}

fn verify_reshape(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = &available[&operand];
    let result = &instruction.results[0];
    let (Some(operand_shape), Some(result_shape)) = (
        shaped_dimensions(operand_type),
        shaped_dimensions(result.ty()),
    ) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    if !shape_elements_compatible(operand_type, result.ty()) {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    if !reshape_quantization_compatible(operand_type, result.ty()) {
        return Err(BuildError::InvalidQuantizedTransform {
            operation: instruction.operation.name(),
        });
    }
    if !static_element_counts_match(operand_shape, result_shape) {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_dynamic_reshape(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let output_shape = instruction.operands[1];
    let operand_type = &available[&operand];
    let Some(Type::Tensor(output_shape_type)) = available.get(&output_shape) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: output_shape,
        });
    };
    let result = &instruction.results[0];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let Some(result_shape) = shaped_dimensions(result.ty()) else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    let rank = u64::try_from(result_shape.len()).ok();
    if output_shape_type.rank() != 1
        || output_shape_type.dimensions()[0] != rank
        || !output_shape_type.element().is_dimension_value()
    {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: output_shape,
        });
    }
    if !shape_elements_compatible(operand_type, result.ty())
        || !static_element_counts_match(operand_shape, result_shape)
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    if !reshape_quantization_compatible(operand_type, result.ty()) {
        return Err(BuildError::InvalidQuantizedTransform {
            operation: instruction.operation.name(),
        });
    }
    Ok(())
}

fn verify_transpose(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = &available[&operand];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let permutation_values = integer_attribute(instruction, "permutation")?;
    let Some(permutation) = validate_dimensions(&permutation_values, operand_shape.len()) else {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    };
    if permutation.len() != operand_shape.len() {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let expected =
        crate::infer_transpose_type(operand_type, &permutation_values).map_err(|error| {
            BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            }
        })?;
    let result = &instruction.results[0];
    if !shape_elements_compatible(operand_type, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    if !transpose_quantization_compatible(operand_type, result.ty(), &permutation) {
        return Err(BuildError::InvalidQuantizedTransform {
            operation: instruction.operation.name(),
        });
    }
    Ok(())
}

fn verify_broadcast_in_dim(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = &available[&operand];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let dimensions = integer_attribute(instruction, "broadcast_dimensions")?;
    let result = &instruction.results[0];
    let Some(result_shape) = shaped_dimensions(result.ty()) else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    let Some(dimensions) = validate_dimensions(&dimensions, result_shape.len()) else {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    };
    if dimensions.len() != operand_shape.len()
        || !shape_elements_compatible(operand_type, result.ty())
    {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    if !broadcast_quantization_compatible(operand_type, result.ty(), &dimensions) {
        return Err(BuildError::InvalidQuantizedTransform {
            operation: instruction.operation.name(),
        });
    }
    for (operand_dimension, &result_dimension) in dimensions.iter().enumerate() {
        if let Some(operand_size) = operand_shape[operand_dimension]
            && operand_size != 1
            && result_shape[result_dimension] != Some(operand_size)
        {
            return Err(BuildError::InvalidResultType {
                operation: instruction.operation.name(),
                result: result.value(),
            });
        }
    }
    Ok(())
}

fn verify_dynamic_broadcast_in_dim(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let output_dimensions = instruction.operands[1];
    let operand_type = &available[&operand];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let Some(Type::Tensor(output_shape_type)) = available.get(&output_dimensions) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: output_dimensions,
        });
    };
    if output_shape_type.rank() != 1
        || output_shape_type.dimensions()[0].is_none()
        || !output_shape_type.element().is_dimension_value()
    {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: output_dimensions,
        });
    }
    let broadcast_dimensions = integer_attribute(instruction, "broadcast_dimensions")?;
    let result = &instruction.results[0];
    let Some(result_shape) = shaped_dimensions(result.ty()) else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    let Some(broadcast_dimensions) = validate_dimensions(&broadcast_dimensions, result_shape.len())
    else {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    };
    if broadcast_dimensions.len() != operand_shape.len()
        || result_shape.len() < operand_shape.len()
        || usize::try_from(output_shape_type.dimensions()[0].unwrap()).ok()
            != Some(result_shape.len())
        || !shape_elements_compatible(operand_type, result.ty())
    {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    for (operand_dimension, &result_dimension) in broadcast_dimensions.iter().enumerate() {
        if let Some(operand_size) = operand_shape[operand_dimension]
            && operand_size != 1
            && matches!(
                (Some(operand_size), result_shape[result_dimension]),
                (Some(lhs), Some(rhs)) if lhs != rhs
            )
        {
            return Err(BuildError::InvalidResultType {
                operation: instruction.operation.name(),
                result: result.value(),
            });
        }
    }
    let expanding = optional_integer_attribute(instruction, "known_expanding_dimensions")?;
    let nonexpanding = optional_integer_attribute(instruction, "known_nonexpanding_dimensions")?;
    let mut hints = HashSet::new();
    let mut hint_count = 0;
    for hint_list in [expanding, nonexpanding].into_iter().flatten() {
        hint_count += hint_list.len();
        for dimension in hint_list {
            let Ok(dimension) = usize::try_from(dimension) else {
                return Err(BuildError::InvalidDimensionNumbers {
                    operation: instruction.operation.name(),
                });
            };
            if dimension >= operand_shape.len() || !hints.insert(dimension) {
                return Err(BuildError::InvalidDimensionNumbers {
                    operation: instruction.operation.name(),
                });
            }
        }
    }
    if hints.len() != hint_count {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    if !broadcast_quantization_compatible(operand_type, result.ty(), &broadcast_dimensions) {
        return Err(BuildError::InvalidQuantizedTransform {
            operation: instruction.operation.name(),
        });
    }
    Ok(())
}

fn verify_reverse(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = &available[&operand];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let dimensions = integer_attribute(instruction, "dimensions")?;
    if validate_dimensions(&dimensions, operand_shape.len()).is_none() {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let result = &instruction.results[0];
    let expected = crate::infer_reverse_type(operand_type, &dimensions).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation.name(),
            error,
        }
    })?;
    if !tensor_element_types_match(operand_type, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_slice(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = &available[&operand];
    if shaped_dimensions(operand_type).is_none() {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    }
    let starts = integer_attribute(instruction, "start_indices")?;
    let limits = integer_attribute(instruction, "limit_indices")?;
    let strides = integer_attribute(instruction, "strides")?;
    let inferred =
        crate::infer_slice_type(operand_type, &starts, &limits, &strides).map_err(|error| {
            BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            }
        })?;
    let result = &instruction.results[0];
    if !tensor_element_types_match(&inferred, result.ty())
        || crate::infer_most_specific_type(&[inferred, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_clamp(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let min = instruction.operands[0];
    let operand = instruction.operands[1];
    let max = instruction.operands[2];
    let operand_type = &available[&operand];
    let inferred = crate::infer_clamp_type(&available[&min], operand_type, &available[&max])
        .map_err(|_| BuildError::IncompatibleOperands {
            operation: instruction.operation.name(),
            lhs: min,
            rhs: max,
        })?;
    let result = &instruction.results[0];
    if !tensor_elements_compatible(operand_type, result.ty())
        || crate::infer_most_specific_type(&[inferred, result.ty().clone()]).is_err()
        || crate::infer_clamp_type(&available[&min], result.ty(), &available[&max]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_select(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let predicate = instruction.operands[0];
    let on_true = instruction.operands[1];
    let on_false = instruction.operands[2];
    let predicate_type = &available[&predicate];
    let true_type = &available[&on_true];
    let false_type = &available[&on_false];
    let inferred =
        crate::infer_select_type(predicate_type, true_type, false_type).map_err(|_| {
            BuildError::IncompatibleOperands {
                operation: instruction.operation.name(),
                lhs: on_true,
                rhs: on_false,
            }
        })?;
    let result = &instruction.results[0];
    if !tensor_elements_compatible(true_type, result.ty())
        || crate::infer_most_specific_type(&[inferred, result.ty().clone()]).is_err()
        || crate::infer_select_type(predicate_type, result.ty(), false_type).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_concatenate(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let Some(Attribute::Integer {
        value: dimension, ..
    }) = instruction
        .operation
        .attributes()
        .and_then(|attributes| attributes.get("dimension"))
    else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "dimension",
        });
    };
    let Some(dimension) = usize::try_from(*dimension).ok() else {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    };
    let types = instruction
        .operands
        .iter()
        .map(|operand| available[operand].clone())
        .collect::<Vec<_>>();
    let expected = crate::infer_concatenate_type(&types, dimension).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation.name(),
            error,
        }
    })?;
    let result = &instruction.results[0];
    if !tensor_element_types_match(&expected, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_pad(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let padding_value = instruction.operands[1];
    let operand_type = &available[&operand];
    let padding_type = &available[&padding_value];
    let edge_low = integer_attribute(instruction, "edge_padding_low")?;
    let edge_high = integer_attribute(instruction, "edge_padding_high")?;
    let interior = integer_attribute(instruction, "interior_padding")?;
    let expected =
        crate::infer_pad_type(operand_type, padding_type, &edge_low, &edge_high, &interior)
            .map_err(|error| BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            })?;
    let result = &instruction.results[0];
    if !tensor_element_types_match(&expected, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_dynamic_pad(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let padding_value = instruction.operands[1];
    let operand_type = &available[&operand];
    let padding_type = &available[&padding_value];
    let result = &instruction.results[0];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let Some(padding_shape) = shaped_dimensions(padding_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: padding_value,
        });
    };
    let Some(result_shape) = shaped_dimensions(result.ty()) else {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    };
    if !matches!(operand_type, Type::Tensor(_))
        || !padding_shape.is_empty()
        || !tensor_element_types_match(operand_type, padding_type)
        || !tensor_element_types_match(operand_type, result.ty())
        || operand_shape.len() != result_shape.len()
    {
        return Err(BuildError::IncompatibleOperands {
            operation: instruction.operation.name(),
            lhs: operand,
            rhs: padding_value,
        });
    }

    let mut padding_vector_type: Option<&Type> = None;
    for &padding_operand in &instruction.operands[2..5] {
        let padding_vector = &available[&padding_operand];
        let Type::Tensor(vector_type) = padding_vector else {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value: padding_operand,
            });
        };
        let Ok(rank) = u64::try_from(operand_shape.len()) else {
            return Err(BuildError::InvalidDimensionNumbers {
                operation: instruction.operation.name(),
            });
        };
        if vector_type.rank() != 1
            || vector_type.dimensions()[0] != Some(rank)
            || !vector_type.element().is_dimension_value()
            || padding_vector_type.is_some_and(|expected| expected != padding_vector)
        {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value: padding_operand,
            });
        }
        padding_vector_type = Some(padding_vector);
    }
    Ok(())
}

pub(crate) fn is_scalar_integer(ty: &Type) -> bool {
    schema::is_scalar_integer(ty)
}

fn verify_start_indices(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
    indices: &[ValueId],
) -> Result<Vec<Type>, BuildError> {
    let mut types = Vec::with_capacity(indices.len());
    for &value in indices {
        let ty = &available[&value];
        if !is_scalar_integer(ty) {
            return Err(BuildError::InvalidOperandType {
                operation: instruction.operation.name(),
                value,
            });
        }
        if types
            .first()
            .is_some_and(|first| !tensor_element_types_match(first, ty))
        {
            return Err(BuildError::IncompatibleOperands {
                operation: instruction.operation.name(),
                lhs: indices[0],
                rhs: value,
            });
        }
        types.push(ty.clone());
    }
    Ok(types)
}

fn verify_dynamic_slice(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = &available[&operand];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let sizes = integer_attribute(instruction, "slice_sizes")?;
    let start_indices = &instruction.operands[1..];
    if sizes.len() != operand_shape.len() || start_indices.len() != operand_shape.len() {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let index_types = verify_start_indices(instruction, available, start_indices)?;
    let expected =
        crate::infer_dynamic_slice_type(operand_type, &index_types, &sizes).map_err(|error| {
            BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            }
        })?;
    let result = &instruction.results[0];
    if !tensor_element_types_match(operand_type, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_dynamic_update_slice(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let update = instruction.operands[1];
    let operand_type = &available[&operand];
    let update_type = &available[&update];
    let (Some(operand_shape), Some(update_shape)) = (
        shaped_dimensions(operand_type),
        shaped_dimensions(update_type),
    ) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let start_indices = &instruction.operands[2..];
    if operand_shape.len() != update_shape.len() || start_indices.len() != operand_shape.len() {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let result = &instruction.results[0];
    let index_types = verify_start_indices(instruction, available, start_indices)?;
    let expected = crate::infer_dynamic_update_slice_type(operand_type, update_type, &index_types)
        .map_err(|error| match error {
            crate::InferenceError::IncompatibleTypes { index: 1 }
            | crate::InferenceError::SliceSizeExceeded { .. } => BuildError::IncompatibleOperands {
                operation: instruction.operation.name(),
                lhs: operand,
                rhs: update,
            },
            error => BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            },
        })?;
    if !tensor_element_types_match(operand_type, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn verify_get_dimension_size(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let Some(shape) = available.get(&operand).and_then(shaped_dimensions) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    let dimension = dimension_attribute(instruction)?;
    if dimension >= shape.len() {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let expected =
        crate::infer_get_dimension_size_type(&available[&operand], dimension).map_err(|error| {
            BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            }
        })?;
    verify_single_result_type(instruction, &expected)
}

fn verify_set_dimension_size(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let size = instruction.operands[1];
    let operand_type = &available[&operand];
    let Some(operand_shape) = shaped_dimensions(operand_type) else {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: operand,
        });
    };
    if !matches!(
        available.get(&size),
        Some(Type::Tensor(tensor))
            if tensor.rank() == 0 && tensor.element() == ElementType::I32
    ) {
        return Err(BuildError::InvalidOperandType {
            operation: instruction.operation.name(),
            value: size,
        });
    }
    let dimension = dimension_attribute(instruction)?;
    if dimension >= operand_shape.len() {
        return Err(BuildError::InvalidDimensionNumbers {
            operation: instruction.operation.name(),
        });
    }
    let expected =
        crate::infer_set_dimension_size_type(operand_type, &available[&size], dimension, None)
            .map_err(|error| BuildError::TypeInference {
                operation: instruction.operation.name(),
                error,
            })?;
    let result = &instruction.results[0];
    if !tensor_element_types_match(operand_type, result.ty())
        || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn dimension_attribute(instruction: &Instruction) -> Result<usize, BuildError> {
    let Some(Attribute::Integer {
        value: dimension, ..
    }) = instruction
        .operation
        .attributes()
        .and_then(|attributes| attributes.get("dimension"))
    else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: "dimension",
        });
    };
    usize::try_from(*dimension).map_err(|_| BuildError::InvalidDimensionNumbers {
        operation: instruction.operation.name(),
    })
}

fn integer_attribute(
    instruction: &Instruction,
    name: &'static str,
) -> Result<Vec<i64>, BuildError> {
    let Some(values) = instruction
        .operation
        .attributes()
        .and_then(|attributes| attributes.get(name))
        .and_then(i64_array)
    else {
        return Err(BuildError::MissingOrInvalidAttribute {
            operation: instruction.operation.name(),
            attribute: name,
        });
    };
    Ok(values)
}

fn optional_integer_attribute(
    instruction: &Instruction,
    name: &'static str,
) -> Result<Option<Vec<i64>>, BuildError> {
    match instruction
        .operation
        .attributes()
        .and_then(|attributes| attributes.get(name))
    {
        None => Ok(None),
        Some(attribute) => {
            i64_array(attribute)
                .map(Some)
                .ok_or(BuildError::MissingOrInvalidAttribute {
                    operation: instruction.operation.name(),
                    attribute: name,
                })
        }
    }
}

fn i64_array(attribute: &Attribute) -> Option<Vec<i64>> {
    let Attribute::DenseArray {
        element: ElementType::I64,
        values,
    } = attribute
    else {
        return None;
    };
    values
        .iter()
        .map(|value| match value {
            crate::ScalarValue::Integer(value) => i64::try_from(*value).ok(),
            _ => None,
        })
        .collect()
}

fn shape_elements_compatible(lhs: &Type, rhs: &Type) -> bool {
    match (lhs, rhs) {
        (Type::Tensor(lhs), Type::Tensor(rhs)) => lhs.element() == rhs.element(),
        (Type::QuantizedTensor(lhs), Type::QuantizedTensor(rhs)) => {
            same_quantization_storage(lhs, rhs)
        }
        _ => false,
    }
}

fn reshape_quantization_compatible(lhs: &Type, rhs: &Type) -> bool {
    let (Type::QuantizedTensor(lhs_tensor), Type::QuantizedTensor(rhs_tensor)) = (lhs, rhs) else {
        return matches!((lhs, rhs), (Type::Tensor(_), Type::Tensor(_)));
    };
    if !same_quantization_storage(lhs_tensor, rhs_tensor)
        || !same_quantization_parameters(lhs_tensor, rhs_tensor)
    {
        return false;
    }
    match (
        lhs_tensor.quantization_dimension(),
        rhs_tensor.quantization_dimension(),
    ) {
        (None, None) => true,
        (Some(lhs_dimension), Some(rhs_dimension)) => {
            let lhs_shape = lhs_tensor.dimensions();
            let rhs_shape = rhs_tensor.dimensions();
            if crate::inference::infer_common_dimension(&[
                (lhs, lhs_dimension),
                (rhs, rhs_dimension),
            ])
            .is_err()
            {
                return false;
            }
            static_element_counts_match(&lhs_shape[..lhs_dimension], &rhs_shape[..rhs_dimension])
                && static_element_counts_match(
                    &lhs_shape[lhs_dimension + 1..],
                    &rhs_shape[rhs_dimension + 1..],
                )
        }
        _ => false,
    }
}

fn transpose_quantization_compatible(lhs: &Type, rhs: &Type, permutation: &[usize]) -> bool {
    let (Type::QuantizedTensor(lhs), Type::QuantizedTensor(rhs)) = (lhs, rhs) else {
        return matches!((lhs, rhs), (Type::Tensor(_), Type::Tensor(_)));
    };
    if !same_quantization_storage(lhs, rhs) || !same_quantization_parameters(lhs, rhs) {
        return false;
    }
    match (lhs.quantization_dimension(), rhs.quantization_dimension()) {
        (None, None) => true,
        (Some(lhs_dimension), Some(rhs_dimension)) => permutation
            .get(rhs_dimension)
            .is_some_and(|dimension| *dimension == lhs_dimension),
        _ => false,
    }
}

fn broadcast_quantization_compatible(lhs: &Type, rhs: &Type, dimensions: &[usize]) -> bool {
    let (Type::QuantizedTensor(lhs), Type::QuantizedTensor(rhs)) = (lhs, rhs) else {
        return matches!((lhs, rhs), (Type::Tensor(_), Type::Tensor(_)));
    };
    if !same_quantization_storage(lhs, rhs) {
        return false;
    }
    match (lhs.quantization_dimension(), rhs.quantization_dimension()) {
        (None, None) => same_quantization_parameters(lhs, rhs),
        (Some(lhs_dimension), Some(rhs_dimension)) => {
            if dimensions.get(lhs_dimension) != Some(&rhs_dimension) {
                return false;
            }
            if lhs.dimensions().get(lhs_dimension) == Some(&Some(1)) {
                let (Some(lhs_scale), Some(&lhs_zero_point)) =
                    (lhs.scales().next(), lhs.zero_points().first())
                else {
                    return false;
                };
                rhs.scales().enumerate().all(|(index, scale)| {
                    scale == lhs_scale && rhs.zero_points().get(index) == Some(&lhs_zero_point)
                })
            } else {
                same_quantization_parameters(lhs, rhs)
            }
        }
        _ => false,
    }
}

fn same_quantization_storage(
    lhs: &crate::QuantizedTensorType,
    rhs: &crate::QuantizedTensorType,
) -> bool {
    same_quantized_element_storage(lhs.element(), rhs.element())
}

fn same_quantized_element_storage(
    lhs: &crate::QuantizedElementType,
    rhs: &crate::QuantizedElementType,
) -> bool {
    lhs.storage_type() == rhs.storage_type()
        && effective_storage_range(lhs) == effective_storage_range(rhs)
        && lhs.expressed_type() == rhs.expressed_type()
}

fn effective_storage_range(tensor: &crate::QuantizedElementType) -> Option<(i128, i128)> {
    let (minimum, maximum) = tensor.storage_type().integer_bounds()?;
    let (storage_min, storage_max) = tensor.storage_range();
    Some((
        storage_min.unwrap_or(minimum),
        storage_max.unwrap_or(maximum),
    ))
}

fn same_quantization_parameters(
    lhs: &crate::QuantizedTensorType,
    rhs: &crate::QuantizedTensorType,
) -> bool {
    lhs.scales().eq(rhs.scales()) && lhs.zero_points() == rhs.zero_points()
}

fn static_element_counts_match(lhs: &[Option<u64>], rhs: &[Option<u64>]) -> bool {
    let lhs_dynamic = lhs.iter().any(Option::is_none);
    let rhs_dynamic = rhs.iter().any(Option::is_none);
    let lhs_zero = lhs.contains(&Some(0));
    let rhs_zero = rhs.contains(&Some(0));
    if lhs_zero || rhs_zero {
        return (lhs_zero || lhs_dynamic) && (rhs_zero || rhs_dynamic);
    }
    if lhs_dynamic && rhs_dynamic {
        return true; // Both sides can have zero elements.
    }
    let mut lhs = lhs
        .iter()
        .filter_map(|dimension| *dimension)
        .collect::<Vec<_>>();
    let mut rhs = rhs
        .iter()
        .filter_map(|dimension| *dimension)
        .collect::<Vec<_>>();
    for lhs_dimension in &mut lhs {
        for rhs_dimension in &mut rhs {
            let divisor = greatest_common_divisor(*lhs_dimension, *rhs_dimension);
            *lhs_dimension /= divisor;
            *rhs_dimension /= divisor;
        }
    }
    (rhs_dynamic || lhs.into_iter().all(|dimension| dimension == 1))
        && (lhs_dynamic || rhs.into_iter().all(|dimension| dimension == 1))
}

fn greatest_common_divisor(mut lhs: u64, mut rhs: u64) -> u64 {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

fn validate_dimensions(dimensions: &[i64], rank: usize) -> Option<Vec<usize>> {
    let mut validated = Vec::with_capacity(dimensions.len());
    for &dimension in dimensions {
        let dimension = usize::try_from(dimension).ok()?;
        if dimension >= rank || validated.contains(&dimension) {
            return None;
        }
        validated.push(dimension);
    }
    Some(validated)
}

fn disjoint_dimensions(lhs: &[usize], rhs: &[usize]) -> bool {
    lhs.iter().all(|dimension| !rhs.contains(dimension))
}

fn compatible_dimensions(lhs: &[Option<u64>], rhs: &[Option<u64>]) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs)
            .all(|(lhs, rhs)| !matches!((lhs, rhs), (Some(lhs), Some(rhs)) if lhs != rhs))
}

fn inferred_shape_matches(expected: &[Option<u64>], actual: &Type) -> bool {
    shaped_dimensions(actual).is_some_and(|shape| {
        compatible_dimensions(expected, shape)
            && expected.iter().enumerate().all(|(dimension, size)| {
                !matches!((*size, crate::inference::bound(actual, dimension)),
                    (Some(size), Some(bound)) if size > bound)
            })
    })
}

fn verify_unary_special_type(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let operand = instruction.operands[0];
    let operand_type = available
        .get(&operand)
        .ok_or(BuildError::UnknownValue(operand))?;
    let expected = match opcode {
        StableHloOp::Abs => crate::infer_abs_type(operand_type),
        StableHloOp::Imag => crate::infer_imag_type(operand_type),
        StableHloOp::Real => crate::infer_real_type(operand_type),
        StableHloOp::IsFinite => crate::infer_is_finite_type(operand_type),
        _ => unreachable!("special unary verification dispatch"),
    }
    .map_err(|_| BuildError::InvalidOperandType {
        operation: instruction.operation.name(),
        value: operand,
    })?;
    let Some(result) = instruction.results.first() else {
        return Err(BuildError::InvalidResultArity {
            operation: instruction.operation.name(),
            expected: 1,
            actual: 0,
        });
    };
    if !compatible_tensor_types(Some(&expected), Some(result.ty())) {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn compatible_tensor_types(lhs: Option<&Type>, rhs: Option<&Type>) -> bool {
    let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
        return false;
    };
    compatible_shapes(lhs, rhs) && tensor_elements_compatible(lhs, rhs)
}

pub(crate) fn compatible_value_types(lhs: &Type, rhs: &Type) -> bool {
    match (lhs, rhs) {
        (
            Type::Tensor(_) | Type::QuantizedTensor(_) | Type::UnrankedTensor(_),
            Type::Tensor(_) | Type::QuantizedTensor(_) | Type::UnrankedTensor(_),
        ) => compatible_tensor_types(Some(lhs), Some(rhs)),
        (Type::Tuple(lhs), Type::Tuple(rhs)) => {
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .zip(rhs)
                    .all(|(lhs, rhs)| compatible_value_types(lhs, rhs))
        }
        _ => lhs == rhs,
    }
}

fn compatible_shapes(lhs: &Type, rhs: &Type) -> bool {
    if matches!(lhs, Type::UnrankedTensor(_)) {
        return matches!(
            rhs,
            Type::Tensor(_) | Type::QuantizedTensor(_) | Type::UnrankedTensor(_)
        );
    }
    if matches!(rhs, Type::UnrankedTensor(_)) {
        return matches!(lhs, Type::Tensor(_) | Type::QuantizedTensor(_));
    }
    let (Some(lhs_shape), Some(rhs_shape)) = (shaped_dimensions(lhs), shaped_dimensions(rhs))
    else {
        return false;
    };
    lhs_shape.len() == rhs_shape.len()
        && lhs_shape
            .iter()
            .zip(rhs_shape)
            .enumerate()
            .all(|(dimension, (&left_size, &right_size))| {
                !matches!((left_size, right_size), (Some(lhs), Some(rhs)) if lhs != rhs)
                    && !matches!((left_size, crate::inference::bound(rhs, dimension)), (Some(size), Some(bound)) if size > bound)
                    && !matches!((right_size, crate::inference::bound(lhs, dimension)), (Some(size), Some(bound)) if size > bound)
            })
}

pub(crate) fn tensor_elements_compatible(lhs: &Type, rhs: &Type) -> bool {
    if let (Some(lhs), Some(rhs)) = (lhs.plain_tensor_element(), rhs.plain_tensor_element()) {
        return lhs == rhs;
    }
    match (
        lhs.quantized_tensor_element(),
        rhs.quantized_tensor_element(),
    ) {
        (Some(lhs), Some(rhs)) => {
            same_quantized_element_storage(lhs, rhs)
                && lhs.quantization_dimension().is_some() == rhs.quantization_dimension().is_some()
        }
        _ => false,
    }
}

fn shaped_dimensions(ty: &Type) -> Option<&[Option<u64>]> {
    match ty {
        Type::Tensor(tensor) => Some(tensor.dimensions()),
        Type::QuantizedTensor(tensor) => Some(tensor.dimensions()),
        _ => None,
    }
}

fn verify_single_result_type(instruction: &Instruction, expected: &Type) -> Result<(), BuildError> {
    let Some(result) = instruction.results.first() else {
        return Err(BuildError::InvalidResultArity {
            operation: instruction.operation.name(),
            expected: 1,
            actual: 0,
        });
    };
    if result.ty() != expected {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation.name(),
            result: result.value(),
        });
    }
    Ok(())
}

/// Errors from building and verifying the IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    InvalidFunctionType(TypeError),
    TypeInference {
        operation: &'static str,
        error: crate::InferenceError,
    },
    InvalidFunctionName,
    InvalidFunctionAttribute(&'static str),
    InvalidMeshSymbol(String),
    InvalidAsyncBody,
    InvalidConstantOperand {
        operation: &'static str,
        value: ValueId,
        reason: &'static str,
    },
    UnknownComputation(ComputationId),
    DuplicateComputationName(String),
    UnknownComputationName {
        operation: &'static str,
        name: String,
    },
    InvalidSymbolReference {
        operation: &'static str,
    },
    InvalidComputationSignature {
        operation: &'static str,
        name: String,
    },
    InvalidDimensionNumbers {
        operation: &'static str,
    },
    InvalidDotAlgorithmPrecision {
        operation: &'static str,
    },
    InvalidDotQuantization {
        operation: &'static str,
    },
    InvalidQuantizedTransform {
        operation: &'static str,
    },
    UnknownValue(ValueId),
    TooManyValues,
    InvalidParameterNumbers,
    InvalidParameterResultCount,
    ParameterInsideRegion,
    EmptyRegion,
    DuplicateValue(ValueId),
    UnknownOrNonDominatingValue {
        instruction: ValueId,
        operand: ValueId,
    },
    InvalidArity {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidResultArity {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidRegionCount {
        operation: &'static str,
        expected: Option<usize>,
        actual: usize,
    },
    InvalidRegionBlockCount {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidRegionSignature {
        operation: &'static str,
        region: usize,
        expected_arguments: usize,
        actual_arguments: usize,
        expected_results: usize,
        actual_results: usize,
    },
    MissingRegionTerminator,
    UnexpectedRegionTerminator,
    ReturnOutsideRegion,
    MissingOrInvalidAttribute {
        operation: &'static str,
        attribute: &'static str,
    },
    TupleIndexOutOfRange {
        operation: &'static str,
        index: i64,
        tuple_size: usize,
    },
    MismatchedOperandResultCount {
        operation: &'static str,
        operands: usize,
        results: usize,
    },
    InvalidVariadicCardinality {
        operation: &'static str,
        operands: usize,
        results: usize,
    },
    InvalidResultId {
        expected: ValueId,
        actual: ValueId,
    },
    NonDominatingOperand {
        instruction: ValueId,
        operand: ValueId,
    },
    InvalidOperandType {
        operation: &'static str,
        value: ValueId,
    },
    IncompatibleOperands {
        operation: &'static str,
        lhs: ValueId,
        rhs: ValueId,
    },
    InvalidResultType {
        operation: &'static str,
        result: ValueId,
    },
    InvalidType(TypeError),
    InvalidAttribute {
        name: String,
        error: AttributeValidationError,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFunctionType(error) => write!(f, "invalid function signature: {error}"),
            Self::TypeInference { operation, error } => {
                write!(f, "{operation} type inference failed: {error}")
            }
            Self::InvalidFunctionName => write!(f, "function symbol names must not be empty"),
            Self::InvalidFunctionAttribute(name) => write!(
                f,
                "invalid or conflicting function/module attribute {name:?}"
            ),
            Self::InvalidMeshSymbol(name) => {
                write!(f, "mesh symbol {name:?} must define a mesh attribute")
            }
            Self::InvalidAsyncBody => write!(
                f,
                "async_start requires exactly one collective or slice operation followed by a return"
            ),
            Self::InvalidConstantOperand {
                operation,
                value,
                reason,
            } => {
                write!(
                    f,
                    "{operation} constant operand {} is invalid: {reason}",
                    value.0
                )
            }
            Self::UnknownComputation(id) => write!(f, "unknown computation {}", id.0),
            Self::DuplicateComputationName(name) => {
                write!(f, "duplicate computation name {name:?}")
            }
            Self::UnknownComputationName { operation, name } => {
                write!(f, "{operation} references unknown computation {name:?}")
            }
            Self::InvalidSymbolReference { operation } => {
                write!(f, "{operation} requires a single function symbol reference")
            }
            Self::InvalidComputationSignature { operation, name } => {
                write!(
                    f,
                    "{operation} signature does not match computation {name:?}"
                )
            }
            Self::InvalidDimensionNumbers { operation } => {
                write!(
                    f,
                    "{operation} has invalid or inconsistent dimension numbers"
                )
            }
            Self::InvalidDotAlgorithmPrecision { operation } => write!(
                f,
                "{operation} algorithm requires default precision configuration"
            ),
            Self::InvalidDotQuantization { operation } => {
                write!(f, "{operation} has incompatible quantized operand types")
            }
            Self::InvalidQuantizedTransform { operation } => {
                write!(f, "{operation} has incompatible quantization parameters")
            }
            Self::UnknownValue(id) => write!(f, "unknown value {}", id.0),
            Self::TooManyValues => write!(f, "computation has too many SSA values"),
            Self::InvalidParameterNumbers => {
                write!(
                    f,
                    "parameter numbers must be unique and contiguous from zero"
                )
            }
            Self::InvalidParameterResultCount => {
                write!(f, "each parameter must define exactly one value")
            }
            Self::ParameterInsideRegion => {
                write!(f, "function parameters cannot be defined inside a region")
            }
            Self::EmptyRegion => write!(f, "operation region has no blocks"),
            Self::DuplicateValue(value) => {
                write!(f, "SSA value {} is defined more than once", value.0)
            }
            Self::UnknownOrNonDominatingValue {
                instruction,
                operand,
            } => write!(
                f,
                "operand {} is not visible at instruction {}",
                operand.0, instruction.0
            ),
            Self::InvalidArity {
                operation,
                expected,
                actual,
            } => write!(f, "{operation} expects {expected} operands, got {actual}"),
            Self::InvalidResultArity {
                operation,
                expected,
                actual,
            } => write!(f, "{operation} expects {expected} results, got {actual}"),
            Self::InvalidRegionCount {
                operation,
                expected: Some(expected),
                actual,
            } => write!(f, "{operation} expects {expected} regions, got {actual}"),
            Self::InvalidRegionCount {
                operation,
                expected: None,
                actual,
            } => write!(f, "{operation} requires more regions, got {actual}"),
            Self::InvalidRegionBlockCount {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation} expects {expected} blocks in each region, got {actual}"
            ),
            Self::InvalidRegionSignature {
                operation,
                region,
                expected_arguments,
                actual_arguments,
                expected_results,
                actual_results,
            } => write!(
                f,
                "{operation} region {region} expects {expected_arguments} arguments and {expected_results} returned values, got {actual_arguments} arguments and {actual_results} returned values"
            ),
            Self::MissingRegionTerminator => {
                write!(
                    f,
                    "region is missing its required stablehlo.return terminator"
                )
            }
            Self::UnexpectedRegionTerminator => {
                write!(
                    f,
                    "stablehlo.return must be the final operation in a region block"
                )
            }
            Self::ReturnOutsideRegion => {
                write!(
                    f,
                    "stablehlo.return is only valid inside an operation region"
                )
            }
            Self::MissingOrInvalidAttribute {
                operation,
                attribute,
            } => write!(f, "{operation} requires a valid {attribute} attribute"),
            Self::TupleIndexOutOfRange {
                operation,
                index,
                tuple_size,
            } => write!(
                f,
                "{operation} tuple index {index} is out of range for tuple size {tuple_size}"
            ),
            Self::MismatchedOperandResultCount {
                operation,
                operands,
                results,
            } => write!(
                f,
                "{operation} requires matching operand and result counts, got {operands} operands and {results} results"
            ),
            Self::InvalidVariadicCardinality {
                operation,
                operands,
                results,
            } => write!(
                f,
                "{operation} has incompatible variadic operand/result counts: {operands} operands, {results} results"
            ),
            Self::InvalidResultId { expected, actual } => write!(
                f,
                "instruction result id {} is not the expected {}",
                actual.0, expected.0
            ),
            Self::NonDominatingOperand {
                instruction,
                operand,
            } => write!(
                f,
                "operand {} is not defined before instruction {}",
                operand.0, instruction.0
            ),
            Self::InvalidOperandType { operation, value } => {
                write!(f, "{operation} does not accept operand {}", value.0)
            }
            Self::IncompatibleOperands {
                operation,
                lhs,
                rhs,
            } => write!(
                f,
                "{operation} operands {} and {} must have the same numeric tensor type",
                lhs.0, rhs.0
            ),
            Self::InvalidResultType { operation, result } => {
                write!(
                    f,
                    "{operation} has an invalid result type at value {}",
                    result.0
                )
            }
            Self::InvalidType(error) => write!(f, "invalid instruction type: {error}"),
            Self::InvalidAttribute { name, error } => {
                write!(f, "invalid attribute {name:?}: {error}")
            }
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidFunctionType(error) => Some(error),
            Self::TypeInference { error, .. } => Some(error),
            _ => None,
        }
    }
}
