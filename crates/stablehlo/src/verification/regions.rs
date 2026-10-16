//! Region signatures and shape relationships for higher-order tensor operations.

use std::collections::HashMap;

use super::{
    BuildError, compatible_dimensions, inferred_shape_matches, integer_attribute,
    optional_integer_attribute, resolve_region_value_type, shaped_dimensions,
    tensor_element_types_match,
};
use crate::{Attribute, Block, ElementType, Instruction, StableHloOp, Type, ValueId};

fn invalid_dimensions(instruction: &Instruction) -> BuildError {
    BuildError::InvalidDimensionNumbers {
        operation: instruction.operation().name(),
    }
}

fn invalid_type(instruction: &Instruction, value: ValueId) -> BuildError {
    BuildError::InvalidOperandType {
        operation: instruction.operation().name(),
        value,
    }
}

pub(super) fn check_result(
    instruction: &Instruction,
    index: usize,
    expected_shape: &[Option<u64>],
    expected_element: &Type,
) -> Result<(), BuildError> {
    let result = &instruction.results()[index];
    if !inferred_shape_matches(expected_shape, result.ty())
        || !tensor_element_types_match(result.ty(), expected_element)
    {
        return Err(BuildError::InvalidResultType {
            operation: instruction.operation().name(),
            result: result.value(),
        });
    }
    Ok(())
}

fn check_scalar(instruction: &Instruction, value: ValueId, ty: &Type) -> Result<(), BuildError> {
    if shaped_dimensions(ty) != Some(&[]) {
        return Err(invalid_type(instruction, value));
    }
    Ok(())
}

fn returned_type<'a>(
    block: &'a Block,
    available: &'a HashMap<ValueId, Type>,
    index: usize,
) -> (ValueId, &'a Type) {
    let value = block
        .instructions()
        .last()
        .expect("verified terminator")
        .operands()[index];
    let ty = resolve_region_value_type(block, available, value).expect("verified region SSA");
    (value, ty)
}

pub(super) fn common_shape(
    instruction: &Instruction,
    inputs: &[ValueId],
    available: &HashMap<ValueId, Type>,
) -> Result<Vec<Option<u64>>, BuildError> {
    let mut common = shaped_dimensions(&available[&inputs[0]])
        .expect("verified tensor operand")
        .to_vec();
    for &value in &inputs[1..] {
        let shape = shaped_dimensions(&available[&value]).expect("verified tensor operand");
        if !compatible_dimensions(&common, shape) {
            return Err(invalid_type(instruction, value));
        }
        for (common, &size) in common.iter_mut().zip(shape) {
            if common.is_none() {
                *common = size;
            }
        }
        for &value in inputs {
            if !inferred_shape_matches(&common, &available[&value]) {
                return Err(invalid_type(instruction, value));
            }
        }
    }
    Ok(common)
}

pub(super) fn common_result_shape(
    instruction: &Instruction,
    mut expected: Vec<Option<u64>>,
) -> Result<Vec<Option<u64>>, BuildError> {
    for result in instruction.results() {
        let actual = shaped_dimensions(result.ty()).expect("verified tensor result");
        if !compatible_dimensions(&expected, actual) {
            return Err(BuildError::InvalidResultType {
                operation: instruction.operation().name(),
                result: result.value(),
            });
        }
        for (expected, &size) in expected.iter_mut().zip(actual) {
            if expected.is_none() {
                *expected = size;
            }
        }
        for result in instruction.results() {
            if !inferred_shape_matches(&expected, result.ty()) {
                return Err(BuildError::InvalidResultType {
                    operation: instruction.operation().name(),
                    result: result.value(),
                });
            }
        }
    }
    Ok(expected)
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    match opcode {
        StableHloOp::Map => verify_map(instruction, available),
        StableHloOp::Sort => verify_sort(instruction, available),
        StableHloOp::Reduce | StableHloOp::ReduceWindow => {
            verify_reduce(instruction, opcode, available)
        }
        StableHloOp::AllReduce | StableHloOp::CollectiveReduce => {
            verify_all_reduce(instruction, available)
        }
        StableHloOp::SelectAndScatter => verify_select_and_scatter(instruction, available),
        StableHloOp::ReduceScatter => {
            let input = &available[&instruction.operands()[0]];
            let accumulator = verify_accumulator(instruction, available, input, 0, 1)?;
            let result = instruction.results()[0].ty();
            if !tensor_element_types_match(accumulator, result) {
                return Err(BuildError::InvalidResultType {
                    operation: opcode.mnemonic(),
                    result: instruction.results()[0].value(),
                });
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn verify_map(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let block = &instruction.regions()[0].blocks()[0];
    let output = instruction.results()[0].ty();
    let dimensions = integer_attribute(instruction, "dimensions")?;
    let inputs = instruction
        .operands()
        .iter()
        .map(|id| available[id].clone())
        .collect::<Vec<_>>();
    let (value, returned) = returned_type(block, available, 0);
    let inferred = crate::infer_map_type(&inputs, returned, &dimensions).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    let witnesses = inputs
        .iter()
        .chain(std::iter::once(output))
        .collect::<Vec<_>>();
    crate::inference::infer_common_shape(&witnesses).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    for (&operand, argument) in instruction.operands().iter().zip(block.arguments()) {
        let input = &available[&operand];
        check_scalar(instruction, argument.value(), argument.ty())?;
        if !tensor_element_types_match(input, argument.ty()) {
            return Err(invalid_type(instruction, argument.value()));
        }
    }
    check_scalar(instruction, value, returned)?;
    if !tensor_element_types_match(&inferred, output) {
        return Err(invalid_type(instruction, value));
    }
    Ok(())
}

fn verify_sort(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    if instruction.operands().is_empty() {
        return Err(invalid_dimensions(instruction));
    }
    let block = &instruction.regions()[0].blocks()[0];
    let shape = common_result_shape(
        instruction,
        common_shape(instruction, instruction.operands(), available)?,
    )?;
    let dimension = match instruction
        .operation()
        .attributes()
        .and_then(|a| a.get("dimension"))
    {
        Some(Attribute::Integer {
            value: dimension, ..
        }) => *dimension,
        None => -1,
        _ => unreachable!("verified dimension attribute"),
    };
    let dimension = i64::try_from(dimension).map_err(|_| invalid_dimensions(instruction))?;
    let witnesses = instruction
        .operands()
        .iter()
        .map(|id| &available[id])
        .chain(instruction.results().iter().map(|result| result.ty()))
        .collect::<Vec<_>>();
    crate::inference::validate_sort_types(&witnesses, dimension).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    for (index, &operand) in instruction.operands()[..instruction.results().len()]
        .iter()
        .enumerate()
    {
        let input = &available[&operand];
        check_result(instruction, index, &shape, input)?;
        for argument in &block.arguments()[2 * index..2 * index + 2] {
            check_scalar(instruction, argument.value(), argument.ty())?;
            if !tensor_element_types_match(input, argument.ty()) {
                return Err(invalid_type(instruction, argument.value()));
            }
        }
    }
    let (value, returned) = returned_type(block, available, 0);
    check_scalar(instruction, value, returned)?;
    if !matches!(returned, Type::Tensor(tensor) if tensor.element() == ElementType::I1) {
        return Err(invalid_type(instruction, value));
    }
    Ok(())
}

fn promotable(input: &Type, accumulator: &Type) -> bool {
    let (source, target) = match (input, accumulator) {
        (Type::Tensor(source), Type::Tensor(target)) => {
            let (source, target) = (source.element(), target.element());
            if !((source.is_integer() && target.is_integer())
                || (source.is_float() && target.is_float())
                || (matches!(source, ElementType::C64 | ElementType::C128)
                    && matches!(target, ElementType::C64 | ElementType::C128)))
            {
                return false;
            }
            (source, target)
        }
        (Type::QuantizedTensor(source), Type::QuantizedTensor(target))
            if source.expressed_type() == target.expressed_type() =>
        {
            (source.storage_type(), target.storage_type())
        }
        _ => return false,
    };
    matches!((source.bit_width(), target.bit_width()), (Some(source), Some(target)) if source <= target)
}

pub(super) fn verify_accumulator<'a>(
    instruction: &'a Instruction,
    available: &'a HashMap<ValueId, Type>,
    input: &Type,
    index: usize,
    count: usize,
) -> Result<&'a Type, BuildError> {
    verify_accumulator_in_region(instruction, available, input, index, count, 0)
}

fn verify_accumulator_in_region<'a>(
    instruction: &'a Instruction,
    available: &'a HashMap<ValueId, Type>,
    input: &Type,
    index: usize,
    count: usize,
    region: usize,
) -> Result<&'a Type, BuildError> {
    let block = &instruction.regions()[region].blocks()[0];
    let (value, returned) = returned_type(block, available, index);
    check_scalar(instruction, value, returned)?;
    for argument_index in [index, count + index] {
        let argument = &block.arguments()[argument_index];
        check_scalar(instruction, argument.value(), argument.ty())?;
        if !tensor_element_types_match(argument.ty(), returned) {
            return Err(invalid_type(instruction, argument.value()));
        }
    }
    if !promotable(input, returned) {
        return Err(invalid_type(instruction, value));
    }
    Ok(returned)
}

fn verify_reduce(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let count = instruction.results().len();
    if opcode == StableHloOp::Reduce {
        return verify_plain_reduce(instruction, available, count);
    }
    let mut inputs = Vec::with_capacity(count);
    let mut accumulators = Vec::with_capacity(count);
    for index in 0..count {
        let operand = instruction.operands()[index];
        let input = &available[&operand];
        let init_id = instruction.operands()[count + index];
        let init = &available[&init_id];
        check_scalar(instruction, init_id, init)?;
        if !tensor_element_types_match(input, init) {
            return Err(invalid_type(instruction, init_id));
        }
        let accumulator = verify_accumulator(instruction, available, input, index, count)?;
        inputs.push(input.clone());
        accumulators.push(accumulator.clone());
    }
    let dimensions = integer_attribute(instruction, "window_dimensions")?;
    let strides = optional_integer_attribute(instruction, "window_strides")?;
    let base = optional_integer_attribute(instruction, "base_dilations")?;
    let dilation = optional_integer_attribute(instruction, "window_dilations")?;
    let rank = shaped_dimensions(&inputs[0])
        .expect("verified ranked input")
        .len();
    let padding = super::window::padding(instruction, rank)?
        .into_iter()
        .enumerate()
        .map(|(dimension, pair)| {
            let pair = pair.expect("static reduction padding");
            let error = || BuildError::TypeInference {
                operation: instruction.operation().name(),
                error: crate::InferenceError::InvalidWindowPadding { dimension },
            };
            Ok([
                i64::try_from(pair[0]).map_err(|_| error())?,
                i64::try_from(pair[1]).map_err(|_| error())?,
            ])
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    let expected = crate::infer_reduce_window_types(
        &inputs,
        &accumulators,
        &dimensions,
        strides.as_deref(),
        base.as_deref(),
        dilation.as_deref(),
        Some(&padding),
    )
    .map_err(|error| BuildError::TypeInference {
        operation: instruction.operation().name(),
        error,
    })?;
    let witnesses = expected
        .iter()
        .chain(instruction.results().iter().map(|result| result.ty()))
        .collect::<Vec<_>>();
    let shape = crate::inference::infer_common_shape(&witnesses).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    for (index, accumulator) in accumulators.iter().enumerate() {
        check_result(instruction, index, &shape, accumulator)?;
    }
    Ok(())
}

fn verify_plain_reduce(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
    count: usize,
) -> Result<(), BuildError> {
    let witness_shape = common_shape(instruction, &instruction.operands()[..count], available)?;
    let mut inputs = Vec::with_capacity(count);
    let mut accumulators = Vec::with_capacity(count);
    for index in 0..count {
        let input = &available[&instruction.operands()[index]];
        let init_id = instruction.operands()[count + index];
        let init = &available[&init_id];
        check_scalar(instruction, init_id, init)?;
        if !tensor_element_types_match(input, init) {
            return Err(invalid_type(instruction, init_id));
        }
        inputs.push(input.clone());
        accumulators.push(verify_accumulator(instruction, available, input, index, count)?.clone());
    }
    let dimensions = integer_attribute(instruction, "dimensions")?;
    let expected = inputs
        .iter()
        .zip(&accumulators)
        .map(|(input, accumulator)| {
            crate::infer_reduce_types(
                std::slice::from_ref(input),
                &dimensions,
                std::slice::from_ref(accumulator),
            )
            .map(|mut results| results.remove(0))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        })?;
    let refined_shape = common_result_shape(
        instruction,
        witness_shape
            .iter()
            .enumerate()
            .filter_map(|(axis, &size)| (!dimensions.contains(&(axis as i64))).then_some(size))
            .collect(),
    )?;
    for (result, expected) in instruction.results().iter().zip(expected) {
        if !tensor_element_types_match(result.ty(), &expected)
            || !inferred_shape_matches(&refined_shape, result.ty())
            || !inferred_shape_matches(&refined_shape, &expected)
            || crate::infer_most_specific_type(&[expected, result.ty().clone()]).is_err()
        {
            return Err(BuildError::InvalidResultType {
                operation: instruction.operation().name(),
                result: result.value(),
            });
        }
    }
    Ok(())
}

fn verify_all_reduce(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let inputs = instruction.operands()[..instruction.results().len()]
        .iter()
        .map(|operand| available[operand].clone())
        .collect::<Vec<_>>();
    let block = &instruction.regions()[0].blocks()[0];
    let (_, accumulator) = returned_type(block, available, 0);
    let inferred = crate::infer_all_reduce_types(&inputs, accumulator).map_err(|error| {
        BuildError::TypeInference {
            operation: instruction.operation().name(),
            error,
        }
    })?;
    for (index, (input, expected)) in inputs.iter().zip(inferred).enumerate() {
        verify_accumulator(instruction, available, input, 0, 1)?;
        check_result(
            instruction,
            index,
            shaped_dimensions(&expected).expect("inferred tensor"),
            &expected,
        )?;
        let result = &instruction.results()[index];
        crate::infer_most_specific_type(&[expected, result.ty().clone()]).map_err(|error| {
            BuildError::TypeInference {
                operation: instruction.operation().name(),
                error,
            }
        })?;
    }
    Ok(())
}

fn verify_select_and_scatter(
    instruction: &Instruction,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let input = &available[&instruction.operands()[0]];
    let source = &available[&instruction.operands()[1]];
    let init_id = instruction.operands()[2];
    let init = &available[&init_id];
    let input_shape = shaped_dimensions(input).expect("verified tensor operand");
    if !tensor_element_types_match(input, source) || !tensor_element_types_match(input, init) {
        return Err(invalid_type(instruction, init_id));
    }
    check_scalar(instruction, init_id, init)?;
    let select = &instruction.regions()[0].blocks()[0];
    for argument in select.arguments() {
        check_scalar(instruction, argument.value(), argument.ty())?;
        if !tensor_element_types_match(input, argument.ty()) {
            return Err(invalid_type(instruction, argument.value()));
        }
    }
    let (value, selected) = returned_type(select, available, 0);
    if !matches!(selected, Type::Tensor(tensor) if tensor.rank() == 0 && tensor.element() == ElementType::I1)
    {
        return Err(invalid_type(instruction, value));
    }
    let accumulator = verify_accumulator_in_region(instruction, available, input, 0, 1, 1)?;
    let dimensions =
        optional_integer_attribute(instruction, "window_dimensions")?.unwrap_or_default();
    let strides = optional_integer_attribute(instruction, "window_strides")?;
    let padding = super::window::padding(instruction, input_shape.len())?
        .into_iter()
        .enumerate()
        .map(|(dimension, pair)| {
            let pair = pair.expect("known select-and-scatter padding");
            let error = || BuildError::TypeInference {
                operation: instruction.operation().name(),
                error: crate::InferenceError::InvalidWindowPadding { dimension },
            };
            Ok([
                i64::try_from(pair[0]).map_err(|_| error())?,
                i64::try_from(pair[1]).map_err(|_| error())?,
            ])
        })
        .collect::<Result<Vec<_>, BuildError>>()?;
    let error = |error| BuildError::TypeInference {
        operation: instruction.operation().name(),
        error,
    };
    let expected = crate::infer_select_and_scatter_type(
        input,
        source,
        accumulator,
        &dimensions,
        strides.as_deref(),
        Some(&padding),
    )
    .map_err(error)?;
    let result = instruction.results()[0].ty();
    crate::inference::validate_select_and_scatter_shapes(
        &[input, result],
        source,
        &dimensions,
        strides.as_deref(),
        Some(&padding),
    )
    .map_err(error)?;
    check_result(
        instruction,
        0,
        shaped_dimensions(&expected).expect("inferred tensor"),
        &expected,
    )?;
    Ok(())
}
