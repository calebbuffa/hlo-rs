use std::collections::HashSet;

use super::*;
use crate::{IndexTransform, OperandView};

const INLINE_BROADCAST_CONSTANT_ELEMENTS: usize = 8;

pub(super) fn lower_module(
    context: &Context,
    module: Ptr<Operation>,
    plan: &mut ExecutionPlanBuilder,
    operation_strategies: &OperationStrategyRegistry,
) -> Result<(), PlanError> {
    let operations = entry_operations(context, module)?;
    let used_values = collect_used_values(context, &operations);
    let mut selections = Vec::with_capacity(operations.len());
    let mut has_operation_strategy = false;
    for &operation in &operations {
        if Operation::get_op::<ParameterOp>(operation, context).is_some()
            || Operation::get_op::<ReturnOp>(operation, context).is_some()
            || Operation::get_op::<TupleOp>(operation, context).is_some()
            || Operation::get_op::<ConstantOp>(operation, context).is_some()
        {
            selections.push(None);
            continue;
        }
        let selected = operation_strategies
            .select(context, operation)
            .map_err(PlanError::OperationStrategy)?;
        has_operation_strategy |= selected.is_some();
        selections.push(selected);
    }
    if !has_operation_strategy
        && can_directly_lower_scalar_constant_arithmetic(context, &operations, &used_values)
    {
        return direct::lower_direct_module(context, module, plan);
    }
    if has_operation_strategy || requires_compositional_lowering(context, &operations) {
        return lower_compositional_module(
            context,
            &operations,
            &selections,
            &used_values,
            operation_strategies,
            plan,
        );
    }
    direct::lower_direct_module(context, module, plan)
}

fn collect_used_values(context: &Context, operations: &[Ptr<Operation>]) -> HashSet<Value> {
    let mut definitions = HashMap::new();
    let mut pending = operations.to_vec();
    while let Some(operation) = pending.pop() {
        for result in operation.deref(context).results() {
            definitions.insert(result, operation);
        }
        if Operation::get_op::<FusionOp>(operation, context).is_some() {
            for region in operation.deref(context).regions() {
                for block in region.deref(context).iter(context) {
                    pending.extend(block.deref(context).iter(context));
                }
            }
        }
    }
    let mut used = HashSet::new();
    let mut values = Vec::new();
    for &operation in operations {
        if Operation::get_op::<ReturnOp>(operation, context).is_some() {
            values.extend(operation.deref(context).operands());
        }
    }
    while let Some(value) = values.pop() {
        if !used.insert(value) {
            continue;
        }
        if let Some(&definition) = definitions.get(&value) {
            values.extend(definition.deref(context).operands());
        }
    }
    used
}

fn requires_compositional_lowering(context: &Context, operations: &[Ptr<Operation>]) -> bool {
    let mut pending = operations.to_vec();
    while let Some(operation) = pending.pop() {
        if Operation::get_op::<ConstantOp>(operation, context).is_some()
            && Shape::rank(
                context,
                operation.deref(context).get_result(0).get_type(context),
            )
            .is_ok_and(|rank| rank > 0)
        {
            return true;
        }
        if Operation::get_op::<BroadcastOp>(operation, context).is_some()
            && operation.deref(context).operands().any(|operand| {
                Shape::rank(context, operand.get_type(context)).is_ok_and(|rank| rank > 0)
            })
        {
            return true;
        }
        let operation_ref = Operation::get_op_dyn(operation, context);
        if let Some(hlo) = pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref) {
            let opcode = hlo.opcode();
            if matches!(
                opcode,
                Opcode::Gather
                    | Opcode::Reshape
                    | Opcode::Transpose
                    | Opcode::Reverse
                    | Opcode::Slice
            ) || matches!(opcode, Opcode::Minimum | Opcode::Maximum | Opcode::Clamp)
                || matches!(
                    opcode,
                    Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
                ) && Shape::element_type(
                    context,
                    operation.deref(context).get_result(0).get_type(context),
                )
                .is_ok_and(|element| {
                    element == ElementType::U1
                        || element.is_float()
                            && !matches!(element, ElementType::F32 | ElementType::F64)
                })
            {
                return true;
            }
        }
        if Operation::get_op::<FusionOp>(operation, context).is_some() {
            for region in operation.deref(context).regions() {
                for block in region.deref(context).iter(context) {
                    pending.extend(block.deref(context).iter(context));
                }
            }
        }
    }
    false
}

fn can_directly_lower_scalar_constant_arithmetic(
    context: &Context,
    operations: &[Ptr<Operation>],
    used_values: &HashSet<Value>,
) -> bool {
    if !low_precision_arithmetic_uses_scalar_constant(context, operations, used_values) {
        return false;
    }
    operations.iter().all(|&operation| {
        if Operation::get_op::<ParameterOp>(operation, context).is_some()
            || Operation::get_op::<ReturnOp>(operation, context).is_some()
            || Operation::get_op::<TupleOp>(operation, context).is_some()
        {
            return true;
        }
        if let Some(constant) = Operation::get_op::<ConstantOp>(operation, context) {
            let result_type = operation.deref(context).get_result(0).get_type(context);
            return Shape::element_type(context, result_type)
                .is_ok_and(|element| matches!(element, ElementType::F16 | ElementType::BF16))
                && constant.literal(context).is_some_and(|literal| {
                    Shape::rank(context, literal.shape).is_ok_and(|rank| rank == 0)
                });
        }
        if let Some(broadcast) = Operation::get_op::<BroadcastOp>(operation, context) {
            let operands: Vec<_> = operation.deref(context).operands().collect();
            let result_type = operation.deref(context).get_result(0).get_type(context);
            return operands.len() == 1
                && Shape::rank(context, operands[0].get_type(context)).is_ok_and(|rank| rank == 0)
                && Shape::element_type(context, result_type)
                    .is_ok_and(|element| matches!(element, ElementType::F16 | ElementType::BF16))
                && broadcast
                    .dimensions(context)
                    .is_some_and(|dimensions| dimensions.0.is_empty());
        }
        let operation_ref = Operation::get_op_dyn(operation, context);
        pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).is_some_and(|hlo| {
            matches!(
                hlo.opcode(),
                Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
            ) && Shape::element_type(
                context,
                operation.deref(context).get_result(0).get_type(context),
            )
            .is_ok_and(|element| matches!(element, ElementType::F16 | ElementType::BF16))
        })
    })
}

fn low_precision_arithmetic_uses_scalar_constant(
    context: &Context,
    operations: &[Ptr<Operation>],
    used_values: &HashSet<Value>,
) -> bool {
    // The direct converter materializes literals and tracks positive zero in-kernel;
    // compositional kernels currently accept only buffer operands.
    let mut scalar_constants = HashSet::new();
    for &operation in operations {
        if let Some(constant) = Operation::get_op::<ConstantOp>(operation, context) {
            let value = operation.deref(context).get_result(0);
            if used_values.contains(&value)
                && constant.literal(context).is_some_and(|literal| {
                    Shape::rank(context, literal.shape).is_ok_and(|rank| rank == 0)
                })
            {
                scalar_constants.insert(value);
            }
            continue;
        }
        if Operation::get_op::<BroadcastOp>(operation, context).is_some() {
            let operands: Vec<_> = operation.deref(context).operands().collect();
            if operands
                .first()
                .is_some_and(|operand| scalar_constants.contains(operand))
            {
                scalar_constants.insert(operation.deref(context).get_result(0));
            }
            continue;
        }
        let operation_ref = Operation::get_op_dyn(operation, context);
        let Some(hlo) = pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref) else {
            continue;
        };
        if matches!(
            hlo.opcode(),
            Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
        ) && Shape::element_type(
            context,
            operation.deref(context).get_result(0).get_type(context),
        )
        .is_ok_and(|element| matches!(element, ElementType::F16 | ElementType::BF16))
            && operation
                .deref(context)
                .operands()
                .any(|operand| scalar_constants.contains(&operand))
        {
            return true;
        }
    }
    false
}

fn entry_operations(
    context: &Context,
    module: Ptr<Operation>,
) -> Result<Vec<Ptr<Operation>>, PlanError> {
    let entry = xlahlo::ir::entry(context, module)
        .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
    let computation = find_computation(context, module, &entry)?;
    let block = computation
        .body(context)
        .deref(context)
        .get_entry_block()
        .ok_or_else(|| PlanError::LoweringFailure("entry block is missing".into()))?;
    Ok(block.deref(context).iter(context).collect())
}

#[derive(Clone)]
struct PlannedValue {
    buffer: BufferId,
    descriptor: BufferDescriptor,
    role: BufferRole,
    transforms: Vec<IndexTransform>,
}

#[derive(Clone)]
struct ConstantValue {
    descriptor: BufferDescriptor,
    bits: Vec<u128>,
    bytes: Vec<u8>,
}

fn lower_compositional_module(
    context: &Context,
    operations: &[Ptr<Operation>],
    selections: &[Option<usize>],
    used_values: &HashSet<Value>,
    operation_strategies: &OperationStrategyRegistry,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    if operations.len() != selections.len() {
        return Err(PlanError::LoweringFailure(
            "operation strategy selection count mismatch".into(),
        ));
    }
    let mut returned_values = None;
    let mut parameter_descriptors = Vec::new();
    for &operation in operations {
        if let Some(parameter) = Operation::get_op::<ParameterOp>(operation, context) {
            let index = parameter
                .index(context)
                .ok_or_else(|| PlanError::LoweringFailure("parameter index is missing".into()))?
                .0 as usize;
            let descriptor = descriptor(
                context,
                operation.deref(context).get_result(0).get_type(context),
            )?;
            if index >= parameter_descriptors.len() {
                parameter_descriptors.resize_with(index + 1, || None);
            }
            if parameter_descriptors[index].replace(descriptor).is_some() {
                return Err(PlanError::LoweringFailure(
                    "duplicate entry parameter index".into(),
                ));
            }
        } else if Operation::is_op::<ReturnOp>(operation, context) {
            if returned_values.is_some() {
                return Err(PlanError::LoweringFailure(
                    "entry computation has multiple returns".into(),
                ));
            }
            let operands: Vec<_> = operation.deref(context).operands().collect();
            let [value] = operands.as_slice() else {
                return Err(PlanError::LoweringFailure(
                    "entry return must have one HLO result value".into(),
                ));
            };
            let values = if Shape::is_tuple(context, value.get_type(context)) {
                let tuple = value
                    .defining_op()
                    .filter(|definition| Operation::is_op::<TupleOp>(*definition, context))
                    .ok_or_else(|| {
                        PlanError::UnsupportedFeature(
                            "entry tuple result must be constructed by xlahlo.tuple".into(),
                        )
                    })?;
                tuple.deref(context).operands().collect()
            } else {
                vec![*value]
            };
            returned_values = Some(values);
        }
    }
    let returned_values = returned_values
        .ok_or_else(|| PlanError::UnsupportedFeature("entry computation has no return".into()))?;
    if parameter_descriptors.iter().any(Option::is_none) {
        return Err(PlanError::UnsupportedFeature(
            "entry parameters must be a contiguous tensor list".into(),
        ));
    }
    let parameter_descriptors: Vec<_> = parameter_descriptors.into_iter().flatten().collect();
    let mut output_indices = HashMap::new();
    for (index, value) in returned_values.iter().copied().enumerate() {
        output_indices.entry(value).or_insert(index);
    }
    let mut values = HashMap::<Value, PlannedValue>::new();
    let mut inline_constants = HashMap::<Value, ConstantValue>::new();
    let mut constants = HashMap::<Value, ConstantValue>::new();
    let logical_views = operations.iter().any(|&operation| {
        let operation_ref = Operation::get_op_dyn(operation, context);
        pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).is_some_and(|hlo| {
            matches!(
                hlo.opcode(),
                Opcode::Reshape | Opcode::Transpose | Opcode::Reverse | Opcode::Slice
            )
        })
    });
    for &operation in operations {
        let Some(parameter) = Operation::get_op::<ParameterOp>(operation, context) else {
            continue;
        };
        let index = parameter
            .index(context)
            .ok_or_else(|| PlanError::LoweringFailure("parameter index is missing".into()))?
            .0 as usize;
        let descriptor = parameter_descriptors
            .get(index)
            .ok_or_else(|| PlanError::LoweringFailure("parameter index is out of range".into()))?
            .clone();
        let buffer = plan.add_buffer(BufferRole::Input(index), descriptor.clone());
        values.insert(
            operation.deref(context).get_result(0),
            PlannedValue {
                buffer,
                descriptor,
                role: BufferRole::Input(index),
                transforms: Vec::new(),
            },
        );
    }

    for (operation_index, (operation, selection)) in operations
        .iter()
        .copied()
        .zip(selections.iter().copied())
        .enumerate()
    {
        if Operation::get_op::<ParameterOp>(operation, context).is_some()
            || Operation::get_op::<ReturnOp>(operation, context).is_some()
            || Operation::get_op::<TupleOp>(operation, context).is_some()
            || Operation::get_op::<ConstantOp>(operation, context).is_some()
                && !used_values.contains(&operation.deref(context).get_result(0))
        {
            continue;
        }
        if Operation::get_op::<ConstantOp>(operation, context).is_some() {
            let result_value = operation.deref(context).get_result(0);
            let result_descriptor = descriptor(context, result_value.get_type(context))?;
            let constant = constant_value(context, operation, &result_descriptor)?;
            constants.insert(result_value, constant.clone());
            if only_used_by_broadcasts(context, operations, selections, result_value)
                && constant.bits.len() <= INLINE_BROADCAST_CONSTANT_ELEMENTS
                && !logical_views
            {
                inline_constants.insert(result_value, constant);
                continue;
            }
            if output_indices.contains_key(&result_value)
                && only_used_by_returns(context, operations, result_value)
            {
                continue;
            }
            let output_index = output_indices.get(&result_value).copied();
            let role = output_index
                .map(BufferRole::Output)
                .unwrap_or(BufferRole::Temporary);
            let result_buffer = plan.add_buffer(role, result_descriptor.clone());
            lower_constant_operation(
                result_buffer,
                &result_descriptor,
                &constant.bits,
                operation_index,
                plan,
            )?;
            values.insert(
                result_value,
                PlannedValue {
                    buffer: result_buffer,
                    descriptor: result_descriptor,
                    role,
                    transforms: Vec::new(),
                },
            );
            continue;
        }
        if logical_views && selection.is_none() {
            let operation_ref = Operation::get_op_dyn(operation, context);
            let opcode =
                pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).map(|hlo| hlo.opcode());
            if opcode.is_some_and(|opcode| {
                matches!(
                    opcode,
                    Opcode::Reshape
                        | Opcode::Transpose
                        | Opcode::Reverse
                        | Opcode::Slice
                        | Opcode::Broadcast
                )
            }) {
                let operands: Vec<_> = operation.deref(context).operands().collect();
                let [operand] = operands.as_slice() else {
                    return Err(PlanError::LoweringFailure(
                        "view requires one tensor operand".into(),
                    ));
                };
                let source = values.get(operand).ok_or_else(|| {
                    PlanError::UnsupportedFeature(
                        "view operand is not a planned tensor value".into(),
                    )
                })?;
                let result_value = operation.deref(context).get_result(0);
                let result_descriptor =
                    logical_descriptor(context, result_value.get_type(context))?;
                let transform =
                    views::transform(context, operation, &source.descriptor, &result_descriptor)?;
                let mut transforms = source.transforms.clone();
                transforms.push(transform);
                values.insert(
                    result_value,
                    PlannedValue {
                        buffer: source.buffer,
                        descriptor: result_descriptor,
                        role: source.role,
                        transforms,
                    },
                );
                continue;
            }
        }
        if Operation::get_op::<BroadcastOp>(operation, context).is_some() && selection.is_none() {
            let operands: Vec<_> = operation.deref(context).operands().collect();
            let [operand] = operands.as_slice() else {
                return Err(PlanError::LoweringFailure(
                    "broadcast instruction arity is not one".into(),
                ));
            };
            let result_value = operation.deref(context).get_result(0);
            let result_descriptor = descriptor(context, result_value.get_type(context))?;
            let broadcast = Operation::get_op::<BroadcastOp>(operation, context)
                .ok_or_else(|| PlanError::LoweringFailure("expected broadcast operation".into()))?;
            let dimensions = broadcast
                .dimensions(context)
                .ok_or_else(|| {
                    PlanError::LoweringFailure("broadcast dimensions attribute is missing".into())
                })?
                .0
                .clone();
            let output_index = output_indices.get(&result_value).copied();
            let role = output_index
                .map(BufferRole::Output)
                .unwrap_or(BufferRole::Temporary);
            let result_buffer = plan.add_buffer(role, result_descriptor.clone());
            if let Some(constant) = inline_constants.get(operand) {
                let indexing =
                    broadcast_indexing(&constant.descriptor, &result_descriptor, &dimensions)?;
                lower_inline_constant_broadcast(
                    constant,
                    indexing,
                    result_buffer,
                    operation_index,
                    plan,
                )?;
            } else {
                let source = values.get(operand).cloned().ok_or_else(|| {
                    PlanError::UnsupportedFeature(
                        "broadcast operand is not a planned tensor value".into(),
                    )
                })?;
                let indexing =
                    broadcast_indexing(&source.descriptor, &result_descriptor, &dimensions)?;
                lower_broadcast_operation(
                    source.buffer,
                    result_buffer,
                    indexing,
                    operation_index,
                    plan,
                )?;
            }
            values.insert(
                result_value,
                PlannedValue {
                    buffer: result_buffer,
                    descriptor: result_descriptor,
                    role,
                    transforms: Vec::new(),
                },
            );
            continue;
        }
        let operands: Vec<_> = operation.deref(context).operands().collect();
        let planned_operands: Vec<_> = operands
            .iter()
            .map(|operand| {
                values.get(operand).cloned().ok_or_else(|| {
                    PlanError::UnsupportedFeature(
                        "operation operand is not a planned tensor value".into(),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let operand_ids: Vec<_> = planned_operands.iter().map(|value| value.buffer).collect();
        let operand_descriptors: Vec<_> = planned_operands
            .iter()
            .map(|value| value.descriptor.clone())
            .collect();
        let operand_views: Vec<_> = planned_operands
            .iter()
            .map(|value| OperandView {
                buffer: value.buffer,
                descriptor: value.descriptor.clone(),
                transforms: value.transforms.clone(),
            })
            .collect();
        let result_values: Vec<_> = operation.deref(context).results().collect();
        if result_values.len() != 1 {
            return Err(PlanError::UnsupportedFeature(
                "compositional operation lowering requires one tensor result".into(),
            ));
        }
        let result_value = result_values[0];
        let result_descriptor = if logical_views {
            logical_descriptor(context, result_value.get_type(context))?
        } else {
            descriptor(context, result_value.get_type(context))?
        };
        let output_index = output_indices.get(&result_value).copied();
        let role = output_index
            .map(BufferRole::Output)
            .unwrap_or(BufferRole::Temporary);
        let result_buffer = plan.add_buffer(role, result_descriptor.clone());
        let selected = selection
            .map(|index| operation_strategies.get(index))
            .cloned();
        if let Some(strategy) = selected {
            if !strategy.supports_indexed_operands()
                && operand_views.iter().any(|view| !view.transforms.is_empty())
            {
                return Err(PlanError::IndexedOperandsUnsupported {
                    strategy: strategy.name().into(),
                });
            }
            strategy.lower(OperationLoweringRequest {
                context,
                operation,
                operands: &operand_ids,
                operand_descriptors: &operand_descriptors,
                operand_views: &operand_views,
                results: &[result_buffer],
                result_descriptors: std::slice::from_ref(&result_descriptor),
                plan,
            })?;
        } else {
            lower_builtin_operation(
                context,
                operation,
                &operand_ids,
                &operand_descriptors,
                &operand_views,
                result_buffer,
                &result_descriptor,
                operation_index,
                plan,
            )?;
        }
        values.insert(
            result_value,
            PlannedValue {
                buffer: result_buffer,
                descriptor: result_descriptor,
                role,
                transforms: Vec::new(),
            },
        );
    }
    for (output_index, value) in returned_values.into_iter().enumerate() {
        let planned = values.get(&value).ok_or_else(|| {
            PlanError::UnsupportedFeature("returned value has no planned buffer".into())
        });
        let planned = match planned {
            Ok(planned) => planned,
            Err(_) => {
                let constant = constants.get(&value).ok_or_else(|| {
                    PlanError::UnsupportedFeature("returned value has no planned buffer".into())
                })?;
                plan.add_output(PlannedOutput::Static(StaticValue {
                    descriptor: constant.descriptor.clone(),
                    bytes: constant.bytes.clone(),
                }));
                continue;
            }
        };
        if !planned.transforms.is_empty() {
            let view = OperandView {
                buffer: planned.buffer,
                descriptor: planned.descriptor.clone(),
                transforms: planned.transforms.clone(),
            };
            let storage = BufferDescriptor::dense_row_major(
                planned.descriptor.element_type,
                planned.transforms[0].source_shape.clone(),
            );
            plan.add_output(PlannedOutput::View(view.output(&storage)?));
            continue;
        }
        match planned.role {
            BufferRole::Input(_) => plan.add_output(PlannedOutput::Alias(planned.buffer)),
            BufferRole::Output(index) if index == output_index => {
                plan.add_output(PlannedOutput::Buffer(planned.buffer))
            }
            BufferRole::Output(_) | BufferRole::Temporary => {
                return Err(PlanError::UnsupportedFeature(
                    "duplicate or non-materialized computed outputs are unsupported".into(),
                ));
            }
        }
    }
    Ok(())
}

fn lower_constant_operation(
    result: BufferId,
    result_descriptor: &BufferDescriptor,
    bits: &[u128],
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let element_type = result_descriptor.element_type;
    lower_constant_fill(
        element_type,
        bits.to_vec(),
        result,
        result_descriptor,
        operation_index,
        plan,
    )
}

fn constant_value(
    context: &Context,
    operation: Ptr<Operation>,
    result_descriptor: &BufferDescriptor,
) -> Result<ConstantValue, PlanError> {
    let constant = Operation::get_op::<ConstantOp>(operation, context)
        .ok_or_else(|| PlanError::LoweringFailure("expected constant operation".into()))?;
    let literal = constant.literal(context).ok_or_else(|| {
        PlanError::LoweringFailure("constant literal attribute is missing".into())
    })?;
    if descriptor(context, literal.shape)? != *result_descriptor {
        return Err(PlanError::LoweringFailure(
            "constant literal shape does not match its result".into(),
        ));
    }
    let bits = literal
        .bits(context)
        .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
    let expected_elements = result_descriptor.num_elements()?;
    if bits.len() != expected_elements {
        return Err(PlanError::LoweringFailure(
            "constant literal element count does not match its result".into(),
        ));
    }
    Ok(ConstantValue {
        descriptor: result_descriptor.clone(),
        bits,
        bytes: literal.storage.clone(),
    })
}

fn lower_constant_fill(
    element_type: ElementType,
    bits: Vec<u128>,
    result: BufferId,
    result_descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    plan.add_kernel(
        format!("xlahlo-constant-{operation_index}"),
        Arc::new(ConstantKernel {
            descriptor: result_descriptor.clone(),
            element_type,
            bits,
        }),
        vec![KernelBinding {
            buffer: result,
            access: BufferAccess::Write,
        }],
        LaunchGeometry::for_elements(result_descriptor.num_elements()?, [CUBE_DIM, 1, 1])?,
        false,
    )
}

fn only_used_by_broadcasts(
    context: &Context,
    operations: &[Ptr<Operation>],
    selections: &[Option<usize>],
    value: Value,
) -> bool {
    let mut found_use = false;
    for (&operation, selection) in operations.iter().zip(selections) {
        if !operation
            .deref(context)
            .operands()
            .any(|operand| operand == value)
        {
            continue;
        }
        if Operation::get_op::<ReturnOp>(operation, context).is_some() {
            continue;
        }
        found_use = true;
        if selection.is_some() || Operation::get_op::<BroadcastOp>(operation, context).is_none() {
            return false;
        }
    }
    found_use
}

fn only_used_by_returns(context: &Context, operations: &[Ptr<Operation>], value: Value) -> bool {
    let mut found_use = false;
    for &operation in operations {
        if !operation
            .deref(context)
            .operands()
            .any(|operand| operand == value)
        {
            continue;
        }
        if Operation::get_op::<ReturnOp>(operation, context).is_some() {
            found_use = true;
            continue;
        }
        return false;
    }
    found_use
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct BroadcastIndexing {
    source: BufferDescriptor,
    result: BufferDescriptor,
    dimensions: Vec<usize>,
    source_strides: Vec<usize>,
    result_strides: Vec<usize>,
}

fn dense_strides(shape: &[i64]) -> Result<Vec<usize>, PlanError> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(usize::try_from(shape[axis]).map_err(|_| PlanError::InvalidShape)?)
            .ok_or(PlanError::SizeOverflow)?;
    }
    Ok(strides)
}

fn broadcast_indexing(
    source: &BufferDescriptor,
    result: &BufferDescriptor,
    dimensions: &[i64],
) -> Result<BroadcastIndexing, PlanError> {
    if source.element_type != result.element_type {
        return Err(PlanError::UnsupportedFeature(
            "broadcast_in_dim requires matching source and result element types".into(),
        ));
    }
    let source_rank = source.shape.len();
    let result_rank = result.shape.len();
    if dimensions.len() != source_rank {
        return Err(PlanError::UnsupportedFeature(format!(
            "broadcast_in_dim dimensions length {} does not match source rank {source_rank}",
            dimensions.len()
        )));
    }
    let mut mapped_dimensions = Vec::with_capacity(source_rank);
    let mut seen = HashSet::with_capacity(source_rank);
    for (source_axis, &axis) in dimensions.iter().enumerate() {
        let axis = usize::try_from(axis).map_err(|_| {
            PlanError::UnsupportedFeature(format!(
                "broadcast_in_dim source dimension {source_axis} maps to a negative result axis"
            ))
        })?;
        if axis >= result_rank {
            return Err(PlanError::UnsupportedFeature(format!(
                "broadcast_in_dim source dimension {source_axis} maps outside result rank {result_rank}"
            )));
        }
        if !seen.insert(axis) {
            return Err(PlanError::UnsupportedFeature(format!(
                "broadcast_in_dim maps multiple source dimensions to result axis {axis}"
            )));
        }
        let source_extent =
            usize::try_from(source.shape[source_axis]).map_err(|_| PlanError::InvalidShape)?;
        let result_extent =
            usize::try_from(result.shape[axis]).map_err(|_| PlanError::InvalidShape)?;
        if source_extent != result_extent && source_extent != 1 {
            return Err(PlanError::UnsupportedFeature(format!(
                "broadcast_in_dim source extent {source_extent} at dimension {source_axis} cannot map to result extent {result_extent} at axis {axis}"
            )));
        }
        mapped_dimensions.push(axis);
    }
    source.num_elements()?;
    result.num_elements()?;
    Ok(BroadcastIndexing {
        source: source.clone(),
        result: result.clone(),
        dimensions: mapped_dimensions,
        source_strides: dense_strides(&source.shape)?,
        result_strides: dense_strides(&result.shape)?,
    })
}

fn mapped_source_index(scope: &Scope, position: Value, indexing: &BroadcastIndexing) -> Value {
    let mut offset = scope.const_usize(0);
    for source_axis in 0..indexing.source.shape.len() {
        let source_extent = indexing.source.shape[source_axis] as usize;
        let result_axis = indexing.dimensions[source_axis];
        let result_extent = indexing.result.shape[result_axis] as usize;
        let coordinate = if source_extent <= 1 || result_extent == 0 {
            scope.const_usize(0)
        } else {
            let stride = scope.const_usize(indexing.result_strides[result_axis].max(1));
            let linear =
                scope.register_with_result(&UDivOp::new(scope.ctx_mut(), position, stride));
            let extent = scope.const_usize(result_extent);
            scope.register_with_result(&URemOp::new(scope.ctx_mut(), linear, extent))
        };
        let source_stride = scope.const_usize(indexing.source_strides[source_axis]);
        let term =
            scope.register_with_result(&IMulOp::new(scope.ctx_mut(), coordinate, source_stride));
        offset = scope.register_with_result(&IAddOp::new(scope.ctx_mut(), offset, term));
    }
    offset
}

fn lower_broadcast_operation(
    source: BufferId,
    result: BufferId,
    indexing: BroadcastIndexing,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    plan.add_kernel(
        format!("xlahlo-broadcast-{operation_index}"),
        Arc::new(BroadcastKernel {
            indexing: indexing.clone(),
        }),
        vec![
            KernelBinding {
                buffer: source,
                access: BufferAccess::Read,
            },
            KernelBinding {
                buffer: result,
                access: BufferAccess::Write,
            },
        ],
        LaunchGeometry::for_elements(indexing.result.num_elements()?, [CUBE_DIM, 1, 1])?,
        false,
    )
}

fn lower_inline_constant_broadcast(
    constant: &ConstantValue,
    indexing: BroadcastIndexing,
    result: BufferId,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    plan.add_kernel(
        format!("xlahlo-broadcast-{operation_index}"),
        Arc::new(InlineBroadcastKernel {
            indexing: indexing.clone(),
            bits: constant.bits.clone(),
        }),
        vec![KernelBinding {
            buffer: result,
            access: BufferAccess::Write,
        }],
        LaunchGeometry::for_elements(indexing.result.num_elements()?, [CUBE_DIM, 1, 1])?,
        false,
    )
}

struct ConstantKernel {
    descriptor: BufferDescriptor,
    element_type: ElementType,
    bits: Vec<u128>,
}

impl KernelMetadata for ConstantKernel {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_constant"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            CODEGEN_VERSION,
            self.descriptor.clone(),
            self.element_type,
            self.bits.clone(),
        ))
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for ConstantKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl ConstantKernel {
    fn define(&self) -> Result<KernelDefinition, PlanError> {
        let elements = self.descriptor.num_elements()?;
        let settings = KernelSettings::new(
            Dim3::new_1d(CUBE_DIM),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let element_type = self
            .descriptor
            .cubecl_element_type()?
            .to_type(builder.ctx());
        let output = builder.buffer(element_type);
        let index_type = ElemType::Index.to_type(builder.ctx());
        let position = builder.register_with_result(&ReadBuiltinOp::new(
            builder.ctx_mut(),
            index_type,
            BuiltinAttr(Builtin::AbsolutePos),
        ));
        let count = builder.const_usize(elements);
        let in_bounds =
            builder.register_with_result(&ULessThanOp::new(builder.ctx_mut(), position, count));
        let guard = IfOp::new(builder.ctx_mut(), in_bounds);
        builder.register(&guard);
        let body = builder.child(OpInserter::new_at_block_end(
            guard.then_block(builder.ctx()),
        ));
        let value = inline_literal_value(&body, self.element_type, &self.bits, position)?;
        let pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), output, position, None));
        body.register(&StoreOp::new(body.ctx_mut(), pointer, value));
        body.register(&YieldOp::new(body.ctx_mut()));
        let otherwise = builder.child(OpInserter::new_at_block_end(
            guard.else_block(builder.ctx()),
        ));
        otherwise.register(&YieldOp::new(otherwise.ctx_mut()));
        builder.register(&cubecl_ir::dialect::branch::ReturnOp::new(
            builder.ctx_mut(),
        ));
        Ok(builder.build())
    }
}

struct BroadcastKernel {
    indexing: BroadcastIndexing,
}

impl KernelMetadata for BroadcastKernel {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_broadcast_in_dim"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((CODEGEN_VERSION, self.indexing.clone()))
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for BroadcastKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl BroadcastKernel {
    fn define(&self) -> Result<KernelDefinition, PlanError> {
        let elements = self.indexing.result.num_elements()?;
        let settings = KernelSettings::new(
            Dim3::new_1d(CUBE_DIM),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let source_type = self
            .indexing
            .source
            .cubecl_element_type()?
            .to_type(builder.ctx());
        let result_type = self
            .indexing
            .result
            .cubecl_element_type()?
            .to_type(builder.ctx());
        let source = builder.buffer(source_type);
        let output = builder.buffer(result_type);
        let index_type = ElemType::Index.to_type(builder.ctx());
        let position = builder.register_with_result(&ReadBuiltinOp::new(
            builder.ctx_mut(),
            index_type,
            BuiltinAttr(Builtin::AbsolutePos),
        ));
        let count = builder.const_usize(elements);
        let in_bounds =
            builder.register_with_result(&ULessThanOp::new(builder.ctx_mut(), position, count));
        let guard = IfOp::new(builder.ctx_mut(), in_bounds);
        builder.register(&guard);
        let body = builder.child(OpInserter::new_at_block_end(
            guard.then_block(builder.ctx()),
        ));
        let source_position = mapped_source_index(&body, position, &self.indexing);
        let source_pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), source, source_position, None));
        let value = body.register_with_result(&LoadOp::new(body.ctx_mut(), source_pointer));
        let output_pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), output, position, None));
        body.register(&StoreOp::new(body.ctx_mut(), output_pointer, value));
        body.register(&YieldOp::new(body.ctx_mut()));
        let otherwise = builder.child(OpInserter::new_at_block_end(
            guard.else_block(builder.ctx()),
        ));
        otherwise.register(&YieldOp::new(otherwise.ctx_mut()));
        builder.register(&cubecl_ir::dialect::branch::ReturnOp::new(
            builder.ctx_mut(),
        ));
        Ok(builder.build())
    }
}

struct InlineBroadcastKernel {
    indexing: BroadcastIndexing,
    bits: Vec<u128>,
}

impl KernelMetadata for InlineBroadcastKernel {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_inline_constant_broadcast"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((CODEGEN_VERSION, self.indexing.clone(), self.bits.clone()))
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for InlineBroadcastKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl InlineBroadcastKernel {
    fn define(&self) -> Result<KernelDefinition, PlanError> {
        let elements = self.indexing.result.num_elements()?;
        let settings = KernelSettings::new(
            Dim3::new_1d(CUBE_DIM),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let result_type = self
            .indexing
            .result
            .cubecl_element_type()?
            .to_type(builder.ctx());
        let output = builder.buffer(result_type);
        let index_type = ElemType::Index.to_type(builder.ctx());
        let position = builder.register_with_result(&ReadBuiltinOp::new(
            builder.ctx_mut(),
            index_type,
            BuiltinAttr(Builtin::AbsolutePos),
        ));
        let count = builder.const_usize(elements);
        let in_bounds =
            builder.register_with_result(&ULessThanOp::new(builder.ctx_mut(), position, count));
        let guard = IfOp::new(builder.ctx_mut(), in_bounds);
        builder.register(&guard);
        let body = builder.child(OpInserter::new_at_block_end(
            guard.then_block(builder.ctx()),
        ));
        let source_index = mapped_source_index(&body, position, &self.indexing);
        let value = inline_literal_value(
            &body,
            self.indexing.source.element_type,
            &self.bits,
            source_index,
        )?;
        let pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), output, position, None));
        body.register(&StoreOp::new(body.ctx_mut(), pointer, value));
        body.register(&YieldOp::new(body.ctx_mut()));
        let otherwise = builder.child(OpInserter::new_at_block_end(
            guard.else_block(builder.ctx()),
        ));
        otherwise.register(&YieldOp::new(otherwise.ctx_mut()));
        builder.register(&cubecl_ir::dialect::branch::ReturnOp::new(
            builder.ctx_mut(),
        ));
        Ok(builder.build())
    }
}

fn inline_literal_value(
    scope: &Scope,
    element_type: ElementType,
    bits: &[u128],
    index: Value,
) -> Result<Value, PlanError> {
    // Empty outputs never execute the guarded body.
    let first = bits.first().copied().unwrap_or_default();
    let first_constant = direct::cube_scalar_constant(scope.ctx_mut(), element_type, first)
        .map_err(PlanError::UnsupportedFeature)?;
    let mut selected = first_constant.deref(scope.ctx()).get_result(0);
    scope.register(&*Operation::get_op_dyn(first_constant, scope.ctx()));
    for (literal_index, &literal_bits) in bits.iter().enumerate().skip(1) {
        let literal_index = scope.const_usize(literal_index);
        let matches =
            scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), index, literal_index));
        let constant = direct::cube_scalar_constant(scope.ctx_mut(), element_type, literal_bits)
            .map_err(PlanError::UnsupportedFeature)?;
        let value = constant.deref(scope.ctx()).get_result(0);
        scope.register(&*Operation::get_op_dyn(constant, scope.ctx()));
        selected = scope.register_with_result(&CubeSelectOp::new(
            scope.ctx_mut(),
            matches,
            value,
            selected,
        ));
    }
    Ok(selected)
}

#[allow(clippy::too_many_arguments)]
fn lower_builtin_operation(
    context: &Context,
    operation: Ptr<Operation>,
    operands: &[BufferId],
    operand_descriptors: &[BufferDescriptor],
    operand_views: &[OperandView],
    result: BufferId,
    result_descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let operation_ref = Operation::get_op_dyn(operation, context);
    let hlo = pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).ok_or_else(|| {
        PlanError::UnsupportedFeature("operation has no supported xlahlo opcode".into())
    })?;
    if hlo.opcode() == Opcode::Gather {
        if operand_views.iter().any(|view| !view.transforms.is_empty()) {
            return gather::lower_gather_view_operation(
                context,
                operation,
                operand_views,
                result,
                result_descriptor,
                operation_index,
                plan,
            );
        }
        return gather::lower_gather_operation(
            context,
            operation,
            operands,
            operand_descriptors,
            result,
            result_descriptor,
            operation_index,
            plan,
        );
    }

    let extrema_kind = match hlo.opcode() {
        Opcode::Minimum => Some(direct::ExtremaKind::Minimum),
        Opcode::Maximum => Some(direct::ExtremaKind::Maximum),
        Opcode::Clamp => Some(direct::ExtremaKind::Clamp),
        _ => None,
    };
    if let Some(kind) = extrema_kind {
        if operand_views.iter().any(|view| !view.transforms.is_empty()) {
            return direct::lower_extrema_view_operation(
                kind,
                operand_views,
                result,
                result_descriptor,
                operation_index,
                plan,
            );
        }
        return direct::lower_extrema_operation(
            kind,
            operands,
            operand_descriptors,
            result,
            result_descriptor,
            operation_index,
            plan,
        );
    }
    if !matches!(
        hlo.opcode(),
        Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
    ) || operands.len() != 2
        || operand_descriptors
            .iter()
            .any(|descriptor| descriptor != result_descriptor)
    {
        return Err(PlanError::UnsupportedFeature(format!(
            "compositional built-in operation {}",
            hlo.opcode()
        )));
    }
    if operand_views.iter().any(|view| !view.transforms.is_empty()) {
        return direct::lower_binary_view_operation(
            hlo.opcode(),
            operand_views,
            result,
            result_descriptor,
            operation_index,
            plan,
        );
    }
    direct::lower_binary_operation(
        hlo.opcode(),
        operands[0],
        operands[1],
        result,
        result_descriptor,
        operation_index,
        plan,
    )
}
