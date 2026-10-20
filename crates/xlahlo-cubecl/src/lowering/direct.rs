use super::*;

mod arithmetic;
mod binary;
mod conversion;
mod extrema;

pub(super) use binary::{lower_binary_operation, lower_binary_view_operation};
pub(super) use conversion::cube_scalar_constant;
pub(super) use extrema::{ExtremaKind, lower_extrema_operation, lower_extrema_view_operation};

pub(super) fn lower_direct_module(
    context: &Context,
    module: Ptr<Operation>,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let program = DirectProgram::analyze(context, module)?;
    let mut input_buffers = Vec::with_capacity(program.inputs.len());
    let mut bindings = Vec::with_capacity(program.inputs.len() + program.outputs.len());
    for (index, descriptor) in program.inputs.iter().enumerate() {
        let buffer = plan.add_buffer(BufferRole::Input(index), descriptor.clone());
        input_buffers.push(buffer);
        bindings.push(KernelBinding {
            buffer,
            access: BufferAccess::Read,
        });
    }
    let mut output_buffers = Vec::with_capacity(program.outputs.len());
    for (index, descriptor) in program.outputs.iter().enumerate() {
        if let Some(input_index) = program.output_aliases[index] {
            let input = *input_buffers.get(input_index).ok_or_else(|| {
                PlanError::LoweringFailure("output alias references an unknown input".into())
            })?;
            plan.add_output(PlannedOutput::Alias(input));
            output_buffers.push(None);
        } else if let Some(bytes) = &program.static_outputs[index] {
            plan.add_output(PlannedOutput::Static(StaticValue {
                descriptor: descriptor.clone(),
                bytes: bytes.clone(),
            }));
            output_buffers.push(None);
        } else {
            let output = plan.add_buffer(BufferRole::Output(index), descriptor.clone());
            plan.add_output(PlannedOutput::Buffer(output));
            output_buffers.push(Some(output));
            bindings.push(KernelBinding {
                buffer: output,
                access: BufferAccess::Write,
            });
        }
    }
    let Some(first_output) = output_buffers.iter().position(Option::is_some) else {
        return Ok(());
    };
    let elements = program.outputs[first_output].num_elements()?;
    let launch = LaunchGeometry::for_elements(elements, [CUBE_DIM, 1, 1])?;
    plan.add_kernel("xlahlo-module", Arc::new(program), bindings, launch, false)
}

struct DirectProgram {
    module_ir: String,
    inputs: Vec<BufferDescriptor>,
    outputs: Vec<BufferDescriptor>,
    output_aliases: Vec<Option<usize>>,
    static_outputs: Vec<Option<Vec<u8>>>,
    tuple_result: bool,
}

impl DirectProgram {
    fn analyze(context: &Context, module: Ptr<Operation>) -> Result<Self, PlanError> {
        let entry = xlahlo::ir::entry(context, module)
            .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
        let mut entry_computation = None;
        for region in module.deref(context).regions() {
            for block in region.deref(context).iter(context) {
                for operation in block.deref(context).iter(context) {
                    if let Some(computation) =
                        Operation::get_op::<ComputationOp>(operation, context)
                        && computation
                            .get_attr_hlo_computation_name(context)
                            .is_some_and(|name| *name.as_ref() == entry)
                    {
                        entry_computation = Some(computation);
                    }
                }
            }
        }
        let computation = entry_computation
            .ok_or_else(|| PlanError::LoweringFailure("entry computation is missing".into()))?;
        let block = computation
            .body(context)
            .deref(context)
            .get_entry_block()
            .ok_or_else(|| PlanError::LoweringFailure("entry block is missing".into()))?;
        let operations: Vec<_> = block.deref(context).iter(context).collect();
        let mut inputs: Vec<Option<BufferDescriptor>> = Vec::new();
        let mut outputs = None;
        let mut output_aliases = None;
        let mut static_outputs = None;
        let mut saw_return = false;
        let mut producer_shapes = Vec::new();
        let mut tuple_operations = Vec::new();
        for &operation in &operations {
            if let Some(parameter) = Operation::get_op::<ParameterOp>(operation, context) {
                let index = parameter
                    .index(context)
                    .map(|index| index.0 as usize)
                    .ok_or_else(|| {
                        PlanError::LoweringFailure("parameter index is missing".into())
                    })?;
                let descriptor = descriptor(
                    context,
                    operation.deref(context).get_result(0).get_type(context),
                )?;
                if index >= inputs.len() {
                    inputs.resize_with(index + 1, || None);
                }
                if inputs[index].replace(descriptor).is_some() {
                    return Err(PlanError::LoweringFailure(
                        "duplicate entry parameter index".into(),
                    ));
                }
                continue;
            }
            if Operation::is_op::<ReturnOp>(operation, context) {
                if saw_return {
                    return Err(PlanError::LoweringFailure(
                        "entry computation has multiple returns".into(),
                    ));
                }
                saw_return = true;
                let return_operands: Vec<_> = operation.deref(context).operands().collect();
                let [returned_value] = return_operands.as_slice() else {
                    return Err(PlanError::LoweringFailure(
                        "entry return must have one HLO result value".into(),
                    ));
                };
                let operands: Vec<_> = if Shape::is_tuple(context, returned_value.get_type(context))
                {
                    let tuple_operation = returned_value
                        .defining_op()
                        .filter(|definition| Operation::is_op::<TupleOp>(*definition, context))
                        .ok_or_else(|| {
                            PlanError::UnsupportedFeature(
                                "entry tuple result must be constructed by xlahlo.tuple".into(),
                            )
                        })?;
                    tuple_operation.deref(context).operands().collect()
                } else {
                    vec![*returned_value]
                };
                let mut result_descriptors = Vec::with_capacity(operands.len());
                let mut result_aliases = Vec::with_capacity(operands.len());
                let mut result_static = Vec::with_capacity(operands.len());
                for value in operands {
                    result_descriptors.push(descriptor(context, value.get_type(context))?);
                    let definition = value.defining_op();
                    let alias = definition
                        .and_then(|definition| {
                            Operation::get_op::<ParameterOp>(definition, context)
                        })
                        .and_then(|parameter| parameter.index(context))
                        .map(|index| index.0 as usize);
                    let static_bytes = if let Some(definition) = definition
                        && let Some(constant) = Operation::get_op::<ConstantOp>(definition, context)
                    {
                        let literal = constant.literal(context).ok_or_else(|| {
                            PlanError::LoweringFailure(
                                "constant literal attribute is missing".into(),
                            )
                        })?;
                        if descriptor(context, literal.shape)?
                            != descriptor(context, value.get_type(context))?
                        {
                            return Err(PlanError::LoweringFailure(
                                "constant literal shape does not match its result".into(),
                            ));
                        }
                        Some(literal.storage.clone())
                    } else {
                        None
                    };
                    result_aliases.push(alias);
                    result_static.push(static_bytes);
                }
                outputs = Some(result_descriptors);
                output_aliases = Some(result_aliases);
                static_outputs = Some(result_static);
                continue;
            }
            if Operation::is_op::<TupleOp>(operation, context) {
                tuple_operations.push(operation);
                continue;
            }
            if Operation::get_op::<ConstantOp>(operation, context).is_some() {
                continue;
            }
            if let Some(broadcast) = Operation::get_op::<BroadcastOp>(operation, context) {
                let operands: Vec<_> = operation.deref(context).operands().collect();
                let [operand] = operands.as_slice() else {
                    return Err(PlanError::LoweringFailure(
                        "broadcast instruction arity is not one".into(),
                    ));
                };
                let source = descriptor(context, operand.get_type(context))?;
                let result = descriptor(
                    context,
                    operation.deref(context).get_result(0).get_type(context),
                )?;
                let dimensions = broadcast
                    .dimensions(context)
                    .ok_or_else(|| {
                        PlanError::LoweringFailure(
                            "broadcast dimensions attribute is missing".into(),
                        )
                    })?
                    .0
                    .clone();
                if !source.shape.is_empty()
                    || !dimensions.is_empty()
                    || source.element_type != result.element_type
                {
                    return Err(PlanError::UnsupportedFeature(
                        "broadcast lowering currently requires a scalar operand".into(),
                    ));
                }
                producer_shapes.push(result);
                continue;
            }
            if let Some(fusion) = Operation::get_op::<FusionOp>(operation, context) {
                collect_fusion_shapes(context, &fusion, &mut producer_shapes)?;
                continue;
            }
            let operation_ref = Operation::get_op_dyn(operation, context);
            let hlo =
                pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).ok_or_else(|| {
                    PlanError::UnsupportedFeature(
                        "entry operation is not a supported xlahlo instruction".into(),
                    )
                })?;
            if !matches!(
                hlo.opcode(),
                Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
            ) {
                return Err(PlanError::UnsupportedFeature(format!(
                    "xlahlo operation {}",
                    hlo.opcode()
                )));
            }
            let operands: Vec<_> = operation.deref(context).operands().collect();
            if operands.len() != 2 {
                return Err(PlanError::LoweringFailure(
                    "binary instruction arity is not two".into(),
                ));
            }
            let result_shape = descriptor(
                context,
                operation.deref(context).get_result(0).get_type(context),
            )?;
            for operand in operands {
                if descriptor(context, operand.get_type(context))? != result_shape {
                    return Err(PlanError::UnsupportedFeature(
                        "elementwise operands and result must have identical dense f32 shapes"
                            .into(),
                    ));
                }
            }
            producer_shapes.push(result_shape);
        }
        if !saw_return {
            return Err(PlanError::UnsupportedFeature(
                "entry computation without a tensor return".into(),
            ));
        }
        let outputs =
            outputs.ok_or_else(|| PlanError::LoweringFailure("entry result is missing".into()))?;
        let output_aliases = output_aliases
            .ok_or_else(|| PlanError::LoweringFailure("entry result aliases are missing".into()))?;
        let static_outputs = static_outputs
            .ok_or_else(|| PlanError::LoweringFailure("entry static results are missing".into()))?;
        if inputs.iter().any(Option::is_none) {
            return Err(PlanError::UnsupportedFeature(
                "entry parameters must be a contiguous tensor list".into(),
            ));
        }
        let inputs: Vec<_> = inputs.into_iter().flatten().collect();
        if outputs.len() != output_aliases.len() || outputs.len() != static_outputs.len() {
            return Err(PlanError::LoweringFailure(
                "entry result alias count does not match result count".into(),
            ));
        }
        let returned_tuple = block.deref(context).iter(context).find_map(|operation| {
            Operation::get_op::<ReturnOp>(operation, context)
                .and_then(|_| operation.deref(context).get_operand(0).defining_op())
                .filter(|definition| Operation::is_op::<TupleOp>(*definition, context))
        });
        if tuple_operations
            .iter()
            .any(|tuple| Some(*tuple) != returned_tuple)
        {
            return Err(PlanError::UnsupportedFeature(
                "only a directly returned tuple may be flattened into outputs".into(),
            ));
        }
        let mut computed_shape = None;
        for (index, output) in outputs.iter().enumerate() {
            if let Some(input_index) = output_aliases[index] {
                if inputs.get(input_index) != Some(output) {
                    return Err(PlanError::UnsupportedFeature(
                        "identity output must preserve its input descriptor".into(),
                    ));
                }
            } else if static_outputs[index].is_none()
                && computed_shape
                    .replace(output)
                    .is_some_and(|shape| shape != output)
            {
                return Err(PlanError::UnsupportedFeature(
                    "computed outputs in one kernel must have identical dense f32 shapes".into(),
                ));
            }
        }
        if outputs.is_empty() && !producer_shapes.is_empty() {
            return Err(PlanError::UnsupportedFeature(
                "no-result computations may not contain executable operations".into(),
            ));
        }
        if let Some(shape) = computed_shape {
            if producer_shapes.iter().any(|producer| producer != shape) {
                return Err(PlanError::UnsupportedFeature(
                    "elementwise instructions and computed outputs must have identical dense f32 shapes"
                        .into(),
                ));
            }
        } else if !producer_shapes.is_empty() {
            return Err(PlanError::UnsupportedFeature(
                "computation has operations but no computed output".into(),
            ));
        }
        Ok(Self {
            module_ir: module.disp(context).to_string(),
            inputs,
            outputs,
            output_aliases,
            static_outputs,
            tuple_result: returned_tuple.is_some(),
        })
    }
}

impl KernelMetadata for DirectProgram {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_module"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            CODEGEN_VERSION,
            self.module_ir.clone(),
            self.inputs.clone(),
            self.outputs.clone(),
            self.output_aliases.clone(),
            self.static_outputs.clone(),
            self.tuple_result,
        ))
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for DirectProgram {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl DirectProgram {
    fn define(&self) -> Result<KernelDefinition, PlanError> {
        let settings = KernelSettings::new(
            Dim3::new_1d(CUBE_DIM),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        xlahlo::register(builder.ctx_mut());
        let mut buffers = Vec::with_capacity(self.inputs.len());
        for descriptor in &self.inputs {
            let element_type = descriptor
                .cubecl_element_type()
                .map_err(|error| PlanError::LoweringFailure(error.to_string()))?
                .to_type(builder.ctx());
            buffers.push(builder.buffer(element_type));
        }
        let mut output_buffers = HashMap::new();
        for (index, descriptor) in self.outputs.iter().enumerate() {
            if self.output_aliases[index].is_none() && self.static_outputs[index].is_none() {
                let output_type = descriptor
                    .cubecl_element_type()
                    .map_err(|error| PlanError::LoweringFailure(error.to_string()))?
                    .to_type(builder.ctx());
                output_buffers.insert(index, builder.buffer(output_type));
            }
        }
        let first_computed = self
            .output_aliases
            .iter()
            .enumerate()
            .find_map(|(index, alias)| {
                (alias.is_none() && self.static_outputs[index].is_none()).then_some(index)
            })
            .ok_or_else(|| PlanError::LoweringFailure("kernel has no computed outputs".into()))?;
        let index_type = ElemType::Index.to_type(builder.ctx());
        let position = builder.register_with_result(&ReadBuiltinOp::new(
            builder.ctx_mut(),
            index_type,
            BuiltinAttr(Builtin::AbsolutePos),
        ));
        let zero_index = builder.const_usize(0);
        let count = builder.const_usize(
            self.outputs[first_computed]
                .num_elements()
                .map_err(|error| PlanError::LoweringFailure(error.to_string()))?,
        );
        let in_bounds =
            builder.register_with_result(&ULessThanOp::new(builder.ctx_mut(), position, count));
        let guard = IfOp::new(builder.ctx_mut(), in_bounds);
        builder.register(&guard);
        let body = guard.then_block(builder.ctx());
        let module = parse_from_str(
            spaced(Operation::top_level_parser()),
            builder.ctx_mut(),
            &self.module_ir,
        )
        .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
        let entry = xlahlo::ir::entry(builder.ctx(), module)
            .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
        let computation = find_computation(builder.ctx(), module, &entry)?;
        let source_block = computation
            .body(builder.ctx())
            .deref(builder.ctx())
            .get_entry_block()
            .ok_or_else(|| PlanError::LoweringFailure("entry block is missing".into()))?;
        if self.tuple_result {
            let return_operation = source_block
                .deref(builder.ctx())
                .iter(builder.ctx())
                .find(|operation| Operation::is_op::<ReturnOp>(*operation, builder.ctx()))
                .ok_or_else(|| PlanError::LoweringFailure("entry return is missing".into()))?;
            let tuple_operation = return_operation
                .deref(builder.ctx())
                .get_operand(0)
                .defining_op()
                .filter(|definition| Operation::is_op::<TupleOp>(*definition, builder.ctx()))
                .ok_or_else(|| {
                    PlanError::LoweringFailure("entry tuple producer is missing".into())
                })?;
            let values: Vec<_> = tuple_operation.deref(builder.ctx()).operands().collect();
            Operation::erase(return_operation, builder.ctx_mut());
            Operation::erase(tuple_operation, builder.ctx_mut());
            let flattened_return = Operation::new(
                builder.ctx_mut(),
                ReturnOp::get_concrete_op_info(),
                vec![],
                values,
                vec![],
                0,
            );
            flattened_return.insert_at_back(source_block, builder.ctx_mut());
        }
        let source_operations: Vec<_> = source_block
            .deref(builder.ctx())
            .iter(builder.ctx())
            .collect();
        for operation in source_operations {
            operation.unlink(builder.ctx());
            operation.insert_at_back(body, builder.ctx_mut());
        }
        let root = builder.scope.state().module.get_operation();
        let mut conversion = conversion::ElementwiseConversion::new(
            buffers.into_iter().enumerate().collect(),
            output_buffers,
            position,
            zero_index,
        );
        apply_dialect_conversion(builder.ctx_mut(), &mut conversion, root)
            .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
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

fn collect_fusion_shapes(
    context: &Context,
    fusion: &FusionOp,
    producer_shapes: &mut Vec<BufferDescriptor>,
) -> Result<(), PlanError> {
    let operation = fusion.get_operation();
    if fusion
        .get_attr_hlo_fusion_kind(context)
        .is_none_or(|kind| *kind != FusionKind::Loop)
        || operation
            .deref(context)
            .attributes
            .0
            .keys()
            .any(|key| key.to_string() != "hlo_fusion_kind")
    {
        return Err(PlanError::UnsupportedFeature(
            "non-loop fusion or fusion metadata".into(),
        ));
    }
    let result = operation.deref(context).get_result(0);
    let result_shape = descriptor(context, result.get_type(context))?;
    let operands: Vec<_> = operation.deref(context).operands().collect();
    let blocks: Vec<_> = fusion.body(context).deref(context).iter(context).collect();
    let [block] = blocks.as_slice() else {
        return Err(PlanError::LoweringFailure(
            "fusion body must contain one block".into(),
        ));
    };
    let block_arguments: Vec<_> = block.deref(context).arguments().collect();
    if block_arguments.len() != operands.len() {
        return Err(PlanError::UnsupportedFeature(
            "fusion parameter count does not match operands".into(),
        ));
    }
    for (argument, operand) in block_arguments.iter().zip(&operands) {
        if descriptor(context, argument.get_type(context))?
            != descriptor(context, operand.get_type(context))?
        {
            return Err(PlanError::UnsupportedFeature(
                "fusion operand and parameter descriptors differ".into(),
            ));
        }
    }
    let mut returned = None;
    for nested in block.deref(context).iter(context) {
        if Operation::is_op::<ReturnOp>(nested, context) {
            let values: Vec<_> = nested.deref(context).operands().collect();
            let [value] = values.as_slice() else {
                return Err(PlanError::UnsupportedFeature(
                    "fusion must return one tensor value".into(),
                ));
            };
            if descriptor(context, value.get_type(context))? != result_shape {
                return Err(PlanError::UnsupportedFeature(
                    "fusion result descriptor differs from its return".into(),
                ));
            }
            returned = Some(());
        } else {
            collect_instruction_shape(context, nested, producer_shapes)?;
        }
    }
    if returned.is_none() {
        return Err(PlanError::LoweringFailure(
            "fusion body has no return".into(),
        ));
    }
    Ok(())
}

fn collect_instruction_shape(
    context: &Context,
    operation: Ptr<Operation>,
    producer_shapes: &mut Vec<BufferDescriptor>,
) -> Result<(), PlanError> {
    if let Some(fusion) = Operation::get_op::<FusionOp>(operation, context) {
        collect_fusion_shapes(context, &fusion, producer_shapes)?;
        return Ok(());
    }
    let operation_ref = Operation::get_op_dyn(operation, context);
    let hlo = pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).ok_or_else(|| {
        PlanError::UnsupportedFeature(
            "fusion contains an unsupported non-xlahlo instruction".into(),
        )
    })?;
    if !matches!(
        hlo.opcode(),
        Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
    ) {
        return Err(PlanError::UnsupportedFeature(format!(
            "xlahlo operation {}",
            hlo.opcode()
        )));
    }
    let operands: Vec<_> = operation.deref(context).operands().collect();
    if operands.len() != 2 {
        return Err(PlanError::LoweringFailure(
            "binary instruction arity is not two".into(),
        ));
    }
    let result_shape = descriptor(
        context,
        operation.deref(context).get_result(0).get_type(context),
    )?;
    for operand in operands {
        let operand_shape = descriptor(context, operand.get_type(context))?;
        if operand_shape != result_shape {
            return Err(PlanError::UnsupportedFeature(
                "elementwise operands and result must have identical dense f32 shapes".into(),
            ));
        }
    }
    producer_shapes.push(result_shape);
    Ok(())
}
