use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GatherSpec {
    offset_dims: Vec<usize>,
    collapsed_slice_dims: Vec<usize>,
    start_index_map: Vec<usize>,
    index_vector_dim: Option<usize>,
    operand_batching_dims: Vec<usize>,
    start_indices_batching_dims: Vec<usize>,
    slice_sizes: Vec<usize>,
}

fn gather_axes(values: &[i64], attribute: &str) -> Result<Vec<usize>, PlanError> {
    values
        .iter()
        .map(|&axis| {
            usize::try_from(axis).map_err(|_| {
                PlanError::LoweringFailure(format!("Gather {attribute} contains a negative axis"))
            })
        })
        .collect()
}

fn gather_spec(
    context: &Context,
    operation: Ptr<Operation>,
    operand: &BufferDescriptor,
    indices: &BufferDescriptor,
    result: &BufferDescriptor,
) -> Result<GatherSpec, PlanError> {
    if !matches!(indices.element_type, ElementType::S32 | ElementType::U32) {
        return Err(PlanError::UnsupportedFeature(
            "Gather currently requires S32 or U32 indices".into(),
        ));
    }
    let gather = Operation::get_op::<GatherOp>(operation, context)
        .ok_or_else(|| PlanError::LoweringFailure("Gather operation wrapper is missing".into()))?;
    let dimensions = gather
        .dimensions(context)
        .ok_or_else(|| PlanError::LoweringFailure("Gather dimensions are missing".into()))?;
    let offset_dims = gather_axes(&dimensions.offset_dims.0, "offset_dims")?;
    let collapsed_slice_dims =
        gather_axes(&dimensions.collapsed_slice_dims.0, "collapsed_slice_dims")?;
    let start_index_map = gather_axes(&dimensions.start_index_map.0, "start_index_map")?;
    let operand_batching_dims =
        gather_axes(&dimensions.operand_batching_dims.0, "operand_batching_dims")?;
    let start_indices_batching_dims = gather_axes(
        &dimensions.start_indices_batching_dims.0,
        "start_indices_batching_dims",
    )?;
    let vector = usize::try_from(dimensions.index_vector_dim)
        .map_err(|_| PlanError::LoweringFailure("Gather index_vector_dim is negative".into()))?;
    let slice_sizes = gather
        .slice_sizes(context)
        .ok_or_else(|| PlanError::LoweringFailure("Gather slice_sizes are missing".into()))?
        .0
        .iter()
        .map(|&size| {
            usize::try_from(size)
                .map_err(|_| PlanError::LoweringFailure("Gather slice size is negative".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let index_vector_dim = if vector == indices.shape.len() {
        None
    } else if vector < indices.shape.len() {
        Some(vector)
    } else {
        return Err(PlanError::LoweringFailure(
            "Gather index_vector_dim is outside the index rank".into(),
        ));
    };
    let vector_size = match index_vector_dim {
        Some(axis) => usize::try_from(indices.shape[axis]).map_err(|_| PlanError::InvalidShape)?,
        None => 1,
    };
    let expected_batch_rank = indices.shape.len() - if index_vector_dim.is_some() { 1 } else { 0 };
    if vector_size != start_index_map.len()
        || operand.shape.len() != slice_sizes.len()
        || operand_batching_dims.len() != start_indices_batching_dims.len()
        || result.shape.len() != offset_dims.len() + expected_batch_rank
    {
        return Err(PlanError::LoweringFailure(
            "Gather dimensions are inconsistent with operand, indices, or result shape".into(),
        ));
    }
    let result_elements = result.num_elements()?;
    for (&extent, &slice_size) in operand.shape.iter().zip(&slice_sizes) {
        let extent = usize::try_from(extent).map_err(|_| PlanError::InvalidShape)?;
        if slice_size > extent {
            return Err(PlanError::LoweringFailure(
                "Gather slice size exceeds its operand dimension".into(),
            ));
        }
        if extent == 0 && result_elements != 0 {
            return Err(PlanError::UnsupportedFeature(
                "Gather cannot read an empty operand dimension for a non-empty result".into(),
            ));
        }
    }
    Ok(GatherSpec {
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        index_vector_dim,
        operand_batching_dims,
        start_indices_batching_dims,
        slice_sizes,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_gather_operation(
    context: &Context,
    operation: Ptr<Operation>,
    operands: &[BufferId],
    operand_descriptors: &[BufferDescriptor],
    result: BufferId,
    result_descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    if operands.len() != 2 || operand_descriptors.len() != 2 {
        return Err(PlanError::LoweringFailure(
            "Gather requires operand and indices inputs".into(),
        ));
    }
    let views: Vec<_> = operands
        .iter()
        .zip(operand_descriptors)
        .map(|(&buffer, descriptor)| crate::OperandView::dense(buffer, descriptor.clone()))
        .collect();
    lower_gather_view_operation(
        context,
        operation,
        &views,
        result,
        result_descriptor,
        operation_index,
        plan,
    )
}

pub(super) fn lower_gather_view_operation(
    context: &Context,
    operation: Ptr<Operation>,
    operands: &[crate::OperandView],
    result: BufferId,
    result_descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let [operand, indices] = operands else {
        return Err(PlanError::LoweringFailure(
            "Gather requires operand and indices inputs".into(),
        ));
    };
    let spec = gather_spec(
        context,
        operation,
        &operand.descriptor,
        &indices.descriptor,
        result_descriptor,
    )?;
    let elements = result_descriptor.num_elements()?;
    let kernel = DirectGatherKernel {
        operand: operand.descriptor.clone(),
        indices: indices.descriptor.clone(),
        result: result_descriptor.clone(),
        spec,
        views: operands.to_vec(),
    };
    plan.add_kernel(
        format!("xlahlo-gather-{operation_index}"),
        Arc::new(kernel),
        vec![
            KernelBinding {
                buffer: operand.buffer,
                access: BufferAccess::Read,
            },
            KernelBinding {
                buffer: indices.buffer,
                access: BufferAccess::Read,
            },
            KernelBinding {
                buffer: result,
                access: BufferAccess::Write,
            },
        ],
        LaunchGeometry::for_elements(elements, [CUBE_DIM, 1, 1])?,
        false,
    )
}

struct DirectGatherKernel {
    operand: BufferDescriptor,
    indices: BufferDescriptor,
    result: BufferDescriptor,
    spec: GatherSpec,
    views: Vec<crate::OperandView>,
}

impl KernelMetadata for DirectGatherKernel {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_gather"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            CODEGEN_VERSION,
            self.operand.clone(),
            self.indices.clone(),
            self.result.clone(),
            self.spec.clone(),
            self.views
                .iter()
                .map(|view| view.transforms.clone())
                .collect::<Vec<_>>(),
        ))
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for DirectGatherKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl DirectGatherKernel {
    fn define(&self) -> Result<KernelDefinition, PlanError> {
        let elements = self.result.num_elements()?;
        let settings = KernelSettings::new(
            Dim3::new_1d(CUBE_DIM),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let operand_buffer =
            builder.buffer(self.operand.cubecl_element_type()?.to_type(builder.ctx()));
        let indices_buffer =
            builder.buffer(self.indices.cubecl_element_type()?.to_type(builder.ctx()));
        let output_buffer =
            builder.buffer(self.result.cubecl_element_type()?.to_type(builder.ctx()));
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
        let output_coordinates = gather_output_coordinates(&body, position, &self.result.shape)?;
        let output_batch_dims: Vec<_> = (0..self.result.shape.len())
            .filter(|axis| !self.spec.offset_dims.contains(axis))
            .collect();
        let index_batch_dims: Vec<_> = (0..self.indices.shape.len())
            .filter(|axis| Some(*axis) != self.spec.index_vector_dim)
            .collect();
        if output_batch_dims.len() != index_batch_dims.len() {
            return Err(PlanError::LoweringFailure(
                "Gather output and index batch ranks differ".into(),
            ));
        }
        let index_axis_to_output_axis: HashMap<_, _> = index_batch_dims
            .iter()
            .copied()
            .zip(output_batch_dims.iter().copied())
            .collect();
        let mut operand_coordinates = vec![body.const_usize(0); self.operand.shape.len()];
        for (&operand_axis, &indices_axis) in self
            .spec
            .operand_batching_dims
            .iter()
            .zip(&self.spec.start_indices_batching_dims)
        {
            let output_axis = *index_axis_to_output_axis
                .get(&indices_axis)
                .ok_or_else(|| {
                    PlanError::LoweringFailure(
                        "Gather batching axis is missing from the output batch".into(),
                    )
                })?;
            operand_coordinates[operand_axis] = output_coordinates[output_axis];
        }
        let indices_strides = dense_strides(&self.indices.shape)?;
        let vector_stride = self.spec.index_vector_dim.map(|axis| indices_strides[axis]);
        for (component, &operand_axis) in self.spec.start_index_map.iter().enumerate() {
            let mut indices_offset = body.const_usize(0);
            for (&indices_axis, &output_axis) in index_batch_dims.iter().zip(&output_batch_dims) {
                let coordinate = output_coordinates[output_axis];
                let stride = body.const_usize(indices_strides[indices_axis]);
                let term =
                    body.register_with_result(&IMulOp::new(body.ctx_mut(), coordinate, stride));
                indices_offset =
                    body.register_with_result(&IAddOp::new(body.ctx_mut(), indices_offset, term));
            }
            if let Some(stride) = vector_stride {
                let component = body.const_usize(component);
                let stride = body.const_usize(stride);
                let term =
                    body.register_with_result(&IMulOp::new(body.ctx_mut(), component, stride));
                indices_offset =
                    body.register_with_result(&IAddOp::new(body.ctx_mut(), indices_offset, term));
            }
            let indices_offset = views::source_position(&body, indices_offset, &self.views[1])?;
            let pointer = body.register_with_result(&IndexOp::new(
                body.ctx_mut(),
                indices_buffer,
                indices_offset,
                None,
            ));
            let index = body.register_with_result(&LoadOp::new(body.ctx_mut(), pointer));
            let max_start = usize::try_from(self.operand.shape[operand_axis])
                .map_err(|_| PlanError::InvalidShape)?
                - self.spec.slice_sizes[operand_axis];
            operand_coordinates[operand_axis] =
                clamp_gather_index(&body, index, self.indices.element_type, max_start)?;
        }
        let window_operand_dims: Vec<_> = (0..self.operand.shape.len())
            .filter(|axis| {
                !self.spec.collapsed_slice_dims.contains(axis)
                    && !self.spec.operand_batching_dims.contains(axis)
            })
            .collect();
        if window_operand_dims.len() != self.spec.offset_dims.len() {
            return Err(PlanError::LoweringFailure(
                "Gather window and output offset ranks differ".into(),
            ));
        }
        for (&operand_axis, &output_axis) in window_operand_dims.iter().zip(&self.spec.offset_dims)
        {
            let window_coordinate = output_coordinates[output_axis];
            operand_coordinates[operand_axis] = body.register_with_result(&IAddOp::new(
                body.ctx_mut(),
                operand_coordinates[operand_axis],
                window_coordinate,
            ));
        }
        let operand_strides = dense_strides(&self.operand.shape)?;
        let mut operand_offset = body.const_usize(0);
        for (axis, &stride) in operand_strides.iter().enumerate() {
            let stride = body.const_usize(stride);
            let term = body.register_with_result(&IMulOp::new(
                body.ctx_mut(),
                operand_coordinates[axis],
                stride,
            ));
            operand_offset =
                body.register_with_result(&IAddOp::new(body.ctx_mut(), operand_offset, term));
        }
        let operand_offset = views::source_position(&body, operand_offset, &self.views[0])?;
        let input_pointer = body.register_with_result(&IndexOp::new(
            body.ctx_mut(),
            operand_buffer,
            operand_offset,
            None,
        ));
        let value = body.register_with_result(&LoadOp::new(body.ctx_mut(), input_pointer));
        let output_pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), output_buffer, position, None));
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

fn dense_strides(shape: &[i64]) -> Result<Vec<usize>, PlanError> {
    let mut strides = vec![0usize; shape.len()];
    let mut running = 1usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = running;
        let extent = usize::try_from(shape[axis]).map_err(|_| PlanError::InvalidShape)?;
        running = running.checked_mul(extent).ok_or(PlanError::SizeOverflow)?;
    }
    Ok(strides)
}

fn gather_output_coordinates(
    scope: &Scope,
    position: Value,
    shape: &[i64],
) -> Result<Vec<Value>, PlanError> {
    let strides = dense_strides(shape)?;
    shape
        .iter()
        .enumerate()
        .map(|(axis, &extent)| {
            let extent = usize::try_from(extent).map_err(|_| PlanError::InvalidShape)?;
            if extent == 0 {
                return Err(PlanError::InvalidShape);
            }
            let stride = scope.const_usize(strides[axis]);
            let linear =
                scope.register_with_result(&UDivOp::new(scope.ctx_mut(), position, stride));
            let extent = scope.const_usize(extent);
            Ok(scope.register_with_result(&URemOp::new(scope.ctx_mut(), linear, extent)))
        })
        .collect()
}

fn integer_constant(scope: &Scope, element_type: ElementType, bits: u32) -> Value {
    let signedness = if element_type.is_signed() {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    };
    let attr = IntegerAttr::new(
        IntegerType::get(scope.ctx(), element_type.bit_width(), signedness),
        APInt::from_u64(u64::from(bits), bw(element_type.bit_width() as usize)),
    );
    scope.register_with_result(&CubeConstantOp::new(scope.ctx_mut(), Box::new(attr)))
}

fn clamp_gather_index(
    scope: &Scope,
    value: Value,
    element_type: ElementType,
    max_start: usize,
) -> Result<Value, PlanError> {
    let max_value = match element_type {
        ElementType::S32 => max_start.min(i32::MAX as usize) as u32,
        ElementType::U32 => max_start.min(u32::MAX as usize) as u32,
        _ => {
            return Err(PlanError::UnsupportedFeature(
                "Gather currently requires S32 or U32 indices".into(),
            ));
        }
    };
    let zero = integer_constant(scope, element_type, 0);
    let upper = integer_constant(scope, element_type, max_value);
    let clamped = if element_type == ElementType::S32 {
        let negative = scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), value, zero));
        let nonnegative =
            scope.register_with_result(&CubeSelectOp::new(scope.ctx_mut(), negative, zero, value));
        let above =
            scope.register_with_result(&SGreaterThanOp::new(scope.ctx_mut(), nonnegative, upper));
        scope.register_with_result(&CubeSelectOp::new(
            scope.ctx_mut(),
            above,
            upper,
            nonnegative,
        ))
    } else {
        let above = scope.register_with_result(&UGreaterThanOp::new(scope.ctx_mut(), value, upper));
        scope.register_with_result(&CubeSelectOp::new(scope.ctx_mut(), above, upper, value))
    };
    let index_type = ElemType::Index.to_type(scope.ctx());
    Ok(scope.register_with_result(&CastOp::new(scope.ctx_mut(), index_type, clamped)))
}
