use super::arithmetic::emit_binary;
use super::*;

pub(in crate::lowering) fn lower_binary_operation(
    opcode: Opcode,
    lhs: BufferId,
    rhs: BufferId,
    output: BufferId,
    descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let operands = [
        crate::OperandView::dense(lhs, descriptor.clone()),
        crate::OperandView::dense(rhs, descriptor.clone()),
    ];
    lower_binary_view_operation(opcode, &operands, output, descriptor, operation_index, plan)
}

pub(in crate::lowering) fn lower_binary_view_operation(
    opcode: Opcode,
    operands: &[crate::OperandView],
    output: BufferId,
    descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let [lhs, rhs] = operands else {
        return Err(PlanError::LoweringFailure(
            "binary operation requires two operands".into(),
        ));
    };
    super::check_binary_primitive(opcode, descriptor.element_type)?;
    let elements = descriptor.num_elements()?;
    plan.add_kernel(
        format!("xlahlo-op-{operation_index}"),
        Arc::new(DirectBinaryKernel {
            opcode,
            descriptor: descriptor.clone(),
            operands: operands.to_vec(),
        }),
        vec![
            KernelBinding {
                buffer: lhs.buffer,
                access: BufferAccess::Read,
            },
            KernelBinding {
                buffer: rhs.buffer,
                access: BufferAccess::Read,
            },
            KernelBinding {
                buffer: output,
                access: BufferAccess::Write,
            },
        ],
        LaunchGeometry::for_elements(elements, [CUBE_DIM, 1, 1])?,
        false,
    )
}

struct DirectBinaryKernel {
    opcode: Opcode,
    descriptor: BufferDescriptor,
    operands: Vec<crate::OperandView>,
}

impl KernelMetadata for DirectBinaryKernel {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_binary"
    }

    fn id(&self) -> KernelId {
        let key = (
            CODEGEN_VERSION,
            self.opcode.to_string(),
            self.descriptor.clone(),
            self.operands
                .iter()
                .map(|operand| operand.transforms.clone())
                .collect::<Vec<_>>(),
        );
        if matches!(
            self.descriptor.element_type,
            ElementType::F8E4M3FN | ElementType::F8E5M2
        ) {
            KernelId::new::<Self>().info((key, "exact-fp8-rounding-v1"))
        } else {
            KernelId::new::<Self>().info(key)
        }
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for DirectBinaryKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl DirectBinaryKernel {
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
        let lhs_buffer = builder.buffer(element_type);
        let rhs_buffer = builder.buffer(element_type);
        let output_buffer = builder.buffer(element_type);
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
        let lhs_index = views::source_position(&body, position, &self.operands[0])?;
        let lhs_pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), lhs_buffer, lhs_index, None));
        let lhs = body.register_with_result(&LoadOp::new(body.ctx_mut(), lhs_pointer));
        let rhs_index = views::source_position(&body, position, &self.operands[1])?;
        let rhs_pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), rhs_buffer, rhs_index, None));
        let rhs = body.register_with_result(&LoadOp::new(body.ctx_mut(), rhs_pointer));
        let result = if self.descriptor.element_type == ElementType::U1 {
            let one = uint8_constant(&body, 1);
            let value = emit_binary(&body, self.opcode, lhs, rhs, ElementType::U8)?;
            body.register_with_result(&BitwiseAndOp::new(body.ctx_mut(), value, one))
        } else {
            emit_binary(&body, self.opcode, lhs, rhs, self.descriptor.element_type)?
        };
        let output_pointer =
            body.register_with_result(&IndexOp::new(body.ctx_mut(), output_buffer, position, None));
        emit_store(&body, output_pointer, result)?;
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

fn uint8_constant(scope: &Scope, value: u8) -> Value {
    let attr = IntegerAttr::new(
        IntegerType::get(scope.ctx(), 8, Signedness::Unsigned),
        APInt::from_u64(u64::from(value), bw(8)),
    );
    scope.register_with_result(&CubeConstantOp::new(scope.ctx_mut(), Box::new(attr)))
}
