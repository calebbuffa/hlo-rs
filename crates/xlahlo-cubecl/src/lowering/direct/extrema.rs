use super::*;
use pliron::r#type::TypeHandle;

#[derive(Clone, Copy)]
pub(in crate::lowering) enum ExtremaKind {
    Minimum,
    Maximum,
    Clamp,
}

pub(in crate::lowering) fn lower_extrema_operation(
    kind: ExtremaKind,
    operands: &[BufferId],
    operand_descriptors: &[BufferDescriptor],
    output: BufferId,
    output_descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    if operands.len() != operand_descriptors.len() {
        return Err(PlanError::LoweringFailure(
            "extrema descriptor count mismatch".into(),
        ));
    }
    let views: Vec<_> = operands
        .iter()
        .zip(operand_descriptors)
        .map(|(&buffer, descriptor)| crate::OperandView::dense(buffer, descriptor.clone()))
        .collect();
    lower_extrema_view_operation(
        kind,
        &views,
        output,
        output_descriptor,
        operation_index,
        plan,
    )
}

pub(in crate::lowering) fn lower_extrema_view_operation(
    kind: ExtremaKind,
    operands: &[crate::OperandView],
    output: BufferId,
    output_descriptor: &BufferDescriptor,
    operation_index: usize,
    plan: &mut ExecutionPlanBuilder,
) -> Result<(), PlanError> {
    let operand_descriptors: Vec<_> = operands
        .iter()
        .map(|view| view.descriptor.clone())
        .collect();
    let expected_operands = if matches!(kind, ExtremaKind::Clamp) {
        3
    } else {
        2
    };
    if operands.len() != expected_operands || operand_descriptors.len() != expected_operands {
        return Err(PlanError::LoweringFailure(
            "extrema operation operand count mismatch".into(),
        ));
    }
    let elements = output_descriptor.num_elements()?;
    if !output_descriptor.element_type.is_float()
        && !output_descriptor.element_type.is_integer()
        && output_descriptor.element_type != ElementType::Pred
    {
        return Err(PlanError::UnsupportedFeature(format!(
            "extrema for element type {:?}",
            output_descriptor.element_type
        )));
    }
    for descriptor in &operand_descriptors {
        if descriptor.element_type != output_descriptor.element_type
            || (descriptor.shape != output_descriptor.shape && !descriptor.shape.is_empty())
        {
            return Err(PlanError::UnsupportedFeature(
                "extrema operands must match the result shape or be scalars".into(),
            ));
        }
    }

    let mut bindings = operands
        .iter()
        .map(|view| KernelBinding {
            buffer: view.buffer,
            access: BufferAccess::Read,
        })
        .collect::<Vec<_>>();
    bindings.push(KernelBinding {
        buffer: output,
        access: BufferAccess::Write,
    });
    plan.add_kernel(
        format!("xlahlo-extrema-{operation_index}"),
        Arc::new(ExtremaKernel {
            kind,
            operand_descriptors: operand_descriptors.to_vec(),
            output_descriptor: output_descriptor.clone(),
            operand_views: operands.to_vec(),
        }),
        bindings,
        LaunchGeometry::for_elements(elements, [CUBE_DIM, 1, 1])?,
        false,
    )
}

struct ExtremaKernel {
    kind: ExtremaKind,
    operand_descriptors: Vec<BufferDescriptor>,
    output_descriptor: BufferDescriptor,
    operand_views: Vec<crate::OperandView>,
}

impl KernelMetadata for ExtremaKernel {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_extrema"
    }

    fn id(&self) -> KernelId {
        let kind = match self.kind {
            ExtremaKind::Minimum => "minimum",
            ExtremaKind::Maximum => "maximum",
            ExtremaKind::Clamp => "clamp",
        };
        KernelId::new::<Self>().info((
            CODEGEN_VERSION,
            kind,
            self.operand_descriptors.clone(),
            self.output_descriptor.clone(),
            self.operand_views
                .iter()
                .map(|view| view.transforms.clone())
                .collect::<Vec<_>>(),
        ))
    }

    fn address_type(&self) -> ElemType {
        AddressType::U32.unsigned_type()
    }
}

impl PlannedKernel for ExtremaKernel {
    fn define_checked(&self) -> Result<KernelDefinition, String> {
        self.define().map_err(|error| error.to_string())
    }
}

impl ExtremaKernel {
    fn define(&self) -> Result<KernelDefinition, PlanError> {
        let elements = self.output_descriptor.num_elements()?;
        let settings = KernelSettings::new(
            Dim3::new_1d(CUBE_DIM),
            ExecutionMode::Unchecked,
            AddressType::U32,
        )
        .kernel_name(self.name());
        let mut builder = KernelBuilder::new(settings);
        let element_type = self
            .output_descriptor
            .cubecl_element_type()?
            .to_type(builder.ctx());
        let input_buffers = self
            .operand_descriptors
            .iter()
            .map(|_| builder.buffer(element_type))
            .collect::<Vec<_>>();
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
        let values = self
            .operand_descriptors
            .iter()
            .zip(input_buffers)
            .zip(&self.operand_views)
            .map(|((descriptor, buffer), view)| {
                let index = if descriptor.shape.is_empty() {
                    body.const_usize(0)
                } else {
                    position
                };
                let index = views::source_position(&body, index, view)?;
                let pointer =
                    body.register_with_result(&IndexOp::new(body.ctx_mut(), buffer, index, None));
                Ok(body.register_with_result(&LoadOp::new(body.ctx_mut(), pointer)))
            })
            .collect::<Result<Vec<_>, PlanError>>()?;
        let result = emit_extrema(
            &body,
            self.kind,
            &values,
            self.output_descriptor.element_type,
        )?;
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

pub(in crate::lowering) fn emit_extrema(
    scope: &Scope,
    kind: ExtremaKind,
    values: &[Value],
    element_type: ElementType,
) -> Result<Value, PlanError> {
    match kind {
        ExtremaKind::Minimum | ExtremaKind::Maximum => {
            let [lhs, rhs] = values else {
                return Err(PlanError::LoweringFailure(
                    "minimum/maximum requires two values".into(),
                ));
            };
            emit_min_max(scope, kind, *lhs, *rhs, element_type)
        }
        ExtremaKind::Clamp => {
            let [minimum, value, maximum] = values else {
                return Err(PlanError::LoweringFailure(
                    "clamp requires three values".into(),
                ));
            };
            emit_clamp(scope, *minimum, *value, *maximum, element_type)
        }
    }
}

fn emit_min_max(
    scope: &Scope,
    kind: ExtremaKind,
    lhs: Value,
    rhs: Value,
    element_type: ElementType,
) -> Result<Value, PlanError> {
    if element_type.is_float() {
        let format = FloatFormat::new(scope.ctx(), element_type)?;
        let lhs = format.to_working_float(scope, lhs);
        let rhs = format.to_working_float(scope, rhs);
        let lhs_order = ordered_float(scope, lhs, format);
        let rhs_order = ordered_float(scope, rhs, format);
        let ordered_rhs = match kind {
            ExtremaKind::Minimum => scope.register_with_result(&ULessThanOp::new(
                scope.ctx_mut(),
                rhs_order.key,
                lhs_order.key,
            )),
            ExtremaKind::Maximum => scope.register_with_result(&ULessThanOp::new(
                scope.ctx_mut(),
                lhs_order.key,
                rhs_order.key,
            )),
            ExtremaKind::Clamp => {
                return Err(PlanError::LoweringFailure(
                    "clamp cannot be lowered as a binary extrema operation".into(),
                ));
            }
        };
        let both_zero = scope.register_with_result(&BoolAndOp::new(
            scope.ctx_mut(),
            lhs_order.zero,
            rhs_order.zero,
        ));
        let not_both_zero = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), both_zero));
        let ordered_rhs = scope.register_with_result(&BoolAndOp::new(
            scope.ctx_mut(),
            ordered_rhs,
            not_both_zero,
        ));
        let rhs_or_ordered =
            scope.register_with_result(&BoolOrOp::new(scope.ctx_mut(), rhs_order.nan, ordered_rhs));
        let not_lhs_nan =
            scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), lhs_order.nan));
        let choose_rhs = scope.register_with_result(&BoolAndOp::new(
            scope.ctx_mut(),
            not_lhs_nan,
            rhs_or_ordered,
        ));
        let result = select_float(scope, choose_rhs, rhs, lhs, format);
        return Ok(format.cast_working_float_to_result(scope, result));
    }
    let choose_rhs = match kind {
        ExtremaKind::Minimum if element_type.is_signed() => {
            scope.register_with_result(&SGreaterThanOp::new(scope.ctx_mut(), lhs, rhs))
        }
        ExtremaKind::Maximum if element_type.is_signed() => {
            scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), lhs, rhs))
        }
        ExtremaKind::Minimum => {
            scope.register_with_result(&UGreaterThanOp::new(scope.ctx_mut(), lhs, rhs))
        }
        ExtremaKind::Maximum => {
            scope.register_with_result(&ULessThanOp::new(scope.ctx_mut(), lhs, rhs))
        }
        ExtremaKind::Clamp => {
            return Err(PlanError::LoweringFailure(
                "clamp cannot be lowered as a binary extrema operation".into(),
            ));
        }
    };
    Ok(scope.register_with_result(&CubeSelectOp::new(scope.ctx_mut(), choose_rhs, rhs, lhs)))
}

fn emit_clamp(
    scope: &Scope,
    minimum: Value,
    value: Value,
    maximum: Value,
    element_type: ElementType,
) -> Result<Value, PlanError> {
    if element_type.is_float() {
        return emit_float_clamp(scope, minimum, value, maximum, element_type);
    }
    let below_minimum = if element_type.is_signed() {
        scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), value, minimum))
    } else {
        scope.register_with_result(&ULessThanOp::new(scope.ctx_mut(), value, minimum))
    };
    let raised = scope.register_with_result(&CubeSelectOp::new(
        scope.ctx_mut(),
        below_minimum,
        minimum,
        value,
    ));
    let below_maximum = if element_type.is_signed() {
        scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), raised, maximum))
    } else {
        scope.register_with_result(&ULessThanOp::new(scope.ctx_mut(), raised, maximum))
    };
    let numeric = scope.register_with_result(&CubeSelectOp::new(
        scope.ctx_mut(),
        below_maximum,
        raised,
        maximum,
    ));
    Ok(numeric)
}

fn emit_float_clamp(
    scope: &Scope,
    minimum: Value,
    value: Value,
    maximum: Value,
    element_type: ElementType,
) -> Result<Value, PlanError> {
    let format = FloatFormat::new(scope.ctx(), element_type)?;
    let minimum = format.to_working_float(scope, minimum);
    let value = format.to_working_float(scope, value);
    let maximum = format.to_working_float(scope, maximum);
    let minimum_order = ordered_float(scope, minimum, format);
    let value_order = ordered_float(scope, value, format);
    let maximum_order = ordered_float(scope, maximum, format);

    let below_minimum = scope.register_with_result(&ULessThanOp::new(
        scope.ctx_mut(),
        value_order.key,
        minimum_order.key,
    ));
    let both_zero = scope.register_with_result(&BoolAndOp::new(
        scope.ctx_mut(),
        value_order.zero,
        minimum_order.zero,
    ));
    let not_both_zero = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), both_zero));
    let below_minimum = scope.register_with_result(&BoolAndOp::new(
        scope.ctx_mut(),
        below_minimum,
        not_both_zero,
    ));
    let stage_one_mask = selection_mask(scope, below_minimum, format);
    let (raised, raised_key) = select_float_and_key(
        scope,
        stage_one_mask,
        minimum,
        value,
        minimum_order.key,
        value_order.key,
        format,
    );
    let raised_zero = float_is_zero(scope, raised, format);
    let below_maximum = scope.register_with_result(&ULessThanOp::new(
        scope.ctx_mut(),
        raised_key,
        maximum_order.key,
    ));
    let both_zero = scope.register_with_result(&BoolAndOp::new(
        scope.ctx_mut(),
        raised_zero,
        maximum_order.zero,
    ));
    let not_both_zero = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), both_zero));
    let below_maximum = scope.register_with_result(&BoolAndOp::new(
        scope.ctx_mut(),
        below_maximum,
        not_both_zero,
    ));
    let numeric = select_float(scope, below_maximum, raised, maximum, format);

    // XLA clamp gives NaNs priority in operand order: minimum, value, maximum.
    let nan_value = select_float(scope, value_order.nan, value, maximum, format);
    let nan_value = select_float(scope, minimum_order.nan, minimum, nan_value, format);
    let any_nan = scope.register_with_result(&BoolOrOp::new(
        scope.ctx_mut(),
        minimum_order.nan,
        value_order.nan,
    ));
    let any_nan =
        scope.register_with_result(&BoolOrOp::new(scope.ctx_mut(), any_nan, maximum_order.nan));
    let result = select_float(scope, any_nan, nan_value, numeric, format);
    Ok(format.cast_working_float_to_result(scope, result))
}

#[derive(Clone, Copy)]
struct FloatFormat {
    working_float_type: TypeHandle,
    result_type: TypeHandle,
    source_uint_type: TypeHandle,
    uint_type: TypeHandle,
    width: u32,
    element_mask: u64,
    sign_bit: u64,
    infinity: u64,
    cast_to_working: bool,
}

impl FloatFormat {
    fn new(context: &Context, element_type: ElementType) -> Result<Self, PlanError> {
        let (
            working_float_kind,
            source_uint_kind,
            uint_kind,
            width,
            element_mask,
            sign_bit,
            infinity,
            cast_to_working,
        ) = match element_type {
            ElementType::F16 => (
                FloatKind::F16,
                UIntKind::U16,
                UIntKind::U32,
                32,
                0xffff,
                0x8000,
                0x7c00,
                false,
            ),
            ElementType::BF16 => (
                FloatKind::BF16,
                UIntKind::U16,
                UIntKind::U32,
                32,
                0xffff,
                0x8000,
                0x7f80,
                false,
            ),
            ElementType::F32 => (
                FloatKind::F32,
                UIntKind::U32,
                UIntKind::U32,
                32,
                0xffff_ffff,
                0x8000_0000,
                0x7f80_0000,
                false,
            ),
            ElementType::F64 => (
                FloatKind::F64,
                UIntKind::U64,
                UIntKind::U64,
                64,
                u64::MAX,
                0x8000_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                false,
            ),
            element if element.is_float() => (
                FloatKind::F32,
                UIntKind::U32,
                UIntKind::U32,
                32,
                u32::MAX as u64,
                0x8000_0000,
                0x7f80_0000,
                true,
            ),
            _ => {
                return Err(PlanError::UnsupportedFeature(format!(
                    "floating extrema for element type {element_type:?}"
                )));
            }
        };
        let result_type = BufferDescriptor::dense_row_major(element_type, vec![])
            .cubecl_element_type()?
            .to_type(context);
        Ok(Self {
            working_float_type: ElemType::Float(working_float_kind).to_type(context),
            result_type,
            source_uint_type: ElemType::UInt(source_uint_kind).to_type(context),
            uint_type: ElemType::UInt(uint_kind).to_type(context),
            width,
            element_mask,
            sign_bit,
            infinity,
            cast_to_working,
        })
    }

    fn to_working_float(self, scope: &Scope, value: Value) -> Value {
        if self.cast_to_working {
            scope.register_with_result(&CastOp::new(
                scope.ctx_mut(),
                self.working_float_type,
                value,
            ))
        } else {
            value
        }
    }

    fn cast_working_float_to_result(self, scope: &Scope, value: Value) -> Value {
        if self.cast_to_working {
            scope.register_with_result(&CastOp::new(scope.ctx_mut(), self.result_type, value))
        } else {
            value
        }
    }

    fn unsigned_constant(self, scope: &Scope, value: u64) -> Value {
        let ty = IntegerType::get(scope.ctx(), self.width, Signedness::Unsigned);
        let attr = IntegerAttr::new(ty, APInt::from_u64(value, bw(self.width as usize)));
        scope.register_with_result(&CubeConstantOp::new(scope.ctx_mut(), Box::new(attr)))
    }

    fn bits(self, scope: &Scope, value: Value) -> Value {
        let bits = scope.register_with_result(&ReinterpretCastOp::new(
            scope.ctx_mut(),
            self.source_uint_type,
            value,
        ));
        if self.source_uint_type == self.uint_type {
            bits
        } else {
            scope.register_with_result(&CastOp::new(scope.ctx_mut(), self.uint_type, bits))
        }
    }

    fn value(self, scope: &Scope, bits: Value) -> Value {
        let bits = if self.source_uint_type == self.uint_type {
            bits
        } else {
            scope.register_with_result(&CastOp::new(scope.ctx_mut(), self.source_uint_type, bits))
        };
        scope.register_with_result(&ReinterpretCastOp::new(
            scope.ctx_mut(),
            self.working_float_type,
            bits,
        ))
    }
}

#[derive(Clone, Copy)]
struct OrderedFloat {
    key: Value,
    nan: Value,
    zero: Value,
}

fn ordered_float(scope: &Scope, value: Value, format: FloatFormat) -> OrderedFloat {
    let bits = format.bits(scope, value);
    let sign_bit = format.unsigned_constant(scope, format.sign_bit);
    let sign = scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), bits, sign_bit));
    let zero = format.unsigned_constant(scope, 0);
    let sign_zero = scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), sign, zero));
    let negative = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), sign_zero));
    let negative_flag =
        scope.register_with_result(&CastOp::new(scope.ctx_mut(), format.uint_type, negative));
    let negative_mask =
        scope.register_with_result(&ISubOp::new(scope.ctx_mut(), zero, negative_flag));
    let element_mask = format.unsigned_constant(scope, format.element_mask);
    let negative_mask = scope.register_with_result(&BitwiseAndOp::new(
        scope.ctx_mut(),
        negative_mask,
        element_mask,
    ));
    let xor_mask =
        scope.register_with_result(&BitwiseOrOp::new(scope.ctx_mut(), negative_mask, sign_bit));
    let key = scope.register_with_result(&BitwiseXorOp::new(scope.ctx_mut(), bits, xor_mask));
    let magnitude_mask = format.unsigned_constant(scope, format.element_mask ^ format.sign_bit);
    let magnitude =
        scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), bits, magnitude_mask));
    let zero_bits = format.unsigned_constant(scope, 0);
    let zero = scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), magnitude, zero_bits));
    let infinity = format.unsigned_constant(scope, format.infinity);
    let nan =
        scope.register_with_result(&UGreaterThanOp::new(scope.ctx_mut(), magnitude, infinity));
    OrderedFloat { key, nan, zero }
}

fn float_is_zero(scope: &Scope, value: Value, format: FloatFormat) -> Value {
    let bits = format.bits(scope, value);
    let magnitude_mask = format.unsigned_constant(scope, format.element_mask ^ format.sign_bit);
    let magnitude =
        scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), bits, magnitude_mask));
    let zero = format.unsigned_constant(scope, 0);
    scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), magnitude, zero))
}

#[derive(Clone, Copy)]
struct SelectionMask {
    mask: Value,
    inverse: Value,
}

fn selection_mask(scope: &Scope, condition: Value, format: FloatFormat) -> SelectionMask {
    let flag =
        scope.register_with_result(&CastOp::new(scope.ctx_mut(), format.uint_type, condition));
    let zero = format.unsigned_constant(scope, 0);
    let mask = scope.register_with_result(&ISubOp::new(scope.ctx_mut(), zero, flag));
    let all_bits = format.unsigned_constant(scope, u64::MAX);
    let inverse = scope.register_with_result(&BitwiseXorOp::new(scope.ctx_mut(), mask, all_bits));
    SelectionMask { mask, inverse }
}

fn select_float(
    scope: &Scope,
    condition: Value,
    on_true: Value,
    on_false: Value,
    format: FloatFormat,
) -> Value {
    let mask = selection_mask(scope, condition, format);
    select_float_with_mask(scope, mask, on_true, on_false, format)
}

fn select_float_with_mask(
    scope: &Scope,
    mask: SelectionMask,
    on_true: Value,
    on_false: Value,
    format: FloatFormat,
) -> Value {
    let true_bits = format.bits(scope, on_true);
    let false_bits = format.bits(scope, on_false);
    let selected = select_integer_with_mask(scope, mask, true_bits, false_bits);
    format.value(scope, selected)
}

fn select_integer_with_mask(
    scope: &Scope,
    mask: SelectionMask,
    on_true: Value,
    on_false: Value,
) -> Value {
    let true_part =
        scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), mask.mask, on_true));
    let false_part =
        scope.register_with_result(&BitwiseAndOp::new(scope.ctx_mut(), mask.inverse, on_false));
    scope.register_with_result(&BitwiseOrOp::new(scope.ctx_mut(), true_part, false_part))
}

fn select_float_and_key(
    scope: &Scope,
    mask: SelectionMask,
    on_true: Value,
    on_false: Value,
    true_key: Value,
    false_key: Value,
    format: FloatFormat,
) -> (Value, Value) {
    (
        select_float_with_mask(scope, mask, on_true, on_false, format),
        select_integer_with_mask(scope, mask, true_key, false_key),
    )
}
