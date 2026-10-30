use super::arithmetic::make_binary_operations;
use super::*;

use std::collections::HashSet;

pub(in crate::lowering) fn cube_scalar_constant(
    context: &mut Context,
    element_type: ElementType,
    bits: u128,
) -> Result<Ptr<Operation>, String> {
    let ty = BufferDescriptor::dense_row_major(element_type, vec![])
        .cubecl_element_type()
        .map_err(|error| error.to_string())?
        .to_type(context);
    let operation = match element_type {
        ElementType::F4E2M1FN => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<cubecl_ir::types::scalar::Float4E2M1S>(bits),
            )),
        ),
        ElementType::F8E4M3 | ElementType::F8E4M3FN => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<rustc_apfloat::ieee::Float8E4M3FNS>(bits),
            )),
        ),
        ElementType::F8E5M2 => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<rustc_apfloat::ieee::Float8E5M2S>(bits),
            )),
        ),
        ElementType::F16 => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<rustc_apfloat::ieee::HalfS>(bits),
            )),
        ),
        ElementType::BF16 => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<rustc_apfloat::ieee::BFloatS>(bits),
            )),
        ),
        ElementType::F32 => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<rustc_apfloat::ieee::SingleS>(bits),
            )),
        ),
        ElementType::F64 => CubeConstantOp::new(
            context,
            Box::new(FloatAttr::new(
                ty,
                APFloat::from_bits::<rustc_apfloat::ieee::DoubleS>(bits),
            )),
        ),
        ElementType::S8
        | ElementType::S16
        | ElementType::S32
        | ElementType::S64
        | ElementType::U1
        | ElementType::U8
        | ElementType::U16
        | ElementType::U32
        | ElementType::U64
        | ElementType::Pred => {
            let width = if matches!(element_type, ElementType::U1 | ElementType::Pred) {
                8
            } else {
                element_type.bit_width()
            };
            let signedness = if element_type.is_signed() {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            let bits = if matches!(element_type, ElementType::U1 | ElementType::Pred) {
                bits & 1
            } else {
                bits
            };
            let attr = IntegerAttr::new(
                IntegerType::get(context, width, signedness),
                APInt::from_u64(bits as u64, bw(width as usize)),
            );
            CubeConstantOp::new(context, Box::new(attr))
        }
        _ => return Err(format!("scalar constant for {element_type:?}")),
    };
    Ok(operation.get_operation())
}

pub(super) struct ElementwiseConversion {
    buffers: HashMap<usize, Value>,
    outputs: HashMap<usize, Value>,
    position: Value,
    zero_index: Value,
    values: HashMap<Value, Value>,
    positive_zero_values: HashSet<Value>,
    scalar_operations: HashMap<Ptr<Operation>, scalar::ScalarOperation>,
    custom_bodies: HashMap<Ptr<Operation>, Arc<dyn crate::ScalarLoweringBody>>,
    copies: HashSet<Ptr<Operation>>,
}

impl ElementwiseConversion {
    pub(super) fn new(
        context: &Context,
        root: Ptr<Operation>,
        buffers: HashMap<usize, Value>,
        outputs: HashMap<usize, Value>,
        position: Value,
        zero_index: Value,
        operation_strategies: &OperationStrategyRegistry,
    ) -> Result<Self, PlanError> {
        let mut pending = vec![root];
        let mut scalar_operations = HashMap::new();
        let mut custom_bodies = HashMap::new();
        let mut copies = HashSet::new();
        while let Some(operation) = pending.pop() {
            if let Some(body) = operation_strategies.scalar_body(context, operation)? {
                custom_bodies.insert(operation, body);
            } else if super::copy_shape(context, operation)?.is_some() {
                copies.insert(operation);
            }
            if let Some(specification) = scalar::ScalarOperation::read(context, operation)? {
                scalar_operations.insert(operation, specification);
            }
            for region in operation.deref(context).regions() {
                for block in region.deref(context).iter(context) {
                    pending.extend(block.deref(context).iter(context));
                }
            }
        }
        Ok(Self {
            buffers,
            outputs,
            position,
            zero_index,
            values: HashMap::new(),
            positive_zero_values: HashSet::new(),
            scalar_operations,
            custom_bodies,
            copies,
        })
    }
}

impl DialectConversion for ElementwiseConversion {
    fn can_convert_op(&self, context: &Context, operation: Ptr<Operation>) -> bool {
        if self.scalar_operations.contains_key(&operation)
            || self.custom_bodies.contains_key(&operation)
            || self.copies.contains(&operation)
        {
            return true;
        }
        if Operation::get_op::<ParameterOp>(operation, context).is_some()
            || Operation::get_op::<ConstantOp>(operation, context).is_some()
            || Operation::get_op::<BroadcastOp>(operation, context).is_some()
            || (Operation::get_op::<ReturnOp>(operation, context).is_some()
                && operation
                    .deref(context)
                    .get_parent_op(context)
                    .is_none_or(|parent| !Operation::is_op::<FusionOp>(parent, context)))
            || Operation::get_op::<FusionOp>(operation, context).is_some()
        {
            return true;
        }
        let operation_ref = Operation::get_op_dyn(operation, context);
        pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref).is_some_and(|hlo| {
            matches!(
                hlo.opcode(),
                Opcode::Add | Opcode::Subtract | Opcode::Multiply | Opcode::Divide
            )
        })
    }

    fn can_convert_type(&self, context: &Context, ty: TypeHandle) -> bool {
        shape::is_array(context, ty) && super::super::descriptor(context, ty).is_ok()
    }

    fn convert_type(
        &mut self,
        context: &mut Context,
        ty: TypeHandle,
    ) -> pliron::result::Result<TypeHandle> {
        super::super::descriptor(context, ty)
            .and_then(|descriptor| descriptor.cubecl_element_type())
            .map(|element_type| element_type.to_type(context))
            .map_err(|error| pliron::arg_error_noloc!("{}", error))
    }

    fn rewrite(
        &mut self,
        context: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operation: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> pliron::result::Result<()> {
        if self.copies.contains(&operation) || self.custom_bodies.contains_key(&operation) {
            let operands = operation
                .deref(context)
                .operands()
                .map(|value| self.values.get(&value).copied().unwrap_or(value))
                .collect::<Vec<_>>();
            let source_result = operation.deref(context).get_result(0);
            let lowered = if let Some(body) = self.custom_bodies.get(&operation) {
                let mut inserter = OpInserter::new_before_operation(operation);
                let scope = Scope::from_context_and_inserter(context, &mut inserter);
                body.emit(&scope, &operands)
                    .map_err(|error| pliron::arg_error_noloc!("{error}"))?
            } else {
                operands[0]
            };
            self.values.insert(source_result, lowered);
            rewriter.replace_operation_with_values(context, operation, vec![lowered]);
            return Ok(());
        }
        if let Some(parameter) = Operation::get_op::<ParameterOp>(operation, context) {
            let source_value = operation.deref(context).get_result(0);
            let index = parameter
                .index(context)
                .ok_or_else(|| pliron::arg_error_noloc!("parameter index is missing"))?
                .0 as usize;
            let buffer = *self
                .buffers
                .get(&index)
                .ok_or_else(|| pliron::arg_error_noloc!("parameter index is out of range"))?;
            let position = if shape::rank(context, source_value.get_type(context))
                .is_ok_and(|rank| rank == 0)
            {
                self.zero_index
            } else {
                self.position
            };
            let pointer = IndexOp::new(context, buffer, position, None);
            let pointer_operation = pointer.get_operation();
            let pointer_value = pointer_operation.deref(context).get_result(0);
            rewriter.insert_operation(context, pointer_operation);
            let load = LoadOp::new(context, pointer_value).get_operation();
            let loaded_value = load.deref(context).get_result(0);
            rewriter.insert_operation(context, load);
            self.values.insert(source_value, loaded_value);
            rewriter.replace_operation(context, operation, load);
            return Ok(());
        }
        if let Some(constant) = Operation::get_op::<ConstantOp>(operation, context) {
            let source_value = operation.deref(context).get_result(0);
            let (element_type, bits) = {
                let literal = constant
                    .literal(context)
                    .ok_or_else(|| pliron::arg_error_noloc!("constant literal is missing"))?;
                let result_type = source_value.get_type(context);
                let literal_descriptor = super::super::descriptor(context, literal.shape)
                    .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
                let result_descriptor = super::super::descriptor(context, result_type)
                    .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
                if shape::rank(context, result_type)
                    .map_err(|error| pliron::arg_error_noloc!("{}", error))?
                    != 0
                    || literal_descriptor != result_descriptor
                {
                    return Err(pliron::arg_error_noloc!(
                        "kernel scalar constant requires a rank-zero literal matching its result"
                    ));
                }
                let bits = literal
                    .bits(context)
                    .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
                let [bits] = bits.as_slice() else {
                    return Err(pliron::arg_error_noloc!(
                        "kernel scalar constant must contain one element"
                    ));
                };
                let element_type = shape::element_type(context, result_type)
                    .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
                (element_type, *bits)
            };
            let lowered = cube_scalar_constant(context, element_type, bits)
                .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
            let lowered_value = lowered.deref(context).get_result(0);
            if element_type.is_float() && bits == 0 {
                self.positive_zero_values.insert(source_value);
                self.positive_zero_values.insert(lowered_value);
            }
            self.values.insert(source_value, lowered_value);
            rewriter.insert_operation(context, lowered);
            rewriter.replace_operation(context, operation, lowered);
            return Ok(());
        }
        if Operation::get_op::<BroadcastOp>(operation, context).is_some() {
            let result = operation.deref(context).get_result(0);
            let operand = operation.deref(context).get_operand(0);
            let value = self.values.get(&operand).copied().unwrap_or(operand);
            if self.positive_zero_values.contains(&operand) {
                self.positive_zero_values.insert(result);
            }
            self.values.insert(result, value);
            rewriter.replace_operation_with_values(context, operation, vec![value]);
            return Ok(());
        }
        if let Some(fusion) = Operation::get_op::<FusionOp>(operation, context) {
            let result = operation.deref(context).get_result(0);
            let operands: Vec<_> = operation.deref(context).operands().collect();
            let blocks: Vec<_> = fusion.body(context).deref(context).iter(context).collect();
            let [block] = blocks.as_slice() else {
                return Err(pliron::arg_error_noloc!(
                    "fusion body must contain one block"
                ));
            };
            let block = *block;
            let arguments: Vec<_> = block.deref(context).arguments().collect();
            if arguments.len() != operands.len() {
                return Err(pliron::arg_error_noloc!(
                    "fusion parameter count does not match operands"
                ));
            }
            let mut replacements = HashMap::new();
            for (argument, operand) in arguments.iter().zip(operands) {
                let value = self.values.get(&operand).copied().unwrap_or(operand);
                if self.positive_zero_values.contains(&operand) {
                    self.positive_zero_values.insert(*argument);
                }
                rewriter.replace_value_uses_with(context, *argument, value);
                replacements.insert(*argument, value);
            }
            let nested: Vec<_> = block.deref(context).iter(context).collect();
            let return_operation = nested
                .last()
                .copied()
                .filter(|nested| Operation::is_op::<ReturnOp>(*nested, context))
                .ok_or_else(|| pliron::arg_error_noloc!("fusion return is missing"))?;
            let returned = return_operation.deref(context).get_operand(0);
            let returned = replacements
                .get(&returned)
                .copied()
                .or_else(|| self.values.get(&returned).copied())
                .unwrap_or(returned);
            for nested in nested
                .into_iter()
                .filter(|nested| *nested != return_operation)
            {
                rewriter.move_operation(
                    context,
                    nested,
                    OpInsertionPoint::BeforeOperation(operation),
                );
            }
            rewriter.erase_operation(context, return_operation);
            rewriter.replace_operation_with_values(context, operation, vec![returned]);
            self.values.insert(result, returned);
            return Ok(());
        }
        if Operation::get_op::<ReturnOp>(operation, context).is_some() {
            let values: Vec<_> = operation.deref(context).operands().collect();
            let mut last_store = None;
            for (index, value) in values.into_iter().enumerate() {
                let Some(output) = self.outputs.get(&index).copied() else {
                    continue;
                };
                let value = self.values.get(&value).copied().unwrap_or(value);
                let pointer = IndexOp::new(context, output, self.position, None);
                let pointer_operation = pointer.get_operation();
                let pointer_value = pointer_operation.deref(context).get_result(0);
                rewriter.insert_operation(context, pointer_operation);
                let stores = make_store_operations(context, pointer_value, value)
                    .map_err(|error| pliron::arg_error_noloc!("{error}"))?;
                for store in stores {
                    rewriter.insert_operation(context, store);
                    last_store = Some(store);
                }
            }
            let store = last_store
                .ok_or_else(|| pliron::arg_error_noloc!("kernel return has no computed outputs"))?;
            rewriter.replace_operation(context, operation, store);
            rewriter.set_insertion_point(OpInsertionPoint::AfterOperation(store));
            let yield_operation = YieldOp::new(context).get_operation();
            rewriter.insert_operation(context, yield_operation);
            return Ok(());
        }
        if let Some(specification) = self.scalar_operations.get(&operation) {
            let operands = operation
                .deref(context)
                .operands()
                .map(|value| self.values.get(&value).copied().unwrap_or(value))
                .collect::<Vec<_>>();
            let source_result = operation.deref(context).get_result(0);
            let mut inserter = OpInserter::new_before_operation(operation);
            let lowered = {
                let scope = Scope::from_context_and_inserter(context, &mut inserter);
                specification
                    .emit(&scope, self.position, &operands)
                    .map_err(|error| pliron::arg_error_noloc!("{}", error))?
            };
            self.values.insert(source_result, lowered);
            rewriter.replace_operation_with_values(context, operation, vec![lowered]);
            return Ok(());
        }
        let operation_ref = Operation::get_op_dyn(operation, context);
        let hlo = pliron::op::op_cast::<dyn HloOpInterface>(&*operation_ref)
            .ok_or_else(|| pliron::arg_error_noloc!("unsupported xlahlo operation"))?;
        let operands: Vec<_> = operation.deref(context).operands().collect();
        let [lhs_source, rhs_source] = operands.as_slice() else {
            return Err(pliron::arg_error_noloc!(
                "binary instruction arity is not two"
            ));
        };
        let lhs_is_positive_zero = self.positive_zero_values.contains(lhs_source);
        let rhs_is_positive_zero = self.positive_zero_values.contains(rhs_source);
        let lhs = self.values.get(lhs_source).copied().unwrap_or(*lhs_source);
        let rhs = self.values.get(rhs_source).copied().unwrap_or(*rhs_source);
        let element_type = shape::element_type(
            context,
            operation.deref(context).get_result(0).get_type(context),
        )
        .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
        if hlo.opcode() == Opcode::Add && (lhs_is_positive_zero || rhs_is_positive_zero) {
            let dynamic_value = if lhs_is_positive_zero { rhs } else { lhs };
            if let Some((lowered_operations, lowered_result)) =
                super::arithmetic::make_add_positive_zero(
                    context,
                    lhs,
                    rhs,
                    dynamic_value,
                    element_type,
                )
            {
                for lowered in lowered_operations {
                    rewriter.insert_operation(context, lowered);
                }
                let source_result = operation.deref(context).get_result(0);
                self.values.insert(source_result, lowered_result);
                rewriter.replace_operation_with_values(context, operation, vec![lowered_result]);
                return Ok(());
            }
        }
        let (lowered_operations, lowered_result) =
            make_binary_operations(context, hlo.opcode(), lhs, rhs, element_type)
                .map_err(|error| pliron::arg_error_noloc!("{}", error))?;
        let source_result = operation.deref(context).get_result(0);
        self.values.insert(source_result, lowered_result);
        for lowered in lowered_operations {
            rewriter.insert_operation(context, lowered);
        }
        rewriter.replace_operation_with_values(context, operation, vec![lowered_result]);
        Ok(())
    }
}
