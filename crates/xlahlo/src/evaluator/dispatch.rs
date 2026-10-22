//! Native opcode dispatch over the evaluator's existing request-local session.

use super::*;

impl Session<'_> {
    pub(super) fn evaluate_value(
        &mut self,
        frame: &mut Frame,
        value: Value,
        path: &[usize],
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        let Some(operation) = value.defining_op() else {
            return if mode == DependencyMode::Recursive {
                EvaluatedValue::unknown(
                    self.ctx,
                    shape::subshape(self.ctx, value.get_type(self.ctx), path)?,
                )
            } else {
                Err(failure(EvaluationError::ShallowDependency))
            };
        };
        let op = opcode(self.ctx, operation)?;
        self.pure(operation)?;
        if let Some(controls) =
            optional_attr::<ControlPredecessors>(self.ctx, operation, "hlo_control_predecessors")?
        {
            let names = controls.0.clone();
            for name in names {
                let mut predecessor = None;
                for candidate in frame.block.deref(self.ctx).iter(self.ctx) {
                    if optional_attr::<InstructionName>(self.ctx, candidate, "hlo_name")?
                        .is_some_and(|id| id.0 == name)
                    {
                        predecessor = Some(candidate);
                        break;
                    }
                }
                let predecessor =
                    predecessor.ok_or_else(|| malformed("unresolved control predecessor"))?;
                let predecessor_value = predecessor.deref(self.ctx).get_result(0);
                self.eval(frame, predecessor_value, &[], mode)?;
            }
        }
        let operands: Vec<_> = operation.deref(self.ctx).operands().collect();
        let target = value.get_type(self.ctx);
        use Opcode::*;
        match op {
            Constant => {
                let literal = attr::<Literal>(self.ctx, operation, "hlo_literal")?;
                Ok(EvaluatedValue::known(self.ctx, literal)?
                    .subvalue(path)?
                    .clone())
            }
            Parameter => {
                if mode == DependencyMode::Shallow {
                    Err(failure(EvaluationError::ShallowDependency))
                } else {
                    EvaluatedValue::unknown(self.ctx, shape::subshape(self.ctx, target, path)?)
                }
            }
            Tuple => {
                if let Some((&index, tail)) = path.split_first() {
                    let input = operands
                        .get(index)
                        .copied()
                        .ok_or_else(|| malformed("tuple path is out of range"))?;
                    self.operand(frame, input, tail, mode)
                } else {
                    let mut children = Vec::with_capacity(operands.len());
                    for input in operands {
                        children.push(self.operand(frame, input, &[], mode)?);
                    }
                    EvaluatedValue::tuple(self.ctx, target, children)
                }
            }
            GetTupleElement => {
                let index = attr::<Index>(self.ctx, operation, "hlo_index")?.0 as usize;
                let mut input_path = vec![index];
                input_path.extend_from_slice(path);
                self.operand(frame, operands[0], &input_path, mode)
            }
            Copy | Domain | OptimizationBarrier => self.operand(frame, operands[0], path, mode),
            AfterAll => {
                for input in operands {
                    self.operand(frame, input, &[], mode)?;
                }
                EvaluatedValue::known(self.ctx, Literal::token(self.ctx))
            }
            AddDependency => {
                self.operand(frame, operands[1], &[], mode)?;
                self.operand(frame, operands[0], path, mode)
            }
            Call | Fusion => {
                let mut args = Vec::with_capacity(operands.len());
                for input in operands {
                    args.push(self.operand(frame, input, &[], mode)?);
                }
                if op == Call {
                    let names = attr::<Callees>(self.ctx, operation, "hlo_callees")?;
                    if names.0.len() != 1 {
                        return Err(malformed("Call needs exactly one callee"));
                    }
                    let callee = self.resolve(&names.0[0])?;
                    self.computation(callee, &args, path)
                } else {
                    self.enter(operation, &args, path)
                }
            }
            Conditional => {
                let selector = self.operand(frame, operands[0], &[], mode)?;
                if !selector.is_known() {
                    return EvaluatedValue::unknown(
                        self.ctx,
                        shape::subshape(self.ctx, target, path)?,
                    );
                }
                let names = attr::<Callees>(self.ctx, operation, "hlo_callees")?;
                if names.0.is_empty() || operands.len() != names.0.len() + 1 {
                    return Err(malformed("Conditional needs one argument per branch"));
                }
                let selector = selector.into_literal(self.ctx)?;
                let selected = if shape::element_type(self.ctx, selector.shape)?
                    == crate::types::ElementType::Pred
                {
                    if literal_semantics::scalar_predicate(self.ctx, &selector)? {
                        0
                    } else {
                        1
                    }
                } else {
                    let index = literal_semantics::scalar_integer(self.ctx, &selector)?;
                    usize::try_from(index)
                        .ok()
                        .filter(|&index| index < names.0.len())
                        .unwrap_or(names.0.len() - 1)
                };
                let name = names
                    .0
                    .get(selected)
                    .ok_or_else(|| malformed("predicate Conditional has no selected branch"))?;
                let argument = self.operand(frame, operands[1 + selected], &[], mode)?;
                let callee = self.resolve(name)?;
                self.computation(callee, &[argument], path)
            }
            While => self.evaluate_while(frame, operation, path, mode),
            Pad | Select => {
                let mut values = Vec::with_capacity(operands.len());
                for input in operands {
                    values.push(self.operand(frame, input, &[], mode)?);
                }
                if values.iter().any(|input| !input.is_known()) {
                    return EvaluatedValue::unknown(self.ctx, target);
                }
                if shape::is_dynamic(self.ctx, target)
                    || values
                        .iter()
                        .any(|input| shape::is_dynamic(self.ctx, input.shape()))
                {
                    return Err(unsupported_value(
                        op,
                        "dynamic raw movement runtime-size policy is not implemented",
                    ));
                }
                let element = shape::element_type(self.ctx, target)?;
                for input in &values[usize::from(op == Select)..] {
                    if shape::element_type(self.ctx, input.shape())? != element {
                        return Err(unsupported_value(
                            op,
                            "mixed-precision raw movement conversion is not implemented",
                        ));
                    }
                }
                let values = values
                    .into_iter()
                    .map(|value| value.into_literal(self.ctx))
                    .collect::<Result<Vec<_>>>()?;
                let literal = if op == Pad {
                    let config = attr::<crate::attributes::PaddingConfig>(
                        self.ctx,
                        operation,
                        "hlo_config",
                    )?;
                    movement::pad(self.ctx, &values[0], &values[1], &config)?
                } else {
                    movement::select(self.ctx, &values[0], &values[1], &values[2])?
                };
                EvaluatedValue::known(self.ctx, literal)
            }
            Concatenate => {
                let mut values = Vec::with_capacity(operands.len());
                for input in operands {
                    values.push(self.operand(frame, input, &[], mode)?);
                }
                if values.iter().any(|input| !input.is_known()) {
                    return EvaluatedValue::unknown(self.ctx, target);
                }
                if shape::is_dynamic(self.ctx, target)
                    || values
                        .iter()
                        .any(|input| shape::is_dynamic(self.ctx, input.shape()))
                {
                    return Err(unsupported_value(
                        op,
                        "dynamic concatenate is not implemented",
                    ));
                }
                let values = values
                    .into_iter()
                    .map(|value| value.into_literal(self.ctx))
                    .collect::<Result<Vec<_>>>()?;
                let inputs = values.iter().collect::<Vec<_>>();
                let axis = usize::try_from(attr::<Index>(self.ctx, operation, "hlo_index")?.0)
                    .map_err(|_| malformed("concatenate axis exceeds host addressing"))?;
                EvaluatedValue::known(self.ctx, movement::concatenate(self.ctx, &inputs, axis)?)
            }
            Reshape | Transpose | Broadcast | Reverse | Slice => {
                let input = self.operand(frame, operands[0], &[], mode)?;
                if !input.is_known() {
                    return EvaluatedValue::unknown(self.ctx, target);
                }
                let input = input.into_literal(self.ctx)?;
                if matches!(op, Reverse | Slice) && shape::is_dynamic(self.ctx, input.shape) {
                    return Err(unsupported_value(
                        op,
                        "dynamic indexed movement runtime-size policy is not implemented",
                    ));
                }
                let literal = match op {
                    Reshape => {
                        let dimensions = shape::dimensions(self.ctx, target)?.to_vec();
                        movement::reshape(self.ctx, &input, &dimensions)?
                    }
                    Transpose | Broadcast => {
                        let axes = attr::<Axes>(self.ctx, operation, "hlo_dimensions")?.0;
                        let axes = axes
                            .into_iter()
                            .map(|axis| {
                                usize::try_from(axis)
                                    .map_err(|_| malformed("negative movement axis"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        if op == Transpose {
                            movement::transpose(self.ctx, &input, &axes)?
                        } else {
                            movement::broadcast(self.ctx, &input, target, &axes)?
                        }
                    }
                    Reverse => {
                        let axes = attr::<Axes>(self.ctx, operation, "hlo_dimensions")?;
                        movement::reverse(self.ctx, &input, &axes)?
                    }
                    Slice => {
                        let geometry = attr::<SliceGeometry>(self.ctx, operation, "hlo_geometry")?;
                        movement::slice(self.ctx, &input, &geometry)?
                    }
                    _ => return Err(malformed("unexpected movement dispatch")),
                };
                EvaluatedValue::known(self.ctx, literal)
            }
            Bitcast => self.bitcast(frame, &operands, target, mode),
            DynamicSlice | DynamicUpdateSlice => {
                self.dynamic_indexed(frame, operation, op, &operands, target, mode)
            }
            Reduce | ReduceWindow if !path.is_empty() => {
                Ok(self.eval(frame, value, &[], mode)?.subvalue(path)?.clone())
            }
            Reduce => self.reduce(frame, operation, &operands, target, mode),
            ReduceWindow => self.reduce_window(frame, operation, &operands, target, mode),
            Abs | Add | And | Clamp | Clz | Compare | Convert | BitcastConvert | Divide
            | Maximum | Minimum | Mulhi | Multiply | Negate | Not | Or | PopulationCount
            | Power | Remainder | ShiftLeft | ShiftRightArithmetic | ShiftRightLogical | Sign
            | Subtract | Xor | Ceil | Floor | Imag | IsFinite | Real | RoundNearestAfz
            | Complex | RoundNearestEven => {
                let mut values = Vec::with_capacity(operands.len());
                for input in operands {
                    values.push(self.operand(frame, input, &[], mode)?);
                }
                if values.iter().any(|input| !input.is_known()) {
                    return EvaluatedValue::unknown(self.ctx, target);
                }
                let values = values
                    .into_iter()
                    .map(|value| value.into_literal(self.ctx))
                    .collect::<Result<Vec<_>>>()?;
                EvaluatedValue::known(self.ctx, self.elementwise(operation, op, target, &values)?)
            }
            Acos
            | Acosh
            | AllGather
            | AllGatherDone
            | AllGatherStart
            | AllReduce
            | AllReduceDone
            | AllReduceStart
            | AllToAll
            | Asin
            | Asinh
            | AsyncDone
            | AsyncStart
            | AsyncUpdate
            | Atan2
            | Atanh
            | BatchNormGrad
            | BatchNormInference
            | BatchNormTraining
            | Cbrt
            | Cholesky
            | CollectiveBroadcast
            | CollectivePermute
            | CollectivePermuteDone
            | CollectivePermuteStart
            | CollectiveReduce
            | Convolution
            | CopyDone
            | CopyStart
            | Cos
            | Cosh
            | CustomCall
            | Dot
            | DynamicReshape
            | Erf
            | Exp
            | Exp2
            | Expm1
            | Fft
            | Gather
            | GetDimensionSize
            | Infeed
            | Iota
            | Log
            | Log1p
            | Log2
            | Logistic
            | Map
            | Outfeed
            | PartitionId
            | RaggedAllToAll
            | RaggedDot
            | Recv
            | RecvDone
            | ReducePrecision
            | ReduceScatter
            | ReplicaId
            | Rng
            | RngBitGenerator
            | RngGetAndUpdateState
            | Rsqrt
            | ScaledDot
            | Scan
            | Scatter
            | SelectAndScatter
            | Send
            | SendDone
            | SetDimensionSize
            | Shuffle
            | Sin
            | Sinh
            | Sort
            | Sqrt
            | StochasticConvert
            | Tan
            | Tanh
            | TopK
            | TriangularSolve => Err(failure(EvaluationError::Unsupported { opcode: op })),
        }
    }
}
