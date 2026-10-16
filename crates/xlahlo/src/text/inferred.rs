use super::*;
use crate::inference;

pub(super) fn can_omit_shape(opcode: Opcode) -> bool {
    use Opcode::*;
    matches!(
        opcode,
        Abs | Acos
            | Acosh
            | Asin
            | Asinh
            | Add
            | AddDependency
            | AfterAll
            | Atan2
            | Atanh
            | BatchNormGrad
            | BatchNormInference
            | BatchNormTraining
            | Broadcast
            | Call
            | Ceil
            | Cholesky
            | Clamp
            | Clz
            | Compare
            | Complex
            | Concatenate
            | Conditional
            | Convolution
            | Copy
            | Cos
            | Cosh
            | OptimizationBarrier
            | Divide
            | Domain
            | Dot
            | Erf
            | Exp
            | Exp2
            | Expm1
            | Fft
            | Floor
            | Gather
            | GetDimensionSize
            | SetDimensionSize
            | GetTupleElement
            | Imag
            | IsFinite
            | Log
            | Log1p
            | Log2
            | Logistic
            | And
            | Not
            | Or
            | Xor
            | Map
            | Maximum
            | Minimum
            | Multiply
            | Mulhi
            | Negate
            | Pad
            | PartitionId
            | PopulationCount
            | Power
            | RaggedDot
            | Real
            | Reduce
            | Remainder
            | ReplicaId
            | Reverse
            | RoundNearestAfz
            | RoundNearestEven
            | Rsqrt
            | Scan
            | ScaledDot
            | Scatter
            | Select
            | ShiftLeft
            | ShiftRightArithmetic
            | ShiftRightLogical
            | Shuffle
            | Sign
            | Sin
            | Sinh
            | Sqrt
            | Cbrt
            | ReduceWindow
            | SelectAndScatter
            | Sort
            | Subtract
            | Tan
            | Tanh
            | Transpose
            | TriangularSolve
            | Tuple
            | While
            | TopK
    )
}

pub(super) fn explicit(shape: &Option<Shape>) -> Result<&Shape> {
    shape
        .as_ref()
        .ok_or_else(|| Error::new("shape", "instruction requires an explicit result shape"))
}

pub(super) fn infer(
    instruction: &HloInstructionProto,
    instructions: &[HloInstructionProto],
    callees: &[ProgramShapeProto],
) -> Result<Shape> {
    use Opcode::*;
    let opcode: Opcode = instruction.opcode.parse()?;
    let operands = instruction
        .operand_ids
        .iter()
        .map(|id| {
            let shape = instructions
                .iter()
                .find(|operand| operand.id == *id)
                .and_then(|operand| operand.shape.as_ref())
                .ok_or_else(|| {
                    Error::new(
                        "shape",
                        "inferred result requires a previously shaped operand",
                    )
                })?;
            Shape::from_proto(shape.clone())
        })
        .collect::<Result<Vec<_>>>()?;
    if opcode.arity().is_some_and(|arity| operands.len() != arity) {
        return Err(Error::new(
            "arity",
            format!("wrong operand count for inferred {opcode}"),
        ));
    }
    let operand = |index: usize| {
        operands
            .get(index)
            .ok_or_else(|| Error::new("arity", "missing inference operand"))
    };
    let dimension = || {
        let [dimension] = instruction.dimensions.as_slice() else {
            return Err(Error::new("dimension", "inference requires one dimension"));
        };
        Ok(*dimension)
    };
    let callee = |index: usize| {
        callees
            .get(index)
            .ok_or_else(|| Error::new("callee", "missing result-inference computation"))
    };
    match opcode {
        Abs | Acos | Acosh | Asin | Asinh | Atanh | Ceil | Cbrt | Clz | Copy | Cos | Cosh
        | Domain | Erf | Exp | Exp2 | Expm1 | Floor | Imag | IsFinite | Log | Log1p | Log2
        | Logistic | Negate | Not | PopulationCount | Real | RoundNearestAfz | RoundNearestEven
        | Rsqrt | Sign | Sin | Sinh | Sqrt | Tan | Tanh => {
            inference::infer_unary(opcode, operand(0)?)
        }
        Add | And | Atan2 | Compare | Complex | Divide | Maximum | Minimum | Multiply | Mulhi
        | Or | Power | Remainder | ShiftLeft | ShiftRightArithmetic | ShiftRightLogical
        | Subtract | Xor => inference::infer_binary(opcode, operand(0)?, operand(1)?),
        Tuple => Shape::tuple(operands),
        GetTupleElement => {
            if !operand(0)?.is_tuple() {
                return Err(Error::new(
                    "shape",
                    "get-tuple-element requires a tuple operand",
                ));
            }
            operand(0)?.subshape(&[usize::try_from(instruction.tuple_index)
                .map_err(|_| Error::new("index", "negative tuple index"))?])
        }
        AfterAll => {
            if operands
                .iter()
                .any(|shape| shape.element_type() != PrimitiveType::TOKEN)
            {
                return Err(Error::new("shape", "after-all requires token operands"));
            }
            Ok(Shape::token())
        }
        AddDependency => {
            if operand(1)?.element_type() != PrimitiveType::TOKEN {
                return Err(Error::new(
                    "shape",
                    "add-dependency requires a token dependency",
                ));
            }
            Ok(operand(0)?.clone())
        }
        ReplicaId | PartitionId => Shape::scalar(PrimitiveType::U32)?.with_default_layout(),
        Clamp => inference::infer_clamp(operand(0)?, operand(1)?, operand(2)?),
        Select => inference::infer_select(operand(0)?, operand(1)?, operand(2)?),
        Concatenate => inference::infer_concatenate(&operands, dimension()?),
        Broadcast => {
            let input = operand(0)?;
            if !input.is_array() || input.dimensions().contains(&crate::UNBOUNDED_SIZE) {
                return Err(Error::new(
                    "shape",
                    "broadcast inference requires a bounded array",
                ));
            }
            let mut dimensions = instruction
                .dimensions
                .iter()
                .copied()
                .map(Dimension::Static)
                .collect::<Vec<_>>();
            for axis in 0..input.rank()? {
                dimensions.push(input.dimension(axis)?);
            }
            Shape::dynamic_array(input.element_type(), dimensions)?.with_default_layout()
        }
        Reverse => inference::infer_reverse(operand(0)?, &instruction.dimensions),
        Transpose => inference::infer_transpose(operand(0)?, &instruction.dimensions),
        Cholesky => inference::infer_cholesky(operand(0)?),
        TriangularSolve => inference::infer_triangular_solve(
            operand(0)?,
            operand(1)?,
            instruction.triangular_solve_options.get_or_default(),
        ),
        Fft => inference::infer_fft(
            operand(0)?,
            instruction
                .fft_type
                .enum_value()
                .map_err(|_| Error::new("fft", "unknown FFT type"))?,
            &instruction.fft_length,
        ),
        TopK => inference::infer_top_k(operand(0)?, instruction.k),
        Pad => inference::infer_pad(
            operand(0)?,
            operand(1)?,
            instruction.padding_config.get_or_default(),
        ),
        GetDimensionSize => inference::infer_get_dimension_size(operand(0)?, dimension()?),
        SetDimensionSize => {
            inference::infer_set_dimension_size(operand(0)?, operand(1)?, dimension()?)
        }
        BatchNormGrad | BatchNormInference | BatchNormTraining => {
            inference::infer_batch_norm(opcode, &operands, instruction.feature_index)
        }
        Dot => inference::infer_dot_with_sparsity(
            operand(0)?,
            operand(1)?,
            instruction.dot_dimension_numbers.get_or_default(),
            None,
            instruction.sparsity_config.get_or_default(),
        ),
        RaggedDot => inference::infer_ragged_dot(
            operand(0)?,
            operand(1)?,
            operand(2)?,
            instruction.ragged_dot_dimension_numbers.get_or_default(),
            None,
        ),
        // The pinned parser infers from operands 0/2, unlike the verifier's 0/1.
        ScaledDot => inference::infer_dot(
            operand(0)?,
            operand(2)?,
            instruction.dot_dimension_numbers.get_or_default(),
            None,
        ),
        Convolution => inference::infer_convolution_with_sparsity(
            operand(0)?,
            operand(1)?,
            instruction.feature_group_count,
            instruction.batch_group_count,
            instruction.window.get_or_default(),
            instruction.convolution_dimension_numbers.get_or_default(),
            None,
            instruction.sparsity_config.get_or_default(),
        ),
        Gather => inference::infer_gather(
            operand(0)?,
            operand(1)?,
            instruction.gather_dimension_numbers.get_or_default(),
            &instruction.gather_slice_sizes,
        ),
        Shuffle => inference::infer_shuffle(
            operand(0)?,
            &instruction.dimensions,
            instruction
                .shuffle_mode
                .as_ref()
                .ok_or_else(|| Error::new("shuffle", "missing shuffle mode"))?,
        ),
        Call => inference::infer_call(&operands, callee(0)?),
        While => inference::infer_while(operand(0)?, callee(0)?, callee(1)?),
        Conditional => inference::infer_conditional(operand(0)?, &operands[1..], callees),
        Map => inference::infer_map(&operands, callee(0)?, &instruction.dimensions),
        Reduce => inference::infer_reduce(&operands, &instruction.dimensions, callee(0)?),
        ReduceWindow => inference::infer_reduce_window(
            &operands,
            instruction.window.get_or_default(),
            callee(0)?,
        ),
        SelectAndScatter => inference::infer_select_and_scatter(
            operand(0)?,
            operand(1)?,
            operand(2)?,
            instruction.window.get_or_default(),
            callee(0)?,
            callee(1)?,
        ),
        Scatter => inference::infer_scatter(
            &operands,
            instruction.scatter_dimension_numbers.get_or_default(),
            callee(0)?,
        ),
        Sort => {
            dimension()?;
            inference::infer_sort_result(&operands)
        }
        Scan => scan_result(&operands, callee(0)?, instruction.num_carries, dimension()?),
        _ => Err(Error::new(
            "unsupported",
            format!("omitted result-shape inference for {opcode}"),
        )),
    }
}

fn scan_result(
    operands: &[Shape],
    body: &ProgramShapeProto,
    carries: i64,
    axis: i64,
) -> Result<Shape> {
    let carries =
        usize::try_from(carries).map_err(|_| Error::new("scan", "negative carry count"))?;
    if carries >= operands.len() || operands.len() != body.parameters.len() {
        return Err(Error::new(
            "scan",
            "inferred scan requires inputs and matching body arity",
        ));
    }
    let axis = usize::try_from(axis).map_err(|_| Error::new("dimension", "negative scan axis"))?;
    let input = &operands[0];
    input.dimension(axis)?;
    let size = input.dimensions()[axis];
    let root = Shape::from_proto(
        body.result
            .as_ref()
            .ok_or_else(|| Error::new("scan", "missing body result"))?
            .clone(),
    )?;
    let insert = |shape: Shape| -> Result<Shape> {
        let rank = shape.rank()?;
        if axis > rank {
            return Err(Error::new("dimension", "scan axis exceeds result rank"));
        }
        let mut dimensions = shape.dimensions().to_vec();
        dimensions.insert(axis, size);
        let mut result = Shape::array(shape.element_type(), dimensions)?;
        if let Some(layout) = shape.layout() {
            let mut order = Vec::new();
            if axis == rank {
                order.push(axis as i64);
            }
            for &dimension in &layout.minor_to_major {
                order.push(if dimension >= axis as i64 {
                    dimension + 1
                } else {
                    dimension
                });
                if dimension == axis as i64 {
                    order.push(axis as i64);
                }
            }
            result = result.with_layout(LayoutProto {
                minor_to_major: order,
                tail_padding_alignment_in_elements: 1,
                ..Default::default()
            })?;
        }
        Ok(result)
    };
    // Native parser insertion resets dynamic flags and retains only minor order.
    if !root.is_tuple() {
        return if carries > 0 { Ok(root) } else { insert(root) };
    }
    let count = root.as_proto().tuple_shapes.len();
    Shape::tuple(
        (0..count)
            .map(|index| {
                let shape = root.subshape(&[index])?;
                if index
                    .checked_add(carries)
                    .is_some_and(|index| index < count)
                {
                    insert(shape)
                } else {
                    Ok(shape)
                }
            })
            .collect::<Result<Vec<_>>>()?,
    )
}
