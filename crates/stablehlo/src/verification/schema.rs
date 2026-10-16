//! Operand and result constraints from StablehloOps.td and Base.td.

use super::{BuildError, compatible_tensor_types};
use crate::{Attribute, ElementType, Instruction, StableHloAttribute, StableHloOp, Type, ValueId};
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Constraint {
    Tensor,
    TensorPerAxis,
    NonQuantized,
    Float,
    Float32Or64,
    FloatComplex,
    FloatQuantized,
    FloatComplexQuantized,
    SignedFloatQuantized,
    SignedFloatComplexQuantized,
    IntFloatComplexQuantized,
    PredInt,
    Int,
    Complex,
    Pred,
    I32,
    Ui32,
    StaticDimension,
    Dimension,
    StaticInt1D,
    StaticInt2D,
    ScalarInt,
    Scalar,
    FloatQuantized1D,
    FloatOrQuantizedPerAxis,
    QuantizedPerAxis,
    ScalarPredIntFloat,
    PredIntFloat,
    IntFloat,
    StaticIntFloat,
    StaticTensorPerAxis,
    StaticNumericQuantized,
    StaticOrBoundedTensorPerAxis,
    TensorToken,
    ControlValue,
    StaticControlValue,
    WhileValue,
    CompositeValue,
    TupleValue,
    CustomValue,
    Tuple,
    Token,
    Future,
    ScatterIndex,
}

pub(super) fn is_scalar_integer(ty: &Type) -> bool {
    Constraint::ScalarInt.accepts(ty)
}

#[derive(Clone, Copy)]
enum Group {
    One(Constraint),
    Many(Constraint),
}

struct Signature {
    operands: &'static [Group],
    results: &'static [Group],
}

use Constraint::*;
use Group::{Many, One};

impl StableHloOp {
    pub(crate) fn accepts_variadic_operand_type(self, group: usize, ty: &Type) -> bool {
        matches!(self.signature().operands.get(group), Some(Many(constraint)) if constraint.accepts(ty))
    }

    pub(crate) fn accepts_fixed_result_type(self, index: usize, ty: &Type) -> bool {
        matches!(self.signature().results.get(index), Some(One(constraint)) if constraint.accepts(ty))
    }

    pub(crate) fn accepts_variadic_result_type(self, group: usize, ty: &Type) -> bool {
        matches!(self.signature().results.get(group), Some(Many(constraint)) if constraint.accepts(ty))
    }

    pub(crate) fn accepts_fixed_operand_type(self, index: usize, ty: &Type) -> bool {
        matches!(self.signature().operands.get(index), Some(One(constraint)) if constraint.accepts(ty))
    }

    pub(crate) fn fixed_operand_arity(self) -> Option<usize> {
        let groups = self.signature().operands;
        groups
            .iter()
            .all(|group| matches!(group, One(_)))
            .then_some(groups.len())
    }

    pub(crate) fn fixed_result_arity(self) -> Option<usize> {
        let groups = self.signature().results;
        groups
            .iter()
            .all(|group| matches!(group, One(_)))
            .then_some(groups.len())
    }

    fn signature(self) -> Signature {
        let (operands, results): (&[Group], &[Group]) = match self {
            Self::Constant => (&[], &[One(StaticTensorPerAxis)]),
            Self::Iota => (&[], &[One(StaticNumericQuantized)]),
            Self::DynamicIota => (&[One(StaticDimension)], &[One(Tensor)]),
            Self::CreateToken => (&[], &[One(Token)]),
            Self::Abs => (
                &[One(SignedFloatComplexQuantized)],
                &[One(SignedFloatQuantized)],
            ),
            Self::Cbrt
            | Self::Cosine
            | Self::Exponential
            | Self::ExponentialMinusOne
            | Self::Log
            | Self::LogPlusOne
            | Self::Logistic
            | Self::Rsqrt
            | Self::Sine
            | Self::Sqrt
            | Self::Tan
            | Self::Tanh => (&[One(FloatComplexQuantized)], &[One(FloatComplexQuantized)]),
            Self::Ceil
            | Self::Floor
            | Self::RoundNearestAfz
            | Self::RoundNearestEven
            | Self::ReducePrecision => (&[One(FloatQuantized)], &[One(FloatQuantized)]),
            Self::Convert => (&[One(NonQuantized)], &[One(NonQuantized)]),
            Self::CountLeadingZeros | Self::PopulationCount => (&[One(Int)], &[One(Int)]),
            Self::Imag | Self::Real => (&[One(FloatComplex)], &[One(Float)]),
            Self::IsFinite => (&[One(FloatQuantized)], &[One(Pred)]),
            Self::Not => (&[One(PredInt)], &[One(PredInt)]),
            Self::Negate => (
                &[One(IntFloatComplexQuantized)],
                &[One(IntFloatComplexQuantized)],
            ),
            Self::Sign => (
                &[One(SignedFloatComplexQuantized)],
                &[One(SignedFloatComplexQuantized)],
            ),
            Self::Add => (
                &[One(TensorPerAxis), One(TensorPerAxis)],
                &[One(TensorPerAxis)],
            ),
            Self::Atan2 => (
                &[One(FloatComplexQuantized), One(FloatComplexQuantized)],
                &[One(FloatComplexQuantized)],
            ),
            Self::Complex => (&[One(Float32Or64), One(Float32Or64)], &[One(Complex)]),
            Self::Divide | Self::Power | Self::Remainder | Self::Subtract => (
                &[One(IntFloatComplexQuantized), One(IntFloatComplexQuantized)],
                &[One(IntFloatComplexQuantized)],
            ),
            Self::Maximum | Self::Minimum | Self::Multiply => {
                (&[One(Tensor), One(Tensor)], &[One(Tensor)])
            }
            Self::ShiftLeft | Self::ShiftRightArithmetic | Self::ShiftRightLogical => {
                (&[One(Int), One(Int)], &[One(Int)])
            }
            Self::And | Self::Or | Self::Xor => (&[One(PredInt), One(PredInt)], &[One(PredInt)]),
            Self::Infeed | Self::Recv => (&[One(Token)], &[Many(StaticControlValue)]),
            Self::Outfeed | Self::Send => (&[Many(TensorPerAxis), One(Token)], &[One(Token)]),
            Self::ReplicaId | Self::PartitionId => (&[], &[One(Ui32)]),
            Self::AfterAll => (&[Many(Token)], &[One(Token)]),
            Self::If => (&[One(Pred)], &[Many(ControlValue)]),
            Self::Case => (&[One(I32)], &[Many(ControlValue)]),
            Self::While => (&[Many(WhileValue)], &[Many(WhileValue)]),
            Self::AllGather
            | Self::AllReduce
            | Self::AllToAll
            | Self::CollectiveBroadcast
            | Self::CollectiveReduce
            | Self::Sort => (&[Many(Tensor)], &[Many(Tensor)]),
            Self::ReduceScatter => (&[One(Tensor)], &[One(Tensor)]),
            Self::Reduce | Self::ReduceWindow => (&[Many(Tensor), Many(Tensor)], &[Many(Tensor)]),
            Self::AsyncStart => (&[Many(Tensor)], &[One(Future)]),
            Self::AsyncDone => (&[One(Future)], &[One(Tensor)]),
            Self::GetTupleElement => (&[One(Tuple)], &[One(TupleValue)]),
            Self::Tuple => (&[Many(TupleValue)], &[One(Tuple)]),
            Self::Compare => (&[One(Tensor), One(Tensor)], &[One(Pred)]),
            Self::Slice
            | Self::Reverse
            | Self::CrossReplicaSum
            | Self::CollectivePermute
            | Self::UnaryEinsum => (&[One(Tensor)], &[One(Tensor)]),
            Self::DynamicSlice => (&[One(Tensor), Many(ScalarInt)], &[One(Tensor)]),
            Self::DynamicUpdateSlice => {
                (&[One(Tensor), One(Tensor), Many(ScalarInt)], &[One(Tensor)])
            }
            Self::BatchNormGrad => (
                &[
                    One(FloatQuantized),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                    One(FloatQuantized),
                ],
                &[
                    One(FloatQuantized),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                ],
            ),
            Self::BatchNormInference => (
                &[
                    One(FloatQuantized),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                ],
                &[One(FloatQuantized)],
            ),
            Self::BatchNormTraining => (
                &[
                    One(FloatQuantized),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                ],
                &[
                    One(FloatQuantized),
                    One(FloatQuantized1D),
                    One(FloatQuantized1D),
                ],
            ),
            Self::BitcastConvert | Self::Transpose => {
                (&[One(TensorPerAxis)], &[One(TensorPerAxis)])
            }
            Self::Broadcast => (&[One(Tensor)], &[One(TensorPerAxis)]),
            Self::BroadcastInDim | Self::Reshape => {
                (&[One(TensorPerAxis)], &[One(StaticOrBoundedTensorPerAxis)])
            }
            Self::DynamicBroadcastInDim | Self::DynamicReshape => (
                &[One(TensorPerAxis), One(StaticDimension)],
                &[One(TensorPerAxis)],
            ),
            Self::Cholesky => (&[One(FloatComplexQuantized)], &[One(FloatComplexQuantized)]),
            Self::Clamp => (&[One(Tensor), One(Tensor), One(Tensor)], &[One(Tensor)]),
            Self::Concatenate | Self::Map => (&[Many(Tensor)], &[One(Tensor)]),
            Self::Composite => (&[Many(CompositeValue)], &[Many(CompositeValue)]),
            Self::Convolution | Self::Dot | Self::DotGeneral => {
                (&[One(Tensor), One(TensorPerAxis)], &[One(TensorPerAxis)])
            }
            Self::CustomCall => (&[Many(CustomValue)], &[Many(CustomValue)]),
            Self::Einsum | Self::TorchIndexSelect => (&[One(Tensor), One(Tensor)], &[One(Tensor)]),
            Self::Fft => (&[One(FloatComplex)], &[One(FloatComplex)]),
            Self::Gather => (&[One(Tensor), One(Int)], &[One(Tensor)]),
            Self::GetDimensionSize => (&[One(TensorPerAxis)], &[One(I32)]),
            Self::Scatter => (
                &[Many(Tensor), One(ScatterIndex), Many(Tensor)],
                &[Many(Tensor)],
            ),
            Self::Select => (&[One(Pred), One(Tensor), One(Tensor)], &[One(Tensor)]),
            Self::SelectAndScatter => (&[One(Tensor), One(Tensor), One(Tensor)], &[One(Tensor)]),
            Self::SetDimensionSize => (&[One(Tensor), One(I32)], &[One(Tensor)]),
            Self::Pad => (&[One(Tensor), One(Scalar)], &[One(Tensor)]),
            Self::TriangularSolve => (
                &[One(FloatComplex), One(FloatComplex)],
                &[One(FloatComplex)],
            ),
            Self::Return => (&[Many(WhileValue)], &[]),
            Self::OptimizationBarrier => (&[Many(TensorToken)], &[Many(TensorToken)]),
            Self::Rng => (
                &[
                    One(ScalarPredIntFloat),
                    One(ScalarPredIntFloat),
                    One(StaticDimension),
                ],
                &[One(PredIntFloat)],
            ),
            Self::RngBitGenerator => (&[One(IntFloat)], &[One(IntFloat), One(StaticIntFloat)]),
            Self::UniformQuantize => (&[One(FloatOrQuantizedPerAxis)], &[One(QuantizedPerAxis)]),
            Self::UniformDequantize => (&[One(QuantizedPerAxis)], &[One(Float)]),
            Self::RealDynamicSlice => (
                &[One(Tensor), One(Dimension), One(Dimension), One(Dimension)],
                &[One(Tensor)],
            ),
            Self::DynamicPad => (
                &[
                    One(Tensor),
                    One(Scalar),
                    One(StaticDimension),
                    One(StaticDimension),
                    One(StaticDimension),
                ],
                &[One(Tensor)],
            ),
            Self::DynamicGather => (&[One(Tensor), One(Int), One(StaticInt1D)], &[One(Tensor)]),
            Self::DynamicConv => (
                &[One(Tensor), One(Tensor), One(StaticInt2D)],
                &[One(Tensor)],
            ),
        };
        Signature { operands, results }
    }
}

impl Constraint {
    fn accepts(self, ty: &Type) -> bool {
        match self {
            Token => return *ty == Type::Token,
            Future => return matches!(ty, Type::Future(_)),
            Tuple => {
                return matches!(ty, Type::Tuple(elements) if elements.iter().all(|ty| TupleValue.accepts(ty)));
            }
            TensorToken => return Tensor.accepts(ty) || *ty == Type::Token,
            ControlValue => return TensorPerAxis.accepts(ty) || *ty == Type::Token,
            StaticControlValue => return StaticTensorPerAxis.accepts(ty) || *ty == Type::Token,
            WhileValue => {
                return ControlValue.accepts(ty)
                    || matches!(ty, Type::Buffer(buffer) if hlo_element(buffer.element()));
            }
            CompositeValue => return ControlValue.accepts(ty) || Tuple.accepts(ty),
            TupleValue => return WhileValue.accepts(ty) || Tuple.accepts(ty),
            CustomValue => {
                return TensorPerAxis.accepts(ty)
                    || (matches!(ty, Type::UnrankedTensor(_))
                        && (ty.plain_tensor_element().is_some_and(hlo_element)
                            || hlo_quantized(ty)))
                    || matches!(ty, Type::Token | Type::Future(_))
                    || matches!(ty, Type::Buffer(buffer) if hlo_element(buffer.element()))
                    || matches!(ty, Type::Tuple(elements) if elements.iter().all(|ty|
                    !matches!(ty, Type::Future(_)) && CustomValue.accepts(ty)));
            }
            _ => {}
        }
        let (shape, element, quantization) = match ty {
            Type::Tensor(tensor) => (tensor.dimensions(), tensor.element(), None),
            Type::QuantizedTensor(tensor) => (
                tensor.dimensions(),
                tensor.storage_type(),
                Some(tensor.quantization_dimension()),
            ),
            _ => return false,
        };
        let static_shape = shape.iter().all(Option::is_some);
        let integer = hlo_int(element);
        let float = element.is_float() && element != ElementType::Tf32;
        let complex = matches!(element, ElementType::C64 | ElementType::C128);
        let pred = element == ElementType::I1;
        let per_tensor = quantization == Some(None);
        if quantization.is_some() && !hlo_quantized(ty) {
            return false;
        }
        let tensor = quantization.is_none() && (integer || float || complex || pred) || per_tensor;
        let tensor_per_axis = tensor || quantization.is_some();
        match self {
            Tensor => tensor,
            TensorPerAxis => tensor_per_axis,
            NonQuantized => tensor && quantization.is_none(),
            Float => quantization.is_none() && float,
            Float32Or64 => {
                quantization.is_none() && matches!(element, ElementType::F32 | ElementType::F64)
            }
            FloatComplex => quantization.is_none() && (float || complex),
            FloatQuantized | FloatQuantized1D => {
                ((quantization.is_none() && float) || per_tensor)
                    && (!matches!(self, FloatQuantized1D) || shape.len() == 1)
            }
            FloatComplexQuantized => (quantization.is_none() && (float || complex)) || per_tensor,
            FloatOrQuantizedPerAxis => (quantization.is_none() && float) || quantization.is_some(),
            QuantizedPerAxis => quantization.is_some(),
            SignedFloatQuantized => {
                (quantization.is_none() && (hlo_signed(element) || float)) || per_tensor
            }
            SignedFloatComplexQuantized => {
                (quantization.is_none() && (hlo_signed(element) || float || complex)) || per_tensor
            }
            IntFloatComplexQuantized | StaticNumericQuantized => {
                ((quantization.is_none() && (integer || float || complex)) || per_tensor)
                    && (!matches!(self, StaticNumericQuantized) || static_shape)
            }
            PredInt => quantization.is_none() && (pred || integer),
            Int => quantization.is_none() && integer,
            Complex => quantization.is_none() && complex,
            Pred => quantization.is_none() && pred,
            I32 => quantization.is_none() && element == ElementType::I32,
            Ui32 => quantization.is_none() && element == ElementType::Ui32,
            Dimension | StaticDimension => {
                quantization.is_none()
                    && (element == ElementType::Index || integer)
                    && shape.len() == 1
                    && (!matches!(self, StaticDimension) || static_shape)
            }
            StaticInt1D | StaticInt2D => {
                quantization.is_none()
                    && integer
                    && static_shape
                    && shape.len() == if matches!(self, StaticInt1D) { 1 } else { 2 }
            }
            ScalarInt => quantization.is_none() && integer && shape.is_empty(),
            Scalar => tensor && shape.is_empty(),
            ScalarPredIntFloat | PredIntFloat => {
                quantization.is_none()
                    && (pred || integer || float)
                    && (!matches!(self, ScalarPredIntFloat) || shape.is_empty())
            }
            IntFloat | StaticIntFloat => {
                quantization.is_none()
                    && (integer || float)
                    && (!matches!(self, StaticIntFloat) || static_shape)
            }
            StaticTensorPerAxis => tensor_per_axis && static_shape,
            StaticOrBoundedTensorPerAxis => {
                tensor_per_axis && (static_shape || single_bounded_dimension(ty, shape))
            }
            ScatterIndex => {
                quantization.is_none() && (element.is_integer() || element == ElementType::Index)
            }
            Token | Future | Tuple | TensorToken | ControlValue | StaticControlValue
            | WhileValue | CompositeValue | TupleValue | CustomValue => unreachable!(),
        }
    }
}

fn hlo_signed(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::I2
            | ElementType::I4
            | ElementType::I8
            | ElementType::I16
            | ElementType::I32
            | ElementType::I64
    )
}

fn hlo_int(element: ElementType) -> bool {
    hlo_signed(element)
        || matches!(
            element,
            ElementType::Ui2
                | ElementType::Ui4
                | ElementType::Ui8
                | ElementType::Ui16
                | ElementType::Ui32
                | ElementType::Ui64
        )
}

fn hlo_element(element: ElementType) -> bool {
    hlo_int(element)
        || element == ElementType::I1
        || (element.is_float() && element != ElementType::Tf32)
        || matches!(element, ElementType::C64 | ElementType::C128)
}

fn hlo_quantized(ty: &Type) -> bool {
    matches!(ty.quantized_tensor_element(), Some(tensor)
        if matches!(tensor.storage_type(), ElementType::I2 | ElementType::I4
            | ElementType::I8 | ElementType::I16 | ElementType::I32
            | ElementType::Ui2 | ElementType::Ui4 | ElementType::Ui8
            | ElementType::Ui16 | ElementType::Ui32)
        && tensor.expressed_type().is_float()
        && tensor.expressed_type() != ElementType::Tf32)
}

fn single_bounded_dimension(ty: &Type, shape: &[Option<u64>]) -> bool {
    let encoding = match ty {
        Type::Tensor(tensor) => tensor.encoding(),
        Type::QuantizedTensor(tensor) => tensor.encoding(),
        _ => None,
    };
    let Some(Attribute::StableHlo(attribute)) = encoding else {
        return false;
    };
    let StableHloAttribute::TypeExtensions(extensions) = attribute.as_ref() else {
        return false;
    };
    let mut dynamic = shape.iter().enumerate().filter(|(_, size)| size.is_none());
    let Some((index, _)) = dynamic.next() else {
        return false;
    };
    dynamic.next().is_none()
        && extensions
            .bounds
            .get(index)
            .is_some_and(|bound| *bound >= 0)
}

fn constraint_at(groups: &[Group], index: usize, count: usize) -> Option<Constraint> {
    let fixed = groups
        .iter()
        .filter(|group| matches!(group, One(_)))
        .count();
    let variadic = groups.len() - fixed;
    let remaining = count.checked_sub(fixed)?;
    if (variadic == 0 && remaining != 0) || (variadic > 0 && !remaining.is_multiple_of(variadic)) {
        return None;
    }
    let group_size = remaining.checked_div(variadic).unwrap_or(0);
    let mut offset = 0;
    for group in groups {
        let (size, constraint) = match group {
            One(constraint) => (1, *constraint),
            Many(constraint) => (group_size, *constraint),
        };
        if index >= offset && index - offset < size {
            return Some(constraint);
        }
        offset += size;
    }
    None
}

pub(super) fn verify(
    instruction: &Instruction,
    opcode: StableHloOp,
    available: &HashMap<ValueId, Type>,
) -> Result<(), BuildError> {
    let signature = opcode.signature();
    for (index, operand) in instruction.operands.iter().enumerate() {
        if !constraint_at(signature.operands, index, instruction.operands.len())
            .is_some_and(|constraint| constraint.accepts(&available[operand]))
        {
            return Err(BuildError::InvalidOperandType {
                operation: opcode.mnemonic(),
                value: *operand,
            });
        }
    }
    for (index, result) in instruction.results.iter().enumerate() {
        if !constraint_at(signature.results, index, instruction.results.len())
            .is_some_and(|constraint| constraint.accepts(result.ty()))
        {
            return Err(BuildError::InvalidResultType {
                operation: opcode.mnemonic(),
                result: result.value(),
            });
        }
    }
    if matches!(
        opcode,
        StableHloOp::And | StableHloOp::Or | StableHloOp::Xor
    ) {
        let first = available.get(&instruction.operands[0]);
        if !compatible_tensor_types(first, available.get(&instruction.operands[1])) {
            return Err(BuildError::IncompatibleOperands {
                operation: opcode.mnemonic(),
                lhs: instruction.operands[0],
                rhs: instruction.operands[1],
            });
        }
        if !compatible_tensor_types(first, Some(instruction.results[0].ty())) {
            return Err(BuildError::InvalidResultType {
                operation: opcode.mnemonic(),
                result: instruction.results[0].value(),
            });
        }
    }
    Ok(())
}
