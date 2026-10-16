//! Source-derived shape inference. Unsupported cases are errors, never guessed
//! shapes. This surface is intentionally narrower than the opcode catalog.

use std::collections::BTreeSet;

use crate::{Dimension, Error, Opcode, PrimitiveType, Result, Shape};

mod collectives;
mod indexing;
pub use collectives::{
    infer_all_gather, infer_all_gather_done, infer_all_gather_start, infer_all_reduce,
    infer_all_reduce_done, infer_all_reduce_start, infer_all_to_all, infer_all_to_all_tuple,
    infer_collective_broadcast, infer_collective_permute, infer_collective_permute_done,
    infer_collective_permute_start, infer_collective_reduce, infer_ragged_all_to_all,
    infer_reduce_scatter,
};
pub use indexing::{infer_gather, infer_scatter};
mod convolution;
pub use convolution::{infer_convolution, infer_convolution_kind, infer_convolution_with_sparsity};
mod dot;
pub use dot::{infer_ragged_dot, infer_scaled_dot};
mod reshape;
mod scan;
pub use reshape::{infer_reshape, infer_static_reshape};
pub use scan::infer_scan;
pub(crate) use scan::verify_scan;

pub fn infer_dynamic_reshape(
    operand: &Shape,
    sizes: &[Shape],
    bounds: Vec<Dimension>,
) -> Result<Shape> {
    let operand = operand.array_view()?;
    for size in sizes {
        let size = size.array_view()?;
        if size.element_type() != PrimitiveType::S32 && size.rank()? != 0 {
            return Err(Error::new(
                "shape",
                "dynamic reshape size must be S32 or scalar in pinned inference",
            ));
        }
    }
    let output = Shape::dynamic_array(operand.element_type(), bounds)?.with_default_layout()?;
    if operand.element_count()? != output.element_count()? {
        return Err(Error::new(
            "shape",
            "dynamic reshape input and output bounds have different element counts",
        ));
    }
    Ok(output)
}

pub fn infer_shuffle(
    operand: &Shape,
    dimensions: &[i64],
    mode: &crate::proto::xla_data::ShuffleMode,
) -> Result<Shape> {
    axes(operand.rank()?, dimensions, false)?;
    if dimensions.is_empty() || !mode.has_rotate() || mode.rotate().shifts.len() != dimensions.len()
    {
        return Err(Error::new(
            "shuffle",
            "shuffle requires nonempty dimensions and one rotate shift per dimension",
        ));
    }
    Ok(operand.clone())
}

pub fn infer_select_and_scatter(
    operand: &Shape,
    source: &Shape,
    initial: &Shape,
    window: &crate::proto::xla_data::Window,
    select: &crate::proto::xla_data::ProgramShapeProto,
    scatter: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<Shape> {
    let window_output = infer_window_output(operand, window, operand.element_type())?;
    if !source.compatible_ignoring_fp_precision(&window_output) {
        return Err(Error::new(
            "shape",
            "select-and-scatter source must match window-reduced operand",
        ));
    }
    if select.parameters.len() != 2 {
        return Err(Error::new(
            "callee",
            "select-and-scatter select requires two parameters",
        ));
    }
    let scalar = Shape::scalar(operand.element_type())?;
    for parameter in &select.parameters {
        if !Shape::from_proto(parameter.clone())?.compatible_ignoring_fp_precision(&scalar) {
            return Err(Error::new(
                "callee",
                "select-and-scatter select parameters must match operand scalar type",
            ));
        }
    }
    let result = Shape::from_proto(
        select
            .result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "select computation missing result"))?
            .clone(),
    )?;
    if !result.compatible(&Shape::scalar(PrimitiveType::PRED)?) {
        return Err(Error::new(
            "callee",
            "select-and-scatter select must return scalar predicate",
        ));
    }
    infer_reduce(
        &[Shape::scalar(source.element_type())?, initial.clone()],
        &[],
        scatter,
    )?;
    Ok(operand.clone())
}

pub fn infer_triangular_solve(
    a: &Shape,
    b: &Shape,
    options: &crate::proto::xla_data::TriangularSolveOptions,
) -> Result<Shape> {
    use crate::proto::xla_data::triangular_solve_options::Transpose;
    infer_cholesky(a)?;
    let output = b;
    let a = a.array_view()?;
    let b = b.array_view()?;
    let rank = a.rank()?;
    if b.rank()? != rank || a.element_type() != b.element_type() {
        return Err(Error::new(
            "shape",
            "triangular solve operands must have identical rank and element type",
        ));
    }
    if !compatible_size(
        a.dimensions()[rank - 1],
        b.dimensions()[rank - if options.left_side { 2 } else { 1 }],
    ) || a.dimensions()[..rank - 2] != b.dimensions()[..rank - 2]
    {
        return Err(Error::new(
            "shape",
            "triangular solve shared or batch dimensions differ",
        ));
    }
    match options.transpose_a.enum_value() {
        Ok(Transpose::NO_TRANSPOSE | Transpose::TRANSPOSE | Transpose::ADJOINT) => {
            Ok(output.clone())
        }
        _ => Err(Error::new(
            "transpose",
            "invalid triangular solve transpose option",
        )),
    }
}

pub fn infer_cholesky(a: &Shape) -> Result<Shape> {
    let array = a.array_view()?;
    let rank = array.rank()?;
    if !(array.element_type().is_floating() || array.element_type().is_complex()) || rank < 2 {
        return Err(Error::new(
            "shape",
            "Cholesky requires a floating or complex array of rank at least two",
        ));
    }
    if !compatible_size(array.dimensions()[rank - 2], array.dimensions()[rank - 1]) {
        return Err(Error::new(
            "shape",
            "matrix minor dimensions must be square",
        ));
    }
    Ok(a.clone())
}

fn compatible_size(a: i64, b: i64) -> bool {
    a == b || a == crate::UNBOUNDED_SIZE || b == crate::UNBOUNDED_SIZE
}

pub fn infer_top_k(operand: &Shape, k: i64) -> Result<Shape> {
    let rank = operand.rank()?;
    if rank == 0 || k < 0 || operand.dimensions()[rank - 1] < k {
        return Err(Error::new(
            "top-k",
            "top-k requires a positive-rank array and 0 <= k <= last dimension",
        ));
    }
    let dimensions = (0..rank)
        .map(|axis| {
            if axis == rank - 1 {
                Ok(Dimension::Static(k))
            } else {
                operand.dimension(axis)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let values = Shape::dynamic_array(operand.element_type(), dimensions)?.with_default_layout()?;
    Shape::tuple(vec![
        values.clone(),
        values.with_element_type(PrimitiveType::S32)?,
    ])
}

pub fn infer_batch_norm(opcode: Opcode, operands: &[Shape], feature_index: i64) -> Result<Shape> {
    let count = match opcode {
        Opcode::BatchNormTraining => 3,
        Opcode::BatchNormInference | Opcode::BatchNormGrad => 5,
        _ => return Err(Error::new("opcode", "expected batch normalization opcode")),
    };
    if operands.len() != count {
        return Err(Error::new(
            "arity",
            "incorrect batch normalization operand count",
        ));
    }
    let input = &operands[0];
    let rank = input.rank()?;
    axes(rank, &[feature_index], false)?;
    let feature = feature_index as usize;
    if !input.element_type().is_floating() {
        return Err(Error::new(
            "type-domain",
            "batch normalization input must be floating point",
        ));
    }
    for (index, parameter) in operands.iter().enumerate().skip(1) {
        let parameter_rank = parameter.rank()?;
        if !parameter.element_type().is_floating() {
            return Err(Error::new(
                "type-domain",
                "batch normalization parameters must be floating point",
            ));
        }
        if opcode == Opcode::BatchNormGrad && index == 4 {
            if parameter_rank != rank
                || !input
                    .dimensions()
                    .iter()
                    .zip(parameter.dimensions())
                    .all(|(&a, &b)| compatible_size(a, b))
            {
                return Err(Error::new(
                    "shape",
                    "batch normalization gradient bounds differ from input",
                ));
            }
        } else {
            let requires_vector = opcode != Opcode::BatchNormInference || index <= 2;
            if parameter_rank == 0
                || (requires_vector && parameter_rank != 1)
                || !compatible_size(parameter.dimensions()[0], input.dimensions()[feature])
            {
                return Err(Error::new(
                    "shape",
                    "batch normalization parameter rank or feature count differs",
                ));
            }
        }
    }
    if opcode == Opcode::BatchNormInference {
        return Ok(input.clone());
    }
    let feature = Shape::dynamic_array(input.element_type(), vec![input.dimension(feature)?])?
        .with_default_layout()?;
    Shape::tuple(vec![input.clone(), feature.clone(), feature])
}

pub(crate) fn infer_sort_result(operands: &[Shape]) -> Result<Shape> {
    fn dimensions(shape: &Shape) -> &[i64] {
        if shape.element_type() == PrimitiveType::BUFFER {
            &shape.as_proto().tuple_shapes[0].dimensions
        } else if shape.is_array() {
            shape.dimensions()
        } else {
            &[]
        }
    }
    let first = operands
        .first()
        .ok_or_else(|| Error::new("arity", "sort requires operands"))?;
    for other in &operands[1..] {
        let first = dimensions(first);
        let other = dimensions(other);
        if first.len() != other.len()
            || first
                .iter()
                .zip(other)
                .any(|(&a, &b)| !compatible_size(a, b))
        {
            return Err(Error::new("shape", "sort operand dimensions differ"));
        }
    }
    if operands.len() == 1 {
        Ok(first.clone())
    } else {
        Shape::tuple(operands.to_vec())
    }
}

pub fn infer_sort(
    operands: &[Shape],
    dimension: i64,
    comparator: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<Shape> {
    let first = operands
        .first()
        .ok_or_else(|| Error::new("arity", "sort requires operands"))?;
    // The pinned verifier checks the upper bound only.
    if dimension >= first.rank()? as i64 {
        return Err(Error::new(
            "dimension",
            "sort dimension exceeds operand rank",
        ));
    }
    if comparator.parameters.len()
        != operands
            .len()
            .checked_mul(2)
            .ok_or_else(|| Error::new("overflow", "sort comparator arity overflow"))?
    {
        return Err(Error::new(
            "callee",
            "sort comparator requires two scalar parameters per operand",
        ));
    }
    let result = Shape::from_proto(
        comparator
            .result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "sort comparator missing result"))?
            .clone(),
    )?;
    if !result.compatible(&Shape::scalar(PrimitiveType::PRED)?) {
        return Err(Error::new(
            "callee",
            "sort comparator must return a predicate scalar",
        ));
    }
    let output = infer_sort_result(operands)?;
    for (index, operand) in operands.iter().enumerate() {
        operand.rank()?;
        for parameter in &comparator.parameters[index * 2..index * 2 + 2] {
            let parameter = Shape::from_proto(parameter.clone())?;
            let same_type = parameter.element_type() == operand.element_type()
                || (parameter.element_type().is_floating() && operand.element_type().is_floating());
            if parameter.rank()? != 0 || !same_type {
                return Err(Error::new(
                    "callee",
                    "sort comparator parameter has incorrect scalar type",
                ));
            }
        }
    }
    Ok(output)
}

pub fn infer_fft(
    operand: &Shape,
    kind: crate::proto::xla_data::FftType,
    lengths: &[i64],
) -> Result<Shape> {
    use crate::proto::xla_data::FftType::*;
    if operand.element_type() == PrimitiveType::BUFFER && matches!(kind, FFT | IFFT) {
        infer_fft(&operand.array_view()?, kind, lengths)?;
        return Ok(operand.clone());
    }
    let rank = operand.rank()?;
    if !(1..=3).contains(&lengths.len()) || rank < lengths.len() {
        return Err(Error::new(
            "fft",
            "FFT requires one to three lengths and sufficient input rank",
        ));
    }
    let element = operand.element_type();
    match kind {
        FFT | IFFT => {
            if !element.is_complex() {
                return Err(Error::new("type-domain", "FFT/IFFT requires complex input"));
            }
            Ok(operand.clone())
        }
        RFFT => {
            if !matches!(element, PrimitiveType::F32 | PrimitiveType::F64) {
                return Err(Error::new("type-domain", "RFFT requires F32 or F64 input"));
            }
            for (index, &size) in lengths.iter().enumerate() {
                let actual = operand.dimensions()[rank - lengths.len() + index];
                if actual != crate::UNBOUNDED_SIZE && actual != size {
                    return Err(Error::new(
                        "fft",
                        "RFFT input inner dimensions must match FFT lengths",
                    ));
                }
            }
            let mut wire = operand
                .with_element_type(if element == PrimitiveType::F32 {
                    PrimitiveType::C64
                } else {
                    PrimitiveType::C128
                })?
                .into_proto();
            if lengths[lengths.len() - 1] != 0 {
                wire.dimensions[rank - 1] = lengths[lengths.len() - 1] / 2 + 1;
                wire.is_dynamic_dimension = (0..rank)
                    .map(|axis| {
                        !matches!(operand.dimension(axis), Ok(Dimension::Static(_)))
                            && (axis != rank - 1
                                || operand.dimension(axis) != Ok(Dimension::Unbounded))
                    })
                    .collect();
            }
            Shape::from_proto(wire)
        }
        IRFFT => {
            if !element.is_complex() {
                return Err(Error::new("type-domain", "IRFFT requires complex input"));
            }
            for (index, &size) in lengths[..lengths.len() - 1].iter().enumerate() {
                if operand.dimensions()[rank - lengths.len() + index] != size {
                    return Err(Error::new(
                        "fft",
                        "IRFFT outer transform dimensions must match FFT lengths",
                    ));
                }
            }
            let last = operand.dimensions()[rank - 1];
            let length = lengths[lengths.len() - 1];
            if (last != 0 || length != 0) && last != crate::UNBOUNDED_SIZE && last != length / 2 + 1
            {
                return Err(Error::new(
                    "fft",
                    "IRFFT inner dimension must equal FFT length/2+1",
                ));
            }
            let mut wire = operand
                .with_element_type(if element == PrimitiveType::C64 {
                    PrimitiveType::F32
                } else {
                    PrimitiveType::F64
                })?
                .into_proto();
            wire.dimensions[rank - 1] = length;
            wire.is_dynamic_dimension = (0..rank)
                .map(|axis| {
                    !matches!(operand.dimension(axis), Ok(Dimension::Static(_)))
                        && (axis != rank - 1 || operand.dimension(axis) != Ok(Dimension::Unbounded))
                })
                .collect();
            Shape::from_proto(wire)
        }
    }
}

pub fn infer_dot(
    lhs: &Shape,
    rhs: &Shape,
    dimensions: &crate::proto::xla_data::DotDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
) -> Result<Shape> {
    infer_dot_with_sparsity(lhs, rhs, dimensions, preferred_element, &Default::default())
}

pub fn infer_dot_with_sparsity(
    lhs: &Shape,
    rhs: &Shape,
    dimensions: &crate::proto::xla_data::DotDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
    sparsity: &crate::proto::xla_data::SparsityConfig,
) -> Result<Shape> {
    let lhs = dot::array(lhs)?;
    let rhs = dot::array(rhs)?;
    let lhs_rank = lhs.rank()?;
    let rhs_rank = rhs.rank()?;
    let (left, right) = dot::validate(&lhs, &rhs, dimensions, sparsity)?;
    let mut output = Vec::new();
    for &axis in &dimensions.lhs_batch_dimensions {
        output.push(lhs.dimension(axis as usize)?);
    }
    for axis in 0..lhs_rank {
        if !left.contains(&(axis as i64)) {
            output.push(lhs.dimension(axis)?);
        }
    }
    for axis in 0..rhs_rank {
        if !right.contains(&(axis as i64)) {
            output.push(rhs.dimension(axis)?);
        }
    }
    let element = dot::element(&lhs, &rhs, preferred_element)?;
    Shape::dynamic_array(element, output)?.with_default_layout()
}

pub fn infer_unary(opcode: Opcode, operand: &Shape) -> Result<Shape> {
    use Opcode::*;
    if opcode == Copy || opcode == Domain {
        return Ok(operand.clone());
    }
    operand.rank()?;
    let ty = operand.element_type();
    let accepted = match opcode {
        Floor | Cbrt | Ceil | Erf | RoundNearestAfz | RoundNearestEven | IsFinite => {
            ty.is_floating()
        }
        Asin | Asinh | Acos | Acosh | Atanh | Cosh | Cos | Sinh | Sin | Exp | Exp2 | Expm1
        | Log | Log1p | Log2 | Logistic | Rsqrt | Sqrt | Tan | Tanh | Real | Imag => {
            ty.is_floating() || ty.is_complex()
        }
        Abs | Sign => ty.is_signed_integral() || ty.is_floating() || ty.is_complex(),
        Clz | PopulationCount => ty.is_integral(),
        Negate => ty.is_integral() || ty.is_floating() || ty.is_complex(),
        Not => ty == PrimitiveType::PRED || ty.is_integral(),
        _ => {
            return Err(Error::new(
                "unsupported",
                format!("unary inference for {opcode}"),
            ));
        }
    };
    if !accepted {
        return Err(Error::new(
            "type-domain",
            format!("{opcode} does not accept {ty:?}"),
        ));
    }
    let result_type = match opcode {
        IsFinite => PrimitiveType::PRED,
        Abs | Real | Imag if ty == PrimitiveType::C64 => PrimitiveType::F32,
        Abs | Real | Imag if ty == PrimitiveType::C128 => PrimitiveType::F64,
        _ => ty,
    };
    if result_type == ty {
        Ok(operand.clone())
    } else {
        operand.with_element_type(result_type)
    }
}

/// Binary inference with the upstream floating-point precision promotion rules.
pub fn infer_binary(opcode: Opcode, lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    infer_binary_in_dim(opcode, lhs, rhs, &[])
}

pub fn infer_binary_in_dim(
    opcode: Opcode,
    lhs: &Shape,
    rhs: &Shape,
    broadcast_dimensions: &[i64],
) -> Result<Shape> {
    use Opcode::*;
    lhs.rank()?;
    rhs.rank()?;
    if !lhs
        .element_type()
        .same_ignoring_fp_precision(rhs.element_type())
    {
        return Err(Error::new(
            "type-domain",
            "incompatible binary element types",
        ));
    }
    let ty = lhs.element_type();
    let accepted = match opcode {
        Add | Maximum | Minimum | Multiply | Mulhi | Compare => true,
        Subtract | Power | Divide | Remainder | ShiftLeft | ShiftRightArithmetic
        | ShiftRightLogical => ty != PrimitiveType::PRED,
        Atan2 => ty.is_floating() || ty.is_complex(),
        And | Or | Xor => ty == PrimitiveType::PRED || ty.is_integral(),
        Complex => {
            matches!(ty, PrimitiveType::F32 | PrimitiveType::F64) && ty == rhs.element_type()
        }
        _ => {
            return Err(Error::new(
                "unsupported",
                format!("binary inference for {opcode}"),
            ));
        }
    };
    if !accepted {
        return Err(Error::new(
            "type-domain",
            format!("{opcode} does not accept {ty:?}"),
        ));
    }
    let element = ty.higher_precision(rhs.element_type())?;
    let output = if lhs.rank()? == rhs.rank()? {
        if !broadcast_dimensions.is_empty()
            && broadcast_dimensions != (0..lhs.rank()? as i64).collect::<Vec<_>>()
        {
            return Err(Error::new(
                "broadcast",
                "same-rank broadcast dimensions must be identity",
            ));
        }
        if lhs.dimensions() == rhs.dimensions()
            && !lhs.dimensions().contains(&crate::UNBOUNDED_SIZE)
        {
            let mut wire = lhs.as_proto().clone();
            wire.is_dynamic_dimension = (0..lhs.rank()?)
                .map(|axis| {
                    !matches!(lhs.dimension(axis), Ok(Dimension::Static(_)))
                        || !matches!(rhs.dimension(axis), Ok(Dimension::Static(_)))
                })
                .collect();
            Shape::from_proto(wire)?
        } else {
            degenerate_broadcast(lhs, rhs)?
        }
    } else {
        let (small, large) = if lhs.rank()? < rhs.rank()? {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };
        axes(large.rank()?, broadcast_dimensions, false)?;
        if broadcast_dimensions.len() != small.rank()?
            || broadcast_dimensions.windows(2).any(|d| d[0] >= d[1])
        {
            return Err(Error::new(
                "broadcast",
                "broadcast dimensions must match lower rank in increasing order",
            ));
        }
        let mut wire = large.as_proto().clone();
        wire.is_dynamic_dimension = (0..large.rank()?)
            .map(|axis| !matches!(large.dimension(axis), Ok(Dimension::Static(_))))
            .collect();
        for (axis, &target) in broadcast_dimensions.iter().enumerate() {
            let target = target as usize;
            let small_size = small.dimensions()[axis];
            let large_size = large.dimensions()[target];
            if small_size != large_size && small_size != 1 && large_size != 1 {
                return Err(Error::new("broadcast", "incompatible mapped dimensions"));
            }
            let small_dynamic = !matches!(small.dimension(axis)?, Dimension::Static(_));
            let large_dynamic = !matches!(large.dimension(target)?, Dimension::Static(_));
            if small_dynamic != large_dynamic
                && small_size != large_size
                && (small_size != 1 || small_dynamic)
                && (large_size != 1 || large_dynamic)
            {
                return Err(Error::new("broadcast", "incompatible mapped dynamism"));
            }
            wire.dimensions[target] = small_size;
            wire.is_dynamic_dimension[target] = small_dynamic;
        }
        degenerate_broadcast(&Shape::from_proto(wire)?, large)?
    };
    match opcode {
        Compare => output.with_element_type(PrimitiveType::PRED),
        Complex => output.with_element_type(if ty == PrimitiveType::F32 {
            PrimitiveType::C64
        } else {
            PrimitiveType::C128
        }),
        _ => output.with_element_type(element),
    }
}

fn degenerate_broadcast(lhs: &Shape, rhs: &Shape) -> Result<Shape> {
    let mut dimensions = Vec::new();
    for axis in 0..lhs.rank()? {
        let (a, b) = (lhs.dimensions()[axis], rhs.dimensions()[axis]);
        let dimension = if a == 1 {
            rhs.dimension(axis)?
        } else if b == 1 {
            lhs.dimension(axis)?
        } else if a == b {
            if lhs.dimension(axis)? == Dimension::Unbounded {
                Dimension::Unbounded
            } else if !matches!(lhs.dimension(axis)?, Dimension::Static(_))
                || !matches!(rhs.dimension(axis)?, Dimension::Static(_))
            {
                Dimension::Bounded(a)
            } else {
                Dimension::Static(a)
            }
        } else if a == crate::UNBOUNDED_SIZE {
            rhs.dimension(axis)?
        } else if b == crate::UNBOUNDED_SIZE {
            lhs.dimension(axis)?
        } else {
            return Err(Error::new(
                "broadcast",
                "incompatible equal-rank dimensions",
            ));
        };
        dimensions.push(dimension);
    }
    Shape::dynamic_array(lhs.element_type(), dimensions)?.with_default_layout()
}

pub fn infer_broadcast(operand: &Shape, output: &Shape, dimensions: &[i64]) -> Result<Shape> {
    let (input_rank, output_rank) = (operand.rank()?, output.rank()?);
    if input_rank > output_rank || dimensions.len() != input_rank {
        return Err(Error::new("broadcast", "broadcast rank mismatch"));
    }
    axes(output_rank, dimensions, false)?;
    if dimensions.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::new(
            "broadcast",
            "broadcast dimensions must increase",
        ));
    }
    if output.dimensions().contains(&crate::UNBOUNDED_SIZE) {
        return Err(Error::new(
            "broadcast",
            "unbounded output is not supported by upstream broadcast inference",
        ));
    }
    if operand.element_type() != output.element_type() {
        return Err(Error::new("type-domain", "broadcast element types differ"));
    }
    for (axis, &target) in dimensions.iter().enumerate() {
        let target = target as usize;
        if operand.dimension(axis)? != Dimension::Unbounded {
            if operand.dimensions()[axis] != 1
                && operand.dimensions()[axis] != output.dimensions()[target]
            {
                return Err(Error::new("broadcast", "mapped dimensions differ"));
            }
            if matches!(operand.dimension(axis)?, Dimension::Bounded(_))
                != matches!(output.dimension(target)?, Dimension::Bounded(_))
            {
                return Err(Error::new("broadcast", "mapped dynamism differs"));
            }
        }
    }
    Ok(output.clone())
}

fn validate_indices(rank: usize, indices: &[Shape]) -> Result<()> {
    if indices.len() == 1 && indices[0].array_view()?.rank()? == 1 {
        indices[0].rank()?;
        if !indices[0].element_type().is_integral() || indices[0].dimensions() != [rank as i64] {
            return Err(Error::new(
                "indices",
                "legacy index vector must be integral and match operand rank",
            ));
        }
    } else {
        if indices.len() != rank {
            return Err(Error::new("indices", "index count must match rank"));
        }
        if let Some(first) = indices.first() {
            let array = first.array_view()?;
            if array.rank()? != 0
                || !array.element_type().is_integral()
                || indices.iter().any(|i| !first.compatible(i))
            {
                return Err(Error::new(
                    "indices",
                    "indices must be integral scalars with the same type",
                ));
            }
        }
    }
    Ok(())
}

pub fn infer_dynamic_slice(operand: &Shape, indices: &[Shape], sizes: &[i64]) -> Result<Shape> {
    let rank = operand.rank()?;
    validate_indices(rank, indices)?;
    if sizes.len() != rank {
        return Err(Error::new("slice", "slice sizes must match rank"));
    }
    let mut dimensions = Vec::new();
    for (axis, &size) in sizes.iter().enumerate() {
        if size < 0
            || (operand.dimension(axis)? != Dimension::Unbounded
                && size > operand.dimensions()[axis])
        {
            return Err(Error::new("slice", "slice size exceeds input bound"));
        }
        dimensions.push(
            if !matches!(operand.dimension(axis)?, Dimension::Static(_))
                && size > 1
                && size == operand.dimensions()[axis]
            {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            },
        );
    }
    Shape::dynamic_array(operand.element_type(), dimensions)?.with_default_layout()
}

pub fn infer_dynamic_update_slice(
    operand: &Shape,
    update: &Shape,
    indices: &[Shape],
) -> Result<Shape> {
    update.rank()?;
    if operand.element_type() == PrimitiveType::BUFFER {
        let output = infer_dynamic_update_slice(&operand.array_view()?, update, indices)?;
        let mut wire = operand.clone().into_proto();
        wire.tuple_shapes[0] = output.into_proto();
        return Shape::from_proto(wire);
    }
    let rank = operand.rank()?;
    validate_indices(rank, indices)?;
    if update.rank()? != rank
        || !update
            .element_type()
            .same_ignoring_fp_precision(operand.element_type())
    {
        return Err(Error::new("slice", "update rank or element type mismatch"));
    }
    let mut wire = operand.as_proto().clone();
    wire.is_dynamic_dimension = Vec::with_capacity(rank);
    for axis in 0..rank {
        if operand.dimension(axis)? != Dimension::Unbounded
            && update.dimensions()[axis] > operand.dimensions()[axis]
        {
            return Err(Error::new("slice", "update exceeds input bound"));
        }
        wire.is_dynamic_dimension.push(
            !matches!(operand.dimension(axis)?, Dimension::Static(_))
                || (!matches!(update.dimension(axis)?, Dimension::Static(_))
                    && update.dimensions()[axis] == operand.dimensions()[axis]),
        );
    }
    Shape::from_proto(wire)
}

pub fn infer_reverse(operand: &Shape, dimensions: &[i64]) -> Result<Shape> {
    axes(operand.rank()?, dimensions, false)?;
    Ok(operand.clone())
}

pub fn infer_transpose(operand: &Shape, dimensions: &[i64]) -> Result<Shape> {
    axes(operand.rank()?, dimensions, true)?;
    let mut wire = operand.as_proto().clone();
    wire.dimensions = dimensions
        .iter()
        .map(|&axis| operand.dimensions()[axis as usize])
        .collect();
    wire.is_dynamic_dimension = dimensions
        .iter()
        .map(|&axis| !matches!(operand.dimension(axis as usize), Ok(Dimension::Static(_))))
        .collect();
    if let Some(layout) = wire.layout.as_mut() {
        for axis in &mut layout.minor_to_major {
            *axis = dimensions
                .iter()
                .position(|n| *n == *axis)
                .ok_or_else(|| Error::new("layout", "missing permutation axis"))?
                as i64;
        }
    }
    Shape::from_proto(wire)
}

pub fn infer_slice(
    operand: &Shape,
    starts: &[i64],
    limits: &[i64],
    strides: &[i64],
) -> Result<Shape> {
    let rank = operand.rank()?;
    if starts.len() != rank || limits.len() != rank || strides.len() != rank {
        return Err(Error::new("slice", "slice vectors must match operand rank"));
    }
    let mut output = Vec::with_capacity(rank);
    for axis in 0..rank {
        let (start, limit, stride) = (starts[axis], limits[axis], strides[axis]);
        if start < 0
            || limit < start
            || stride <= 0
            || (operand.dimension(axis)? != Dimension::Unbounded
                && limit > operand.dimensions()[axis])
        {
            return Err(Error::new(
                "slice",
                format!("invalid bounds/stride at axis {axis}"),
            ));
        }
        let extent = limit - start;
        let size = extent / stride + i64::from(extent % stride != 0);
        output.push(
            if size != 1 && matches!(operand.dimension(axis)?, Dimension::Bounded(_)) {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            },
        );
    }
    Shape::dynamic_array(operand.element_type(), output)?.with_default_layout()
}

pub fn infer_convert(operand: &Shape, element: PrimitiveType) -> Result<Shape> {
    operand.with_element_type(element)
}

pub fn infer_reduce_precision(
    operand: &Shape,
    exponent_bits: i32,
    mantissa_bits: i32,
) -> Result<Shape> {
    let array = operand.array_view()?;
    if !array.element_type().is_floating() || exponent_bits < 1 || mantissa_bits < 0 {
        return Err(Error::new(
            "precision",
            "reduce-precision requires floating input, positive exponent bits and nonnegative mantissa bits",
        ));
    }
    Ok(operand.clone())
}

pub fn infer_stochastic_convert(
    operand: &Shape,
    random: &Shape,
    element: PrimitiveType,
) -> Result<Shape> {
    let rank = operand.rank()?;
    if !operand.element_type().is_floating()
        || random.element_type().is_signed_integral()
        || !random.element_type().is_integral()
        || operand.element_type().bit_width() != random.element_type().bit_width()
    {
        return Err(Error::new(
            "type-domain",
            "stochastic-convert requires floating data and equal-width unsigned random values",
        ));
    }
    let same_layout = match (random.layout(), operand.layout()) {
        (None, None) => true,
        (Some(a), Some(b)) => crate::shape::layouts_equal(a, b),
        _ => false,
    };
    if random.rank()? != rank
        || random.dimensions() != operand.dimensions()
        || !same_layout
        || (0..rank).any(|axis| random.dimension(axis) != operand.dimension(axis))
    {
        return Err(Error::new(
            "shape",
            "stochastic-convert input shapes differ",
        ));
    }
    operand.with_element_type(element)
}

pub fn infer_concatenate(operands: &[Shape], dimension: i64) -> Result<Shape> {
    let first = operands
        .first()
        .ok_or_else(|| Error::new("arity", "concatenate requires operands"))?;
    let rank = first.rank()?;
    axes(rank, &[dimension], false)?;
    let mut dimensions = (0..rank)
        .map(|axis| first.dimension(axis))
        .collect::<Result<Vec<_>>>()?;
    let mut element = first.element_type();
    for other in &operands[1..] {
        if other.rank()? != rank {
            return Err(Error::new("shape", "concatenate operand ranks differ"));
        }
        if !other
            .element_type()
            .same_ignoring_fp_precision(first.element_type())
        {
            return Err(Error::new(
                "type-domain",
                "incompatible concatenate element types",
            ));
        }
        // The pinned implementation compares every operand with the first,
        // rather than accumulating the highest precision.
        element = other
            .element_type()
            .higher_precision(first.element_type())?;
        for (axis, current) in dimensions.iter_mut().enumerate() {
            let next = other.dimension(axis)?;
            *current = if axis == dimension as usize {
                match (*current, next) {
                    (Dimension::Unbounded, _) | (_, Dimension::Unbounded) => Dimension::Unbounded,
                    (Dimension::Static(a), Dimension::Static(b)) => {
                        Dimension::Static(a.checked_add(b).ok_or_else(|| {
                            Error::new("overflow", "concatenate dimension overflow")
                        })?)
                    }
                    (
                        Dimension::Static(a) | Dimension::Bounded(a),
                        Dimension::Static(b) | Dimension::Bounded(b),
                    ) => Dimension::Bounded(
                        a.checked_add(b)
                            .ok_or_else(|| Error::new("overflow", "concatenate bound overflow"))?,
                    ),
                }
            } else {
                match (*current, next) {
                    (Dimension::Unbounded, other) | (other, Dimension::Unbounded) => other,
                    (Dimension::Static(a), Dimension::Static(b)) if a == b => Dimension::Static(a),
                    (
                        Dimension::Static(a) | Dimension::Bounded(a),
                        Dimension::Static(b) | Dimension::Bounded(b),
                    ) if a == b => Dimension::Bounded(a),
                    _ => {
                        return Err(Error::new(
                            "dimension",
                            "concatenate non-concatenated dimensions differ",
                        ));
                    }
                }
            };
        }
    }
    Shape::dynamic_array(element, dimensions)?.with_default_layout()
}

pub fn infer_pad(
    operand: &Shape,
    value: &Shape,
    config: &crate::proto::xla_data::PaddingConfig,
) -> Result<Shape> {
    let value = value.array_view()?;
    if operand.rank()? != config.dimensions.len() || value.rank()? != 0 {
        return Err(Error::new(
            "padding",
            "pad requires scalar padding value and one configuration per axis",
        ));
    }
    if !operand
        .element_type()
        .same_ignoring_fp_precision(value.element_type())
    {
        return Err(Error::new("type-domain", "incompatible pad element types"));
    }
    let mut dimensions = Vec::new();
    for (axis, padding) in config.dimensions.iter().enumerate() {
        if padding.interior_padding < 0 {
            return Err(Error::new("padding", "negative interior padding"));
        }
        let input = operand.dimension(axis)?;
        dimensions.push(match input {
            Dimension::Unbounded => Dimension::Unbounded,
            Dimension::Static(size) | Dimension::Bounded(size) => {
                let padded = (size - 1)
                    .max(0)
                    .checked_mul(padding.interior_padding)
                    .and_then(|n| n.checked_add(size))
                    .and_then(|n| n.checked_add(padding.edge_padding_low))
                    .and_then(|n| n.checked_add(padding.edge_padding_high))
                    .ok_or_else(|| Error::new("overflow", "padding result overflow"))?;
                if padded < 0 {
                    return Err(Error::new("padding", "padding results in negative extent"));
                }
                if matches!(input, Dimension::Bounded(_)) {
                    Dimension::Bounded(padded)
                } else {
                    Dimension::Static(padded)
                }
            }
        });
    }
    Shape::dynamic_array(
        operand
            .element_type()
            .higher_precision(value.element_type())?,
        dimensions,
    )?
    .with_default_layout()
}

pub fn infer_clamp(minimum: &Shape, operand: &Shape, maximum: &Shape) -> Result<Shape> {
    operand.rank()?;
    for bound in [minimum, maximum] {
        let scalar = bound.rank()? == 0;
        if !bound
            .element_type()
            .same_ignoring_fp_precision(operand.element_type())
        {
            return Err(Error::new(
                "type-domain",
                "incompatible clamp element types",
            ));
        }
        if !scalar && !bound.compatible_ignoring_fp_precision(operand) {
            return Err(Error::new(
                "shape",
                "clamp bounds must be scalars or match operand shape",
            ));
        }
    }
    if minimum.rank()? != 0
        && maximum.rank()? != 0
        && !minimum.compatible_ignoring_fp_precision(maximum)
    {
        return Err(Error::new("shape", "clamp bound shapes are incompatible"));
    }
    Ok(operand.clone())
}

pub fn infer_select(predicate: &Shape, on_true: &Shape, on_false: &Shape) -> Result<Shape> {
    let scalar = predicate.rank()? == 0;
    on_true.rank()?;
    on_false.rank()?;
    if predicate.element_type() != PrimitiveType::PRED
        || !on_true.compatible_ignoring_fp_precision(on_false)
        || (!scalar
            && (!predicate
                .with_element_type(on_true.element_type())?
                .compatible(on_true)
                || !predicate
                    .with_element_type(on_false.element_type())?
                    .compatible(on_false)))
    {
        return Err(Error::new(
            "shape",
            "select predicate or value shapes do not match",
        ));
    }
    let element = on_true
        .element_type()
        .higher_precision(on_false.element_type())?;
    let mut wire = if scalar {
        on_true.with_element_type(element)?.into_proto()
    } else {
        predicate.with_element_type(element)?.into_proto()
    };
    wire.is_dynamic_dimension = vec![false; on_true.rank()?];
    for axis in 0..on_true.rank()? {
        let left = on_true.dimension(axis)?;
        let right = on_false.dimension(axis)?;
        wire.is_dynamic_dimension[axis] =
            if left == Dimension::Unbounded || right == Dimension::Unbounded {
                wire.dimensions[axis] = if left == Dimension::Unbounded {
                    on_false.dimensions()[axis]
                } else {
                    on_true.dimensions()[axis]
                };
                !matches!(left, Dimension::Static(_)) && !matches!(right, Dimension::Static(_))
            } else {
                (!scalar && !matches!(predicate.dimension(axis), Ok(Dimension::Static(_))))
                    || !matches!(on_true.dimension(axis), Ok(Dimension::Static(_)))
                    || !matches!(on_false.dimension(axis), Ok(Dimension::Static(_)))
            };
    }
    if let Some(layout) = wire.layout.as_mut() {
        layout.element_size_in_bits = on_true
            .layout()
            .map_or(0, |layout| layout.element_size_in_bits);
    }
    Shape::from_proto(wire)
}

pub fn infer_map(
    operands: &[Shape],
    callee: &crate::proto::xla_data::ProgramShapeProto,
    dimensions: &[i64],
) -> Result<Shape> {
    let first = operands
        .first()
        .ok_or_else(|| Error::new("arity", "map requires operands"))?;
    let mut prototype = first.array_view()?;
    let mut prototype_is_buffer = first.element_type() == PrimitiveType::BUFFER;
    for other in &operands[1..] {
        other.rank()?;
        if !prototype_is_buffer
            && other.rank()? == prototype.rank()?
            && other
                .dimensions()
                .iter()
                .zip(prototype.dimensions())
                .all(|(&a, &b)| compatible_size(a, b))
        {
            continue;
        }
        if other
            .element_type()
            .same_ignoring_fp_precision(prototype.element_type())
        {
            if other.rank()? == 0 {
                continue;
            }
            if prototype.rank()? == 0 {
                prototype = other.clone();
                prototype_is_buffer = false;
                continue;
            }
        }
        return Err(Error::new(
            "shape",
            "map operands must have matching dimensions or compatible scalar broadcasts",
        ));
    }
    if dimensions != (0..prototype.rank()? as i64).collect::<Vec<_>>() {
        return Err(Error::new(
            "dimension",
            "map requires all dimensions in increasing order",
        ));
    }
    if operands.len() != callee.parameters.len() {
        return Err(Error::new(
            "callee",
            "map operand count differs from parameter count",
        ));
    }
    for (operand, parameter) in operands.iter().zip(&callee.parameters) {
        let operand = operand.array_view()?;
        let parameter = Shape::from_proto(parameter.clone())?.array_view()?;
        if parameter.rank()? != 0
            || !parameter
                .element_type()
                .same_ignoring_fp_precision(operand.element_type())
        {
            return Err(Error::new(
                "callee",
                "map parameter must be scalar with operand element type",
            ));
        }
    }
    let result = Shape::from_proto(
        callee
            .result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "map computation has no result"))?
            .clone(),
    )?
    .array_view()?;
    if result.rank()? != 0 {
        return Err(Error::new("callee", "map computation must return a scalar"));
    }
    Shape::dynamic_array(
        result.element_type(),
        (0..prototype.rank()?)
            .map(|axis| prototype.dimension(axis))
            .collect::<Result<Vec<_>>>()?,
    )?
    .with_default_layout()
}

pub fn infer_reduce(
    operands: &[Shape],
    dimensions: &[i64],
    reducer: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<Shape> {
    if operands.is_empty() || !operands.len().is_multiple_of(2) {
        return Err(Error::new(
            "arity",
            "reduce requires nonzero even operand count",
        ));
    }
    let count = operands.len() / 2;
    let rank = operands[0].rank()?;
    axes(rank, dimensions, false)?;
    for input in &operands[..count] {
        if input.rank()? != rank || input.dimensions() != operands[0].dimensions() {
            return Err(Error::new("shape", "reduce input dimensions differ"));
        }
    }
    if reducer.parameters.len() != operands.len() {
        return Err(Error::new(
            "callee",
            "reducer requires two parameters per input",
        ));
    }
    let result = Shape::from_proto(
        reducer
            .result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "reducer has no result"))?
            .clone(),
    )?;
    let results = if result.is_tuple() {
        result
            .as_proto()
            .tuple_shapes
            .iter()
            .cloned()
            .map(Shape::from_proto)
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![result.clone()]
    };
    if results.len() != count || (count > 1 && !result.is_tuple()) {
        return Err(Error::new(
            "callee",
            "reducer result cardinality differs from input count",
        ));
    }
    for (index, result) in results.iter().enumerate() {
        let accumulator = Shape::from_proto(reducer.parameters[index].clone())?;
        let input_parameter = Shape::from_proto(reducer.parameters[count + index].clone())?;
        let input = &operands[index];
        let initial = &operands[count + index];
        if result.rank()? != 0
            || !result.compatible(&accumulator)
            || !result.compatible_ignoring_fp_precision(initial)
            || !result.compatible_ignoring_fp_precision(&input_parameter)
            || !Shape::scalar(input.element_type())?
                .compatible_ignoring_fp_precision(&input_parameter)
        {
            return Err(Error::new(
                "callee",
                "reducer scalar, accumulator, input and initial-value types differ",
            ));
        }
    }
    let output_dimensions = (0..rank)
        .filter(|axis| !dimensions.contains(&(*axis as i64)))
        .map(|axis| operands[0].dimension(axis))
        .collect::<Result<Vec<_>>>()?;
    let outputs = results
        .iter()
        .map(|result| {
            Shape::dynamic_array(result.element_type(), output_dimensions.clone())
                .and_then(Shape::with_default_layout)
        })
        .collect::<Result<Vec<_>>>()?;
    if result.is_tuple() {
        Shape::tuple(outputs)
    } else {
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::new("callee", "reduction has no result"))
    }
}

pub fn infer_window_output(
    operand: &Shape,
    window: &crate::proto::xla_data::Window,
    element: PrimitiveType,
) -> Result<Shape> {
    if operand.rank()? != window.dimensions.len() {
        return Err(Error::new(
            "window",
            "window rank differs from operand rank",
        ));
    }
    fn dilate(size: i64, factor: i64) -> Result<i64> {
        if size == 0 {
            return Ok(0);
        }
        (size - 1)
            .checked_mul(factor)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::new("overflow", "dilated window extent overflow"))
    }
    let mut dimensions = Vec::new();
    for (axis, window) in window.dimensions.iter().enumerate() {
        if window.size <= 0
            || window.stride <= 0
            || window.base_dilation <= 0
            || window.window_dilation <= 0
        {
            return Err(Error::new(
                "window",
                "window sizes, strides and dilations must be positive",
            ));
        }
        let input = operand.dimension(axis)?;
        dimensions.push(match input {
            Dimension::Unbounded => Dimension::Unbounded,
            Dimension::Static(size) | Dimension::Bounded(size) => {
                let base = dilate(size, window.base_dilation)?
                    .checked_add(window.padding_low)
                    .and_then(|n| n.checked_add(window.padding_high))
                    .ok_or_else(|| Error::new("overflow", "padded window extent overflow"))?;
                if base < 0 {
                    return Err(Error::new("window", "negative padded input extent"));
                }
                let size = dilate(window.size, window.window_dilation)?;
                let count = if base == 0 || size > base {
                    0
                } else {
                    (base - size) / window.stride + 1
                };
                if matches!(input, Dimension::Bounded(_)) {
                    Dimension::Bounded(count)
                } else {
                    Dimension::Static(count)
                }
            }
        });
    }
    Shape::dynamic_array(element, dimensions)?.with_default_layout()
}

pub fn infer_reduce_window(
    operands: &[Shape],
    window: &crate::proto::xla_data::Window,
    reducer: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<Shape> {
    let signature = infer_reduce(operands, &[], reducer)?;
    let outputs = operands[..operands.len() / 2]
        .iter()
        .zip(&operands[operands.len() / 2..])
        .map(|(input, initial)| infer_window_output(input, window, initial.element_type()))
        .collect::<Result<Vec<_>>>()?;
    if signature.is_tuple() {
        Shape::tuple(outputs)
    } else {
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::new("arity", "reduce-window requires operands"))
    }
}

pub fn infer_call(
    operands: &[Shape],
    callee: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<Shape> {
    if operands.len() != callee.parameters.len() {
        return Err(Error::new(
            "callee",
            "call operand count differs from parameter count",
        ));
    }
    for (operand, parameter) in operands.iter().zip(&callee.parameters) {
        if !operand.compatible(&Shape::from_proto(parameter.clone())?) {
            return Err(Error::new(
                "callee",
                "call operand and parameter shapes differ",
            ));
        }
    }
    Shape::from_proto(
        callee
            .result
            .as_ref()
            .ok_or_else(|| Error::new("callee", "callee has no result"))?
            .clone(),
    )
}

/// While's wire callee ordering is body, then condition.
pub fn infer_while(
    initial: &Shape,
    body: &crate::proto::xla_data::ProgramShapeProto,
    condition: &crate::proto::xla_data::ProgramShapeProto,
) -> Result<Shape> {
    let body_result = infer_call(std::slice::from_ref(initial), body)?;
    let condition_result = infer_call(std::slice::from_ref(initial), condition)?;
    if !condition_result.compatible(&Shape::scalar(PrimitiveType::PRED)?) {
        return Err(Error::new(
            "callee",
            "while condition must return PRED scalar",
        ));
    }
    if !body_result.compatible(initial) {
        return Err(Error::new(
            "callee",
            "while body result and state shapes differ",
        ));
    }
    Ok(initial.clone())
}

pub fn infer_conditional(
    selector: &Shape,
    operands: &[Shape],
    branches: &[crate::proto::xla_data::ProgramShapeProto],
) -> Result<Shape> {
    if selector.rank()? != 0
        || !matches!(
            selector.element_type(),
            PrimitiveType::PRED | PrimitiveType::S32
        )
        || branches.is_empty()
        || (selector.element_type() == PrimitiveType::PRED && branches.len() != 2)
        || operands.len() != branches.len()
    {
        return Err(Error::new(
            "callee",
            "invalid conditional selector or branch cardinality",
        ));
    }
    let mut results = operands
        .iter()
        .zip(branches)
        .map(|(operand, branch)| infer_call(std::slice::from_ref(operand), branch));
    let mut result = results
        .next()
        .ok_or_else(|| Error::new("callee", "conditional has no branches"))??;
    for other in results {
        let other = other?;
        if !result.compatible(&other) {
            return Err(Error::new(
                "callee",
                "conditional branch result shapes differ",
            ));
        }
        fn merge(
            a: &mut crate::proto::xla_data::ShapeProto,
            b: &crate::proto::xla_data::ShapeProto,
        ) {
            if a.element_type.enum_value_or_default() == PrimitiveType::BUFFER {
                return;
            }
            for (a, b) in a.tuple_shapes.iter_mut().zip(&b.tuple_shapes) {
                merge(a, b);
            }
            a.is_dynamic_dimension = (0..a.dimensions.len())
                .map(|axis| {
                    a.is_dynamic_dimension.get(axis).copied().unwrap_or(false)
                        || b.is_dynamic_dimension.get(axis).copied().unwrap_or(false)
                })
                .collect();
        }
        let mut wire = result.into_proto();
        merge(&mut wire, other.as_proto());
        result = Shape::from_proto(wire)?;
    }
    Ok(result)
}

pub fn infer_bitcast_convert(operand: &Shape, element: PrimitiveType) -> Result<Shape> {
    let rank = operand.rank()?;
    if !element.is_array_element() || operand.element_type().is_complex() != element.is_complex() {
        return Err(Error::new(
            "type-domain",
            "bitcast-convert requires array types, both complex or both noncomplex",
        ));
    }
    let width = |ty: PrimitiveType| -> Result<u32> {
        if ty == PrimitiveType::PRED {
            Ok(8)
        } else {
            ty.bit_width()
                .ok_or_else(|| Error::new("type-domain", "missing bit width"))
        }
    };
    let input_bits = width(operand.element_type())?;
    let output_bits = width(element)?;
    if !input_bits
        .max(output_bits)
        .is_multiple_of(input_bits.min(output_bits))
    {
        return Err(Error::new(
            "bitcast",
            "element storage widths are not divisible",
        ));
    }
    let ratio = i64::from(input_bits.max(output_bits) / input_bits.min(output_bits));
    let mut wire = operand.as_proto().clone();
    wire.element_type = element.into();
    wire.is_dynamic_dimension = (0..rank)
        .map(|axis| !matches!(operand.dimension(axis), Ok(Dimension::Static(_))))
        .collect();
    if input_bits != output_bits
        && wire
            .layout
            .as_ref()
            .is_some_and(|layout| !layout.tiles.is_empty() || !layout.split_configs.is_empty())
    {
        return Err(Error::new(
            "unsupported",
            "rank-changing bitcast of tiled/split layout",
        ));
    }
    if input_bits > output_bits {
        wire.dimensions.push(ratio);
        wire.is_dynamic_dimension.push(false);
        if let Some(layout) = wire.layout.as_mut() {
            layout.minor_to_major.insert(0, rank as i64);
        }
    } else if input_bits < output_bits {
        if wire.dimensions.last() != Some(&ratio) {
            return Err(Error::new(
                "bitcast",
                "last input dimension must equal storage-width ratio",
            ));
        }
        wire.dimensions.pop();
        wire.is_dynamic_dimension.pop();
        if let Some(layout) = wire.layout.as_mut() {
            layout
                .minor_to_major
                .retain(|&axis| axis != rank as i64 - 1);
        }
    }
    Shape::from_proto(wire)
}

pub fn infer_get_dimension_size(operand: &Shape, dimension: i64) -> Result<Shape> {
    let operand = operand.array_view()?;
    axes(operand.rank()?, &[dimension], false)?;
    if operand.dimensions()[dimension as usize] > i64::from(i32::MAX) {
        return Err(Error::new(
            "dimension",
            "dimension exceeds upstream S32 size limit",
        ));
    }
    Shape::scalar(PrimitiveType::S32)?.with_default_layout()
}

pub fn infer_set_dimension_size(operand: &Shape, size: &Shape, dimension: i64) -> Result<Shape> {
    infer_get_dimension_size(operand, dimension)?;
    let size = size.array_view()?;
    if size.rank()? != 0 || size.element_type() != PrimitiveType::S32 {
        return Err(Error::new(
            "type-domain",
            "dimension size must be S32 scalar",
        ));
    }
    if operand.element_type() == PrimitiveType::BUFFER {
        let output = infer_set_dimension_size(&operand.array_view()?, &size, dimension)?;
        let mut wire = operand.as_proto().clone();
        wire.tuple_shapes[0] = output.into_proto();
        return Shape::from_proto(wire);
    }
    let mut wire = operand.as_proto().clone();
    wire.is_dynamic_dimension = (0..operand.rank()?)
        .map(|axis| {
            axis == dimension as usize
                || !matches!(operand.dimension(axis), Ok(Dimension::Static(_)))
        })
        .collect();
    Shape::from_proto(wire)
}

pub fn axes(rank: usize, dimensions: &[i64], permutation: bool) -> Result<()> {
    let mut seen = BTreeSet::new();
    for &axis in dimensions {
        if axis < 0 || usize::try_from(axis).map_or(true, |n| n >= rank) || !seen.insert(axis) {
            return Err(Error::new("dimension", "duplicate or out-of-range axis"));
        }
    }
    if permutation && dimensions.len() != rank {
        return Err(Error::new("dimension", "incomplete permutation"));
    }
    Ok(())
}
