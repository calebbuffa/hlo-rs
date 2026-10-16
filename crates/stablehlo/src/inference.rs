//! Type refinement, branch joins and operation-result inference.

use std::fmt;

use crate::verification::tensor_elements_compatible;
use crate::{
    Attribute, QuantizedTensorType, StableHloAttribute, TensorType, Type, TypeError, TypeExtensions,
};

mod all_reduce;
mod all_to_all;
mod batch_norm;
mod bitcast;
mod collective_identity;
mod collective_shape;
mod concatenate;
mod control_flow;
mod convert;
mod convolution;
mod dimension_size;
mod dot;
mod dynamic_slice;
mod elementwise;
mod fft;
mod gather;
mod linear_algebra;
mod map;
mod pad;
mod quantization;
mod reduce;
mod reduce_window;
mod rng;
mod scatter;
mod select_and_scatter;
mod selection;
mod shape_ops;
mod shape_witnesses;
mod slice;
mod sort;
mod transpose;
mod window;
pub use all_reduce::infer_all_reduce_types;
pub use all_to_all::infer_all_to_all_types;
pub(crate) use batch_norm::validate as validate_batch_norm_types;
pub use batch_norm::{
    infer_batch_norm_grad_types, infer_batch_norm_inference_type, infer_batch_norm_training_types,
};
pub use bitcast::infer_bitcast_convert_type;
pub(crate) use bitcast::validate as validate_bitcast_types;
pub(crate) use collective_identity::validate_dynamic_root;
pub use collective_identity::{infer_collective_broadcast_types, infer_collective_permute_type};
pub(crate) use collective_shape::validate_exchange_shape;
pub use collective_shape::{infer_all_gather_types, infer_reduce_scatter_type};
pub use concatenate::infer_concatenate_type;
pub(crate) use control_flow::infer_conditional_types;
pub use control_flow::{
    infer_case_types, infer_get_tuple_element_type, infer_if_types,
    infer_optimization_barrier_types, infer_tuple_type, infer_while_types,
};
pub use convert::infer_convert_type;
pub(crate) use convert::validate as validate_convert_types;
pub use convolution::{ConvolutionOptions, infer_convolution_type};
pub(crate) use convolution::{
    infer_shape as infer_convolution_shape, validate_result as validate_convolution_result,
};
pub use dimension_size::{infer_get_dimension_size_type, infer_set_dimension_size_type};
pub use dot::{infer_dot_general_type, infer_dot_type};
pub use dynamic_slice::{infer_dynamic_slice_type, infer_dynamic_update_slice_type};
pub use elementwise::{
    ComparisonType, infer_abs_type, infer_compare_type, infer_complex_type, infer_imag_type,
    infer_is_finite_type, infer_real_type,
};
pub use fft::{FftType, infer_fft_type};
pub(crate) use gather::validate_result as validate_gather_result;
pub use gather::{infer_dynamic_gather_type, infer_gather_type};
pub use linear_algebra::{infer_cholesky_type, infer_triangular_solve_type};
pub use map::infer_map_type;
pub use pad::infer_pad_type;
pub use quantization::{infer_uniform_dequantize_type, infer_uniform_quantize_type};
pub use reduce::infer_reduce_types;
pub use reduce_window::infer_reduce_window_types;
pub(crate) use rng::validate_rng_state;
pub use rng::{RngAlgorithm, RngDistribution, infer_rng_bit_generator_types, infer_rng_type};
pub use scatter::infer_scatter_types;
pub(crate) use scatter::validate as validate_scatter_shapes;
pub use select_and_scatter::infer_select_and_scatter_type;
pub(crate) use select_and_scatter::validate as validate_select_and_scatter_shapes;
pub use selection::{infer_clamp_type, infer_select_type};
pub use shape_ops::{infer_broadcast_type, infer_reverse_type};
pub(crate) use shape_witnesses::common_dimension as infer_common_dimension;
pub(crate) use shape_witnesses::common_shape as infer_common_shape;
pub use slice::infer_slice_type;
pub use sort::infer_sort_types;
pub(crate) use sort::validate as validate_sort_types;
pub use transpose::infer_transpose_type;

/// Failure to combine StableHLO value types.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceError {
    NonIntegralBitcastRatio {
        input: u64,
        output: u64,
    },
    InvalidBitcastMinorExtent,
    UnrepresentableRank {
        rank: u64,
    },
    InvalidShapeExtent {
        dimension: usize,
        size: i64,
    },
    InvalidGroupSize {
        size: i64,
    },
    InvalidCollectiveExtent {
        dimension: usize,
    },
    EmptyBranches,
    BranchResultCountMismatch {
        branch: usize,
        expected: usize,
        actual: usize,
    },
    TupleIndexOutOfBounds {
        index: i32,
        size: usize,
    },
    InvalidWindowSource {
        dimension: usize,
    },
    InvalidScatterDimensions,
    UpdateCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidSplitCount {
        count: i64,
    },
    NonDivisibleSplit {
        dimension: usize,
        size: u64,
        count: u64,
    },
    InvalidConvolutionDimensions,
    InvalidConvolutionGroups,
    NonPositiveWindowValue {
        attribute: &'static str,
        dimension: usize,
        value: i64,
    },
    InvalidWindowPadding {
        dimension: usize,
    },
    InvalidGatherDimensions,
    EmptyInputs,
    InvalidType {
        index: usize,
        error: TypeError,
    },
    IncompatibleTypes {
        index: usize,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    DimensionMismatch {
        dimension: usize,
        lhs: u64,
        rhs: u64,
    },
    BoundExceeded {
        dimension: usize,
        size: u64,
        bound: u64,
    },
    DimensionOutOfBounds {
        dimension: usize,
        rank: usize,
    },
    DimensionOverflow {
        dimension: usize,
    },
    AttributeLengthMismatch {
        attribute: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidSliceRange {
        dimension: usize,
        start: i64,
        limit: i64,
        stride: i64,
    },
    InvalidSliceSize {
        dimension: usize,
        size: i64,
    },
    SliceSizeExceeded {
        dimension: usize,
        size: u64,
        size_or_bound: u64,
    },
    SliceLimitExceeded {
        dimension: usize,
        limit: i64,
        size_or_bound: u64,
    },
    NegativeInteriorPadding {
        dimension: usize,
        value: i64,
    },
    NegativePaddingResult {
        dimension: usize,
    },
    InvalidPermutation {
        rank: usize,
        permutation: Vec<i64>,
    },
    InvalidDimensions {
        rank: usize,
        dimensions: Vec<i64>,
    },
    AccumulatorCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonScalarAccumulator {
        index: usize,
        rank: usize,
    },
    InvalidFftRank {
        operand_rank: usize,
        fft_rank: usize,
    },
    InvalidFftLength {
        dimension: usize,
        length: i64,
    },
    InvalidBroadcastSize {
        dimension: usize,
        size: i64,
    },
    DimensionPairCountMismatch,
    InvalidDotRank {
        lhs: usize,
        rhs: usize,
    },
    InvalidMatrixRank {
        index: usize,
        rank: usize,
    },
    InvalidSortDimension {
        dimension: i64,
        rank: usize,
    },
    InvalidMapDimensions {
        rank: usize,
        dimensions: Vec<i64>,
    },
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIntegralBitcastRatio { input, output } => write!(
                f,
                "bitcast element widths {input} and {output} do not have an integral ratio"
            ),
            Self::InvalidBitcastMinorExtent => {
                f.write_str("bitcast minor extent cannot satisfy the element-width ratio")
            }
            Self::UnrepresentableRank { rank } => {
                write!(f, "cannot allocate a result shape with rank {rank}")
            }
            Self::InvalidShapeExtent { dimension, size } => {
                write!(
                    f,
                    "shape extent {size} at dimension {dimension} is negative or outside the shape element's range"
                )
            }
            Self::InvalidGroupSize { size } => {
                write!(f, "process-group size must be positive, got {size}")
            }
            Self::InvalidCollectiveExtent { dimension } => write!(
                f,
                "collective dimension {dimension} has no positive extent satisfying its group-size geometry"
            ),
            Self::EmptyBranches => write!(f, "case requires at least one branch"),
            Self::BranchResultCountMismatch {
                branch,
                expected,
                actual,
            } => write!(
                f,
                "branch {branch} returns {actual} values, expected {expected}"
            ),
            Self::TupleIndexOutOfBounds { index, size } => write!(
                f,
                "tuple index {index} is out of bounds for {size} elements"
            ),
            Self::InvalidWindowSource { dimension } => write!(
                f,
                "source extent at dimension {dimension} cannot match the input window geometry"
            ),
            Self::InvalidScatterDimensions => {
                write!(f, "inconsistent scatter dimension maps or update geometry")
            }
            Self::UpdateCountMismatch { expected, actual } => {
                write!(f, "scatter needs {expected} update tensors, got {actual}")
            }
            Self::InvalidSplitCount { count } => {
                write!(f, "all-to-all split_count must be positive, got {count}")
            }
            Self::NonDivisibleSplit {
                dimension,
                size,
                count,
            } => write!(
                f,
                "all-to-all split dimension {dimension} has extent {size}, not divisible by {count}"
            ),
            Self::InvalidConvolutionDimensions => write!(
                f,
                "convolution requires equal ranks of at least two and complete dimension permutations"
            ),
            Self::InvalidConvolutionGroups => write!(
                f,
                "convolution group counts, divisibility or grouped feature extents are inconsistent"
            ),
            Self::NonPositiveWindowValue {
                attribute,
                dimension,
                value,
            } => write!(f, "{attribute}[{dimension}] must be positive, got {value}"),
            Self::InvalidWindowPadding { dimension } => {
                write!(f, "padding values for dimension {dimension} must fit int64")
            }
            Self::InvalidGatherDimensions => {
                write!(f, "inconsistent gather dimension maps or slice sizes")
            }
            Self::InvalidMapDimensions { rank, dimensions } => write!(
                f,
                "map dimensions {dimensions:?} must be all axes 0..{rank} in order"
            ),
            Self::InvalidSortDimension { dimension, rank } => {
                write!(f, "sort dimension {dimension} is outside [-{rank}, {rank})")
            }
            Self::InvalidMatrixRank { index, rank } => write!(
                f,
                "matrix operand {index} must have rank at least 2, got {rank}"
            ),
            Self::EmptyInputs => write!(f, "type inference requires at least one input"),
            Self::InvalidType { index, error } => {
                write!(f, "invalid inference input {index}: {error}")
            }
            Self::IncompatibleTypes { index } => write!(f, "incompatible inference input {index}"),
            Self::RankMismatch { expected, actual } => {
                write!(f, "mismatched tensor ranks: {expected} and {actual}")
            }
            Self::DimensionMismatch {
                dimension,
                lhs,
                rhs,
            } => write!(f, "mismatched dimension {dimension}: {lhs} and {rhs}"),
            Self::BoundExceeded {
                dimension,
                size,
                bound,
            } => write!(f, "dimension {dimension} size {size} exceeds bound {bound}"),
            Self::DimensionOutOfBounds { dimension, rank } => {
                write!(f, "dimension {dimension} is outside tensor rank {rank}")
            }
            Self::DimensionOverflow { dimension } => write!(
                f,
                "inferred size or bound of dimension {dimension} exceeds i64::MAX"
            ),
            Self::AttributeLengthMismatch {
                attribute,
                expected,
                actual,
            } => write!(f, "{attribute} has {actual} elements, expected {expected}"),
            Self::InvalidSliceRange {
                dimension,
                start,
                limit,
                stride,
            } => write!(
                f,
                "invalid slice range {start}:{limit}:{stride} in dimension {dimension}"
            ),
            Self::InvalidSliceSize { dimension, size } => {
                write!(f, "negative slice size {size} in dimension {dimension}")
            }
            Self::SliceSizeExceeded {
                dimension,
                size,
                size_or_bound,
            } => write!(
                f,
                "slice size {size} exceeds size or bound {size_or_bound} in dimension {dimension}"
            ),
            Self::SliceLimitExceeded {
                dimension,
                limit,
                size_or_bound,
            } => write!(
                f,
                "slice limit {limit} exceeds size or bound {size_or_bound} in dimension {dimension}"
            ),
            Self::NegativeInteriorPadding { dimension, value } => write!(
                f,
                "negative interior padding {value} in dimension {dimension}"
            ),
            Self::NegativePaddingResult { dimension } => write!(
                f,
                "padding produces a negative size or bound in dimension {dimension}"
            ),
            Self::InvalidPermutation { rank, permutation } => write!(
                f,
                "{permutation:?} is not a permutation of dimensions 0..{rank}"
            ),
            Self::InvalidDimensions { rank, dimensions } => write!(
                f,
                "{dimensions:?} must contain unique dimensions in 0..{rank}"
            ),
            Self::AccumulatorCountMismatch { expected, actual } => {
                write!(f, "reducer has {actual} accumulators, expected {expected}")
            }
            Self::NonScalarAccumulator { index, rank } => write!(
                f,
                "reducer accumulator {index} has rank {rank}, expected a scalar"
            ),
            Self::InvalidFftRank {
                operand_rank,
                fft_rank,
            } => write!(
                f,
                "FFT rank {fft_rank} must be in 1..=3 and no greater than operand rank {operand_rank}"
            ),
            Self::InvalidFftLength { dimension, length } => write!(
                f,
                "FFT length {length} at index {dimension} must be nonnegative"
            ),
            Self::InvalidBroadcastSize { dimension, size } => write!(
                f,
                "broadcast size {size} in dimension {dimension} must be nonnegative"
            ),
            Self::DimensionPairCountMismatch => write!(
                f,
                "dot batching and contracting dimension pairs must have equal counts"
            ),
            Self::InvalidDotRank { lhs, rhs } => write!(
                f,
                "legacy dot operand ranks must be 1 or 2, got {lhs} and {rhs}"
            ),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidType { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Refines compatible types with static dimensions and the tightest bounds.
///
/// Tensor element metadata comes from the first input, including quantization
/// parameters; compatibility does not require identical scales or zero points.
/// Tuples are combined recursively. Invalid types and contradictory static
/// dimensions or bounds are rejected.
pub fn infer_most_specific_type(types: &[Type]) -> Result<Type, InferenceError> {
    infer(types, Specificity::Most)
}

/// Joins branch types, retaining only shared static dimensions and upper bounds.
///
/// Unlike a general shape union, StableHLO rejects unequal static dimensions.
/// A static dimension joined with a bounded dynamic one becomes dynamic, with
/// the larger of that size and the bound.
pub fn infer_least_specific_type(types: &[Type]) -> Result<Type, InferenceError> {
    infer(types, Specificity::Least)
}

#[derive(Clone, Copy)]
enum Specificity {
    Most,
    Least,
}

fn infer(types: &[Type], specificity: Specificity) -> Result<Type, InferenceError> {
    let Some(first) = types.first() else {
        return Err(InferenceError::EmptyInputs);
    };
    for (index, ty) in types.iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
    }
    if let Type::Tuple(elements) = first {
        let mut outputs = Vec::with_capacity(elements.len());
        for (index, ty) in types.iter().enumerate() {
            if !matches!(ty, Type::Tuple(other) if other.len() == elements.len()) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
        }
        for index in 0..elements.len() {
            let elements = types
                .iter()
                .map(|ty| {
                    let Type::Tuple(elements) = ty else {
                        unreachable!("checked tuple inputs");
                    };
                    elements[index].clone()
                })
                .collect::<Vec<_>>();
            outputs.push(infer(&elements, specificity)?);
        }
        return Ok(Type::Tuple(outputs));
    }
    if let Some(unranked) = types
        .iter()
        .find(|ty| matches!(ty, Type::UnrankedTensor(_)))
    {
        for (index, ty) in types.iter().enumerate() {
            if !tensor_elements_compatible(unranked, ty) {
                return Err(InferenceError::IncompatibleTypes { index });
            }
        }
        if matches!(specificity, Specificity::Most) {
            let ranked = types
                .iter()
                .filter(|ty| dimensions(ty).is_some())
                .cloned()
                .collect::<Vec<_>>();
            if !ranked.is_empty() {
                return infer(&ranked, specificity);
            }
        }
        return Ok(unranked.clone());
    }
    let Some(shape) = dimensions(first) else {
        for (index, ty) in types.iter().enumerate() {
            if ty != first {
                return Err(InferenceError::IncompatibleTypes { index });
            }
        }
        return Ok(first.clone());
    };
    let mut sizes = shape.to_vec();
    let mut output_bounds = (0..shape.len())
        .map(|index| bound(first, index))
        .collect::<Vec<_>>();
    let any_bounds = types.iter().any(|ty| bounds(ty).is_some());
    for (index, ty) in types.iter().enumerate().skip(1) {
        let Some(shape) = dimensions(ty) else {
            return Err(InferenceError::IncompatibleTypes { index });
        };
        if !tensor_elements_compatible(first, ty) {
            return Err(InferenceError::IncompatibleTypes { index });
        }
        if shape.len() != sizes.len() {
            return Err(InferenceError::RankMismatch {
                expected: sizes.len(),
                actual: shape.len(),
            });
        }
        for (dimension, (&rhs, lhs)) in shape.iter().zip(&mut sizes).enumerate() {
            let (size, upper) = merge_dimension(
                dimension,
                *lhs,
                rhs,
                output_bounds[dimension],
                bound(ty, dimension),
                specificity,
            )?;
            *lhs = size;
            output_bounds[dimension] = upper;
        }
    }
    let encoding = if any_bounds {
        bounds_encoding(output_bounds)
    } else {
        encoding(first).cloned()
    };
    let result = with_shape_and_encoding(first, sizes, encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 0, error })?;
    Ok(result)
}

fn bounds_encoding(output_bounds: Vec<Option<u64>>) -> Option<Attribute> {
    output_bounds.iter().any(Option::is_some).then(|| {
        Attribute::StableHlo(Box::new(StableHloAttribute::TypeExtensions(
            TypeExtensions {
                bounds: output_bounds
                    .into_iter()
                    .map(|bound| bound.map_or(-1, |bound| bound as i64))
                    .collect(),
            },
        )))
    })
}

fn with_shape_and_encoding(
    first: &Type,
    sizes: Vec<Option<u64>>,
    encoding: Option<Attribute>,
) -> Type {
    match first {
        Type::Tensor(tensor) => {
            let mut tensor = TensorType::new(tensor.element(), sizes);
            if let Some(encoding) = encoding {
                tensor = tensor.with_encoding(encoding);
            }
            Type::Tensor(tensor)
        }
        Type::QuantizedTensor(tensor) => {
            let mut tensor = QuantizedTensorType::from_element(sizes, tensor.element().clone());
            if let Some(encoding) = encoding {
                tensor = tensor.with_encoding(encoding);
            }
            Type::QuantizedTensor(tensor)
        }
        _ => unreachable!("checked tensor type"),
    }
}

fn merge_dimension(
    dimension: usize,
    lhs: Option<u64>,
    rhs: Option<u64>,
    left_bound: Option<u64>,
    right_bound: Option<u64>,
    specificity: Specificity,
) -> Result<(Option<u64>, Option<u64>), InferenceError> {
    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
        return if lhs == rhs {
            Ok((Some(lhs), None))
        } else {
            Err(InferenceError::DimensionMismatch {
                dimension,
                lhs,
                rhs,
            })
        };
    }
    match specificity {
        Specificity::Most => {
            if let Some(size) = lhs.or(rhs) {
                for bound in [left_bound, right_bound].into_iter().flatten() {
                    if size > bound {
                        return Err(InferenceError::BoundExceeded {
                            dimension,
                            size,
                            bound,
                        });
                    }
                }
                Ok((Some(size), None))
            } else {
                Ok((
                    None,
                    match (left_bound, right_bound) {
                        (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
                        _ => left_bound.or(right_bound),
                    },
                ))
            }
        }
        Specificity::Least => Ok((
            None,
            match (lhs.or(left_bound), rhs.or(right_bound)) {
                (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
                _ => None,
            },
        )),
    }
}

pub(crate) fn dimensions(ty: &Type) -> Option<&[Option<u64>]> {
    match ty {
        Type::Tensor(tensor) => Some(tensor.dimensions()),
        Type::QuantizedTensor(tensor) => Some(tensor.dimensions()),
        _ => None,
    }
}

fn encoding(ty: &Type) -> Option<&Attribute> {
    match ty {
        Type::Tensor(tensor) => tensor.encoding(),
        Type::QuantizedTensor(tensor) => tensor.encoding(),
        _ => None,
    }
}

fn bounds(ty: &Type) -> Option<&[i64]> {
    match encoding(ty) {
        Some(Attribute::StableHlo(attribute)) => match attribute.as_ref() {
            StableHloAttribute::TypeExtensions(extensions) => Some(&extensions.bounds),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn bound(ty: &Type, dimension: usize) -> Option<u64> {
    bounds(ty)?
        .get(dimension)
        .and_then(|&bound| u64::try_from(bound).ok())
}
