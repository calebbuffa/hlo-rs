use super::{
    InferenceError, Specificity, bound, dimensions, merge_dimension, window::output_shape,
    with_shape_and_encoding,
};
use crate::{ConvDimensionNumbers, StableHloOp, Type};

/// Convolution shape attributes. Omitted window vectors use unit strides and
/// dilations, zero padding and no reversal, as in StableHLO.
#[derive(Clone, Copy, Debug)]
pub struct ConvolutionOptions<'a> {
    pub dimension_numbers: &'a ConvDimensionNumbers,
    pub window_strides: Option<&'a [i64]>,
    pub padding: Option<&'a [[i64; 2]]>,
    pub lhs_dilation: Option<&'a [i64]>,
    pub rhs_dilation: Option<&'a [i64]>,
    pub window_reversal: Option<&'a [bool]>,
    pub feature_group_count: i64,
    pub batch_group_count: i64,
}

impl<'a> ConvolutionOptions<'a> {
    pub fn new(dimension_numbers: &'a ConvDimensionNumbers) -> Self {
        Self {
            dimension_numbers,
            window_strides: None,
            padding: None,
            lhs_dilation: None,
            rhs_dilation: None,
            window_reversal: None,
            feature_group_count: 1,
            batch_group_count: 1,
        }
    }
}

/// Infers convolution dimensions without propagating operand encodings.
///
/// Follows `inferConvolutionOp` shape inference. The result template selects
/// elements and quantization metadata, not shape or encoding. Precision and
/// quantization relationships remain operation-verifier responsibilities.
pub fn infer_convolution_type(
    lhs: &Type,
    rhs: &Type,
    result_template: &Type,
    options: ConvolutionOptions<'_>,
) -> Result<Type, InferenceError> {
    for (index, ty) in [lhs, rhs, result_template].into_iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        let accepts = if index == 2 {
            StableHloOp::Convolution.accepts_fixed_result_type(0, ty)
        } else {
            StableHloOp::Convolution.accepts_fixed_operand_type(index, ty)
        };
        if !accepts {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    if let (Type::Tensor(a), Type::Tensor(b)) = (lhs, rhs)
        && a.element() != b.element()
        && !(a.element().is_float()
            && a.element().bit_width() == Some(8)
            && b.element().is_float()
            && b.element().bit_width() == Some(8))
    {
        return Err(InferenceError::IncompatibleTypes { index: 1 });
    }
    let rank = dimensions(lhs).expect("checked operand domain").len();
    let padding = options.padding.map_or_else(
        || vec![Some([0, 0]); rank.saturating_sub(2)],
        |values| {
            values
                .iter()
                .map(|p| Some([i128::from(p[0]), i128::from(p[1])]))
                .collect()
        },
    );
    let shape = infer_shape(lhs, rhs, options, &padding)?;
    let result = with_shape_and_encoding(result_template, shape, None);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 2, error })?;
    Ok(result)
}

fn permutation(
    first: i64,
    second: i64,
    spatial: &[i64],
    rank: usize,
) -> Result<Vec<usize>, InferenceError> {
    if spatial.len() + 2 != rank {
        return Err(InferenceError::InvalidConvolutionDimensions);
    }
    let values = [first, second]
        .into_iter()
        .chain(spatial.iter().copied())
        .collect::<Vec<_>>();
    let mut seen = vec![false; rank];
    let mut axes = Vec::with_capacity(rank);
    for value in values {
        let axis =
            usize::try_from(value).map_err(|_| InferenceError::InvalidConvolutionDimensions)?;
        let slot = seen
            .get_mut(axis)
            .ok_or(InferenceError::InvalidConvolutionDimensions)?;
        if std::mem::replace(slot, true) {
            return Err(InferenceError::InvalidConvolutionDimensions);
        }
        axes.push(axis);
    }
    Ok(axes)
}

pub(crate) fn axes(
    d: &ConvDimensionNumbers,
    rank: usize,
) -> Result<[Vec<usize>; 3], InferenceError> {
    Ok([
        permutation(
            d.input_batch_dimension,
            d.input_feature_dimension,
            &d.input_spatial_dimensions,
            rank,
        )?,
        permutation(
            d.kernel_input_feature_dimension,
            d.kernel_output_feature_dimension,
            &d.kernel_spatial_dimensions,
            rank,
        )?,
        permutation(
            d.output_batch_dimension,
            d.output_feature_dimension,
            &d.output_spatial_dimensions,
            rank,
        )?,
    ])
}

fn quotient(
    ty: &Type,
    axis: usize,
    divisor: u64,
) -> Result<(Option<u64>, Option<u64>), InferenceError> {
    let size = dimensions(ty).expect("checked tensor")[axis];
    if size.is_some_and(|size| !size.is_multiple_of(divisor)) {
        return Err(InferenceError::InvalidConvolutionGroups);
    }
    Ok((
        size.map(|size| size / divisor),
        bound(ty, axis).map(|upper| upper / divisor),
    ))
}

pub(crate) fn infer_shape(
    lhs: &Type,
    rhs: &Type,
    options: ConvolutionOptions<'_>,
    padding: &[Option<[i128; 2]>],
) -> Result<Vec<Option<u64>>, InferenceError> {
    for (index, ty) in [lhs, rhs].into_iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
    }
    let a = dimensions(lhs).ok_or(InferenceError::IncompatibleTypes { index: 0 })?;
    let b = dimensions(rhs).ok_or(InferenceError::IncompatibleTypes { index: 1 })?;
    let rank = a.len();
    if rank < 2 || b.len() != rank {
        return Err(InferenceError::InvalidConvolutionDimensions);
    }
    let [input, kernel, output] = axes(options.dimension_numbers, rank)?;
    if options.feature_group_count <= 0
        || options.batch_group_count <= 0
        || (options.feature_group_count > 1 && options.batch_group_count > 1)
    {
        return Err(InferenceError::InvalidConvolutionGroups);
    }
    let feature = options.feature_group_count as u64;
    let batch = options.batch_group_count as u64;
    let (input_feature, input_feature_bound) = quotient(lhs, input[1], feature)?;
    quotient(rhs, kernel[1], feature)?;
    quotient(rhs, kernel[1], batch)?;
    merge_dimension(
        input[1],
        input_feature,
        b[kernel[0]],
        input_feature_bound,
        bound(rhs, kernel[0]),
        Specificity::Most,
    )?;
    let (input_batch, _) = quotient(lhs, input[0], batch)?;
    if let Some(reversals) = options.window_reversal
        && reversals.len() != rank - 2
    {
        return Err(InferenceError::AttributeLengthMismatch {
            attribute: "window_reversal",
            expected: rank - 2,
            actual: reversals.len(),
        });
    }
    let ones = vec![1; rank - 2];
    let spatial_a = input[2..].iter().map(|&axis| a[axis]).collect::<Vec<_>>();
    let spatial_b = kernel[2..].iter().map(|&axis| b[axis]).collect::<Vec<_>>();
    let spatial = output_shape(
        &spatial_a,
        &spatial_b,
        options.window_strides.unwrap_or(&ones),
        options.lhs_dilation.unwrap_or(&ones),
        options.rhs_dilation.unwrap_or(&ones),
        padding,
    )?;
    let mut shape = vec![None; rank];
    shape[output[0]] = input_batch;
    shape[output[1]] = b[kernel[1]];
    for (&axis, &size) in output[2..].iter().zip(&spatial) {
        shape[axis] = size;
    }
    Ok(shape)
}

pub(crate) fn validate_result(
    lhs: &Type,
    rhs: &Type,
    result: &Type,
    options: ConvolutionOptions<'_>,
) -> Result<(), InferenceError> {
    let rank = dimensions(lhs)
        .ok_or(InferenceError::IncompatibleTypes { index: 0 })?
        .len();
    let result_shape = dimensions(result).ok_or(InferenceError::IncompatibleTypes { index: 2 })?;
    if result_shape.len() != rank {
        return Err(InferenceError::InvalidConvolutionDimensions);
    }
    let [input, kernel, output] = axes(options.dimension_numbers, rank)?;
    let (batch, upper) = quotient(lhs, input[0], options.batch_group_count as u64)?;
    merge_dimension(
        output[0],
        batch,
        result_shape[output[0]],
        upper,
        bound(result, output[0]),
        Specificity::Most,
    )?;
    let rhs_shape = dimensions(rhs).expect("checked rhs");
    let (feature, _) = merge_dimension(
        output[1],
        rhs_shape[kernel[1]],
        result_shape[output[1]],
        bound(rhs, kernel[1]),
        bound(result, output[1]),
        Specificity::Most,
    )?;
    let group = options.batch_group_count.max(options.feature_group_count) as u64;
    if feature.is_some_and(|size| !size.is_multiple_of(group)) {
        return Err(InferenceError::InvalidConvolutionGroups);
    }
    Ok(())
}
