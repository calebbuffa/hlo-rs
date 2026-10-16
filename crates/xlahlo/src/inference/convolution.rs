use super::{axes, infer_window_output};
use crate::proto::hlo::ConvolutionKind;
use crate::proto::xla_data::{ConvolutionDimensionNumbers, SparsityConfig, Window};
use crate::{Dimension, Error, PrimitiveType, Result, Shape, UNBOUNDED_SIZE};

pub fn infer_convolution(
    lhs: &Shape,
    rhs: &Shape,
    feature_groups: i64,
    batch_groups: i64,
    window: &Window,
    dimensions: &ConvolutionDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
) -> Result<Shape> {
    infer_convolution_with_sparsity(
        lhs,
        rhs,
        feature_groups,
        batch_groups,
        window,
        dimensions,
        preferred_element,
        &Default::default(),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Preserves the upstream convolution inference inputs."
)]
pub fn infer_convolution_with_sparsity(
    lhs: &Shape,
    rhs: &Shape,
    feature_groups: i64,
    batch_groups: i64,
    window: &Window,
    dimensions: &ConvolutionDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
    sparsity: &SparsityConfig,
) -> Result<Shape> {
    infer_convolution_kind(
        lhs,
        rhs,
        feature_groups,
        batch_groups,
        window,
        dimensions,
        preferred_element,
        sparsity,
        ConvolutionKind::CONVOLUTION_KIND_UNSET,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Preserves the upstream convolution inference inputs."
)]
pub fn infer_convolution_kind(
    lhs: &Shape,
    rhs: &Shape,
    feature_groups: i64,
    batch_groups: i64,
    window: &Window,
    dimensions: &ConvolutionDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
    sparsity: &SparsityConfig,
    kind: ConvolutionKind,
) -> Result<Shape> {
    let lhs = super::dot::array(lhs)?;
    let rhs = super::dot::array(rhs)?;
    let spatial_rank = dimensions.input_spatial_dimensions.len();
    let rank = spatial_rank
        .checked_add(2)
        .ok_or_else(|| Error::new("overflow", "convolution rank overflow"))?;
    if lhs.rank()? != rank
        || rhs.rank()? != rank
        || dimensions.kernel_spatial_dimensions.len() != spatial_rank
        || dimensions.output_spatial_dimensions.len() != spatial_rank
        || window.dimensions.len() != spatial_rank
    {
        return Err(Error::new(
            "dimensions",
            "convolution input, kernel, output and window ranks differ",
        ));
    }
    for (first, second, spatial) in [
        (
            dimensions.input_batch_dimension,
            dimensions.input_feature_dimension,
            &dimensions.input_spatial_dimensions,
        ),
        (
            dimensions.kernel_input_feature_dimension,
            dimensions.kernel_output_feature_dimension,
            &dimensions.kernel_spatial_dimensions,
        ),
        (
            dimensions.output_batch_dimension,
            dimensions.output_feature_dimension,
            &dimensions.output_spatial_dimensions,
        ),
    ] {
        let mut permutation = vec![first, second];
        permutation.extend(spatial);
        axes(rank, &permutation, true)?;
    }
    if feature_groups <= 0 || batch_groups <= 0 || (feature_groups > 1 && batch_groups > 1) {
        return Err(Error::new(
            "groups",
            "convolution group counts must be positive and cannot both exceed one",
        ));
    }
    let input_features = super::dot::sparse_size(
        lhs.dimensions()[dimensions.input_feature_dimension as usize],
        dimensions.input_feature_dimension,
        sparsity.lhs.as_ref(),
    )?;
    let input_batch = lhs.dimensions()[dimensions.input_batch_dimension as usize];
    let kernel_input = super::dot::sparse_size(
        rhs.dimensions()[dimensions.kernel_input_feature_dimension as usize],
        dimensions.kernel_input_feature_dimension,
        sparsity.rhs.as_ref(),
    )?;
    let kernel_output = rhs.dimensions()[dimensions.kernel_output_feature_dimension as usize];
    if kernel_output % batch_groups != 0
        || kernel_output % feature_groups > 0
        || input_batch % batch_groups != 0
    {
        return Err(Error::new(
            "groups",
            "convolution feature/batch geometry is incompatible with group counts",
        ));
    }
    if kind == ConvolutionKind::CONVOLUTION_KIND_WGRAD {
        if input_batch != rhs.dimensions()[dimensions.output_batch_dimension as usize] {
            return Err(Error::new(
                "convolution",
                "weight-gradient input and rhs batch bounds differ",
            ));
        }
        let mut output = vec![0; rank];
        output[dimensions.kernel_output_feature_dimension as usize] =
            rhs.dimensions()[dimensions.output_feature_dimension as usize];
        output[dimensions.kernel_input_feature_dimension as usize] =
            input_features / feature_groups;
        for (i, &axis) in dimensions.kernel_spatial_dimensions.iter().enumerate() {
            output[axis as usize] = window.dimensions[i].size;
        }
        let mut dynamic = vec![false; rank];
        if !matches!(
            lhs.dimension(dimensions.input_batch_dimension as usize)?,
            Dimension::Static(_)
        ) {
            dynamic[dimensions.output_batch_dimension as usize] = true;
        }
        for (i, &axis) in dimensions.input_spatial_dimensions.iter().enumerate() {
            if !matches!(lhs.dimension(axis as usize)?, Dimension::Static(_)) {
                dynamic[dimensions.output_spatial_dimensions[i] as usize] = true;
            }
        }
        let output = output
            .into_iter()
            .zip(dynamic)
            .map(|(bound, dynamic)| {
                if bound == UNBOUNDED_SIZE {
                    Dimension::Unbounded
                } else if dynamic {
                    Dimension::Bounded(bound)
                } else {
                    Dimension::Static(bound)
                }
            })
            .collect();
        return Shape::dynamic_array(super::dot::element(&lhs, &rhs, preferred_element)?, output)?
            .with_default_layout();
    }
    let dgrad = kind == ConvolutionKind::CONVOLUTION_KIND_DGRAD;
    if input_features % feature_groups != 0
        || input_features / feature_groups != if dgrad { kernel_output } else { kernel_input }
    {
        return Err(Error::new(
            "groups",
            "convolution input features do not match grouped kernel features",
        ));
    }
    if !matches!(
        rhs.dimension(dimensions.kernel_output_feature_dimension as usize)?,
        Dimension::Static(_)
    ) {
        return Err(Error::new(
            "dynamic",
            "dynamic convolution kernel output features are unsupported upstream",
        ));
    }
    let mut input_spatial = Vec::with_capacity(spatial_rank);
    for (index, (&input, &kernel)) in dimensions
        .input_spatial_dimensions
        .iter()
        .zip(&dimensions.kernel_spatial_dimensions)
        .enumerate()
    {
        if rhs.dimensions()[kernel as usize] != window.dimensions[index].size {
            return Err(Error::new(
                "window",
                "convolution window sizes must match kernel spatial dimensions",
            ));
        }
        let size = lhs.dimensions()[input as usize];
        input_spatial.push(if size == UNBOUNDED_SIZE {
            Dimension::Unbounded
        } else {
            Dimension::Static(size)
        });
    }
    let spatial = if dgrad {
        let mut output = Vec::new();
        for (axis, w) in input_spatial.into_iter().zip(&window.dimensions) {
            if axis == Dimension::Unbounded {
                output.push(Dimension::Unbounded);
                continue;
            }
            let Dimension::Static(size) = axis else {
                return Err(Error::new("convolution", "unexpected bounded spatial base"));
            };
            let lhs = (i128::from(size) - 1) * i128::from(w.base_dilation);
            let kernel = (i128::from(w.size) - 1) * i128::from(w.window_dilation) + 1;
            let size = lhs
                .checked_mul(i128::from(w.stride))
                .and_then(|v| v.checked_add(kernel))
                .and_then(|v| v.checked_sub(i128::from(w.padding_low)))
                .and_then(|v| v.checked_sub(i128::from(w.padding_high)))
                .and_then(|v| i64::try_from(v).ok())
                .filter(|&v| v >= 0)
                .ok_or_else(|| {
                    Error::new(
                        "convolution",
                        "data-gradient spatial bound is negative or overflows",
                    )
                })?;
            output.push(Dimension::Static(size));
        }
        Shape::dynamic_array(lhs.element_type(), output)?
    } else {
        infer_window_output(
            &Shape::dynamic_array(lhs.element_type(), input_spatial)?,
            window,
            lhs.element_type(),
        )?
    };
    let mut output = vec![Dimension::Static(0); rank];
    let batch_size = input_batch / batch_groups;
    output[dimensions.output_batch_dimension as usize] = if batch_size == UNBOUNDED_SIZE {
        Dimension::Unbounded
    } else if matches!(
        lhs.dimension(dimensions.input_batch_dimension as usize)?,
        Dimension::Bounded(_)
    ) {
        Dimension::Bounded(batch_size)
    } else {
        Dimension::Static(batch_size)
    };
    output[dimensions.output_feature_dimension as usize] =
        Dimension::Static(if dgrad { kernel_input } else { kernel_output });
    for (index, &axis) in dimensions.output_spatial_dimensions.iter().enumerate() {
        let size = spatial.dimensions()[index];
        let dynamic = !matches!(
            lhs.dimension(dimensions.input_spatial_dimensions[index] as usize)?,
            Dimension::Static(_)
        ) || !matches!(
            rhs.dimension(dimensions.kernel_spatial_dimensions[index] as usize)?,
            Dimension::Static(_)
        );
        output[axis as usize] = if size == UNBOUNDED_SIZE {
            Dimension::Unbounded
        } else if dynamic {
            Dimension::Bounded(size)
        } else {
            Dimension::Static(size)
        };
    }
    let element = super::dot::element(&lhs, &rhs, preferred_element)?;
    Shape::dynamic_array(element, output)?.with_default_layout()
}
