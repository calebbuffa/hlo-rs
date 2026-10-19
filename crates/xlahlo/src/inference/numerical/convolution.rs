//! Forward and gradient convolution geometry with optional sparse operands.
use super::{axes, dot, element, infer_window_output, output, require};
use crate::{
    Diagnostic,
    attributes::{ConvolutionDimensionNumbers, ConvolutionKind, SparsityConfig, Window},
    shape::{Dimension, Shape, UNBOUNDED_SIZE},
    types::ElementType,
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

/// Infer a forward convolution with explicit feature and batch group counts.
#[allow(clippy::too_many_arguments)]
pub fn infer_convolution(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    feature_groups: i64,
    batch_groups: i64,
    window: &Window,
    d: &ConvolutionDimensionNumbers,
    preferred: Option<ElementType>,
) -> Result<TypeHandle> {
    infer_convolution_with_sparsity(
        ctx,
        lhs,
        rhs,
        feature_groups,
        batch_groups,
        window,
        d,
        preferred,
        None,
    )
}

/// Infer a forward convolution with optional sparse feature dimensions.
#[allow(clippy::too_many_arguments)]
pub fn infer_convolution_with_sparsity(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    feature_groups: i64,
    batch_groups: i64,
    window: &Window,
    d: &ConvolutionDimensionNumbers,
    preferred: Option<ElementType>,
    sparsity: Option<&SparsityConfig>,
) -> Result<TypeHandle> {
    infer_convolution_kind(
        ctx,
        lhs,
        rhs,
        feature_groups,
        batch_groups,
        window,
        d,
        preferred,
        sparsity,
        ConvolutionKind::Unset,
    )
}

/// Infer forward, data-gradient, or weight-gradient convolution geometry.
#[allow(clippy::too_many_arguments)]
pub fn infer_convolution_kind(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    feature_groups: i64,
    batch_groups: i64,
    window: &Window,
    d: &ConvolutionDimensionNumbers,
    preferred: Option<ElementType>,
    sparsity: Option<&SparsityConfig>,
    kind: ConvolutionKind,
) -> Result<TypeHandle> {
    require(
        Shape::is_array(ctx, lhs),
        "convolution lhs must be an array",
    )?;
    require(
        Shape::is_array(ctx, rhs),
        "convolution rhs must be an array",
    )?;
    require(
        feature_groups > 0,
        &format!("convolution feature group count must be positive; got {feature_groups}"),
    )?;
    require(
        batch_groups > 0,
        &format!("convolution batch group count must be positive; got {batch_groups}"),
    )?;
    require(
        feature_groups <= 1 || batch_groups <= 1,
        "convolution batch and feature group counts cannot both exceed one",
    )?;
    let spatial_rank = d.input_spatial_dimensions.0.len();
    require(
        d.kernel_spatial_dimensions.0.len() == spatial_rank,
        "convolution input and kernel spatial ranks differ",
    )?;
    require(
        d.output_spatial_dimensions.0.len() == spatial_rank,
        "convolution input and output spatial ranks differ",
    )?;
    require(
        window.0.len() == spatial_rank,
        "convolution window and dimension-number ranks differ",
    )?;
    let rank = spatial_rank.checked_add(2).ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "convolution rank overflow".into()
        })
    })?;
    let ld = Shape::dimensions(ctx, lhs)?;
    let rd = Shape::dimensions(ctx, rhs)?;
    require(
        ld.len() == rank && rd.len() == rank,
        "convolution operand rank differs from dimension numbers",
    )?;
    let permutations = [
        (
            d.input_batch_dimension,
            d.input_feature_dimension,
            &d.input_spatial_dimensions.0,
        ),
        (
            d.kernel_input_feature_dimension,
            d.kernel_output_feature_dimension,
            &d.kernel_spatial_dimensions.0,
        ),
        (
            d.output_batch_dimension,
            d.output_feature_dimension,
            &d.output_spatial_dimensions.0,
        ),
    ]
    .map(|(first, second, spatial)| {
        let mut permutation = vec![first, second];
        permutation.extend(spatial);
        permutation
    });
    require(
        permutations
            .iter()
            .flatten()
            .all(|&axis| axis >= 0 && (axis as usize) < rank),
        "convolution dimension number out of range",
    )?;
    for permutation in &permutations {
        // These are permutations, not sorted lists.
        axes(rank, permutation, false)?;
    }
    let input_features = dot::sparse_size(
        ld[d.input_feature_dimension as usize],
        d.input_feature_dimension,
        sparsity.and_then(|s| s.lhs.as_ref()),
    )?;
    let input_batch = ld[d.input_batch_dimension as usize];
    let kernel_input = dot::sparse_size(
        rd[d.kernel_input_feature_dimension as usize],
        d.kernel_input_feature_dimension,
        sparsity.and_then(|s| s.rhs.as_ref()),
    )?;
    let kernel_output = rd[d.kernel_output_feature_dimension as usize];
    require(
        kernel_output % batch_groups == 0,
        "convolution kernel output features must be divisible by batch groups",
    )?;
    require(
        kernel_output % feature_groups == 0,
        "convolution kernel output features must be divisible by feature groups",
    )?;
    require(
        input_batch % batch_groups == 0,
        "convolution input batch must be divisible by batch groups",
    )?;
    if kind == ConvolutionKind::Wgrad {
        require(
            input_batch == rd[d.output_batch_dimension as usize],
            "weight-gradient input and rhs batch bounds differ",
        )?;
        let mut bounds = vec![0; rank];
        bounds[d.kernel_output_feature_dimension as usize] =
            rd[d.output_feature_dimension as usize];
        bounds[d.kernel_input_feature_dimension as usize] = input_features / feature_groups;
        for (i, &axis) in d.kernel_spatial_dimensions.0.iter().enumerate() {
            bounds[axis as usize] = window.0[i].size;
        }
        let mut dynamic = vec![false; rank];
        if !matches!(
            Shape::dimension(ctx, lhs, d.input_batch_dimension as usize)?,
            Dimension::Static(_)
        ) {
            dynamic[d.output_batch_dimension as usize] = true;
        }
        for (i, &axis) in d.input_spatial_dimensions.0.iter().enumerate() {
            if !matches!(
                Shape::dimension(ctx, lhs, axis as usize)?,
                Dimension::Static(_)
            ) {
                dynamic[d.output_spatial_dimensions.0[i] as usize] = true;
            }
        }
        let dims = bounds
            .into_iter()
            .zip(dynamic)
            .map(|(n, dynamic)| {
                if n == UNBOUNDED_SIZE {
                    Dimension::Unbounded
                } else if dynamic {
                    Dimension::Bounded(n)
                } else {
                    Dimension::Static(n)
                }
            })
            .collect();
        return output(ctx, dot::dot_element(ctx, lhs, rhs, preferred)?, dims);
    }
    let dgrad = kind == ConvolutionKind::Dgrad;
    require(
        input_features % feature_groups == 0
            && input_features / feature_groups == if dgrad { kernel_output } else { kernel_input },
        "convolution input features do not match grouped kernel features",
    )?;
    let mut input_spatial = Vec::new();
    for (index, (&input, &kernel)) in d
        .input_spatial_dimensions
        .0
        .iter()
        .zip(&d.kernel_spatial_dimensions.0)
        .enumerate()
    {
        require(
            rd[kernel as usize] == window.0[index].size,
            "convolution window sizes must match kernel spatial dimensions",
        )?;
        let size = ld[input as usize];
        input_spatial.push(if size == UNBOUNDED_SIZE {
            Dimension::Unbounded
        } else {
            Dimension::Static(size)
        });
    }
    let spatial = if dgrad {
        let mut dims = Vec::new();
        for (axis, w) in input_spatial.into_iter().zip(&window.0) {
            if axis == Dimension::Unbounded {
                dims.push(Dimension::Unbounded);
                continue;
            }
            let Dimension::Static(size) = axis else {
                return pliron::verify_err_noloc!(Diagnostic::Inference {
                    message: "unexpected bounded spatial base".into()
                });
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
                    pliron::verify_error_noloc!(Diagnostic::Inference {
                        message: "data-gradient spatial bound is negative or overflows".into()
                    })
                })?;
            dims.push(Dimension::Static(size));
        }
        Shape::dynamic_array(ctx, element(ctx, lhs)?, dims)?
    } else {
        infer_window_output(
            ctx,
            Shape::dynamic_array(ctx, element(ctx, lhs)?, input_spatial)?,
            window,
            element(ctx, lhs)?,
        )?
    };
    require(
        matches!(
            Shape::dimension(ctx, rhs, d.kernel_output_feature_dimension as usize)?,
            Dimension::Static(_)
        ),
        "convolution kernel output feature dimension must be static",
    )?;
    let mut dims = vec![Dimension::Static(0); rank];
    let batch_size = input_batch / batch_groups;
    dims[d.output_batch_dimension as usize] = if batch_size == UNBOUNDED_SIZE {
        Dimension::Unbounded
    } else if matches!(
        Shape::dimension(ctx, lhs, d.input_batch_dimension as usize)?,
        Dimension::Bounded(_)
    ) {
        Dimension::Bounded(batch_size)
    } else {
        Dimension::Static(batch_size)
    };
    dims[d.output_feature_dimension as usize] =
        Dimension::Static(if dgrad { kernel_input } else { kernel_output });
    let spatial_bounds = Shape::dimensions(ctx, spatial)?;
    for (i, &axis) in d.output_spatial_dimensions.0.iter().enumerate() {
        let size = spatial_bounds[i];
        let dynamic = !matches!(
            Shape::dimension(ctx, lhs, d.input_spatial_dimensions.0[i] as usize)?,
            Dimension::Static(_)
        ) || !matches!(
            Shape::dimension(ctx, rhs, d.kernel_spatial_dimensions.0[i] as usize)?,
            Dimension::Static(_)
        );
        dims[axis as usize] = if size == UNBOUNDED_SIZE {
            Dimension::Unbounded
        } else if dynamic {
            Dimension::Bounded(size)
        } else {
            Dimension::Static(size)
        };
    }
    output(ctx, dot::dot_element(ctx, lhs, rhs, preferred)?, dims)
}
