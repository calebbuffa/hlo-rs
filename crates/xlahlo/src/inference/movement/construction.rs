use crate::{
    Diagnostic,
    attributes::PaddingConfig,
    shape::{self, Dimension, UNBOUNDED_SIZE},
    types::tensor,
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

use super::super::require;

/// Infer the merged dimensions of arrays concatenated along one axis.
///
/// Each later operand's precision is compared with the first operand; the last
/// such comparison determines the result element type.
pub fn infer_concatenate(ctx: &Context, inputs: &[TypeHandle], axis: usize) -> Result<TypeHandle> {
    require(!inputs.is_empty(), "concatenate needs an operand")?;
    let first = tensor(ctx, inputs[0])?;
    let rank = first.dimensions.len();
    require(
        axis < rank,
        &format!("concatenate axis {axis} out of bounds for rank {rank}"),
    )?;
    let mut dimensions = (0..rank)
        .map(|axis| shape::dimension(ctx, inputs[0], axis))
        .collect::<Result<Vec<_>>>()?;
    let mut element = first.element;
    for &input in &inputs[1..] {
        let input = tensor(ctx, input)?;
        require(
            input.dimensions.len() == rank,
            "concatenate operand ranks differ",
        )?;
        require(
            input.element.same_ignoring_fp_precision(first.element),
            "incompatible concatenate element types",
        )?;
        element = input.element.higher_precision(first.element)?;
        for (dimension, current) in dimensions.iter_mut().enumerate() {
            let next = if input.dimensions[dimension] == UNBOUNDED_SIZE {
                Dimension::Unbounded
            } else if input.dynamic_dimensions[dimension] {
                Dimension::Bounded(input.dimensions[dimension])
            } else {
                Dimension::Static(input.dimensions[dimension])
            };
            if dimension == axis {
                *current = match (*current, next) {
                    (Dimension::Unbounded, _) | (_, Dimension::Unbounded) => Dimension::Unbounded,
                    (Dimension::Static(a), Dimension::Static(b)) => {
                        Dimension::Static(a.checked_add(b).ok_or_else(|| {
                            pliron::verify_error_noloc!(Diagnostic::Inference {
                                message: format!(
                                    "concatenate dimension {axis} overflows adding {a} and {b}"
                                ),
                            })
                        })?)
                    }
                    (
                        Dimension::Static(a) | Dimension::Bounded(a),
                        Dimension::Static(b) | Dimension::Bounded(b),
                    ) => Dimension::Bounded(a.checked_add(b).ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: format!(
                                "concatenate dimension {axis} overflows adding {a} and {b}"
                            ),
                        })
                    })?),
                };
            } else {
                *current = match (*current, next) {
                    (Dimension::Static(a), Dimension::Static(b)) if a == b => Dimension::Static(a),
                    (Dimension::Static(a), Dimension::Bounded(b))
                    | (Dimension::Bounded(a), Dimension::Static(b))
                    | (Dimension::Bounded(a), Dimension::Bounded(b))
                        if a == b =>
                    {
                        Dimension::Bounded(a)
                    }
                    (Dimension::Unbounded, Dimension::Unbounded) => Dimension::Unbounded,
                    (Dimension::Unbounded, Dimension::Static(b))
                    | (Dimension::Static(b), Dimension::Unbounded) => Dimension::Static(b),
                    (Dimension::Unbounded, Dimension::Bounded(b))
                    | (Dimension::Bounded(b), Dimension::Unbounded) => Dimension::Bounded(b),
                    _ => {
                        return pliron::verify_err_noloc!(Diagnostic::Inference {
                            message: format!(
                                "concatenate non-axis dimension {dimension} differs: {current:?} versus {next:?}"
                            ),
                        });
                    }
                };
            }
        }
    }
    shape::with_default_layout(ctx, shape::dynamic_array(ctx, element, dimensions)?)
}
/// Infer dimensions after applying edge and interior padding.
pub fn infer_pad(
    ctx: &Context,
    input: TypeHandle,
    value: TypeHandle,
    config: &PaddingConfig,
) -> Result<TypeHandle> {
    let input = tensor(ctx, input)?;
    let value = tensor(ctx, shape::array_view(ctx, value)?)?;
    require(value.dimensions.is_empty(), "padding value must be scalar")?;
    require(
        !value.dynamic_dimensions.iter().any(|&dynamic| dynamic),
        "dynamic padding scalar is unsupported",
    )?;
    require(
        input.element.same_ignoring_fp_precision(value.element),
        "incompatible pad element types",
    )?;
    require(
        config.0.len() == input.dimensions.len(),
        "pad requires one config per input axis",
    )?;
    require(
        config.0.iter().all(|padding| padding.interior_padding >= 0),
        "negative interior padding",
    )?;
    let mut dimensions = Vec::with_capacity(config.0.len());
    for (axis, padding) in config.0.iter().enumerate() {
        if input.dimensions[axis] == UNBOUNDED_SIZE {
            dimensions.push(Dimension::Unbounded);
            continue;
        }
        let extent = (input.dimensions[axis] - 1)
            .max(0)
            .checked_mul(padding.interior_padding)
            .and_then(|size| size.checked_add(input.dimensions[axis]))
            .and_then(|size| size.checked_add(padding.edge_padding_low))
            .and_then(|size| size.checked_add(padding.edge_padding_high))
            .ok_or_else(|| pliron::verify_error_noloc!(Diagnostic::Inference {
                message: format!("padding result overflows at axis {axis}: extent {}, low {}, high {}, interior {}", input.dimensions[axis], padding.edge_padding_low, padding.edge_padding_high, padding.interior_padding),
            }))?;
        require(extent >= 0, "padding results in negative extent")?;
        dimensions.push(if input.dynamic_dimensions[axis] {
            Dimension::Bounded(extent)
        } else {
            Dimension::Static(extent)
        });
    }
    let element = input.element.higher_precision(value.element)?;
    shape::with_default_layout(ctx, shape::dynamic_array(ctx, element, dimensions)?)
}
/// Validate an iota result type and its varying dimension.
pub fn infer_iota(ctx: &Context, result: TypeHandle, axis: usize) -> Result<TypeHandle> {
    let output = tensor(ctx, result)?;
    require(
        !output.dimensions.is_empty(),
        "iota requires non-scalar result",
    )?;
    require(
        axis < output.dimensions.len(),
        &format!(
            "iota dimension {axis} out of range for rank {}",
            output.dimensions.len()
        ),
    )?;
    require(
        output.element.is_integer() || output.element.is_float() || output.element.is_complex(),
        "iota requires numerical element type",
    )?;
    Ok(result)
}
