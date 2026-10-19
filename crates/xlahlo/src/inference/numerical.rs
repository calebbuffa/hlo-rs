//! Numerical shape inference on native, interned pliron types.
//!
//! Callable operations accept [`crate::inference::Signature`] values in HLO
//! parameter order. These functions validate semantic geometry without resolving
//! symbols or constructing operations. Result types are interned in the supplied
//! context; layout-preserving operations retain the original type handles.
use super::{axes, require};
use crate::{
    Diagnostic,
    attributes::{FftType, FftType::*, TriangularSolveOptions, Window, WindowDimension},
    opcode::{Opcode, Opcode::*},
    shape::{Dimension, Shape, UNBOUNDED_SIZE},
    types::ElementType,
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};
use std::{cell::Ref, ops::Deref};

mod callable;
mod convolution;
mod dot;
pub(crate) use callable::infer_sort_result;
pub use callable::{
    infer_map, infer_reduce, infer_reduce_window, infer_scan, infer_select_and_scatter, infer_sort,
    verify_scan,
};
pub use convolution::{infer_convolution, infer_convolution_kind, infer_convolution_with_sparsity};
pub use dot::{infer_dot, infer_dot_with_sparsity, infer_ragged_dot, infer_scaled_dot};

fn array(ctx: &Context, shape: TypeHandle) -> Result<TypeHandle> {
    Shape::array_view(ctx, shape)
}
fn element(ctx: &Context, shape: TypeHandle) -> Result<ElementType> {
    Shape::element_type(ctx, array(ctx, shape)?)
}
fn dimensions(ctx: &Context, shape: TypeHandle) -> Result<Vec<Dimension>> {
    let shape = array(ctx, shape)?;
    (0..Shape::rank(ctx, shape)?)
        .map(|axis| Shape::dimension(ctx, shape, axis))
        .collect()
}
fn output(ctx: &Context, element: ElementType, dimensions: Vec<Dimension>) -> Result<TypeHandle> {
    Shape::with_default_layout(ctx, Shape::dynamic_array(ctx, element, dimensions)?)
}
fn compatible_size(a: i64, b: i64) -> bool {
    a == b || a == UNBOUNDED_SIZE || b == UNBOUNDED_SIZE
}
pub(crate) enum Elements<'a> {
    Tuple(Ref<'a, [TypeHandle]>),
    Single([TypeHandle; 1]),
}

impl Deref for Elements<'_> {
    type Target = [TypeHandle];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Tuple(types) => types,
            Self::Single(types) => types,
        }
    }
}

pub(crate) fn elements(ctx: &Context, shape: TypeHandle) -> Elements<'_> {
    if Shape::is_tuple(ctx, shape) {
        Elements::Tuple(Shape::children(ctx, shape))
    } else {
        Elements::Single([shape])
    }
}
fn same_dimensions(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    let a = Shape::dimensions(ctx, array(ctx, a)?)?;
    let b = Shape::dimensions(ctx, array(ctx, b)?)?;
    Ok(a.len() == b.len() && a.iter().zip(b.iter()).all(|(&a, &b)| compatible_size(a, b)))
}

/// Validate triangular-solve geometry and return the right-hand side type.
pub fn infer_triangular_solve(
    ctx: &Context,
    a: TypeHandle,
    b: TypeHandle,
    options: &TriangularSolveOptions,
) -> Result<TypeHandle> {
    require(
        Shape::is_array_or_buffer(ctx, a) && Shape::is_array_or_buffer(ctx, b),
        "triangular solve operands must have matching floating or complex element types",
    )?;
    let at = element(ctx, a)?;
    require(
        (at.is_float() || at.is_complex()) && at == element(ctx, b)?,
        "triangular solve operands must have matching floating or complex element types",
    )?;
    let ad = Shape::dimensions(ctx, array(ctx, a)?)?;
    let bd = Shape::dimensions(ctx, array(ctx, b)?)?;
    let rank = ad.len();
    require(
        rank >= 2,
        "triangular solve matrix operand must have rank at least two",
    )?;
    require(
        bd.len() == rank,
        "triangular solve operands must have identical rank",
    )?;
    require(
        compatible_size(ad[rank - 2], ad[rank - 1]),
        "triangular solve matrix dimensions must be square",
    )?;
    require(
        compatible_size(
            ad[rank - 1],
            bd[rank - if options.left_side { 2 } else { 1 }],
        ) && ad[..rank - 2] == bd[..rank - 2],
        "triangular solve shared or batch dimensions differ",
    )?;
    Ok(b)
}

/// Validate a square floating or complex matrix and preserve its type and layout.
pub fn infer_cholesky(ctx: &Context, a: TypeHandle) -> Result<TypeHandle> {
    require(
        Shape::is_array_or_buffer(ctx, a),
        "Cholesky requires a floating or complex array of rank at least two",
    )?;
    let et = element(ctx, a)?;
    let d = Shape::dimensions(ctx, array(ctx, a)?)?;
    require(
        (et.is_float() || et.is_complex()) && d.len() >= 2,
        "Cholesky requires a floating or complex array of rank at least two",
    )?;
    require(
        compatible_size(d[d.len() - 2], d[d.len() - 1]),
        "matrix minor dimensions must be square",
    )?;
    Ok(a)
}

/// Infer the tuple of top-k values and signed 32-bit indices along the final axis.
pub fn infer_top_k(ctx: &Context, operand: TypeHandle, k: i64) -> Result<TypeHandle> {
    let d = Shape::dimensions(ctx, operand)?;
    require(
        !d.is_empty() && k >= 0 && d[d.len() - 1] >= k,
        "top-k requires a positive-rank array and 0 <= k <= last dimension",
    )?;
    let mut dims = dimensions(ctx, operand)?;
    *dims.last_mut().ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "top-k missing dimension".into()
        })
    })? = Dimension::Static(k);
    let values = output(ctx, element(ctx, operand)?, dims)?;
    Shape::tuple(
        ctx,
        vec![
            values,
            Shape::with_element_type(ctx, values, ElementType::S32)?,
        ],
    )
}

/// Infer training, inference, or gradient batch normalization for `opcode`.
///
/// Parameters follow HLO operand order; training and gradient return tuples.
pub fn infer_batch_norm(
    ctx: &Context,
    opcode: Opcode,
    operands: &[TypeHandle],
    feature_index: i64,
) -> Result<TypeHandle> {
    let count = match opcode {
        BatchNormTraining => 3,
        BatchNormInference | BatchNormGrad => 5,
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Inference {
                message: "expected batch normalization opcode".into()
            });
        }
    };
    require(
        operands.len() == count,
        "incorrect batch normalization operand count",
    )?;
    let input = operands[0];
    let d = Shape::dimensions(ctx, input)?;
    let mut parameters = operands.to_vec();
    let order: &[usize] = match opcode {
        BatchNormTraining => &[2, 1],
        BatchNormInference => &[2, 1, 3, 4],
        BatchNormGrad => &[1, 2, 3, 4],
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Inference {
                message: "expected batch normalization opcode".into()
            });
        }
    };
    for &index in order {
        if opcode == BatchNormInference && index >= 3 {
            parameters[index] = array(ctx, parameters[index])?;
        } else {
            Shape::rank(ctx, parameters[index])?;
        }
    }
    axes(d.len(), &[feature_index], false)?;
    let feature = feature_index as usize;
    if opcode == BatchNormGrad {
        require(
            Shape::rank(ctx, parameters[4])? == d.len(),
            "batch normalization gradient and input ranks differ",
        )?;
    }
    let vector_indices: &[usize] = if opcode == BatchNormGrad {
        &[2, 1, 3]
    } else {
        &[2, 1]
    };
    for &index in vector_indices {
        require(
            Shape::rank(ctx, parameters[index])? == 1,
            "batch normalization parameter must be a vector",
        )?;
    }
    let et = element(ctx, input)?;
    require(
        et.is_float(),
        "batch normalization input must be floating point",
    )?;
    if opcode == BatchNormGrad {
        require(
            element(ctx, parameters[4])?.is_float(),
            "batch normalization gradient must be floating point",
        )?;
    }
    let type_order: &[usize] = if opcode == BatchNormGrad {
        &[4, 1, 2, 3]
    } else {
        order
    };
    for &index in type_order {
        require(
            element(ctx, parameters[index])?.same_ignoring_fp_precision(et),
            "batch normalization parameters must match the input floating type",
        )?;
    }
    let dimension_order: &[usize] = if opcode == BatchNormGrad {
        &[2, 1, 3]
    } else {
        order
    };
    for &index in dimension_order {
        let pd = Shape::dimensions(ctx, parameters[index])?;
        require(
            !pd.is_empty() && compatible_size(pd[0], d[feature]),
            "batch normalization parameter rank or feature count differs",
        )?;
    }
    if opcode == BatchNormGrad {
        require(
            same_dimensions(ctx, input, parameters[4])?,
            "batch normalization gradient bounds differ from input",
        )?;
    }
    if opcode == BatchNormInference {
        return Ok(input);
    }
    let feature = output(ctx, et, vec![Shape::dimension(ctx, input, feature)?])?;
    Shape::tuple(ctx, vec![input, feature, feature])
}

/// Infer a complex or real FFT, including dynamic dimensions and layout conversion.
pub fn infer_fft(
    ctx: &Context,
    operand: TypeHandle,
    kind: FftType,
    lengths: &[i64],
) -> Result<TypeHandle> {
    require(
        (1..=3).contains(&lengths.len()),
        "FFT requires one to three lengths",
    )?;
    let et = element(ctx, operand)?;
    match kind {
        FFT | IFFT => require(et.is_complex(), "FFT/IFFT requires complex input")?,
        RFFT => require(
            matches!(et, ElementType::F32 | ElementType::F64),
            "RFFT requires F32 or F64 input",
        )?,
        IRFFT => require(et.is_complex(), "IRFFT requires complex input")?,
    }
    let input = array(ctx, operand)?;
    let d = Shape::dimensions(ctx, input)?;
    let rank = d.len();
    require(
        rank >= lengths.len(),
        "FFT input rank is smaller than FFT length rank",
    )?;
    match kind {
        FFT | IFFT => {
            require(et.is_complex(), "FFT/IFFT requires complex input")?;
            Ok(operand)
        }
        RFFT | IRFFT => {
            require(
                if kind == RFFT {
                    matches!(et, ElementType::F32 | ElementType::F64)
                } else {
                    et.is_complex()
                },
                "real FFT input element type is invalid",
            )?;
            for (index, &size) in lengths.iter().enumerate() {
                let actual = d[rank - lengths.len() + index];
                if kind == RFFT {
                    require(
                        compatible_size(actual, size),
                        "RFFT input inner dimensions must match FFT lengths",
                    )?;
                } else if index + 1 < lengths.len() {
                    require(
                        actual == size,
                        "IRFFT outer transform dimensions must match FFT lengths",
                    )?;
                } else {
                    require(
                        (actual == 0 && size == 0)
                            || actual == UNBOUNDED_SIZE
                            || actual == size / 2 + 1,
                        "IRFFT inner dimension must equal FFT length/2+1",
                    )?;
                }
            }
            require(
                Shape::is_array(ctx, operand),
                "real FFT cannot change the element type of a buffer shape",
            )?;
            let new_element = match (kind, et) {
                (RFFT, ElementType::F32) => ElementType::C64,
                (RFFT, _) => ElementType::C128,
                (_, ElementType::C64) => ElementType::F32,
                _ => ElementType::F64,
            };
            let mut dims = dimensions(ctx, input)?;
            let length = lengths[lengths.len() - 1];
            if kind == IRFFT || length != 0 {
                let size = if kind == RFFT { length / 2 + 1 } else { length };
                dims[rank - 1] = if matches!(dims[rank - 1], Dimension::Bounded(_)) {
                    Dimension::Bounded(size)
                } else {
                    Dimension::Static(size)
                };
            }
            let result = Shape::dynamic_array(ctx, new_element, dims)?;
            match Shape::layout(ctx, Shape::with_element_type(ctx, input, new_element)?)? {
                Some(layout) => Shape::with_layout(ctx, result, layout.clone()),
                None => Ok(result),
            }
        }
    }
}

/// Build and validate a typed window from spatial sizes, strides, and dilations.
pub fn infer_window_from_dimensions(
    sizes: &[i64],
    strides: &[i64],
    padding: &[(i64, i64)],
    base_dilation: &[i64],
    window_dilation: &[i64],
    reversal: Option<&[bool]>,
) -> Result<Window> {
    for count in [
        strides.len(),
        padding.len(),
        base_dilation.len(),
        window_dilation.len(),
        reversal.map_or(0, <[bool]>::len),
    ] {
        require(
            count == 0 || count == sizes.len(),
            "window metadata count differs from its dimension count",
        )?;
    }
    Ok(Window(
        sizes
            .iter()
            .enumerate()
            .map(|(axis, &size)| {
                let (padding_low, padding_high) = padding.get(axis).copied().unwrap_or((0, 0));
                WindowDimension {
                    size,
                    stride: strides.get(axis).copied().unwrap_or(1),
                    padding_low,
                    padding_high,
                    base_dilation: base_dilation.get(axis).copied().unwrap_or(1),
                    window_dilation: window_dilation.get(axis).copied().unwrap_or(1),
                    window_reversal: reversal
                        .and_then(|values| values.get(axis))
                        .copied()
                        .unwrap_or(false),
                }
            })
            .collect(),
    ))
}

/// Infer windowed output dimensions with checked dilation and padding arithmetic.
pub fn infer_window_output(
    ctx: &Context,
    operand: TypeHandle,
    window: &Window,
    element: ElementType,
) -> Result<TypeHandle> {
    require(
        Shape::rank(ctx, operand)? == window.0.len(),
        "window rank differs from operand rank",
    )?;
    fn dilate(size: i64, factor: i64) -> Result<i64> {
        if size == 0 {
            return Ok(0);
        }
        (size - 1)
            .checked_mul(factor)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: "dilated window extent overflow".into()
                })
            })
    }
    let mut dims = Vec::new();
    for (axis, w) in window.0.iter().enumerate() {
        require(
            w.size > 0 && w.stride > 0 && w.base_dilation > 0 && w.window_dilation > 0,
            "window sizes, strides and dilations must be positive",
        )?;
        let input = Shape::dimension(ctx, operand, axis)?;
        dims.push(match input {
            Dimension::Unbounded => Dimension::Unbounded,
            Dimension::Static(size) | Dimension::Bounded(size) => {
                let base = dilate(size, w.base_dilation)?
                    .checked_add(w.padding_low)
                    .and_then(|n| n.checked_add(w.padding_high))
                    .ok_or_else(|| {
                        pliron::verify_error_noloc!(Diagnostic::Inference {
                            message: "padded window extent overflow".into()
                        })
                    })?;
                require(base >= 0, "negative padded input extent")?;
                let size = dilate(w.size, w.window_dilation)?;
                let count = if base == 0 || size > base {
                    0
                } else {
                    (base - size) / w.stride + 1
                };
                if matches!(input, Dimension::Bounded(_)) {
                    Dimension::Bounded(count)
                } else {
                    Dimension::Static(count)
                }
            }
        });
    }
    output(ctx, element, dims)
}
