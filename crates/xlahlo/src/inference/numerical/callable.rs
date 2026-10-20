//! Scalar and tuple contracts for callable numerical operations.
use super::{
    array, axes, dimensions, element, elements, infer_window_output, output, require,
    same_dimensions,
};
use crate::{
    Diagnostic,
    attributes::Window,
    inference::Signature,
    shape::{Dimension, Shape},
    types::ElementType,
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

/// Validate matching sort dimensions and preserve each operand's type.
pub(crate) fn infer_sort_result(ctx: &Context, operands: &[TypeHandle]) -> Result<TypeHandle> {
    let &first = operands.first().ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "sort requires operands".into()
        })
    })?;
    array(ctx, first)?;
    for &other in &operands[1..] {
        require(
            same_dimensions(ctx, first, other)?,
            "sort operand dimensions differ",
        )?;
    }
    if operands.len() == 1 {
        Ok(first)
    } else {
        Shape::tuple(ctx, operands.to_vec())
    }
}

/// Infer sort results after checking interleaved scalar comparator parameter pairs.
pub fn infer_sort(
    ctx: &Context,
    operands: &[TypeHandle],
    dimension: i64,
    comparator: &Signature,
) -> Result<TypeHandle> {
    let &first = operands.first().ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "sort requires operands".into()
        })
    })?;
    require(
        Shape::compatible(ctx, comparator.1, Shape::scalar(ctx, ElementType::Pred)?),
        "sort comparator must return a predicate scalar",
    )?;
    require(
        comparator.0.len()
            == operands.len().checked_mul(2).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: "sort comparator arity overflow".into()
                })
            })?,
        "sort comparator requires two scalar parameters per operand",
    )?;
    for (index, &operand) in operands.iter().enumerate() {
        let et = element(ctx, operand)?;
        for &parameter in &comparator.0[index * 2..index * 2 + 2] {
            require(
                Shape::is_array(ctx, parameter)
                    && Shape::rank(ctx, parameter)? == 0
                    && element(ctx, parameter)?.same_ignoring_fp_precision(et),
                "sort comparator parameter has incorrect scalar type",
            )?;
        }
    }
    let output = infer_sort_result(ctx, operands)?;
    // HLO sort validation intentionally checks only the upper bound.
    require(
        dimension < Shape::rank(ctx, array(ctx, first)?)? as i64,
        "sort dimension exceeds operand rank",
    )?;
    Ok(output)
}

/// Infer a scalar computation mapped over all operand dimensions.
pub fn infer_map(
    ctx: &Context,
    operands: &[TypeHandle],
    callee: &Signature,
    dims: &[i64],
) -> Result<TypeHandle> {
    let &first = operands.first().ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "map requires operands".into()
        })
    })?;
    let mut prototype = array(ctx, first)?;
    let mut prototype_is_buffer = !Shape::is_array(ctx, first);
    for &other in &operands[1..] {
        let other_rank = Shape::rank(ctx, other)?;
        if !prototype_is_buffer && same_dimensions(ctx, prototype, other)? {
            continue;
        }
        if element(ctx, other)?.same_ignoring_fp_precision(element(ctx, prototype)?) {
            if other_rank == 0 {
                continue;
            }
            if Shape::rank(ctx, prototype)? == 0 {
                prototype = other;
                prototype_is_buffer = false;
                continue;
            }
        }
        return pliron::verify_err_noloc!(Diagnostic::Inference {
            message: "map operands must have matching dimensions or compatible scalar broadcasts"
                .into(),
        });
    }
    let rank = Shape::rank(ctx, prototype)?;
    require(
        dims.len() == rank,
        "map cannot be applied to a subset of dimensions",
    )?;
    require(
        dims.iter().enumerate().all(|(i, &axis)| axis == i as i64),
        "map requires monotonically increasing dimension numbers",
    )?;
    require(
        operands.len() == callee.0.len(),
        "map operand count differs from parameter count",
    )?;
    let result = array(ctx, callee.1)?;
    require(
        Shape::rank(ctx, result)? == 0,
        "map computation must return a scalar",
    )?;
    for (&operand, &parameter) in operands.iter().zip(&callee.0) {
        let parameter = array(ctx, parameter)?;
        require(
            Shape::rank(ctx, parameter)? == 0
                && element(ctx, parameter)?.same_ignoring_fp_precision(element(ctx, operand)?),
            "map parameter must be scalar with operand element type",
        )?;
    }
    output(ctx, element(ctx, result)?, dimensions(ctx, prototype)?)
}

/// Infer a variadic reduction from inputs followed by scalar initial values.
///
/// Reducer parameters contain accumulators followed by input values; a tuple
/// reducer result retains its tuple structure even for a single component.
pub fn infer_reduce(
    ctx: &Context,
    operands: &[TypeHandle],
    dims: &[i64],
    reducer: &Signature,
) -> Result<TypeHandle> {
    require(
        !operands.is_empty() && operands.len().is_multiple_of(2),
        "reduce requires nonzero even operand count",
    )?;
    let count = operands.len() / 2;
    let first = array(ctx, operands[0])?;
    let rank = Shape::rank(ctx, first)?;
    for &input in &operands[..count] {
        require(
            same_dimensions(ctx, input, first)?,
            "reduce input dimensions differ",
        )?;
    }
    require(
        dims.iter().all(|&axis| axis >= 0 && (axis as usize) < rank),
        "reducing out-of-range dimension",
    )?;
    require(
        reducer.0.len() == operands.len(),
        "reducer requires two parameters per input",
    )?;
    let results = elements(ctx, reducer.1);
    require(
        results.len() == count && (count == 1 || Shape::is_tuple(ctx, reducer.1)),
        "reducer result cardinality differs from input count",
    )?;
    for (index, &result) in results.iter().enumerate() {
        let accumulator = reducer.0[index];
        let input_parameter = reducer.0[count + index];
        require(
            Shape::rank(ctx, result)? == 0
                && Shape::compatible(ctx, result, accumulator)
                && Shape::compatible_ignoring_fp_precision(ctx, result, operands[count + index])
                && Shape::compatible_ignoring_fp_precision(ctx, result, input_parameter)
                && Shape::compatible_ignoring_fp_precision(
                    ctx,
                    Shape::scalar(ctx, element(ctx, operands[index])?)?,
                    input_parameter,
                ),
            "reducer scalar, accumulator, input and initial-value types differ",
        )?;
    }
    let reduced = axes(rank, dims, false)?;
    let dims = dimensions(ctx, first)?
        .into_iter()
        .enumerate()
        .filter_map(|(axis, d)| (!reduced[axis]).then_some(d))
        .collect::<Vec<_>>();
    let outputs = results
        .iter()
        .map(|&result| output(ctx, element(ctx, result)?, dims.clone()))
        .collect::<Result<Vec<_>>>()?;
    if Shape::is_tuple(ctx, reducer.1) {
        Shape::tuple(ctx, outputs)
    } else {
        Ok(outputs[0])
    }
}

/// Infer a variadic window reduction, taking output element types from initial values.
pub fn infer_reduce_window(
    ctx: &Context,
    operands: &[TypeHandle],
    window: &Window,
    reducer: &Signature,
) -> Result<TypeHandle> {
    let signature = infer_reduce(ctx, operands, &[], reducer)?;
    let count = operands.len() / 2;
    let outputs = operands[..count]
        .iter()
        .zip(&operands[count..])
        .map(|(&input, &initial)| infer_window_output(ctx, input, window, element(ctx, initial)?))
        .collect::<Result<Vec<_>>>()?;
    if Shape::is_tuple(ctx, signature) {
        Shape::tuple(ctx, outputs)
    } else {
        Ok(outputs[0])
    }
}

/// Validate selection and scattering scalar computations and infer the operand-shaped result.
#[allow(clippy::too_many_arguments)]
pub fn infer_select_and_scatter(
    ctx: &Context,
    operand: TypeHandle,
    source: TypeHandle,
    initial: TypeHandle,
    window: &Window,
    select: &Signature,
    scatter: &Signature,
) -> Result<TypeHandle> {
    Shape::rank(ctx, operand)?;
    require(
        select.0.len() == 2,
        "select-and-scatter select requires two parameters",
    )?;
    require(
        Shape::compatible(ctx, select.1, Shape::scalar(ctx, ElementType::Pred)?),
        "select-and-scatter select must return scalar predicate",
    )?;
    let scalar = Shape::scalar(ctx, element(ctx, operand)?)?;
    for &parameter in &select.0 {
        require(
            Shape::compatible_ignoring_fp_precision(ctx, parameter, scalar),
            "select-and-scatter select parameters must match operand scalar type",
        )?;
    }
    infer_reduce(
        ctx,
        &[Shape::scalar(ctx, element(ctx, source)?)?, initial],
        &[],
        scatter,
    )?;
    let expected = infer_window_output(ctx, operand, window, element(ctx, operand)?)?;
    require(
        Shape::compatible_ignoring_fp_precision(ctx, source, expected),
        "select-and-scatter source must match window-reduced operand",
    )?;
    Ok(operand)
}

fn remove_axis(ctx: &Context, shape: TypeHandle, axis: usize) -> Result<TypeHandle> {
    let mut dims = dimensions(ctx, shape)?;
    require(axis < dims.len(), "scan axis exceeds input/output rank")?;
    dims.remove(axis);
    Shape::dynamic_array(ctx, element(ctx, shape)?, dims)
}

/// Check scan body slices and trailing carries against declared input and output types.
///
/// `strict_dynamic` requires exact dynamic flags; mixed floating precision is
/// relaxed only under that policy. Complete computation verification always
/// uses this strict shape policy, independently of layout sensitivity.
#[allow(clippy::too_many_arguments)]
pub fn verify_scan(
    ctx: &Context,
    operands: &[TypeHandle],
    result: TypeHandle,
    body: &Signature,
    num_carries: i64,
    dimension: i64,
    strict_dynamic: bool,
    allow_mixed_precision: bool,
) -> Result<()> {
    let carries = usize::try_from(num_carries).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "negative carry count".into()
        })
    })?;
    let axis = usize::try_from(dimension).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "negative scan axis".into()
        })
    })?;
    let roots = elements(ctx, body.1);
    let results = elements(ctx, result);
    require(
        carries <= operands.len()
            && carries <= results.len()
            && !(carries == operands.len() && carries == results.len())
            && body.0.len() == operands.len()
            && roots.len() == results.len(),
        "scan requires matching operand/parameter, root/result and carry counts and at least one input or output",
    )?;
    let inputs = operands.len() - carries;
    let outputs = results.len() - carries;
    let size_source = if inputs > 0 { operands[0] } else { results[0] };
    let bound = Shape::dimensions(ctx, size_source)?
        .get(axis)
        .copied()
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: "scan axis exceeds rank".into()
            })
        })?;
    let same = |a, b| {
        if strict_dynamic {
            if allow_mixed_precision {
                Shape::equal_ignoring_layout_and_fp_precision(ctx, a, b)
            } else {
                Shape::equal_ignoring_layout(ctx, a, b)
            }
        } else {
            Shape::compatible(ctx, a, b)
        }
    };
    for (index, (&operand, &parameter)) in operands.iter().zip(&body.0).enumerate() {
        let expected = if index < inputs {
            let slice = remove_axis(ctx, operand, axis)?;
            require(
                Shape::dimensions(ctx, operand)?[axis] == bound,
                "scan inputs have different scan dimension extents",
            )?;
            slice
        } else {
            operand
        };
        require(
            same(expected, parameter),
            "scan operand slice/carry differs from body parameter",
        )?;
    }
    for index in 0..carries {
        require(
            same(body.0[index + inputs], roots[index + outputs]),
            "scan body carry parameter and root differ",
        )?;
    }
    for (index, (&root, &result)) in roots.iter().zip(results.iter()).enumerate() {
        let expected = if index < outputs {
            let slice = remove_axis(ctx, result, axis)?;
            require(
                Shape::dimensions(ctx, result)?[axis] == bound,
                "scan outputs have different scan dimension extents",
            )?;
            slice
        } else {
            result
        };
        require(
            same(root, expected),
            "scan result slice/carry differs from body root",
        )?;
    }
    Ok(())
}

/// Infer scanned outputs followed by carries from the body result.
///
/// A scan with no scanned inputs must supply `scan_size`. Inference allows
/// compatible dynamic dimensions; complete computation verification additionally
/// requires exact body slice and carry shapes, ignoring layouts.
pub fn infer_scan(
    ctx: &Context,
    operands: &[TypeHandle],
    body: &Signature,
    num_carries: i64,
    dimension: i64,
    scan_size: Option<Dimension>,
) -> Result<TypeHandle> {
    let carries = usize::try_from(num_carries).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "negative carry count".into()
        })
    })?;
    let axis = usize::try_from(dimension).map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "negative scan axis".into()
        })
    })?;
    let roots = elements(ctx, body.1);
    require(
        carries <= operands.len() && carries <= roots.len(),
        "carry count exceeds operand/root count",
    )?;
    let size = if operands.len() > carries {
        Shape::dimension(ctx, operands[0], axis)?
    } else {
        scan_size.ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: "scan without inputs requires an explicit iteration extent".into()
            })
        })?
    };
    require(
        scan_size.is_none_or(|requested| requested == size),
        "explicit scan extent differs from input extent",
    )?;
    let outputs = roots.len() - carries;
    let results = roots
        .iter()
        .enumerate()
        .map(|(index, &shape)| {
            if index >= outputs {
                return Ok(shape);
            }
            let mut dims = dimensions(ctx, shape)?;
            require(axis <= dims.len(), "scan axis exceeds result rank")?;
            dims.insert(axis, size);
            output(ctx, element(ctx, shape)?, dims)
        })
        .collect::<Result<Vec<_>>>()?;
    let result = if Shape::is_tuple(ctx, body.1) {
        Shape::tuple(ctx, results)?
    } else {
        *results.first().ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Inference {
                message: "scan missing result".into()
            })
        })?
    };
    verify_scan(
        ctx,
        operands,
        result,
        body,
        num_carries,
        dimension,
        false,
        false,
    )?;
    Ok(result)
}
