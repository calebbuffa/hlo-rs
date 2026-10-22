//! Native shape inference for elementwise HLO operations.
use super::require;
use crate::{
    Diagnostic,
    opcode::Opcode,
    shape::{self, UNBOUNDED_SIZE},
    types::{ElementType, TensorType, tensor},
};
use pliron::{
    common_traits::Verify,
    context::Context,
    result::Result,
    r#type::{Type, TypeHandle},
};

fn rebuilt(
    ctx: &Context,
    base: TypeHandle,
    element: ElementType,
    dimensions: Vec<i64>,
    dynamic_dimensions: Vec<bool>,
) -> Result<TypeHandle> {
    let mut ty = tensor(ctx, base)?.clone();
    ty.element = element;
    ty.dimensions = dimensions;
    ty.dynamic_dimensions = dynamic_dimensions;
    if element.storage_bit_width() >= 8 {
        for layout in &mut ty.layout {
            layout.element_size_in_bits = 0;
        }
    }
    ty.verify(ctx)?;
    Ok(TensorType::instantiate(ty, ctx).into())
}

pub(crate) fn unary_result_element(opcode: Opcode, element: ElementType) -> Result<ElementType> {
    use Opcode::*;
    let accepted = match opcode {
        Ceil | Cbrt | Erf | Floor | RoundNearestAfz | RoundNearestEven | IsFinite => {
            element.is_float()
        }
        Acos | Acosh | Asin | Asinh | Atanh | Cos | Cosh | Exp | Exp2 | Expm1 | Log | Log1p
        | Log2 | Logistic | Real | Imag | Rsqrt | Sin | Sinh | Sqrt | Tan | Tanh => {
            element.is_float() || element.is_complex()
        }
        Abs | Sign => element.is_signed() || element.is_float() || element.is_complex(),
        Clz | PopulationCount => element.is_integer(),
        Negate => element.is_integer() || element.is_float() || element.is_complex(),
        Not => element == ElementType::Pred || element.is_integer(),
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Inference {
                message: format!("unsupported unary inference for {opcode}"),
            });
        }
    };
    require(
        accepted,
        "unary opcode does not accept operand element type",
    )?;
    let result_element = match (opcode, element) {
        (IsFinite, _) => ElementType::Pred,
        (Abs | Real | Imag, ElementType::C64) => ElementType::F32,
        (Abs | Real | Imag, ElementType::C128) => ElementType::F64,
        _ => element,
    };
    Ok(result_element)
}

/// Infer a unary elementwise operation's result type.
pub fn infer_unary(ctx: &Context, opcode: Opcode, operand: TypeHandle) -> Result<TypeHandle> {
    let input = tensor(ctx, operand)?;
    let result_element = unary_result_element(opcode, input.element)?;
    if result_element == input.element {
        return Ok(operand);
    }
    rebuilt(
        ctx,
        operand,
        result_element,
        input.dimensions.to_vec(),
        input.dynamic_dimensions.to_vec(),
    )
}

/// Infer a binary elementwise operation without explicit broadcast dimensions.
pub fn infer_binary(
    ctx: &Context,
    opcode: Opcode,
    lhs: TypeHandle,
    rhs: TypeHandle,
) -> Result<TypeHandle> {
    infer_binary_in_dim(ctx, opcode, lhs, rhs, &[])
}

/// Return the common non-scalar array shape for implicit scalar broadcasting.
pub fn infer_scalar_broadcast(ctx: &Context, shapes: &[TypeHandle]) -> Result<Option<TypeHandle>> {
    let mut selected: Option<TypeHandle> = None;
    for &shape in shapes {
        if !shape::is_array(ctx, shape) || shape::rank(ctx, shape)? == 0 {
            continue;
        }
        if let Some(first) = selected {
            require(
                shape::compatible(ctx, first, shape),
                "implicit scalar broadcast requires matching array dimensions",
            )?;
        } else {
            selected = Some(shape);
        }
    }
    Ok(selected)
}

/// Infer a binary result while mapping lower-rank dimensions to higher-rank axes.
pub fn infer_binary_in_dim(
    ctx: &Context,
    opcode: Opcode,
    lhs: TypeHandle,
    rhs: TypeHandle,
    broadcast_dimensions: &[i64],
) -> Result<TypeHandle> {
    use Opcode::*;
    let left = tensor(ctx, lhs)?;
    let right = tensor(ctx, rhs)?;
    let (a, b) = (left.element, right.element);
    require(
        matches!(
            opcode,
            Add | And
                | Atan2
                | Complex
                | Compare
                | Divide
                | Maximum
                | Minimum
                | Mulhi
                | Multiply
                | Or
                | Power
                | Remainder
                | ShiftLeft
                | ShiftRightArithmetic
                | ShiftRightLogical
                | Subtract
                | Xor
        ),
        "unsupported binary inference opcode",
    )?;
    require(
        !matches!(
            opcode,
            Subtract
                | Power
                | Divide
                | Remainder
                | ShiftLeft
                | ShiftRightArithmetic
                | ShiftRightLogical
        ) || (a != ElementType::Pred && b != ElementType::Pred),
        "arithmetic element types are required",
    )?;
    require(
        opcode != Atan2 || a.is_float() || a.is_complex(),
        "atan2 requires floating or complex elements",
    )?;
    require(
        opcode != Complex || a.is_float(),
        "complex composition requires a floating lhs",
    )?;
    require(
        !matches!(opcode, And | Or | Xor) || a == ElementType::Pred || a.is_integer(),
        "logical op requires predicate or integral lhs",
    )?;
    require(
        a.same_ignoring_fp_precision(b),
        "incompatible binary element types",
    )?;
    let accepted = match opcode {
        Add | Maximum | Minimum | Multiply | Mulhi | Compare => true,
        Subtract | Power | Divide | Remainder | ShiftLeft | ShiftRightArithmetic
        | ShiftRightLogical => a != ElementType::Pred,
        Atan2 => a.is_float() || a.is_complex(),
        And | Or | Xor => a == ElementType::Pred || a.is_integer(),
        Complex => matches!(a, ElementType::F32 | ElementType::F64) && a == b,
        _ => false,
    };
    require(
        accepted,
        "binary opcode does not accept operand element type",
    )?;
    let result_element = match opcode {
        Compare => ElementType::Pred,
        Complex => {
            if a == ElementType::F32 {
                ElementType::C64
            } else {
                ElementType::C128
            }
        }
        _ => a.higher_precision(b)?,
    };

    let left_rank = left.dimensions.len();
    let right_rank = right.dimensions.len();
    let (dimensions, dynamic, base) = if left_rank == right_rank {
        if !broadcast_dimensions.is_empty()
            && broadcast_dimensions != (0..left_rank as i64).collect::<Vec<_>>()
        {
            return pliron::verify_err_noloc!(Diagnostic::Inference {
                message: "same-rank broadcast dimensions must be identity".to_owned(),
            });
        }
        let mut dimensions = Vec::with_capacity(left_rank);
        let mut dynamic = Vec::with_capacity(left_rank);
        for axis in 0..left_rank {
            let x = left.dimensions[axis];
            let y = right.dimensions[axis];
            require(
                x == y || x == 1 || y == 1 || x == UNBOUNDED_SIZE || y == UNBOUNDED_SIZE,
                "incompatible equal-rank dimensions",
            )?;
            let (size, is_dynamic) = if x == 1 {
                (y, right.dynamic_dimensions[axis])
            } else if y == 1 {
                (x, left.dynamic_dimensions[axis])
            } else if x == UNBOUNDED_SIZE {
                (y, right.dynamic_dimensions[axis])
            } else if y == UNBOUNDED_SIZE {
                (x, left.dynamic_dimensions[axis])
            } else {
                (
                    x,
                    left.dynamic_dimensions[axis] || right.dynamic_dimensions[axis],
                )
            };
            dimensions.push(size);
            dynamic.push(is_dynamic);
        }
        (dimensions, dynamic, lhs)
    } else {
        let (small, large) = if left_rank < right_rank {
            (&*left, &*right)
        } else {
            (&*right, &*left)
        };
        require(
            !broadcast_dimensions.is_empty() || small.dimensions.is_empty(),
            "different-rank binary operands require explicit broadcast dimensions",
        )?;
        require(
            broadcast_dimensions.len() == small.dimensions.len(),
            "broadcast dimensions must match lower-rank operand rank",
        )?;
        let mut dimensions = large.dimensions.clone();
        let mut dynamic = large.dynamic_dimensions.clone();
        let mut previous = None;
        for (axis, &target) in broadcast_dimensions.iter().enumerate() {
            let target = usize::try_from(target).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Inference {
                    message: "negative broadcast dimension".to_owned(),
                })
            })?;
            require(
                target < dimensions.len(),
                "broadcast dimension out of range",
            )?;
            require(
                previous.is_none_or(|previous| previous < target),
                "broadcast dimensions must increase",
            )?;
            previous = Some(target);
            let small_size = small.dimensions[axis];
            let large_size = dimensions[target];
            require(
                small_size == large_size || small_size == 1 || large_size == 1,
                "incompatible mapped dimensions",
            )?;
            let small_dynamic = small.dynamic_dimensions[axis];
            let large_dynamic = dynamic[target];
            require(
                small_dynamic == large_dynamic
                    || small_size == large_size
                    || small_size == 1 && !small_dynamic
                    || large_size == 1 && !large_dynamic,
                "incompatible mapped dynamism",
            )?;
            let (size, is_dynamic) = if small_size == 1 {
                (large_size, large_dynamic)
            } else if large_size == 1 {
                (small_size, small_dynamic)
            } else {
                (small_size, small_dynamic || large_dynamic)
            };
            dimensions[target] = size;
            dynamic[target] = is_dynamic;
        }
        (
            dimensions,
            dynamic,
            if left_rank < right_rank { rhs } else { lhs },
        )
    };

    let same_geometry = left_rank == right_rank
        && left.dimensions == right.dimensions
        && !left.dimensions.contains(&UNBOUNDED_SIZE);
    if same_geometry {
        rebuilt(ctx, base, result_element, dimensions, dynamic)
    } else {
        let result = rebuilt(ctx, base, result_element, dimensions, dynamic)?;
        shape::with_default_layout(ctx, result)
    }
}

/// Preserve operand geometry and layout while replacing its element type.
pub fn infer_convert(
    ctx: &Context,
    operand: TypeHandle,
    element: ElementType,
) -> Result<TypeHandle> {
    shape::with_element_type(ctx, operand, element)
}

/// Validate the numeric domain and bit counts for reduce-precision.
pub fn infer_reduce_precision(
    ctx: &Context,
    operand: TypeHandle,
    exponent_bits: i32,
    mantissa_bits: i32,
) -> Result<TypeHandle> {
    let element = tensor(ctx, operand)?.element;
    require(
        element.is_float(),
        "reduce-precision requires floating-point element type",
    )?;
    require(exponent_bits >= 1, "exponent_bits must be at least one")?;
    require(mantissa_bits >= 0, "mantissa_bits must be nonnegative")?;
    Ok(operand)
}

/// Infer stochastic conversion after checking its random tensor contract.
pub fn infer_stochastic_convert(
    ctx: &Context,
    operand: TypeHandle,
    random: TypeHandle,
    element: ElementType,
) -> Result<TypeHandle> {
    let input = tensor(ctx, operand)?;
    let random_type = tensor(ctx, random)?;
    require(
        random_type.element.is_integer() && !random_type.element.is_signed(),
        "stochastic-convert random values must be unsigned integers",
    )?;
    require(
        input.element.is_float(),
        "stochastic-convert requires floating-point operand data",
    )?;
    require(
        input.element.bit_width() == random_type.element.bit_width(),
        "stochastic-convert operand and random values must have equal bit widths",
    )?;
    require(
        input.dimensions == random_type.dimensions
            && input.dynamic_dimensions == random_type.dynamic_dimensions,
        "stochastic-convert input shapes differ",
    )?;
    let input_layout = shape::layout(ctx, operand)?;
    let random_layout = shape::layout(ctx, random)?;
    require(
        match (input_layout, random_layout) {
            (None, None) => true,
            (Some(a), Some(b)) => crate::shape::layouts_equal(ctx, &a, &b),
            _ => false,
        },
        "stochastic-convert input layouts differ",
    )?;
    shape::with_element_type(ctx, operand, element)
}

/// Infer a clamp result, allowing scalar or shape-compatible bounds.
pub fn infer_clamp(
    ctx: &Context,
    minimum: TypeHandle,
    operand: TypeHandle,
    maximum: TypeHandle,
) -> Result<TypeHandle> {
    let (minimum_ty, operand_ty, maximum_ty) = (
        tensor(ctx, minimum)?,
        tensor(ctx, operand)?,
        tensor(ctx, maximum)?,
    );
    require(
        minimum_ty
            .element
            .same_ignoring_fp_precision(operand_ty.element)
            && maximum_ty
                .element
                .same_ignoring_fp_precision(operand_ty.element),
        "incompatible clamp element types",
    )?;
    require(
        (minimum_ty.dimensions.is_empty()
            || shape::compatible_ignoring_fp_precision(ctx, minimum, operand))
            && (maximum_ty.dimensions.is_empty()
                || shape::compatible_ignoring_fp_precision(ctx, maximum, operand))
            && (minimum_ty.dimensions.is_empty()
                || maximum_ty.dimensions.is_empty()
                || shape::compatible_ignoring_fp_precision(ctx, minimum, maximum)),
        "clamp bounds have incompatible shapes",
    )?;
    Ok(operand)
}

/// Infer a select result, including scalar predicates and dynamic extents.
pub fn infer_select(
    ctx: &Context,
    predicate: TypeHandle,
    on_true: TypeHandle,
    on_false: TypeHandle,
) -> Result<TypeHandle> {
    let (predicate_ty, true_ty, false_ty) = (
        tensor(ctx, predicate)?,
        tensor(ctx, on_true)?,
        tensor(ctx, on_false)?,
    );
    require(
        shape::compatible_ignoring_fp_precision(ctx, on_true, on_false),
        "select values must have compatible shapes",
    )?;
    require(
        predicate_ty.element == ElementType::Pred,
        "select predicate must have predicate element type",
    )?;
    let scalar_predicate = predicate_ty.dimensions.is_empty();
    require(
        scalar_predicate
            || shape::compatible(
                ctx,
                shape::with_element_type(ctx, predicate, true_ty.element)?,
                on_true,
            ) && shape::compatible(
                ctx,
                shape::with_element_type(ctx, predicate, false_ty.element)?,
                on_false,
            ),
        "select predicate and values must have compatible shapes",
    )?;
    let element = true_ty.element.higher_precision(false_ty.element)?;
    let mut dynamic = Vec::with_capacity(true_ty.dimensions.len());
    for axis in 0..true_ty.dimensions.len() {
        let left_unbounded = true_ty.dimensions[axis] == UNBOUNDED_SIZE;
        let right_unbounded = false_ty.dimensions[axis] == UNBOUNDED_SIZE;
        if left_unbounded || right_unbounded {
            dynamic.push(true_ty.dynamic_dimensions[axis] && false_ty.dynamic_dimensions[axis]);
        } else {
            dynamic.push(
                (!scalar_predicate && predicate_ty.dynamic_dimensions[axis])
                    || true_ty.dynamic_dimensions[axis]
                    || false_ty.dynamic_dimensions[axis],
            );
        }
    }
    let mut dimensions = if scalar_predicate {
        true_ty.dimensions.clone()
    } else {
        predicate_ty.dimensions.clone()
    };
    for (axis, (&left, &right)) in true_ty
        .dimensions
        .iter()
        .zip(&false_ty.dimensions)
        .enumerate()
    {
        if left == UNBOUNDED_SIZE {
            dimensions[axis] = right;
        } else if right == UNBOUNDED_SIZE {
            dimensions[axis] = left;
        }
    }
    let base = if scalar_predicate { on_true } else { predicate };
    let mut result = tensor(ctx, base)?.clone();
    result.element = element;
    result.dimensions = dimensions;
    result.dynamic_dimensions = dynamic;
    if let Some(layout) = result.layout.first_mut() {
        layout.element_size_in_bits =
            shape::layout(ctx, on_true)?.map_or(0, |true_layout| true_layout.element_size_in_bits);
    }
    result.verify(ctx)?;
    Ok(TensorType::instantiate(result, ctx).into())
}

/// Infer a bitcast-convert result, including storage-width geometry changes.
pub fn infer_bitcast_convert(
    ctx: &Context,
    operand: TypeHandle,
    element: ElementType,
) -> Result<TypeHandle> {
    let source = tensor(ctx, operand)?;
    require(
        source.element.is_complex() == element.is_complex(),
        "bitcast-convert requires both types complex or both noncomplex",
    )?;
    let input_bits = source.element.storage_bit_width();
    let output_bits = element.storage_bit_width();
    require(
        input_bits.max(output_bits) % input_bits.min(output_bits) == 0,
        "element storage widths are not divisible",
    )?;
    let ratio = i64::from(input_bits.max(output_bits) / input_bits.min(output_bits));
    let mut result = source.clone();
    result.element = element;
    if input_bits > output_bits {
        result.dimensions.push(ratio);
        result.dynamic_dimensions.push(false);
        if let Some(layout) = result.layout.first_mut() {
            layout
                .minor_to_major
                .insert(0, source.dimensions.len() as i64);
        }
    } else if input_bits < output_bits {
        require(
            result.dimensions.last() == Some(&ratio),
            "last input dimension must equal storage-width ratio",
        )?;
        let deleted_axis = result.dimensions.len() - 1;
        result.dimensions.pop();
        result.dynamic_dimensions.pop();
        if let Some(layout) = result.layout.first_mut() {
            let physical_axis = layout
                .minor_to_major
                .iter()
                .position(|&axis| axis == deleted_axis as i64)
                .map(|position| layout.minor_to_major.len() - 1 - position);
            layout
                .minor_to_major
                .retain(|&axis| axis != deleted_axis as i64);
            if let Some(physical_axis) = physical_axis {
                layout.split_configs.retain_mut(|split| {
                    if split.dimension == physical_axis as i64 {
                        false
                    } else {
                        if split.dimension > physical_axis as i64 {
                            split.dimension -= 1;
                        }
                        true
                    }
                });
            }
        }
    }
    result.verify(ctx)?;
    Ok(TensorType::instantiate(result, ctx).into())
}
