//! Native elementwise inference.
use crate::Diagnostic;
use crate::types::{ComplexType, QuantizedElementType, TensorType, primitive_element};
use crate::{attributes::TypeExtensionsAttr, types::ElementType};
use pliron::attribute::AttrObj;
use pliron::{context::Context, result::Result, r#type::TypeHandle};

/// Infers an elementwise result for the operation identified by `mnemonic`.
///
/// Inputs must be supported ranked tensors with compatible ranks, element
/// types, and dimensions; dynamic dimensions (`-1`) may match known extents.
/// The mnemonic controls the accepted element domain and result element type
/// (for example, `is_finite` returns `i1`, while `real` on a complex tensor
/// returns its component type). Empty inputs, unsupported mnemonics/domains,
/// and incompatible shapes or elements return an error.
pub fn infer_elementwise(
    ctx: &Context,
    mnemonic: &str,
    inputs: &[TypeHandle],
) -> Result<TypeHandle> {
    let first = *inputs.first().ok_or_else(|| {
        Diagnostic::EmptyOperands {
            operation: "elementwise",
            role: "input",
        }
        .into_error()
    })?;
    let first_ty = first.deref(ctx);
    first_ty.verify(ctx)?;
    let tensor = first_ty
        .downcast_ref::<TensorType>()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if tensor.dimensions.iter().any(|&d| d < -1) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    if !crate::verification::elementwise::valid_stablehlo_element(ctx, tensor.element) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let first_element_ty = tensor.element.deref(ctx);
    let first_element = primitive_element(ctx, tensor.element);
    let q = first_element_ty.downcast_ref::<QuantizedElementType>();
    let quantized = q.is_some_and(|q| q.quantization_dimension().is_none());
    let quantized_any = q.is_some();
    let complex = first_element_ty
        .downcast_ref::<ComplexType>()
        .map(|c| c.component);
    let valid_complex = matches!(first_element, Some(ElementType::C64 | ElementType::C128));
    let is_float = first_element.is_some_and(|e| e.is_float() && e != ElementType::Tf32);
    let is_integer = first_element.is_some_and(crate::verification::elementwise::valid_hlo_integer);
    let is_predicate = first_element == Some(ElementType::I1);
    let unsigned = first_element.is_some_and(ElementType::is_unsigned_integer);
    let numeric = is_float || valid_complex || quantized || (is_integer && !is_predicate);
    let allowed = match mnemonic {
        "abs" | "sign" => {
            (is_integer && !unsigned && !is_predicate) || is_float || valid_complex || quantized
        }
        "negate" | "divide" | "power" | "remainder" | "subtract" => numeric || is_integer,
        "reduce_precision" => is_float || quantized,
        "ceil" | "floor" | "round_nearest_afz" | "round_nearest_even" | "is_finite" => {
            is_float || quantized
        }
        "not" | "and" | "or" | "xor" => is_integer || is_predicate,
        "count_leading_zeros"
        | "popcnt"
        | "shift_left"
        | "shift_right_arithmetic"
        | "shift_right_logical" => is_integer,
        "real" | "imag" => is_float || valid_complex,
        "complex" => matches!(first_element, Some(ElementType::F32 | ElementType::F64)),
        "add" => numeric || quantized_any || is_predicate || is_integer,
        "maximum" | "minimum" | "multiply" => numeric || is_predicate || is_integer,
        _ => is_float || valid_complex || quantized,
    };
    if !allowed {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let (shape, output_encoding) = if inputs.len() == 1 {
        let (shape, element, encoding) = parts(ctx, first)?;
        if !crate::verification::elementwise::element_types_compatible(ctx, element, tensor.element)
        {
            return Err(Diagnostic::IncompatibleElement.into());
        }
        (shape, encoding)
    } else {
        merge_shape_types(ctx, inputs)?
    };
    let element = match mnemonic {
        "abs" | "real" | "imag" if valid_complex => {
            let Some(component) = complex else {
                return Err(Diagnostic::InternalInvariant {
                    context: "complex element component missing after validation",
                }
                .into());
            };
            component
        }
        "is_finite" => ElementType::I1.native(ctx),
        "complex" => ComplexType::get(ctx, tensor.element).into(),
        _ => tensor.element,
    };
    let output = result_tensor(ctx, shape, element, output_encoding);
    checked_result(ctx, output)
}

fn result_tensor(
    ctx: &Context,
    shape: Vec<i64>,
    element: TypeHandle,
    encoding: Option<AttrObj>,
) -> TypeHandle {
    let result = TensorType::get(ctx, shape, element);
    if let Some(encoding) = encoding {
        result.deref(ctx).with_encoding(ctx, encoding)
    } else {
        result.into()
    }
}
fn checked_result(ctx: &Context, result: TypeHandle) -> Result<TypeHandle> {
    result.deref(ctx).verify(ctx)?;
    Ok(result)
}
fn parts(ctx: &Context, ty: TypeHandle) -> Result<(Vec<i64>, TypeHandle, Option<AttrObj>)> {
    let object = ty.deref(ctx);
    object.verify(ctx)?;
    let tensor = object
        .downcast_ref::<TensorType>()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    if tensor.dimensions.iter().any(|&d| d < -1) {
        return Err(Diagnostic::InvalidDimensions.into());
    }
    if !crate::verification::elementwise::valid_stablehlo_element(ctx, tensor.element) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    Ok((
        tensor.dimensions.clone(),
        tensor.element,
        tensor.encoding().cloned(),
    ))
}
fn merge_shape_types(ctx: &Context, types: &[TypeHandle]) -> Result<(Vec<i64>, Option<AttrObj>)> {
    let first = *types
        .first()
        .ok_or_else(|| Diagnostic::UnsupportedType.into_error())?;
    let (mut sizes, _, first_encoding) = parts(ctx, first)?;
    let rank = sizes.len();
    let any_bounds = types.iter().any(|ty| {
        ty.deref(ctx)
            .downcast_ref::<TensorType>()
            .is_some_and(|tensor| tensor.bounds().is_some())
    });
    let mut bounds = (0..rank)
        .map(|axis| bound_for_axis(ctx, first, axis))
        .collect::<Vec<_>>();
    for &ty in &types[1..] {
        let (shape, element, _) = parts(ctx, ty)?;
        let (_, first_element, _) = parts(ctx, first)?;
        if !crate::verification::elementwise::element_types_compatible(ctx, first_element, element)
        {
            return Err(Diagnostic::IncompatibleElement.into());
        }
        if shape.len() != rank {
            return Err(Diagnostic::IncompatibleShape.into());
        }
        for axis in 0..rank {
            let left = sizes[axis];
            let right = shape[axis];
            let left_bound = bounds[axis];
            let right_bound = bound_for_axis(ctx, ty, axis);
            match (left >= 0, right >= 0) {
                (true, true) if left != right => return Err(Diagnostic::IncompatibleShape.into()),
                (true, _) => {
                    if [left_bound, right_bound]
                        .into_iter()
                        .flatten()
                        .any(|b| left > b)
                    {
                        return Err(Diagnostic::IncompatibleShape.into());
                    }
                    bounds[axis] = None;
                }
                (_, true) => {
                    if [left_bound, right_bound]
                        .into_iter()
                        .flatten()
                        .any(|b| right > b)
                    {
                        return Err(Diagnostic::IncompatibleShape.into());
                    }
                    sizes[axis] = right;
                    bounds[axis] = None;
                }
                _ => {
                    bounds[axis] = match (left_bound, right_bound) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    };
                }
            }
        }
    }
    let encoding = if any_bounds {
        bounds.iter().any(Option::is_some).then(|| {
            Box::new(TypeExtensionsAttr {
                bounds: bounds.into_iter().map(|b| b.unwrap_or(-1)).collect(),
            }) as AttrObj
        })
    } else {
        first_encoding
    };
    Ok((sizes, encoding))
}
fn bound_for_axis(ctx: &Context, ty: TypeHandle, axis: usize) -> Option<i64> {
    ty.deref(ctx)
        .downcast_ref::<TensorType>()?
        .bounds()?
        .get(axis)
        .copied()
        .filter(|&bound| bound >= 0)
}

/// Infers the elementwise absolute-value result type for a supported tensor.
///
/// The input element type must be a supported signed integer, floating-point,
/// complex, or uniform-quantized type.
pub fn infer_abs_type(ctx: &Context, operand: TypeHandle) -> Result<TypeHandle> {
    infer_elementwise(ctx, "abs", &[operand])
}
/// Infers the real-part result type of a floating-point or complex tensor.
///
/// A complex element type becomes its floating-point component type; a
/// floating-point element type is preserved.
pub fn infer_real_type(ctx: &Context, operand: TypeHandle) -> Result<TypeHandle> {
    infer_elementwise(ctx, "real", &[operand])
}
/// Infers the imaginary-part result type of a floating-point or complex tensor.
///
/// A complex element type becomes its floating-point component type; a
/// floating-point element type is preserved.
pub fn infer_imag_type(ctx: &Context, operand: TypeHandle) -> Result<TypeHandle> {
    infer_elementwise(ctx, "imag", &[operand])
}
/// Infers the `is_finite` result, preserving shape and replacing the element
/// type with `i1`.
///
/// The operand must have a supported floating-point or uniform-quantized
/// element type.
pub fn infer_is_finite_type(ctx: &Context, operand: TypeHandle) -> Result<TypeHandle> {
    infer_elementwise(ctx, "is_finite", &[operand])
}
/// Infers the complex result formed from matching `f32` or `f64` components.
///
/// The operands must have compatible tensor shapes and identical component
/// element types; the resulting elements are `complex<f32>` or `complex<f64>`.
pub fn infer_complex_type(ctx: &Context, lhs: TypeHandle, rhs: TypeHandle) -> Result<TypeHandle> {
    let (shape, element, encoding) = parts(ctx, lhs)?;
    let (_other_shape, other_element, _) = parts(ctx, rhs)?;
    let _ = merge_shape_types(ctx, &[lhs, rhs])?;
    if element != other_element {
        return Err(Diagnostic::IncompatibleElement.into());
    }
    let component = primitive_element(ctx, element)
        .filter(|e| matches!(e, ElementType::F32 | ElementType::F64))
        .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?;
    let complex = ElementType::complex(component)
        .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?
        .native(ctx);
    checked_result(ctx, result_tensor(ctx, shape, complex, encoding))
}
/// Infers the boolean tensor result of comparing two compatible tensors.
///
/// `comparison` selects the requested comparison domain; `None` leaves the
/// comparison kind unspecified. Inputs must have compatible shapes and
/// compatible element types. Per-axis quantized inputs are not accepted.
pub fn infer_compare_type(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    comparison: Option<crate::attributes::ComparisonKind>,
) -> Result<TypeHandle> {
    let (ls, le, encoding) = parts(ctx, lhs)?;
    let (_rs, re, _) = parts(ctx, rhs)?;
    let _ = merge_shape_types(ctx, &[lhs, rhs])?;
    if !crate::verification::elementwise::element_types_compatible(ctx, le, re) {
        return Err(Diagnostic::IncompatibleElement.into());
    }
    let object = le.deref(ctx);
    let element = primitive_element(ctx, le);
    let quantized = object
        .downcast_ref::<QuantizedElementType>()
        .is_some_and(|q| q.quantization_dimension().is_none());
    let per_axis_quantized = object
        .downcast_ref::<QuantizedElementType>()
        .is_some_and(|q| q.quantization_dimension().is_some());
    if per_axis_quantized {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let valid = match (element, comparison, quantized) {
        (_, None, _) => true,
        (
            Some(e),
            Some(
                crate::attributes::ComparisonKind::NoType
                | crate::attributes::ComparisonKind::Signed
                | crate::attributes::ComparisonKind::Unsigned,
            ),
            false,
        ) => e.is_integer(),
        (Some(e), Some(crate::attributes::ComparisonKind::Float), false) => {
            e.is_float() || e.complex_component().is_some()
        }
        (
            Some(e),
            Some(
                crate::attributes::ComparisonKind::TotalOrder
                | crate::attributes::ComparisonKind::WeakOrder,
            ),
            false,
        ) => e.is_float(),
        (_, Some(crate::attributes::ComparisonKind::WeakOrder), true) => false,
        (_, Some(_), true) => true,
        _ => false,
    };
    if !valid {
        return Err(Diagnostic::InvalidDomain.into());
    }
    if element.is_none() && !quantized {
        return Err(Diagnostic::InvalidDomain.into());
    }
    checked_result(
        ctx,
        result_tensor(ctx, ls, ElementType::I1.native(ctx), encoding),
    )
}
/// Infers the result type of selecting between two value tensors.
///
/// The predicate must be an `i1` tensor. The value operands must have
/// compatible elements and shapes; a non-scalar predicate must also have a
/// shape compatible with the values. Per-axis quantized values are rejected.
pub fn infer_select_type(
    ctx: &Context,
    predicate: TypeHandle,
    on_true: TypeHandle,
    on_false: TypeHandle,
) -> Result<TypeHandle> {
    let (ps, pe, _) = parts(ctx, predicate)?;
    if primitive_element(ctx, pe) != Some(ElementType::I1) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let (_ts, te, _encoding) = parts(ctx, on_true)?;
    let (_fs, fe, _) = parts(ctx, on_false)?;
    if [te, fe].iter().any(|element| {
        element
            .deref(ctx)
            .downcast_ref::<QuantizedElementType>()
            .is_some_and(|q| q.quantization_dimension().is_some())
    }) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    if !crate::verification::elementwise::element_types_compatible(ctx, te, fe) {
        return Err(Diagnostic::IncompatibleElement.into());
    }
    let value_result = merge_shape_types(ctx, &[on_true, on_false])?;
    if !ps.is_empty() {
        let pred_witness = result_tensor(ctx, ps, te, parts(ctx, predicate)?.2);
        let _ = merge_shape_types(ctx, &[on_true, on_false, pred_witness])?;
    }
    let (shape, merged_encoding) = value_result;
    checked_result(ctx, result_tensor(ctx, shape, te, merged_encoding))
}
/// Infers the result type of clamping `operand` between `min` and `max`.
///
/// Bound elements must be compatible with the operand element type, and
/// non-scalar bounds must have shapes compatible with the operand. The result
/// preserves the operand's shape, element type, and encoding.
pub fn infer_clamp_type(
    ctx: &Context,
    min: TypeHandle,
    operand: TypeHandle,
    max: TypeHandle,
) -> Result<TypeHandle> {
    let (shape, element, encoding) = parts(ctx, operand)?;
    let mut witnesses = vec![operand];
    for bound in [min, max] {
        let (bound_shape, bound_element, _) = parts(ctx, bound)?;
        if [element, bound_element].iter().any(|element| {
            element
                .deref(ctx)
                .downcast_ref::<QuantizedElementType>()
                .is_some_and(|q| q.quantization_dimension().is_some())
        }) {
            return Err(Diagnostic::InvalidDomain.into());
        }
        if !crate::verification::elementwise::element_types_compatible(ctx, element, bound_element)
        {
            return Err(Diagnostic::IncompatibleElement.into());
        }
        if !bound_shape.is_empty() {
            witnesses.push(bound);
        }
    }
    let _ = merge_shape_types(ctx, &witnesses)?;
    checked_result(ctx, result_tensor(ctx, shape, element, encoding))
}
/// Infers a conversion result with the requested primitive `element` type.
///
/// The input must have a supported primitive element type. Index, `tf32`, and
/// signless integer element types are not accepted as either source or target.
/// The result preserves shape and tensor encoding.
pub fn infer_convert_type(
    ctx: &Context,
    operand: TypeHandle,
    element: ElementType,
) -> Result<TypeHandle> {
    let (shape, input, encoding) = parts(ctx, operand)?;
    let allowed = |element: ElementType| {
        element != ElementType::Index
            && element != ElementType::Tf32
            && !matches!(
                element,
                ElementType::Si1
                    | ElementType::Si2
                    | ElementType::Si4
                    | ElementType::Si8
                    | ElementType::Si16
                    | ElementType::Si32
                    | ElementType::Si64
            )
    };
    if !primitive_element(ctx, input).is_some_and(allowed) || !allowed(element) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    checked_result(
        ctx,
        result_tensor(ctx, shape, element.native(ctx), encoding),
    )
}
/// Infers a uniform dequantization result using the quantized type's expressed
/// element type.
///
/// The operand must be a tensor with a uniform quantized element type. The
/// result keeps its shape and drops the tensor encoding.
pub fn infer_uniform_dequantize_type(ctx: &Context, operand: TypeHandle) -> Result<TypeHandle> {
    let (shape, element, _) = parts(ctx, operand)?;
    let expressed = {
        let ty = element.deref(ctx);
        ty.downcast_ref::<QuantizedElementType>()
            .map(|q| q.expressed)
            .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?
    };
    checked_result(ctx, result_tensor(ctx, shape, expressed, None))
}
/// Infers a uniform quantization result using the quantized element type in
/// `result_template`.
///
/// The operand must be floating-point (except `tf32`) or already quantized,
/// and its expressed type must match the template's expressed type. The
/// operand shape is used for the result; the template shape is not copied.
pub fn infer_uniform_quantize_type(
    ctx: &Context,
    operand: TypeHandle,
    result_template: TypeHandle,
) -> Result<TypeHandle> {
    let (shape, input, _) = parts(ctx, operand)?;
    let (_result_shape, result_element, _) = parts(ctx, result_template)?;
    let expressed_result = {
        let ty = result_element.deref(ctx);
        ty.downcast_ref::<QuantizedElementType>()
            .map(|q| q.expressed)
            .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?
    };
    let expressed = {
        let ty = input.deref(ctx);
        ty.downcast_ref::<QuantizedElementType>()
            .map_or(input, |q| q.expressed)
    };
    let valid = primitive_element(ctx, input)
        .is_some_and(|e| e.is_float() && e != ElementType::Tf32)
        || input
            .deref(ctx)
            .downcast_ref::<QuantizedElementType>()
            .is_some();
    if !valid || expressed_result != expressed {
        return Err(Diagnostic::InvalidDomain.into());
    }
    checked_result(ctx, result_tensor(ctx, shape, result_element, None))
}
/// Infers a bitcast-conversion shape and element type from an operand and a
/// result template.
///
/// The element widths must divide one another, complex and non-complex domains
/// cannot be mixed, and `index`/`tf32` are unsupported. Narrowing appends a
/// dimension for the width ratio; widening consumes a final dimension equal
/// to that ratio. The result uses the template element type.
pub fn infer_bitcast_convert_type(
    ctx: &Context,
    operand: TypeHandle,
    result_template: TypeHandle,
) -> Result<TypeHandle> {
    let (mut shape, input, _) = parts(ctx, operand)?;
    let (_, output, _) = parts(ctx, result_template)?;
    let mut bounds = (0..shape.len())
        .map(|axis| bound_for_axis(ctx, operand, axis))
        .collect::<Vec<_>>();
    let bitcast_element = |element: TypeHandle| {
        let object = element.deref(ctx);
        object
            .downcast_ref::<QuantizedElementType>()
            .and_then(|q| q.storage_type(ctx))
            .or_else(|| primitive_element(ctx, element))
    };
    let (input_element, input_complex) = {
        let e = bitcast_element(input).ok_or_else(|| Diagnostic::InvalidDomain.into_error())?;
        (e, e.complex_component().is_some())
    };
    let (output_element, output_complex) = {
        let e = bitcast_element(output).ok_or_else(|| Diagnostic::InvalidDomain.into_error())?;
        (e, e.complex_component().is_some())
    };
    if input_complex != output_complex
        || matches!(input_element, ElementType::Index | ElementType::Tf32)
        || matches!(output_element, ElementType::Index | ElementType::Tf32)
    {
        return Err(Diagnostic::InvalidDomain.into());
    }
    let (iw, ow) = (
        input_element
            .bit_width()
            .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?,
        output_element
            .bit_width()
            .ok_or_else(|| Diagnostic::InvalidDomain.into_error())?,
    );
    if !iw.max(ow).is_multiple_of(iw.min(ow)) {
        return Err(Diagnostic::InvalidDomain.into());
    }
    if iw > ow {
        shape.push(i64::from(iw / ow));
        bounds.push(None);
    } else if iw < ow {
        let Some(last) = shape.pop() else {
            return Err(Diagnostic::InvalidDimensions.into());
        };
        let required = i64::from(ow / iw);
        if (last >= 0 && last != required)
            || bounds
                .last()
                .copied()
                .flatten()
                .is_some_and(|bound| required > bound)
        {
            return Err(Diagnostic::InvalidDimensions.into());
        }
        bounds.pop();
    }
    let encoding = bounds.iter().any(Option::is_some).then(|| {
        Box::new(TypeExtensionsAttr {
            bounds: bounds.into_iter().map(|b| b.unwrap_or(-1)).collect(),
        }) as AttrObj
    });
    checked_result(ctx, result_tensor(ctx, shape, output, encoding))
}
