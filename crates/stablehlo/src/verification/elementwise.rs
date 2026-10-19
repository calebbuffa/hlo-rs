use crate::types::{ElementType, QuantizedElementType, TensorType, primitive_element};
use pliron::r#type::TypeHandle;
use pliron::{
    common_traits::Verify, context::Context, op::Op, operation::Operation, result::Result,
    r#type::Typed,
};

pub(crate) fn verify_elementwise(
    op: &dyn Op,
    ctx: &Context,
    mnemonic: &str,
    arity: usize,
) -> Result<()> {
    crate::verification::verify_elementwise(op, ctx, mnemonic, arity)?;
    let ir = op.get_operation().deref(ctx);
    let types = ir
        .operands()
        .map(|value| value.get_type(ctx))
        .chain((0..ir.get_num_results()).map(|index| ir.get_result(index).get_type(ctx)))
        .collect::<Vec<_>>();
    if !compatible_witnesses(ctx, &types) {
        return crate::verification::invalid(
            op,
            ctx,
            "elementwise operand/result bounds or shapes are incompatible",
        );
    }
    Ok(())
}

fn compatible_witnesses(ctx: &Context, types: &[TypeHandle]) -> bool {
    let tensors = types
        .iter()
        .map(|ty| {
            let object = ty.deref(ctx);
            object.downcast_ref::<TensorType>().map(|tensor| {
                (
                    tensor.dimensions.clone(),
                    tensor.bounds().map(<[i64]>::to_vec),
                )
            })
        })
        .collect::<Option<Vec<_>>>();
    let Some(tensors) = tensors else {
        return false;
    };
    let Some((first_shape, _)) = tensors.first() else {
        return false;
    };
    if tensors
        .iter()
        .any(|(shape, _)| shape.len() != first_shape.len())
    {
        return false;
    }
    for axis in 0..first_shape.len() {
        let mut fixed = None;
        let mut upper = None;
        for (shape, bounds) in &tensors {
            let dimension = shape[axis];
            if dimension >= 0 {
                if fixed.is_some_and(|prior| prior != dimension) {
                    return false;
                }
                fixed = Some(dimension);
            }
            if let Some(bound) = bounds
                .as_ref()
                .and_then(|values| values.get(axis))
                .copied()
                .filter(|&value| value >= 0)
            {
                upper = Some(upper.map_or(bound, |prior: i64| prior.min(bound)));
            }
        }
        if fixed
            .zip(upper)
            .is_some_and(|(dimension, bound)| dimension > bound)
        {
            return false;
        }
    }
    true
}

pub(crate) fn verify_convert(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "convert arity mismatch");
    }
    let input = ir.get_operand(0).get_type(ctx).deref(ctx);
    let result = ir.get_result(0).get_type(ctx).deref(ctx);
    input.verify(ctx)?;
    result.verify(ctx)?;
    let (Some(a), Some(b)) = (
        input.downcast_ref::<TensorType>(),
        result.downcast_ref::<TensorType>(),
    ) else {
        return crate::verification::invalid(op, ctx, "convert requires ranked tensors");
    };
    let Some(ae) = primitive_element(ctx, a.element) else {
        return crate::verification::invalid(op, ctx, "convert requires ordinary tensor elements");
    };
    let Some(be) = primitive_element(ctx, b.element) else {
        return crate::verification::invalid(op, ctx, "convert requires ordinary tensor elements");
    };
    let allowed = |e: ElementType| {
        e != ElementType::Index
            && e != ElementType::Tf32
            && !matches!(
                e,
                ElementType::Si1
                    | ElementType::Si2
                    | ElementType::Si4
                    | ElementType::Si8
                    | ElementType::Si16
                    | ElementType::Si32
                    | ElementType::Si64
            )
    };
    if !allowed(ae)
        || !allowed(be)
        || a.dimensions.len() != b.dimensions.len()
        || a.dimensions
            .iter()
            .zip(&b.dimensions)
            .any(|(&x, &y)| x >= 0 && y >= 0 && x != y)
        || !compatible_witnesses(
            ctx,
            &[
                ir.get_operand(0).get_type(ctx),
                ir.get_result(0).get_type(ctx),
            ],
        )
    {
        return crate::verification::invalid(
            op,
            ctx,
            "convert operand and result shapes/elements are invalid",
        );
    }
    if a.dimensions.iter().all(|&d| d >= 0) && b.dimensions.iter().all(|&d| d >= 0) {
        crate::verification::one_tensor(op, ctx, 1, false)?;
    }
    Ok(())
}

pub(crate) fn valid_stablehlo_element(ctx: &Context, element: TypeHandle) -> bool {
    let ty = element.deref(ctx);
    if let Some(q) = ty.downcast_ref::<QuantizedElementType>() {
        return q.verify(ctx).is_ok()
            && matches!(
                q.storage_type(ctx),
                Some(
                    ElementType::I2
                        | ElementType::I4
                        | ElementType::I8
                        | ElementType::I16
                        | ElementType::I32
                        | ElementType::Ui2
                        | ElementType::Ui4
                        | ElementType::Ui8
                        | ElementType::Ui16
                        | ElementType::Ui32
                )
            )
            && q.expressed_type(ctx)
                .is_some_and(|expressed| expressed.is_float() && expressed != ElementType::Tf32);
    }
    primitive_element(ctx, element).is_some_and(|e| {
        e == ElementType::I1
            || valid_hlo_integer(e)
            || (e.is_float() && e != ElementType::Tf32)
            || matches!(e, ElementType::C64 | ElementType::C128)
    })
}
pub(crate) fn valid_hlo_integer(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::I2
            | ElementType::I4
            | ElementType::I8
            | ElementType::I16
            | ElementType::I32
            | ElementType::I64
            | ElementType::Ui2
            | ElementType::Ui4
            | ElementType::Ui8
            | ElementType::Ui16
            | ElementType::Ui32
            | ElementType::Ui64
    )
}
fn tensor_domain_element(ctx: &Context, element: TypeHandle) -> bool {
    let object = element.deref(ctx);
    object
        .downcast_ref::<QuantizedElementType>()
        .is_none_or(|quantized| quantized.quantization_dimension().is_none())
}
fn tensor_parts_any(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    ty.verify(ctx).ok()?;
    let t = ty.downcast_ref::<TensorType>()?;
    if t.dimensions.iter().any(|&d| d < -1) {
        return None;
    }
    Some((t.dimensions.clone(), t.element))
}

pub(crate) fn element_types_compatible(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    if a == b {
        return true;
    }
    let (a, b) = (a.deref(ctx), b.deref(ctx));
    let (Some(a), Some(b)) = (
        a.downcast_ref::<QuantizedElementType>(),
        b.downcast_ref::<QuantizedElementType>(),
    ) else {
        return false;
    };
    let (Some(a_storage), Some(b_storage)) = (a.storage_type(ctx), b.storage_type(ctx)) else {
        return false;
    };
    let effective_range = |element: ElementType, ty: &QuantizedElementType| {
        let (minimum, maximum) = element.integer_bounds()?;
        let (minimum_override, maximum_override) = ty.storage_range();
        Some((
            minimum_override.unwrap_or(minimum),
            maximum_override.unwrap_or(maximum),
        ))
    };
    a_storage == b_storage
        && effective_range(a_storage, a) == effective_range(b_storage, b)
        && a.expressed_type(ctx) == b.expressed_type(ctx)
        && a.quantization_dimension().is_some() == b.quantization_dimension().is_some()
}
fn compatible_shapes(a: &[i64], b: &[i64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| x == y || x == -1 || y == -1)
}
pub(crate) fn verify_select(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 3 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "select arity mismatch");
    }
    let (Some((ps, pe)), Some((ts, te)), Some((fs, fe)), Some((rs, re))) = (
        tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_operand(1).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_operand(2).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_result(0).get_type(ctx)),
    ) else {
        return crate::verification::invalid(op, ctx, "select requires ranked tensor types");
    };
    if !valid_stablehlo_element(ctx, pe)
        || !tensor_domain_element(ctx, pe)
        || !valid_stablehlo_element(ctx, te)
        || !tensor_domain_element(ctx, te)
        || !valid_stablehlo_element(ctx, fe)
        || !tensor_domain_element(ctx, fe)
        || !valid_stablehlo_element(ctx, re)
        || !tensor_domain_element(ctx, re)
        || primitive_element(ctx, pe) != Some(ElementType::I1)
        || !element_types_compatible(ctx, te, fe)
        || !element_types_compatible(ctx, te, re)
        || !compatible_shapes(&ts, &fs)
        || !compatible_shapes(&ts, &rs)
        || (!ps.is_empty()
            && (!compatible_shapes(&ps, &ts)
                || !compatible_shapes(&ps, &fs)
                || !compatible_shapes(&ps, &rs)))
        || !compatible_witnesses(
            ctx,
            &[
                ir.get_operand(1).get_type(ctx),
                ir.get_operand(2).get_type(ctx),
                ir.get_result(0).get_type(ctx),
            ],
        )
        || (!ps.is_empty()
            && !compatible_witnesses(
                ctx,
                &[
                    ir.get_operand(0).get_type(ctx),
                    ir.get_operand(1).get_type(ctx),
                    ir.get_operand(2).get_type(ctx),
                    ir.get_result(0).get_type(ctx),
                ],
            ))
    {
        return crate::verification::invalid(
            op,
            ctx,
            "select operand/result types are incompatible",
        );
    }
    Ok(())
}
pub(crate) fn verify_clamp(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 3 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "clamp arity mismatch");
    }
    let parts = ir
        .operands()
        .map(|v| tensor_parts_any(ctx, v.get_type(ctx)))
        .collect::<Option<Vec<_>>>();
    let Some(parts) = parts else {
        return crate::verification::invalid(op, ctx, "clamp requires ranked tensor operands");
    };
    let Some((out_shape, out_element)) = tensor_parts_any(ctx, ir.get_result(0).get_type(ctx))
    else {
        return crate::verification::invalid(op, ctx, "clamp result must be ranked");
    };
    let (value_shape, value_element) = &parts[1];
    if parts
        .iter()
        .any(|(_, e)| !valid_stablehlo_element(ctx, *e) || !tensor_domain_element(ctx, *e))
        || !valid_stablehlo_element(ctx, out_element)
        || !tensor_domain_element(ctx, out_element)
        || !element_types_compatible(ctx, *value_element, out_element)
        || parts.iter().any(|(s, e)| {
            !element_types_compatible(ctx, *e, *value_element)
                || (!s.is_empty() && !compatible_shapes(s, value_shape))
        })
        || !compatible_shapes(value_shape, &out_shape)
        || !compatible_witnesses(
            ctx,
            &ir.operands()
                .map(|value| value.get_type(ctx))
                .chain(std::iter::once(ir.get_result(0).get_type(ctx)))
                .filter(|ty| {
                    ty.deref(ctx)
                        .downcast_ref::<TensorType>()
                        .is_some_and(|tensor| !tensor.dimensions.is_empty())
                })
                .collect::<Vec<_>>(),
        )
    {
        return crate::verification::invalid(op, ctx, "clamp tensor types must be compatible");
    }
    Ok(())
}
pub(crate) fn verify_compare(
    op: &dyn Op,
    ctx: &Context,
    dir: Option<u64>,
    mode: Option<u64>,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "compare arity mismatch");
    }
    let (Some((as_, ae)), Some((bs, be)), Some((rs, re))) = (
        tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_operand(1).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_result(0).get_type(ctx)),
    ) else {
        return crate::verification::invalid(op, ctx, "compare requires ranked tensors");
    };
    let Some(dir) = dir else {
        return crate::verification::invalid(op, ctx, "compare direction attribute is invalid");
    };
    let elem = primitive_element(ctx, ae);
    let supported = valid_stablehlo_element(ctx, ae)
        && valid_stablehlo_element(ctx, be)
        && valid_stablehlo_element(ctx, re)
        && [ae, be]
            .iter()
            .all(|element| tensor_domain_element(ctx, *element));
    let quant = ae
        .deref(ctx)
        .downcast_ref::<QuantizedElementType>()
        .is_some();
    let valid_mode = supported
        && (mode.is_none()
            || match (elem, mode.unwrap_or_default(), quant) {
                (_, 0, true) => true,
                (_, 1, true) => true,
                (_, 2, true) => true,
                (_, 3, true) => true,
                (_, 4, true) => true,
                (_, 5, true) => false,
                (Some(e), 0, false) => e.is_integer(),
                (Some(e), 1, false) => e.is_float() || e.complex_component().is_some(),
                (Some(e), 2, false) => e.is_float(),
                (Some(e), 3, false) => e.is_integer(),
                (Some(e), 4, false) => e.is_integer(),
                (Some(e), 5, false) => e.is_float(),
                _ => false,
            });
    if dir > 5
        || !valid_mode
        || !element_types_compatible(ctx, ae, be)
        || !compatible_shapes(&as_, &bs)
        || primitive_element(ctx, re) != Some(ElementType::I1)
        || !compatible_shapes(&as_, &rs)
        || !compatible_witnesses(
            ctx,
            &[
                ir.get_operand(0).get_type(ctx),
                ir.get_operand(1).get_type(ctx),
                ir.get_result(0).get_type(ctx),
            ],
        )
    {
        return crate::verification::invalid(
            op,
            ctx,
            "compare operand/result types or comparison mode are invalid",
        );
    }
    Ok(())
}

pub(crate) fn verify_iota(
    op: &dyn Op,
    ctx: &Context,
    dimension: Option<usize>,
    dynamic: bool,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != usize::from(dynamic) || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "iota arity mismatch");
    }
    let Some(dimension) = dimension else {
        return crate::verification::invalid(op, ctx, "iota dimension is missing or invalid");
    };
    let result = ir.get_result(0).get_type(ctx).deref(ctx);
    if result.verify(ctx).is_err() {
        return crate::verification::invalid(op, ctx, "iota result type is invalid");
    }
    let Some(t) = result.downcast_ref::<TensorType>() else {
        return crate::verification::invalid(op, ctx, "iota result must be ranked");
    };
    if dimension >= t.dimensions.len() || (!dynamic && t.dimensions.iter().any(|&d| d < 0)) {
        return crate::verification::invalid(op, ctx, "iota dimension or shape is invalid");
    }
    let element = t.element.deref(ctx);
    if !valid_stablehlo_element(ctx, t.element) {
        return crate::verification::invalid(op, ctx, "iota result element is unsupported");
    }
    if let Some(q) = element.downcast_ref::<QuantizedElementType>() {
        if q.quantization_dimension().is_some() {
            return crate::verification::invalid(
                op,
                ctx,
                "iota does not allow per-axis quantization",
            );
        }
    } else {
        let Some(e) = primitive_element(ctx, t.element) else {
            return crate::verification::invalid(op, ctx, "iota result element is unsupported");
        };
        if e == ElementType::Index
            || e == ElementType::Tf32
            || matches!(
                e,
                ElementType::Si1
                    | ElementType::Si2
                    | ElementType::Si4
                    | ElementType::Si8
                    | ElementType::Si16
                    | ElementType::Si32
                    | ElementType::Si64
            )
            || (!dynamic && !e.is_numeric())
        {
            return crate::verification::invalid(op, ctx, "iota result element is invalid");
        }
    }
    if dynamic {
        let Some(input) = tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)) else {
            return crate::verification::invalid(
                op,
                ctx,
                "dynamic_iota dimension size must be a ranked tensor",
            );
        };
        if input.0.len() != 1
            || !primitive_element(ctx, input.1).is_some_and(ElementType::is_dimension_value)
            || input.0[0] != t.dimensions.len() as i64
        {
            return crate::verification::invalid(
                op,
                ctx,
                "dynamic_iota dimension size must contain one dimension size per result axis",
            );
        }
        if let Some(definition) = ir.get_operand(0).defining_op()
            && let Some(constant) =
                Operation::get_op_dyn(definition, ctx).downcast_ref::<crate::ops::ConstantOp>()
            && let (Some(bits), Some(splat)) = (constant.bits(ctx), constant.is_splat(ctx))
        {
            let values = if splat {
                vec![*bits.first().unwrap_or(&0); t.dimensions.len()]
            } else {
                bits
            };
            if values.len() != t.dimensions.len() {
                return crate::verification::invalid(
                    op,
                    ctx,
                    "dynamic_iota constant shape length differs from result rank",
                );
            }
            for (axis, bits) in values.into_iter().enumerate() {
                let Some(size) = primitive_element(ctx, input.1).and_then(|element| {
                    if element == ElementType::Index {
                        Some(bits as i64)
                    } else {
                        element
                            .integer_from_bits(u128::from(bits))
                            .and_then(|size| i64::try_from(size).ok())
                    }
                }) else {
                    return crate::verification::invalid(
                        op,
                        ctx,
                        "dynamic_iota constant shape element is invalid",
                    );
                };
                let bound = t
                    .bounds()
                    .and_then(|bounds| bounds.get(axis))
                    .copied()
                    .filter(|&bound| bound >= 0);
                if size < 0
                    || (t.dimensions[axis] >= 0 && t.dimensions[axis] != size)
                    || bound.is_some_and(|bound| size > bound)
                {
                    return crate::verification::invalid(
                        op,
                        ctx,
                        "dynamic_iota constant shape is incompatible with result",
                    );
                }
            }
        }
    }
    Ok(())
}

fn element_width(ctx: &Context, element: TypeHandle) -> Option<(u32, bool)> {
    let object = element.deref(ctx);
    let e = object
        .downcast_ref::<QuantizedElementType>()
        .and_then(|quantized| quantized.storage_type(ctx))
        .or_else(|| primitive_element(ctx, element))?;
    Some((e.bit_width()?, e.complex_component().is_some()))
}
pub(crate) fn verify_bitcast(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "bitcast_convert arity mismatch");
    }
    let (Some((a, ae)), Some((b, be))) = (
        tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_result(0).get_type(ctx)),
    ) else {
        return crate::verification::invalid(
            op,
            ctx,
            "bitcast_convert requires ranked ordinary tensors",
        );
    };
    if !valid_stablehlo_element(ctx, ae) || !valid_stablehlo_element(ctx, be) {
        return crate::verification::invalid(op, ctx, "bitcast_convert elements are unsupported");
    }
    let (Some((aw, ac)), Some((bw, bc))) = (element_width(ctx, ae), element_width(ctx, be)) else {
        return crate::verification::invalid(op, ctx, "bitcast_convert element width is invalid");
    };
    if ac != bc || !aw.max(bw).is_multiple_of(aw.min(bw)) {
        return crate::verification::invalid(
            op,
            ctx,
            "bitcast_convert element widths are incompatible",
        );
    }
    let axis_witness = |left_axis: usize, right_axis: usize, required: Option<i64>| {
        let left = a[left_axis];
        let right = b[right_axis];
        if required.is_some_and(|required| {
            (left >= 0 && left != required) || (right >= 0 && right != required)
        }) {
            return false;
        }
        if required.is_none() && left >= 0 && right >= 0 && left != right {
            return false;
        }
        let left_bound = ir
            .get_operand(0)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<TensorType>()
            .and_then(|tensor| tensor.bounds())
            .and_then(|bounds| bounds.get(left_axis))
            .copied()
            .filter(|&bound| bound >= 0);
        let right_bound = ir
            .get_result(0)
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<TensorType>()
            .and_then(|tensor| tensor.bounds())
            .and_then(|bounds| bounds.get(right_axis))
            .copied()
            .filter(|&bound| bound >= 0);
        if let Some(required) = required {
            return left_bound.is_none_or(|bound| required <= bound)
                && right_bound.is_none_or(|bound| required <= bound);
        }
        let fixed = if left >= 0 {
            Some(left)
        } else if right >= 0 {
            Some(right)
        } else {
            None
        };
        fixed
            .zip(left_bound)
            .is_none_or(|(size, bound)| size <= bound)
            && fixed
                .zip(right_bound)
                .is_none_or(|(size, bound)| size <= bound)
    };
    let dimension_bound = |is_result: bool, axis: usize| {
        let ty = if is_result {
            ir.get_result(0).get_type(ctx)
        } else {
            ir.get_operand(0).get_type(ctx)
        };
        ty.deref(ctx)
            .downcast_ref::<TensorType>()
            .and_then(|tensor| tensor.bounds())
            .and_then(|bounds| bounds.get(axis))
            .copied()
            .filter(|&bound| bound >= 0)
    };
    let required_minor = |shape: &[i64], is_result: bool, axis: usize, ratio: i64| {
        shape
            .get(axis)
            .is_some_and(|&dimension| dimension == ratio || dimension == -1)
            && dimension_bound(is_result, axis).is_none_or(|bound| ratio <= bound)
    };
    let valid = if aw == bw {
        a.len() == b.len() && (0..a.len()).all(|axis| axis_witness(axis, axis, None))
    } else if aw > bw {
        b.len() == a.len() + 1
            && (0..a.len()).all(|axis| axis_witness(axis, axis, None))
            && required_minor(&b, true, a.len(), (aw / bw) as i64)
    } else {
        a.len() == b.len() + 1
            && (0..b.len()).all(|axis| axis_witness(axis, axis, None))
            && required_minor(&a, false, b.len(), (bw / aw) as i64)
    };
    if !valid {
        return crate::verification::invalid(
            op,
            ctx,
            "bitcast_convert shapes do not satisfy minor-dimension rules",
        );
    }
    Ok(())
}
pub(crate) fn verify_reduce_precision(
    op: &dyn Op,
    ctx: &Context,
    attr: Option<i32>,
    mant: Option<i32>,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "reduce_precision arity mismatch");
    }
    let (Some((a, ae)), Some((b, be))) = (
        tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_result(0).get_type(ctx)),
    ) else {
        return crate::verification::invalid(op, ctx, "reduce_precision requires ranked tensors");
    };
    let af = primitive_element(ctx, ae).is_some_and(|e| e.is_float() && e != ElementType::Tf32)
        || ae
            .deref(ctx)
            .downcast_ref::<QuantizedElementType>()
            .is_some_and(|q| q.quantization_dimension().is_none());
    let bf = primitive_element(ctx, be).is_some_and(|e| e.is_float() && e != ElementType::Tf32)
        || be
            .deref(ctx)
            .downcast_ref::<QuantizedElementType>()
            .is_some_and(|q| q.quantization_dimension().is_none());
    if !(af
        && bf
        && element_types_compatible(ctx, ae, be)
        && compatible_shapes(&a, &b)
        && compatible_witnesses(
            ctx,
            &[
                ir.get_operand(0).get_type(ctx),
                ir.get_result(0).get_type(ctx),
            ],
        ))
        || attr.is_none_or(|v| v <= 0)
        || mant.is_none_or(|v| v < 0)
    {
        return crate::verification::invalid(
            op,
            ctx,
            "reduce_precision element, shape, or bit attributes are invalid",
        );
    }
    Ok(())
}
pub(crate) fn verify_uniform_quantize(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "uniform_quantize arity mismatch");
    }
    let (Some((a, ae)), Some((b, be))) = (
        tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_result(0).get_type(ctx)),
    ) else {
        return crate::verification::invalid(op, ctx, "uniform_quantize requires ranked tensors");
    };
    let result_expressed = {
        let ty = be.deref(ctx);
        ty.downcast_ref::<QuantizedElementType>()
            .map(|q| q.expressed)
    };
    let input_expressed = {
        let ty = ae.deref(ctx);
        ty.downcast_ref::<QuantizedElementType>()
            .map(|q| q.expressed)
    };
    let expressed = input_expressed.unwrap_or(ae);
    if result_expressed != Some(expressed)
        || !compatible_shapes(&a, &b)
        || !compatible_witnesses(
            ctx,
            &[
                ir.get_operand(0).get_type(ctx),
                ir.get_result(0).get_type(ctx),
            ],
        )
        || (!primitive_element(ctx, ae).is_some_and(ElementType::is_float)
            && input_expressed.is_none())
    {
        return crate::verification::invalid(
            op,
            ctx,
            "uniform_quantize input/result quantization types or shapes are incompatible",
        );
    }
    Ok(())
}
pub(crate) fn verify_uniform_dequantize(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
        return crate::verification::invalid(op, ctx, "uniform_dequantize arity mismatch");
    }
    let (Some((a, ae)), Some((b, be))) = (
        tensor_parts_any(ctx, ir.get_operand(0).get_type(ctx)),
        tensor_parts_any(ctx, ir.get_result(0).get_type(ctx)),
    ) else {
        return crate::verification::invalid(op, ctx, "uniform_dequantize requires ranked tensors");
    };
    let expressed = {
        let ty = ae.deref(ctx);
        ty.downcast_ref::<QuantizedElementType>()
            .map(|q| q.expressed)
    };
    if expressed != Some(be)
        || !primitive_element(ctx, be).is_some_and(ElementType::is_float)
        || !compatible_shapes(&a, &b)
        || !compatible_witnesses(
            ctx,
            &[
                ir.get_operand(0).get_type(ctx),
                ir.get_result(0).get_type(ctx),
            ],
        )
    {
        return crate::verification::invalid(
            op,
            ctx,
            "uniform_dequantize input/result types or shapes are incompatible",
        );
    }
    Ok(())
}
