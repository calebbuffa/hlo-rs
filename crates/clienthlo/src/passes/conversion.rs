use super::{
    builder::{Builder, ranked, scalar_words},
    is_chlo, unsupported,
};
use crate::{Result, ops as chlo};
use pliron::{
    builtin::attributes::{IntegerAttr, VecAttr},
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
        rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    r#type::Typed,
    value::Value,
};
use stablehlo::{
    attributes::ComparisonDirection as Direction,
    ops as hlo,
    types::{ElementType, TensorType, primitive_element, tensor},
};

macro_rules! broadcast_dispatch {
    ($name:expr, $builder:expr, $lhs:expr, $rhs:expr, $ty:expr;
     $($source:literal => $target:ident),* $(,)?) => {
        match $name {
            $($source => {
                let op = hlo::$target::new($builder.ctx, $lhs, $rhs, $ty);
                $builder.emit(op)
            },)*
            _ => Err(unsupported($builder.ctx, $builder.source, "no broadcast decomposition for this operation")),
        }
    };
}

fn dimensions(ctx: &Context, op: Ptr<Operation>) -> Result<Option<Vec<usize>>> {
    let expected = format!(
        "{}_dimensions",
        Operation::get_opid(op, ctx).to_string().replace('.', "_")
    );
    for (key, value) in &op.deref(ctx).attributes.0 {
        if key.to_string() == expected {
            let dims = value
                .downcast_ref::<VecAttr>()
                .and_then(|list| {
                    list.0
                        .iter()
                        .map(|value| {
                            value
                                .downcast_ref::<IntegerAttr>()?
                                .value()
                                .to_string_decimal(true)
                                .parse()
                                .ok()
                        })
                        .collect::<Option<Vec<usize>>>()
                })
                .ok_or_else(|| {
                    unsupported(
                        ctx,
                        op,
                        "broadcast dimensions must be nonnegative integer axes",
                    )
                })?;
            if dims.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(unsupported(
                    ctx,
                    op,
                    "broadcast dimensions must be strictly increasing",
                ));
            }
            return Ok(Some(dims));
        }
    }
    Ok(None)
}

const BASIC_BROADCASTS: &[&str] = &[
    "chlo.broadcast_add",
    "chlo.broadcast_and",
    "chlo.broadcast_atan2",
    "chlo.broadcast_divide",
    "chlo.broadcast_maximum",
    "chlo.broadcast_minimum",
    "chlo.broadcast_multiply",
    "chlo.broadcast_or",
    "chlo.broadcast_power",
    "chlo.broadcast_remainder",
    "chlo.broadcast_shift_left",
    "chlo.broadcast_shift_right_arithmetic",
    "chlo.broadcast_shift_right_logical",
    "chlo.broadcast_subtract",
    "chlo.broadcast_xor",
    "chlo.broadcast_complex",
    "chlo.broadcast_compare",
    "chlo.broadcast_next_after",
    "chlo.broadcast_polygamma",
    "chlo.broadcast_zeta",
];

pub(super) fn preflight(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let name = Operation::get_opid(op, ctx).to_string();
    let ir = op.deref(ctx);
    if !BASIC_BROADCASTS.contains(&name.as_str())
        && !matches!(
            name.as_str(),
            "chlo.broadcast_select"
                | "chlo.square"
                | "chlo.conj"
                | "chlo.tan"
                | "chlo.mulhi"
                | "chlo.constant"
                | "chlo.constant_like"
                | "chlo.atan"
                | "chlo.atanh"
                | "chlo.is_inf"
                | "chlo.is_pos_inf"
                | "chlo.is_neg_inf"
                | "chlo.next_after"
                | "chlo.top_k"
                | "chlo.asin"
                | "chlo.acos"
                | "chlo.acosh"
                | "chlo.asinh"
                | "chlo._asin_acos_kernel"
                | "chlo.bessel_i1e"
                | "chlo.cosh"
                | "chlo.sinh"
                | "chlo.erf"
                | "chlo.erfc"
                | "chlo.erf_inv"
                | "chlo.lgamma"
                | "chlo.digamma"
                | "chlo.polygamma"
                | "chlo.zeta"
                | "chlo.scan"
                | "chlo.ragged_dot"
        )
    {
        return Err(unsupported(
            ctx,
            op,
            "numerical or region decomposition is not implemented",
        ));
    }
    if ir.get_parent_block().is_none() {
        return Err(unsupported(
            ctx,
            op,
            "legalization requires an operation attached to a block",
        ));
    }
    for ty in ir
        .operands()
        .chain(ir.results())
        .map(|value| value.get_type(ctx))
    {
        let (_, element) = ranked(ctx, ty).ok_or_else(|| {
            unsupported(
                ctx,
                op,
                "unranked tensors cannot be legalized to ranked StableHLO",
            )
        })?;
        if primitive_element(ctx, element).is_none() && name == "chlo.scan" {
            return Err(unsupported(
                ctx,
                op,
                "upstream scan legalization requires an element type with a zero initializer",
            ));
        }
    }
    let known = |key: &str| match name.as_str() {
        n if BASIC_BROADCASTS.contains(&n) => {
            key == format!("{}_dimensions", n.replace('.', "_"))
                || (n == "chlo.broadcast_compare"
                    && matches!(key, "chlo_comparison_direction" | "chlo_comparison_type"))
        }
        "chlo.constant" | "chlo.constant_like" => key == "value",
        "chlo.top_k" => matches!(key, "chlo_top_k_k" | "chlo_top_k_is_stable"),
        "chlo.ragged_dot" => matches!(
            key,
            "chlo_ragged_dot_dimension_numbers" | "chlo_precision_config"
        ),
        "chlo.scan" => matches!(
            key,
            "chlo_scan_dimension"
                | "chlo_scan_dim_size"
                | "chlo_scan_is_reverse"
                | "chlo_scan_is_associative"
                | "chlo_scan_operand_segment_sizes"
                | "chlo_scan_result_segment_sizes"
        ),
        _ => false,
    };
    for key in ir.attributes.0.keys() {
        if !known(key.as_ref()) {
            return Err(unsupported(
                ctx,
                op,
                "operation metadata has no legalization mapping",
            ));
        }
    }
    if BASIC_BROADCASTS.contains(&name.as_str()) {
        let axes = dimensions(ctx, op)?;
        let (lhs, le) = ranked(ctx, ir.get_operand(0).get_type(ctx))
            .ok_or_else(|| unsupported(ctx, op, "broadcast operands must be ranked"))?;
        let (rhs, re) = ranked(ctx, ir.get_operand(1).get_type(ctx))
            .ok_or_else(|| unsupported(ctx, op, "broadcast operands must be ranked"))?;
        let (out, _) = ranked(ctx, ir.get_result(0).get_type(ctx))
            .ok_or_else(|| unsupported(ctx, op, "broadcast result must be ranked"))?;
        if out.len() != lhs.len().max(rhs.len())
            || axes.as_ref().is_some_and(|axes| {
                axes.len() != lhs.len().min(rhs.len()) || axes.iter().any(|&axis| axis >= out.len())
            })
        {
            return Err(unsupported(
                ctx,
                op,
                "broadcast ranks and explicit mapping must agree",
            ));
        }
        if name == "chlo.broadcast_next_after" {
            return next_after_domain(ctx, op);
        }
        if matches!(
            name.as_str(),
            "chlo.broadcast_polygamma" | "chlo.broadcast_zeta"
        ) {
            return real_float(ctx, op);
        }
        let lhs = TensorType::get(ctx, out.clone(), le).into();
        let rhs = TensorType::get(ctx, out, re).into();
        let inferred = if name == "chlo.broadcast_compare" {
            let compare = chlo::BroadcastCompareOp::from_operation(op);
            if primitive_element(ctx, le)
                .is_some_and(|element| element.complex_component().is_some())
                && !matches!(
                    compare.comparison_direction(ctx),
                    Some(
                        crate::attributes::ComparisonDirection::Eq
                            | crate::attributes::ComparisonDirection::Ne
                    )
                )
            {
                return Err(unsupported(
                    ctx,
                    op,
                    "StableHLO complex comparisons require EQ or NE",
                ));
            }
            let kind = compare
                .comparison_kind(ctx)
                .filter(|kind| *kind != crate::attributes::ComparisonKind::NoType)
                .map(Into::into);
            stablehlo::inference::infer_compare_type(ctx, lhs, rhs, kind)
        } else {
            stablehlo::inference::infer_elementwise(
                ctx,
                name.trim_start_matches("chlo.broadcast_"),
                &[lhs, rhs],
            )
        };
        inferred.map_err(|_| {
            unsupported(
                ctx,
                op,
                "operand types are outside the target StableHLO operation's domain",
            )
        })?;
        return Ok(());
    }
    match name.as_str() {
        "chlo.scan" => super::scan::preflight(ctx, op),
        "chlo.ragged_dot" => {
            let source = chlo::RaggedDotOp::from_operation(op);
            let d = source
                .dimension_numbers(ctx)
                .ok_or_else(|| unsupported(ctx, op, "missing ragged dot dimensions"))?
                .dimensions();
            if !d
                .lhs_batching_dimensions
                .contains(&d.lhs_ragged_dimensions[0])
            {
                let (shape, _) = ranked(ctx, source.group_sizes(ctx).get_type(ctx))
                    .ok_or_else(|| unsupported(ctx, op, "group sizes must be ranked"))?;
                if shape.last().is_none_or(|&n| n < 0) {
                    return Err(unsupported(
                        ctx,
                        op,
                        "dynamic ragged group count is not supported upstream",
                    ));
                }
            }
            Ok(())
        }
        "chlo.broadcast_select" => {
            let (output, _) = ranked(ctx, ir.get_result(0).get_type(ctx))
                .ok_or_else(|| unsupported(ctx, op, "select requires a ranked result"))?;
            for value in ir.operands() {
                let (shape, _) = ranked(ctx, value.get_type(ctx))
                    .ok_or_else(|| unsupported(ctx, op, "select requires ranked operands"))?;
                if shape.len() > output.len() {
                    return Err(unsupported(
                        ctx,
                        op,
                        "select operand rank exceeds the broadcast result rank",
                    ));
                }
            }
            Ok(())
        }
        "chlo.square" | "chlo.conj" | "chlo.tan" | "chlo.mulhi" => Ok(()),
        "chlo.constant" => {
            let constant = chlo::ConstantOp::from_operation(op);
            let value = constant
                .value(ctx)
                .ok_or_else(|| unsupported(ctx, op, "missing constant value"))?;
            if let Some(dense) = value.downcast_ref::<stablehlo::attributes::DenseElementsAttr>()
                && dense
                    .values
                    .0
                    .iter()
                    .any(|value| scalar_words(ctx, value).is_none())
            {
                return Err(unsupported(
                    ctx,
                    op,
                    "constant scalar representation is not supported",
                ));
            }
            Ok(())
        }
        "chlo.constant_like" => {
            let value = chlo::ConstantLikeOp::from_operation(op)
                .value(ctx)
                .ok_or_else(|| unsupported(ctx, op, "missing constant value"))?;
            if scalar_words(ctx, &value).is_none() {
                return Err(unsupported(
                    ctx,
                    op,
                    "constant scalar representation is not supported",
                ));
            }
            Ok(())
        }
        "chlo.atan"
        | "chlo.atanh"
        | "chlo.asin"
        | "chlo.acos"
        | "chlo.acosh"
        | "chlo.asinh"
        | "chlo._asin_acos_kernel"
        | "chlo.cosh"
        | "chlo.sinh" => Ok(()),
        "chlo.bessel_i1e" | "chlo.erf" | "chlo.erfc" => ieee_float(ctx, op),
        "chlo.is_inf" | "chlo.is_pos_inf" | "chlo.is_neg_inf" | "chlo.erf_inv" | "chlo.lgamma"
        | "chlo.digamma" | "chlo.polygamma" | "chlo.zeta" => real_float(ctx, op),
        "chlo.next_after" => next_after_domain(ctx, op),
        "chlo.top_k" => {
            let (shape, element) = ranked(ctx, ir.get_operand(0).get_type(ctx))
                .ok_or_else(|| unsupported(ctx, op, "top_k requires a ranked tensor"))?;
            if primitive_element(ctx, element)
                .is_some_and(|element| element.complex_component().is_some())
            {
                return Err(unsupported(
                    ctx,
                    op,
                    "top_k requires an ordered real or integer element type",
                ));
            }
            if shape.is_empty() {
                return Err(unsupported(ctx, op, "top_k requires a non-scalar input"));
            }
            Ok(())
        }
        _ => Err(unsupported(
            ctx,
            op,
            "numerical or region decomposition is not implemented",
        )),
    }
}

fn next_after_domain(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    real_float(ctx, op)?;
    let (_, element) = ranked(ctx, op.deref(ctx).get_operand(0).get_type(ctx))
        .ok_or_else(|| unsupported(ctx, op, "next_after requires ranked tensors"))?;
    if primitive_element(ctx, element).and_then(ElementType::bit_width) == Some(6) {
        return Err(unsupported(
            ctx,
            op,
            "upstream bitwise next_after requires an integer width supported by StableHLO",
        ));
    }
    Ok(())
}

fn real_float(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let (_, element) = ranked(ctx, op.deref(ctx).get_operand(0).get_type(ctx))
        .ok_or_else(|| unsupported(ctx, op, "real decomposition requires a ranked tensor"))?;
    if primitive_element(ctx, element).is_some_and(|e| e.is_float() && e != ElementType::Tf32) {
        Ok(())
    } else {
        Err(unsupported(
            ctx,
            op,
            "this decomposition requires StableHLO floating elements",
        ))
    }
}

fn ieee_float(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let (_, element) = ranked(ctx, op.deref(ctx).get_operand(0).get_type(ctx))
        .ok_or_else(|| unsupported(ctx, op, "real decomposition requires a ranked tensor"))?;
    if matches!(
        primitive_element(ctx, element),
        Some(ElementType::F16 | ElementType::Bf16 | ElementType::F32 | ElementType::F64)
    ) {
        Ok(())
    } else {
        Err(unsupported(
            ctx,
            op,
            "this decomposition currently supports real f16, bf16, f32 and f64",
        ))
    }
}

pub(super) struct Conversion;
impl DialectConversion for Conversion {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        is_chlo(ctx, op)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands: &OperandsInfo,
    ) -> Result<()> {
        let name = Operation::get_opid(op, ctx).to_string();
        let operands: Vec<_> = op.deref(ctx).operands().collect();
        let results: Vec<_> = op.deref(ctx).result_types().collect();
        let mut b = Builder {
            ctx,
            rewriter,
            source: op,
        };
        let values = if BASIC_BROADCASTS.contains(&name.as_str()) {
            let [lhs, rhs] = broadcast_pair(&mut b, operands[0], operands[1], results[0])?;
            let value = match name.as_str() {
                "chlo.broadcast_next_after" => next_after(&mut b, lhs, rhs)?,
                "chlo.broadcast_polygamma" | "chlo.broadcast_zeta" => super::math::lower(
                    &mut b,
                    name.trim_start_matches("chlo.broadcast_"),
                    &[lhs, rhs],
                )?,
                "chlo.broadcast_compare" => {
                    let source = chlo::BroadcastCompareOp::from_operation(op);
                    let direction = source
                        .comparison_direction(b.ctx)
                        .ok_or_else(|| unsupported(b.ctx, op, "comparison direction is missing"))?;
                    let kind = source
                        .comparison_kind(b.ctx)
                        .filter(|kind| *kind != crate::attributes::ComparisonKind::NoType)
                        .map(Into::into);
                    let compare =
                        hlo::CompareOp::new_inferred(b.ctx, lhs, rhs, direction.into(), kind)?;
                    b.emit(compare)?
                }
                name => broadcast_dispatch!(name, b, lhs, rhs, results[0];
                    "chlo.broadcast_add" => AddOp,
                    "chlo.broadcast_and" => AndOp,
                    "chlo.broadcast_atan2" => Atan2Op,
                    "chlo.broadcast_divide" => DivideOp,
                    "chlo.broadcast_maximum" => MaximumOp,
                    "chlo.broadcast_minimum" => MinimumOp,
                    "chlo.broadcast_multiply" => MultiplyOp,
                    "chlo.broadcast_or" => OrOp,
                    "chlo.broadcast_power" => PowerOp,
                    "chlo.broadcast_remainder" => RemainderOp,
                    "chlo.broadcast_shift_left" => ShiftLeftOp,
                    "chlo.broadcast_shift_right_arithmetic" => ShiftRightArithmeticOp,
                    "chlo.broadcast_shift_right_logical" => ShiftRightLogicalOp,
                    "chlo.broadcast_subtract" => SubtractOp,
                    "chlo.broadcast_xor" => XorOp,
                    "chlo.broadcast_complex" => ComplexOp,
                )?,
            };
            vec![value]
        } else {
            match name.as_str() {
                "chlo.constant" => {
                    let value = chlo::ConstantOp::from_operation(op)
                        .value(b.ctx)
                        .ok_or_else(|| unsupported(b.ctx, op, "constant value is missing"))?;
                    vec![b.elements(results[0], value)?]
                }
                "chlo.constant_like" => {
                    let value = chlo::ConstantLikeOp::from_operation(op)
                        .value(b.ctx)
                        .ok_or_else(|| unsupported(b.ctx, op, "constant value is missing"))?;
                    let (_, element) = ranked(b.ctx, results[0])
                        .ok_or_else(|| unsupported(b.ctx, op, "constant_like must be ranked"))?;
                    let bits = scalar_words(b.ctx, &value).ok_or_else(|| {
                        unsupported(b.ctx, op, "constant scalar representation is not supported")
                    })?;
                    let ty = TensorType::get(b.ctx, vec![], element).into();
                    let scalar = hlo::ConstantOp::new(b.ctx, ty, bits, true);
                    let scalar = b.emit(scalar)?;
                    let (shape, _) = ranked(b.ctx, results[0])
                        .ok_or_else(|| unsupported(b.ctx, op, "constant_like must be ranked"))?;
                    let shape = if shape.iter().any(|&d| d < 0) {
                        Some(b.shape(operands[0])?)
                    } else {
                        None
                    };
                    vec![b.broadcast(scalar, results[0], vec![], shape)?]
                }
                "chlo.broadcast_select" => vec![broadcast_select(&mut b, &operands, results[0])?],
                "chlo.top_k" => top_k(&mut b, operands[0], &results)?,
                "chlo.scan" => super::scan::lower(&mut b)?,
                "chlo.ragged_dot" => vec![super::ragged::lower(&mut b)?],
                "chlo.mulhi" => vec![mulhi(&mut b, operands[0], operands[1])?],
                "chlo.next_after" => vec![next_after(&mut b, operands[0], operands[1])?],
                "chlo.asin"
                | "chlo.acos"
                | "chlo.acosh"
                | "chlo.asinh"
                | "chlo._asin_acos_kernel"
                | "chlo.bessel_i1e"
                | "chlo.cosh"
                | "chlo.sinh"
                | "chlo.erf"
                | "chlo.erfc"
                | "chlo.erf_inv"
                | "chlo.lgamma"
                | "chlo.digamma"
                | "chlo.polygamma"
                | "chlo.zeta" => vec![super::math::lower(
                    &mut b,
                    name.trim_start_matches("chlo."),
                    &operands,
                )?],
                _ => vec![unary(&mut b, &name, operands[0], results[0])?],
            }
        };
        let values = values
            .into_iter()
            .zip(results)
            .map(|(value, ty)| b.convert(value, ty))
            .collect::<Result<Vec<_>>>()?;
        b.rewriter.replace_operation_with_values(b.ctx, op, values);
        Ok(())
    }
}

fn next_after(b: &mut Builder<'_>, x: Value, y: Value) -> Result<Value> {
    let (shape, element) = ranked(b.ctx, x.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "next_after requires ranked tensors"))?;
    let primitive = primitive_element(b.ctx, element).ok_or_else(|| {
        unsupported(
            b.ctx,
            b.source,
            "next_after requires floating-point elements",
        )
    })?;
    let width = primitive
        .bit_width()
        .ok_or_else(|| unsupported(b.ctx, b.source, "floating element width is missing"))?;
    let unsigned = ElementType::integer(width, pliron::builtin::types::Signedness::Unsigned)
        .ok_or_else(|| unsupported(b.ctx, b.source, "next_after needs same-width integer bits"))?;
    let bits_type = tensor(b.ctx, &shape, unsigned);
    let xb = b.bitcast(x, bits_type)?;
    let yb = b.bitcast(y, bits_type)?;
    let one = b.integer(1, unsigned)?;
    let one = b.broadcast_like(one, xb)?;
    if primitive == ElementType::F8E8M0FnUz {
        // E8M0 is unsigned and has neither zero nor a sign bit.
        let ascending = b.compare(x, y, Direction::Lt)?;
        let up = b.add(xb, one)?;
        let down = b.sub(xb, one)?;
        let bits = b.select(ascending, up, down)?;
        let stepped = b.bitcast(bits, x.get_type(b.ctx))?;
        let equal = b.compare(x, y, Direction::Eq)?;
        let stepped = b.select(equal, y, stepped)?;
        let x_nan = b.compare(x, x, Direction::Ne)?;
        let y_nan = b.compare(y, y, Direction::Ne)?;
        let nan = b.or(x_nan, y_nan)?;
        let propagated = b.add(x, y)?;
        return b.select(nan, propagated, stepped);
    }
    let sign = b.integer((1u64 << (width - 1)) as i64, unsigned)?;
    let sign = b.broadcast_like(sign, xb)?;
    let zero = b.float(0.0, x)?;
    let equal = b.compare(x, y, Direction::Eq)?;
    let is_zero = b.compare(x, zero, Direction::Eq)?;
    let positive = b.compare(x, zero, Direction::Gt)?;
    let ascending = b.compare(x, y, Direction::Lt)?;
    let increment = b.compare(positive, ascending, Direction::Eq)?;
    let up = b.add(xb, one)?;
    let down = b.sub(xb, one)?;
    let stepped = b.select(increment, up, down)?;
    let sign_y = b.and(yb, sign)?;
    let subnormal = b.or(sign_y, one)?;
    let stepped = b.select(is_zero, subnormal, stepped)?;
    let stepped = if matches!(
        primitive,
        ElementType::F8E4M3FnUz | ElementType::F8E4M3B11FnUz | ElementType::F8E5M2FnUz
    ) {
        // FNUZ reserves the negative-zero encoding for NaN.
        let negative_zero = b.compare(stepped, sign, Direction::Eq)?;
        let zero = b.integer(0, unsigned)?;
        let zero = b.broadcast_like(zero, xb)?;
        b.select(negative_zero, zero, stepped)?
    } else {
        stepped
    };
    let stepped = b.bitcast(stepped, x.get_type(b.ctx))?;
    let stepped = b.select(equal, y, stepped)?;
    let x_nan = b.compare(x, x, Direction::Ne)?;
    let y_nan = b.compare(y, y, Direction::Ne)?;
    let nan = b.or(x_nan, y_nan)?;
    let propagated = b.add(x, y)?;
    b.select(nan, propagated, stepped)
}

fn broadcast_pair(
    b: &mut Builder<'_>,
    lhs: Value,
    rhs: Value,
    result: pliron::r#type::TypeHandle,
) -> Result<[Value; 2]> {
    let (a, ae) = ranked(b.ctx, lhs.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "left operand must be ranked"))?;
    let (c, ce) = ranked(b.ctx, rhs.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "right operand must be ranked"))?;
    let (out, _) = ranked(b.ctx, result)
        .ok_or_else(|| unsupported(b.ctx, b.source, "result must be ranked"))?;
    let explicit = dimensions(b.ctx, b.source)?;
    let smaller_lhs = a.len() <= c.len();
    let axes = |rank: usize, small: bool| -> Vec<usize> {
        if small && let Some(explicit) = &explicit {
            explicit.clone()
        } else {
            (out.len() - rank..out.len()).collect()
        }
    };
    let aa = axes(a.len(), smaller_lhs);
    let ca = axes(c.len(), !smaller_lhs);
    let shape = if out.iter().any(|&extent| extent < 0) {
        let one = b.integer(1, ElementType::I64)?;
        let mut sizes = Vec::new();
        for (axis, &extent) in out.iter().enumerate() {
            let size = if extent >= 0 {
                b.integer(extent, ElementType::I64)?
            } else {
                let x = if let Some(index) = aa.iter().position(|&a| a == axis) {
                    b.dimension(lhs, index, a[index])?
                } else {
                    one
                };
                let y = if let Some(index) = ca.iter().position(|&a| a == axis) {
                    b.dimension(rhs, index, c[index])?
                } else {
                    one
                };
                let condition = b.compare(x, one, Direction::Eq)?;
                b.select(condition, y, x)?
            };
            sizes.push(size);
        }
        Some(b.shape_vector(sizes)?)
    } else {
        None
    };
    let at = TensorType::get(b.ctx, out.clone(), ae).into();
    let ct = TensorType::get(b.ctx, out, ce).into();
    Ok([
        b.broadcast(lhs, at, aa, shape)?,
        b.broadcast(rhs, ct, ca, shape)?,
    ])
}

fn broadcast_select(
    b: &mut Builder<'_>,
    operands: &[Value],
    result: pliron::r#type::TypeHandle,
) -> Result<Value> {
    let (out, _) = ranked(b.ctx, result)
        .ok_or_else(|| unsupported(b.ctx, b.source, "select requires a ranked result"))?;
    let mut shapes = Vec::new();
    for &value in operands {
        let (shape, _) = ranked(b.ctx, value.get_type(b.ctx))
            .ok_or_else(|| unsupported(b.ctx, b.source, "select requires ranked operands"))?;
        shapes.push(shape);
    }
    let shape = if out.iter().any(|&extent| extent < 0) {
        let one = b.integer(1, ElementType::I64)?;
        let mut sizes = Vec::new();
        for (axis, &extent) in out.iter().enumerate() {
            let mut size = if extent >= 0 {
                b.integer(extent, ElementType::I64)?
            } else {
                one
            };
            if extent < 0 {
                for (&value, shape) in operands.iter().zip(&shapes) {
                    if let Some(index) = axis.checked_sub(out.len() - shape.len()) {
                        let next = b.dimension(value, index, shape[index])?;
                        let condition = b.compare(size, one, Direction::Eq)?;
                        size = b.select(condition, next, size)?;
                    }
                }
            }
            sizes.push(size);
        }
        Some(b.shape_vector(sizes)?)
    } else {
        None
    };
    let mut values = Vec::new();
    for (&value, dims) in operands.iter().zip(&shapes) {
        let (_, element) = ranked(b.ctx, value.get_type(b.ctx))
            .ok_or_else(|| unsupported(b.ctx, b.source, "select requires ranked operands"))?;
        let ty = TensorType::get(b.ctx, out.clone(), element).into();
        values.push(b.broadcast(
            value,
            ty,
            (out.len() - dims.len()..out.len()).collect(),
            shape,
        )?);
    }
    b.select(values[0], values[1], values[2])
}

fn unary(
    b: &mut Builder<'_>,
    name: &str,
    x: Value,
    result: pliron::r#type::TypeHandle,
) -> Result<Value> {
    match name {
        "chlo.square" => {
            if ranked(b.ctx, x.get_type(b.ctx))
                .and_then(|(_, e)| primitive_element(b.ctx, e))
                .is_some_and(|e| e.complex_component().is_some())
            {
                let real = b.component(x, false)?;
                let imag = b.component(x, true)?;
                let same = {
                    let a = b.abs(real)?;
                    let c = b.abs(imag)?;
                    b.compare(a, c, Direction::Eq)?
                };
                let zero = b.float(0.0, real)?;
                let difference = b.sub(real, imag)?;
                let sum = b.add(real, imag)?;
                let product = b.mul(difference, sum)?;
                let r = b.select(same, zero, product)?;
                let i = b.mul(real, imag)?;
                let i = b.add(i, i)?;
                b.complex(r, i)
            } else {
                b.mul(x, x)
            }
        }
        "chlo.tan" => {
            let op = hlo::TanOp::new(b.ctx, x, result);
            b.emit(op)
        }
        "chlo.conj" => {
            let (_, element) = ranked(b.ctx, result).ok_or_else(|| {
                unsupported(b.ctx, b.source, "conjugation requires ranked tensors")
            })?;
            let element = primitive_element(b.ctx, element).ok_or_else(|| {
                unsupported(b.ctx, b.source, "conjugation requires primitive elements")
            })?;
            if let Some(component) = element.complex_component() {
                let (shape, _) = ranked(b.ctx, result).ok_or_else(|| {
                    unsupported(b.ctx, b.source, "conjugation requires ranked tensors")
                })?;
                let ty = tensor(b.ctx, &shape, component);
                let real = hlo::RealOp::new(b.ctx, x, ty);
                let real = b.emit(real)?;
                let imaginary = hlo::ImagOp::new(b.ctx, x, ty);
                let imaginary = b.emit(imaginary)?;
                let imaginary = b.neg(imaginary)?;
                let op = hlo::ComplexOp::new(b.ctx, real, imaginary, result);
                b.emit(op)
            } else {
                Ok(x)
            }
        }
        "chlo.atan" => {
            if ranked(b.ctx, x.get_type(b.ctx))
                .and_then(|(_, e)| primitive_element(b.ctx, e))
                .is_some_and(|e| e.complex_component().is_some())
            {
                return super::math::lower(b, "atan", &[x]);
            }
            let one = b.float(1.0, x)?;
            b.atan2(x, one)
        }
        "chlo.atanh" => {
            if ranked(b.ctx, x.get_type(b.ctx))
                .and_then(|(_, e)| primitive_element(b.ctx, e))
                .is_some_and(|e| e.complex_component().is_some())
            {
                return super::math::lower(b, "atanh", &[x]);
            }
            let half = b.float(0.5, x)?;
            let positive = b.log1p(x)?;
            let negative = b.neg(x)?;
            let negative = b.log1p(negative)?;
            let difference = b.sub(positive, negative)?;
            b.mul(half, difference)
        }
        "chlo.is_inf" | "chlo.is_pos_inf" | "chlo.is_neg_inf" => {
            let x = if name == "chlo.is_inf" { b.abs(x)? } else { x };
            let infinity = b.float(
                if name == "chlo.is_neg_inf" {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                },
                x,
            )?;
            b.compare(x, infinity, Direction::Eq)
        }
        _ => Err(unsupported(
            b.ctx,
            b.source,
            "unimplemented unary decomposition",
        )),
    }
}

fn top_k(
    b: &mut Builder<'_>,
    input: Value,
    results: &[pliron::r#type::TypeHandle],
) -> Result<Vec<Value>> {
    use pliron::basic_block::BasicBlock;
    let (shape, element) = ranked(b.ctx, input.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "top_k requires ranked tensors"))?;
    let axis = shape.len() - 1;
    let index_type = tensor(b.ctx, &shape, ElementType::I32);
    let dynamic = shape.iter().any(|&d| d < 0);
    let indices = if dynamic {
        let sizes = b.shape(input)?;
        let iota = hlo::DynamicIotaOp::new(b.ctx, sizes, index_type, axis);
        b.emit(iota)?
    } else {
        let iota = hlo::IotaOp::new(b.ctx, index_type, axis);
        b.emit(iota)?
    };
    let stable = chlo::TopKOp::from_operation(b.source).is_stable(b.ctx);
    let sort = hlo::SortOp::new(
        b.ctx,
        vec![input, indices],
        vec![input.get_type(b.ctx), index_type],
        axis as u64,
        stable,
    );
    let scalar = TensorType::get(b.ctx, vec![], element).into();
    let index_scalar = tensor(b.ctx, &[], ElementType::I32);
    let block = BasicBlock::new(
        b.ctx,
        None,
        vec![scalar, scalar, index_scalar, index_scalar],
    );
    block.insert_at_back(sort.comparator_region(b.ctx), b.ctx);
    let arguments: Vec<_> = block.deref(b.ctx).arguments().collect();
    let predicate = hlo::CompareOp::new_inferred(
        b.ctx,
        arguments[0],
        arguments[1],
        Direction::Gt,
        primitive_element(b.ctx, element)
            .filter(|e| e.is_float())
            .map(|_| stablehlo::attributes::ComparisonKind::TotalOrder),
    )?;
    let predicate_value = predicate.get_operation().deref(b.ctx).get_result(0);
    predicate.get_operation().insert_at_back(block, b.ctx);
    hlo::ReturnOp::new(b.ctx, predicate_value)
        .get_operation()
        .insert_at_back(block, b.ctx);
    let sort_operation = sort.get_operation();
    b.emit(sort)?;
    let sorted: Vec<_> = sort_operation.deref(b.ctx).results().collect();
    let mut values = Vec::new();
    for (value, &result) in sorted.into_iter().zip(results) {
        let (output, _) = ranked(b.ctx, result)
            .ok_or_else(|| unsupported(b.ctx, b.source, "top_k requires ranked results"))?;
        if dynamic {
            let zero = b.integer(0, ElementType::I64)?;
            let one = b.integer(1, ElementType::I64)?;
            let start = b.shape_vector(vec![zero; shape.len()])?;
            let strides = b.shape_vector(vec![one; shape.len()])?;
            let mut limits = Vec::new();
            for (dim, &extent) in output.iter().enumerate() {
                limits.push(b.dimension(input, dim, extent)?);
            }
            let limits = b.shape_vector(limits)?;
            let slice = hlo::RealDynamicSliceOp::new(b.ctx, value, start, limits, strides, result);
            values.push(b.emit(slice)?);
        } else {
            let limits = output.iter().map(|&d| d as u64).collect();
            let slice = hlo::SliceOp::new(
                b.ctx,
                value,
                result,
                vec![0; shape.len()],
                limits,
                vec![1; shape.len()],
            );
            values.push(b.emit(slice)?);
        }
    }
    Ok(values)
}

fn mulhi(b: &mut Builder<'_>, lhs: Value, rhs: Value) -> Result<Value> {
    let (shape, element) = ranked(b.ctx, lhs.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "mulhi requires ranked tensors"))?;
    let element = primitive_element(b.ctx, element)
        .ok_or_else(|| unsupported(b.ctx, b.source, "mulhi requires integer elements"))?;
    let (width, signedness) = element
        .integer_parts()
        .ok_or_else(|| unsupported(b.ctx, b.source, "mulhi requires integer elements"))?;
    let signed = signedness != pliron::builtin::types::Signedness::Unsigned;
    if width < 64 {
        let wide = ElementType::integer(width * 2, signedness)
            .ok_or_else(|| unsupported(b.ctx, b.source, "no double-width integer type"))?;
        let ty = tensor(b.ctx, &shape, wide);
        let a = b.convert(lhs, ty)?;
        let c = b.convert(rhs, ty)?;
        let product = b.mul(a, c)?;
        let amount = b.integer(i64::from(width), wide)?;
        let amount = b.broadcast_like(amount, a)?;
        let high = if signed {
            b.sar(product, amount)?
        } else {
            b.shr(product, amount)?
        };
        return b.convert(high, lhs.get_type(b.ctx));
    }
    let ty = tensor(b.ctx, &shape, ElementType::Ui64);
    let a = b.convert(lhs, ty)?;
    let c = b.convert(rhs, ty)?;
    let mask = b.integer(0xffff_ffff, ElementType::Ui64)?;
    let mask = b.broadcast_like(mask, a)?;
    let amount = b.integer(32, ElementType::Ui64)?;
    let amount = b.broadcast_like(amount, a)?;
    let al = b.and(a, mask)?;
    let cl = b.and(c, mask)?;
    let ah = b.shr(a, amount)?;
    let ch = b.shr(c, amount)?;
    let low = b.mul(al, cl)?;
    let carry = b.shr(low, amount)?;
    let cross = b.mul(al, ch)?;
    let middle = b.add(cross, carry)?;
    let middle_low = b.and(middle, mask)?;
    let other_cross = b.mul(ah, cl)?;
    let other_middle = b.add(other_cross, middle_low)?;
    let upper = b.mul(ah, ch)?;
    let carry1 = b.shr(middle, amount)?;
    let carry2 = b.shr(other_middle, amount)?;
    let high = b.add(upper, carry1)?;
    let mut high = b.add(high, carry2)?;
    if signed {
        let zero = b.integer(0, element)?;
        let zero = b.broadcast_like(zero, lhs)?;
        let negative_a = b.compare(lhs, zero, Direction::Lt)?;
        let negative_c = b.compare(rhs, zero, Direction::Lt)?;
        let zero_unsigned = b.convert(zero, ty)?;
        let correction = b.select(negative_a, c, zero_unsigned)?;
        high = b.sub(high, correction)?;
        let correction = b.select(negative_c, a, zero_unsigned)?;
        high = b.sub(high, correction)?;
    }
    b.convert(high, lhs.get_type(b.ctx))
}
