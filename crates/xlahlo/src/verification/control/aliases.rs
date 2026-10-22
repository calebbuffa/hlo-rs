use super::*;

pub(super) fn indexed(ctx: &Context, shape: TypeHandle, index: &[i64]) -> Result<TypeHandle> {
    let index = index
        .iter()
        .map(|&i| {
            usize::try_from(i).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "negative shape index".into()
                })
            })
        })
        .collect::<Result<Vec<_>>>()?;
    shape::subshape(ctx, shape, &index)
}
pub(crate) fn verify_aliases(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<()> {
    if !matches!(
        opcode,
        Opcode::Call
            | Opcode::Fusion
            | Opcode::CustomCall
            | Opcode::AsyncStart
            | Opcode::AsyncUpdate
    ) {
        return Ok(());
    }
    let Some(aliases) =
        optional_attr::<OutputOperandAliases>(ctx, op, crate::attributes::keys::ALIASES)?
    else {
        return Ok(());
    };
    let target = optional_attr::<StringAttr>(ctx, op, crate::attributes::keys::TARGET)?;
    let target = target.as_ref().map(|t| t.as_str());
    let skip =
        optional_attr::<FrontendAttributes>(ctx, op, crate::attributes::keys::FRONTEND_ATTRIBUTES)?
            .is_some_and(|a| a.get("xla_skip_custom_call_alias_shape_check").is_some());
    for alias in &aliases.0 {
        let operand = usize::try_from(alias.operand_index)
            .ok()
            .and_then(|i| operands.get(i))
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "alias operand index out of range".into()
                })
            })?;
        let mut operand = indexed(ctx, *operand, &alias.operand_shape_index.0)?;
        let mut output = indexed(ctx, result, &alias.output_shape_index.0)?;
        if opcode == Opcode::CustomCall && skip {
            return Ok(());
        }
        if opcode == Opcode::CustomCall && target.is_some_and(|t| matches!(t, "Pin" | "Unpin")) {
            if operand.deref(ctx).is::<BufferType>() {
                operand = shape::array_view(ctx, operand)?;
            }
            if output.deref(ctx).is::<BufferType>() {
                output = shape::array_view(ctx, output)?;
            }
        }
        let equal = if opcode == Opcode::Fusion {
            same(ctx, operand, output)
        } else if options.layout_sensitive
            && opcode == Opcode::CustomCall
            && shape::is_array_or_buffer(ctx, operand)
            && shape::dimensions(ctx, shape::array_view(ctx, operand)?)?
                .iter()
                .all(|&d| d == 1)
        {
            same_ignoring_memory(ctx, operand, output, false)?
        } else if options.layout_sensitive && opcode == Opcode::Call {
            call_alias_equal(ctx, operand, output)?
        } else if options.layout_sensitive {
            options.same_shapes(ctx, operand, output)
        } else if opcode == Opcode::CustomCall {
            custom_call_alias_equal(ctx, operand, output)?
        } else {
            same(ctx, operand, output)
        };
        require(equal, "aliased output and operand subshapes differ")?;
    }
    Ok(())
}

// Aliasing ignores dynamic flags, but unbounded extents are not wildcard bounds.
fn custom_call_alias_equal(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    if !same(ctx, a, b) {
        return Ok(false);
    }
    let mut pending = vec![(a, b)];
    while let Some((a, b)) = pending.pop() {
        if shape::is_array_or_buffer(ctx, a)
            && *shape::dimensions(ctx, shape::array_view(ctx, a)?)?
                != *shape::dimensions(ctx, shape::array_view(ctx, b)?)?
        {
            return Ok(false);
        }
        pending.extend(
            shape::children(ctx, a)
                .iter()
                .copied()
                .zip(shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}
/// Complete fusion alias layout checks after the module caller index is built.
///
/// Set `relaxed` for a non-entry computation with no callers. An inline enclosing
/// Fusion also enables relaxation. Only memory spaces and tiles are ignored;
/// alias indices, geometry, and all other layout fields remain checked.
pub(crate) fn verify_fusion_aliases(
    ctx: &Context,
    op: Ptr<Operation>,
    options: VerifyOptions,
    relaxed: bool,
) -> Result<()> {
    require(
        Operation::is_op::<FusionOp>(op, ctx),
        "fusion alias hook requires a fusion",
    )?;
    let data = op.deref(ctx);
    require(data.get_num_results() == 1, "fusion requires one result")?;
    let Some(aliases) =
        optional_attr::<OutputOperandAliases>(ctx, op, crate::attributes::keys::ALIASES)?
    else {
        return Ok(());
    };
    let relaxed = relaxed
        || data
            .get_parent_op(ctx)
            .is_some_and(|parent| Operation::is_op::<FusionOp>(parent, ctx));
    for alias in &aliases.0 {
        let operand = usize::try_from(alias.operand_index).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "negative alias operand index".into()
            })
        })?;
        require(
            operand < data.get_num_operands(),
            "alias operand index out of range",
        )?;
        let operand = indexed(
            ctx,
            data.get_operand(operand).get_type(ctx),
            &alias.operand_shape_index.0,
        )?;
        let result = indexed(ctx, data.get_type(0), &alias.output_shape_index.0)?;
        let equal = if !options.layout_sensitive {
            same(ctx, operand, result)
        } else if relaxed {
            fusion_alias_equal(ctx, operand, result)?
        } else {
            options.same_shapes(ctx, operand, result)
        };
        require(equal, "aliased output and operand subshapes differ")?;
    }
    Ok(())
}
fn call_alias_equal(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    if !same(ctx, a, b) {
        return Ok(false);
    }
    let mut pending = vec![(a, b)];
    while let Some((a, b)) = pending.pop() {
        if !shape::is_array_or_buffer(ctx, a) {
            pending.extend(
                shape::children(ctx, a)
                    .iter()
                    .copied()
                    .zip(shape::children(ctx, b).iter().copied()),
            );
            continue;
        }
        match (shape::layout(ctx, a)?, shape::layout(ctx, b)?) {
            (Some(a), Some(b)) if !layouts_equal(ctx, &a, &b) => return Ok(false),
            (Some(_), None) | (None, Some(_)) => return Ok(false),
            _ => {}
        }
        pending.extend(
            shape::children(ctx, a)
                .iter()
                .copied()
                .zip(shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}
fn fusion_alias_equal(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
    if !shape::equal_ignoring_layout(ctx, a, b) {
        return Ok(false);
    }
    let mut pending = vec![(a, b)];
    while let Some((a, b)) = pending.pop() {
        if !shape::is_array_or_buffer(ctx, a) {
            pending.extend(
                shape::children(ctx, a)
                    .iter()
                    .copied()
                    .zip(shape::children(ctx, b).iter().copied()),
            );
            continue;
        }
        match (shape::layout(ctx, a)?, shape::layout(ctx, b)?) {
            (Some(a), Some(b)) => {
                let (mut a, mut b) = ((*a).clone(), (*b).clone());
                a.memory_space = 0;
                b.memory_space = 0;
                a.tiles.clear();
                b.tiles.clear();
                if !layouts_equal(ctx, &a, &b) {
                    return Ok(false);
                }
            }
            (Some(_), None) | (None, Some(_)) => return Ok(false),
            _ => {}
        }
        pending.extend(
            shape::children(ctx, a)
                .iter()
                .copied()
                .zip(shape::children(ctx, b).iter().copied()),
        );
    }
    Ok(true)
}
