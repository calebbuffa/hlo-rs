use super::*;

pub(crate) fn opcode(ctx: &Context, op: Ptr<Operation>) -> Option<Opcode> {
    let dynamic = Operation::get_op_dyn(op, ctx);
    op_cast::<dyn HloOpInterface>(dynamic.as_ref()).map(|hlo| hlo.opcode())
}

pub(crate) fn copy_prefetch_index(ctx: &Context, op: Ptr<Operation>) -> Result<Option<i32>> {
    match optional_attr::<OptionalInteger>(ctx, op, "hlo_cross_program_prefetch_index")?.as_deref()
    {
        None | Some(OptionalInteger::Absent) => Ok(None),
        Some(OptionalInteger::Value(index)) => i32::try_from(*index).map(Some).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "hlo_cross_program_prefetch_index",
                message: format!("prefetch index {index} is outside the signed int32 domain"),
            })
        }),
    }
}

pub(crate) fn feed_config(ctx: &Context, op: Ptr<Operation>, key: &'static str) -> Result<String> {
    Ok(optional_attr::<StringAttr>(ctx, op, key)?
        .map(|config| config.as_str().to_owned())
        .unwrap_or_default())
}

pub(super) fn same_frontend_attributes(
    ctx: &Context,
    left: Ptr<Operation>,
    right: Ptr<Operation>,
) -> Result<bool> {
    let left = optional_attr::<FrontendAttributes>(
        ctx,
        left,
        crate::attributes::keys::FRONTEND_ATTRIBUTES,
    )?;
    let right = optional_attr::<FrontendAttributes>(
        ctx,
        right,
        crate::attributes::keys::FRONTEND_ATTRIBUTES,
    )?;
    fn entries(attributes: Option<&FrontendAttributes>) -> BTreeMap<&str, &str> {
        attributes
            .into_iter()
            .flat_map(|attributes| &attributes.0)
            .map(|pair| (pair.key.as_str(), pair.value.as_str()))
            .collect()
    }
    Ok(entries(left.as_deref()) == entries(right.as_deref()))
}

pub(in crate::verification::control) fn infer(
    ctx: &Context,
    op: Ptr<Operation>,
    kind: Opcode,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<TypeHandle> {
    match kind {
        CopyStart => {
            copy_prefetch_index(ctx, op)?;
            infer::infer_copy_start(ctx, operands[0])
        }
        CopyDone => {
            let destination = infer::infer_copy_done(ctx, operands[0])?;
            if options.layout_sensitive {
                require(
                    same_ignoring_memory(
                        ctx,
                        destination,
                        infer::infer_get_tuple_element(ctx, operands[0], 1)?,
                        true,
                    )?,
                    "copy context source and destination differ",
                )?;
            }
            Ok(destination)
        }
        Send => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            infer::infer_send(ctx, operands[0])
        }
        SendDone => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            Ok(shape::token(ctx))
        }
        Recv => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            infer::infer_recv(ctx, infer::infer_get_tuple_element(ctx, result, 0)?)
        }
        RecvDone => {
            channel(ctx, op)?;
            flag(ctx, op, "hlo_host_transfer")?;
            infer::infer_recv_done(ctx, operands[0])
        }
        Infeed => {
            feed_config(ctx, op, "hlo_infeed_config")?;
            require(
                operands[0].deref(ctx).is::<TokenType>(),
                "infeed requires a token operand",
            )?;
            shape::tuple(
                ctx,
                vec![
                    infer::infer_get_tuple_element(ctx, result, 0)?,
                    shape::token(ctx),
                ],
            )
        }
        Outfeed => {
            feed_config(ctx, op, "hlo_outfeed_config")?;
            require(
                operands[1].deref(ctx).is::<TokenType>(),
                "outfeed requires a token second operand",
            )?;
            let host: Ref<'_, TypeAttr> = required(ctx, op, "hlo_outfeed_shape")?;
            require(
                options.same_shapes(ctx, host.get_type(ctx), operands[0]),
                "outfeed host shape differs from data operand",
            )?;
            Ok(shape::token(ctx))
        }
        AsyncStart => {
            let callees: Ref<'_, Callees> = required(ctx, op, crate::attributes::keys::CALLEES)?;
            require(
                callees.0.len() == 1,
                "async-start requires one wrapped computation",
            )?;
            let (bound, _) = infer::async_context(ctx, result)?;
            require(
                bound.len() == operands.len()
                    && bound
                        .iter()
                        .zip(operands)
                        .all(|(&a, &b)| options.same_shapes(ctx, a, b)),
                "async-start operand differs from bound parameter shape",
            )?;
            Ok(result)
        }
        AsyncUpdate => {
            let previous = operands.first().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "async-update needs a predecessor".into()
                })
            })?;
            let (previous_bound, previous_output) = infer::async_context(ctx, *previous)?;
            let (bound, output) = infer::async_context(ctx, result)?;
            require(
                prefix(ctx, previous_output, output, options)?
                    && previous_bound.len() <= bound.len(),
                "async-update must extend predecessor parameter/result prefixes",
            )?;
            for (&a, &b) in previous_bound.iter().zip(bound.iter()) {
                require(
                    prefix(ctx, a, b, options)?,
                    "async-update must extend bound parameter prefixes",
                )?;
            }
            require(
                operands.len() - 1 == bound.len() - previous_bound.len(),
                "async-update new operands do not match newly bound parameters",
            )?;
            require(
                operands[1..]
                    .iter()
                    .zip(&bound[previous_bound.len()..])
                    .all(|(&a, &b)| options.same_shapes(ctx, a, b)),
                "async-update new operand shape differs from newly bound parameter",
            )?;
            Ok(result)
        }
        AsyncDone => {
            infer::async_context(ctx, operands[0])?;
            Ok(result)
        }
        _ => Err(pliron::verify_error_noloc!(Diagnostic::Verification {
            message: "unsupported async inference opcode".into()
        })),
    }
}

fn prefix(ctx: &Context, a: TypeHandle, b: TypeHandle, options: VerifyOptions) -> Result<bool> {
    let mut work = vec![(a, b)];
    while let Some((a, b)) = work.pop() {
        if options.same_shapes(ctx, a, b) {
            continue;
        }
        if shape::is_tuple(ctx, a) && shape::children(ctx, a).is_empty() {
            continue;
        }
        if !shape::is_tuple(ctx, a) || !shape::is_tuple(ctx, b) {
            return Ok(false);
        }
        let a = shape::children(ctx, a);
        let b = shape::children(ctx, b);
        if a.len() > b.len() {
            return Ok(false);
        }
        work.extend(a.iter().copied().zip(b.iter().copied()));
    }
    Ok(true)
}

pub(in crate::verification::control) fn verify_callees(
    ctx: &Context,
    _: Ptr<Operation>,
    kind: Opcode,
    result: TypeHandle,
    _: &[TypeHandle],
    resolved: &[&Signature],
    options: VerifyOptions,
) -> Result<()> {
    if kind != Opcode::AsyncStart && resolved.is_empty() {
        return Ok(());
    }
    require(
        resolved.len() == 1,
        "async instruction requires one wrapped computation",
    )?;
    let callee = &resolved[0];
    if kind == Opcode::AsyncDone {
        require(
            options.same_shapes(ctx, result, callee.1),
            "async-done requires full computation result shape",
        )
    } else {
        let (parameters, output) = infer::async_context(ctx, result)?;
        require(
            parameters.len() <= callee.0.len(),
            "bound parameters must prefix wrapped computation",
        )?;
        for (&a, &b) in parameters.iter().zip(&callee.0) {
            require(
                prefix(ctx, a, b, options)?,
                "bound parameters must prefix wrapped computation",
            )?;
        }
        require(
            prefix(ctx, output, callee.1, options)?,
            "bound result must prefix wrapped computation result",
        )
    }
}
