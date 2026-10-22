use super::*;

pub(crate) fn required<'a, T: Attribute>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    name: &'static str,
) -> Result<Ref<'a, T>> {
    optional_attr(ctx, op, name)?.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: name,
            message: "required attribute is missing".into(),
        })
    })
}
pub(crate) fn rng_delta(ctx: &Context, op: Ptr<Operation>) -> Result<i64> {
    Ok(optional_attr::<NumericalInteger>(ctx, op, "hlo_delta")?.map_or(0, |delta| delta.0))
}
pub(crate) fn custom_call_api_version(
    ctx: &Context,
    op: Ptr<Operation>,
) -> Result<CustomCallApiVersion> {
    Ok(
        optional_attr::<CustomCallApiVersion>(ctx, op, "hlo_api_version")?
            .map_or(CustomCallApiVersion::Unspecified, |version| *version),
    )
}
pub(crate) fn custom_call_schedule(
    ctx: &Context,
    op: Ptr<Operation>,
) -> Result<CustomCallSchedule> {
    Ok(
        optional_attr::<CustomCallSchedule>(ctx, op, crate::attributes::keys::SCHEDULE)?
            .map_or(CustomCallSchedule::None, |schedule| *schedule),
    )
}
pub(crate) fn custom_call_padding_kind(ctx: &Context, op: Ptr<Operation>) -> Result<PaddingKind> {
    Ok(optional_attr::<PaddingKind>(ctx, op, "hlo_padding_type")?
        .map_or(PaddingKind::Invalid, |padding| *padding))
}
pub(crate) fn flag(ctx: &Context, op: Ptr<Operation>, name: &'static str) -> Result<bool> {
    match optional_attr::<Bools>(ctx, op, name)? {
        None => Ok(false),
        Some(value) => {
            if value.0.len() != 1 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: name,
                    message: "boolean attribute requires exactly one value".into(),
                }));
            }
            Ok(value.0[0])
        }
    }
}
pub(crate) fn channel(ctx: &Context, op: Ptr<Operation>) -> Result<Option<i64>> {
    Ok(
        match optional_attr::<OptionalInteger>(ctx, op, crate::attributes::keys::CHANNEL)?
            .as_deref()
        {
            None | Some(OptionalInteger::Absent) => None,
            Some(OptionalInteger::Value(value)) => Some(*value),
        },
    )
}
pub(crate) fn same_ignoring_memory(
    ctx: &Context,
    left: TypeHandle,
    right: TypeHandle,
    splits: bool,
) -> Result<bool> {
    if !shape::equal_ignoring_layout(ctx, left, right) {
        return Ok(false);
    }
    let mut pending = vec![(left, right)];
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
        let a = if a.deref(ctx).is::<BufferType>() {
            shape::array_view(ctx, a)?
        } else {
            a
        };
        let b = if b.deref(ctx).is::<BufferType>() {
            shape::array_view(ctx, b)?
        } else {
            b
        };
        let (mut a_layout, mut b_layout) = (
            shape::layout(ctx, a)?.as_deref().cloned(),
            shape::layout(ctx, b)?.as_deref().cloned(),
        );
        for layout in [&mut a_layout, &mut b_layout].into_iter().flatten() {
            layout.memory_space = 0;
            if splits {
                layout.split_configs.clear();
            }
        }
        if match (&a_layout, &b_layout) {
            (Some(a), Some(b)) => !layouts_equal(ctx, a, b),
            (None, None) => false,
            _ => true,
        } {
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
pub(super) fn compare(
    ctx: &Context,
    actual: TypeHandle,
    expected: TypeHandle,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let exact = matches!(
        opcode,
        Call | Conditional
            | Constant
            | CopyDone
            | CopyStart
            | CustomCall
            | GetTupleElement
            | Infeed
            | Outfeed
            | OptimizationBarrier
            | Parameter
            | Recv
            | RecvDone
            | Send
            | SendDone
            | Tuple
            | While
            | Fusion
    );
    let equal = if exact && options.layout_sensitive {
        if matches!(
            opcode,
            Opcode::Constant | Opcode::Infeed | Opcode::Send | Opcode::Recv | Opcode::CopyStart
        ) {
            same_minor_layout(ctx, actual, expected)?
        } else {
            options.same_shapes(ctx, actual, expected)
        }
    } else if !exact && options.allow_mixed_precision {
        shape::compatible_ignoring_fp_precision(ctx, actual, expected)
    } else {
        same(ctx, actual, expected)
    };
    require(equal, "declared result differs from inferred shape")
}
fn same_minor_layout(ctx: &Context, a: TypeHandle, b: TypeHandle) -> Result<bool> {
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
        let (a, b) = (
            if a.deref(ctx).is::<BufferType>() {
                shape::array_view(ctx, a)?
            } else {
                a
            },
            if b.deref(ctx).is::<BufferType>() {
                shape::array_view(ctx, b)?
            } else {
                b
            },
        );
        match (shape::layout(ctx, a)?, shape::layout(ctx, b)?) {
            (Some(a), Some(b)) => {
                if a.minor_to_major != b.minor_to_major
                    || a.physical_shape.len() != b.physical_shape.len()
                    || !a
                        .physical_shape
                        .iter()
                        .zip(&b.physical_shape)
                        .all(|(&a, &b)| shape::equal_including_layout(ctx, a, b))
                {
                    return Ok(false);
                }
            }
            (None, None) => {}
            _ => return Ok(false),
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
