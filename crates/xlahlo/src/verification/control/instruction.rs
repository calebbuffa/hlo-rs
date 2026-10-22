use super::*;

pub(crate) fn instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    let expected = match opcode {
        Parameter => {
            let _: Ref<'_, Index> = required(ctx, op, crate::attributes::keys::INDEX)?;
            result
        }
        Constant => {
            let literal: Ref<'_, Literal> = required(ctx, op, crate::attributes::keys::LITERAL)?;
            literal.verify(ctx)?;
            literal.shape()
        }
        Tuple => infer::infer_tuple(ctx, &operands)?,
        GetTupleElement => infer::infer_get_tuple_element(
            ctx,
            operands[0],
            i64::from(required::<Index>(ctx, op, crate::attributes::keys::INDEX)?.0),
        )?,
        Domain => {
            // Sharding-domain consistency is checked with the enclosing graph.
            let _: Option<Ref<'_, crate::sharding::Sharding>> =
                optional_attr(ctx, op, crate::attributes::keys::DOMAIN_ENTRY_SHARDING)?;
            let _: Option<Ref<'_, crate::sharding::Sharding>> =
                optional_attr(ctx, op, crate::attributes::keys::DOMAIN_EXIT_SHARDING)?;
            operands[0]
        }
        OptimizationBarrier => operands[0],
        AfterAll => shape::token(ctx),
        AddDependency => {
            require(
                operands[1].deref(ctx).is::<TokenType>(),
                "add-dependency requires a token second operand",
            )?;
            operands[0]
        }
        ReplicaId | PartitionId => shape::scalar(ctx, ElementType::U32)?,
        Call | While | Conditional => {
            let callees: Ref<'_, Callees> = required(ctx, op, crate::attributes::keys::CALLEES)?;
            require(
                match opcode {
                    Call => callees.0.len() == 1,
                    While => callees.0.len() == 2,
                    _ => !callees.0.is_empty(),
                },
                "invalid callee cardinality",
            )?;
            if opcode == Conditional {
                let (selector, branches) = operands.split_first().ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "conditional requires a selector".into()
                    })
                })?;
                infer::validate_conditional_selector(
                    ctx,
                    *selector,
                    branches.len(),
                    callees.0.len(),
                )?;
            }
            if opcode == Call && flag(ctx, op, crate::attributes::keys::IS_COMPOSITE)? {
                let attrs: Ref<'_, FrontendAttributes> =
                    required(ctx, op, crate::attributes::keys::FRONTEND_ATTRIBUTES)?;
                require(
                    attrs.get("composite.name").is_some_and(|s| !s.is_empty())
                        && attrs
                            .get("composite.attributes")
                            .is_none_or(|s| !s.is_empty())
                        && attrs
                            .get("composite.version")
                            .is_none_or(|s| s.trim().parse::<i64>().is_ok_and(|n| n >= 0)),
                    "invalid composite name, attributes or version",
                )?;
            }
            result
        }
        CustomCall => {
            verify_custom_call(ctx, op, result, &operands, options)?;
            result
        }
        Rng => {
            let distribution: Ref<'_, RandomDistribution> = required(ctx, op, "hlo_distribution")?;
            infer::infer_rng(
                ctx,
                &operands,
                result,
                *distribution,
                options.allow_mixed_precision,
            )?
        }
        RngBitGenerator => {
            let _: Ref<'_, RandomAlgorithm> = required(ctx, op, "hlo_algorithm")?;
            infer::infer_rng_bit_generator(ctx, operands[0], result)?
        }
        RngGetAndUpdateState => {
            rng_delta(ctx, op)?;
            infer::infer_rng_get_and_update_state(ctx)?
        }
        AllGather
        | AllGatherStart
        | AllGatherDone
        | AllReduce
        | AllReduceStart
        | AllReduceDone
        | AllToAll
        | ReduceScatter
        | RaggedAllToAll
        | CollectiveBroadcast
        | CollectiveReduce
        | CollectivePermute
        | CollectivePermuteStart
        | CollectivePermuteDone => collective::infer(ctx, op, opcode, result, &operands, options)?,
        AsyncStart | AsyncUpdate | AsyncDone | CopyStart | CopyDone | Send | SendDone | Recv
        | RecvDone | Infeed | Outfeed => {
            asynchronous::infer(ctx, op, opcode, result, &operands, options)?
        }
        _ => {
            return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "opcode is not a control-family instruction".into()
            }));
        }
    };
    if !matches!(opcode, AsyncStart | AsyncUpdate) {
        verify_aliases(ctx, op, opcode, result, &operands, options)?;
    }
    compare(ctx, result, expected, opcode, options)
}
/// Validate local callee signatures resolved by the module's symbol index.
///
/// The slice must include every callee in attribute order, even when empty.
/// While uses body then condition; inline Fusion requires no external callee.
pub(crate) fn verify_callees(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
    resolved: &[&Signature],
) -> Result<()> {
    if opcode == Fusion {
        return require(
            resolved.is_empty(),
            "inline fusion cannot call an external computation",
        );
    }
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    let parameters = |operands: &[TypeHandle], callee: &Signature| -> Result<()> {
        require(
            operands.len() == callee.0.len(),
            "operand/parameter cardinality mismatch",
        )?;
        require(
            operands
                .iter()
                .zip(&callee.0)
                .all(|(&a, &b)| options.same_shapes(ctx, a, b)),
            "operand and parameter shapes differ",
        )
    };
    match opcode {
        Call => {
            require(resolved.len() == 1, "call requires exactly one computation")?;
            parameters(&operands, resolved[0])?;
            compare(ctx, result, resolved[0].1, opcode, options)
        }
        While => {
            require(
                resolved.len() == 2,
                "while requires body and condition computations",
            )?;
            for callee in resolved {
                parameters(&operands, callee)?;
            }
            infer::infer_while(ctx, operands[0], resolved[0], resolved[1])?;
            compare(ctx, result, resolved[0].1, opcode, options)
        }
        Conditional => {
            let (selector, branches) = operands.split_first().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "conditional requires a selector".into()
                })
            })?;
            infer::validate_conditional_selector(ctx, *selector, branches.len(), resolved.len())?;
            for (&operand, branch) in branches.iter().zip(resolved) {
                parameters(&[operand], branch)?;
                compare(ctx, result, branch.1, opcode, options)?;
            }
            Ok(())
        }
        AllReduce | AllReduceStart | ReduceScatter | CollectiveReduce => {
            collective::verify_reducer(ctx, op, &operands, resolved, options)
        }
        AsyncStart | AsyncUpdate | AsyncDone => {
            asynchronous::verify_callees(ctx, op, opcode, result, &operands, resolved, options)
        }
        CustomCall => Ok(()),
        _ => require(resolved.is_empty(), "unexpected called computations"),
    }
}
fn verify_custom_call(
    ctx: &Context,
    op: Ptr<Operation>,
    result: TypeHandle,
    operands: &[TypeHandle],
    options: VerifyOptions,
) -> Result<()> {
    let target: Ref<'_, StringAttr> = required(ctx, op, crate::attributes::keys::TARGET)?;
    let target = target.as_str();
    if flag(ctx, op, crate::attributes::keys::CONSTRAIN_LAYOUT)?
        && !matches!(target, "LayoutConstraint" | "control_dep")
    {
        require(
            shape::has_complete_layout(ctx, result)?,
            "layout-constrained custom call requires result layout",
        )?;
        let constraints: Ref<'_, ShapeList> =
            required(ctx, op, crate::attributes::keys::OPERAND_LAYOUTS)?;
        require(
            constraints.0.len() == operands.len(),
            "layout-constrained custom call requires operand layouts",
        )?;
        for (&operand, &constraint) in operands.iter().zip(&constraints.0) {
            require(
                shape::has_complete_layout(ctx, constraint)? && same(ctx, operand, constraint),
                "operand layout constraint is missing layout or has incompatible geometry",
            )?;
        }
    }
    if target == "GetRngSeed" {
        require(
            operands.is_empty() && same(ctx, result, shape::scalar(ctx, ElementType::U64)?),
            "GetRngSeed requires no operands and a U64 scalar result",
        )?;
    }
    if matches!(target, "Pin" | "Unpin") {
        require(operands.len() == 1, "Pin/Unpin requires one operand")?;
        let (array, buffer) = if target == "Pin" {
            (operands[0], result)
        } else {
            (result, operands[0])
        };
        require(
            shape::is_array(ctx, array) && buffer.deref(ctx).is::<BufferType>(),
            "Pin/Unpin requires an array and a buffer",
        )?;
        let inner = shape::array_view(ctx, buffer)?;
        require(
            if options.layout_sensitive {
                options.same_shapes(ctx, array, inner)
            } else {
                shape::equal_ignoring_layout(ctx, array, inner)
            },
            "Pin/Unpin requires equal array shapes",
        )?;
        let aliases: Ref<'_, OutputOperandAliases> =
            required(ctx, op, crate::attributes::keys::ALIASES)?;
        require(aliases.0.len() == 1, "Pin/Unpin requires exactly one alias")?;
    } else if target == "CreateBuffer" {
        require(
            operands.is_empty() && result.deref(ctx).is::<BufferType>(),
            "CreateBuffer requires no operands and a buffer result",
        )?;
    }
    custom_call_schedule(ctx, op)?;
    custom_call_api_version(ctx, op)?;
    custom_call_padding_kind(ctx, op)?;
    let _: Option<Ref<'_, Window>> = optional_attr(ctx, op, crate::attributes::keys::WINDOW)?;
    let _: Option<Ref<'_, ConvolutionDimensionNumbers>> =
        optional_attr(ctx, op, crate::attributes::keys::DIMENSIONS)?;
    let _: Option<Ref<'_, PrecisionConfig>> =
        optional_attr(ctx, op, crate::attributes::keys::PRECISION)?;
    for key in [
        crate::attributes::keys::FEATURE_GROUPS,
        crate::attributes::keys::BATCH_GROUPS,
    ] {
        let _: Option<Ref<'_, NumericalInteger>> = optional_attr(ctx, op, key)?;
    }
    flag(ctx, op, crate::attributes::keys::HAS_SIDE_EFFECT)?;
    Ok(())
}
