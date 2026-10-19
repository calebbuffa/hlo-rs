//! Local geometry and indexed, resolved computation contracts for numerical HLO.
use super::{VerifyOptions, instruction_types, optional_attr};
use crate::{
    Diagnostic,
    attributes::{
        Axes, BatchNormOptions, BlockScalingConfig, Callees, CholeskyOptions,
        ConvolutionDimensionNumbers, ConvolutionKind, DotDimensionNumbers, FftType,
        OptionalInteger, OptionalTensorBlockScalingConfig, OptionalTensorSparsityConfig,
        PrecisionConfig, RaggedDotDimensionNumbers, SortOptions, SparsityConfig, TriState,
        TriangularSolveOptions, Window,
    },
    inference::{Signature, numerical as infer, require},
    opcode::{Opcode, Opcode::*},
    ops::numerical::NumericalInteger,
    shape::Shape,
    types::ElementType,
};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
};
use std::{cell::Ref, collections::BTreeSet};

fn required<'a, T: Attribute>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    name: &'static str,
) -> Result<Ref<'a, T>> {
    optional_attr(ctx, op, name)?.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: name,
            message: format!("missing {name}"),
        })
    })
}
fn integer(ctx: &Context, op: Ptr<Operation>, name: &'static str) -> Result<i64> {
    Ok(required::<NumericalInteger>(ctx, op, name)?.0)
}
fn dimension(ctx: &Context, op: Ptr<Operation>) -> Result<i64> {
    let dims = required::<Axes>(ctx, op, "hlo_dimensions")?;
    if dims.0.len() != 1 {
        return pliron::verify_err_noloc!(Diagnostic::InvalidAttribute {
            attribute: "hlo_dimensions",
            message: format!(
                "instruction requires exactly one dimension; got {}",
                dims.0.len()
            ),
        });
    }
    Ok(dims.0[0])
}
fn scalar(ctx: &Context, shape: TypeHandle) -> Result<TypeHandle> {
    Shape::scalar(
        ctx,
        Shape::element_type(ctx, Shape::array_view(ctx, shape)?)?,
    )
}
fn same(ctx: &Context, a: TypeHandle, b: TypeHandle, options: VerifyOptions) -> bool {
    if options.allow_mixed_precision {
        Shape::compatible_ignoring_fp_precision(ctx, a, b)
    } else {
        Shape::compatible(ctx, a, b)
    }
}
fn callee_count(ctx: &Context, op: Ptr<Operation>, opcode: Opcode) -> Result<()> {
    let actual = optional_attr::<Callees>(ctx, op, "hlo_callees")?.map_or(0, |c| c.0.len());
    let expected = expected_callees(opcode);
    if actual != expected {
        return pliron::verify_err_noloc!(Diagnostic::InvalidAttribute {
            attribute: "hlo_callees",
            message: format!(
                "incorrect numerical computation count for {opcode:?}: expected {expected}, got {actual}"
            ),
        });
    }
    Ok(())
}

fn expected_callees(opcode: Opcode) -> usize {
    match opcode {
        Opcode::Reduce | Opcode::ReduceWindow | Opcode::Map | Opcode::Scan | Opcode::Sort => 1,
        Opcode::SelectAndScatter => 2,
        _ => 0,
    }
}

fn auxiliary_operands<'a>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    operands: &[TypeHandle],
) -> Result<Option<Ref<'a, SparsityConfig>>> {
    require(
        operands.len() >= 2,
        "dot/convolution requires at least two operands",
    )?;
    let sparsity = optional_attr::<SparsityConfig>(ctx, op, "hlo_sparsity")?;
    let scaling = optional_attr::<BlockScalingConfig>(ctx, op, "hlo_block_scaling")?;
    let mut seen = BTreeSet::new();
    let mut check = |raw_index: i64, attribute: &'static str| -> Result<()> {
        let index = usize::try_from(raw_index).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute,
                message: format!("negative auxiliary operand index {raw_index}"),
            })
        })?;
        if index < 2 || index >= operands.len() || !seen.insert(index) {
            return pliron::verify_err_noloc!(Diagnostic::InvalidAttribute {
                attribute,
                message: format!(
                    "auxiliary operand index is repeated or out of range: {index} for {} operands",
                    operands.len()
                ),
            });
        }
        require(
            Shape::is_array_or_buffer(ctx, operands[index]),
            "auxiliary operand must be an array",
        )
    };
    if let Some(config) = &sparsity {
        for side in [&config.lhs, &config.rhs] {
            if let OptionalTensorSparsityConfig::Present(side) = side {
                check(i64::from(side.idx), "hlo_sparsity")?;
            }
        }
    }
    if let Some(config) = &scaling {
        for side in [&config.lhs, &config.rhs] {
            if let OptionalTensorBlockScalingConfig::Present(side) = side {
                check(i64::from(side.scale_idx), "hlo_block_scaling")?;
                if let OptionalInteger::Value(index) = side.zero_idx {
                    check(index, "hlo_block_scaling")?;
                }
            }
        }
    }
    require(
        seen.len() == operands.len() - 2,
        "every extra operand must be referenced exactly once",
    )?;
    Ok(sparsity)
}

fn local_reducer(ctx: &Context, operands: &[TypeHandle], result: TypeHandle) -> Result<Signature> {
    require(
        !operands.is_empty() && operands.len().is_multiple_of(2),
        "reduce requires nonzero even operand count",
    )?;
    let count = operands.len() / 2;
    let leaves = infer::elements(ctx, result);
    require(
        leaves.len() == count,
        "reduction result cardinality differs from input count",
    )?;
    let roots = leaves
        .iter()
        .map(|&ty| scalar(ctx, ty))
        .collect::<Result<Vec<_>>>()?;
    let root = if Shape::is_tuple(ctx, result) {
        Shape::tuple(ctx, roots.clone())?
    } else {
        roots[0]
    };
    let mut params = roots;
    for &ty in &operands[..count] {
        params.push(scalar(ctx, ty)?);
    }
    Ok((params, root))
}

/// Verify numerical attributes, operand geometry, and the declared result type.
///
/// Callable bodies are checked separately by [`verify_callees`] after resolution.
pub(crate) fn instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    callee_count(ctx, op, opcode)?;
    if !options.allow_mixed_precision
        && matches!(
            opcode,
            Reduce
                | Map
                | SelectAndScatter
                | BatchNormTraining
                | BatchNormInference
                | BatchNormGrad
        )
    {
        let mut precision = None;
        for &operand in &operands {
            let et = Shape::element_type(ctx, Shape::array_view(ctx, operand)?)?;
            if et.is_float() {
                require(
                    precision.is_none_or(|previous| previous == et),
                    "mixed floating point operand precision requires allow_mixed_precision",
                )?;
                precision = Some(et);
            }
        }
    }
    let expected = match opcode {
        Dot => {
            let sparsity = auxiliary_operands(ctx, op, &operands)?;
            let dims = required::<DotDimensionNumbers>(ctx, op, "hlo_dimensions")?;
            required::<PrecisionConfig>(ctx, op, "hlo_precision")?;
            infer::infer_dot_with_sparsity(
                ctx,
                operands[0],
                operands[1],
                &dims,
                Some(Shape::element_type(ctx, result)?),
                sparsity.as_deref(),
            )?
        }
        RaggedDot => {
            require(operands.len() == 3, "ragged-dot requires three operands")?;
            required::<PrecisionConfig>(ctx, op, "hlo_precision")?;
            infer::infer_ragged_dot(
                ctx,
                operands[0],
                operands[1],
                operands[2],
                &*required::<RaggedDotDimensionNumbers>(ctx, op, "hlo_dimensions")?,
                Some(Shape::element_type(ctx, result)?),
            )?
        }
        ScaledDot => {
            require(operands.len() == 4, "scaled-dot requires four operands")?;
            required::<PrecisionConfig>(ctx, op, "hlo_precision")?;
            infer::infer_scaled_dot(
                ctx,
                operands[0],
                operands[1],
                operands[2],
                operands[3],
                &*required::<DotDimensionNumbers>(ctx, op, "hlo_dimensions")?,
                Some(Shape::element_type(ctx, result)?),
            )?
        }
        Convolution => {
            let sparsity = auxiliary_operands(ctx, op, &operands)?;
            required::<PrecisionConfig>(ctx, op, "hlo_precision")?;
            required::<ConvolutionKind>(ctx, op, "hlo_kind")?;
            // HLO instruction verification ignores gradient metadata; the explicit
            // inference API remains available for gradient geometry.
            infer::infer_convolution_with_sparsity(
                ctx,
                operands[0],
                operands[1],
                integer(ctx, op, "hlo_feature_groups")?.max(1),
                integer(ctx, op, "hlo_batch_groups")?.max(1),
                &*required::<Window>(ctx, op, "hlo_window")?,
                &*required::<ConvolutionDimensionNumbers>(ctx, op, "hlo_dimensions")?,
                Some(Shape::element_type(ctx, result)?),
                sparsity.as_deref(),
            )?
        }
        Cholesky => {
            required::<CholeskyOptions>(ctx, op, "hlo_options")?;
            infer::infer_cholesky(ctx, operands[0])?
        }
        TriangularSolve => infer::infer_triangular_solve(
            ctx,
            operands[0],
            operands[1],
            &*required::<TriangularSolveOptions>(ctx, op, "hlo_options")?,
        )?,
        TopK => {
            required::<SortOptions>(ctx, op, "hlo_options")?;
            infer::infer_top_k(ctx, operands[0], integer(ctx, op, "hlo_k")?)?
        }
        Fft => infer::infer_fft(
            ctx,
            operands[0],
            *required::<FftType>(ctx, op, "hlo_kind")?,
            &required::<Axes>(ctx, op, "hlo_lengths")?.0,
        )?,
        BatchNormTraining | BatchNormInference | BatchNormGrad => {
            let config = required::<BatchNormOptions>(ctx, op, "hlo_options")?;
            infer::infer_batch_norm(ctx, opcode, &operands, config.feature_index)?
        }
        Sort => {
            let output = infer::infer_sort_result(ctx, &operands)?;
            require(
                dimension(ctx, op)?
                    < Shape::rank(ctx, Shape::array_view(ctx, operands[0])?)? as i64,
                "sort dimension exceeds operand rank",
            )?;
            output
        }
        Reduce | ReduceWindow => {
            let reducer = local_reducer(ctx, &operands, result)?;
            if opcode == Reduce {
                infer::infer_reduce(
                    ctx,
                    &operands,
                    &required::<Axes>(ctx, op, "hlo_dimensions")?.0,
                    &reducer,
                )?
            } else {
                infer::infer_reduce_window(
                    ctx,
                    &operands,
                    &*required::<Window>(ctx, op, "hlo_window")?,
                    &reducer,
                )?
            }
        }
        Map => {
            let params = operands
                .iter()
                .map(|&ty| scalar(ctx, ty))
                .collect::<Result<Vec<_>>>()?;
            let rank = operands
                .iter()
                .map(|&ty| Shape::rank(ctx, Shape::array_view(ctx, ty)?))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            infer::infer_map(
                ctx,
                &operands,
                &(params, scalar(ctx, result)?),
                &(0..rank as i64).collect::<Vec<_>>(),
            )?
        }
        SelectAndScatter => {
            let select = (
                vec![scalar(ctx, operands[0])?; 2],
                Shape::scalar(ctx, ElementType::Pred)?,
            );
            let scatter = (
                vec![scalar(ctx, operands[2])?, scalar(ctx, operands[1])?],
                scalar(ctx, operands[2])?,
            );
            infer::infer_select_and_scatter(
                ctx,
                operands[0],
                operands[1],
                operands[2],
                &*required::<Window>(ctx, op, "hlo_window")?,
                &select,
                &scatter,
            )?
        }
        Scan => {
            let axis = dimension(ctx, op)?;
            let carries = integer(ctx, op, "hlo_num_carries")?;
            let carries = usize::try_from(carries).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "hlo_num_carries",
                    message: format!(
                        "scan carry count {carries} must be nonnegative and fit usize"
                    )
                })
            })?;
            let results = infer::elements(ctx, result);
            require(
                carries <= operands.len()
                    && carries <= results.len()
                    && !(carries == operands.len() && carries == results.len()),
                "scan requires valid carry counts and at least one input or output",
            )?;
            let inputs = operands.len() - carries;
            let outputs = results.len() - carries;
            let source = if inputs > 0 { operands[0] } else { results[0] };
            let axis = usize::try_from(axis).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "hlo_dimensions",
                    message: format!("scan axis {axis} must be nonnegative and fit usize")
                })
            })?;
            let bound = Shape::dimensions(ctx, source)?
                .get(axis)
                .copied()
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Inference {
                        message: "scan axis exceeds rank".into()
                    })
                })?;
            for &shape in operands[..inputs].iter().chain(&results[..outputs]) {
                require(
                    Shape::dimensions(ctx, shape)?.get(axis) == Some(&bound),
                    "scan input/output extents differ or axis exceeds rank",
                )?;
            }
            let associative = *required::<TriState>(ctx, op, "hlo_is_associative")?;
            require(
                !options.layout_sensitive || associative == TriState::True,
                "non-associative scan must be expanded before layout-sensitive verification",
            )?;
            result
        }
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Verification {
                message: "opcode is not in the numerical family".into()
            });
        }
    };
    require(
        same(ctx, result, expected, options),
        "declared result differs from inferred shape",
    )
}

/// Verify computation signatures already resolved by the enclosing graph index.
/// Signatures are borrowed in `Callees` order; missing required callees are errors.
/// No parameter vectors are copied and no symbol lookup or graph traversal occurs here.
pub(crate) fn verify_callees(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
    signatures: &[&Signature],
) -> Result<()> {
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    let expected_count = expected_callees(opcode);
    require(
        signatures.len() == expected_count,
        "incorrect resolved numerical computation count",
    )?;
    if expected_count == 0 {
        return Ok(());
    }
    let expected = match opcode {
        Reduce => infer::infer_reduce(
            ctx,
            &operands,
            &required::<Axes>(ctx, op, "hlo_dimensions")?.0,
            signatures[0],
        )?,
        ReduceWindow => infer::infer_reduce_window(
            ctx,
            &operands,
            &*required::<Window>(ctx, op, "hlo_window")?,
            signatures[0],
        )?,
        Map => {
            let rank = operands
                .iter()
                .map(|&ty| Shape::rank(ctx, Shape::array_view(ctx, ty)?))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            infer::infer_map(
                ctx,
                &operands,
                signatures[0],
                &(0..rank as i64).collect::<Vec<_>>(),
            )?
        }
        Sort => infer::infer_sort(ctx, &operands, dimension(ctx, op)?, signatures[0])?,
        SelectAndScatter => infer::infer_select_and_scatter(
            ctx,
            operands[0],
            operands[1],
            operands[2],
            &*required::<Window>(ctx, op, "hlo_window")?,
            signatures[0],
            signatures[1],
        )?,
        Scan => {
            infer::verify_scan(
                ctx,
                &operands,
                result,
                signatures[0],
                integer(ctx, op, "hlo_num_carries")?,
                dimension(ctx, op)?,
                options.layout_sensitive,
                options.allow_mixed_precision,
            )?;
            result
        }
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Verification {
                message: "unexpected callable numerical opcode".into()
            });
        }
    };
    if !options.allow_mixed_precision && matches!(opcode, Map | Reduce | ReduceWindow) {
        for (&operand, &parameter) in operands.iter().zip(&signatures[0].0) {
            require(
                Shape::element_type(ctx, Shape::array_view(ctx, operand)?)?
                    == Shape::element_type(ctx, Shape::array_view(ctx, parameter)?)?,
                "callee parameter and operand element types differ",
            )?;
        }
    }
    require(
        same(ctx, result, expected, options),
        "declared result differs from inferred callee shape",
    )
}
