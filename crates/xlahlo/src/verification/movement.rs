//! Native verification for shape and data-movement operations.
use crate::{
    Diagnostic,
    attributes::{
        Axes, Callees, GatherDimensionNumbers, Index, IndexFlags, PaddingConfig,
        ScatterDimensionNumbers, ShuffleMode, Slice,
    },
    inference::{self, Signature, require},
    opcode::Opcode,
    ops::{control::ConstantOp, movement::CopyOp},
    shape::{Dimension, Shape},
    types::{TupleType, tensor},
};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
    value::DefiningEntity,
};
use std::{cell::Ref, collections::HashSet};

use super::{VerifyOptions, instruction_types, optional_attr};

pub(crate) fn reshape_inferred_dimension(ctx: &Context, op: Ptr<Operation>) -> Result<i64> {
    let dimensions = optional_attr::<Axes>(ctx, op, "hlo_dimensions")?;
    match dimensions
        .as_deref()
        .map(|dimensions| dimensions.0.as_slice())
    {
        None | Some([]) => Ok(-1),
        Some([dimension]) if *dimension >= -1 => Ok(*dimension),
        _ => pliron::verify_err_noloc!(Diagnostic::InvalidAttribute {
            attribute: "hlo_dimensions",
            message:
                "reshape requires at most one inferred dimension, which must be -1 or nonnegative"
                    .into(),
        }),
    }
}

fn required_attr<'a, T: Attribute>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    name: &'static str,
) -> Result<Ref<'a, T>> {
    optional_attr(ctx, op, name)?.ok_or_else(|| {
        pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: name,
            message: "required movement attribute is missing".into(),
        })
    })
}

fn arrays(ctx: &Context, input: TypeHandle, output: TypeHandle) -> Result<()> {
    tensor(ctx, Shape::array_view(ctx, input)?)?;
    tensor(ctx, Shape::array_view(ctx, output)?)?;
    Ok(())
}

fn compatible_element(
    ctx: &Context,
    left: TypeHandle,
    right: TypeHandle,
    allow_mixed_precision: bool,
) -> Result<bool> {
    let left = tensor(ctx, Shape::array_view(ctx, left)?)?.element;
    let right = tensor(ctx, Shape::array_view(ctx, right)?)?.element;
    Ok(left == right || allow_mixed_precision && left.same_ignoring_fp_precision(right))
}

fn equivalent(
    ctx: &Context,
    options: VerifyOptions,
    declared: TypeHandle,
    expected: TypeHandle,
) -> Result<()> {
    require(
        options.same_shapes(ctx, declared, expected),
        "declared result differs from inferred movement shape",
    )
}

fn declared_bounds(ctx: &Context, shape: TypeHandle) -> Result<Vec<Dimension>> {
    (0..Shape::rank(ctx, shape)?)
        .map(|axis| Shape::dimension(ctx, shape, axis))
        .collect()
}

fn verify_bitcast(
    ctx: &Context,
    op: Ptr<Operation>,
    input: TypeHandle,
    output: TypeHandle,
    options: VerifyOptions,
) -> Result<()> {
    if !options.layout_sensitive {
        return Ok(());
    }
    if options.size_of(ctx, input)? != options.size_of(ctx, output)? {
        require(
            options.allow_bitcast_to_have_different_size
                && !Shape::is_dynamic(ctx, input)
                && !Shape::is_dynamic(ctx, output)
                && Shape::array_data_size(ctx, input)? == Shape::array_data_size(ctx, output)?,
            "bitcast operand and result have different shape sizes",
        )?;
    }
    let input_array = Shape::array_view(ctx, input)?;
    let output_array = Shape::array_view(ctx, output)?;
    if let (Some(input_layout), Some(output_layout)) = (
        Shape::layout(ctx, input_array)?,
        Shape::layout(ctx, output_array)?,
    ) && (input_layout.memory_space == 5) != (output_layout.memory_space == 5)
    {
        let data = op.deref(ctx);
        let mut value = data.get_operand(0);
        let mut visited = HashSet::new();
        loop {
            match value.defining_entity() {
                DefiningEntity::Op(definition) if Operation::is_op::<CopyOp>(definition, ctx) => {
                    if !visited.insert(definition) {
                        return pliron::verify_err_noloc!(Diagnostic::Verification {
                            message: "bitcast constant-origin traversal encountered a copy cycle"
                                .into(),
                        });
                    }
                    instruction_types(ctx, definition, Opcode::Copy)?;
                    value = definition.deref(ctx).get_operand(0);
                }
                DefiningEntity::Op(definition)
                    if Operation::is_op::<ConstantOp>(definition, ctx) =>
                {
                    break;
                }
                _ => {
                    return pliron::verify_err_noloc!(Diagnostic::Verification {
                        message: "bitcast cannot cross host/device memory spaces except through constant copies".into(),
                    });
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn instruction(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    options: VerifyOptions,
) -> Result<()> {
    let (operands, result) = instruction_types(ctx, op, opcode)?;
    if matches!(
        opcode,
        Opcode::DynamicReshape | Opcode::DynamicSlice | Opcode::Concatenate
    ) {
        require(
            !operands.is_empty(),
            &format!("{} requires at least one operand, got 0", opcode.name()),
        )?;
    }
    let inferred = match opcode {
        Opcode::Copy => {
            let expected = inference::movement::infer_copy(ctx, operands[0])?;
            if options.layout_sensitive
                && Shape::is_array(ctx, result)
                && Shape::is_array(ctx, expected)
                && let (Some(output_layout), Some(input_layout)) =
                    (Shape::layout(ctx, result)?, Shape::layout(ctx, expected)?)
                && output_layout.memory_space != input_layout.memory_space
                && (output_layout.memory_space == 5 || input_layout.memory_space == 5)
            {
                if output_layout.memory_space == 5 {
                    // Host-bound copies may transpose before memory-space propagation.
                    require(
                        Shape::element_type(ctx, result)? == Shape::element_type(ctx, expected)?,
                        "host-bound copy must preserve the element type",
                    )?;
                } else {
                    let mut layout = output_layout.clone();
                    layout.memory_space = input_layout.memory_space;
                    let normalized = Shape::with_layout(ctx, result, layout)?;
                    require(
                        Shape::equal_including_layout(ctx, normalized, expected),
                        "host-to-device copy must preserve shape and layout except memory space",
                    )?;
                }
                return Ok(());
            }
            equivalent(ctx, options, result, expected)?;
            return Ok(());
        }
        Opcode::Broadcast => {
            let dimensions = required_attr::<Axes>(ctx, op, "hlo_dimensions")?;
            arrays(ctx, operands[0], result)?;
            let input = tensor(ctx, Shape::array_view(ctx, operands[0])?)?;
            let output = tensor(ctx, Shape::array_view(ctx, result)?)?;
            require(
                input.element == output.element,
                "broadcast element types differ",
            )?;
            inference::movement::verify_broadcast(
                ctx,
                operands[0],
                result,
                &dimensions,
                options.verify_broadcast_dimensions_order,
            )?;
            result
        }
        Opcode::Reshape => {
            arrays(ctx, operands[0], result)?;
            let input = tensor(ctx, Shape::array_view(ctx, operands[0])?)?;
            let output = tensor(ctx, Shape::array_view(ctx, result)?)?;
            require(
                input.element == output.element,
                "reshape cannot change element type",
            )?;
            let dimension = reshape_inferred_dimension(ctx, op)?;
            let dimension = if dimension == -1 {
                None
            } else {
                Some(usize::try_from(dimension).map_err(|_| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "hlo_dimensions",
                        message: "reshape inferred dimension does not fit the host index type"
                            .into(),
                    })
                })?)
            };
            let expected = inference::movement::infer_reshape_with_dimension(
                ctx,
                operands[0],
                &output.dimensions,
                dimension,
            )?;
            require(
                Shape::compatible(ctx, result, expected),
                "reshape changes element count or dynamic bounds",
            )?;
            if options.verify_reshape_is_bitcast {
                require(
                    Shape::reshape_is_bitcast(ctx, operands[0], result, false)?,
                    "reshape is not a physical bitcast",
                )?;
            }
            result
        }
        Opcode::DynamicReshape => {
            arrays(ctx, operands[0], result)?;
            let input = tensor(ctx, Shape::array_view(ctx, operands[0])?)?;
            let output = tensor(ctx, Shape::array_view(ctx, result)?)?;
            require(
                input.element == output.element,
                "dynamic reshape cannot change element type",
            )?;
            require(
                operands.len() - 1 == output.dimensions.len(),
                &format!(
                    "dynamic reshape requires one size operand per result dimension: got {} for rank {}",
                    operands.len() - 1,
                    output.dimensions.len()
                ),
            )?;
            for (axis, &size) in operands[1..].iter().enumerate() {
                require(
                    Shape::element_type(ctx, size)? == crate::types::ElementType::S32,
                    &format!(
                        "dynamic reshape size operand for axis {axis} must have S32 element type"
                    ),
                )?;
            }
            let bounds = declared_bounds(ctx, result)?;
            inference::movement::infer_dynamic_reshape(ctx, operands[0], &operands[1..], bounds)?
        }
        Opcode::Bitcast => {
            arrays(ctx, operands[0], result)?;
            verify_bitcast(ctx, op, operands[0], result, options)?;
            return Ok(());
        }
        Opcode::Transpose => {
            let dimensions = required_attr::<Axes>(ctx, op, "hlo_dimensions")?;
            arrays(ctx, operands[0], result)?;
            let expected = inference::movement::infer_transpose(ctx, operands[0], &dimensions)?;
            require(
                tensor(ctx, expected)?.dimensions == tensor(ctx, result)?.dimensions,
                "transpose result dimensions differ from permuted operand",
            )?;
            expected
        }
        Opcode::Reverse => {
            let dimensions = required_attr::<Axes>(ctx, op, "hlo_dimensions")?;
            let expected = inference::movement::infer_reverse(ctx, operands[0], &dimensions)?;
            equivalent(ctx, options, result, expected)?;
            return Ok(());
        }
        Opcode::Slice => {
            let geometry = required_attr::<Slice>(ctx, op, "hlo_geometry")?;
            inference::movement::infer_slice(ctx, operands[0], &geometry)?
        }
        Opcode::DynamicSlice => {
            let sizes = required_attr::<Axes>(ctx, op, "hlo_slice_sizes")?;
            inference::movement::infer_dynamic_slice(ctx, operands[0], &operands[1..], &sizes)?
        }
        Opcode::DynamicUpdateSlice => {
            require(
                operands.len() >= 2,
                "dynamic update slice needs operand and update",
            )?;
            require(
                compatible_element(ctx, operands[0], operands[1], options.allow_mixed_precision)?,
                "dynamic update element type mismatch",
            )?;
            let expected = inference::movement::infer_dynamic_update_slice(
                ctx,
                operands[0],
                operands[1],
                &operands[2..],
            )?;
            equivalent(ctx, options, result, expected)?;
            return Ok(());
        }
        Opcode::Concatenate => {
            let axis = required_attr::<Index>(ctx, op, "hlo_index")?;
            for &operand in &operands[1..] {
                require(
                    compatible_element(ctx, operands[0], operand, options.allow_mixed_precision)?,
                    "incompatible concatenate element types",
                )?;
            }
            inference::movement::infer_concatenate(ctx, &operands, axis.0 as usize)?
        }
        Opcode::Pad => {
            let config = required_attr::<PaddingConfig>(ctx, op, "hlo_config")?;
            require(
                compatible_element(ctx, operands[0], operands[1], options.allow_mixed_precision)?,
                "incompatible pad element types",
            )?;
            inference::movement::infer_pad(ctx, operands[0], operands[1], &config)?
        }
        Opcode::Iota => {
            let axis = required_attr::<Index>(ctx, op, "hlo_index")?;
            inference::movement::infer_iota(ctx, result, axis.0 as usize)?;
            return Ok(());
        }
        Opcode::Gather => {
            let dimensions = required_attr::<GatherDimensionNumbers>(ctx, op, "hlo_dimensions")?;
            let slice_sizes = required_attr::<Axes>(ctx, op, "hlo_slice_sizes")?;
            let _flags = required_attr::<IndexFlags>(ctx, op, "hlo_flags")?;
            inference::movement::infer_gather(
                ctx,
                operands[0],
                operands[1],
                &dimensions,
                &slice_sizes,
            )?
        }
        Opcode::Scatter => {
            let dimensions = required_attr::<ScatterDimensionNumbers>(ctx, op, "hlo_dimensions")?;
            let _callees = required_attr::<Callees>(ctx, op, "hlo_callees")?;
            let _flags = required_attr::<IndexFlags>(ctx, op, "hlo_flags")?;
            require(
                operands.len() >= 3 && !operands.len().is_multiple_of(2),
                "scatter needs N inputs, indices, and N updates",
            )?;
            let count = operands.len() / 2;
            for (&input, &update) in operands[..count].iter().zip(&operands[count + 1..]) {
                require(
                    compatible_element(ctx, input, update, options.allow_mixed_precision)?,
                    "scatter input and update element types differ",
                )?;
            }
            inference::movement::infer_variadic_scatter(
                ctx,
                &operands[..count],
                operands[count],
                &operands[count + 1..],
                &dimensions,
            )?
        }
        Opcode::GetDimensionSize => {
            let axis = required_attr::<Index>(ctx, op, "hlo_index")?;
            inference::movement::infer_get_dimension_size(ctx, operands[0], axis.0 as usize)?
        }
        Opcode::SetDimensionSize => {
            let axis = required_attr::<Index>(ctx, op, "hlo_index")?;
            inference::movement::infer_set_dimension_size(
                ctx,
                operands[0],
                operands[1],
                axis.0 as usize,
            )?
        }
        Opcode::Shuffle => {
            let dimensions = required_attr::<Axes>(ctx, op, "hlo_dimensions")?;
            let mode = required_attr::<ShuffleMode>(ctx, op, "hlo_mode")?;
            inference::movement::infer_shuffle(ctx, operands[0], &dimensions, &mode)?
        }
        _ => {
            return pliron::verify_err_noloc!(Diagnostic::Verification {
                message: format!("{} is not a movement opcode", opcode.name()),
            });
        }
    };
    equivalent(ctx, options, result, inferred)
}

/// Validate the reducer resolved for a variadic scatter.
pub(crate) fn verify_callees(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
    _options: VerifyOptions,
    resolved: &[&Signature],
) -> Result<()> {
    require(
        opcode == Opcode::Scatter,
        "movement callee hook only accepts scatter",
    )?;
    let (operands, _) = instruction_types(ctx, op, opcode)?;
    require(
        operands.len() >= 3 && !operands.len().is_multiple_of(2),
        "scatter operand partition is malformed",
    )?;
    let count = operands.len() / 2;
    require(
        resolved.len() == 1,
        "scatter requires exactly one reducer computation",
    )?;
    let (parameters, result) = resolved[0];
    require(
        parameters.len() == count * 2,
        "scatter reducer parameter count must be twice the variadic arity",
    )?;
    let result_ty = result.deref(ctx);
    let results = if let Some(tuple) = result_ty.downcast_ref::<TupleType>() {
        require(
            tuple.elements.len() == count,
            "scatter reducer result tuple arity mismatch",
        )?;
        tuple.elements.as_slice()
    } else {
        require(count == 1, "variadic scatter reducer must return a tuple")?;
        std::slice::from_ref(result)
    };
    for (index, &result) in results.iter().enumerate() {
        let accumulator = tensor(ctx, result)?;
        let parameter = tensor(ctx, parameters[index])?;
        let update_parameter = tensor(ctx, parameters[count + index])?;
        let input = tensor(ctx, operands[index])?;
        let update = tensor(ctx, operands[count + 1 + index])?;
        require(
            accumulator.dimensions.is_empty()
                && parameter.dimensions.is_empty()
                && update_parameter.dimensions.is_empty(),
            &format!("scatter reducer lane {index} requires scalar parameters and result"),
        )?;
        require(
            accumulator.element == parameter.element,
            &format!(
                "scatter reducer accumulator parameter {index} must exactly match its result type"
            ),
        )?;
        require(
            accumulator
                .element
                .same_ignoring_fp_precision(input.element)
                && update_parameter
                    .element
                    .same_ignoring_fp_precision(update.element)
                && accumulator
                    .element
                    .same_ignoring_fp_precision(update_parameter.element),
            &format!(
                "scatter reducer lane {index} has incompatible accumulator/input/update types"
            ),
        )?;
    }
    Ok(())
}
