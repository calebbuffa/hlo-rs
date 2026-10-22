//! Static and bounded-dynamic `Reduce` evaluation through the request-local session.

use super::*;
use crate::{
    attributes::{Axes, Callees},
    literal::semantics::{geometry, read_element},
    shape::Dimension,
};

impl Session<'_> {
    pub(super) fn reduce(
        &mut self,
        frame: &mut Frame,
        operation: Ptr<Operation>,
        operands: &[Value],
        target: TypeHandle,
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        let dimensions = attr::<Axes>(self.ctx, operation, crate::attributes::keys::DIMENSIONS)?;
        let callees = attr::<Callees>(self.ctx, operation, crate::attributes::keys::CALLEES)?;
        if callees.0.len() != 1 || operands.is_empty() || !operands.len().is_multiple_of(2) {
            return Err(malformed(
                "Reduce requires one reducer and paired inputs and initial values",
            ));
        }
        let component_count = operands.len() / 2;
        let mut values = Vec::with_capacity(operands.len());
        for &operand in operands {
            values.push(self.operand(frame, operand, &[], mode)?);
        }
        if values.iter().any(|value| !value.is_known()) {
            return EvaluatedValue::unknown(self.ctx, target);
        }
        let literals = values
            .into_iter()
            .map(|value| value.into_literal(self.ctx))
            .collect::<Result<Vec<_>>>()?;
        let inputs = &literals[..component_count];
        let initial = &literals[component_count..];
        let output_shapes = if shape::is_tuple(self.ctx, target) {
            shape::children(self.ctx, target).to_vec()
        } else {
            vec![target]
        };
        if output_shapes.len() != component_count {
            return Err(malformed(
                "Reduce result component count differs from its operands",
            ));
        }
        let rank = shape::rank(self.ctx, inputs[0].shape)?;
        for (input, init) in inputs.iter().zip(initial) {
            if !shape::is_array(self.ctx, input.shape)
                || shape::rank(self.ctx, input.shape)? != rank
                || shape::rank(self.ctx, init.shape)? != 0
                || shape::is_dynamic(self.ctx, init.shape)
            {
                return Err(malformed(
                    "Reduce requires equally ranked array inputs and scalar initial values",
                ));
            }
        }

        let mut reduced = vec![false; rank];
        for &axis in &dimensions.0 {
            let axis = usize::try_from(axis)
                .ok()
                .filter(|&axis| axis < rank)
                .ok_or_else(|| malformed("Reduce dimension is outside the input rank"))?;
            if std::mem::replace(&mut reduced[axis], true) {
                return Err(malformed("Reduce dimension is repeated"));
            }
        }
        let input_geometry = inputs
            .iter()
            .map(|literal| geometry(self.ctx, literal))
            .collect::<Result<Vec<_>>>()?;
        let input_dimensions = input_geometry[0].dimensions().to_vec();
        if input_geometry
            .iter()
            .any(|geometry| geometry.dimensions() != input_dimensions)
        {
            return Err(malformed("Reduce input runtime dimensions differ"));
        }
        let output_dimensions = input_dimensions
            .iter()
            .zip(&reduced)
            .filter_map(|(&dimension, &is_reduced)| (!is_reduced).then_some(dimension))
            .collect::<Vec<_>>();
        for &shape in &output_shapes {
            if !shape::is_array(self.ctx, shape) {
                return Err(malformed("Reduce result component is not an array"));
            }
            validate_output_dimensions(self.ctx, shape, &output_dimensions, Opcode::Reduce)?;
        }

        let mut results = output_shapes
            .iter()
            .map(|&shape| allocate_output(self.ctx, shape, &output_dimensions, Opcode::Reduce))
            .collect::<Result<Vec<_>>>()?;
        let output_geometry = results
            .iter()
            .map(|literal| geometry(self.ctx, literal))
            .collect::<Result<Vec<_>>>()?;
        let input_scalar_shapes = inputs
            .iter()
            .map(|literal| shape::scalar(self.ctx, shape::element_type(self.ctx, literal.shape)?))
            .collect::<Result<Vec<_>>>()?;
        let input_order = minor_to_major(self.ctx, inputs[0].shape)?;
        let output_order = minor_to_major(self.ctx, results[0].shape)?;
        let callee = self.resolve(&callees.0[0])?;
        let result_is_tuple = shape::is_tuple(self.ctx, target);
        let mut unknown_result = false;

        visit_index_space(
            &output_dimensions,
            &output_order,
            &mut vec![0; output_dimensions.len()],
            0,
            &mut |output_index| {
                if unknown_result {
                    return Ok(false);
                }
                let mut input_index = vec![0; rank];
                let mut output_axis = 0;
                for (axis, &is_reduced) in reduced.iter().enumerate() {
                    if !is_reduced {
                        input_index[axis] = output_index[output_axis];
                        output_axis += 1;
                    }
                }
                let mut accumulators = initial.to_vec();
                visit_reduction_space(
                    &input_dimensions,
                    &reduced,
                    &input_order,
                    &mut input_index,
                    0,
                    &mut |index| {
                        let mut args = Vec::with_capacity(component_count * 2);
                        args.extend(
                            accumulators
                                .iter()
                                .cloned()
                                .map(|literal| EvaluatedValue::known(self.ctx, literal))
                                .collect::<Result<Vec<_>>>()?,
                        );
                        for (component, input) in inputs.iter().enumerate() {
                            let input_geometry = &input_geometry[component];
                            let start = input_geometry.byte_offset(index)?;
                            let end = start
                                .checked_add(input_geometry.element_bytes())
                                .ok_or_else(|| malformed("Reduce input byte range overflow"))?;
                            let bytes = input
                                .storage
                                .get(start..end)
                                .ok_or_else(|| malformed("Reduce input storage is incomplete"))?;
                            args.push(EvaluatedValue::known(
                                self.ctx,
                                Literal::from_storage_bytes(
                                    self.ctx,
                                    input_scalar_shapes[component],
                                    bytes.to_vec(),
                                )?,
                            )?);
                        }

                        let reduced_value = self.computation(callee, &args, &[])?;
                        if !reduced_value.is_known() {
                            unknown_result = true;
                            return Ok(false);
                        }
                        let mut next = Vec::with_capacity(component_count);
                        for component in 0..component_count {
                            let value = if result_is_tuple {
                                reduced_value.subvalue(&[component])?.clone()
                            } else {
                                reduced_value.clone()
                            };
                            let literal = value.into_literal(self.ctx)?;
                            if shape::rank(self.ctx, literal.shape)? != 0 {
                                return Err(malformed("Reduce reducer result is not scalar"));
                            }
                            next.push(literal);
                        }
                        accumulators = next;
                        Ok(true)
                    },
                )?;
                if unknown_result {
                    return Ok(false);
                }
                for (component, accumulator) in accumulators.iter().enumerate() {
                    let bytes = read_element(self.ctx, accumulator, &[])?;
                    let start = output_geometry[component].byte_offset(output_index)?;
                    let end = start
                        .checked_add(bytes.len())
                        .ok_or_else(|| malformed("Reduce output byte range overflow"))?;
                    results[component]
                        .storage
                        .get_mut(start..end)
                        .ok_or_else(|| malformed("Reduce output storage is incomplete"))?
                        .copy_from_slice(bytes);
                }
                Ok(true)
            },
        )?;

        if unknown_result {
            return EvaluatedValue::unknown(self.ctx, target);
        }
        let children = results
            .into_iter()
            .map(|literal| EvaluatedValue::known(self.ctx, literal))
            .collect::<Result<Vec<_>>>()?;
        if result_is_tuple {
            EvaluatedValue::tuple(self.ctx, target, children)
        } else {
            children
                .into_iter()
                .next()
                .ok_or_else(|| malformed("Reduce result is missing"))
        }
    }
}

pub(super) fn allocate_output(
    ctx: &Context,
    shape: TypeHandle,
    runtime_dimensions: &[usize],
    opcode: Opcode,
) -> Result<Literal> {
    validate_output_dimensions(ctx, shape, runtime_dimensions, opcode)?;
    let bounds = shape::dimensions(ctx, shape)?;
    if bounds.len() != runtime_dimensions.len() {
        return Err(malformed("reduction output runtime rank mismatch"));
    }
    let is_dynamic = shape::is_dynamic(ctx, shape);
    let mut dynamic_sizes = if is_dynamic {
        Vec::with_capacity(bounds.len())
    } else {
        Vec::new()
    };
    for (axis, (&bound, &runtime)) in bounds.iter().zip(runtime_dimensions).enumerate() {
        let bound = usize::try_from(bound)
            .map_err(|_| malformed("unbounded reduction output is unsupported"))?;
        if runtime > bound {
            return Err(malformed(
                "reduction output runtime dimension exceeds its bound",
            ));
        }
        match shape::dimension(ctx, shape, axis)? {
            Dimension::Static(_) if runtime != bound => {
                return Err(malformed(
                    "reduction runtime dimension differs from a static output extent",
                ));
            }
            Dimension::Bounded(_) => {}
            Dimension::Unbounded => {
                return Err(unsupported_value(
                    opcode,
                    "unbounded array allocation is not implemented",
                ));
            }
            Dimension::Static(_) => {}
        }
        if is_dynamic {
            dynamic_sizes.push(
                i64::try_from(runtime)
                    .map_err(|_| malformed("reduction runtime dimension exceeds i64"))?,
            );
        }
    }
    let element_count = usize::try_from(shape::element_count(ctx, shape)?)
        .map_err(|_| malformed("reduction output exceeds host addressing"))?;
    let bytes_per_element = shape::element_type(ctx, shape)?.bit_width().div_ceil(8) as usize;
    let byte_count = element_count
        .checked_mul(bytes_per_element)
        .ok_or_else(|| malformed("reduction output byte size overflow"))?;
    Literal::from_dynamic_storage_bytes(ctx, shape, vec![0; byte_count], dynamic_sizes)
}

pub(super) fn validate_output_dimensions(
    ctx: &Context,
    shape: TypeHandle,
    runtime_dimensions: &[usize],
    opcode: Opcode,
) -> Result<()> {
    let bounds = shape::dimensions(ctx, shape)?;
    if bounds.len() != runtime_dimensions.len() {
        return Err(malformed("reduction output runtime rank mismatch"));
    }
    for (axis, (&bound, &runtime)) in bounds.iter().zip(runtime_dimensions).enumerate() {
        let bound =
            usize::try_from(bound).map_err(|_| malformed("negative reduction output bound"))?;
        if runtime > bound {
            return Err(malformed(
                "reduction output runtime dimension exceeds its bound",
            ));
        }
        match shape::dimension(ctx, shape, axis)? {
            Dimension::Static(_) if runtime != bound => {
                return Err(malformed(
                    "reduction runtime dimension differs from a static output extent",
                ));
            }
            Dimension::Unbounded => {
                return Err(unsupported_value(
                    opcode,
                    "unbounded array allocation is not implemented",
                ));
            }
            Dimension::Static(_) | Dimension::Bounded(_) => {}
        }
    }
    Ok(())
}

fn minor_to_major(ctx: &Context, shape: TypeHandle) -> Result<Vec<usize>> {
    shape::layout(ctx, shape)?
        .ok_or_else(|| malformed("Reduce array has no layout"))?
        .minor_to_major
        .iter()
        .map(|&axis| {
            usize::try_from(axis).map_err(|_| malformed("Reduce layout has a negative axis"))
        })
        .collect()
}

fn visit_index_space(
    dimensions: &[usize],
    minor_to_major: &[usize],
    index: &mut [usize],
    axis: usize,
    visitor: &mut impl FnMut(&[usize]) -> Result<bool>,
) -> Result<bool> {
    if axis == minor_to_major.len() {
        return visitor(index);
    }
    let dimension = minor_to_major[minor_to_major.len() - 1 - axis];
    for coordinate in 0..dimensions[dimension] {
        index[dimension] = coordinate;
        if !visit_index_space(dimensions, minor_to_major, index, axis + 1, visitor)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn visit_reduction_space(
    dimensions: &[usize],
    reduced: &[bool],
    minor_to_major: &[usize],
    index: &mut [usize],
    axis: usize,
    visitor: &mut impl FnMut(&[usize]) -> Result<bool>,
) -> Result<bool> {
    if axis == minor_to_major.len() {
        return visitor(index);
    }
    let dimension = minor_to_major[minor_to_major.len() - 1 - axis];
    if !reduced[dimension] {
        return visit_reduction_space(
            dimensions,
            reduced,
            minor_to_major,
            index,
            axis + 1,
            visitor,
        );
    }
    for coordinate in 0..dimensions[dimension] {
        index[dimension] = coordinate;
        if !visit_reduction_space(
            dimensions,
            reduced,
            minor_to_major,
            index,
            axis + 1,
            visitor,
        )? {
            return Ok(false);
        }
    }
    Ok(true)
}
