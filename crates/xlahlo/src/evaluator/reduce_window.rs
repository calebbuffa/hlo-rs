// Copyright 2017 The OpenXLA Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Derived from OpenXLA `xla/hlo/evaluator/hlo_evaluator.cc`.
// Modified: Rust static and bounded-dynamic evaluation through the native evaluator session.
//
//! Static and bounded-dynamic `ReduceWindow` evaluation through the request-local session.
//!
//! The window coordinate arithmetic follows OpenXLA's
//! `xla/hlo/evaluator/hlo_evaluator.cc` `IterateThroughWindow` and
//! `HandleReduceWindow` paths. The pinned `IterateThroughWindow` does not
//! inspect `window_reversal`; traversal therefore retains its ordinary
//! lexicographic order even when the metadata flag is true. Reducer computations
//! run through the existing evaluator session so frame binding, recursion
//! detection, effect rejection, scalar semantics, and loop limits remain shared.

use super::*;
use crate::{
    attributes::{Callees, Window, WindowDimension},
    inference::numerical::infer_window_output,
    literal::semantics::{geometry, read_element},
};

#[derive(Clone, Copy)]
struct AxisTraversal {
    first: usize,
    step: usize,
    count: usize,
}

impl Session<'_> {
    pub(super) fn reduce_window(
        &mut self,
        frame: &mut Frame,
        operation: Ptr<Operation>,
        operands: &[Value],
        target: TypeHandle,
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        let window = attr::<Window>(self.ctx, operation, "hlo_window")?;
        let callees = attr::<Callees>(self.ctx, operation, "hlo_callees")?;
        if callees.0.len() != 1 || operands.is_empty() || !operands.len().is_multiple_of(2) {
            return Err(malformed(
                "ReduceWindow requires one reducer and paired inputs and initial values",
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
                "ReduceWindow result component count differs from its operands",
            ));
        }
        if window.0.len() != shape::rank(self.ctx, inputs[0].shape)? {
            return Err(malformed(
                "ReduceWindow window rank differs from its input rank",
            ));
        }
        for (input, init) in inputs.iter().zip(initial) {
            if !shape::is_array(self.ctx, input.shape)
                || shape::rank(self.ctx, init.shape)? != 0
                || shape::rank(self.ctx, input.shape)? != window.0.len()
                || shape::is_dynamic(self.ctx, init.shape)
            {
                return Err(malformed(
                    "ReduceWindow requires ranked array inputs and scalar initial values",
                ));
            }
        }

        let input_geometry = inputs
            .iter()
            .map(|literal| geometry(self.ctx, literal))
            .collect::<Result<Vec<_>>>()?;
        let input_scalar_shapes = inputs
            .iter()
            .map(|literal| shape::scalar(self.ctx, shape::element_type(self.ctx, literal.shape)?))
            .collect::<Result<Vec<_>>>()?;
        let input_dimensions = input_geometry[0].dimensions().to_vec();
        if input_geometry
            .iter()
            .any(|geometry| geometry.dimensions() != input_dimensions)
        {
            return Err(malformed("ReduceWindow input runtime dimensions differ"));
        }
        let active_input_shape = shape::array(
            self.ctx,
            shape::element_type(self.ctx, inputs[0].shape)?,
            input_dimensions
                .iter()
                .map(|&dimension| {
                    i64::try_from(dimension)
                        .map_err(|_| malformed("ReduceWindow runtime dimension exceeds i64"))
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let active_output_shape = infer_window_output(
            self.ctx,
            active_input_shape,
            &window,
            shape::element_type(self.ctx, initial[0].shape)?,
        )?;
        let output_dimensions = shape::dimensions(self.ctx, active_output_shape)?
            .iter()
            .map(|&dimension| {
                usize::try_from(dimension)
                    .map_err(|_| malformed("negative ReduceWindow output dimension"))
            })
            .collect::<Result<Vec<_>>>()?;
        for &shape in &output_shapes {
            if !shape::is_array(self.ctx, shape) {
                return Err(malformed("ReduceWindow result component is not an array"));
            }
            super::reduce::validate_output_dimensions(
                self.ctx,
                shape,
                &output_dimensions,
                Opcode::ReduceWindow,
            )?;
        }

        let mut results = output_shapes
            .iter()
            .map(|&shape| {
                if !shape::is_array(self.ctx, shape) {
                    return Err(malformed("ReduceWindow result component is not an array"));
                }
                super::reduce::allocate_output(
                    self.ctx,
                    shape,
                    &output_dimensions,
                    Opcode::ReduceWindow,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let output_geometries = results
            .iter()
            .map(|literal| geometry(self.ctx, literal))
            .collect::<Result<Vec<_>>>()?;
        let callee = self.resolve(&callees.0[0])?;
        let component_results_are_tuple = shape::is_tuple(self.ctx, target);
        let mut unknown_result = false;

        visit_indices(&output_dimensions, &mut |output_index| {
            if unknown_result {
                return Ok(false);
            }
            let traversals = window
                .0
                .iter()
                .enumerate()
                .map(|(axis, dimension)| {
                    axis_traversal(output_index[axis], dimension, input_dimensions[axis])
                })
                .collect::<Result<Vec<_>>>()?;
            let mut accumulators = initial.to_vec();
            visit_window(
                &traversals,
                0,
                &mut Vec::with_capacity(traversals.len()),
                &mut |input_index| {
                    let mut args = Vec::with_capacity(component_count * 2);
                    args.extend(
                        accumulators
                            .iter()
                            .cloned()
                            .map(|literal| EvaluatedValue::known(self.ctx, literal))
                            .collect::<Result<Vec<_>>>()?,
                    );
                    for (component, input) in inputs.iter().enumerate() {
                        let geometry = &input_geometry[component];
                        let start = geometry.byte_offset(input_index)?;
                        let end = start
                            .checked_add(geometry.element_bytes())
                            .ok_or_else(|| malformed("ReduceWindow input byte range overflow"))?;
                        let bytes = input
                            .storage
                            .get(start..end)
                            .ok_or_else(|| malformed("ReduceWindow input storage is incomplete"))?
                            .to_vec();
                        args.push(EvaluatedValue::known(
                            self.ctx,
                            Literal::from_storage_bytes(
                                self.ctx,
                                input_scalar_shapes[component],
                                bytes,
                            )?,
                        )?);
                    }

                    let reduced = self.computation(callee, &args, &[])?;
                    if !reduced.is_known() {
                        unknown_result = true;
                        return Ok(false);
                    }
                    let mut next = Vec::with_capacity(component_count);
                    for component in 0..component_count {
                        let value = if component_results_are_tuple {
                            reduced.subvalue(&[component])?.clone()
                        } else {
                            reduced.clone()
                        };
                        let literal = value.into_literal(self.ctx)?;
                        if shape::rank(self.ctx, literal.shape)? != 0 {
                            return Err(malformed(
                                "ReduceWindow reducer result component is not scalar",
                            ));
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
                let offset = output_geometries[component].byte_offset(output_index)?;
                let end = offset
                    .checked_add(bytes.len())
                    .ok_or_else(|| malformed("ReduceWindow output byte range overflow"))?;
                results[component]
                    .storage
                    .get_mut(offset..end)
                    .ok_or_else(|| malformed("ReduceWindow output storage is incomplete"))?
                    .copy_from_slice(bytes);
            }
            Ok(true)
        })?;

        if unknown_result {
            return EvaluatedValue::unknown(self.ctx, target);
        }
        let children = results
            .into_iter()
            .map(|literal| EvaluatedValue::known(self.ctx, literal))
            .collect::<Result<Vec<_>>>()?;
        if component_results_are_tuple {
            EvaluatedValue::tuple(self.ctx, target, children)
        } else {
            children
                .into_iter()
                .next()
                .ok_or_else(|| malformed("ReduceWindow result is missing"))
        }
    }
}

fn axis_traversal(
    output_index: usize,
    dimension: &WindowDimension,
    base_limit: usize,
) -> Result<AxisTraversal> {
    if dimension.size <= 0
        || dimension.stride <= 0
        || dimension.window_dilation <= 0
        || dimension.base_dilation <= 0
    {
        return Err(malformed("ReduceWindow has nonpositive window geometry"));
    }
    if base_limit == 0 {
        return Ok(AxisTraversal {
            first: 0,
            step: 1,
            count: 0,
        });
    }
    let start = i128::try_from(output_index)
        .map_err(|_| malformed("ReduceWindow output index exceeds wide arithmetic"))?
        .checked_mul(i128::from(dimension.stride))
        .and_then(|index| index.checked_sub(i128::from(dimension.padding_low)))
        .ok_or_else(|| malformed("ReduceWindow start coordinate overflow"))?;
    let window_dilation = i128::from(dimension.window_dilation);
    let base_dilation = i128::from(dimension.base_dilation);
    let window_size = i128::from(dimension.size);
    let minimum_window = if start < 0 {
        start
            .checked_neg()
            .and_then(|distance| distance.checked_add(window_dilation - 1))
            .ok_or_else(|| malformed("ReduceWindow window bound overflow"))?
            / window_dilation
    } else {
        0
    };
    let Some(first_window) =
        first_congruent_window(start, window_dilation, base_dilation, minimum_window)?
    else {
        return Ok(AxisTraversal {
            first: 0,
            step: 1,
            count: 0,
        });
    };
    if first_window >= window_size {
        return Ok(AxisTraversal {
            first: 0,
            step: 1,
            count: 0,
        });
    }
    let first_numerator = first_window
        .checked_mul(window_dilation)
        .and_then(|coordinate| start.checked_add(coordinate))
        .ok_or_else(|| malformed("ReduceWindow input coordinate overflow"))?;
    if first_numerator % base_dilation != 0 {
        return Err(malformed(
            "ReduceWindow congruence solver returned a nondivisible coordinate",
        ));
    }
    let first_base = first_numerator / base_dilation;
    let base_limit = i128::try_from(base_limit)
        .map_err(|_| malformed("ReduceWindow input dimension exceeds wide arithmetic"))?;
    if first_base >= base_limit {
        return Ok(AxisTraversal {
            first: 0,
            step: 1,
            count: 0,
        });
    }
    let gcd = gcd(window_dilation, base_dilation);
    let window_step = base_dilation / gcd;
    let base_step = window_dilation / gcd;
    let count_by_window = (window_size - 1 - first_window) / window_step + 1;
    let count_by_base = (base_limit - 1 - first_base) / base_step + 1;
    let count = usize::try_from(count_by_window.min(count_by_base))
        .map_err(|_| malformed("ReduceWindow window count exceeds host addressing"))?;
    Ok(AxisTraversal {
        first: usize::try_from(first_base)
            .map_err(|_| malformed("ReduceWindow base index exceeds host addressing"))?,
        step: usize::try_from(base_step)
            .map_err(|_| malformed("ReduceWindow base step exceeds host addressing"))?,
        count,
    })
}

fn first_congruent_window(
    start: i128,
    window_dilation: i128,
    base_dilation: i128,
    minimum_window: i128,
) -> Result<Option<i128>> {
    let divisor = gcd(window_dilation, base_dilation);
    let start_remainder = start.rem_euclid(base_dilation);
    let target = (base_dilation - start_remainder).rem_euclid(base_dilation);
    if target % divisor != 0 {
        return Ok(None);
    }

    let period = base_dilation / divisor;
    let residue = if period == 1 {
        0
    } else {
        let reduced_dilation = window_dilation / divisor;
        let reduced_target = target / divisor;
        let inverse = modular_inverse(reduced_dilation, period)?;
        reduced_target
            .checked_mul(inverse)
            .ok_or_else(|| malformed("ReduceWindow congruence multiplication overflow"))?
            .rem_euclid(period)
    };
    if residue >= minimum_window {
        return Ok(Some(residue));
    }
    let distance = minimum_window
        .checked_sub(residue)
        .ok_or_else(|| malformed("ReduceWindow congruence bound underflow"))?;
    let periods = distance
        .checked_add(period - 1)
        .ok_or_else(|| malformed("ReduceWindow congruence bound overflow"))?
        / period;
    let offset = periods
        .checked_mul(period)
        .ok_or_else(|| malformed("ReduceWindow congruence bound overflow"))?;
    Ok(Some(residue.checked_add(offset).ok_or_else(|| {
        malformed("ReduceWindow congruence bound overflow")
    })?))
}

fn modular_inverse(value: i128, modulus: i128) -> Result<i128> {
    let (mut old_remainder, mut remainder) = (value, modulus);
    let (mut old_coefficient, mut coefficient) = (1_i128, 0_i128);
    while remainder != 0 {
        let quotient = old_remainder / remainder;
        (old_remainder, remainder) = (
            remainder,
            old_remainder
                .checked_sub(
                    quotient
                        .checked_mul(remainder)
                        .ok_or_else(|| malformed("ReduceWindow inverse overflow"))?,
                )
                .ok_or_else(|| malformed("ReduceWindow inverse overflow"))?,
        );
        (old_coefficient, coefficient) = (
            coefficient,
            old_coefficient
                .checked_sub(
                    quotient
                        .checked_mul(coefficient)
                        .ok_or_else(|| malformed("ReduceWindow inverse overflow"))?,
                )
                .ok_or_else(|| malformed("ReduceWindow inverse overflow"))?,
        );
    }
    if old_remainder != 1 {
        return Err(malformed(
            "ReduceWindow reduced dilation has no modular inverse",
        ));
    }
    Ok(old_coefficient.rem_euclid(modulus))
}

fn gcd(mut left: i128, mut right: i128) -> i128 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn visit_indices(
    dimensions: &[usize],
    visitor: &mut impl FnMut(&[usize]) -> Result<bool>,
) -> Result<()> {
    if dimensions.contains(&0) {
        return Ok(());
    }
    let mut index = vec![0; dimensions.len()];
    loop {
        if !visitor(&index)? {
            return Ok(());
        }
        let mut axis = dimensions.len();
        loop {
            if axis == 0 {
                return Ok(());
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < dimensions[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

fn visit_window(
    traversals: &[AxisTraversal],
    axis: usize,
    index: &mut Vec<usize>,
    visitor: &mut impl FnMut(&[usize]) -> Result<bool>,
) -> Result<bool> {
    if axis == traversals.len() {
        return visitor(index);
    }
    let traversal = traversals[axis];
    for ordinal in 0..traversal.count {
        let coordinate = traversal
            .step
            .checked_mul(ordinal)
            .and_then(|offset| traversal.first.checked_add(offset))
            .ok_or_else(|| malformed("ReduceWindow input coordinate overflow"))?;
        index.push(coordinate);
        let keep_going = visit_window(traversals, axis + 1, index, visitor)?;
        index.pop();
        if !keep_going {
            return Ok(false);
        }
    }
    Ok(true)
}
