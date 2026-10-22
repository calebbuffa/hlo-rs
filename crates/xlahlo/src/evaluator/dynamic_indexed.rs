//! Bounded-dynamic native DynamicSlice and DynamicUpdateSlice evaluation.
//!
//! This draft reuses the evaluator's request-local session and literal geometry
//! contract. Bounds determine physical strides; runtime sizes restrict logical
//! reads and writes. No source value is mutated.

use super::*;
use crate::{
    attributes::Axes,
    literal::semantics::{LiteralGeometry, geometry},
    shape::Dimension,
    types::ElementType,
};

impl Session<'_> {
    pub(super) fn dynamic_indexed(
        &mut self,
        frame: &mut Frame,
        operation: Ptr<Operation>,
        opcode: Opcode,
        operands: &[Value],
        target: TypeHandle,
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        let required = if opcode == Opcode::DynamicSlice { 1 } else { 2 };
        if operands.len() < required {
            return Err(malformed("dynamic indexed movement requires data operands"));
        }
        let values = operands
            .iter()
            .map(|&operand| self.operand(frame, operand, &[], mode))
            .collect::<Result<Vec<_>>>()?;
        if values.iter().any(|value| !value.is_known()) {
            return EvaluatedValue::unknown(self.ctx, target);
        }
        let literals = values
            .into_iter()
            .map(|value| value.into_literal(self.ctx))
            .collect::<Result<Vec<_>>>()?;
        let source = &literals[0];
        if !shape::is_array(self.ctx, source.shape) || !shape::is_array(self.ctx, target) {
            return Err(unsupported_value(
                opcode,
                "native dynamic indexed movement requires array literals",
            ));
        }
        let rank = shape::rank(self.ctx, source.shape)?;
        let source_geometry = geometry(self.ctx, source)?;
        let starts = extract_start_indices(
            self.ctx,
            if opcode == Opcode::DynamicSlice {
                &literals[1..]
            } else {
                &literals[2..]
            },
            rank,
        )?;

        let result = if opcode == Opcode::DynamicSlice {
            let sizes = attr::<Axes>(self.ctx, operation, crate::attributes::keys::SLICE_SIZES)?;
            dynamic_slice(self.ctx, source, target, &source_geometry, &starts, &sizes)?
        } else {
            let update = &literals[1];
            let update_geometry = geometry(self.ctx, update)?;
            dynamic_update_slice(
                self.ctx,
                source,
                update,
                target,
                &source_geometry,
                &update_geometry,
                &starts,
            )?
        };
        EvaluatedValue::known(self.ctx, result)
    }
}

fn dynamic_slice(
    ctx: &Context,
    source: &Literal,
    target: TypeHandle,
    source_geometry: &LiteralGeometry,
    starts: &[i128],
    sizes: &Axes,
) -> Result<Literal> {
    let rank = source_geometry.dimensions().len();
    if starts.len() != rank || sizes.0.len() != rank || shape::rank(ctx, target)? != rank {
        return Err(malformed("DynamicSlice rank mismatch"));
    }
    if shape::element_type(ctx, source.shape)? != shape::element_type(ctx, target)? {
        return Err(unsupported_value(
            Opcode::DynamicSlice,
            "element format conversion is not implemented",
        ));
    }

    let mut output_dimensions = Vec::with_capacity(rank);
    let mut clamped_starts = Vec::with_capacity(rank);
    for (axis, (&start, &slice_size)) in starts.iter().zip(&sizes.0).enumerate() {
        let requested = usize::try_from(slice_size)
            .map_err(|_| malformed("DynamicSlice size is negative or exceeds host addressing"))?;
        let bound = match shape::dimension(ctx, target, axis)? {
            Dimension::Static(bound) | Dimension::Bounded(bound) => usize::try_from(bound)
                .map_err(|_| malformed("negative DynamicSlice output bound"))?,
            Dimension::Unbounded => {
                return Err(unsupported_value(
                    Opcode::DynamicSlice,
                    "unbounded output allocation is not implemented",
                ));
            }
        };
        if requested != bound {
            return Err(malformed(
                "DynamicSlice result bound differs from its slice size",
            ));
        }
        let active_input = source_geometry.dimensions()[axis];
        let active_output = match shape::dimension(ctx, target, axis)? {
            Dimension::Bounded(_) => requested.min(active_input),
            Dimension::Static(_) if requested <= active_input => requested,
            Dimension::Static(_) => {
                return Err(unsupported_value(
                    Opcode::DynamicSlice,
                    "runtime input extent is smaller than a static slice size",
                ));
            }
            Dimension::Unbounded => {
                return Err(unsupported_value(
                    Opcode::DynamicSlice,
                    "unbounded output allocation is not implemented",
                ));
            }
        };
        let max_start = active_input.saturating_sub(active_output);
        clamped_starts.push(clamp_index(start, max_start)?);
        output_dimensions.push(active_output);
    }

    let mut result = allocate_result(ctx, target, &output_dimensions, Opcode::DynamicSlice)?;
    let output_geometry = geometry(ctx, &result)?;
    visit_indices(&output_dimensions, &mut |index| {
        let input_index = index
            .iter()
            .zip(&clamped_starts)
            .map(|(&coordinate, &start)| {
                coordinate
                    .checked_add(start)
                    .ok_or_else(|| malformed("DynamicSlice coordinate overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let from = source_geometry.byte_offset(&input_index)?;
        let to = output_geometry.byte_offset(index)?;
        copy_raw_element(
            source,
            &mut result,
            from,
            to,
            source_geometry.element_bytes(),
        )
    })?;
    Ok(result)
}

fn dynamic_update_slice(
    ctx: &Context,
    source: &Literal,
    update: &Literal,
    target: TypeHandle,
    source_geometry: &LiteralGeometry,
    update_geometry: &LiteralGeometry,
    starts: &[i128],
) -> Result<Literal> {
    let rank = source_geometry.dimensions().len();
    let source_bounds = shape::dimensions(ctx, source.shape)?;
    let target_bounds = shape::dimensions(ctx, target)?;
    if starts.len() != rank
        || update_geometry.dimensions().len() != rank
        || shape::rank(ctx, target)? != rank
        || source_bounds.as_ref() != target_bounds.as_ref()
    {
        return Err(malformed("DynamicUpdateSlice rank or bounds mismatch"));
    }
    if shape::element_type(ctx, source.shape)? != shape::element_type(ctx, target)?
        || shape::element_type(ctx, update.shape)? != shape::element_type(ctx, target)?
    {
        return Err(unsupported_value(
            Opcode::DynamicUpdateSlice,
            "raw update requires identical element formats",
        ));
    }

    let output_dimensions = (0..rank)
        .map(|axis| match shape::dimension(ctx, target, axis)? {
            Dimension::Static(bound) => usize::try_from(bound)
                .map_err(|_| malformed("negative DynamicUpdateSlice output bound")),
            Dimension::Bounded(_) => Ok(source_geometry.dimensions()[axis]),
            Dimension::Unbounded => Err(unsupported_value(
                Opcode::DynamicUpdateSlice,
                "unbounded output allocation is not implemented",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut result = allocate_result(ctx, target, &output_dimensions, Opcode::DynamicUpdateSlice)?;
    let output_geometry = geometry(ctx, &result)?;
    copy_allocation(
        source,
        &mut result,
        source_geometry.strides(),
        output_geometry.strides(),
        source_geometry.element_bytes(),
        &shape::dimensions(ctx, source.shape)?
            .iter()
            .map(|&bound| {
                usize::try_from(bound).map_err(|_| malformed("unbounded DynamicUpdateSlice input"))
            })
            .collect::<Result<Vec<_>>>()?,
    )?;

    let mut clamped_starts = Vec::with_capacity(rank);
    for axis in 0..rank {
        let update_extent = update_geometry.dimensions()[axis];
        let available = output_dimensions[axis];
        if update_extent > available {
            return Err(unsupported_value(
                Opcode::DynamicUpdateSlice,
                "runtime update extent exceeds the active input extent",
            ));
        }
        clamped_starts.push(clamp_index(starts[axis], available - update_extent)?);
    }
    visit_indices(update_geometry.dimensions(), &mut |index| {
        let destination = index
            .iter()
            .zip(&clamped_starts)
            .map(|(&coordinate, &start)| {
                coordinate
                    .checked_add(start)
                    .ok_or_else(|| malformed("DynamicUpdateSlice coordinate overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let from = update_geometry.byte_offset(index)?;
        let to = output_geometry.byte_offset(&destination)?;
        copy_raw_element(
            update,
            &mut result,
            from,
            to,
            update_geometry.element_bytes(),
        )
    })?;
    Ok(result)
}

fn extract_start_indices(ctx: &Context, indices: &[Literal], rank: usize) -> Result<Vec<i128>> {
    let Some(first) = indices.first() else {
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(malformed("dynamic indexed movement has no start indices"));
    };
    if indices.len() == 1 && shape::rank(ctx, first.shape)? == 1 {
        let geometry = geometry(ctx, first)?;
        if geometry.dimensions() != [rank] {
            return Err(malformed(
                "dynamic start-index vector has the wrong runtime length",
            ));
        }
        let element = shape::element_type(ctx, first.shape)?;
        if !element.is_integer() {
            return Err(malformed("dynamic start indices must be integers"));
        }
        return (0..rank)
            .map(|index| {
                let offset =
                    checked_offset(&[index], geometry.strides(), geometry.element_bytes())?;
                let end = offset
                    .checked_add(geometry.element_bytes())
                    .ok_or_else(|| malformed("dynamic start-index byte range overflow"))?;
                let bytes = first
                    .storage
                    .get(offset..end)
                    .ok_or_else(|| malformed("dynamic start-index storage is incomplete"))?;
                decode_integer(element, bytes)
            })
            .collect();
    }
    if indices.len() != rank {
        return Err(malformed(
            "dynamic start-index count differs from operand rank",
        ));
    }
    indices
        .iter()
        .map(|literal| {
            if shape::rank(ctx, literal.shape)? != 0 {
                return Err(malformed("dynamic start indices must be scalars"));
            }
            let element = shape::element_type(ctx, literal.shape)?;
            if !element.is_integer() {
                return Err(malformed("dynamic start indices must be integers"));
            }
            let geometry = geometry(ctx, literal)?;
            let offset = checked_offset(&[], geometry.strides(), geometry.element_bytes())?;
            let end = offset
                .checked_add(geometry.element_bytes())
                .ok_or_else(|| malformed("dynamic start-index byte range overflow"))?;
            let bytes = literal
                .storage
                .get(offset..end)
                .ok_or_else(|| malformed("dynamic start-index storage is incomplete"))?;
            decode_integer(element, bytes)
        })
        .collect()
}

fn decode_integer(element: ElementType, bytes: &[u8]) -> Result<i128> {
    if !element.is_integer() || bytes.len() > 8 {
        return Err(malformed(
            "dynamic start index has an unsupported integer width",
        ));
    }
    let mut encoded = [0; 8];
    encoded[..bytes.len()].copy_from_slice(bytes);
    let bits = u64::from_le_bytes(encoded);
    let width = element.bit_width();
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let bits = bits & mask;
    if element.is_signed() && bits & (1u64 << (width - 1)) != 0 {
        Ok(i128::from(bits) - (1i128 << width))
    } else {
        Ok(i128::from(bits))
    }
}

fn clamp_index(index: i128, maximum: usize) -> Result<usize> {
    let maximum =
        i128::try_from(maximum).map_err(|_| malformed("dynamic index bound exceeds i128"))?;
    usize::try_from(index.clamp(0, maximum))
        .map_err(|_| malformed("clamped dynamic index exceeds host addressing"))
}

fn allocate_result(
    ctx: &Context,
    shape: TypeHandle,
    runtime_dimensions: &[usize],
    opcode: Opcode,
) -> Result<Literal> {
    if shape::rank(ctx, shape)? != runtime_dimensions.len() {
        return Err(malformed("dynamic indexed result rank mismatch"));
    }
    let bounds = shape::dimensions(ctx, shape)?;
    let mut element_count = 1usize;
    let mut dynamic_sizes = if shape::is_dynamic(ctx, shape) {
        Vec::with_capacity(bounds.len())
    } else {
        Vec::new()
    };
    for (axis, &runtime) in runtime_dimensions.iter().enumerate() {
        let bound = match shape::dimension(ctx, shape, axis)? {
            Dimension::Static(bound) | Dimension::Bounded(bound) => usize::try_from(bound)
                .map_err(|_| malformed("negative dynamic indexed result bound"))?,
            Dimension::Unbounded => {
                return Err(unsupported_value(
                    opcode,
                    "unbounded output allocation is not implemented",
                ));
            }
        };
        if runtime > bound {
            return Err(malformed("dynamic indexed result exceeds its bound"));
        }
        if matches!(shape::dimension(ctx, shape, axis)?, Dimension::Static(_)) && runtime != bound {
            return Err(unsupported_value(
                opcode,
                "runtime extent conflicts with a static result dimension",
            ));
        }
        element_count = element_count
            .checked_mul(bound)
            .ok_or_else(|| malformed("dynamic indexed result element count overflow"))?;
        if shape::is_dynamic(ctx, shape) {
            dynamic_sizes.push(
                i64::try_from(runtime)
                    .map_err(|_| malformed("dynamic indexed runtime extent exceeds i64"))?,
            );
        }
    }
    let bytes = element_count
        .checked_mul(shape::element_type(ctx, shape)?.bit_width().div_ceil(8) as usize)
        .ok_or_else(|| malformed("dynamic indexed result byte count overflow"))?;
    Literal::from_dynamic_storage_bytes(ctx, shape, vec![0; bytes], dynamic_sizes)
}

fn copy_raw_element(
    source: &Literal,
    destination: &mut Literal,
    source_offset: usize,
    destination_offset: usize,
    width: usize,
) -> Result<()> {
    let source_end = source_offset
        .checked_add(width)
        .ok_or_else(|| malformed("dynamic indexed source byte range overflow"))?;
    let destination_end = destination_offset
        .checked_add(width)
        .ok_or_else(|| malformed("dynamic indexed destination byte range overflow"))?;
    let source_bytes = source
        .storage
        .get(source_offset..source_end)
        .ok_or_else(|| malformed("dynamic indexed source storage is incomplete"))?;
    let destination_bytes = destination
        .storage
        .get_mut(destination_offset..destination_end)
        .ok_or_else(|| malformed("dynamic indexed destination storage is incomplete"))?;
    destination_bytes.copy_from_slice(source_bytes);
    Ok(())
}

fn copy_allocation(
    source: &Literal,
    destination: &mut Literal,
    source_strides: &[usize],
    destination_strides: &[usize],
    width: usize,
    bounds: &[usize],
) -> Result<()> {
    visit_indices(bounds, &mut |index| {
        let from = checked_offset(index, source_strides, width)?;
        let to = checked_offset(index, destination_strides, width)?;
        copy_raw_element(source, destination, from, to, width)
    })
}

fn checked_offset(index: &[usize], strides: &[usize], width: usize) -> Result<usize> {
    if index.len() != strides.len() {
        return Err(malformed("dynamic indexed storage rank mismatch"));
    }
    let elements = index
        .iter()
        .zip(strides)
        .try_fold(0usize, |sum, (&coordinate, &stride)| {
            coordinate
                .checked_mul(stride)
                .and_then(|term| sum.checked_add(term))
        })
        .ok_or_else(|| malformed("dynamic indexed storage offset overflow"))?;
    elements
        .checked_mul(width)
        .ok_or_else(|| malformed("dynamic indexed storage byte offset overflow"))
}

fn visit_indices(
    dimensions: &[usize],
    visitor: &mut impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    if dimensions.is_empty() {
        return visitor(&[]);
    }
    if dimensions.contains(&0) {
        return Ok(());
    }
    let mut index = vec![0; dimensions.len()];
    loop {
        visitor(&index)?;
        let mut axis = dimensions.len();
        while axis > 0 {
            axis -= 1;
            index[axis] += 1;
            if index[axis] < dimensions[axis] {
                break;
            }
            index[axis] = 0;
        }
        if axis == 0 && index[0] == 0 {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_offset_reports_overflow() {
        assert!(checked_offset(&[usize::MAX, 1], &[1, 1], 1).is_err());
    }

    #[test]
    fn dynamic_slice_rejects_a_result_shape_mismatch() {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let source_shape = shape::array(&ctx, ElementType::S32, vec![4]).unwrap();
        let source = Literal::from_bits(&ctx, source_shape, &[1, 2, 3, 4]).unwrap();
        let source_geometry = geometry(&ctx, &source).unwrap();
        let target = shape::array(&ctx, ElementType::S32, vec![2]).unwrap();
        assert!(
            dynamic_slice(
                &ctx,
                &source,
                target,
                &source_geometry,
                &[0],
                &Axes(vec![3]),
            )
            .is_err()
        );
    }
}
