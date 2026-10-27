use super::{
    builder::{Builder, ranked},
    unsupported,
};
use crate::{Result, ops::RaggedDotOp};
use pliron::{
    op::Op,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use stablehlo::{
    attributes::{ComparisonDirection as D, DotDimensionNumbersAttr},
    ops as hlo,
    types::{ElementType, TensorType},
};

pub(super) fn lower(b: &mut Builder<'_>) -> Result<Value> {
    let source = RaggedDotOp::from_operation(b.source);
    let lhs = source.lhs(b.ctx);
    let rhs = source.rhs(b.ctx);
    let groups = source.group_sizes(b.ctx);
    let dimensions = source
        .dimension_numbers(b.ctx)
        .ok_or_else(|| unsupported(b.ctx, b.source, "ragged dot dimensions are missing"))?
        .dimensions();
    let precision = source.precision_config(b.ctx).map(|p| {
        p.into_iter()
            .map(|p| match p {
                crate::attributes::Precision::Default => stablehlo::attributes::Precision::Default,
                crate::attributes::Precision::High => stablehlo::attributes::Precision::High,
                crate::attributes::Precision::Highest => stablehlo::attributes::Precision::Highest,
            })
            .collect::<Vec<_>>()
    });
    let result = b.source.deref(b.ctx).get_result(0).get_type(b.ctx);
    let mut dot_dimensions = DotDimensionNumbersAttr {
        lhs_batching_dimensions: dimensions.lhs_batching_dimensions.clone(),
        rhs_batching_dimensions: dimensions.rhs_batching_dimensions,
        lhs_contracting_dimensions: dimensions.lhs_contracting_dimensions.clone(),
        rhs_contracting_dimensions: dimensions.rhs_contracting_dimensions,
    };
    let ragged = dimensions.lhs_ragged_dimensions[0] as usize;
    let dot = |b: &mut Builder<'_>, lhs, rhs, ty, dimensions| {
        let mut op = hlo::DotGeneralOp::new(b.ctx, lhs, rhs, ty, dimensions);
        if let Some(precision) = &precision {
            op = op.with_precision(b.ctx, precision.clone());
        }
        b.emit(op)
    };
    if dimensions
        .lhs_batching_dimensions
        .contains(&(ragged as i64))
    {
        return dot(b, lhs, rhs, result, dot_dimensions);
    }
    let contracting = dimensions
        .lhs_contracting_dimensions
        .contains(&(ragged as i64));
    let (lhs_shape, _) = ranked(b.ctx, lhs.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "ragged dot requires ranked operands"))?;
    let (group_shape, _) = ranked(b.ctx, groups.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "ragged group sizes must be ranked"))?;
    let group_axis = group_shape.len() - 1;
    let count = group_shape[group_axis];
    if count < 0 {
        return Err(unsupported(
            b.ctx,
            b.source,
            "dynamic ragged group count is not supported upstream",
        ));
    }
    let (output_shape, output_element) = ranked(b.ctx, result)
        .ok_or_else(|| unsupported(b.ctx, b.source, "ragged dot result must be ranked"))?;
    let mut start_sizes = Vec::new();
    for (axis, &extent) in group_shape[..group_axis].iter().enumerate() {
        start_sizes.push(b.dimension(groups, axis, extent)?);
    }
    let start_ty = tensor_type(
        b,
        &group_shape[..group_axis],
        ElementType::I64.native(b.ctx),
    );
    let start_shape = b.shape_vector(start_sizes)?;
    let mut start = b.zero(start_ty, Some(start_shape))?;
    let index_ty = tensor_type(b, &lhs_shape, ElementType::I64.native(b.ctx));
    let lhs_sizes = b.shape(lhs)?;
    let iota = hlo::DynamicIotaOp::new(b.ctx, lhs_sizes, index_ty, ragged);
    let indices = b.emit(iota)?;
    let lhs_zero = b.zero(lhs.get_type(b.ctx), Some(lhs_sizes))?;
    let mut prefix = dimensions.lhs_batching_dimensions.clone();
    if contracting {
        prefix.extend(
            dimensions
                .lhs_contracting_dimensions
                .iter()
                .copied()
                .take_while(|&dim| dim != ragged as i64),
        );
    } else {
        prefix.extend((0..ragged as i64).filter(|dim| {
            !dimensions.lhs_batching_dimensions.contains(dim)
                && !dimensions.lhs_contracting_dimensions.contains(dim)
        }));
    }
    let axes: Vec<_> = prefix
        .into_iter()
        .take(group_axis)
        .map(|dim| dim as usize)
        .collect();
    let rhs_group = if contracting {
        None
    } else {
        Some(dimensions.rhs_group_dimensions[0] as usize)
    };
    if let Some(group) = rhs_group {
        for dim in dot_dimensions
            .rhs_batching_dimensions
            .iter_mut()
            .chain(&mut dot_dimensions.rhs_contracting_dimensions)
        {
            if *dim > group as i64 {
                *dim -= 1;
            }
        }
    }
    let dot_shape = if contracting {
        &output_shape[1..]
    } else {
        &output_shape
    };
    let dot_ty = tensor_type(b, dot_shape, output_element);
    let mut partials = Vec::new();
    for group in 0..count {
        let index = b.integer(group, ElementType::I64)?;
        let group_size = b.slice_axis(groups, group_axis, index)?;
        let size_ty = tensor_type(
            b,
            &group_shape[..group_axis],
            ElementType::I64.native(b.ctx),
        );
        let group_size = b.convert(group_size, size_ty)?;
        let end = b.add(start, group_size)?;
        let from = b.broadcast(start, index_ty, axes.clone(), Some(lhs_sizes))?;
        let until = b.broadcast(end, index_ty, axes.clone(), Some(lhs_sizes))?;
        let ge = b.compare(indices, from, D::Ge)?;
        let lt = b.compare(indices, until, D::Lt)?;
        let mask = b.and(ge, lt)?;
        let masked = b.select(mask, lhs, lhs_zero)?;
        let right = if let Some(axis) = rhs_group {
            b.slice_axis(rhs, axis, index)?
        } else {
            rhs
        };
        let partial = dot(b, masked, right, dot_ty, dot_dimensions.clone())?;
        partials.push(partial);
        start = end;
    }
    if contracting {
        if partials.is_empty() {
            return zero_output(b, result, lhs, rhs, &dot_dimensions, true);
        }
        let mut expanded = Vec::new();
        let mut shape = vec![1];
        shape.extend(dot_shape);
        let ty = tensor_type(b, &shape, output_element);
        for partial in partials {
            let one = b.integer(1, ElementType::I64)?;
            let mut sizes = vec![one];
            for (dim, &extent) in dot_shape.iter().enumerate() {
                sizes.push(b.dimension(partial, dim, extent)?);
            }
            expanded.push(b.reshape(partial, ty, sizes)?);
        }
        let op = hlo::ConcatenateOp::new(b.ctx, expanded, result, 0);
        b.emit(op)
    } else {
        let mut partials = partials.into_iter();
        let Some(mut sum) = partials.next() else {
            return zero_output(b, result, lhs, rhs, &dot_dimensions, false);
        };
        for next in partials {
            sum = b.add(sum, next)?;
        }
        Ok(sum)
    }
}

fn tensor_type(b: &Builder<'_>, shape: &[i64], element: TypeHandle) -> TypeHandle {
    TensorType::get(b.ctx, shape.to_vec(), element).into()
}

fn zero_output(
    b: &mut Builder<'_>,
    result: TypeHandle,
    lhs: Value,
    rhs: Value,
    dimensions: &DotDimensionNumbersAttr,
    contracting: bool,
) -> Result<Value> {
    let mut sizes = Vec::new();
    if contracting {
        sizes.push(b.integer(0, ElementType::I64)?);
    }
    let (ls, _) = ranked(b.ctx, lhs.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "ragged lhs must be ranked"))?;
    let (rs, _) = ranked(b.ctx, rhs.get_type(b.ctx))
        .ok_or_else(|| unsupported(b.ctx, b.source, "ragged rhs must be ranked"))?;
    for &axis in &dimensions.lhs_batching_dimensions {
        sizes.push(b.dimension(lhs, axis as usize, ls[axis as usize])?);
    }
    for (axis, &extent) in ls.iter().enumerate() {
        if !dimensions.lhs_batching_dimensions.contains(&(axis as i64))
            && !dimensions
                .lhs_contracting_dimensions
                .contains(&(axis as i64))
        {
            sizes.push(b.dimension(lhs, axis, extent)?);
        }
    }
    // In the noncontracting mode rhs has a group axis. Derive its position
    // from the original geometry, not the adjusted dimensions used for a slice.
    let original = RaggedDotOp::from_operation(b.source)
        .dimension_numbers(b.ctx)
        .ok_or_else(|| unsupported(b.ctx, b.source, "missing ragged dimensions"))?
        .dimensions();
    for (axis, &extent) in rs.iter().enumerate() {
        if !original.rhs_batching_dimensions.contains(&(axis as i64))
            && !original.rhs_contracting_dimensions.contains(&(axis as i64))
            && !original.rhs_group_dimensions.contains(&(axis as i64))
        {
            sizes.push(b.dimension(rhs, axis, extent)?);
        }
    }
    let shape = b.shape_vector(sizes)?;
    b.zero(result, Some(shape))
}
