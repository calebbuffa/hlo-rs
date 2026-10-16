use super::*;
use crate::proto::xla_data::{
    DotDimensionNumbers, RaggedDotDimensionNumbers, SparsityConfig,
    sparsity_config::TensorSparsityConfig,
};

pub(super) fn element(
    lhs: &Shape,
    rhs: &Shape,
    preferred: Option<PrimitiveType>,
) -> Result<PrimitiveType> {
    match preferred {
        Some(element) => Ok(element),
        None => lhs.element_type().higher_precision(rhs.element_type()),
    }
}

pub(super) fn validate(
    lhs: &Shape,
    rhs: &Shape,
    dimensions: &DotDimensionNumbers,
    sparsity: &SparsityConfig,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let left = dimensions
        .lhs_contracting_dimensions
        .iter()
        .chain(&dimensions.lhs_batch_dimensions)
        .copied()
        .collect::<Vec<_>>();
    let right = dimensions
        .rhs_contracting_dimensions
        .iter()
        .chain(&dimensions.rhs_batch_dimensions)
        .copied()
        .collect::<Vec<_>>();
    axes(lhs.rank()?, &left, false)?;
    axes(rhs.rank()?, &right, false)?;
    if dimensions.lhs_contracting_dimensions.len() != dimensions.rhs_contracting_dimensions.len()
        || dimensions.lhs_batch_dimensions.len() != dimensions.rhs_batch_dimensions.len()
    {
        return Err(Error::new(
            "dot",
            "left and right contraction/batch cardinalities differ",
        ));
    }
    for (&a, &b) in dimensions
        .lhs_contracting_dimensions
        .iter()
        .zip(&dimensions.rhs_contracting_dimensions)
    {
        let a = sparse_size(lhs.dimensions()[a as usize], a, sparsity.lhs.as_ref())?;
        let b = sparse_size(rhs.dimensions()[b as usize], b, sparsity.rhs.as_ref())?;
        if a != b && a != crate::UNBOUNDED_SIZE && b != crate::UNBOUNDED_SIZE {
            return Err(Error::new(
                "dot",
                "paired contracting dimension sizes differ",
            ));
        }
    }
    for (&a, &b) in dimensions
        .lhs_batch_dimensions
        .iter()
        .zip(&dimensions.rhs_batch_dimensions)
    {
        let (a, b) = (lhs.dimensions()[a as usize], rhs.dimensions()[b as usize]);
        if a != b && a != crate::UNBOUNDED_SIZE && b != crate::UNBOUNDED_SIZE {
            return Err(Error::new("dot", "paired batch dimension sizes differ"));
        }
    }
    Ok((left, right))
}

pub(super) fn array(shape: &Shape) -> Result<Shape> {
    shape.array_view()
}

pub(super) fn sparse_size(
    size: i64,
    axis: i64,
    sparsity: Option<&TensorSparsityConfig>,
) -> Result<i64> {
    let Some(config) = sparsity.filter(|c| c.dimension == axis) else {
        return Ok(size);
    };
    if config.num_non_zero <= 0 || config.block_size <= 0 {
        return Err(Error::new(
            "sparsity",
            "nonzero count and block size must be positive",
        ));
    }
    if size == crate::UNBOUNDED_SIZE {
        return Ok(size);
    }
    if size % config.num_non_zero != 0 {
        return Err(Error::new(
            "sparsity",
            "stored contracting size must be divisible by nonzero count",
        ));
    }
    i64::try_from(
        i128::from(size) * i128::from(config.block_size) / i128::from(config.num_non_zero),
    )
    .map_err(|_| Error::new("overflow", "expanded sparse contracting size exceeds i64"))
}

pub fn infer_ragged_dot(
    lhs: &Shape,
    rhs: &Shape,
    group_sizes: &Shape,
    dimensions: &RaggedDotDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
) -> Result<Shape> {
    let lhs = array(lhs)?;
    let rhs = array(rhs)?;
    let groups = array(group_sizes)?;
    let dot = dimensions
        .dot_dimension_numbers
        .as_ref()
        .cloned()
        .unwrap_or_default();
    let (left, right) = validate(&lhs, &rhs, &dot, &Default::default())?;
    let &[ragged] = dimensions.lhs_ragged_dimensions.as_slice() else {
        return Err(Error::new(
            "ragged-dot",
            "exactly one lhs ragged dimension is required",
        ));
    };
    axes(lhs.rank()?, &[ragged], false)?;
    let batch = dot.lhs_batch_dimensions.contains(&ragged);
    let contracting = dot.lhs_contracting_dimensions.contains(&ragged);
    let mut prefix = dot.lhs_batch_dimensions.clone();
    let candidates = if batch {
        prefix.clear();
        dot.lhs_batch_dimensions.clone()
    } else if contracting {
        dot.lhs_contracting_dimensions.clone()
    } else {
        (0..lhs.rank()? as i64)
            .filter(|a| {
                !dot.lhs_batch_dimensions.contains(a) && !dot.lhs_contracting_dimensions.contains(a)
            })
            .collect()
    };
    prefix.extend(candidates.into_iter().take_while(|&a| a != ragged));
    if groups.rank()? == 0 {
        return Err(Error::new(
            "ragged-dot",
            "group sizes must have positive rank",
        ));
    }

    if groups.rank()? != 1
        && (groups.rank()? != prefix.len() + 1
            || prefix
                .iter()
                .enumerate()
                .any(|(i, &axis)| groups.dimensions()[i] != lhs.dimensions()[axis as usize]))
    {
        return Err(Error::new(
            "ragged-dot",
            "group sizes prefix differs from required lhs dimensions",
        ));
    }
    let group_count = *groups
        .dimensions()
        .last()
        .ok_or_else(|| Error::new("ragged-dot", "missing group count"))?;
    for &axis in &dimensions.rhs_group_dimensions {
        axes(rhs.rank()?, &[axis], false)?;
        if dot.rhs_batch_dimensions.contains(&axis)
            || dot.rhs_contracting_dimensions.contains(&axis)
        {
            return Err(Error::new(
                "ragged-dot",
                "rhs group dimension cannot be batch or contracting",
            ));
        }
    }
    if batch || contracting {
        if !dimensions.rhs_group_dimensions.is_empty() {
            return Err(Error::new(
                "ragged-dot",
                "ragged batch/contracting modes require zero rhs group dimensions",
            ));
        }
    } else {
        let &[axis] = dimensions.rhs_group_dimensions.as_slice() else {
            return Err(Error::new(
                "ragged-dot",
                "noncontracting mode requires exactly one rhs group dimension",
            ));
        };
        let rhs_count = rhs.dimensions()[axis as usize];
        if group_count != rhs_count
            && group_count != crate::UNBOUNDED_SIZE
            && rhs_count != crate::UNBOUNDED_SIZE
        {
            return Err(Error::new(
                "ragged-dot",
                "rhs group bound differs from group count",
            ));
        }
    }
    let mut output = Vec::new();
    if contracting {
        output.push(groups.dimension(groups.rank()? - 1)?);
    }
    for &axis in &dot.lhs_batch_dimensions {
        output.push(lhs.dimension(axis as usize)?);
    }
    for axis in 0..lhs.rank()? {
        if !left.contains(&(axis as i64)) {
            output.push(lhs.dimension(axis)?);
        }
    }
    for axis in 0..rhs.rank()? {
        if !right.contains(&(axis as i64))
            && !dimensions.rhs_group_dimensions.contains(&(axis as i64))
        {
            output.push(rhs.dimension(axis)?);
        }
    }
    Shape::dynamic_array(element(&lhs, &rhs, preferred_element)?, output)?.with_default_layout()
}

pub fn infer_scaled_dot(
    lhs: &Shape,
    rhs: &Shape,
    lhs_scale: &Shape,
    rhs_scale: &Shape,
    dimensions: &DotDimensionNumbers,
    preferred_element: Option<PrimitiveType>,
) -> Result<Shape> {
    let lhs = array(lhs)?;
    let rhs = array(rhs)?;
    let lhs_scale = array(lhs_scale)?;
    let rhs_scale = array(rhs_scale)?;
    validate(&lhs, &rhs, dimensions, &Default::default())?;
    for (operand, scale, batch, contracting) in [
        (
            &lhs,
            &lhs_scale,
            &dimensions.lhs_batch_dimensions,
            &dimensions.lhs_contracting_dimensions,
        ),
        (
            &rhs,
            &rhs_scale,
            &dimensions.rhs_batch_dimensions,
            &dimensions.rhs_contracting_dimensions,
        ),
    ] {
        if scale.rank()? != 0 {
            let categories = batch.iter().chain(contracting).copied().collect::<Vec<_>>();
            axes(scale.rank()?, &categories, false)?;
        }
        let dummy = operand.element_type() == PrimitiveType::BF16
            && scale.element_type() == PrimitiveType::BF16
            && scale.dimensions().iter().all(|&d| d == 1)
            && operand.dimensions().iter().any(|&d| d != 1);
        if dummy {
            continue;
        }
        if scale.rank()? != operand.rank()? || contracting.len() != 1 {
            return Err(Error::new(
                "scaled-dot",
                "scale rank must match data rank and exactly one contracting axis is required",
            ));
        }
        for (&data, &scale) in operand.dimensions().iter().zip(scale.dimensions()) {
            if scale == 0 || data.checked_rem(scale) != Some(0) {
                return Err(Error::new(
                    "scaled-dot",
                    "data bounds must be divisible by nonzero scale bounds",
                ));
            }
        }
    }
    if lhs_scale.rank()? == 0 && rhs_scale.rank()? == 0 {
        return Err(Error::new(
            "scaled-dot",
            "at least one scale must be nonscalar",
        ));
    }
    super::infer_dot(&lhs, &rhs, dimensions, preferred_element)
}
