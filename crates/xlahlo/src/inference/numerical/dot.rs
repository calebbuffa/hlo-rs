//! General, sparse, ragged, and block-scaled dot-product inference.
use super::{array, axes, compatible_size, element, output, require};
use crate::{
    Diagnostic,
    attributes::{
        DotDimensionNumbers, RaggedDotDimensionNumbers, SparsityConfig, TensorSparsityConfig,
    },
    shape::{Shape, UNBOUNDED_SIZE},
    types::ElementType,
};
use pliron::{context::Context, result::Result, r#type::TypeHandle};

pub(super) fn dot_element(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    preferred: Option<ElementType>,
) -> Result<ElementType> {
    match preferred {
        Some(et) => Ok(et),
        None => element(ctx, lhs)?.higher_precision(element(ctx, rhs)?),
    }
}

pub(super) fn sparse_size(
    size: i64,
    axis: i64,
    config: Option<&TensorSparsityConfig>,
) -> Result<i64> {
    let Some(config) = config.filter(|c| c.dimension == axis) else {
        return Ok(size);
    };
    require(
        config.num_non_zero > 0 && config.block_size > 0,
        "nonzero count and block size must be positive",
    )?;
    if size == UNBOUNDED_SIZE {
        return Ok(size);
    }
    require(
        size % config.num_non_zero == 0,
        "stored contracting size must be divisible by nonzero count",
    )?;
    i64::try_from(
        i128::from(size) * i128::from(config.block_size) / i128::from(config.num_non_zero),
    )
    .map_err(|_| {
        pliron::verify_error_noloc!(Diagnostic::Inference {
            message: "expanded sparse contracting size exceeds i64".into()
        })
    })
}

pub(super) fn validate(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    d: &DotDimensionNumbers,
    sparsity: Option<&SparsityConfig>,
) -> Result<(Vec<bool>, Vec<bool>)> {
    let ld = Shape::dimensions(ctx, array(ctx, lhs)?)?;
    let rd = Shape::dimensions(ctx, array(ctx, rhs)?)?;
    let left: Vec<_> = d
        .lhs_contracting_dimensions
        .0
        .iter()
        .chain(&d.lhs_batch_dimensions.0)
        .copied()
        .collect();
    let right: Vec<_> = d
        .rhs_contracting_dimensions
        .0
        .iter()
        .chain(&d.rhs_batch_dimensions.0)
        .copied()
        .collect();
    require(
        left.iter().all(|&a| a >= 0 && (a as usize) < ld.len())
            && right.iter().all(|&a| a >= 0 && (a as usize) < rd.len()),
        &format!(
            "a dot dimension number is out of range: lhs axes {left:?} for rank {}, rhs axes {right:?} for rank {}",
            ld.len(),
            rd.len(),
        ),
    )?;
    let left = axes(ld.len(), &left, false)?;
    let right = axes(rd.len(), &right, false)?;
    require(
        d.lhs_contracting_dimensions.0.len() == d.rhs_contracting_dimensions.0.len(),
        "left and right contracting dimension cardinalities differ",
    )?;
    for (&a, &b) in d
        .lhs_contracting_dimensions
        .0
        .iter()
        .zip(&d.rhs_contracting_dimensions.0)
    {
        let a = sparse_size(ld[a as usize], a, sparsity.and_then(|s| s.lhs.as_ref()))?;
        let b = sparse_size(rd[b as usize], b, sparsity.and_then(|s| s.rhs.as_ref()))?;
        require(
            compatible_size(a, b),
            &format!(
                "paired contracting dimension sizes differ: expanded lhs extent {a}, rhs extent {b}"
            ),
        )?;
    }
    require(
        d.lhs_batch_dimensions.0.len() == d.rhs_batch_dimensions.0.len(),
        "left and right batch dimension cardinalities differ",
    )?;
    for (&a, &b) in d
        .lhs_batch_dimensions
        .0
        .iter()
        .zip(&d.rhs_batch_dimensions.0)
    {
        require(
            compatible_size(ld[a as usize], rd[b as usize]),
            "paired batch dimension sizes differ",
        )?;
    }
    Ok((left, right))
}

/// Infer a generalized dot, optionally selecting an explicit result element type.
///
/// Both operands must be arrays, not buffer-wrapped arrays.
pub fn infer_dot(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    d: &DotDimensionNumbers,
    preferred: Option<ElementType>,
) -> Result<TypeHandle> {
    infer_dot_with_sparsity(ctx, lhs, rhs, d, preferred, None)
}

/// Infer a dot after expanding sparse contracting extents with checked arithmetic.
///
/// Data operands must be arrays; sparsity does not permit buffer operands.
pub fn infer_dot_with_sparsity(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    d: &DotDimensionNumbers,
    preferred: Option<ElementType>,
    sparsity: Option<&SparsityConfig>,
) -> Result<TypeHandle> {
    require(Shape::is_array(ctx, lhs), "dot lhs must be an array")?;
    require(Shape::is_array(ctx, rhs), "dot rhs must be an array")?;
    let (left, right) = validate(ctx, lhs, rhs, d, sparsity)?;
    let mut dims = Vec::new();
    for &axis in &d.lhs_batch_dimensions.0 {
        dims.push(Shape::dimension(ctx, lhs, axis as usize)?);
    }
    for (axis, used) in left.into_iter().enumerate() {
        if !used {
            dims.push(Shape::dimension(ctx, lhs, axis)?);
        }
    }
    for (axis, used) in right.into_iter().enumerate() {
        if !used {
            dims.push(Shape::dimension(ctx, rhs, axis)?);
        }
    }
    output(ctx, dot_element(ctx, lhs, rhs, preferred)?, dims)
}

/// Infer ragged-dot output for ragged batch, contracting, or noncontracting axes.
///
/// Data operands and group sizes must be arrays, not buffer-wrapped arrays.
pub fn infer_ragged_dot(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    group_sizes: TypeHandle,
    d: &RaggedDotDimensionNumbers,
    preferred: Option<ElementType>,
) -> Result<TypeHandle> {
    require(Shape::is_array(ctx, lhs), "ragged-dot lhs must be an array")?;
    require(Shape::is_array(ctx, rhs), "ragged-dot rhs must be an array")?;
    require(
        Shape::is_array(ctx, group_sizes),
        "ragged-dot group sizes must be an array",
    )?;
    let groups = group_sizes;
    let dot = &d.dot_dimension_numbers;
    let (left, right) = validate(ctx, lhs, rhs, dot, None)?;
    require(
        d.lhs_ragged_dimensions.0.len() == 1,
        "exactly one lhs ragged dimension is required",
    )?;
    let ragged = d.lhs_ragged_dimensions.0[0];
    axes(left.len(), &[ragged], false)?;
    let batch = dot.lhs_batch_dimensions.0.contains(&ragged);
    let contracting = dot.lhs_contracting_dimensions.0.contains(&ragged);
    let mut prefix = Vec::new();
    if !batch {
        prefix.extend_from_slice(&dot.lhs_batch_dimensions.0);
    }
    if batch {
        prefix.extend(
            dot.lhs_batch_dimensions
                .0
                .iter()
                .copied()
                .take_while(|&a| a != ragged),
        );
    } else if contracting {
        prefix.extend(
            dot.lhs_contracting_dimensions
                .0
                .iter()
                .copied()
                .take_while(|&a| a != ragged),
        );
    } else {
        prefix.extend(
            (0..left.len())
                .filter(|&a| !left[a])
                .map(|a| a as i64)
                .take_while(|&a| a != ragged),
        );
    }
    let gd = Shape::dimensions(ctx, groups)?;
    require(!gd.is_empty(), "group sizes must have positive rank")?;
    let ld = Shape::dimensions(ctx, lhs)?;
    require(
        gd.len() == 1
            || (gd.len() == prefix.len() + 1
                && prefix
                    .iter()
                    .enumerate()
                    .all(|(i, &a)| gd[i] == ld[a as usize])),
        "group sizes prefix differs from required lhs dimensions",
    )?;
    for &axis in &d.rhs_group_dimensions.0 {
        axes(right.len(), &[axis], false)?;
        require(
            !right[axis as usize],
            "rhs group dimension cannot be batch or contracting",
        )?;
    }
    if batch || contracting {
        require(
            d.rhs_group_dimensions.0.is_empty(),
            "ragged batch/contracting modes require zero rhs group dimensions",
        )?;
    } else {
        require(
            d.rhs_group_dimensions.0.len() == 1,
            "noncontracting mode requires exactly one rhs group dimension",
        )?;
        let rd = Shape::dimensions(ctx, rhs)?;
        require(
            compatible_size(gd[gd.len() - 1], rd[d.rhs_group_dimensions.0[0] as usize]),
            "rhs group bound differs from group count",
        )?;
    }
    let mut dims = Vec::new();
    if contracting {
        dims.push(Shape::dimension(ctx, groups, gd.len() - 1)?);
    }
    for &axis in &dot.lhs_batch_dimensions.0 {
        dims.push(Shape::dimension(ctx, lhs, axis as usize)?);
    }
    for (axis, used) in left.into_iter().enumerate() {
        if !used {
            dims.push(Shape::dimension(ctx, lhs, axis)?);
        }
    }
    for (axis, used) in right.into_iter().enumerate() {
        if !used && !d.rhs_group_dimensions.0.contains(&(axis as i64)) {
            dims.push(Shape::dimension(ctx, rhs, axis)?);
        }
    }
    output(ctx, dot_element(ctx, lhs, rhs, preferred)?, dims)
}

/// Infer a dot with explicit block-scale operands and validate scale geometry.
///
/// After scale validation, the original data operands must satisfy the dot
/// array contract; buffer views used to inspect geometry do not change it.
pub fn infer_scaled_dot(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    lhs_scale: TypeHandle,
    rhs_scale: TypeHandle,
    d: &DotDimensionNumbers,
    preferred: Option<ElementType>,
) -> Result<TypeHandle> {
    let lhs_array = array(ctx, lhs)?;
    let rhs_array = array(ctx, rhs)?;
    let lhs_scale = array(ctx, lhs_scale)?;
    let rhs_scale = array(ctx, rhs_scale)?;
    validate(ctx, lhs_array, rhs_array, d, None)?;
    for (operand, scale, batch, contracting) in [
        (
            lhs_array,
            lhs_scale,
            &d.lhs_batch_dimensions.0,
            &d.lhs_contracting_dimensions.0,
        ),
        (
            rhs_array,
            rhs_scale,
            &d.rhs_batch_dimensions.0,
            &d.rhs_contracting_dimensions.0,
        ),
    ] {
        let od = Shape::dimensions(ctx, operand)?;
        let sd = Shape::dimensions(ctx, scale)?;
        if !sd.is_empty() {
            let categories: Vec<_> = batch.iter().chain(contracting).copied().collect();
            axes(sd.len(), &categories, false)?;
        }
        let dummy = element(ctx, operand)? == ElementType::BF16
            && element(ctx, scale)? == ElementType::BF16
            && sd.iter().all(|&n| n == 1)
            && od.iter().any(|&n| n != 1);
        if dummy {
            continue;
        }
        require(
            sd.len() == od.len() && contracting.len() == 1,
            "scale rank must match data rank and exactly one contracting axis is required",
        )?;
        for (&data, &scale) in od.iter().zip(sd.iter()) {
            require(
                scale != 0 && data.checked_rem(scale) == Some(0),
                "data bounds must be divisible by nonzero scale bounds",
            )?;
        }
    }
    require(
        Shape::rank(ctx, lhs_scale)? != 0 || Shape::rank(ctx, rhs_scale)? != 0,
        "at least one scale must be nonscalar",
    )?;
    infer_dot(ctx, lhs, rhs, d, preferred)
}
