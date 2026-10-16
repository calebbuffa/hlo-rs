use super::{
    InferenceError, Specificity, bound, bounds, bounds_encoding, dimensions, encoding,
    merge_dimension, with_shape_and_encoding,
};
use crate::{DotDimensionNumbers, Type};

/// Infers dot-general dimensions and bounds using the pinned `inferDotGeneralOp`.
///
/// `result_template` supplies the result element metadata, including quantization;
/// its shape and encoding are replaced. StableHLO dot results do not necessarily
/// have the operand element type. Precision, algorithm and quantization
/// relationships are checked by operation verification, not this shape helper.
/// Scalar results drop nonempty input bounds instead of retaining an invalid
/// non-scalar bounds encoding through the C++ helper's empty-bounds fast path.
pub fn infer_dot_general_type(
    lhs: &Type,
    rhs: &Type,
    dimensions: &DotDimensionNumbers,
    result_template: &Type,
) -> Result<Type, InferenceError> {
    for (index, ty) in [lhs, rhs, result_template].into_iter().enumerate() {
        ty.verify()
            .map_err(|error| InferenceError::InvalidType { index, error })?;
        if super::dimensions(ty).is_none() {
            return Err(InferenceError::IncompatibleTypes { index });
        }
    }
    let lhs_shape = super::dimensions(lhs).expect("verified tensor");
    let rhs_shape = super::dimensions(rhs).expect("verified tensor");
    if dimensions.lhs_batching_dimensions.len() != dimensions.rhs_batching_dimensions.len()
        || dimensions.lhs_contracting_dimensions.len()
            != dimensions.rhs_contracting_dimensions.len()
    {
        return Err(InferenceError::DimensionPairCountMismatch);
    }
    let (lhs_batch, lhs_contract) = axes(
        lhs_shape.len(),
        &dimensions.lhs_batching_dimensions,
        &dimensions.lhs_contracting_dimensions,
    )?;
    let (rhs_batch, rhs_contract) = axes(
        rhs_shape.len(),
        &dimensions.rhs_batching_dimensions,
        &dimensions.rhs_contracting_dimensions,
    )?;
    let mut sizes = Vec::new();
    let mut output_bounds = Vec::new();
    for (&left, &right) in lhs_batch.iter().zip(&rhs_batch) {
        let (size, upper) = merge_dimension(
            sizes.len(),
            lhs_shape[left],
            rhs_shape[right],
            bound(lhs, left),
            bound(rhs, right),
            Specificity::Most,
        )?;
        sizes.push(size);
        output_bounds.push(upper);
    }
    for (&left, &right) in lhs_contract.iter().zip(&rhs_contract) {
        merge_dimension(
            left,
            lhs_shape[left],
            rhs_shape[right],
            bound(lhs, left),
            bound(rhs, right),
            Specificity::Most,
        )?;
    }
    for (ty, batch, contract) in [
        (lhs, lhs_batch, lhs_contract),
        (rhs, rhs_batch, rhs_contract),
    ] {
        for (axis, &size) in super::dimensions(ty)
            .expect("verified tensor")
            .iter()
            .enumerate()
        {
            if !batch.contains(&axis) && !contract.contains(&axis) {
                sizes.push(size);
                output_bounds.push(bound(ty, axis));
            }
        }
    }
    let output_encoding = if sizes.is_empty() {
        let prototype = if encoding(lhs).is_some() { lhs } else { rhs };
        if bounds(prototype).is_some_and(|bounds| !bounds.is_empty()) {
            None
        } else {
            encoding(prototype).cloned()
        }
    } else {
        bounds_encoding(output_bounds)
    };
    let result = with_shape_and_encoding(result_template, sizes, output_encoding);
    result
        .verify()
        .map_err(|error| InferenceError::InvalidType { index: 2, error })?;
    Ok(result)
}

/// Infers legacy rank-one/rank-two dot results using the same shape rules.
pub fn infer_dot_type(
    lhs: &Type,
    rhs: &Type,
    result_template: &Type,
) -> Result<Type, InferenceError> {
    let lhs_rank = dimensions(lhs)
        .ok_or(InferenceError::IncompatibleTypes { index: 0 })?
        .len();
    let rhs_rank = dimensions(rhs)
        .ok_or(InferenceError::IncompatibleTypes { index: 1 })?
        .len();
    if !(1..=2).contains(&lhs_rank) || !(1..=2).contains(&rhs_rank) {
        return Err(InferenceError::InvalidDotRank {
            lhs: lhs_rank,
            rhs: rhs_rank,
        });
    }
    infer_dot_general_type(
        lhs,
        rhs,
        &DotDimensionNumbers {
            lhs_batching_dimensions: vec![],
            rhs_batching_dimensions: vec![],
            lhs_contracting_dimensions: vec![(lhs_rank - 1) as i64],
            rhs_contracting_dimensions: vec![0],
        },
        result_template,
    )
}

fn axes(
    rank: usize,
    batch: &[i64],
    contract: &[i64],
) -> Result<(Vec<usize>, Vec<usize>), InferenceError> {
    let mut used = vec![false; rank];
    let mut parse = |values: &[i64]| {
        values
            .iter()
            .map(|&axis| {
                let entry = usize::try_from(axis)
                    .ok()
                    .and_then(|axis| used.get_mut(axis))
                    .ok_or_else(|| InferenceError::InvalidDimensions {
                        rank,
                        dimensions: values.to_vec(),
                    })?;
                if std::mem::replace(entry, true) {
                    return Err(InferenceError::InvalidDimensions {
                        rank,
                        dimensions: values.to_vec(),
                    });
                }
                Ok(axis as usize)
            })
            .collect::<Result<Vec<_>, _>>()
    };
    Ok((parse(batch)?, parse(contract)?))
}
