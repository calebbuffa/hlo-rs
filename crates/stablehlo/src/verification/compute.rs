//! Shared verification rules for native compute and reduction operations.
use crate::attributes::{ConvDimensionNumbersAttr, DotDimensionNumbersAttr};

fn compatible(a: i64, b: i64) -> bool {
    a < 0 || b < 0 || a == b
}
fn axes(rank: usize, batch: &[i64], contract: &[i64]) -> Option<Vec<bool>> {
    let mut used = vec![false; rank];
    for &d in batch.iter().chain(contract) {
        let axis = usize::try_from(d).ok()?;
        if axis >= rank || used[axis] {
            return None;
        }
        used[axis] = true;
    }
    Some(used)
}

/// Checks dot-general axis uniqueness, paired extent equality and result rank.
pub fn validate_dot_general(
    lhs: &[i64],
    rhs: &[i64],
    result: &[i64],
    dims: &DotDimensionNumbersAttr,
) -> bool {
    if dims.lhs_batching_dimensions.len() != dims.rhs_batching_dimensions.len()
        || dims.lhs_contracting_dimensions.len() != dims.rhs_contracting_dimensions.len()
    {
        return false;
    }
    let Some(lused) = axes(
        lhs.len(),
        &dims.lhs_batching_dimensions,
        &dims.lhs_contracting_dimensions,
    ) else {
        return false;
    };
    let Some(rused) = axes(
        rhs.len(),
        &dims.rhs_batching_dimensions,
        &dims.rhs_contracting_dimensions,
    ) else {
        return false;
    };
    for (&a, &b) in dims
        .lhs_batching_dimensions
        .iter()
        .zip(&dims.rhs_batching_dimensions)
        .chain(
            dims.lhs_contracting_dimensions
                .iter()
                .zip(&dims.rhs_contracting_dimensions),
        )
    {
        if !compatible(lhs[a as usize], rhs[b as usize]) {
            return false;
        }
    }
    let mut shape = Vec::new();
    shape.extend(
        dims.lhs_batching_dimensions
            .iter()
            .zip(&dims.rhs_batching_dimensions)
            .map(|(&a, &b)| {
                if lhs[a as usize] < 0 {
                    rhs[b as usize]
                } else {
                    lhs[a as usize]
                }
            }),
    );
    shape.extend(
        lhs.iter()
            .enumerate()
            .filter_map(|(i, &d)| (!lused[i]).then_some(d)),
    );
    shape.extend(
        rhs.iter()
            .enumerate()
            .filter_map(|(i, &d)| (!rused[i]).then_some(d)),
    );
    shape.len() == result.len() && shape.iter().zip(result).all(|(&a, &b)| compatible(a, b))
}

/// Verifies that convolution dimension records are complete permutations.
pub fn valid_convolution_dimensions(rank: usize, dims: &ConvDimensionNumbersAttr) -> bool {
    fn permutation(rank: usize, first: i64, second: i64, spatial: &[i64]) -> bool {
        if spatial.len() + 2 != rank {
            return false;
        }
        let mut seen = vec![false; rank];
        for &d in [first, second].iter().chain(spatial) {
            let Ok(axis) = usize::try_from(d) else {
                return false;
            };
            if axis >= rank || seen[axis] {
                return false;
            }
            seen[axis] = true
        }
        true
    }
    permutation(
        rank,
        dims.input_batch_dimension,
        dims.input_feature_dimension,
        &dims.input_spatial_dimensions,
    ) && permutation(
        rank,
        dims.kernel_input_feature_dimension,
        dims.kernel_output_feature_dimension,
        &dims.kernel_spatial_dimensions,
    ) && permutation(
        rank,
        dims.output_batch_dimension,
        dims.output_feature_dimension,
        &dims.output_spatial_dimensions,
    )
}

/// Verifies a declared reduce dimension list against the operand rank.
pub fn valid_reduction_dimensions(rank: usize, dimensions: &[u64]) -> bool {
    let mut used = vec![false; rank];
    dimensions.iter().all(|&d| {
        let i = d as usize;
        if i >= rank || used[i] {
            false
        } else {
            used[i] = true;
            true
        }
    })
}
