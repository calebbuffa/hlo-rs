//! Replica-group derivation from `stablehlo/dialect/ReplicaGroupUtils.cpp`.

use super::{AxisRefAttr, Diagnostic, MeshAttr};
use pliron::{common_traits::Verify, context::Context, result::Result};

struct GroupLayout {
    sizes: Vec<u64>,
    permutation: Vec<usize>,
    group_size: u64,
    total: u64,
}

impl MeshAttr {
    /// Number of devices in each group selected by full or split mesh axes.
    ///
    /// Validates references and mesh geometry without expanding device groups.
    pub fn replica_group_size(&self, ctx: &Context, axes: &[AxisRefAttr]) -> Result<u64> {
        Ok(self.group_layout(ctx, axes)?.group_size)
    }

    /// Expands mesh communication axes to ordered, explicit replica groups.
    ///
    /// Noncommunication dimensions precede communication dimensions; references
    /// determine the order within each group. Missing or empty device IDs use
    /// lexicographical IDs. Invalid and overlapping sub-axes return an error.
    pub fn replica_groups(&self, ctx: &Context, axes: &[AxisRefAttr]) -> Result<Vec<Vec<i64>>> {
        let layout = self.group_layout(ctx, axes)?;
        let original_strides = strides(&layout.sizes)?;
        let transposed_sizes = layout
            .permutation
            .iter()
            .map(|&axis| layout.sizes[axis])
            .collect::<Vec<_>>();
        let transposed_strides = strides(&transposed_sizes)?;
        let groups = usize::try_from(layout.total / layout.group_size)
            .map_err(|_| Diagnostic::MeshTooLarge)?;
        let group_size =
            usize::try_from(layout.group_size).map_err(|_| Diagnostic::MeshTooLarge)?;
        let ids = self
            .device_ids
            .first()
            .map(|ids| ids.values.as_slice())
            .filter(|ids| !ids.is_empty());
        let mut output = Vec::new();
        output
            .try_reserve_exact(groups)
            .map_err(|_| Diagnostic::MeshTooLarge)?;
        for group_index in 0..groups {
            let mut group = Vec::new();
            group
                .try_reserve_exact(group_size)
                .map_err(|_| Diagnostic::MeshTooLarge)?;
            for member in 0..group_size {
                let linear = u64::try_from(group_index)
                    .ok()
                    .and_then(|index| index.checked_mul(layout.group_size))
                    .and_then(|start| {
                        u64::try_from(member)
                            .ok()
                            .and_then(|offset| start.checked_add(offset))
                    })
                    .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?;
                let mut original = 0_u64;
                for (dimension, &axis) in layout.permutation.iter().enumerate() {
                    let stride = transposed_strides
                        .get(dimension)
                        .copied()
                        .filter(|&value| value > 0)
                        .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?;
                    let size = transposed_sizes
                        .get(dimension)
                        .copied()
                        .filter(|&value| value > 0)
                        .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?;
                    let original_stride = original_strides
                        .get(axis)
                        .copied()
                        .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?;
                    let coordinate = (linear / stride) % size;
                    let term = checked_product(coordinate, original_stride)?;
                    original = original
                        .checked_add(term)
                        .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?;
                }
                let device = match ids {
                    Some(ids) => {
                        let device_index =
                            usize::try_from(original).map_err(|_| Diagnostic::MeshTooLarge)?;
                        *ids.get(device_index)
                            .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?
                    }
                    None => i64::try_from(original).map_err(|_| Diagnostic::MeshTooLarge)?,
                };
                group.push(device);
            }
            output.push(group);
        }
        Ok(output)
    }

    fn group_layout(&self, ctx: &Context, references: &[AxisRefAttr]) -> Result<GroupLayout> {
        self.verify(ctx)?;
        let total = self.axes.iter().try_fold(1_u64, |total, axis| {
            checked_product(total, axis.size as u64)
        })?;
        if let Some(ids) = self
            .device_ids
            .first()
            .map(|ids| ids.values.as_slice())
            .filter(|ids| !ids.is_empty())
            && u64::try_from(ids.len()).ok() != Some(total)
        {
            return Err(Diagnostic::InvalidMesh("device ID count does not match mesh size").into());
        }
        if let Some(ids) = self.device_ids.first().map(|ids| &ids.values) {
            let mut seen = std::collections::HashSet::new();
            if ids.iter().any(|&id| id < 0 || !seen.insert(id)) {
                return Err(
                    Diagnostic::InvalidMesh("device IDs must be nonnegative and unique").into(),
                );
            }
        }
        for reference in references {
            reference
                .verify(ctx)
                .map_err(|_| invalid_axis(reference, "invalid axis reference"))?;
            let Some(axis) = self.axes.iter().find(|axis| axis.name == reference.name) else {
                return Err(Diagnostic::UnknownMeshAxis(reference.name.as_str().to_owned()).into());
            };
            if let Some(sub) = reference.sub_axis_info.first() {
                let end = checked_product(sub.pre_size as u64, sub.size as u64)?;
                if !(axis.size as u64).is_multiple_of(end) {
                    return Err(invalid_axis(
                        reference,
                        "sub-axis pre_size * size must divide the full axis size",
                    )
                    .into());
                }
            }
        }
        let mut sizes = Vec::new();
        let mut segments = Vec::new();
        for axis in &self.axes {
            let full_size = axis.size as u64;
            let mut boundaries = vec![1, full_size];
            for reference in references
                .iter()
                .filter(|reference| reference.name == axis.name)
            {
                if let Some(sub) = reference.sub_axis_info.first() {
                    boundaries.push(sub.pre_size as u64);
                    boundaries.push(checked_product(sub.pre_size as u64, sub.size as u64)?);
                }
            }
            boundaries.sort_unstable();
            boundaries.dedup();
            if full_size == 1 {
                segments.push((axis.name.as_str(), 1, 1, sizes.len()));
                sizes.push(1);
            } else {
                for pair in boundaries.windows(2) {
                    if !pair[1].is_multiple_of(pair[0]) {
                        return Err(Diagnostic::InvalidMesh(
                            "sub-axis boundaries do not form an integral factorization",
                        )
                        .into());
                    }
                    let size = pair[1] / pair[0];
                    segments.push((axis.name.as_str(), pair[0], size, sizes.len()));
                    sizes.push(size);
                }
            }
        }
        let mut grouped = Vec::new();
        for reference in references {
            let (pre, size) = match reference.sub_axis_info.first() {
                Some(sub) => (sub.pre_size as u64, sub.size as u64),
                None => {
                    let Some(axis) = self.axes.iter().find(|axis| axis.name == reference.name)
                    else {
                        return Err(Diagnostic::UnknownMeshAxis(
                            reference.name.as_str().to_owned(),
                        )
                        .into());
                    };
                    (1, axis.size as u64)
                }
            };
            let Some(&(_, _, _, index)) = segments.iter().find(|&&(name, start, length, _)| {
                name == reference.name.as_str() && start == pre && length == size
            }) else {
                return Err(
                    invalid_axis(reference, "invalid or overlapping communication axis").into(),
                );
            };
            if grouped.contains(&index) {
                return Err(
                    invalid_axis(reference, "duplicate or overlapping communication axis").into(),
                );
            }
            grouped.push(index);
        }
        let group_size = grouped.iter().try_fold(1_u64, |size, &axis| {
            let axis_size = sizes
                .get(axis)
                .copied()
                .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())?;
            checked_product(size, axis_size)
        })?;
        let permutation = (0..sizes.len())
            .filter(|axis| !grouped.contains(axis))
            .chain(grouped.iter().copied())
            .collect();
        Ok(GroupLayout {
            sizes,
            permutation,
            group_size,
            total,
        })
    }
}

fn invalid_axis(reference: &AxisRefAttr, reason: &'static str) -> Diagnostic {
    Diagnostic::InvalidMeshAxisReference {
        axis: reference.name.as_str().to_owned(),
        reason,
    }
}

fn checked_product(lhs: u64, rhs: u64) -> Result<u64> {
    lhs.checked_mul(rhs)
        .filter(|&size| size <= i64::MAX as u64)
        .ok_or_else(|| Diagnostic::MeshTooLarge.into_error())
}

fn strides(sizes: &[u64]) -> Result<Vec<u64>> {
    let mut product = 1;
    let mut output = vec![1; sizes.len()];
    for (size, stride) in sizes.iter().zip(&mut output).rev() {
        *stride = product;
        product = checked_product(product, *size)?;
    }
    Ok(output)
}
