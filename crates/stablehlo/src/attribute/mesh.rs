//! Replica-group derivation from `stablehlo/dialect/ReplicaGroupUtils.cpp`.

use super::stablehlo::{AttributeError, AxisRef, Mesh, StableHloAttribute};

struct GroupLayout {
    sizes: Vec<u64>,
    permutation: Vec<usize>,
    group_size: u64,
    total: u64,
}

impl Mesh {
    /// Number of devices in each group selected by full or split mesh axes.
    ///
    /// Validates references and mesh geometry without expanding device groups.
    pub fn replica_group_size(&self, axes: &[AxisRef]) -> Result<u64, AttributeError> {
        Ok(self.group_layout(axes)?.group_size)
    }

    /// Expands mesh communication axes to ordered, explicit replica groups.
    ///
    /// Noncommunication dimensions precede communication dimensions; references
    /// determine the order within each group. Missing or empty device IDs use
    /// lexicographical IDs. Invalid and overlapping sub-axes return an error.
    pub fn replica_groups(&self, axes: &[AxisRef]) -> Result<Vec<Vec<i64>>, AttributeError> {
        let layout = self.group_layout(axes)?;
        let original_strides = strides(&layout.sizes)?;
        let transposed_sizes = layout
            .permutation
            .iter()
            .map(|&axis| layout.sizes[axis])
            .collect::<Vec<_>>();
        let transposed_strides = strides(&transposed_sizes)?;
        let groups = usize::try_from(layout.total / layout.group_size)
            .map_err(|_| AttributeError::MeshTooLarge)?;
        let group_size =
            usize::try_from(layout.group_size).map_err(|_| AttributeError::MeshTooLarge)?;
        let ids = self.device_ids.as_deref().filter(|ids| !ids.is_empty());
        let mut output = Vec::new();
        output
            .try_reserve_exact(groups)
            .map_err(|_| AttributeError::MeshTooLarge)?;
        for group_index in 0..groups {
            let mut group = Vec::new();
            group
                .try_reserve_exact(group_size)
                .map_err(|_| AttributeError::MeshTooLarge)?;
            for member in 0..group_size {
                let linear = group_index as u64 * layout.group_size + member as u64;
                let original = layout
                    .permutation
                    .iter()
                    .enumerate()
                    .map(|(dimension, &axis)| {
                        let coordinate =
                            (linear / transposed_strides[dimension]) % transposed_sizes[dimension];
                        coordinate * original_strides[axis]
                    })
                    .sum::<u64>();
                group.push(match ids {
                    Some(ids) => ids[original as usize],
                    None => original as i64,
                });
            }
            output.push(group);
        }
        Ok(output)
    }

    fn group_layout(&self, references: &[AxisRef]) -> Result<GroupLayout, AttributeError> {
        StableHloAttribute::Mesh(self.clone()).verify()?;
        let total = self.axes.iter().try_fold(1_u64, |total, axis| {
            checked_product(total, axis.size as u64)
        })?;
        if let Some(ids) = self.device_ids.as_deref().filter(|ids| !ids.is_empty())
            && u64::try_from(ids.len()).ok() != Some(total)
        {
            return Err(AttributeError::InvalidMesh(
                "device ID count does not match mesh size",
            ));
        }
        if let Some(ids) = &self.device_ids {
            let mut seen = std::collections::HashSet::new();
            if ids.iter().any(|&id| id < 0 || !seen.insert(id)) {
                return Err(AttributeError::InvalidMesh(
                    "device IDs must be nonnegative and unique",
                ));
            }
        }
        for reference in references {
            StableHloAttribute::AxisRef(reference.clone()).verify()?;
            let Some(axis) = self.axes.iter().find(|axis| axis.name == reference.name) else {
                return Err(AttributeError::UnknownMeshAxis(reference.name.clone()));
            };
            if let Some(sub) = &reference.sub_axis_info {
                let end = checked_product(sub.pre_size as u64, sub.size as u64)?;
                if !(axis.size as u64).is_multiple_of(end) {
                    return Err(invalid_axis(
                        reference,
                        "sub-axis pre_size * size must divide the full axis size",
                    ));
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
                if let Some(sub) = &reference.sub_axis_info {
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
                        return Err(AttributeError::InvalidMesh(
                            "sub-axis boundaries do not form an integral factorization",
                        ));
                    }
                    let size = pair[1] / pair[0];
                    segments.push((axis.name.as_str(), pair[0], size, sizes.len()));
                    sizes.push(size);
                }
            }
        }
        let mut grouped = Vec::new();
        for reference in references {
            let (pre, size) = match &reference.sub_axis_info {
                Some(sub) => (sub.pre_size as u64, sub.size as u64),
                None => (
                    1,
                    self.axes
                        .iter()
                        .find(|axis| axis.name == reference.name)
                        .expect("checked axis reference")
                        .size as u64,
                ),
            };
            let Some(&(_, _, _, index)) = segments.iter().find(|&&(name, start, length, _)| {
                name == reference.name && start == pre && length == size
            }) else {
                return Err(invalid_axis(
                    reference,
                    "invalid or overlapping communication axis",
                ));
            };
            if grouped.contains(&index) {
                return Err(invalid_axis(
                    reference,
                    "duplicate or overlapping communication axis",
                ));
            }
            grouped.push(index);
        }
        let group_size = grouped
            .iter()
            .try_fold(1_u64, |size, &axis| checked_product(size, sizes[axis]))?;
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

fn invalid_axis(reference: &AxisRef, reason: &'static str) -> AttributeError {
    AttributeError::InvalidMeshAxisReference {
        axis: reference.name.clone(),
        reason,
    }
}

fn checked_product(lhs: u64, rhs: u64) -> Result<u64, AttributeError> {
    lhs.checked_mul(rhs)
        .filter(|&size| size <= i64::MAX as u64)
        .ok_or(AttributeError::MeshTooLarge)
}

fn strides(sizes: &[u64]) -> Result<Vec<u64>, AttributeError> {
    let mut product = 1;
    let mut output = vec![1; sizes.len()];
    for (size, stride) in sizes.iter().zip(&mut output).rev() {
        *stride = product;
        product = checked_product(product, *size)?;
    }
    Ok(output)
}
