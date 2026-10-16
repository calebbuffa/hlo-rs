use crate::{
    AxisRef, Error, Mesh, Result,
    proto::xla_data::{MeshAxesReplicaGroupListProto, ReplicaGroup},
};

/// Compact collective groups formed by an ordered span of mesh axes/sub-axes.
#[derive(Clone, Debug)]
pub struct MeshAxesReplicaGroupList {
    wire: MeshAxesReplicaGroupListProto,
    mesh: Mesh,
    size: i64,
    reshape: Vec<i64>,
    permutation: Vec<i32>,
}

impl MeshAxesReplicaGroupList {
    pub fn from_proto(wire: MeshAxesReplicaGroupListProto) -> Result<Self> {
        let mesh = Mesh::from_proto(wire.mesh.as_ref().cloned().unwrap_or_default())?;
        let axes = wire
            .axes
            .iter()
            .cloned()
            .map(AxisRef::from_proto)
            .collect::<Result<Vec<_>>>()?;
        let mut size = 1_i64;
        for (index, axis) in axes.iter().enumerate() {
            size = size
                .checked_mul(axis.size(&mesh)?)
                .ok_or_else(|| Error::new("overflow", "mesh group size overflow"))?;
            if axes[..index]
                .iter()
                .any(|other| !axis.can_coexist_without_overlap(other))
            {
                return Err(Error::new(
                    "collective-group",
                    "mesh group axes overlap or cannot coexist",
                ));
            }
            if index > 0 && axes[index - 1].can_merge(axis) {
                return Err(Error::new(
                    "collective-group",
                    "adjacent mesh group sub-axes must be merged",
                ));
            }
        }
        let mut reshape = Vec::new();
        let mut grouped = vec![0usize; axes.len()];
        for (dimension, full) in mesh.as_proto().axes.iter().enumerate() {
            let mut refs = axes
                .iter()
                .enumerate()
                .filter(|(_, axis)| axis.as_proto().mesh_axis_index == dimension as i64)
                .collect::<Vec<_>>();
            refs.sort_by_key(|(_, axis)| {
                axis.as_proto()
                    .sub_axis_info
                    .as_ref()
                    .map_or(1, |sub| sub.pre_size)
            });
            let mut prefix = 1;
            for (index, axis) in refs {
                let (pre, width) = axis
                    .as_proto()
                    .sub_axis_info
                    .as_ref()
                    .map_or((1, full.size), |sub| (sub.pre_size, sub.size));
                if prefix != pre {
                    reshape.push(pre / prefix);
                }
                grouped[index] = reshape.len();
                reshape.push(width);
                prefix = pre
                    .checked_mul(width)
                    .ok_or_else(|| Error::new("overflow", "mesh group sub-axis size overflow"))?;
            }
            if prefix != full.size {
                reshape.push(full.size / prefix);
            }
        }
        let permutation = (0..reshape.len())
            .filter(|axis| !grouped.contains(axis))
            .chain(grouped.iter().copied())
            .map(|axis| {
                i32::try_from(axis)
                    .map_err(|_| Error::new("overflow", "mesh group rank exceeds i32"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            wire,
            mesh,
            size,
            reshape,
            permutation,
        })
    }

    pub fn as_proto(&self) -> &MeshAxesReplicaGroupListProto {
        &self.wire
    }

    pub fn into_proto(self) -> MeshAxesReplicaGroupListProto {
        self.wire
    }

    pub fn num_replica_groups(&self) -> i64 {
        self.mesh.device_count() / self.size
    }

    pub fn num_devices_per_group(&self) -> i64 {
        self.size
    }

    pub fn device_at(&self, group: i64, member: i64) -> Result<i64> {
        if group < 0 || group >= self.num_replica_groups() || member < 0 || member >= self.size {
            return Err(Error::new(
                "collective-group",
                "mesh group/member index out of range",
            ));
        }
        let linear = group
            .checked_mul(self.size)
            .and_then(|index| index.checked_add(member))
            .ok_or_else(|| Error::new("overflow", "mesh group device offset overflow"))?;
        let index = if self.reshape.is_empty() {
            linear
        } else {
            crate::sharding::iota_value(&self.reshape, &self.permutation, linear)
        };
        self.mesh.device_at(index)
    }

    /// Materialize groups only when the total number of device entries fits the limit.
    pub fn replica_groups(&self, max_entries: usize) -> Result<Vec<ReplicaGroup>> {
        if usize::try_from(self.mesh.device_count()).map_or(true, |count| count > max_entries) {
            return Err(Error::new(
                "resource-limit",
                "mesh groups exceed materialization limit",
            ));
        }
        (0..self.num_replica_groups())
            .map(|group| {
                Ok(ReplicaGroup {
                    replica_ids: (0..self.size)
                        .map(|member| self.device_at(group, member))
                        .collect::<Result<_>>()?,
                    ..Default::default()
                })
            })
            .collect()
    }
}
