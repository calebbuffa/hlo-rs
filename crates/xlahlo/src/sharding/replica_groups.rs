//! Compact replica-group assignments derived from iota transforms or named mesh axes.
use crate::Diagnostic;

use crate::{
    attributes::{ReplicaGroup, ReplicaGroups},
    sharding::{AxisRef, Mesh, iota_value, product, validate_iota},
};
use pliron::{common_traits::Verify, context::Context, derive::pliron_attr, result::Result};

#[pliron_attr(
    name = "xlahlo.iota_replica_group_list",
    format = "`<` $num_replica_groups `,` $num_devices_per_group `,` `[` vec($iota_reshape_dims, CharSpace(`,`)) `]` `,` `[` vec($iota_transpose_perm, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// A compact rectangular group assignment using reshape and transpose of device IDs.
pub struct IotaReplicaGroupList {
    /// Positive number of replica groups.
    pub num_replica_groups: i64,
    /// Positive number of participating devices in each group.
    pub num_devices_per_group: i64,
    /// Positive reshape dimensions whose product is the total device count.
    pub iota_reshape_dims: Vec<i64>,
    /// Permutation of the reshape axes used to assign consecutive device IDs.
    pub iota_transpose_perm: Vec<i64>,
}
impl Verify for IotaReplicaGroupList {
    fn verify(&self, _: &Context) -> Result<()> {
        let count = product(&[self.num_replica_groups, self.num_devices_per_group])?;
        validate_iota(&self.iota_reshape_dims, &self.iota_transpose_perm, count)
    }
}
impl IotaReplicaGroupList {
    /// Resolve one device without materializing all groups.
    pub fn device_at(&self, group: i64, member: i64) -> Result<i64> {
        let count = product(&[self.num_replica_groups, self.num_devices_per_group])?;
        validate_iota(&self.iota_reshape_dims, &self.iota_transpose_perm, count)?;
        if group < 0
            || group >= self.num_replica_groups
            || member < 0
            || member >= self.num_devices_per_group
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "replica_groups",
                message: "group/member out of range".into(),
            }));
        }
        Ok(iota_value(
            &self.iota_reshape_dims,
            &self.iota_transpose_perm,
            group * self.num_devices_per_group + member,
        ))
    }
}

#[pliron_attr(
    name = "xlahlo.mesh_axes_replica_group_list",
    format = "`<` $mesh `,` `[` vec($axes, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Replica groups varying along selected nonoverlapping named-mesh axes.
pub struct MeshAxesReplicaGroupList {
    /// Validated mesh providing device identities.
    pub mesh: Mesh,
    /// Nonoverlapping mesh-axis factors along which members of a group vary.
    pub axes: Vec<AxisRef>,
}

impl Verify for MeshAxesReplicaGroupList {
    fn verify(&self, _: &Context) -> Result<()> {
        self.geometry().map(|_| ())
    }
}
impl MeshAxesReplicaGroupList {
    /// Construct and validate a compact mesh-axis grouping.
    pub fn new(mesh: Mesh, axes: Vec<AxisRef>) -> Result<Self> {
        let value = Self { mesh, axes };
        value.geometry()?;
        Ok(value)
    }
    fn geometry(&self) -> Result<(i64, Vec<i64>, Vec<i64>)> {
        self.mesh.validate()?;
        let mut size = 1i64;
        for (index, axis) in self.axes.iter().enumerate() {
            size = size.checked_mul(axis.size(&self.mesh)?).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "replica_groups",
                    message: "mesh group size overflow".into(),
                })
            })?;
            for other in &self.axes[..index] {
                if !axis.can_coexist_without_overlap(other)? {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "replica_groups",
                        message: "mesh group axes overlap or cannot coexist".into(),
                    }));
                }
            }
            if index > 0 && self.axes[index - 1].can_merge(axis)? {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "replica_groups",
                    message: "adjacent group sub-axes must be merged".into(),
                }));
            }
        }
        let mut reshape = Vec::new();
        let mut grouped = vec![0usize; self.axes.len()];
        for (dimension, full) in self.mesh.axes.iter().enumerate() {
            let mut refs: Vec<_> = self
                .axes
                .iter()
                .enumerate()
                .filter(|(_, axis)| axis.mesh_axis_index == dimension as i64)
                .collect();
            refs.sort_by_key(|(_, axis)| axis.sub_axis_info.first().map_or(1, |sub| sub.pre_size));
            let mut prefix = 1;
            for (index, axis) in refs {
                let (pre, width) = axis
                    .sub_axis_info
                    .first()
                    .map_or((1, full.size), |sub| (sub.pre_size, sub.size));
                if prefix != pre {
                    reshape.push(pre / prefix);
                }
                grouped[index] = reshape.len();
                reshape.push(width);
                prefix = pre.checked_mul(width).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "replica_groups",
                        message: "mesh group sub-axis size overflow".into(),
                    })
                })?;
            }
            if prefix != full.size {
                reshape.push(full.size / prefix);
            }
        }
        let permutation = (0..reshape.len())
            .filter(|axis| !grouped.contains(axis))
            .chain(grouped.iter().copied())
            .map(|axis| axis as i64)
            .collect();
        Ok((size, reshape, permutation))
    }
    /// Return the product of the grouped axis sizes.
    pub fn num_devices_per_group(&self) -> Result<i64> {
        Ok(self.geometry()?.0)
    }
    /// Return the number of groups covering the mesh.
    pub fn num_replica_groups(&self) -> Result<i64> {
        Ok(self.mesh.device_count()? / self.geometry()?.0)
    }
    /// Resolve one device without materializing all groups.
    pub fn device_at(&self, group: i64, member: i64) -> Result<i64> {
        let (size, reshape, permutation) = self.geometry()?;
        if group < 0 || group >= self.mesh.device_count()? / size || member < 0 || member >= size {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "replica_groups",
                message: "mesh group/member out of range".into(),
            }));
        }
        let linear = group
            .checked_mul(size)
            .and_then(|i| i.checked_add(member))
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "replica_groups",
                    message: "mesh group offset overflow".into(),
                })
            })?;
        self.mesh.device_at(if reshape.is_empty() {
            linear
        } else {
            iota_value(&reshape, &permutation, linear)
        })
    }
    /// Materialize groups only when total entries do not exceed the caller's limit.
    ///
    /// Mesh validation and group geometry are computed once, not per device.
    pub fn replica_groups(&self, max_entries: usize) -> Result<ReplicaGroups> {
        let (size, reshape, permutation) = self.geometry()?;
        let count = self.mesh.device_count()?;
        if usize::try_from(count).map_or(true, |count| count > max_entries) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "replica_groups",
                message: "mesh groups exceed materialization limit".into(),
            }));
        }
        Ok(ReplicaGroups(
            (0..count / size)
                .map(|group| {
                    ReplicaGroup(
                        (0..size)
                            .map(|member| {
                                let linear = group * size + member;
                                let index = if reshape.is_empty() {
                                    linear
                                } else {
                                    iota_value(&reshape, &permutation, linear)
                                };
                                self.mesh.device_at_validated(index)
                            })
                            .collect(),
                    )
                })
                .collect(),
        ))
    }
}
