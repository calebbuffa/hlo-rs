use std::collections::{BTreeMap, BTreeSet};

use super::{error, product};
use crate::{
    Error, Result,
    proto::xla_data::{AxisRefProto, MeshProto, NamedShardingProto},
};

#[derive(Clone, Debug)]
pub struct Mesh {
    wire: MeshProto,
    count: i64,
}

pub(super) fn validate_iota(dimensions: &[i64], permutation: &[i64], count: i64) -> Result<()> {
    if dimensions.is_empty()
        || dimensions.len() != permutation.len()
        || product(dimensions)? != count
    {
        return Err(error("invalid iota transform shape"));
    }
    let mut seen = BTreeSet::new();
    for &axis in permutation {
        if axis < 0 || axis as usize >= dimensions.len() || !seen.insert(axis) {
            return Err(error("invalid iota transform permutation"));
        }
    }
    Ok(())
}

pub(crate) fn iota_value<P: Copy + Into<i64>>(
    dimensions: &[i64],
    permutation: &[P],
    mut index: i64,
) -> i64 {
    let mut coordinates = vec![0; dimensions.len()];
    for &axis in permutation.iter().rev() {
        let axis = axis.into() as usize;
        coordinates[axis] = index % dimensions[axis];
        index /= dimensions[axis];
    }
    dimensions
        .iter()
        .zip(coordinates)
        .fold(0, |value, (&size, coordinate)| value * size + coordinate)
}

pub(super) fn iota_key<P: Copy + Into<i64>>(
    dimensions: &[i64],
    permutation: &[P],
    count: i64,
) -> BTreeMap<i64, i128> {
    let mut strides = vec![1_i64; dimensions.len()];
    for axis in (1..dimensions.len()).rev() {
        strides[axis - 1] = strides[axis] * dimensions[axis];
    }
    // Mixed-radix digits are floor(i / stride) - size * floor(i / (stride * size)).
    // Combining their coefficients makes equivalent compact reshapes comparable
    // without enumerating a potentially enormous mesh.
    let mut coefficients = BTreeMap::<i64, i128>::new();
    let mut stride = 1;
    for &axis in permutation.iter().rev() {
        let axis = axis.into() as usize;
        if stride < count {
            *coefficients.entry(stride).or_default() += i128::from(strides[axis]);
        }
        stride *= dimensions[axis];
        if stride < count {
            *coefficients.entry(stride).or_default() -=
                i128::from(strides[axis]) * i128::from(dimensions[axis]);
        }
    }
    coefficients.retain(|_, coefficient| *coefficient != 0);
    coefficients
}

impl Mesh {
    pub fn from_proto(wire: MeshProto) -> Result<Self> {
        let count = if wire.axes.is_empty() {
            if wire.device_ids.len() > 1 {
                return Err(error("single-device mesh has multiple devices"));
            }
            wire.device_ids.len() as i64
        } else {
            let mut names = BTreeSet::new();
            for axis in &wire.axes {
                if !names.insert(&axis.name) {
                    return Err(error("mesh axis names must be unique"));
                }
            }
            let count = product(&wire.axes.iter().map(|axis| axis.size).collect::<Vec<_>>())?;
            if wire.device_ids.is_empty() {
                if let Some(transform) = wire.iota_transform.as_ref() {
                    validate_iota(&transform.reshape_dims, &transform.transpose_perm, count)?;
                }
            } else {
                if wire.iota_transform.is_some() || wire.device_ids.len() as i64 != count {
                    return Err(error(
                        "mesh assignment cardinality or exclusive representation is invalid",
                    ));
                }
                let mut devices = BTreeSet::new();
                let mut identity = true;
                for (index, &device) in wire.device_ids.iter().enumerate() {
                    if device < 0 || device >= count || !devices.insert(device) {
                        return Err(error("mesh devices must permute the device iota"));
                    }
                    identity &= device == index as i64;
                }
                if identity {
                    return Err(error(
                        "explicit mesh assignment must not be an identity iota",
                    ));
                }
            }
            count
        };
        Ok(Self { wire, count })
    }

    pub fn as_proto(&self) -> &MeshProto {
        &self.wire
    }
    pub fn into_proto(self) -> MeshProto {
        self.wire
    }
    pub fn device_count(&self) -> i64 {
        self.count
    }

    /// Device geometry and ordering, independent of mesh axis names and wire form.
    pub fn device_assignment_equal(&self, other: &Self) -> bool {
        if self.count != other.count
            || self.wire.axes.iter().map(|axis| axis.size).ne(other
                .wire
                .axes
                .iter()
                .map(|axis| axis.size))
        {
            return false;
        }
        if self.wire.device_ids.is_empty() && other.wire.device_ids.is_empty() {
            let key = |mesh: &Self| {
                if let Some(transform) = mesh
                    .wire
                    .iota_transform
                    .as_ref()
                    .filter(|_| !mesh.wire.axes.is_empty())
                {
                    iota_key(
                        &transform.reshape_dims,
                        &transform.transpose_perm,
                        mesh.count,
                    )
                } else if mesh.count > 1 {
                    BTreeMap::from([(1, 1)])
                } else {
                    BTreeMap::new()
                }
            };
            return key(self) == key(other);
        }
        (0..self.count).all(|index| self.device_at(index) == other.device_at(index))
    }

    pub fn device_at(&self, index: i64) -> Result<i64> {
        if index < 0 || index >= self.count {
            return Err(error("mesh linear device index is out of range"));
        }
        Ok(if !self.wire.device_ids.is_empty() {
            self.wire.device_ids[index as usize]
        } else if let Some(transform) = self.wire.iota_transform.as_ref() {
            iota_value(&transform.reshape_dims, &transform.transpose_perm, index)
        } else {
            index
        })
    }

    fn covers_all(&self, axes: &[AxisRef]) -> bool {
        let mut next = 0;
        for (index, axis) in self.wire.axes.iter().enumerate() {
            if axes
                .get(next)
                .is_some_and(|axis| axis.wire.mesh_axis_index == index as i64)
            {
                if axes[next].wire.sub_axis_info.is_some() {
                    return false;
                }
                next += 1;
            } else if axis.size != 1 {
                return false;
            }
        }
        next == axes.len()
    }
}

#[derive(Clone, Debug)]
pub struct AxisRef {
    wire: AxisRefProto,
}

impl AxisRef {
    pub fn from_proto(wire: AxisRefProto) -> Result<Self> {
        if wire.mesh_axis_index < 0 {
            return Err(error("negative mesh axis reference"));
        }
        if let Some(sub) = wire.sub_axis_info.as_ref() {
            if sub.pre_size < 1 || sub.size <= 1 {
                return Err(error(
                    "sub-axis pre-size must be positive and size greater than one",
                ));
            }
            sub.pre_size
                .checked_mul(sub.size)
                .ok_or_else(|| Error::new("overflow", "sub-axis size overflow"))?;
        }
        Ok(Self { wire })
    }

    pub fn as_proto(&self) -> &AxisRefProto {
        &self.wire
    }
    pub fn into_proto(self) -> AxisRefProto {
        self.wire
    }

    pub fn validate(&self, mesh: &Mesh) -> Result<()> {
        let axis = mesh
            .wire
            .axes
            .get(self.wire.mesh_axis_index as usize)
            .ok_or_else(|| error("axis reference exceeds mesh rank"))?;
        if let Some(sub) = self.wire.sub_axis_info.as_ref()
            && (axis.size % (sub.pre_size * sub.size) != 0 || sub.size >= axis.size)
        {
            return Err(error("sub-axis must properly divide its full mesh axis"));
        }
        Ok(())
    }

    pub fn size(&self, mesh: &Mesh) -> Result<i64> {
        self.validate(mesh)?;
        Ok(self.wire.sub_axis_info.as_ref().map_or(
            mesh.wire.axes[self.wire.mesh_axis_index as usize].size,
            |sub| sub.size,
        ))
    }

    fn interval(&self) -> Option<(i64, i64)> {
        self.wire
            .sub_axis_info
            .as_ref()
            .map(|sub| (sub.pre_size, sub.pre_size * sub.size))
    }

    pub fn can_coexist_without_overlap(&self, other: &Self) -> bool {
        if self.wire.mesh_axis_index != other.wire.mesh_axis_index {
            return true;
        }
        match (self.interval(), other.interval()) {
            (Some((a, b)), Some((c, d))) => {
                if b > c && d > a {
                    false
                } else {
                    a.max(c) % b.min(d) == 0
                }
            }
            _ => false,
        }
    }

    pub fn can_merge(&self, other: &Self) -> bool {
        self.wire.mesh_axis_index == other.wire.mesh_axis_index
            && matches!((self.interval(),other.interval()),(Some((_,end)),Some((start,_))) if end==start)
    }

    fn order(&self) -> (i64, i64, i64) {
        let (pre, size) = self
            .wire
            .sub_axis_info
            .as_ref()
            .map_or((1, i64::MAX), |sub| (sub.pre_size, sub.size));
        (self.wire.mesh_axis_index, pre, size)
    }
}

#[derive(Clone, Debug)]
pub struct NamedSharding {
    wire: NamedShardingProto,
    mesh: Mesh,
    dimensions: Vec<Vec<AxisRef>>,
    unreduced: Vec<AxisRef>,
    manual: Vec<AxisRef>,
}

impl NamedSharding {
    pub fn from_proto(wire: NamedShardingProto) -> Result<Self> {
        let mesh = Mesh::from_proto(wire.mesh.as_ref().cloned().unwrap_or_default())?;
        let mut seen = BTreeMap::<i64, Vec<AxisRef>>::new();
        let mut verify = |axes: &[AxisRefProto], sorted: bool| -> Result<Vec<AxisRef>> {
            let axes = axes
                .iter()
                .cloned()
                .map(AxisRef::from_proto)
                .collect::<Result<Vec<_>>>()?;
            for (index, axis) in axes.iter().enumerate() {
                axis.validate(&mesh)?;
                if index > 0
                    && (axes[index - 1].can_merge(axis)
                        || (sorted && axes[index - 1].order() > axis.order()))
                {
                    return Err(error(
                        "named sharding axes must be normalized and required lists sorted",
                    ));
                }
                let others = seen.entry(axis.wire.mesh_axis_index).or_default();
                if others
                    .iter()
                    .any(|other| !axis.can_coexist_without_overlap(other))
                {
                    return Err(error("named sharding axes overlap or cannot coexist"));
                }
                others.push(axis.clone());
            }
            Ok(axes)
        };
        let dimensions = wire
            .dim_shardings
            .iter()
            .map(|dimension| verify(&dimension.axes, false))
            .collect::<Result<Vec<_>>>()?;
        verify(&wire.replicated_axes, true)?;
        let unreduced = verify(&wire.unreduced_axes, true)?;
        let manual = verify(&wire.manual_axes, true)?;
        for dimension in &dimensions {
            product(
                &dimension
                    .iter()
                    .map(|axis| axis.size(&mesh))
                    .collect::<Result<Vec<_>>>()?,
            )?;
        }
        Ok(Self {
            wire,
            mesh,
            dimensions,
            unreduced,
            manual,
        })
    }

    pub fn as_proto(&self) -> &NamedShardingProto {
        &self.wire
    }
    pub fn into_proto(self) -> NamedShardingProto {
        self.wire
    }
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub(super) fn to_legacy_proto(&self) -> Result<crate::proto::xla_data::OpSharding> {
        use crate::proto::xla_data::{OpMetadata, OpSharding, op_sharding::Type};
        let mut wire = OpSharding {
            metadata: self.wire.metadata.clone(),
            ..Default::default()
        };
        // The legacy representation transports reduction through metadata.
        wire.metadata
            .retain(|metadata| metadata.op_type != "sdy::reduction_op");
        let reduction = self.wire.reduction_op.enum_value().unwrap_or_default() as i32;
        if reduction != 0 {
            wire.metadata.push(OpMetadata {
                op_type: "sdy::reduction_op".into(),
                op_name: if reduction == 1 { "MAX" } else { "MIN" }.into(),
                ..Default::default()
            });
        }
        if self.is_replicated() {
            return Ok(wire);
        }
        if self.is_single_device() {
            wire.type_ = Type::MAXIMAL.into();
            wire.tile_assignment_devices = vec![self.mesh.device_at(0)?];
            return Ok(wire);
        }
        wire.type_ = Type::OTHER.into();
        let mut positions = BTreeMap::new();
        let mut next = 0_i64;
        let mut append = |axes: &[AxisRef]| -> Result<i64> {
            let mut size = 1_i64;
            for axis in axes {
                positions.insert(axis.order(), next);
                next += 1;
                size = size
                    .checked_mul(axis.size(&self.mesh)?)
                    .ok_or_else(|| Error::new("overflow", "named sharding size overflow"))?;
            }
            Ok(size)
        };
        for dimension in &self.dimensions {
            wire.tile_assignment_dimensions.push(append(dimension)?);
        }
        for (axes, ty) in [
            (&self.manual, Type::MANUAL),
            (&self.unreduced, Type::UNREDUCED),
        ] {
            if !axes.is_empty() {
                wire.tile_assignment_dimensions.push(append(axes)?);
                wire.last_tile_dims.push(ty.into());
            }
        }
        let mut axes = Vec::new();
        for (index, axis) in self.mesh.wire.axes.iter().enumerate() {
            let mut cuts = BTreeSet::from([1, axis.size]);
            for reference in self
                .wire
                .dim_shardings
                .iter()
                .flat_map(|dim| &dim.axes)
                .chain(&self.wire.replicated_axes)
                .chain(&self.wire.manual_axes)
                .chain(&self.wire.unreduced_axes)
            {
                if reference.mesh_axis_index == index as i64
                    && let Some(sub) = reference.sub_axis_info.as_ref()
                {
                    cuts.insert(sub.pre_size);
                    cuts.insert(sub.pre_size * sub.size);
                }
            }
            if cuts.len() <= 2 {
                axes.push(AxisRef::from_proto(AxisRefProto {
                    mesh_axis_index: index as i64,
                    ..Default::default()
                })?);
            } else {
                let cuts = cuts.into_iter().collect::<Vec<_>>();
                for window in cuts.windows(2) {
                    axes.push(AxisRef::from_proto(AxisRefProto {
                        mesh_axis_index: index as i64,
                        sub_axis_info: Some(crate::proto::xla_data::axis_ref_proto::SubAxis {
                            pre_size: window[0],
                            size: window[1] / window[0],
                            ..Default::default()
                        })
                        .into(),
                        ..Default::default()
                    })?);
                }
            }
        }
        let dimensions = axes
            .iter()
            .map(|axis| axis.size(&self.mesh))
            .collect::<Result<Vec<_>>>()?;
        let mut permutation = vec![0; axes.len()];
        let mut replicated_size = 1_i64;
        for (index, axis) in axes.iter().enumerate() {
            let position = if let Some(&position) = positions.get(&axis.order()) {
                position
            } else {
                let position = next;
                next += 1;
                replicated_size = replicated_size
                    .checked_mul(dimensions[index])
                    .ok_or_else(|| Error::new("overflow", "replicated size overflow"))?;
                position
            };
            permutation[position as usize] = index as i64;
        }
        if replicated_size > 1 {
            wire.tile_assignment_dimensions.push(replicated_size);
            wire.last_tile_dims.push(Type::REPLICATED.into());
        }
        validate_iota(&dimensions, &permutation, self.mesh.count)?;
        let simple = self.mesh.wire.device_ids.is_empty()
            && self
                .mesh
                .wire
                .iota_transform
                .as_ref()
                .is_none_or(|transform| {
                    iota_key(
                        &transform.reshape_dims,
                        &transform.transpose_perm,
                        self.mesh.count,
                    ) == BTreeMap::from([(1, 1)])
                });
        if simple {
            wire.iota_reshape_dims = dimensions;
            wire.iota_transpose_perm = permutation
                .into_iter()
                .map(|axis| {
                    i32::try_from(axis)
                        .map_err(|_| Error::new("overflow", "iota permutation exceeds i32"))
                })
                .collect::<Result<_>>()?;
        } else {
            let count = usize::try_from(self.mesh.count)
                .map_err(|_| Error::new("resource-limit", "mesh exceeds address space"))?;
            wire.tile_assignment_devices
                .try_reserve_exact(count)
                .map_err(|_| {
                    Error::new(
                        "resource-limit",
                        "cannot allocate converted mesh assignment",
                    )
                })?;
            for index in 0..self.mesh.count {
                wire.tile_assignment_devices
                    .push(
                        self.mesh
                            .device_at(iota_value(&dimensions, &permutation, index))?,
                    );
            }
        }
        Ok(wire)
    }

    /// Pinned named-sharding equality excludes metadata, explicit replication
    /// axes and mesh names, and normalizes unrecognized reductions to SUM.
    pub fn equal_semantics(&self, other: &Self) -> bool {
        if self.is_replicated() && other.is_replicated() {
            return true;
        }
        let axes_equal = |left: &[AxisRef], right: &[AxisRef]| {
            left.iter()
                .map(AxisRef::order)
                .eq(right.iter().map(AxisRef::order))
        };
        let reduction =
            |wire: &NamedShardingProto| wire.reduction_op.enum_value().unwrap_or_default();
        self.mesh.device_assignment_equal(&other.mesh)
            && self.dimensions.len() == other.dimensions.len()
            && self
                .dimensions
                .iter()
                .zip(&other.dimensions)
                .enumerate()
                .all(|(index, (left, right))| {
                    axes_equal(left, right)
                        && self.wire.dim_shardings[index].is_closed
                            == other.wire.dim_shardings[index].is_closed
                })
            && axes_equal(&self.unreduced, &other.unreduced)
            && axes_equal(&self.manual, &other.manual)
            && reduction(&self.wire) == reduction(&other.wire)
    }
    pub fn is_single_device(&self) -> bool {
        self.mesh.wire.axes.is_empty() && self.mesh.count == 1
    }
    pub fn is_replicated(&self) -> bool {
        !self.is_single_device()
            && self.unreduced.is_empty()
            && self.manual.is_empty()
            && self
                .dimensions
                .iter()
                .flatten()
                .all(|axis| axis.size(&self.mesh).is_ok_and(|size| size == 1))
    }
    pub fn is_manual(&self) -> bool {
        !self.manual.is_empty() && self.mesh.covers_all(&self.manual)
    }
    pub fn is_unreduced(&self) -> bool {
        !self.unreduced.is_empty() && self.mesh.covers_all(&self.unreduced)
    }

    pub(super) fn validate_shape(&self, shape: &crate::Shape, devices: Option<i64>) -> Result<()> {
        if self.is_replicated() || self.is_manual() || self.is_unreduced() {
            return Ok(());
        }
        if self.is_single_device() {
            let device = self.mesh.device_at(0)?;
            if device < 0 || devices.is_some_and(|count| device >= count) {
                return Err(error("single-device named sharding is out of range"));
            }
            return Ok(());
        }
        if self.mesh.count == 1 {
            return Err(error("tiled named sharding has only one device"));
        }
        if shape.array_view()?.rank()? != self.dimensions.len() {
            return Err(error("named sharding rank differs from shape"));
        }
        if devices.is_some_and(|count| count != self.mesh.count) {
            return Err(error(
                "named sharding device count differs from configuration",
            ));
        }
        Ok(())
    }
}
