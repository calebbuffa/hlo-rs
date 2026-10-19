use super::{Sharding, ShardingType};
use crate::Diagnostic;
use crate::{
    attributes::{Metadata, OpMetadata},
    shape::Shape,
};
use pliron::{
    common_traits::Verify, context::Context, derive::pliron_attr, result::Result,
    r#type::TypeHandle,
};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn product(dimensions: &[i64]) -> Result<i64> {
    dimensions.iter().try_fold(1i64, |n, &d| {
        if d <= 0 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "tile dimensions must be positive".into(),
            }));
        }
        n.checked_mul(d).ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "tile assignment product overflow".into(),
            })
        })
    })
}
pub(crate) fn validate_iota(dimensions: &[i64], permutation: &[i64], count: i64) -> Result<()> {
    if dimensions.is_empty()
        || dimensions.len() != permutation.len()
        || product(dimensions)? != count
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "named_sharding",
            message: "invalid iota transform shape".into(),
        }));
    }
    let mut seen = BTreeSet::new();
    if permutation
        .iter()
        .any(|&axis| axis < 0 || axis as usize >= dimensions.len() || !seen.insert(axis))
    {
        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
            attribute: "named_sharding",
            message: "invalid iota permutation".into(),
        }));
    }
    Ok(())
}
pub(crate) fn iota_value(dimensions: &[i64], permutation: &[i64], mut index: i64) -> i64 {
    let mut coordinates = vec![0; dimensions.len()];
    for &axis in permutation.iter().rev() {
        coordinates[axis as usize] = index % dimensions[axis as usize];
        index /= dimensions[axis as usize];
    }
    dimensions
        .iter()
        .zip(coordinates)
        .fold(0, |value, (&size, coordinate)| value * size + coordinate)
}
pub(super) fn iota_key(dimensions: &[i64], permutation: &[i64], count: i64) -> BTreeMap<i64, i128> {
    let mut strides = vec![1i64; dimensions.len()];
    for axis in (1..dimensions.len()).rev() {
        strides[axis - 1] = strides[axis] * dimensions[axis];
    }
    let mut coefficients = BTreeMap::<i64, i128>::new();
    let mut stride = 1;
    for &axis in permutation.iter().rev() {
        let axis = axis as usize;
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

#[pliron_attr(name = "hlo.mesh_axis", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// One named dimension and its positive extent in a device mesh.
pub struct MeshAxis {
    pub name: String,
    pub size: i64,
}
#[pliron_attr(
    name = "hlo.iota_transform",
    format = "`<` `[` vec($reshape_dims, CharSpace(`,`)) `]` `,` `[` vec($transpose_perm, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Compact device-ID assignment formed by reshaping and transposing an iota.
pub struct IotaTransform {
    pub reshape_dims: Vec<i64>,
    pub transpose_perm: Vec<i64>,
}
#[pliron_attr(
    name = "hlo.mesh",
    format = "`<` `[` vec($axes, CharSpace(`,`)) `]` `,` `[` vec($device_ids, CharSpace(`,`)) `]` `,` `[` vec($iota_transform, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Named device mesh with explicit IDs or a compact iota assignment.
pub struct Mesh {
    pub axes: Vec<MeshAxis>,
    pub device_ids: Vec<i64>,
    pub iota_transform: Vec<IotaTransform>,
}

impl Verify for Mesh {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
impl Mesh {
    /// Take mesh components and validate dimensions and device assignment.
    pub fn new(
        axes: Vec<MeshAxis>,
        device_ids: Vec<i64>,
        iota_transform: Vec<IotaTransform>,
    ) -> Result<Self> {
        let mesh = Self {
            axes,
            device_ids,
            iota_transform,
        };
        mesh.validate()?;
        Ok(mesh)
    }
    /// Check axis names, dimensions, and the exclusive device-assignment representation.
    pub fn validate(&self) -> Result<()> {
        if self.iota_transform.len() > 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "mesh has multiple iota transforms".into(),
            }));
        }
        if self.axes.is_empty() {
            if self.device_ids.len() > 1 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "single-device mesh has multiple devices".into(),
                }));
            }
            return Ok(());
        }
        let mut names = BTreeSet::new();
        if self.axes.iter().any(|axis| !names.insert(&axis.name)) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "mesh axis names must be unique".into(),
            }));
        }
        let count = product(&self.axes.iter().map(|axis| axis.size).collect::<Vec<_>>())?;
        if self.device_ids.is_empty() {
            if let Some(transform) = self.iota_transform.first() {
                validate_iota(&transform.reshape_dims, &transform.transpose_perm, count)?;
            }
        } else {
            if !self.iota_transform.is_empty() || self.device_ids.len() as i64 != count {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "invalid mesh assignment cardinality or exclusive representation"
                        .into(),
                }));
            }
            let mut devices = BTreeSet::new();
            if self
                .device_ids
                .iter()
                .any(|&device| device < 0 || device >= count || !devices.insert(device))
            {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "mesh devices must permute iota".into(),
                }));
            }
            if self
                .device_ids
                .iter()
                .enumerate()
                .all(|(i, &device)| device == i as i64)
            {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "explicit mesh assignment must not be identity".into(),
                }));
            }
        }
        Ok(())
    }
    /// Return mesh cardinality, rejecting invalid extents and overflow.
    pub fn device_count(&self) -> Result<i64> {
        if self.axes.is_empty() {
            i64::try_from(self.device_ids.len()).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "mesh cardinality exceeds i64".into(),
                })
            })
        } else {
            self.axes.iter().try_fold(1i64, |count, axis| {
                if axis.size <= 0 {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "mesh axis extent must be positive".into(),
                    }));
                }
                count.checked_mul(axis.size).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "mesh cardinality overflow".into(),
                    })
                })
            })
        }
    }
    /// Resolve a checked linear position without expanding compact assignments.
    pub fn device_at(&self, index: i64) -> Result<i64> {
        self.validate()?;
        if index < 0 || index >= self.device_count()? {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "mesh device index out of range".into(),
            }));
        }
        Ok(if !self.device_ids.is_empty() {
            self.device_ids[index as usize]
        } else if let Some(t) = self.iota_transform.first() {
            iota_value(&t.reshape_dims, &t.transpose_perm, index)
        } else {
            index
        })
    }
    /// Compare device order independently from axis names and reshape representation.
    pub fn device_assignment_equal(&self, other: &Self) -> Result<bool> {
        self.validate()?;
        other.validate()?;
        let count = self.device_count()?;
        if count != other.device_count()?
            || self
                .axes
                .iter()
                .map(|a| a.size)
                .ne(other.axes.iter().map(|a| a.size))
        {
            return Ok(false);
        }
        if self.device_ids.is_empty() && other.device_ids.is_empty() {
            let key = |mesh: &Self| {
                if let Some(t) = mesh
                    .iota_transform
                    .first()
                    .filter(|_| !mesh.axes.is_empty())
                {
                    iota_key(&t.reshape_dims, &t.transpose_perm, count)
                } else if count > 1 {
                    BTreeMap::from([(1, 1)])
                } else {
                    BTreeMap::new()
                }
            };
            return Ok(key(self) == key(other));
        }
        for index in 0..count {
            if self.device_at(index)? != other.device_at(index)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn covers_all(&self, axes: &[AxisRef]) -> bool {
        let mut next = 0;
        for (index, axis) in self.axes.iter().enumerate() {
            if axes
                .get(next)
                .is_some_and(|a| a.mesh_axis_index == index as i64)
            {
                if !axes[next].sub_axis_info.is_empty() {
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

#[pliron_attr(name = "hlo.sub_axis", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// A proper mesh-axis factor starting at a multiplicative prefix.
pub struct SubAxis {
    pub pre_size: i64,
    pub size: i64,
}
#[pliron_attr(
    name = "hlo.axis_ref",
    format = "`<` $mesh_axis_index `,` `[` vec($sub_axis_info, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// A full mesh axis or one proper factor of that axis.
pub struct AxisRef {
    pub mesh_axis_index: i64,
    pub sub_axis_info: Vec<SubAxis>,
}
impl Verify for AxisRef {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate_fields()
    }
}
impl AxisRef {
    /// Construct an axis reference, validating index and optional sub-axis fields.
    pub fn new(mesh_axis_index: i64, sub_axis_info: Vec<SubAxis>) -> Result<Self> {
        let axis = Self {
            mesh_axis_index,
            sub_axis_info,
        };
        axis.validate_fields()?;
        Ok(axis)
    }
    fn validate_fields(&self) -> Result<()> {
        if self.mesh_axis_index < 0 || self.sub_axis_info.len() > 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "invalid mesh axis reference".into(),
            }));
        }
        if let Some(sub) = self.sub_axis_info.first() {
            if sub.pre_size < 1 || sub.size <= 1 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "sub-axis pre-size must be positive and size greater than one".into(),
                }));
            }
            sub.pre_size.checked_mul(sub.size).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "sub-axis size overflow".into(),
                })
            })?;
        }
        Ok(())
    }
    /// Check that the reference exists in the mesh and its factor properly divides the axis.
    pub fn validate(&self, mesh: &Mesh) -> Result<()> {
        self.validate_fields()?;
        let axis = mesh
            .axes
            .get(self.mesh_axis_index as usize)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "axis reference exceeds mesh rank".into(),
                })
            })?;
        if axis.size <= 0 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "referenced mesh axis extent must be positive".into(),
            }));
        }
        if let Some(sub) = self.sub_axis_info.first()
            && (axis.size % (sub.pre_size * sub.size) != 0 || sub.size >= axis.size)
        {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "sub-axis must properly divide full axis".into(),
            }));
        }
        Ok(())
    }
    /// Return the checked axis or sub-axis extent.
    pub fn size(&self, mesh: &Mesh) -> Result<i64> {
        self.validate(mesh)?;
        Ok(self
            .sub_axis_info
            .first()
            .map_or(mesh.axes[self.mesh_axis_index as usize].size, |sub| {
                sub.size
            }))
    }
    fn interval(&self) -> Option<(i64, i64)> {
        self.sub_axis_info
            .first()
            .map(|sub| (sub.pre_size, sub.pre_size * sub.size))
    }
    /// Whether two references can occupy disjoint compatible factors.
    pub fn can_coexist_without_overlap(&self, other: &Self) -> Result<bool> {
        self.validate_fields()?;
        other.validate_fields()?;
        if self.mesh_axis_index != other.mesh_axis_index {
            return Ok(true);
        }
        Ok(match (self.interval(), other.interval()) {
            (Some((a, b)), Some((c, d))) => !(b > c && d > a) && a.max(c) % b.min(d) == 0,
            _ => false,
        })
    }
    /// Whether this sub-axis immediately precedes another on the same mesh axis.
    pub fn can_merge(&self, other: &Self) -> Result<bool> {
        self.validate_fields()?;
        other.validate_fields()?;
        Ok(self.mesh_axis_index == other.mesh_axis_index
            && matches!((self.interval(), other.interval()), (Some((_,end)),Some((start,_))) if end == start))
    }
    fn order(&self) -> (i64, i64, i64) {
        let (pre, size) = self
            .sub_axis_info
            .first()
            .map_or((1, i64::MAX), |sub| (sub.pre_size, sub.size));
        (self.mesh_axis_index, pre, size)
    }
}

#[pliron_attr(
    name = "hlo.dimension_sharding",
    format = "`<` `[` vec($axes, CharSpace(`,`)) `]` `,` $is_closed `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Named mesh axes assigned to one logical dimension and its closedness policy.
pub struct DimensionSharding {
    pub axes: Vec<AxisRef>,
    pub is_closed: bool,
}
#[pliron_attr(name = "hlo.reduction_op", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
/// Reduction semantics associated with unreduced mesh axes.
pub enum ReductionOp {
    #[default]
    Sum,
    Max,
    Min,
}
#[pliron_attr(
    name = "hlo.named_sharding",
    format = "`<` $mesh `,` `[` vec($dim_shardings, CharSpace(`,`)) `]` `,` `[` vec($replicated_axes, CharSpace(`,`)) `]` `,` `[` vec($unreduced_axes, CharSpace(`,`)) `]` `,` `[` vec($manual_axes, CharSpace(`,`)) `]` `,` $reduction_op `,` $metadata `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Sharding expressed as nonoverlapping named mesh axes and subgroup roles.
pub struct NamedSharding {
    pub mesh: Mesh,
    pub dim_shardings: Vec<DimensionSharding>,
    pub replicated_axes: Vec<AxisRef>,
    pub unreduced_axes: Vec<AxisRef>,
    pub manual_axes: Vec<AxisRef>,
    pub reduction_op: ReductionOp,
    pub metadata: Metadata,
}
impl Verify for NamedSharding {
    fn verify(&self, _: &Context) -> Result<()> {
        self.validate()
    }
}
impl NamedSharding {
    /// Validate mesh geometry, normalized axis lists, and nonoverlapping assignments.
    pub fn validate(&self) -> Result<()> {
        self.mesh.validate()?;
        let mut seen = BTreeMap::<i64, Vec<&AxisRef>>::new();
        for (axes, sorted) in self
            .dim_shardings
            .iter()
            .map(|dim| (dim.axes.as_slice(), false))
            .chain([
                (self.replicated_axes.as_slice(), true),
                (self.unreduced_axes.as_slice(), true),
                (self.manual_axes.as_slice(), true),
            ])
        {
            for (index, axis) in axes.iter().enumerate() {
                axis.validate(&self.mesh)?;
                if index > 0
                    && (axes[index - 1].can_merge(axis)?
                        || sorted && axes[index - 1].order() > axis.order())
                {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "named sharding axes must be normalized and required lists sorted"
                            .into(),
                    }));
                }
                let others = seen.entry(axis.mesh_axis_index).or_default();
                for other in others.iter() {
                    if !axis.can_coexist_without_overlap(other)? {
                        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                            attribute: "named_sharding",
                            message: "named sharding axes overlap or cannot coexist".into(),
                        }));
                    }
                }
                others.push(axis);
            }
        }
        for dim in &self.dim_shardings {
            dim.axes.iter().try_fold(1i64, |size, axis| {
                size.checked_mul(axis.size(&self.mesh)?).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "tile assignment product overflow".into(),
                    })
                })
            })?;
        }
        Ok(())
    }
    /// Compare device assignment and effective sharding roles, ignoring ordinary metadata.
    pub fn equal_semantics(&self, other: &Self) -> Result<bool> {
        self.validate()?;
        other.validate()?;
        if self.is_replicated()? && other.is_replicated()? {
            return Ok(true);
        }
        let axes_equal = |a: &[AxisRef], b: &[AxisRef]| {
            a.iter()
                .map(AxisRef::order)
                .eq(b.iter().map(AxisRef::order))
        };
        Ok(self.mesh.device_assignment_equal(&other.mesh)?
            && self.dim_shardings.len() == other.dim_shardings.len()
            && self
                .dim_shardings
                .iter()
                .zip(&other.dim_shardings)
                .all(|(a, b)| axes_equal(&a.axes, &b.axes) && a.is_closed == b.is_closed)
            && axes_equal(&self.unreduced_axes, &other.unreduced_axes)
            && axes_equal(&self.manual_axes, &other.manual_axes)
            && self.reduction_op == other.reduction_op)
    }
    /// Whether this is a maximal assignment to one device rather than a mesh.
    pub fn is_single_device(&self) -> Result<bool> {
        self.validate()?;
        Ok(self.mesh.axes.is_empty() && self.mesh.device_count()? == 1)
    }
    /// Whether no effective mesh axis shards or partially reduces the value.
    pub fn is_replicated(&self) -> Result<bool> {
        self.validate()?;
        if self.is_single_device()?
            || !self.unreduced_axes.is_empty()
            || !self.manual_axes.is_empty()
        {
            return Ok(false);
        }
        for axis in self.dim_shardings.iter().flat_map(|dim| &dim.axes) {
            if axis.size(&self.mesh)? != 1 {
                return Ok(false);
            }
        }
        Ok(true)
    }
    /// Whether manual axes cover the entire mesh.
    pub fn is_manual(&self) -> Result<bool> {
        self.validate()?;
        Ok(!self.manual_axes.is_empty() && self.mesh.covers_all(&self.manual_axes))
    }
    /// Whether unreduced axes cover the entire mesh.
    pub fn is_unreduced(&self) -> Result<bool> {
        self.validate()?;
        Ok(!self.unreduced_axes.is_empty() && self.mesh.covers_all(&self.unreduced_axes))
    }
    /// Check structural validity plus result rank and configured device-count constraints.
    pub fn validate_shape(
        &self,
        ctx: &Context,
        shape: TypeHandle,
        devices: Option<i64>,
    ) -> Result<()> {
        self.validate()?;
        if self.is_replicated()? || self.is_manual()? || self.is_unreduced()? {
            return Ok(());
        }
        if self.is_single_device()? {
            let device = self.mesh.device_at(0)?;
            if device < 0 || devices.is_some_and(|count| device >= count) {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "single-device named sharding out of range".into(),
                }));
            }
            return Ok(());
        }
        let count = self.mesh.device_count()?;
        if count == 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "tiled named sharding has one device".into(),
            }));
        }
        if Shape::rank(ctx, Shape::array_view(ctx, shape)?)? != self.dim_shardings.len() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "named sharding rank differs from shape".into(),
            }));
        }
        if devices.is_some_and(|n| n != count) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "named_sharding",
                message: "named sharding device count differs from configuration".into(),
            }));
        }
        Ok(())
    }
    /// Convert named mesh axes into a native tile assignment with subgroup dimensions.
    pub fn to_tile_assignment(&self) -> Result<Sharding> {
        self.validate()?;
        let mut sharding = Sharding {
            metadata: self.metadata.clone(),
            ..Sharding::default()
        };
        sharding
            .metadata
            .0
            .retain(|m| m.op_type != "sdy::reduction_op");
        if self.reduction_op != ReductionOp::Sum {
            sharding
                .metadata
                .0
                .push(OpMetadata::reduction(self.reduction_op == ReductionOp::Max));
        }
        if self.is_replicated()? {
            return Ok(sharding);
        }
        if self.is_single_device()? {
            sharding.kind = ShardingType::Maximal;
            sharding.tile_assignment_devices = vec![self.mesh.device_at(0)?];
            return Ok(sharding);
        }
        sharding.kind = ShardingType::Other;
        let mut positions = BTreeMap::new();
        let mut next = 0i64;
        let mut append = |axes: &[AxisRef]| -> Result<i64> {
            let mut size = 1i64;
            for axis in axes {
                positions.insert(axis.order(), next);
                next += 1;
                size = size.checked_mul(axis.size(&self.mesh)?).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "named sharding size overflow".into(),
                    })
                })?;
            }
            Ok(size)
        };
        for dim in &self.dim_shardings {
            sharding.tile_assignment_dimensions.push(append(&dim.axes)?);
        }
        for (axes, ty) in [
            (&self.manual_axes, ShardingType::Manual),
            (&self.unreduced_axes, ShardingType::Unreduced),
        ] {
            if !axes.is_empty() {
                sharding.tile_assignment_dimensions.push(append(axes)?);
                sharding.last_tile_dims.push(ty);
            }
        }
        let mut axes = Vec::new();
        for (index, full) in self.mesh.axes.iter().enumerate() {
            let mut cuts = BTreeSet::from([1, full.size]);
            for axis in self
                .dim_shardings
                .iter()
                .flat_map(|dim| &dim.axes)
                .chain(&self.replicated_axes)
                .chain(&self.manual_axes)
                .chain(&self.unreduced_axes)
            {
                if axis.mesh_axis_index == index as i64
                    && let Some(sub) = axis.sub_axis_info.first()
                {
                    cuts.insert(sub.pre_size);
                    cuts.insert(sub.pre_size * sub.size);
                }
            }
            if cuts.len() <= 2 {
                axes.push(AxisRef::new(index as i64, Vec::new())?);
            } else {
                let cuts: Vec<_> = cuts.into_iter().collect();
                for pair in cuts.windows(2) {
                    axes.push(AxisRef::new(
                        index as i64,
                        vec![SubAxis {
                            pre_size: pair[0],
                            size: pair[1] / pair[0],
                        }],
                    )?);
                }
            }
        }
        let dimensions = axes
            .iter()
            .map(|axis| axis.size(&self.mesh))
            .collect::<Result<Vec<_>>>()?;
        let mut permutation = vec![0; axes.len()];
        let mut replicated_size = 1i64;
        for (index, axis) in axes.iter().enumerate() {
            let position = if let Some(&p) = positions.get(&axis.order()) {
                p
            } else {
                let p = next;
                next += 1;
                replicated_size =
                    replicated_size
                        .checked_mul(dimensions[index])
                        .ok_or_else(|| {
                            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                                attribute: "named_sharding",
                                message: "replicated size overflow".into(),
                            })
                        })?;
                p
            };
            let slot = usize::try_from(position)
                .ok()
                .and_then(|position| permutation.get_mut(position))
                .ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "mesh axes do not form a tile-assignment permutation".into(),
                    })
                })?;
            *slot = index as i64;
        }
        if replicated_size > 1 {
            sharding.tile_assignment_dimensions.push(replicated_size);
            sharding.last_tile_dims.push(ShardingType::Replicated);
        }
        let device_count = self.mesh.device_count()?;
        validate_iota(&dimensions, &permutation, device_count)?;
        let simple = self.mesh.device_ids.is_empty()
            && self.mesh.iota_transform.first().is_none_or(|t| {
                iota_key(&t.reshape_dims, &t.transpose_perm, device_count)
                    == BTreeMap::from([(1, 1)])
            });
        if simple {
            sharding.iota_reshape_dims = dimensions;
            sharding.iota_transpose_perm = permutation;
        } else {
            let count = usize::try_from(device_count).map_err(|_| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "named_sharding",
                    message: "mesh exceeds address space".into(),
                })
            })?;
            sharding
                .tile_assignment_devices
                .try_reserve_exact(count)
                .map_err(|_| {
                    pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "named_sharding",
                        message: "cannot allocate converted mesh".into(),
                    })
                })?;
            for index in 0..device_count {
                sharding
                    .tile_assignment_devices
                    .push(
                        self.mesh
                            .device_at(iota_value(&dimensions, &permutation, index))?,
                    );
            }
        }
        Ok(sharding)
    }
}
