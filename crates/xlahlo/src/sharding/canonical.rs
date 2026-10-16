use super::{Kind, Sharding, named};
use crate::{
    Error, Result,
    proto::xla_data::{
        OpSharding,
        op_sharding::{ShardGroupType, Type},
    },
};

#[derive(Clone, Debug)]
pub(super) struct Tiling {
    pub rank: usize,
    pub count: i64,
    dimensions: Vec<i64>,
    types: Vec<Type>,
    partial: bool,
    axes: Vec<usize>,
}

impl Tiling {
    pub(super) fn classify(wire: &OpSharding, types: &[Type], rank: usize, count: i64) -> Kind {
        let source = &wire.tile_assignment_dimensions;
        let all_unit = source[..rank].iter().all(|&size| size == 1);
        if wire.replicate_on_last_tile_dim && (source.len() == 1 || all_unit) {
            return Kind::Leaf(Type::REPLICATED);
        }
        if all_unit
            && types.first().is_some_and(|first| {
                matches!(first, Type::REPLICATED | Type::MANUAL | Type::UNREDUCED)
                    && types.iter().all(|ty| ty == first)
            })
        {
            return Kind::Leaf(types[0]);
        }
        let mut dimensions = source[..rank].to_vec();
        let mut axes = (0..rank).collect::<Vec<_>>();
        let mut normalized = Vec::new();
        if wire.replicate_on_last_tile_dim {
            if source[rank] > 1 {
                dimensions.push(source[rank]);
                axes.push(rank);
                normalized.push(Type::REPLICATED);
            }
        } else {
            for ty in [
                Type::MAXIMAL,
                Type::TUPLE,
                Type::OTHER,
                Type::MANUAL,
                Type::UNREDUCED,
                Type::REPLICATED,
                Type::UNKNOWN,
            ] {
                let group = (rank..source.len())
                    .filter(|&axis| types[axis - rank] == ty && source[axis] > 1)
                    .collect::<Vec<_>>();
                if !group.is_empty() {
                    dimensions.push(group.iter().map(|&axis| source[axis]).product());
                    axes.extend(group);
                    normalized.push(ty);
                }
            }
        }
        if normalized.len() == 1 && dimensions.last() == Some(&count) {
            if normalized[0] == Type::REPLICATED {
                return Kind::Leaf(Type::REPLICATED);
            }
            if normalized[0] == Type::MANUAL {
                return Kind::Leaf(Type::MANUAL);
            }
        }
        let partial = normalized == [Type::REPLICATED];
        Kind::Tiled(Self {
            rank,
            count,
            dimensions,
            types: normalized,
            partial,
            axes,
        })
    }

    fn value(&self, wire: &OpSharding, mut index: i64) -> i64 {
        let source = &wire.tile_assignment_dimensions;
        let mut coordinates = vec![0; source.len()];
        for &axis in self.axes.iter().rev() {
            coordinates[axis] = index % source[axis];
            index /= source[axis];
        }
        let linear = source
            .iter()
            .zip(coordinates)
            .fold(0, |value, (&size, coordinate)| value * size + coordinate);
        if wire.iota_reshape_dims.is_empty() {
            wire.tile_assignment_devices[linear as usize]
        } else {
            named::iota_value(&wire.iota_reshape_dims, &wire.iota_transpose_perm, linear)
        }
    }

    fn equal(&self, wire: &OpSharding, other: &Self, other_wire: &OpSharding) -> Result<bool> {
        if self.dimensions != other.dimensions
            || self.types != other.types
            || self.partial != other.partial
        {
            return Ok(false);
        }
        if self.axes == other.axes
            && wire.tile_assignment_dimensions == other_wire.tile_assignment_dimensions
        {
            if !wire.iota_reshape_dims.is_empty()
                && !other_wire.iota_reshape_dims.is_empty()
                && named::iota_key(
                    &wire.iota_reshape_dims,
                    &wire.iota_transpose_perm,
                    self.count,
                ) == named::iota_key(
                    &other_wire.iota_reshape_dims,
                    &other_wire.iota_transpose_perm,
                    self.count,
                )
            {
                return Ok(true);
            }
            if wire.iota_reshape_dims.is_empty()
                && other_wire.iota_reshape_dims.is_empty()
                && wire.tile_assignment_devices == other_wire.tile_assignment_devices
            {
                return Ok(true);
            }
        }
        let group = if self.types.last() == Some(&Type::REPLICATED) {
            *self.dimensions.last().expect("replication dimension")
        } else {
            1
        };
        if group == 1 {
            return Ok((0..self.count)
                .all(|index| self.value(wire, index) == other.value(other_wire, index)));
        }
        let mut left = Vec::new();
        let mut right = Vec::new();
        let size = usize::try_from(group)
            .map_err(|_| Error::new("resource-limit", "replication group exceeds address space"))?;
        left.try_reserve_exact(size).map_err(|_| {
            Error::new(
                "resource-limit",
                "cannot allocate sharding replication group",
            )
        })?;
        right.try_reserve_exact(size).map_err(|_| {
            Error::new(
                "resource-limit",
                "cannot allocate sharding replication group",
            )
        })?;
        for start in (0..self.count).step_by(size) {
            left.clear();
            right.clear();
            for index in start..start + group {
                left.push(self.value(wire, index));
                right.push(other.value(other_wire, index));
            }
            left.sort_unstable();
            right.sort_unstable();
            if left != right {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn reduction(wire: &OpSharding) -> i32 {
    let mut reduction = 0;
    for metadata in &wire.metadata {
        if metadata.op_type == "sdy::reduction_op" {
            match metadata.op_name.as_str() {
                "MAX" => reduction = 1,
                "MIN" => reduction = 2,
                _ => (),
            }
        }
    }
    reduction
}

fn group(wire: &OpSharding) -> Option<(i64, bool)> {
    wire.is_shard_group.then_some((
        wire.shard_group_id,
        wire.shard_group_type == ShardGroupType::AS.into(),
    ))
}

impl Sharding {
    /// Logical equality after HLO constructor normalization. Metadata and unknown
    /// protobuf fields do not participate. Allocation failures are explicit.
    pub fn equal_semantics(&self, other: &Self) -> Result<bool> {
        match (&self.kind, &other.kind) {
            (Kind::Named(left), Kind::Named(right)) => return Ok(left.equal_semantics(right)),
            (Kind::Named(left), _) => {
                return Self::from_proto(left.to_legacy_proto()?)?.equal_semantics(other);
            }
            (_, Kind::Named(right)) => {
                return self.equal_semantics(&Self::from_proto(right.to_legacy_proto()?)?);
            }
            _ => (),
        }
        if reduction(&self.wire) != reduction(&other.wire)
            || group(&self.wire) != group(&other.wire)
        {
            return Ok(false);
        }
        match (&self.kind, &other.kind) {
            (Kind::Leaf(left), Kind::Leaf(right)) => Ok(left == right),
            (Kind::Single(left), Kind::Single(right)) => Ok(left == right),
            (Kind::Tuple(left), Kind::Tuple(right)) => {
                if left.len() != right.len() {
                    return Ok(false);
                }
                for (left, right) in left.iter().zip(right) {
                    if !left.equal_semantics(right)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (Kind::Tiled(left), Kind::Tiled(right)) => left.equal(&self.wire, right, &other.wire),
            _ => Ok(false),
        }
    }
}
