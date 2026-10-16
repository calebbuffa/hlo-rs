use std::collections::BTreeSet;

use crate::{
    Error, PrimitiveType, Result, Shape,
    proto::xla_data::{OpSharding, op_sharding::Type},
};

mod canonical;
mod named;
pub(crate) use named::iota_value;
pub use named::{AxisRef, Mesh, NamedSharding};

/// Validated legacy and named HLO sharding import. Wire fields are retained separately
/// from constructor-normalized semantics, including unknown fields.
#[derive(Clone, Debug)]
pub struct Sharding {
    wire: OpSharding,
    kind: Kind,
}

#[derive(Clone, Debug)]
enum Kind {
    Named(Box<NamedSharding>),
    Leaf(Type),
    Single(i64),
    Tuple(Vec<Sharding>),
    Tiled(canonical::Tiling),
}

fn error(message: &str) -> Error {
    Error::new("sharding", message)
}

fn product(dimensions: &[i64]) -> Result<i64> {
    dimensions.iter().try_fold(1_i64, |product, &size| {
        if size <= 0 {
            return Err(error("tile dimensions must be positive"));
        }
        product
            .checked_mul(size)
            .ok_or_else(|| Error::new("overflow", "tile assignment product overflow"))
    })
}

impl Sharding {
    pub fn from_proto(wire: OpSharding) -> Result<Self> {
        Self::parse(wire, 0)
    }

    fn parse(wire: OpSharding, depth: usize) -> Result<Self> {
        enum Task {
            Visit(OpSharding, usize),
            FinishTuple(OpSharding, usize),
        }
        let mut tasks = vec![Task::Visit(wire, depth)];
        let mut parsed = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                Task::FinishTuple(wire, count) => {
                    let start = parsed
                        .len()
                        .checked_sub(count)
                        .ok_or_else(|| error("tuple sharding parse stack underflow"))?;
                    let children = parsed.split_off(start);
                    parsed.push(Self {
                        wire,
                        kind: Kind::Tuple(children),
                    });
                    continue;
                }
                Task::Visit(wire, depth) => {
                    if depth >= 100 {
                        return Err(Error::new("resource-limit", "sharding nesting limit"));
                    }
                    if let Some(named) = wire.named_sharding.as_ref() {
                        parsed.push(Self {
                            kind: Kind::Named(Box::new(NamedSharding::from_proto(named.clone())?)),
                            wire,
                        });
                        continue;
                    }
                    let ty = wire
                        .type_
                        .enum_value()
                        .map_err(|_| error("unknown sharding type"))?;
                    if ty == Type::TUPLE {
                        if wire
                            .metadata
                            .iter()
                            .any(|metadata| metadata.op_type != "sdy::reduction_op")
                        {
                            return Err(error("tuple sharding cannot have metadata"));
                        }
                        tasks.push(Task::FinishTuple(wire.clone(), wire.tuple_shardings.len()));
                        tasks.extend(
                            wire.tuple_shardings
                                .iter()
                                .rev()
                                .cloned()
                                .map(|child| Task::Visit(child, depth + 1)),
                        );
                        continue;
                    }
                    let kind = match ty {
                        Type::REPLICATED | Type::MANUAL | Type::UNKNOWN | Type::UNREDUCED => {
                            Kind::Leaf(ty)
                        }
                        Type::MAXIMAL => {
                            if wire.tile_assignment_devices.len() != 1 {
                                return Err(error("maximal sharding requires exactly one device"));
                            }
                            Kind::Single(wire.tile_assignment_devices[0])
                        }
                        Type::OTHER => {
                            let iota = !wire.iota_reshape_dims.is_empty();
                            if iota {
                                if !wire.tile_assignment_devices.is_empty()
                                    || wire.iota_reshape_dims.len()
                                        != wire.iota_transpose_perm.len()
                                {
                                    return Err(error(
                                        "iota sharding requires exclusive compact assignment and matching permutation rank",
                                    ));
                                }
                            } else if wire.tile_assignment_devices.is_empty() {
                                return Err(error("tiled sharding requires device assignments"));
                            }
                            if (iota && wire.iota_reshape_dims.iter().all(|&size| size == 1))
                                || (!iota && wire.tile_assignment_devices.len() == 1)
                            {
                                Kind::Leaf(Type::REPLICATED)
                            } else {
                                if wire.tile_assignment_dimensions.is_empty() {
                                    return Err(error("tile assignment dimensions are missing"));
                                }
                                let count = product(&wire.tile_assignment_dimensions)?;
                                let expected = if iota {
                                    let mut seen = BTreeSet::new();
                                    for &axis in &wire.iota_transpose_perm {
                                        if axis < 0
                                            || axis as usize >= wire.iota_reshape_dims.len()
                                            || !seen.insert(axis)
                                        {
                                            return Err(error(
                                                "invalid iota transpose permutation",
                                            ));
                                        }
                                    }
                                    product(&wire.iota_reshape_dims)?
                                } else {
                                    i64::try_from(wire.tile_assignment_devices.len())
                                        .map_err(|_| error("device count exceeds i64"))?
                                };
                                if count != expected {
                                    return Err(error(
                                        "tile assignment cardinality differs from its dimensions",
                                    ));
                                }
                                if !wire.last_tile_dims.is_empty()
                                    && wire.replicate_on_last_tile_dim
                                {
                                    return Err(error(
                                        "subgroups and legacy partial replication are exclusive",
                                    ));
                                }
                                let subgroups = wire
                                    .last_tile_dims
                                    .iter()
                                    .map(|ty| {
                                        ty.enum_value().map_err(|_| error("unknown subgroup type"))
                                    })
                                    .collect::<Result<Vec<_>>>()?;
                                let subgroup_count = if wire.replicate_on_last_tile_dim {
                                    1
                                } else {
                                    subgroups.len()
                                };
                                let rank = wire
                                    .tile_assignment_dimensions
                                    .len()
                                    .checked_sub(subgroup_count)
                                    .ok_or_else(|| error("more subgroups than tile dimensions"))?;
                                canonical::Tiling::classify(&wire, &subgroups, rank, count)
                            }
                        }
                        Type::TUPLE => return Err(error("tuple sharding task was not scheduled")),
                    };
                    parsed.push(Self { wire, kind });
                }
            }
        }
        parsed
            .pop()
            .ok_or_else(|| error("sharding parse produced no result"))
    }

    pub fn as_proto(&self) -> &OpSharding {
        &self.wire
    }
    pub fn into_proto(self) -> OpSharding {
        self.wire
    }

    /// Repeat a non-tuple sharding over the flat leaves of a tuple instruction.
    /// Empty (including nested-empty) tuples retain a single unused entry.
    pub fn normalize_tuple(&self, shape: &Shape) -> Result<Self> {
        if !shape.is_tuple() || matches!(self.kind, Kind::Tuple(_)) {
            return Ok(self.clone());
        }
        Self::from_proto(OpSharding {
            type_: Type::TUPLE.into(),
            tuple_shardings: vec![self.wire.clone(); shape.leaf_count().max(1)],
            ..Default::default()
        })
    }

    /// `num_devices` is supplied only for upstream's SPMD device-number policy.
    /// Non-tuple shardings applied to tuples are repeated over leaves, matching
    /// HLO instruction import's `NormalizeTupleSharding`.
    pub fn validate(&self, shape: &Shape, num_devices: Option<i64>) -> Result<()> {
        if num_devices.is_some_and(|n| n <= 0) {
            return Err(error("device count must be positive"));
        }
        if shape.element_type() == PrimitiveType::TOKEN {
            return Ok(());
        }
        if shape.is_tuple() {
            fn leaves(shape: &Shape, output: &mut Vec<Shape>) -> Result<()> {
                if shape.is_tuple() {
                    for index in 0..shape.as_proto().tuple_shapes.len() {
                        leaves(&shape.subshape(&[index])?, output)?;
                    }
                } else {
                    output.push(shape.clone());
                }
                Ok(())
            }
            let mut shapes = Vec::new();
            leaves(shape, &mut shapes)?;
            if let Kind::Tuple(elements) = &self.kind {
                if elements.len() != shapes.len() && !(shapes.is_empty() && elements.len() == 1) {
                    return Err(error("tuple sharding leaf count differs from shape"));
                }
                for (element, shape) in elements.iter().zip(shapes) {
                    element.validate_leaf(&shape, num_devices)?;
                }
                Ok(())
            } else {
                for shape in shapes {
                    self.validate_leaf(&shape, num_devices)?;
                }
                Ok(())
            }
        } else {
            self.validate_leaf(shape, num_devices)
        }
    }

    fn validate_leaf(&self, shape: &Shape, num_devices: Option<i64>) -> Result<()> {
        let in_range = |device| {
            if device < 0 || num_devices.is_some_and(|count| device >= count) {
                Err(error("device assignment is out of range"))
            } else {
                Ok(())
            }
        };
        match &self.kind {
            Kind::Named(named) => named.validate_shape(shape, num_devices),
            Kind::Leaf(_) => Ok(()),
            Kind::Tuple(_) => Err(error("tuple sharding applied to a non-tuple leaf")),
            Kind::Single(device) => in_range(*device),
            Kind::Tiled(tiling) => {
                let canonical::Tiling { rank, count, .. } = tiling;
                let shape = shape.array_view()?;
                if shape.rank()? != *rank {
                    return Err(error("tile assignment data rank differs from shape"));
                }
                if num_devices.is_some_and(|devices| devices != *count) {
                    return Err(error(
                        "tile assignment does not cover the configured device count",
                    ));
                }
                let mut seen = BTreeSet::new();
                for &device in &self.wire.tile_assignment_devices {
                    in_range(device)?;
                    if !seen.insert(device) {
                        return Err(error("device assignment contains duplicates"));
                    }
                }
                Ok(())
            }
        }
    }
}
