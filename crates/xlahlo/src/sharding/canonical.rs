use super::{Sharding, ShardingType as T, named};
use crate::Diagnostic;
use pliron::result::Result;

pub(super) enum Kind {
    Leaf(T),
    Single(i64),
    Tuple,
    Tiled(Tiling),
}
pub(super) struct Tiling {
    pub rank: usize,
    pub count: i64,
    dimensions: Vec<i64>,
    types: Vec<T>,
    axes: Vec<usize>,
}

impl Sharding {
    pub(super) fn classify(&self) -> Result<Kind> {
        match self.kind {
            T::Replicated | T::Manual | T::Unknown | T::Unreduced => {
                return Ok(Kind::Leaf(self.kind));
            }
            T::Tuple => return Ok(Kind::Tuple),
            T::Maximal => {
                if self.tile_assignment_devices.len() != 1 {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "sharding",
                        message: "maximal sharding requires one device".into(),
                    }));
                }
                return Ok(Kind::Single(self.tile_assignment_devices[0]));
            }
            T::Other => (),
        }
        let iota = !self.iota_reshape_dims.is_empty();
        if iota {
            if !self.tile_assignment_devices.is_empty()
                || self.iota_reshape_dims.len() != self.iota_transpose_perm.len()
            {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "sharding",
                    message:
                        "iota requires exclusive compact assignment and matching permutation rank"
                            .into(),
                }));
            }
        } else if self.tile_assignment_devices.is_empty() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "tiled sharding requires devices".into(),
            }));
        }
        if iota && self.iota_reshape_dims.iter().all(|&n| n == 1)
            || !iota && self.tile_assignment_devices.len() == 1
        {
            return Ok(Kind::Leaf(T::Replicated));
        }
        if self.tile_assignment_dimensions.is_empty() {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "missing tile dimensions".into(),
            }));
        }
        let count = named::product(&self.tile_assignment_dimensions)?;
        let expected = if iota {
            named::validate_iota(&self.iota_reshape_dims, &self.iota_transpose_perm, count)?;
            count
        } else {
            self.tile_assignment_devices.len() as i64
        };
        if expected != count {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "tile assignment cardinality mismatch".into(),
            }));
        }
        if !self.last_tile_dims.is_empty() && self.replicate_on_last_tile_dim {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "subgroups and legacy partial replication are exclusive".into(),
            }));
        }
        let subgroups = if self.replicate_on_last_tile_dim {
            1
        } else {
            self.last_tile_dims.len()
        };
        let rank = self
            .tile_assignment_dimensions
            .len()
            .checked_sub(subgroups)
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "sharding",
                    message: "more subgroups than tile dimensions".into(),
                })
            })?;
        let source = &self.tile_assignment_dimensions;
        let all_unit = source[..rank].iter().all(|&n| n == 1);
        if self.replicate_on_last_tile_dim && (source.len() == 1 || all_unit) {
            return Ok(Kind::Leaf(T::Replicated));
        }
        if all_unit
            && self.last_tile_dims.first().is_some_and(|ty| {
                matches!(ty, T::Replicated | T::Manual | T::Unreduced)
                    && self.last_tile_dims.iter().all(|other| other == ty)
            })
        {
            return Ok(Kind::Leaf(self.last_tile_dims[0]));
        }
        let mut dimensions = source[..rank].to_vec();
        let mut axes = (0..rank).collect::<Vec<_>>();
        let mut types = Vec::new();
        if self.replicate_on_last_tile_dim {
            if source[rank] > 1 {
                dimensions.push(source[rank]);
                axes.push(rank);
                types.push(T::Replicated);
            }
        } else {
            for ty in [
                T::Maximal,
                T::Tuple,
                T::Other,
                T::Manual,
                T::Unreduced,
                T::Replicated,
                T::Unknown,
            ] {
                let group: Vec<_> = (rank..source.len())
                    .filter(|&axis| self.last_tile_dims[axis - rank] == ty && source[axis] > 1)
                    .collect();
                if !group.is_empty() {
                    dimensions.push(group.iter().map(|&axis| source[axis]).product());
                    axes.extend(group);
                    types.push(ty);
                }
            }
        }
        if types.len() == 1
            && dimensions.last() == Some(&count)
            && matches!(types[0], T::Replicated | T::Manual)
        {
            return Ok(Kind::Leaf(types[0]));
        }
        Ok(Kind::Tiled(Tiling {
            rank,
            count,
            dimensions,
            types,
            axes,
        }))
    }
    /// Compare normalized assignments, subgroup roles, and reduction markers.
    pub fn equal_semantics(&self, other: &Self) -> Result<bool> {
        self.equal_at_depth(other, 0)
    }
    fn equal_at_depth(&self, other: &Self, depth: usize) -> Result<bool> {
        if depth >= 100 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "sharding nesting limit".into(),
            }));
        }
        match (self.named_sharding.first(), other.named_sharding.first()) {
            (Some(a), Some(b)) => return a.equal_semantics(b),
            (Some(a), None) => return a.to_tile_assignment()?.equal_at_depth(other, depth + 1),
            (None, Some(b)) => return self.equal_at_depth(&b.to_tile_assignment()?, depth + 1),
            _ => (),
        }
        let reduction = |s: &Self| {
            s.metadata
                .0
                .iter()
                .filter(|m| m.op_type == "sdy::reduction_op")
                .filter_map(|m| match m.op_name.as_str() {
                    "MAX" => Some(1),
                    "MIN" => Some(2),
                    _ => None,
                })
                .next_back()
                .unwrap_or(0)
        };
        if reduction(self) != reduction(other) || self.shard_group != other.shard_group {
            return Ok(false);
        }
        match (self.classify()?, other.classify()?) {
            (Kind::Leaf(a), Kind::Leaf(b)) => Ok(a == b),
            (Kind::Single(a), Kind::Single(b)) => Ok(a == b),
            (Kind::Tuple, Kind::Tuple) => {
                if self.tuple_shardings.len() != other.tuple_shardings.len() {
                    return Ok(false);
                }
                for (a, b) in self.tuple_shardings.iter().zip(&other.tuple_shardings) {
                    if !a.equal_at_depth(b, depth + 1)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (Kind::Tiled(a), Kind::Tiled(b)) => a.equal(self, &b, other),
            _ => Ok(false),
        }
    }
    /// Compare domain constraints, allowing uniform tuple assignments to collapse.
    pub fn equal_domain_semantics(&self, other: &Self) -> Result<bool> {
        fn single(s: &Sharding) -> Result<Option<&Sharding>> {
            if s.kind != T::Tuple {
                return Ok(Some(s));
            }
            let Some(first) = s.tuple_shardings.first() else {
                return Ok(None);
            };
            for child in &s.tuple_shardings[1..] {
                if !first.equal_semantics(child)? {
                    return Ok(None);
                }
            }
            Ok(Some(first))
        }
        match (single(self)?, single(other)?) {
            (Some(a), Some(b)) => a.equal_semantics(b),
            _ => self.equal_semantics(other),
        }
    }
}

impl Tiling {
    fn value(&self, s: &Sharding, mut index: i64) -> i64 {
        let source = &s.tile_assignment_dimensions;
        let mut coordinates = vec![0; source.len()];
        for &axis in self.axes.iter().rev() {
            coordinates[axis] = index % source[axis];
            index /= source[axis];
        }
        let linear = source
            .iter()
            .zip(coordinates)
            .fold(0, |value, (&size, coordinate)| value * size + coordinate);
        if s.iota_reshape_dims.is_empty() {
            s.tile_assignment_devices[linear as usize]
        } else {
            named::iota_value(&s.iota_reshape_dims, &s.iota_transpose_perm, linear)
        }
    }
    fn equal(&self, a: &Sharding, other: &Self, b: &Sharding) -> Result<bool> {
        if self.dimensions != other.dimensions || self.types != other.types {
            return Ok(false);
        }
        if self.axes == other.axes && a.tile_assignment_dimensions == b.tile_assignment_dimensions {
            if !a.iota_reshape_dims.is_empty()
                && !b.iota_reshape_dims.is_empty()
                && named::iota_key(&a.iota_reshape_dims, &a.iota_transpose_perm, self.count)
                    == named::iota_key(&b.iota_reshape_dims, &b.iota_transpose_perm, self.count)
            {
                return Ok(true);
            }
            if a.iota_reshape_dims.is_empty()
                && b.iota_reshape_dims.is_empty()
                && a.tile_assignment_devices == b.tile_assignment_devices
            {
                return Ok(true);
            }
        }
        let group = if self.types.last() == Some(&T::Replicated) {
            *self.dimensions.last().ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "sharding",
                    message: "missing replication dimension".into(),
                })
            })?
        } else {
            1
        };
        if group == 1 {
            return Ok((0..self.count).all(|i| self.value(a, i) == other.value(b, i)));
        }
        let size = usize::try_from(group).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "replication group exceeds address space".into(),
            })
        })?;
        let (mut left, mut right) = (Vec::new(), Vec::new());
        left.try_reserve_exact(size).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "cannot allocate replication group".into(),
            })
        })?;
        right.try_reserve_exact(size).map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "cannot allocate replication group".into(),
            })
        })?;
        for start in (0..self.count).step_by(size) {
            left.clear();
            right.clear();
            for index in start..start + group {
                left.push(self.value(a, index));
                right.push(other.value(b, index));
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
