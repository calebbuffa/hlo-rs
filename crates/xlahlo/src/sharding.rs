//! Native tile-assignment and named-mesh sharding attributes.
use crate::Diagnostic;
use crate::{attributes::Metadata, shape::Shape, types::TokenType};
use pliron::{
    common_traits::Verify, context::Context, derive::pliron_attr, result::Result,
    r#type::TypeHandle,
};
use std::collections::BTreeSet;

mod canonical;
mod named;
use canonical::Kind;
pub use named::{
    AxisRef, DimensionSharding, IotaTransform, Mesh, MeshAxis, NamedSharding, ReductionOp, SubAxis,
};
pub(crate) use named::{iota_value, product, validate_iota};

#[pliron_attr(name = "xlahlo.sharding_type", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Replication, maximal, tuple, tiled, manual, unknown, or unreduced assignment.
pub enum ShardingType {
    Replicated,
    Maximal,
    Tuple,
    Other,
    Manual,
    Unknown,
    Unreduced,
}

#[pliron_attr(name = "xlahlo.shard_group", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Constraint identifying values that must shard together or share a sharding pattern.
pub enum ShardGroup {
    As(i64),
    Like(i64),
}

#[pliron_attr(
    name = "xlahlo.sharding",
    format = "`<` $kind `,` `[` vec($tile_shape, CharSpace(`,`)) `]` `,` `[` vec($tile_assignment_dimensions, CharSpace(`,`)) `]` `,` `[` vec($tile_assignment_devices, CharSpace(`,`)) `]` `,` `[` vec($tuple_shardings, CharSpace(`,`)) `]` `,` $replicate_on_last_tile_dim `,` $metadata `,` `[` vec($last_tile_dims, CharSpace(`,`)) `]` `,` `[` vec($iota_reshape_dims, CharSpace(`,`)) `]` `,` `[` vec($iota_transpose_perm, CharSpace(`,`)) `]` `,` `[` vec($shard_group, CharSpace(`,`)) `]` `,` `[` vec($named_sharding, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Native tuple, tile-assignment, or named-mesh sharding annotation.
pub struct Sharding {
    pub kind: ShardingType,
    pub tile_shape: Vec<TypeHandle>,
    pub tile_assignment_dimensions: Vec<i64>,
    pub tile_assignment_devices: Vec<i64>,
    pub tuple_shardings: Vec<Sharding>,
    pub replicate_on_last_tile_dim: bool,
    pub metadata: Metadata,
    pub last_tile_dims: Vec<ShardingType>,
    pub iota_reshape_dims: Vec<i64>,
    pub iota_transpose_perm: Vec<i64>,
    pub shard_group: Vec<ShardGroup>,
    pub named_sharding: Vec<NamedSharding>,
}

impl Default for Sharding {
    fn default() -> Self {
        Self {
            kind: ShardingType::Replicated,
            tile_shape: Vec::new(),
            tile_assignment_dimensions: Vec::new(),
            tile_assignment_devices: Vec::new(),
            tuple_shardings: Vec::new(),
            replicate_on_last_tile_dim: false,
            metadata: Metadata::default(),
            last_tile_dims: Vec::new(),
            iota_reshape_dims: Vec::new(),
            iota_transpose_perm: Vec::new(),
            shard_group: Vec::new(),
            named_sharding: Vec::new(),
        }
    }
}

impl Sharding {
    /// Construct fully replicated sharding.
    pub fn replicated() -> Self {
        Self::default()
    }
    /// Construct a maximal assignment; contextual validation checks the device range.
    pub fn maximal(device: i64) -> Self {
        Self {
            kind: ShardingType::Maximal,
            tile_assignment_devices: vec![device],
            ..Self::default()
        }
    }
    /// Take flattened tuple-leaf sharding assignments.
    pub fn tuple(elements: Vec<Self>) -> Self {
        Self {
            kind: ShardingType::Tuple,
            tuple_shardings: elements,
            ..Self::default()
        }
    }
    /// Expand a non-tuple annotation to the tuple shape's flattened leaves.
    pub fn normalize_tuple(&self, ctx: &Context, shape: TypeHandle) -> Result<Self> {
        self.verify(ctx)?;
        if !Shape::is_tuple(ctx, shape) || self.kind == ShardingType::Tuple {
            return Ok(self.clone());
        }
        Ok(Self::tuple(vec![
            self.clone();
            Shape::leaf_count(ctx, shape).max(1)
        ]))
    }
    /// Validate the annotation against a shape and optional configured device count.
    pub fn validate(
        &self,
        ctx: &Context,
        shape: TypeHandle,
        num_devices: Option<i64>,
    ) -> Result<()> {
        self.verify(ctx)?;
        if num_devices.is_some_and(|n| n <= 0) {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "device count must be positive".into(),
            }));
        }
        if shape.deref(ctx).is::<TokenType>() {
            return Ok(());
        }
        if Shape::is_tuple(ctx, shape) {
            let mut leaves = Vec::new();
            let mut pending = vec![shape];
            while let Some(shape) = pending.pop() {
                if Shape::is_tuple(ctx, shape) {
                    pending.extend(Shape::children(ctx, shape).iter().rev().copied());
                } else {
                    leaves.push(shape);
                }
            }
            if self.kind == ShardingType::Tuple && self.named_sharding.is_empty() {
                if self.tuple_shardings.len() != leaves.len()
                    && !(leaves.is_empty() && self.tuple_shardings.len() == 1)
                {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "sharding",
                        message: "tuple sharding leaf count differs from shape".into(),
                    }));
                }
                for (sharding, leaf) in self.tuple_shardings.iter().zip(leaves) {
                    sharding.validate_leaf(ctx, leaf, num_devices)?;
                }
            } else {
                for leaf in leaves {
                    self.validate_leaf(ctx, leaf, num_devices)?;
                }
            }
            return Ok(());
        }
        self.validate_leaf(ctx, shape, num_devices)
    }
    fn validate_leaf(&self, ctx: &Context, shape: TypeHandle, devices: Option<i64>) -> Result<()> {
        let in_range = |device: i64| -> Result<()> {
            if device < 0 || devices.is_some_and(|count| device >= count) {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "sharding",
                    message: "device assignment out of range".into(),
                }));
            }
            Ok(())
        };
        if let Some(named) = self.named_sharding.first() {
            return named.validate_shape(ctx, shape, devices);
        }
        match self.classify()? {
            Kind::Leaf(_) => Ok(()),
            Kind::Single(device) => in_range(device),
            Kind::Tuple => Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "sharding",
                message: "tuple sharding applied to non-tuple leaf".into(),
            })),
            Kind::Tiled(tiling) => {
                if Shape::rank(ctx, Shape::array_view(ctx, shape)?)? != tiling.rank {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "sharding",
                        message: "tile assignment data rank differs from shape".into(),
                    }));
                }
                if devices.is_some_and(|n| n != tiling.count) {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "sharding",
                        message: "tile assignment does not cover configured devices".into(),
                    }));
                }
                let mut seen = BTreeSet::new();
                for &device in &self.tile_assignment_devices {
                    in_range(device)?;
                    if !seen.insert(device) {
                        return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                            attribute: "sharding",
                            message: "duplicate device assignment".into(),
                        }));
                    }
                }
                Ok(())
            }
        }
    }
}

impl Verify for Sharding {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let mut pending = vec![(self, 0)];
        while let Some((sharding, depth)) = pending.pop() {
            if depth >= 100 {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "sharding",
                    message: "sharding nesting limit".into(),
                }));
            }
            if sharding.named_sharding.len() > 1
                || sharding.shard_group.len() > 1
                || sharding.tile_shape.len() > 1
            {
                return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "sharding",
                    message: "multiple singleton sharding fields".into(),
                }));
            }
            if let Some(named) = sharding.named_sharding.first() {
                named.verify(ctx)?;
                continue;
            }
            if sharding.kind == ShardingType::Tuple {
                if sharding
                    .metadata
                    .0
                    .iter()
                    .any(|m| m.op_type != "sdy::reduction_op")
                {
                    return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                        attribute: "sharding",
                        message: "tuple sharding cannot have metadata".into(),
                    }));
                }
                pending.extend(
                    sharding
                        .tuple_shardings
                        .iter()
                        .map(|child| (child, depth + 1)),
                );
            }
            sharding.classify()?;
        }
        Ok(())
    }
}
