use super::{ParseOptions, Parser, metadata, reject_fields, strings};
use crate::{
    Error, Result, Sharding,
    proto::{
        hlo::HloInstructionProto,
        xla_data::{
            AxisRefProto, MeshProto, NamedShardingProto, OpSharding,
            axis_ref_proto::SubAxis,
            mesh_proto::{IotaTransform, MeshAxis},
            named_sharding_proto::{DimensionSharding, ReductionOpProto as ReductionOp},
            op_sharding::{ShardGroupType, Type},
        },
    },
};
use protobuf::EnumFull;
use std::collections::BTreeSet;

pub fn parse(source: &str) -> Result<Sharding> {
    let mut parser = Parser::new(source, ParseOptions::default())?;
    let wire = parser.sharding()?;
    parser.finish()?;
    Sharding::from_proto(wire)
}

impl Parser {
    pub(super) fn sharding(&mut self) -> Result<OpSharding> {
        self.expect("{")?;
        let wire = if self.peek() == "{" || self.peek() == "}" {
            let mut wire = OpSharding {
                type_: Type::TUPLE.into(),
                ..Default::default()
            };
            if !self.take("}") {
                loop {
                    self.expect("{")?;
                    if self.peek() == "{" || self.peek() == "}" {
                        return Err(self.error("tuple shardings must be flat non-tuple leaves"));
                    }
                    wire.tuple_shardings.push(self.single_sharding()?);
                    if self.take("}") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            wire
        } else {
            self.single_sharding()?
        };
        Sharding::from_proto(wire.clone())?;
        Ok(wire)
    }

    fn iota_assignment(&mut self) -> Result<(Vec<i64>, Vec<i64>)> {
        let dimensions = self.integers("[", "]")?;
        if dimensions.is_empty() {
            return Err(self.error("iota reshape dimensions must not be empty"));
        }
        let permutation = if dimensions.len() == 1 {
            vec![0]
        } else {
            self.expect("T")?;
            self.integers("(", ")")?
        };
        Ok((dimensions, permutation))
    }

    fn single_sharding(&mut self) -> Result<OpSharding> {
        if matches!(self.peek(), "mesh" | "single_device" | "maximal_mesh") {
            return Ok(OpSharding {
                named_sharding: Some(self.named_sharding()?).into(),
                ..Default::default()
            });
        }
        let mut wire = OpSharding::default();
        let mut modes = BTreeSet::new();
        let mut shard_as = false;
        while !self.take("}") {
            let name = self.word()?;
            match name.as_str() {
                "replicated" | "maximal" | "manual" | "unknown" | "unreduced" => {
                    modes.insert(name);
                }
                "last_tile_dim_replicate" => wire.replicate_on_last_tile_dim = true,
                "shard_as" | "shard_like" => {
                    shard_as |= name == "shard_as";
                    wire.is_shard_group = true;
                    wire.shard_group_id = self.integer()?;
                }
                "device" => {
                    self.expect("=")?;
                    wire.tile_assignment_devices = vec![self.integer()?];
                }
                "devices" => {
                    self.expect("=")?;
                    wire.tile_assignment_dimensions = self.integers("[", "]")?;
                    if wire.tile_assignment_dimensions.is_empty() {
                        return Err(self.error("tile dimensions must not be empty"));
                    }
                    if self.take("<=") {
                        let (dimensions, permutation) = self.iota_assignment()?;
                        wire.iota_reshape_dims = dimensions;
                        wire.iota_transpose_perm = permutation
                            .into_iter()
                            .map(|axis| {
                                i32::try_from(axis)
                                    .map_err(|_| self.error("iota permutation exceeds i32"))
                            })
                            .collect::<Result<_>>()?;
                    } else {
                        wire.tile_assignment_devices = vec![self.integer()?];
                        while self.take(",") {
                            wire.tile_assignment_devices.push(self.integer()?);
                        }
                    }
                }
                "last_tile_dims" => {
                    self.expect("=")?;
                    self.expect("{")?;
                    if !self.take("}") {
                        loop {
                            let ty = match self.word()?.as_str() {
                                "replicated" => Type::REPLICATED,
                                "maximal" => Type::MAXIMAL,
                                "manual" => Type::MANUAL,
                                "unreduced" => Type::UNREDUCED,
                                _ => return Err(self.error("invalid subgroup type")),
                            };
                            wire.last_tile_dims.push(ty.into());
                            if self.take("}") {
                                break;
                            }
                            self.expect(",")?;
                        }
                    }
                }
                "metadata" => {
                    self.expect("=")?;
                    wire.metadata.extend(self.metadata_list()?);
                }
                _ => return Err(self.error(format!("unknown sharding attribute {name}"))),
            }
        }
        wire.shard_group_type = if shard_as || !wire.is_shard_group {
            ShardGroupType::AS
        } else {
            ShardGroupType::LIKE
        }
        .into();
        wire.type_ = if modes.contains("replicated") {
            Type::REPLICATED
        } else if modes.contains("maximal") {
            Type::MAXIMAL
        } else if modes.contains("manual") {
            Type::MANUAL
        } else if modes.contains("unknown") {
            Type::UNKNOWN
        } else if modes.contains("unreduced") {
            Type::UNREDUCED
        } else {
            Type::OTHER
        }
        .into();
        if wire.type_ != Type::OTHER.into() {
            if wire.type_ == Type::MAXIMAL.into() {
                if wire.tile_assignment_devices.len() != 1 {
                    return Err(self.error("maximal sharding requires exactly one device"));
                }
            } else if !wire.tile_assignment_devices.is_empty() {
                return Err(self.error("non-maximal leaf mode cannot assign devices"));
            }
            wire.tile_assignment_dimensions.clear();
            wire.iota_reshape_dims.clear();
            wire.iota_transpose_perm.clear();
            wire.last_tile_dims.clear();
            wire.replicate_on_last_tile_dim = false;
        } else {
            if wire.tile_assignment_dimensions.is_empty() {
                return Err(self.error("tiled sharding requires dimensions"));
            }
            if wire.iota_reshape_dims.is_empty() && wire.tile_assignment_devices.len() <= 1 {
                return Err(self.error("explicit tiled text requires more than one device"));
            }
        }
        Ok(wire)
    }

    pub(super) fn mesh(&mut self) -> Result<MeshProto> {
        let kind = self.word()?;
        if matches!(kind.as_str(), "single_device" | "maximal_mesh") {
            self.expect("[")?;
            self.expect("device_id")?;
            self.expect("=")?;
            let device = self.integer()?;
            self.expect("]")?;
            return Ok(MeshProto {
                device_ids: vec![device],
                ..Default::default()
            });
        }
        if kind != "mesh" {
            return Err(self.error("expected mesh"));
        }
        self.expect("[")?;
        let mut mesh = MeshProto::default();
        if !self.take("]") {
            loop {
                let name = self.utf8_string()?;
                self.expect("=")?;
                let size = self.integer()?;
                mesh.axes.push(MeshAxis {
                    name,
                    size,
                    ..Default::default()
                });
                if self.take("]") {
                    break;
                }
                self.expect(",")?;
            }
        }
        if !mesh.axes.is_empty()
            && self.peek() == ","
            && self
                .tokens
                .get(self.index + 1)
                .is_some_and(|token| token.value == "device_ids")
        {
            self.expect(",")?;
            self.expect("device_ids")?;
            self.expect("=")?;
            self.expect("(")?;
            if self.peek() == "[" {
                let (reshape_dims, transpose_perm) = self.iota_assignment()?;
                mesh.iota_transform = Some(IotaTransform {
                    reshape_dims,
                    transpose_perm,
                    ..Default::default()
                })
                .into();
                self.expect(")")?;
            } else {
                if !self.take(")") {
                    loop {
                        mesh.device_ids.push(self.integer()?);
                        if self.take(")") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                if mesh.device_ids.is_empty() {
                    return Err(self.error("explicit mesh device list must not be empty"));
                }
            }
        }
        crate::Mesh::from_proto(mesh.clone())?;
        Ok(mesh)
    }

    fn axis_ref(&mut self, mesh: &MeshProto) -> Result<AxisRefProto> {
        let name = self.utf8_string()?;
        let index = mesh
            .axes
            .iter()
            .position(|axis| axis.name == name)
            .map(|index| index as i64)
            .or_else(|| name.parse::<i64>().ok())
            .ok_or_else(|| self.error("unknown mesh axis"))?;
        let mut axis = AxisRefProto {
            mesh_axis_index: index,
            ..Default::default()
        };
        if self.take(":") {
            self.expect("(")?;
            let pre_size = self.integer()?;
            self.expect(")")?;
            let size = self.integer()?;
            axis.sub_axis_info = Some(SubAxis {
                pre_size,
                size,
                ..Default::default()
            })
            .into();
        }
        Ok(axis)
    }

    pub(super) fn axes(
        &mut self,
        mesh: &MeshProto,
        dimension: bool,
    ) -> Result<(Vec<AxisRefProto>, bool)> {
        self.expect("{")?;
        let mut axes = Vec::new();
        let mut closed = true;
        if !self.take("}") {
            loop {
                if dimension && self.take("?") {
                    closed = false;
                } else {
                    axes.push(self.axis_ref(mesh)?);
                }
                if self.take("}") {
                    break;
                }
                if !closed {
                    return Err(self.error("open dimension marker must be last"));
                }
                self.expect(",")?;
            }
        }
        Ok((axes, closed))
    }

    fn reduction(&mut self) -> Result<ReductionOp> {
        Ok(match self.word()?.as_str() {
            "sum" => ReductionOp::SUM,
            "max" => ReductionOp::MAX,
            "min" => ReductionOp::MIN,
            _ => return Err(self.error("expected sum, max, or min reduction")),
        })
    }

    fn named_sharding(&mut self) -> Result<NamedShardingProto> {
        let mesh = self.mesh()?;
        let mut wire = NamedShardingProto {
            mesh: Some(mesh.clone()).into(),
            ..Default::default()
        };
        if mesh.axes.is_empty() && mesh.device_ids.len() == 1 {
            if self.take(",") {
                self.expect("metadata")?;
                self.expect("=")?;
                wire.metadata = self.metadata_list()?;
            }
            self.expect("}")?;
            return Ok(wire);
        }
        self.take(",");
        if self.take("replicated") {
        } else if self.take("manual") {
            wire.manual_axes = (0..mesh.axes.len())
                .map(|index| AxisRefProto {
                    mesh_axis_index: index as i64,
                    ..Default::default()
                })
                .collect();
        } else if self.take("unreduced") {
            if self.take("=") {
                wire.reduction_op = self.reduction()?.into();
            }
            wire.unreduced_axes = if self.peek() == "{" {
                self.axes(&mesh, false)?.0
            } else {
                (0..mesh.axes.len())
                    .map(|index| AxisRefProto {
                        mesh_axis_index: index as i64,
                        ..Default::default()
                    })
                    .collect()
            };
        } else {
            self.expect("[")?;
            if !self.take("]") {
                loop {
                    let (axes, is_closed) = self.axes(&mesh, true)?;
                    wire.dim_shardings.push(DimensionSharding {
                        axes,
                        is_closed,
                        ..Default::default()
                    });
                    if self.take("]") {
                        break;
                    }
                    self.expect(",")?;
                }
            }
            while self.take(",") {
                let name = self.word()?;
                self.expect("=")?;
                match name.as_str() {
                    "replicated" => wire.replicated_axes = self.axes(&mesh, false)?.0,
                    "manual" => wire.manual_axes = self.axes(&mesh, false)?.0,
                    "unreduced" => {
                        if self.peek() != "{" {
                            wire.reduction_op = self.reduction()?.into();
                        }
                        wire.unreduced_axes = self.axes(&mesh, false)?.0;
                    }
                    "metadata" => wire.metadata.extend(self.metadata_list()?),
                    _ => return Err(self.error("unknown named sharding attribute")),
                }
            }
            self.expect("}")?;
            return Ok(wire);
        }
        if self.take(",") {
            self.expect("metadata")?;
            self.expect("=")?;
            wire.metadata = self.metadata_list()?;
        }
        self.expect("}")?;
        Ok(wire)
    }

    pub(super) fn domain(&mut self, instruction: &mut HloInstructionProto) -> Result<()> {
        self.expect("{")?;
        let mut seen = BTreeSet::new();
        while !self.take("}") {
            self.take(",");
            let name = self.word()?;
            if !seen.insert(name.clone()) {
                return Err(self.error("duplicate domain attribute"));
            }
            self.expect("=")?;
            match name.as_str() {
                "kind" => {
                    if self.utf8_string()? != "sharding" {
                        return Err(self.error("unsupported domain kind"));
                    }
                }
                "entry" => instruction.domain_exit_sharding = Some(self.sharding()?).into(),
                "exit" => instruction.domain_entry_sharding = Some(self.sharding()?).into(),
                _ => return Err(self.error("unknown domain attribute")),
            }
        }
        if seen.len() != 3 {
            return Err(self.error("domain requires kind, entry and exit"));
        }
        Ok(())
    }
}

fn integers<P: ToString>(values: &[P]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn iota<P: ToString>(dimensions: &[i64], permutation: &[P]) -> String {
    let mut result = format!("[{}]", integers(dimensions));
    if dimensions.len() > 1 {
        result.push_str(&format!("T({})", integers(permutation)));
    }
    result
}

pub fn print(sharding: &Sharding) -> Result<String> {
    print_wire(sharding.as_proto())
}

fn print_wire(wire: &OpSharding) -> Result<String> {
    Sharding::from_proto(wire.clone())?;
    if wire.named_sharding.is_some() || wire.type_.enum_value() != Ok(Type::TUPLE) {
        return print_wire_non_tuple(wire);
    }
    reject_fields(
        wire,
        &[
            "type",
            "metadata",
            "is_shard_group",
            "shard_group_id",
            "shard_group_type",
            "tuple_shardings",
        ],
    )?;
    if !wire.metadata.is_empty() || wire.is_shard_group {
        return Err(Error::new(
            "unsupported",
            "outer tuple sharding metadata/group cannot be expressed in HLO text",
        ));
    }
    if wire
        .tuple_shardings
        .iter()
        .any(|element| element.named_sharding.is_none() && element.type_ == Type::TUPLE.into())
    {
        return Err(Error::new(
            "unsupported",
            "nested tuple shardings have no upstream text representation",
        ));
    }
    Ok(format!(
        "{{{}}}",
        wire.tuple_shardings
            .iter()
            .map(print_wire_non_tuple)
            .collect::<Result<Vec<_>>>()?
            .join(",")
    ))
}

fn print_wire_non_tuple(wire: &OpSharding) -> Result<String> {
    Sharding::from_proto(wire.clone())?;
    if let Some(named) = wire.named_sharding.as_ref() {
        reject_fields(wire, &["named_sharding"])?;
        return print_named(named);
    }
    let ty = wire
        .type_
        .enum_value()
        .map_err(|_| Error::new("sharding", "unknown sharding type"))?;
    if ty == Type::TUPLE {
        return Err(Error::new(
            "unsupported",
            "nested tuple sharding cannot be printed as a leaf",
        ));
    }
    let mut allowed = vec![
        "type",
        "metadata",
        "is_shard_group",
        "shard_group_id",
        "shard_group_type",
    ];
    match ty {
        Type::TUPLE => allowed.push("tuple_shardings"),
        Type::MAXIMAL => allowed.push("tile_assignment_devices"),
        Type::OTHER => allowed.extend([
            "tile_assignment_dimensions",
            "tile_assignment_devices",
            "iota_reshape_dims",
            "iota_transpose_perm",
            "last_tile_dims",
            "replicate_on_last_tile_dim",
        ]),
        _ => (),
    }
    reject_fields(wire, &allowed)?;
    let mut result = match ty {
        Type::MAXIMAL => format!("maximal device={}", wire.tile_assignment_devices[0]),
        Type::OTHER => {
            let assignment = if wire.iota_reshape_dims.is_empty() {
                if wire.tile_assignment_devices.len() <= 1 {
                    return Err(Error::new(
                        "unsupported",
                        "singleton explicit OTHER wire sharding normalizes to replicated",
                    ));
                }
                integers(&wire.tile_assignment_devices)
            } else {
                format!(
                    "<={}",
                    iota(&wire.iota_reshape_dims, &wire.iota_transpose_perm)
                )
            };
            format!(
                "devices=[{}]{assignment}",
                integers(&wire.tile_assignment_dimensions)
            )
        }
        _ => ty.descriptor().name().to_ascii_lowercase(),
    };
    if wire.replicate_on_last_tile_dim {
        result.push_str(" last_tile_dim_replicate");
    }
    if !wire.last_tile_dims.is_empty() {
        let types = wire
            .last_tile_dims
            .iter()
            .map(|ty| {
                ty.enum_value()
                    .map(|ty| ty.descriptor().name().to_ascii_lowercase())
                    .map_err(|_| Error::new("sharding", "unknown subgroup type"))
            })
            .collect::<Result<Vec<_>>>()?;
        if types
            .iter()
            .any(|ty| matches!(ty.as_str(), "tuple" | "other" | "unknown"))
        {
            return Err(Error::new(
                "unsupported",
                "subgroup type has no upstream text token",
            ));
        }
        result.push_str(&format!(" last_tile_dims={{{}}}", types.join(",")));
    }
    if wire.is_shard_group {
        result.push_str(&format!(
            " {} {}",
            if wire.shard_group_type == ShardGroupType::AS.into() {
                "shard_as"
            } else {
                "shard_like"
            },
            wire.shard_group_id
        ));
    }
    if !wire.metadata.is_empty() {
        result.push_str(&format!(
            " metadata={}",
            metadata::print_list(&wire.metadata)?
        ));
    }
    Ok(format!("{{{result}}}"))
}

pub(super) fn print_axes(axes: &[AxisRefProto], mesh: &MeshProto, closed: bool) -> Result<String> {
    let mut result = Vec::new();
    for axis in axes {
        reject_fields(axis, &["mesh_axis_index", "sub_axis_info"])?;
        let mut text = strings::quote(mesh.axes[axis.mesh_axis_index as usize].name.as_bytes())?;
        if let Some(sub) = axis.sub_axis_info.as_ref() {
            reject_fields(sub, &["pre_size", "size"])?;
            text.push_str(&format!(":({}){}", sub.pre_size, sub.size));
        }
        result.push(text);
    }
    if !closed {
        result.push("?".into());
    }
    Ok(format!("{{{}}}", result.join(",")))
}

pub(super) fn print_mesh(mesh: &MeshProto) -> Result<String> {
    crate::Mesh::from_proto(mesh.clone())?;
    reject_fields(mesh, &["axes", "device_ids", "iota_transform"])?;
    if mesh.axes.is_empty() && mesh.device_ids.len() == 1 {
        if mesh.iota_transform.is_some() {
            return Err(Error::new(
                "unsupported",
                "single-device mesh iota transform has no text representation",
            ));
        }
        return Ok(format!("single_device[device_id={}]", mesh.device_ids[0]));
    }
    let axes = mesh
        .axes
        .iter()
        .map(|axis| {
            reject_fields(axis, &["name", "size"])?;
            Ok(format!(
                "{}={}",
                strings::quote(axis.name.as_bytes())?,
                axis.size
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut text = format!("mesh[{}]", axes.join(","));
    if let Some(transform) = mesh.iota_transform.as_ref() {
        reject_fields(transform, &["reshape_dims", "transpose_perm"])?;
        if mesh.axes.is_empty() {
            return Err(Error::new(
                "unsupported",
                "empty mesh iota transform has no text representation",
            ));
        }
        text.push_str(&format!(
            ", device_ids=({})",
            iota(&transform.reshape_dims, &transform.transpose_perm)
        ));
    } else if !mesh.device_ids.is_empty() {
        text.push_str(&format!(", device_ids=({})", integers(&mesh.device_ids)));
    }
    Ok(text)
}

fn print_named(wire: &NamedShardingProto) -> Result<String> {
    reject_fields(
        wire,
        &[
            "mesh",
            "dim_shardings",
            "replicated_axes",
            "unreduced_axes",
            "manual_axes",
            "reduction_op",
            "metadata",
        ],
    )?;
    let mesh = wire
        .mesh
        .as_ref()
        .ok_or_else(|| Error::new("unsupported", "named sharding missing mesh wire presence"))?;
    reject_fields(mesh, &["axes", "device_ids", "iota_transform"])?;
    let mut result = if mesh.axes.is_empty() && mesh.device_ids.len() == 1 {
        if !wire.dim_shardings.is_empty()
            || !wire.replicated_axes.is_empty()
            || !wire.unreduced_axes.is_empty()
            || !wire.manual_axes.is_empty()
            || wire.reduction_op != ReductionOp::SUM.into()
        {
            return Err(Error::new(
                "unsupported",
                "single-device named sharding fields cannot be preserved in text",
            ));
        }
        print_mesh(mesh)?
    } else {
        let mut text = print_mesh(mesh)?;
        let dimensions = wire
            .dim_shardings
            .iter()
            .map(|dimension| {
                reject_fields(dimension, &["axes", "is_closed"])?;
                print_axes(&dimension.axes, mesh, dimension.is_closed)
            })
            .collect::<Result<Vec<_>>>()?;
        text.push_str(&format!(", [{}]", dimensions.join(",")));
        if !wire.replicated_axes.is_empty() {
            text.push_str(&format!(
                ", replicated={}",
                print_axes(&wire.replicated_axes, mesh, true)?
            ));
        }
        let reduction = wire.reduction_op.enum_value().map_err(|_| {
            Error::new(
                "unsupported",
                "unknown named reduction cannot be retained in text",
            )
        })?;
        if !wire.unreduced_axes.is_empty() || reduction != ReductionOp::SUM {
            text.push_str(&format!(
                ", unreduced={}{}",
                reduction.descriptor().name().to_ascii_lowercase(),
                print_axes(&wire.unreduced_axes, mesh, true)?
            ));
        }
        if !wire.manual_axes.is_empty() {
            text.push_str(&format!(
                ", manual={}",
                print_axes(&wire.manual_axes, mesh, true)?
            ));
        }
        text
    };
    if !wire.metadata.is_empty() {
        result.push_str(&format!(
            ", metadata={}",
            metadata::print_list(&wire.metadata)?
        ));
    }
    Ok(format!("{{{result}}}"))
}

pub(super) fn print_fields(instruction: &HloInstructionProto, result: &mut String) -> Result<()> {
    if let Some(sharding) = instruction.sharding.as_ref() {
        result.push_str(&format!(", sharding={}", print_wire(sharding)?));
    }
    if instruction.opcode == "domain" {
        let entry = instruction
            .domain_exit_sharding
            .as_ref()
            .ok_or_else(|| Error::new("unsupported", "domain text requires entry sharding"))?;
        let exit = instruction
            .domain_entry_sharding
            .as_ref()
            .ok_or_else(|| Error::new("unsupported", "domain text requires exit sharding"))?;
        result.push_str(&format!(
            ", domain={{kind=\"sharding\", entry={}, exit={}}}",
            print_wire(entry)?,
            print_wire(exit)?
        ));
    }
    if let Some(metadata) = instruction.metadata.as_ref() {
        result.push_str(&format!(", metadata={}", metadata::print(metadata)?));
    }
    Ok(())
}
