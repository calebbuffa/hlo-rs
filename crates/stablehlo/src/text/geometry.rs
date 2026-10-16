use std::collections::BTreeMap;

use crate::{
    Attribute, AxisRef, ConvDimensionNumbers, Mesh, MeshAxis, MeshReference, ReplicaGroupMeshAxes,
    ResultAccuracy, ResultAccuracyMode, StableHloAttribute, SubAxisInfo,
};

use super::{
    TextError, TextErrorKind,
    attributes::{list, quote},
    cursor::Cursor,
};

impl Cursor<'_> {
    pub(super) fn convolution_attribute(&mut self) -> Result<StableHloAttribute, TextError> {
        self.expect("<")?;
        let dimensions = if self.take("raw") {
            let mut dimensions = ConvDimensionNumbers::default();
            self.fields_body(|cursor, key| {
                match key {
                    "input_batch_dimension" => dimensions.input_batch_dimension = cursor.i64()?,
                    "input_feature_dimension" => {
                        dimensions.input_feature_dimension = cursor.i64()?
                    }
                    "input_spatial_dimensions" => {
                        dimensions.input_spatial_dimensions = cursor.int_list()?
                    }
                    "kernel_input_feature_dimension" => {
                        dimensions.kernel_input_feature_dimension = cursor.i64()?
                    }
                    "kernel_output_feature_dimension" => {
                        dimensions.kernel_output_feature_dimension = cursor.i64()?
                    }
                    "kernel_spatial_dimensions" => {
                        dimensions.kernel_spatial_dimensions = cursor.int_list()?
                    }
                    "output_batch_dimension" => dimensions.output_batch_dimension = cursor.i64()?,
                    "output_feature_dimension" => {
                        dimensions.output_feature_dimension = cursor.i64()?
                    }
                    "output_spatial_dimensions" => {
                        dimensions.output_spatial_dimensions = cursor.int_list()?
                    }
                    _ => {
                        return Err(
                            cursor.error(TextErrorKind::Expected("convolution dimension field"))
                        );
                    }
                }
                Ok(())
            })?;
            dimensions
        } else {
            let dimensions = self.convolution_dimension_numbers()?;
            self.expect(">")?;
            dimensions
        };
        Ok(StableHloAttribute::ConvDimensionNumbers(dimensions))
    }

    pub(super) fn convolution_dimension_numbers(
        &mut self,
    ) -> Result<ConvDimensionNumbers, TextError> {
        let (input_batch_dimension, input_feature_dimension, input_spatial_dimensions) =
            self.convolution_dims("b", "f")?;
        self.expect("x")?;
        let (
            kernel_input_feature_dimension,
            kernel_output_feature_dimension,
            kernel_spatial_dimensions,
        ) = self.convolution_dims("i", "o")?;
        self.expect("->")?;
        let (output_batch_dimension, output_feature_dimension, output_spatial_dimensions) =
            self.convolution_dims("b", "f")?;
        Ok(ConvDimensionNumbers {
            input_batch_dimension,
            input_feature_dimension,
            input_spatial_dimensions,
            kernel_input_feature_dimension,
            kernel_output_feature_dimension,
            kernel_spatial_dimensions,
            output_batch_dimension,
            output_feature_dimension,
            output_spatial_dimensions,
        })
    }

    fn convolution_dims(
        &mut self,
        first: &'static str,
        second: &'static str,
    ) -> Result<(i64, i64, Vec<i64>), TextError> {
        self.expect("[")?;
        let mut first_dim = None;
        let mut second_dim = None;
        let mut spatial = BTreeMap::new();
        let mut index = 0_i64;
        loop {
            self.space();
            if self
                .input
                .as_bytes()
                .get(self.offset)
                .is_some_and(u8::is_ascii_digit)
            {
                let number = self.i64()?;
                if spatial.insert(number, index).is_some() {
                    return Err(self.error(TextErrorKind::Expected("unique spatial dimension")));
                }
            } else {
                let keyword = self.word()?;
                let target = if keyword == first {
                    &mut first_dim
                } else if keyword == second {
                    &mut second_dim
                } else {
                    return Err(self.error(TextErrorKind::Expected("convolution dimension marker")));
                };
                if target.replace(index).is_some() {
                    return Err(self.error(TextErrorKind::Expected("unique nonspatial dimension")));
                }
            }
            index += 1;
            if self.take("]") {
                break;
            }
            self.expect(",")?;
        }
        // Validate dense numbering before allocating by any user-provided index.
        for (expected, &actual) in spatial.keys().enumerate() {
            if i64::try_from(expected).ok() != Some(actual) {
                return Err(self.error(TextErrorKind::Expected(
                    "spatial dimensions numbered consecutively from zero",
                )));
            }
        }
        Ok((
            first_dim.ok_or_else(|| self.error(TextErrorKind::Expected(first)))?,
            second_dim.ok_or_else(|| self.error(TextErrorKind::Expected(second)))?,
            spatial.into_values().collect(),
        ))
    }

    pub(super) fn accuracy_attribute(
        &mut self,
        depth: usize,
    ) -> Result<StableHloAttribute, TextError> {
        self.expect("<")?;
        let mut atol = 0.0;
        let mut rtol = 0.0;
        let mut ulps = 0;
        if self.take("atol") {
            self.expect("=")?;
            atol = self.float()?;
            self.expect(",")?;
        }
        if self.take("rtol") {
            self.expect("=")?;
            rtol = self.float()?;
            self.expect(",")?;
        }
        if self.take("ulps") {
            self.expect("=")?;
            ulps = self.i64()?;
            self.expect(",")?;
        }
        self.expect("mode")?;
        self.expect("=")?;
        let crate::Attribute::Enum {
            dialect,
            name,
            value,
        } = self.attribute(depth + 1)?
        else {
            return Err(self.error(TextErrorKind::Expected("StableHLO result accuracy mode")));
        };
        if dialect != "stablehlo" || name != "result_accuracy_mode" {
            return Err(self.error(TextErrorKind::Expected("StableHLO result accuracy mode")));
        }
        let mode = match value.as_str() {
            "DEFAULT" => ResultAccuracyMode::Default,
            "HIGHEST" => ResultAccuracyMode::Highest,
            "TOLERANCE" => ResultAccuracyMode::Tolerance,
            _ => return Err(self.error(TextErrorKind::Expected("DEFAULT, HIGHEST or TOLERANCE"))),
        };
        self.expect(">")?;
        Ok(StableHloAttribute::ResultAccuracy(ResultAccuracy::new(
            atol, rtol, ulps, mode,
        )))
    }

    fn sub_axis(&mut self) -> Result<SubAxisInfo, TextError> {
        // ODS prints the nested SubAxisInfo without a dialect prefix.
        self.expect("(")?;
        let pre_size = self.i64()?;
        self.expect(")")?;
        let size = self.i64()?;
        Ok(SubAxisInfo { pre_size, size })
    }

    pub(super) fn mesh_attribute(
        &mut self,
        name: &str,
        depth: usize,
    ) -> Result<StableHloAttribute, TextError> {
        match name {
            "sub_axis_info" => Ok(StableHloAttribute::SubAxisInfo(self.sub_axis()?)),
            "axis_ref" => {
                let mut name = String::new();
                let mut sub_axis_info = None;
                let seen = self.fields(|cursor, key| {
                    match key {
                        "name" => name = cursor.string()?,
                        "sub_axis_info" => sub_axis_info = Some(cursor.sub_axis()?),
                        _ => {
                            return Err(
                                cursor.error(TextErrorKind::Expected("name or sub_axis_info"))
                            );
                        }
                    }
                    Ok(())
                })?;
                self.required(&seen, &["name"])?;
                Ok(StableHloAttribute::AxisRef(AxisRef {
                    name,
                    sub_axis_info,
                }))
            }
            "mesh_axis" => {
                let mut name = String::new();
                let mut size = 0;
                let seen = self.fields(|cursor, key| {
                    match key {
                        "name" => name = cursor.string()?,
                        "size" => size = cursor.i64()?,
                        _ => return Err(cursor.error(TextErrorKind::Expected("name or size"))),
                    }
                    Ok(())
                })?;
                self.required(&seen, &["name", "size"])?;
                Ok(StableHloAttribute::MeshAxis(MeshAxis { name, size }))
            }
            "mesh" => {
                self.expect("<")?;
                self.expect("axes")?;
                self.expect("=")?;
                self.expect("[")?;
                let mut axes = Vec::new();
                if !self.take("]") {
                    loop {
                        match self.attribute(depth + 1)? {
                            Attribute::StableHlo(attribute) => match *attribute {
                                StableHloAttribute::MeshAxis(axis) => axes.push(axis),
                                _ => {
                                    return Err(
                                        self.error(TextErrorKind::Expected("mesh_axis attribute"))
                                    );
                                }
                            },
                            _ => {
                                return Err(
                                    self.error(TextErrorKind::Expected("mesh_axis attribute"))
                                );
                            }
                        }
                        if self.take("]") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                let device_ids = if self.take(",") {
                    self.expect("device_ids")?;
                    self.expect("=")?;
                    Some(self.mesh_device_ids(depth + 1)?)
                } else {
                    None
                };
                self.expect(">")?;
                Ok(StableHloAttribute::Mesh(Mesh { axes, device_ids }))
            }
            "replica_group_mesh_axes" => {
                let mut mesh = None;
                let mut axes = Vec::new();
                let seen = self.fields(|cursor, key| {
                    match key {
                        "mesh" => {
                            mesh = Some(match cursor.attribute(depth + 1)? {
                                Attribute::SymbolRef(path) if path.len() == 1 => {
                                    MeshReference::Symbol(path[0].clone())
                                }
                                Attribute::StableHlo(attribute) => match *attribute {
                                    StableHloAttribute::Mesh(mesh) => MeshReference::Inline(mesh),
                                    _ => {
                                        return Err(cursor.error(TextErrorKind::Expected(
                                            "mesh attribute or flat symbol",
                                        )));
                                    }
                                },
                                _ => {
                                    return Err(cursor.error(TextErrorKind::Expected(
                                        "mesh attribute or flat symbol",
                                    )));
                                }
                            });
                        }
                        "axes" => match cursor.attribute(depth + 1)? {
                            Attribute::Array(attributes) => {
                                for attribute in attributes {
                                    match attribute {
                                        Attribute::StableHlo(attribute) => match *attribute {
                                            StableHloAttribute::AxisRef(axis) => axes.push(axis),
                                            _ => {
                                                return Err(cursor.error(TextErrorKind::Expected(
                                                    "axis_ref attribute",
                                                )));
                                            }
                                        },
                                        _ => {
                                            return Err(cursor.error(TextErrorKind::Expected(
                                                "axis_ref attribute",
                                            )));
                                        }
                                    }
                                }
                            }
                            _ => {
                                return Err(cursor.error(TextErrorKind::Expected("axis_ref array")));
                            }
                        },
                        _ => return Err(cursor.error(TextErrorKind::Expected("mesh or axes"))),
                    }
                    Ok(())
                })?;
                self.required(&seen, &["mesh", "axes"])?;
                Ok(StableHloAttribute::ReplicaGroupMeshAxes(
                    ReplicaGroupMeshAxes {
                        mesh: mesh.ok_or_else(|| self.error(TextErrorKind::Expected("mesh")))?,
                        axes,
                    },
                ))
            }
            _ => Err(self.error(TextErrorKind::Unsupported("this mesh attribute"))),
        }
    }

    fn mesh_device_ids(&mut self, depth: usize) -> Result<Vec<i64>, TextError> {
        let Attribute::DenseElements { ty, values } = self.attribute(depth)? else {
            return Err(self.error(TextErrorKind::Expected("dense device IDs")));
        };
        let crate::Type::Tensor(tensor) = ty.as_ref() else {
            return Err(self.error(TextErrorKind::Expected("rank-one i64 tensor")));
        };
        let [Some(_)] = tensor.dimensions() else {
            return Err(self.error(TextErrorKind::Expected("static rank-one tensor")));
        };
        if tensor.element() != crate::ElementType::I64 || tensor.encoding().is_some() {
            return Err(self.error(TextErrorKind::Expected("rank-one i64 tensor")));
        }
        values
            .into_iter()
            .map(|value| match value {
                crate::ScalarValue::Integer(value) => {
                    i64::try_from(value).map_err(|_| self.error(TextErrorKind::InvalidNumber))
                }
                _ => Err(self.error(TextErrorKind::Expected("integer device ID"))),
            })
            .collect()
    }
}

pub(super) fn print(attribute: &StableHloAttribute, depth: usize) -> Result<String, TextError> {
    if depth >= 128 {
        return Err(TextError {
            offset: 0,
            kind: TextErrorKind::NestingLimit,
        });
    }
    let output = match attribute {
        StableHloAttribute::ConvDimensionNumbers(dimensions) => {
            // The reference's raw format is lossless even for dimension sets
            // that cannot be rendered as compressed permutations.
            format!(
                "#stablehlo.conv<raw input_batch_dimension = {}, input_feature_dimension = {}, input_spatial_dimensions = {}, kernel_input_feature_dimension = {}, kernel_output_feature_dimension = {}, kernel_spatial_dimensions = {}, output_batch_dimension = {}, output_feature_dimension = {}, output_spatial_dimensions = {}>",
                dimensions.input_batch_dimension,
                dimensions.input_feature_dimension,
                list(&dimensions.input_spatial_dimensions),
                dimensions.kernel_input_feature_dimension,
                dimensions.kernel_output_feature_dimension,
                list(&dimensions.kernel_spatial_dimensions),
                dimensions.output_batch_dimension,
                dimensions.output_feature_dimension,
                list(&dimensions.output_spatial_dimensions)
            )
        }
        StableHloAttribute::ResultAccuracy(accuracy) => {
            let mode = match accuracy.mode {
                ResultAccuracyMode::Default => "DEFAULT",
                ResultAccuracyMode::Highest => "HIGHEST",
                ResultAccuracyMode::Tolerance => "TOLERANCE",
            };
            format!(
                "#stablehlo.result_accuracy<atol = {}, rtol = {}, ulps = {}, mode = #stablehlo.result_accuracy_mode<{mode}>>",
                float(accuracy.atol_bits),
                float(accuracy.rtol_bits),
                accuracy.ulps
            )
        }
        StableHloAttribute::SubAxisInfo(info) => {
            format!("#stablehlo<sub_axis_info{}>", sub_axis(info))
        }
        StableHloAttribute::AxisRef(axis) => format!(
            "#stablehlo.axis_ref<name = {}{}>",
            quote(&axis.name),
            axis.sub_axis_info
                .as_ref()
                .map_or(String::new(), |info| format!(
                    ", sub_axis_info = {}",
                    sub_axis(info)
                ))
        ),
        StableHloAttribute::MeshAxis(axis) => format!(
            "#stablehlo.mesh_axis<name = {}, size = {}>",
            quote(&axis.name),
            axis.size
        ),
        StableHloAttribute::Mesh(mesh) => {
            let axes = mesh
                .axes
                .iter()
                .map(|axis| print(&StableHloAttribute::MeshAxis(axis.clone()), depth + 1))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            let devices = mesh.device_ids.as_ref().map_or(String::new(), |ids| {
                format!(
                    ", device_ids = dense<{}> : tensor<{}xi64>",
                    list(ids),
                    ids.len()
                )
            });
            format!("#stablehlo.mesh<axes = [{axes}]{devices}>")
        }
        StableHloAttribute::ReplicaGroupMeshAxes(groups) => {
            let mesh = match &groups.mesh {
                MeshReference::Inline(mesh) => {
                    print(&StableHloAttribute::Mesh(mesh.clone()), depth + 1)?
                }
                MeshReference::Symbol(name) => format!("@{}", quote(name)),
            };
            let axes = groups
                .axes
                .iter()
                .map(|axis| print(&StableHloAttribute::AxisRef(axis.clone()), depth + 1))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            format!("#stablehlo.replica_group_mesh_axes<mesh = {mesh}, axes = [{axes}]>")
        }
        _ => {
            return Err(TextError {
                offset: 0,
                kind: TextErrorKind::Unsupported("this geometry attribute"),
            });
        }
    };
    Ok(output)
}

fn float(bits: u64) -> String {
    super::floats::print_f64(bits)
}

fn sub_axis(info: &SubAxisInfo) -> String {
    format!("({}){}", info.pre_size, info.size)
}
