use crate::attribute::stablehlo::*;

use super::*;

impl PayloadDecoder<'_, '_> {
    pub(super) fn stablehlo_attribute(
        &self,
        cursor: &mut Cursor<'_>,
        depth: usize,
    ) -> Result<Attribute, BytecodeError> {
        let tag = cursor.varint()?;
        let enumeration: Option<(&str, &[&str])> = match tag {
            2 => Some((
                "comparison_direction",
                &["EQ", "NE", "GE", "GT", "LE", "LT"],
            )),
            3 => Some((
                "comparison_type",
                &[
                    "NOTYPE",
                    "FLOAT",
                    "TOTALORDER",
                    "SIGNED",
                    "UNSIGNED",
                    "WEAKORDER",
                ],
            )),
            6 => Some(("fft_type", &["FFT", "IFFT", "RFFT", "IRFFT"])),
            8 => Some(("precision", &["DEFAULT", "HIGH", "HIGHEST"])),
            9 => Some(("rng_algorithm", &["DEFAULT", "THREE_FRY", "PHILOX"])),
            10 => Some(("rng_distribution", &["", "UNIFORM", "NORMAL"])),
            12 => Some((
                "transpose",
                &["TRANSPOSE_INVALID", "NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"],
            )),
            16 => Some(("result_accuracy_mode", &["DEFAULT", "HIGHEST", "TOLERANCE"])),
            _ => None,
        };
        if let Some((name, values)) = enumeration {
            let index = cursor.varint()?;
            let value = usize::try_from(index)
                .ok()
                .and_then(|index| values.get(index))
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    cursor.error(Kind::InvalidPayload("defined StableHLO enum value"))
                })?;
            return Ok(Attribute::Enum {
                dialect: "stablehlo".into(),
                name: name.into(),
                value: (*value).into(),
            });
        }
        let value = match tag {
            1 => StableHloAttribute::ChannelHandle(ChannelHandle {
                handle: Self::signed(cursor)?,
                channel_type: Self::signed(cursor)?,
            }),
            4 => StableHloAttribute::ConvDimensionNumbers(ConvDimensionNumbers {
                input_batch_dimension: Self::signed(cursor)?,
                input_feature_dimension: Self::signed(cursor)?,
                input_spatial_dimensions: Self::signed_list(cursor)?,
                kernel_input_feature_dimension: Self::signed(cursor)?,
                kernel_output_feature_dimension: Self::signed(cursor)?,
                kernel_spatial_dimensions: Self::signed_list(cursor)?,
                output_batch_dimension: Self::signed(cursor)?,
                output_feature_dimension: Self::signed(cursor)?,
                output_spatial_dimensions: Self::signed_list(cursor)?,
            }),
            5 => StableHloAttribute::DotDimensionNumbers(DotDimensionNumbers {
                lhs_batching_dimensions: Self::signed_list(cursor)?,
                rhs_batching_dimensions: Self::signed_list(cursor)?,
                lhs_contracting_dimensions: Self::signed_list(cursor)?,
                rhs_contracting_dimensions: Self::signed_list(cursor)?,
            }),
            7 => StableHloAttribute::GatherDimensionNumbers(GatherDimensionNumbers {
                offset_dims: Self::signed_list(cursor)?,
                collapsed_slice_dims: Self::signed_list(cursor)?,
                operand_batching_dims: Self::signed_list(cursor)?,
                start_indices_batching_dims: Self::signed_list(cursor)?,
                start_index_map: Self::signed_list(cursor)?,
                index_vector_dim: Self::signed(cursor)?,
            }),
            11 => StableHloAttribute::ScatterDimensionNumbers(ScatterDimensionNumbers {
                update_window_dims: Self::signed_list(cursor)?,
                inserted_window_dims: Self::signed_list(cursor)?,
                input_batching_dims: Self::signed_list(cursor)?,
                scatter_indices_batching_dims: Self::signed_list(cursor)?,
                scatter_dims_to_operand_dims: Self::signed_list(cursor)?,
                index_vector_dim: Self::signed(cursor)?,
            }),
            13 => {
                let bounds = Self::signed_list(cursor)?
                    .into_iter()
                    .map(|bound| {
                        if bound == i64::MIN {
                            Ok(-1)
                        } else if bound >= 0 {
                            Ok(bound)
                        } else {
                            Err(cursor.error(Kind::InvalidPayload(
                                "nonnegative bound or dynamic sentinel",
                            )))
                        }
                    })
                    .collect::<Result<_, _>>()?;
                StableHloAttribute::TypeExtensions(TypeExtensions { bounds })
            }
            14 => StableHloAttribute::OutputOperandAlias(OutputOperandAlias {
                output_tuple_indices: Self::signed_list(cursor)?,
                operand_index: Self::signed(cursor)?,
                operand_tuple_indices: Self::signed_list(cursor)?,
            }),
            15 => StableHloAttribute::DotAlgorithm(Box::new(DotAlgorithm {
                lhs_precision_type: self.type_ref(cursor, depth)?,
                rhs_precision_type: self.type_ref(cursor, depth)?,
                accumulation_type: self.type_ref(cursor, depth)?,
                lhs_component_count: Self::signed(cursor)?,
                rhs_component_count: Self::signed(cursor)?,
                num_primitive_operations: Self::signed(cursor)?,
                allow_imprecise_accumulation: cursor.byte()? != 0,
            })),
            17 => {
                let atol_bits = Self::bits(cursor, 64)?;
                let rtol_bits = Self::bits(cursor, 64)?;
                let ulps = Self::signed(cursor)?;
                let mode = match self.attr_ref(cursor, depth)? {
                    Attribute::Enum {
                        dialect,
                        name,
                        value,
                    } if dialect == "stablehlo" && name == "result_accuracy_mode" => match value
                        .as_str()
                    {
                        "DEFAULT" => ResultAccuracyMode::Default,
                        "HIGHEST" => ResultAccuracyMode::Highest,
                        "TOLERANCE" => ResultAccuracyMode::Tolerance,
                        _ => return Err(cursor.error(Kind::InvalidPayload("result accuracy mode"))),
                    },
                    _ => {
                        return Err(
                            cursor.error(Kind::InvalidPayload("result accuracy mode attribute"))
                        );
                    }
                };
                StableHloAttribute::ResultAccuracy(ResultAccuracy {
                    atol_bits,
                    rtol_bits,
                    ulps,
                    mode,
                })
            }
            18 => {
                let mesh = match self.attr_ref(cursor, depth)? {
                    Attribute::StableHlo(value) => match *value {
                        StableHloAttribute::Mesh(value) => MeshReference::Inline(value),
                        _ => return Err(cursor.error(Kind::InvalidPayload("mesh attribute"))),
                    },
                    Attribute::SymbolRef(value) if value.len() == 1 => {
                        MeshReference::Symbol(value[0].clone())
                    }
                    _ => {
                        return Err(
                            cursor.error(Kind::InvalidPayload("mesh or flat symbol attribute"))
                        );
                    }
                };
                let axes = self
                    .attrs(cursor, depth)?
                    .into_iter()
                    .map(|value| match value {
                        Attribute::StableHlo(value) => match *value {
                            StableHloAttribute::AxisRef(value) => Ok(value),
                            _ => {
                                Err(cursor.error(Kind::InvalidPayload("axis reference attribute")))
                            }
                        },
                        _ => Err(cursor.error(Kind::InvalidPayload("axis reference attribute"))),
                    })
                    .collect::<Result<_, _>>()?;
                StableHloAttribute::ReplicaGroupMeshAxes(ReplicaGroupMeshAxes { mesh, axes })
            }
            19 => StableHloAttribute::SubAxisInfo(SubAxisInfo {
                pre_size: Self::signed(cursor)?,
                size: Self::signed(cursor)?,
            }),
            20 => {
                let name = String::from_utf8(self.string(cursor)?)
                    .map_err(|_| cursor.error(Kind::InvalidUtf8))?;
                let sub_axis_info = if cursor.byte()? != 0 {
                    match self.attr_ref(cursor, depth)? {
                        Attribute::StableHlo(value) => match *value {
                            StableHloAttribute::SubAxisInfo(value) => Some(value),
                            _ => {
                                return Err(
                                    cursor.error(Kind::InvalidPayload("sub-axis attribute"))
                                );
                            }
                        },
                        _ => return Err(cursor.error(Kind::InvalidPayload("sub-axis attribute"))),
                    }
                } else {
                    None
                };
                StableHloAttribute::AxisRef(AxisRef {
                    name,
                    sub_axis_info,
                })
            }
            21 => StableHloAttribute::MeshAxis(MeshAxis {
                name: String::from_utf8(self.string(cursor)?)
                    .map_err(|_| cursor.error(Kind::InvalidUtf8))?,
                size: Self::signed(cursor)?,
            }),
            22 => {
                let axes = match self.attr_ref(cursor, depth)? {
                    Attribute::Array(values) => values
                        .into_iter()
                        .map(|value| match value {
                            Attribute::StableHlo(value) => match *value {
                                StableHloAttribute::MeshAxis(value) => Ok(value),
                                _ => Err(cursor.error(Kind::InvalidPayload("mesh axis attribute"))),
                            },
                            _ => Err(cursor.error(Kind::InvalidPayload("mesh axis attribute"))),
                        })
                        .collect::<Result<_, _>>()?,
                    _ => return Err(cursor.error(Kind::InvalidPayload("mesh axis array"))),
                };
                let device_ids = if cursor.byte()? != 0 {
                    match self.attr_ref(cursor, depth)? {
                        Attribute::DenseElements { ty, values } if matches!(ty.as_ref(), Type::Tensor(ty) if ty.rank() == 1 && ty.element() == ElementType::I64) => {
                            Some(
                                values
                                    .into_iter()
                                    .map(|value| match value {
                                        crate::ScalarValue::Integer(value) => i64::try_from(value)
                                            .map_err(|_| {
                                                cursor
                                                    .error(Kind::InvalidPayload("64-bit device id"))
                                            }),
                                        _ => {
                                            Err(cursor
                                                .error(Kind::InvalidPayload("integer device id")))
                                        }
                                    })
                                    .collect::<Result<_, _>>()?,
                            )
                        }
                        _ => {
                            return Err(
                                cursor.error(Kind::InvalidPayload("rank-one i64 device ids"))
                            );
                        }
                    }
                } else {
                    None
                };
                StableHloAttribute::Mesh(Mesh { axes, device_ids })
            }
            _ => return Err(cursor.error(Kind::Unsupported("StableHLO attribute code"))),
        };
        Ok(Attribute::StableHlo(Box::new(value)))
    }
}
