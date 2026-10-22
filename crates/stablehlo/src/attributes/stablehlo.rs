use super::*;
use crate::Diagnostic;
use pliron::{builtin::attributes::StringAttr, derive::pliron_attr};
use pliron::{location::Location, verify_err};

#[pliron_attr(
    name = "stablehlo.dot_dimension_numbers",
    format = "`<` `[` vec($lhs_batching_dimensions, CharSpace(`,`)) `]` `,` `[` vec($rhs_batching_dimensions, CharSpace(`,`)) `]` `,` `[` vec($lhs_contracting_dimensions, CharSpace(`,`)) `]` `,` `[` vec($rhs_contracting_dimensions, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dimension selections that define dot batching and contracting axes.
pub struct DotDimensionNumbersAttr {
    /// Batch dimensions selected from the left-hand operand.
    pub lhs_batching_dimensions: Vec<i64>,
    /// Batch dimensions selected from the right-hand operand.
    pub rhs_batching_dimensions: Vec<i64>,
    /// Contracting dimensions selected from the left-hand operand.
    pub lhs_contracting_dimensions: Vec<i64>,
    /// Contracting dimensions selected from the right-hand operand.
    pub rhs_contracting_dimensions: Vec<i64>,
}
impl Verify for DotDimensionNumbersAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.lhs_batching_dimensions.iter().any(|&v| v < 0)
            || self.rhs_batching_dimensions.iter().any(|&v| v < 0)
            || self.lhs_contracting_dimensions.iter().any(|&v| v < 0)
            || self.rhs_contracting_dimensions.iter().any(|&v| v < 0)
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid dot_dimension_numbers fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.gather_dimension_numbers",
    format = "`<` `[` vec($offset_dims, CharSpace(`,`)) `]` `,` `[` vec($collapsed_slice_dims, CharSpace(`,`)) `]` `,` `[` vec($operand_batching_dims, CharSpace(`,`)) `]` `,` `[` vec($start_indices_batching_dims, CharSpace(`,`)) `]` `,` `[` vec($start_index_map, CharSpace(`,`)) `]` `,` $index_vector_dim `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dimension selections that define how gather indices address an operand.
pub struct GatherDimensionNumbersAttr {
    /// Result dimensions occupied by operand window dimensions.
    pub offset_dims: Vec<i64>,
    /// Operand dimensions removed because each slice has extent one.
    pub collapsed_slice_dims: Vec<i64>,
    /// Operand dimensions paired with start-index batch dimensions.
    pub operand_batching_dims: Vec<i64>,
    /// Start-indices dimensions paired with operand batch dimensions.
    pub start_indices_batching_dims: Vec<i64>,
    /// Operand dimensions addressed by entries of the index vector.
    pub start_index_map: Vec<i64>,
    /// Dimension of `start_indices` containing index-vector entries.
    pub index_vector_dim: i64,
}
impl Verify for GatherDimensionNumbersAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.offset_dims.iter().any(|&v| v < 0)
            || self.collapsed_slice_dims.iter().any(|&v| v < 0)
            || self.operand_batching_dims.iter().any(|&v| v < 0)
            || self.start_indices_batching_dims.iter().any(|&v| v < 0)
            || self.start_index_map.iter().any(|&v| v < 0)
            || self.index_vector_dim < 0
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid gather_dimension_numbers fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.scatter_dimension_numbers",
    format = "`<` `[` vec($update_window_dims, CharSpace(`,`)) `]` `,` `[` vec($inserted_window_dims, CharSpace(`,`)) `]` `,` `[` vec($input_batching_dims, CharSpace(`,`)) `]` `,` `[` vec($scatter_indices_batching_dims, CharSpace(`,`)) `]` `,` `[` vec($scatter_dims_to_operand_dims, CharSpace(`,`)) `]` `,` $index_vector_dim `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dimension selections that define scatter update and index mappings.
pub struct ScatterDimensionNumbersAttr {
    /// Update dimensions that correspond to operand window dimensions.
    pub update_window_dims: Vec<i64>,
    /// Operand dimensions inserted as size-one windows.
    pub inserted_window_dims: Vec<i64>,
    /// Input dimensions paired with scatter-index batch dimensions.
    pub input_batching_dims: Vec<i64>,
    /// Scatter-index dimensions paired with input batch dimensions.
    pub scatter_indices_batching_dims: Vec<i64>,
    /// Operand dimensions addressed by scatter-index vector entries.
    pub scatter_dims_to_operand_dims: Vec<i64>,
    /// Dimension of `scatter_indices` containing index-vector entries.
    pub index_vector_dim: i64,
}
impl Verify for ScatterDimensionNumbersAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.update_window_dims.iter().any(|&v| v < 0)
            || self.inserted_window_dims.iter().any(|&v| v < 0)
            || self.input_batching_dims.iter().any(|&v| v < 0)
            || self.scatter_indices_batching_dims.iter().any(|&v| v < 0)
            || self.scatter_dims_to_operand_dims.iter().any(|&v| v < 0)
            || self.index_vector_dim < 0
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid scatter_dimension_numbers fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.conv_dimension_numbers",
    format = "`<` $input_batch_dimension `,` $input_feature_dimension `,` `[` vec($input_spatial_dimensions, CharSpace(`,`)) `]` `,` $kernel_input_feature_dimension `,` $kernel_output_feature_dimension `,` `[` vec($kernel_spatial_dimensions, CharSpace(`,`)) `]` `,` $output_batch_dimension `,` $output_feature_dimension `,` `[` vec($output_spatial_dimensions, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Input, kernel, and output axis positions for a convolution.
pub struct ConvDimensionNumbersAttr {
    /// Batch dimension of the convolution input.
    pub input_batch_dimension: i64,
    /// Feature/channel dimension of the convolution input.
    pub input_feature_dimension: i64,
    /// Spatial dimensions of the convolution input.
    pub input_spatial_dimensions: Vec<i64>,
    /// Input-feature dimension of the convolution kernel.
    pub kernel_input_feature_dimension: i64,
    /// Output-feature dimension of the convolution kernel.
    pub kernel_output_feature_dimension: i64,
    /// Spatial dimensions of the convolution kernel.
    pub kernel_spatial_dimensions: Vec<i64>,
    /// Batch dimension of the convolution result.
    pub output_batch_dimension: i64,
    /// Feature/channel dimension of the convolution result.
    pub output_feature_dimension: i64,
    /// Spatial dimensions of the convolution result.
    pub output_spatial_dimensions: Vec<i64>,
}
impl Verify for ConvDimensionNumbersAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.input_batch_dimension < 0
            || self.input_feature_dimension < 0
            || self.input_spatial_dimensions.iter().any(|&v| v < 0)
            || self.kernel_input_feature_dimension < 0
            || self.kernel_output_feature_dimension < 0
            || self.kernel_spatial_dimensions.iter().any(|&v| v < 0)
            || self.output_batch_dimension < 0
            || self.output_feature_dimension < 0
            || self.output_spatial_dimensions.iter().any(|&v| v < 0)
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid conv_dimension_numbers fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.dot_algorithm",
    format = "`<` $lhs_precision_type `,` $rhs_precision_type `,` $accumulation_type `,` $lhs_component_count `,` $rhs_component_count `,` $num_primitive_operations `,` $allow_imprecise_accumulation `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Operand formats and operation counts for a dot algorithm.
pub struct DotAlgorithmAttr {
    /// Element type used to interpret the left-hand operand precision.
    pub lhs_precision_type: TypeHandle,
    /// Element type used to interpret the right-hand operand precision.
    pub rhs_precision_type: TypeHandle,
    /// Element type used for accumulation.
    pub accumulation_type: TypeHandle,
    /// Number of components used to represent each left-hand value.
    pub lhs_component_count: i64,
    /// Number of components used to represent each right-hand value.
    pub rhs_component_count: i64,
    /// Number of primitive operations in the algorithm.
    pub num_primitive_operations: i64,
    /// Whether accumulation may use reduced precision.
    pub allow_imprecise_accumulation: bool,
}
impl Verify for DotAlgorithmAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.lhs_component_count < 0
            || self.rhs_component_count < 0
            || self.num_primitive_operations < 0
            || self.lhs_component_count == 0
            || self.rhs_component_count == 0
            || self.num_primitive_operations == 0
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid dot_algorithm fields"
                }
            );
        }
        self.lhs_precision_type.deref(ctx).verify(ctx)?;
        self.rhs_precision_type.deref(ctx).verify(ctx)?;
        self.accumulation_type.deref(ctx).verify(ctx)?;
        let _ = ctx;
        if !is_known_dot_algorithm(ctx, self) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "unknown dot algorithm type/count combination"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.output_operand_alias",
    format = "`<` `[` vec($output_tuple_indices, CharSpace(`,`)) `]` `,` $operand_index `,` `[` vec($operand_tuple_indices, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Identifies an output value that aliases storage from an operand.
pub struct OutputOperandAliasAttr {
    /// Path through nested result tuples to the aliased output.
    pub output_tuple_indices: Vec<i64>,
    /// Zero-based operand number providing the aliased storage.
    pub operand_index: i64,
    /// Path through nested operand tuples to the aliased value.
    pub operand_tuple_indices: Vec<i64>,
}
impl Verify for OutputOperandAliasAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.output_tuple_indices.iter().any(|&v| v < 0)
            || self.operand_index < 0
            || self.operand_tuple_indices.iter().any(|&v| v < 0)
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid output_operand_alias fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.channel_type", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i64)]
/// Direction of communication for a StableHLO channel.
pub enum ChannelType {
    /// Send data between devices.
    DeviceToDevice = 1,
    /// Send data from a device to the host.
    DeviceToHost = 2,
    /// Receive data from the host to a device.
    HostToDevice = 3,
}

#[pliron_attr(
    name = "stablehlo.channel_handle",
    format = "`<` $handle `,` $channel_type `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Identifier and category of a communication channel.
pub struct ChannelHandleAttr {
    /// Identifier of the communication channel.
    pub handle: i64,
    /// Channel category used by the operation.
    pub channel_type: ChannelType,
}
impl Verify for ChannelHandleAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.type_extensions",
    format = "`<` `[` vec($bounds, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Optional per-dimension bounds for tensor type extensions.
pub struct TypeExtensionsAttr {
    /// Per-dimension bounds; `-1` represents an unspecified bound.
    pub bounds: Vec<i64>,
}
impl Verify for TypeExtensionsAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.bounds.iter().any(|&b| b < -1) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid type_extensions fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.result_accuracy",
    format = "`<` $atol_bits `,` $rtol_bits `,` $ulps `,` $mode `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Absolute, relative, and ULP tolerances for result comparison.
pub struct ResultAccuracyAttr {
    /// Exact `f64` bits of the absolute error tolerance.
    pub atol_bits: u64,
    /// Exact `f64` bits of the relative error tolerance.
    pub rtol_bits: u64,
    /// Maximum permitted error in units in the last place.
    pub ulps: i64,
    /// Policy for interpreting the supplied tolerances.
    pub mode: ResultAccuracyMode,
}
impl ResultAccuracyAttr {
    /// Encodes tolerances and mode in the native representation.
    pub fn new(atol: f64, rtol: f64, ulps: i64, mode: ResultAccuracyMode) -> Self {
        Self {
            atol_bits: atol.to_bits(),
            rtol_bits: rtol.to_bits(),
            ulps,
            mode,
        }
    }
    /// Decodes the absolute tolerance from its stored bit pattern.
    pub fn atol(&self) -> f64 {
        f64::from_bits(self.atol_bits)
    }
    /// Decodes the relative tolerance from its stored bit pattern.
    pub fn rtol(&self) -> f64 {
        f64::from_bits(self.rtol_bits)
    }
}
impl Verify for ResultAccuracyAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.atol().is_sign_negative() || self.rtol().is_sign_negative() || self.ulps < 0 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "result accuracy tolerances/mode are invalid"
                }
            );
        }
        if self.mode != ResultAccuracyMode::Tolerance
            && (self.atol() != 0.0 || self.rtol() != 0.0 || self.ulps != 0)
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "DEFAULT and HIGHEST require zero tolerances"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.sub_axis_info",
    format = "`<` $pre_size `,` $size `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Size metadata selecting a sub-axis within a mesh axis.
pub struct SubAxisInfoAttr {
    /// Size of the enclosing axis before this sub-axis.
    pub pre_size: i64,
    /// Extent of this sub-axis.
    pub size: i64,
}
impl Verify for SubAxisInfoAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.pre_size <= 0 || self.size <= 0 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid sub_axis_info_info fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.axis_ref",
    format = "`<` $name `,` `[` vec($sub_axis_info, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Reference to a mesh axis, optionally narrowed to a sub-axis.
pub struct AxisRefAttr {
    /// Name of the referenced mesh axis.
    pub name: StringAttr,
    /// Optional single sub-axis selection.
    pub sub_axis_info: Vec<SubAxisInfoAttr>,
}
impl Verify for AxisRefAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.name.as_str().is_empty() || self.sub_axis_info.len() > 1 {
            return verify_err!(
                Location::Unknown,
                Diagnostic::InvalidMesh("invalid axis reference fields")
            );
        }
        for value in &self.sub_axis_info {
            value.verify(ctx)?;
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.mesh_axis", format = "`<` $name `,` $size `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Named axis and its positive extent in a device mesh.
pub struct MeshAxisAttr {
    /// Unique name of this mesh axis.
    pub name: StringAttr,
    /// Positive number of devices along this axis.
    pub size: i64,
}
impl Verify for MeshAxisAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.size < 0 || self.name.as_str().is_empty() || self.size <= 0 {
            return verify_err!(
                Location::Unknown,
                Diagnostic::InvalidMesh("invalid mesh axis fields")
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.mesh",
    format = "`<` `[` vec($axes, CharSpace(`,`)) `]` `,` `[` vec($device_ids, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Mesh topology and optional device identifier assignment.
pub struct MeshAttr {
    /// Mesh axes, each with a unique name and positive extent.
    pub axes: Vec<MeshAxisAttr>,
    /// Optional device-id array describing the device assignment.
    pub device_ids: Vec<IndexTensorAttr>,
}
impl Verify for MeshAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let mut names = std::collections::HashSet::new();
        for axis in &self.axes {
            axis.verify(ctx)?;
            if !names.insert(axis.name.as_str()) {
                return verify_err!(
                    Location::Unknown,
                    Diagnostic::DuplicateMeshAxis(axis.name.as_str().to_owned())
                );
            }
        }
        if self.device_ids.len() > 1 {
            return verify_err!(
                Location::Unknown,
                Diagnostic::InvalidMesh("mesh has multiple device ID arrays")
            );
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.replica_group_mesh_axes",
    format = "`<` $mesh `,` `[` vec($axes, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Mesh reference paired with the axes used to form replica groups.
pub struct ReplicaGroupMeshAxesAttr {
    /// Inline or symbolic mesh referenced by these axes.
    pub mesh: MeshReferenceAttr,
    /// Axes whose products determine replica-group membership.
    pub axes: Vec<AxisRefAttr>,
}
impl Verify for ReplicaGroupMeshAxesAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.mesh.verify(ctx)?;
        for axis in &self.axes {
            axis.verify(ctx)?;
            if let Some(mesh) = self.mesh.inline.first()
                && !mesh
                    .axes
                    .iter()
                    .any(|candidate| candidate.name == axis.name)
            {
                return verify_err!(
                    Location::Unknown,
                    Diagnostic::UnknownMeshAxis(axis.name.as_str().to_owned())
                );
            }
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.index_tensor",
    format = "`<` `[` vec($values, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Flattened nonnegative device or coordinate indices.
pub struct IndexTensorAttr {
    /// Nonnegative flattened coordinate values.
    pub values: Vec<i64>,
}
impl Verify for IndexTensorAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.values.iter().any(|&v| v < 0) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid index_tensor fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.layout",
    format = "`<` `[` vec($minor_to_major, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Dimension order describing a ranked tensor layout.
pub struct LayoutAttr {
    /// Dimension order from minor (fastest varying) to major.
    pub minor_to_major: Vec<i64>,
}
impl Verify for LayoutAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.minor_to_major.iter().any(|&v| v < 0) {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid layout fields"
                }
            );
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.array_of_layout",
    format = "`<` `[` vec($layouts, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Ordered collection of layout descriptors.
pub struct ArrayOfLayoutAttr {
    /// Layout descriptors in array order.
    pub layouts: Vec<LayoutAttr>,
}
impl Verify for ArrayOfLayoutAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for value in &self.layouts {
            value.verify(ctx)?;
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.tiling",
    format = "`<` `[` vec($tiles, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Tile sizes for a tiled operation configuration.
pub struct TilingAttr {
    /// Tile sizes represented as index tensors.
    pub tiles: Vec<IndexTensorAttr>,
}
impl Verify for TilingAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for value in &self.tiles {
            value.verify(ctx)?;
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.array_of_tiling",
    format = "`<` `[` vec($tilings, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Ordered collection of tiling descriptors.
pub struct ArrayOfTilingAttr {
    /// Tiling descriptors in array order.
    pub tilings: Vec<TilingAttr>,
}
impl Verify for ArrayOfTilingAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for value in &self.tilings {
            value.verify(ctx)?;
        }
        let _ = ctx;
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.fft_type", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Fourier transform kind used by StableHLO FFT operations.
pub enum FftTypeAttr {
    /// Complex-to-complex forward transform.
    Fft,
    /// Complex-to-complex inverse transform.
    Ifft,
    /// Real-to-complex forward transform.
    Rfft,
    /// Complex-to-real inverse transform.
    Irfft,
}

#[pliron_attr(
    name = "stablehlo.mesh_reference",
    format = "`<` `[` vec($inline, CharSpace(`,`)) `]` `,` `[` vec($symbol, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Either an inline device mesh or a symbolic mesh path.
pub struct MeshReferenceAttr {
    /// Inline mesh form; valid references contain exactly one inline mesh or symbol.
    pub inline: Vec<MeshAttr>,
    /// Symbolic mesh path; valid references contain exactly one inline mesh or symbol.
    pub symbol: Vec<StringAttr>,
}
impl Verify for MeshReferenceAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.inline.len() + self.symbol.len() != 1
            || self.symbol.first().is_some_and(|s| s.as_str().is_empty())
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "mesh reference requires one inline mesh or nonempty symbol"
                }
            );
        }
        for mesh in &self.inline {
            mesh.verify(ctx)?;
        }
        Ok(())
    }
}
#[pliron_attr(name = "stablehlo.precision", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Precision preference for floating-point operations.
pub enum Precision {
    /// Use the operation's default precision.
    Default,
    /// Request high precision.
    High,
    /// Request the highest supported precision.
    Highest,
}
#[pliron_attr(name = "stablehlo.result_accuracy_mode", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Policy for interpreting result-accuracy tolerances.
pub enum ResultAccuracyMode {
    /// Use the default accuracy policy, with zero explicit tolerances.
    Default,
    /// Request highest accuracy, with zero explicit tolerances.
    Highest,
    /// Apply the supplied absolute, relative, and ULP tolerances.
    Tolerance,
}
#[pliron_attr(name = "stablehlo.comparison_direction", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Relational operator used for elementwise comparison.
pub enum ComparisonDirection {
    /// Equal to.
    Eq,
    /// Not equal to.
    Ne,
    /// Greater than or equal to.
    Ge,
    /// Greater than.
    Gt,
    /// Less than or equal to.
    Le,
    /// Less than.
    Lt,
}
#[pliron_attr(name = "stablehlo.comparison_type", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Numeric ordering semantics used for elementwise comparison.
pub enum ComparisonType {
    /// No explicitly selected comparison type.
    NoType,
    /// Floating-point comparison semantics.
    Float,
    /// Total ordering over values, including exceptional floating-point values.
    TotalOrder,
    /// Signed integer comparison semantics.
    Signed,
    /// Unsigned integer comparison semantics.
    Unsigned,
    /// Weak ordering semantics.
    WeakOrder,
}

#[pliron_attr(name = "stablehlo.rng_algorithm", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Random number generator algorithm supported by StableHLO.
pub enum RngAlgorithmAttr {
    /// Implementation-selected algorithm.
    Default,
    /// ThreeFry counter-based algorithm.
    ThreeFry,
    /// Philox counter-based algorithm.
    Philox,
}

#[pliron_attr(name = "stablehlo.rng_distribution", format, verifier = "succ")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
/// Distribution used to generate random values.
pub enum RngDistributionAttr {
    /// Uniform distribution.
    Uniform,
    /// Normal distribution.
    Normal,
}
fn is_known_dot_algorithm(ctx: &Context, algorithm: &DotAlgorithmAttr) -> bool {
    let (Some(lhs), Some(rhs), Some(accumulation)) = (
        crate::types::primitive_element(ctx, algorithm.lhs_precision_type),
        crate::types::primitive_element(ctx, algorithm.rhs_precision_type),
        crate::types::primitive_element(ctx, algorithm.accumulation_type),
    ) else {
        return false;
    };
    if algorithm.lhs_component_count != 1 || algorithm.rhs_component_count != 1 {
        return false;
    }

    if is_dot_f8(lhs)
        && is_dot_f8(rhs)
        && accumulation == crate::types::ElementType::F32
        && algorithm.num_primitive_operations == 1
    {
        return true;
    }
    if algorithm.allow_imprecise_accumulation {
        return false;
    }

    use crate::types::ElementType::{Bf16, F8E4M3Fn, F16, F32, F64, Tf32};
    matches!(
        (lhs, rhs, accumulation, algorithm.num_primitive_operations),
        (F16, F16, F16 | F32, 1)
            | (Bf16, Bf16, Bf16 | F32, 1)
            | (Bf16, Bf16, F32, 3 | 6 | 9)
            | (F8E4M3Fn, F8E4M3Fn, F32, 3 | 4)
            | (Tf32, Tf32, F32, 1 | 3)
            | (F32, F32, F32, 1)
            | (F64, F64, F64, 1)
    )
}

fn is_dot_f8(element: crate::types::ElementType) -> bool {
    matches!(
        element,
        crate::types::ElementType::F8E3M4
            | crate::types::ElementType::F8E4M3
            | crate::types::ElementType::F8E4M3Fn
            | crate::types::ElementType::F8E5M2
            | crate::types::ElementType::F8E4M3FnUz
            | crate::types::ElementType::F8E4M3B11FnUz
            | crate::types::ElementType::F8E5M2FnUz
    )
}
