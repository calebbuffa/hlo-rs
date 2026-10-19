use super::*;
use crate::Diagnostic;
use pliron::{builtin::attributes::StringAttr, derive::pliron_attr};
use pliron::{location::Location, verify_err};

#[pliron_attr(
    name = "stablehlo.dot_dimension_numbers",
    format = "`<` `[` vec($lhs_batching_dimensions, CharSpace(`,`)) `]` `,` `[` vec($rhs_batching_dimensions, CharSpace(`,`)) `]` `,` `[` vec($lhs_contracting_dimensions, CharSpace(`,`)) `]` `,` `[` vec($rhs_contracting_dimensions, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DotDimensionNumbersAttr {
    pub lhs_batching_dimensions: Vec<i64>,
    pub rhs_batching_dimensions: Vec<i64>,
    pub lhs_contracting_dimensions: Vec<i64>,
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
pub struct GatherDimensionNumbersAttr {
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub operand_batching_dims: Vec<i64>,
    pub start_indices_batching_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
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
pub struct ScatterDimensionNumbersAttr {
    pub update_window_dims: Vec<i64>,
    pub inserted_window_dims: Vec<i64>,
    pub input_batching_dims: Vec<i64>,
    pub scatter_indices_batching_dims: Vec<i64>,
    pub scatter_dims_to_operand_dims: Vec<i64>,
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
pub struct ConvDimensionNumbersAttr {
    pub input_batch_dimension: i64,
    pub input_feature_dimension: i64,
    pub input_spatial_dimensions: Vec<i64>,
    pub kernel_input_feature_dimension: i64,
    pub kernel_output_feature_dimension: i64,
    pub kernel_spatial_dimensions: Vec<i64>,
    pub output_batch_dimension: i64,
    pub output_feature_dimension: i64,
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
pub struct DotAlgorithmAttr {
    pub lhs_precision_type: TypeHandle,
    pub rhs_precision_type: TypeHandle,
    pub accumulation_type: TypeHandle,
    pub lhs_component_count: i64,
    pub rhs_component_count: i64,
    pub num_primitive_operations: i64,
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
pub struct OutputOperandAliasAttr {
    pub output_tuple_indices: Vec<i64>,
    pub operand_index: i64,
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

#[pliron_attr(
    name = "stablehlo.channel_handle",
    format = "`<` $handle `,` $channel_type `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChannelHandleAttr {
    pub handle: i64,
    pub channel_type: i64,
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
pub struct TypeExtensionsAttr {
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
pub struct ResultAccuracyAttr {
    pub atol_bits: u64,
    pub rtol_bits: u64,
    pub ulps: i64,
    pub mode: u8,
}
impl ResultAccuracyAttr {
    pub fn new(atol: f64, rtol: f64, ulps: i64, mode: ResultAccuracyMode) -> Self {
        Self {
            atol_bits: atol.to_bits(),
            rtol_bits: rtol.to_bits(),
            ulps,
            mode: mode as u8,
        }
    }
    pub fn atol(&self) -> f64 {
        f64::from_bits(self.atol_bits)
    }
    pub fn rtol(&self) -> f64 {
        f64::from_bits(self.rtol_bits)
    }
}
impl Verify for ResultAccuracyAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.atol().is_sign_negative()
            || self.rtol().is_sign_negative()
            || self.ulps < 0
            || self.mode > 2
        {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "result accuracy tolerances/mode are invalid"
                }
            );
        }
        if self.mode < 2 && (self.atol() != 0.0 || self.rtol() != 0.0 || self.ulps != 0) {
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
pub struct SubAxisInfoAttr {
    pub pre_size: i64,
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
pub struct AxisRefAttr {
    pub name: StringAttr,
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
pub struct MeshAxisAttr {
    pub name: StringAttr,
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
pub struct MeshAttr {
    pub axes: Vec<MeshAxisAttr>,
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
pub struct ReplicaGroupMeshAxesAttr {
    pub mesh: MeshReferenceAttr,
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
pub struct IndexTensorAttr {
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
pub struct LayoutAttr {
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
pub struct ArrayOfLayoutAttr {
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
pub struct TilingAttr {
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
pub struct ArrayOfTilingAttr {
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

#[pliron_attr(name = "stablehlo.precision", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrecisionAttr {
    pub value: u8,
}
impl Verify for PrecisionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 2 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid precision enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.comparison_direction", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComparisonDirectionAttr {
    pub value: u8,
}
impl Verify for ComparisonDirectionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 5 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid comparison_direction enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.comparison_type", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComparisonTypeAttr {
    pub value: u8,
}
impl Verify for ComparisonTypeAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 5 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid comparison_type enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.fft_type", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FftTypeAttr {
    pub value: u8,
}
impl Verify for FftTypeAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 3 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid fft_type enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.rng_algorithm", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngAlgorithmAttr {
    pub value: u8,
}
impl Verify for RngAlgorithmAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 2 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid rng_algorithm enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.rng_distribution", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngDistributionAttr {
    pub value: u8,
}
impl Verify for RngDistributionAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 1 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid rng_distribution enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(name = "stablehlo.result_accuracy_mode", format = "`<` $value `>`")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResultAccuracyModeAttr {
    pub value: u8,
}
impl Verify for ResultAccuracyModeAttr {
    fn verify(&self, _: &Context) -> Result<()> {
        if self.value > 2 {
            return verify_err!(
                Location::Unknown,
                crate::Diagnostic::InvalidAttribute {
                    attribute: "StableHLO attribute",
                    reason: "invalid result_accuracy_mode enumeration"
                }
            );
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "stablehlo.mesh_reference",
    format = "`<` `[` vec($inline, CharSpace(`,`)) `]` `,` `[` vec($symbol, CharSpace(`,`)) `]` `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MeshReferenceAttr {
    pub inline: Vec<MeshAttr>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Precision {
    Default,
    High,
    Highest,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResultAccuracyMode {
    Default,
    Highest,
    Tolerance,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComparisonDirection {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComparisonType {
    NoType,
    Float,
    TotalOrder,
    Signed,
    Unsigned,
    WeakOrder,
}
impl PrecisionAttr {
    pub fn new(value: Precision) -> Self {
        Self { value: value as u8 }
    }
}
impl ComparisonDirectionAttr {
    pub fn new(value: ComparisonDirection) -> Self {
        Self { value: value as u8 }
    }
}
impl ComparisonTypeAttr {
    pub fn new(value: ComparisonType) -> Self {
        Self { value: value as u8 }
    }
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
