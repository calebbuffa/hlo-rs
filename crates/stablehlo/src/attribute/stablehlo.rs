use std::fmt;

use crate::Type;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum StableHloAttribute {
    ScatterDimensionNumbers(ScatterDimensionNumbers),
    GatherDimensionNumbers(GatherDimensionNumbers),
    DotAlgorithm(Box<DotAlgorithm>),
    DotDimensionNumbers(DotDimensionNumbers),
    OutputOperandAlias(OutputOperandAlias),
    ChannelHandle(ChannelHandle),
    TypeExtensions(TypeExtensions),
    ConvDimensionNumbers(ConvDimensionNumbers),
    ResultAccuracy(ResultAccuracy),
    SubAxisInfo(SubAxisInfo),
    AxisRef(AxisRef),
    MeshAxis(MeshAxis),
    Mesh(Mesh),
    ReplicaGroupMeshAxes(ReplicaGroupMeshAxes),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScatterDimensionNumbers {
    pub update_window_dims: Vec<i64>,
    pub inserted_window_dims: Vec<i64>,
    pub input_batching_dims: Vec<i64>,
    pub scatter_indices_batching_dims: Vec<i64>,
    pub scatter_dims_to_operand_dims: Vec<i64>,
    pub index_vector_dim: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct GatherDimensionNumbers {
    pub offset_dims: Vec<i64>,
    pub collapsed_slice_dims: Vec<i64>,
    pub operand_batching_dims: Vec<i64>,
    pub start_indices_batching_dims: Vec<i64>,
    pub start_index_map: Vec<i64>,
    pub index_vector_dim: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DotAlgorithm {
    pub lhs_precision_type: Type,
    pub rhs_precision_type: Type,
    pub accumulation_type: Type,
    pub lhs_component_count: i64,
    pub rhs_component_count: i64,
    pub num_primitive_operations: i64,
    pub allow_imprecise_accumulation: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotDimensionNumbers {
    pub lhs_batching_dimensions: Vec<i64>,
    pub rhs_batching_dimensions: Vec<i64>,
    pub lhs_contracting_dimensions: Vec<i64>,
    pub rhs_contracting_dimensions: Vec<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    Default,
    High,
    Highest,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct OutputOperandAlias {
    pub output_tuple_indices: Vec<i64>,
    pub operand_index: i64,
    pub operand_tuple_indices: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChannelHandle {
    pub handle: i64,
    pub channel_type: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TypeExtensions {
    pub bounds: Vec<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConvDimensionNumbers {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResultAccuracyMode {
    Default,
    Highest,
    Tolerance,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResultAccuracy {
    pub atol_bits: u64,
    pub rtol_bits: u64,
    pub ulps: i64,
    pub mode: ResultAccuracyMode,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubAxisInfo {
    pub pre_size: i64,
    pub size: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AxisRef {
    pub name: String,
    pub sub_axis_info: Option<SubAxisInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshAxis {
    pub name: String,
    pub size: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mesh {
    pub axes: Vec<MeshAxis>,
    pub device_ids: Option<Vec<i64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplicaGroupMeshAxes {
    pub mesh: MeshReference,
    pub axes: Vec<AxisRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MeshReference {
    Inline(Mesh),
    Symbol(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexTensor(pub Vec<i64>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout(pub Vec<i64>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayOfLayout(pub Vec<Layout>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// A tiling is an array of rank-one index tensors, one tensor per tile.
pub struct Tiling(pub Vec<IndexTensor>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayOfTiling(pub Vec<Tiling>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttributeError {
    NegativeDimension { field: &'static str, index: usize },
    NonPositive { field: &'static str, value: i64 },
    InvalidTolerance,
    InvalidToleranceMode,
    EmptyName(&'static str),
    DuplicateMeshAxis(String),
    UnknownMeshAxis(String),
    InvalidMesh(&'static str),
    InvalidMeshAxisReference { axis: String, reason: &'static str },
    MeshTooLarge,
    InvalidDimensionBound { index: usize, bound: i64 },
    UnknownDotAlgorithm,
}

impl StableHloAttribute {
    /// Verifies local attribute constraints declared in StablehloAttrs.td and
    /// TypeInference.cpp. Shape-dependent constraints are verified with ops.
    pub fn verify(&self) -> Result<(), AttributeError> {
        match self {
            Self::ScatterDimensionNumbers(dimensions) => {
                verify_dimensions("update_window_dims", &dimensions.update_window_dims)?;
                verify_dimensions("inserted_window_dims", &dimensions.inserted_window_dims)?;
                verify_dimensions("input_batching_dims", &dimensions.input_batching_dims)?;
                verify_dimensions(
                    "scatter_indices_batching_dims",
                    &dimensions.scatter_indices_batching_dims,
                )?;
                verify_dimensions(
                    "scatter_dims_to_operand_dims",
                    &dimensions.scatter_dims_to_operand_dims,
                )?;
                verify_nonnegative("index_vector_dim", dimensions.index_vector_dim)
            }
            Self::GatherDimensionNumbers(dimensions) => {
                verify_dimensions("offset_dims", &dimensions.offset_dims)?;
                verify_dimensions("collapsed_slice_dims", &dimensions.collapsed_slice_dims)?;
                verify_dimensions("operand_batching_dims", &dimensions.operand_batching_dims)?;
                verify_dimensions(
                    "start_indices_batching_dims",
                    &dimensions.start_indices_batching_dims,
                )?;
                verify_dimensions("start_index_map", &dimensions.start_index_map)?;
                verify_nonnegative("index_vector_dim", dimensions.index_vector_dim)
            }
            Self::DotAlgorithm(algorithm) => {
                for (field, value) in [
                    ("lhs_component_count", algorithm.lhs_component_count),
                    ("rhs_component_count", algorithm.rhs_component_count),
                    (
                        "num_primitive_operations",
                        algorithm.num_primitive_operations,
                    ),
                ] {
                    verify_positive(field, value)?;
                }
                if !is_known_dot_algorithm(algorithm) {
                    return Err(AttributeError::UnknownDotAlgorithm);
                }
                Ok(())
            }
            Self::DotDimensionNumbers(dimensions) => {
                verify_dimensions(
                    "lhs_batching_dimensions",
                    &dimensions.lhs_batching_dimensions,
                )?;
                verify_dimensions(
                    "rhs_batching_dimensions",
                    &dimensions.rhs_batching_dimensions,
                )?;
                verify_dimensions(
                    "lhs_contracting_dimensions",
                    &dimensions.lhs_contracting_dimensions,
                )?;
                verify_dimensions(
                    "rhs_contracting_dimensions",
                    &dimensions.rhs_contracting_dimensions,
                )
            }
            Self::OutputOperandAlias(alias) => {
                verify_nonnegative_list("output_tuple_indices", &alias.output_tuple_indices)?;
                verify_nonnegative("operand_index", alias.operand_index)?;
                verify_nonnegative_list("operand_tuple_indices", &alias.operand_tuple_indices)
            }
            Self::TypeExtensions(extensions) => {
                for (index, bound) in extensions.bounds.iter().copied().enumerate() {
                    if bound < -1 {
                        return Err(AttributeError::InvalidDimensionBound { index, bound });
                    }
                }
                Ok(())
            }
            Self::ConvDimensionNumbers(dimensions) => {
                for (field, value) in [
                    ("input_batch_dimension", dimensions.input_batch_dimension),
                    (
                        "input_feature_dimension",
                        dimensions.input_feature_dimension,
                    ),
                    (
                        "kernel_input_feature_dimension",
                        dimensions.kernel_input_feature_dimension,
                    ),
                    (
                        "kernel_output_feature_dimension",
                        dimensions.kernel_output_feature_dimension,
                    ),
                    ("output_batch_dimension", dimensions.output_batch_dimension),
                    (
                        "output_feature_dimension",
                        dimensions.output_feature_dimension,
                    ),
                ] {
                    verify_nonnegative(field, value)?;
                }
                for (field, values) in [
                    (
                        "input_spatial_dimensions",
                        &dimensions.input_spatial_dimensions,
                    ),
                    (
                        "kernel_spatial_dimensions",
                        &dimensions.kernel_spatial_dimensions,
                    ),
                    (
                        "output_spatial_dimensions",
                        &dimensions.output_spatial_dimensions,
                    ),
                ] {
                    verify_dimensions(field, values)?;
                }
                Ok(())
            }
            Self::ResultAccuracy(accuracy) => accuracy.verify(),
            Self::SubAxisInfo(info) => {
                verify_positive("sub-axis pre_size", info.pre_size)?;
                verify_positive("sub-axis size", info.size)
            }
            Self::AxisRef(axis) => {
                if axis.name.is_empty() {
                    return Err(AttributeError::EmptyName("axis"));
                }
                if let Some(info) = &axis.sub_axis_info {
                    StableHloAttribute::SubAxisInfo(info.clone()).verify()?;
                }
                Ok(())
            }
            Self::MeshAxis(axis) => {
                if axis.name.is_empty() {
                    return Err(AttributeError::EmptyName("mesh axis"));
                }
                verify_positive("mesh axis size", axis.size)
            }
            Self::Mesh(mesh) => {
                let mut names = std::collections::HashSet::new();
                for axis in &mesh.axes {
                    StableHloAttribute::MeshAxis(axis.clone()).verify()?;
                    if !names.insert(&axis.name) {
                        return Err(AttributeError::DuplicateMeshAxis(axis.name.clone()));
                    }
                }
                Ok(())
            }
            Self::ReplicaGroupMeshAxes(replica_groups) => {
                if matches!(&replica_groups.mesh, MeshReference::Symbol(name) if name.is_empty()) {
                    return Err(AttributeError::EmptyName("mesh symbol"));
                }
                if let MeshReference::Inline(mesh) = &replica_groups.mesh {
                    StableHloAttribute::Mesh(mesh.clone()).verify()?;
                    for axis in &replica_groups.axes {
                        if !mesh
                            .axes
                            .iter()
                            .any(|mesh_axis| mesh_axis.name == axis.name)
                        {
                            return Err(AttributeError::UnknownMeshAxis(axis.name.clone()));
                        }
                    }
                }
                for axis in &replica_groups.axes {
                    StableHloAttribute::AxisRef(axis.clone()).verify()?;
                }
                Ok(())
            }
            Self::ChannelHandle(_) => Ok(()),
        }
    }
}

impl ResultAccuracy {
    pub fn new(atol: f64, rtol: f64, ulps: i64, mode: ResultAccuracyMode) -> Self {
        Self {
            atol_bits: atol.to_bits(),
            rtol_bits: rtol.to_bits(),
            ulps,
            mode,
        }
    }

    pub fn atol(&self) -> f64 {
        f64::from_bits(self.atol_bits)
    }

    pub fn rtol(&self) -> f64 {
        f64::from_bits(self.rtol_bits)
    }

    fn verify(&self) -> Result<(), AttributeError> {
        if self.atol().is_sign_negative() || self.rtol().is_sign_negative() || self.ulps < 0 {
            return Err(AttributeError::InvalidTolerance);
        }
        if matches!(
            self.mode,
            ResultAccuracyMode::Default | ResultAccuracyMode::Highest
        ) && (self.atol() != 0.0 || self.rtol() != 0.0 || self.ulps != 0)
        {
            return Err(AttributeError::InvalidToleranceMode);
        }
        Ok(())
    }
}

fn verify_dimensions(field: &'static str, dimensions: &[i64]) -> Result<(), AttributeError> {
    for (index, dimension) in dimensions.iter().copied().enumerate() {
        verify_nonnegative(field, dimension)
            .map_err(|_| AttributeError::NegativeDimension { field, index })?;
    }
    Ok(())
}

fn verify_nonnegative_list(field: &'static str, values: &[i64]) -> Result<(), AttributeError> {
    for (index, value) in values.iter().copied().enumerate() {
        verify_nonnegative(field, value)
            .map_err(|_| AttributeError::NegativeDimension { field, index })?;
    }
    Ok(())
}

fn verify_nonnegative(field: &'static str, value: i64) -> Result<(), AttributeError> {
    if value < 0 {
        return Err(AttributeError::NonPositive { field, value });
    }
    Ok(())
}

fn verify_positive(field: &'static str, value: i64) -> Result<(), AttributeError> {
    if value <= 0 {
        return Err(AttributeError::NonPositive { field, value });
    }
    Ok(())
}

impl fmt::Display for AttributeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeDimension { field, index } => {
                write!(f, "{field}[{index}] must be nonnegative")
            }
            Self::NonPositive { field, value } => {
                write!(f, "{field} has invalid value {value}")
            }
            Self::InvalidTolerance => write!(f, "result accuracy tolerances must be nonnegative"),
            Self::InvalidToleranceMode => {
                write!(
                    f,
                    "DEFAULT and HIGHEST result accuracy require zero tolerances"
                )
            }
            Self::EmptyName(field) => write!(f, "{field} name must not be empty"),
            Self::DuplicateMeshAxis(name) => {
                write!(f, "mesh has duplicate axis {name:?}")
            }
            Self::UnknownMeshAxis(name) => {
                write!(f, "mesh does not define axis {name:?}")
            }
            Self::InvalidMesh(reason) => write!(f, "invalid mesh: {reason}"),
            Self::InvalidMeshAxisReference { axis, reason } => {
                write!(f, "invalid mesh axis reference {axis:?}: {reason}")
            }
            Self::MeshTooLarge => write!(
                f,
                "mesh dimensions or group storage exceed representable limits"
            ),
            Self::InvalidDimensionBound { index, bound } => {
                write!(f, "dimension bound {index} has invalid value {bound}")
            }
            Self::UnknownDotAlgorithm => {
                write!(f, "dot algorithm type/count combination is not specified")
            }
        }
    }
}

impl std::error::Error for AttributeError {}

fn is_known_dot_algorithm(algorithm: &DotAlgorithm) -> bool {
    let (Type::Element(lhs), Type::Element(rhs), Type::Element(accumulation)) = (
        &algorithm.lhs_precision_type,
        &algorithm.rhs_precision_type,
        &algorithm.accumulation_type,
    ) else {
        return false;
    };
    if algorithm.lhs_component_count != 1 || algorithm.rhs_component_count != 1 {
        return false;
    }

    if is_dot_f8(*lhs) && is_dot_f8(*rhs) && *accumulation == crate::ElementType::F32 {
        return algorithm.num_primitive_operations == 1;
    }
    if algorithm.allow_imprecise_accumulation {
        return false;
    }

    use crate::ElementType::{Bf16, F8E4M3Fn, F16, F32, F64, Tf32};
    matches!(
        (
            *lhs,
            *rhs,
            *accumulation,
            algorithm.num_primitive_operations
        ),
        (F16, F16, F16 | F32, 1)
            | (Bf16, Bf16, Bf16 | F32, 1)
            | (Bf16, Bf16, F32, 3 | 6 | 9)
            | (F8E4M3Fn, F8E4M3Fn, F32, 3 | 4)
            | (Tf32, Tf32, F32, 1 | 3)
            | (F32, F32, F32, 1)
            | (F64, F64, F64, 1)
    )
}

fn is_dot_f8(element: crate::ElementType) -> bool {
    matches!(
        element,
        crate::ElementType::F8E3M4
            | crate::ElementType::F8E4M3
            | crate::ElementType::F8E4M3Fn
            | crate::ElementType::F8E5M2
            | crate::ElementType::F8E4M3FnUz
            | crate::ElementType::F8E4M3B11FnUz
            | crate::ElementType::F8E5M2FnUz
    )
}
