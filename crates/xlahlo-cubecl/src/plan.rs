use std::sync::Arc;

use cubecl_ir::{ElemType, settings::ExecutionMode};
use cubecl_runtime::{
    id::KernelId,
    kernel::{CubeKernel, KernelDefinition, KernelMetadata},
};
use pliron::{
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use xlahlo::types::ElementType;

/// Stable identifier for a logical device buffer in an execution plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

impl BufferId {
    /// Index in the execution plan's buffer table.
    pub fn index(self) -> usize {
        self.0
    }
}

/// The externally visible or internal role of a planned buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferRole {
    /// Caller-provided input, numbered in entry-parameter order.
    Input(usize),
    /// Scratch storage produced and consumed inside the plan.
    Temporary,
    /// Caller-visible output, numbered in result order.
    Output(usize),
}

/// Dense tensor storage contract for one planned buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferDescriptor {
    pub element_type: ElementType,
    pub shape: Vec<i64>,
    pub layout: BufferLayout,
}

impl BufferDescriptor {
    pub fn dense_row_major(element_type: ElementType, shape: Vec<i64>) -> Self {
        Self {
            element_type,
            shape,
            layout: BufferLayout::DenseRowMajor,
        }
    }
}

/// Layout contract supported by the current plan ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferLayout {
    /// Contiguous row-major storage with descending logical axes.
    DenseRowMajor,
}

impl BufferDescriptor {
    /// Number of logical elements in the tensor.
    pub fn num_elements(&self) -> Result<usize, PlanError> {
        if self.shape.iter().any(|&extent| extent < 0) {
            return Err(PlanError::InvalidShape);
        }
        self.shape.iter().try_fold(1usize, |count, &extent| {
            count
                .checked_mul(usize::try_from(extent).map_err(|_| PlanError::InvalidShape)?)
                .ok_or(PlanError::SizeOverflow)
        })
    }

    /// Number of storage bytes required by the dense tensor.
    pub fn byte_size(&self) -> Result<usize, PlanError> {
        self.num_elements()?
            .checked_mul(self.element_type.storage_bit_width().div_ceil(8) as usize)
            .ok_or(PlanError::SizeOverflow)
    }

    pub fn cubecl_element_type(&self) -> Result<ElemType, PlanError> {
        match self.element_type {
            ElementType::F4E2M1FN => Ok(ElemType::Float(cubecl_ir::FloatKind::E2M1)),
            ElementType::F6E3M2FN => Ok(ElemType::Float(cubecl_ir::FloatKind::E3M2)),
            ElementType::F6E2M3FN => Ok(ElemType::Float(cubecl_ir::FloatKind::E2M3)),
            ElementType::F8E4M3 | ElementType::F8E4M3FN => {
                Ok(ElemType::Float(cubecl_ir::FloatKind::E4M3))
            }
            ElementType::F8E5M2 => Ok(ElemType::Float(cubecl_ir::FloatKind::E5M2)),
            ElementType::F8E8M0FNU => Ok(ElemType::Float(cubecl_ir::FloatKind::UE8M0)),
            ElementType::F16 => Ok(ElemType::Float(cubecl_ir::FloatKind::F16)),
            ElementType::BF16 => Ok(ElemType::Float(cubecl_ir::FloatKind::BF16)),
            ElementType::F32 => Ok(ElemType::Float(cubecl_ir::FloatKind::F32)),
            ElementType::F64 => Ok(ElemType::Float(cubecl_ir::FloatKind::F64)),
            ElementType::S8 => Ok(ElemType::Int(cubecl_ir::IntKind::I8)),
            ElementType::S16 => Ok(ElemType::Int(cubecl_ir::IntKind::I16)),
            ElementType::S32 => Ok(ElemType::Int(cubecl_ir::IntKind::I32)),
            ElementType::S64 => Ok(ElemType::Int(cubecl_ir::IntKind::I64)),
            ElementType::U1 | ElementType::U8 | ElementType::Pred => {
                Ok(ElemType::UInt(cubecl_ir::UIntKind::U8))
            }
            ElementType::U16 => Ok(ElemType::UInt(cubecl_ir::UIntKind::U16)),
            ElementType::U32 => Ok(ElemType::UInt(cubecl_ir::UIntKind::U32)),
            ElementType::U64 => Ok(ElemType::UInt(cubecl_ir::UIntKind::U64)),
            _ => Err(PlanError::UnsupportedElementType(self.element_type)),
        }
    }
}

/// Read/write behavior for one ordered kernel argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
}

/// One kernel argument, in exact CubeCL registration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelBinding {
    pub buffer: BufferId,
    pub access: BufferAccess,
}

/// Per-kernel launch geometry. `cube_count: None` explicitly means no launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchGeometry {
    pub cube_dim: [u32; 3],
    pub cube_count: Option<[u32; 3]>,
    pub skip_reason: Option<SkipReason>,
}

impl LaunchGeometry {
    pub fn dispatch(cube_dim: [u32; 3], cube_count: [u32; 3]) -> Result<Self, PlanError> {
        if cube_dim.contains(&0) || cube_count.contains(&0) {
            return Err(PlanError::ZeroLaunchDimension);
        }
        Ok(Self {
            cube_dim,
            cube_count: Some(cube_count),
            skip_reason: None,
        })
    }

    pub fn skip(reason: SkipReason) -> Self {
        Self {
            cube_dim: [1, 1, 1],
            cube_count: None,
            skip_reason: Some(reason),
        }
    }

    pub fn for_elements(elements: usize, cube_dim: [u32; 3]) -> Result<Self, PlanError> {
        if cube_dim.contains(&0) {
            return Err(PlanError::ZeroLaunchDimension);
        }
        if elements == 0 {
            return Ok(Self::skip(SkipReason::EmptyOutput));
        }
        let x_cubes = elements.div_ceil(cube_dim[0] as usize);
        let x_cubes = u32::try_from(x_cubes).map_err(|_| PlanError::SizeOverflow)?;
        Self::dispatch(cube_dim, [x_cubes, 1, 1])
    }
}

/// Why a kernel step is intentionally not dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkipReason {
    EmptyOutput,
    StaticOnly,
    NoWork,
}

/// Inclusive kernel-step interval during which a buffer must remain live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferLifetime {
    pub first_kernel: usize,
    pub last_kernel: usize,
    /// Keep the allocation available after the final kernel for a returned output/alias.
    pub retained_for_output: bool,
}

/// Immutable output value returned by a compiled module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PlannedOutput {
    /// A newly produced output buffer.
    Buffer(BufferId),
    /// An explicitly declared alias of an existing input or planned buffer.
    Alias(BufferId),
    /// Logical view over an existing retained physical allocation.
    View(crate::OutputView),
    /// A compile-time value requiring no device allocation or launch.
    Static(StaticValue),
}

/// Raw bytes and tensor metadata for a static result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StaticValue {
    pub descriptor: BufferDescriptor,
    pub bytes: Vec<u8>,
}

/// A Pliron transformation over CubeCL IR, run in registered priority order.
///
/// The root is the CubeCL kernel module inside `KernelDefinition::body`.
pub trait CubeclTargetPass: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn priority(&self) -> i32;
    /// Stable identity for configuration that changes the transformation result.
    fn cache_key(&self) -> Vec<u8>;
    fn run(&self, context: &mut Context, root: Ptr<Operation>) -> Result<(), String>;
}

/// A kernel implementation whose Pliron definition can fail explicitly.
pub trait PlannedKernel: KernelMetadata {
    fn define_checked(&self) -> Result<KernelDefinition, String>;
}

/// One scheduled CubeCL kernel with explicit ABI and launch contract.
pub struct KernelStep {
    name: String,
    kernel: Arc<dyn PlannedKernel>,
    bindings: Vec<KernelBinding>,
    launch: LaunchGeometry,
    side_effecting: bool,
    passes: Arc<[Arc<dyn CubeclTargetPass>]>,
}

impl KernelStep {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kernel(&self) -> &dyn CubeKernel {
        self
    }

    pub fn bindings(&self) -> &[KernelBinding] {
        &self.bindings
    }

    pub fn launch(&self) -> LaunchGeometry {
        self.launch
    }

    pub fn side_effecting(&self) -> bool {
        self.side_effecting
    }
}

impl KernelMetadata for KernelStep {
    fn name(&self) -> &'static str {
        "xlahlo_cubecl_plan_step"
    }

    fn id(&self) -> KernelId {
        let pass_signature: Vec<_> = self
            .passes
            .iter()
            .map(|pass| (pass.name(), pass.priority(), pass.cache_key()))
            .collect();
        KernelId::new::<Self>().info((
            self.kernel.id(),
            self.name.clone(),
            self.launch,
            self.bindings.clone(),
            pass_signature,
        ))
    }

    fn address_type(&self) -> ElemType {
        self.kernel.address_type()
    }
}

impl CubeKernel for KernelStep {
    fn define(&self) -> KernelDefinition {
        self.define_checked()
            .unwrap_or_else(|error| panic!("kernel step {} failed: {error}", self.name))
    }
}

impl KernelStep {
    /// Define the target kernel, run target passes, and verify its ABI and IR.
    pub fn define_checked(&self) -> Result<KernelDefinition, KernelDefinitionError> {
        let definition =
            self.kernel
                .define_checked()
                .map_err(|message| KernelDefinitionError::Kernel {
                    step: self.name.clone(),
                    message,
                })?;
        let root = definition.body.state().module.get_operation();
        for pass in self.passes.iter() {
            pass.run(definition.body.ctx_mut(), root)
                .map_err(|message| KernelDefinitionError::Pass {
                    step: self.name.clone(),
                    pass: pass.name(),
                    priority: pass.priority(),
                    message,
                })?;
        }
        if definition.info.metadata.num_meta() != self.bindings.len()
            || !definition.info.scalars.is_empty()
            || definition.info.has_dynamic_meta
        {
            return Err(KernelDefinitionError::BindingAbi {
                step: self.name.clone(),
                actual_buffers: definition.info.metadata.num_meta(),
                actual_scalars: definition.info.scalars.len(),
                expected_buffers: self.bindings.len(),
                dynamic_metadata: definition.info.has_dynamic_meta,
            });
        }
        if self.launch.cube_count.is_some() {
            if definition.settings.execution_mode != ExecutionMode::Unchecked {
                return Err(KernelDefinitionError::ExecutionMode {
                    step: self.name.clone(),
                });
            }
            let expected = cubecl_ir::settings::Dim3::new_3d(
                self.launch.cube_dim[0],
                self.launch.cube_dim[1],
                self.launch.cube_dim[2],
            );
            if definition.settings.cube_dim != expected {
                return Err(KernelDefinitionError::LaunchGeometry {
                    step: self.name.clone(),
                    expected: self.launch.cube_dim,
                    actual: definition.settings.cube_dim.into(),
                });
            }
            if definition.settings.address_type.unsigned_type() != self.kernel.address_type() {
                return Err(KernelDefinitionError::AddressType {
                    step: self.name.clone(),
                });
            }
        }
        let context = definition.body.ctx_mut();
        let mut pending = vec![root];
        let mut operations = Vec::new();
        while let Some(operation) = pending.pop() {
            operations.push(operation);
            for region in operation.deref(context).regions() {
                for block in region.deref(context).iter(context) {
                    pending.extend(block.deref(context).iter(context));
                }
            }
        }
        for operation in operations.into_iter().rev() {
            pliron::op::verify_op(&*Operation::get_op_dyn(operation, context), context).map_err(
                |error| KernelDefinitionError::InvalidTargetIr {
                    step: self.name.clone(),
                    operation: Operation::get_opid(operation, context).to_string(),
                    message: error.to_string(),
                },
            )?;
        }
        Ok(definition)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KernelDefinitionError {
    #[error("defining kernel step {step} failed: {message}")]
    Kernel { step: String, message: String },
    #[error("CubeCL target pass {pass} at priority {priority} failed in {step}: {message}")]
    Pass {
        step: String,
        pass: &'static str,
        priority: i32,
        message: String,
    },
    #[error(
        "kernel step {step} exposes {actual_buffers} buffers and {actual_scalars} scalar fields, but the plan declares {expected_buffers} buffer bindings (dynamic metadata: {dynamic_metadata})"
    )]
    BindingAbi {
        step: String,
        actual_buffers: usize,
        actual_scalars: usize,
        expected_buffers: usize,
        dynamic_metadata: bool,
    },
    #[error(
        "kernel step {step} launch dimension mismatch: expected {expected:?}, actual {actual:?}"
    )]
    LaunchGeometry {
        step: String,
        expected: [u32; 3],
        actual: (u32, u32, u32),
    },
    #[error("kernel step {step} must use unchecked execution mode")]
    ExecutionMode { step: String },
    #[error("kernel step {step} address type disagrees with KernelMetadata")]
    AddressType { step: String },
    #[error("kernel step {step} generated invalid CubeCL IR at {operation}: {message}")]
    InvalidTargetIr {
        step: String,
        operation: String,
        message: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("tensor shape contains a negative extent")]
    InvalidShape,
    #[error("tensor element count or byte size overflows host addressing")]
    SizeOverflow,
    #[error("element type {0:?} has no scalar CubeCL buffer representation")]
    UnsupportedElementType(ElementType),
    #[error("CubeCL launch dimensions must all be nonzero")]
    ZeroLaunchDimension,
    #[error("kernel step {kernel} refers to unknown buffer {buffer}")]
    UnknownBuffer { kernel: usize, buffer: usize },
    #[error("buffer {buffer} has conflicting roles")]
    ConflictingBufferRole { buffer: usize },
    #[error("input buffer indices must be unique and contiguous from zero")]
    InvalidInputOrder,
    #[error("output buffer indices must be unique and match result positions")]
    InvalidOutputOrder,
    #[error("buffer {buffer} is written by multiple kernel steps")]
    MultipleProducers { buffer: usize },
    #[error("buffer {buffer} is read before it is produced")]
    ReadBeforeWrite { buffer: usize },
    #[error("buffer {buffer} has no producer")]
    MissingProducer { buffer: usize },
    #[error("buffer {buffer} is read-only input but a kernel writes it")]
    WriteToInput { buffer: usize },
    #[error("output refers to unknown buffer {0}")]
    UnknownOutputBuffer(usize),
    #[error("output buffer {0} is not produced by the plan")]
    UnproducedOutput(usize),
    #[error("static output has {actual} bytes, expected {expected}")]
    InvalidStaticSize { expected: usize, actual: usize },
    #[error("a skipped kernel cannot declare side effects")]
    SkippedSideEffect,
    #[error("kernel step names must be unique")]
    DuplicateKernelName,
    #[error("lowering does not support {0}")]
    UnsupportedFeature(String),
    #[error(
        "operation strategy {strategy} requires dense operands and does not support logical indexing"
    )]
    IndexedOperandsUnsupported { strategy: String },
    #[error("operation lowering strategy failed: {0}")]
    OperationStrategy(String),
    #[error("lowering failed: {0}")]
    LoweringFailure(String),
    #[error("kernel launch geometry and skip reason are inconsistent")]
    InvalidLaunchContract,
    #[error("empty-output kernel must bind only zero-element write targets")]
    InvalidEmptyLaunch,
}

struct BufferEntry {
    role: BufferRole,
    descriptor: BufferDescriptor,
    lifetime: Option<BufferLifetime>,
}

/// Builder for a validated, caller-executable module schedule.
pub struct ExecutionPlanBuilder {
    buffers: Vec<BufferEntry>,
    kernels: Vec<KernelStep>,
    outputs: Vec<PlannedOutput>,
    passes: Arc<[Arc<dyn CubeclTargetPass>]>,
}

impl ExecutionPlanBuilder {
    pub(crate) fn new(passes: Vec<Arc<dyn CubeclTargetPass>>) -> Self {
        Self {
            buffers: Vec::new(),
            kernels: Vec::new(),
            outputs: Vec::new(),
            passes: passes.into(),
        }
    }

    pub fn add_buffer(&mut self, role: BufferRole, descriptor: BufferDescriptor) -> BufferId {
        let id = BufferId(self.buffers.len());
        self.buffers.push(BufferEntry {
            role,
            descriptor,
            lifetime: None,
        });
        id
    }

    pub fn add_kernel(
        &mut self,
        name: impl Into<String>,
        kernel: Arc<dyn PlannedKernel>,
        bindings: Vec<KernelBinding>,
        launch: LaunchGeometry,
        side_effecting: bool,
    ) -> Result<(), PlanError> {
        let name = name.into();
        if name.trim().is_empty() || self.kernels.iter().any(|step| step.name == name) {
            return Err(PlanError::DuplicateKernelName);
        }
        if launch.cube_count.is_none() && side_effecting {
            return Err(PlanError::SkippedSideEffect);
        }
        self.kernels.push(KernelStep {
            name,
            kernel,
            bindings,
            launch,
            side_effecting,
            passes: self.passes.clone(),
        });
        Ok(())
    }

    pub fn add_output(&mut self, output: PlannedOutput) {
        self.outputs.push(output);
    }

    pub fn build(mut self) -> Result<ExecutionPlan, PlanError> {
        let mut input_indices = Vec::new();
        let mut output_indices = Vec::new();
        for (index, entry) in self.buffers.iter().enumerate() {
            match entry.role {
                BufferRole::Input(input) => input_indices.push((input, index)),
                BufferRole::Output(output) => output_indices.push((output, index)),
                BufferRole::Temporary => {}
            }
            entry.descriptor.byte_size()?;
        }
        input_indices.sort_unstable();
        output_indices.sort_unstable();
        if input_indices
            .iter()
            .enumerate()
            .any(|(expected, &(actual, _))| expected != actual)
        {
            return Err(PlanError::InvalidInputOrder);
        }
        if output_indices.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(PlanError::InvalidOutputOrder);
        }

        let mut producers = vec![None; self.buffers.len()];
        for (kernel_index, step) in self.kernels.iter().enumerate() {
            match (step.launch.cube_count, step.launch.skip_reason) {
                (Some(cubes), None) if cubes.iter().all(|&count| count > 0) => {}
                (None, Some(SkipReason::EmptyOutput)) => {
                    let has_empty_output = step.bindings.iter().any(|binding| {
                        binding.access == BufferAccess::Write
                            && self
                                .buffers
                                .get(binding.buffer.index())
                                .is_some_and(|buffer| {
                                    matches!(buffer.descriptor.num_elements(), Ok(0))
                                })
                    });
                    let invalid_write = step.bindings.iter().any(|binding| {
                        matches!(
                            binding.access,
                            BufferAccess::Write | BufferAccess::ReadWrite
                        ) && self
                            .buffers
                            .get(binding.buffer.index())
                            .is_some_and(|buffer| {
                                !matches!(buffer.descriptor.num_elements(), Ok(0))
                            })
                    });
                    if !has_empty_output || invalid_write {
                        return Err(PlanError::InvalidEmptyLaunch);
                    }
                }
                (None, Some(SkipReason::StaticOnly | SkipReason::NoWork))
                    if !step.side_effecting
                        && step.bindings.iter().all(|binding| {
                            !matches!(
                                binding.access,
                                BufferAccess::Write | BufferAccess::ReadWrite
                            )
                        }) => {}
                _ => return Err(PlanError::InvalidLaunchContract),
            }
            for binding in &step.bindings {
                let buffer_index = binding.buffer.index();
                let Some(entry) = self.buffers.get(buffer_index) else {
                    return Err(PlanError::UnknownBuffer {
                        kernel: kernel_index,
                        buffer: buffer_index,
                    });
                };
                if matches!(entry.role, BufferRole::Input(_))
                    && matches!(
                        binding.access,
                        BufferAccess::Write | BufferAccess::ReadWrite
                    )
                {
                    return Err(PlanError::WriteToInput {
                        buffer: buffer_index,
                    });
                }
                if matches!(
                    binding.access,
                    BufferAccess::Write | BufferAccess::ReadWrite
                ) && producers[buffer_index].replace(kernel_index).is_some()
                {
                    return Err(PlanError::MultipleProducers {
                        buffer: buffer_index,
                    });
                }
                let lifetime = self.buffers[buffer_index]
                    .lifetime
                    .get_or_insert(BufferLifetime {
                        first_kernel: kernel_index,
                        last_kernel: kernel_index,
                        retained_for_output: false,
                    });
                lifetime.last_kernel = kernel_index;
            }
        }
        for (buffer_index, entry) in self.buffers.iter().enumerate() {
            match entry.role {
                BufferRole::Input(_) => {}
                BufferRole::Temporary | BufferRole::Output(_) => {
                    let producer = producers[buffer_index].ok_or(PlanError::MissingProducer {
                        buffer: buffer_index,
                    })?;
                    let first_access =
                        self.kernels
                            .iter()
                            .enumerate()
                            .find_map(|(kernel_index, step)| {
                                step.bindings
                                    .iter()
                                    .find(|binding| binding.buffer.index() == buffer_index)
                                    .map(|binding| (kernel_index, binding.access))
                            });
                    if first_access.is_some_and(|(kernel_index, access)| {
                        kernel_index == producer && access != BufferAccess::Write
                    }) {
                        return Err(PlanError::ReadBeforeWrite {
                            buffer: buffer_index,
                        });
                    }
                    if first_access.is_some_and(|(kernel_index, access)| {
                        kernel_index < producer
                            && matches!(access, BufferAccess::Read | BufferAccess::ReadWrite)
                    }) {
                        return Err(PlanError::ReadBeforeWrite {
                            buffer: buffer_index,
                        });
                    }
                }
            }
        }
        for &(output_index, buffer_index) in &output_indices {
            if !matches!(
                self.outputs.get(output_index),
                Some(PlannedOutput::Buffer(id)) if id.index() == buffer_index
            ) {
                return Err(PlanError::InvalidOutputOrder);
            }
        }
        for (output_index, output) in self.outputs.iter().enumerate() {
            match output {
                PlannedOutput::Buffer(id) => {
                    let Some(entry) = self.buffers.get(id.index()) else {
                        return Err(PlanError::UnknownOutputBuffer(id.index()));
                    };
                    if matches!(entry.role, BufferRole::Input(_)) || producers[id.index()].is_none()
                    {
                        return Err(PlanError::UnproducedOutput(id.index()));
                    }
                    if entry.role != BufferRole::Output(output_index) {
                        return Err(PlanError::InvalidOutputOrder);
                    }
                }
                PlannedOutput::Alias(id) => {
                    if self.buffers.get(id.index()).is_none() {
                        return Err(PlanError::UnknownOutputBuffer(id.index()));
                    }
                }
                PlannedOutput::View(view) => {
                    let entry = self
                        .buffers
                        .get(view.buffer.index())
                        .ok_or(PlanError::UnknownOutputBuffer(view.buffer.index()))?;
                    if !matches!(entry.role, BufferRole::Input(_))
                        && producers[view.buffer.index()].is_none()
                    {
                        return Err(PlanError::UnproducedOutput(view.buffer.index()));
                    }
                    view.validate(&entry.descriptor)?;
                }
                PlannedOutput::Static(value) => {
                    let expected = value.descriptor.byte_size()?;
                    if expected != value.bytes.len() {
                        return Err(PlanError::InvalidStaticSize {
                            expected,
                            actual: value.bytes.len(),
                        });
                    }
                }
            }
        }
        let mut retained = vec![false; self.buffers.len()];
        for output in &self.outputs {
            if let PlannedOutput::Buffer(id) | PlannedOutput::Alias(id) = output {
                retained[id.index()] = true;
            } else if let PlannedOutput::View(view) = output {
                retained[view.buffer.index()] = true;
            }
        }
        for (index, buffer) in self.buffers.iter_mut().enumerate() {
            if let Some(lifetime) = &mut buffer.lifetime {
                lifetime.retained_for_output = retained[index];
            }
        }
        Ok(ExecutionPlan {
            buffers: self
                .buffers
                .into_iter()
                .map(|entry| PlannedBuffer {
                    role: entry.role,
                    descriptor: entry.descriptor,
                    lifetime: entry.lifetime,
                })
                .collect(),
            kernels: self.kernels,
            outputs: self.outputs,
        })
    }
}

/// A complete, immutable schedule. It owns no device allocations or runtime.
pub struct ExecutionPlan {
    buffers: Vec<PlannedBuffer>,
    kernels: Vec<KernelStep>,
    outputs: Vec<PlannedOutput>,
}

impl ExecutionPlan {
    pub fn buffers(&self) -> &[PlannedBuffer] {
        &self.buffers
    }

    pub fn kernels(&self) -> &[KernelStep] {
        &self.kernels
    }

    pub fn outputs(&self) -> &[PlannedOutput] {
        &self.outputs
    }

    pub fn has_device_work(&self) -> bool {
        self.kernels
            .iter()
            .any(|kernel| kernel.launch.cube_count.is_some())
    }

    /// Preflight every scheduled kernel definition and registered target pass.
    pub fn validate_kernel_definitions(&self) -> Result<(), KernelDefinitionError> {
        for step in &self.kernels {
            if step.launch.cube_count.is_some() {
                step.define_checked()?;
            }
        }
        Ok(())
    }
}

/// Validated buffer metadata owned by an execution plan.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlannedBuffer {
    pub role: BufferRole,
    pub descriptor: BufferDescriptor,
    pub lifetime: Option<BufferLifetime>,
}
