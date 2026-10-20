//! Owned module proto metadata, independent of execution configuration.
use crate::{
    attributes::{Bytes, Payload},
    original_value::OriginalArray,
    sharding::Sharding,
    verification::{optional_attr, require},
};
use pliron::{
    attribute::Attribute,
    builtin::ops::ModuleOp,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_attr,
    identifier::Identifier,
    irfmt::parsers::spaced,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    result::Result,
};
use std::collections::HashSet;

#[pliron_attr(
    name = "xlahlo.cross_program_prefetch",
    format = "`<` $parameter `,` `[` vec($index, CharSpace(`,`)) `]` `,` $offset `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// An entry parameter subshape prefetched to an alternate-memory byte offset.
pub struct CrossProgramPrefetch {
    pub parameter: i64,
    pub index: Vec<i64>,
    /// Proto3's absent offset is zero.
    pub offset: i64,
}
impl Verify for CrossProgramPrefetch {
    fn verify(&self, _: &Context) -> Result<()> {
        require(
            self.parameter >= 0 && self.index.iter().all(|&i| i >= 0),
            "invalid cross-program prefetch parameter or shape index",
        )
    }
}

#[pliron_attr(name = "xlahlo.module_profile_info", format, verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// HloModuleProto.ProfileInfo, NOT OpMetadata.ProfileInfo.
///
/// Enum numbers (including unknown future values) and IEEE f64 bits are exact.
/// Profile types: 0 invalid, 1 flag, 2 fusion, 3 layout, 4 dot, 5 flagnet,
/// 6 sharding, 7 schedule. Other enums follow xla_data.proto.
pub struct ModuleProfileInfo {
    pub profile_type: i32,
    pub relative_speedup_bits: u64,
    pub profile_source: i32,
    pub compilation_event: i32,
    pub fingerprint: String,
    pub profile_generation_strategy: i32,
    pub original_changelist: i64,
    pub changelist: i64,
}

#[pliron_attr(
    name = "xlahlo.computation_device",
    format = "`[` vec($replica_device_ids, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Physical device IDs in replica order for one logical computation.
pub struct ComputationDevice {
    pub replica_device_ids: Vec<i64>,
}

#[pliron_attr(
    name = "xlahlo.device_assignment",
    format = "`<` $replica_count `,` $computation_count `,` `[` vec($computation_devices, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Computation-major DeviceAssignmentProto geometry.
pub struct DeviceAssignment {
    pub replica_count: i32,
    pub computation_count: i32,
    pub computation_devices: Vec<ComputationDevice>,
}
impl Verify for DeviceAssignment {
    fn verify(&self, _: &Context) -> Result<()> {
        require(
            self.replica_count > 0
                && self.computation_count > 0
                && usize::try_from(self.computation_count) == Ok(self.computation_devices.len())
                && self
                    .computation_devices
                    .iter()
                    .all(|c| usize::try_from(self.replica_count) == Ok(c.replica_device_ids.len())),
            "device assignment dimensions must be positive and match device lists",
        )
    }
}

#[pliron_attr(name = "xlahlo.file_location", format, verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Stack-frame source location; name IDs must be valid 1-based references.
pub struct FileLocation {
    pub file_name_id: i32,
    pub function_name_id: i32,
    pub line: i32,
    pub end_line: i32,
    pub column: i32,
    pub end_column: i32,
}

#[pliron_attr(name = "xlahlo.stack_frame", format, verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Stack-frame location and parent IDs; only the parent may be zero (no parent).
pub struct StackFrame {
    pub file_location_id: i32,
    pub parent_frame_id: i32,
}

#[pliron_attr(
    name = "xlahlo.stack_frame_index",
    format = "`<` `[` vec($file_names, CharSpace(`,`)) `]` `,` `[` vec($function_names, CharSpace(`,`)) `]` `,` `[` vec($file_locations, CharSpace(`,`)) `]` `,` `[` vec($stack_frames, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Flat stack-frame index with checked references and acyclic parent links.
pub struct StackFrameIndex {
    pub file_names: Vec<String>,
    pub function_names: Vec<String>,
    pub file_locations: Vec<FileLocation>,
    pub stack_frames: Vec<StackFrame>,
}
impl Verify for StackFrameIndex {
    fn verify(&self, _: &Context) -> Result<()> {
        let valid = |id: i32, len| id > 0 && usize::try_from(id).is_ok_and(|id| id <= len);
        for location in &self.file_locations {
            require(
                valid(location.file_name_id, self.file_names.len())
                    && valid(location.function_name_id, self.function_names.len()),
                "stack file/function name ID out of range",
            )?;
        }
        for frame in &self.stack_frames {
            require(
                valid(frame.file_location_id, self.file_locations.len())
                    && (frame.parent_frame_id == 0
                        || valid(frame.parent_frame_id, self.stack_frames.len())),
                "stack location/parent frame ID out of range",
            )?;
        }
        require(
            self.file_names.iter().collect::<HashSet<_>>().len() == self.file_names.len()
                && self.function_names.iter().collect::<HashSet<_>>().len()
                    == self.function_names.len()
                && self.file_locations.iter().collect::<HashSet<_>>().len()
                    == self.file_locations.len()
                && self.stack_frames.iter().collect::<HashSet<_>>().len()
                    == self.stack_frames.len(),
            "duplicate stack-frame index entry",
        )?;
        let mut colors = vec![0u8; self.stack_frames.len()];
        for start in 0..colors.len() {
            if colors[start] != 0 {
                continue;
            }
            let mut path = Vec::new();
            let mut next = Some(start);
            while let Some(i) = next {
                require(colors[i] != 1, "cycle in stack frame parents")?;
                if colors[i] == 2 {
                    break;
                }
                colors[i] = 1;
                path.push(i);
                next = usize::try_from(self.stack_frames[i].parent_frame_id)
                    .ok()
                    .and_then(|id| id.checked_sub(1));
            }
            for i in path {
                colors[i] = 2;
            }
        }
        Ok(())
    }
}

#[pliron_attr(name = "xlahlo.x64_config", format, verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Debug-only x64 reconstruction strings, preserved even when not interpreted.
pub struct X64Config {
    pub xprecision_64bit_fragment: String,
    pub original_log_instruction_name: String,
    pub xprecision_64bit_original_type: String,
}

#[pliron_attr(
    name = "xlahlo.debug_attributes",
    format = "`<` $log_mode `,` $callback_id `,` $op_id `,` $partitioned `,` $operands_sharding `,` opt($x64_config) `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// DebugAttributesProto, distinct from ModuleConfig.debug_options.
pub struct DebugAttributes {
    /// 0 none, 1 default, 2 fusion debugger, 3 no fusion, 4 guaranteed.
    /// Unknown enum numbers are retained, not silently converted to none.
    pub log_mode: i32,
    pub callback_id: i64,
    pub op_id: i64,
    pub partitioned: bool,
    pub operands_sharding: String,
    pub x64_config: Option<X64Config>,
}

#[pliron_attr(
    name = "xlahlo.debug_attribute_table_entry",
    format = "`<` opt($original_array) `,` `[` vec($debug_attributes, CharSpace(`,`)) `]` `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Debug attributes for an original array; preserves proto message presence.
pub struct DebugAttributeTableEntry {
    pub original_array: Option<OriginalArray>,
    pub debug_attributes: Vec<DebugAttributes>,
}
fn verify_original(array: Option<&OriginalArray>) -> Result<()> {
    require(
        array.is_none_or(|a| a.shape_index.iter().all(|&i| i >= 0)),
        "negative original array shape index",
    )
}
impl Verify for DebugAttributeTableEntry {
    fn verify(&self, _: &Context) -> Result<()> {
        verify_original(self.original_array.as_ref())
    }
}

#[pliron_attr(name = "xlahlo.recovery_module", format = "$0")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Immutable native module assembly snapshot, deep-copied on materialization.
///
/// This is native assembly, not serialized HloModuleProto bytes.
pub struct RecoveryModule(pub String);
impl RecoveryModule {
    /// Capture a verified builtin module without retaining mutable IR handles.
    pub fn capture(ctx: &Context, module: Ptr<Operation>) -> Result<Self> {
        require(
            Operation::is_op::<ModuleOp>(module, ctx),
            "recovery module requires builtin.module",
        )?;
        crate::verification::verify(ctx, module)?;
        Ok(Self(module.disp(ctx).to_string()))
    }

    /// Parse a fresh module into a registered destination context and verify it.
    pub fn materialize(&self, ctx: &mut Context) -> Result<Ptr<Operation>> {
        let module = parse_from_str(spaced(Operation::top_level_parser()), ctx, &self.0)?;
        require(
            Operation::is_op::<ModuleOp>(module, ctx),
            "recovery module requires builtin.module",
        )?;
        crate::verification::verify(ctx, module)?;
        Ok(module)
    }
}
impl Verify for RecoveryModule {
    fn verify(&self, _: &Context) -> Result<()> {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        self.materialize(&mut ctx).map(|_| ())
    }
}

#[pliron_attr(
    name = "xlahlo.original_value_recovery_entry",
    format = "`<` opt($old_original_array) `,` opt($new_original_array) `,` opt($recovery_module) `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Recovery mapping and optional recursively owned recovery graph.
pub struct OriginalValueRecoveryEntry {
    pub old_original_array: Option<OriginalArray>,
    pub new_original_array: Option<OriginalArray>,
    pub recovery_module: Option<RecoveryModule>,
}
impl Verify for OriginalValueRecoveryEntry {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_original(self.old_original_array.as_ref())?;
        verify_original(self.new_original_array.as_ref())?;
        if let Some(module) = &self.recovery_module {
            module.verify(ctx)?;
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "xlahlo.original_value_recovery_table",
    format = "`[` vec($0, CharSpace(`,`)) `]`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Ordered recovery entries; no optimizer or recovery execution is implied.
pub struct OriginalValueRecoveryTable(pub Vec<OriginalValueRecoveryEntry>);
impl Verify for OriginalValueRecoveryTable {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for entry in &self.0 {
            entry.verify(ctx)?;
        }
        Ok(())
    }
}

#[pliron_attr(
    name = "xlahlo.module_metadata",
    format = "`<` `[` vec($cross_program_prefetches, CharSpace(`,`)) `]` `,` opt($spmd_output_sharding) `,` `[` vec($profile_info, CharSpace(`,`)) `]` `,` `[` vec($spmd_parameters_shardings, CharSpace(`,`)) `]` `,` opt($device_assignment) `,` $use_auto_spmd_partitioning `,` opt($stack_frame_index) `,` opt($original_value_recovery_table) `,` $device_type `,` `[` vec($debug_attributes, CharSpace(`,`)) `]` `,` $backend_config `>`"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Additive HloModuleProto metadata attached under `hlo_module_metadata`.
pub struct ModuleMetadata {
    pub cross_program_prefetches: Vec<CrossProgramPrefetch>,
    pub spmd_output_sharding: Option<Sharding>,
    pub profile_info: Vec<ModuleProfileInfo>,
    pub spmd_parameters_shardings: Vec<Sharding>,
    pub device_assignment: Option<DeviceAssignment>,
    pub use_auto_spmd_partitioning: bool,
    pub stack_frame_index: Option<StackFrameIndex>,
    pub original_value_recovery_table: Option<OriginalValueRecoveryTable>,
    pub device_type: String,
    pub debug_attributes: Vec<DebugAttributeTableEntry>,
    pub backend_config: Bytes,
}
impl Verify for ModuleMetadata {
    fn verify(&self, ctx: &Context) -> Result<()> {
        for prefetch in &self.cross_program_prefetches {
            prefetch.verify(ctx)?;
        }
        for sharding in self
            .spmd_output_sharding
            .iter()
            .chain(&self.spmd_parameters_shardings)
        {
            sharding.verify(ctx)?;
        }
        if let Some(assignment) = &self.device_assignment {
            assignment.verify(ctx)?;
        }
        if let Some(index) = &self.stack_frame_index {
            index.verify(ctx)?;
        }
        if let Some(table) = &self.original_value_recovery_table {
            table.verify(ctx)?;
        }
        for entry in &self.debug_attributes {
            entry.verify(ctx)?;
        }
        Ok(())
    }
}

/// Attach verified metadata to a builtin module (entry-shape checks run at graph verification).
pub fn set_metadata(ctx: &Context, module: Ptr<Operation>, metadata: ModuleMetadata) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "metadata requires builtin.module",
    )?;
    metadata.verify(ctx)?;
    module
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_module_metadata"), metadata);
    Ok(())
}

/// Read an owned metadata snapshot, preserving absence as the proto defaults.
pub fn metadata(ctx: &Context, module: Ptr<Operation>) -> Result<ModuleMetadata> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "metadata requires builtin.module",
    )?;
    Ok(
        optional_attr::<ModuleMetadata>(ctx, module, "hlo_module_metadata")?
            .map(|metadata| metadata.clone())
            .unwrap_or_default(),
    )
}

#[pliron_attr(
    name = "xlahlo.backend_config_payload",
    format = "$0",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Instruction field 99; presence takes precedence over legacy backend_config.
pub struct BackendConfigPayload(pub Payload);

/// Attach instruction field 99 without normalizing inline bytes or IDs.
pub fn set_backend_config_payload(
    ctx: &Context,
    instruction: Ptr<Operation>,
    payload: BackendConfigPayload,
) -> Result<()> {
    require(
        instruction.deref(ctx).get_num_results() == 1
            && Operation::get_op_dyn(instruction, ctx).get_opid().dialect
                == Identifier::new("xlahlo").into(),
        "backend config payload requires an HLO instruction",
    )?;
    instruction
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_backend_config_payload"), payload);
    Ok(())
}

/// Read field 99, distinguishing absent attribute from a present empty payload.
pub fn backend_config_payload(
    ctx: &Context,
    instruction: Ptr<Operation>,
) -> Result<Option<BackendConfigPayload>> {
    Ok(
        optional_attr::<BackendConfigPayload>(ctx, instruction, "hlo_backend_config_payload")?
            .map(|payload| payload.clone()),
    )
}

/// Resolve field 99 exactly, falling back to legacy bytes only if absent.
///
/// `Some(Payload::Absent)` represents a present but empty Payload message.
pub fn resolve_backend_config(
    payload: Option<&BackendConfigPayload>,
    legacy: &Bytes,
    payloads: &super::MetadataPayloads,
) -> Result<Bytes> {
    match payload.map(|payload| &payload.0) {
        None => Ok(legacy.clone()),
        Some(Payload::Absent) => Ok(Bytes::default()),
        Some(Payload::Value(bytes)) => Ok(bytes.clone()),
        Some(Payload::Id(id)) => {
            let bytes = usize::try_from(*id).ok().and_then(|id| payloads.0.get(id));
            bytes.cloned().ok_or_else(|| {
                crate::verification::error("backend config payload ID is out of range")
            })
        }
    }
}

pub(super) fn register(ctx: &mut Context) {
    macro_rules! attrs {($($ty:ty),+)=>{$(<$ty>::register::<$ty>(ctx);)+};}
    attrs!(
        CrossProgramPrefetch,
        ModuleProfileInfo,
        ComputationDevice,
        DeviceAssignment,
        FileLocation,
        StackFrame,
        StackFrameIndex,
        X64Config,
        DebugAttributes,
        DebugAttributeTableEntry,
        RecoveryModule,
        OriginalValueRecoveryEntry,
        OriginalValueRecoveryTable,
        ModuleMetadata,
        BackendConfigPayload
    );
}
