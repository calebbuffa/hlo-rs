//! Module-owned HLO configuration on native pliron operations.
use crate::{
    Diagnostic,
    attributes::{Axes, Bytes, StringPair},
    verification::require,
};
use pliron::{
    attribute::Attribute,
    builtin::{attributes::IdentifierAttr, ops::ModuleOp},
    combine::Parser,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_attr,
    identifier::Identifier,
    irfmt::parsers::delimited_list_parser,
    location::Located,
    operation::Operation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, ListSeparator, Printable},
    result::Result,
    r#type::TypeHandle,
};
use std::collections::BTreeMap;

#[cfg(test)]
mod clone_with_replacements;
mod control;
mod fusion_parameter;
pub mod instruction_order;
#[cfg(test)]
mod instruction_replacement;
mod metadata;
#[cfg(test)]
mod schedule_update;
pub use control::*;
pub use fusion_parameter::*;
pub use metadata::*;

#[pliron_attr(
    name = "xlahlo.module_config",
    format = "`<` $replica_count `,` $num_partitions `,` $use_spmd_partitioning `,` $alias_passthrough_params `,` `[` vec($allow_spmd_sharding_propagation_to_parameters, CharSpace(`,`)) `]` `,` `[` vec($allow_spmd_sharding_propagation_to_output, CharSpace(`,`)) `]` `,` $debug_options `,` opt($entry_computation_layout) `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Module-wide execution topology and alias/sharding policies.
pub struct ModuleConfig {
    /// Number of replicated program instances.
    pub replica_count: i64,
    /// Number of partitions per program instance; must be positive.
    pub num_partitions: i64,
    /// Whether sharding is interpreted using SPMD partitioning.
    pub use_spmd_partitioning: bool,
    /// Whether passthrough parameters may alias outputs.
    pub alias_passthrough_params: bool,
    /// Per-parameter permission to propagate SPMD sharding.
    pub allow_spmd_sharding_propagation_to_parameters: Vec<bool>,
    /// Per-output permission to propagate SPMD sharding.
    pub allow_spmd_sharding_propagation_to_output: Vec<bool>,
    /// Backend-specific string options read by XLA passes.
    pub debug_options: DebugOptions,
    /// Explicit entry parameter and result shapes, including physical layouts.
    pub entry_computation_layout: Option<EntryComputationLayout>,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            replica_count: 1,
            num_partitions: 1,
            use_spmd_partitioning: false,
            alias_passthrough_params: false,
            allow_spmd_sharding_propagation_to_parameters: vec![false],
            allow_spmd_sharding_propagation_to_output: vec![false],
            debug_options: DebugOptions::default(),
            entry_computation_layout: None,
        }
    }
}

impl ModuleConfig {
    /// Whether channel IDs carry semantic information in this module.
    ///
    /// This mirrors `HloModuleConfig::ChannelIdSensitive()` in pinned OpenXLA.
    pub fn channel_id_sensitive(&self) -> bool {
        !self.use_spmd_partitioning
    }
}

#[pliron_attr(name = "xlahlo.debug_options", verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Debug options supported by xlahlo's native module configuration.
pub struct DebugOptions {
    /// Backend extra options, keyed and ordered lexicographically.
    pub xla_backend_extra_options: BTreeMap<String, String>,
}

impl Printable for DebugOptions {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        let pairs: Vec<_> = self
            .xla_backend_extra_options
            .iter()
            .map(|(key, value)| StringPair {
                key: key.clone(),
                value: value.clone(),
            })
            .collect();
        pliron::irfmt::printers::enclosed(
            "[",
            "]",
            pliron::irfmt::printers::list_with_sep(&pairs, ListSeparator::CharSpace(',')),
        )
        .fmt(ctx, state, f)
    }
}

impl Parsable for DebugOptions {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let (pairs, _) = delimited_list_parser('[', ']', ',', StringPair::parser(()))
            .parse_stream(state_stream)
            .into_result()?;
        let mut xla_backend_extra_options = BTreeMap::new();
        for pair in pairs {
            if xla_backend_extra_options
                .insert(pair.key, pair.value)
                .is_some()
            {
                return pliron::input_err!(
                    state_stream.loc(),
                    "duplicate xla_backend_extra_options key"
                )
                .into_parse_result();
            }
        }
        Ok(Self {
            xla_backend_extra_options,
        })
        .into_parse_result()
    }
}

#[pliron_attr(
    name = "xlahlo.entry_computation_layout",
    format = "`<` `[` vec($parameter_shapes, CharSpace(`,`)) `]` `,` $result_shape `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Explicit entry computation parameter/result shapes with their native layouts.
pub struct EntryComputationLayout {
    /// Parameter shapes in entry parameter-number order.
    pub parameter_shapes: Vec<TypeHandle>,
    /// Entry result shape.
    pub result_shape: TypeHandle,
}

impl EntryComputationLayout {
    /// Construct an entry layout from its parameter and result shapes.
    pub fn new(parameter_shapes: Vec<TypeHandle>, result_shape: TypeHandle) -> Self {
        Self {
            parameter_shapes,
            result_shape,
        }
    }
}

#[pliron_attr(name = "xlahlo.alias_kind", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Strength of an entry parameter/output buffer alias.
pub enum AliasKind {
    /// The compiler may reuse the input allocation.
    MayAlias,
    /// Input and output must use the same allocation and exact shape/layout.
    MustAlias,
}

#[pliron_attr(
    name = "xlahlo.input_output_alias",
    format = "`<` $output_shape_index `,` $parameter_number `,` $parameter_shape_index `,` $kind `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// An alias between array leaves of an entry parameter and the entry result.
pub struct AliasEntry {
    /// Tuple path into the entry result.
    pub output_shape_index: Axes,
    /// Zero-based entry parameter number.
    pub parameter_number: i64,
    /// Tuple path into the parameter.
    pub parameter_shape_index: Axes,
    /// Whether allocation sharing is optional or required.
    pub kind: AliasKind,
}

#[pliron_attr(
    name = "xlahlo.input_output_aliases",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Entry input/output aliases, attached under `hlo_input_output_aliases`.
pub struct InputOutputAliases(pub Vec<AliasEntry>);

#[pliron_attr(
    name = "xlahlo.buffer_donor",
    format = "`<` $parameter_number `,` $parameter_shape_index `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// An entry parameter array leaf whose allocation may be donated.
pub struct BufferDonorEntry {
    /// Zero-based entry parameter number.
    pub parameter_number: i64,
    /// Tuple path into the parameter.
    pub parameter_shape_index: Axes,
}

#[pliron_attr(
    name = "xlahlo.buffer_donors",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Donated entry parameter buffers, attached under `hlo_buffer_donors`.
pub struct BufferDonors(pub Vec<BufferDonorEntry>);

#[pliron_attr(
    name = "xlahlo.schedule_sequence",
    format = "`<` $computation `,` `[` vec($instructions, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Execution order of named instructions in a computation.
pub struct InstructionSequence {
    /// Computation symbol, not a display name.
    pub computation: Identifier,
    /// Instruction names in execution order, excluding the return terminator.
    pub instructions: Vec<String>,
}

#[pliron_attr(
    name = "xlahlo.schedule",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Module execution schedule, attached under `hlo_schedule`.
pub struct Schedule(pub Vec<InstructionSequence>);

#[pliron_attr(name = "xlahlo.instruction_name", format = "$0", verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Globally unique instruction name, attached under `hlo_name`.
pub struct InstructionName(pub String);

#[pliron_attr(
    name = "xlahlo.control_predecessors",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Named control dependencies, attached under `hlo_control_predecessors`.
pub struct ControlPredecessors(pub Vec<String>);

#[pliron_attr(name = "xlahlo.execution_thread", format = "$0", verifier = "succ")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Computation execution thread; an empty string denotes `main`.
pub struct ExecutionThread(pub String);

#[pliron_attr(
    name = "xlahlo.metadata_payloads",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Module-owned payload bytes referenced by metadata and backend-config IDs.
pub struct MetadataPayloads(pub Vec<Bytes>);

/// Attach typed configuration to a builtin module.
///
/// Returns an error for a non-module operation or invalid configuration.
pub fn set_config(ctx: &Context, module: Ptr<Operation>, config: ModuleConfig) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "configuration requires builtin.module",
    )?;
    config.verify(ctx)?;
    module
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_module_config"), config);
    Ok(())
}

/// Return an owned configuration snapshot, or defaults if none is attached.
///
/// Returns an error for a non-module operation or an ill-typed attribute.
pub fn config(ctx: &Context, module: Ptr<Operation>) -> Result<ModuleConfig> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "configuration requires builtin.module",
    )?;
    let data = module.deref(ctx);
    match data.attributes.0.get(&Identifier::new("hlo_module_config")) {
        None => Ok(ModuleConfig::default()),
        Some(value) => value
            .downcast_ref::<ModuleConfig>()
            .cloned()
            .ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                    attribute: "hlo_module_config",
                    message: "expected an ir::ModuleConfig attribute".to_owned(),
                })
            }),
    }
}

/// Set the explicit entry computation symbol of a builtin module.
pub fn set_entry(ctx: &Context, module: Ptr<Operation>, entry: Identifier) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "entry requires builtin.module",
    )?;
    module
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_entry"), IdentifierAttr::new(entry));
    Ok(())
}

/// Return the explicit entry symbol, rejecting missing or ill-typed metadata.
pub fn entry(ctx: &Context, module: Ptr<Operation>) -> Result<Identifier> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "entry requires builtin.module",
    )?;
    module
        .deref(ctx)
        .attributes
        .get::<IdentifierAttr>(&Identifier::new("hlo_entry"))
        .map(|name| name.as_ref().clone())
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "hlo_entry",
                message: "expected an entry computation symbol; set it with ir::set_entry"
                    .to_owned(),
            })
        })
}

pub(crate) fn register(ctx: &mut Context) {
    metadata::register(ctx);
    fusion_parameter::register(ctx);
    instruction_order::register(ctx);
    macro_rules! attrs {($($ty:ty),+)=>{$(<$ty>::register::<$ty>(ctx);)+};}
    attrs!(
        ModuleConfig,
        DebugOptions,
        EntryComputationLayout,
        AliasKind,
        AliasEntry,
        InputOutputAliases,
        BufferDonorEntry,
        BufferDonors,
        InstructionSequence,
        Schedule,
        InstructionName,
        ControlPredecessors,
        ExecutionThread,
        MetadataPayloads
    );
}
