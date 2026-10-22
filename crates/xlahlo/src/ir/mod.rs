//! Module-owned HLO configuration on native pliron operations.
use crate::{
    Diagnostic,
    attributes::{Axes, Bytes, StringPair},
    verification::require,
};
use pliron::{
    attribute::Attribute,
    builtin::ops::ModuleOp,
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

mod clone_with_replacements;
mod computation_cloning;
mod control;
mod fusion_parameter;
pub mod instruction_order;
mod instruction_replacement;
mod metadata;
mod original_value;
mod schedule_update;
pub use computation_cloning::{
    ComputationClone, ComputationCloneRequest, ComputationCloneRoot, ComputationReplacements,
    clone_computation_with_replacements, clone_computations_with_replacements,
};
pub use control::*;
pub use fusion_parameter::*;
pub use instruction_replacement::{
    DefusedFusion, InlineCallOptions, InlinedCall, InstructionReplacement,
    ReplaceInstructionOptions, WhileInductionForwarding, WhileLoopAnnotationBatch,
    WhileLoopAnnotationRequest, annotate_while_loops, defuse, inline_call,
    remove_instruction_and_unused_operands, replace_all_uses_with, replace_instruction,
};
pub use metadata::*;
pub use original_value::{OriginalArray, OriginalValue, OriginalValueElement};
pub use schedule_update::{update_schedule, update_schedule_with_threads};

#[pliron_attr(name = "xlahlo.module_config", verifier = "succ")]
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
    /// Preserve Shardy manual-computation calls during HLO round trips.
    pub use_shardy_partitioner: bool,
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
            use_shardy_partitioner: false,
        }
    }
}

impl ModuleConfig {
    /// Pinned `HloModuleConfig::use_shardy_partitioner`, independent of SPMD.
    pub fn use_shardy_partitioner(&self) -> bool {
        self.use_shardy_partitioner
    }

    /// Set whether Shardy manual-computation calls are preserved on HLO
    /// round trips; this setting is independent of SPMD partitioning.
    pub fn set_use_shardy_partitioner(&mut self, enabled: bool) {
        self.use_shardy_partitioner = enabled;
    }

    /// Whether channel IDs carry semantic information in this module.
    ///
    /// This mirrors `HloModuleConfig::ChannelIdSensitive()` in pinned OpenXLA.
    pub fn channel_id_sensitive(&self) -> bool {
        !self.use_spmd_partitioning
    }
}

impl Printable for ModuleConfig {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "<{}, {}, {}, {}, [",
            self.replica_count,
            self.num_partitions,
            self.use_spmd_partitioning,
            self.alias_passthrough_params
        )?;
        pliron::irfmt::printers::list_with_sep(
            &self.allow_spmd_sharding_propagation_to_parameters,
            ListSeparator::CharSpace(','),
        )
        .fmt(ctx, state, f)?;
        write!(f, "], [")?;
        pliron::irfmt::printers::list_with_sep(
            &self.allow_spmd_sharding_propagation_to_output,
            ListSeparator::CharSpace(','),
        )
        .fmt(ctx, state, f)?;
        write!(f, "], ")?;
        self.debug_options.fmt(ctx, state, f)?;
        write!(f, ", ")?;
        if let Some(layout) = &self.entry_computation_layout {
            layout.fmt(ctx, state, f)?;
        }
        write!(f, ", {}>", self.use_shardy_partitioner)
    }
}

impl Parsable for ModuleConfig {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(stream: &mut StateStream<'a>, _: ()) -> ParseResult<'a, Self> {
        use pliron::{
            combine::{attempt, optional, parser::char::char},
            irfmt::parsers::spaced,
        };
        let ((replicas, partitions, spmd, alias, parameters, output, debug, layout, shardy), _) =
            spaced(char('<'))
                .with((
                    i64::parser(()).skip(spaced(char(','))),
                    i64::parser(()).skip(spaced(char(','))),
                    bool::parser(()).skip(spaced(char(','))),
                    bool::parser(()).skip(spaced(char(','))),
                    delimited_list_parser('[', ']', ',', bool::parser(())).skip(spaced(char(','))),
                    delimited_list_parser('[', ']', ',', bool::parser(())).skip(spaced(char(','))),
                    DebugOptions::parser(()).skip(spaced(char(','))),
                    optional(EntryComputationLayout::parser(())),
                    optional(attempt(spaced(char(','))).with(bool::parser(()))),
                ))
                .skip(spaced(char('>')))
                .parse_stream(stream)
                .into_result()?;
        Ok(Self {
            replica_count: replicas,
            num_partitions: partitions,
            use_spmd_partitioning: spmd,
            alias_passthrough_params: alias,
            allow_spmd_sharding_propagation_to_parameters: parameters,
            allow_spmd_sharding_propagation_to_output: output,
            debug_options: debug,
            entry_computation_layout: layout,
            use_shardy_partitioner: shardy.unwrap_or(false),
        })
        .into_parse_result()
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

#[pliron_attr(name = "xlahlo.computation_name", format = "$0", verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
/// Source-faithful computation symbol such as `main.15` or `region-0`.
///
/// The text is stored and printed verbatim (quoted by the attribute printer),
/// so punctuation survives a print/parse round trip without mangling.
/// [`ComputationName::validate`] rejects empty names and names containing
/// whitespace or control characters; it is checked wherever a symbol is
/// defined (`ComputationOp`, `set_entry`).
pub struct ComputationName(pub String);

impl ComputationName {
    /// Wrap source text verbatim; call [`Self::validate`] before accepting it.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the exact source text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Reject empty names and names containing whitespace or control characters.
    pub fn validate(&self) -> Result<()> {
        require(
            !self.0.is_empty() && !self.0.chars().any(|c| c.is_whitespace() || c.is_control()),
            "computation names must be non-empty without whitespace or control characters",
        )
    }
}

impl AsRef<str> for ComputationName {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ComputationName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<Identifier> for ComputationName {
    fn from(name: Identifier) -> Self {
        Self(name.to_string())
    }
}

impl From<&Identifier> for ComputationName {
    fn from(name: &Identifier) -> Self {
        Self(name.to_string())
    }
}

impl From<&str> for ComputationName {
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}

impl From<String> for ComputationName {
    fn from(name: String) -> Self {
        Self(name)
    }
}

impl PartialEq<str> for ComputationName {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for ComputationName {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

#[pliron_attr(name = "xlahlo.module_name", format = "$0", verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Source HloModule name, attached to the builtin module under `hlo_module_name`.
pub struct ModuleName(pub String);

#[pliron_attr(
    name = "xlahlo.schedule_sequence",
    format = "`<` $computation `,` `[` vec($instructions, CharSpace(`,`)) `]` `>`",
    verifier = "succ"
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Execution order of named instructions in a computation.
pub struct InstructionSequence {
    /// Computation symbol, not a display name.
    pub computation: ComputationName,
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
pub fn set_entry(
    ctx: &Context,
    module: Ptr<Operation>,
    entry: impl Into<ComputationName>,
) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "entry requires builtin.module",
    )?;
    let entry = entry.into();
    entry.validate()?;
    module
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_entry"), entry);
    Ok(())
}

/// Return the explicit entry symbol, rejecting missing or ill-typed metadata.
pub fn entry(ctx: &Context, module: Ptr<Operation>) -> Result<ComputationName> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "entry requires builtin.module",
    )?;
    module
        .deref(ctx)
        .attributes
        .get::<ComputationName>(&Identifier::new("hlo_entry"))
        .cloned()
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "hlo_entry",
                message: "expected an entry computation symbol; set it with ir::set_entry"
                    .to_owned(),
            })
        })
}

/// Record the source HloModule name on a builtin module.
pub fn set_module_name(
    ctx: &Context,
    module: Ptr<Operation>,
    name: impl Into<String>,
) -> Result<()> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "module name requires builtin.module",
    )?;
    module
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_module_name"), ModuleName(name.into()));
    Ok(())
}

/// Return the source HloModule name, if one was recorded.
pub fn module_name(ctx: &Context, module: Ptr<Operation>) -> Result<Option<String>> {
    require(
        Operation::is_op::<ModuleOp>(module, ctx),
        "module name requires builtin.module",
    )?;
    let op = module.deref(ctx);
    let key = Identifier::new("hlo_module_name");
    if !op.attributes.0.contains_key(&key) {
        return Ok(None);
    }
    op.attributes
        .get::<ModuleName>(&key)
        .map(|name| Some(name.0.clone()))
        .ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "hlo_module_name",
                message: "expected an ir::ModuleName attribute".to_owned(),
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
        ComputationName,
        ModuleName,
        ControlPredecessors,
        ExecutionThread,
        MetadataPayloads
    );
}

#[cfg(test)]
mod module_config_tests {

    use super::*;
    use crate::{
        attributes::Index,
        ops::{ComputationOp, ReturnOp, control::ParameterOp},
        shape,
        types::ElementType,
    };
    use pliron::{
        basic_block::BasicBlock, builtin::types::FunctionType, irfmt::parsers::spaced, op::Op,
        parsable::parse_from_str,
    };

    #[test]
    fn computation_names_are_source_faithful_and_validated() {
        for ok in ["main", "main.15", "region-0.3", "a:b/c"] {
            let name = ComputationName::from(ok);
            assert!(name.validate().is_ok(), "{ok}");
            assert_eq!(name.as_str(), ok);
            assert_eq!(name, ok);
        }
        for bad in ["", "a b", "a\tb", "a\nb"] {
            assert!(ComputationName::from(bad).validate().is_err(), "{bad:?}");
        }
        let ctx = &mut Context::new();
        crate::register(ctx);
        let module =
            pliron::builtin::ops::ModuleOp::new(ctx, pliron::identifier::Identifier::new("m"));
        let module = module.get_operation();
        assert!(set_entry(ctx, module, "").is_err());
        set_entry(ctx, module, "main.15").unwrap();
        assert_eq!(entry(ctx, module).unwrap(), "main.15");
        assert_eq!(module_name(ctx, module).unwrap(), None);
        set_module_name(ctx, module, "jit_fn-1").unwrap();
        assert_eq!(
            module_name(ctx, module).unwrap().as_deref(),
            Some("jit_fn-1")
        );
    }

    #[test]
    fn shardy_config_defaults_and_legacy_assembly_are_backward_compatible() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let legacy = "<1, 1, false, false, [false], [false], [], >";
        let parsed = parse_from_str(spaced(ModuleConfig::parser(())), ctx, legacy).unwrap();
        assert_eq!(parsed, ModuleConfig::default());
        assert!(!parsed.use_shardy_partitioner());
        assert!(parsed.channel_id_sensitive());
        assert!(
            parse_from_str(
                spaced(ModuleConfig::parser(())),
                ctx,
                "<1, 1, false, false, [false], [false], [], , 1>"
            )
            .is_err()
        );
    }

    #[test]
    fn shardy_config_roundtrips_both_values_and_preserves_legacy_layout() {
        let ctx = &mut Context::new();
        crate::register(ctx);
        let ty = shape::array(ctx, ElementType::F32, vec![2]).unwrap();
        let mut config = ModuleConfig {
            entry_computation_layout: Some(EntryComputationLayout::new(vec![ty], ty)),
            ..Default::default()
        };
        for enabled in [false, true] {
            config.set_use_shardy_partitioner(enabled);
            assert_eq!(config.use_shardy_partitioner(), enabled);
            assert!(!config.use_spmd_partitioning);
            config.verify(ctx).unwrap();
            let text = config.disp(ctx).to_string();
            let parsed = parse_from_str(spaced(ModuleConfig::parser(())), ctx, &text).unwrap();
            assert_eq!(parsed, config);
        }
        let text = config.disp(ctx).to_string();
        let legacy = format!("{}>", text.strip_suffix(", true>").unwrap());
        let recovered = parse_from_str(spaced(ModuleConfig::parser(())), ctx, &legacy).unwrap();
        config.set_use_shardy_partitioner(false);
        assert_eq!(recovered, config);
    }

    #[test]
    fn module_shardy_config_fresh_parse_and_verification_preserve_true_and_false() {
        for enabled in [false, true] {
            let ctx = &mut Context::new();
            crate::register(ctx);
            let ty = shape::array(ctx, ElementType::F32, vec![2]).unwrap();
            let module = ModuleOp::new(ctx, Identifier::new("config")).get_operation();
            set_entry(ctx, module, Identifier::new("main")).unwrap();
            let container = module
                .deref(ctx)
                .get_region(0)
                .deref(ctx)
                .get_entry_block()
                .unwrap();
            let signature = FunctionType::get(ctx, vec![ty], vec![ty]).into();
            let computation = ComputationOp::new(ctx, Identifier::new("main"), signature);
            computation.get_operation().insert_at_back(container, ctx);
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(computation.body(ctx), ctx);
            let parameter = ParameterOp::new(ctx, ty, vec![], Index(0));
            parameter.get_operation().insert_at_back(block, ctx);
            ReturnOp::new(ctx, parameter.result(ctx))
                .get_operation()
                .insert_at_back(block, ctx);
            let config = ModuleConfig {
                use_shardy_partitioner: enabled,
                ..Default::default()
            };
            set_config(ctx, module, config.clone()).unwrap();
            assert_eq!(super::config(ctx, module).unwrap(), config);
            crate::verification::verify(ctx, module).unwrap();
            let text = module.disp(ctx).to_string();
            Operation::erase(module, ctx);
            assert!(ctx.is_ir_empty());
            let fresh = &mut Context::new();
            crate::register(fresh);
            let parsed =
                parse_from_str(spaced(Operation::top_level_parser()), fresh, &text).unwrap();
            crate::verification::verify(fresh, parsed).unwrap();
            assert_eq!(
                super::config(fresh, parsed)
                    .unwrap()
                    .use_shardy_partitioner(),
                enabled
            );
            Operation::erase(parsed, fresh);
            assert!(fresh.is_ir_empty());
        }
    }
}
