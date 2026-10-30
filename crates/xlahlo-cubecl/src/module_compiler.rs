use std::sync::{Arc, RwLock};

use pliron::{
    context::{Context, Ptr},
    operation::Operation,
};

use crate::{
    KernelOverrideError, KernelOverrideRegistry, RegistrationError,
    plan::{
        BufferDescriptor, BufferId, CubeclTargetPass, ExecutionPlan, ExecutionPlanBuilder,
        KernelDefinitionError, PlanError,
    },
};

/// Stable categories for unsupported compile contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnsupportedKind {
    /// Runtime-dependent dimensions cannot be represented by this lowering.
    DynamicShape,
    /// Required addressing exceeds the supported target index width.
    AddressWidth,
    /// The logical element format has no supported lowering for this operation.
    ElementType,
    /// Physical layout metadata cannot be represented by the plan.
    PhysicalLayout,
    /// Operand or result precision combinations are unsupported.
    MixedPrecision,
    /// Tuple structure or control flow is outside the supported contract.
    TupleOrControl,
    /// A selected strategy does not support logical operand views.
    IndexedStrategy,
    /// Canonical index semantics could not be lowered.
    CanonicalIndex,
    /// The operation or its configuration has no supported implementation.
    Operation,
}

/// Errors produced by module strategy selection or execution-plan validation.
#[derive(Debug, thiserror::Error)]
pub enum ModuleCompileError {
    /// Native verification failed before lowering; the original diagnostic is retained.
    #[error("xlahlo module verification failed: {0}")]
    Verification(#[source] pliron::result::Error),
    /// No implementation supports this operation or module contract.
    #[error("xlahlo module is unsupported: {feature}")]
    Unsupported {
        /// Unsupported source feature or requirement.
        feature: String,
    },
    /// A known compatibility gap with a machine-readable classification.
    #[error("xlahlo module is unsupported ({kind:?}): {feature}")]
    UnsupportedContract {
        /// Compatibility category, independent of the diagnostic wording.
        kind: crate::UnsupportedKind,
        /// Unsupported source feature or requirement.
        feature: String,
    },
    /// Module-level override selection or lowering failed.
    #[error("lowering strategy dispatch failed: {0}")]
    Strategy(#[from] KernelOverrideError<PlanError>),
    /// An operation-strategy callback failed.
    #[error("operation lowering strategy failed: {0}")]
    OperationStrategy(String),
    /// Lowering produced an invalid execution plan.
    #[error("execution plan is invalid: {0}")]
    Plan(#[source] PlanError),
    /// A generated kernel failed definition, target-pass, or ABI validation.
    #[error("kernel definition validation failed: {0}")]
    KernelDefinition(#[from] KernelDefinitionError),
}

impl From<PlanError> for ModuleCompileError {
    fn from(error: PlanError) -> Self {
        match error {
            PlanError::UnsupportedContract { kind, feature } => {
                Self::UnsupportedContract { kind, feature }
            }
            PlanError::UnsupportedFeature(feature) => Self::Unsupported { feature },
            PlanError::UnsupportedElementType(element) => Self::UnsupportedContract {
                kind: crate::UnsupportedKind::ElementType,
                feature: format!("element type {element:?}"),
            },
            error => Self::Plan(error),
        }
    }
}

impl ModuleCompileError {
    /// Stable compatibility category; validation and implementation failures return None.
    pub fn unsupported_kind(&self) -> Option<crate::UnsupportedKind> {
        match self {
            Self::Unsupported { .. } => Some(crate::UnsupportedKind::Operation),
            Self::UnsupportedContract { kind, .. } => Some(*kind),
            Self::Plan(error) => error.unsupported_kind(),
            Self::Strategy(KernelOverrideError::Lowering { source, .. }) => {
                source.unsupported_kind()
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    use crate::UnsupportedKind;

    #[test]
    fn compatibility_categories_survive_each_plan_error_boundary() {
        for kind in [
            UnsupportedKind::DynamicShape,
            UnsupportedKind::AddressWidth,
            UnsupportedKind::ElementType,
            UnsupportedKind::PhysicalLayout,
            UnsupportedKind::MixedPrecision,
            UnsupportedKind::TupleOrControl,
        ] {
            let error = ModuleCompileError::from(PlanError::UnsupportedContract {
                kind,
                feature: "unsupported native contract".into(),
            });
            assert_eq!(error.unsupported_kind(), Some(kind));
            assert!(error.to_string().contains("unsupported native contract"));
        }
        assert_eq!(
            ModuleCompileError::from(PlanError::CanonicalIndexUnsupported("unproved".into()))
                .unsupported_kind(),
            Some(UnsupportedKind::CanonicalIndex),
        );
        assert_eq!(
            ModuleCompileError::from(PlanError::ReadBeforeWrite { buffer: 0 }).unsupported_kind(),
            None,
        );
    }
}

/// Per-operation inputs and preallocated results for a compositional lowering strategy.
pub struct OperationLoweringRequest<'a> {
    /// Context owning the verified source operation; borrowed only during lowering.
    pub context: &'a Context,
    /// Source instruction selected for this strategy.
    pub operation: Ptr<Operation>,
    /// Physical operand buffer identifiers, in source operand order.
    pub operands: &'a [BufferId],
    /// Logical operand descriptors in the same order as `operands`.
    pub operand_descriptors: &'a [BufferDescriptor],
    /// Logical operand indexing derived exclusively from xlahlo operations.
    pub operand_views: &'a [crate::OperandView],
    /// Preallocated result buffers to be produced by the strategy.
    pub results: &'a [BufferId],
    /// Required logical descriptors for `results`.
    pub result_descriptors: &'a [BufferDescriptor],
    /// Shared schedule to which the strategy appends kernels and dependencies.
    pub plan: &'a mut ExecutionPlanBuilder,
}

/// An owned, pure scalar implementation suitable for a generated fused kernel.
pub trait ScalarLoweringBody: Send + Sync + 'static {
    /// Full semantic identity, including any captured function binding.
    /// Equal keys must emit equivalent code; include all captured code-generation settings.
    fn cache_key(&self) -> String;
    /// Emit a pure scalar DAG in `scope` using the supplied ordered operands.
    ///
    /// Return a value owned by this scope. Do not capture source-context pointers;
    /// the body may be used after the source module and registry have been dropped.
    fn emit(
        &self,
        scope: &cubecl_ir::Scope,
        operands: &[pliron::value::Value],
    ) -> Result<pliron::value::Value, String>;
}

/// Validated logical geometry for a pure, nonsegmented ordered fold.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OrderedReductionGeometry {
    /// Logical input tensor folded by the callback.
    pub input: BufferDescriptor,
    /// Logical result tensor after the fold.
    pub output: BufferDescriptor,
    /// Input axes traversed by the ordered fold.
    pub axes: Vec<usize>,
    /// Whether reduced axes remain as unit extents in the output.
    pub keepdim: bool,
    /// Element formats of the carried scalar accumulator values.
    pub accumulators: Vec<xlahlo::types::ElementType>,
}

/// Owned callbacks retained after the source module and registry are dropped.
/// Each callback emits its actual scalar DAG; neither combine semantics nor
/// index independence is inferred by the compiler.
pub trait OrderedReductionLoweringBody: Send + Sync + 'static {
    /// Complete semantic identity, including captured bindings and geometry.
    fn cache_key(&self) -> String;
    /// Validated logical geometry of the ordered fold.
    fn geometry(&self) -> OrderedReductionGeometry;
    /// Emit initial accumulator values in geometry order into `scope`.
    fn init(&self, scope: &cubecl_ir::Scope) -> Result<Vec<pliron::value::Value>, String>;
    /// `index` is the real zero-based logical fold position with Index type.
    fn accumulate(
        &self,
        scope: &cubecl_ir::Scope,
        previous: &[pliron::value::Value],
        input: pliron::value::Value,
        index: pliron::value::Value,
    ) -> Result<Vec<pliron::value::Value>, String>;
    /// `count` is UInt32 and includes every logically folded element.
    fn finalize(
        &self,
        scope: &cubecl_ir::Scope,
        accumulators: &[pliron::value::Value],
        count: pliron::value::Value,
    ) -> Result<pliron::value::Value, String>;
}

/// A priority-selected lowering implementation for one verified xlahlo operation.
///
/// Implementations may provide whole-buffer kernels and an optional scalar body.
pub trait OperationLoweringStrategy: Send + Sync + 'static {
    /// Nonempty name, unique within the compiler's operation strategy registry.
    fn name(&self) -> &'static str;
    /// Selection priority; the unique highest-priority match wins.
    fn priority(&self) -> i32;
    /// Opt into applying `operand_views` when loading operands.
    fn supports_indexed_operands(&self) -> bool {
        false
    }
    /// Inspect a verified instruction without mutation.
    ///
    /// Errors abort compilation; they are not interpreted as non-matches.
    fn matches(&self, context: &Context, operation: Ptr<Operation>) -> Result<bool, String>;
    /// Supply a pure scalar body after validating the complete operation contract.
    /// Returning `None` keeps the operation on its normal planned-kernel route.
    fn scalar_body(
        &self,
        _context: &Context,
        _operation: Ptr<Operation>,
    ) -> Result<Option<Arc<dyn ScalarLoweringBody>>, String> {
        Ok(None)
    }
    /// Supply a pure nonsegmented ordered fold after exact operation validation.
    /// None retains the existing compositional lowering.
    fn ordered_reduction_body(
        &self,
        _context: &Context,
        _operation: Ptr<Operation>,
    ) -> Result<Option<Arc<dyn OrderedReductionLoweringBody>>, String> {
        Ok(None)
    }
    /// Append kernels that produce the requested results using the declared ABI.
    ///
    /// Errors abort compilation and discard the compiler's in-progress plan.
    fn lower(&self, request: OperationLoweringRequest<'_>) -> Result<(), PlanError>;
}

#[derive(Default, Clone)]
pub(crate) struct OperationStrategyRegistry {
    strategies: Vec<Arc<dyn OperationLoweringStrategy>>,
    pub(crate) fusion_policy: Option<Arc<dyn crate::TargetFusionPolicy>>,
}

impl OperationStrategyRegistry {
    fn register(
        &mut self,
        strategy: Arc<dyn OperationLoweringStrategy>,
    ) -> Result<(), RegistrationError> {
        if strategy.name().trim().is_empty() {
            return Err(RegistrationError::EmptyName);
        }
        if self
            .strategies
            .iter()
            .any(|registered| registered.name() == strategy.name())
        {
            return Err(RegistrationError::DuplicateName(strategy.name().to_owned()));
        }
        self.strategies.push(strategy);
        self.strategies
            .sort_by_key(|registered| registered.priority());
        Ok(())
    }

    pub(crate) fn select(
        &self,
        context: &Context,
        operation: Ptr<Operation>,
    ) -> Result<Option<usize>, String> {
        let mut matches = Vec::new();
        for (index, strategy) in self.strategies.iter().enumerate() {
            let matched = strategy
                .matches(context, operation)
                .map_err(|message| format!("matching {} failed: {message}", strategy.name()))?;
            if matched {
                matches.push((index, strategy.priority(), strategy.name()));
            }
        }
        let Some(highest) = matches.iter().map(|(_, priority, _)| *priority).max() else {
            return Ok(None);
        };
        let selected: Vec<_> = matches
            .into_iter()
            .filter(|(_, priority, _)| *priority == highest)
            .collect();
        if selected.len() != 1 {
            return Err(format!(
                "ambiguous operation strategies for {}: {}",
                Operation::get_opid(operation, context),
                selected
                    .iter()
                    .map(|(_, _, name)| *name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        Ok(Some(selected[0].0))
    }

    pub(crate) fn get(&self, index: usize) -> &Arc<dyn OperationLoweringStrategy> {
        &self.strategies[index]
    }

    pub(crate) fn scalar_body(
        &self,
        context: &Context,
        operation: Ptr<Operation>,
    ) -> Result<Option<Arc<dyn ScalarLoweringBody>>, PlanError> {
        self.select(context, operation)
            .map_err(PlanError::OperationStrategy)?
            .map(|index| self.get(index).scalar_body(context, operation))
            .transpose()
            .map(Option::flatten)
            .map_err(PlanError::OperationStrategy)
    }

    pub(crate) fn ordered_reduction_body(
        &self,
        context: &Context,
        operation: Ptr<Operation>,
    ) -> Result<Option<Arc<dyn OrderedReductionLoweringBody>>, PlanError> {
        self.select(context, operation)
            .map_err(PlanError::OperationStrategy)?
            .map(|index| self.get(index).ordered_reduction_body(context, operation))
            .transpose()
            .map(Option::flatten)
            .map_err(PlanError::OperationStrategy)
    }
}

/// Builder for the extensible xlahlo module compiler.
///
/// Strategies match a verified Pliron module and append concrete CubeCL
/// kernel steps, buffer contracts, launch geometries, and outputs to the same
/// plan. Equal-priority matching strategies are rejected; no-match is an
/// explicit error rather than an implicit fallback.
pub struct ModuleCompilerBuilder {
    strategies: KernelOverrideRegistry<ExecutionPlanBuilder, PlanError>,
    target_passes: Vec<Arc<dyn CubeclTargetPass>>,
    operation_strategies: Arc<RwLock<OperationStrategyRegistry>>,
    affine_output_views: bool,
}

impl Default for ModuleCompilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleCompilerBuilder {
    /// Create a builder with the built-in module lowerer at the lowest priority.
    ///
    /// Additional operation strategies and target passes are initially empty.
    pub fn new() -> Self {
        let mut builder = Self {
            strategies: KernelOverrideRegistry::new(),
            target_passes: Vec::new(),
            operation_strategies: Arc::new(RwLock::new(OperationStrategyRegistry::default())),
            affine_output_views: false,
        };
        let operation_strategies = Arc::clone(&builder.operation_strategies);
        if let Err(error) = builder.register_strategy(
            "xlahlo-cubecl.module",
            i32::MIN,
            |_, _| Ok(true),
            move |context, module, plan| {
                let strategies = operation_strategies.read().map_err(|_| {
                    PlanError::OperationStrategy("registry lock was poisoned".into())
                })?;
                crate::lowering::lower_module(context, module, plan, &strategies)
            },
        ) {
            panic!("failed to register the built-in xlahlo module lowerer: {error}");
        }
        builder
    }

    /// Require runtime-representable affine views at the output ABI.
    ///
    /// Computed composed views are materialized in the existing plan. Input-backed
    /// composed views are rejected rather than silently losing their alias contract.
    pub fn require_affine_output_views(mut self) -> Self {
        self.affine_output_views = true;
        self
    }

    /// Apply target profitability/resource policy to optional Gather/consumer
    /// and ordered producer/fold/epilogue fusion. No policy preserves existing
    /// behavior. Exact strategy selection and semantic legality take precedence;
    /// other emitter families are not repartitioned by this policy.
    pub fn set_fusion_policy(
        &mut self,
        policy: Arc<dyn crate::TargetFusionPolicy>,
    ) -> Result<(), RegistrationError> {
        self.operation_strategies
            .write()
            .map_err(|_| RegistrationError::PoisonedRegistry)?
            .fusion_policy = Some(policy);
        Ok(())
    }

    /// Register a module-level lowering strategy.
    ///
    /// Matchers receive the verified root Pliron module. The unique highest
    /// priority match is selected before lowering begins.
    pub fn register_strategy<M, L>(
        &mut self,
        name: impl Into<String>,
        priority: i32,
        matches: M,
        lower: L,
    ) -> Result<(), RegistrationError>
    where
        M: Fn(&Context, Ptr<Operation>) -> Result<bool, String> + 'static,
        L: Fn(&Context, Ptr<Operation>, &mut ExecutionPlanBuilder) -> Result<(), PlanError>
            + 'static,
    {
        self.strategies.register(name, priority, matches, lower)
    }

    /// Register a Pliron pass over each generated CubeCL kernel module.
    ///
    /// Passes execute in ascending priority and then registration order.
    pub fn register_target_pass(
        &mut self,
        pass: Arc<dyn CubeclTargetPass>,
    ) -> Result<(), RegistrationError> {
        if self
            .target_passes
            .iter()
            .any(|existing| existing.name() == pass.name())
        {
            return Err(RegistrationError::DuplicateName(pass.name().to_owned()));
        }
        self.target_passes.push(pass);
        self.target_passes
            .sort_by_key(|registered| registered.priority());
        Ok(())
    }

    /// Register a compositional lowering for matching individual xlahlo operations.
    ///
    /// The strategy receives planned operand/result buffers and appends its
    /// kernels to the same execution schedule as the built-in generic lowerer.
    pub fn register_operation_strategy(
        &mut self,
        strategy: Arc<dyn OperationLoweringStrategy>,
    ) -> Result<(), RegistrationError> {
        self.operation_strategies
            .write()
            .map_err(|_| RegistrationError::PoisonedRegistry)?
            .register(strategy)
    }

    /// Finish registration and consume the builder without compiling source IR.
    pub fn build(self) -> ModuleCompiler {
        ModuleCompiler {
            strategies: self.strategies,
            target_passes: self.target_passes,
            affine_output_views: self.affine_output_views,
        }
    }
}

/// Verified module compiler with pluggable lowering strategies and target passes.
pub struct ModuleCompiler {
    strategies: KernelOverrideRegistry<ExecutionPlanBuilder, PlanError>,
    target_passes: Vec<Arc<dyn CubeclTargetPass>>,
    affine_output_views: bool,
}

impl ModuleCompiler {
    /// Configure a compiler with the built-in lowerer and optional extensions.
    pub fn builder() -> ModuleCompilerBuilder {
        ModuleCompilerBuilder::new()
    }

    /// Verify and lower a complete xlahlo module to an explicit execution plan.
    ///
    /// The module must belong to `context`. Success owns the kernel artifacts and
    /// metadata, not a runtime or device allocations. The source IR is not mutated.
    /// Verification failures retain their typed diagnostics; unsupported contracts
    /// can be classified with [`ModuleCompileError::unsupported_kind`].
    pub fn compile(
        &self,
        context: &Context,
        module: Ptr<Operation>,
    ) -> Result<ExecutionPlan, ModuleCompileError> {
        let phase = crate::compile_profile::Phase::start("native_verify");
        xlahlo::verification::verify(context, module).map_err(ModuleCompileError::Verification)?;
        drop(phase);
        let mut target = ExecutionPlanBuilder::new(self.target_passes.clone());
        let phase = crate::compile_profile::Phase::start("strategy_lowering");
        let lowered = self.strategies.lower(context, module, &mut target);
        drop(phase);
        match lowered {
            Ok(Some(_)) => {
                let phase = crate::compile_profile::Phase::start("entry_layout");
                target.realize_entry_layout(context, module, self.affine_output_views)?;
                drop(phase);
                let phase = crate::compile_profile::Phase::start("plan_build");
                let plan = target.build()?;
                drop(phase);
                let phase = crate::compile_profile::Phase::start("kernel_definition_validation");
                plan.validate_kernel_definitions()?;
                drop(phase);
                Ok(plan)
            }
            Ok(None) => Err(ModuleCompileError::Unsupported {
                feature: format!(
                    "no lowering strategy matched module operation {}",
                    Operation::get_opid(module, context)
                ),
            }),
            Err(KernelOverrideError::Lowering {
                source: PlanError::UnsupportedContract { kind, feature },
                ..
            }) => Err(ModuleCompileError::UnsupportedContract { kind, feature }),
            Err(KernelOverrideError::Lowering {
                source: PlanError::UnsupportedFeature(feature),
                ..
            }) => Err(ModuleCompileError::Unsupported { feature }),
            Err(KernelOverrideError::Lowering {
                source: source @ PlanError::UnsupportedElementType(_),
                ..
            }) => Err(source.into()),
            Err(KernelOverrideError::Lowering {
                source: PlanError::OperationStrategy(message),
                ..
            }) => Err(ModuleCompileError::OperationStrategy(message)),
            Err(KernelOverrideError::Lowering {
                source: source @ PlanError::IndexedOperandsUnsupported { .. },
                ..
            }) => Err(ModuleCompileError::Plan(source)),
            Err(KernelOverrideError::Lowering {
                source: source @ PlanError::CanonicalIndexUnsupported(_),
                ..
            }) => Err(ModuleCompileError::Plan(source)),
            Err(error) => Err(ModuleCompileError::Strategy(error)),
        }
    }
}
