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

/// Errors produced by module strategy selection or execution-plan validation.
#[derive(Debug, thiserror::Error)]
pub enum ModuleCompileError {
    #[error("xlahlo module verification failed: {0}")]
    Verification(String),
    #[error("xlahlo module is unsupported: {feature}")]
    Unsupported { feature: String },
    #[error("lowering strategy dispatch failed: {0}")]
    Strategy(#[from] KernelOverrideError<PlanError>),
    #[error("operation lowering strategy failed: {0}")]
    OperationStrategy(String),
    #[error("execution plan is invalid: {0}")]
    Plan(#[from] PlanError),
    #[error("kernel definition validation failed: {0}")]
    KernelDefinition(#[from] KernelDefinitionError),
}

/// Per-operation inputs and preallocated results for a compositional lowering strategy.
pub struct OperationLoweringRequest<'a> {
    pub context: &'a Context,
    pub operation: Ptr<Operation>,
    pub operands: &'a [BufferId],
    pub operand_descriptors: &'a [BufferDescriptor],
    /// Logical operand indexing derived exclusively from xlahlo operations.
    pub operand_views: &'a [crate::OperandView],
    pub results: &'a [BufferId],
    pub result_descriptors: &'a [BufferDescriptor],
    pub plan: &'a mut ExecutionPlanBuilder,
}

/// A priority-selected lowering implementation for one verified xlahlo operation.
///
/// The module compiler invokes these strategies within the generic operation
/// traversal, so a custom kernel can consume generic results and feed later
/// built-in or custom operations without replacing the whole module compiler.
pub trait OperationLoweringStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn priority(&self) -> i32;
    /// Opt into applying `operand_views` when loading operands.
    fn supports_indexed_operands(&self) -> bool {
        false
    }
    fn matches(&self, context: &Context, operation: Ptr<Operation>) -> Result<bool, String>;
    fn lower(&self, request: OperationLoweringRequest<'_>) -> Result<(), PlanError>;
}

#[derive(Default)]
pub(crate) struct OperationStrategyRegistry {
    strategies: Vec<Arc<dyn OperationLoweringStrategy>>,
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
}

impl Default for ModuleCompilerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleCompilerBuilder {
    pub fn new() -> Self {
        let mut builder = Self {
            strategies: KernelOverrideRegistry::new(),
            target_passes: Vec::new(),
            operation_strategies: Arc::new(RwLock::new(OperationStrategyRegistry::default())),
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

    pub fn build(self) -> ModuleCompiler {
        ModuleCompiler {
            strategies: self.strategies,
            target_passes: self.target_passes,
        }
    }
}

/// Verified module compiler with pluggable lowering strategies and target passes.
pub struct ModuleCompiler {
    strategies: KernelOverrideRegistry<ExecutionPlanBuilder, PlanError>,
    target_passes: Vec<Arc<dyn CubeclTargetPass>>,
}

impl ModuleCompiler {
    pub fn builder() -> ModuleCompilerBuilder {
        ModuleCompilerBuilder::new()
    }

    /// Verify and lower a complete xlahlo module to an explicit execution plan.
    pub fn compile(
        &self,
        context: &Context,
        module: Ptr<Operation>,
    ) -> Result<ExecutionPlan, ModuleCompileError> {
        xlahlo::verification::verify(context, module)
            .map_err(|error| ModuleCompileError::Verification(error.to_string()))?;
        let mut target = ExecutionPlanBuilder::new(self.target_passes.clone());
        match self.strategies.lower(context, module, &mut target) {
            Ok(Some(_)) => {
                let plan = target.build()?;
                plan.validate_kernel_definitions()?;
                Ok(plan)
            }
            Ok(None) => Err(ModuleCompileError::Unsupported {
                feature: format!(
                    "no lowering strategy matched module operation {}",
                    Operation::get_opid(module, context)
                ),
            }),
            Err(KernelOverrideError::Lowering {
                source: PlanError::UnsupportedFeature(feature),
                ..
            }) => Err(ModuleCompileError::Unsupported { feature }),
            Err(KernelOverrideError::Lowering {
                source: PlanError::UnsupportedElementType(element_type),
                ..
            }) => Err(ModuleCompileError::Unsupported {
                feature: format!("element type {element_type:?}"),
            }),
            Err(KernelOverrideError::Lowering {
                source: PlanError::OperationStrategy(message),
                ..
            }) => Err(ModuleCompileError::OperationStrategy(message)),
            Err(KernelOverrideError::Lowering {
                source: source @ PlanError::IndexedOperandsUnsupported { .. },
                ..
            }) => Err(ModuleCompileError::Plan(source)),
            Err(error) => Err(ModuleCompileError::Strategy(error)),
        }
    }
}
