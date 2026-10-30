//! Compile supported native XLA HLO modules into owned CubeCL execution plans.
//!
//! See the crate README for supported operations, execution-plan ownership,
//! and current lowering limitations.
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod compile_profile;
mod float_cast;
mod fusion_policy;
mod lowering;
mod module_compiler;
mod plan;
mod view;
pub use float_cast::cast_float as emit_scalar_float_cast;
pub use fusion_policy::{FusionBudget, FusionEstimate, TargetFusionPolicy};
pub use lowering::checked_offsets::add_checked_offset_validation;
pub use lowering::emit_store as emit_scalar_store;
pub use lowering::reduction_fusion::normalize_narrow_float_casts;
pub use lowering::scalar_body::OwnedScalarBody;
pub use module_compiler::{
    ModuleCompileError, ModuleCompiler, ModuleCompilerBuilder, UnsupportedKind,
};
pub use module_compiler::{
    OperationLoweringRequest, OperationLoweringStrategy, OrderedReductionGeometry,
    OrderedReductionLoweringBody, ScalarLoweringBody,
};
pub use plan::{
    BufferAccess, BufferDescriptor, BufferId, BufferInitializer, BufferLayout, BufferLifetime,
    BufferRole, CHECKED_STATUS_COUNTER_OVERFLOW, CHECKED_STATUS_DESCENDING,
    CHECKED_STATUS_OUT_OF_BOUNDS, CHECKED_STATUS_PENDING, CHECKED_STATUS_SUCCESS,
    CHECKED_STATUS_VERSION, CHECKED_STATUS_WORDS, CheckedCompletion, CubeclTargetPass, DeviceCount,
    ExecutionPlan, ExecutionPlanBuilder, InitializerRequirements, KernelArgumentLayout,
    KernelBinding, KernelDefinitionError, KernelStep, LaunchGeometry, MemoryRequirements,
    OutputContract, PlanError, PlannedBuffer, PlannedKernel, PlannedOutput, PrivateStorageGroup,
    SkipReason, StaticOutputView, StaticValue, StorageUpdate, TensorArgument,
};
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
};
pub use view::{IndexTransform, IndexTransformKind, OperandView, OutputIndexMap, OutputView};

type Matcher = dyn Fn(&Context, Ptr<Operation>) -> Result<bool, String>;
type Lowering<T, E> = dyn Fn(&Context, Ptr<Operation>, &mut T) -> Result<(), E>;

struct Override<T: ?Sized, E> {
    name: String,
    priority: i32,
    matches: Box<Matcher>,
    lower: Box<Lowering<T, E>>,
}

/// Failure to register a named lowering strategy or access its registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistrationError {
    /// The supplied name is empty or contains only whitespace.
    EmptyName,
    /// A strategy with this exact name is already registered.
    DuplicateName(String),
    /// A panic while holding a strategy registry lock poisoned that registry.
    PoisonedRegistry,
}

impl std::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyName => f.write_str("kernel override name cannot be empty"),
            Self::DuplicateName(name) => write!(f, "kernel override {name} is already registered"),
            Self::PoisonedRegistry => f.write_str("operation strategy registry lock was poisoned"),
        }
    }
}

impl std::error::Error for RegistrationError {}

/// Failure to select or execute a custom lowering override.
///
/// Lowering failures retain the backend error as their standard error source.
#[derive(Debug)]
pub enum KernelOverrideError<E> {
    /// A matcher failed; this is not treated as a non-match.
    Matcher {
        /// Registered name of the failing matcher.
        name: String,
        /// Diagnostic supplied by the matcher.
        message: String,
    },
    /// Multiple matching overrides share the highest priority.
    Ambiguous {
        /// Source operation identifier.
        operation: String,
        /// Names of equally ranked matching overrides.
        candidates: Vec<String>,
    },
    /// The selected override failed while constructing its plan.
    Lowering {
        /// Registered name of the selected override.
        name: String,
        /// Original backend failure.
        source: E,
    },
}

impl<E: std::fmt::Display> std::fmt::Display for KernelOverrideError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Matcher { name, message } => {
                write!(f, "matching kernel override {name} failed: {message}")
            }
            Self::Ambiguous {
                operation,
                candidates,
            } => write!(
                f,
                "ambiguous kernel overrides for {operation}: {}",
                candidates.join(", ")
            ),
            Self::Lowering { name, source } => {
                write!(f, "kernel override {name} failed to lower: {source}")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for KernelOverrideError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lowering { source, .. } => Some(source),
            Self::Matcher { .. } | Self::Ambiguous { .. } => None,
        }
    }
}

/// Priority-ordered custom-kernel registry.
///
/// Matchers inspect only the shared pliron operation and context. `T` is the
/// backend lowering state supplied by the eventual CubeCL code generator.
pub struct KernelOverrideRegistry<T: ?Sized, E> {
    overrides: Vec<Override<T, E>>,
}

/// A registry match selected for one post-canonicalization HLO operation.
pub struct SelectedOverride<'a, T: ?Sized, E> {
    entry: &'a Override<T, E>,
    operation: Ptr<Operation>,
}

impl<'a, T: ?Sized, E> SelectedOverride<'a, T, E> {
    /// Registered name of the uniquely selected override.
    pub fn name(&self) -> &str {
        &self.entry.name
    }

    /// Selection priority; higher values take precedence.
    pub fn priority(&self) -> i32 {
        self.entry.priority
    }

    /// Run the selected lowerer and return its name on success.
    ///
    /// A failure retains its source but does not roll back mutations to `target`.
    /// The operation must still belong to `context` and remain valid.
    pub fn lower(
        &self,
        context: &Context,
        target: &mut T,
    ) -> Result<&'a str, KernelOverrideError<E>> {
        (self.entry.lower)(context, self.operation, target).map_err(|source| {
            KernelOverrideError::Lowering {
                name: self.entry.name.clone(),
                source,
            }
        })?;
        Ok(&self.entry.name)
    }
}

impl<T: ?Sized, E> Default for KernelOverrideRegistry<T, E> {
    fn default() -> Self {
        Self {
            overrides: Vec::new(),
        }
    }
}

impl<T: ?Sized, E> KernelOverrideRegistry<T, E> {
    /// Create an empty registry with no implicit fallback.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a uniquely named matcher and lowerer.
    ///
    /// Higher priorities win. Empty names and duplicate exact names are rejected.
    /// Callbacks must not retain borrowed source IR beyond its context lifetime.
    pub fn register<M, L>(
        &mut self,
        name: impl Into<String>,
        priority: i32,
        matches: M,
        lower: L,
    ) -> Result<(), RegistrationError>
    where
        M: Fn(&Context, Ptr<Operation>) -> Result<bool, String> + 'static,
        L: Fn(&Context, Ptr<Operation>, &mut T) -> Result<(), E> + 'static,
    {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(RegistrationError::EmptyName);
        }
        if self.overrides.iter().any(|entry| entry.name == name) {
            return Err(RegistrationError::DuplicateName(name));
        }
        self.overrides.push(Override {
            name,
            priority,
            matches: Box::new(matches),
            lower: Box::new(lower),
        });
        Ok(())
    }

    /// Lowers with the unique highest-priority matching override.
    ///
    /// No match returns `Ok(None)` so the caller can use generic lowering.
    /// Equal-priority matches are rejected rather than resolved by registration
    /// order, keeping kernel selection deterministic.
    pub fn lower(
        &self,
        context: &Context,
        operation: Ptr<Operation>,
        target: &mut T,
    ) -> Result<Option<&str>, KernelOverrideError<E>> {
        let Some(selected) = self.select(context, operation)? else {
            return Ok(None);
        };
        selected.lower(context, target).map(Some)
    }

    /// Selects an override independently from lowering so callers can make
    /// the decision once after canonicalization and retain that choice.
    pub fn select(
        &self,
        context: &Context,
        operation: Ptr<Operation>,
    ) -> Result<Option<SelectedOverride<'_, T, E>>, KernelOverrideError<E>> {
        let Some(index) = self.select_index(context, operation)? else {
            return Ok(None);
        };
        Ok(Some(SelectedOverride {
            entry: &self.overrides[index],
            operation,
        }))
    }

    /// Number of registered overrides.
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Whether no overrides are registered.
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    fn select_index(
        &self,
        context: &Context,
        operation: Ptr<Operation>,
    ) -> Result<Option<usize>, KernelOverrideError<E>> {
        let mut best_priority = None;
        let mut best = Vec::new();
        for (index, entry) in self.overrides.iter().enumerate() {
            let matches = (entry.matches)(context, operation).map_err(|message| {
                KernelOverrideError::Matcher {
                    name: entry.name.clone(),
                    message,
                }
            })?;
            if !matches {
                continue;
            }
            match best_priority {
                None => {
                    best_priority = Some(entry.priority);
                    best.push(index);
                }
                Some(priority) if entry.priority > priority => {
                    best_priority = Some(entry.priority);
                    best.clear();
                    best.push(index);
                }
                Some(priority) if entry.priority == priority => best.push(index),
                Some(_) => {}
            }
        }

        if best.len() > 1 {
            let operation_name = Operation::get_opid(operation, context).to_string();
            return Err(KernelOverrideError::Ambiguous {
                operation: operation_name,
                candidates: best
                    .into_iter()
                    .map(|index| self.overrides[index].name.clone())
                    .collect(),
            });
        }
        Ok(best.into_iter().next())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{builtin::ops::ModuleOp, identifier::Identifier, op::Op};

    #[test]
    fn highest_priority_matching_override_is_selected_after_matching() {
        let mut context = Context::new();
        let operation = ModuleOp::new(&mut context, Identifier::new("registry")).get_operation();
        let mut registry = KernelOverrideRegistry::<usize, String>::new();
        registry
            .register(
                "generic",
                1,
                |_, _| Ok(true),
                |_, _, target| {
                    *target = 1;
                    Ok(())
                },
            )
            .unwrap();
        registry
            .register(
                "tuned",
                10,
                |_, _| Ok(true),
                |_, _, target| {
                    *target = 10;
                    Ok(())
                },
            )
            .unwrap();
        let selected = registry.select(&context, operation).unwrap().unwrap();
        assert_eq!(selected.name(), "tuned");
        assert_eq!(selected.priority(), 10);
        let mut target = 0;
        assert_eq!(selected.lower(&context, &mut target).unwrap(), "tuned");
        assert_eq!(target, 10);
    }

    #[test]
    fn equal_priority_matches_are_rejected_and_no_match_falls_through() {
        let mut context = Context::new();
        let operation = ModuleOp::new(&mut context, Identifier::new("registry")).get_operation();
        let mut registry = KernelOverrideRegistry::<(), String>::new();
        registry
            .register("first", 4, |_, _| Ok(true), |_, _, _| Ok(()))
            .unwrap();
        registry
            .register("second", 4, |_, _| Ok(true), |_, _, _| Ok(()))
            .unwrap();
        assert!(matches!(
            registry.select(&context, operation),
            Err(KernelOverrideError::Ambiguous { candidates, .. })
                if candidates == ["first", "second"]
        ));

        let mut fallback = KernelOverrideRegistry::<(), String>::new();
        fallback
            .register("not-applicable", 1, |_, _| Ok(false), |_, _, _| Ok(()))
            .unwrap();
        assert!(fallback.select(&context, operation).unwrap().is_none());
    }

    #[test]
    fn matcher_errors_are_not_silently_treated_as_no_match() {
        let mut context = Context::new();
        let operation = ModuleOp::new(&mut context, Identifier::new("registry")).get_operation();
        let mut registry = KernelOverrideRegistry::<(), String>::new();
        registry
            .register(
                "broken-matcher",
                1,
                |_, _| Err("unsupported shape metadata".to_owned()),
                |_, _, _| Ok(()),
            )
            .unwrap();

        assert!(matches!(
            registry.select(&context, operation),
            Err(KernelOverrideError::Matcher { name, message })
                if name == "broken-matcher" && message == "unsupported shape metadata"
        ));
    }
}
