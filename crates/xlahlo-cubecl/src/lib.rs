//! CubeCL lowering hooks over the shared pliron dialect IR.
//! Lower supported xlahlo modules into explicit CubeCL execution plans.
//!
//! See the crate README for supported operations, execution-plan ownership,
//! and current lowering limitations.
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

mod float_cast;
mod lowering;
mod module_compiler;
mod plan;
mod view;
pub use module_compiler::{ModuleCompileError, ModuleCompiler, ModuleCompilerBuilder};
pub use module_compiler::{OperationLoweringRequest, OperationLoweringStrategy};
pub use plan::{
    BufferAccess, BufferDescriptor, BufferId, BufferLayout, BufferLifetime, BufferRole,
    CubeclTargetPass, ExecutionPlan, ExecutionPlanBuilder, KernelBinding, KernelDefinitionError,
    KernelStep, LaunchGeometry, PlanError, PlannedBuffer, PlannedKernel, PlannedOutput, SkipReason,
    StaticValue,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistrationError {
    EmptyName,
    DuplicateName(String),
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

#[derive(Debug)]
pub enum KernelOverrideError<E> {
    Matcher {
        name: String,
        message: String,
    },
    Ambiguous {
        operation: String,
        candidates: Vec<String>,
    },
    Lowering {
        name: String,
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

impl<E: std::error::Error + 'static> std::error::Error for KernelOverrideError<E> {}

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
    pub fn name(&self) -> &str {
        &self.entry.name
    }

    pub fn priority(&self) -> i32 {
        self.entry.priority
    }

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
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn len(&self) -> usize {
        self.overrides.len()
    }

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
