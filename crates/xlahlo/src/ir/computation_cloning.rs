//! Detached computation cloning backed by the shared CloneWithReplacements engine.
//!
//! Based on OpenXLA's `HloComputation::CloneWithReplacements` in
//! `hlo/ir/hlo_computation.cc`.
use super::InstructionSequence;
use super::clone_with_replacements as engine;
use crate::{
    ops::ComputationOp,
    verification::{VerifyOptions, require},
};
use pliron::{
    context::{Context, Ptr},
    irbuild::cloning::IrMapping,
    op::Op,
    operation::Operation,
    result::Result,
};
use std::collections::{HashMap, HashSet};

/// Select the clone's root before applying instruction replacements.
#[derive(Clone, Debug, Default)]
pub enum ComputationCloneRoot {
    /// Keep the original root, or its selected replacement.
    #[default]
    Keep,
    /// Use a source instruction, or its selected replacement.
    Instruction(Ptr<Operation>),
    /// Construct a tuple from selected source instructions, including an empty tuple.
    Tuple(Vec<Ptr<Operation>>),
}

/// Instruction substitutions for one detached computation clone.
///
/// Replacement lookup occurs once at each operand edge, not recursively.
/// Templates must be detached; their external dependency closure is cloned,
/// never captured as live caller SSA. The source and templates remain unchanged.
/// A removed instruction must have no surviving data use. Replacement parameter
/// indices, including extra parameters, must form a contiguous signature.
#[derive(Default)]
pub struct ComputationReplacements {
    /// Source instructions to replace or remove. `Some` selects a detached
    /// replacement template; `None` removes the source instruction when it has
    /// no surviving data uses.
    pub instructions: HashMap<Ptr<Operation>, Option<Ptr<Operation>>>,
    /// Independent parameter Clone() occurrences, not shared mapping entries.
    /// Attached extras are allowed and retain their original owners.
    pub extra_parameters: Vec<Ptr<Operation>>,
    /// Pointer-identified control edges for otherwise unscoped detached templates.
    pub template_control_edges: Vec<(Ptr<Operation>, Ptr<Operation>)>,
    /// Root policy applied after instruction substitutions and cloning.
    pub root: ComputationCloneRoot,
}

/// An owned detached computation and its async-owned dependency bundle.
///
/// Publish every computation into the same destination module, or call
/// [`Self::dispose`] while all remain detached. There is no automatic Drop:
/// disposal requires the allocating Context. Partial publication is the caller's
/// responsibility; this low-level API does not rewrite callsites transactionally.
#[must_use = "publish the complete detached bundle or dispose it with its Context"]
pub struct ComputationClone {
    /// The detached clone of the requested source computation.
    pub computation: ComputationOp,
    /// Detached computations cloned to satisfy owned asynchronous dependencies.
    pub embedded_computations: Vec<ComputationOp>,
    /// Only async-owned cloned dependencies receive sequences here, when their
    /// source wrappers were scheduled. No main clone sequence is invented.
    /// Install these alongside the computations, only in a scheduled module.
    pub embedded_schedules: Vec<InstructionSequence>,
    /// Maps selected templates/dependencies, not replaced originals, to clones.
    /// Independent extra parameter occurrences intentionally have no entry.
    pub mapping: IrMapping,
}

impl ComputationClone {
    /// Dispose a collectively owned detached batch, or return every owner intact.
    ///
    /// Batch outputs can reference an async dependency owned by another output.
    /// Keep them together: publish all members or dispose the complete batch.
    /// Every member is checked before any erasure; partial publication returns
    /// the original owners and an error without changing either attached or
    /// detached computations. There is no verifier-policy selection here.
    #[allow(
        clippy::result_large_err,
        reason = "Disposal errors must return every original owner intact."
    )]
    pub fn dispose_all(
        clones: Vec<Self>,
        ctx: &mut Context,
    ) -> std::result::Result<(), (Vec<Self>, pliron::result::Error)> {
        let detached = clones.iter().all(|clone| {
            std::iter::once(&clone.computation)
                .chain(&clone.embedded_computations)
                .all(|computation| {
                    computation
                        .get_operation()
                        .deref(ctx)
                        .get_parent_block()
                        .is_none()
                })
        });
        if let Err(error) = require(
            detached,
            "batch disposal requires every computation to be detached",
        ) {
            return Err((clones, error));
        }
        erase_batch(ctx, clones);
        Ok(())
    }

    /// Dispose the complete detached bundle. On an attached member, return both
    /// the error and the still-owned bundle without erasing any computation.
    pub fn dispose(
        self,
        ctx: &mut Context,
    ) -> std::result::Result<(), (Box<Self>, pliron::result::Error)> {
        let detached = std::iter::once(&self.computation)
            .chain(&self.embedded_computations)
            .all(|computation| {
                computation
                    .get_operation()
                    .deref(ctx)
                    .get_parent_block()
                    .is_none()
            });
        if let Err(error) = require(
            detached,
            "dispose requires every cloned computation to be detached",
        ) {
            return Err((Box::new(self), error));
        }
        Operation::erase(self.computation.get_operation(), ctx);
        for computation in self.embedded_computations {
            Operation::erase(computation.get_operation(), ctx);
        }
        Ok(())
    }
}

fn erase_batch(ctx: &mut Context, clones: Vec<ComputationClone>) {
    let mut erased = HashSet::new();
    // Named computation dependencies are symbolic, not cross-computation SSA.
    for clone in clones {
        for computation in std::iter::once(clone.computation).chain(clone.embedded_computations) {
            if erased.insert(computation.get_operation()) {
                Operation::erase(computation.get_operation(), ctx);
            }
        }
    }
}

/// One request in a same-module detached clone batch.
///
/// Source computations must be distinct and attached to the same module.
/// All sources and replacement templates must outlive the complete batch call.
pub struct ComputationCloneRequest<'a> {
    /// Attached source computation to clone.
    pub source: &'a ComputationOp,
    /// Requested symbol for the clone; the actual name may be uniquified.
    pub name: crate::ir::ComputationName,
    /// Replacement templates, parameter substitutions, and selected clone root.
    pub replacements: &'a ComputationReplacements,
}

/// Clone multiple computations with shared name reservations and dependency cache.
///
/// Uses the same engine as [`clone_computation_with_replacements`]. No source,
/// template, or module is changed. On failure, every allocation from earlier
/// requests is erased, as well as the failing request's own staged allocations.
/// Legacy verifier callbacks are rejected before any cloning, including an
/// empty batch. `None` remains structural cloning, not implicit verification.
///
/// Output order matches request order. Read actual uniquified symbols from the
/// returned computations. Names are unique across the destination and every
/// detached output. Async-owned dependencies and their schedules can be shared
/// across outputs: the first bundle owns a dependency used by later bundles.
/// The returned vector therefore has **collective ownership**. Publish every
/// primary and embedded computation together, or use
/// [`ComputationClone::dispose_all`]; do not dispose an individual referenced
/// bundle. Partial publication is not rolled back automatically.
///
/// Main clone schedules are not invented. Install embedded schedules only when
/// publishing to an already scheduled module. Callsite/signature publication
/// and its final verification remain consumer responsibilities.
pub fn clone_computations_with_replacements(
    ctx: &mut Context,
    requests: &[ComputationCloneRequest<'_>],
    verification: Option<VerifyOptions>,
) -> Result<Vec<ComputationClone>> {
    if let Some(options) = verification {
        super::instruction_replacement::require_rehearsable_policy(options)?;
    }
    let mut module = None;
    let mut sources = HashSet::new();
    for request in requests {
        require(
            sources.insert(request.source.get_operation()),
            "duplicate batch clone source",
        )?;
        let owner =
            engine::owning_module(ctx, request.source.get_operation()).ok_or_else(|| {
                crate::verification::error("batch source must be attached to a module")
            })?;
        require(
            module.is_none_or(|module| module == owner),
            "batch sources must share a module",
        )?;
        module = Some(owner);
    }
    let mut context = engine::ComputationCloneContext::new(module, "clone".into());
    let mut outputs = Vec::with_capacity(requests.len());
    for request in requests {
        let replacements = engine_replacements(request.replacements);
        match engine::clone_in_context(
            ctx,
            request.source,
            request.name.clone(),
            &replacements,
            verification,
            &mut context,
        ) {
            Ok(clone) => outputs.push(public_clone(clone)),
            Err(error) => {
                drop(context);
                erase_batch(ctx, outputs);
                return Err(error);
            }
        }
    }
    Ok(outputs)
}

fn engine_replacements(replacements: &ComputationReplacements) -> engine::Replacements {
    engine::Replacements {
        instructions: replacements.instructions.clone(),
        extra_parameters: replacements.extra_parameters.clone(),
        template_control_edges: replacements.template_control_edges.clone(),
        root: match &replacements.root {
            ComputationCloneRoot::Keep => engine::CloneRoot::Keep,
            ComputationCloneRoot::Instruction(root) => engine::CloneRoot::Instruction(*root),
            ComputationCloneRoot::Tuple(roots) => engine::CloneRoot::Tuple(roots.clone()),
        },
    }
}

fn public_clone(cloned: engine::ComputationClone) -> ComputationClone {
    ComputationClone {
        computation: cloned.computation,
        embedded_computations: cloned.embedded_computations,
        embedded_schedules: cloned.embedded_schedules,
        mapping: cloned.mapping,
    }
}

/// Clone a computation without changing the source, templates, or module.
///
/// Retained parameters are replaced with caller-supplied Parameter templates
/// carrying their new indices; tuple-result compaction uses
/// [`ComputationCloneRoot::Tuple`]. Callsite operands/results and their aliases
/// must be rebuilt explicitly by the consumer before final module verification.
/// Intrinsic instruction aliases are copied, not inferred or rebased.
///
/// Ordinary same-module named callees remain shared. AsyncStart owns a fresh
/// wrapped-computation bundle. Logical order, nested Fusion parameters, controls,
/// typed attributes and OriginalValue follow the shared cloning engine.
///
/// `None` performs structural cloning without selecting a verifier policy.
/// `Some` validates in the source module context with the supplied policy.
/// Legacy Context/Ptr-sensitive verifier callbacks cannot be rehearsed faithfully
/// and are rejected. Successful detached verification is not verification of
/// later publication or of callsites with a changed signature.
pub fn clone_computation_with_replacements(
    ctx: &mut Context,
    source: &ComputationOp,
    name: impl Into<crate::ir::ComputationName>,
    replacements: &ComputationReplacements,
    verification: Option<VerifyOptions>,
) -> Result<ComputationClone> {
    let name = name.into();
    if let Some(options) = verification {
        super::instruction_replacement::require_rehearsable_policy(options)?;
    }
    let replacements = engine_replacements(replacements);
    let cloned = engine::clone_with_replacements(ctx, source, name, &replacements, verification)?;
    Ok(public_clone(cloned))
}
