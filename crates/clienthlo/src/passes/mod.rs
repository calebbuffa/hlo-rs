//! CHLO legalization using pliron's pass manager and dialect-conversion driver.
//!
//! Successful legalization removes every `chlo.*` operation under the supplied
//! root and verifies the resulting IR. Unsupported decompositions are errors,
//! not partial success. Existing non-CHLO operations are left in place.

mod builder;
mod conversion;
mod math;
mod ragged;
mod scan;

use crate::{Diagnostic, Result};
use pliron::{
    context::{Context, Ptr},
    irbuild::{IRStatus, dialect_conversion::apply_dialect_conversion},
    linked_list::ContainsLinkedList,
    operation::Operation,
    pass::{AnalysisManager, Pass, PassResult},
    printable::Printable,
};

/// Converts all pinned upstream CHLO operation kinds to native StableHLO operations.
///
/// Run this pass on a containing module or function, not an individual CHLO
/// operation. Successful execution verifies the result and leaves no nested
/// `chlo.*` operations. Replacements contain only StableHLO operations; existing
/// foreign operations are preserved rather than legalized.
///
/// The pass preflights every nested CHLO operation before rewriting, so known
/// unsupported inputs fail without modifying the IR. An unexpected rewrite or
/// verification error can leave partially changed IR; errors are not transactional.
/// Source/target element domains and upstream shape restrictions still apply;
/// see the crate README for their scope.
///
/// # Errors
///
/// [`Pass::run`] returns [`Diagnostic::UnsupportedLowering`] for a rejected root,
/// unsupported source representation, or metadata without a mapping. Invalid
/// input and replacement IR return verification errors. Analysis caches are
/// invalidated when an error follows mutation.
#[derive(Default)]
pub struct ChloToStablehloPass;

impl Pass for ChloToStablehloPass {
    fn name(&self) -> &str {
        "chlo-legalize-to-stablehlo"
    }

    fn run(
        &mut self,
        root: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        if is_chlo(ctx, root) {
            return Err(unsupported(
                ctx,
                root,
                "run legalization on a containing module or function",
            ));
        }
        stablehlo::verification::verify_module_fast(ctx, root)?;
        for operation in operations(ctx, root) {
            if is_chlo(ctx, operation) {
                conversion::preflight(ctx, operation)?;
            }
        }
        let mut result = PassResult::default();
        result.ir_changed = IRStatus::Changed;
        let ir_changed = apply_dialect_conversion(ctx, &mut conversion::Conversion, root)
            .inspect_err(|_| analyses.retain_preserved(&result))?;
        for operation in operations(ctx, root) {
            if is_chlo(ctx, operation) {
                analyses.retain_preserved(&result);
                return Err(unsupported(
                    ctx,
                    operation,
                    "CHLO operation remains after legalization",
                ));
            }
        }
        result.ir_changed = ir_changed;
        stablehlo::verification::verify_module_fast(ctx, root)
            .inspect_err(|_| analyses.retain_preserved(&result))?;
        Ok(result)
    }
}

fn operations(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut pending = vec![root];
    let mut visited = Vec::new();
    while let Some(operation) = pending.pop() {
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
        visited.push(operation);
    }
    visited
}

fn is_chlo(ctx: &Context, operation: Ptr<Operation>) -> bool {
    Operation::get_opid(operation, ctx)
        .to_string()
        .starts_with("chlo.")
}

fn unsupported(
    ctx: &Context,
    operation: Ptr<Operation>,
    reason: &'static str,
) -> pliron::result::Error {
    Diagnostic::UnsupportedLowering {
        operation: Operation::get_opid(operation, ctx).to_string(),
        location: operation.deref(ctx).loc().disp(ctx).to_string(),
        reason,
    }
    .into()
}
