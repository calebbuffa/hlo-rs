//! Shared native-IREE fixture capture for dialect integration tests.

use pliron::{
    common_traits::Named,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    r#type::Typed,
    value::Value,
};
use std::collections::HashSet;

/// Captures a printed operation in a closed MLIR module when requested.
///
/// The function signature explicitly supplies every external SSA value,
/// including captures used only by nested regions. Fixtures are deterministic
/// and distinct across tests. The oracle rejects unregistered dialects and
/// verifies whole functions, not isolated snippets.
pub fn record(ctx: &Context, op: Ptr<Operation>, assembly: &str) {
    let Some(directory) = std::env::var_os("PLIRON_IREE_FIXTURES") else {
        return;
    };
    fn walk(
        ctx: &Context,
        op: Ptr<Operation>,
        defined: &mut HashSet<Value>,
        used: &mut Vec<Value>,
    ) {
        let raw = op.deref(ctx);
        defined.extend(raw.results());
        used.extend(raw.operands());
        for region in raw.regions() {
            for block in region.deref(ctx).iter(ctx) {
                defined.extend(block.deref(ctx).arguments());
                for nested in block.deref(ctx).iter(ctx) {
                    walk(ctx, nested, defined, used);
                }
            }
        }
    }
    let mut defined = HashSet::new();
    let mut used = Vec::new();
    walk(ctx, op, &mut defined, &mut used);
    let mut seen = HashSet::new();
    let external: Vec<_> = used
        .into_iter()
        .filter(|v| !defined.contains(v) && seen.insert(*v))
        .collect();
    let parameters = external
        .iter()
        .map(|v| {
            let ty = String::try_from(pliron_mlir_interop::MlirPrinter::new(ctx, &v.get_type(ctx)))
                .unwrap();
            format!("%{}: {ty}", v.unique_name(ctx))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let name = Operation::get_op_dyn(op, ctx).get_opid().to_string();
    let assembly = if name == "tensor.parallel_insert_slice" {
        let dest = op.deref(ctx).get_operand(1);
        let ty = String::try_from(pliron_mlir_interop::MlirPrinter::new(
            ctx,
            &dest.get_type(ctx),
        ))
        .unwrap();
        let nested = assembly.replace(&format!("%{}", dest.unique_name(ctx)), "%shared");
        format!(
            "%parallel_result = scf.forall (%iv) in (1) shared_outs(%shared = %{}) -> ({ty}) {{\nscf.forall.in_parallel {{\n{nested}\n}}\n}}",
            dest.unique_name(ctx)
        )
    } else {
        assembly.to_owned()
    };
    let module =
        format!("module {{\nfunc.func @fixture({parameters}) {{\n{assembly}\nreturn\n}}\n}}\n");
    // Stable FNV-1a, not RandomState (which would change names between runs).
    let hash = module.bytes().fold(0xcbf29ce484222325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join(format!("{name}-{hash:016x}.mlir")), module).unwrap();
}
