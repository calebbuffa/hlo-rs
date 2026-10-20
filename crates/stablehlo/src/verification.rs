//! Local operation checks and fast verification for complete StableHLO modules.

use crate::types::*;
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    region::Region as PlironRegion,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
    verify_err,
};

pub(crate) fn invalid(op: &dyn Op, ctx: &Context, msg: &'static str) -> Result<()> {
    let location = op.loc(ctx);
    verify_err!(
        location.clone(),
        crate::Diagnostic::InvalidOperation {
            operation: op.get_opid().to_string(),
            location: format!("{location:?}"),
            reason: msg,
        }
    )
}
pub(crate) fn one_tensor(op: &dyn Op, ctx: &Context, n: usize, same: bool) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != n || ir.get_num_results() != 1 {
        return invalid(op, ctx, "operand/result count mismatch");
    }
    let result_type = ir.get_result(0).get_type(ctx);
    if !valid_tensor_type(ctx, result_type) {
        return invalid(op, ctx, "result must be a supported static ranked tensor");
    }
    for i in 0..n {
        let input_type = ir.get_operand(i).get_type(ctx);
        if same {
            if input_type != result_type {
                return invalid(op, ctx, "operand and result tensor types must agree");
            }
        } else if !valid_tensor_type(ctx, input_type) {
            return invalid(op, ctx, "operand must be a supported static ranked tensor");
        }
    }
    Ok(())
}

/// Fast verification entry point. It checks every operation locally and the SSA
/// use-before-def rule in one linear walk rather than invoking pliron's quadratic verifier.
pub fn verify_module_fast(ctx: &Context, root: Ptr<Operation>) -> Result<()> {
    use pliron::linked_list::ContainsLinkedList;
    use std::collections::{HashMap, HashSet};
    // Arena identities are internal integers, not attacker-selected strings.
    // Avoid SipHash on the per-operand dominance-index hot path.
    #[derive(Default)]
    struct ArenaHasher(u64);
    impl std::hash::Hasher for ArenaHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            for &byte in bytes {
                self.write_u64(u64::from(byte));
            }
        }
        fn write_u64(&mut self, value: u64) {
            self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517cc1b727220a95);
        }
        fn write_usize(&mut self, value: usize) {
            self.write_u64(value as u64);
        }
    }
    type ArenaMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<ArenaHasher>>;
    type Block = Ptr<BasicBlock>;
    type Region = Ptr<PlironRegion>;
    #[derive(Default)]
    struct Index {
        operations: ArenaMap<Ptr<Operation>, (Block, usize)>,
        blocks: ArenaMap<Block, (Region, usize)>,
        dominators: ArenaMap<Region, Vec<Vec<bool>>>,
        // These operations have no semantic attributes: their local contracts
        // depend only on opcode and the three interned operand/result types.
        arithmetic: HashSet<(bool, [TypeHandle; 3])>,
        arithmetic_last: [Option<[TypeHandle; 3]>; 2],
        regions: HashSet<Region>,
        checked_types: HashSet<TypeHandle>,
        last_type: Option<TypeHandle>,
    }
    impl Index {
        fn check_type(&mut self, ctx: &Context, ty: TypeHandle) -> Result<()> {
            if self.last_type != Some(ty) && !self.checked_types.contains(&ty) {
                ty.deref(ctx).verify(ctx)?;
                self.checked_types.insert(ty);
            }
            self.last_type = Some(ty);
            Ok(())
        }
        fn region(
            &mut self,
            ctx: &Context,
            region: Region,
            pending: &mut Vec<Region>,
        ) -> Result<()> {
            if !self.regions.insert(region) {
                return verify_err!(
                    pliron::location::Location::Unknown,
                    crate::Diagnostic::CycleDetected {
                        graph: "region containment",
                    }
                );
            }
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            self.blocks.reserve(blocks.len());
            for (bi, &block) in blocks.iter().enumerate() {
                self.blocks.insert(block, (region, bi));
                let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
                self.operations.reserve(operations.len());
                for (position, ptr) in operations.into_iter().enumerate() {
                    self.operations.insert(ptr, (block, position));
                    pending.extend(ptr.deref(ctx).regions());
                }
            }
            // The overwhelmingly common single-block case needs no matrix.
            if blocks.len() <= 1 {
                return Ok(());
            }
            let mut predecessors = vec![Vec::new(); blocks.len()];
            for (bi, &block) in blocks.iter().enumerate() {
                if let Some(last) = block.deref(ctx).iter(ctx).last() {
                    for successor in last.deref(ctx).successors() {
                        if let Some(&(r, si)) = self.blocks.get(&successor)
                            && r == region
                        {
                            predecessors[si].push(bi);
                        }
                    }
                }
            }
            let mut dom = vec![vec![true; blocks.len()]; blocks.len()];
            dom[0].fill(false);
            dom[0][0] = true;
            loop {
                let mut changed = false;
                for bi in 1..blocks.len() {
                    let mut next = vec![!predecessors[bi].is_empty(); blocks.len()];
                    for &pred in &predecessors[bi] {
                        for (slot, &value) in next.iter_mut().zip(&dom[pred]) {
                            *slot &= value;
                        }
                    }
                    next[bi] = true;
                    if next != dom[bi] {
                        dom[bi] = next;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            self.dominators.insert(region, dom);
            Ok(())
        }
        #[inline]
        fn dominates(
            &self,
            ctx: &Context,
            value: Value,
            mut use_block: Block,
            mut position: usize,
        ) -> bool {
            let defining_op = value.defining_op();
            let definition = if let Some(def) = defining_op {
                self.operations.get(&def).copied()
            } else {
                value.defining_block().map(|block| (block, 0))
            };
            let Some((def_block, def_position)) = definition else {
                return false;
            };
            if def_block == use_block {
                return defining_op.is_none() || def_position < position;
            }
            let Some(&(def_region, def_index)) = self.blocks.get(&def_block) else {
                return false;
            };
            loop {
                let Some(&(region, use_index)) = self.blocks.get(&use_block) else {
                    return false;
                };
                if def_region == region {
                    if def_block == use_block {
                        return defining_op.is_none() || def_position < position;
                    }
                    return self
                        .dominators
                        .get(&region)
                        .and_then(|dom| dom.get(use_index))
                        .and_then(|row| row.get(def_index))
                        .copied()
                        .unwrap_or(false);
                }
                let anchor = region.deref(ctx).get_parent_op();
                let Some(&(block, parent_position)) = self.operations.get(&anchor) else {
                    return false;
                };
                use_block = block;
                position = parent_position;
            }
        }
    }
    type FunctionSignature = (Vec<TypeHandle>, Vec<TypeHandle>);
    type SymbolTable = HashMap<String, Ptr<Operation>>;
    type Signatures = HashMap<Ptr<Operation>, FunctionSignature>;
    fn walk_region(
        ctx: &Context,
        region: Ptr<PlironRegion>,
        symbols: Option<&SymbolTable>,
        signatures: &Signatures,
        index: &mut Index,
    ) -> Result<()> {
        use pliron::linked_list::ContainsLinkedList;
        let mut pending = vec![region];
        let mut visited_regions = HashSet::new();
        while let Some(region) = pending.pop() {
            if !visited_regions.insert(region) {
                return verify_err!(
                    pliron::location::Location::Unknown,
                    crate::Diagnostic::CycleDetected {
                        graph: "region containment",
                    }
                );
            }
            for block_ptr in region.deref(ctx).iter(ctx) {
                for arg in block_ptr.deref(ctx).arguments() {
                    index.check_type(ctx, arg.get_type(ctx))?;
                }
                for (i, ptr) in block_ptr.deref(ctx).iter(ctx).enumerate() {
                    let op = ptr.deref(ctx);
                    for ty in op.result_types() {
                        index.check_type(ctx, ty)?;
                    }
                    let mut previous_operand = None;
                    for v in op.operands() {
                        if previous_operand != Some(v) && !index.dominates(ctx, v, block_ptr, i) {
                            return invalid(
                                &*Operation::get_op_dyn(ptr, ctx),
                                ctx,
                                "operand definition must dominate its use",
                            );
                        }
                        previous_operand = Some(v);
                    }
                    let add = Operation::is_op::<crate::ops::AddOp>(ptr, ctx);
                    let multiply = Operation::is_op::<crate::ops::MultiplyOp>(ptr, ctx);
                    if (add || multiply)
                        && op.get_num_operands() == 2
                        && op.get_num_results() == 1
                        && op.regions().next().is_none()
                        && op.successors().next().is_none()
                        && op.attributes.0.is_empty()
                    {
                        let lhs = op.get_operand(0);
                        let rhs = op.get_operand(1);
                        let lhs_type = lhs.get_type(ctx);
                        let rhs_type = if lhs == rhs {
                            lhs_type
                        } else {
                            rhs.get_type(ctx)
                        };
                        let types = [lhs_type, rhs_type, op.get_type(0)];
                        let slot = usize::from(multiply);
                        let key = (multiply, types);
                        if index.arithmetic_last[slot] != Some(types)
                            && !index.arithmetic.contains(&key)
                        {
                            let dynop = Operation::get_op_dyn(ptr, ctx);
                            dynop.verify(ctx)?;
                            dynop.verify_interfaces(ctx)?;
                            index.arithmetic.insert(key);
                        }
                        index.arithmetic_last[slot] = Some(types);
                        continue;
                    }
                    let dynop = Operation::get_op_dyn(ptr, ctx);
                    dynop.verify(ctx)?;
                    dynop.verify_interfaces(ctx)?;
                    if let Some(call) = dynop.downcast_ref::<crate::ops::CallOp>()
                        && let (Some(name), Some(symbols)) =
                            (call.get_attr_stablehlo_call_callee(ctx), symbols)
                    {
                        let Some(callee) = symbols.get(name.as_str()) else {
                            return invalid(
                                &*dynop,
                                ctx,
                                "call references an unresolved function symbol",
                            );
                        };
                        let Some((inputs, outputs)) = signatures.get(callee) else {
                            return verify_err!(
                                dynop.loc(ctx),
                                crate::Diagnostic::InternalInvariant {
                                    context: "resolved function is missing its validated signature",
                                }
                            );
                        };
                        if inputs.len() != op.get_num_operands()
                            || outputs.len() != op.get_num_results()
                            || inputs
                                .iter()
                                .zip(op.operands())
                                .any(|(a, b)| *a != b.get_type(ctx))
                            || outputs
                                .iter()
                                .zip(op.results())
                                .any(|(a, b)| *a != b.get_type(ctx))
                        {
                            return invalid(&*dynop, ctx, "call signature does not match callee");
                        }
                    }
                    if let Some(call) = dynop.downcast_ref::<crate::ops::CustomCallOp>()
                        && let Some(symbols) = symbols
                    {
                        for name in call.called_computations(ctx)? {
                            if !symbols.contains_key(&name) {
                                return invalid(
                                    &*dynop,
                                    ctx,
                                    "custom_call references an unresolved function symbol",
                                );
                            }
                        }
                    }
                    pending.extend(op.regions());
                }
            }
        }
        Ok(())
    }
    let mut index = Index::default();
    let mut pending: Vec<_> = root.deref(ctx).regions().collect();
    while let Some(region) = pending.pop() {
        index.region(ctx, region, &mut pending)?;
    }
    let root_ir = root.deref(ctx);
    let root_dyn = Operation::get_op_dyn(root, ctx);
    root_dyn.verify(ctx)?;
    root_dyn.verify_interfaces(ctx)?;
    let is_module = root_dyn.downcast_ref::<crate::ops::ModuleOp>().is_some();
    let is_function = root_dyn.downcast_ref::<crate::ops::FuncOp>().is_some();
    let mut symbols: SymbolTable = HashMap::new();
    let mut functions = Vec::new();
    let mut signatures = Signatures::new();
    if is_module {
        if root_ir.regions().count() != 1 {
            return invalid(&*root_dyn, ctx, "module must contain one body region");
        }
        let blocks: Vec<_> = root_ir.get_region(0).deref(ctx).iter(ctx).collect();
        if blocks.len() != 1 {
            return invalid(&*root_dyn, ctx, "module must contain one body block");
        }
        let Some(module_block) = blocks.first().copied() else {
            return invalid(&*root_dyn, ctx, "module must contain one body block");
        };
        for ptr in module_block.deref(ctx).iter(ctx) {
            if Operation::get_op_dyn(ptr, ctx)
                .downcast_ref::<crate::ops::FuncOp>()
                .is_some()
            {
                functions.push(ptr)
            } else {
                return invalid(
                    &*Operation::get_op_dyn(ptr, ctx),
                    ctx,
                    "module body may contain only functions",
                );
            }
        }
        for ptr in &functions {
            let dynop = Operation::get_op_dyn(*ptr, ctx);
            let Some(fun) = dynop.downcast_ref::<crate::ops::FuncOp>() else {
                continue;
            };
            let Some(name) = fun.get_attr_stablehlo_function_name(ctx) else {
                return invalid(&*dynop, ctx, "function requires a symbol name");
            };
            if name.as_str().is_empty() || symbols.insert(name.as_str().to_owned(), *ptr).is_some()
            {
                return invalid(&*dynop, ctx, "function symbol is empty or duplicated");
            }
        }
    } else if is_function {
        functions.push(root)
    }
    for &ptr in &functions {
        let Some(fun) = Operation::get_op::<crate::ops::FuncOp>(ptr, ctx) else {
            return verify_err!(
                Operation::get_op_dyn(ptr, ctx).loc(ctx),
                crate::Diagnostic::InternalInvariant {
                    context: "function list contained a non-function operation",
                }
            );
        };
        let Some(sig) = fun.get_attr_stablehlo_function_signature(ctx) else {
            if is_module {
                return invalid(
                    &*Operation::get_op_dyn(ptr, ctx),
                    ctx,
                    "function requires a signature",
                );
            }
            continue;
        };
        let handle = sig.get_type(ctx);
        index.check_type(ctx, handle)?;
        let object = handle.deref(ctx);
        let Some(ft) = object.downcast_ref::<crate::types::FunctionType>() else {
            return invalid(
                &*Operation::get_op_dyn(ptr, ctx),
                ctx,
                "function signature must be a StableHLO function type",
            );
        };
        signatures.insert(ptr, (ft.inputs.clone(), ft.outputs.clone()));
    }
    if is_module {
        let mut edges: HashMap<Ptr<Operation>, Vec<Ptr<Operation>>> = HashMap::new();
        for &function in &functions {
            let mut regions: Vec<_> = function.deref(ctx).regions().collect();
            let mut visited = HashSet::new();
            let mut callees = Vec::new();
            while let Some(region) = regions.pop() {
                if !visited.insert(region) {
                    return verify_err!(
                        Operation::get_op_dyn(function, ctx).loc(ctx),
                        crate::Diagnostic::CycleDetected {
                            graph: "region containment",
                        }
                    );
                }
                for block in region.deref(ctx).iter(ctx) {
                    for operation in block.deref(ctx).iter(ctx) {
                        let dynamic = Operation::get_op_dyn(operation, ctx);
                        if let Some(call) = dynamic.downcast_ref::<crate::ops::CallOp>()
                            && let Some(name) = call.get_attr_stablehlo_call_callee(ctx)
                            && let Some(&callee) = symbols.get(name.as_str())
                        {
                            callees.push(callee);
                        }
                        regions.extend(operation.deref(ctx).regions());
                    }
                }
            }
            edges.insert(function, callees);
        }
        let mut complete = HashSet::new();
        let mut active = HashSet::new();
        for &function in &functions {
            let mut stack = vec![(function, false)];
            while let Some((current, exiting)) = stack.pop() {
                if exiting {
                    active.remove(&current);
                    complete.insert(current);
                    continue;
                }
                if complete.contains(&current) {
                    continue;
                }
                if !active.insert(current) {
                    return verify_err!(
                        Operation::get_op_dyn(current, ctx).loc(ctx),
                        crate::Diagnostic::CycleDetected {
                            graph: "function call graph",
                        }
                    );
                }
                stack.push((current, true));
                if let Some(callees) = edges.get(&current) {
                    for &callee in callees {
                        if active.contains(&callee) {
                            return verify_err!(
                                Operation::get_op_dyn(callee, ctx).loc(ctx),
                                crate::Diagnostic::CycleDetected {
                                    graph: "function call graph",
                                }
                            );
                        }
                        if !complete.contains(&callee) {
                            stack.push((callee, false));
                        }
                    }
                }
            }
        }
    }
    let symbol_view = is_module.then_some(&symbols);
    if !functions.is_empty() {
        for ptr in &functions {
            let f = ptr.deref(ctx);
            if f.regions().count() != 1 {
                return invalid(
                    &*Operation::get_op_dyn(*ptr, ctx),
                    ctx,
                    "function requires one body region",
                );
            }
            let blocks: Vec<_> = f.get_region(0).deref(ctx).iter(ctx).collect();
            if blocks.is_empty() {
                continue;
            }
            let function = Operation::get_op_dyn(*ptr, ctx);
            let signature = signatures.get(ptr);
            for (block_index, block_ptr) in blocks.iter().enumerate() {
                let block = block_ptr.deref(ctx);
                let Some(last) = block.iter(ctx).last() else {
                    return invalid(
                        &*function,
                        ctx,
                        "function body contains an unterminated block",
                    );
                };
                if Operation::get_op_dyn(last, ctx)
                    .downcast_ref::<crate::ops::ReturnOp>()
                    .is_none()
                {
                    return invalid(
                        &*Operation::get_op_dyn(last, ctx),
                        ctx,
                        "function block must end in stablehlo.return",
                    );
                }
                if let Some((inputs, outputs)) = signature {
                    if block_index == 0
                        && !block
                            .arguments()
                            .map(|v| v.get_type(ctx))
                            .eq(inputs.iter().copied())
                    {
                        return invalid(
                            &*function,
                            ctx,
                            "function entry arguments do not match signature",
                        );
                    }
                    let ret = last.deref(ctx);
                    if !ret
                        .operands()
                        .map(|v| v.get_type(ctx))
                        .eq(outputs.iter().copied())
                    {
                        return invalid(
                            &*Operation::get_op_dyn(last, ctx),
                            ctx,
                            "function return types do not match signature",
                        );
                    }
                }
            }
            walk_region(ctx, f.get_region(0), symbol_view, &signatures, &mut index)?;
        }
    } else {
        for reg in root_ir.regions() {
            walk_region(ctx, reg, None, &signatures, &mut index)?
        }
    }
    Ok(())
}

pub(crate) fn verify_elementwise(
    op: &dyn Op,
    ctx: &Context,
    mnemonic: &str,
    arity: usize,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != arity || ir.get_num_results() != 1 {
        return invalid(op, ctx, "elementwise operation arity mismatch");
    }
    let inputs: Vec<_> = ir.operands().map(|v| v.get_type(ctx)).collect();
    let Ok(expected) = crate::inference::infer_elementwise(ctx, mnemonic, &inputs) else {
        return invalid(op, ctx, "elementwise operand domain or shape mismatch");
    };
    if !crate::inference::compatible_tensor_types(ctx, expected, ir.get_result(0).get_type(ctx)) {
        return invalid(
            op,
            ctx,
            "elementwise result is incompatible with inferred tensor type",
        );
    }
    Ok(())
}

pub(crate) mod compute;
pub(crate) mod control;
pub(crate) mod elementwise;
pub(crate) mod shape;
