//! Creation-capable extension of instruction replacement's rehearsal/commit plan.
//!
//! Existing operations keep their identities. Newly created operation trees are
//! adopted from the verified rehearsal rather than cloned with SSA fallbacks.
use crate::verification::{VerifyOptions, error, require, verify_with_options};
use pliron::{
    attribute::AttributeDict,
    basic_block::BasicBlock,
    context::{Context, Ptr},
    irbuild::{
        cloning::{IrMapping, clone_operation},
        listener::DummyListener,
        rewriter::IRRewriter,
    },
    linked_list::ContainsLinkedList,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
};
use std::collections::{HashMap, HashSet};

pub(super) fn tree(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut operations = vec![root];
    let mut index = 0;
    while index < operations.len() {
        let operation = operations[index];
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                operations.extend(block.deref(ctx).iter(ctx));
            }
        }
        index += 1;
    }
    operations
}

pub(crate) struct LiveMapping {
    values: HashMap<Value, Value>,
}

impl LiveMapping {
    pub fn value(&self, value: Value) -> Result<Value> {
        self.values
            .get(&value)
            .copied()
            .ok_or_else(|| error("staged output value does not survive"))
    }
}

struct OperationEdit {
    operation: Ptr<Operation>,
    operands: Vec<Value>,
    attributes: AttributeDict,
}

struct Addition {
    operation: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    before: Option<Ptr<Operation>>,
}

struct EditPlan {
    operations: Vec<OperationEdit>,
    arguments: Vec<(Ptr<BasicBlock>, Vec<usize>, AttributeDict)>,
    additions: Vec<Addition>,
    added_operands: Vec<(Ptr<Operation>, Vec<Value>)>,
    erase: Vec<Ptr<Operation>>,
}

impl EditPlan {
    fn commit(self, ctx: &mut Context) {
        // All graph ownership, indices, policies and projected outputs have
        // already been checked. No fallible helper runs after publication.
        for addition in self.additions {
            addition.operation.unlink(ctx);
            if let Some(before) = addition.before {
                addition.operation.insert_before(ctx, before);
            } else {
                addition.operation.insert_at_back(addition.block, ctx);
            }
        }
        for (operation, operands) in self.added_operands {
            for (index, value) in operands.into_iter().enumerate() {
                if operation.deref(ctx).get_operand(index) != value {
                    Operation::replace_operand(operation, ctx, index, value);
                }
            }
        }
        for edit in self.operations {
            let old_count = edit.operation.deref(ctx).get_num_operands();
            for (index, value) in edit.operands.iter().copied().enumerate() {
                if index < old_count {
                    if edit.operation.deref(ctx).get_operand(index) != value {
                        Operation::replace_operand(edit.operation, ctx, index, value);
                    }
                } else {
                    Operation::push_operand(edit.operation, ctx, value);
                }
            }
            for index in (edit.operands.len()..old_count).rev() {
                Operation::remove_operand(edit.operation, ctx, index);
            }
            edit.operation.deref_mut(ctx).attributes = edit.attributes;
        }
        for operation in self.erase {
            Operation::erase(operation, ctx);
        }
        for (block, indices, attributes) in self.arguments {
            for index in indices.into_iter().rev() {
                BasicBlock::remove_argument(block, ctx, index);
            }
            block.deref_mut(ctx).attributes = attributes;
        }
    }
}

fn prepare(
    ctx: &Context,
    original: &[Ptr<Operation>],
    initial: &IrMapping,
    scratch: Ptr<Operation>,
) -> Result<(EditPlan, LiveMapping)> {
    let surviving = tree(ctx, scratch);
    let surviving_set: HashSet<_> = surviving.iter().copied().collect();
    let mut reverse = HashMap::new();
    let mut reverse_blocks = HashMap::new();
    let mut values = HashMap::new();
    let mut removed = HashSet::new();
    let mut arguments = Vec::new();
    for &operation in original {
        let copy = initial
            .lookup_op(operation)
            .ok_or_else(|| error("rehearsal omitted an original operation"))?;
        reverse.insert(copy, operation);
        if !surviving_set.contains(&copy) {
            removed.insert(operation);
            continue;
        }
        let expected_parent = operation
            .deref(ctx)
            .get_parent_block()
            .and_then(|block| initial.lookup_block(block));
        require(
            copy.deref(ctx).get_parent_block() == expected_parent,
            "moving surviving operations requires an explicit relocation edit",
        )?;
        require(
            operation.deref(ctx).num_regions() == copy.deref(ctx).num_regions(),
            "editing changed a surviving operation's region count",
        )?;
        require(
            operation.deref(ctx).get_num_results() == copy.deref(ctx).get_num_results(),
            "editing changed a surviving operation's result count",
        )?;
        for index in 0..operation.deref(ctx).get_num_results() {
            let old = operation.deref(ctx).get_result(index);
            let new = copy.deref(ctx).get_result(index);
            require(
                old.get_type(ctx) == new.get_type(ctx),
                "editing changed a surviving result type",
            )?;
            values.insert(new, old);
        }
        for region in operation.deref(ctx).regions() {
            let index = operation
                .deref(ctx)
                .regions()
                .position(|candidate| candidate == region)
                .ok_or_else(|| error("unknown original region"))?;
            let copied_region = copy.deref(ctx).get_region(index);
            let expected: Vec<_> = region
                .deref(ctx)
                .iter(ctx)
                .map(|block| {
                    initial
                        .lookup_block(block)
                        .ok_or_else(|| error("missing staged block"))
                })
                .collect::<Result<_>>()?;
            require(
                copied_region.deref(ctx).iter(ctx).collect::<Vec<_>>() == expected,
                "editing surviving region blocks requires an explicit structural edit",
            )?;
            for block in region.deref(ctx).iter(ctx) {
                let cloned = initial
                    .lookup_block(block)
                    .ok_or_else(|| error("rehearsal omitted an original block"))?;
                reverse_blocks.insert(cloned, block);
                let remaining: HashSet<_> = cloned.deref(ctx).arguments().collect();
                let mut indices = Vec::new();
                let mut mapped = HashSet::new();
                for (index, argument) in block.deref(ctx).arguments().enumerate() {
                    let copy = initial
                        .lookup_value(argument)
                        .ok_or_else(|| error("rehearsal omitted a block argument"))?;
                    if remaining.contains(&copy) {
                        values.insert(copy, argument);
                        mapped.insert(copy);
                    } else {
                        indices.push(index);
                    }
                }
                require(
                    mapped == remaining,
                    "adding arguments to surviving blocks requires an explicit signature edit",
                )?;
                arguments.push((block, indices, cloned.deref(ctx).attributes.clone()));
            }
        }
    }
    let mut new_operations = Vec::new();
    for &operation in &surviving {
        require(
            operation.deref(ctx).get_num_successors() == 0,
            "native staged editing cannot introduce block successors",
        )?;
        if reverse.contains_key(&operation) {
            continue;
        }
        new_operations.push(operation);
        for index in 0..operation.deref(ctx).get_num_results() {
            let value = operation.deref(ctx).get_result(index);
            values.insert(value, value);
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                for value in block.deref(ctx).arguments() {
                    values.insert(value, value);
                }
            }
        }
    }
    for (&copy, &block) in &reverse_blocks {
        let expected = block
            .deref(ctx)
            .iter(ctx)
            .filter_map(|operation| {
                initial
                    .lookup_op(operation)
                    .filter(|copy| surviving_set.contains(copy))
            })
            .collect::<Vec<_>>();
        let actual = copy
            .deref(ctx)
            .iter(ctx)
            .filter(|operation| reverse.contains_key(operation))
            .collect::<Vec<_>>();
        require(
            expected == actual,
            "reordering surviving operations requires an explicit relocation edit",
        )?;
    }
    let live = LiveMapping { values };
    let mut additions = Vec::new();
    let mut added_operands = Vec::new();
    for operation in new_operations {
        let parent = operation
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| error("new staged operation is detached"))?;
        if let Some(&block) = reverse_blocks.get(&parent) {
            let mut seen = false;
            let before = parent.deref(ctx).iter(ctx).find_map(|candidate| {
                if candidate == operation {
                    seen = true;
                    None
                } else if seen {
                    reverse.get(&candidate).copied()
                } else {
                    None
                }
            });
            additions.push(Addition {
                operation,
                block,
                before,
            });
        }
        let operands = operation
            .deref(ctx)
            .operands()
            .map(|value| live.value(value))
            .collect::<Result<Vec<_>>>()?;
        added_operands.push((operation, operands));
    }
    let mut operations = Vec::new();
    for &operation in original {
        if removed.contains(&operation) {
            continue;
        }
        let copy = initial
            .lookup_op(operation)
            .ok_or_else(|| error("surviving operation has no staged identity"))?;
        let operands = copy
            .deref(ctx)
            .operands()
            .map(|value| live.value(value))
            .collect::<Result<Vec<_>>>()?;
        operations.push(OperationEdit {
            operation,
            operands,
            attributes: copy.deref(ctx).attributes.clone(),
        });
    }
    let erase = original
        .iter()
        .rev()
        .copied()
        .filter(|operation| {
            if !removed.contains(operation) {
                return false;
            }
            let mut parent = operation
                .deref(ctx)
                .get_parent_block()
                .and_then(|block| block.deref(ctx).get_parent_op(ctx));
            while let Some(owner) = parent {
                if removed.contains(&owner) {
                    return false;
                }
                parent = owner
                    .deref(ctx)
                    .get_parent_block()
                    .and_then(|block| block.deref(ctx).get_parent_op(ctx));
            }
            true
        })
        .collect();
    Ok((
        EditPlan {
            operations,
            arguments,
            additions,
            added_operands,
            erase,
        },
        live,
    ))
}

/// Shared creation-capable rehearsal boundary. `project` resolves every output
/// handle before commit so no rejected edit can publish a detached partial tree.
pub(crate) fn staged_edit<T, R>(
    ctx: &mut Context,
    module: Ptr<Operation>,
    verification: Option<VerifyOptions>,
    edit: impl FnOnce(&mut Context, Ptr<Operation>, &IrMapping) -> Result<T>,
    project: impl FnOnce(T, &LiveMapping) -> Result<R>,
) -> Result<R> {
    staged_edit_if(ctx, module, verification, edit, |_| true, project)
}

/// A refused replacement must not publish even equivalent cloned attributes.
pub(crate) fn staged_edit_if<T, R>(
    ctx: &mut Context,
    module: Ptr<Operation>,
    verification: Option<VerifyOptions>,
    edit: impl FnOnce(&mut Context, Ptr<Operation>, &IrMapping) -> Result<T>,
    changed: impl FnOnce(&T) -> bool,
    project: impl FnOnce(T, &LiveMapping) -> Result<R>,
) -> Result<R> {
    if let Some(options) = verification {
        super::super::require_rehearsable_policy(options)?;
        verify_with_options(ctx, module, options)?;
    }
    let original = tree(ctx, module);
    let mut defined = HashSet::new();
    for &operation in &original {
        require(
            operation.deref(ctx).get_num_successors() == 0,
            "native staged editing cannot contain block successors",
        )?;
        for index in 0..operation.deref(ctx).get_num_results() {
            defined.insert(operation.deref(ctx).get_result(index));
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                defined.extend(block.deref(ctx).arguments());
            }
        }
    }
    for &operation in &original {
        require(
            operation
                .deref(ctx)
                .operands()
                .all(|value| defined.contains(&value)),
            "staged module captures an external SSA value",
        )?;
    }
    let mut mapping = IrMapping::new();
    let scratch = clone_operation(
        module,
        ctx,
        &mut IRRewriter::<DummyListener>::default(),
        &mut mapping,
    );
    let result = (|| {
        let result = edit(ctx, scratch, &mapping)?;
        if !changed(&result) {
            return Ok((
                None,
                project(
                    result,
                    &LiveMapping {
                        values: HashMap::new(),
                    },
                )?,
            ));
        }
        if let Some(options) = verification {
            verify_with_options(ctx, scratch, options)?;
        }
        let (plan, live) = prepare(ctx, &original, &mapping, scratch)?;
        let output = project(result, &live)?;
        Ok((Some(plan), output))
    })();
    match result {
        Ok((plan, output)) => {
            if let Some(plan) = plan {
                plan.commit(ctx);
            }
            Operation::erase(scratch, ctx);
            Ok(output)
        }
        Err(error) => {
            Operation::erase(scratch, ctx);
            Err(error)
        }
    }
}
