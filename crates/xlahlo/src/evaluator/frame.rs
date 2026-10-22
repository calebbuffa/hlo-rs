use super::{EvaluatedValue, EvaluationError, failure};
use crate::{
    attributes::Index,
    ops::{ComputationOp, ReturnOp, control::ParameterOp},
    verification::optional_attr,
};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
    result::Result,
    value::Value,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct Key {
    pub value: Value,
    pub path: Vec<usize>,
}

pub(super) struct Frame {
    pub owner: Ptr<Operation>,
    pub block: Ptr<BasicBlock>,
    pub root: Value,
    pub members: HashSet<Value>,
    pub bindings: HashMap<Value, EvaluatedValue>,
    pub substitutions: HashMap<Value, EvaluatedValue>,
    pub memo: HashMap<Key, EvaluatedValue>,
    pub active: HashSet<Key>,
}

impl Frame {
    pub fn new(ctx: &Context, owner: Ptr<Operation>) -> Result<Self> {
        let regions: Vec<_> = owner.deref(ctx).regions().collect();
        if regions.len() != 1 {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation frame requires one region".into(),
            }));
        }
        let blocks: Vec<_> = regions[0].deref(ctx).iter(ctx).collect();
        if blocks.len() != 1 {
            return Err(failure(EvaluationError::Malformed {
                message: "evaluation frame requires one block".into(),
            }));
        }
        let block = blocks[0];
        let mut members = HashSet::new();
        for index in 0..block.deref(ctx).get_num_arguments() {
            members.insert(block.deref(ctx).get_argument(index));
        }
        let mut root = None;
        for operation in block.deref(ctx).iter(ctx) {
            if Operation::is_op::<ReturnOp>(operation, ctx) {
                if root.replace(operation.deref(ctx).get_operand(0)).is_some() {
                    return Err(failure(EvaluationError::Malformed {
                        message: "evaluation frame has multiple returns".into(),
                    }));
                }
            } else {
                for result in operation.deref(ctx).results() {
                    members.insert(result);
                }
            }
        }
        let root = root.ok_or_else(|| {
            failure(EvaluationError::Malformed {
                message: "evaluation frame has no return".into(),
            })
        })?;
        Ok(Self {
            owner,
            block,
            root,
            members,
            bindings: HashMap::new(),
            substitutions: HashMap::new(),
            memo: HashMap::new(),
            active: HashSet::new(),
        })
    }

    pub fn bind(&mut self, ctx: &Context, args: &[EvaluatedValue]) -> Result<()> {
        if Operation::is_op::<ComputationOp>(self.owner, ctx) {
            let mut parameters = HashMap::new();
            for operation in self.block.deref(ctx).iter(ctx) {
                if Operation::is_op::<ParameterOp>(operation, ctx) {
                    let index = optional_attr::<Index>(ctx, operation, "hlo_index")?
                        .ok_or_else(|| {
                            failure(EvaluationError::Malformed {
                                message: "parameter has no typed index".into(),
                            })
                        })?
                        .0;
                    let index = usize::try_from(index).map_err(|_| {
                        failure(EvaluationError::Malformed {
                            message: "parameter index exceeds host addressing".into(),
                        })
                    })?;
                    if parameters
                        .insert(index, operation.deref(ctx).get_result(0))
                        .is_some()
                    {
                        return Err(failure(EvaluationError::Malformed {
                            message: "duplicate parameter index".into(),
                        }));
                    }
                }
            }
            if parameters.len() != args.len() {
                return Err(failure(EvaluationError::Malformed {
                    message: "computation argument count mismatch".into(),
                }));
            }
            for (index, arg) in args.iter().enumerate() {
                let value = parameters.get(&index).copied().ok_or_else(|| {
                    failure(EvaluationError::Malformed {
                        message: "computation parameter numbering has a gap".into(),
                    })
                })?;
                self.bindings.insert(value, arg.clone());
            }
        } else {
            if self.block.deref(ctx).get_num_arguments() != args.len() {
                return Err(failure(EvaluationError::Malformed {
                    message: "inline fusion argument count mismatch".into(),
                }));
            }
            for (index, arg) in args.iter().enumerate() {
                self.bindings
                    .insert(self.block.deref(ctx).get_argument(index), arg.clone());
            }
        }
        Ok(())
    }
}
