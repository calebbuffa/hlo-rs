//! Native control, communication, and stateful operation verification.
use super::{
    VerifyOptions, instruction_types, optional_attr, require, returned, same, single_block,
    verify_result_accuracy,
};
use crate::Diagnostic;
use crate::{
    attributes::*,
    inference::{Signature, control as infer},
    literal::Literal,
    opcode::{Opcode, Opcode::*},
    ops::{
        ReturnOp,
        control::{FusionOp, ShapeList},
        numerical::NumericalInteger,
    },
    shape::{self, layouts_equal},
    types::{BufferType, ElementType, TokenType},
};
use pliron::result::Result;
use pliron::{
    attribute::Attribute,
    builtin::attributes::StringAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
};
use std::{
    cell::Ref,
    collections::{HashMap, HashSet},
};

/// Resolve each async instruction to its wrapped computation using verifier
/// producer tracing over the supplied module operation index.
pub fn async_wrapped_computations(
    ctx: &Context,
    operations: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
) -> Result<HashMap<Ptr<Operation>, Ptr<Operation>>> {
    asynchronous::wrapped_computations(ctx, operations, computations)
}

/// Resolve each async instruction to its chain start using verifier producer
/// tracing over the supplied module operation index.
pub fn async_chain_starts(
    ctx: &Context,
    operations: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<crate::ir::ComputationName, Ptr<Operation>>,
) -> Result<HashMap<Ptr<Operation>, Ptr<Operation>>> {
    asynchronous::chain_starts(ctx, operations, computations)
}

/// Async ownership, producer tracing, and channel-pair validation.
pub(crate) mod asynchronous;
/// Collective group, reducer, and module consistency validation.
pub(crate) mod collective;
mod deadlocks;

mod aliases;
mod common;
mod effects;
mod fusion;
mod instruction;

use aliases::indexed;
pub(crate) use aliases::{verify_aliases, verify_fusion_aliases};
use common::compare;
pub(super) use common::required;
pub(crate) use common::{channel, flag, same_ignoring_memory};
pub(crate) use common::{
    custom_call_api_version, custom_call_padding_kind, custom_call_schedule, rng_delta,
};
pub(crate) use effects::{has_local_side_effect, has_local_side_effect_with_context};
pub(crate) use fusion::fusion;
pub(crate) use instruction::{instruction, verify_callees};
