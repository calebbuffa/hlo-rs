//! Asynchronous binding prefixes, ownership, and channel pairing.
use super::{
    VerifyOptions, channel, custom_call_api_version, custom_call_padding_kind,
    custom_call_schedule, flag, optional_attr, require, required, rng_delta, same,
    same_ignoring_memory,
};
use crate::Diagnostic;
use crate::{
    attributes::{
        Algorithm, BatchNormOptions, BlockScalingConfig, Bytes, Callees, FrontendAttributes, Index,
        IndexFlags, OperandPrecision, OptionalInteger, OutputOperandAliases, PrecisionConfig,
        ResultAccuracy, SparsityConfig,
    },
    inference::{Signature, control as infer},
    ir::{self, ExecutionThread, MetadataPayloads},
    literal::Literal,
    opcode::{Opcode, Opcode::*},
    ops::{HloOpInterface, ReturnOp, numerical::NumericalInteger},
    shape,
    sharding::Sharding,
    types::{BufferType, TokenType},
};
use pliron::result::Result;
use pliron::{
    builtin::attributes::{StringAttr, TypeAttr},
    context::{Context, Ptr},
    identifier::Identifier,
    op::op_cast,
    operation::Operation,
    r#type::{TypeHandle, Typed},
    value::{DefiningEntity, Value},
};
use std::{
    cell::{Ref, RefCell},
    collections::{BTreeMap, HashMap, HashSet},
};

mod families;
mod graph;
mod lifecycle;

use families::same_frontend_attributes;
pub(crate) use families::{copy_prefetch_index, feed_config, opcode};
pub(super) use families::{infer, verify_callees};
use graph::{Graph, build_graph, direct, intermediary, target};
use lifecycle::execution_thread;
pub(crate) use lifecycle::{chain_starts, verify_graph, wrapped_computations};
