//! Control flow, tuples, stateful operations, and communication.
//!
//! Constructors retain the declared result type. Use the operation verifier for
//! local checks and the module verifier for callee, ownership, and channel rules.
use super::HloOpInterface;
use crate::{
    attributes::{Axes, Callees, FusionKind, Index, RandomAlgorithm, RandomDistribution},
    literal::Literal,
    opcode::Opcode,
};
use pliron::{
    attribute::Attribute,
    builtin::attributes::StringAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    region::Region,
    result::Result,
    r#type::TypeHandle,
    value::Value,
};

/// Declared operand layouts for a layout-constrained custom call.
#[pliron::derive::pliron_attr(
    name = "hlo.shape_list",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShapeList(pub Vec<TypeHandle>);

/// Per-operand slice bounds for in-place collective-permute buffers.
#[pliron::derive::pliron_attr(
    name = "hlo.dynamic_slice_sizes",
    format = "`[` vec($0, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DynamicSliceSizes(pub Vec<Axes>);

native_instruction!(ParameterOp, "hlo.parameter", Parameter, crate::verification::control::instruction, index: Index);
native_instruction!(ConstantOp, "hlo.constant", Constant, crate::verification::control::instruction, literal: Literal);
native_instruction!(
    TupleOp,
    "hlo.tuple",
    Tuple,
    crate::verification::control::instruction,
);
native_instruction!(GetTupleElementOp, "hlo.get_tuple_element", GetTupleElement, crate::verification::control::instruction, index: Index);
native_instruction!(CallOp, "hlo.call", Call, crate::verification::control::instruction, callees: Callees);
native_instruction!(WhileOp, "hlo.while", While, crate::verification::control::instruction, callees: Callees);
native_instruction!(ConditionalOp, "hlo.conditional", Conditional, crate::verification::control::instruction, callees: Callees);
native_instruction!(CustomCallOp, "hlo.custom_call", CustomCall, crate::verification::control::instruction, target: StringAttr);
native_instruction!(
    DomainOp,
    "hlo.domain",
    Domain,
    crate::verification::control::instruction,
);
native_instruction!(
    OptimizationBarrierOp,
    "hlo.optimization_barrier",
    OptimizationBarrier,
    crate::verification::control::instruction,
);
native_instruction!(
    AfterAllOp,
    "hlo.after_all",
    AfterAll,
    crate::verification::control::instruction,
);
native_instruction!(
    AddDependencyOp,
    "hlo.add_dependency",
    AddDependency,
    crate::verification::control::instruction,
);
native_instruction!(
    ReplicaIdOp,
    "hlo.replica_id",
    ReplicaId,
    crate::verification::control::instruction,
);
native_instruction!(
    PartitionIdOp,
    "hlo.partition_id",
    PartitionId,
    crate::verification::control::instruction,
);
native_instruction!(RngOp, "hlo.rng", Rng, crate::verification::control::instruction, distribution: RandomDistribution);
native_instruction!(RngBitGeneratorOp, "hlo.rng_bit_generator", RngBitGenerator, crate::verification::control::instruction, algorithm: RandomAlgorithm);
native_instruction!(
    RngGetAndUpdateStateOp,
    "hlo.rng_get_and_update_state",
    RngGetAndUpdateState,
    crate::verification::control::instruction,
);

mod asynchronous;
mod collective;
pub use asynchronous::{
    AsyncDoneOp, AsyncStartOp, AsyncUpdateOp, CopyDoneOp, CopyStartOp, InfeedOp, OutfeedOp,
    RecvDoneOp, RecvOp, SendDoneOp, SendOp,
};
pub use collective::{
    AllGatherDoneOp, AllGatherOp, AllGatherStartOp, AllReduceDoneOp, AllReduceOp, AllReduceStartOp,
    AllToAllOp, CollectiveBroadcastOp, CollectivePermuteDoneOp, CollectivePermuteOp,
    CollectivePermuteStartOp, CollectiveReduceOp, RaggedAllToAllOp, ReduceScatterOp,
};

/// A fused computation represented by one isolated, single-block region.
#[pliron_op(name = "hlo.fusion", format, interfaces = [pliron::builtin::op_interfaces::IsolatedFromAboveInterface], attributes = (hlo_fusion_kind: FusionKind))]
pub struct FusionOp;

impl FusionOp {
    /// The HLO opcode represented by this operation.
    pub const OPCODE: Opcode = Opcode::Fusion;
    /// Create a fusion with an empty body; populate its block and return value.
    pub fn new(
        ctx: &mut Context,
        result: TypeHandle,
        operands: Vec<Value>,
        kind: FusionKind,
    ) -> Self {
        let value = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                operands,
                vec![],
                1,
            ),
        };
        value.set_attr_hlo_fusion_kind(ctx, kind);
        value
    }
    /// The isolated region whose block arguments bind the fusion operands.
    pub fn body(&self, ctx: &Context) -> Ptr<Region> {
        self.op.deref(ctx).get_region(0)
    }
    /// The sole result of the fused computation.
    pub fn result(&self, ctx: &Context) -> Value {
        self.op.deref(ctx).get_result(0)
    }
}

impl Verify for FusionOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::control::fusion(ctx, self, Default::default())
    }
}

#[pliron::derive::op_interface_impl]
impl HloOpInterface for FusionOp {
    fn opcode(&self) -> Opcode {
        Self::OPCODE
    }
    fn verify_with_options(
        &self,
        ctx: &Context,
        options: crate::verification::VerifyOptions,
    ) -> Result<()> {
        crate::verification::control::fusion(ctx, self, options)
    }
}

/// Register all control-family operations and their family-specific attributes.
pub(crate) fn register(ctx: &mut Context) {
    ShapeList::register::<ShapeList>(ctx);
    DynamicSliceSizes::register::<DynamicSliceSizes>(ctx);
    macro_rules! ops { ($($ty:ty),* $(,)?) => { $(<$ty>::register(ctx);)* }; }
    ops!(
        ParameterOp,
        ConstantOp,
        TupleOp,
        GetTupleElementOp,
        CallOp,
        WhileOp,
        ConditionalOp,
        CustomCallOp,
        DomainOp,
        OptimizationBarrierOp,
        AfterAllOp,
        AddDependencyOp,
        ReplicaIdOp,
        PartitionIdOp,
        RngOp,
        RngBitGeneratorOp,
        RngGetAndUpdateStateOp,
        FusionOp
    );
    collective::register(ctx);
    asynchronous::register(ctx);
}
