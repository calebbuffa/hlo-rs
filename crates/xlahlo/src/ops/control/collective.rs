//! Collective communication operations.
use crate::attributes::{Axes, Callees, ReplicaGroups, SourceTargetPairs};
use pliron::{context::Context, op::Op};

native_instruction!(AllGatherOp, "hlo.all_gather", AllGather, crate::verification::control::instruction, dimensions: Axes, groups: ReplicaGroups);
native_instruction!(AllGatherStartOp, "hlo.all_gather_start", AllGatherStart, crate::verification::control::instruction, dimensions: Axes, groups: ReplicaGroups);
native_instruction!(
    AllGatherDoneOp,
    "hlo.all_gather_done",
    AllGatherDone,
    crate::verification::control::instruction,
);
native_instruction!(AllReduceOp, "hlo.all_reduce", AllReduce, crate::verification::control::instruction, callees: Callees, groups: ReplicaGroups);
native_instruction!(AllReduceStartOp, "hlo.all_reduce_start", AllReduceStart, crate::verification::control::instruction, callees: Callees, groups: ReplicaGroups);
native_instruction!(
    AllReduceDoneOp,
    "hlo.all_reduce_done",
    AllReduceDone,
    crate::verification::control::instruction,
);
native_instruction!(ReduceScatterOp, "hlo.reduce_scatter", ReduceScatter, crate::verification::control::instruction, dimensions: Axes, callees: Callees, groups: ReplicaGroups);
native_instruction!(AllToAllOp, "hlo.all_to_all", AllToAll, crate::verification::control::instruction, dimensions: Axes, groups: ReplicaGroups);
native_instruction!(RaggedAllToAllOp, "hlo.ragged_all_to_all", RaggedAllToAll, crate::verification::control::instruction, groups: ReplicaGroups);
native_instruction!(CollectiveBroadcastOp, "hlo.collective_broadcast", CollectiveBroadcast, crate::verification::control::instruction, groups: ReplicaGroups);
native_instruction!(CollectiveReduceOp, "hlo.collective_reduce", CollectiveReduce, crate::verification::control::instruction, callees: Callees, groups: ReplicaGroups);
native_instruction!(CollectivePermuteOp, "hlo.collective_permute", CollectivePermute, crate::verification::control::instruction, pairs: SourceTargetPairs);
native_instruction!(CollectivePermuteStartOp, "hlo.collective_permute_start", CollectivePermuteStart, crate::verification::control::instruction, pairs: SourceTargetPairs);
native_instruction!(
    CollectivePermuteDoneOp,
    "hlo.collective_permute_done",
    CollectivePermuteDone,
    crate::verification::control::instruction,
);

pub(super) fn register(ctx: &mut Context) {
    macro_rules! ops { ($($ty:ty),* $(,)?) => { $(<$ty>::register(ctx);)* }; }
    ops!(
        AllGatherOp,
        AllGatherStartOp,
        AllGatherDoneOp,
        AllReduceOp,
        AllReduceStartOp,
        AllReduceDoneOp,
        ReduceScatterOp,
        AllToAllOp,
        RaggedAllToAllOp,
        CollectiveBroadcastOp,
        CollectiveReduceOp,
        CollectivePermuteOp,
        CollectivePermuteStartOp,
        CollectivePermuteDoneOp
    );
    crate::verification::control::collective::register(ctx);
}
