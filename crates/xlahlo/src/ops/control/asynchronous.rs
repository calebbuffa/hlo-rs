//! Asynchronous contexts, channel transfers, and host feeds.
use crate::attributes::{Bools, Callees, OptionalInteger};
use pliron::{context::Context, op::Op};

native_instruction!(AsyncStartOp, "hlo.async_start", AsyncStart, crate::verification::control::instruction, callees: Callees);
native_instruction!(
    AsyncUpdateOp,
    "hlo.async_update",
    AsyncUpdate,
    crate::verification::control::instruction,
);
native_instruction!(
    AsyncDoneOp,
    "hlo.async_done",
    AsyncDone,
    crate::verification::control::instruction,
);
native_instruction!(
    CopyStartOp,
    "hlo.copy_start",
    CopyStart,
    crate::verification::control::instruction,
);
native_instruction!(
    CopyDoneOp,
    "hlo.copy_done",
    CopyDone,
    crate::verification::control::instruction,
);
native_instruction!(SendOp, "hlo.send", Send, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(SendDoneOp, "hlo.send_done", SendDone, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(RecvOp, "hlo.recv", Recv, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(RecvDoneOp, "hlo.recv_done", RecvDone, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(
    InfeedOp,
    "hlo.infeed",
    Infeed,
    crate::verification::control::instruction,
);
native_instruction!(
    OutfeedOp,
    "hlo.outfeed",
    Outfeed,
    crate::verification::control::instruction,
);

pub(super) fn register(ctx: &mut Context) {
    macro_rules! ops { ($($ty:ty),* $(,)?) => { $(<$ty>::register(ctx);)* }; }
    ops!(
        AsyncStartOp,
        AsyncUpdateOp,
        AsyncDoneOp,
        CopyStartOp,
        CopyDoneOp,
        SendOp,
        SendDoneOp,
        RecvOp,
        RecvDoneOp,
        InfeedOp,
        OutfeedOp
    );
}
