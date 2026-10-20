//! Asynchronous contexts, channel transfers, and host feeds.
use crate::attributes::{Bools, Callees, OptionalInteger};
use pliron::{
    builtin::attributes::StringAttr, context::Context, identifier::Identifier, op::Op,
    result::Result,
};

native_instruction!(AsyncStartOp, "xlahlo.async_start", AsyncStart, crate::verification::control::instruction, callees: Callees);
native_instruction!(
    AsyncUpdateOp,
    "xlahlo.async_update",
    AsyncUpdate,
    crate::verification::control::instruction,
);
native_instruction!(
    AsyncDoneOp,
    "xlahlo.async_done",
    AsyncDone,
    crate::verification::control::instruction,
);
native_instruction!(
    CopyStartOp,
    "xlahlo.copy_start",
    CopyStart,
    crate::verification::control::instruction,
);

impl CopyStartOp {
    /// Optional cross-program prefetch index, preserving the proto's signed
    /// int32 domain and the distinction between absence and zero.
    pub fn cross_program_prefetch_index(&self, ctx: &Context) -> Result<Option<i32>> {
        crate::verification::control::asynchronous::copy_prefetch_index(ctx, self.op)
    }

    /// Set or clear the optional prefetch index without changing copy shapes.
    pub fn set_cross_program_prefetch_index(&self, ctx: &Context, index: Option<i32>) {
        self.op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_cross_program_prefetch_index"),
            index.map_or(OptionalInteger::Absent, |index| {
                OptionalInteger::Value(i64::from(index))
            }),
        );
    }
}

native_instruction!(
    CopyDoneOp,
    "xlahlo.copy_done",
    CopyDone,
    crate::verification::control::instruction,
);
native_instruction!(SendOp, "xlahlo.send", Send, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(SendDoneOp, "xlahlo.send_done", SendDone, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(RecvOp, "xlahlo.recv", Recv, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(RecvDoneOp, "xlahlo.recv_done", RecvDone, crate::verification::control::instruction, channel: OptionalInteger, host_transfer: Bools);
native_instruction!(
    InfeedOp,
    "xlahlo.infeed",
    Infeed,
    crate::verification::control::instruction,
);
native_instruction!(
    OutfeedOp,
    "xlahlo.outfeed",
    Outfeed,
    crate::verification::control::instruction,
);

impl InfeedOp {
    /// Opaque host-feed configuration; omitted metadata is the empty string.
    pub fn infeed_config(&self, ctx: &Context) -> Result<String> {
        crate::verification::control::asynchronous::feed_config(ctx, self.op, "hlo_infeed_config")
    }

    pub fn set_infeed_config(&self, ctx: &Context, config: String) {
        self.op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_infeed_config"),
            StringAttr::new(config),
        );
    }
}

impl OutfeedOp {
    /// Opaque host-feed configuration; omitted metadata is the empty string.
    pub fn outfeed_config(&self, ctx: &Context) -> Result<String> {
        crate::verification::control::asynchronous::feed_config(ctx, self.op, "hlo_outfeed_config")
    }

    pub fn set_outfeed_config(&self, ctx: &Context, config: String) {
        self.op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_outfeed_config"),
            StringAttr::new(config),
        );
    }
}

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
