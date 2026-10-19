//! Shared native verification predicates for control and token operations.
use crate::verification::invalid;
use pliron::{context::Context, op::Op, result::Result, r#type::Typed};

pub(crate) fn verify_token_values(op: &dyn Op, ctx: &Context) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.results().any(|v| {
        v.get_type(ctx)
            .deref(ctx)
            .downcast_ref::<crate::types::TokenType>()
            .is_none()
    }) || ir.operands().any(|v| {
        v.get_type(ctx)
            .deref(ctx)
            .downcast_ref::<crate::types::TokenType>()
            .is_none()
    }) {
        return invalid(
            op,
            ctx,
            "token operation operands and results must have token type",
        );
    }
    Ok(())
}
