//! Region terminator and iteration-index query.

use super::{AttentionOp, CustomOp, OnlineAttentionOp};
use crate::support::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
};
use pliron_arith::shaped;
use pliron_mlir_builtin::attributes::IndexAttr;

/// Terminator of an `iree_linalg_ext` region.
///
/// The enclosing operation defines what must be yielded.
#[pliron_op(
    name = "iree_linalg_ext.yield",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [
        pliron::builtin::op_interfaces::NResultsInterface<0>,
        pliron::builtin::op_interfaces::IsTerminatorInterface
    ]
)]
pub struct YieldOp;

impl YieldOp {
    /// Allocates the terminator. The enclosing operation verifies the yielded types.
    pub fn new(ctx: &mut Context, values: Vec<Value>) -> Self {
        Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], values, vec![], 0),
        }
    }
}

impl Verify for YieldOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let ok = {
            let ir = op.deref(ctx);
            ir.get_num_results() == 0 && ir.num_regions() == 0 && ir.get_num_successors() == 0
        };
        ensure(
            ctx,
            op,
            ok,
            "yield must have no results, regions, or successors",
        )
    }
}

/// Current index of iteration-space dimension `dim` inside the region of
/// `custom_op`, `attention`, or `online_attention`.
#[pliron_op(
    name = "iree_linalg_ext.index",
    format = "attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (iree_index_dim: IndexAttr)
)]
pub struct IndexOp;

impl IndexOp {
    /// Allocates the operation with an `index` result.
    pub fn new(ctx: &mut Context, dim: u32) -> Self {
        let index = shaped::index(ctx);
        let op = new_plain(ctx, |op| Self { op }, vec![index], vec![]);
        op.set_attr_iree_index_dim(
            ctx,
            IndexAttr {
                value: i64::from(dim),
            },
        );
        op
    }

    /// Returns the queried dimension, or `None` when the attribute is absent.
    pub fn dim(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_iree_index_dim(ctx).map(|a| a.value)
    }

    /// Returns the result.
    pub fn result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}

/// Returns the loop count of an operation whose region may contain [`IndexOp`].
fn loop_count(ctx: &Context, parent: Ptr<Operation>) -> Option<usize> {
    let concrete = Operation::get_op_dyn(parent, ctx);
    if let Some(custom) = concrete.downcast_ref::<CustomOp>() {
        return Some(custom.iterator_types(ctx).len());
    }
    if let Some(attention) = concrete.downcast_ref::<AttentionOp>() {
        return attention.indexing_maps(ctx).first().map(|m| m.dimensions);
    }
    if let Some(attention) = concrete.downcast_ref::<OnlineAttentionOp>() {
        return attention.indexing_maps(ctx).first().map(|m| m.dimensions);
    }
    None
}

impl Verify for IndexOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let ok = {
            let ir = op.deref(ctx);
            ir.get_num_operands() == 0 && ir.get_num_results() == 1 && ir.num_regions() == 0
        };
        ensure(
            ctx,
            op,
            ok,
            "expected no operands, one result, and no regions",
        )?;
        ensure(
            ctx,
            op,
            shaped::is_index(ctx, self.result(ctx).get_type(ctx)),
            "result must be index",
        )?;
        let dim = self
            .dim(ctx)
            .ok_or_else(|| fail(ctx, op, "missing dimension"))?;
        ensure(ctx, op, dim >= 0, "dimension must be nonnegative")?;
        let parent = parent_op(ctx, op).ok_or_else(|| {
            fail(
                ctx,
                op,
                "must be nested in custom_op, attention, or online_attention",
            )
        })?;
        let loops = loop_count(ctx, parent).ok_or_else(|| {
            fail(
                ctx,
                op,
                "must be nested in custom_op, attention, or online_attention",
            )
        })?;
        ensure(
            ctx,
            op,
            (dim as usize) < loops,
            "dimension is outside the iteration space",
        )
    }
}
