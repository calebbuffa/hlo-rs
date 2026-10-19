//! Unit coverage for local effect policy, independent of graph validity.
use super::{has_local_side_effect, has_local_side_effect_with_context};
use crate::{
    attributes::{
        Bools, Callees, Index, OptionalInteger, RandomDistribution, ReplicaGroup, ReplicaGroups,
    },
    ops::control::{AllReduceOp, AllReduceStartOp, CollectiveBroadcastOp, ParameterOp, RngOp},
    types::{ElementType, TensorType},
};
use pliron::{context::Context, identifier::Identifier, op::Op};

#[test]
fn local_effects_preserve_rng_collective_and_channel_context_policies() {
    let mut ctx = Context::new();
    crate::register(&mut ctx);
    let f = TensorType::get(&ctx, ElementType::F32, vec![]).unwrap();
    let p = ParameterOp::new(&mut ctx, f, vec![], Index(0));
    let pv = p.result(&ctx);
    let rng = RngOp::new(&mut ctx, f, vec![pv, pv], RandomDistribution::Uniform);
    assert!(has_local_side_effect(&ctx, rng.get_operation()).unwrap());
    let broadcast = CollectiveBroadcastOp::new(
        &mut ctx,
        f,
        vec![pv],
        ReplicaGroups(vec![ReplicaGroup(vec![0])]),
    );
    assert!(has_local_side_effect(&ctx, broadcast.get_operation()).unwrap());
    let start = AllReduceStartOp::new(
        &mut ctx,
        f,
        vec![pv],
        Callees(vec![Identifier::new("reducer")]),
        ReplicaGroups(vec![ReplicaGroup(vec![0])]),
    );
    assert!(has_local_side_effect(&ctx, start.get_operation()).unwrap());
    let reduce = AllReduceOp::new(
        &mut ctx,
        f,
        vec![pv],
        Callees(vec![Identifier::new("reducer")]),
        ReplicaGroups(vec![ReplicaGroup(vec![0])]),
    );
    let op = reduce.get_operation();
    assert!(!has_local_side_effect(&ctx, op).unwrap());
    op.deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_channel"), OptionalInteger::Value(0));
    assert!(has_local_side_effect(&ctx, op).unwrap());
    assert!(!has_local_side_effect_with_context(&ctx, op, false).unwrap());
    op.deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_constrain_layout"), Bools(vec![true]));
    assert!(has_local_side_effect_with_context(&ctx, op, false).unwrap());
    op.deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_constrain_layout"), Bools(vec![]));
    assert!(has_local_side_effect(&ctx, op).is_err());
}
