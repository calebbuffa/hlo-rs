#[path = "control_support/mod.rs"]
mod control_support;
#[path = "support/mod.rs"]
mod support;

#[path = "ops/control/core.rs"]
mod control;
#[path = "ops/control/async.rs"]
mod control_async;
#[path = "ops/control/async_collective_identity.rs"]
mod control_async_collective_identity;
#[path = "ops/control/async_fusion_identity.rs"]
mod control_async_fusion_identity;
#[path = "ops/control/async_identity.rs"]
mod control_async_identity;
#[path = "ops/control/async_index_identity.rs"]
mod control_async_index_identity;
#[path = "ops/control/async_numerical_identity.rs"]
mod control_async_numerical_identity;
#[path = "ops/control/async_transfer_identity.rs"]
mod control_async_transfer_identity;
#[path = "ops/control/collective_roots.rs"]
mod control_collective_roots;
#[path = "ops/control/collectives.rs"]
mod control_collectives;
#[path = "ops/control/custom_call_aliases.rs"]
mod control_custom_call_aliases;
#[path = "ops/control/datatypes.rs"]
mod control_datatypes;
#[path = "ops/control/dependencies.rs"]
mod control_dependencies;
#[path = "ops/control/rng.rs"]
mod control_rng;
#[path = "ops/elementwise.rs"]
mod elementwise;
#[path = "ops/movement.rs"]
mod movement;
#[path = "ops/numerical.rs"]
mod numerical_ops;
#[path = "ops/numerical/sort_stability.rs"]
mod sort_stability;
#[path = "ops/numerical/topk_stability.rs"]
mod topk_stability;
