#[path = "control_support/mod.rs"]
mod control_support;
#[path = "support/mod.rs"]
mod support;

#[path = "ir/computation_cloning/batch_api.rs"]
mod computation_cloning_batch_api;
#[path = "ir/computation_cloning/public_api.rs"]
mod computation_cloning_public_api;
#[path = "ir/metadata/copy_prefetch.rs"]
mod copy_prefetch_metadata;
#[path = "ir/metadata/custom_call.rs"]
mod custom_call_metadata;
#[path = "ir/metadata/feed_config.rs"]
mod feed_config_metadata;
#[path = "ir/fusion_parameter.rs"]
mod fusion_parameter;
#[path = "ir/graph.rs"]
mod graph;
#[path = "ir/metadata/module.rs"]
mod module_metadata;
#[path = "ir/metadata/rng_state.rs"]
mod rng_state_metadata;
