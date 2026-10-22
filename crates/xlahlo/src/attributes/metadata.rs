//! Instruction identity, profiling, scheduling, and statistics attributes.

use super::*;
use crate::Diagnostic;
use pliron::{common_traits::Verify, context::Context, derive::pliron_attr, result::Result};

// Floating metadata uses exact encodings for pliron's Eq/Hash contract.
record!(ProfileInfo, "xlahlo.profile_info", "Instruction profile provenance and exact f64 relative-speedup encoding.", {
    /// Profile category identifiers supplied by the frontend.
    profile_type: Axes,
    /// Exact IEEE 754 binary64 encoding of the relative speedup estimate.
    relative_speedup_bits: u64,
    /// Source identifier for the collected profile.
    profile_source: i32,
    /// Compilation event associated with this profile.
    compilation_event: i32,
    /// Identifier of the strategy that generated this profile.
    profile_generation_strategy: i32,
});
list!(
    ProfileInfos,
    "xlahlo.profile_infos",
    ProfileInfo,
    "Optional singleton instruction profile record."
);
#[pliron_attr(name = "xlahlo.op_metadata", format)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Instruction identity, source location, profiling, scheduling, and payload metadata.
pub struct OpMetadata {
    /// Frontend operation type used to produce this instruction.
    pub op_type: String,
    /// Frontend operation name.
    pub op_name: String,
    /// Source file containing the originating operation.
    pub source_file: String,
    /// Starting source line recorded by the frontend.
    pub source_line: i32,
    /// Ending source line recorded by the frontend.
    pub source_end_line: i32,
    /// Starting source column recorded by the frontend.
    pub source_column: i32,
    /// Ending source column recorded by the frontend.
    pub source_end_column: i32,
    /// Frontend profile category identifiers.
    pub profile_type: Axes,
    /// Estimated generated machine-code size in bytes.
    pub size_of_generated_code_in_bytes: i64,
    /// Estimated memory working-set size in bytes.
    pub size_of_memory_working_set_in_bytes: i64,
    /// Optional singleton profiling record; multiple entries fail verification.
    pub profile_info: ProfileInfos,
    /// Name shared by deduplicated copies of the originating operation.
    pub deduplicated_name: String,
    /// Identifier of the associated source stack frame.
    pub stack_frame_id: i32,
    /// Frontend-provided name used for scheduling diagnostics.
    pub scheduling_name: String,
    /// Absent, inline, or module-referenced auxiliary metadata.
    pub metadata_payload: Payload,
}
impl Verify for OpMetadata {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.profile_info.0.len() > 1 {
            return Err(pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: "op_metadata",
                message: "multiple instruction profile records".into(),
            }));
        }
        self.metadata_payload.verify(ctx)
    }
}
list!(
    Metadata,
    "xlahlo.metadata",
    OpMetadata,
    "Instruction metadata records associated with a sharding annotation."
);
impl Default for OpMetadata {
    fn default() -> Self {
        Self {
            op_type: String::new(),
            op_name: String::new(),
            source_file: String::new(),
            source_line: 0,
            source_end_line: 0,
            source_column: 0,
            source_end_column: 0,
            profile_type: Axes(Vec::new()),
            size_of_generated_code_in_bytes: 0,
            size_of_memory_working_set_in_bytes: 0,
            profile_info: ProfileInfos::default(),
            deduplicated_name: String::new(),
            stack_frame_id: 0,
            scheduling_name: String::new(),
            metadata_payload: Payload::Absent,
        }
    }
}
impl OpMetadata {
    /// Construct the named-sharding marker for a non-sum reduction.
    pub(crate) fn reduction(max: bool) -> Self {
        Self {
            op_type: "sdy::reduction_op".into(),
            op_name: if max { "MAX" } else { "MIN" }.into(),
            ..Self::default()
        }
    }
}
record!(Statistic, "xlahlo.statistic", "Named visualization statistic with an exact f64 value encoding.", {
    /// Human-readable statistic label.
    stat_name: String,
    /// Exact IEEE 754 binary64 encoding of the statistic value.
    stat_val_bits: u64,
});
list!(
    Statistics,
    "xlahlo.statistics",
    Statistic,
    "Named visualization statistics."
);
record!(StatisticsViz, "xlahlo.statistics_viz", "Visualization statistics and the selected statistic index.", {
    /// Index of the statistic selected for visualization.
    stat_index_to_visualize: i64,
    /// Ordered statistics available for visualization.
    statistics: Statistics,
});
