//! Control-flow, communication, and backend configuration attributes.

use super::*;
use pliron::derive::{def_attribute, pliron_attr};

#[pliron_attr(name = "xlahlo.custom_call_schedule", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Scheduling constraint applied to a custom call.
pub enum CustomCallSchedule {
    /// Imposes no additional placement preference.
    None,
    /// Prefers the latest legal position in the schedule.
    Latest,
    /// Prefers the earliest legal position in the schedule.
    Earliest,
}
#[pliron_attr(name = "xlahlo.custom_call_api_version", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// ABI version used to invoke a custom-call target.
pub enum CustomCallApiVersion {
    /// No ABI version was explicitly selected.
    Unspecified,
    /// Original custom-call ABI without a status return.
    Original,
    /// Status-returning custom-call ABI.
    StatusReturning,
    /// Unified status-returning custom-call ABI.
    StatusReturningUnified,
    /// Typed foreign-function interface ABI.
    TypedFfi,
}
#[def_attribute("xlahlo.collective_op_group_mode")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Interpretation of device IDs in collective replica groups.
pub enum CollectiveOpGroupMode {
    /// Groups replica IDs within each partition.
    CrossReplica,
    /// Groups partition IDs within each replica.
    CrossPartition,
    /// Communicates across both replica and partition dimensions.
    CrossReplicaAndPartition,
    /// Interprets participants as flattened global device IDs.
    FlattenedId,
}
#[def_attribute("xlahlo.channel_type")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i64)]
/// Communication direction represented by a channel handle.
pub enum ChannelKind {
    /// Communication between devices.
    DeviceToDevice = 1,
    /// Communication from a device to the host.
    DeviceToHost = 2,
    /// Communication from the host to a device.
    HostToDevice = 3,
}

impl From<ChannelKind> for i64 {
    fn from(value: ChannelKind) -> Self {
        value as i64
    }
}

impl TryFrom<i64> for ChannelKind {
    type Error = ChannelKindValueError;

    fn try_from(value: i64) -> std::result::Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::DeviceToDevice),
            2 => Ok(Self::DeviceToHost),
            3 => Ok(Self::HostToDevice),
            _ => Err(ChannelKindValueError(value)),
        }
    }
}

/// An integer outside the channel-direction identifiers `1`, `2`, and `3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown channel kind identifier {0}; expected 1, 2, or 3")]
pub struct ChannelKindValueError(i64);

impl ChannelKindValueError {
    /// Returns the rejected numeric channel-direction identifier.
    pub const fn value(self) -> i64 {
        self.0
    }
}

#[pliron_attr(name = "xlahlo.channel_handle", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Communication channel identity and channel kind.
pub struct ChannelHandle {
    /// Numeric identity shared by matching communication operations.
    pub handle: i64,
    /// Direction of communication between the host and devices.
    pub channel_kind: ChannelKind,
}
#[pliron_attr(name = "xlahlo.payload", format, verifier = "succ")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Absent metadata, inline exact bytes, or an index into module metadata payloads.
pub enum Payload {
    /// No metadata payload is attached.
    Absent,
    /// Exact bytes stored directly in this attribute.
    Value(Bytes),
    /// Index of a payload stored in the enclosing module.
    Id(i64),
}
record!(SliceDimensions, "xlahlo.slice_dimensions", "Start, exclusive limit, and stride along one slice axis.", {
    /// Inclusive first selected index.
    start: i64,
    /// Exclusive upper bound on selected indices.
    limit: i64,
    /// Positive distance between selected indices.
    stride: i64,
});
record!(ReducePrecision, "xlahlo.reduce_precision", "Exponent and mantissa widths retained by reduce-precision.", {
    /// Number of exponent bits retained; must be at least one.
    exponent_bits: i32,
    /// Number of fraction bits retained; must be nonnegative.
    mantissa_bits: i32,
});
record!(BatchNormOptions, "xlahlo.batch_norm_options", "Batch normalization feature axis and exact f32 epsilon encoding.", {
    /// Exact IEEE 754 binary32 encoding of the variance-stabilizing epsilon.
    epsilon_bits: u32,
    /// Input axis containing feature channels.
    feature_index: i64,
});
record!(KnownInitStep, "xlahlo.known_init_step", "Known initial value and increment of a loop induction variable.", {
    /// Initial induction-variable value.
    init: i64,
    /// Increment applied after each loop iteration.
    step: i64,
});
#[pliron_attr(name = "xlahlo.optional_integer", format, verifier = "succ")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// An integer whose absence is distinct from every numeric value.
pub enum OptionalInteger {
    /// No integer was specified.
    Absent,
    /// The explicitly specified integer, including zero or negative values.
    Value(i64),
}
impl OptionalInteger {
    /// Borrow the integer, if present.
    pub fn as_ref(&self) -> Option<&i64> {
        match self {
            Self::Absent => None,
            Self::Value(value) => Some(value),
        }
    }
    /// Whether an integer is present.
    pub fn is_some(&self) -> bool {
        self.as_ref().is_some()
    }
    /// Whether the integer is absent.
    pub fn is_none(&self) -> bool {
        self.as_ref().is_none()
    }
}

impl From<Option<i64>> for OptionalInteger {
    fn from(value: Option<i64>) -> Self {
        match value {
            Some(value) => Self::Value(value),
            None => Self::Absent,
        }
    }
}

impl From<OptionalInteger> for Option<i64> {
    fn from(value: OptionalInteger) -> Self {
        match value {
            OptionalInteger::Value(value) => Some(value),
            OptionalInteger::Absent => None,
        }
    }
}

record!(DynamicVariable, "xlahlo.dynamic_variable", "A loop-carried tuple member with optional known initial value and step.", {
    /// Index of this variable in the loop-carried state tuple.
    tuple_index: i64,
    /// Known initial value, if established by analysis.
    init: OptionalInteger,
    /// Known per-iteration increment, if established by analysis.
    step: OptionalInteger,
});
list!(
    DynamicVariables,
    "xlahlo.dynamic_variables",
    DynamicVariable,
    "Loop-carried variables described by a while-loop backend configuration."
);
list!(
    KnownInitSteps,
    "xlahlo.known_init_steps",
    KnownInitStep,
    "Optional singleton initial-value and step record for a loop."
);
record!(WhileLoopBackendConfig, "xlahlo.while_loop_backend_config", "Known trip-count and induction-variable information for a while loop.", {
    /// Exact iteration count when known.
    known_trip_count: OptionalInteger,
    /// Optional singleton initial-value and step annotation.
    known_init_step: KnownInitSteps,
    /// State tuple index of the known induction variable, if any.
    known_induction_variable: OptionalInteger,
    /// Additional loop-carried variables with known initial values or steps.
    dynamic_variables: DynamicVariables,
});

record!(SourceTarget, "xlahlo.source_target", "One directed device edge of a collective permute.", {
    /// Device ID sending the value.
    source: i64,
    /// Device ID receiving the value.
    target: i64,
});
list!(
    SourceTargetPairs,
    "xlahlo.source_target_pairs",
    SourceTarget,
    "Directed device edges of a collective permute."
);
list!(
    ReplicaGroup,
    "xlahlo.replica_group",
    i64,
    "Device identifiers participating in one collective group."
);
list!(
    ReplicaGroups,
    "xlahlo.replica_groups",
    ReplicaGroup,
    "Explicit collective groups, preserving device order."
);
list!(
    ParameterReplication,
    "xlahlo.parameter_replication",
    bool,
    "Replication flags for parameter shape leaves."
);
record!(OutputOperandAliasing, "xlahlo.output_operand_aliasing", "An output subshape aliasing a particular operand subshape.", {
    /// Tuple path selecting the aliased output subshape.
    output_shape_index: Axes,
    /// Index of the operand supplying the shared buffer.
    operand_index: i64,
    /// Tuple path selecting the operand's aliased subshape.
    operand_shape_index: Axes,
});
list!(
    OutputOperandAliases,
    "xlahlo.output_operand_aliases",
    OutputOperandAliasing,
    "Declared output-to-operand buffer aliases."
);
