//! Named attribute schemas from the pinned StablehloOps.td.

use crate::{Attribute, BuildError, ElementType, StableHloAttribute, StableHloOp, Type};

#[derive(Clone, Copy)]
enum Kind {
    Elements,
    Integer,
    Nonnegative,
    Positive,
    I32,
    I32Nonnegative,
    I32Positive,
    Float,
    String,
    Bool,
    Unit,
    Integers,
    Bools,
    DenseI64,
    Array,
    Dictionary,
    Symbol,
    Symbols,
    BackendConfig,
    ReplicaGroups,
    Precision,
    DotDimensions,
    DotAlgorithm,
    ConvDimensions,
    GatherDimensions,
    ScatterDimensions,
    Channel,
    Accuracy,
    Layouts,
    Tilings,
    Aliases,
    ApiVersion,
    CompositeName,
    Enum(&'static [&'static str]),
}

struct Rule {
    name: &'static str,
    kind: Kind,
    required: bool,
}

const fn required(name: &'static str, kind: Kind) -> Rule {
    Rule {
        name,
        kind,
        required: true,
    }
}

const fn optional(name: &'static str, kind: Kind) -> Rule {
    Rule {
        name,
        kind,
        required: false,
    }
}

use Kind::*;

pub(super) fn verify(
    opcode: StableHloOp,
    attributes: &crate::Attributes,
) -> Result<(), BuildError> {
    let rules: &[Rule] = match opcode {
        StableHloOp::Constant => &[required("value", Elements)],
        StableHloOp::Iota | StableHloOp::DynamicIota => &[required("iota_dimension", Nonnegative)],
        StableHloOp::Cbrt
        | StableHloOp::Cosine
        | StableHloOp::Exponential
        | StableHloOp::ExponentialMinusOne
        | StableHloOp::Log
        | StableHloOp::LogPlusOne
        | StableHloOp::Logistic
        | StableHloOp::Rsqrt
        | StableHloOp::Sine
        | StableHloOp::Sqrt
        | StableHloOp::Tan
        | StableHloOp::Tanh => &[optional("result_accuracy", Accuracy)],
        StableHloOp::Infeed => &[optional("infeed_config", String), optional("layout", Array)],
        StableHloOp::Outfeed => &[optional("outfeed_config", String)],
        StableHloOp::Send | StableHloOp::Recv => &[
            required("channel_handle", Channel),
            optional("is_host_transfer", Bool),
            optional("source_target_pairs", DenseI64),
        ],
        StableHloOp::AllGather => &[
            required("all_gather_dim", Nonnegative),
            required("replica_groups", ReplicaGroups),
            optional("channel_handle", Channel),
            optional("use_global_device_ids", Unit),
        ],
        StableHloOp::AllReduce => &[
            required("replica_groups", ReplicaGroups),
            optional("channel_handle", Channel),
            optional("use_global_device_ids", Unit),
        ],
        StableHloOp::ReduceScatter => &[
            required("scatter_dimension", Nonnegative),
            required("replica_groups", ReplicaGroups),
            optional("channel_handle", Channel),
            optional("use_global_device_ids", Unit),
        ],
        StableHloOp::AllToAll => &[
            required("split_dimension", Nonnegative),
            required("concat_dimension", Nonnegative),
            required("split_count", Positive),
            required("replica_groups", ReplicaGroups),
            optional("channel_handle", Channel),
        ],
        StableHloOp::Reduce | StableHloOp::Map | StableHloOp::Reverse => {
            &[required("dimensions", Integers)]
        }
        StableHloOp::GetTupleElement => &[required("index", I32Nonnegative)],
        StableHloOp::Compare => &[
            required(
                "comparison_direction",
                Enum(&["EQ", "NE", "GE", "GT", "LE", "LT"]),
            ),
            optional(
                "compare_type",
                Enum(&[
                    "NOTYPE",
                    "FLOAT",
                    "TOTALORDER",
                    "SIGNED",
                    "UNSIGNED",
                    "WEAKORDER",
                ]),
            ),
        ],
        StableHloOp::Slice => &[
            required("start_indices", Integers),
            required("limit_indices", Integers),
            required("strides", Integers),
        ],
        StableHloOp::DynamicSlice => &[required("slice_sizes", Integers)],
        StableHloOp::BatchNormGrad
        | StableHloOp::BatchNormInference
        | StableHloOp::BatchNormTraining => &[
            required("epsilon", Float),
            required("feature_index", Nonnegative),
        ],
        StableHloOp::Broadcast => &[required("broadcast_sizes", Integers)],
        StableHloOp::BroadcastInDim => &[required("broadcast_dimensions", Integers)],
        StableHloOp::DynamicBroadcastInDim => &[
            required("broadcast_dimensions", Integers),
            optional("known_expanding_dimensions", Integers),
            optional("known_nonexpanding_dimensions", Integers),
        ],
        StableHloOp::Cholesky => &[optional("lower", Bool)],
        StableHloOp::Concatenate
        | StableHloOp::GetDimensionSize
        | StableHloOp::SetDimensionSize => &[required("dimension", Nonnegative)],
        StableHloOp::CollectiveBroadcast => &[
            required("replica_groups", ReplicaGroups),
            optional("channel_handle", Channel),
            optional("has_dynamic_root", Unit),
        ],
        StableHloOp::CollectiveReduce => &[
            required("replica_groups", ReplicaGroups),
            optional("channel_handle", Channel),
            optional("use_global_device_ids", Unit),
            optional("has_dynamic_root", Unit),
        ],
        StableHloOp::CollectivePermute => &[
            required("source_target_pairs", DenseI64),
            optional("channel_handle", Channel),
        ],
        StableHloOp::Composite => &[
            required("name", CompositeName),
            optional("composite_attributes", Dictionary),
            required("decomposition", Symbol),
            optional("version", I32),
        ],
        StableHloOp::Convolution => &[
            optional("window_strides", Integers),
            optional("padding", DenseI64),
            optional("lhs_dilation", Integers),
            optional("rhs_dilation", Integers),
            optional("window_reversal", Bools),
            required("dimension_numbers", ConvDimensions),
            required("feature_group_count", Positive),
            required("batch_group_count", Positive),
            optional("precision_config", Precision),
        ],
        StableHloOp::DynamicConv => &[
            optional("window_strides", Integers),
            optional("lhs_dilation", Integers),
            optional("rhs_dilation", Integers),
            optional("window_reversal", Bools),
            required("dimension_numbers", ConvDimensions),
            required("feature_group_count", Positive),
            required("batch_group_count", Positive),
            optional("precision_config", Precision),
        ],
        StableHloOp::CrossReplicaSum => &[required("replica_groups", DenseI64)],
        StableHloOp::CustomCall => &[
            required("call_target_name", String),
            optional("has_side_effect", Bool),
            optional("backend_config", BackendConfig),
            optional("api_version", ApiVersion),
            optional("called_computations", Symbols),
            optional("operand_layouts", Layouts),
            optional("result_layouts", Layouts),
            optional("output_operand_aliases", Aliases),
            optional("result_tilings", Tilings),
        ],
        StableHloOp::Dot => &[optional("precision_config", Precision)],
        StableHloOp::DotGeneral => &[
            required("dot_dimension_numbers", DotDimensions),
            optional("precision_config", Precision),
            optional("algorithm", DotAlgorithm),
        ],
        StableHloOp::Einsum | StableHloOp::UnaryEinsum => &[required("einsum_config", String)],
        StableHloOp::Fft => &[
            required("fft_type", Enum(&["FFT", "IFFT", "RFFT", "IRFFT"])),
            required("fft_length", Integers),
        ],
        StableHloOp::Gather => &[
            required("dimension_numbers", GatherDimensions),
            required("slice_sizes", Integers),
            optional("indices_are_sorted", Bool),
        ],
        StableHloOp::DynamicGather => &[
            required("dimension_numbers", GatherDimensions),
            optional("indices_are_sorted", Bool),
        ],
        StableHloOp::Scatter => &[
            required("scatter_dimension_numbers", ScatterDimensions),
            optional("indices_are_sorted", Bool),
            optional("unique_indices", Bool),
        ],
        StableHloOp::SelectAndScatter => &[
            optional("window_dimensions", Integers),
            optional("window_strides", Integers),
            optional("padding", DenseI64),
        ],
        StableHloOp::Sort => &[optional("dimension", Integer), optional("is_stable", Bool)],
        StableHloOp::Pad => &[
            required("edge_padding_low", Integers),
            required("edge_padding_high", Integers),
            required("interior_padding", Integers),
        ],
        StableHloOp::Transpose => &[required("permutation", Integers)],
        StableHloOp::TriangularSolve => &[
            required("left_side", Bool),
            required("lower", Bool),
            required("unit_diagonal", Bool),
            required(
                "transpose_a",
                Enum(&["TRANSPOSE_INVALID", "NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"]),
            ),
        ],
        StableHloOp::ReduceWindow => &[
            required("window_dimensions", Integers),
            optional("window_strides", Integers),
            optional("base_dilations", Integers),
            optional("window_dilations", Integers),
            optional("padding", DenseI64),
        ],
        StableHloOp::TorchIndexSelect => {
            &[required("dim", Integer), required("batch_dims", Integer)]
        }
        StableHloOp::Rng => &[required("rng_distribution", Enum(&["UNIFORM", "NORMAL"]))],
        StableHloOp::RngBitGenerator => &[required(
            "rng_algorithm",
            Enum(&["DEFAULT", "THREE_FRY", "PHILOX"]),
        )],
        StableHloOp::ReducePrecision => &[
            required("exponent_bits", I32Positive),
            required("mantissa_bits", I32Nonnegative),
        ],
        StableHloOp::CreateToken
        | StableHloOp::Abs
        | StableHloOp::Ceil
        | StableHloOp::Convert
        | StableHloOp::CountLeadingZeros
        | StableHloOp::Floor
        | StableHloOp::Imag
        | StableHloOp::IsFinite
        | StableHloOp::Not
        | StableHloOp::Negate
        | StableHloOp::PopulationCount
        | StableHloOp::Real
        | StableHloOp::RoundNearestAfz
        | StableHloOp::RoundNearestEven
        | StableHloOp::Sign
        | StableHloOp::Add
        | StableHloOp::Atan2
        | StableHloOp::Complex
        | StableHloOp::Divide
        | StableHloOp::Maximum
        | StableHloOp::Minimum
        | StableHloOp::Multiply
        | StableHloOp::Power
        | StableHloOp::Remainder
        | StableHloOp::ShiftLeft
        | StableHloOp::ShiftRightArithmetic
        | StableHloOp::ShiftRightLogical
        | StableHloOp::Subtract
        | StableHloOp::And
        | StableHloOp::Or
        | StableHloOp::Xor
        | StableHloOp::ReplicaId
        | StableHloOp::PartitionId
        | StableHloOp::AfterAll
        | StableHloOp::If
        | StableHloOp::Case
        | StableHloOp::While
        | StableHloOp::AsyncStart
        | StableHloOp::AsyncDone
        | StableHloOp::Tuple
        | StableHloOp::DynamicUpdateSlice
        | StableHloOp::Clamp
        | StableHloOp::Reshape
        | StableHloOp::DynamicReshape
        | StableHloOp::BitcastConvert
        | StableHloOp::Select
        | StableHloOp::Return
        | StableHloOp::OptimizationBarrier
        | StableHloOp::UniformQuantize
        | StableHloOp::UniformDequantize
        | StableHloOp::RealDynamicSlice
        | StableHloOp::DynamicPad => &[],
    };
    for rule in rules {
        if match attributes.get(rule.name) {
            Some(value) => {
                let enum_name = match rule.name {
                    "compare_type" => "comparison_type",
                    "transpose_a" => "transpose",
                    name => name,
                };
                !rule.kind.accepts(value)
                    || matches!((&rule.kind, value), (Enum(_), Attribute::Enum { name, .. }) if name != enum_name)
            }
            None => rule.required,
        } {
            return Err(BuildError::MissingOrInvalidAttribute {
                operation: opcode.mnemonic(),
                attribute: rule.name,
            });
        }
    }
    Ok(())
}

impl Kind {
    fn accepts(self, attribute: &Attribute) -> bool {
        match self {
            Elements => matches!(
                attribute,
                Attribute::DenseElements { .. } | Attribute::SparseElements { .. }
            ),
            Integer => matches!(
                attribute,
                Attribute::Integer {
                    ty: ElementType::I64,
                    ..
                }
            ),
            Nonnegative => {
                matches!(attribute, Attribute::Integer { ty: ElementType::I64, value } if *value >= 0)
            }
            Positive => {
                matches!(attribute, Attribute::Integer { ty: ElementType::I64, value } if *value > 0)
            }
            I32 => matches!(
                attribute,
                Attribute::Integer {
                    ty: ElementType::I32,
                    ..
                }
            ),
            I32Nonnegative => {
                matches!(attribute, Attribute::Integer { ty: ElementType::I32, value } if (0..=i128::from(i32::MAX)).contains(value))
            }
            I32Positive => {
                matches!(attribute, Attribute::Integer { ty: ElementType::I32, value } if (1..=i128::from(i32::MAX)).contains(value))
            }
            Float => matches!(
                attribute,
                Attribute::Float {
                    ty: ElementType::F32,
                    ..
                }
            ),
            String => matches!(attribute, Attribute::String(_) | Attribute::Bytes(_)),
            Bool => matches!(attribute, Attribute::Bool(_)),
            ApiVersion => {
                matches!(attribute, Attribute::Integer { ty: ElementType::I32, value } if (0..=4).contains(value))
            }
            CompositeName => {
                matches!(attribute, Attribute::String(name) if valid_composite_name(name))
            }
            Unit => matches!(attribute, Attribute::Unit),
            Integers => matches!(
                attribute,
                Attribute::DenseArray {
                    element: ElementType::I64,
                    ..
                }
            ),
            Bools => matches!(
                attribute,
                Attribute::DenseArray {
                    element: ElementType::I1,
                    ..
                }
            ),
            DenseI64 => {
                matches!(attribute, Attribute::DenseElements { ty, .. } if matches!(ty.as_ref(),
                    Type::Tensor(tensor) if tensor.element() == ElementType::I64))
            }
            Array => matches!(attribute, Attribute::Array(_)),
            Dictionary => matches!(attribute, Attribute::Dictionary(_)),
            Symbol => {
                matches!(attribute, Attribute::SymbolRef(path) if path.len() == 1 && !path[0].is_empty())
            }
            Symbols => {
                matches!(attribute, Attribute::Array(values) if values.iter().all(|value| Symbol.accepts(value)))
            }
            BackendConfig => matches!(
                attribute,
                Attribute::String(_) | Attribute::Bytes(_) | Attribute::Dictionary(_)
            ),
            ReplicaGroups => {
                DenseI64.accepts(attribute)
                    || matches!(attribute,
                Attribute::StableHlo(value) if matches!(value.as_ref(), StableHloAttribute::ReplicaGroupMeshAxes(_)))
            }
            Enum(values) => matches!(attribute, Attribute::Enum { dialect, value, .. }
                if dialect == "stablehlo" && values.contains(&value.as_str())),
            Aliases => matches!(attribute, Attribute::Array(values) if values.iter().all(|value|
                matches!(value, Attribute::StableHlo(value) if matches!(value.as_ref(), StableHloAttribute::OutputOperandAlias(_))))),
            Precision => matches!(attribute, Attribute::Array(values) if values.len() <= 2 &&
                values.iter().all(|value| matches!(value,
                    Attribute::Enum { dialect, name, value }
                        if dialect == "stablehlo" && name == "precision" &&
                            ["DEFAULT", "HIGH", "HIGHEST"].contains(&value.as_str())))),
            Layouts => matches!(attribute, Attribute::Array(values) if
                values.iter().all(crate::attribute::builtin::is_index_tensor)),
            Tilings => matches!(attribute, Attribute::Array(values) if values.iter().all(|tiling|
                matches!(tiling, Attribute::Array(tiles) if
                    tiles.iter().all(crate::attribute::builtin::is_index_tensor)))),
            DotDimensions | DotAlgorithm | ConvDimensions | GatherDimensions
            | ScatterDimensions | Channel | Accuracy => {
                let Attribute::StableHlo(value) = attribute else {
                    return false;
                };
                matches!(
                    (self, value.as_ref()),
                    (DotDimensions, StableHloAttribute::DotDimensionNumbers(_))
                        | (DotAlgorithm, StableHloAttribute::DotAlgorithm(_))
                        | (ConvDimensions, StableHloAttribute::ConvDimensionNumbers(_))
                        | (
                            GatherDimensions,
                            StableHloAttribute::GatherDimensionNumbers(_)
                        )
                        | (
                            ScatterDimensions,
                            StableHloAttribute::ScatterDimensionNumbers(_)
                        )
                        | (Channel, StableHloAttribute::ChannelHandle(_))
                        | (Accuracy, StableHloAttribute::ResultAccuracy(_))
                )
            }
        }
    }
}

fn valid_composite_name(name: &str) -> bool {
    let mut parts = name.split('.');
    let Some(namespace) = parts.next() else {
        return false;
    };
    if !namespace
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphabetic)
        || !namespace
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return false;
    }
    let mut count = 0;
    for part in parts {
        if part.is_empty()
            || !part
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
        {
            return false;
        }
        count += 1;
    }
    count > 0
}
