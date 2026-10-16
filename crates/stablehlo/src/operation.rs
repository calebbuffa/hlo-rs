use crate::Attributes;

/// Operation names from the pinned StableHLO dialect.
///
/// This catalog makes the dialect surface explicit. A catalog entry is not by
/// itself proof of operation verification, inference, or execution semantics.
macro_rules! stablehlo_ops {
    ($($variant:ident => $name:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum StableHloOp {
            $($variant),+
        }

        impl StableHloOp {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            pub const fn mnemonic(self) -> &'static str {
                match self {
                    $(Self::$variant => $name),+
                }
            }
        }
    }
}

stablehlo_ops! {
    Constant => "constant",
    Iota => "iota",
    DynamicIota => "dynamic_iota",
    CreateToken => "create_token",
    Abs => "abs",
    Cbrt => "cbrt",
    Ceil => "ceil",
    Convert => "convert",
    CountLeadingZeros => "count_leading_zeros",
    Cosine => "cosine",
    Exponential => "exponential",
    ExponentialMinusOne => "exponential_minus_one",
    Floor => "floor",
    Imag => "imag",
    IsFinite => "is_finite",
    Log => "log",
    LogPlusOne => "log_plus_one",
    Logistic => "logistic",
    Not => "not",
    Negate => "negate",
    PopulationCount => "popcnt",
    Real => "real",
    RoundNearestAfz => "round_nearest_afz",
    RoundNearestEven => "round_nearest_even",
    Rsqrt => "rsqrt",
    Sign => "sign",
    Sine => "sine",
    Sqrt => "sqrt",
    Tan => "tan",
    Tanh => "tanh",
    Add => "add",
    Atan2 => "atan2",
    Complex => "complex",
    Divide => "divide",
    Maximum => "maximum",
    Minimum => "minimum",
    Multiply => "multiply",
    Power => "power",
    Remainder => "remainder",
    ShiftLeft => "shift_left",
    ShiftRightArithmetic => "shift_right_arithmetic",
    ShiftRightLogical => "shift_right_logical",
    Subtract => "subtract",
    And => "and",
    Or => "or",
    Xor => "xor",
    Infeed => "infeed",
    Outfeed => "outfeed",
    Send => "send",
    Recv => "recv",
    ReplicaId => "replica_id",
    PartitionId => "partition_id",
    AfterAll => "after_all",
    If => "if",
    Case => "case",
    While => "while",
    AllGather => "all_gather",
    AllReduce => "all_reduce",
    ReduceScatter => "reduce_scatter",
    AllToAll => "all_to_all",
    Reduce => "reduce",
    AsyncStart => "async_start",
    AsyncDone => "async_done",
    GetTupleElement => "get_tuple_element",
    Tuple => "tuple",
    Compare => "compare",
    Slice => "slice",
    DynamicSlice => "dynamic_slice",
    DynamicUpdateSlice => "dynamic_update_slice",
    BatchNormGrad => "batch_norm_grad",
    BatchNormInference => "batch_norm_inference",
    BatchNormTraining => "batch_norm_training",
    BitcastConvert => "bitcast_convert",
    Broadcast => "broadcast",
    BroadcastInDim => "broadcast_in_dim",
    DynamicBroadcastInDim => "dynamic_broadcast_in_dim",
    Cholesky => "cholesky",
    Clamp => "clamp",
    Concatenate => "concatenate",
    CollectiveBroadcast => "collective_broadcast",
    CollectiveReduce => "collective_reduce",
    CollectivePermute => "collective_permute",
    Composite => "composite",
    Convolution => "convolution",
    CrossReplicaSum => "cross-replica-sum",
    CustomCall => "custom_call",
    Dot => "dot",
    DotGeneral => "dot_general",
    Einsum => "einsum",
    UnaryEinsum => "unary_einsum",
    Fft => "fft",
    Gather => "gather",
    GetDimensionSize => "get_dimension_size",
    Map => "map",
    Reshape => "reshape",
    DynamicReshape => "dynamic_reshape",
    Scatter => "scatter",
    Select => "select",
    SelectAndScatter => "select_and_scatter",
    SetDimensionSize => "set_dimension_size",
    Sort => "sort",
    Reverse => "reverse",
    Pad => "pad",
    Transpose => "transpose",
    TriangularSolve => "triangular_solve",
    ReduceWindow => "reduce_window",
    Return => "return",
    TorchIndexSelect => "torch_index_select",
    OptimizationBarrier => "optimization_barrier",
    Rng => "rng",
    RngBitGenerator => "rng_bit_generator",
    UniformQuantize => "uniform_quantize",
    UniformDequantize => "uniform_dequantize",
    ReducePrecision => "reduce_precision",
    RealDynamicSlice => "real_dynamic_slice",
    DynamicPad => "dynamic_pad",
    DynamicGather => "dynamic_gather",
    DynamicConv => "dynamic_conv"
}

/// StableHLO operation record with generic attributes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Parameter {
        number: u32,
        name: String,
    },
    StableHlo {
        opcode: StableHloOp,
        attributes: Attributes,
    },
    Call {
        attributes: Attributes,
    },
}

impl Operation {
    /// Returns a fixed operand count for operations whose ODS signature has no
    /// variadic operand groups.
    pub fn expected_arity(&self) -> Option<usize> {
        match self {
            Self::Parameter { .. } => Some(0),
            Self::Call { .. } => None,
            Self::StableHlo { opcode, .. } => opcode.fixed_operand_arity(),
        }
    }

    /// Returns the result count when the StableHLO operation has fixed
    /// cardinality. Variadic result operations are left unconstrained here.
    pub fn expected_result_arity(&self) -> Option<usize> {
        match self {
            Self::Parameter { .. } => Some(1),
            Self::Call { .. } => None,
            Self::StableHlo { opcode, .. } => opcode.fixed_result_arity(),
        }
    }

    /// Returns the exact number of regions declared by the operation's ODS.
    pub fn expected_region_arity(&self) -> Option<usize> {
        match self {
            Self::Parameter { .. } | Self::Call { .. } => Some(0),
            Self::StableHlo { opcode, .. } => match opcode {
                StableHloOp::If | StableHloOp::While | StableHloOp::SelectAndScatter => Some(2),
                StableHloOp::AllReduce
                | StableHloOp::ReduceScatter
                | StableHloOp::Reduce
                | StableHloOp::AsyncStart
                | StableHloOp::CollectiveReduce
                | StableHloOp::Map
                | StableHloOp::Scatter
                | StableHloOp::Sort
                | StableHloOp::ReduceWindow => Some(1),
                StableHloOp::Case | StableHloOp::Composite => None,
                _ => Some(0),
            },
        }
    }

    pub fn minimum_region_arity(&self) -> usize {
        match self {
            Self::StableHlo {
                opcode: StableHloOp::Case,
                ..
            } => 1,
            _ => 0,
        }
    }

    pub fn expected_region_block_count(&self) -> Option<usize> {
        match self {
            Self::StableHlo {
                opcode: StableHloOp::Composite,
                ..
            } => None,
            _ if self.expected_region_arity().is_some_and(|count| count > 0)
                || matches!(
                    self,
                    Self::StableHlo {
                        opcode: StableHloOp::Case,
                        ..
                    }
                ) =>
            {
                Some(1)
            }
            _ => None,
        }
    }

    pub fn requires_region_return(&self) -> bool {
        matches!(
            self,
            Self::StableHlo {
                opcode: StableHloOp::If
                    | StableHloOp::Case
                    | StableHloOp::While
                    | StableHloOp::AllReduce
                    | StableHloOp::ReduceScatter
                    | StableHloOp::Reduce
                    | StableHloOp::AsyncStart
                    | StableHloOp::CollectiveReduce
                    | StableHloOp::Map
                    | StableHloOp::Scatter
                    | StableHloOp::SelectAndScatter
                    | StableHloOp::Sort
                    | StableHloOp::ReduceWindow,
                ..
            }
        )
    }

    /// Whether a region may capture values from its enclosing computation.
    ///
    /// `async_start` is isolated from surrounding values. Its operands are
    /// represented inside the region by corresponding block arguments.
    pub fn is_isolated_from_above(&self) -> bool {
        matches!(
            self,
            Self::StableHlo {
                opcode: StableHloOp::AsyncStart,
                ..
            }
        )
    }

    pub fn is_parameter(&self) -> bool {
        matches!(self, Self::Parameter { .. })
    }

    /// Returns true only when purity is explicitly established by the dialect
    /// contract understood by this crate.
    pub fn is_pure(&self) -> bool {
        matches!(
            self,
            Self::StableHlo {
                opcode: StableHloOp::Abs
                    | StableHloOp::Add
                    | StableHloOp::And
                    | StableHloOp::Atan2
                    | StableHloOp::Cbrt
                    | StableHloOp::Ceil
                    | StableHloOp::Compare
                    | StableHloOp::Complex
                    | StableHloOp::Convert
                    | StableHloOp::Cosine
                    | StableHloOp::CountLeadingZeros
                    | StableHloOp::Divide
                    | StableHloOp::Exponential
                    | StableHloOp::ExponentialMinusOne
                    | StableHloOp::Floor
                    | StableHloOp::Imag
                    | StableHloOp::IsFinite
                    | StableHloOp::Log
                    | StableHloOp::LogPlusOne
                    | StableHloOp::Logistic
                    | StableHloOp::Maximum
                    | StableHloOp::Minimum
                    | StableHloOp::Multiply
                    | StableHloOp::Negate
                    | StableHloOp::Not
                    | StableHloOp::Or
                    | StableHloOp::PopulationCount
                    | StableHloOp::Power
                    | StableHloOp::Remainder
                    | StableHloOp::RoundNearestAfz
                    | StableHloOp::RoundNearestEven
                    | StableHloOp::Rsqrt
                    | StableHloOp::ShiftLeft
                    | StableHloOp::ShiftRightArithmetic
                    | StableHloOp::ShiftRightLogical
                    | StableHloOp::Sign
                    | StableHloOp::Sine
                    | StableHloOp::Sqrt
                    | StableHloOp::Subtract
                    | StableHloOp::Tan
                    | StableHloOp::Tanh
                    | StableHloOp::Xor,
                ..
            }
        )
    }

    pub fn attributes(&self) -> Option<&Attributes> {
        match self {
            Self::Parameter { .. } => None,
            Self::StableHlo { attributes, .. } | Self::Call { attributes } => Some(attributes),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Parameter { .. } => "parameter",
            Self::Call { .. } => "func.call",
            Self::StableHlo { opcode, .. } => opcode.mnemonic(),
        }
    }

    pub fn stablehlo(opcode: StableHloOp, attributes: Attributes) -> Self {
        Self::StableHlo { opcode, attributes }
    }

    pub fn parameter(number: u32, name: impl Into<String>) -> Self {
        Self::Parameter {
            number,
            name: name.into(),
        }
    }

    /// Constructs a direct `func.call` to a module-local function symbol.
    pub fn call(callee: impl Into<String>) -> Self {
        Self::Call {
            attributes: [(
                "callee".into(),
                crate::Attribute::SymbolRef(vec![callee.into()]),
            )]
            .into(),
        }
    }
}
