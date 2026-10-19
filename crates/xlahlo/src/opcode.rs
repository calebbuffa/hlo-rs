//! OpenXLA HLO opcode names, operand arities and capabilities.
use pliron::derive::pliron_attr;
use std::{fmt, str::FromStr};

macro_rules! catalog {
    ($($variant:ident: $name:literal, $arity:expr;)+) => {
        #[pliron_attr(name = "hlo.opcode", format, verifier = "succ")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        /// Instruction kinds in the OpenXLA HLO catalog.
        pub enum Opcode { $(#[doc = concat!("The `", $name, "` instruction.")] $variant,)+ }
        impl Opcode {
            /// Every supported opcode, in upstream catalog order.
            pub const ALL: &'static [Self] = &[$(Self::$variant,)+];
            /// Return the exact upstream HLO spelling.
            pub const fn name(self) -> &'static str {match self {$(Self::$variant=>$name,)+}}
            /// Return fixed operand arity, or `None` for a variadic instruction.
            pub const fn arity(self) -> Option<usize> {match self {$(Self::$variant=>$arity,)+}}
        }
        impl FromStr for Opcode {
            type Err = pliron::result::Error;
            fn from_str(name: &str) -> Result<Self, Self::Err> {
                match name {
                    $($name=>Ok(Self::$variant),)+
                    _=>pliron::input_err_noloc!(crate::Diagnostic::UnknownOpcode {
                        name: name.to_owned(),
                    }),
                }
            }
        }
    };
}

catalog! {
    Abs: "abs", Some(1);
    Acos: "acos", Some(1);
    Acosh: "acosh", Some(1);
    Add: "add", Some(2);
    AddDependency: "add-dependency", Some(2);
    AfterAll: "after-all", None;
    AllGather: "all-gather", None;
    AllGatherDone: "all-gather-done", Some(1);
    AllGatherStart: "all-gather-start", None;
    AllReduce: "all-reduce", None;
    AllReduceDone: "all-reduce-done", Some(1);
    AllReduceStart: "all-reduce-start", None;
    AllToAll: "all-to-all", None;
    And: "and", Some(2);
    Asin: "asin", Some(1);
    Asinh: "asinh", Some(1);
    AsyncDone: "async-done", Some(1);
    AsyncStart: "async-start", None;
    AsyncUpdate: "async-update", None;
    Atan2: "atan2", Some(2);
    Atanh: "atanh", Some(1);
    BatchNormGrad: "batch-norm-grad", Some(5);
    BatchNormInference: "batch-norm-inference", Some(5);
    BatchNormTraining: "batch-norm-training", Some(3);
    Bitcast: "bitcast", Some(1);
    BitcastConvert: "bitcast-convert", Some(1);
    Broadcast: "broadcast", Some(1);
    Call: "call", None;
    Cbrt: "cbrt", Some(1);
    Ceil: "ceil", Some(1);
    Cholesky: "cholesky", Some(1);
    Clamp: "clamp", Some(3);
    Clz: "count-leading-zeros", Some(1);
    CollectiveBroadcast: "collective-broadcast", None;
    CollectivePermute: "collective-permute", None;
    CollectivePermuteDone: "collective-permute-done", Some(1);
    CollectivePermuteStart: "collective-permute-start", None;
    CollectiveReduce: "collective-reduce", None;
    Compare: "compare", Some(2);
    Complex: "complex", Some(2);
    Concatenate: "concatenate", None;
    Conditional: "conditional", None;
    Constant: "constant", Some(0);
    Convert: "convert", Some(1);
    Convolution: "convolution", None;
    Copy: "copy", Some(1);
    CopyDone: "copy-done", Some(1);
    CopyStart: "copy-start", Some(1);
    Cos: "cosine", Some(1);
    Cosh: "cosh", Some(1);
    CustomCall: "custom-call", None;
    Divide: "divide", Some(2);
    Domain: "domain", Some(1);
    Dot: "dot", None;
    DynamicReshape: "dynamic-reshape", None;
    DynamicSlice: "dynamic-slice", None;
    DynamicUpdateSlice: "dynamic-update-slice", None;
    Erf: "erf", Some(1);
    Exp: "exponential", Some(1);
    Exp2: "exp2", Some(1);
    Expm1: "exponential-minus-one", Some(1);
    Fft: "fft", Some(1);
    Floor: "floor", Some(1);
    Fusion: "fusion", None;
    Gather: "gather", Some(2);
    GetDimensionSize: "get-dimension-size", Some(1);
    GetTupleElement: "get-tuple-element", Some(1);
    Imag: "imag", Some(1);
    Infeed: "infeed", Some(1);
    Iota: "iota", Some(0);
    IsFinite: "is-finite", Some(1);
    Log: "log", Some(1);
    Log1p: "log-plus-one", Some(1);
    Log2: "log2", Some(1);
    Logistic: "logistic", Some(1);
    Map: "map", None;
    Maximum: "maximum", Some(2);
    Minimum: "minimum", Some(2);
    Mulhi: "mulhi", Some(2);
    Multiply: "multiply", Some(2);
    Negate: "negate", Some(1);
    Not: "not", Some(1);
    OptimizationBarrier: "opt-barrier", Some(1);
    Or: "or", Some(2);
    Outfeed: "outfeed", Some(2);
    Pad: "pad", Some(2);
    Parameter: "parameter", Some(0);
    PartitionId: "partition-id", Some(0);
    PopulationCount: "popcnt", Some(1);
    Power: "power", Some(2);
    RaggedAllToAll: "ragged-all-to-all", Some(6);
    RaggedDot: "ragged-dot", Some(3);
    Real: "real", Some(1);
    Recv: "recv", Some(1);
    RecvDone: "recv-done", Some(1);
    Reduce: "reduce", None;
    ReducePrecision: "reduce-precision", Some(1);
    ReduceScatter: "reduce-scatter", None;
    ReduceWindow: "reduce-window", None;
    Remainder: "remainder", Some(2);
    ReplicaId: "replica-id", Some(0);
    Reshape: "reshape", Some(1);
    Reverse: "reverse", Some(1);
    Rng: "rng", None;
    RngBitGenerator: "rng-bit-generator", Some(1);
    RngGetAndUpdateState: "rng-get-and-update-state", Some(0);
    RoundNearestAfz: "round-nearest-afz", Some(1);
    RoundNearestEven: "round-nearest-even", Some(1);
    Rsqrt: "rsqrt", Some(1);
    ScaledDot: "scaled-dot", Some(4);
    Scan: "scan", None;
    Scatter: "scatter", None;
    Select: "select", Some(3);
    SelectAndScatter: "select-and-scatter", Some(3);
    Send: "send", Some(2);
    SendDone: "send-done", Some(1);
    SetDimensionSize: "set-dimension-size", Some(2);
    ShiftLeft: "shift-left", Some(2);
    ShiftRightArithmetic: "shift-right-arithmetic", Some(2);
    ShiftRightLogical: "shift-right-logical", Some(2);
    Shuffle: "shuffle", Some(1);
    Sign: "sign", Some(1);
    Sin: "sine", Some(1);
    Sinh: "sinh", Some(1);
    Slice: "slice", Some(1);
    Sort: "sort", None;
    Sqrt: "sqrt", Some(1);
    StochasticConvert: "stochastic-convert", Some(2);
    Subtract: "subtract", Some(2);
    Tan: "tan", Some(1);
    Tanh: "tanh", Some(1);
    TopK: "topk", Some(1);
    Transpose: "transpose", Some(1);
    TriangularSolve: "triangular-solve", Some(2);
    Tuple: "tuple", None;
    While: "while", Some(1);
    Xor: "xor", Some(2);
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Opcode {
    pub const fn supports_result_accuracy(self) -> bool {
        matches!(
            self,
            Self::Acos
                | Self::Acosh
                | Self::Asin
                | Self::Asinh
                | Self::Atanh
                | Self::Cbrt
                | Self::Cos
                | Self::Cosh
                | Self::Erf
                | Self::Exp
                | Self::Exp2
                | Self::Expm1
                | Self::Log
                | Self::Log1p
                | Self::Log2
                | Self::Logistic
                | Self::Rsqrt
                | Self::Sin
                | Self::Sinh
                | Self::Sqrt
                | Self::Tan
                | Self::Tanh
        )
    }
}
