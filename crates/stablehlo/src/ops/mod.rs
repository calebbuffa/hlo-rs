//! StableHLO operation wrappers grouped behind one public operations namespace.
//!
//! Constructors that accept an explicit result type use that type as provided and
//! do not run operation verification. Builders named `new_inferred` document the
//! inference and input checks they perform; those checks do not replace full
//! operation verification. Call [`pliron::common_traits::Verify::verify`] before
//! relying on an operation's invariants.
//!
//! Operand, result, and region accessors assume the declared operation structure
//! is present and may panic on malformed IR. Attribute accessors return `None`
//! for absent or incorrectly typed attributes unless a documented default applies.
//! Borrowed attribute guards must be dropped before mutating their operation.
//! Raw setters update storage without revalidating the enclosing operation.
//! Macro-generated attribute-key modules remain private; the public namespace
//! contains only operation wrappers and their authoring options.
//!
//! ```compile_fail
//! use stablehlo::ops::constant_op_attr_names;
//! ```

use crate::Diagnostic;
use pliron::result::Result;

fn builder_option<T>(value: Option<T>, operation: &'static str, reason: &'static str) -> Result<T> {
    value.ok_or_else(|| {
        Diagnostic::InvalidOperation {
            operation: operation.to_owned(),
            location: "<builder>".to_owned(),
            reason,
        }
        .into()
    })
}

fn builder_result<T>(value: Result<T>, operation: &'static str) -> Result<T> {
    value.map_err(|error| Diagnostic::builder_failure(operation, error))
}

mod constants;
mod control;
mod elementwise;
mod shape;
mod tensor;
pub use constants::{ConstantOp, DynamicIotaOp};
pub use control::{
    AfterAllOp, AsyncDoneOp, AsyncStartOp, CallOp, CaseOp, CompositeOp, CreateTokenOp,
    CustomCallOp, FuncOp, GetTupleElementOp, IfOp, InfeedOp, ModuleOp, OptimizationBarrierOp,
    OutfeedOp, PartitionIdOp, RecvOp, ReplicaIdOp, ReturnOp, SendOp, TupleOp, WhileOp,
};
pub use elementwise::{
    AbsOp, AddOp, AndOp, Atan2Op, BitcastConvertOp, CbrtOp, CeilOp, ClampOp, CompareOp, ComplexOp,
    ConvertOp, CosineOp, CountLeadingZerosOp, DivideOp, Exp2Op, ExponentialMinusOneOp,
    ExponentialOp, FloorOp, ImagOp, IotaOp, IsFiniteOp, Log2Op, LogOp, LogPlusOneOp, LogisticOp,
    MaximumOp, MinimumOp, MultiplyOp, NegateOp, NotOp, OrOp, PopulationCountOp, PowerOp, RealOp,
    ReducePrecisionOp, RemainderOp, RoundNearestAfzOp, RoundNearestEvenOp, RsqrtOp, SelectOp,
    ShiftLeftOp, ShiftRightArithmeticOp, ShiftRightLogicalOp, SignOp, SineOp, SqrtOp, SubtractOp,
    TanOp, TanhOp, UniformDequantizeOp, UniformQuantizeOp, XorOp,
};
pub use shape::{
    BroadcastInDimOp, BroadcastOp, DynamicBroadcastInDimOp, DynamicReshapeOp, GetDimensionSizeOp,
    ReshapeOp, ReverseOp, SetDimensionSizeOp, TransposeOp,
};
pub use tensor::{
    ConcatenateOp, DynamicPadOp, DynamicSliceOp, DynamicUpdateSliceOp, PadOp, RealDynamicSliceOp,
    SliceOp,
};
mod collective;
mod compute;
mod indexing;
mod reduction;
pub use collective::{
    AllGatherOp, AllReduceOp, AllToAllOp, CollectiveBroadcastOp, CollectivePermuteOp,
    CollectiveReduceOp, CrossReplicaSumOp, ReduceScatterOp,
};
pub use compute::{
    BatchNormGradOp, BatchNormGradOperands, BatchNormInferenceOp, BatchNormInferenceOperands,
    BatchNormOptions, BatchNormTrainingOp, CholeskyOp, ConvolutionOp, ConvolutionOptions,
    DotGeneralOp, DotOp, DynamicConvOp, DynamicConvolutionOptions, EinsumOp, FftOp,
    RngBitGeneratorOp, RngOp, TransposeMode, TriangularSolveOp, TriangularSolveOptions,
    UnaryEinsumOp,
};
pub use indexing::{DynamicGatherOp, GatherOp, ScatterOp, TorchIndexSelectOp};
pub use reduction::{
    MapOp, ReduceOp, ReduceWindowOp, ReduceWindowOptions, SelectAndScatterOp,
    SelectAndScatterOptions, SortOp,
};
