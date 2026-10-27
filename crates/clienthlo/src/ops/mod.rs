//! CHLO operation definitions, builders, and verification.
//!
//! `new` constructors allocate operations with caller-supplied result types and
//! do not verify them. Use [`pliron::common_traits::Verify::verify`] before
//! consuming externally supplied or manually edited IR. Inferred constructors
//! check the constraints needed to determine result types; they are not a
//! substitute for full verification unless explicitly documented otherwise.
//!
//! Operand and result accessors assume the operation has its declared arity;
//! malformed IR can panic when an operand or result is missing. Attribute
//! accessors return `None` for absent or incorrectly typed attributes unless a
//! documented dialect default applies. Drop borrowed attribute guards before
//! mutating the operation. Raw setters do not revalidate the enclosing operation.
//!
//! Attribute storage names, including macro-generated key modules, are private
//! implementation details. Use typed operation accessors instead.
//!
//! ```compile_fail
//! use clienthlo::ops::broadcast_add_op_attr_names;
//! ```
//!
//! ```compile_fail
//! use clienthlo::ops::scan_op_attr_names;
//! ```

mod definitions;

pub(crate) use definitions::constant_element_value;
pub use definitions::{
    AcosOp, AcoshOp, AsinAcosKernelOp, AsinOp, AsinhOp, AtanOp, AtanhOp, BesselI1eOp,
    BroadcastAddOp, BroadcastAndOp, BroadcastAtan2Op, BroadcastCompareOp, BroadcastCompareOptions,
    BroadcastComplexOp, BroadcastDivideOp, BroadcastMaximumOp, BroadcastMinimumOp,
    BroadcastMultiplyOp, BroadcastNextAfterOp, BroadcastOrOp, BroadcastPolygammaOp,
    BroadcastPowerOp, BroadcastRemainderOp, BroadcastSelectOp, BroadcastShiftLeftOp,
    BroadcastShiftRightArithmeticOp, BroadcastShiftRightLogicalOp, BroadcastSubtractOp,
    BroadcastXorOp, BroadcastZetaOp, ConjOp, ConstantLikeOp, ConstantOp, CoshOp, DigammaOp,
    ErfInvOp, ErfOp, ErfcOp, IsInfOp, IsNegInfOp, IsPosInfOp, LgammaOp, MulhiOp, NextAfterOp,
    PolygammaOp, RaggedDotOp, RaggedDotOptions, ScanOp, SinhOp, SquareOp, TanOp, TopKOp,
    TopKOptions, ZetaOp,
};
