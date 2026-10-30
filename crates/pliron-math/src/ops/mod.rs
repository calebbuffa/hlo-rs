//! `math` operation definitions, builders, and verification.
//!
//! `new` constructors allocate operations without verification and take the
//! result type from the first operand, except classification operations,
//! which produce `i1` shaped like the operand. Use
//! [`pliron::common_traits::Verify::verify`] before consuming externally
//! supplied or manually edited IR.
//!
//! Operand and result accessors assume the operation has its declared arity;
//! malformed IR can panic when an operand or result is missing. Fast-math flags
//! default to `none` when absent.

mod definitions;

pub use definitions::{
    AbsFOp, AbsIOp, AcosOp, AcoshOp, AsinOp, AsinhOp, Atan2Op, AtanOp, AtanhOp, CbrtOp, CeilOp,
    ClampFOp, CopySignOp, CosOp, CoshOp, CountLeadingZerosOp, CountTrailingZerosOp, CtPopOp, ErfOp,
    ErfcOp, Exp2Op, ExpM1Op, ExpOp, FPowIOp, FloorOp, FmaOp, IPowIOp, IsFiniteOp, IsInfOp, IsNaNOp,
    IsNormalOp, Log1pOp, Log2Op, Log10Op, LogOp, PowFOp, RoundEvenOp, RoundOp, RsqrtOp, SinOp,
    SincosOp, SinhOp, SqrtOp, TanOp, TanhOp, TruncOp,
};
