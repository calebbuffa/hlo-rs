//! `arith` operation definitions, builders, and verification.
//!
//! `new` constructors allocate operations without verification. Binary
//! constructors take the result type from the left operand; comparison
//! constructors produce an `i1` result shaped like the operands; cast
//! constructors take an explicit result type, while `new_inferred` keeps the
//! input's container and replaces its element. Use
//! [`pliron::common_traits::Verify::verify`] before consuming externally
//! supplied or manually edited IR.
//!
//! Operand and result accessors assume the operation has its declared arity;
//! malformed IR can panic when an operand or result is missing. Optional flags
//! default to `none` when absent.

mod definitions;

pub use definitions::{
    AddFOp, AddIOp, AddUIExtendedOp, AndIOp, BitcastOp, CeilDivSIOp, CeilDivUIOp, CmpFOp, CmpIOp,
    ConstantOp, ConvertFOp, DivFOp, DivSIOp, DivUIOp, ExtFOp, ExtSIOp, ExtUIOp, FPToSIOp, FPToUIOp,
    FloorDivSIOp, FlushDenormalsOp, IndexCastOp, IndexCastUIOp, MaxNumFOp, MaxSIOp, MaxUIOp,
    MaximumFOp, MaximumNumFOp, MinNumFOp, MinSIOp, MinUIOp, MinimumFOp, MinimumNumFOp, MulFOp,
    MulIOp, MulSIExtendedOp, MulUIExtendedOp, NegFOp, OrIOp, RemFOp, RemSIOp, RemUIOp, SIToFPOp,
    ScalingExtFOp, ScalingTruncFOp, SelectOp, ShLIOp, ShRSIOp, ShRUIOp, SubFOp, SubIOp,
    SubUIExtendedOp, TruncFOp, TruncIOp, UIToFPOp, XOrIOp,
};
