//! MLIR textual export for `math` operations.
//!
//! Every operation implements [`ToMlirOp`] and prints in MLIR's generic form.
//! The `math_fastmath` storage key becomes the upstream `fastmath` property
//! (`#arith.fastmath<..>`) and is omitted when `none`.

use crate::ops::*;
use pliron::{
    attribute::Attribute, context::Context, derive::op_interface_impl, op::Op, printable::State,
    result::Result,
};
use pliron_arith::{attributes::FastMathFlagsAttr, mlir::print_generic};
use pliron_mlir_interop::ToMlirOp;
use std::fmt;

/// Translates one `math`-owned storage key to an MLIR property.
///
/// Returns `Ok(None)` for keys `math` does not own and `Ok(Some((name, None)))`
/// when the property equals its default and is omitted.
///
/// # Errors
///
/// Returns an error when a `math` key holds an attribute of the wrong type.
pub fn property(
    _ctx: &Context,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    if key != "math_fastmath" {
        return Ok(None);
    }
    let flags = attr
        .downcast_ref::<FastMathFlagsAttr>()
        .ok_or_else(|| {
            pliron::input_error_noloc!(pliron_mlir_interop::Error::Untranslatable(format!(
                "attribute {key} has an unexpected type"
            )))
        })?
        .value;
    Ok(Some((
        "fastmath",
        (!flags.is_none()).then(|| format!("#arith.fastmath<{}>", flags.mlir_spelling())),
    )))
}

macro_rules! operations {
    ($($ty:ty),* $(,)?) => {$(
        #[op_interface_impl]
        impl ToMlirOp for $ty {
            fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                let name = self.get_opid().to_string();
                print_generic(ctx, self.get_operation(), state, f, &|ctx, key, attr| {
                    if matches!(name.as_str(), "math.absi" | "math.ctlz" | "math.cttz"
                        | "math.ctpop" | "math.ipowi" | "math.isnan" | "math.isinf"
                        | "math.isfinite" | "math.isnormal")
                    {
                        return Err(pliron::input_error_noloc!(
                            pliron_mlir_interop::Error::Untranslatable(
                                format!("attribute {key} is not a property of {name}")
                            )
                        ));
                    }
                    property(ctx, key, attr)
                })
            }
        }
    )*};
}

operations!(
    AbsFOp,
    CeilOp,
    FloorOp,
    RoundOp,
    RoundEvenOp,
    TruncOp,
    ExpOp,
    Exp2Op,
    ExpM1Op,
    LogOp,
    Log2Op,
    Log10Op,
    Log1pOp,
    SqrtOp,
    RsqrtOp,
    CbrtOp,
    SinOp,
    CosOp,
    TanOp,
    AsinOp,
    AcosOp,
    AtanOp,
    SinhOp,
    CoshOp,
    TanhOp,
    AsinhOp,
    AcoshOp,
    AtanhOp,
    ErfOp,
    ErfcOp,
    Atan2Op,
    PowFOp,
    CopySignOp,
    FmaOp,
    ClampFOp,
    SincosOp,
    FPowIOp,
    AbsIOp,
    CountLeadingZerosOp,
    CountTrailingZerosOp,
    CtPopOp,
    IPowIOp,
    IsNaNOp,
    IsInfOp,
    IsFiniteOp,
    IsNormalOp,
);
