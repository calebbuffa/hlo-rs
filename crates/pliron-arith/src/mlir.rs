//! MLIR textual export for `arith` operations.
//!
//! Every operation implements [`ToMlirOp`] and prints in MLIR's generic form,
//! translating typed storage to upstream property names:
//!
//! | storage key | MLIR property | text |
//! |---|---|---|
//! | `arith_fastmath` | `fastmath` | `#arith.fastmath<nnan,ninf>` |
//! | `arith_overflow_flags` | `overflowFlags` | `#arith.overflow<nsw, nuw>` |
//! | `arith_cmpf_predicate`, `arith_cmpi_predicate` | `predicate` | `1 : i64` |
//! | `arith_roundingmode` | `roundingmode` | `1 : i32` |
//! | `arith_is_exact` | `isExact` | `unit` |
//! | `arith_nneg` | `nonNeg` | `unit` |
//! | `arith_constant_value` | `value` | builtin attribute text |
//!
//! Flags that are `none` or `false` are omitted, matching MLIR's defaults.

use crate::{attributes::*, ops::*};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    derive::op_interface_impl,
    op::Op,
    operation::Operation,
    printable::State,
    result::Result,
};
use pliron_mlir_interop::{ToMlirOp, printers::GenericOp};
use std::fmt;

/// Re-export of the framework translation interface implemented by every operation.
pub use pliron_mlir_interop::ToMlirOp as MlirOperation;

fn untranslatable(message: String) -> pliron::result::Error {
    pliron::input_error_noloc!(pliron_mlir_interop::Error::Untranslatable(message))
}

/// Translates one `arith`-owned storage key to an MLIR property.
///
/// Returns `Ok(None)` when the key is not an `arith` key, `Ok(Some((name, None)))`
/// when the property equals its default and is omitted.
///
/// # Errors
///
/// Returns an error when an `arith` key holds an attribute of the wrong type.
pub fn property(
    ctx: &Context,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    let wrong = || untranslatable(format!("attribute {key} has an unexpected type"));
    Ok(Some(match key {
        "arith_fastmath" => {
            let flags = attr
                .downcast_ref::<FastMathFlagsAttr>()
                .ok_or_else(wrong)?
                .value;
            (
                "fastmath",
                (!flags.is_none()).then(|| format!("#arith.fastmath<{}>", flags.mlir_spelling())),
            )
        }
        "arith_overflow_flags" => {
            let flags = attr
                .downcast_ref::<IntegerOverflowFlagsAttr>()
                .ok_or_else(wrong)?
                .value;
            (
                "overflowFlags",
                (!flags.is_none()).then(|| format!("#arith.overflow<{}>", flags.mlir_spelling())),
            )
        }
        "arith_cmpf_predicate" => {
            let value = attr
                .downcast_ref::<CmpFPredicateAttr>()
                .ok_or_else(wrong)?
                .value;
            ("predicate", Some(format!("{} : i64", u64::from(value))))
        }
        "arith_cmpi_predicate" => {
            let value = attr
                .downcast_ref::<CmpIPredicateAttr>()
                .ok_or_else(wrong)?
                .value;
            ("predicate", Some(format!("{} : i64", u64::from(value))))
        }
        "arith_roundingmode" => {
            let value = attr
                .downcast_ref::<RoundingModeAttr>()
                .ok_or_else(wrong)?
                .value;
            ("roundingmode", Some(format!("{} : i32", u64::from(value))))
        }
        "arith_is_exact" | "arith_nneg" => {
            let set = bool::from(
                attr.downcast_ref::<pliron::builtin::attributes::BoolAttr>()
                    .ok_or_else(wrong)?
                    .clone(),
            );
            let name = if key == "arith_is_exact" {
                "isExact"
            } else {
                "nonNeg"
            };
            (name, set.then(|| "unit".to_owned()))
        }
        "arith_constant_value" => (
            "value",
            Some(stablehlo::mlir::assembly::print_attribute(ctx, attr)?),
        ),
        _ => return Ok(None),
    }))
}

/// Prints `op` in generic form after verifying it.
///
/// Attribute keys are translated by `translate`, which returns the MLIR property
/// name and text (or `None` text to omit a default). Keys it does not recognize
/// are rejected, so silently dropped semantics cannot reach MLIR.
///
/// # Errors
///
/// Returns an error when verification fails, an attribute is not translatable,
/// or an operand or result type has no MLIR translation.
pub fn print_generic(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
    translate: &Translate<'_>,
) -> Result<()> {
    let name = Operation::get_op_dyn(op, ctx).get_opid().to_string();
    print_generic_named(ctx, op, &name, state, f, translate)
}

/// Translates one storage key to an MLIR property name and text; see
/// [`print_generic`].
pub type Translate<'a> =
    dyn Fn(&Context, &str, &dyn Attribute) -> Result<Option<(&'static str, Option<String>)>> + 'a;

/// Like [`print_generic`], but prints `name` instead of the pliron operation
/// name, for upstream names that are not valid pliron identifiers (such as
/// `iree_linalg_ext.winograd.input_transform`).
///
/// # Errors
///
/// Same as [`print_generic`].
pub fn print_generic_named(
    ctx: &Context,
    op: Ptr<Operation>,
    name: &str,
    state: &State,
    f: &mut fmt::Formatter<'_>,
    translate: &Translate<'_>,
) -> Result<()> {
    Operation::get_op_dyn(op, ctx).verify(ctx)?;
    let mut properties = Vec::new();
    let mut translated = std::collections::HashSet::new();
    {
        let raw = op.deref(ctx);
        let mut entries = raw.attributes.0.iter().collect::<Vec<_>>();
        entries.sort_by(|(a, _), (b, _)| a.as_ref().cmp(b.as_ref()));
        for (key, attr) in entries {
            if AsRef::<str>::as_ref(key) == "builtin_given_names" {
                continue;
            }
            match translate(ctx, key.as_ref(), &**attr)? {
                Some((property, text)) => {
                    if !translated.insert(property) {
                        return Err(untranslatable(format!(
                            "duplicate property {property} on {name}"
                        )));
                    }
                    if let Some(text) = text {
                        properties.push((property, text));
                    }
                }
                None => {
                    return Err(untranslatable(format!(
                        "unmapped attribute {key} on {name}"
                    )));
                }
            }
        }
    }
    let mut generic = GenericOp::new(op, name);
    for (property, text) in properties {
        generic = generic.prop_raw(property, text);
    }
    generic.print(ctx, state, f)
}

fn property_for(
    ctx: &Context,
    name: &str,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    let mnemonic = name.strip_prefix("arith.").unwrap_or(name);
    let allowed = match key {
        "arith_fastmath" => matches!(
            mnemonic,
            "addf"
                | "subf"
                | "mulf"
                | "divf"
                | "remf"
                | "negf"
                | "flush_denormals"
                | "maximumf"
                | "minimumf"
                | "maxnumf"
                | "minnumf"
                | "maximumnumf"
                | "minimumnumf"
                | "cmpf"
                | "extf"
                | "truncf"
                | "convertf"
                | "scaling_extf"
                | "scaling_truncf"
        ),
        "arith_overflow_flags" => matches!(mnemonic, "addi" | "subi" | "muli" | "shli" | "trunci"),
        "arith_roundingmode" => matches!(
            mnemonic,
            "addf" | "subf" | "mulf" | "divf" | "truncf" | "convertf" | "scaling_truncf"
        ),
        "arith_is_exact" => matches!(mnemonic, "divsi" | "divui" | "shrsi" | "shrui"),
        "arith_nneg" => matches!(mnemonic, "extui" | "uitofp" | "index_castui"),
        "arith_cmpf_predicate" => mnemonic == "cmpf",
        "arith_cmpi_predicate" => mnemonic == "cmpi",
        "arith_constant_value" => mnemonic == "constant",
        _ => false,
    };
    if !allowed {
        return Err(untranslatable(format!(
            "attribute {key} is not a property of {name}"
        )));
    }
    property(ctx, key, attr)
}

macro_rules! operations {
    ($($ty:ty),* $(,)?) => {$(
        #[op_interface_impl]
        impl ToMlirOp for $ty {
            fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                let name = self.get_opid().to_string();
                print_generic(ctx, self.get_operation(), state, f, &|ctx, key, attr| {
                    property_for(ctx, &name, key, attr)
                })
            }
        }
    )*};
}

operations!(
    ConstantOp,
    AddFOp,
    SubFOp,
    MulFOp,
    DivFOp,
    RemFOp,
    NegFOp,
    FlushDenormalsOp,
    MaximumFOp,
    MinimumFOp,
    MaxNumFOp,
    MinNumFOp,
    MaximumNumFOp,
    MinimumNumFOp,
    AddIOp,
    AddUIExtendedOp,
    SubUIExtendedOp,
    MulSIExtendedOp,
    MulUIExtendedOp,
    SubIOp,
    MulIOp,
    ShLIOp,
    DivSIOp,
    DivUIOp,
    CeilDivSIOp,
    CeilDivUIOp,
    FloorDivSIOp,
    RemSIOp,
    RemUIOp,
    AndIOp,
    OrIOp,
    XOrIOp,
    ShRSIOp,
    ShRUIOp,
    MaxSIOp,
    MinSIOp,
    MaxUIOp,
    MinUIOp,
    CmpFOp,
    CmpIOp,
    SelectOp,
    ExtFOp,
    TruncFOp,
    ConvertFOp,
    ScalingExtFOp,
    ScalingTruncFOp,
    ExtSIOp,
    ExtUIOp,
    TruncIOp,
    SIToFPOp,
    UIToFPOp,
    FPToSIOp,
    FPToUIOp,
    IndexCastOp,
    IndexCastUIOp,
    BitcastOp,
);
