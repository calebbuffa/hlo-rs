//! Verified generic-form MLIR export for every tensor operation.

use crate::ops::*;
use pliron::{
    attribute::Attribute,
    builtin::attributes::{OperandSegmentSizesAttr, UnitAttr},
    context::Context,
    derive::op_interface_impl,
    op::Op,
    printable::State,
    result::Result,
};
use pliron_arith::mlir::print_generic;
use pliron_linalg::attributes::I64ArrayAttr;
use pliron_mlir_interop::{ToMlirOp, printers::mlir_dense_array};
use stablehlo::attributes::IndexAttr;
use std::fmt;

fn untranslatable(message: &str) -> pliron::result::Error {
    pliron::input_error_noloc!(pliron_mlir_interop::Error::Untranslatable(
        message.to_owned()
    ))
}

fn property(
    _ctx: &Context,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    let name = match key {
        "tensor_operand_segment_sizes" => "operandSegmentSizes",
        "tensor_concat_dim" => "dim",
        "tensor_static_offsets" => "static_offsets",
        "tensor_static_sizes" => "static_sizes",
        "tensor_static_strides" => "static_strides",
        "tensor_reassociation" => "reassociation",
        "tensor_static_output_shape" => "static_output_shape",
        "tensor_static_low" => "static_low",
        "tensor_static_high" => "static_high",
        "tensor_nofold" => "nofold",
        "tensor_gather_dims" => "gather_dims",
        "tensor_scatter_dims" => "scatter_dims",
        "tensor_unique" => "unique",
        _ => return Ok(None),
    };
    let text = match name {
        "operandSegmentSizes" => {
            let a = attr
                .downcast_ref::<OperandSegmentSizesAttr>()
                .ok_or_else(|| {
                    untranslatable("tensor operand segments require i32 array storage")
                })?;
            mlir_dense_array("i32", a.0.iter().map(ToString::to_string))
        }
        "dim" => {
            let a = attr
                .downcast_ref::<IndexAttr>()
                .ok_or_else(|| untranslatable("tensor concat dim requires i64 storage"))?;
            format!("{} : i64", a.value)
        }
        "nofold" | "unique" => {
            if attr.downcast_ref::<UnitAttr>().is_none() {
                return Err(untranslatable("tensor flag requires unit storage"));
            }
            "unit".to_owned()
        }
        "reassociation" => {
            let groups = attr
                .downcast_ref::<pliron::builtin::attributes::VecAttr>()
                .ok_or_else(|| untranslatable("tensor reassociation requires array storage"))?;
            let groups: Result<Vec<_>> = groups
                .0
                .iter()
                .map(|g| {
                    let values = &g
                        .downcast_ref::<I64ArrayAttr>()
                        .ok_or_else(|| {
                            untranslatable(
                                "tensor reassociation group requires integer-array storage",
                            )
                        })?
                        .values;
                    Ok(format!(
                        "[{}]",
                        values
                            .iter()
                            .map(|n| format!("{n} : i64"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })
                .collect();
            format!("[{}]", groups?.join(", "))
        }
        _ => {
            let a = attr
                .downcast_ref::<I64ArrayAttr>()
                .ok_or_else(|| untranslatable("tensor property requires integer-array storage"))?;
            mlir_dense_array("i64", a.values.iter().map(ToString::to_string))
        }
    };
    Ok(Some((name, Some(text))))
}

macro_rules! export {
    ($($ty:ty),+ $(,)?) => {$(
        #[op_interface_impl]
        impl ToMlirOp for $ty {
            fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                let name = self.get_opid().to_string();
                print_generic(ctx, self.get_operation(), state, f, &|ctx, key, attr| {
                    let owner = match key {
                        "tensor_operand_segment_sizes" => matches!(name.as_str(),
                            "tensor.extract_slice" | "tensor.insert_slice"
                            | "tensor.parallel_insert_slice" | "tensor.pad"),
                        "tensor_concat_dim" => name == "tensor.concat",
                        "tensor_static_offsets" | "tensor_static_sizes" | "tensor_static_strides" =>
                            matches!(name.as_str(), "tensor.extract_slice" | "tensor.insert_slice"
                                | "tensor.parallel_insert_slice"),
                        "tensor_reassociation" => matches!(name.as_str(),
                            "tensor.expand_shape" | "tensor.collapse_shape"),
                        "tensor_static_output_shape" => name == "tensor.expand_shape",
                        "tensor_static_low" | "tensor_static_high" | "tensor_nofold" => name == "tensor.pad",
                        "tensor_gather_dims" => name == "tensor.gather",
                        "tensor_scatter_dims" => name == "tensor.scatter",
                        "tensor_unique" => matches!(name.as_str(), "tensor.gather" | "tensor.scatter"),
                        _ => false,
                    };
                    if !owner {
                        return Err(untranslatable(&format!("attribute {key} is not a property of {name}")));
                    }
                    property(ctx, key, attr)
                })
            }
        }
    )+};
}
export!(
    BitcastOp,
    CastOp,
    ConcatOp,
    DimOp,
    EmptyOp,
    ExtractOp,
    ExtractSliceOp,
    FromElementsOp,
    GatherOp,
    GenerateOp,
    InsertOp,
    InsertSliceOp,
    RankOp,
    ReshapeOp,
    ExpandShapeOp,
    CollapseShapeOp,
    PadOp,
    ParallelInsertSliceOp,
    ScatterOp,
    SplatOp,
    YieldOp,
);
