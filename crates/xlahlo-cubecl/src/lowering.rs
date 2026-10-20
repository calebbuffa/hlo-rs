use std::collections::HashMap;
use std::sync::Arc;

mod compositional;
mod direct;
mod gather;
pub(crate) mod views;

use cubecl_core::compute::KernelBuilder;
use cubecl_ir::{
    AddressType, Builtin, ElemType, FloatKind, OpInserter, Scope, UIntKind,
    apfloat::APFloat,
    attributes::FloatAttr,
    dialect::{
        bitwise::{BitwiseAndOp, BitwiseOrOp, BitwiseXorOp},
        branch::{IfOp, YieldOp},
        cmp::{IEqualOp, SGreaterThanOp, SLessThanOp, UGreaterThanOp, ULessThanOp},
        general::{
            BoolAndOp, BoolNotOp, BoolOrOp, BuiltinAttr, CastOp, ReadBuiltinOp, ReinterpretCastOp,
            SelectOp as CubeSelectOp,
        },
        math::{FAddOp, FDivOp, FMulOp, FSubOp, IAddOp, IMulOp, ISubOp, SDivOp, UDivOp, URemOp},
        memory::{IndexOp, LoadOp, StoreOp},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_runtime::{
    id::KernelId,
    kernel::{KernelDefinition, KernelMetadata},
};
use pliron::{
    builtin::{
        attributes::IntegerAttr,
        ops::ConstantOp as CubeConstantOp,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::{
            DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
        },
        inserter::{Inserter, OpInsertionPoint},
        rewriter::Rewriter,
    },
    irfmt::parsers::spaced,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    parsable::parse_from_str,
    printable::Printable,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};
use xlahlo::{
    attributes::FusionKind,
    opcode::Opcode,
    ops::{
        ComputationOp, HloOpInterface, ReturnOp,
        control::{ConstantOp, FusionOp, ParameterOp, TupleOp},
        movement::{BroadcastOp, GatherOp},
    },
    shape::{Dimension, Layout, Shape},
    types::ElementType,
};

use crate::module_compiler::{OperationLoweringRequest, OperationStrategyRegistry};
use crate::plan::{
    BufferAccess, BufferDescriptor, BufferId, BufferRole, ExecutionPlanBuilder, KernelBinding,
    LaunchGeometry, PlanError, PlannedKernel, PlannedOutput, StaticValue,
};

const CUBE_DIM: u32 = 256;
const CODEGEN_VERSION: u32 = 1;

pub(crate) fn lower_module(
    context: &Context,
    module: Ptr<Operation>,
    plan: &mut ExecutionPlanBuilder,
    operation_strategies: &OperationStrategyRegistry,
) -> Result<(), PlanError> {
    compositional::lower_module(context, module, plan, operation_strategies)
}

fn find_computation(
    context: &Context,
    module: Ptr<Operation>,
    entry: &pliron::identifier::Identifier,
) -> Result<ComputationOp, PlanError> {
    for region in module.deref(context).regions() {
        for block in region.deref(context).iter(context) {
            for operation in block.deref(context).iter(context) {
                if let Some(computation) = Operation::get_op::<ComputationOp>(operation, context)
                    && computation
                        .get_attr_hlo_computation_name(context)
                        .is_some_and(|name| *name.as_ref() == *entry)
                {
                    return Ok(computation);
                }
            }
        }
    }
    Err(PlanError::LoweringFailure(
        "entry computation is missing".into(),
    ))
}

fn descriptor(context: &Context, ty: TypeHandle) -> Result<BufferDescriptor, PlanError> {
    tensor_descriptor(context, ty, false)
}

fn logical_descriptor(context: &Context, ty: TypeHandle) -> Result<BufferDescriptor, PlanError> {
    tensor_descriptor(context, ty, true)
}

fn tensor_descriptor(
    context: &Context,
    ty: TypeHandle,
    logical: bool,
) -> Result<BufferDescriptor, PlanError> {
    if !Shape::is_array(context, ty) || Shape::is_dynamic(context, ty) {
        return Err(PlanError::UnsupportedFeature(
            "non-static tensor shape".into(),
        ));
    }
    let element_type = Shape::element_type(context, ty)
        .map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
    let rank =
        Shape::rank(context, ty).map_err(|error| PlanError::LoweringFailure(error.to_string()))?;
    if Shape::layout(context, ty)
        .map_err(|error| PlanError::LoweringFailure(error.to_string()))?
        .is_some_and(|layout| {
            let mut expected = Layout::descending(rank);
            if logical {
                expected.minor_to_major = layout.minor_to_major.clone();
            }
            *layout != expected
        })
    {
        return Err(PlanError::UnsupportedFeature(
            "non-dense row-major tensor layout".into(),
        ));
    }
    let shape = (0..rank)
        .map(|axis| match Shape::dimension(context, ty, axis) {
            Ok(Dimension::Static(extent)) => Ok(extent),
            Ok(_) => Err(PlanError::UnsupportedFeature(
                "dynamic tensor dimension".into(),
            )),
            Err(error) => Err(PlanError::LoweringFailure(error.to_string())),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = BufferDescriptor::dense_row_major(element_type, shape);
    descriptor.cubecl_element_type()?;
    Ok(descriptor)
}
