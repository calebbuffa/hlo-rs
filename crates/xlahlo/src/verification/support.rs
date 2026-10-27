use super::elementwise;
use crate::{
    Diagnostic,
    attributes::ResultAccuracy,
    opcode::Opcode::{self, *},
    types::is_hlo_type,
};
use pliron::{
    attribute::Attribute,
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
};
use std::cell::Ref;

pub(crate) fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return pliron::verify_err_noloc!(Diagnostic::Verification {
            message: message.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn error(message: impl Into<String>) -> pliron::result::Error {
    pliron::verify_error_noloc!(Diagnostic::Verification {
        message: message.into(),
    })
}

pub(crate) fn instruction_error(
    ctx: &Context,
    op: Ptr<Operation>,
    mut error: pliron::result::Error,
) -> pliron::result::Error {
    if error.loc.is_unknown() {
        error.loc = op.deref(ctx).loc().clone();
    }
    let operation = Operation::get_op_dyn(op, ctx).get_opid().to_string();
    error.err = Box::new(Diagnostic::Operation {
        operation,
        source: error.err,
    });
    error
}

pub(crate) fn optional_attr<'a, T: Attribute>(
    ctx: &'a Context,
    op: Ptr<Operation>,
    name: &'static str,
) -> Result<Option<Ref<'a, T>>> {
    let data = op.deref(ctx);
    if data.attributes.0.is_empty() {
        return Ok(None);
    }
    let key = pliron::identifier::Identifier::new(name);
    if !data.attributes.0.contains_key(&key) {
        return Ok(None);
    }
    Ref::filter_map(data, |data| data.attributes.get::<T>(&key))
        .map(Some)
        .map_err(|_| {
            pliron::verify_error_noloc!(Diagnostic::InvalidAttribute {
                attribute: name,
                message: format!("expected attribute type {}", std::any::type_name::<T>()),
            })
        })
}

pub(crate) fn same(ctx: &Context, lhs: TypeHandle, rhs: TypeHandle) -> bool {
    lhs == rhs || crate::shape::compatible(ctx, lhs, rhs)
}

pub(crate) fn verify_result_accuracy(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
) -> Result<()> {
    if let Some(accuracy) =
        optional_attr::<ResultAccuracy>(ctx, op, crate::attributes::keys::RESULT_ACCURACY)?
    {
        accuracy.validate()?;
        // These are precisely the pinned InstructionVerifier unary visitor paths.
        let unary = elementwise::is_unary(opcode)
            || matches!(opcode, Convert | Copy | Domain | ReducePrecision);
        require(
            !unary || *accuracy == ResultAccuracy::Default || opcode.supports_result_accuracy(),
            "result accuracy is unsupported by this opcode",
        )?;
    }
    Ok(())
}

pub(crate) fn instruction_types(
    ctx: &Context,
    op: Ptr<Operation>,
    opcode: Opcode,
) -> Result<(Vec<TypeHandle>, TypeHandle)> {
    let data = op.deref(ctx);
    if data.get_num_results() != 1 || data.num_regions() != 0 || data.get_num_successors() != 0 {
        return Err(error(format!(
            "{} requires one result, no regions and no successors; got {} results, {} regions and {} successors",
            opcode.name(),
            data.get_num_results(),
            data.num_regions(),
            data.get_num_successors(),
        )));
    }
    if let Some(arity) = opcode.arity()
        && data.get_num_operands() != arity
    {
        return Err(error(format!(
            "{} operand arity mismatch: expected {arity}, got {}",
            opcode.name(),
            data.get_num_operands(),
        )));
    }
    let result = data.get_type(0);
    require(is_hlo_type(ctx, result), "result must be an HLO type")?;
    let operands: Vec<_> = data.operand_types(ctx).collect();
    require(
        operands.iter().all(|&ty| is_hlo_type(ctx, ty)),
        "operands must be HLO types",
    )?;
    verify_result_accuracy(ctx, op, opcode)?;
    Ok((operands, result))
}
