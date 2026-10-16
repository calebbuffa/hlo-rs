//! Function/module attributes and module-context mesh verification.

use std::collections::HashMap;

use super::{BuildError, collectives};
use crate::{
    Attribute, Attributes, Computation, Instruction, Mesh, MeshReference, Operation,
    StableHloAttribute,
};

fn verify_attributes(attributes: &Attributes) -> Result<(), BuildError> {
    for (name, attribute) in attributes {
        attribute
            .verify()
            .map_err(|error| BuildError::InvalidAttribute {
                name: name.clone(),
                error,
            })?;
    }
    Ok(())
}

pub(crate) fn verify_module_attributes(attributes: &Attributes) -> Result<(), BuildError> {
    verify_attributes(attributes)?;
    if attributes.contains_key("sym_name") {
        return Err(BuildError::InvalidFunctionAttribute("sym_name"));
    }
    Ok(())
}

pub(crate) fn verify_function_attributes(
    computation: &Computation,
    attributes: &Attributes,
) -> Result<(), BuildError> {
    verify_attributes(attributes)?;
    for name in ["sym_name", "function_type"] {
        if attributes.contains_key(name) {
            return Err(BuildError::InvalidFunctionAttribute(name));
        }
    }
    if let Some(visibility) = attributes.get("sym_visibility")
        && !matches!(visibility, Attribute::String(value) if matches!(value.as_str(), "public" | "private" | "nested"))
    {
        return Err(BuildError::InvalidFunctionAttribute("sym_visibility"));
    }
    let signature = computation.function_type();
    if computation.is_declaration()
        && !matches!(attributes.get("sym_visibility"), Some(Attribute::String(value))
            if matches!(value.as_str(), "private" | "nested"))
    {
        return Err(BuildError::InvalidFunctionAttribute("sym_visibility"));
    }
    verify_signature_attributes(
        attributes,
        signature.inputs().len(),
        signature.outputs().len(),
    )
}

pub(crate) fn verify_signature_attributes(
    attributes: &Attributes,
    inputs: usize,
    outputs: usize,
) -> Result<(), BuildError> {
    for (name, expected) in [("arg_attrs", inputs), ("res_attrs", outputs)] {
        if let Some(attributes) = attributes.get(name)
            && !matches!(attributes, Attribute::Array(values) if values.len() == expected
                && values.iter().all(|value| matches!(value, Attribute::Dictionary(_))))
        {
            return Err(BuildError::InvalidFunctionAttribute(name));
        }
    }
    Ok(())
}

pub(crate) fn verify_resolved_mesh(
    instruction: &Instruction,
    enclosing: &Computation,
    mesh: &Mesh,
) -> Result<(), BuildError> {
    let mut resolved = instruction.clone();
    let Operation::StableHlo { opcode, attributes } = &mut resolved.operation else {
        unreachable!("mesh groups on a StableHLO operation");
    };
    let Some(Attribute::StableHlo(attribute)) = attributes.get_mut("replica_groups") else {
        unreachable!("verified mesh group attribute");
    };
    let StableHloAttribute::ReplicaGroupMeshAxes(groups) = attribute.as_mut() else {
        unreachable!("verified mesh group attribute");
    };
    groups.mesh = MeshReference::Inline(mesh.clone());
    attribute
        .verify()
        .map_err(|error| BuildError::InvalidAttribute {
            name: "replica_groups".into(),
            error: crate::AttributeValidationError::StableHlo(error),
        })?;
    let opcode = *opcode;
    let available = instruction
        .operands()
        .iter()
        .map(|&value| {
            (
                value,
                enclosing
                    .value_type(value)
                    .expect("verified SSA operand")
                    .clone(),
            )
        })
        .collect::<HashMap<_, _>>();
    collectives::verify(&resolved, opcode, &available)
}
