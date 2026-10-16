use std::collections::{HashMap, HashSet};

use super::{ENTRY_ATTRIBUTE, error, ir_error};
use crate::text::{TextError, TextErrorKind, attributes, types};
use crate::{Attribute, Attributes, Computation, Instruction, Module, Operation, Type, ValueId};

pub(in crate::text) fn print(module: &Module) -> Result<String, TextError> {
    module.verify().map_err(ir_error)?;
    let mut attributes = module.attributes().clone();
    if attributes.contains_key(ENTRY_ATTRIBUTE) {
        return Err(error(TextErrorKind::DuplicateField(ENTRY_ATTRIBUTE.into())));
    }
    if !module.name().is_empty() {
        attributes.insert("sym_name".into(), Attribute::String(module.name().into()));
    }
    if let Some(entry) = module.entry() {
        attributes.insert(
            ENTRY_ATTRIBUTE.into(),
            Attribute::SymbolRef(vec![module.computations()[entry.0 as usize].name().into()]),
        );
    }
    let mut output = String::from("\"builtin.module\"() ({\n");
    if module.computations().is_empty() {
        output.push_str("  ^bb0:\n");
    }
    for function in module.computations() {
        output.push_str(&print_function(function)?);
    }
    output.push_str(&format!(
        "}}) {} : () -> (){}\n",
        print_attributes(attributes, 0)?,
        print_location(module.location(), 0)?,
    ));
    Ok(output)
}

fn print_attributes(values: Attributes, depth: usize) -> Result<String, TextError> {
    attributes::print(&Attribute::Dictionary(values), depth)
}

fn print_function(function: &Computation) -> Result<String, TextError> {
    if function
        .instructions()
        .iter()
        .any(|instruction| !instruction.called_computations().is_empty())
    {
        return Err(error(TextErrorKind::Unsupported(
            "legacy numeric called-computation references",
        )));
    }
    let mut metadata = function.attributes().clone();
    metadata.insert("sym_name".into(), Attribute::String(function.name().into()));
    metadata.insert(
        "function_type".into(),
        Attribute::Type(Box::new(Type::Function(function.function_type()))),
    );
    let mut output = String::from("  \"func.func\"() ({");
    if !function.is_declaration() {
        let mut names = Names::default();
        let mut parameters = function
            .instructions()
            .iter()
            .filter_map(|instruction| {
                if let Operation::Parameter { number, name } = instruction.operation() {
                    Some((
                        *number,
                        name,
                        instruction.results()[0].clone(),
                        instruction.location(),
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        parameters.sort_unstable_by_key(|(number, _, _, _)| *number);
        let mut arguments = Vec::new();
        for (_, name, result, location) in parameters {
            if !valid_ssa_name(name) || !names.used.insert(name.clone()) {
                return Err(error(TextErrorKind::Unsupported(
                    "parameter names that cannot be represented as unique MLIR SSA names",
                )));
            }
            names
                .values
                .insert(result.value(), (name.clone(), result.ty().clone()));
            arguments.push(format!(
                "%{name}: {}{}",
                types::print(result.ty(), 2)?,
                print_location(location, 2)?
            ));
        }
        output.push_str(&format!("\n    ^bb0({}):\n", arguments.join(", ")));
        for instruction in function.instructions() {
            if !instruction.operation().is_parameter() {
                print_instruction(instruction, &mut names, &mut output, 3)?;
            }
        }
        let operands = function
            .roots()
            .iter()
            .map(|value| names.value(*value).map(|(name, _)| format!("%{name}")))
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = function
            .roots()
            .iter()
            .map(|value| names.value(*value).map(|(_, ty)| ty.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        output.push_str(&format!(
            "      \"func.return\"({}) : {} -> (){}\n  ",
            operands.join(", "),
            print_types(&inputs, 2)?,
            print_location(function.return_location(), 2)?,
        ));
    }
    output.push_str(&format!(
        "}}) {} : () -> (){}\n",
        print_attributes(metadata, 1)?,
        print_location(function.location(), 1)?,
    ));
    Ok(output)
}

#[derive(Clone, Default)]
struct Names {
    values: HashMap<ValueId, (String, Type)>,
    used: HashSet<String>,
}

impl Names {
    fn value(&self, value: ValueId) -> Result<&(String, Type), TextError> {
        self.values.get(&value).ok_or_else(|| {
            error(TextErrorKind::InvalidIr(Box::new(
                crate::BuildError::UnknownValue(value),
            )))
        })
    }

    fn define(&mut self, value: ValueId, ty: &Type) -> String {
        let mut name = format!("v{}", value.0);
        while !self.used.insert(name.clone()) {
            name.push('_');
        }
        self.values.insert(value, (name.clone(), ty.clone()));
        name
    }
}

fn print_instruction(
    instruction: &Instruction,
    names: &mut Names,
    output: &mut String,
    depth: usize,
) -> Result<(), TextError> {
    if depth >= 128 {
        return Err(error(TextErrorKind::NestingLimit));
    }
    if !instruction.called_computations().is_empty() {
        return Err(error(TextErrorKind::Unsupported(
            "legacy numeric called-computation references",
        )));
    }
    let operation_name = match instruction.operation() {
        Operation::StableHlo { opcode, .. } => format!("stablehlo.{}", opcode.mnemonic()),
        Operation::Call { .. } => "func.call".into(),
        Operation::Parameter { .. } => {
            return Err(error(TextErrorKind::Unsupported(
                "nested parameter operation",
            )));
        }
    };
    let operands = instruction
        .operands()
        .iter()
        .map(|value| names.value(*value).map(|(name, _)| format!("%{name}")))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = instruction
        .operands()
        .iter()
        .map(|value| names.value(*value).map(|(_, ty)| ty.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut result_names = Vec::new();
    // Results are not in scope inside the operation's own regions.
    let mut result_scope = names.clone();
    for result in instruction.results() {
        result_names.push(format!(
            "%{}",
            result_scope.define(result.value(), result.ty())
        ));
    }
    let indentation = "  ".repeat(depth);
    output.push_str(&indentation);
    if !result_names.is_empty() {
        output.push_str(&format!("{} = ", result_names.join(", ")));
    }
    output.push_str(&format!("\"{operation_name}\"({})", operands.join(", ")));
    if !instruction.regions().is_empty() {
        output.push_str(" (");
        for (index, region) in instruction.regions().iter().enumerate() {
            if index > 0 {
                output.push_str(", ");
            }
            output.push_str("{\n");
            for (block_index, block) in region.blocks().iter().enumerate() {
                let mut scope = names.clone();
                let args = block
                    .arguments()
                    .iter()
                    .map(|argument| {
                        let name = scope.define(argument.value(), argument.ty());
                        Ok(format!(
                            "%{name}: {}{}",
                            types::print(argument.ty(), depth + 1)?,
                            print_location(argument.location(), depth + 1)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, TextError>>()?;
                output.push_str(&format!(
                    "{indentation}  ^bb{block_index}({}):\n",
                    args.join(", ")
                ));
                for nested in block.instructions() {
                    print_instruction(nested, &mut scope, output, depth + 2)?;
                }
            }
            output.push_str(&format!("{indentation}}}"));
        }
        output.push(')');
    }
    if let Some(metadata) = instruction.operation().attributes()
        && !metadata.is_empty()
    {
        output.push_str(&format!(" {}", print_attributes(metadata.clone(), depth)?));
    }
    let results = instruction
        .results()
        .iter()
        .map(|result| result.ty().clone())
        .collect::<Vec<_>>();
    output.push_str(&format!(
        " : {} -> {}{}\n",
        print_types(&inputs, depth)?,
        print_types(&results, depth)?,
        print_location(instruction.location(), depth)?,
    ));
    *names = result_scope;
    Ok(())
}

fn print_location(location: Option<&crate::Location>, depth: usize) -> Result<String, TextError> {
    location
        .map(|location| {
            crate::text::locations::print(location, depth).map(|text| format!(" {text}"))
        })
        .transpose()
        .map(Option::unwrap_or_default)
}

fn print_types(values: &[Type], depth: usize) -> Result<String, TextError> {
    Ok(format!(
        "({})",
        values
            .iter()
            .map(|ty| types::print(ty, depth + 1))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ")
    ))
}

pub(super) fn valid_ssa_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    if bytes[0].is_ascii_digit() {
        return bytes.iter().all(u8::is_ascii_digit);
    }
    (bytes[0].is_ascii_alphabetic() || matches!(bytes[0], b'_' | b'$' | b'.' | b'-'))
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$' | b'.' | b'-'))
}
