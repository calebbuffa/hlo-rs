use crate::{Error, Module, Result, Shape, VerifyOptions};
use std::collections::BTreeSet;

fn index(index: &[i64]) -> Result<Vec<usize>> {
    index
        .iter()
        .map(|&i| {
            usize::try_from(i)
                .map_err(|_| Error::new("shape-index", "negative metadata shape index"))
        })
        .collect()
}

pub(super) fn verify_aliases_and_donors(module: &Module, options: VerifyOptions) -> Result<()> {
    let actual = module.program_shape(module.entry()?)?;
    let program = module
        .as_proto()
        .host_program_shape
        .as_ref()
        .unwrap_or(&actual);
    let parameter = |number: i64, path: &[i64]| -> Result<Shape> {
        let shape = usize::try_from(number)
            .ok()
            .and_then(|i| program.parameters.get(i))
            .ok_or_else(|| Error::new("alias", "entry parameter number out of range"))?;
        Shape::from_proto(shape.clone())?.subshape(&index(path)?)
    };
    let output = Shape::from_proto(
        program
            .result
            .as_ref()
            .ok_or_else(|| Error::new("signature", "missing entry result"))?
            .clone(),
    )?;
    let mut outputs = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    if let Some(config) = module.as_proto().input_output_alias.as_ref() {
        for alias in &config.entries {
            let input = parameter(alias.parameter_number, &alias.parameter_shape_index)?;
            let output = output.subshape(&index(&alias.output_shape_index)?)?;
            if !input.is_array() || !output.is_array() {
                return Err(Error::new(
                    "alias",
                    "module aliases must reference array subshapes",
                ));
            }
            if !outputs.insert(alias.output_shape_index.clone())
                || !inputs.insert((alias.parameter_number, alias.parameter_shape_index.clone()))
            {
                return Err(Error::new(
                    "alias",
                    "module alias input or output is used more than once",
                ));
            }
            if options.layout_sensitive {
                let size = |shape: &Shape| options.size_of(shape);
                if size(&input)? != size(&output)? {
                    return Err(Error::new(
                        "alias",
                        "aliased input and output have different target sizes",
                    ));
                }
            }
        }
    }
    if let Some(config) = module.as_proto().buffer_donor.as_ref() {
        for donor in &config.entries {
            if !parameter(donor.parameter_number, &donor.parameter_shape_index)?.is_array() {
                return Err(Error::new("donor", "buffer donors must reference arrays"));
            }
            if inputs.contains(&(donor.parameter_number, donor.parameter_shape_index.clone())) {
                return Err(Error::new(
                    "donor",
                    "an aliased parameter cannot also be a buffer donor",
                ));
            }
        }
    }
    Ok(())
}
pub(super) fn verify_instruction(
    instruction: &crate::proto::hlo::HloInstructionProto,
    shape: &Shape,
    options: VerifyOptions,
) -> Result<()> {
    if let Some(value) = instruction.original_value.as_ref() {
        crate::OriginalValue::from_proto(value.clone())?.verify_shape(shape)?;
    }
    if options.verify_instruction_name_unchanged {
        let scheduling_name = &instruction.metadata.get_or_default().scheduling_name;
        if !scheduling_name.is_empty()
            && scheduling_name != &instruction.name
            && !instruction.name.contains(".clone")
            && !instruction.name.contains(".remat")
        {
            return Err(Error::new(
                "instruction-name",
                "instruction name differs from its scheduling name",
            ));
        }
    }
    fn host(shape: &crate::proto::xla_data::ShapeProto) -> bool {
        if shape.element_type.enum_value() == Ok(crate::PrimitiveType::BUFFER) {
            return host(&shape.array_storage());
        }
        shape.layout.as_ref().is_some_and(|l| l.memory_space == 5)
            || shape.logical_children().iter().any(host)
    }
    if options.verify_no_host_memory_space && host(shape.as_proto()) {
        return Err(Error::new(
            "memory-space",
            "instruction shape contains host memory space",
        ));
    }
    Ok(())
}
