use super::{LayoutComparison, same_shapes};
use crate::proto::xla_data::ShapeProto;
use crate::{Error, InstructionId, Module, PrimitiveType, Result, Shape, VerifyOptions};
use std::collections::BTreeSet;

fn indices(shape: &ShapeProto) -> Vec<Vec<i64>> {
    let mut pending = vec![(shape, Vec::new())];
    let mut result = Vec::new();
    while let Some((shape, path)) = pending.pop() {
        match shape.element_type.enum_value_or_default() {
            PrimitiveType::BUFFER => result.push(path.clone()),
            PrimitiveType::TUPLE => {
                for (index, child) in shape.tuple_shapes.iter().enumerate() {
                    let mut child_path = path.clone();
                    child_path.push(index as i64);
                    pending.push((child, child_path));
                }
            }
            _ => {}
        }
    }
    result
}

fn no_buffers(shape: &Shape) -> Result<()> {
    if !indices(shape.as_proto()).is_empty() {
        return Err(Error::new(
            "buffer",
            "buffers are not allowed in this operation context",
        ));
    }
    Ok(())
}

struct Writers<'a> {
    module: &'a Module,
    roots: BTreeSet<InstructionId>,
    while_roots: BTreeSet<InstructionId>,
}

impl Writers<'_> {
    fn unique(&self, id: InstructionId, index: Vec<i64>) -> Result<()> {
        let mut worklist = vec![(id, index)];
        let mut writers = 0;
        while let Some((id, index)) = worklist.pop() {
            for user in self.module.users(id)? {
                let instruction = self.module.instruction(user)?;
                match instruction.opcode.as_str() {
                    "fusion"
                    | "call"
                    | "custom-call"
                    | "while"
                    | "conditional"
                    | "dynamic-update-slice" => writers += 1,
                    "get-tuple-element" if index.first() == Some(&instruction.tuple_index) => {
                        worklist.push((user, index[1..].to_vec()));
                    }
                    "tuple" if self.while_roots.contains(&user) => writers += 1,
                    "tuple" if self.roots.contains(&user) => {}
                    "tuple" => {
                        let operands = self.module.operands(user)?;
                        let position = operands
                            .iter()
                            .position(|&operand| operand == id)
                            .ok_or_else(|| {
                                Error::new(
                                    "buffer",
                                    "tuple user does not reference buffer producer",
                                )
                            })?;
                        let mut next = vec![position as i64];
                        next.extend(&index);
                        worklist.push((user, next));
                    }
                    _ => {}
                }
                if writers > 1 {
                    return Err(Error::new(
                        "buffer",
                        "an SSA buffer value has more than one writer or unpin",
                    )
                    .context(&self.module.instruction(id)?.name));
                }
            }
        }
        Ok(())
    }

    fn all_unique(&self, id: InstructionId) -> Result<()> {
        for index in indices(self.module.shape(id)?.as_proto()) {
            self.unique(id, index)?;
        }
        Ok(())
    }

    fn custom_call(&self, id: InstructionId, options: VerifyOptions) -> Result<()> {
        let instruction = self.module.instruction(id)?;
        let shape = self.module.shape(id)?;
        let operands = self
            .module
            .operands(id)?
            .iter()
            .map(|&id| self.module.shape(id))
            .collect::<Result<Vec<_>>>()?;
        match instruction.custom_call_target.as_str() {
            "Pin" | "Unpin" => {
                let [operand] = operands.as_slice() else {
                    return Err(Error::new("buffer", "Pin/Unpin requires one operand"));
                };
                let (array, buffer) = if instruction.custom_call_target == "Pin" {
                    (operand, &shape)
                } else {
                    (&shape, operand)
                };
                if !array.is_array() || buffer.element_type() != PrimitiveType::BUFFER {
                    return Err(Error::new(
                        "buffer",
                        "Pin/Unpin requires an array and a buffer",
                    ));
                }
                let inner = buffer.array_view()?;
                let equal = if options.layout_sensitive {
                    same_shapes(array, &inner, LayoutComparison::Exact)
                } else {
                    array.equal_ignoring_layout(&inner)
                };
                if !equal || instruction.output_operand_aliasing.len() != 1 {
                    return Err(Error::new(
                        "buffer",
                        "Pin/Unpin requires equal array shapes and exactly one alias",
                    ));
                }
                if instruction.custom_call_target == "Pin" {
                    self.unique(id, vec![])?;
                }
                return Ok(());
            }
            "CreateBuffer" => {
                if !operands.is_empty() || shape.element_type() != PrimitiveType::BUFFER {
                    return Err(Error::new(
                        "buffer",
                        "CreateBuffer requires no operands and a buffer result",
                    ));
                }
                return self.unique(id, vec![]);
            }
            _ => {}
        }
        for (operand, shape) in operands.iter().enumerate() {
            for index in indices(shape.as_proto()) {
                let count = instruction
                    .output_operand_aliasing
                    .iter()
                    .filter(|alias| {
                        alias.operand_index == operand as i64 && alias.operand_shape_index == index
                    })
                    .count();
                if count != 1 {
                    return Err(Error::new(
                        "buffer",
                        "each buffer operand must occur in exactly one output alias",
                    ));
                }
            }
        }
        for index in indices(shape.as_proto()) {
            if instruction
                .output_operand_aliasing
                .iter()
                .filter(|alias| alias.output_shape_index == index)
                .count()
                != 1
            {
                return Err(Error::new(
                    "buffer",
                    "each buffer result must occur in exactly one operand alias",
                ));
            }
            self.unique(id, index)?;
        }
        Ok(())
    }
}

pub(super) fn verify(module: &Module, options: VerifyOptions) -> Result<()> {
    let mut writers = Writers {
        module,
        roots: BTreeSet::new(),
        while_roots: BTreeSet::new(),
    };
    let mut asynchronous = BTreeSet::new();
    for computation in module.computations() {
        writers.roots.insert(module.root(computation)?);
        for id in module.instructions(computation)? {
            let instruction = module.instruction(id)?;
            for &callee in &instruction.called_computation_ids {
                let callee = module.computation_handle(callee);
                if instruction.opcode == "while" {
                    writers.while_roots.insert(module.root(callee)?);
                }
                if matches!(
                    instruction.opcode.as_str(),
                    "async-start" | "async-update" | "async-done"
                ) {
                    asynchronous.insert(callee);
                }
            }
        }
    }
    for computation in module.computations() {
        if asynchronous.contains(&computation) {
            let root = module.root(computation)?;
            if module.instruction(root)?.opcode == "custom-call" {
                writers.custom_call(root, options)?;
            }
            continue;
        }
        for id in module.instructions(computation)? {
            let instruction = module.instruction(id)?;
            let operands = module.operands(id)?;
            let stream_call = instruction.opcode == "call"
                && instruction
                    .frontend_attributes
                    .as_ref()
                    .is_some_and(|attrs| attrs.map.contains_key("_xla_stream_annotation"));
            if matches!(
                instruction.opcode.as_str(),
                "async-start" | "async-update" | "async-done"
            ) || stream_call
            {
                continue;
            }
            match instruction.opcode.as_str() {
                "custom-call" => writers.custom_call(id, options)?,
                "while" => writers.all_unique(id)?,
                "parameter" => {
                    if computation == module.entry()? {
                        no_buffers(&module.shape(id)?)?;
                    }
                    writers.all_unique(id)?;
                }
                "tuple" | "get-tuple-element" => {}
                "dynamic-update-slice"
                    if module.shape(operands[0])?.element_type() == PrimitiveType::BUFFER =>
                {
                    writers.all_unique(id)?;
                    for &operand in &operands[1..] {
                        no_buffers(&module.shape(operand)?)?;
                    }
                    if module.shape(id)?.element_type() != PrimitiveType::BUFFER {
                        return Err(Error::new(
                            "buffer",
                            "buffer dynamic-update-slice must return a buffer",
                        ));
                    }
                }
                _ => {
                    no_buffers(&module.shape(id)?)?;
                    for operand in operands {
                        no_buffers(&module.shape(operand)?)?;
                    }
                }
            }
        }
    }
    Ok(())
}
