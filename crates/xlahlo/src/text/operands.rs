use super::*;

pub(super) fn resolve(
    computation: &mut HloComputationProto,
    names: &mut BTreeMap<String, i64>,
    name: &str,
    indices: &[i64],
    consumer: i64,
    next: &mut i64,
) -> Result<i64> {
    let mut current = *names
        .get(name)
        .ok_or_else(|| Error::new("reference", format!("undefined operand {name}")))?;
    if indices.is_empty() {
        return Ok(current);
    }
    let position = |id| {
        computation
            .instructions
            .iter()
            .position(|i| i.id == id)
            .ok_or_else(|| Error::new("reference", "missing projection instruction"))
    };
    if position(current)? >= position(consumer)? {
        return Err(Error::new(
            "reference",
            "compact projection requires a previously defined operand",
        ));
    }
    let mut prefix = name.to_owned();
    for &index in indices {
        if index < 0 {
            return Err(Error::new("index", "negative tuple projection index"));
        }
        write!(prefix, "#{index}").map_err(|e| Error::new("print", e.to_string()))?;
        if let Some(&cached) = names.get(&prefix) {
            current = cached;
            continue;
        }
        let id = *next;
        *next = next
            .checked_add(1)
            .ok_or_else(|| Error::new("resource-limit", "instruction count overflow"))?;
        let position = computation
            .instructions
            .iter()
            .position(|i| i.id == consumer)
            .ok_or_else(|| Error::new("reference", "missing projection consumer"))?;
        computation.instructions.insert(
            position,
            HloInstructionProto {
                id,
                name: prefix.replace('#', "_"),
                opcode: Opcode::GetTupleElement.name().into(),
                operand_ids: vec![current],
                tuple_index: index,
                ..Default::default()
            },
        );
        names.insert(prefix.clone(), id);
        current = id;
    }
    Ok(current)
}
