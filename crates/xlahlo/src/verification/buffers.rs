use super::{VerifyOptions, error, optional_attr, require};
use crate::{
    attributes::{Callees, FrontendAttributes, Index, OutputOperandAliases},
    ir,
    opcode::Opcode,
    ops::{HloOpInterface, ReturnOp},
    shape::Shape,
    types::{BufferType, TupleType},
};
use pliron::{
    builtin::{attributes::StringAttr, ops::ModuleOp},
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::{DefiningEntity, Value},
};
use std::collections::{HashMap, HashSet};

fn indices(ctx: &Context, shape: TypeHandle) -> Vec<Vec<i64>> {
    let mut pending = vec![(shape, Vec::new())];
    let mut result = Vec::new();
    while let Some((shape, path)) = pending.pop() {
        let ty = shape.deref(ctx);
        if ty.is::<BufferType>() {
            result.push(path);
        } else if let Some(tuple) = ty.downcast_ref::<TupleType>() {
            for (i, &child) in tuple.elements.iter().enumerate() {
                let mut child_path = path.clone();
                child_path.push(i as i64);
                pending.push((child, child_path));
            }
        }
    }
    result
}

fn no_buffers(ctx: &Context, shape: TypeHandle) -> Result<()> {
    require(
        indices(ctx, shape).is_empty(),
        "buffers are not allowed in this operation context",
    )
}

struct Writers<'a> {
    ctx: &'a Context,
    opcodes: HashMap<Ptr<Operation>, Opcode>,
    roots: HashSet<Ptr<Operation>>,
    while_roots: HashSet<Ptr<Operation>>,
}

impl Writers<'_> {
    fn unique(&self, value: Value, index: Vec<i64>) -> Result<()> {
        let ctx = self.ctx;
        let mut worklist = vec![(value, index)];
        let mut writers = 0;
        while let Some((value, index)) = worklist.pop() {
            let users: HashSet<_> = value
                .uses(ctx)
                .iter()
                .map(|usage| usage.user_op())
                .collect();
            for user in users {
                let Some(opcode) = self.opcodes.get(&user) else {
                    continue;
                };
                let data = user.deref(ctx);
                match opcode {
                    Opcode::Fusion
                    | Opcode::Call
                    | Opcode::CustomCall
                    | Opcode::While
                    | Opcode::Conditional
                    | Opcode::DynamicUpdateSlice => writers += 1,
                    Opcode::GetTupleElement => {
                        let selected = optional_attr::<Index>(ctx, user, "hlo_index")?
                            .ok_or_else(|| error("missing tuple index"))?;
                        if index
                            .first()
                            .is_some_and(|&i| i >= 0 && i as u64 == u64::from(selected.0))
                        {
                            worklist.push((data.get_result(0), index[1..].to_vec()));
                        }
                    }
                    Opcode::Tuple if self.while_roots.contains(&user) => writers += 1,
                    Opcode::Tuple if self.roots.contains(&user) => {}
                    Opcode::Tuple => {
                        let position = data
                            .operands()
                            .position(|operand| operand == value)
                            .ok_or_else(|| error("tuple user does not reference buffer"))?;
                        let mut next = vec![position as i64];
                        next.extend(&index);
                        worklist.push((data.get_result(0), next));
                    }
                    _ => {}
                }
                require(
                    writers <= 1,
                    "an SSA buffer value has more than one writer or unpin",
                )?;
            }
        }
        Ok(())
    }

    fn all_unique(&self, value: Value) -> Result<()> {
        for index in indices(self.ctx, value.get_type(self.ctx)) {
            self.unique(value, index)?;
        }
        Ok(())
    }

    fn custom_call(&self, op: Ptr<Operation>, options: VerifyOptions) -> Result<()> {
        let ctx = self.ctx;
        let data = op.deref(ctx);
        let shape = data.get_type(0);
        let operands: Vec<_> = data.operand_types(ctx).collect();
        let target = optional_attr::<StringAttr>(ctx, op, "hlo_target")?
            .ok_or_else(|| error("missing custom-call target"))?;
        let aliases = optional_attr::<OutputOperandAliases>(ctx, op, "hlo_aliases")?;
        let aliases = aliases
            .as_deref()
            .map_or(&[][..], |aliases| aliases.0.as_slice());
        match target.as_str() {
            "Pin" | "Unpin" => {
                require(operands.len() == 1, "Pin/Unpin requires one operand")?;
                let (array, buffer) = if target.as_str() == "Pin" {
                    (operands[0], shape)
                } else {
                    (shape, operands[0])
                };
                require(
                    array.deref(ctx).is::<crate::types::TensorType>()
                        && buffer.deref(ctx).is::<BufferType>(),
                    "Pin/Unpin requires an array and a buffer",
                )?;
                let inner = Shape::array_view(ctx, buffer)?;
                let equal = if options.layout_sensitive {
                    Shape::equal_including_layout(ctx, array, inner)
                } else {
                    Shape::equal_ignoring_layout(ctx, array, inner)
                };
                require(
                    equal && aliases.len() == 1,
                    "Pin/Unpin requires equal array shapes and exactly one alias",
                )?;
                if target.as_str() == "Pin" {
                    self.unique(data.get_result(0), vec![])?;
                }
                return Ok(());
            }
            "CreateBuffer" => {
                require(
                    operands.is_empty() && shape.deref(ctx).is::<BufferType>(),
                    "CreateBuffer requires no operands and a buffer result",
                )?;
                return self.unique(data.get_result(0), vec![]);
            }
            _ => {}
        }
        for (operand, &shape) in operands.iter().enumerate() {
            for index in indices(ctx, shape) {
                require(
                    aliases
                        .iter()
                        .filter(|alias| {
                            alias.operand_index == operand as i64
                                && alias.operand_shape_index.0 == index
                        })
                        .count()
                        == 1,
                    "each buffer operand must occur in exactly one output alias",
                )?;
            }
        }
        for index in indices(ctx, shape) {
            require(
                aliases
                    .iter()
                    .filter(|alias| alias.output_shape_index.0 == index)
                    .count()
                    == 1,
                "each buffer result must occur in exactly one operand alias",
            )?;
            self.unique(data.get_result(0), index)?;
        }
        Ok(())
    }
}

pub(super) fn verify(
    ctx: &Context,
    root: Ptr<Operation>,
    instructions: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
    computations: &HashMap<pliron::identifier::Identifier, Ptr<Operation>>,
    options: VerifyOptions,
) -> Result<()> {
    let mut shapes = HashSet::new();
    let mut has_buffers = false;
    for &(op, _) in instructions {
        for ty in op.deref(ctx).result_types() {
            if shapes.insert(ty) && !indices(ctx, ty).is_empty() {
                has_buffers = true;
            }
        }
    }
    if !has_buffers {
        return Ok(());
    }
    let entry = if Operation::is_op::<ModuleOp>(root, ctx) {
        computations[&ir::entry(ctx, root)?]
    } else {
        root
    };
    let mut writers = Writers {
        ctx,
        opcodes: HashMap::new(),
        roots: HashSet::new(),
        while_roots: HashSet::new(),
    };
    let mut roots = HashMap::new();
    let mut asynchronous = HashSet::new();
    for &(op, _) in instructions {
        let dynamic = Operation::get_op_dyn(op, ctx);
        if let Some(hlo) = op_cast::<dyn HloOpInterface>(dynamic.as_ref()) {
            writers.opcodes.insert(op, hlo.opcode());
        }
        if Operation::is_op::<ReturnOp>(op, ctx)
            && let DefiningEntity::Op(definition) = op.deref(ctx).get_operand(0).defining_entity()
        {
            writers.roots.insert(definition);
            if let Some(parent) = op.deref(ctx).get_parent_op(ctx) {
                roots.insert(parent, definition);
            }
        }
    }
    for (&op, &opcode) in &writers.opcodes {
        if matches!(
            opcode,
            Opcode::While | Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone
        ) && let Some(callees) = optional_attr::<Callees>(ctx, op, "hlo_callees")?
        {
            for callee in &callees.0 {
                let computation = *computations
                    .get(callee)
                    .ok_or_else(|| error("buffer callee not found"))?;
                if opcode == Opcode::While {
                    if let Some(&root) = roots.get(&computation) {
                        writers.while_roots.insert(root);
                    }
                } else {
                    asynchronous.insert(computation);
                }
            }
        }
    }
    for &(op, owner) in instructions {
        let Some(&opcode) = writers.opcodes.get(&op) else {
            continue;
        };
        if owner.is_some_and(|owner| asynchronous.contains(&owner)) {
            if owner.is_some_and(|owner| roots.get(&owner) == Some(&op))
                && opcode == Opcode::CustomCall
            {
                writers.custom_call(op, options)?;
            }
            continue;
        }
        let stream_call = opcode == Opcode::Call
            && optional_attr::<FrontendAttributes>(ctx, op, "hlo_frontend_attributes")?
                .is_some_and(|attrs| attrs.get("_xla_stream_annotation").is_some());
        if stream_call
            || matches!(
                opcode,
                Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone
            )
        {
            continue;
        }
        let data = op.deref(ctx);
        match opcode {
            Opcode::CustomCall => writers.custom_call(op, options)?,
            Opcode::While => writers.all_unique(data.get_result(0))?,
            Opcode::Parameter => {
                if owner == Some(entry) {
                    no_buffers(ctx, data.get_type(0))?;
                }
                writers.all_unique(data.get_result(0))?;
            }
            Opcode::Tuple | Opcode::GetTupleElement => {}
            Opcode::DynamicUpdateSlice
                if data
                    .get_operand(0)
                    .get_type(ctx)
                    .deref(ctx)
                    .is::<BufferType>() =>
            {
                writers.all_unique(data.get_result(0))?;
                for operand in data.operands().skip(1) {
                    no_buffers(ctx, operand.get_type(ctx))?;
                }
                require(
                    data.get_type(0).deref(ctx).is::<BufferType>(),
                    "buffer dynamic-update-slice must return a buffer",
                )?;
            }
            _ => {
                no_buffers(ctx, data.get_type(0))?;
                for operand in data.operands() {
                    no_buffers(ctx, operand.get_type(ctx))?;
                }
            }
        }
        for region in data.regions() {
            for block in region.deref(ctx).iter(ctx) {
                for argument in block.deref(ctx).arguments() {
                    writers.all_unique(argument)?;
                }
            }
        }
    }
    Ok(())
}
