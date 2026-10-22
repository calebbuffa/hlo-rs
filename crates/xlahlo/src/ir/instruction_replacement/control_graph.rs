//! Control-dependency graph relay, splice and write-back.

use super::*;

pub(super) struct ControlNode {
    pub(super) value: Value,
    pub(super) name: Option<String>,
    pub(super) incoming: Vec<usize>,
}

pub(super) struct ControlGraph {
    pub(super) block: Ptr<BasicBlock>,
    pub(super) nodes: Vec<ControlNode>,
}

impl ControlGraph {
    pub(super) fn read(ctx: &Context, block: Ptr<BasicBlock>) -> Result<Self> {
        let parent = block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("control body is detached"))?;
        let mut raw = Vec::new();
        if Operation::is_op::<FusionOp>(parent, ctx) {
            for (index, parameter) in fusion_parameters(ctx, block)?.0.into_iter().enumerate() {
                raw.push((
                    block.deref(ctx).get_argument(index),
                    parameter.name,
                    parameter.control_predecessors,
                ));
            }
        } else {
            require(
                Operation::is_op::<ComputationOp>(parent, ctx),
                "control body must be HLO",
            )?;
        }
        for instruction in block.deref(ctx).iter(ctx) {
            if Operation::is_op::<ReturnOp>(instruction, ctx) {
                continue;
            }
            raw.push((
                result(ctx, instruction)?,
                optional_attr::<InstructionName>(ctx, instruction, "hlo_name")?
                    .map(|name| name.0.clone()),
                optional_attr::<ControlPredecessors>(ctx, instruction, "hlo_control_predecessors")?
                    .map(|controls| controls.0.clone())
                    .unwrap_or_default(),
            ));
        }
        let mut names = HashMap::new();
        for (index, (_, name, _)) in raw.iter().enumerate() {
            if let Some(name) = name {
                require(
                    !name.is_empty() && names.insert(name.clone(), index).is_none(),
                    "invalid or duplicate control endpoint name",
                )?;
            }
        }
        let nodes = raw
            .into_iter()
            .map(|(value, name, controls)| {
                let incoming = controls
                    .into_iter()
                    .map(|name| {
                        names
                            .get(&name)
                            .copied()
                            .ok_or_else(|| error("unknown control predecessor"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(ControlNode {
                    value,
                    name,
                    incoming,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { block, nodes })
    }

    pub(super) fn index(&self, value: Value) -> Result<usize> {
        self.nodes
            .iter()
            .position(|node| node.value == value)
            .ok_or_else(|| error("control endpoint is outside its body"))
    }

    pub(super) fn has_controls(&self, index: usize) -> bool {
        !self.nodes[index].incoming.is_empty()
            || self.nodes.iter().any(|node| node.incoming.contains(&index))
    }

    pub(super) fn relay(&mut self, from: usize, to: usize) {
        let incoming = self.nodes[from].incoming.clone();
        for predecessor in incoming {
            if !self.nodes[to].incoming.contains(&predecessor) {
                self.nodes[to].incoming.push(predecessor);
            }
        }
        for node in &mut self.nodes {
            if node.incoming.contains(&from) && !node.incoming.contains(&to) {
                node.incoming.push(to);
            }
        }
        self.drop_edges(from);
    }

    pub(super) fn splice(&mut self, index: usize) {
        let incoming = self.nodes[index].incoming.clone();
        for node in &mut self.nodes {
            if node.incoming.contains(&index) {
                for &predecessor in &incoming {
                    if !node.incoming.contains(&predecessor) {
                        node.incoming.push(predecessor);
                    }
                }
            }
        }
        self.drop_edges(index);
    }

    pub(super) fn drop_edges(&mut self, index: usize) {
        for node in &mut self.nodes {
            node.incoming.retain(|&predecessor| predecessor != index);
        }
        self.nodes[index].incoming.clear();
    }

    pub(super) fn ensure_names(&mut self, ctx: &Context, module: Ptr<Operation>) -> Result<()> {
        let mut occupied = HashSet::new();
        let mut pending = vec![module];
        while let Some(operation) = pending.pop() {
            if let Some(name) = optional_attr::<InstructionName>(ctx, operation, "hlo_name")? {
                occupied.insert(name.0.clone());
            }
            for region in operation.deref(ctx).regions() {
                for block in region.deref(ctx).iter(ctx) {
                    if Operation::is_op::<FusionOp>(operation, ctx) {
                        occupied.extend(
                            fusion_parameters(ctx, block)?
                                .0
                                .into_iter()
                                .filter_map(|parameter| parameter.name),
                        );
                    }
                    pending.extend(block.deref(ctx).iter(ctx));
                }
            }
        }
        for index in 0..self.nodes.len() {
            if self.nodes[index].name.is_some()
                || !self.nodes.iter().any(|node| node.incoming.contains(&index))
            {
                continue;
            }
            let mut suffix = 0;
            let name = loop {
                let name = format!("control_relay.{suffix}");
                if occupied.insert(name.clone()) {
                    break name;
                }
                suffix += 1;
            };
            self.nodes[index].name = Some(name);
        }
        Ok(())
    }

    pub(super) fn write(&self, ctx: &Context) -> Result<()> {
        let parent = self
            .block
            .deref(ctx)
            .get_parent_op(ctx)
            .ok_or_else(|| error("control body detached"))?;
        let mut parameters = if Operation::is_op::<FusionOp>(parent, ctx) {
            Some(fusion_parameters(ctx, self.block)?)
        } else {
            None
        };
        for node in &self.nodes {
            let controls = node
                .incoming
                .iter()
                .map(|&index| {
                    self.nodes[index]
                        .name
                        .clone()
                        .ok_or_else(|| error("control predecessor has no name"))
                })
                .collect::<Result<Vec<_>>>()?;
            match node.value.defining_entity() {
                DefiningEntity::Op(operation) => {
                    let mut ir = operation.deref_mut(ctx);
                    if let Some(name) = &node.name {
                        ir.attributes
                            .set(Identifier::new("hlo_name"), InstructionName(name.clone()));
                    } else {
                        ir.attributes.0.remove(&Identifier::new("hlo_name"));
                    }
                    let key = Identifier::new("hlo_control_predecessors");
                    if !controls.is_empty() || ir.attributes.0.contains_key(&key) {
                        ir.attributes.set(key, ControlPredecessors(controls));
                    }
                }
                DefiningEntity::Block(block) => {
                    let index = argument_index(ctx, block, node.value)?;
                    let parameter = parameters
                        .as_mut()
                        .ok_or_else(|| error("ordinary block argument is not an instruction"))?
                        .0
                        .get_mut(index)
                        .ok_or_else(|| error("missing fusion parameter"))?;
                    parameter.name = node.name.clone();
                    parameter.control_predecessors = controls;
                }
            }
        }
        if let Some(parameters) = parameters {
            set_fusion_parameters(ctx, self.block, parameters)?;
        }
        Ok(())
    }
}

pub(super) fn same_custom_target(
    ctx: &Context,
    old: Ptr<Operation>,
    new: Option<Ptr<Operation>>,
) -> Result<bool> {
    let Some(new) = new else {
        return Ok(false);
    };
    Ok(
        optional_attr::<pliron::builtin::attributes::StringAttr>(ctx, old, "hlo_target")?
            .as_deref()
            == optional_attr::<pliron::builtin::attributes::StringAttr>(ctx, new, "hlo_target")?
                .as_deref(),
    )
}
