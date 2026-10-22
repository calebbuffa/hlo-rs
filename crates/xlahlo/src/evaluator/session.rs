//! Request-local frames, dependency evaluation, and computation lookup.

use super::*;

impl<'a> Session<'a> {
    pub(super) fn new(ctx: &'a Context, module: Ptr<Operation>) -> Result<Self> {
        let body = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .ok_or_else(|| malformed("module has no computation block"))?;
        let mut callees = HashMap::new();
        for operation in body.deref(ctx).iter(ctx) {
            if let Some(computation) = Operation::get_op::<ComputationOp>(operation, ctx) {
                let name = computation
                    .get_attr_hlo_computation_name(ctx)
                    .ok_or_else(|| malformed("computation has no name"))?;
                if callees.insert(name.clone(), operation).is_some() {
                    return Err(malformed("duplicate computation name"));
                }
            }
        }
        Ok(Self {
            ctx,
            module,
            callees,
            channel_sensitive: crate::ir::config(ctx, module)?.channel_id_sensitive(),
            stack: Vec::new(),
            effects: effects::SideEffects::compute(module, ctx, &mut AnalysisManager::default())?,
            while_state: while_loop::WhileState::new(-1),
            float_context: None,
        })
    }

    pub(super) fn owner(&self, value: Value) -> Result<Ptr<Operation>> {
        let block = match value.defining_entity() {
            DefiningEntity::Op(operation) => operation.deref(self.ctx).get_parent_block(),
            DefiningEntity::Block(block) => Some(block),
        }
        .ok_or_else(|| malformed("evaluation value is detached"))?;
        let owner = block
            .deref(self.ctx)
            .get_parent_op(self.ctx)
            .ok_or_else(|| malformed("evaluation value has no frame owner"))?;
        if !Operation::is_op::<ComputationOp>(owner, self.ctx)
            && !Operation::is_op::<FusionOp>(owner, self.ctx)
        {
            return Err(malformed(
                "evaluation value has no native computation/fusion frame",
            ));
        }
        let mut parent = owner;
        loop {
            if parent == self.module {
                return Ok(owner);
            }
            parent = parent
                .deref(self.ctx)
                .get_parent_op(self.ctx)
                .ok_or_else(|| malformed("evaluation value is outside the supplied module"))?;
        }
    }

    pub(super) fn pure(&self, operation: Ptr<Operation>) -> Result<()> {
        if effects::has_local_side_effect(self.ctx, operation, self.channel_sensitive)? {
            return Err(failure(EvaluationError::Effectful {
                opcode: opcode(self.ctx, operation)?,
            }));
        }
        Ok(())
    }

    pub(super) fn computation(
        &mut self,
        callee: Ptr<Operation>,
        args: &[EvaluatedValue],
        path: &[usize],
    ) -> Result<EvaluatedValue> {
        if !self.callees.values().any(|&operation| operation == callee) {
            return Err(malformed("computation is outside the supplied module"));
        }
        self.enter(callee, args, path)
    }

    pub(super) fn enter(
        &mut self,
        owner: Ptr<Operation>,
        args: &[EvaluatedValue],
        path: &[usize],
    ) -> Result<EvaluatedValue> {
        if self.stack.contains(&owner) {
            return Err(failure(EvaluationError::Recursion));
        }
        let mut frame = Frame::new(self.ctx, owner)?;
        frame.bind(self.ctx, args)?;
        let bindings = std::mem::take(&mut frame.bindings);
        for (value, argument) in bindings {
            frame.bindings.insert(
                value,
                argument.retarget(self.ctx, value.get_type(self.ctx))?,
            );
        }
        // Visit transitive effects even when unused, but let Conditional choose
        // its branch before deciding which nested effect would execute.
        let mut effectful = Vec::new();
        for operation in frame.block.deref(self.ctx).iter(self.ctx) {
            if !Operation::is_op::<crate::ops::ReturnOp>(operation, self.ctx)
                && self.effects.has_side_effect(operation)?
            {
                effectful.push(operation.deref(self.ctx).get_result(0));
            }
        }
        self.stack.push(owner);
        let root = frame.root;
        let result = (|| {
            for value in effectful {
                self.eval(&mut frame, value, &[], DependencyMode::Recursive)?;
            }
            self.eval(&mut frame, root, path, DependencyMode::Recursive)
        })();
        let _ = self.stack.pop();
        result
    }

    pub(super) fn eval(
        &mut self,
        frame: &mut Frame,
        value: Value,
        path: &[usize],
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        if !frame.members.contains(&value) {
            return Err(malformed("value escapes its isolated evaluation frame"));
        }
        if let Some(replacement) = frame
            .substitutions
            .get(&value)
            .or_else(|| frame.bindings.get(&value))
        {
            return Ok(replacement.subvalue(path)?.clone());
        }
        let key = Key {
            value,
            path: path.to_vec(),
        };
        if let Some(cached) = frame.memo.get(&key) {
            return Ok(cached.clone());
        }
        if !frame.active.insert(key.clone()) {
            return Err(failure(EvaluationError::Recursion));
        }
        let result = self.evaluate_value(frame, value, path, mode);
        frame.active.remove(&key);
        let result = result?;
        let target = shape::subshape(self.ctx, value.get_type(self.ctx), path)?;
        let result = result.retarget(self.ctx, target)?;
        frame.memo.insert(key, result.clone());
        Ok(result)
    }

    pub(super) fn operand(
        &mut self,
        frame: &mut Frame,
        value: Value,
        path: &[usize],
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        if mode == DependencyMode::Shallow
            && !frame.substitutions.contains_key(&value)
            && !frame.bindings.contains_key(&value)
        {
            let constant = if let Some(operation) = value.defining_op() {
                opcode(self.ctx, operation)? == Opcode::Constant
            } else {
                false
            };
            if !constant {
                return Err(failure(EvaluationError::ShallowDependency));
            }
        }
        let result = self.eval(frame, value, path, mode)?;
        if mode == DependencyMode::Shallow && !result.is_known() {
            return Err(failure(EvaluationError::ShallowDependency));
        }
        Ok(result)
    }

    pub(super) fn resolve(&self, name: &crate::ir::ComputationName) -> Result<Ptr<Operation>> {
        self.callees
            .get(name)
            .copied()
            .ok_or_else(|| malformed("unresolved computation"))
    }
}
