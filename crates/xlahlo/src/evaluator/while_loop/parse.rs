//! Conservative, request/frame-local proofs, not a public trip-count API.
//!
//! Pinned evaluator.cc236-815 supplies the source patterns. Checked range
//! admission intentionally declines overflow/wrapping cases rather than
//! reproducing unchecked C++ trip-count arithmetic.
use super::*;
use crate::types::ElementType;
use std::collections::HashMap;

#[derive(Clone)]
enum Node {
    Input(Vec<usize>, TypeHandle),
    Known(EvaluatedValue),
    Expression {
        operation: Ptr<Operation>,
        opcode: Opcode,
        args: Vec<Node>,
    },
}

#[derive(Clone)]
struct Scope {
    // Zero denotes source resolution; bound environments must not share cycle keys.
    id: usize,
    owner: Ptr<Operation>,
    bindings: HashMap<Value, Node>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct Visit {
    scope: usize,
    owner: Ptr<Operation>,
    value: Value,
    path: Vec<usize>,
}

/// Never cached or exposed: a proof is consumed within the same borrowed
/// request/frame that resolved its constants, substitutions and callers.
pub(super) struct Proof {
    shape: TypeHandle,
    index: usize,
    element: ElementType,
    final_value: i128,
}

impl Proof {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn materialize(self, ctx: &Context) -> Result<EvaluatedValue> {
        let mut children = Vec::new();
        let types = shape::children(ctx, self.shape).to_vec();
        for (index, ty) in types.into_iter().enumerate() {
            let value = if index == self.index {
                EvaluatedValue::known(
                    ctx,
                    literal_semantics::integer_scalar(ctx, self.element, self.final_value)?,
                )?
                .retarget(ctx, ty)?
            } else {
                EvaluatedValue::unknown(ctx, ty)?
            };
            children.push(value);
        }
        EvaluatedValue::tuple(ctx, self.shape, children)
    }
}

pub(super) struct Parser<'s, 'c, 'f> {
    session: &'s mut Session<'c>,
    actual: &'f mut Frame,
    active: HashSet<Visit>,
    calls: Vec<Ptr<Operation>>,
    input: Option<Value>,
    next_scope: usize,
}

impl<'s, 'c, 'f> Parser<'s, 'c, 'f> {
    pub fn new(session: &'s mut Session<'c>, actual: &'f mut Frame) -> Self {
        Self {
            session,
            actual,
            active: HashSet::new(),
            calls: Vec::new(),
            input: None,
            next_scope: 1,
        }
    }

    pub fn context(&self) -> &'c Context {
        self.session.ctx
    }

    pub fn prove(
        &mut self,
        operation: Ptr<Operation>,
        initial: &EvaluatedValue,
    ) -> Result<Option<Proof>> {
        let ctx = self.context();
        let (input, body, condition) = self.session.while_parts(operation)?;
        self.input = Some(input);
        let shape = operation.deref(ctx).get_result(0).get_type(ctx);
        if !shape::is_tuple(ctx, shape) {
            return Ok(None);
        }
        // An authoritative replacement for the source tuple is not a license
        // to pattern-match a different, original tuple expression.
        if self.actual.substitutions.contains_key(&input)
            || input.defining_op().map(|op| opcode(ctx, op)).transpose()? != Some(Opcode::Tuple)
        {
            return Ok(None);
        }
        let parameter = Node::Input(vec![], shape);
        let Some(body_scope) = self.bound_scope(body, std::slice::from_ref(&parameter))? else {
            return Ok(None);
        };
        let Some(condition_scope) =
            self.bound_scope(condition, std::slice::from_ref(&parameter))?
        else {
            return Ok(None);
        };
        let body_root = Frame::new(ctx, body)?.root;
        let condition_root = Frame::new(ctx, condition)?.root;
        let types = shape::children(ctx, shape).to_vec();
        let mut invariant = HashSet::new();
        for index in 0..types.len() {
            if matches!(
                self.visit(&body_scope, body_root, &[index])?,
                Some(Node::Input(ref path, _)) if path.as_slice() == [index]
            ) {
                invariant.insert(index);
            }
        }
        let Some(node) = self.visit(&condition_scope, condition_root, &[])? else {
            return Ok(None);
        };
        let Node::Expression {
            operation: comparison,
            opcode: Opcode::Compare,
            args,
        } = node
        else {
            // Boolean passthrough can describe a loop, but the pinned scalar
            // IV constructor admits integers, not Pred. No fabricated proof.
            return Ok(None);
        };
        if args.len() != 2 {
            return Err(malformed("comparison has invalid arity in While proof"));
        }
        let mut direction = attr::<Direction>(ctx, comparison, "hlo_direction")?;
        let candidate = |node: &Node| match node {
            Node::Input(path, ty) if path.len() == 1 => Some((path[0], *ty)),
            _ => None,
        };
        let (index, ty, bound) = if let Some((index, ty)) = candidate(&args[0])
            && !invariant.contains(&index)
        {
            let Some(bound) = self.integer(&args[1], initial, &invariant)? else {
                return Ok(None);
            };
            (index, ty, bound)
        } else if let Some((index, ty)) = candidate(&args[1])
            && !invariant.contains(&index)
        {
            let Some(bound) = self.integer(&args[0], initial, &invariant)? else {
                return Ok(None);
            };
            direction = reverse(direction);
            (index, ty, bound)
        } else {
            return Ok(None);
        };
        let element = shape::element_type(ctx, ty)?;
        if !element.is_integer() || shape::rank(ctx, ty)? != 0 || shape::is_dynamic(ctx, ty) {
            return Ok(None);
        }
        let init = match known_integer(ctx, initial.subvalue(&[index])?)? {
            Some(init) => init,
            None => {
                let scope = Scope {
                    id: 0,
                    owner: self.actual.owner,
                    bindings: HashMap::new(),
                };
                let Some(node) = self.visit(&scope, input, &[index])? else {
                    return Ok(None);
                };
                let Some(init) = self.integer(&node, initial, &invariant)? else {
                    return Ok(None);
                };
                init
            }
        };
        let Some(update) = self.visit(&body_scope, body_root, &[index])? else {
            return Ok(None);
        };
        let Some(step) = self.affine_step(&update, index, initial, &invariant)? else {
            return Ok(None);
        };
        let Some(final_value) = checked_final(init, bound, step, direction, element) else {
            return Ok(None);
        };
        Ok(Some(Proof {
            shape,
            index,
            element,
            final_value,
        }))
    }

    fn bound_scope(&mut self, owner: Ptr<Operation>, args: &[Node]) -> Result<Option<Scope>> {
        let ctx = self.context();
        let frame = Frame::new(ctx, owner)?;
        let mut bindings = HashMap::new();
        if Operation::is_op::<ComputationOp>(owner, ctx) {
            for op in frame.block.deref(ctx).iter(ctx) {
                if Operation::is_op::<crate::ops::control::ParameterOp>(op, ctx) {
                    let index = attr::<Index>(ctx, op, "hlo_index")?.0 as usize;
                    let Some(argument) = args.get(index) else {
                        return Err(malformed("symbolic computation argument count mismatch"));
                    };
                    bindings.insert(op.deref(ctx).get_result(0), argument.clone());
                }
            }
        } else {
            if frame.block.deref(ctx).get_num_arguments() != args.len() {
                return Err(malformed("symbolic fusion argument count mismatch"));
            }
            for (index, argument) in args.iter().enumerate() {
                bindings.insert(frame.block.deref(ctx).get_argument(index), argument.clone());
            }
        }
        if bindings.len() != args.len() {
            return Err(malformed("symbolic frame parameter count mismatch"));
        }
        let id = self.next_scope;
        self.next_scope += 1;
        Ok(Some(Scope {
            id,
            owner,
            bindings,
        }))
    }

    fn visit(&mut self, scope: &Scope, value: Value, path: &[usize]) -> Result<Option<Node>> {
        if self.active.len() >= 128 {
            return Ok(None);
        }
        if scope.owner == self.actual.owner
            && let Some(value) = self
                .actual
                .substitutions
                .get(&value)
                .or_else(|| self.actual.bindings.get(&value))
        {
            return Ok(Some(Node::Known(value.subvalue(path)?.clone())));
        }
        if let Some(node) = scope.bindings.get(&value) {
            return project(self.context(), node.clone(), path).map(Some);
        }
        let key = Visit {
            scope: scope.id,
            owner: scope.owner,
            value,
            path: path.to_vec(),
        };
        if !self.active.insert(key.clone()) {
            return Ok(None);
        }
        let result = self.visit_inner(scope, value, path);
        self.active.remove(&key);
        result
    }

    fn visit_inner(&mut self, scope: &Scope, value: Value, path: &[usize]) -> Result<Option<Node>> {
        let ctx = self.context();
        let Some(operation) = value.defining_op() else {
            return Ok(None);
        };
        let op = opcode(ctx, operation)?;
        self.session.pure(operation)?;
        if optional_attr::<ControlPredecessors>(ctx, operation, "hlo_control_predecessors")?
            .is_some_and(|controls| !controls.0.is_empty())
        {
            return Ok(None);
        }
        let operands: Vec<_> = operation.deref(ctx).operands().collect();
        match op {
            Opcode::Constant => Ok(Some(Node::Known(
                EvaluatedValue::known(ctx, attr::<Literal>(ctx, operation, "hlo_literal")?)?
                    .subvalue(path)?
                    .clone(),
            ))),
            Opcode::Parameter => self.resolve_unique_caller(scope, operation, path),
            Opcode::Copy | Opcode::Domain | Opcode::OptimizationBarrier => {
                self.visit(scope, operands[0], path)
            }
            Opcode::GetTupleElement => {
                let mut input_path = vec![attr::<Index>(ctx, operation, "hlo_index")?.0 as usize];
                input_path.extend_from_slice(path);
                self.visit(scope, operands[0], &input_path)
            }
            Opcode::Tuple => {
                if let Some((&index, tail)) = path.split_first() {
                    let input = operands
                        .get(index)
                        .ok_or_else(|| malformed("symbolic tuple path is invalid"))?;
                    self.visit(scope, *input, tail)
                } else {
                    Ok(None)
                }
            }
            Opcode::Call | Opcode::Fusion => {
                let owner = if op == Opcode::Fusion {
                    operation
                } else {
                    let names = attr::<Callees>(ctx, operation, "hlo_callees")?;
                    if names.0.len() != 1 {
                        return Err(malformed("symbolic Call requires one callee"));
                    }
                    self.session.resolve(&names.0[0])?
                };
                if self.calls.contains(&owner) {
                    return Ok(None);
                }
                let mut args = Vec::new();
                for input in operands {
                    let Some(argument) = self.visit(scope, input, &[])? else {
                        return Ok(None);
                    };
                    args.push(argument);
                }
                let Some(inner) = self.bound_scope(owner, &args)? else {
                    return Ok(None);
                };
                self.calls.push(owner);
                let root = Frame::new(ctx, owner)?.root;
                let result = self.visit(&inner, root, path);
                let _ = self.calls.pop();
                result
            }
            Opcode::Add
            | Opcode::Subtract
            | Opcode::Multiply
            | Opcode::Divide
            | Opcode::Remainder
            | Opcode::Negate
            | Opcode::Abs
            | Opcode::Sign
            | Opcode::Not
            | Opcode::And
            | Opcode::Or
            | Opcode::Xor
            | Opcode::ShiftLeft
            | Opcode::ShiftRightArithmetic
            | Opcode::ShiftRightLogical
            | Opcode::Clz
            | Opcode::PopulationCount
            | Opcode::Minimum
            | Opcode::Maximum
            | Opcode::Power
            | Opcode::Convert
            | Opcode::BitcastConvert
            | Opcode::Compare
            | Opcode::Select
            | Opcode::Clamp => {
                if !path.is_empty() {
                    return Ok(None);
                }
                let ty = value.get_type(ctx);
                if !shape::is_array(ctx, ty)
                    || shape::is_dynamic(ctx, ty)
                    || shape::element_count(ctx, ty)? != 1
                {
                    return Ok(None);
                }
                let element = shape::element_type(ctx, ty)?;
                if !element.is_integer() && element != ElementType::Pred {
                    return Ok(None);
                }
                let mut args = Vec::new();
                for input in operands {
                    let Some(argument) = self.visit(scope, input, &[])? else {
                        return Ok(None);
                    };
                    args.push(argument);
                }
                Ok(Some(Node::Expression {
                    operation,
                    opcode: op,
                    args,
                }))
            }
            _ => Ok(None),
        }
    }

    fn resolve_unique_caller(
        &mut self,
        scope: &Scope,
        parameter: Ptr<Operation>,
        path: &[usize],
    ) -> Result<Option<Node>> {
        let ctx = self.context();
        let mut callers = Vec::new();
        for &callee in self.session.callees.values() {
            self.collect_callers(callee, scope.owner, &mut callers)?;
        }
        if callers.len() != 1 {
            return Ok(None);
        }
        let (caller_owner, caller) = callers[0];
        let index = attr::<Index>(ctx, parameter, "hlo_index")?.0 as usize;
        let caller_scope = Scope {
            id: 0,
            owner: caller_owner,
            bindings: HashMap::new(),
        };
        match opcode(ctx, caller)? {
            Opcode::Call => {
                let inputs: Vec<_> = caller.deref(ctx).operands().collect();
                let input = inputs
                    .get(index)
                    .ok_or_else(|| malformed("caller parameter index is invalid"))?;
                self.visit(&caller_scope, *input, path)
            }
            Opcode::While if index == 0 && !path.is_empty() => {
                let (input, body, _) = self.session.while_parts(caller)?;
                let ty = input.get_type(ctx);
                let Some(body_scope) = self.bound_scope(body, &[Node::Input(vec![], ty)])? else {
                    return Ok(None);
                };
                let root = Frame::new(ctx, body)?.root;
                if !matches!(
                    self.visit(&body_scope, root, path)?,
                    Some(Node::Input(ref forwarded, _)) if forwarded == path
                ) {
                    return Ok(None);
                }
                self.visit(&caller_scope, input, path)
            }
            _ => Ok(None),
        }
    }

    fn collect_callers(
        &self,
        owner: Ptr<Operation>,
        target: Ptr<Operation>,
        callers: &mut Vec<(Ptr<Operation>, Ptr<Operation>)>,
    ) -> Result<()> {
        let ctx = self.context();
        let frame = Frame::new(ctx, owner)?;
        for operation in frame.block.deref(ctx).iter(ctx) {
            if Operation::is_op::<crate::ops::ReturnOp>(operation, ctx) {
                continue;
            }
            if let Some(names) = optional_attr::<Callees>(ctx, operation, "hlo_callees")?
                && names
                    .0
                    .iter()
                    .any(|name| self.session.callees.get(name) == Some(&target))
            {
                callers.push((owner, operation));
            }
            if opcode(ctx, operation)? == Opcode::Fusion {
                self.collect_callers(operation, target, callers)?;
            }
        }
        Ok(())
    }

    fn resolve_scalar(
        &mut self,
        node: &Node,
        initial: &EvaluatedValue,
        invariant: &HashSet<usize>,
    ) -> Result<Option<Scalar>> {
        let ctx = self.context();
        match node {
            Node::Known(value) => scalar(ctx, value),
            Node::Input(path, _) => {
                if path.first().is_some_and(|index| invariant.contains(index)) {
                    if let Some(value) = scalar(ctx, initial.subvalue(path)?)? {
                        return Ok(Some(value));
                    }
                    let Some(input) = self.input else {
                        return Ok(None);
                    };
                    let scope = Scope {
                        id: 0,
                        owner: self.actual.owner,
                        bindings: HashMap::new(),
                    };
                    let Some(node) = self.visit(&scope, input, path)? else {
                        return Ok(None);
                    };
                    // Source resolution may recurse through callers, never
                    // through this loop's symbolic Input environment.
                    if matches!(node, Node::Input(..)) {
                        return Ok(None);
                    }
                    self.resolve_scalar(&node, initial, invariant)
                } else {
                    Ok(None)
                }
            }
            Node::Expression {
                operation,
                opcode,
                args,
            } => {
                let mut values = Vec::new();
                for node in args {
                    let Some(value) = self.resolve_scalar(node, initial, invariant)? else {
                        return Ok(None);
                    };
                    values.push(value);
                }
                let ty = operation.deref(ctx).get_result(0).get_type(ctx);
                let element = shape::element_type(ctx, ty)?;
                self.session
                    .scalar(*operation, *opcode, element, &values)
                    .map(Some)
            }
        }
    }

    fn integer(
        &mut self,
        node: &Node,
        initial: &EvaluatedValue,
        invariant: &HashSet<usize>,
    ) -> Result<Option<i64>> {
        let Some(value) = self.resolve_scalar(node, initial, invariant)? else {
            return Ok(None);
        };
        known_integer(
            self.context(),
            &EvaluatedValue::known(self.context(), value.to_literal(self.context())?)?,
        )
    }

    fn affine_step(
        &mut self,
        node: &Node,
        index: usize,
        initial: &EvaluatedValue,
        invariant: &HashSet<usize>,
    ) -> Result<Option<i64>> {
        if matches!(node, Node::Input(path, _) if path.as_slice() == [index]) {
            return Ok(Some(0));
        }
        let Node::Expression { opcode, args, .. } = node else {
            return Ok(None);
        };
        if args.len() != 2 || !matches!(opcode, Opcode::Add | Opcode::Subtract) {
            return Ok(None);
        }
        if matches!(&args[0], Node::Input(path, _) if path.as_slice() == [index]) {
            let Some(value) = self.integer(&args[1], initial, invariant)? else {
                return Ok(None);
            };
            return Ok(if *opcode == Opcode::Subtract {
                value.checked_neg()
            } else {
                Some(value)
            });
        }
        if *opcode == Opcode::Add
            && matches!(&args[1], Node::Input(path, _) if path.as_slice() == [index])
        {
            return self.integer(&args[0], initial, invariant);
        }
        Ok(None)
    }
}

fn project(ctx: &Context, node: Node, path: &[usize]) -> Result<Node> {
    match node {
        Node::Input(mut prefix, ty) => {
            let ty = shape::subshape(ctx, ty, path)?;
            prefix.extend_from_slice(path);
            Ok(Node::Input(prefix, ty))
        }
        Node::Known(value) => Ok(Node::Known(value.subvalue(path)?.clone())),
        expression if path.is_empty() => Ok(expression),
        _ => Err(malformed("cannot project a symbolic numerical expression")),
    }
}

fn scalar(ctx: &Context, value: &EvaluatedValue) -> Result<Option<Scalar>> {
    if !value.is_known()
        || !shape::is_array(ctx, value.shape())
        || shape::is_dynamic(ctx, value.shape())
        || shape::element_count(ctx, value.shape())? != 1
    {
        return Ok(None);
    }
    let element = shape::element_type(ctx, value.shape())?;
    if !element.is_integer() && element != ElementType::Pred {
        return Ok(None);
    }
    let literal = value.clone().into_literal(ctx)?;
    let index = vec![0; shape::rank(ctx, value.shape())?];
    Scalar::read(ctx, &literal, &index).map(Some)
}

fn known_integer(ctx: &Context, value: &EvaluatedValue) -> Result<Option<i64>> {
    let Some(value) = scalar(ctx, value)? else {
        return Ok(None);
    };
    if !value.element().is_integer()
        || (!value.element().is_signed() && value.bits() > i64::MAX as u128)
    {
        return Ok(None);
    }
    literal_semantics::scalar_integer(ctx, &value.to_literal(ctx)?).map(Some)
}

fn reverse(direction: Direction) -> Direction {
    match direction {
        Direction::Lt => Direction::Gt,
        Direction::Le => Direction::Ge,
        Direction::Gt => Direction::Lt,
        Direction::Ge => Direction::Le,
        Direction::Eq => Direction::Eq,
        Direction::Ne => Direction::Ne,
    }
}

pub(super) fn checked_final(
    init: i64,
    bound: i64,
    step: i64,
    direction: Direction,
    element: ElementType,
) -> Option<i128> {
    let (init, bound, step) = (i128::from(init), i128::from(bound), i128::from(step));
    let inclusive = matches!(direction, Direction::Le | Direction::Ge);
    let (distance, stride) = match direction {
        Direction::Lt | Direction::Le if step > 0 => (bound - init, step),
        Direction::Gt | Direction::Ge if step < 0 => (init - bound, -step),
        _ => return None,
    };
    let trips = if distance < 0 || (distance == 0 && !inclusive) {
        0
    } else if inclusive {
        distance / stride + 1
    } else {
        (distance + stride - 1) / stride
    };
    let final_value = init.checked_add(trips.checked_mul(step)?)?;
    let width = element.bit_width();
    if !element.is_integer() {
        return None;
    }
    let (minimum, maximum) = if element.is_signed() {
        (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
    } else {
        (0, (1i128 << width) - 1)
    };
    if trips > i128::from(i64::MAX)
        || init < minimum
        || init > maximum
        || final_value < minimum
        || final_value > maximum
        || i64::try_from(final_value).is_err()
    {
        return None;
    }
    Some(final_value)
}
