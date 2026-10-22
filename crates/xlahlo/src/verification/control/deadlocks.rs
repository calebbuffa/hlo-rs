//! Schedule-order communication checks using native source/target metadata.
use super::{asynchronous, flag, optional_attr, require, required};
use crate::Diagnostic;
use crate::{
    attributes::{Callees, FrontendAttributes, OptionalInteger, SourceTargetPairs},
    ir::{self, InstructionName, Schedule},
    opcode::Opcode,
    ops::ComputationOp,
};
use pliron::result::Result;
use pliron::{
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    operation::Operation,
};
use std::{
    cell::Ref,
    collections::{HashMap, HashSet},
};

fn conflicts(pairs: impl Iterator<Item = (i64, i64)>) -> Result<()> {
    let mut edges = HashMap::new();
    let mut sources = HashSet::new();
    let mut targets = HashSet::new();
    for (source, target) in pairs {
        require(
            source >= 0 && target >= -1,
            "negative send/recv source or target",
        )?;
        require(
            sources.insert(source) && targets.insert(target),
            "duplicate send/recv source or target",
        )?;
        edges.insert(source, target);
    }
    let mut acyclic = HashSet::new();
    for &source in edges.keys() {
        let mut path = HashSet::new();
        let mut current = Some(source);
        while let Some(source) = current {
            if acyclic.contains(&source) {
                break;
            }
            require(path.insert(source), "cyclic send/recv source-target pairs")?;
            current = edges.get(&source).copied();
        }
        acyclic.extend(path);
    }
    Ok(())
}

struct Transfer<'a> {
    signature: Ref<'a, SourceTargetPairs>,
    pairs: Vec<(i64, i64)>,
}
#[derive(Clone, Copy, Default, PartialEq)]
enum Expect {
    #[default]
    Neither,
    Send,
    Recv,
}
#[derive(Default)]
struct State<'a> {
    expect: Expect,
    sends: HashMap<Ptr<Operation>, Transfer<'a>>,
    recvs: HashMap<Ptr<Operation>, Transfer<'a>>,
}
impl<'a> State<'a> {
    fn transfer(&mut self, id: Ptr<Operation>, send: bool, transfer: Transfer<'a>) -> Result<()> {
        let expected = if send { Expect::Send } else { Expect::Recv };
        let opposite = if send { Expect::Recv } else { Expect::Send };
        let (same, other) = if send {
            (&mut self.sends, &mut self.recvs)
        } else {
            (&mut self.recvs, &mut self.sends)
        };
        if self.expect == Expect::Neither {
            same.insert(id, transfer);
            self.expect = opposite;
        } else if self.expect == opposite {
            for pending in same.values() {
                conflicts(transfer.pairs.iter().chain(&pending.pairs).copied())?;
            }
            same.insert(id, transfer);
        } else if self.expect == expected {
            let matched = other
                .iter()
                .find(|(_, candidate)| *candidate.signature == *transfer.signature)
                .map(|(&id, _)| id);
            if let Some(matched) = matched {
                other.remove(&matched);
                if other.is_empty() {
                    self.expect = Expect::Neither;
                }
            } else {
                same.insert(id, transfer);
            }
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        require(
            self.sends.is_empty() == self.recvs.is_empty(),
            "unmatched send or recv at computation boundary",
        )?;
        let mut unmatched = false;
        for send in self.sends.values() {
            let matched = self
                .recvs
                .iter()
                .find(|(_, recv)| *send.signature == *recv.signature)
                .map(|(&id, _)| id);
            if let Some(id) = matched {
                self.recvs.remove(&id);
            } else {
                unmatched = true;
            }
        }
        for recv in self.recvs.values() {
            let matched = self
                .sends
                .iter()
                .find(|(_, send)| *recv.signature == *send.signature)
                .map(|(&id, _)| id);
            if let Some(id) = matched {
                self.sends.remove(&id);
            } else {
                unmatched = true;
            }
        }
        require(!unmatched, "send/recv source-target pairs do not match")
    }
}

pub(super) fn verify(
    ctx: &Context,
    all: &[(Ptr<Operation>, Option<Ptr<Operation>>)],
) -> Result<()> {
    let mut computations = HashMap::new();
    let mut instructions = HashMap::<_, Vec<_>>::new();
    let mut wrapped = HashSet::new();
    let mut entry = None;
    for &(op, owner) in all {
        if let Some(comp) = Operation::get_op::<ComputationOp>(op, ctx) {
            let name = comp.get_attr_hlo_computation_name(ctx).ok_or_else(|| {
                pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "missing computation name".into()
                })
            })?;
            computations.insert(name.clone(), op);
            let block = super::super::single_block(ctx, comp.body(ctx))?;
            instructions.insert(
                op,
                block
                    .deref(ctx)
                    .iter(ctx)
                    .filter(|&op| asynchronous::opcode(ctx, op).is_some())
                    .collect(),
            );
            if entry.is_none() {
                entry = Some(op);
            }
        }
        let _ = owner;
    }
    if let Some(&(root, _)) = all.first()
        && Operation::is_op::<ModuleOp>(root, ctx)
    {
        let name = ir::entry(ctx, root)?;
        entry = computations.get(&name).copied();
        if let Some(schedule) =
            optional_attr::<Schedule>(ctx, root, crate::attributes::keys::SCHEDULE)?
        {
            let mut named = HashMap::new();
            for &(op, _) in all {
                if let Some(name) =
                    optional_attr::<InstructionName>(ctx, op, crate::attributes::keys::NAME)?
                {
                    named.insert(name.0.clone(), op);
                }
            }
            for sequence in &schedule.0 {
                let computation = computations.get(&sequence.computation).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "scheduled computation not found".into()
                    })
                })?;
                let ordered = sequence
                    .instructions
                    .iter()
                    .map(|name| {
                        named.get(name.as_str()).copied().ok_or_else(|| {
                            pliron::verify_error_noloc!(Diagnostic::Verification {
                                message: "scheduled instruction not found".into()
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                instructions.insert(*computation, ordered);
            }
        }
    }
    for &(op, _) in all {
        if matches!(
            asynchronous::opcode(ctx, op),
            Some(Opcode::AsyncStart | Opcode::AsyncUpdate | Opcode::AsyncDone)
        ) && let Some(callees) =
            optional_attr::<Callees>(ctx, op, crate::attributes::keys::CALLEES)?
        {
            for name in &callees.0 {
                if let Some(&callee) = computations.get(name) {
                    wrapped.insert(callee);
                }
            }
        }
    }
    enum Work {
        Visit(Ptr<Operation>, usize, usize),
        Enter(Ptr<Operation>, usize),
        Exit(usize, usize),
    }
    let mut states = vec![State::default()];
    let mut pending = vec![Work::Visit(
        entry.ok_or_else(|| {
            pliron::verify_error_noloc!(Diagnostic::Verification {
                message: "missing entry computation".into()
            })
        })?,
        0,
        0,
    )];
    let mut active = HashSet::new();
    while let Some(work) = pending.pop() {
        let (computation, index, state) = match work {
            Work::Exit(state, parent) => {
                if states[parent].expect != Expect::Neither {
                    states[state].finish()?;
                }
                continue;
            }
            Work::Enter(comp, parent) => {
                let isolate = wrapped.contains(&comp)
                    && instructions.get(&comp).is_some_and(|ops| {
                        ops.iter()
                            .filter(|&&op| asynchronous::opcode(ctx, op) != Some(Opcode::Parameter))
                            .count()
                            > 1
                    });
                let state = if isolate {
                    let state = states.len();
                    states.push(State::default());
                    pending.push(Work::Exit(state, parent));
                    state
                } else {
                    parent
                };
                (comp, 0, state)
            }
            Work::Visit(comp, index, state) => (comp, index, state),
        };
        let Some(&op) = instructions
            .get(&computation)
            .and_then(|ops| ops.get(index))
        else {
            active.remove(&computation);
            continue;
        };
        if index == 0 {
            require(
                active.insert(computation),
                "recursive computation calls are forbidden",
            )?;
        }
        pending.push(Work::Visit(computation, index + 1, state));
        if let Some(callees) = optional_attr::<Callees>(ctx, op, crate::attributes::keys::CALLEES)?
            && !callees.0.is_empty()
        {
            for name in callees.0.iter().rev() {
                let callee = computations.get(name).ok_or_else(|| {
                    pliron::verify_error_noloc!(Diagnostic::Verification {
                        message: "callee not found".into()
                    })
                })?;
                pending.push(Work::Enter(*callee, state));
            }
            continue;
        }
        match asynchronous::opcode(ctx, op) {
            Some(kind @ (Opcode::Send | Opcode::Recv)) => {
                let frontend = optional_attr::<FrontendAttributes>(
                    ctx,
                    op,
                    crate::attributes::keys::FRONTEND_ATTRIBUTES,
                )?;
                if flag(ctx, op, "hlo_host_transfer")?
                    || frontend
                        .as_ref()
                        .is_some_and(|a| a.get("_xla_send_recv_pipeline").is_some())
                    || optional_attr::<OptionalInteger>(ctx, op, "hlo_pipeline")?.is_some()
                {
                    continue;
                }
                let signature: Ref<'_, SourceTargetPairs> =
                    required(ctx, op, "hlo_source_target_pairs")?;
                let pairs = signature
                    .0
                    .iter()
                    .map(|p| (p.source, p.target))
                    .collect::<Vec<_>>();
                conflicts(pairs.iter().copied())?;
                states[state].transfer(op, kind == Opcode::Send, Transfer { signature, pairs })?;
            }
            Some(
                Opcode::AllGather
                | Opcode::AllReduce
                | Opcode::AllToAll
                | Opcode::RaggedAllToAll
                | Opcode::CollectivePermute
                | Opcode::ReduceScatter
                | Opcode::CollectiveBroadcast,
            ) if states[state].expect != Expect::Neither => {
                return Err(pliron::verify_error_noloc!(Diagnostic::Verification {
                    message: "collective interrupts unmatched send/recv".into()
                }));
            }
            _ => {}
        }
    }
    if states[0].expect != Expect::Neither {
        states[0].finish()?;
    }
    Ok(())
}
