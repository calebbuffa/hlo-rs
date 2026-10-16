use std::collections::{BTreeMap, BTreeSet};

use crate::{ComputationId, Error, InstructionId, Module, Result};

const PAIRS: &str = "_xla_send_recv_source_target_pairs";
const PIPELINE: &str = "_xla_send_recv_pipeline";

#[derive(Clone)]
struct Transfer {
    signature: String,
    pairs: Vec<(i64, i64)>,
}

fn conflicts(pairs: impl Iterator<Item = (i64, i64)>) -> Result<()> {
    let mut edges = BTreeMap::new();
    let mut sources = BTreeSet::new();
    let mut targets = BTreeSet::new();
    for (source, target) in pairs {
        if source < 0 || target < -1 {
            return Err(Error::new(
                "deadlock",
                "negative send/recv source or target",
            ));
        }
        // The pinned duplicate helper stores int64 IDs in std::set<int>.
        if !sources.insert(source as i32) || !targets.insert(target as i32) {
            return Err(Error::new(
                "deadlock",
                "duplicate send/recv source or target",
            ));
        }
        edges.insert(source, target);
    }
    let mut acyclic = BTreeSet::new();
    for &source in edges.keys() {
        let mut path = BTreeSet::new();
        let mut current = Some(source);
        while let Some(source) = current {
            if acyclic.contains(&source) {
                break;
            }
            if !path.insert(source) {
                return Err(Error::new(
                    "deadlock",
                    "cyclic send/recv source-target pairs",
                ));
            }
            current = edges.get(&source).copied();
        }
        acyclic.extend(path);
    }
    Ok(())
}

#[derive(Clone, Copy, Default, PartialEq)]
enum Expect {
    #[default]
    Neither,
    Send,
    Recv,
}

#[derive(Default)]
struct State {
    expect: Expect,
    sends: BTreeMap<InstructionId, Transfer>,
    recvs: BTreeMap<InstructionId, Transfer>,
}

impl State {
    fn transfer(&mut self, id: InstructionId, send: bool, transfer: Transfer) -> Result<()> {
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
                .find(|(_, candidate)| candidate.signature == transfer.signature)
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
        if self.sends.is_empty() != self.recvs.is_empty() {
            return Err(Error::new(
                "deadlock",
                "unmatched send or recv at computation boundary",
            ));
        }
        let mut unmatched = false;
        for send in self.sends.values() {
            let matched = self
                .recvs
                .iter()
                .find(|(_, recv)| send.signature == recv.signature)
                .map(|(&id, _)| id);
            if let Some(matched) = matched {
                self.recvs.remove(&matched);
            } else {
                unmatched = true;
            }
        }
        for recv in self.recvs.values() {
            let matched = self
                .sends
                .iter()
                .find(|(_, send)| recv.signature == send.signature)
                .map(|(&id, _)| id);
            if let Some(matched) = matched {
                self.sends.remove(&matched);
            } else {
                unmatched = true;
            }
        }
        if unmatched {
            return Err(Error::new(
                "deadlock",
                "send/recv source-target strings do not match",
            ));
        }
        Ok(())
    }
}

pub(super) fn verify(module: &Module) -> Result<()> {
    let instructions: BTreeMap<_, _> = module
        .computations()
        .map(|computation| Ok((computation, module.instructions(computation)?)))
        .collect::<Result<_>>()?;
    let wrapped: BTreeSet<_> = module
        .as_proto()
        .computations
        .iter()
        .flat_map(|computation| &computation.instructions)
        .filter(|instruction| {
            matches!(
                instruction.opcode.as_str(),
                "async-start" | "async-update" | "async-done"
            )
        })
        .flat_map(|instruction| instruction.called_computation_ids.iter().copied())
        .collect();
    enum Work {
        Visit {
            computation: ComputationId,
            index: usize,
            state: usize,
        },
        Enter {
            computation: ComputationId,
            parent: usize,
        },
        Exit {
            state: usize,
            parent: usize,
        },
    }
    let mut states = vec![State::default()];
    let mut pending = vec![Work::Visit {
        computation: module.entry()?,
        index: 0,
        state: 0,
    }];
    while let Some(work) = pending.pop() {
        let (computation, index, state) = match work {
            Work::Exit { state, parent } => {
                // The pinned predicate intentionally observes the parent's
                // expectation, not the isolated group's expectation.
                if states[parent].expect != Expect::Neither {
                    states[state].finish()?;
                }
                states.pop();
                continue;
            }
            Work::Enter {
                computation,
                parent,
            } => {
                let data = module.computation(computation)?;
                let isolate = wrapped.contains(&data.id)
                    && data.instructions.iter().any(|instruction| {
                        instruction.id != data.root_id && instruction.opcode != "parameter"
                    });
                let state = if isolate {
                    let state = states.len();
                    states.push(State::default());
                    pending.push(Work::Exit { state, parent });
                    state
                } else {
                    parent
                };
                (computation, 0, state)
            }
            Work::Visit {
                computation,
                index,
                state,
            } => (computation, index, state),
        };
        let Some(&id) = instructions[&computation].get(index) else {
            continue;
        };
        pending.push(Work::Visit {
            computation,
            index: index + 1,
            state,
        });
        let instruction = module.instruction(id)?;
        if !instruction.called_computation_ids.is_empty() {
            pending.extend(
                instruction
                    .called_computation_ids
                    .iter()
                    .rev()
                    .map(|&callee| Work::Enter {
                        computation: module.computation_handle(callee),
                        parent: state,
                    }),
            );
            continue;
        }
        match instruction.opcode.as_str() {
            "send" | "recv" => {
                let attributes = instruction.frontend_attributes.get_or_default();
                if instruction.is_host_transfer || attributes.map.contains_key(PIPELINE) {
                    continue;
                }
                let signature = attributes.map.get(PAIRS).ok_or_else(|| {
                    Error::new("deadlock", "send/recv lacks source-target pairs attribute")
                        .context(&instruction.name)
                })?;
                let pairs = crate::text::collective::source_target_pairs(signature)?;
                conflicts(pairs.iter().copied())
                    .map_err(|error| error.context(&instruction.name))?;
                states[state]
                    .transfer(
                        id,
                        instruction.opcode == "send",
                        Transfer {
                            signature: signature.clone(),
                            pairs,
                        },
                    )
                    .map_err(|error| error.context(&instruction.name))?;
            }
            "all-gather"
            | "all-reduce"
            | "all-to-all"
            | "ragged-all-to-all"
            | "collective-permute"
            | "reduce-scatter"
            | "collective-broadcast"
                if states[state].expect != Expect::Neither =>
            {
                return Err(
                    Error::new("deadlock", "collective interrupts unmatched send/recv")
                        .context(&instruction.name),
                );
            }
            _ => {}
        }
    }
    if states[0].expect != Expect::Neither {
        states[0].finish()?;
    }
    Ok(())
}
