//! Program IR fechado, VM determinística, Thought JIT e deoptimization.

use std::collections::{BTreeMap, BTreeSet};

use super::ids::{FactorId, ProgramId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    Atom(String),
    Bool(bool),
    Unit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    Atom,
    Bool,
    Unit,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum OpCode {
    Identity,
    TransitiveBefore,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Guard {
    ContextRequired(String),
    InputCount(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramStatus {
    Candidate,
    Compiled,
    Deprecated,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    pub id: ProgramId,
    pub inputs: Vec<ValueType>,
    pub output: ValueType,
    pub opcode: OpCode,
    pub guards: Vec<Guard>,
    pub provenance: Vec<String>,
    pub status: ProgramStatus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceStep {
    Retrieve(FactorId),
    Compare(FactorId, FactorId),
    Infer(String),
    Verify(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReasoningTrace {
    pub steps: Vec<TraceStep>,
    pub result: Value,
    pub verified: bool,
    pub operations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramDispatch {
    Interpreted,
    CompiledHit,
    Deoptimized,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmOutcome {
    pub result: Value,
    pub dispatch: ProgramDispatch,
    pub trace: ReasoningTrace,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProgramVm;

impl ProgramVm {
    pub fn execute(
        &self,
        program: &Program,
        inputs: &[Value],
        context: &BTreeSet<String>,
    ) -> Option<VmOutcome> {
        let guard_holds = program.guards.iter().all(|guard| match guard {
            Guard::ContextRequired(required) => context.contains(required),
            Guard::InputCount(count) => inputs.len() == *count,
        });
        let dispatch = if program.status == ProgramStatus::Compiled && guard_holds {
            ProgramDispatch::CompiledHit
        } else if program.status == ProgramStatus::Compiled {
            ProgramDispatch::Deoptimized
        } else {
            ProgramDispatch::Interpreted
        };
        let result = interpret(program.opcode, inputs)?;
        let steps = match program.opcode {
            OpCode::Identity => vec![TraceStep::Infer("identity".to_string())],
            OpCode::TransitiveBefore => vec![
                TraceStep::Retrieve(FactorId(0)),
                TraceStep::Compare(FactorId(0), FactorId(1)),
                TraceStep::Infer("transitive-before".to_string()),
                TraceStep::Verify("relation-chain".to_string()),
            ],
        };
        Some(VmOutcome {
            result: result.clone(),
            dispatch,
            trace: ReasoningTrace {
                operations: steps.len() as u64,
                steps,
                result,
                verified: true,
            },
        })
    }
}

fn interpret(opcode: OpCode, inputs: &[Value]) -> Option<Value> {
    match opcode {
        OpCode::Identity => inputs.first().cloned(),
        OpCode::TransitiveBefore => match inputs {
            [Value::Atom(left), Value::Atom(_middle), Value::Atom(right)] => {
                Some(Value::Atom(format!("{left} before {right}")))
            }
            _ => None,
        },
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProgramStats {
    pub calls: u64,
    pub failures: u64,
    pub compute_saved: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramLibrary {
    compile_after: u32,
    traces: BTreeMap<(OpCode, Vec<Guard>), u32>,
    programs: BTreeMap<ProgramId, Program>,
    stats: BTreeMap<ProgramId, ProgramStats>,
}

impl ProgramLibrary {
    pub fn new(compile_after: u32) -> Option<Self> {
        (compile_after > 0).then(|| Self {
            compile_after,
            traces: BTreeMap::new(),
            programs: BTreeMap::new(),
            stats: BTreeMap::new(),
        })
    }

    pub fn record_verified_trace(
        &self,
        opcode: OpCode,
        guards: &[Guard],
        trace: &ReasoningTrace,
    ) -> Option<(Self, Option<Program>)> {
        if !trace.verified {
            return None;
        }
        let key = (opcode, guards.to_vec());
        let mut next = self.clone();
        let count = next.traces.entry(key.clone()).or_default();
        *count = count.saturating_add(1);
        let compiled = (*count == next.compile_after).then(|| {
            let id = ProgramId(next.programs.len() as u64 + 1);
            let program = Program {
                id,
                inputs: match opcode {
                    OpCode::Identity => vec![ValueType::Atom],
                    OpCode::TransitiveBefore => vec![ValueType::Atom; 3],
                },
                output: ValueType::Atom,
                opcode,
                guards: guards.to_vec(),
                provenance: vec!["verified-reasoning-trace".to_string()],
                status: ProgramStatus::Compiled,
            };
            next.programs.insert(id, program.clone());
            next.stats.insert(id, ProgramStats::default());
            program
        });
        Some((next, compiled))
    }

    pub fn record_outcome(&self, id: ProgramId, outcome: &VmOutcome) -> Option<Self> {
        let mut next = self.clone();
        let stats = next.stats.get_mut(&id)?;
        stats.calls = stats.calls.saturating_add(1);
        if outcome.dispatch == ProgramDispatch::Deoptimized {
            stats.failures = stats.failures.saturating_add(1);
        } else if outcome.dispatch == ProgramDispatch::CompiledHit {
            stats.compute_saved = stats
                .compute_saved
                .saturating_add(outcome.trace.operations.saturating_sub(1));
        }
        Some(next)
    }

    pub fn program(&self, id: ProgramId) -> Option<&Program> {
        self.programs.get(&id)
    }

    pub fn stats(&self, id: ProgramId) -> Option<ProgramStats> {
        self.stats.get(&id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_reasoning_compiles_and_guard_failure_deoptimizes_without_changing_result() {
        let vm = ProgramVm;
        let candidate = Program {
            id: ProgramId(0),
            inputs: vec![ValueType::Atom; 3],
            output: ValueType::Atom,
            opcode: OpCode::TransitiveBefore,
            guards: vec![Guard::ContextRequired("ordered".to_string())],
            provenance: vec!["planner".to_string()],
            status: ProgramStatus::Candidate,
        };
        let inputs = [
            Value::Atom("A".to_string()),
            Value::Atom("B".to_string()),
            Value::Atom("C".to_string()),
        ];
        let interpreted = vm.execute(&candidate, &inputs, &BTreeSet::new()).unwrap();
        let library = ProgramLibrary::new(2).unwrap();
        let (library, first) = library
            .record_verified_trace(candidate.opcode, &candidate.guards, &interpreted.trace)
            .unwrap();
        let (library, compiled) = library
            .record_verified_trace(candidate.opcode, &candidate.guards, &interpreted.trace)
            .unwrap();
        let compiled = compiled.unwrap();
        let deoptimized = vm.execute(&compiled, &inputs, &BTreeSet::new()).unwrap();
        let fast = vm
            .execute(&compiled, &inputs, &BTreeSet::from(["ordered".to_string()]))
            .unwrap();

        assert!(first.is_none());
        assert_eq!(deoptimized.dispatch, ProgramDispatch::Deoptimized);
        assert_eq!(deoptimized.result, interpreted.result);
        assert_eq!(fast.dispatch, ProgramDispatch::CompiledHit);
        assert_eq!(library.program(compiled.id), Some(&compiled));

        let (library, repeated_compile) = library
            .record_verified_trace(candidate.opcode, &candidate.guards, &interpreted.trace)
            .unwrap();
        assert!(repeated_compile.is_none());
        assert_eq!(library.stats(compiled.id), Some(ProgramStats::default()));
    }
}
