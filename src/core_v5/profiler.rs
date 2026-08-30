//! Thought profiler, JIT de trajetórias verificadas e ISA cognitiva extensível.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CognitiveOpcode {
    Bind,
    Compare,
    Predict,
    Route,
    Cause,
    Query,
    Search,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CognitiveMacro {
    pub id: String,
    pub trace: Vec<CognitiveOpcode>,
    pub guards: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MacroDispatch {
    FastPath { macro_id: String },
    Deoptimized { macro_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThoughtProfiler {
    compile_after: u32,
    observed: BTreeMap<(Vec<CognitiveOpcode>, BTreeSet<String>), u32>,
    macros: BTreeMap<(Vec<CognitiveOpcode>, BTreeSet<String>), CognitiveMacro>,
}

impl ThoughtProfiler {
    pub fn new(compile_after: u32) -> Option<Self> {
        (compile_after > 0).then(|| Self {
            compile_after,
            observed: BTreeMap::new(),
            macros: BTreeMap::new(),
        })
    }

    pub fn record_verified_trace(
        &self,
        trace: &[CognitiveOpcode],
        guards: &BTreeSet<String>,
    ) -> Option<(Self, Option<CognitiveMacro>)> {
        if trace.is_empty() {
            return None;
        }
        let key = (trace.to_vec(), guards.clone());
        if let Some(existing) = self.macros.get(&key) {
            return Some((self.clone(), Some(existing.clone())));
        }
        let mut next = self.clone();
        let count = next.observed.entry(key.clone()).or_default();
        *count = count.saturating_add(1);
        let compiled = (*count >= next.compile_after).then(|| {
            let macro_ = CognitiveMacro {
                id: format!("NEW_OP_{}", next.macros.len()),
                trace: key.0.clone(),
                guards: key.1.clone(),
            };
            next.macros.insert(key, macro_.clone());
            macro_
        });
        Some((next, compiled))
    }

    pub fn dispatch(&self, macro_: &CognitiveMacro, context: &BTreeSet<String>) -> MacroDispatch {
        if macro_.guards.is_subset(context) {
            MacroDispatch::FastPath {
                macro_id: macro_.id.clone(),
            }
        } else {
            MacroDispatch::Deoptimized {
                macro_id: macro_.id.clone(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_repetition_extends_isa_and_guard_failure_deoptimizes() {
        let profiler = ThoughtProfiler::new(2).unwrap();
        let trace = [CognitiveOpcode::Bind, CognitiveOpcode::Compare];
        let guards = BTreeSet::from(["dense-key".to_string()]);
        let (profiler, first) = profiler.record_verified_trace(&trace, &guards).unwrap();
        let (profiler, second) = profiler.record_verified_trace(&trace, &guards).unwrap();
        let macro_ = second.expect("two verified runs compile a cognitive macro");

        assert!(first.is_none());
        assert!(matches!(
            profiler.dispatch(&macro_, &BTreeSet::new()),
            MacroDispatch::Deoptimized { .. }
        ));
    }
}
