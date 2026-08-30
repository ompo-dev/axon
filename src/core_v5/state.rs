//! Fronteiras explícitas entre os substratos cognitivos da V5/Ω.

use std::collections::{BTreeMap, BTreeSet};

use super::cost::CostVector;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CognitiveSubstrate {
    SemanticMesh,
    DynamicWorld,
    EpisodicMemory,
    ProgramFabric,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CognitiveEvent {
    pub id: u64,
    pub parents: BTreeSet<u64>,
    pub reads: BTreeSet<CognitiveSubstrate>,
    pub writes: BTreeSet<CognitiveSubstrate>,
    pub payload_ref: String,
    pub cost: CostVector,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MultiSubstrateState {
    cells: BTreeMap<CognitiveSubstrate, BTreeMap<String, String>>,
}

impl MultiSubstrateState {
    pub fn apply(&self, event: &CognitiveEvent) -> Self {
        let mut next = self.clone();
        for substrate in &event.writes {
            next.cells
                .entry(*substrate)
                .or_default()
                .insert(event.payload_ref.clone(), event.id.to_string());
        }
        next
    }

    pub fn contains(&self, substrate: CognitiveSubstrate, key: &str) -> bool {
        self.cells
            .get(&substrate)
            .is_some_and(|cells| cells.contains_key(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_v5::CostVector;

    #[test]
    fn event_mutates_only_its_declared_substrates() {
        let event = CognitiveEvent {
            id: 1,
            parents: BTreeSet::new(),
            reads: BTreeSet::from([CognitiveSubstrate::SemanticMesh]),
            writes: BTreeSet::from([CognitiveSubstrate::ProgramFabric]),
            payload_ref: "repeat-A-B".to_string(),
            cost: CostVector::declared(1, 8, 8, 0, 1),
        };
        let state = MultiSubstrateState::default().apply(&event);

        assert!(state.contains(CognitiveSubstrate::ProgramFabric, "repeat-A-B"));
        assert!(!state.contains(CognitiveSubstrate::SemanticMesh, "repeat-A-B"));
    }
}
