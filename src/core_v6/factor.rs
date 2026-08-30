//! Factor Fabric: grafo tipado com representação, validade e estado local.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use super::ids::{FactorId, RevisionId};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FactorKind {
    Semantic,
    Relation,
    Dynamic,
    Probabilistic,
    Program,
    Constraint,
    Memory,
    Associative,
    Goal,
    Neural,
    Physical,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RepresentationKind {
    Binary,
    Ternary,
    HyperVector,
    Graph,
    Symbolic,
    Probabilistic,
    Program,
    Continuous,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateRef(pub String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidityDomain {
    pub conditions: BTreeSet<String>,
    pub valid_from: Option<u64>,
    pub valid_until: Option<u64>,
}

impl ValidityDomain {
    pub fn universal() -> Self {
        Self {
            conditions: BTreeSet::new(),
            valid_from: None,
            valid_until: None,
        }
    }

    pub fn applies_to(&self, context: &BTreeSet<String>, timestamp: u64) -> bool {
        self.conditions.is_subset(context)
            && self.valid_from.is_none_or(|start| timestamp >= start)
            && self.valid_until.is_none_or(|end| timestamp <= end)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalityHint {
    pub region: u32,
    pub hot: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearningMeta {
    pub plasticity: f32,
    pub stability: f32,
    pub fast_weight: f32,
    pub slow_weight: f32,
    pub evidence_count: u32,
    pub last_update: u64,
}

impl Default for LearningMeta {
    fn default() -> Self {
        Self {
            plasticity: 1.0,
            stability: 0.0,
            fast_weight: 0.0,
            slow_weight: 0.0,
            evidence_count: 0,
            last_update: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Factor {
    pub id: FactorId,
    pub kind: FactorKind,
    pub inputs: BTreeSet<FactorId>,
    pub outputs: BTreeSet<FactorId>,
    pub representation: RepresentationKind,
    pub state: StateRef,
    pub uncertainty: f32,
    pub validity: ValidityDomain,
    pub provenance: Vec<RevisionId>,
    pub learning: LearningMeta,
    pub locality: LocalityHint,
    pub revision: RevisionId,
}

impl Factor {
    pub fn try_new(
        id: FactorId,
        kind: FactorKind,
        representation: RepresentationKind,
        state: impl Into<String>,
        uncertainty: f32,
        revision: RevisionId,
    ) -> Result<Self, FactorError> {
        if !uncertainty.is_finite() || !(0.0..=1.0).contains(&uncertainty) {
            return Err(FactorError::InvalidUncertainty);
        }
        Ok(Self {
            id,
            kind,
            inputs: BTreeSet::new(),
            outputs: BTreeSet::new(),
            representation,
            state: StateRef(state.into()),
            uncertainty,
            validity: ValidityDomain::universal(),
            provenance: vec![revision],
            learning: LearningMeta::default(),
            locality: LocalityHint {
                region: 0,
                hot: false,
            },
            revision,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RelationKind {
    DependsOn,
    Causes,
    Supports,
    Contradicts,
    Retrieves,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FactorEdge {
    pub from: FactorId,
    pub to: FactorId,
    pub relation: RelationKind,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FactorGraph {
    factors: BTreeMap<FactorId, Factor>,
    edges: BTreeSet<FactorEdge>,
}

impl FactorGraph {
    pub fn insert(&self, factor: Factor) -> Result<Self, FactorError> {
        if self.factors.contains_key(&factor.id) {
            return Err(FactorError::DuplicateId(factor.id));
        }
        let mut next = self.clone();
        next.factors.insert(factor.id, factor);
        Ok(next)
    }

    pub fn connect(&self, edge: FactorEdge) -> Result<Self, FactorError> {
        if !self.factors.contains_key(&edge.from) || !self.factors.contains_key(&edge.to) {
            return Err(FactorError::UnknownFactor);
        }
        let mut next = self.clone();
        next.edges.insert(edge);
        if let Some(from) = next.factors.get_mut(&edge.from) {
            from.outputs.insert(edge.to);
        }
        if let Some(to) = next.factors.get_mut(&edge.to) {
            to.inputs.insert(edge.from);
        }
        Ok(next)
    }

    pub fn factor(&self, id: FactorId) -> Option<&Factor> {
        self.factors.get(&id)
    }

    pub fn len(&self) -> usize {
        self.factors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FactorError {
    InvalidUncertainty,
    DuplicateId(FactorId),
    UnknownFactor,
}

impl Display for FactorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUncertainty => write!(f, "factor uncertainty must be finite in 0..=1"),
            Self::DuplicateId(id) => write!(f, "factor id {} already exists", id.0),
            Self::UnknownFactor => write!(f, "all factor edge endpoints must exist"),
        }
    }
}

impl Error for FactorError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn factor(id: u64) -> Factor {
        Factor::try_new(
            FactorId(id),
            FactorKind::Semantic,
            RepresentationKind::Graph,
            format!("state-{id}"),
            0.1,
            RevisionId(1),
        )
        .unwrap()
    }

    #[test]
    fn factor_ids_are_stable_and_edges_only_touch_declared_factors() {
        let graph = FactorGraph::default()
            .insert(factor(1))
            .unwrap()
            .insert(factor(2))
            .unwrap()
            .connect(FactorEdge {
                from: FactorId(1),
                to: FactorId(2),
                relation: RelationKind::Causes,
            })
            .unwrap();

        assert_eq!(graph.factor(FactorId(1)).unwrap().id, FactorId(1));
        assert!(
            graph
                .factor(FactorId(1))
                .unwrap()
                .outputs
                .contains(&FactorId(2))
        );
        assert_eq!(
            graph.connect(FactorEdge {
                from: FactorId(1),
                to: FactorId(9),
                relation: RelationKind::Causes,
            }),
            Err(FactorError::UnknownFactor)
        );
    }
}
