//! Estado explícito de mundo e alterações reversíveis com proveniência.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Relation {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ObjectWorld {
    pub objects: BTreeSet<String>,
    pub relations: BTreeSet<Relation>,
    pub assumptions: BTreeSet<String>,
}

impl ObjectWorld {
    pub fn with_relation(mut self, relation: Relation) -> Self {
        self.objects.insert(relation.subject.clone());
        self.objects.insert(relation.object.clone());
        self.relations.insert(relation);
        self
    }

    pub fn with_assumption(mut self, assumption: impl Into<String>) -> Self {
        self.assumptions.insert(assumption.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuralOperator {
    AddLatentVariable(String),
    RemoveAssumption(String),
    InvertCausality,
    ChangeCoordinates(String),
    ChangeScale(String),
    AddDimension(String),
    RemoveDimension(String),
    RedefineObject(String),
    CreateOperator(String),
    BreakSymmetry(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldMutation {
    pub operator: StructuralOperator,
    pub before: ObjectWorld,
    pub after: ObjectWorld,
    pub provenance: String,
}

impl WorldMutation {
    pub fn apply(&self) -> ObjectWorld {
        self.after.clone()
    }

    pub fn undo(&self) -> ObjectWorld {
        self.before.clone()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReversibleState {
    values: BTreeMap<String, String>,
    journal: Vec<StateMutation>,
    undo_stack: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateMutation {
    pub key: String,
    pub before: Option<String>,
    pub after: String,
    pub provenance: String,
}

impl ReversibleState {
    pub fn with_value(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.values.insert(key.into(), value.into());
        self
    }

    pub fn apply(
        &self,
        key: impl Into<String>,
        after: impl Into<String>,
        provenance: impl Into<String>,
    ) -> Self {
        let key = key.into();
        let after = after.into();
        let mut next = self.clone();
        next.journal.push(StateMutation {
            before: next.values.get(&key).cloned(),
            key: key.clone(),
            after: after.clone(),
            provenance: provenance.into(),
        });
        next.undo_stack.push(next.journal.len() - 1);
        next.values.insert(key, after);
        next
    }

    pub fn undo_last(&self) -> Option<Self> {
        let mutation_index = *self.undo_stack.last()?;
        let mutation = self.journal.get(mutation_index)?.clone();
        let mut previous = self.clone();
        previous.undo_stack.pop();
        match mutation.before {
            Some(value) => {
                previous.values.insert(mutation.key, value);
            }
            None => {
                previous.values.remove(&mutation.key);
            }
        }
        Some(previous)
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    pub fn journal_len(&self) -> usize {
        self.journal.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_undo_restores_an_object_world_exactly() {
        let before = ObjectWorld::default()
            .with_relation(Relation::new("A", "causes", "B"))
            .with_assumption("A-causes-B");
        let after = before
            .clone()
            .with_relation(Relation::new("C", "explains", "A"));
        let mutation = WorldMutation {
            operator: StructuralOperator::AddLatentVariable("C".to_string()),
            before: before.clone(),
            after,
            provenance: "counterfactual-failure".to_string(),
        };

        assert_eq!(mutation.undo(), before);
        assert_eq!(mutation.apply().relations.len(), 2);
    }

    #[test]
    fn reversible_state_restores_the_prior_value_without_destroying_provenance() {
        let state = ReversibleState::default().with_value("rule", "old");
        let changed = state.apply("rule", "new", "verification-failed");

        assert_eq!(changed.get("rule"), Some("new"));
        assert_eq!(changed.journal_len(), 1);
        let restored = changed.undo_last().unwrap();
        assert_eq!(restored.get("rule"), state.get("rule"));
        assert_eq!(restored.journal_len(), 1);
        assert!(restored.undo_last().is_none());
    }
}
