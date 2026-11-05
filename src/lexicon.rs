//! Sense-aware dictionary views over the existing sourced, temporal episode journal.
use crate::{
    BeliefResult, BeliefStatus, EpisodeData, EpisodeError, EpisodeStore, EpisodeValue, FactKey,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const PREFIX: &str = "lexicon/";
pub const MAX_FACTS: usize = 250_000;
pub const MAX_VISITS: usize = 4096;
pub const MAX_EPISODES: usize = 500_000;
pub fn store_limits() -> crate::EpisodeLimits {
    crate::EpisodeLimits {
        max_file_bytes: 268_435_456,
        max_episodes: MAX_EPISODES,
        ..crate::EpisodeLimits::default()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Relation {
    Word,
    Definition,
    IsA,
    NotA,
}

/// Synonyms denote the same explicit sense; shared spelling does not merge senses.
pub fn fact(
    context: &str,
    subject: &str,
    relation: Relation,
    object: &str,
) -> Result<(FactKey, EpisodeValue), EpisodeError> {
    identifier(context)?;
    identifier(subject)?;
    let (predicate, value) = match relation {
        Relation::Word => {
            identifier(object)?;
            (
                format!("word/{}", normalize(object)),
                EpisodeValue::Boolean(true),
            )
        }
        Relation::Definition => {
            if object.is_empty() || object.len() > 4096 || object.chars().any(char::is_control) {
                return Err(EpisodeError::Invalid("definition"));
            }
            ("definition".into(), EpisodeValue::Text(object.into()))
        }
        Relation::IsA | Relation::NotA => {
            identifier(object)?;
            (
                format!("is-a/{object}"),
                EpisodeValue::Boolean(relation == Relation::IsA),
            )
        }
    };
    Ok((
        FactKey::new(subject, predicate, format!("{PREFIX}{context}")),
        value,
    ))
}
fn identifier(s: &str) -> Result<(), EpisodeError> {
    if s.is_empty() || s.len() > 128 || s.trim() != s || s.chars().any(char::is_control) {
        return Err(EpisodeError::Invalid("lexicon identifier"));
    }
    Ok(())
}
fn normalize(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Resolution {
    Unique(String),
    Ambiguous(Vec<String>),
    Conflict,
    Unknown,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Entailment {
    Supported(Vec<u64>),
    Refuted(Vec<u64>),
    Conflict,
    Unknown,
    Budget,
}
/// Immutable temporal snapshot. Definitions are evidence, never executable instructions.
pub struct LexiconView {
    words: BTreeMap<String, BTreeMap<String, BeliefResult>>,
    edges: BTreeMap<String, BTreeMap<String, BeliefResult>>,
    definitions: BTreeMap<String, BeliefResult>,
    senses: BTreeSet<String>,
    pub facts: usize,
    pub known_at: u64,
}
impl LexiconView {
    pub fn snapshot(
        store: &EpisodeStore,
        context: &str,
        at: Option<u64>,
        known_at: Option<u64>,
    ) -> Result<Self, EpisodeError> {
        identifier(context)?;
        let known_at = known_at.unwrap_or(store.len() as u64);
        if known_at > store.len() as u64 || known_at > MAX_EPISODES as u64 {
            return Err(EpisodeError::Invalid("dictionary journal limit"));
        }
        let context = format!("{PREFIX}{context}");
        let mut keys = BTreeSet::new();
        for id in 1..=known_at {
            if let EpisodeData::Fact { key, .. } = store.episode(id)?.data()
                && key.context == context
            {
                keys.insert(key.clone());
                if keys.len() > MAX_FACTS {
                    return Err(EpisodeError::Invalid("dictionary fact limit"));
                }
            }
        }
        let mut view = Self {
            words: BTreeMap::new(),
            edges: BTreeMap::new(),
            definitions: BTreeMap::new(),
            senses: BTreeSet::new(),
            facts: keys.len(),
            known_at,
        };
        for key in keys {
            let belief = store.belief(&key, at, Some(known_at))?;
            if belief.status == BeliefStatus::Missing {
                continue;
            }
            identifier(&key.subject)?;
            view.senses.insert(key.subject.clone());
            if let Some(word) = key.predicate.strip_prefix("word/") {
                identifier(word)?;
                boolean(&belief)?;
                view.words
                    .entry(normalize(word))
                    .or_default()
                    .insert(key.subject, belief);
            } else if let Some(parent) = key.predicate.strip_prefix("is-a/") {
                identifier(parent)?;
                boolean(&belief)?;
                view.senses.insert(parent.into());
                view.edges
                    .entry(key.subject)
                    .or_default()
                    .insert(parent.into(), belief);
            } else if key.predicate == "definition" {
                if belief.evidence.iter().any(|e| !matches!(&e.value, EpisodeValue::Text(t) if !t.is_empty() && t.len() <= 4096 && !t.chars().any(char::is_control))) { return Err(EpisodeError::Invalid("dictionary definition type")); }
                view.definitions.insert(key.subject, belief);
            } else {
                return Err(EpisodeError::Invalid("dictionary predicate"));
            }
        }
        Ok(view)
    }
    pub fn resolve(&self, word: &str) -> Resolution {
        let Some(senses) = self.words.get(&normalize(word)) else {
            return Resolution::Unknown;
        };
        if senses.values().any(|b| b.status == BeliefStatus::Conflict) {
            return Resolution::Conflict;
        }
        let active = senses
            .iter()
            .filter(|(_, b)| positive(b))
            .map(|(s, _)| s.clone())
            .collect::<Vec<_>>();
        match active.len() {
            0 => Resolution::Unknown,
            1 => Resolution::Unique(active[0].clone()),
            _ => Resolution::Ambiguous(active),
        }
    }
    pub fn definition(&self, sense: &str) -> Option<&BeliefResult> {
        self.definitions.get(sense)
    }
    pub fn word_evidence(&self, word: &str, sense: &str) -> Vec<u64> {
        self.words
            .get(&normalize(word))
            .and_then(|s| s.get(sense))
            .map_or_else(Vec::new, ids)
    }
    /// Open-world, bounded ISA traversal. A cycle cannot prove an unrelated claim.
    pub fn classify(&self, sense: &str, parent: &str, limit: usize) -> Entailment {
        if limit == 0 || limit > MAX_VISITS {
            return Entailment::Budget;
        }
        if !self.senses.contains(sense) || !self.senses.contains(parent) {
            return Entailment::Unknown;
        }
        let direct = self.edges.get(sense).and_then(|e| e.get(parent));
        if direct.is_some_and(|b| b.status == BeliefStatus::Conflict) {
            return Entailment::Conflict;
        }
        let negative = direct.filter(|b| !positive(b)).map(ids);
        let mut queue = VecDeque::from([sense.to_owned()]);
        let mut predecessors = BTreeMap::from([(sense.to_owned(), None)]);
        let mut blocked = false;
        while let Some(node) = queue.pop_front() {
            if node == parent {
                let mut proof = Vec::new();
                let mut cursor = &node;
                while let Some((previous, evidence)) = &predecessors[cursor] {
                    proof.extend(evidence);
                    cursor = previous;
                }
                proof.sort_unstable();
                proof.dedup();
                return if negative.is_some() {
                    Entailment::Conflict
                } else {
                    Entailment::Supported(proof)
                };
            }
            for (next, b) in self.edges.get(&node).into_iter().flatten() {
                if b.status == BeliefStatus::Conflict {
                    blocked = true;
                    continue;
                }
                if positive(b) && !predecessors.contains_key(next) {
                    if predecessors.len() == limit {
                        return Entailment::Budget;
                    }
                    predecessors.insert(next.clone(), Some((node.clone(), ids(b))));
                    queue.push_back(next.clone());
                }
            }
        }
        if blocked {
            Entailment::Conflict
        } else if let Some(proof) = negative {
            Entailment::Refuted(proof)
        } else {
            Entailment::Unknown
        }
    }
}
fn boolean(b: &BeliefResult) -> Result<(), EpisodeError> {
    if b.evidence
        .iter()
        .any(|e| !matches!(e.value, EpisodeValue::Boolean(_)))
    {
        return Err(EpisodeError::Invalid("dictionary relation type"));
    }
    Ok(())
}
fn positive(b: &BeliefResult) -> bool {
    b.status == BeliefStatus::Consistent
        && b.evidence
            .first()
            .is_some_and(|e| e.value == EpisodeValue::Boolean(true))
}
fn ids(b: &BeliefResult) -> Vec<u64> {
    b.evidence.iter().map(|e| e.episode_id).collect()
}
