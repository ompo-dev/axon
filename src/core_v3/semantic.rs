use super::vector::HyperVector;

/// A compositional fact, stored separately from temporal dynamics and raw episodes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub signature: HyperVector,
}

impl SemanticFact {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        signature: HyperVector,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            signature,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SemanticMesh {
    facts: Vec<SemanticFact>,
}

impl SemanticMesh {
    pub fn bind(&self, fact: SemanticFact) -> Self {
        let mut facts = self.facts.clone();
        if !facts.contains(&fact) {
            facts.push(fact);
        }
        Self { facts }
    }

    pub fn facts_for(&self, subject: &str) -> Vec<&SemanticFact> {
        self.facts
            .iter()
            .filter(|fact| fact.subject == subject)
            .collect()
    }
}
