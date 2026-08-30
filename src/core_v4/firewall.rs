#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum KnowledgeTier {
    New,
    Fast,
    Stable,
    Consolidated,
    Protected,
}

/// A self-contained reversible description. Integration with a concrete store
/// will apply `before`/`after`; the journal never destroys either value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KnowledgeMutation {
    pub id: u64,
    pub subject: String,
    pub before: String,
    pub after: String,
}

impl KnowledgeMutation {
    pub fn new(
        subject: impl Into<String>,
        before: impl Into<String>,
        after: impl Into<String>,
    ) -> Self {
        let subject = subject.into();
        let before = before.into();
        let after = after.into();
        Self {
            id: mutation_id(&subject, &before, &after),
            subject,
            before,
            after,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FirewallDecision {
    Apply { mutation: KnowledgeMutation },
    ForkCandidate { mutation: KnowledgeMutation },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemoryFirewall;

impl MemoryFirewall {
    /// Protected knowledge never receives an in-place update from one observation.
    pub fn propose(&self, tier: KnowledgeTier, mutation: KnowledgeMutation) -> FirewallDecision {
        if tier == KnowledgeTier::Protected {
            FirewallDecision::ForkCandidate { mutation }
        } else {
            FirewallDecision::Apply { mutation }
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReversibleJournal {
    entries: Vec<KnowledgeMutation>,
}

impl ReversibleJournal {
    pub fn append(&self, mutation: KnowledgeMutation) -> Self {
        let mut entries = self.entries.clone();
        entries.push(mutation);
        Self { entries }
    }

    /// Returns the complete mutation payload required by a concrete store to restore `before`.
    pub fn rollback(&self, mutation_id: u64) -> Option<KnowledgeMutation> {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.id == mutation_id)
            .cloned()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn mutation_id(subject: &str, before: &str, after: &str) -> u64 {
    [subject, before, after]
        .into_iter()
        .flat_map(str::bytes)
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}
