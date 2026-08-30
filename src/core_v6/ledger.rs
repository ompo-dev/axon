//! Ledger epistemológico imutável, revisões e conflito paraconsistente.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

use super::factor::ValidityDomain;
use super::ids::{ClaimId, RevisionId};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EpistemicStatus {
    Observed,
    Reported,
    Hypothesis,
    Supported,
    Verified,
    Current,
    DomainLimited,
    Superseded,
    Deprecated,
    Refuted,
    Unknown,
    Conflicted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TruthValue {
    True,
    False,
    Both,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Claim {
    pub id: ClaimId,
    /// Identidade estável da crença, independente de sua formulação/revisão.
    pub key: String,
    pub proposition: String,
    pub status: EpistemicStatus,
    pub confidence_milliunits: u32,
    pub estimated_cost: u64,
    pub evidence: Vec<String>,
    pub derived_from: BTreeSet<ClaimId>,
    pub validity: ValidityDomain,
    pub revision: RevisionId,
}

impl Claim {
    pub fn new(
        id: ClaimId,
        proposition: impl Into<String>,
        status: EpistemicStatus,
        confidence_milliunits: u32,
        revision: RevisionId,
    ) -> Result<Self, LedgerError> {
        let proposition = proposition.into();
        Self::for_key(
            id,
            proposition.clone(),
            proposition,
            status,
            confidence_milliunits,
            revision,
        )
    }

    pub fn for_key(
        id: ClaimId,
        key: impl Into<String>,
        proposition: impl Into<String>,
        status: EpistemicStatus,
        confidence_milliunits: u32,
        revision: RevisionId,
    ) -> Result<Self, LedgerError> {
        if confidence_milliunits > 1_000 {
            return Err(LedgerError::InvalidConfidence);
        }
        Ok(Self {
            id,
            key: key.into(),
            proposition: proposition.into(),
            status,
            confidence_milliunits,
            estimated_cost: 1,
            evidence: Vec::new(),
            derived_from: BTreeSet::new(),
            validity: ValidityDomain::universal(),
            revision,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimView {
    pub claim: Claim,
    pub effective_status: EpistemicStatus,
    pub superseded_by: Option<ClaimId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NegativeKnowledge {
    failed: BTreeMap<String, Vec<String>>,
}

impl NegativeKnowledge {
    pub fn record(&self, strategy: impl Into<String>, constraint: impl Into<String>) -> Self {
        let strategy = strategy.into();
        let mut next = self.clone();
        next.failed
            .entry(strategy)
            .or_default()
            .push(constraint.into());
        next
    }

    pub fn constraints_for(&self, strategy: &str) -> &[String] {
        self.failed.get(strategy).map_or(&[], Vec::as_slice)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EpistemicLedger {
    claims: BTreeMap<ClaimId, Claim>,
    proposition_history: BTreeMap<String, Vec<ClaimId>>,
    supersedes: BTreeMap<ClaimId, ClaimId>,
    conflicts: BTreeSet<(ClaimId, ClaimId)>,
    negative: NegativeKnowledge,
}

impl EpistemicLedger {
    pub fn add(&self, claim: Claim) -> Result<Self, LedgerError> {
        if self.claims.contains_key(&claim.id) {
            return Err(LedgerError::DuplicateClaim(claim.id));
        }
        let mut next = self.clone();
        next.proposition_history
            .entry(claim.key.clone())
            .or_default()
            .push(claim.id);
        next.claims.insert(claim.id, claim);
        Ok(next)
    }

    /// Liga revisões sem sobrescrever nem mutar o objeto histórico anterior.
    pub fn supersede(&self, old: ClaimId, new: Claim) -> Result<Self, LedgerError> {
        if !self.claims.contains_key(&old) {
            return Err(LedgerError::UnknownClaim(old));
        }
        let old_claim = self.claims.get(&old).expect("checked above");
        if old_claim.key != new.key {
            return Err(LedgerError::PropositionMismatch);
        }
        let mut next = self.add(new.clone())?;
        next.supersedes.insert(new.id, old);
        Ok(next)
    }

    pub fn claim(&self, id: ClaimId) -> Option<&Claim> {
        self.claims.get(&id)
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn view(&self, id: ClaimId) -> Option<ClaimView> {
        let claim = self.claims.get(&id)?.clone();
        let superseded_by = self
            .supersedes
            .iter()
            .find_map(|(new, old)| (*old == id).then_some(*new));
        let effective_status = if superseded_by.is_some() {
            EpistemicStatus::Superseded
        } else if self
            .conflicts
            .iter()
            .any(|(left, right)| *left == id || *right == id)
        {
            EpistemicStatus::Conflicted
        } else {
            claim.status
        };
        Some(ClaimView {
            claim,
            effective_status,
            superseded_by,
        })
    }

    pub fn current(&self, proposition: &str) -> Option<ClaimView> {
        self.proposition_history
            .get(proposition)?
            .iter()
            .rev()
            .find_map(|id| {
                let view = self.view(*id)?;
                (view.effective_status != EpistemicStatus::Superseded).then_some(view)
            })
    }

    pub fn history(&self, proposition: &str) -> Vec<ClaimView> {
        self.proposition_history
            .get(proposition)
            .into_iter()
            .flatten()
            .filter_map(|id| self.view(*id))
            .collect()
    }

    /// Escolhe a crença não supersedida compatível com o contexto pelo domínio
    /// mais específico e, em empate, pelo menor custo estimado.
    pub fn best_valid(
        &self,
        key: &str,
        context: &BTreeSet<String>,
        timestamp: u64,
    ) -> Option<ClaimView> {
        self.history(key)
            .into_iter()
            .filter(|view| view.effective_status != EpistemicStatus::Superseded)
            .filter(|view| view.claim.validity.applies_to(context, timestamp))
            .max_by(|left, right| {
                left.claim
                    .validity
                    .conditions
                    .len()
                    .cmp(&right.claim.validity.conditions.len())
                    .then_with(|| right.claim.estimated_cost.cmp(&left.claim.estimated_cost))
                    .then_with(|| {
                        left.claim
                            .confidence_milliunits
                            .cmp(&right.claim.confidence_milliunits)
                    })
                    .then_with(|| right.claim.id.cmp(&left.claim.id))
            })
    }

    pub fn conflict(&self, left: ClaimId, right: ClaimId) -> Result<Self, LedgerError> {
        if !self.claims.contains_key(&left) {
            return Err(LedgerError::UnknownClaim(left));
        }
        if !self.claims.contains_key(&right) {
            return Err(LedgerError::UnknownClaim(right));
        }
        let mut next = self.clone();
        next.conflicts.insert(ordered_pair(left, right));
        Ok(next)
    }

    pub fn truth_value(&self, positive: ClaimId, negative: Option<ClaimId>) -> TruthValue {
        match (self.claims.contains_key(&positive), negative) {
            (false, _) => TruthValue::Unknown,
            (true, Some(negative))
                if self.conflicts.contains(&ordered_pair(positive, negative)) =>
            {
                TruthValue::Both
            }
            (true, _) => TruthValue::True,
        }
    }

    pub fn record_failure(
        &self,
        strategy: impl Into<String>,
        constraint: impl Into<String>,
    ) -> Self {
        Self {
            negative: self.negative.record(strategy, constraint),
            ..self.clone()
        }
    }

    pub fn failed_constraints(&self, strategy: &str) -> &[String] {
        self.negative.constraints_for(strategy)
    }
}

fn ordered_pair(left: ClaimId, right: ClaimId) -> (ClaimId, ClaimId) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LedgerError {
    InvalidConfidence,
    DuplicateClaim(ClaimId),
    UnknownClaim(ClaimId),
    PropositionMismatch,
}

impl Display for LedgerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfidence => write!(f, "claim confidence must be in 0..=1000"),
            Self::DuplicateClaim(id) => write!(f, "claim id {} already exists", id.0),
            Self::UnknownClaim(id) => write!(f, "claim id {} does not exist", id.0),
            Self::PropositionMismatch => write!(
                f,
                "only a revision of the same claim key can supersede a claim"
            ),
        }
    }
}

impl Error for LedgerError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(id: u64, proposition: &str, status: EpistemicStatus) -> Claim {
        Claim::new(ClaimId(id), proposition, status, 900, RevisionId(id)).unwrap()
    }

    #[test]
    fn supersession_preserves_immutable_history_and_exposes_the_current_revision() {
        let old = Claim::for_key(
            ClaimId(1),
            "capital(Freland)",
            "capital(Freland)=Noma",
            EpistemicStatus::Current,
            900,
            RevisionId(1),
        )
        .unwrap();
        let new = Claim::for_key(
            ClaimId(2),
            "capital(Freland)",
            "capital(Freland)=Zora",
            EpistemicStatus::Current,
            900,
            RevisionId(2),
        )
        .unwrap();
        let ledger = EpistemicLedger::default()
            .add(old.clone())
            .unwrap()
            .supersede(old.id, new.clone())
            .unwrap();

        assert_eq!(ledger.claim(old.id), Some(&old));
        assert_eq!(
            ledger.view(old.id).unwrap().effective_status,
            EpistemicStatus::Superseded
        );
        assert_eq!(ledger.current(&new.key).unwrap().claim.id, new.id);
        assert_eq!(ledger.history(&new.key).len(), 2);
    }

    #[test]
    fn contradiction_is_local_and_does_not_explode_the_ledger() {
        let positive = claim(1, "rain", EpistemicStatus::Observed);
        let negative = claim(2, "not-rain", EpistemicStatus::Reported);
        let ledger = EpistemicLedger::default()
            .add(positive.clone())
            .unwrap()
            .add(negative.clone())
            .unwrap()
            .conflict(positive.id, negative.id)
            .unwrap();

        assert_eq!(
            ledger.truth_value(positive.id, Some(negative.id)),
            TruthValue::Both
        );
        assert_eq!(
            ledger.view(positive.id).unwrap().effective_status,
            EpistemicStatus::Conflicted
        );
        assert_eq!(
            ledger
                .record_failure("a-b-c", "counterexample-x")
                .failed_constraints("a-b-c"),
            ["counterexample-x"]
        );
    }

    #[test]
    fn domain_limited_knowledge_selects_the_cheapest_specific_valid_model() {
        let mut cheap = Claim::for_key(
            ClaimId(1),
            "trajectory-model",
            "newtonian",
            EpistemicStatus::DomainLimited,
            900,
            RevisionId(1),
        )
        .unwrap();
        cheap.validity.conditions.insert("weak-field".to_string());
        cheap.estimated_cost = 1;
        let mut broad = Claim::for_key(
            ClaimId(2),
            "trajectory-model",
            "relativistic",
            EpistemicStatus::Current,
            950,
            RevisionId(2),
        )
        .unwrap();
        broad.estimated_cost = 10;
        let ledger = EpistemicLedger::default()
            .add(cheap)
            .unwrap()
            .add(broad)
            .unwrap();

        assert_eq!(
            ledger
                .best_valid(
                    "trajectory-model",
                    &BTreeSet::from(["weak-field".to_string()]),
                    0
                )
                .unwrap()
                .claim
                .proposition,
            "newtonian"
        );
        assert_eq!(
            ledger
                .best_valid("trajectory-model", &BTreeSet::new(), 0)
                .unwrap()
                .claim
                .proposition,
            "relativistic"
        );
    }
}
