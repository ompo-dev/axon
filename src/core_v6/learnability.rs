//! Decide entre resolver, adaptar, buscar evidência ou reenquadrar.

use std::collections::{BTreeMap, BTreeSet};

use crate::core_v5::{CostVector, CostWeights};

use super::ids::FactorId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnabilityEvidence {
    pub residual: f32,
    pub uncertainty: f32,
    pub compatible_worlds: u32,
    pub discriminating_evidence: f32,
    pub residual_persistence: f32,
    pub expected_adapt_gain: f32,
}

impl LearnabilityEvidence {
    pub fn valid(self) -> bool {
        [
            self.residual,
            self.uncertainty,
            self.discriminating_evidence,
            self.residual_persistence,
            self.expected_adapt_gain,
        ]
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Experiment {
    pub id: String,
    /// Previsões das hipóteses concorrentes; nunca contém o mundo verdadeiro.
    pub predicted_outcomes: BTreeMap<String, bool>,
    pub cost: CostVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InformationRequest {
    pub candidates: Vec<Experiment>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReframeRequest {
    pub target_region: BTreeSet<FactorId>,
    pub residual_signature: String,
    pub max_operations: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LearnabilityDecision {
    Solve,
    Adapt,
    NeedInformation(InformationRequest),
    Reframe(ReframeRequest),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LearnabilityGate;

impl LearnabilityGate {
    pub fn decide(
        &self,
        evidence: LearnabilityEvidence,
        information: InformationRequest,
        reframe: ReframeRequest,
    ) -> Option<LearnabilityDecision> {
        if !evidence.valid() {
            return None;
        }
        if evidence.residual <= 0.10 {
            return Some(LearnabilityDecision::Solve);
        }
        if evidence.expected_adapt_gain >= 0.50 && evidence.residual_persistence < 0.70 {
            return Some(LearnabilityDecision::Adapt);
        }
        if evidence.uncertainty >= 0.70
            && evidence.compatible_worlds >= 2
            && evidence.discriminating_evidence <= 0.20
        {
            return Some(LearnabilityDecision::NeedInformation(information));
        }
        if evidence.residual_persistence >= 0.70
            && evidence.uncertainty <= 0.40
            && evidence.expected_adapt_gain <= 0.20
        {
            return Some(LearnabilityDecision::Reframe(reframe));
        }
        Some(LearnabilityDecision::Solve)
    }

    pub fn choose_experiment<'a>(&self, request: &'a InformationRequest) -> Option<&'a Experiment> {
        let origin = request.candidates.first()?.cost.origin;
        if request
            .candidates
            .iter()
            .any(|candidate| candidate.cost.origin != origin)
        {
            return None;
        }
        request
            .candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .predicted_outcomes
                    .values()
                    .collect::<BTreeSet<_>>()
                    .len()
                    > 1
            })
            .min_by(|left, right| {
                let left_cost = left.cost.weighted_total(CostWeights::default());
                let right_cost = right.cost.weighted_total(CostWeights::default());
                left_cost
                    .total_cmp(&right_cost)
                    .then_with(|| left.id.cmp(&right.id))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> InformationRequest {
        InformationRequest {
            candidates: vec![
                Experiment {
                    id: "correlation".to_string(),
                    predicted_outcomes: BTreeMap::from([
                        ("direct".to_string(), true),
                        ("reverse".to_string(), true),
                    ]),
                    cost: CostVector::declared(1, 1, 0, 0, 1),
                },
                Experiment {
                    id: "intervene".to_string(),
                    predicted_outcomes: BTreeMap::from([
                        ("direct".to_string(), true),
                        ("reverse".to_string(), false),
                    ]),
                    cost: CostVector::declared(1, 1, 0, 0, 1),
                },
            ],
        }
    }

    fn reframe() -> ReframeRequest {
        ReframeRequest {
            target_region: BTreeSet::from([FactorId(1)]),
            residual_signature: "structured".to_string(),
            max_operations: 10,
        }
    }

    #[test]
    fn gate_distinguishes_solve_adapt_seek_and_reframe() {
        let gate = LearnabilityGate;
        let decide = |evidence| gate.decide(evidence, request(), reframe()).unwrap();

        assert_eq!(
            decide(LearnabilityEvidence {
                residual: 0.05,
                uncertainty: 0.0,
                compatible_worlds: 1,
                discriminating_evidence: 1.0,
                residual_persistence: 0.0,
                expected_adapt_gain: 0.0
            }),
            LearnabilityDecision::Solve
        );
        assert_eq!(
            decide(LearnabilityEvidence {
                residual: 0.4,
                uncertainty: 0.2,
                compatible_worlds: 1,
                discriminating_evidence: 1.0,
                residual_persistence: 0.2,
                expected_adapt_gain: 0.8
            }),
            LearnabilityDecision::Adapt
        );
        assert!(matches!(
            decide(LearnabilityEvidence {
                residual: 0.6,
                uncertainty: 0.9,
                compatible_worlds: 3,
                discriminating_evidence: 0.1,
                residual_persistence: 0.3,
                expected_adapt_gain: 0.1
            }),
            LearnabilityDecision::NeedInformation(_)
        ));
        assert!(matches!(
            decide(LearnabilityEvidence {
                residual: 0.9,
                uncertainty: 0.2,
                compatible_worlds: 1,
                discriminating_evidence: 0.9,
                residual_persistence: 0.9,
                expected_adapt_gain: 0.1
            }),
            LearnabilityDecision::Reframe(_)
        ));
    }

    #[test]
    fn experiment_selection_uses_hypothesis_disagreement_not_an_oracle_label() {
        let gate = LearnabilityGate;
        assert_eq!(gate.choose_experiment(&request()).unwrap().id, "intervene");
    }
}
