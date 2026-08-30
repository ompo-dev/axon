//! Crédito local, traces de elegibilidade e consolidação fast/slow.

use super::factor::Factor;
use super::ids::FactorId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EligibilityTrace {
    pub factor: FactorId,
    pub value: f32,
    pub decay: f32,
    pub timestamp: u64,
}

impl EligibilityTrace {
    pub fn advance(self, activity: f32, timestamp: u64) -> Self {
        Self {
            factor: self.factor,
            value: (self.value * self.decay + activity).clamp(0.0, 1.0),
            timestamp,
            ..self
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TeachingSignal {
    pub target: FactorId,
    pub dimensions: Vec<f32>,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LocalLearner;

impl LocalLearner {
    pub fn apply(
        &self,
        factor: &Factor,
        trace: EligibilityTrace,
        teaching: &TeachingSignal,
        timestamp: u64,
    ) -> Option<Factor> {
        if trace.factor != factor.id
            || teaching.target != factor.id
            || !trace.value.is_finite()
            || !trace.decay.is_finite()
            || !teaching.confidence.is_finite()
            || teaching.dimensions.iter().any(|value| !value.is_finite())
            || !(0.0..=1.0).contains(&trace.value)
            || !(0.0..=1.0).contains(&trace.decay)
        {
            return None;
        }
        let credit = teaching.dimensions.iter().copied().sum::<f32>()
            * trace.value
            * teaching.confidence.clamp(0.0, 1.0);
        if !credit.is_finite() {
            return None;
        }
        let mut next = factor.clone();
        let updated_weight = next.learning.fast_weight + next.learning.plasticity * credit;
        if !updated_weight.is_finite() {
            return None;
        }
        next.learning.fast_weight = updated_weight;
        next.learning.evidence_count = next.learning.evidence_count.saturating_add(1);
        next.learning.last_update = timestamp;
        Some(next)
    }

    pub fn consolidate(&self, factor: &Factor, coherent: bool) -> Factor {
        let mut next = factor.clone();
        if coherent {
            next.learning.slow_weight += 0.1 * next.learning.fast_weight;
            next.learning.stability = (next.learning.stability + 0.1).min(1.0);
            next.learning.plasticity = (next.learning.plasticity * 0.9).max(0.01);
        } else {
            next.learning.fast_weight *= 0.5;
            next.learning.stability *= 0.9;
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_v6::{Factor, FactorKind, RepresentationKind, RevisionId};

    #[test]
    fn teaching_changes_only_the_eligible_target_and_coherence_consolidates_slow_state() {
        let factor = Factor::try_new(
            FactorId(1),
            FactorKind::Dynamic,
            RepresentationKind::Continuous,
            "state",
            0.1,
            RevisionId(1),
        )
        .unwrap();
        let learner = LocalLearner;
        let trace = EligibilityTrace {
            factor: FactorId(1),
            value: 0.5,
            decay: 0.9,
            timestamp: 0,
        };
        let updated = learner
            .apply(
                &factor,
                trace,
                &TeachingSignal {
                    target: FactorId(1),
                    dimensions: vec![1.0],
                    confidence: 1.0,
                },
                1,
            )
            .unwrap();

        assert!(updated.learning.fast_weight > factor.learning.fast_weight);
        assert!(
            learner
                .apply(
                    &factor,
                    trace,
                    &TeachingSignal {
                        target: FactorId(2),
                        dimensions: vec![1.0],
                        confidence: 1.0
                    },
                    1
                )
                .is_none()
        );
        assert!(learner.consolidate(&updated, true).learning.slow_weight > 0.0);
        assert!(
            learner
                .apply(
                    &factor,
                    trace,
                    &TeachingSignal {
                        target: FactorId(1),
                        dimensions: vec![f32::NAN],
                        confidence: 1.0,
                    },
                    1,
                )
                .is_none()
        );
    }
}
