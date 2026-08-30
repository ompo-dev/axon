//! Scheduler V5 orientado a valor cognitivo por custo de informação.

use std::collections::BTreeSet;

use super::cost::{CostError, CostOrigin, CostVector, CostWeights};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermodynamicBudget {
    pub max_weighted_cost: f64,
    pub origin: CostOrigin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThermalCandidate {
    pub id: String,
    pub active_concepts: BTreeSet<String>,
    pub estimated_cost: CostVector,
    pub utility_milliunits: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThermalPlan {
    pub id: String,
    pub active_concepts: BTreeSet<String>,
    pub estimated_cost: CostVector,
    pub value_per_cost: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionAudit {
    pub within_budget: bool,
    pub estimated_weighted_cost: f64,
    pub actual_weighted_cost: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermodynamicScheduler {
    weights: CostWeights,
}

impl ThermodynamicScheduler {
    pub const fn new(weights: CostWeights) -> Self {
        Self { weights }
    }

    /// Seleciona somente custos da mesma proveniência e dentro do orçamento.
    /// A métrica é uma política declarada, não uma medição termodinâmica física.
    pub fn schedule(
        &self,
        budget: ThermodynamicBudget,
        candidates: &[ThermalCandidate],
    ) -> Option<ThermalPlan> {
        candidates
            .iter()
            .filter(|candidate| candidate.estimated_cost.origin == budget.origin)
            .filter_map(|candidate| {
                let cost = candidate.estimated_cost.weighted_total(self.weights);
                (cost > 0.0 && cost <= budget.max_weighted_cost).then(|| ThermalPlan {
                    id: candidate.id.clone(),
                    active_concepts: candidate.active_concepts.clone(),
                    estimated_cost: candidate.estimated_cost,
                    value_per_cost: candidate.utility_milliunits as f64 / cost,
                })
            })
            .max_by(|left, right| {
                left.value_per_cost
                    .total_cmp(&right.value_per_cost)
                    .then_with(|| right.id.cmp(&left.id))
            })
    }

    pub fn audit(
        &self,
        plan: &ThermalPlan,
        actual: CostVector,
        budget: ThermodynamicBudget,
    ) -> Result<ExecutionAudit, CostError> {
        plan.estimated_cost.ensure_comparable_to(actual)?;
        if actual.origin != budget.origin {
            return Err(CostError::MixedOrigins);
        }
        let estimated_weighted_cost = plan.estimated_cost.weighted_total(self.weights);
        let actual_weighted_cost = actual.weighted_total(self.weights);
        Ok(ExecutionAudit {
            within_budget: actual_weighted_cost <= budget.max_weighted_cost,
            estimated_weighted_cost,
            actual_weighted_cost,
        })
    }
}

impl Default for ThermodynamicScheduler {
    fn default() -> Self {
        Self::new(CostWeights::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, concepts: &[&str], cost: CostVector, utility: u64) -> ThermalCandidate {
        ThermalCandidate {
            id: id.to_string(),
            active_concepts: concepts
                .iter()
                .map(|concept| (*concept).to_string())
                .collect(),
            estimated_cost: cost,
            utility_milliunits: utility,
        }
    }

    #[test]
    fn scheduler_prefers_value_per_moved_byte_and_keeps_the_active_set_small() {
        let budget = ThermodynamicBudget {
            max_weighted_cost: 100.0,
            origin: CostOrigin::Declared,
        };
        let plan = ThermodynamicScheduler::default()
            .schedule(
                budget,
                &[
                    candidate(
                        "wake-all",
                        &["a", "b", "c", "d"],
                        CostVector::declared(8, 20, 0, 0, 1),
                        20,
                    ),
                    candidate(
                        "local-reflex",
                        &["a"],
                        CostVector::declared(2, 2, 0, 0, 1),
                        10,
                    ),
                ],
            )
            .unwrap();

        assert_eq!(plan.id, "local-reflex");
        assert_eq!(plan.active_concepts.len(), 1);
    }

    #[test]
    fn audit_rejects_a_real_cost_that_exceeds_its_budget() {
        let scheduler = ThermodynamicScheduler::default();
        let budget = ThermodynamicBudget {
            max_weighted_cost: 20.0,
            origin: CostOrigin::Measured,
        };
        let plan = scheduler
            .schedule(
                budget,
                &[candidate(
                    "measured",
                    &["a"],
                    CostVector::measured(1, 1, 0, 0, 1, 1),
                    10,
                )],
            )
            .unwrap();

        let audit = scheduler
            .audit(&plan, CostVector::measured(10, 10, 0, 0, 1, 1), budget)
            .unwrap();
        assert!(!audit.within_budget);
    }
}
